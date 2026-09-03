"""Base platform adapter interface; every platform adapter inherits from BasePlatformAdapter."""

import asyncio
import inspect
import ipaddress
import logging
import os
import random
import re
import socket as _socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from utils import normalize_proxy_url

logger = logging.getLogger(__name__)


def _consume_detached_handler_exception(task: "asyncio.Task") -> None:
    """Done-callback retrieving a detached fatal-error handler's exception, so handler
    tasks left running after their carrier was cancelled (``_notify_fatal_error``)
    never log "Task exception was never retrieved"."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Detached fatal-error handler task failed: %s", exc, exc_info=exc)


# Audio extensions Hermes recognizes for native audio delivery. Telegram's narrower
# attachment/voice sets stay separate: MPEG-2 Layer II is audio to Hermes but not to sendAudio.
_AUDIO_MIME_TYPES = {
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".mp3": "audio/mpeg", ".m2a": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/m4a", ".flac": "audio/flac"}
_AUDIO_EXTS = frozenset(_AUDIO_MIME_TYPES)
# Outbound dispatch partition for MEDIA/local files (image batch vs send_video).
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
# Telegram sendAudio accepts only MP3 / M4A; others go via sendVoice (Opus/OGG) or as a document.
_TELEGRAM_AUDIO_ATTACHMENT_EXTS = frozenset({'.mp3', '.m4a'})
_TELEGRAM_VOICE_EXTS = frozenset({'.ogg', '.opus'})


def transcode_to_ogg_opus(path: str, *, bitrate: str = "32k") -> "str | None":
    """Best-effort ffmpeg transcode to Ogg/Opus (voip-tuned) for native voice bubbles: a NEW
    temp ``.ogg`` path (caller owns cleanup), or ``None`` when ffmpeg is missing/fails so
    callers keep their document fallback. Blocking — use ``asyncio.to_thread``."""
    import shutil as _shutil
    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    fd, ogg_path = tempfile.mkstemp(prefix="voice_transcode_", suffix=".ogg")
    os.close(fd)
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path),
             "-acodec", "libopus", "-ac", "1", "-b:a", bitrate, "-vbr", "on",
             "-application", "voip", "-compression_level", "10", ogg_path],
            capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.getsize(ogg_path) > 0:
            return ogg_path
    except Exception:
        logger.debug("voice transcode to Ogg/Opus failed for %s", path, exc_info=True)
    try:
        os.unlink(ogg_path)
    except OSError:
        pass
    return None
_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS = 30.0
# Delivery-time history is best-effort dedup metadata: stay well below the Discord heartbeat
# watchdog window and fail open rather than withhold a legitimate attachment.
_HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS = 5.0
# Timed-out reads can't be cancelled mid-SQLite; isolate and cap them so wedged dedup work
# can't consume the shared executor or spawn unbounded threads.
_HISTORY_MEDIA_LOOKUP_MAX_WORKERS = 2
_HISTORY_MEDIA_LOOKUP_ADMISSION = threading.BoundedSemaphore(_HISTORY_MEDIA_LOOKUP_MAX_WORKERS)


def _platform_name(platform) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Platform-aware thread metadata for adapter sends. Telegram DM topics route with
    ``message_thread_id`` + a reply anchor; anchorless synthetic/resumed sends fall back to
    ``direct_messages_topic_id`` when supported."""
    thread_id = getattr(source, "thread_id", None)
    platform = _platform_name(getattr(source, "platform", None))
    metadata = {"thread_id": thread_id} if thread_id is not None else {}
    # Slack workspace identity is durable routing state: carry it on every outbound path
    # so a multi-workspace Socket Mode gateway never falls back to its primary WebClient.
    scope_id = getattr(source, "scope_id", None) if platform == "slack" else None
    if scope_id:
        metadata["slack_team_id"] = str(scope_id)
    if not metadata:
        return None
    if platform == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        if str(thread_id) not in {"", "1"}:
            metadata["direct_messages_topic_id"] = str(thread_id)
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    # Routed profile for shared state.db namespaces (multiplex / profile_routes);
    # outbound prune paths must not assume the adapter's static profile stamp.
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile:
        metadata["hermes_profile"] = profile
    return metadata


def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    return notify_metadata


def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics."""
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None))
    thread_id = getattr(source, "thread_id", None)
    raw_message = getattr(event, "raw_message", None)
    if (platform == "slack" and isinstance(raw_message, dict)
            and raw_message.get("_hermes_no_thread_response")):
        # Slack reaction handoffs create a new top-level message in the target channel;
        # returning message_id would make _resolve_thread_ts() reply in a nonexistent thread.
        return None
    if platform == "telegram" and thread_id:
        # Forum topics route by topic metadata (no reply); DM-topic lanes reply to the
        # triggering user message — replying to Telegram's topic seed/anchor can render
        # the bot response outside the active lane.
        if getattr(source, "chat_type", None) != "dm":
            return None
        return getattr(event, "message_id", None) or getattr(event, "reply_to_message_id", None)
    if platform == "feishu" and thread_id and getattr(event, "reply_to_message_id", None):
        return getattr(event, "reply_to_message_id", None)
    return getattr(event, "message_id", None)


def _media_failure_text(kind: str, file_name: "str | None" = None) -> str:
    """User-facing "couldn't deliver" notice; ``file_name`` is the only name ever shown."""
    suffix = f" ({file_name})" if file_name else ""
    return f"⚠️ Couldn't deliver the {kind} attachment{suffix}."


def should_send_media_as_audio(platform, ext: str, is_voice: bool = False) -> bool:
    """True when a media file should use the platform's audio sender. Telegram: explicit
    ``is_voice`` ([[audio_as_voice]]) routes ANY format to the voice sender (adapter transcodes
    non-Opus); otherwise only sendAudio's MP3/M4A qualify — a plain Opus/OGG attachment is never
    turned into a voice bubble, everything else → document. Other platforms: any audio ext."""
    normalized_ext = (ext or "").lower()
    if normalized_ext not in _AUDIO_EXTS:
        return False
    if _platform_name(platform) != "telegram":
        return True
    return is_voice or normalized_ext in _TELEGRAM_AUDIO_ATTACHMENT_EXTS


def build_auto_tts_output_path(platform) -> str:
    """Unique temp output path for gateway auto-TTS: ``.ogg`` for ``OPUS_VOICE_PLATFORMS``
    (the tool's ``_repair_ogg_container`` then guarantees real Opus bytes), else ``.mp3``.
    Platform-awareness lives HERE because ``_clear_session_env`` wipes the TTS tool's
    ``HERMES_SESSION_PLATFORM`` contextvar before the post-handler auto-TTS block runs."""
    from tools.tts_tool import OPUS_VOICE_PLATFORMS
    ext = "ogg" if _platform_name(platform) in OPUS_VOICE_PLATFORMS else "mp3"
    audio_path = os.path.join(
        tempfile.gettempdir(), "hermes_voice", f"tts_reply_{uuid.uuid4().hex[:12]}.{ext}")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    return audio_path


def utf16_len(s: str) -> int:
    """UTF-16 code units in *s* — Telegram's 4 096 limit counts those, so astral chars
    (emoji, CJK Ext B) cost **two** units although Python's ``len()`` counts one."""
    return len(s.encode("utf-16-le")) // 2


def _custom_unit_to_cp(s: str, budget: int, len_fn) -> int:
    """Largest codepoint offset *n* with ``len_fn(s[:n]) <= budget`` (binary search)."""
    if len_fn(s) <= budget:
        return len(s)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _prefix_within_utf16_limit(s: str, limit: int) -> str:
    """Longest prefix of *s* with UTF-16 length ≤ *limit*; never splits a surrogate pair."""
    return s[:_custom_unit_to_cp(s, limit, utf16_len)]


def is_network_accessible(host: str) -> bool:
    """True if *host* would expose the server beyond loopback (incl. IPv4-mapped
    ::ffff:127.0.0.1); hostnames are resolved and DNS failure fails closed (True)."""
    try:
        addr = ipaddress.ip_address(host)
        # ::ffff:127.0.0.1 reports is_loopback=False; check the mapped IPv4 explicitly.
        mapped = getattr(addr, "ipv4_mapped", None)
        return not (addr.is_loopback or (mapped and mapped.is_loopback))
    except ValueError:
        pass  # hostname — resolve below
    try:
        resolved = _socket.getaddrinfo(host, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
        # Network-accessible if any resolved address is non-loopback.
        return any(not ipaddress.ip_address(sockaddr[0]).is_loopback for *_, sockaddr in resolved)
    except (_socket.gaierror, OSError):
        return True


def _detect_macos_system_proxy() -> str | None:
    """Read the macOS system HTTP(S) proxy via ``scutil --proxy``: ``http://host:port``
    when an HTTP(S) proxy is enabled, else None (non-macOS or any subprocess error)."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["scutil", "--proxy"], timeout=3, text=True, encoding='utf-8', errors='replace', stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    props = {
        key.strip(): val.strip()
        for key, sep, val in (line.strip().partition(" : ") for line in out.splitlines()) if sep}
    # Prefer HTTPS, fall back to HTTP
    for enable_key, host_key, port_key in (
        ("HTTPSEnable", "HTTPSProxy", "HTTPSPort"), ("HTTPEnable", "HTTPProxy", "HTTPPort")):
        if props.get(enable_key) == "1" and props.get(host_key) and props.get(port_key):
            return f"http://{props[host_key]}:{props[port_key]}"
    return None


def _split_host_port(value: str) -> tuple[str, int | None]:
    raw = str(value or "").strip()
    if not raw:
        return "", None
    if "://" in raw:
        parsed = urlsplit(raw)
        return (parsed.hostname or "").lower().rstrip("."), parsed.port
    if raw.startswith("[") and "]" in raw:
        host, _, rest = raw[1:].partition("]")
        port = int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
        return host.lower().rstrip("."), port
    host, _, maybe_port = raw.rpartition(":")
    if raw.count(":") == 1 and maybe_port.isdigit():
        return host.lower().rstrip("."), int(maybe_port)
    return raw.lower().strip("[]").rstrip("."), None


def _no_proxy_entries() -> list[str]:
    return [
        part.strip() for key in ("NO_PROXY", "no_proxy")
        for part in os.environ.get(key, "").split(",") if part.strip()]


def _ip_or_none(value: str, parse=ipaddress.ip_address):
    """``parse(value)`` or None on ``ValueError`` (``parse`` is ip_address / ip_network)."""
    try:
        return parse(value)
    except ValueError:
        return None


def _no_proxy_entry_matches(entry: str, host: str, port: int | None = None) -> bool:
    token = str(entry or "").strip().lower()
    if not token:
        return False
    if token == "*":
        return True
    token_host, token_port = _split_host_port(token)
    if token_port is not None and (port is None or token_port != port):
        return False
    if not token_host:
        return False
    host_ip = _ip_or_none(host)
    network = _ip_or_none(token_host, lambda v: ipaddress.ip_network(v, strict=False))
    if network is not None:
        return host_ip is not None and host_ip in network
    token_ip = _ip_or_none(token_host)
    if token_ip is not None:
        return host_ip == token_ip
    if token_host.startswith("*."):
        return host.endswith(token_host[1:])
    if token_host.startswith("."):
        return host == token_host[1:] or host.endswith(token_host)
    return host == token_host or host.endswith(f".{token_host}")


def should_bypass_proxy(target_hosts: str | list[str] | tuple[str, ...] | set[str] | None) -> bool:
    """Return True when NO_PROXY/no_proxy matches at least one target host.

    Supports exact hosts, domain suffixes, wildcard suffixes, IP literals,
    CIDR ranges, optional host:port entries, and ``*``.
    """
    entries = _no_proxy_entries()
    if not entries or not target_hosts:
        return False
    candidates = [target_hosts] if isinstance(target_hosts, str) else list(target_hosts)
    return any(
        host and any(_no_proxy_entry_matches(entry, host, port) for entry in entries)
        for host, port in map(_split_host_port, map(str, candidates)))


def resolve_proxy_url(
    platform_env_var: str | None = None, *,
    target_hosts: str | list[str] | tuple[str, ...] | set[str] | None = None) -> str | None:
    """Proxy URL: *platform_env_var* (e.g. ``DISCORD_PROXY``) first, then HTTPS_PROXY /
    HTTP_PROXY / ALL_PROXY (any case), then the macOS system proxy — the latter two only when
    ``gateway.trust_env`` is true. None when nothing is found or NO_PROXY matches a target."""
    value = (os.environ.get(platform_env_var) or "").strip() if platform_env_var else ""
    if not value:
        if not gateway_trust_env():
            # trust_env false: only the explicit per-platform var above is honored.
            return None
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy"):
            value = (os.environ.get(key) or "").strip()
            if value:
                break
    proxy = normalize_proxy_url(value or _detect_macos_system_proxy())
    if proxy and should_bypass_proxy(target_hosts):
        return None
    return proxy


def _aiohttp_socks_connector(proxy_url: str):
    """``aiohttp_socks.ProxyConnector`` for ``proxy_url``, or None when aiohttp_socks is
    missing (SOCKS then logs a warning; HTTP callers fall back to ``proxy=``).
    ``rdns=True`` forces remote DNS through the proxy — required by many SOCKS
    implementations (Shadowrocket, Clash) and essential against GFW DNS pollution."""
    try:
        from aiohttp_socks import ProxyConnector
        return ProxyConnector.from_url(proxy_url, rdns=True)
    except ImportError:
        if proxy_url.lower().startswith("socks"):
            logger.warning(
                "aiohttp_socks not installed — SOCKS proxy %s ignored. "
                "Run: pip install aiohttp-socks", proxy_url)
        return None


def proxy_kwargs_for_bot(proxy_url: str | None) -> dict:
    """Kwargs for ``commands.Bot()`` / ``discord.Client()``: SOCKS → ``{"connector"}``,
    HTTP → ``{"proxy": url}``, None → ``{}``."""
    if not proxy_url:
        return {}
    if proxy_url.lower().startswith("socks"):
        connector = _aiohttp_socks_connector(proxy_url)
        return {"connector": connector} if connector is not None else {}
    return {"proxy": proxy_url}


def _config_section(name: str) -> dict:
    """Read-only ``config.yaml`` section ``name``; ``{}`` when unreadable/missing/not a dict."""
    try:
        from hermes_cli.config import load_config_readonly as _load_config
        cfg = _load_config()  # read-only: .get() only, never mutated
    except Exception:
        return {}
    section = cfg.get(name) if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def gateway_trust_env() -> bool:
    """``gateway.trust_env`` from config.yaml (default True): whether gateway
    ``aiohttp.ClientSession``s honor HTTP(S)_PROXY / NO_PROXY / SSL_CERT_FILE. Set false
    when the gateway inherits a proxy env it must not use. Fail-open to default."""
    value = _config_section("gateway").get("trust_env", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value) if value is not None else True


def proxy_kwargs_for_aiohttp(proxy_url: str | None) -> tuple[dict, dict]:
    """``(session_kwargs, request_kwargs)`` for a standalone ``aiohttp.ClientSession``.
    With aiohttp-socks every scheme uses a connector (libraries like mautrix never forward
    per-request ``proxy=``); without it HTTP falls back to ``({}, {"proxy": url})``, SOCKS is ignored."""
    if not proxy_url:
        return {}, {}
    connector = _aiohttp_socks_connector(proxy_url)
    if connector is not None:
        return {"connector": connector}, {}
    return ({}, {}) if proxy_url.lower().startswith("socks") else ({}, {"proxy": proxy_url})


def is_host_excluded_by_no_proxy(hostname: str, no_proxy_value: str | None = None) -> bool:
    """Return True when ``hostname`` matches a ``NO_PROXY`` entry (comma/whitespace
    separated; leading-dot and ``*.`` entries match the apex domain and subdomains)."""
    if no_proxy_value is None:
        no_proxy_value = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    lower_hostname = hostname.lower()
    for entry in re.split(r"[\s,]+", no_proxy_value.strip()):
        normalized = entry.strip().lower()
        if not normalized:
            continue
        if normalized == "*":
            return True
        normalized = normalized[2:] if normalized.startswith("*.") else normalized.removeprefix(".")
        if lower_hostname == normalized or lower_hostname.endswith(f".{normalized}"):
            return True
    return False


import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Callable, Awaitable, Tuple, Union
from enum import Enum

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import fence_state_after
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_default_hermes_root, get_hermes_dir, get_hermes_home

if TYPE_CHECKING:
    from agent.display import ToolPreview

@dataclass
class AudioFormat:
    """Declared PCM format for a streaming-TTS session: every ``write_streaming_tts``
    chunk must be raw little-endian PCM at this rate / channels / sample width."""
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (int16 = 2)


@dataclass
class StreamingTTSHandle:
    """Opaque handle returned by ``begin_streaming_tts``; adapters may extend it with
    platform state. The base fields are consumer bookkeeping / cancellation."""
    chat_id: str = ""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    # True once the first PCM chunk is written: a later failure then ends cleanly instead of
    # falling back to whole-file TTS (don't replay already-audible output).
    audible: bool = False
    aborted: bool = False  # set by abort_streaming_tts; late chunks are dropped


def streaming_tts_turn_key(session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> str | None:
    """Per-turn streaming-TTS suppression key — turn-scoped (not chat-scoped) so
    overlapping turns in one chat can't suppress each other's fallback paths.
    ``turn_marker`` is normally the run generation, else the event's message/update id."""
    if not session_key:
        return None
    if turn_marker is None and event is not None:
        turn_marker = getattr(event, "message_id", None) or getattr(event, "platform_update_id", None)
    return None if turn_marker is None else f"{session_key}:{turn_marker}"


def streaming_tts_should_skip_whole_file(
    completed_turns: set[str], session_key: str | None, turn_marker: Any = None, *,
    event: Any = None) -> bool:
    """Pure, turn-scoped auto-TTS suppression decision (testable without the adapter stack)."""
    turn_key = streaming_tts_turn_key(session_key, turn_marker, event=event)
    return bool(turn_key and turn_key in completed_turns)


GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE = (
    "Secure secret entry is not supported over messaging. "
    "Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually.")


def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo)."""
    raw = str(url) if max_len > 0 and url is not None else ""
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw[:max_len]
    safe = raw
    if parsed.scheme and parsed.netloc:
        # Strip potential embedded credentials (user:pass@host).
        path = parsed.path or ""
        basename = path.rsplit("/", 1)[-1]
        tail = "" if path in ("", "/") else f"/.../{basename}" if basename else "/..."
        safe = f"{parsed.scheme}://{parsed.netloc.rsplit('@', 1)[-1]}{tail}"
    if len(safe) <= max_len:
        return safe
    if max_len <= 3:
        return "." * max_len
    return f"{safe[:max_len - 3]}..."


async def _ssrf_redirect_guard(response):
    """Re-validate each redirect target: a public URL that 302-redirects to
    http://169.254.169.254/ would otherwise bypass the pre-flight is_safe_url() check.
    Async because httpx.AsyncClient awaits response event hooks."""
    from tools.url_safety import is_safe_url, redirect_target_from_response
    redirect_url = redirect_target_from_response(response)
    if redirect_url and not is_safe_url(redirect_url):
        raise ValueError(f"Blocked redirect to private/internal address: {safe_url_for_log(redirect_url)}")


# Inbound images are cached locally so the vision tool can read them by path (platform URLs
# are ephemeral). Import-time default; tests monkeypatch it, getters re-resolve per call.
IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")


def _resolve_cache_dir(constant_name: str, new_subpath: str, old_name: str) -> Path:
    """Resolve fresh via get_hermes_dir (active profile) unless a test monkeypatched
    the module constant away from its import-time default; create the directory."""
    d = get_hermes_dir(new_subpath, old_name)
    current = globals().get(constant_name)
    default = _CACHE_DIR_IMPORT_DEFAULTS.get(constant_name)
    if current is not None and default is not None and current != default:
        d = Path(current)
    d.mkdir(parents=True, exist_ok=True)
    return d

# Inbound media size cap: payloads are buffered fully in memory before hitting the cache, so
# an uncapped upload (Discord Nitro: 500 MB) or huge remote URL can OOM-kill the gateway.
# Enforced in ``cache_*_from_bytes`` and ``cache_*_from_url``; ``gateway.max_inbound_media_bytes``.
DEFAULT_INBOUND_MEDIA_MAX_BYTES = 128 * 1024 * 1024


def get_inbound_media_max_bytes() -> int:
    """Max inbound media bytes held in memory (``gateway.max_inbound_media_bytes``);
    ``0`` / negative / unparseable disables the cap; unreadable config → default."""
    try:
        return int(_config_section("gateway")["max_inbound_media_bytes"])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_INBOUND_MEDIA_MAX_BYTES


def validate_inbound_media_size(
    size: int, *, media_type: str = "media", max_bytes: Optional[int] = None) -> None:
    """Raise ``ValueError`` if an inbound payload exceeds the cap (``max_bytes`` of ``0``
    disables it; pass it explicitly to resolve the limit once across an incremental read)."""
    limit = get_inbound_media_max_bytes() if max_bytes is None else max_bytes
    if limit and size > limit:
        raise ValueError(f"Inbound {media_type} payload is too large ({size} bytes > {limit} bytes)")


async def _read_httpx_body_with_limit(response, *, media_type: str) -> bytes:
    """Read an httpx streaming body under the media cap: reject an oversized
    ``Content-Length`` early, then re-check the running total per chunk so a
    lying/absent header can't smuggle an unbounded body past the cap."""
    max_bytes = get_inbound_media_max_bytes()
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            logger.debug("Ignoring invalid Content-Length for inbound %s: %r", media_type, content_length)
        else:
            validate_inbound_media_size(declared_size, media_type=media_type, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        validate_inbound_media_size(total, media_type=media_type, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _cache_dir_accessors(kind: str, constant_name: str, new_subpath: str, old_name: str):
    """``(get_<kind>_cache_dir, cleanup_<kind>_cache)`` pair: the getter resolves and creates
    the directory; ``cleanup(max_age_hours=24)`` deletes older files and returns the count."""
    def get_dir() -> Path:
        return _resolve_cache_dir(constant_name, new_subpath, old_name)

    def cleanup(max_age_hours: int = 24) -> int:
        return _cleanup_cache_dir(get_dir(), max_age_hours)
    get_dir.__name__ = get_dir.__qualname__ = f"get_{kind}_cache_dir"
    cleanup.__name__ = cleanup.__qualname__ = f"cleanup_{kind}_cache"
    return get_dir, cleanup


get_image_cache_dir, cleanup_image_cache = _cache_dir_accessors(
    "image", "IMAGE_CACHE_DIR", "cache/images", "image_cache")


def _looks_like_image(data: bytes) -> bool:
    """Return True if *data* starts with a known image magic-byte sequence."""
    if len(data) < 4:
        return False
    return (
        data[:8] == b"\x89PNG\r\n\x1a\n"
        or data[:3] == b"\xff\xd8\xff"
        or data[:6] in {b"GIF87a", b"GIF89a"}
        or data[:2] == b"BM"
        or (data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP"))


def _write_cache_file(cache_dir: Path, prefix: str, ext: str, data: bytes) -> str:
    """Write ``data`` to ``<cache_dir>/<prefix>_<uuid12><ext>``; return the path string."""
    filepath = cache_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    filepath.write_bytes(data)
    return str(filepath)


def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """Save raw image bytes to the cache and return the absolute path; raises
    ValueError when *data* isn't an image (e.g. an upstream HTML error page)."""
    validate_inbound_media_size(len(data), media_type="image")
    if not _looks_like_image(data):
        snippet = data[:80].decode("utf-8", errors="replace")
        raise ValueError(f"Refusing to cache non-image data as {ext} (starts with: {snippet!r})")
    return _write_cache_file(get_image_cache_dir(), "img", ext, data)


async def _cache_media_from_url(
    url: str, ext: str, retries: int, *, media_type: str, accept: str, cache_fn, log_label: str,
) -> str:
    """Shared downloader behind ``cache_image_from_url`` / ``cache_audio_from_url``:
    SSRF-checked (pre-flight + per-redirect; raises ValueError), size-capped, and
    retried with linear backoff on timeouts / 429 / 5xx."""
    from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
    import httpx
    if not is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL (SSRF protection): {safe_url_for_log(url)}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)", "Accept": accept}
    async with create_ssrf_safe_async_client(
        timeout=30.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]},
    ) as client:
        for attempt in range(retries + 1):
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    content = await _read_httpx_body_with_limit(response, media_type=media_type)
                return cache_fn(content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    logger.debug(
                        "%s cache retry %d/%d for %s (%.1fs): %s",
                        log_label, attempt + 1, retries, safe_url_for_log(url), wait, exc)
                    await asyncio.sleep(wait)
                    continue
                raise


async def cache_image_from_url(url: str, ext: str = ".jpg", retries: int = 2) -> str:
    """Download an image URL into the image cache; return the absolute path."""
    return await _cache_media_from_url(
        url, ext, retries, media_type="image", accept="image/*,*/*;q=0.8",
        cache_fn=cache_image_from_bytes, log_label="Media")


def _cleanup_cache_dir(cache_dir: Path, max_age_hours: int) -> int:
    """Delete files in *cache_dir* older than *max_age_hours*; return the count removed."""
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# Audio cache utilities (same pattern as images; feeds the STT tool).
AUDIO_CACHE_DIR = get_hermes_dir("cache/audio", "audio_cache")
get_audio_cache_dir, cleanup_audio_cache = _cache_dir_accessors(
    "audio", "AUDIO_CACHE_DIR", "cache/audio", "audio_cache")


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """Save raw audio bytes to the cache (container-sniffed ext); return the path."""
    # tools.audio_container is the ONE owner of container detection (outbound TTS repair + here).
    from tools.audio_container import sniff_audio_ext
    validate_inbound_media_size(len(data), media_type="audio")
    return _write_cache_file(get_audio_cache_dir(), "audio", sniff_audio_ext(data, ext), data)


async def cache_audio_from_url(url: str, ext: str = ".ogg", retries: int = 2) -> str:
    """Download an audio URL into the audio cache; return the absolute path."""
    return await _cache_media_from_url(
        url, ext, retries, media_type="audio", accept="audio/*,*/*;q=0.8",
        cache_fn=cache_audio_from_bytes, log_label="Audio")


# Video cache utilities (same pattern; referenced by local path).
VIDEO_CACHE_DIR = get_hermes_dir("cache/videos", "video_cache")
get_video_cache_dir, cleanup_video_cache = _cache_dir_accessors(
    "video", "VIDEO_CACHE_DIR", "cache/videos", "video_cache")

SUPPORTED_VIDEO_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}


def cache_video_from_bytes(data: bytes, ext: str = ".mp4") -> str:
    """Save raw video bytes to the cache and return the absolute file path."""
    validate_inbound_media_size(len(data), media_type="video")
    return _write_cache_file(get_video_cache_dir(), "video", ext, data)


# Document / screenshot cache utilities (same pattern; referenced by local path).
DOCUMENT_CACHE_DIR = get_hermes_dir("cache/documents", "document_cache")
SCREENSHOT_CACHE_DIR = get_hermes_dir("cache/screenshots", "browser_screenshots")
get_document_cache_dir, cleanup_document_cache = _cache_dir_accessors(
    "document", "DOCUMENT_CACHE_DIR", "cache/documents", "document_cache")
get_screenshot_cache_dir, cleanup_screenshot_cache = _cache_dir_accessors(
    "screenshot", "SCREENSHOT_CACHE_DIR", "cache/screenshots", "browser_screenshots")

# Import-time defaults; _resolve_cache_dir compares against these to detect a test monkeypatch.
_CACHE_DIR_IMPORT_DEFAULTS = {
    "IMAGE_CACHE_DIR": IMAGE_CACHE_DIR, "AUDIO_CACHE_DIR": AUDIO_CACHE_DIR,
    "VIDEO_CACHE_DIR": VIDEO_CACHE_DIR, "DOCUMENT_CACHE_DIR": DOCUMENT_CACHE_DIR,
    "SCREENSHOT_CACHE_DIR": SCREENSHOT_CACHE_DIR}

_HERMES_HOME = get_hermes_home()
_HERMES_ROOT = get_default_hermes_root()
MEDIA_DELIVERY_ALLOW_DIRS_ENV = "HERMES_MEDIA_ALLOW_DIRS"
MEDIA_DELIVERY_TRUST_RECENT_ENV = "HERMES_MEDIA_TRUST_RECENT_FILES"
MEDIA_DELIVERY_TRUST_RECENT_SECONDS_ENV = "HERMES_MEDIA_TRUST_RECENT_SECONDS"
# Strict mode = allowlist+recency validation. Off by default (symmetric with inbound; the
# denylist still blocks credential / system paths); set true on public-facing gateways.
MEDIA_DELIVERY_STRICT_ENV = "HERMES_MEDIA_DELIVERY_STRICT"
# Canonical cache subdirs holding deliverable artifacts; also used to enumerate
# per-profile cache roots at check time (_media_delivery_allowed_roots).
_MEDIA_DELIVERY_CACHE_SUBDIRS = ("images", "audio", "videos", "documents", "screenshots")
MEDIA_DELIVERY_SAFE_ROOTS = (
    IMAGE_CACHE_DIR, AUDIO_CACHE_DIR, VIDEO_CACHE_DIR, DOCUMENT_CACHE_DIR, SCREENSHOT_CACHE_DIR,
    *(_HERMES_HOME / d for d in (
        "image_cache", "audio_cache", "video_cache", "document_cache", "browser_screenshots")),
    # Canonical cache layout, alongside the legacy *_cache dirs (installs may have both).
    *(_HERMES_HOME / "cache" / d for d in _MEDIA_DELIVERY_CACHE_SUBDIRS))

# Recency window (seconds) for trusting freshly-produced files: build artifacts land seconds
# before delivery; pre-existing host files (/etc/passwd, ~/.ssh/id_rsa) are days/months old.
_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS = 600

# Hard denylist applied even to "recent" files: credentials, system state, process
# introspection. The cache-dir allowlist still beats it (an operator may allow a root here).
_MEDIA_DELIVERY_DENIED_PREFIXES = (
    "/etc", "/proc", "/sys", "/dev", "/root", "/boot", "/var/log", "/var/lib", "/var/run")

# Credential / config dirs denied under $HOME (Library/Keychains = macOS), resolved at check time.
_MEDIA_DELIVERY_DENIED_HOME_SUBPATHS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config", ".azure", ".gcloud",
    "Library/Keychains")

# Credential / secret stores at the HERMES_HOME root, denied per-file (not the whole tree, so
# skills/, logs/ and agent-written files under ~/.hermes stay deliverable; cache subdirs are
# allowlisted BEFORE this denylist). Mirrors the read guard in agent/file_safety.py so the exfil
# side never trails the write side. google_token.json's mtime bumps every turn (defeats the strict
# recency window); pairing/ and mcp-tokens/ (live MCP OAuth tokens) are denied as whole trees.
_ROOT_CREDENTIAL_PATHS = (
    ".env", "auth.json", "auth.lock", "credentials", "config.yaml", ".anthropic_oauth.json",
    "google_token.json", "google_oauth_pending.json", os.path.join("auth", "google_oauth.json"),
    "webhook_subscriptions.json", os.path.join("cache", "bws_cache.json"),
    os.path.join("cache", "bws_cache.enc.json"), "pairing", "mcp-tokens")


def _profile_cache_roots() -> List[Path]:
    """Per-profile cache roots ``<root>/profiles/<name>/cache/{images,...}`` (the static safe
    roots cover only the active HERMES_HOME). Enumerated at check time so profiles created after
    startup count and are allowlisted BEFORE the ``/root`` denylist (HERMES_HOME symlinked)."""
    try:
        profile_dirs = [p for p in (_HERMES_ROOT / "profiles").iterdir() if p.is_dir()]
    except OSError:
        return []
    return [p / "cache" / subdir for p in profile_dirs for subdir in _MEDIA_DELIVERY_CACHE_SUBDIRS]


def _kanban_attachment_roots() -> List[Path]:
    """Return durable Kanban attachment roots without importing kanban_db."""
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        return [Path(override).expanduser()]
    home_override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    root = Path(home_override).expanduser() if home_override else _HERMES_ROOT
    roots = [root / "kanban" / "attachments"]
    boards_root = root / "kanban" / "boards"
    try:
        board_dirs = [
            path for path in boards_root.iterdir()
            if path.is_dir() and not path.is_symlink()
            and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", path.name)
            and (path / "kanban.db").is_file()]
    except OSError:
        return roots
    roots.extend(path / "attachments" for path in board_dirs)
    return roots


def _media_delivery_allowed_roots() -> List[Path]:
    """Return roots from which model-emitted local media may be delivered."""
    roots = [*map(Path, MEDIA_DELIVERY_SAFE_ROOTS), *_profile_cache_roots(),
             *_kanban_attachment_roots()]
    roots.extend(
        root for chunk in os.environ.get(MEDIA_DELIVERY_ALLOW_DIRS_ENV, "").split(os.pathsep)
        for raw_root in chunk.split(",")
        if (root := Path(os.path.expanduser(raw_root.strip()))).is_absolute())
    return roots


def _media_delivery_recency_seconds() -> float:
    """Recency window (seconds) for trusting fresh files; 0 = pure-allowlist mode."""
    raw = os.environ.get(MEDIA_DELIVERY_TRUST_RECENT_ENV, "1").strip().lower()
    if raw in ("0", "false", "no", "off", ""):
        return 0.0
    custom = os.environ.get(MEDIA_DELIVERY_TRUST_RECENT_SECONDS_ENV, "").strip()
    try:
        return max(0.0, float(custom)) if custom else float(_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS)
    except (TypeError, ValueError):
        return float(_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS)


def _media_delivery_denied_paths() -> List[Path]:
    """Return absolute denylist paths under which delivery is never allowed."""
    denied = [Path(p) for p in _MEDIA_DELIVERY_DENIED_PREFIXES]
    home = Path(os.path.expanduser("~"))
    denied.extend(home / sub for sub in _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS)
    for hermes_root in (_HERMES_HOME, _HERMES_ROOT):
        denied.extend(hermes_root / rel for rel in _ROOT_CREDENTIAL_PATHS)
    return denied


def _path_under_denied_prefix(resolved: Path) -> bool:
    """True if ``resolved`` lives under a deny-listed system path — except a denied prefix that
    IS the running user's own home: ``/root`` is listed so a non-root gateway can't deliver
    another user's home, but a root-run gateway's own deliverables live under ``$HOME=/root``.
    Credential sub-dirs (``~/.ssh``, ``~/.hermes/.env``) stay blocked (more-specific entries)."""
    try:
        home = Path(os.path.expanduser("~")).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        home = None
    for denied in _media_delivery_denied_paths():
        try:
            resolved_denied = denied.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        hit = resolved == resolved_denied or _path_is_within(resolved, resolved_denied)
        if hit and resolved_denied != home:
            return True
    return False


def _file_is_recently_produced(resolved: Path, window_seconds: float) -> bool:
    """True if mtime is within ``window_seconds`` — a session-scoped trust signal: agents
    produce artifacts seconds before sending; pre-existing host files are days/months old."""
    if window_seconds <= 0:
        return False
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) <= window_seconds


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware TERMINAL_* read: the per-turn scope carries the ACTIVE profile's settings while
    os.getenv reads whatever a prior turn pinned into the process env. Only ImportError falls
    back — a refusal scope must raise rather than rebuild another profile's policy from the env."""
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.getenv(name, default)
    return terminal_env(name, default)


def _parse_docker_volume_mounts() -> List[Tuple[Path, Path]]:
    """Parse ``TERMINAL_DOCKER_VOLUMES`` (JSON list of ``host:container[:mode]``) into
    ``(host_path, container_path)``; named volumes / non-absolute hosts can't resolve here."""
    raw = _tenv("TERMINAL_DOCKER_VOLUMES", "").strip()
    try:
        import json as _json
        parsed = _json.loads(raw) if raw else []
    except Exception:
        return []
    mounts: List[Tuple[Path, Path]] = []
    if not isinstance(parsed, list):
        return mounts
    for entry in parsed:
        spec = entry.strip() if isinstance(entry, str) else ""
        # Prefer the first ':/' so absolute container paths are unambiguous.
        sep = spec.find(":/")
        if sep <= 0:
            continue
        container_raw = spec[sep + 1:].split(":", 1)[0]  # starts with /
        # Skip named volumes (no absolute/drive host path).
        host_expanded = os.path.expanduser(spec[:sep])
        if not (host_expanded.startswith("/") or (len(host_expanded) > 1 and host_expanded[1] == ":")):
            continue
        try:
            host_path = Path(host_expanded).resolve(strict=False)
            container_path = Path(container_raw)
        except (OSError, RuntimeError, ValueError):
            continue
        if container_path.is_absolute():
            mounts.append((host_path, container_path))
    return mounts


def _docker_sandbox_dir_candidates(session_key: str = "") -> List[str]:
    """Candidate host sandbox dir names for the delivering session, best first. Mirrors
    ``_resolve_container_task_id`` (tools/terminal_tool.py): containers are PROFILE-scoped
    (``default``, else ``profile:<name>``); legacy ``session:<key>`` sandboxes stay as a fallback.
    The key is passed explicitly because delivery runs after the turn's contextvars were cleared."""
    candidates: List[str] = []
    try:
        from tools.environments.path_utils import sanitize_task_id_for_path
    except Exception:
        return ["default"]
    # Explicit trusted-profiles opt-in: one shared container identity.
    shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip()
    if shared:
        candidates.append(sanitize_task_id_for_path(f"shared:{shared}"))
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name() or "default"
    except Exception:
        profile = "default"
    if profile != "default":
        candidates.append(sanitize_task_id_for_path(f"profile:{profile}"))
    candidates.append("default")
    if session_key:
        # Bug-window legacy layout: per-session sandboxes.
        candidates.append(sanitize_task_id_for_path(f"session:{session_key}"))
    return candidates


_TRUTHY = {"1", "true", "yes", "on"}


def _docker_env_active() -> bool:
    return _tenv("TERMINAL_ENV", "").strip().lower() == "docker"


def _docker_persistent_active() -> bool:
    """Docker backend with persistent containers (the default) enabled."""
    return _docker_env_active() and _tenv("TERMINAL_CONTAINER_PERSISTENT", "true").strip().lower() in _TRUTHY


def _docker_persistent_sandbox_roots(session_key: str, leaf: str) -> List[Path]:
    """Existing ``<sandbox>/docker/<candidate>/<leaf>`` host dirs in candidate order;
    the translator tries each until the file resolves. Empty unless Docker + persistent."""
    if not _docker_persistent_active():
        return []
    try:
        from tools.environments.base import get_sandbox_dir
        base = get_sandbox_dir() / "docker"
        return [
            cand for name in _docker_sandbox_dir_candidates(session_key)
            if (cand := (base / name / leaf).resolve(strict=False)).is_dir()]
    except Exception:
        return []


def _default_docker_workspace_host_roots(session_key: str = "") -> List[Path]:
    """Existing host candidates for ``/workspace``: the explicit cwd mount
    (``TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE``) if set, else the persistent sandbox layouts."""
    if not _docker_persistent_active():
        return []
    if _tenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").strip().lower() in _TRUTHY:
        cwd = _tenv("TERMINAL_CWD") or os.getcwd()
        try:
            host = Path(os.path.expanduser(cwd)).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return []
        return [host] if host.is_dir() else []
    return _docker_persistent_sandbox_roots(session_key, "workspace")


def _cache_dir_container_mounts() -> List[Tuple[Path, Path]]:
    """(host, container) pairs for the auto-mounted Hermes cache dirs (``/root/.hermes/...`` in
    MEDIA tags); longer prefixes than the ``/root`` home mount, so longest-prefix match wins."""
    if not _docker_env_active():
        return []
    try:
        from tools.credential_files import get_cache_directory_mounts
        return [(Path(m["host_path"]), Path(m["container_path"])) for m in get_cache_directory_mounts()]
    except Exception:
        return []


def _warn_unresolved_docker_media(candidate: Path, session_key: str, reason: str) -> None:
    """Name WHY a container-absolute MEDIA path failed translation (otherwise the only signal is
    the generic "Skipping unsafe MEDIA directive path" line). Docker-only; host rejections quiet."""
    if not _docker_env_active():
        return
    logger.warning(
        "Docker MEDIA path %s did not resolve to a host sandbox file (%s%s); "
        "the producing container's sandbox directory may not exist yet or "
        "was pruned",
        _log_safe_path(str(candidate)), reason,
        f", session_key={session_key}" if session_key else "")


def _translate_docker_container_media_path(candidate: Path, session_key: str = "") -> Optional[Path]:
    """Translate a container-absolute path to its host path via longest-prefix match
    over ``docker_volumes``, the auto-mounted cache dirs (``/root/.hermes/...``), the
    persistent ``/workspace`` host root, and the persistent ``/root`` home mount."""
    if not candidate.is_absolute():
        return None
    # In-process gateways (Desktop, `hermes serve`) may not have bridged terminal.*
    # config into TERMINAL_* env; run the idempotent bridge so mount parsing sees it.
    try:
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
    except Exception:
        pass
    mounts = [*_parse_docker_volume_mounts(), *_cache_dir_container_mounts()]
    mounted = {c.as_posix() for _, c in mounts}
    # Synthetic /workspace mounts: profile-scoped layout first, then legacy per-session.
    if "/workspace" not in mounted:
        mounts.extend((root, Path("/workspace")) for root in _default_docker_workspace_host_roots(session_key))
    # Synthetic /root home mounts only catch stray home writes like /root/out.png (cache mounts
    # are longer prefixes). /root/.hermes/* that missed a cache mount is the container's credential
    # surface (.env, auth.json, ...); translating via the home mount would dodge the host denylist.
    if "/root" not in mounted and not candidate.as_posix().startswith("/root/.hermes"):
        mounts.extend(
            (root, Path("/root")) for root in _docker_persistent_sandbox_roots(session_key, "home"))
    if not mounts:
        _warn_unresolved_docker_media(candidate, session_key, "no sandbox mounts resolved")
        return None
    # Longest container-prefix match; equal-length prefixes are tried in insertion order.
    candidate_posix = candidate.as_posix()
    matched: List[Tuple[Path, Path, int]] = [
        (host_root, container_root, len(container_posix))
        for host_root, container_root in mounts
        for container_posix in (container_root.as_posix().rstrip("/") or "/",)
        if candidate_posix == container_posix or candidate_posix.startswith(container_posix + "/")]
    if not matched:
        _warn_unresolved_docker_media(candidate, session_key, "no mounted prefix matches")
        return None
    matched.sort(key=lambda m: -m[2])
    for host_root, container_root, _score in matched:
        try:
            translated = (host_root / candidate.relative_to(container_root)).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if translated == host_root or _path_is_within(translated, host_root):
            return translated
    _warn_unresolved_docker_media(candidate, session_key, "host file missing from sandbox")
    return None


def validate_media_delivery_path(path: str, session_key: str = "") -> Optional[str]:
    """Return a safe absolute file path for native media delivery, else None.

    Default: any existing regular file outside the credential / system denylist (symmetric
    with inbound). Strict (``HERMES_MEDIA_DELIVERY_STRICT=1``, for public bots where one
    user's prompt injection must not exfiltrate host secrets): MUST be under a Hermes cache,
    an operator root (``HERMES_MEDIA_ALLOW_DIRS``), or freshly produced within the recency
    window. Symlinks are resolved before any check."""
    candidate = _normalize_media_tag_path(path)
    if not candidate:
        return None
    try:
        expanded = Path(os.path.expanduser(candidate))
    except (OSError, RuntimeError, ValueError):
        # expanduser raises ValueError("embedded null byte") for a ~\x00 path.
        return None
    if not expanded.is_absolute():
        return None
    # Docker agents emit MEDIA:/workspace/... — map container paths to host paths first.
    resolved = _translate_docker_container_media_path(expanded, session_key=session_key)
    if resolved is None:
        try:
            resolved = expanded.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None
    if not resolved.is_file():
        return None
    # Cache / operator allowlist is trusted unconditionally, regardless of mode.
    for root in _media_delivery_allowed_roots():
        try:
            resolved_root = root.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if _path_is_within(resolved, resolved_root):
            return str(resolved)
    # Non-strict (default): accept anything not denylisted — /etc, /proc, ~/.ssh, ~/.aws and the
    # Hermes-root secret stores stay rejected (MEDIA:/etc/passwd, MEDIA:~/.hermes/google_token.json)
    if os.environ.get(MEDIA_DELIVERY_STRICT_ENV, "0").strip().lower() not in _TRUTHY:
        return None if _path_under_denied_prefix(resolved) else str(resolved)
    # Strict: fall back to recency trust for freshly-produced files (pandoc -o /tmp/x.pdf);
    # system / credential paths stay blocked even when "recent".
    window = _media_delivery_recency_seconds()
    if (window > 0 and not _path_under_denied_prefix(resolved)
            and _file_is_recently_produced(resolved, window)):
        return str(resolved)
    return None


# Neutralise control chars and Unicode line separators (NEL, LS, PS) that splitlines()
# / log aggregators treat as breaks, so a model-emitted path can't forge a log line.
_LOG_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")


def _log_safe_path(path: str) -> str:
    """Return a single-line, length-bounded path for log output."""
    return _LOG_UNSAFE_CHARS.sub("?", str(path))[:200]


def _validated_delivery_path(raw_path, session_key: str, label: str) -> Optional[str]:
    """``validate_media_delivery_path`` plus the shared "Skipping unsafe ..." warning."""
    raw = str(raw_path)
    safe_path = validate_media_delivery_path(raw, session_key=session_key)
    if not safe_path:
        logger.warning("Skipping unsafe %s: %s", label, _log_safe_path(raw))
    return safe_path


SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv",
    ".log": "text/plain", ".json": "application/json", ".xml": "application/xml",
    ".yaml": "application/yaml", ".yml": "application/yaml", ".toml": "application/toml",
    ".ini": "text/plain", ".cfg": "text/plain", ".zip": "application/zip",
    ".doc": "application/msword", ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ts": "text/plain", ".py": "text/plain", ".sh": "text/plain"}

# Files safe to inline into the prompt when small. Deliberately an extension gate, NOT a blind
# UTF-8 decode — PDF/zip/docx can start with decodable ASCII. Non-members are still cached by path.
_TEXT_INJECT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".jsonl", ".ndjson", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs",
    ".java", ".kt", ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".swift", ".m",
    ".scala", ".clj", ".ex", ".exs", ".erl", ".sql", ".graphql", ".proto", ".tf", ".hcl",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".rst", ".tex", ".srt", ".vtt", ".diff",
    ".patch"}

# Image extensions platforms may deliver as "documents" (file-picker uploads, stickers/
# screenshots wrapped as files); routed through the image cache / vision path instead of rejected.
SUPPORTED_IMAGE_DOCUMENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif"}

# Media-delivery extension allowlist — SINGLE SOURCE OF TRUTH for both extractors (MEDIA: tags,
# bare paths) and the cleanup regexes, so a tag is only stripped when its extension is deliverable
# and an unknown-extension path survives in the body. Dispatch partition lives in gateway/run.py.
MEDIA_DELIVERY_EXTS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",  # images (embed inline)
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp",  # video (embed inline where supported)
    ".mp3", ".m2a", ".wav", ".ogg", ".opus", ".m4a", ".flac",  # audio (voice/audio where supported)
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".epub",  # documents (file attachments)
    ".xlsx", ".xls", ".ods", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # spreadsheets/data
    ".kmz", ".kml", ".geojson", ".gpx",  # geospatial / GIS
    ".pptx", ".ppt", ".odp", ".key",  # presentations
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".apk", ".ipa",  # archives
    ".html", ".htm")  # web / rendered output

# Bare extensions (no dot) longest-first so a shorter ext never matches as a prefix of a longer one.
_MEDIA_EXT_ALTERNATION = "|".join(sorted((e.lstrip(".") for e in MEDIA_DELIVERY_EXTS), key=len, reverse=True))

# Anchored ``MEDIA:<path>`` cleanup pattern shared by the non-streaming dispatch path and the
# streaming consumer. Strips only a tag whose path ends in a known deliverable extension
# (optionally quoted/backticked); unknown-extension tags stay for the bare-path detector. Rules:
#   * Path anchors: ``~/``, ``/``, ``X:\`` or ``X:/`` (Windows drive letter).
#   * Up to 3 quote/emphasis markers each side (``**MEDIA:/x.pdf**``); code / inline-code /
#     blockquote contexts are neutralised earlier by ``_mask_protected_spans``.
#   * Non-greedy path forms with ``MEDIA:`` as a boundary, so glued tags
#     (``MEDIA:/a.pngMEDIA:/b.png``) or trailing prose never merge into one invalid path.
#   * Sentence-final ``.`` is a boundary only before whitespace/EOL (``\.(?=\s|$)``): ``data.csv.``
#     yields ``data.csv`` while ``archive.tar.gz`` extends past ``.tar``.
#   * CJK full-width punctuation terminates paths too (``MEDIA:D:\path\早报.pdf（782.6 KB）``).
_MEDIA_CJK_TERMINATORS = "（）〈〉《》：，。；！？、\u201c\u201d\u2018\u2019【】"

MEDIA_TAG_CLEANUP_RE = re.compile(
    r'''[`"'*_]{0,3}MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+?`|"[^"\n]+?"|'[^'\n]+?'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])\S+?(?:[^\S\n]+\S+?)*?\.(?:''' + _MEDIA_EXT_ALTERNATION + r'''))'''
    r'''(?=[\s`"'*_,;:)\]}\[''' + _MEDIA_CJK_TERMINATORS + r''']|MEDIA:|\.(?:\s|$)|$)[`"'*_]{0,3}\.?''',
    re.IGNORECASE)

# Extension-less (Caddyfile) or unknown-extension (.py, .log) paths are delivered via this
# pattern only after ``validate_media_delivery_path`` accepts them (on disk, not denylisted,
# strict-mode rules honored), so prompt-injection paths that don't validate stay visible. The
# bare path class is a whitespace-bounded tempered-greedy token: a tag glued to the next
# ``MEDIA:`` or to prose must not absorb it. Spaced paths (``MEDIA:/data/map data.kmz``) are
# recovered by ``_match_extensionless_path`` with on-disk validation as the oracle.
MEDIA_EXTENSIONLESS_TAG_RE = re.compile(
    r'''[`"'*_]{0,3}MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])[^\s\n`"']+?)'''
    r'''(?=[`"'\s,;:)\]}''' + _MEDIA_CJK_TERMINATORS + r''']|MEDIA:|$)'''
    r'''[`"'*_]{0,3}\s*''',
    re.IGNORECASE)


def _match_extensionless_path(scan_text: str, match: "re.Match") -> Optional[Tuple[str, int]]:
    """Resolve an extensionless MEDIA tag match to a validated on-disk path: the captured
    path first, then extended forward across single spaces (max 8 tokens, never past a
    newline or the next ``MEDIA:``). Returns ``(safe_path, end_offset)`` or ``None``."""
    path = _normalize_media_tag_path(match.group("path"))
    if not path:
        return None
    safe = validate_media_delivery_path(path)
    if safe:
        return safe, match.end("path")
    start = match.start("path")
    segment = scan_text[start:].split("\n", 1)[0]
    nxt = segment.find("MEDIA:", 1)
    if nxt != -1:
        segment = segment[:nxt]
    pos = match.end("path") - start
    for _ in range(8):
        token = re.match(r"[ \t]*[^ \t]+", segment[pos:])
        if not token:
            break
        tok_end = pos + token.end()
        safe = validate_media_delivery_path(_normalize_media_tag_path(segment[:tok_end]))
        if safe:
            return safe, start + tok_end
        pos = tok_end
    return None


def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _path_lacks_deliverable_extension(path: str) -> bool:
    """True when ``path`` has no extension or one outside MEDIA_DELIVERY_EXTS — such paths
    take the validated delivery pass so nonexistent / denylisted ones stay visible."""
    return Path(path).suffix.lower() not in MEDIA_DELIVERY_EXTS


def _has_media_directives(text: str) -> bool:
    return "MEDIA:" in text or "[[audio_as_voice]]" in text or "[[as_document]]" in text


def _mask_media_scan_text(text: str) -> str:
    """Offset-preserving mask of protected spans (code, quotes, JSON string values).
    BasePlatformAdapter is defined later in this module; resolved at call time."""
    return BasePlatformAdapter._mask_json_string_media(
        BasePlatformAdapter._mask_protected_spans(text))


def _real_media_tag_spans(masked: str) -> list:
    """(start, end) spans of deliverable MEDIA tags on a masked copy: known-extension tags
    unconditionally, extension-less / unknown ones only if validate_media_delivery_path accepts."""
    spans: list = [m.span() for m in MEDIA_TAG_CLEANUP_RE.finditer(masked)]
    for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(masked):
        path = _normalize_media_tag_path(match.group("path"))
        if not path or not _path_lacks_deliverable_extension(path):
            continue
        resolved = _match_extensionless_path(masked, match)
        if resolved is not None:
            spans.append((match.start(), resolved[1]))
    return spans


_FENCED_CODE_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`[^`\n]+`')


def _code_spans(content: str) -> list:
    """(start, end) spans of fenced code blocks and inline code in ``content``."""
    return [m.span() for m in _FENCED_CODE_RE.finditer(content)] + [m.span() for m in _INLINE_CODE_RE.finditer(content)]


def _blank_spans(text: str, spans: list) -> str:
    """Replace every non-newline char inside ``spans`` with a space (offsets preserved)."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = [c if c == '\n' else ' ' for c in chars[start:end]]
    return ''.join(chars)


def _delete_spans(text: str, spans: list) -> str:
    """Delete ``spans`` from ``text``, merging overlapping/nested ones first so multi-pattern
    matches over the same tag never double-delete adjacent text."""
    if not spans:
        return text
    merged: list = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    chars = list(text)
    for start, end in reversed(merged):
        del chars[start:end]
    return "".join(chars)


def _strip_media_tag_directives(text: str) -> str:
    """Remove MEDIA: tags and [[audio_as_voice]] / [[as_document]] markers. Protected spans
    are mask-located only — tags inside them are neither stripped nor mangled, matching
    ``extract_media`` so display and delivery agree."""
    if not _has_media_directives(text):
        return text
    cleaned = text.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
    return _delete_spans(cleaned, _real_media_tag_spans(_mask_media_scan_text(cleaned)))


def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """Save raw document bytes to the cache as ``doc_{uuid12}_{original_name}`` and return
    the absolute path; raises ValueError if the sanitized path escapes the cache directory."""
    cache_dir = get_document_cache_dir()
    # Sanitize: strip directory components, null bytes, and control characters
    safe_name = (Path(filename).name if filename else "document").replace("\x00", "").strip()
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "document"
    filepath = cache_dir / f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    # Final safety check: ensure path stays inside cache dir
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    filepath.write_bytes(data)
    return str(filepath)


# Unified media caching: classify attachment bytes by ext/MIME, route to cache_*_from_bytes.
@dataclass
class CachedMedia:
    """Result of caching one attachment's bytes."""
    path: str                 # absolute cache path, agent-visible (sandbox-translated)
    media_type: str           # MIME type recorded on the MessageEvent
    kind: str                 # "image" | "video" | "audio" | "document"
    display_name: str         # human-readable name for transcript notes

    def context_note(self) -> str:
        """One-line transcript annotation pointing the agent at the file."""
        return f"[{self.kind} '{self.display_name}' saved at: {self.path}]"


# MIME -> extension reverse lookup; first match across image, video, document tables wins.
_MIME_TO_EXT: Dict[str, str] = {}
for _table in (SUPPORTED_IMAGE_DOCUMENT_TYPES, SUPPORTED_VIDEO_TYPES, SUPPORTED_DOCUMENT_TYPES):
    for _ext, _mime in _table.items():
        _MIME_TO_EXT.setdefault(_mime, _ext)
del _table, _ext, _mime


def _resolve_media_ext(filename: str, mime_type: str) -> str:
    """Best-effort file extension from filename, then MIME fallback."""
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    return ext or _MIME_TO_EXT.get((mime_type or "").lower(), "")


def cache_media_bytes(
    data: bytes, *, filename: str = "", mime_type: str = "", default_kind: Optional[str] = None,
) -> Optional[CachedMedia]:
    """Classify and cache raw attachment bytes; return a CachedMedia or None. ``default_kind``
    biases classification when extension/MIME are ambiguous (Telegram native photo, no name).
    Anything not image/video/audio is a document; only images failing validation return None."""
    from tools.credential_files import to_agent_visible_cache_path
    ext = _resolve_media_ext(filename, mime_type)
    mime = (mime_type or "").lower()
    display = re.sub(r"[^\w.\- ]", "_", filename) if filename else (ext.lstrip(".") or "file")
    # (kind, ext->mime table, default ext, cache fn, whether a matching caller MIME passes through)
    for kind, table, default_ext, cache_fn, passthrough in (
            ("image", SUPPORTED_IMAGE_DOCUMENT_TYPES, ".jpg", cache_image_from_bytes, True),
            ("video", SUPPORTED_VIDEO_TYPES, ".mp4", cache_video_from_bytes, False),
            ("audio", _AUDIO_MIME_TYPES, ".ogg", cache_audio_from_bytes, True)):
        if not (mime.startswith(f"{kind}/") or ext in table or default_kind == kind):
            continue
        kind_ext = ext if ext in table else default_ext
        try:
            path = cache_fn(data, ext=kind_ext)
        except ValueError:
            if kind != "image":
                raise
            return None
        out_mime = mime if passthrough and mime.startswith(f"{kind}/") else table[kind_ext]
        return CachedMedia(to_agent_visible_cache_path(path), out_mime, kind, display)
    # Any other file type is cached and surfaced as a local path: once a user is authorized
    # to message the agent, the extension allowlist must not silently drop their uploads.
    # Unknown types get application/octet-stream (or the caller's MIME) so the agent knows
    # to reach for terminal tools.
    fallback_name = filename or (f"document{ext}" if ext else "document.bin")
    path = cache_document_from_bytes(data, fallback_name)
    out_mime = SUPPORTED_DOCUMENT_TYPES.get(ext) or mime or "application/octet-stream"
    return CachedMedia(to_agent_visible_cache_path(path), out_mime, "document", display or fallback_name)


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class MessageEvent:
    """Incoming message from a platform — the normalized shape all adapters produce."""
    text: str
    message_type: MessageType = MessageType.TEXT
    # Author, mirrored from ``source`` for per-message prompt builders; None for non-IM sources.
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    source: SessionSource = None
    raw_message: Any = None
    message_id: Optional[str] = None
    # Platform update id (Telegram ``update_id``); ``/restart`` records it so the new gateway
    # advances the offset past it and never re-processes it if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: List[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay): {prompt_id, option_id, label?,
    # prompt_message_id?}. RelayAdapter routes it to the approval / slash-confirm / clarify
    # resolvers BEFORE normal dispatch; native adapters never set it (buttons resolve in-process).
    prompt_response: Optional[Dict[str, Any]] = None
    # Auto-loaded skill(s) for topic/channel bindings; a single name or ordered list.
    auto_skill: Optional[str | list[str]] = None
    # Per-channel ephemeral system prompt; applied at API call time, never persisted to transcript.
    channel_prompt: Optional[str] = None
    # Channel context recovered by history backfill (messages missed under require_mention);
    # separate from ``text`` so run.py's sender-prefix logic sees only the trigger message.
    channel_context: Optional[str] = None
    # Set for synthetic events (e.g. background-process notifications) that must bypass user authorization.
    internal: bool = False
    # Free-form per-event metadata (e.g. ``whatsapp_from_owner=True``); plugins must ``.get()``.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / pending control prompts? Kept last for positional
    # compat. Proactive plugin events set False so untrusted payload text stays conversational.
    allow_gateway_control: bool = True

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and (self.text or "").lstrip().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = (self.text or "").lstrip().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = (self.text or "").lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")


@dataclass
class TextDebounceState:
    event: MessageEvent
    task: asyncio.Task | None
    first_ts: float
    last_ts: float

    def cancel_timer(self, *, unless: "asyncio.Task | None" = None) -> None:
        """Cancel the pending flush timer (if live and not ``unless``)."""
        if self.task is not None and self.task is not unless and not self.task.done():
            self.task.cancel()


def _append_text(existing: Optional[str], new: Optional[str]) -> str:
    """``existing\\nnew`` when both non-empty; the non-empty one otherwise."""
    return f"{existing}\n{new}" if existing else new


@dataclass
class _ExtractedResponse:
    """Deliverable parts of a handler response (see ``_extract_response_content``)."""
    text_content: str
    images: list
    media_files: list
    local_files: list
    force_document_attachments: bool
    pre_extract: str


_PLAINTEXT_GATEWAY_RESTART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+hermes[.!?\s]*$", re.IGNORECASE))


def coerce_plaintext_gateway_command(event: "MessageEvent") -> None:
    """Rewrite a tiny set of DM plaintext admin phrases into slash commands, keeping ``restart
    gateway`` out of the LLM/tool path (a self-restart from inside the running agent leaves the
    gateway stuck in ``draining`` waiting on that agent). Narrow: DM text, exact phrases only."""
    try:
        if event is None or event.message_type != MessageType.TEXT:
            return
        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return
        if getattr(getattr(event, "source", None), "chat_type", None) != "dm":
            return
        if any(pattern.match(text) for pattern in _PLAINTEXT_GATEWAY_RESTART_PATTERNS):
            event.text = "/restart"
    except Exception:
        return


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    # Adapter-specific metadata. Cross-layer contract: Telegram edit-overflow partials set
    # raw_response["partial_overflow"] (delivered_chunks, total_chunks, last_message_id,
    # delivered_prefix, continuation_message_ids) so the stream consumer sends the missing tail.
    raw_response: Any = None
    retryable: bool = False  # transient connection error — base retries automatically
    retry_after: Optional[float] = None  # server-requested delay (Telegram FloodWait) beats our backoff
    # Extra ids (send order) when an oversized payload was split across platform messages;
    # ``message_id`` is then the LAST visible id so later edits target the newest chunk.
    continuation_message_ids: tuple = ()
    # SEND_ERROR_KINDS member (only when ``success`` is False) so consumers branch without
    # substring-matching ``error``; producers set it via :func:`classify_send_error`.
    error_kind: Optional[str] = None


# Platform-neutral send-failure categories for ``SendResult.error_kind``, so the gateway
# decides once whether a failure is worth surfacing: too_long (size cap; adapter usually
# splits), bad_format (markup/entities rejected; plain-text retry fixes), forbidden (blocked/
# kicked/no permission — the bot CANNOT reach the user), not_found (chat/thread/message gone),
# rate_limited (flood control), transient (connection-level, safe to retry), unknown.
SEND_ERROR_KINDS = frozenset(
    {"too_long", "bad_format", "forbidden", "not_found", "rate_limited", "transient", "unknown"})

# ``not_found`` substrings split by blast radius: chat-level = whole target dead; thread/topic/
# message-level leaves the parent chat reachable and must NOT mark it dead.
_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS = ("chat not found",)
_SUBCHAT_NOT_FOUND_SUBSTRINGS = (
    "message to edit not found", "message to reply not found", "thread not found", "topic_deleted",
    "message_id_invalid")


def _error_blob(exc: Optional[BaseException] = None, error_text: str = "") -> str:
    """Lowercased blob (error_text + str(exc) + exception class name) that both
    send-error classifiers match against — one builder so they can never drift."""
    parts = [error_text] if error_text else []
    if exc is not None:
        parts.extend(p for p in (str(exc), exc.__class__.__name__) if p)
    return " ".join(parts).lower()


def _any_in(blob: str, *needles: str) -> bool:
    return any(n in blob for n in needles)


# Ordered (kind, predicate) table for classify_send_error — first match wins.
_SEND_ERROR_CLASSIFIERS: Tuple[Tuple[str, Callable[[str], bool]], ...] = (
    ("too_long", lambda b: _any_in(b, "message_too_long", "too long", "message is too long")),
    ("bad_format", lambda b: (
        _any_in(b, "can't parse entities", "cant parse entities", "can't find end", "unsupported start tag")
        or ("entity" in b and "parse" in b)
        or ("bad request" in b and "entit" in b))),
    ("forbidden", lambda b: _any_in(
        b, "forbidden", "bot was blocked", "blocked by the user", "user is deactivated",
        "not enough rights", "have no rights", "not a member")),
    ("not_found", lambda b: _any_in(b, *_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS, *_SUBCHAT_NOT_FOUND_SUBSTRINGS)),
    ("rate_limited", lambda b: _any_in(b, "flood", "too many requests", "retry after", "rate limit")),
    ("transient", lambda b: _any_in(b, *_RETRYABLE_ERROR_PATTERNS, "connecttimeout")))


def classify_send_error(exc: Optional[BaseException], error_text: str = "") -> str:
    """Map a send exception / error string to a :data:`SEND_ERROR_KINDS` value.
    Conservative substring matching: anything unrecognized is ``"unknown"`` so an
    unclassified failure is never mistaken for a benign one."""
    blob = _error_blob(exc, error_text)
    return next((kind for kind, matches in _SEND_ERROR_CLASSIFIERS if matches(blob)), "unknown")


def is_chat_level_not_found(exc: Optional[BaseException] = None, error_text: str = "") -> bool:
    """Whether a ``not_found`` failure means the *whole chat* is gone — only that should mark
    a delivery target dead; a deleted forum topic or edited-away message leaves the parent
    chat reachable. When both markers are present the sub-chat reading wins."""
    blob = _error_blob(exc, error_text)
    return (not _any_in(blob, *_SUBCHAT_NOT_FOUND_SUBSTRINGS)
            and _any_in(blob, *_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS))


class EphemeralReply(str):
    """System-notice reply that auto-deletes after ``ttl_seconds`` on platforms implementing
    ``delete_message`` (others leave it). ``None`` ttl uses ``display.ephemeral_system_ttl``
    (``0`` disables globally). Subclassing ``str`` keeps it transparent to everything that
    treats handler results as text; ``isinstance`` lets the send path schedule deletion."""
    ttl_seconds: Optional[int]

    def __new__(cls, text: str, ttl_seconds: Optional[int] = None):
        instance = super().__new__(cls, text)
        instance.ttl_seconds = ttl_seconds
        return instance

    @property
    def text(self) -> str:
        """The underlying text (explicit form of ``str(reply)``)."""
        return str.__str__(self)


def merge_pending_message_event(
    pending_messages: Dict[str, MessageEvent], session_key: str, event: MessageEvent, *,
    merge_text: bool = False) -> None:
    """Store or merge a pending event for a session: photo bursts/albums (several near-simultaneous
    PHOTO events) merge into the queued event so the next turn sees the whole burst; with
    ``merge_text``, rapid follow-up TEXT events are appended instead of replacing the turn."""
    existing = pending_messages.get(session_key)
    if existing:
        existing_is_photo = getattr(existing, "message_type", None) == MessageType.PHOTO
        incoming_is_photo = event.message_type == MessageType.PHOTO
        both_photo = existing_is_photo and incoming_is_photo
        incoming_has_media = bool(event.media_urls)

        def _padded_inline_flags(msg: MessageEvent) -> List[Optional[bool]]:
            flags = list(getattr(msg, "media_text_inlined", []) or [])
            return flags + [None] * (len(msg.media_urls) - len(flags))
        incoming_inline_flags: List[Optional[bool]] = []
        if incoming_has_media:
            existing.media_text_inlined = _padded_inline_flags(existing)
            incoming_inline_flags = _padded_inline_flags(event)
        # A photo burst always absorbs (even media-less PHOTO events); otherwise merge only
        # when media is involved on either side. Captions merge in every absorbing case.
        if both_photo or existing.media_urls or incoming_has_media:
            if both_photo or incoming_has_media:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
                existing.media_text_inlined.extend(incoming_inline_flags)
            if event.text:
                existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
            if existing_is_photo or incoming_is_photo:
                existing.message_type = MessageType.PHOTO
            elif (
                getattr(existing, "message_type", None) == MessageType.TEXT
                and event.message_type != MessageType.TEXT):
                existing.message_type = event.message_type
            # Drop the *derived* STT transcript cache now that the event changed; the echo
            # ledger (``_gateway_pending_stt_echoed``) must survive or notes echo twice.
            for attr in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts"):
                if hasattr(existing, attr):
                    delattr(existing, attr)
            return
        if (
            merge_text
            and getattr(existing, "message_type", None) == MessageType.TEXT
            and event.message_type == MessageType.TEXT):
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            return
    pending_messages[session_key] = event


# Substrings marking a transient *connection* failure worth retrying. Plain "timeout" /
# "readtimeout" / "writetimeout" are excluded on purpose: a read/write timeout on a
# non-idempotent send may have reached the server, so a retry risks duplicate delivery;
# "connecttimeout" never connected. Platforms that know better set SendResult.retryable.
_RETRYABLE_ERROR_PATTERNS = (
    "connecterror", "connectionerror", "connectionreset", "connectionrefused", "connecttimeout",
    "network", "broken pipe", "remotedisconnected", "eoferror")

# Handlers return a plain string (normal reply), an ``EphemeralReply`` (auto-deletion), or
# ``None`` when the response was already delivered (e.g. via streaming).
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[Union[str, "EphemeralReply"]]]]


def resolve_channel_prompt(config_extra: dict, channel_id: str, parent_id: str | None = None) -> str | None:
    """Resolve a per-channel ephemeral prompt from ``config.extra["channel_prompts"]``:
    exact *channel_id* first, then *parent_id* (forum threads / child channels inherit
    the parent prompt). Blank prompts count as absent."""
    prompts = config_extra.get("channel_prompts") or {}
    if not isinstance(prompts, dict):
        return None
    for key in (channel_id, parent_id):
        prompt = prompts.get(key) if key else None
        if prompt is not None and (prompt := str(prompt).strip()):
            return prompt
    return None


def resolve_channel_skills(
    config_extra: dict, channel_id: str, parent_id: str | None = None) -> list[str] | None:
    """Resolve auto-loaded skill(s) for a channel/thread from ``channel_skill_bindings``
    (a list of ``{id: "<channel/forum id>", skills: [...]}`` entries; ``skill: "<name>"``
    single string also accepted). Exact *channel_id* match first, then *parent_id*
    (threads inherit the parent channel's binding). Returns a deduplicated,
    order-preserving list or None."""
    bindings = config_extra.get("channel_skill_bindings") or []
    if not isinstance(bindings, list) or not bindings:
        return None
    ids_to_check = {str(key) for key in (channel_id, parent_id) if key}
    if not ids_to_check:
        return None
    for entry in bindings:
        if not isinstance(entry, dict) or str(entry.get("id", "")) not in ids_to_check:
            continue
        skills = entry.get("skills") or entry.get("skill")
        if isinstance(skills, str):
            return [skills.strip()] if skills.strip() else None
        if isinstance(skills, list) and skills:
            seen = dict.fromkeys(
                nm for name in skills if isinstance(name, str) and (nm := name.strip()))
            return list(seen) or None
    return None


def _split_post_delivery_entry(entry: Any) -> Tuple[Optional[int], Any]:
    """``(generation, callback)`` from a post-delivery slot; legacy bare callbacks have no generation."""
    return entry if isinstance(entry, tuple) and len(entry) == 2 else (None, entry)


def _lazy_attr(obj: Any, name: str, factory: Callable[[], Any]) -> Any:
    """``getattr(obj, name)`` or create it via ``factory`` — the getattr-guard for
    tests that build adapters via ``object.__new__`` and never run ``__init__``."""
    value = getattr(obj, name, None)
    if value is None:
        value = factory()
        setattr(obj, name, value)
    return value


def _strip_media_directives(text: str) -> str:
    """Backstop strip of delivery directives ([[audio_as_voice]], [[as_document]],
    MEDIA:<path>) so they never render as text; run ``extract_media`` first."""
    return _strip_media_tag_directives(text) if text else text


class BasePlatformAdapter(ABC):
    """Base class for platform adapters: connect/auth, receive, send, handle media."""

    # ``format_message`` renders triple-backtick fences as real code blocks (tool-progress
    # renders a terminal command as a bare fenced block; plain-text platforms get the preview).
    supports_code_blocks: bool = False

    # Typing indicator renders TEXT (status line by the bot name) rather than a textless
    # bubble; when True the gateway feeds per-tool phrases via set_status_text().
    supports_status_text: bool = False

    def set_status_text(self, chat_id: str, text: Optional[str]) -> None:
        """Set or clear (``None``) the live working-state phrase for a chat.

        In-memory only: the next typing refresh renders it; a no-op store on adapters
        that never read ``_status_text``.
        """
        store = _lazy_attr(self, "_status_text", dict)
        if text:
            store[str(chat_id)] = text
        else:
            store.pop(str(chat_id), None)

    # Can wake a fresh turn AFTER a turn ends (background/detached-subagent completions).
    # False for stateless request/response adapters (API server); propagated to
    # ``HERMES_SESSION_ASYNC_DELIVERY`` so tools never promise a delivery they can't keep.
    supports_async_delivery: bool = True

    # ``send()`` chunks long content natively via ``truncate_message()``; when True the
    # delivery router skips gateway-level truncation. Set True only when verified.
    splits_long_messages: bool = False

    # Prefix users can always TYPE to reach Hermes commands ("!" where the client
    # intercepts a leading "/": Slack threads, Matrix) so instruction text names the working form.
    typed_command_prefix: str = "/"

    # ``in_channel`` continuable-cron surface: job delivered FLAT into a channel, plain replies
    # continue it via the whole-channel bucket ``(platform, chat_id, None)``; needs a flat-reply
    # outbound gate too (Slack ``reply_in_thread: false``). False fails SAFE -> ``thread``.
    supports_inchannel_continuable: bool = False

    # A human is present to answer "session restored — what next?"; non-interactive event
    # platforms (webhook) set False so auto-resume finishes the work instead of asking nobody.
    interactive_resume: bool = True

    # Back-reference to the running ``GatewayRunner`` (injected by gateway/run.py); on the base
    # so ``build_source`` resolves the inbound profile via ``runner._profile_name_for_source``.
    gateway_runner = None  # type: ignore[assignment]  # set by gateway/run.py

    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
        # Gateway fan-out for platform-native reaction events (set_reaction_handler).
        self._reaction_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        # Runner-owned boundary for normalized events (+ internal SessionSource):
        # authorization/profile state never lives in an SDK adapter.
        self._platform_event_handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]] = None
        # Rewrites ``event.source.thread_id`` before session keying (Telegram DM topics).
        self._topic_recovery_fn: Optional[Callable[[Any], Optional[str]]] = None
        self._running, self._fatal_error_retryable = False, True
        self._fatal_error_code: Optional[str] = None
        self._fatal_error_message: Optional[str] = None
        self._fatal_error_handler: Optional[Callable[["BasePlatformAdapter"], Awaitable[None] | None]] = None
        # Strong refs to shielded fatal-error handler tasks (asyncio keeps only weak refs).
        self._detached_fatal_tasks: set = set()
        # Cross-HERMES_HOME lock takeover, armed by GatewayRunner only for the initial
        # connect of an explicit ``gateway run --replace``; reconnects fail safe.
        self._platform_lock_takeover_allowed = self._platform_lock_takeover_attempted = False
        # Per-session interrupt Event + owner Task so /stop, /new, /reset cancel the right
        # task; without the owner map an old task's finally could drop a newer guard.
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        self._session_tasks: Dict[str, asyncio.Task] = {}
        # Legacy env knob; the runner syncs the busy_input_mode-derived value after
        # construction. Default "interrupt" so a pre-sync read never silently queues.
        self._busy_text_mode: str = (
            os.environ.get("HERMES_GATEWAY_BUSY_TEXT_MODE", "interrupt").strip().lower() or "interrupt")
        self._busy_text_debounce_seconds: float = _float_env("HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS", 0.35)
        self._busy_text_hard_cap_seconds: float = _float_env("HERMES_GATEWAY_BUSY_TEXT_HARD_CAP_SECONDS", 1.0)
        self._text_debounce: dict[str, TextDebounceState] = {}
        # handle_message() tasks; shutdown cancels them so a replaced gateway stops working.
        self._background_tasks: set[asyncio.Task] = set()
        # Post-delivery one-shots keyed by session_key: bare callback (legacy) or
        # ``(generation, callback)`` so a stale run can't clear a fresher run's callback.
        self._post_delivery_callbacks: Dict[str, Any] = {}
        self._expected_cancelled_tasks: set[asyncio.Task] = set()
        self._busy_session_handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]] = None
        # Owning multiplex profile (None on primary/single-profile); ``source.profile`` is
        # stamped only after adapter ingress — see _session_key_profile.
        self._owner_profile: Optional[str] = None
        # Registered by GatewayRunner; adapters that fetch external context (Slack thread
        # history) mark non-allowlisted senders unverified (prompt-injection mitigation).
        self._authorization_check: Optional[Callable[[str, Optional[str], Optional[str]], bool]] = None
        # Auto-TTS on voice input: global default (``voice.auto_tts``) plus per-chat
        # opt-in (``/voice on|tts``, fires even if default False) / opt-out (``/voice off``).
        self._auto_tts_default: bool = False
        self._auto_tts_enabled_chats, self._auto_tts_disabled_chats = set(), set()
        # Turn keys where streaming TTS already delivered audio; whole-file auto-TTS skips them.
        self._streaming_tts_completed_turns: set[str] = set()
        # Chats whose typing indicator is paused (approval waits); _keep_typing skips them.
        self._typing_paused: set = set()
        # Per-chat working-state phrase read by text-rendering typing indicators (Slack);
        # the regular _keep_typing refresh picks it up, so updates cost no extra API calls.
        self._status_text: Dict[str, str] = {}

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        """Length function for message size; override where the platform counts
        differently from ``len`` (Telegram: UTF-16 code units)."""
        return len

    def max_message_length_for_chat(self, chat_id: str) -> int:
        """Per-chat max length in ``message_len_fn_for_chat`` units. Default: the adapter
        scalar ``MAX_MESSAGE_LENGTH`` (4096 when absent); the relay adapter overrides
        because one adapter fronts N platforms with different caps."""
        try:
            return int(getattr(self, "MAX_MESSAGE_LENGTH", 4096) or 4096)
        except (TypeError, ValueError):
            return 4096

    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        """Per-chat length function (companion to max_message_length_for_chat); the relay
        adapter overrides it per the chat's fronting platform."""
        return self.message_len_fn

    @property
    def enforces_own_access_policy(self) -> bool:
        """Whether this adapter enforces its own config-driven access policy at intake
        (``dm_policy``/``group_policy``/``allow_from``: WeCom, Weixin, QQBot, WhatsApp…).

        The gateway env allowlist runs *after* the adapter; with no env allowlist it trusts
        this flag ONLY when the effective policy is a real ``"allowlist"`` — never ``"open"``
        (the default), which would be a network-exposed fail-open (SECURITY.md §2.6). Open
        access still requires ``{PLATFORM}_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOW_ALL_USERS``.
        """
        return False

    @property
    def authorization_is_upstream(self) -> bool:
        """Whether inbound was already authorized by a TRUSTED UPSTREAM (relay only).

        The Team Gateway connector authenticates the WebSocket and resolves owner-only author
        binding BEFORE delivery, so the no-allowlist default-deny would be wrong. This is
        authorization DELEGATED, not ABSENT — every network-exposed direct adapter leaves it False.
        """
        return False

    def supports_draft_streaming(
        self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None) -> bool:
        """Whether native streaming-draft updates (``send_draft``) work for this chat type;
        ``chat_id`` lets the relay adapter answer per negotiated capabilities. Consumers
        fall back to ``send`` + ``edit_message`` when False or ``send_draft`` raises."""
        return False

    def prefers_fresh_final_streaming(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Whether the stream consumer should finalize with a *fresh* final message (best-effort
        deleting the preview) instead of final-editing it (Telegram: keeps rich rendering)."""
        return False

    def streaming_overflow_limit(self) -> Optional[int]:
        """Max single-message length (``message_len_fn`` units) the stream consumer may
        accumulate before splitting, for rich send/draft paths exceeding the legacy cap
        (Telegram Rich Messages: 32,768 vs 4,096). ``None`` = use ``MAX_MESSAGE_LENGTH``."""
        return None

    async def send_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send or update an animated streaming-draft preview.

        Reuse one non-zero ``draft_id`` across a single response so the platform animates;
        different responses in the same chat need different ids. Drafts have no message_id
        (no edit/reply/delete) — the final answer goes out as a regular ``send``. Must be
        overridden by adapters returning True from :meth:`supports_draft_streaming`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement send_draft")

    # ── Structured stream-event rendering (gateway/stream_events.py): presentation-only,
    # nothing rendered here is persisted, so what an adapter "eats" never changes history.

    def render_message_event(self, event: Any, sink: Any) -> None:
        """Render a MessageChunk / MessageStop / Commentary onto the sink (a
        GatewayStreamConsumer), mapping 1:1 onto its existing primitives."""
        from gateway.stream_events import MessageChunk, MessageStop, Commentary
        if isinstance(event, MessageChunk) and event.text:
            sink.on_delta(event.text)
        elif isinstance(event, MessageStop) and not event.final:
            # Intermediate stop (text → tool → text) = segment break; the terminal
            # stop is signalled by the gateway via finish(), not here.
            sink.on_segment_break()
        elif isinstance(event, Commentary) and event.text:
            sink.on_commentary(event.text)

    def format_tool_event(self, event: Any, *, mode: str = "all", preview_max_len: int = 40) -> Optional[str]:
        """Rendered chrome for a ToolCallChunk, or None to eat it (adapters without editing/
        rich text override to None). ``mode`` is the tool-progress mode ("all"/"new"/
        "verbose"); ``preview_max_len`` mirrors ``tool_preview_length`` (0 = no cap in verbose)."""
        from gateway.stream_events import ToolCallChunk
        if not isinstance(event, ToolCallChunk):
            return None
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(event.tool_name, default="⚙️")
        if mode == "verbose" and event.args:
            import json
            args_str = json.dumps(event.args, ensure_ascii=False, default=str)
            if preview_max_len > 0 and len(args_str) > preview_max_len:
                args_str = args_str[:preview_max_len - 3] + "..."
            return f"{emoji} {event.tool_name}({list(event.args.keys())})\n{args_str}"
        if event.preview:
            if mode == "verbose":
                return f"{emoji} {event.tool_name}: \"{event.preview}\""
            # "all" / "new": short capped preview (default 40; progress bubbles persist as messages).
            from agent.display import prepare_tool_preview
            cap = preview_max_len if preview_max_len > 0 else 40
            prepared = prepare_tool_preview(
                event.tool_name, event.args, fallback=event.preview, max_len=cap)
            return f"{emoji} {event.tool_name}: \"{self.format_tool_preview(prepared)}\""
        return f"{emoji} {event.tool_name}..."

    def format_tool_preview(self, preview: "ToolPreview") -> str:
        """Platform-native formatting of a compact tool preview; rich-text adapters may use
        the preview's metadata (e.g. a URL shortened for display)."""
        return preview.text

    @property
    def has_fatal_error(self) -> bool:
        return self._fatal_error_message is not None

    @property
    def fatal_error_message(self) -> Optional[str]:
        return self._fatal_error_message

    @property
    def fatal_error_code(self) -> Optional[str]:
        return self._fatal_error_code

    @property
    def fatal_error_retryable(self) -> bool:
        return self._fatal_error_retryable

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        """Whether auto-TTS fires for ``chat_id``: explicit ``/voice on|tts`` wins,
        then explicit ``/voice off``, then the global ``voice.auto_tts`` default."""
        if chat_id in self._auto_tts_enabled_chats:
            return True
        return chat_id not in self._auto_tts_disabled_chats and bool(self._auto_tts_default)

    def set_fatal_error_handler(self, handler: Callable[["BasePlatformAdapter"], Awaitable[None] | None]) -> None:
        self._fatal_error_handler = handler

    def _mark_connected(self) -> None:
        self._running = True
        self._fatal_error_code = self._fatal_error_message = None
        self._fatal_error_retryable = True
        self._write_runtime_status_safe("connected", platform_state="connected", error_code=None, error_message=None)

    def _mark_disconnected(self) -> None:
        self._running = False
        if not self.has_fatal_error:
            self._write_runtime_status_safe(
                "disconnected", platform_state="disconnected", error_code=None, error_message=None)

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._running = False
        self._fatal_error_code, self._fatal_error_message = code, message
        self._fatal_error_retryable = retryable
        self._write_runtime_status_safe("fatal", platform_state="fatal", error_code=code, error_message=message)

    def _write_runtime_status_safe(self, context: str, **kwargs) -> None:
        """Write runtime status; log first failure per context at warning, rest at debug
        (failures — permissions, ENOSPC — must neither be silent nor spam reconnect loops)."""
        try:
            from gateway.status import write_runtime_status
            # Multiplexed secondary adapters share the runtime status file; their
            # runner stamps a ``<profile>:<platform>`` key so profiles don't clobber.
            platform_key = getattr(self, "_runtime_status_platform_key", None) or self.platform.value
            write_runtime_status(platform=platform_key, **kwargs)
        except Exception as exc:
            logged = _lazy_attr(self, "_status_write_logged", set)  # object.__new__ in tests
            key = (self.platform.value, context)
            first = key not in logged
            logged.add(key)
            (logger.warning if first else logger.debug)(
                "Failed to write runtime status (%s) for %s: %s" + (" (further failures at debug level)" if first else ""),
                context, self.platform.value, exc)

    async def _notify_fatal_error(self) -> None:
        handler = self._fatal_error_handler
        if not handler:
            return
        result = handler(self)
        if asyncio.iscoroutine(result):
            # Detached + shielded: often awaited from an adapter-owned task that the
            # handler's ``disconnect()`` cancels; unshielded, the handler died mid-flight
            # (adapter popped but never queued for reconnect). Strong ref so the loop's
            # weak-ref task table can't GC it.
            task = asyncio.ensure_future(result)
            _tasks = _lazy_attr(self, "_detached_fatal_tasks", set)
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Carrier cancelled (typically by our own teardown inside the handler): let it
                # finish detached and consume its exception to avoid "never retrieved" noise.
                if not task.done():
                    task.add_done_callback(_consume_detached_handler_exception)
                raise

    def _acquire_platform_lock(self, scope: str, identity: str, resource_desc: str) -> bool:
        """Acquire a scoped lock for this adapter; True on success. A live cross-HERMES_HOME
        holder is replaced only when the runner armed this adapter for its initial
        ``--replace`` connect (the status module validates ownership and terminates)."""
        from gateway.status import (
            acquire_scoped_lock, scoped_lock_owner_label, take_over_scoped_lock_holder)
        self._platform_lock_scope, self._platform_lock_identity = scope, identity
        lock_meta = {"platform": self.platform.value}
        acquired, existing = acquire_scoped_lock(scope, identity, metadata=lock_meta)
        if acquired:
            return True
        if (bool(getattr(self, "_platform_lock_takeover_allowed", False))
                and not bool(getattr(self, "_platform_lock_takeover_attempted", False))
                and isinstance(existing, dict)):
            # Consume the authority before any I/O: one adapter connect gets at most one
            # termination attempt, even if lock re-acquire or later initialization fails.
            self._platform_lock_takeover_allowed = False
            self._platform_lock_takeover_attempted = True
            owner_pid = take_over_scoped_lock_holder(existing)
            if owner_pid is not None:
                logger.warning(
                    "[%s] %s was held by gateway PID %d — explicit --replace handoff completed",
                    self.name, resource_desc, owner_pid)
                acquired, existing = acquire_scoped_lock(scope, identity, metadata=lock_meta)
                if acquired:
                    logger.info("[%s] Acquired %s after taking over PID %d", self.name, resource_desc, owner_pid)
                    return True
        owner_pid = existing.get('pid') if isinstance(existing, dict) else None
        # Scoped locks are machine-global: name the owning profile when known so the
        # operator can tell WHICH gateway holds the credential.
        owner_profile = scoped_lock_owner_label(existing)
        pid_part = f" (PID {owner_pid})" if owner_pid else ""
        holder = f" by the '{owner_profile}' profile gateway{pid_part}" if owner_profile else pid_part
        remedy = (f" Stop that gateway first (hermes --profile {owner_profile} gateway stop)."
                  if owner_profile else " Stop the other gateway first.")
        message = f"{resource_desc} already in use{holder}.{remedy}"
        logger.error('[%s] %s', self.name, message)
        self._set_fatal_error(f'{scope}_lock', message, retryable=True)
        return False

    def _release_platform_lock(self) -> None:
        """Release the scoped lock acquired by _acquire_platform_lock."""
        identity = getattr(self, '_platform_lock_identity', None)
        if not identity:
            return
        from gateway.status import release_scoped_lock
        release_scoped_lock(self._platform_lock_scope, identity)
        self._platform_lock_identity = None

    def _wire_plugin_handlers(self, native: Any = None) -> None:
        """Invoke plugin-registered native handler factories (``ctx.register_platform_handler``)
        with ``(native, adapter)``; adapters call this from ``connect()`` once the native
        client exists. Each factory is isolated so a bad plugin can't block connecting."""
        platform_name = getattr(self.platform, "value", str(self.platform))
        try:
            from hermes_cli.plugins import get_plugin_manager
            factories = get_plugin_manager().get_platform_handler_factories(platform_name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[%s] Could not load plugin handler factories: %s", self.name, e)
            return
        for factory, plugin_name in factories:
            try:
                factory(native, self)
                logger.info("[%s] Wired native handlers from plugin '%s'", self.name, plugin_name)
            except Exception as exc:
                logger.error(
                    "[%s] Plugin '%s' handler factory raised: %s",
                    self.name, plugin_name, exc, exc_info=True)

    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        return self.platform.value.title()

    @property
    def is_connected(self) -> bool:
        """Check if adapter is currently connected."""
        return self._running

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the incoming-message handler (MessageEvent -> optional response str)."""
        self._message_handler = handler

    def set_platform_event_handler(
        self, handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]]) -> None:
        """Install the gateway-owned normalized platform-event boundary: adapters pass stable
        dicts plus an internal ``SessionSource``; the runner owns authorization and plugin
        dispatch, so no callback = fail closed."""
        self._platform_event_handler = handler

    def set_topic_recovery_fn(self, fn: Optional[Callable[[Any], Optional[str]]]) -> None:
        """Install a thread_id-recovery hook (Telegram DM topic mode): called with
        ``event.source`` before session keying; a non-None return replaces
        ``source.thread_id``. ``None`` clears the hook."""
        # getattr-guard: tests build adapters via object.__new__ (no __init__).
        self._topic_recovery_fn = fn  # type: ignore[attr-defined]

    def _apply_topic_recovery(self, event: MessageEvent) -> None:
        """Rewrite ``event.source.thread_id`` in place if the hook returns one."""
        recover = getattr(self, "_topic_recovery_fn", None)
        source = getattr(event, "source", None)
        if recover is None or source is None:
            return
        try:
            recovered = recover(source)
        except Exception:
            logger.debug("topic recovery hook failed", exc_info=True)
            return
        if recovered is None or str(recovered) == str(source.thread_id or ""):
            return
        try:
            event.source = dataclasses.replace(source, thread_id=str(recovered))
        except Exception:
            logger.debug("topic recovery rewrite failed", exc_info=True)

    def set_busy_session_handler(self, handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]]) -> None:
        """Set an optional handler for messages arriving during active sessions."""
        self._busy_session_handler = handler

    def set_reaction_handler(self, handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]) -> None:
        """Set the handler for platform-native emoji-reaction events: a normalised dict
        (``platform``, ``event_name`` "reaction:added"/"reaction:removed", ``reaction``,
        ``user_id``, ``item_user_id``, ``channel_id``, ``message_ts``, ``event_ts``,
        ``raw_event``) fanned out via ``HookRegistry.emit``."""
        # getattr-guard: tests build adapters via object.__new__ (no __init__).
        self._reaction_handler = handler  # type: ignore[attr-defined]

    def set_authorization_check(
        self, callback: Optional[Callable[[str, Optional[str], Optional[str]], bool]]) -> None:
        """Register ``(user_id, chat_type, chat_id) -> bool``; adapters that pull
        external context (Slack thread replies) use it to flag non-allowlisted
        senders as unverified background rather than authoritative input."""
        self._authorization_check = callback

    def _is_sender_authorized(
        self, user_id: Optional[str], chat_type: Optional[str] = None,
        chat_id: Optional[str] = None, *, is_bot: bool = False, thread_id: Optional[str] = None,
    ) -> Optional[bool]:
        """True/False from the registered check, or ``None`` when no check exists ("trust
        unknown", legacy). ``is_bot``/``thread_id`` are forwarded as keywords only when set
        so legacy three-positional callbacks keep working. Only literal booleans propagate:
        a truthy non-boolean is "unknown", never coerced into an authorization that gates
        a credentialed side effect."""
        if not user_id or self._authorization_check is None:
            return None
        extra: Dict[str, Any] = {**({"is_bot": True} if is_bot else {}),
                                 **({"thread_id": thread_id} if thread_id is not None else {})}
        try:
            result = self._authorization_check(user_id, chat_type, chat_id, **extra)
            if result is True or result is False:
                return result
            logger.warning(
                "[%s] Authorization check returned %s for user %s; treating as unknown",
                self.name, type(result).__name__, user_id)
            return None
        except Exception:
            logger.warning(
                "[%s] Authorization check raised for user %s; treating as unknown",
                self.name, user_id, exc_info=True)
            return None

    def set_session_store(self, session_store: Any) -> None:
        """Set the session store (e.g. Slack checks for an active thread session
        before handling un-mentioned replies)."""
        self._session_store = session_store

    def set_owner_profile(self, profile_name: Optional[str]) -> None:
        """Declare the owning multiplex profile (secondary profiles only); read by
        :meth:`_session_key_profile` so adapter-level keys leave ``agent:main:``."""
        self._owner_profile = None if (name := (profile_name or "").strip() or None) == "default" else name

    def _session_key_profile(self, source: Optional[Any] = None) -> Optional[str]:
        """Profile namespace for an adapter-derived session key. Ingress runs BEFORE the
        runner stamps ``source.profile``, so without this every bot in a multiplexed gateway
        shares one ``agent:main:`` lane. Order: ``source.profile`` → ``_owner_profile`` →
        session-store resolver; getattr-guarded (object.__new__ in tests) and type-checked
        so a MagicMock never lands in the key."""
        for candidate in (
            getattr(source, "profile", None) if source is not None else None,
            getattr(self, "_owner_profile", None)):
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        store = getattr(self, "_session_store", None)
        resolver = getattr(store, "_resolve_profile_for_key", None) if store else None
        if callable(resolver):
            try:
                resolved = resolver(source)
            except Exception:
                return None
            if isinstance(resolved, str) and resolved.strip():
                return resolved
        return None

    # ── Inbound text batching (adapters that merge split messages): subclasses supply
    # ``_pending_text_batches`` / ``_pending_text_batch_tasks`` dicts and ``_flush_text_batch(key)``.

    def _event_session_key(self, event: "MessageEvent") -> str:
        """Adapter-level session key for ``event``, profile-namespaced like the agent run."""
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source))

    def _text_batch_key(self, event: "MessageEvent") -> str:
        """Session-scoped key for text batching (subclasses may override)."""
        return self._event_session_key(event)

    def _enqueue_text_event(self, event: "MessageEvent") -> None:
        """Buffer a text event (merging into a pending one) and restart the flush timer."""
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(self._flush_text_batch(key))

    def _history_media_paths_for_session(self, session_key: str) -> Optional[set]:
        """Return media paths already delivered in prior turns of this session
        (MEDIA: tags / image_generate payloads), so an echoed old tag isn't re-sent."""
        store = getattr(self, "_session_store", None)
        if not store:
            return None
        try:
            # Transcripts are keyed by session_id, not gateway session_key; map via
            # the routing index, falling back to the raw key for stores that accept either.
            peek = getattr(store, "peek_session_id", None)
            session_id = peek(session_key) if callable(peek) else None
            transcript = store.load_transcript(session_id or session_key)
        except Exception:
            return None
        if not transcript:
            return None
        # Exclude the CURRENT TURN entirely (from the last user message onward): rows are
        # persisted as produced, so this turn's tool results are already there and a
        # text_to_speech media_tag would dedup away its own attachment.
        history = list(transcript)
        last_user_idx = next(
            (i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
        if last_user_idx is not None:
            history = history[:last_user_idx]
        else:
            # No user row (unusual store shape): at least drop the trailing reply.
            last_reply = next((msg for msg in reversed(history) if msg.get("role") == "assistant"), None)
            if last_reply is not None:
                history.remove(last_reply)
        if not history:
            return None
        # Avoid circular import: gateway.run already imports this module.
        from gateway.run import _collect_history_media_paths
        return _collect_history_media_paths(history)

    async def _bounded_history_media_paths_for_session(self, session_key: str) -> Optional[set]:
        """Run best-effort history lookup in a bounded isolated daemon thread."""
        def _fail_open(reason: str, *, exc_info: bool = False) -> None:
            logger.warning(
                "[%s] " + reason + " %s; delivering bare local file path(s) without history dedup",
                self.name, session_key, exc_info=exc_info)
        admission = _HISTORY_MEDIA_LOOKUP_ADMISSION
        if not admission.acquire(blocking=False):
            _fail_open("Media-delivery history lookup capacity exhausted for")
            return None
        loop = asyncio.get_running_loop()
        result_future = loop.create_future()

        def _publish_result(result=None, error=None):
            if not result_future.done():
                (result_future.set_exception(error) if error is not None
                 else result_future.set_result(result))

        def _worker():
            result, error = None, None
            try:
                result = self._history_media_paths_for_session(session_key)
            except BaseException as exc:
                error = exc
            try:
                loop.call_soon_threadsafe(_publish_result, result, error)
            except RuntimeError:
                pass  # Event loop already closed during gateway shutdown.
            finally:
                admission.release()
        try:
            threading.Thread(target=_worker, name="media-history-lookup", daemon=True).start()
        except Exception:
            # start() failed (thread exhaustion): the worker never ran, so release the permit
            # here or it leaks. Plain Exception on purpose — don't eat KeyboardInterrupt/SystemExit.
            admission.release()
            _fail_open("Could not start media-delivery history lookup worker for", exc_info=True)
            return None
        try:
            return await asyncio.wait_for(result_future, timeout=_HISTORY_MEDIA_LOOKUP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            _fail_open("Timed out loading media-delivery history for")
            return None
        except Exception:
            # Best-effort/fail-open: never let a lookup failure kill media delivery.
            _fail_open("Media-delivery history lookup failed for", exc_info=True)
            return None

    @abstractmethod
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect and start receiving; True on success. ``is_reconnect`` is True when the
        reconnect watcher re-establishes a dropped platform: adapters with a server-side
        update queue (Telegram) must preserve it so outage-time messages aren't discarded."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""

    @abstractmethod
    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send ``content`` (may be markdown) to a chat; returns SendResult with message id."""

    # True for surfaces that need an explicit finalize edit to close the message
    # lifecycle (DingTalk AI Cards), so the stream consumer never skips it.
    REQUIRES_EDIT_FINALIZE: bool = False

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a fresh thread under ``parent_chat_id`` for a CLI→platform session handoff;
        return its id as a string, or ``None`` when threading is unsupported/failed (the
        watcher then uses ``parent_chat_id`` directly). Thread-capable adapters override."""
        return None

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        """Edit a sent message. Optional: return success=False and callers send anew.
        ``finalize`` marks the last edit of a streamed response; surfaces with a distinct
        "in progress" state (DingTalk AI Cards) use it to close the message and should also
        set ``REQUIRES_EDIT_FINALIZE`` so the final edit is routed even when content is unchanged."""
        return SendResult(success=False, error="Not supported")

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a sent message; ``True`` on success. Optional: platforms without a
        deletion API return ``False`` and callers leave the message in place.
        Used by the stream consumer's fresh-final cleanup to remove stale previews."""
        return False

    def _get_ephemeral_system_ttl_default(self) -> int:
        """Default :class:`EphemeralReply` TTL from ``display.ephemeral_system_ttl``
        (``0`` = no auto-delete); non-fatal if config is unreadable."""
        try:
            return int(_config_section("display").get("ephemeral_system_ttl", 0))
        except (TypeError, ValueError):
            return 0

    def _schedule_ephemeral_delete(self, chat_id: str, message_id: str, ttl_seconds: int) -> None:
        """Spawn a detached task that deletes ``message_id`` after ``ttl_seconds``; best-effort
        (gateway restart, permission denied, Telegram 48h window) swallowed at debug level."""
        async def _run_delete() -> None:
            try:
                await asyncio.sleep(max(1, int(ttl_seconds)))
                await self.delete_message(chat_id=chat_id, message_id=message_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("[%s] Ephemeral delete failed for %s/%s: %s", self.name, chat_id, message_id, e)
        coro = _run_delete()
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop (unit tests): close the coroutine to avoid a never-awaited warning.
            coro.close()

    # ── Shared interactive-prompt formatting cores: ``_format_exec_approval`` templates;
    # adapters override them to keep their historical wording byte-identical.
    _EA_HEADER: str = "⚠️ Command Approval Required\n\n"
    _EA_CODE_OPEN: str = "```\n"
    _EA_CODE_CLOSE: str = "\n```\n"
    _EA_REASON_LABEL: str = "Reason: "
    _EA_SMART_DENY_LINE: str = "\n\nSmart DENY: owner override applies to this one operation only."
    _EA_CMD_BUDGET: int = 3000

    @staticmethod
    def _truncate_preview(text: str, budget: int, suffix: str = "...") -> str:
        """Truncate ``text`` to ``budget`` chars, appending ``suffix`` when cut."""
        text = str(text or "")
        return text[:budget] + suffix if len(text) > budget else text

    def _ea_escape(self, text: str) -> str:
        """Escape hook for command preview/reason; HTML-mode platforms (Telegram) override."""
        return text

    def _format_exec_approval(
        self, command: str, description: str = "dangerous command", smart_denied: bool = False) -> str:
        """Shared exec-approval prompt text: header + fenced (truncated) command + reason,
        plus the smart-deny line. Buttons/trailing instructions stay platform-local."""
        cmd_preview = self._truncate_preview(str(command or ""), self._EA_CMD_BUDGET)
        text = (
            f"{self._EA_HEADER}"
            f"{self._EA_CODE_OPEN}{self._ea_escape(cmd_preview)}{self._EA_CODE_CLOSE}"
            f"{self._EA_REASON_LABEL}{self._ea_escape(description)}")
        return text + self._EA_SMART_DENY_LINE if smart_denied else text

    @staticmethod
    def _format_choice_page(options: list, page: int, per_page: int) -> "tuple[list, Dict[str, Any]]":
        """Shared picker pagination: clamp ``page``, slice ``options``, return
        ``(page_options, meta)`` with ``page``/``total_pages``/``start``/``end``/
        ``total``/``page_info`` (the `` (N–M of T)`` suffix, empty for one page)."""
        total = len(options)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))
        start, end = page * per_page, min(page * per_page + per_page, total)
        page_info = f" ({start + 1}–{end} of {total})" if total_pages > 1 else ""
        meta: Dict[str, Any] = {
            "page": page, "total_pages": total_pages, "start": start, "end": end,
            "total": total, "page_info": page_info}
        return options[start:end], meta

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a three-option slash-command confirmation (e.g. ``/reload-mcp``). Button
        adapters override to render Approve Once / Always Approve / Cancel and MUST resolve
        via ``GatewayRunner._resolve_slash_confirm(confirm_id, choice)`` with ``"once"``/
        ``"always"``/``"cancel"``. Default (not supported) falls through to the gateway text
        fallback (``/approve``/``/always``/``/cancel``)."""
        return SendResult(success=False, error="Not supported")

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a clarify prompt; button-capable adapters SHOULD override. Multiple choice
        (``choices`` non-empty): one button per choice plus "Other"; callbacks MUST resolve via
        ``tools.clarify_gateway.resolve_gateway_clarify(clarify_id, response)`` and "Other"
        calls ``mark_awaiting_text(clarify_id)``. Open-ended: send the question as text; the
        gateway text-intercept resolves the next message. Default: numbered text list +
        ``mark_awaiting_text`` so replies aren't lost."""
        if choices:
            # Multi-select flag lives on the pending entry; look it up by id so the
            # signature stays adapter-compatible.
            try:
                from tools import clarify_gateway as _cg
                with _cg._lock:
                    _entry = _cg._entries.get(clarify_id)
                _is_multi = bool(_entry and getattr(_entry, "multi_select", False))
            except Exception:
                _is_multi = False
            hint = (
                "Multiple selections allowed — reply with the numbers separated by commas or "
                "spaces (e.g. \"1, 3\"), the option text, or your own answer."
                if _is_multi else "Reply with the number, the option text, or your own answer.")
            numbered = [f"  {i}. {choice}" for i, choice in enumerate(choices, start=1)]
            text = "\n".join([f"❓ {question}", "", *numbered, "", hint])
            # Text fallback: let the gateway intercept capture the typed reply.
            from tools.clarify_gateway import mark_awaiting_text
            mark_awaiting_text(clarify_id)
        else:
            text = f"❓ {question}"
        return await self.send(chat_id=chat_id, content=text, metadata=metadata)

    async def send_private_notice(
        self, chat_id: str, user_id: Optional[str], content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a notice privately when the platform supports it; default is a normal send."""
        return await self.send(chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator; ``metadata`` carries platform context (Slack thread_id)."""

    async def stop_typing(self, chat_id: str) -> None:
        """Stop a persistent typing indicator; override where typing runs as a loop."""

    async def _stop_typing_with_metadata(self, chat_id: str, metadata=None) -> None:
        """Stop typing, forwarding ``metadata`` only if ``stop_typing`` accepts it (Slack AI
        status is per thread, so dropping metadata could clear a sibling thread; introspecting
        keeps legacy ``stop_typing(chat_id)`` adapters working)."""
        if metadata:
            try:
                params = inspect.signature(self.stop_typing).parameters
                accepts_metadata = "metadata" in params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
            except (TypeError, ValueError):
                accepts_metadata = False
            if accepts_metadata:
                await self.stop_typing(chat_id, metadata=metadata)
                return
        await self.stop_typing(chat_id)

    async def send_multiple_images(
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """Send a batch of ``(url, alt)`` images (``http(s)://`` or ``file://``) individually
        (GIFs via ``send_animation``, local files via ``send_image_file``); override to
        bundle into one native call (Signal)."""
        from urllib.parse import unquote as _unquote
        for image_url, alt_text in images:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            try:
                logger.info(
                    "[%s] Sending image: %s (alt=%s)",
                    self.name, safe_url_for_log(image_url), alt_text[:30] if alt_text else "")
                caption = alt_text if alt_text else None
                if image_url.startswith("file://"):
                    sender, url_kw = self.send_image_file, {"image_path": _unquote(image_url[7:])}
                elif self._is_animation_url(image_url):
                    sender, url_kw = self.send_animation, {"animation_url": image_url}
                else:
                    sender, url_kw = self.send_image, {"image_url": image_url}
                img_result = await sender(chat_id=chat_id, **url_kw, caption=caption, metadata=metadata)
                if not img_result.success:
                    logger.error("[%s] Failed to send image: %s", self.name, img_result.error)
            except Exception as img_err:
                logger.error("[%s] Error sending image: %s", self.name, img_err, exc_info=True)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image natively; default falls back to sending the URL as text."""
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a GIF as a native animation (auto-plays inline); default falls back to send_image."""
        return await self.send_image(
            chat_id=chat_id, image_url=animation_url, caption=caption, reply_to=reply_to, metadata=metadata)

    @staticmethod
    def _is_animation_url(url: str) -> bool:
        """Check if a URL points to an animated GIF (vs a static image)."""
        return url.lower().split('?')[0].endswith('.gif')

    @staticmethod
    def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
        """Extract ``![alt](url)`` and ``<img src=...>`` image URLs from a response;
        returns ``([(url, alt_text), ...], content with those tags removed)``."""
        cleaned = content
        md_pattern = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
        # Only extract URLs that look like actual images
        images = [
            (m.group(2), m.group(1)) for m in re.finditer(md_pattern, content)
            if any(m.group(2).lower().endswith(ext) or ext in m.group(2).lower() for ext in
                   ['.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery'])]
        # <img src="url"> / <img src="url"></img> / <img src="url"/>
        html_pattern = r'<img\s+src=["\']?(https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
        images.extend((match.group(1), "") for match in re.finditer(html_pattern, content))
        # Remove only the tags we extracted, not every markdown image.
        if images:
            extracted_urls = {url for url, _ in images}
            def _remove_if_extracted(match):
                url = match.group(2) if match.lastindex >= 2 else match.group(1)
                return '' if url in extracted_urls else match.group(0)
            cleaned = re.sub(md_pattern, _remove_if_extracted, cleaned)
            cleaned = re.sub(html_pattern, _remove_if_extracted, cleaned)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()  # leftover blank lines
        return images, cleaned

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send audio as a native voice message (Telegram bubble / Discord attachment).
        Default: friendly failure notice."""
        return await self._send_media_fallback_notice(
            "send_voice", "audio", audio_path, chat_id, caption, reply_to, metadata)

    async def _send_media_fallback_notice(
        self, method: str, kind: str, path: str, chat_id: str, caption: Optional[str],
        reply_to: Optional[str], metadata: Optional[Dict[str, Any]], *, file_name: Optional[str] = None,
    ) -> SendResult:
        """Shared default for send_voice/send_video/send_document/send_image_file. The local
        path is logged but NEVER echoed into chat (it would leak the host layout); only the
        caller-supplied ``file_name`` is shown."""
        logger.warning("[%s] %s fallback: native %s send unavailable for %s", self.name, method, kind, path)
        text = _media_failure_text(kind, file_name)
        text = f"{caption}\n{text}" if caption else text
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    def prepare_tts_text(self, text: str) -> str:
        """Turn chat Markdown into a transcript-like spoken script: reasoning blocks
        removed, headings/bullets flattened, units expanded (``°C`` → degrees Celsius).
        Chunking and delivery limits are the TTS tool's job."""
        try:
            from tools.tts_text_normalize import prepare_spoken_text
            return prepare_spoken_text(text, max_chars=None)
        except Exception:
            # Keep auto-TTS best-effort if the normalizer ever fails.
            text = re.sub(r'<think[\s>].*?</think>', ' ', text, flags=re.DOTALL)
            return re.sub(r'[*_`#\[\]()]', '', text).strip()

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Play auto-TTS audio; override for invisible playback (Web UI). Default: send_voice."""
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    # ── Streaming TTS adapter contract: voice-capable adapters (LiveKit, Discord voice) accept
    # PCM chunks while the LLM generates. Defaults report "unsupported" (whole-file fallback).

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        """Return True when this adapter can accept streaming PCM for *chat_id*."""
        return False

    async def begin_streaming_tts(
        self, chat_id: str, audio_format: AudioFormat, metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTTSHandle]:
        """Open a streaming-audio session; returns an opaque handle for the
        write/finish/abort calls, or ``None`` to decline (whole-file fallback)."""
        return None

    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        """Write one PCM chunk to the adapter's outbound audio track."""

    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *, interrupted: bool = False) -> None:
        """Signal normal end of the audio stream."""

    async def abort_streaming_tts(self, handle: StreamingTTSHandle, error: Optional[str] = None) -> None:
        """Abort the stream due to an error or cancellation. Must be idempotent: late producer
        chunks after abort are silently dropped, not raised. Restores state to "not streaming"."""

    def _streaming_tts_turn_key(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> str | None:
        return streaming_tts_turn_key(session_key, turn_marker, event=event)

    def _mark_streaming_tts_completed_turn(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> None:
        turn_key = self._streaming_tts_turn_key(session_key, turn_marker, event=event)
        if turn_key is not None:
            _lazy_attr(self, "_streaming_tts_completed_turns", set).add(turn_key)

    def _streaming_tts_turn_completed(self, session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> bool:
        return streaming_tts_should_skip_whole_file(
            getattr(self, "_streaming_tts_completed_turns", set()), session_key, turn_marker, event=event)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a video natively (inline playable). Default: friendly failure notice."""
        return await self._send_media_fallback_notice(
            "send_video", "video", video_path, chat_id, caption, reply_to, metadata)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a document/file natively. Default: friendly failure notice."""
        return await self._send_media_fallback_notice(
            "send_document", "file", file_path, chat_id, caption, reply_to, metadata, file_name=file_name)

    async def _notify_media_delivery_failure(
        self, chat_id: str, media_path: str, *, is_voice: bool = False,
        metadata: Optional[Dict[str, Any]] = None) -> None:
        """User-visible notice when a MEDIA attachment upload failed: the tag was
        already stripped from the text, so silence would be a silent drop."""
        ext = Path(media_path).suffix.lower()
        text = (_media_failure_text("audio")
                if is_voice or should_send_media_as_audio(self.platform, ext, is_voice=is_voice)
                else _media_failure_text("video") if ext in _VIDEO_EXTS
                else _media_failure_text("file", os.path.basename(media_path)))
        try:
            notice = await self.send(chat_id=chat_id, content=text, metadata=metadata)
            failed, problem = not notice.success, notice.error
        except Exception as notify_err:
            failed, problem = True, notify_err
        if failed:
            logger.debug("[%s] Could not send media-delivery-failure notice: %s", self.name, problem)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local image file natively (send_image takes a URL). Default: friendly notice."""
        return await self._send_media_fallback_notice(
            "send_image_file", "image", image_path, chat_id, caption, reply_to, metadata)

    @staticmethod
    def validate_media_delivery_path(path: str, session_key: str = "") -> Optional[str]:
        """Return a resolved path if it is safe for native attachment upload."""
        return validate_media_delivery_path(path, session_key=session_key)

    @staticmethod
    def filter_media_delivery_paths(media_files, session_key: str = "") -> List[Tuple[str, bool]]:
        """Drop unsafe MEDIA paths and normalize accepted paths."""
        return [
            (safe_path, bool(is_voice)) for media_path, is_voice in media_files or []
            if (safe_path := _validated_delivery_path(media_path, session_key, "MEDIA directive path"))]

    @staticmethod
    def filter_local_delivery_paths(file_paths, session_key: str = "") -> List[str]:
        """Drop unsafe bare local file paths and normalize accepted paths."""
        safe_paths = (_validated_delivery_path(p, session_key, "local file path") for p in file_paths or [])
        return [p for p in safe_paths if p]

    @staticmethod
    def _mask_protected_spans(content: str) -> str:
        """Blank fenced code, inline code and blockquotes (length-preserving, so
        regex offsets stay valid) to prevent MEDIA: false positives; backtick-quoted
        paths inside MEDIA: tags are left scannable."""
        spans: list = [m.span() for m in _FENCED_CODE_RE.finditer(content)]
        for m in _INLINE_CODE_RE.finditer(content):
            start = m.start()
            if re.search(r'MEDIA:\s*$', content[max(0, start - 20):start]):
                continue  # This is a MEDIA path quote, not inline code
            # A whole tag in inline code (`MEDIA:/path.csv`) is a real directive — models
            # format paths as code — so deliver it IF the path validates; prose examples
            # with non-existent paths stay masked, fenced blocks always.
            inner = m.group(0)[1:-1].strip()
            if inner.upper().startswith("MEDIA:"):
                candidate = _normalize_media_tag_path(inner[6:])
                if candidate and validate_media_delivery_path(candidate):
                    continue  # Real deliverable tag in inline code — keep it scannable
            spans.append((start, m.end()))
        spans.extend(m.span() for m in re.finditer(r'^>.*$', content, re.MULTILINE))
        return _blank_spans(content, spans)

    @staticmethod
    def _mask_json_string_media(content: str) -> str:
        """Blank ``MEDIA:<bare-path>`` tags inside JSON string *values* (stored tool-result
        text like ``{"result": "MEDIA:/x/stale.png"}``) so they are never re-delivered. Only
        spans opened by a value-context quote (``:,{[`` before the ``"``) count, and only bare
        paths (``/``, ``~/``, ``X:\\``) — ``MEDIA:"..."`` quoted tags and line-start/prose tags
        are untouched. Offsets are preserved (blanked with spaces)."""
        if '"' not in content or "MEDIA:" not in content:
            return content
        # JSON value-context string: a quote preceded by : , { or [ (optional ws),
        # capturing the (escape-aware) string body up to the closing quote.
        spans = [
            m.span(1) for m in re.finditer(r'(?<=[:,{\[])\s*"((?:[^"\\\n]|\\.)*)"', content)
            if re.search(r'MEDIA:\s*(?:~/|/|[A-Za-z]:[/\\])', m.group(1))]
        return _blank_spans(content, spans)

    @staticmethod
    def extract_media(content: str) -> Tuple[List[Tuple[str, bool]], str]:
        """Extract ``MEDIA:<path>`` tags and strip the ``[[audio_as_voice]]`` /
        ``[[as_document]]`` directives; returns ``([(path, is_voice), ...], cleaned)``.
        Both directives are message-global; ``[[as_document]]`` (unmodified sendDocument
        delivery for large images) is detected by dispatch sites on the original response
        and only stripped here."""
        media = []
        has_voice_tag = "[[audio_as_voice]]" in content
        cleaned = content.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
        # Scan a masked copy so example/stored MEDIA paths (code blocks, quotes, JSON string
        # values) are never delivered; dedupe on the expanded path so a file referenced
        # twice is uploaded once.
        scan_content = _mask_media_scan_text(content)
        seen_paths: set = set()

        def _add(path: str) -> None:
            # is_voice only for audio files: flagging an image is_voice would
            # push it out of the photo batch into send_document.
            if path not in seen_paths:
                seen_paths.add(path)
                media.append((path, has_voice_tag and os.path.splitext(path)[1].lower() in _AUDIO_EXTS))
        for match in MEDIA_TAG_CLEANUP_RE.finditer(scan_content):
            path = _normalize_media_tag_path(match.group("path"))
            if path:
                try:
                    _add(os.path.expanduser(path))
                except (OSError, RuntimeError, ValueError):
                    continue  # crafted ~\x00 path: skip it, keep the rest
        for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(scan_content):
            path = _normalize_media_tag_path(match.group("path"))
            if not path or not _path_lacks_deliverable_extension(path):
                continue
            resolved = _match_extensionless_path(scan_content, match)
            if resolved is not None:
                _add(resolved[0])
        # Locate real tag spans on a masked copy of ``cleaned``, then delete exactly
        # those spans from the unmasked text so protected spans survive verbatim.
        if media:
            spans = _real_media_tag_spans(_mask_media_scan_text(cleaned))
            if spans:
                cleaned = re.sub(r'\n{3,}', '\n\n', _delete_spans(cleaned, spans)).strip()
        return media, cleaned

    @staticmethod
    def strip_media_directives_for_display(text: str) -> str:
        """Strip MEDIA: directives from streamed/display text. Known-extension tags are
        removed unconditionally (as ``MEDIA_TAG_CLEANUP_RE``); extension-less tags only when
        ``validate_media_delivery_path`` accepts the path, so undeliverable paths stay visible."""
        if not _has_media_directives(text):
            return text
        return re.sub(r'\n{3,}', '\n\n', _strip_media_tag_directives(text)).rstrip()

    @staticmethod
    def extract_local_files(content: str) -> Tuple[List[str], str]:
        """Detect bare local file paths (absolute, ``~/`` or drive-letter) with deliverable
        extensions and return ``(expanded_paths, cleaned_text)``. Candidates must exist on
        disk (``os.path.isfile``) so URLs and hallucinated paths are ignored; paths inside
        fenced or inline code are skipped so code samples are never mutilated. Dispatch by
        type lives in ``gateway/run.py``."""
        ext_part = '|'.join(e.lstrip('.') for e in MEDIA_DELIVERY_EXTS)
        # Lookbehind rejects URL/relative-path matches (https://…/img.png, ./foo.png);
        # the alternation anchors Unix absolute, ``~/`` and Windows drive paths.
        path_re = re.compile(
            r'(?<![/:\w.])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.(?:' + ext_part + r')\b',
            re.IGNORECASE)
        code_spans = _code_spans(content)
        unique: dict = {}  # expanded_path -> raw_match_text, deduped in discovery order
        for match in path_re.finditer(content):
            if any(s <= match.start() < e for s, e in code_spans):
                continue
            raw = match.group(0)
            expanded = os.path.expanduser(raw)
            if os.path.isfile(expanded):
                unique.setdefault(expanded, raw)
            else:
                # Most common reason a promised file never arrives — log the gap.
                logger.info("Skipping bare file path in reply (no file on disk): %s", _log_safe_path(raw))
        cleaned = content
        for raw in unique.values():
            cleaned = cleaned.replace(raw, '')
        return list(unique), re.sub(r'\n{3,}', '\n\n', cleaned).strip() if unique else cleaned

    async def _keep_typing(
        self, chat_id: str, interval: float = 2.0, metadata=None,
        stop_event: asyncio.Event | None = None) -> None:
        """Refresh the typing indicator every ``interval`` seconds until cancelled.

        Platform typing state expires after ~5s. Chats in ``_typing_paused`` are skipped
        (approval waits — Slack's setStatus disables the compose box). Each ``send_typing``
        is bounded by a sub-interval timeout (below ``interval``) so one slow round-trip
        is abandoned before the next tick rather than letting the bubble lapse.
        """
        _send_typing_timeout = max(0.25, min(1.5, interval - 0.25))
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                if chat_id not in self._typing_paused:
                    try:
                        await asyncio.wait_for(
                            self.send_typing(chat_id, metadata=metadata),
                            timeout=_send_typing_timeout)
                    except asyncio.TimeoutError:
                        pass  # Slow network — abandon this tick, stay on schedule.
                    except Exception as typing_err:
                        logger.debug("[%s] send_typing error (non-fatal): %s", self.name, typing_err)
                if stop_event is None:
                    await asyncio.sleep(interval)
                    continue
                loop = asyncio.get_running_loop()
                deadline = loop.time() + interval
                while not stop_event.is_set():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    # Poll rather than wait_for(stop_event.wait()): cancelling that can
                    # wedge shutdown on Python 3.11/pytest-asyncio; sleep cancels immediately.
                    await asyncio.sleep(min(0.25, remaining))
        except asyncio.CancelledError:
            pass  # Normal cancellation when handler completes
        finally:
            # A send_typing after an outer stop_typing() may have recreated the
            # platform typing loop; cancelling this task alone won't clean it up.
            if hasattr(self, "stop_typing"):
                await self._stop_typing_quietly(chat_id, metadata)
            self._typing_paused.discard(chat_id)
            # getattr-guard: tests build adapters via object.__new__ without _status_text.
            getattr(self, "_status_text", {}).pop(str(chat_id), None)

    async def _stop_typing_refresh(
        self, chat_id: str, typing_task: asyncio.Task | None = None, *, metadata=None,
        timeout: float = 0.5, stop_attempts: int = 2) -> None:
        """Stop the refresh task and platform typing state as one operation."""
        self._typing_paused.add(chat_id)
        try:
            if typing_task is not None and not typing_task.done():
                typing_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(typing_task), timeout=timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # Cancelled; slow adapter cleanup must not block delivery/shutdown.
            if not hasattr(self, "stop_typing"):
                return
            for attempt in range(max(1, stop_attempts)):
                if attempt:
                    await asyncio.sleep(0)
                await self._stop_typing_quietly(chat_id, metadata)
        finally:
            self._typing_paused.discard(chat_id)

    async def _stop_typing_quietly(self, chat_id: str, metadata=None) -> None:
        """Best-effort platform stop_typing; adapter errors are swallowed."""
        try:
            await self._stop_typing_with_metadata(chat_id, metadata)
        except Exception:
            pass

    def pause_typing_for_chat(self, chat_id: str) -> None:
        """Pause typing for a chat (approval waits); GIL-safe from the sync agent thread."""
        self._typing_paused.add(chat_id)

    def resume_typing_for_chat(self, chat_id: str) -> None:
        """Resume typing indicator for a chat after approval resolves."""
        self._typing_paused.discard(chat_id)

    async def interrupt_session_activity(self, session_key: str, chat_id: str, metadata=None) -> None:
        """Signal the active session loop to stop and clear typing immediately."""
        interrupt_event = self._active_sessions.get(session_key) if session_key else None
        if interrupt_event is not None:
            interrupt_event.set()
        await self._stop_typing_quietly(chat_id, metadata)

    def register_post_delivery_callback(
        self, session_key: str, callback: Callable, *, generation: int | None = None) -> None:
        """Register a deferred callback to fire after the main response.

        Same-key registrations are chained (both fire, in order, with per-callback
        exception isolation) so independent features coexist. ``generation`` ties
        the callback to a gateway run; stale generations never overwrite a fresher slot.
        """
        if not session_key or not callable(callback):
            return
        existing = self._post_delivery_callbacks.get(session_key)
        if existing is not None:
            existing_gen, existing_cb = _split_post_delivery_entry(existing)
            if existing_gen is not None and generation is not None and int(generation) < int(existing_gen):
                return
            # Same-or-newer generation: chain so both fire in registration order.
            if callable(existing_cb) and (
                existing_gen is None or generation is None or int(existing_gen) == int(generation)):
                _chain = (existing_cb, callback)

                async def _chained() -> None:
                    # Must be async: the invoker awaits awaitable callbacks, and a sync
                    # wrapper would silently drop coroutines returned by async hooks.
                    for _cb in _chain:
                        try:
                            _result = _cb()
                            if inspect.isawaitable(_result):
                                await _result
                        except Exception:
                            logger.debug("Post-delivery callback failed", exc_info=True)
                callback = _chained
        self._post_delivery_callbacks[session_key] = (
            callback if generation is None else (int(generation), callback))

    def pop_post_delivery_callback(
        self, session_key: str, *, generation: int | None = None) -> Callable | None:
        """Pop a deferred callback, optionally requiring generation ownership."""
        entry = self._post_delivery_callbacks.get(session_key) if session_key else None
        if entry is None:
            return None
        entry_generation, callback = _split_post_delivery_entry(entry)
        if generation is not None and (entry_generation is None or int(entry_generation) != int(generation)):
            return None
        self._post_delivery_callbacks.pop(session_key, None)
        return callback if callable(callback) else None

    # ── Processing lifecycle hooks ──────────────────────────────────────────
    # Subclasses override these to react to processing events (e.g. Discord
    # 👀/✅/❌ reactions). Adapters exposing ``_add_reaction(chat_id, message_id,
    # emoji)`` / ``_remove_reaction(chat_id, message_id)`` can instead set the
    # emoji attributes below; left ``None`` the hook stays a no-op.
    _ACK_EMOJI: Optional[str] = None
    _OK_EMOJI: Optional[str] = None
    _FAIL_EMOJI: Optional[str] = None

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Hook called when background processing begins."""

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Hook called when background processing completes.

        Default: opt-in reaction ack — with ``_OK_EMOJI``/``_FAIL_EMOJI`` set and
        ``_add_reaction``/``_remove_reaction`` present, swap the in-progress reaction
        for the outcome one. Remove-then-add is deterministic whether the platform
        replaces or stacks a sender's reactions. CANCELLED leaves it unreacted.
        """
        if self._OK_EMOJI is None and self._FAIL_EMOJI is None:
            return
        add: Any = getattr(self, "_add_reaction", None)
        remove: Any = getattr(self, "_remove_reaction", None)
        enabled = getattr(self, "_reactions_enabled", None)
        if not callable(add) or not callable(remove) or (callable(enabled) and not enabled()):
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not chat_id or not message_id:
            return
        await remove(chat_id, message_id)
        emoji = (
            self._OK_EMOJI if outcome == ProcessingOutcome.SUCCESS
            else self._FAIL_EMOJI if outcome == ProcessingOutcome.FAILURE else None)
        if emoji:
            await add(chat_id, message_id, emoji)

    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        try:
            await hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s hook failed: %s", self.name, hook_name, e)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        """Return True if the error string looks like a transient network failure."""
        lowered = (error or "").lower()
        return any(pat in lowered for pat in _RETRYABLE_ERROR_PATTERNS)

    @staticmethod
    def _is_timeout_error(error: Optional[str]) -> bool:
        """Return True for read/write timeouts — NOT retryable and NOT a plain-text
        fallback trigger, because the request may already have been delivered."""
        lowered = (error or "").lower()
        return any(pat in lowered for pat in ("timed out", "readtimeout", "writetimeout"))

    def _unwrap_ephemeral(self, response: Any) -> Tuple[Optional[str], int]:
        """Unwrap a str/None/:class:`EphemeralReply` response into ``(text, ttl)``.

        ``ttl > 0`` means the caller should schedule ``_schedule_ephemeral_delete``
        after a successful send; forced to 0 when the adapter doesn't override
        ``delete_message`` so non-supporting platforms degrade to normal sends.
        """
        if not isinstance(response, EphemeralReply):
            return response, 0
        ttl = response.ttl_seconds
        if ttl is None:
            try:
                ttl = int(self._get_ephemeral_system_ttl_default())
            except Exception:
                ttl = 0
        if ttl and ttl > 0 and type(self).delete_message is BasePlatformAdapter.delete_message:
            ttl = 0
        return response.text, int(ttl or 0)

    async def _dispatch_inline_reply(self, event: MessageEvent, *, log_cmd: Optional[str] = None) -> None:
        """Call the handler and send its reply inline, with retry, threading and
        ephemeral deletion — no session lifecycle (active-session bypass paths)."""
        thread_meta = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
        response = await self._message_handler(event)
        text, eph_ttl = self._unwrap_ephemeral(response)
        if not text:
            return
        if log_cmd is not None:
            logger.info(
                "[%s] Sending command '/%s' response (%d chars) to %s",
                self.name, log_cmd, len(text), event.source.chat_id)
        result = await self._send_with_retry(
            chat_id=event.source.chat_id, content=text, reply_to=_reply_anchor_for_event(event),
            metadata=_mark_notify_metadata(thread_meta))
        if eph_ttl > 0 and result.success and result.message_id:
            self._schedule_ephemeral_delete(
                chat_id=event.source.chat_id, message_id=result.message_id, ttl_seconds=eph_ttl)

    def _final_delivery_adapter(self, source: Optional[SessionSource]) -> "BasePlatformAdapter":
        """Return the runner's current adapter for a new final-response send.

        A reconnect can swap the registry adapter while this task is in flight; an
        unsent final response belongs on the replacement transport, but message IDs,
        edits and deletes stay owned by the old one (nothing is migrated).
        """
        runner = getattr(self, "gateway_runner", None)
        resolve = getattr(runner, "_adapter_for_source", None)
        if not callable(resolve):
            return self
        try:
            live_adapter = resolve(source)
        except Exception:
            logger.debug("[%s] Failed to resolve live adapter for final delivery", self.name)
            return self
        if isinstance(live_adapter, BasePlatformAdapter) and live_adapter.platform == self.platform:
            return live_adapter
        return self

    async def _send_with_retry(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Any = None,
        max_retries: int = 2, base_delay: float = 2.0) -> "SendResult":
        """Send with exponential-backoff retry on transient network errors; permanent
        failures fall back to a plain-text send, exhausted retries notify the user."""
        async def _send(text: str) -> "SendResult":
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)
        result = await _send(content)
        if result.success:
            return result
        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)
        # Timeouts: not safe to retry (may have delivered) and not a formatting error.
        if not is_network and self._is_timeout_error(error_str):
            return result
        if is_network:
            # A server-requested retry_after (Telegram FloodWait) overrides backoff, once per send.
            server_retry_after = result.retry_after
            for attempt in range(1, max_retries + 1):
                backoff = base_delay * (2 ** (attempt - 1))
                if server_retry_after is not None:
                    backoff = server_retry_after
                delay = backoff + random.uniform(0, 1)
                server_retry_after = None
                logger.warning(
                    "[%s] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                    self.name, attempt, max_retries, delay, error_str)
                await asyncio.sleep(delay)
                result = await _send(content)
                if result.success:
                    logger.info("[%s] Send succeeded on retry %d", self.name, attempt)
                    return result
                error_str = result.error or ""
                if result.retry_after is not None:
                    server_retry_after = result.retry_after
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break  # non-transient now — fall through to plain-text fallback
            else:  # retries exhausted — notify user
                logger.error("[%s] Failed to deliver response after %d retries: %s", self.name, max_retries, error_str)
                notice = (
                    "\u26a0\ufe0f Message delivery failed after multiple attempts. "
                    "Please try again \u2014 your request was processed but the response could not be sent.")
                try:
                    await _send(notice)
                except Exception as notify_err:
                    logger.debug("[%s] Could not send delivery-failure notice: %s", self.name, notify_err)
                return result
        # Non-network / post-retry formatting failure: try plain text as fallback
        logger.warning("[%s] Send failed: %s — trying plain-text fallback", self.name, error_str)
        fallback_result = await _send(f"(Response formatting failed, plain text:)\n\n{content[:3500]}")
        if not fallback_result.success:
            logger.error("[%s] Fallback send also failed: %s", self.name, fallback_result.error)
        return fallback_result

    @staticmethod
    def _merge_caption(existing_text: Optional[str], new_text: str) -> str:
        """Merge a new caption into existing text unless an identical (whitespace-
        normalised) caption already exists — exact match per caption, not
        substring, so "Meeting" is not swallowed by "Meeting agenda"."""
        if not existing_text:
            return new_text
        if new_text.strip() in [c.strip() for c in existing_text.split("\n\n")]:
            return existing_text
        return f"{existing_text}\n\n{new_text}".strip()

    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        return _lazy_attr(self, "_text_debounce", dict)

    def _is_queue_text_debounce_candidate(self, event: MessageEvent) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT
            and not getattr(event, "internal", False)
            and not event.is_command()
            and bool((event.text or "").strip()))
        if result:
            logger.debug(
                "[%s] Queue-text debounce candidate accepted: session=%s text_len=%d", self.name,
                getattr(event, "session_key", "?"), len(event.text or ""))
        return result

    def _can_merge_text_debounce_events(self, existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
            if sender:
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(source, "chat_id", None):
                return (platform, "dm", str(source.chat_id))
            return None
        existing_sender = _identity(existing)
        return existing_sender is not None and existing_sender == _identity(event)

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(
            state.last_ts + self._busy_text_debounce_seconds,
            state.first_ts + self._busy_text_hard_cap_seconds)
        return max(0.0, deadline - time.monotonic())

    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution in shared sessions: flush the current
            # buffer as the next pending turn; the new sender starts a fresh burst.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    merge_pending_message_event(self._pending_messages, session_key, event, merge_text=True)
                return
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            if event.text:
                state.event.text = _append_text(state.event.text, event.text)
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now
        state.cancel_timer()
        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))

    async def _flush_text_debounce(self, session_key: str, delay: float) -> None:
        """Timer task that flushes the debounced text buffer."""
        try:
            await asyncio.sleep(delay)
            await self._flush_text_debounce_now(session_key)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            state = self._text_debounce_store().get(session_key)
            if state is not None and state.task is current:
                state.task = None

    async def _flush_text_debounce_now(self, session_key: str) -> bool:
        """Force-flush one debounced busy-text burst into the pending slot."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        existing_pending = self._pending_messages.get(session_key)
        if (
            existing_pending is not None
            and not self._can_merge_text_debounce_events(existing_pending, state.event)):
            return False
        store.pop(session_key, None)
        merge_pending_message_event(self._pending_messages, session_key, state.event, merge_text=True)
        return True

    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()

    # ── Session task + guard ownership helpers ──────────────────────────────
    # Paired with the _session_tasks owner map so lifecycle reconciliation is
    # deterministic across normal completion, /stop /new /reset bypass commands,
    # and stale-lock self-heal on the next inbound message.

    def _release_session_guard(self, session_key: str, *, guard: Optional[asyncio.Event] = None) -> None:
        """Release the session guard; with ``guard`` given, only if the entry is
        still that exact Event (so an old task's unwind can't clear the temporary
        guard a reset-like command swapped in)."""
        current_guard = self._active_sessions.get(session_key)
        if current_guard is None or (guard is not None and current_guard is not guard):
            return
        del self._active_sessions[session_key]

    def _session_task_is_stale(self, session_key: str) -> bool:
        """True if the recorded owner task for ``session_key`` has exited. No owner
        task at all is NOT stale (guards installed outside handle_message, as tests
        do, must not be healed); only owner-recorded-then-exited counts."""
        done = getattr(self._session_tasks.get(session_key), "done", None)
        return bool(done and done())

    def _heal_stale_session_lock(self, session_key: str) -> bool:
        """Clear a stale session lock; True if healed, False if no lock or the owner
        is still alive. On-entry safety net: without it a split-brain (guard held,
        nothing processing) traps the chat in "Interrupting..." until restart."""
        if session_key not in self._active_sessions or not self._session_task_is_stale(session_key):
            return False
        logger.warning(
            "[%s] Healing stale session lock for %s (owner task is done/absent)",
            self.name, session_key)
        self._active_sessions.pop(session_key, None)
        self._pending_messages.pop(session_key, None)
        self._session_tasks.pop(session_key, None)
        self._discard_text_debounce(session_key)
        return True

    def _start_session_processing(
        self, event: MessageEvent, session_key: str, *,
        interrupt_event: Optional[asyncio.Event] = None) -> bool:
        """Spawn a background processing task under the session guard; True on
        success. If ``create_task`` is stubbed with a non-Task sentinel (tests),
        the guard is rolled back and False returned — no half-installed lock."""
        guard = interrupt_event or asyncio.Event()
        self._active_sessions[session_key] = guard
        task = asyncio.create_task(self._process_message_background(event, session_key))
        self._session_tasks[session_key] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            # Tests stub create_task() with unhashable sentinels lacking lifecycle callbacks.
            self._session_tasks.pop(session_key, None)
            self._release_session_guard(session_key, guard=guard)
            return False
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)
        return True

    async def cancel_session_processing(
        self, session_key: str, *, release_guard: bool = True, discard_pending: bool = True,
    ) -> None:
        """Cancel in-flight processing for one session. ``release_guard=False`` keeps
        the guard so reset-like commands finish atomically before follow-ups start;
        the await is bounded (5s) so a wedged finally block can't stall dispatch."""
        task = self._session_tasks.pop(session_key, None)
        if task is not None and not task.done():
            logger.debug("[%s] Cancelling active processing for session %s", self.name, session_key)
            self._expected_cancelled_tasks.add(task)
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Cancelled task for %s did not exit within 5s; "
                    "unblocking dispatch and letting the task unwind in the background",
                    self.name, session_key)
            except Exception:
                logger.debug(
                    "[%s] Session cancellation raised while unwinding %s", self.name, session_key,
                    exc_info=True)
        if discard_pending:
            self._pending_messages.pop(session_key, None)
            self._discard_text_debounce(session_key)
        if release_guard:
            self._release_session_guard(session_key)

    async def _drain_pending_after_session_command(
        self, session_key: str, command_guard: asyncio.Event) -> None:
        """Tail of /stop, /new, /reset: release the command-scoped guard, then
        spawn a fresh processing task for any follow-up queued meanwhile."""
        await self._flush_text_debounce_now(session_key)
        pending_event = self._pending_messages.pop(session_key, None)
        self._release_session_guard(session_key, guard=command_guard)
        if pending_event is not None:
            self._start_session_processing(pending_event, session_key)

    async def _dispatch_active_session_command(self, event: MessageEvent, session_key: str, cmd: str) -> None:
        """Dispatch a reset-like bypass command (/stop, /new, /reset): keep a guard
        installed while the runner handles it (racing follow-ups stay queued), cancel
        the old task AFTER the response is sent, then release and drain once."""
        logger.debug("[%s] Command '/%s' bypassing active-session guard for %s", self.name, cmd, session_key)
        current_guard = self._active_sessions.get(session_key)
        command_guard = asyncio.Event()
        self._active_sessions[session_key] = command_guard
        try:
            # Send BEFORE cancelling the old task so cancellation side effects
            # can't drop the "/new" confirmation; cancel AFTER for deterministic ordering.
            await self._dispatch_inline_reply(event, log_cmd=cmd)
            await self.cancel_session_processing(session_key, release_guard=False, discard_pending=False)
        except Exception:
            # On failure restore the original guard so the session isn't left half-reset.
            if self._active_sessions.get(session_key) is command_guard:
                if session_key in self._session_tasks and current_guard is not None:
                    self._active_sessions[session_key] = current_guard
                else:
                    self._release_session_guard(session_key, guard=command_guard)
            raise
        await self._drain_pending_after_session_command(session_key, command_guard)

    async def handle_message(self, event: MessageEvent) -> None:
        """Process an incoming message; returns quickly by spawning a background
        task so new messages (and interrupts) can arrive while an agent runs."""
        if not self._message_handler:
            return
        if event.allow_gateway_control:
            coerce_plaintext_gateway_command(event)
        # Topic recovery is Telegram-DM-only; skip the executor hop for group traffic.
        if (getattr(self, "_topic_recovery_fn", None) is not None
                and event.source.platform == Platform.TELEGRAM and event.source.chat_type == "dm"):
            await asyncio.to_thread(self._apply_topic_recovery, event)
        session_key = self._event_session_key(event)
        expected_session_key = str((event.metadata or {}).get("gateway_session_key") or "").strip()
        if expected_session_key and session_key != expected_session_key:
            logger.warning(
                "Dropping internally routed event: expected session=%s derived=%s",
                expected_session_key, session_key)
            return
        # On-entry self-heal: clear a guard whose owner task already exited.
        if session_key in self._active_sessions:
            self._heal_stale_session_lock(session_key)
        if session_key in self._active_sessions:
            await self._handle_message_while_active(event, session_key)
            return
        # Guard is installed synchronously BEFORE the task spawns (grammY sequentialize)
        # so a second message can't race in; the owner task is recorded atomically.
        self._start_session_processing(event, session_key)

    async def _handle_message_while_active(self, event: MessageEvent, session_key: str) -> None:
        """Route a message that arrived while ``session_key`` is busy: bypass
        commands / clarify replies dispatch inline, everything else is queued."""
        # Bypass commands run inline: queued they would leak as user text (/stop, /new)
        # or deadlock (/approve, /deny).
        cmd = event.get_command()
        from hermes_cli.commands import (is_interrupt_then_dispatch, should_bypass_active_session)
        if should_bypass_active_session(cmd):
            try:
                # /stop, /new, /reset take the handoff path (cancel + response + drain);
                # other bypass commands dispatch directly without cancelling the task.
                if cmd and is_interrupt_then_dispatch(cmd):
                    self._discard_text_debounce(session_key)
                    await self._dispatch_active_session_command(event, session_key, cmd)
                else:
                    logger.debug(
                        "[%s] Command '/%s' bypassing active-session guard for %s",
                        self.name, cmd, session_key)
                    await self._dispatch_inline_reply(event)
            except Exception as e:
                logger.error("[%s] Command '/%s' dispatch failed: %s", self.name, cmd, e, exc_info=True)
            return
        # Clarify bypass: while blocked on clarify_tool, the next non-command message
        # must reach the runner's text-intercept, not the queue.
        if not cmd and event.allow_gateway_control:
            try:
                from tools import clarify_gateway as _clarify_mod
                _has_text_clarify = _clarify_mod.get_pending_for_session(
                    session_key, include_choice_prompts=True) is not None
            except Exception:
                _has_text_clarify = False
            if _has_text_clarify:
                logger.debug("[%s] Routing message to clarify text-intercept for %s", self.name, session_key)
                try:
                    await self._dispatch_inline_reply(event)
                except Exception as e:
                    logger.error("[%s] Clarify text-intercept dispatch failed: %s", self.name, e, exc_info=True)
                return
        if self._busy_session_handler is not None:
            try:
                if await self._busy_session_handler(event, session_key):
                    return
            except Exception as e:
                logger.error("[%s] Busy-session handler failed: %s", self.name, e, exc_info=True)
        # Photo bursts/albums: queue without interrupting; they run after the current task.
        if event.message_type == MessageType.PHOTO:
            logger.debug("[%s] Queuing photo follow-up for session %s without interrupt", self.name, session_key)
            merge_pending_message_event(self._pending_messages, session_key, event)
            return
        if self._is_queue_text_debounce_candidate(event):
            logger.debug(
                "[%s] New text message while session %s is active — "
                "debouncing follow-up (busy_text_mode=queue, window=%.2fs)",
                self.name, session_key, self._busy_text_debounce_seconds)
            await self._queue_text_debounce(session_key, event)
        else:
            logger.debug(
                "[%s] New message while session %s is active — queuing follow-up "
                "(no interrupt, will cascade after current turn)",
                self.name, session_key)
            merge_pending_message_event(
                self._pending_messages, session_key, event,
                merge_text=event.message_type == MessageType.TEXT)

    @staticmethod
    def _get_human_delay() -> float:
        """Random human-like pacing delay in seconds, from HERMES_HUMAN_DELAY_MODE
        ("off" default | "natural" 800-2500ms | "custom" via
        HERMES_HUMAN_DELAY_MIN_MS / HERMES_HUMAN_DELAY_MAX_MS)."""
        mode = os.getenv("HERMES_HUMAN_DELAY_MODE", "off").lower()
        if mode == "off":
            return 0.0
        if mode == "natural":
            return random.uniform(800 / 1000.0, 2500 / 1000.0)
        def _ms(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default  # custom mode tolerates malformed env vars
        return random.uniform(
            _ms("HERMES_HUMAN_DELAY_MIN_MS", 800) / 1000.0,
            _ms("HERMES_HUMAN_DELAY_MAX_MS", 2500) / 1000.0)

    async def _synthesize_auto_tts(self, text_content: str) -> Tuple[List[str], Optional[str]]:
        """Synthesize auto-TTS audio -> ``(existing_paths, requested_path)``; empty/None
        on failure (logged, never raised). The path is built platform-aware here
        because HERMES_SESSION_PLATFORM is already cleared post-handler."""
        paths: List[str] = []
        requested_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, check_tts_requirements
            if check_tts_requirements():
                import json as _json
                speech_text = self.prepare_tts_text(text_content)
                if not speech_text:
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = build_auto_tts_output_path(self.platform)
                tts_data = _json.loads(await asyncio.to_thread(
                    text_to_speech_tool, text=speech_text, output_path=requested_path))
                if tts_data.get("success", True):
                    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
                    paths = [str(path) for path in raw_tts_paths if path and Path(path).exists()]
        except Exception as tts_err:
            logger.warning("[%s] Auto-TTS failed: %s", self.name, tts_err)
        return paths, requested_path

    def _wants_auto_tts(
        self, event: MessageEvent, session_key: str, interrupt_event: asyncio.Event,
        text_content: str, media_files: list) -> bool:
        """Auto-TTS on voice input (voice-first), gated by /voice or voice.auto_tts;
        skipped when streaming TTS already delivered audio this turn."""
        return bool(
            self._should_auto_tts_for_chat(event.source.chat_id)
            and event.message_type == MessageType.VOICE
            and text_content
            and not media_files
            and not self._streaming_tts_turn_completed(
                session_key, getattr(interrupt_event, "_hermes_run_generation", None), event=event))

    async def _play_tts_file(
        self, event: MessageEvent, text_content: str, tts_path: str, first: bool,
        metadata: Dict[str, Any], record_delivery: Callable) -> bool:
        """Play one synthesized TTS file. Returns True when the ORIGINAL reply text rode
        along as a Telegram caption (first file, ≤1024 chars) so the text send is skipped."""
        caption = None
        if first and self.platform == Platform.TELEGRAM and text_content and text_content[:1024] == text_content:
            caption = text_content
        tts_result = await self.play_tts(
            chat_id=event.source.chat_id, audio_path=tts_path, caption=caption, metadata=metadata)
        record_delivery(tts_result)
        return bool(caption and getattr(tts_result, "success", False))

    async def _record_delivery_obligation(
        self, event: MessageEvent, session_key: str, text_content: str,
        delivery_adapter: "BasePlatformAdapter", is_ephemeral_response: bool) -> Optional[str]:
        """Record the final response in the ledger BEFORE the send so a crash before
        platform ACK redelivers on next boot; best-effort, skips slash-command and
        ephemeral replies. Returns the obligation id or None."""
        if is_ephemeral_response or str(event.text or "").lstrip().startswith(
            ("/", self.typed_command_prefix or "!")):
            return None
        try:
            from gateway.delivery_ledger import (
                compute_obligation_id, ledger_enabled, mark_attempting, record_obligation)
            if not await asyncio.to_thread(ledger_enabled):
                return None
            obligation_id = compute_obligation_id(
                session_key, str(getattr(event, "message_id", "") or ""), text_content)
            await asyncio.to_thread(
                record_obligation, obligation_id=obligation_id, session_key=session_key,
                platform=str(getattr(event.source.platform, "value", event.source.platform)),
                chat_id=event.source.chat_id, thread_id=getattr(event.source, "thread_id", None),
                content=text_content,
                adapter_profile=getattr(delivery_adapter, "_owner_profile", None))
            await asyncio.to_thread(mark_attempting, obligation_id)
            return obligation_id
        except Exception:
            logger.debug("delivery ledger record failed", exc_info=True)
            return None

    async def _finalize_delivery_obligation(
        self, obligation_id: str, result: Any, event: MessageEvent,
        delivery_adapter: "BasePlatformAdapter") -> None:
        """Mark the ledger row delivered/failed (best-effort). On ``send_path_degraded``
        with a replacement adapter live, signal another redelivery sweep (the watcher's
        may have run before this failure landed; atomic claiming keeps it idempotent)."""
        try:
            from gateway.delivery_ledger import mark_delivered, mark_failed
            if getattr(result, "success", False):
                await asyncio.to_thread(mark_delivered, obligation_id)
                return
            _delivery_error = str(getattr(result, "error", "") or "")
            await asyncio.to_thread(mark_failed, obligation_id, _delivery_error)
            if _delivery_error == "send_path_degraded":
                _live_adapter = self._final_delivery_adapter(event.source)
                _runtime_redeliver = getattr(
                    getattr(self, "gateway_runner", None),
                    "_redeliver_failed_obligations_for_platform", None)
                if _live_adapter is not delivery_adapter and callable(_runtime_redeliver):
                    await _runtime_redeliver(
                        event.source.platform,
                        profile=getattr(delivery_adapter, "_owner_profile", None))
        except Exception:
            logger.debug("delivery ledger update failed", exc_info=True)

    async def _deliver_media_attachments(
        self, event: MessageEvent, media_files: list, local_files: list, *,
        force_document_attachments: bool, human_delay: float, metadata: Dict[str, Any]) -> None:
        """Deliver MEDIA-tag files and auto-detected local files by type: images are
        batched via ``send_multiple_images`` unless ``[[as_document]]`` forced document
        delivery; other MEDIA files route audio → send_voice, video → send_video, else
        send_document (local files never go to send_voice). Every failure is reported."""
        from urllib.parse import quote as _quote

        def _as_image(path: str) -> bool:
            return Path(path).suffix.lower() in _IMAGE_EXTS and not force_document_attachments
        _image_paths = [p for p, is_voice in media_files if not is_voice and _as_image(p)]
        _image_paths += [p for p in local_files if _as_image(p)]
        if _image_paths:
            await self._send_image_batch(
                event, [(f"file://{_quote(p)}", "") for p in _image_paths], metadata, human_delay)
        chat_id = event.source.chat_id

        async def _send_one(path: str, *, is_voice: bool, media_tag: bool) -> None:
            """MEDIA-tag files (``media_tag``) may route to send_voice; bare local files never do."""
            ext = Path(path).suffix.lower()
            if media_tag and should_send_media_as_audio(self.platform, ext, is_voice=is_voice):
                result = await self.send_voice(chat_id=chat_id, audio_path=path, metadata=metadata, is_voice=is_voice)
            elif ext in _VIDEO_EXTS:
                if media_tag:
                    logger.info("[%s] Sending video attachment (%s) to %s", self.name, ext, chat_id)
                result = await self.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
            else:
                result = await self.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            if not result.success:
                label = "media" if media_tag else "local file"
                logger.warning("[%s] Failed to send %s (%s): %s", self.name, label, ext, result.error)
                await self._notify_media_delivery_failure(chat_id, path, is_voice=is_voice, metadata=metadata)
        _non_image_media = [(p, v) for p, v in media_files if v or not _as_image(p)]
        if _non_image_media:
            logger.info("[%s] Delivering %d non-image MEDIA attachment(s)", self.name, len(_non_image_media))
        queue = [(p, v, True) for p, v in _non_image_media]
        queue += [(p, False, False) for p in local_files if not _as_image(p)]
        for path, is_voice, media_tag in queue:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            try:
                await _send_one(path, is_voice=is_voice, media_tag=media_tag)
            except Exception as err:
                if media_tag:
                    logger.warning("[%s] Error sending media: %s", self.name, err)
                else:
                    logger.error("[%s] Error sending local file %s: %s", self.name, path, err)

    async def _send_image_batch(
        self, event: MessageEvent, images: list, metadata: Dict[str, Any], human_delay: float) -> None:
        """Batch-send images; a failure is logged (never raised) so other attachments still go."""
        try:
            await self.send_multiple_images(
                chat_id=event.source.chat_id, images=images, metadata=metadata, human_delay=human_delay)
        except Exception as batch_err:
            logger.warning("[%s] Error batching images: %s", self.name, batch_err, exc_info=True)

    async def _send_final_text(
        self, event: MessageEvent, session_key: str, text_content: str, metadata: Dict[str, Any],
        is_ephemeral_response: bool, ephemeral_ttl: int, record_delivery: Callable) -> None:
        """Send the final text on the CURRENT transport (a reconnect may have replaced
        this adapter), ledger-bracketed; the message-id owner owns the ephemeral delete."""
        delivery_adapter = self._final_delivery_adapter(event.source)
        logger.info(
            "[%s] Sending response (%d chars) to %s", delivery_adapter.name, len(text_content),
            event.source.chat_id)
        _obligation_id = await self._record_delivery_obligation(
            event, session_key, text_content, delivery_adapter, is_ephemeral_response)
        result = await delivery_adapter._send_with_retry(
            chat_id=event.source.chat_id, content=text_content,
            reply_to=_reply_anchor_for_event(event), metadata=metadata)
        record_delivery(result)
        if _obligation_id is not None:
            await self._finalize_delivery_obligation(_obligation_id, result, event, delivery_adapter)
        if ephemeral_ttl and ephemeral_ttl > 0 and result.success and result.message_id:
            delivery_adapter._schedule_ephemeral_delete(
                chat_id=event.source.chat_id, message_id=result.message_id,
                ttl_seconds=ephemeral_ttl)

    async def _notify_turn_error(self, event: MessageEvent, e: BaseException) -> Optional[dict]:
        """Tell the user a turn failed rather than leaving radio silence (last resort:
        a failing notice is logged, never raised). Returns the thread metadata used."""
        _thread_metadata = None
        try:
            error_detail = str(e)[:300] if str(e) else "no details available"
            _thread_metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
            await self.send(
                chat_id=event.source.chat_id,
                content=(
                    f"Sorry, I encountered an error ({type(e).__name__}).\n{error_detail}\n"
                    "Try again or use /reset to start a fresh session."),
                metadata=_thread_metadata)
        except Exception as notify_err:
            logger.error(
                "[%s] Failed to send error notification to user: %s", self.name, notify_err, exc_info=True)
        return _thread_metadata

    async def _deliver_attachments(
        self, event: MessageEvent, extracted: "_ExtractedResponse", metadata: Dict[str, Any], *,
        anything_sent: bool) -> None:
        """Send extracted image URLs, MEDIA files and bare local files (human-paced),
        then fail loudly if a non-empty response produced nothing deliverable."""
        human_delay = self._get_human_delay()
        images, media_files, local_files = extracted.images, extracted.media_files, extracted.local_files
        if images:
            logger.info("[%s] Extracted %d image(s) to send as attachments", self.name, len(images))
            await self._send_image_batch(event, images, metadata, human_delay)
        await self._deliver_media_attachments(
            event, media_files, local_files,
            force_document_attachments=extracted.force_document_attachments,
            human_delay=human_delay, metadata=metadata)
        if not (anything_sent or images or local_files or media_files) and extracted.pre_extract.strip():
            logger.error(
                "[%s] response_delivery_dropped: non-empty response "
                "(%d chars) produced no delivered message or attachment "
                "for %s (empty after extract, recovery yielded nothing).",
                self.name, len(extracted.pre_extract), event.source.chat_id)

    def _start_typing_refresh(
        self, event: MessageEvent, interrupt_event: asyncio.Event, metadata: Optional[dict],
    ) -> Optional[asyncio.Task]:
        """Spawn the typing-refresh task, or None when ``typing_indicator=False``.
        ``stop_event`` is passed only when the (possibly overridden) ``_keep_typing`` accepts it."""
        if not getattr(self.config, "typing_indicator", True):
            return None
        kwargs: Dict[str, Any] = {"metadata": metadata}
        try:
            sig = inspect.signature(self._keep_typing)
        except (TypeError, ValueError):
            sig = None
        if sig is None or "stop_event" in sig.parameters:
            kwargs["stop_event"] = interrupt_event
        return asyncio.create_task(self._keep_typing(event.source.chat_id, **kwargs))

    async def _extract_response_content(
        self, response: str, event: MessageEvent, session_key: str, *, is_ephemeral_response: bool,
    ) -> "_ExtractedResponse":
        """Split a handler response into deliverable text + attachments. Order matters:
        MEDIA tags → image URLs → residual directives → bare local paths (skipped for
        ephemeral notices so config paths stay text; unknown-extension MEDIA tags survive
        the strip for the bare-path detector). History dedup is bare-path only, off-loop,
        fail-open. If extraction empties a non-empty response, the text is recovered."""
        # Captured before extract_media strips it: images then go via send_document (no recompression).
        force_document = "[[as_document]]" in response
        pre_extract = response
        media_files, response = self.extract_media(response)
        media_files = self.filter_media_delivery_paths(media_files, session_key=session_key)
        images, text_content = self.extract_images(response)
        text_content = _strip_media_directives(text_content).strip()
        if images:
            logger.info("[%s] extract_images found %d image(s) in response (%d chars)", self.name, len(images), len(response))
        local_files = []
        if not is_ephemeral_response:
            local_files, text_content = self.extract_local_files(text_content)
            local_files = self.filter_local_delivery_paths(local_files, session_key=session_key)
            _history_media_paths = (
                await self._bounded_history_media_paths_for_session(session_key)
                if local_files else None)
            if _history_media_paths:
                _suppressed = [p for p in local_files if p in _history_media_paths]
                if _suppressed:
                    logger.info(
                        "[%s] Suppressing %d bare local file path(s) already "
                        "delivered in this session: %s", self.name, len(_suppressed), _suppressed)
                local_files = [p for p in local_files if p not in _history_media_paths]
            if local_files:
                logger.info("[%s] extract_local_files found %d file(s) in response", self.name, len(local_files))
        if not (text_content or images or local_files or media_files):
            _recovered = _strip_media_directives(response).strip()
            if _recovered:
                logger.warning(
                    "[%s] response_delivery_recovered: extract pipeline "
                    "reduced a non-empty response (%d chars) to empty with "
                    "no attachment; delivering recovered original to %s",
                    self.name, len(pre_extract), event.source.chat_id)
                text_content = _recovered
        return _ExtractedResponse(
            text_content=text_content, images=images, media_files=media_files,
            local_files=local_files, force_document_attachments=force_document, pre_extract=pre_extract)

    async def _fire_post_delivery_callback(self, session_key: str, interrupt_event: asyncio.Event) -> None:
        """Run the one-shot post-delivery callback (bounded, errors swallowed). The
        generation is read HERE: it is stamped on the interrupt event DURING the handler
        await, so an earlier snapshot would let stale runs fire a fresher run's callbacks."""
        _post_cb = self.pop_post_delivery_callback(
            session_key, generation=getattr(interrupt_event, "_hermes_run_generation", None))
        if callable(_post_cb):
            try:
                _post_result = _post_cb()
                if inspect.isawaitable(_post_result):
                    await asyncio.wait_for(_post_result, timeout=_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, Exception):
                pass

    def _finish_session_task(self, session_key: str, interrupt_event: asyncio.Event) -> None:
        """End-of-task guard/ownership reconciliation. A late ``_pending_messages``
        arrival must not drop: re-queue it if another task already owns the session
        (drain handoff), else spawn the drain task and leave it the guard. With
        nothing pending, release the guard only if we still own the session."""
        late_pending = self._pending_messages.pop(session_key, None)
        current_task = asyncio.current_task()
        if late_pending is not None:
            existing_task = self._session_tasks.get(session_key)
            if existing_task is not None and existing_task is not current_task:
                self._pending_messages[session_key] = late_pending
            else:
                logger.debug(
                    "[%s] Late-arrival pending message during cleanup — spawning drain task",
                    self.name)
                self._spawn_drain_task(late_pending, session_key)
        elif current_task is not None and self._session_tasks.get(session_key) is current_task:
            self._cleanup_finished_session_task(session_key, interrupt_event)

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Background task that actually processes the message."""
        delivery_attempted = delivery_succeeded = False  # feeds the processing-complete hook

        def _record_delivery(result):
            nonlocal delivery_attempted, delivery_succeeded
            if result is not None:
                delivery_attempted = True
                delivery_succeeded = delivery_succeeded or bool(getattr(result, "success", False))
        # Reuse the interrupt event handle_message() installed; new Event only if removed externally.
        interrupt_event = self._active_sessions.get(session_key) or asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        _thread_metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
        typing_task = self._start_typing_refresh(event, interrupt_event, _thread_metadata)
        try:
            await self._run_processing_hook("on_processing_start", event)
            response = await self._message_handler(event)
            is_ephemeral_response = isinstance(response, EphemeralReply)
            # Unwrap EphemeralReply for downstream text processing; TTL applies after send.
            response, _ephemeral_ttl = self._unwrap_ephemeral(response)
            # None/empty is normal (streamed/queued). Suppress a stale response when the
            # session was interrupted by a still-pending message.
            if (response and interrupt_event.is_set() and session_key in self._pending_messages):
                logger.info(
                    "[%s] Suppressing stale response for interrupted session %s",
                    self.name, session_key)
                response = None
            if not response:
                logger.debug("[%s] Handler returned empty/None response for %s", self.name, event.source.chat_id)
            else:
                extracted = await self._extract_response_content(
                    response, event, session_key, is_ephemeral_response=is_ephemeral_response)
                text_content, media_files = extracted.text_content, extracted.media_files
                # Final user-visible content gets notify=True; typing/status
                # metadata stays unmarked so progress bubbles remain thread-strict.
                _final_thread_metadata = _mark_notify_metadata(_thread_metadata)
                _tts_paths, _tts_requested_path = (
                    await self._synthesize_auto_tts(text_content)
                    if self._wants_auto_tts(event, session_key, interrupt_event, text_content, media_files)
                    else ([], None))
                # TTS plays before text; generated files are removed afterwards.
                _tts_caption_delivered = False
                for _tts_index, _tts_path in enumerate(_tts_paths):
                    try:
                        _tts_caption_delivered |= await self._play_tts_file(
                            event, text_content, _tts_path, _tts_index == 0, _final_thread_metadata,
                            _record_delivery)
                    finally:
                        try:
                            os.remove(_tts_path)
                        except OSError:
                            pass
                if not _tts_paths and _tts_requested_path is not None:
                    try:
                        os.remove(_tts_requested_path)
                    except OSError:
                        pass
                if text_content and not _tts_caption_delivered:
                    await self._send_final_text(
                        event, session_key, text_content, _final_thread_metadata,
                        is_ephemeral_response, _ephemeral_ttl, _record_delivery)
                await self._deliver_attachments(
                    event, extracted, _final_thread_metadata,
                    anything_sent=delivery_attempted or _tts_caption_delivered)
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            # Clean up the per-turn streaming-TTS flag.
            self._streaming_tts_completed_turns.discard(
                self._streaming_tts_turn_key(
                    session_key, getattr(interrupt_event, "_hermes_run_generation", None),
                    event=event) or "")
            await self._run_processing_hook(
                "on_processing_complete", event,
                ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE)
            # The active drain owns debounce state: force-flush an unfired timer so this
            # task hands off the follow-up to a fresh drain task. Clear the Event BEFORE
            # the stop-typing await so a concurrent inbound still sees a live guard.
            await self._flush_text_debounce_now(session_key)
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                logger.debug("[%s] Processing queued follow-up message", self.name)
                _active = self._active_sessions.get(session_key)
                if _active is not None:
                    _active.clear()
                await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
                self._spawn_drain_task(pending_event, session_key)
                return  # Drain task owns the session now.
        except asyncio.CancelledError:
            outcome = (
                ProcessingOutcome.CANCELLED if asyncio.current_task() in self._expected_cancelled_tasks
                else ProcessingOutcome.FAILURE)
            await self._run_processing_hook("on_processing_complete", event, outcome)
            raise
        except BaseException as e:
            await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)
            _thread_metadata = (await self._notify_turn_error(event, e)) or _thread_metadata
            # SystemExit/KeyboardInterrupt must propagate; other BaseExceptions are
            # contained so this task never logs "exception was never retrieved".
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
        finally:
            # Stop typing BEFORE the post-delivery callback: a stuck callback must not
            # leave the refresh running indefinitely.
            await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
            await self._fire_post_delivery_callback(session_key, interrupt_event)
            # Callback work or a late refresh may have recreated platform typing — one
            # final bounded stop before releasing the guard.
            await self._stop_typing_refresh(
                event.source.chat_id, None, metadata=_thread_metadata, stop_attempts=1)
            # Flush any timer that missed the in-band drain, then reconcile ownership.
            await self._flush_text_debounce_now(session_key)
            self._finish_session_task(session_key, interrupt_event)

    def _spawn_drain_task(self, pending_event: MessageEvent, session_key: str) -> None:
        """Hand the session to a fresh task for a queued follow-up — never recurse (chained
        follow-ups grew the C stack to SIGSEGV). Clearing, not deleting, the Event keeps
        the guard live for concurrent inbound; ownership moves so stale-lock detection works."""
        _active = self._active_sessions.get(session_key)
        if _active is not None:
            _active.clear()
        drain_task = asyncio.create_task(self._process_message_background(pending_event, session_key))
        self._session_tasks[session_key] = drain_task
        try:
            self._background_tasks.add(drain_task)
            drain_task.add_done_callback(self._background_tasks.discard)
        except TypeError:
            pass  # Tests stub create_task() with non-hashable sentinels; tolerate.

    def _cleanup_finished_session_task(
        self, session_key: str, interrupt_event: Optional[asyncio.Event]) -> None:
        """Release a finished owner task's guard, dropping its ``_session_tasks`` entry
        ONLY if the guard was actually released: if a concurrent path swapped guards,
        the done-task entry lets ``_session_task_is_stale`` heal the orphan."""
        self._release_session_guard(session_key, guard=interrupt_event)
        if session_key not in self._active_sessions:
            self._session_tasks.pop(session_key, None)

    async def cancel_background_tasks(self) -> None:
        """Cancel in-flight background tasks (shutdown/replacement); 5s bound each,
        stragglers are untracked and left to unwind."""
        # Re-drain (max 5 rounds) until the task set stabilizes: a message arriving mid-gather
        # would spawn a task the final clear() would untrack.
        for _ in range(5):
            tasks = [task for task in self._background_tasks if not task.done()]
            if not tasks:
                break
            for task in tasks:
                self._expected_cancelled_tasks.add(task)
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True),
                    timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] %d background task(s) did not exit within 5s; "
                    "releasing tracking and letting them unwind in the background",
                    self.name, len([t for t in tasks if not t.done()]))
                break
        self._background_tasks.clear()
        self._expected_cancelled_tasks.clear()
        self._session_tasks.clear()
        # Flush pending messages to disk before clearing.
        try:
            from gateway.shutdown_flush import flush_pending_to_file
            flush_pending_to_file(self._pending_messages, reason="adapter_shutdown")
        except Exception:
            pass
        self._pending_messages.clear()
        self._active_sessions.clear()
        for state in list(self._text_debounce_store().values()):
            state.cancel_timer()
        self._text_debounce_store().clear()

    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()

    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        return self._pending_messages.pop(session_key, None)

    def build_source(
        self, chat_id: str, chat_name: Optional[str] = None, chat_type: str = "dm",
        user_id: Optional[str] = None, user_name: Optional[str] = None,
        thread_id: Optional[str] = None, chat_topic: Optional[str] = None,
        user_id_alt: Optional[str] = None, chat_id_alt: Optional[str] = None, is_bot: bool = False,
        scope_id: Optional[str] = None, guild_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None, message_id: Optional[str] = None,
        role_authorized: bool = False, auto_thread_created: bool = False,
        auto_thread_initial_name: Optional[str] = None) -> SessionSource:
        """Build a SessionSource; with ``gateway.profile_routes`` configured the matching
        profile is stamped on ``source.profile`` for per-profile HERMES_HOME isolation."""
        def _opt(value) -> Optional[str]:
            return str(value) if value else None
        fields = dict(
            platform=self.platform, chat_id=str(chat_id), chat_name=chat_name, chat_type=chat_type,
            user_id=_opt(user_id), user_name=user_name, thread_id=_opt(thread_id),
            chat_topic=(chat_topic or "").strip() or None,
            user_id_alt=user_id_alt, chat_id_alt=chat_id_alt, is_bot=is_bot,
            scope_id=_opt(scope_id), guild_id=_opt(guild_id), parent_chat_id=_opt(parent_chat_id),
            message_id=_opt(message_id))
        profile = None  # from configured routes; None when no match / no routes
        profile_route_rejected = False
        runner = getattr(self, "gateway_runner", None)
        if runner is not None:
            from gateway.profile_routing import ProfileRouteRejected
            try:
                profile = runner._profile_name_for_source(SessionSource(**fields))
            except ProfileRouteRejected:
                profile_route_rejected = True
            except Exception:
                logger.warning(
                    "Profile resolution failed for %s/%s, defaulting to active profile",
                    self.platform, chat_id, exc_info=True)
        source = SessionSource(
            **fields, profile=profile, role_authorized=role_authorized,
            auto_thread_created=auto_thread_created,
            auto_thread_initial_name=auto_thread_initial_name)
        # Both transport-only, kept out of to_dict(): the live receiving adapter is
        # authoritative this turn even if profile_routes picks another runtime; the
        # fail-closed reject flag is consumed before auth so rejected routes never 500.
        source._transport_adapter_ref = weakref.ref(self)
        source.profile_route_rejected = profile_route_rejected
        return source

    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/channel; dict with at least ``name``
        and ``type`` ("dm", "group", "channel")."""

    def toolsets_for_source(self, source: "SessionSource") -> Optional[List[str]]:
        """Per-source toolset override REPLACING ``platform_toolsets.<platform>``, or None
        (default); validated via ``_get_platform_tools`` (webhook adapter pins per-route)."""
        return None

    def format_message(self, content: str) -> str:
        """Format a message for this platform (override for e.g. Telegram
        MarkdownV2); default returns content as-is."""
        return content

    @staticmethod
    def truncate_message(
        content: str, max_length: int = 4096, len_fn: Optional["Callable[[str], int]"] = None,
    ) -> List[str]:
        """Split a long message into chunks, preserving code-block boundaries: a split
        inside a fence closes it at the chunk end and reopens it (same language tag) in
        the next; multi-chunk output gets ``(1/3)`` indicators. ``len_fn`` overrides
        ``len`` (``utf16_len`` for Telegram, which counts UTF-16 code units)."""
        _len = len_fn or len
        if _len(content) <= max_length:
            return [content]
        INDICATOR_RESERVE = 10   # room for " (XX/XX)"
        FENCE_CLOSE = "\n```"
        chunks: List[str] = []
        remaining = content
        carry_lang: Optional[str] = None  # language tag ("" ok) when previous chunk ended mid-fence
        while remaining:
            prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
            # Body budget after prefix, closing fence and indicator; floored so a tiny
            # max_length (0/1 from a relay capability descriptor) can't stall the loop.
            headroom = max_length - INDICATOR_RESERVE - _len(prefix) - _len(FENCE_CLOSE)
            if headroom < 1:
                headroom = max(1, max_length // 2)
            # Remainder fits in one final chunk; close a reopened fence if still open.
            if _len(prefix) + _len(remaining) <= max_length - INDICATOR_RESERVE:
                final_chunk = prefix + remaining
                if carry_lang is not None and fence_state_after(remaining, True, carry_lang)[0]:
                    final_chunk += FENCE_CLOSE
                chunks.append(final_chunk)
                break
            # Natural split (newline, then space). A custom _len measures headroom in its
            # own units: map to the largest codepoint offset that fits the budget.
            _cp_limit = (
                _custom_unit_to_cp(remaining, headroom, _len) if _len is not len else headroom)
            region = remaining[:_cp_limit]
            split_at = region.rfind("\n")
            if split_at < _cp_limit // 2:
                split_at = region.rfind(" ")
            if split_at < 1:
                # Floor at one codepoint: a zero _cp_limit (max_length 0/1, or a surrogate
                # pair wider than the utf16 budget) would never shrink ``remaining`` and
                # spin forever; exceeding max_length by one codepoint beats a hang.
                split_at = max(1, _cp_limit)
            # Don't split inside an inline code span: an unpaired backtick breaks MarkdownV2.
            candidate = remaining[:split_at]
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    safe_split = max(
                        candidate.rfind(" ", 0, last_bt), candidate.rfind("\n", 0, last_bt))
                    if safe_split > _cp_limit // 4:
                        split_at = safe_split
            chunk_body = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()
            full_chunk = prefix + chunk_body
            # Walk only chunk_body (not the prepended prefix) for the fence state.
            in_code, lang = fence_state_after(chunk_body, carry_lang is not None, carry_lang or "")
            if in_code:
                full_chunk += FENCE_CLOSE  # Close the orphaned fence so the chunk stands alone
                carry_lang = lang
            else:
                carry_lang = None
            chunks.append(full_chunk)
        if len(chunks) > 1:
            total = len(chunks)
            chunks = [f"{chunk} ({i + 1}/{total})" for i, chunk in enumerate(chunks)]
        return chunks
