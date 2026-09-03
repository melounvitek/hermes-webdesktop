"""Standalone per-platform senders and error helpers for send_message."""

import asyncio
import logging
import os
import re
import time

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.send_message_tool")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".m2a", ".wav", ".m4a", ".flac"}
_VOICE_EXTS = {".ogg", ".opus"}
# Telegram's sendAudio only accepts MP3 / M4A; other audio goes via sendVoice
# (Opus/OGG) or falls back to document delivery.
_TELEGRAM_SEND_AUDIO_EXTS = {".mp3", ".m4a"}

# Extensions that carry a native caption on the media bubble (photo/video/document).
# Voice/audio notes are excluded: a caption on a voice note reads as a separate label,
# so the accompanying text stays its own message.
_CAPTIONABLE_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx", ".zip",
}

# Native caption length limits (characters). Telegram's photo/video cap is 1024;
# WhatsApp/Discord are far more generous, so one conservative shared ceiling.
_TELEGRAM_CAPTION_LIMIT = 1024
_DEFAULT_CAPTION_LIMIT = 4096


def _media_caption_split(text, media_files, *, max_caption_len):
    """Decide whether the accompanying text rides on the media bubble.

    Single chokepoint for ``MEDIA:<path> caption`` across every sender. Returns
    ``(caption, "")`` — text becomes the native caption, no separate body — only for
    exactly one captionable file (not a voice/audio note) whose text fits
    ``max_caption_len``. Otherwise ``(None, text)``: text goes as a separate message
    (multi-file caption→file association is ambiguous).

    Length is measured in codepoints: never under-counts Telegram's UTF-16 units for
    BMP text, so over-counting only fails safe. The Telegram sender re-checks the
    *formatted* caption since escaping inflates it.
    """
    stripped = (text or "").strip()
    media = media_files or []
    if not stripped or len(media) != 1:
        return None, text
    media_path, is_voice = media[0]
    if is_voice or os.path.splitext(media_path)[1].lower() not in _CAPTIONABLE_EXTS:
        return None, text
    if len(stripped) > max_caption_len:
        return None, text
    return stripped, ""


_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)",
    re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\s*=\s*([^\s,;]+)",
    re.IGNORECASE,
)


def _sanitize_error_text(text) -> str:
    """Redact secrets from error text before surfacing it to users/models."""
    redacted = redact_sensitive_text(text)
    redacted = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", redacted)
    return _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", redacted)


def _error(message: str) -> dict:
    """Build a standardized error payload with redacted content."""
    return {"error": _sanitize_error_text(message)}


def _success(platform: str, chat_id, warnings=None, **fields) -> dict:
    """Standard success payload; ``warnings`` is only included when non-empty."""
    result = {"success": True, "platform": platform, "chat_id": chat_id, **fields}
    if warnings:
        result["warnings"] = warnings
    return result


def _display_chat_id(platform_name: str, chat_id: str) -> str:
    """Return a result-safe chat identifier for tool transcripts/log consumers."""
    if platform_name == "signal" and str(chat_id).startswith("group:"):
        return "group:***"
    return chat_id


_NO_DELIVERABLE = "No deliverable text or media remained after processing MEDIA tags"

_TELEGRAM_TRANSIENT_MARKERS = (
    "bad gateway", "502", "too many requests", "429",
    "service unavailable", "503", "gateway timeout", "504",
)


def _telegram_retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when the error is final.
    Honours Telegram's ``retry_after``; timeouts are never retried (the send
    may have gone through); 5xx/429 back off exponentially."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return 1.0

    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return None
    if any(marker in text for marker in _TELEGRAM_TRANSIENT_MARKERS):
        return float(2 ** attempt)
    return None


async def _send_telegram_message_with_retry(bot, *, attempts: int = 3, **kwargs):
    """``bot.send_message`` with bounded retries on transient failures."""
    for attempt in range(attempts):
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:
            delay = _telegram_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts - 1:
                raise
            logger.warning(
                "Transient Telegram send failure (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, attempts, delay, _sanitize_error_text(exc),
            )
            await asyncio.sleep(delay)


def _is_telegram_thread_not_found(error: Exception) -> bool:
    """Mirror of the gateway adapter's ``_is_thread_not_found_error``."""
    return "thread not found" in str(error).lower()


def _telegram_bot(token):
    """Build a Bot honouring TELEGRAM_PROXY (config ``telegram.proxy_url``);
    without it the standalone path bypasses the proxy and times out where
    api.telegram.org is blocked. Falls back to a direct connection."""
    from telegram import Bot

    try:
        from gateway.platforms.base import resolve_proxy_url
        proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
    except Exception:
        proxy = None
    if proxy:
        try:
            from telegram.request import HTTPXRequest
            logger.info("send_message: standalone Telegram send routed through proxy %s", proxy)
            return Bot(
                token=token,
                request=HTTPXRequest(proxy=proxy),
                get_updates_request=HTTPXRequest(proxy=proxy),
            )
        except Exception as proxy_err:
            logger.warning("send_message: failed to attach Telegram proxy (%s), falling back to direct connection", proxy_err)
    return Bot(token=token)


def _telegram_thread_kwargs(thread_id):
    """Map a topic id to ``message_thread_id`` kwargs. Forum "General" is
    thread "1" on incoming updates but Bot API rejects message_thread_id=1
    ("Message thread not found"), so it maps to no thread — same as the adapter."""
    if thread_id is None:
        return {}
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        effective = TelegramAdapter._message_thread_id_for_send(str(thread_id))
    except Exception:
        # Explicit mapping if the adapter import fails (python-telegram-bot missing).
        effective = None if str(thread_id) == "1" else int(thread_id)
    return {} if effective is None else {"message_thread_id": effective}


def _strip_mdv2_safe(text):
    """Strip MarkdownV2 escapes for the plain-text fallback; identity if unavailable."""
    try:
        from plugins.platforms.telegram.adapter import _strip_mdv2
        return _strip_mdv2(text)
    except Exception:
        return text


async def _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **kwargs):
    """Pick the Bot API media method by extension: photo (unless forced to
    document), video, voice note, sendAudio (MP3/M4A only), else document."""
    if ext in _IMAGE_EXTS and not force_document:
        return await bot.send_photo(chat_id=chat_id, photo=f, **kwargs)
    if ext in _VIDEO_EXTS:
        return await bot.send_video(chat_id=chat_id, video=f, **kwargs)
    if ext in _VOICE_EXTS and is_voice:
        return await bot.send_voice(chat_id=chat_id, voice=f, **kwargs)
    if ext in _TELEGRAM_SEND_AUDIO_EXTS:
        return await bot.send_audio(chat_id=chat_id, audio=f, **kwargs)
    return await bot.send_document(chat_id=chat_id, document=f, **kwargs)


async def _telegram_send_text_chunk(bot, chat_id, chunk, parse_mode, has_html, text_kwargs):
    """Send one formatted text chunk with the adapter-matching fallbacks:
    thread-not-found -> retry without ``message_thread_id`` (dropped from
    ``text_kwargs`` for later chunks too); parse failure -> plain text."""
    async def send(text, mode):
        return await _send_telegram_message_with_retry(
            bot, chat_id=chat_id, text=text, parse_mode=mode, **text_kwargs)

    try:
        return await send(chunk, parse_mode)
    except Exception as md_error:
        if _is_telegram_thread_not_found(md_error) and text_kwargs.get("message_thread_id") is not None:
            logger.warning(
                "Thread %s not found in _send_telegram, retrying without message_thread_id",
                text_kwargs.get("message_thread_id"),
            )
            text_kwargs.pop("message_thread_id", None)
            return await send(chunk, parse_mode)
        err_text = str(md_error).lower()
        if "parse" in err_text or "markdown" in err_text or "html" in err_text:
            logger.warning(
                "Parse mode %s failed in _send_telegram, falling back to plain text: %s",
                parse_mode, _sanitize_error_text(md_error),
            )
            return await send(chunk if has_html else _strip_mdv2_safe(chunk), None)
        raise


async def _telegram_send_one_media(
    bot, chat_id, media_path, is_voice, *, caption, parse_mode, has_html, thread_kwargs, force_document
):
    """Upload one file with the adapter-matching fallbacks (thread-not-found ->
    no ``message_thread_id``; caption parse failure -> plain caption). Each
    retry re-seeks the file because the first attempt consumed it."""
    ext = os.path.splitext(media_path)[1].lower()
    voice_note = ext in _VOICE_EXTS and is_voice
    with open(media_path, "rb") as f:
        media_kwargs = dict(thread_kwargs)
        # ``caption`` is only set for a single captionable file, so this never
        # double-captions a multi-file send or a voice note.
        if caption is not None and not voice_note:
            media_kwargs["caption"] = caption
            media_kwargs["parse_mode"] = parse_mode
        if voice_note or ext in _TELEGRAM_SEND_AUDIO_EXTS:
            try:
                from plugins.platforms.telegram.adapter import _probe_voice_duration_seconds
                duration = await asyncio.to_thread(_probe_voice_duration_seconds, media_path)
                if duration is not None:
                    media_kwargs["duration"] = duration
            except Exception:
                pass

        async def _send(**kw):
            return await _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **kw)

        try:
            return await _send(**media_kwargs)
        except Exception as media_err:
            if _is_telegram_thread_not_found(media_err) and media_kwargs.get("message_thread_id"):
                logger.warning(
                    "Thread %s not found for media send, retrying without message_thread_id",
                    media_kwargs["message_thread_id"],
                )
                f.seek(0)
                media_kwargs.pop("message_thread_id", None)
                return await _send(**media_kwargs)
            err_text = str(media_err).lower()
            if media_kwargs.get("parse_mode") and ("parse" in err_text or "caption" in err_text):
                logger.warning(
                    "Caption parse failed for media send, retrying plain: %s",
                    _sanitize_error_text(media_err),
                )
                f.seek(0)
                media_kwargs.pop("parse_mode", None)
                if not has_html and media_kwargs.get("caption"):
                    media_kwargs["caption"] = _strip_mdv2_safe(media_kwargs["caption"])
                return await _send(**media_kwargs)
            raise


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False, force_document=False):
    """Send via Telegram Bot API (one-shot, no polling needed).

    Markdown is converted to MarkdownV2 via the gateway adapter's ``format_message``
    so bold/links/headers render; a message that already contains HTML tags skips
    that and is sent with ``parse_mode='HTML'``. Parse failures fall back to plain
    text so the message still delivers.
    """
    try:
        from telegram.constants import ParseMode

        # Auto-detect HTML tags: if present, skip MarkdownV2 and send as HTML.
        _has_html = bool(re.search(r'<[a-zA-Z/][^>]*>', message))
        if _has_html:
            formatted = message
            send_parse_mode = ParseMode.HTML
        else:
            try:
                from plugins.platforms.telegram.adapter import TelegramAdapter
                formatted = TelegramAdapter.__new__(TelegramAdapter).format_message(message)
            except Exception:
                formatted = message  # formatting unavailable: send as-is
            send_parse_mode = ParseMode.MARKDOWN_V2

        bot = _telegram_bot(token)
        from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

        # Telegram accepts a numeric chat_id OR an @username string; never force-int.
        int_chat_id = normalize_telegram_chat_id(chat_id)
        media_files = media_files or []
        thread_kwargs = _telegram_thread_kwargs(thread_id)
        # disable_web_page_preview is only valid for send_message, not media sends.
        text_kwargs = dict(thread_kwargs)
        if disable_link_previews:
            text_kwargs["disable_web_page_preview"] = True

        last_msg = None
        warnings = []

        # MEDIA caption: a single captionable file + short text rides on the bubble as
        # its *formatted* caption. Formatting can inflate a raw <1024 string past
        # Telegram's cap, so re-check in UTF-16 units and fall back to a separate body.
        _tg_caption = None
        from gateway.platforms.base import BasePlatformAdapter, utf16_len
        _cap, _ = _media_caption_split(message, media_files, max_caption_len=_TELEGRAM_CAPTION_LIMIT)
        if _cap is not None and utf16_len(formatted) <= _TELEGRAM_CAPTION_LIMIT:
            _tg_caption = formatted
            formatted = ""  # suppress the separate text send below

        if formatted.strip():
            # Chunk *after* formatting, in UTF-16 units: MarkdownV2/HTML escaping inflates
            # text, so a raw-<4096 message can exceed the limit once formatted.
            for chunk in BasePlatformAdapter.truncate_message(formatted, 4096, len_fn=utf16_len):
                last_msg = await _telegram_send_text_chunk(
                    bot, int_chat_id, chunk, send_parse_mode, _has_html, text_kwargs)

        for media_path, is_voice in media_files:
            if not os.path.exists(media_path):
                warning = f"Media file not found, skipping: {media_path}"
                logger.warning(warning)
                warnings.append(warning)
                # Caption mode suppressed the text send; if the file it was meant to
                # caption is gone, deliver the words on their own.
                if _tg_caption is not None and last_msg is None:
                    try:
                        last_msg = await _send_telegram_message_with_retry(
                            bot, chat_id=int_chat_id, text=_tg_caption,
                            parse_mode=send_parse_mode, **text_kwargs
                        )
                        _tg_caption = None  # delivered — don't re-caption a later file
                    except Exception as _cap_err:
                        logger.warning(
                            "Telegram caption-fallback send failed for missing media: %s",
                            _sanitize_error_text(_cap_err),
                        )
                continue

            try:
                last_msg = await _telegram_send_one_media(
                    bot, int_chat_id, media_path, is_voice,
                    caption=_tg_caption, parse_mode=send_parse_mode, has_html=_has_html,
                    thread_kwargs=thread_kwargs, force_document=force_document,
                )
            except Exception as e:
                warning = _sanitize_error_text(f"Failed to send media {media_path}: {e}")
                logger.error(warning)
                warnings.append(warning)

        if last_msg is None:
            if warnings:
                return {"error": _NO_DELIVERABLE, "warnings": warnings}
            return {"error": _NO_DELIVERABLE}
        return _success("telegram", chat_id, warnings, message_id=str(last_msg.message_id))
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return _error(f"Telegram send failed: {e}")


def _live_runner():
    """Return the in-process gateway runner, or None (standalone/cron)."""
    try:
        from gateway.run import _gateway_runner_ref
        return _gateway_runner_ref()
    except Exception:
        return None


def _plugin_standalone_sender(platform_name, *, label=None, discover=True):
    """Return ``(standalone_sender_fn, None)`` for a registered platform plugin,
    or ``(None, error_dict)``. ``discover`` runs the idempotent plugin scan first."""
    from gateway.platform_registry import platform_registry

    if discover:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    entry = platform_registry.get(platform_name)
    if entry is None or entry.standalone_sender_fn is None:
        return None, {"error": f"{label or platform_name} plugin not registered or missing standalone_sender_fn"}
    return entry.standalone_sender_fn, None


async def _registry_standalone_send(platform_name, pconfig, chat_id, message, thread_id=None):
    """One-shot text send through a migrated plugin's ``standalone_sender_fn``
    (the former inline ``_send_<platform>`` helpers now live in the plugins)."""
    sender, err = _plugin_standalone_sender(platform_name)
    if err:
        return err
    return await sender(pconfig, chat_id, message, thread_id=thread_id)


async def _resolve_slack_user_target(token, chat_id):
    """Resolve ``user:U...`` / ``user_name:<handle>`` to a D... DM conversation.

    chat.postMessage needs a conversation ID, so user targets go through
    conversations.open; ``user_name:`` first maps to a user ID via users.list
    (stable handle match only). Other ids pass through unchanged.
    Returns ``(chat_id, None)`` or ``(None, error_dict)``.
    """
    if not (chat_id.startswith("user:") or chat_id.startswith("user_name:")):
        return chat_id, None
    try:
        import aiohttp
    except ImportError:
        return None, {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url()
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        base_url = "https://slack.com/api"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async def post_api(session, method, payload):
            async with session.post(f"{base_url}/{method}", headers=headers, json=payload, **_req_kw) as resp:
                return await resp.json()

        async def resolve_user_name(session, name):
            query = name.strip().lstrip("@").lower()
            matches = []
            cursor = None
            for _page in range(20):
                payload = {"limit": 200}
                if cursor:
                    payload["cursor"] = cursor
                data = await post_api(session, "users.list", payload)
                if not data.get("ok"):
                    return None, f"Slack users.list error: {data.get('error', 'unknown')}"
                for member in data.get("members", []):
                    if member.get("deleted") or member.get("is_bot"):
                        continue
                    # Match the stable handle only: display/real names are
                    # mutable and non-unique enough to DM the wrong person.
                    if str(member.get("name", "")).strip().lower() == query:
                        matches.append(member)
                cursor = (data.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
            if not matches:
                return None, f"Could not resolve Slack user '@{name}'."
            if len(matches) > 1:
                return None, f"Slack user '@{name}' matched multiple Slack users. Use a Slack user ID instead."
            return matches[0].get("id"), None

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            if chat_id.startswith("user_name:"):
                user_id, error = await resolve_user_name(session, chat_id[len("user_name:"):])
                if error:
                    return None, _error(error)
                chat_id = f"user:{user_id}"

            user_id = chat_id[len("user:"):]
            opened = await post_api(session, "conversations.open", {"users": user_id})
            if not opened.get("ok"):
                return None, _error(
                    f"Slack conversations.open error: {opened.get('error', 'unknown')}. "
                    "Check bot permissions (im:write)."
                )
            dm_id = (opened.get("channel") or {}).get("id")
            if not dm_id:
                return None, _error("Slack conversations.open did not return a DM channel ID")
            return dm_id, None
    except Exception as e:
        return None, _error(f"Slack DM resolution failed: {e}")


async def _signal_send_batch(post, scheduler, rl, idx, n_batches, att_batch, batch_message):
    """Send one Signal batch under the scheduler with rate-limit retries.
    Returns None on success, False when retries were exhausted (batch lost),
    or an error dict for a non-rate-limit RPC error."""
    n = len(att_batch)
    for attempt in range(1, rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            await scheduler.acquire(n)
            _rpc_t0 = time.monotonic()
            data = await post(att_batch, batch_message)
            _rpc_duration = time.monotonic() - _rpc_t0
            if "error" not in data:
                await scheduler.report_rpc_duration(_rpc_duration, n)
                return None

            err = data["error"]
            if not rl._is_signal_rate_limit_error(err):
                return _error(f"Signal RPC error on batch {idx + 1}/{n_batches}: {err}")

            server_retry_after = rl._extract_retry_after_seconds(err)
            scheduler.feedback(server_retry_after, n)
            retry_after_label = f"{server_retry_after:.0f}s" if server_retry_after else "unknown"

            if attempt >= rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                logger.error(
                    "Signal: rate-limit retries exhausted on batch %d/%d "
                    "(%d attachments lost, server retry_after=%s)",
                    idx + 1, n_batches, n, retry_after_label,
                )
                return False
            logger.warning(
                "Signal: rate-limited on batch %d/%d "
                "(attempt %d/%d, server retry_after=%s); "
                "scheduler will pace the retry",
                idx + 1, n_batches, attempt, rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS, retry_after_label,
            )
        except Exception as e:
            if attempt >= rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                logger.error(
                    "Signal: send error on batch %d/%d after %d attempts: %s",
                    idx + 1, n_batches, attempt, str(e)
                )
                return False
            logger.warning(
                "Signal: transient error on batch %d/%d (attempt %d/%d): %s; will retry",
                idx + 1, n_batches, attempt, rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS, str(e)
            )


async def _send_signal(extra, chat_id, message, media_files=None):
    """Send via signal-cli JSON-RPC, text and/or attachments.

    Attachments go in batches of SIGNAL_MAX_ATTACHMENTS_PER_MSG metered by the
    process-wide SignalAttachmentScheduler — the same bucket the gateway
    adapter uses, so tool sends and inbound replies share rate-limit state.
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}

    from gateway.platforms import signal_rate_limit as rl
    from gateway.platforms.signal_format import markdown_to_signal

    try:
        http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        account = extra.get("account", "")
        if not account:
            return {"error": "Signal account not configured"}

        valid_media = media_files or []
        attachment_paths = []
        for media_path, _is_voice in valid_media:
            if os.path.exists(media_path):
                attachment_paths.append(media_path)
            else:
                logger.warning("Signal media file not found, skipping: %s", media_path)

        # No attachments still means one (text-only) batch; with attachments
        # the text rides on batch #0 so it isn't repeated per batch.
        per_batch = rl.SIGNAL_MAX_ATTACHMENTS_PER_MSG
        att_batches = [
            attachment_paths[i:i + per_batch] for i in range(0, len(attachment_paths), per_batch)
        ] or [[]]
        n_batches = len(att_batches)

        plain_text, text_styles = markdown_to_signal(message)

        def _rpc_params(text):
            params = {"account": account, "message": text}
            if chat_id.startswith("group:"):
                params["groupId"] = chat_id[6:]
            else:
                params["recipient"] = [chat_id]
            return params

        async def _rpc_send(params, *, id_prefix, timeout):
            payload = {
                "jsonrpc": "2.0",
                "method": "send",
                "params": params,
                "id": f"{id_prefix}_{int(time.time() * 1000)}",
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(f"{http_url}/api/v1/rpc", json=payload)

        async def _post(batch_attachments, batch_message):
            params = _rpc_params(batch_message)
            if batch_message and text_styles:
                if len(text_styles) == 1:
                    params["textStyle"] = text_styles[0]
                else:
                    params["textStyles"] = text_styles
            if batch_attachments:
                params["attachments"] = batch_attachments
            timeout = rl._signal_send_timeout(len(batch_attachments) if batch_attachments else 0)
            resp = await _rpc_send(params, id_prefix="send", timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        scheduler = rl.get_scheduler()
        logger.info(
            "send_message Signal: scheduler state=%s, %d attachment(s) in %d batch(es)",
            scheduler.state(), len(attachment_paths), n_batches,
        )
        failed_batches: list[int] = []
        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            if n > 0:
                estimated = scheduler.estimate_wait(n)
                if estimated >= rl.SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                    # Best-effort one-shot RPC for a user-facing pacing notice.
                    notice = (
                        f"(More images coming — pausing ~{rl._format_wait(estimated)} "
                        f"for Signal rate limit, batch {idx + 1}/{n_batches}.)"
                    )
                    try:
                        await _rpc_send(_rpc_params(notice), id_prefix="notice", timeout=30.0)
                    except Exception as _e:
                        logger.warning("Signal: inline notice failed: %s", _e)

            outcome = await _signal_send_batch(
                _post, scheduler, rl, idx, n_batches, att_batch, plain_text if idx == 0 else "")
            if outcome is False:
                failed_batches.append(idx + 1)
            elif outcome is not None:
                return outcome

        warnings = []
        if len(attachment_paths) < len(valid_media):
            warnings.append("Some media files were skipped (not found on disk)")
        if failed_batches:
            warnings.append(
                f"Signal rate-limited {len(failed_batches)} batch(es) "
                f"(#{', #'.join(str(b) for b in failed_batches)})"
            )

        if failed_batches and len(failed_batches) == n_batches:
            return _error(
                f"Signal: every batch ({n_batches}) hit rate limit; "
                f"no attachments delivered"
            )

        return _success("signal", _display_chat_id("signal", chat_id), warnings)
    except Exception as e:
        return _error(f"Signal send failed: {e}")


async def _send_matrix_via_adapter(pconfig, chat_id, message, media_files=None, thread_id=None):
    """Send via the Matrix adapter so native media uploads are preserved.

    Prefer the live gateway adapter: one persistent olm/megolm session for all sends.
    Ephemeral per-send connects re-init E2EE and claim one-time keys, which under
    bursts exhausts recipient OTKs and silently drops messages — so the ephemeral
    connect/disconnect path is only for standalone/cron.
    """
    media_files = media_files or []
    metadata = {"thread_id": thread_id} if thread_id else None

    # A runner that exists but whose adapter lookup fails is logged, not
    # swallowed: a silent fall-through would recreate the reconnect storm.
    live_adapter = None
    runner = _live_runner()
    if runner is not None:
        try:
            from gateway.config import Platform
            live_adapter = runner.adapters.get(Platform.MATRIX)
        except Exception:
            logger.warning(
                "Matrix: live gateway adapter lookup failed; falling back to an "
                "ephemeral connect (may re-init E2EE per send)",
                exc_info=True,
            )
            live_adapter = None

    if live_adapter is not None:
        # Owned by the gateway — must NOT be disconnected; return before the
        # ephemeral adapter (and its ``finally`` disconnect) exists.
        return await _matrix_send_core(live_adapter, chat_id, message, media_files, metadata)

    # --- Fallback: ephemeral adapter (standalone / cron context) ---
    try:
        from plugins.platforms.matrix.adapter import MatrixAdapter
    except ImportError:
        return {"error": "Matrix dependencies not installed. Run: pip install 'mautrix[encryption]'"}

    adapter = MatrixAdapter(pconfig)
    try:
        connected = await adapter.connect()
        if not connected:
            return _error("Matrix connect failed")
        return await _matrix_send_core(adapter, chat_id, message, media_files, metadata)
    except Exception as e:
        return _error(f"Matrix send failed: {e}")
    finally:
        try:
            await adapter.disconnect()
        except Exception:
            pass


async def _matrix_send_core(adapter, chat_id, message, media_files, metadata):
    """Core send logic shared by live and ephemeral Matrix adapters."""
    last_result = None

    if message.strip():
        last_result = await adapter.send(chat_id, message, metadata=metadata)
        if not last_result.success:
            return _error(f"Matrix send failed: {last_result.error}")

    for media_path, is_voice in media_files:
        if not os.path.exists(media_path):
            return _error(f"Media file not found: {media_path}")

        ext = os.path.splitext(media_path)[1].lower()
        if ext in _IMAGE_EXTS:
            method = adapter.send_image_file
        elif ext in _VIDEO_EXTS:
            method = adapter.send_video
        elif (ext in _VOICE_EXTS and is_voice) or ext in _AUDIO_EXTS:
            method = adapter.send_voice
        else:
            method = adapter.send_document
        last_result = await method(chat_id, media_path, metadata=metadata)

        if not last_result.success:
            return _error(f"Matrix media send failed: {last_result.error}")

    if last_result is None:
        return {"error": _NO_DELIVERABLE}
    return _success("matrix", chat_id, message_id=last_result.message_id)


async def _send_weixin(pconfig, chat_id, message, media_files=None):
    """Send via Weixin iLink using the native adapter helper."""
    try:
        from gateway.platforms.weixin import check_weixin_requirements, send_weixin_direct
        if not check_weixin_requirements():
            return {"error": "Weixin requirements not met. Need aiohttp + cryptography."}
    except ImportError:
        return {"error": "Weixin adapter not available."}

    try:
        return await send_weixin_direct(
            extra=pconfig.extra, token=pconfig.token, chat_id=chat_id,
            message=message, media_files=media_files,
        )
    except Exception as e:
        return _error(f"Weixin send failed: {e}")


async def _send_bluebubbles(extra, chat_id, message):
    """Send via BlueBubbles iMessage server using the adapter's REST API."""
    try:
        from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
        if not check_bluebubbles_requirements():
            return {"error": "BlueBubbles requirements not met (need aiohttp + httpx)."}
    except ImportError:
        return {"error": "BlueBubbles adapter not available."}

    try:
        from gateway.config import PlatformConfig
        adapter = BlueBubblesAdapter(PlatformConfig(extra=extra))
        connected = await adapter.connect()
        if not connected:
            return _error("BlueBubbles: failed to connect to server")
        try:
            result = await adapter.send(chat_id, message)
            if not result.success:
                return _error(f"BlueBubbles send failed: {result.error}")
            return _success("bluebubbles", chat_id, message_id=result.message_id)
        finally:
            await adapter.disconnect()
    except Exception as e:
        return _error(f"BlueBubbles send failed: {e}")


async def _send_qqbot(pconfig, chat_id, message):
    """Send via the QQ Bot Open Platform REST API (no WebSocket needed)."""
    try:
        import httpx
    except ImportError:
        return _error("QQBot direct send requires httpx. Run: pip install httpx")

    # Profile-scoped secret lookup so a multiplex profile never borrows
    # another profile's QQ credentials.
    from gateway.config import _getenv

    extra = pconfig.extra or {}
    appid = extra.get("app_id") or _getenv("QQ_APP_ID", "")
    secret = (pconfig.token or extra.get("client_secret")
              or _getenv("QQ_CLIENT_SECRET", ""))
    if not appid or not secret:
        return _error("QQBot: QQ_APP_ID / QQ_CLIENT_SECRET not configured.")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            token_resp = await client.post(
                "https://bots.qq.com/app/getAppAccessToken",
                json={"appId": str(appid), "clientSecret": str(secret)},
            )
            if token_resp.status_code != 200:
                return _error(f"QQBot token request failed: {token_resp.status_code}")
            access_token = token_resp.json().get("access_token")
            if not access_token:
                return _error("QQBot: no access_token in response")

            # QQ Bot API has separate endpoints for guild channels, C2C (private)
            # and groups; try them in that order, first 2xx wins.
            headers = {
                "Authorization": f"QQBot {access_token}",
                "Content-Type": "application/json",
            }
            payload = {"content": message[:4000], "msg_type": 0}
            endpoints = (
                ("channel", f"https://api.sgroup.qq.com/channels/{chat_id}/messages"),
                ("c2c", f"https://api.sgroup.qq.com/v2/users/{chat_id}/messages"),
                ("group", f"https://api.sgroup.qq.com/v2/groups/{chat_id}/messages"),
            )
            statuses = []
            for kind, url in endpoints:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in {200, 201}:
                    return _success("qqbot", chat_id, message_id=resp.json().get("id"))
                statuses.append(f"{kind}={resp.status_code}")
            return _error(f"QQBot send failed: {' '.join(statuses)}")
    except Exception as e:
        return _error(f"QQBot send failed: {e}")


async def _send_yuanbao(chat_id, message, media_files=None):
    """Send via the running Yuanbao adapter's persistent WebSocket (no
    throwaway client possible). chat_id: ``group:<code>``, ``direct:<id>`` or ``<id>``."""
    try:
        from gateway.platforms.yuanbao import get_active_adapter, send_yuanbao_direct
    except ImportError:
        return _error("Yuanbao adapter module not available.")

    adapter = get_active_adapter()
    if adapter is None:
        return _error(
            "Yuanbao adapter is not running. "
            "Start the gateway with yuanbao platform enabled first."
        )

    try:
        return await send_yuanbao_direct(adapter, chat_id, message, media_files=media_files)
    except Exception as e:
        return _error(f"Yuanbao send failed: {e}")
