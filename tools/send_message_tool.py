"""Send Message Tool -- cross-channel messaging via platform APIs.

Sends a message to a user or channel on any connected messaging platform
(Telegram, Discord, Slack). Supports listing available targets and resolving
human-friendly channel names to IDs. Works in both CLI and gateway contexts.
"""

import asyncio
import json
import logging
import os

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

from tools.send_message_targets import (  # noqa: F401
    _BUZZ_UUID_RE,
    _E164_TARGET_RE,
    _EMAIL_TARGET_RE,
    _FEISHU_TARGET_RE,
    _HOME_CHANNEL_ENV_OVERRIDES,
    _NUMERIC_TOPIC_RE,
    _PHONE_PLATFORMS,
    _PHOTON_DM_GUID_RE,
    _SLACK_MENTION_RE,
    _SLACK_TARGET_RE,
    _SLACK_THREAD_TARGET_RE,
    _SLACK_USER_ID_RE,
    _SLACK_USER_NAME_RE,
    _TELEGRAM_TOPIC_TARGET_RE,
    _WEIXIN_TARGET_RE,
    _WHATSAPP_JID_RE,
    _YUANBAO_TARGET_RE,
    _parse_target_ref,
    resolve_send_target,
)
from tools.send_message_senders import (  # noqa: F401
    _AUDIO_EXTS,
    _CAPTIONABLE_EXTS,
    _DEFAULT_CAPTION_LIMIT,
    _GENERIC_SECRET_ASSIGN_RE,
    _IMAGE_EXTS,
    _TELEGRAM_CAPTION_LIMIT,
    _TELEGRAM_SEND_AUDIO_EXTS,
    _URL_SECRET_QUERY_RE,
    _VIDEO_EXTS,
    _VOICE_EXTS,
    _display_chat_id,
    _error,
    _is_telegram_thread_not_found,
    _live_runner,
    _matrix_send_core,
    _media_caption_split,
    _plugin_standalone_sender,
    _registry_standalone_send,
    _resolve_slack_user_target,
    _sanitize_error_text,
    _send_bluebubbles,
    _send_matrix_via_adapter,
    _send_qqbot,
    _send_signal,
    _send_telegram,
    _send_telegram_message_with_retry,
    _send_weixin,
    _send_yuanbao,
    _telegram_retry_delay,
)

def prepare_send_message_platforms() -> None:
    """Load enabled standalone plugins before tool schemas/cache keys are built."""
    from hermes_cli.plugins import discover_plugins

    discover_plugins()


def send_message_tool(args, **kw):
    """Handle cross-channel send_message tool calls."""
    action = args.get("action", "send")
    if action == "list":
        return _handle_list()
    if action in ("react", "unreact"):
        return _handle_react(args, remove=action == "unreact")
    return _handle_send(args)


def _split_target(target: str):
    """Split ``platform[:ref]`` into ``(platform_name, target_ref)``."""
    parts = target.split(":", 1)
    return parts[0].strip().lower(), (parts[1].strip() if len(parts) > 1 else None)


def _live_adapter(platform):
    """Return the running gateway's adapter for ``platform``, or None."""
    runner = _live_runner()
    return runner.adapters.get(platform) if runner is not None else None


def _handle_list():
    """Return formatted list of available messaging targets."""
    try:
        from gateway.channel_directory import format_directory_for_display
        return json.dumps({"targets": format_directory_for_display()})
    except Exception as e:
        return json.dumps(_error(f"Failed to load channel directory: {e}"))


def _handle_react(args, remove=False):
    """Attach (or with ``remove=True`` retract) an emoji reaction via a live
    gateway adapter exposing ``add_reaction(chat_id, emoji, message_id)`` /
    ``remove_reaction(chat_id, message_id)`` (e.g. photon/iMessage tapbacks).
    No standalone fallback: reacting needs the adapter's live message-id state.
    """
    target = args.get("target", "")
    emoji = (args.get("emoji") or "").strip()
    message_id = (args.get("message_id") or "").strip() or None
    if not target or (not remove and not emoji):
        return tool_error(
            "Both 'target' and 'emoji' are required when action='react'"
            if not remove
            else "'target' is required when action='unreact'"
        )

    platform_name, target_ref = _split_target(target)
    chat_id = None
    prepare_send_message_platforms()
    if target_ref:
        # Platform-native ids (e.g. photon GUIDs) match no parser/directory
        # entry; hand them to the adapter unchanged and let it validate.
        chat_id, _thread_id, resolution_error = resolve_send_target(
            platform_name, target_ref, pass_unresolved_references=True
        )
        if resolution_error:
            return tool_error(resolution_error)

    try:
        from gateway.config import Platform, load_gateway_config
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return tool_error(f"Unknown platform: {platform_name}")

    if not chat_id:
        try:
            home = load_gateway_config().get_home_channel(platform)
        except Exception:
            home = None
        if not home:
            return tool_error(
                f"No chat specified and no home channel set for {platform_name}. "
                f"Use '{platform_name}:chat_id'."
            )
        chat_id = home.chat_id

    adapter = _live_adapter(platform)
    if adapter is None:
        return tool_error(
            f"Reactions require a live {platform_name} adapter in the running "
            "gateway (not available from cron/standalone contexts)."
        )
    react_fn = getattr(adapter, "remove_reaction" if remove else "add_reaction", None)
    if not callable(react_fn):
        return tool_error(
            f"Platform '{platform_name}' does not support message reactions."
        )

    kwargs = {"chat_id": chat_id, "message_id": message_id}
    if not remove:
        kwargs["emoji"] = emoji
    try:
        from model_tools import _run_async
        result = _run_async(react_fn(**kwargs))
    except Exception as e:
        return json.dumps(_error(f"Reaction failed: {e}"))
    if isinstance(result, dict):
        return json.dumps(result)
    return json.dumps({"success": bool(result)})


def _handle_send(args):
    """Send a message to a platform target."""
    target = args.get("target", "")
    message = args.get("message", "")
    if not target or not message:
        return tool_error("Both 'target' and 'message' are required when action='send'")

    platform_name, target_ref = _split_target(target)
    chat_id = None
    thread_id = None

    prepare_send_message_platforms()
    if target_ref:
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_name, target_ref
        )
        if resolution_error:
            return tool_error(resolution_error)

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
    except Exception as e:
        return json.dumps(_error(f"Failed to load gateway config: {e}"))

    platform, pconfig, entry, err = _resolve_platform_config(platform_name, config)
    if err:
        return tool_error(err)

    from gateway.platforms.base import BasePlatformAdapter

    # Capture [[as_document]] before extract_media strips it: image files then
    # go through send_document so the original bytes survive (Telegram's
    # sendPhoto recompresses to 1280px).
    force_document_attachments = "[[as_document]]" in message

    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)

    used_home_channel = not chat_id
    if used_home_channel:
        chat_id, err = _home_chat_id(config, platform, platform_name)
        if err:
            return tool_error(err)

    duplicate_skip = _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id)
    if duplicate_skip:
        return json.dumps(duplicate_skip)

    if platform_name == "slack" and chat_id:
        chat_id, resolve_err = _slack_dm_chat_id(pconfig, chat_id)
        if resolve_err:
            return json.dumps(resolve_err)

    try:
        from model_tools import _run_async
        send_kwargs = {
            "thread_id": thread_id,
            "media_files": media_files,
            "force_document": force_document_attachments,
        }
        # Only custom plugin handlers receive the complete typed request; the
        # built-in call contract stays exact.
        if entry is not None and entry.send_message_handler is not None:
            send_kwargs["args"] = args
        result = _run_async(
            _send_to_platform(platform, pconfig, chat_id, cleaned_message, **send_kwargs)
        )
        if isinstance(result, dict) and result.get("success"):
            if used_home_channel:
                result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"
            if mirror_text and _mirror_sent_message(platform_name, chat_id, mirror_text, thread_id):
                result["mirrored"] = True

        if isinstance(result, dict) and "error" in result:
            result["error"] = _sanitize_error_text(result["error"])
        return json.dumps(result)
    except Exception as e:
        return json.dumps(_error(f"Send failed: {e}"))


def _resolve_platform_config(platform_name, config):
    """Return ``(platform, pconfig, registry_entry, error)`` for a send.

    Plugin platforms must be registered; disabled/missing platforms error,
    except Weixin, which may be configured purely via .env (synthesized
    pconfig so send_message and cron delivery work without a gateway.yaml entry).
    """
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry

    entry = platform_registry.get(platform_name)
    is_builtin = platform_name in {member.value for member in Platform}
    if not is_builtin and entry is None:
        return None, None, None, f"Unknown or unregistered plugin platform: {platform_name}"
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return None, None, None, f"Unknown platform: {platform_name}"

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        pconfig = _weixin_env_pconfig() if platform_name == "weixin" else None
        if pconfig is None:
            return None, None, None, (
                f"Platform '{platform_name}' is not configured. Set up credentials in "
                "~/.hermes/config.yaml or environment variables."
            )
    return platform, pconfig, entry, None


def _home_chat_id(config, platform, platform_name):
    """Return ``(home chat_id, None)`` or ``(None, actionable error)``.
    Weixin additionally honours the WEIXIN_HOME_CHANNEL env var."""
    home = config.get_home_channel(platform)
    if not home and platform_name == "weixin":
        wx_home = os.getenv("WEIXIN_HOME_CHANNEL", "").strip()
        if wx_home:
            from gateway.config import HomeChannel
            home = HomeChannel(platform=platform, chat_id=wx_home, name="Weixin Home")
    if home:
        return home.chat_id, None
    home_env = _HOME_CHANNEL_ENV_OVERRIDES.get(
        platform_name, f"{platform_name.upper()}_HOME_CHANNEL"
    )
    return None, (
        f"No home channel set for {platform_name} to determine where to send the message. "
        f"Either specify a channel directly with '{platform_name}:CHANNEL_NAME', "
        f"or set a home channel via: hermes config set {home_env} <channel_id>"
    )


def _slack_dm_chat_id(pconfig, chat_id):
    """Open Slack user targets as DM conversations: ``user:U...`` /
    ``user_name:@handle`` from the parser, or a bare U... id from session
    metadata / home-channel config. chat.postMessage needs a conversation ID.
    Returns ``(chat_id, None)`` or ``(None, error_dict)``."""
    dm_target = chat_id
    if dm_target.startswith("U") and _SLACK_USER_ID_RE.fullmatch(dm_target):
        dm_target = f"user:{dm_target}"
    if not dm_target.startswith(("user:", "user_name:")):
        return chat_id, None
    from model_tools import _run_async
    return _run_async(_resolve_slack_user_target(pconfig.token, dm_target))


def _mirror_sent_message(platform_name, chat_id, mirror_text, thread_id):
    """Best-effort mirror of the sent message into the target's gateway session."""
    try:
        from gateway.mirror import mirror_to_session
        from gateway.session_context import get_session_env
        return bool(mirror_to_session(
            platform_name,
            chat_id,
            mirror_text,
            source_label=get_session_env("HERMES_SESSION_PLATFORM", "cli"),
            thread_id=thread_id,
            user_id=get_session_env("HERMES_SESSION_USER_ID", "") or None,
        ))
    except Exception:
        return False


def _weixin_env_pconfig():
    """Synthesize a Weixin PlatformConfig from .env secrets, or None."""
    wx_token = get_secret("WEIXIN_TOKEN", "").strip()
    wx_account = get_secret("WEIXIN_ACCOUNT_ID", "").strip()
    if not (wx_token and wx_account):
        return None
    from gateway.config import PlatformConfig
    return PlatformConfig(
        enabled=True,
        token=wx_token,
        extra={
            "account_id": wx_account,
            "base_url": get_secret("WEIXIN_BASE_URL", "").strip(),
            "cdn_base_url": get_secret("WEIXIN_CDN_BASE_URL", "").strip(),
        },
    )


def _describe_media_for_mirror(media_files):
    """Return a human-readable mirror summary when a message only contains media."""
    if not media_files:
        return ""
    if len(media_files) != 1:
        return f"[Sent {len(media_files)} media attachments]"
    media_path, is_voice = media_files[0]
    ext = os.path.splitext(media_path)[1].lower()
    if is_voice and ext in _VOICE_EXTS:
        return "[Sent voice message]"
    for exts, kind in ((_IMAGE_EXTS, "image"), (_VIDEO_EXTS, "video"), (_AUDIO_EXTS, "audio")):
        if ext in exts:
            return f"[Sent {kind} attachment]"
    return "[Sent document attachment]"


def _get_cron_auto_delivery_target():
    """Return the cron scheduler's auto-delivery target for the current run, if any."""
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    if not platform or not chat_id:
        return None
    thread_id = get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip() or None
    return {"platform": platform, "chat_id": chat_id, "thread_id": thread_id}


def _maybe_skip_cron_duplicate_send(platform_name: str, chat_id: str, thread_id: str | None):
    """Skip redundant cron send_message calls when the scheduler will auto-deliver there."""
    auto_target = _get_cron_auto_delivery_target()
    if not auto_target or not (
        auto_target["platform"] == platform_name
        and str(auto_target["chat_id"]) == str(chat_id)
        and auto_target.get("thread_id") == thread_id
    ):
        return None

    target_label = f"{platform_name}:{chat_id}"
    if thread_id is not None:
        target_label += f":{thread_id}"

    return {
        "success": True,
        "skipped": True,
        "reason": "cron_auto_delivery_duplicate_target",
        "target": target_label,
        "note": (
            f"Skipped send_message to {target_label}. This cron job will already auto-deliver "
            "its final response to that same target. Put the intended user-facing content in "
            "your final response instead, or use a different target if you want an additional message."
        ),
    }


def _bounded_send_error(detail, max_chars=900):
    """Bound untrusted adapter/plugin error detail returned by send_message."""
    text = str(detail or "send failed")
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


async def _send_live_adapter_media(
    adapter,
    chat_id,
    message,
    media_files,
    *,
    thread_id=None,
    metadata=None,
    force_document=False,
):
    """Deliver text and every media descriptor through adapter media APIs.
    Adapters that only inherit the BasePlatformAdapter stub for a media kind
    are treated as unsupported rather than silently no-op'd."""
    caption, separate_text = _media_caption_split(
        message, media_files, max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    last_result = None
    if separate_text and separate_text.strip():
        last_result = await adapter.send(
            chat_id=chat_id, content=separate_text, metadata=metadata
        )
        if not last_result.success:
            return {"error": f"Adapter send failed: {_bounded_send_error(last_result.error)}"}

    total = len(media_files)
    for index, descriptor in enumerate(media_files):
        if not isinstance(descriptor, (list, tuple)) or not descriptor:
            return {"error": f"Adapter media send failed: invalid media descriptor {index + 1}/{total}"}
        media_path = descriptor[0]
        is_voice = bool(descriptor[1]) if len(descriptor) > 1 else False
        if not isinstance(media_path, str) or not media_path:
            return {"error": f"Adapter media send failed: invalid media descriptor {index + 1}/{total}"}
        if not os.path.exists(media_path):
            return {"error": f"Adapter media send failed: media file {index + 1}/{total} was not found"}

        ext = os.path.splitext(media_path)[1].lower()
        kwargs = {
            "caption": caption if index == 0 else None,
            "reply_to": thread_id,
            "metadata": metadata,
        }
        if force_document:
            method_name, media_kind = "send_document", "document"
        elif ext in _IMAGE_EXTS:
            method_name, media_kind = "send_image_file", "image"
        elif ext in _VIDEO_EXTS:
            method_name, media_kind = "send_video", "video"
        elif is_voice or ext in _AUDIO_EXTS:
            method_name, media_kind = "send_voice", "audio"
        else:
            method_name, media_kind = "send_document", "document"

        from gateway.platforms.base import BasePlatformAdapter

        adapter_method = getattr(type(adapter), method_name, None)
        base_fallback = getattr(BasePlatformAdapter, method_name)
        if adapter_method is None or adapter_method is base_fallback:
            return {
                "error": (
                    f"Live adapter does not implement native {media_kind} delivery; "
                    f"media file {index + 1}/{total} was not sent"
                )
            }
        try:
            last_result = await getattr(adapter, method_name)(chat_id, media_path, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "error": (
                    f"Adapter media send failed after {index}/{total} files: "
                    f"{_bounded_send_error(exc)}"
                )
            }
        if not last_result.success:
            detail = _bounded_send_error(last_result.error or "media send failed")
            return {
                "error": f"Adapter media send failed after {index}/{total} files: {detail}"
            }

    if last_result is None:
        return {"error": "No deliverable text or media remained after processing MEDIA tags"}
    return {
        "success": True,
        "message_id": last_result.message_id,
        "media_delivered": True,
    }


async def _send_via_adapter(
    platform,
    pconfig,
    chat_id,
    chunk,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Send via the live in-process gateway adapter (``_gateway_runner_ref``),
    else the plugin's ``standalone_sender_fn`` (gateway not in this process,
    e.g. cron — the runner weakref is None), else a descriptive error naming
    both options. Media descriptors go through the adapter's native media
    APIs under the same cross-loop rules as text.
    """
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    runner = _live_runner()
    adapter = None
    if runner is not None:
        try:
            adapter = runner.adapters.get(platform)
        except Exception:
            adapter = None
    if adapter is not None:
        try:
            metadata = {}
            if thread_id:
                metadata["thread_id"] = thread_id
            if platform_name == "ntfy" and chat_id:
                metadata["publish_topic"] = chat_id
            metadata = metadata or None
            # adapter.send() uses queues/tasks bound to the gateway's loop.
            # Awaiting it from another loop (the tool worker thread) deadlocks
            # on a cross-loop Future, so dispatch onto the gateway loop instead.
            gateway_loop = getattr(runner, "_gateway_loop", None)
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            cross_loop = gateway_loop is not None and current_loop is not gateway_loop

            async def _dispatch(make_coro, log_message):
                if not cross_loop:
                    return await make_coro()  # same loop / no gateway loop (CLI, tests)
                if not gateway_loop.is_running():
                    return {"error": "Gateway loop is not running; cannot dispatch adapter send"}
                from agent.async_utils import safe_schedule_threadsafe
                fut = safe_schedule_threadsafe(
                    make_coro(), gateway_loop, logger=logger, log_message=log_message
                )
                if fut is None:
                    return {"error": "Gateway loop unavailable for send dispatch"}
                # shield: a cancelled caller (agent interrupt) must not cancel the
                # already-enqueued gateway send, or a retry would duplicate it.
                # No timeout here — the adapter's request timeout and the outer
                # _run_async timeout bound the wait.
                return await asyncio.shield(asyncio.wrap_future(fut))

            if media_files:
                return await _dispatch(
                    lambda: _send_live_adapter_media(
                        adapter, chat_id, chunk, media_files,
                        thread_id=thread_id, metadata=metadata, force_document=force_document,
                    ),
                    "send_message: failed to schedule media send on gateway loop",
                )
            result = await _dispatch(
                lambda: adapter.send(chat_id=chat_id, content=chunk, metadata=metadata),
                "send_message: failed to schedule on gateway loop",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"error": f"Plugin platform send failed: {_bounded_send_error(e)}"}
        if isinstance(result, dict):
            return result
        if result.success:
            return {"success": True, "message_id": result.message_id}
        return {"error": f"Adapter send failed: {_bounded_send_error(result.error)}"}

    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name)
    except Exception:
        entry = None

    if entry is not None and entry.standalone_sender_fn is not None:
        try:
            result = await entry.standalone_sender_fn(
                pconfig,
                chat_id,
                chunk,
                thread_id=thread_id,
                media_files=media_files,
                force_document=force_document,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Plugin standalone send for %s raised", platform_name, exc_info=True)
            return {"error": f"Plugin standalone send failed: {_bounded_send_error(e)}"}

        if isinstance(result, dict) and (result.get("success") or result.get("error")):
            if result.get("error"):
                return {**result, "error": _bounded_send_error(result["error"])}
            return result
        return {
            "error": (
                f"Plugin standalone send for '{platform_name}' returned an "
                f"invalid result: expected a dict with 'success' or 'error' "
                f"keys, got {type(result).__name__}"
            )
        }

    return {
        "error": (
            f"No live adapter for platform '{platform_name}'. Is the gateway "
            f"running with this platform connected? For out-of-process delivery "
            f"(e.g. cron in a separate process), the platform plugin must "
            f"register a standalone_sender_fn on its PlatformEntry."
        )
    }


async def _send_chunks(chunks, send_one):
    """Send chunks in order via ``send_one(chunk, is_last)``; stop at the first
    error dict, otherwise return the last result."""
    last_result = None
    for i, chunk in enumerate(chunks):
        result = await send_one(chunk, i == len(chunks) - 1)
        if isinstance(result, dict) and result.get("error"):
            return result
        last_result = result
    return last_result


def _platform_max_length(platform):
    """Max message length for chunking: the adapter constant for Signal (its
    raw JSON-RPC path never sees SignalAdapter's own chunking, so signal-cli
    would reject long sends whole), the registry's ``max_message_length`` for
    plugins (Slack, Feishu, ...), else None (no chunking)."""
    from gateway.config import Platform

    if platform == Platform.SIGNAL:
        try:
            from gateway.platforms.signal import MAX_MESSAGE_LENGTH
            return MAX_MESSAGE_LENGTH
        except ImportError:
            return 8000
    try:
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform.value)
        if entry and entry.max_message_length > 0:
            return entry.max_message_length
    except Exception:
        pass
    return None


# Plugin platforms whose media (and, for Discord, all) sends go straight to the
# registry ``standalone_sender_fn`` — bypassing the live adapter on purpose:
# Discord's plugin ``_standalone_send`` handles forum channels, threads and
# multipart uploads (historically the only Discord path); Slack uploads via
# files_upload_v2; WhatsApp posts each file to the Baileys bridge /send-media so
# images/videos/audio arrive as native bubbles, not documents.
# platform -> (error label, run discover_plugins first, caption-capable,
#              media_files sentinel for non-final chunks, forward force_document)
_PLUGIN_STANDALONE_MEDIA = {
    "discord": ("Discord", False, True, [], False),
    "feishu": ("Feishu", True, False, None, False),
    "slack": ("Slack", True, True, [], False),
    "whatsapp": ("WhatsApp", True, True, None, True),
}


async def _send_plugin_standalone(
    platform_name, pconfig, chat_id, message, chunks, media_files, *, thread_id, max_len, force_document
):
    """Chunked send through a plugin's standalone_sender_fn, with the single
    captionable file + short text case riding as the media caption."""
    label, discover, captionable, empty_media, pass_force = _PLUGIN_STANDALONE_MEDIA[platform_name]
    sender, err = _plugin_standalone_sender(platform_name, label=label, discover=discover)
    if err:
        return err
    extra = {"force_document": force_document} if pass_force else {}
    if captionable:
        # Cap on the platform's own message limit so the caption is deliverable.
        caption, _ = _media_caption_split(
            message, media_files, max_caption_len=(max_len or _DEFAULT_CAPTION_LIMIT)
        )
        if caption is not None:
            return await sender(
                pconfig, chat_id, "", thread_id=thread_id, media_files=media_files,
                caption=caption, **extra,
            )
    return await _send_chunks(
        chunks,
        lambda chunk, is_last: sender(
            pconfig, chat_id, chunk, thread_id=thread_id,
            media_files=media_files if is_last else empty_media, **extra,
        ),
    )


# Native-media chunked routes for built-in platforms; media rides on the final
# chunk, non-final chunks get the platform's empty-media sentinel.
# platform -> (media required, empty-media sentinel,
#              sender(platform, pconfig, chat_id, chunk, media, thread_id, force_document))
# Matrix: ALL sends go through the native adapter so text is encrypted in E2EE
#   rooms too (the raw-HTTP standalone path is not encryption-aware).
# Signal: attachments ride the JSON-RPC ``attachments`` param.
# Yuanbao: media needs the running gateway adapter's WebSocket.
# Slack (text; media was intercepted above): prefer the live adapter — it is
#   multi-workspace aware and honors adapter gates like ignored_channels, while
#   the standalone Web-API path may only have a token list — else the plugin's
#   standalone sender, keeping MEDIA delivery on the cron fallback.
# WeCom: native media only through the live gateway adapter.
# Names resolve at call time so tests can monkeypatch e.g. ``_send_signal``.
_CHUNKED_ROUTES = {
    "matrix": (False, [], lambda p, pc, cid, chunk, media, tid, fd: _send_matrix_via_adapter(
        pc, cid, chunk, media_files=media, thread_id=tid)),
    "signal": (True, [], lambda p, pc, cid, chunk, media, tid, fd: _send_signal(
        pc.extra, cid, chunk, media_files=media)),
    "yuanbao": (True, None, lambda p, pc, cid, chunk, media, tid, fd: _send_yuanbao(
        cid, chunk, media_files=media)),
    "slack": (False, [], lambda p, pc, cid, chunk, media, tid, fd: _send_via_adapter(
        p, pc, cid, chunk, thread_id=tid, media_files=media, force_document=fd)),
    "wecom": (True, None, lambda p, pc, cid, chunk, media, tid, fd: _send_via_adapter(
        p, pc, cid, chunk, thread_id=tid, media_files=media, force_document=fd)),
}

# Text-only senders for built-in platforms (generic path; media is dropped
# with a warning). Signature: (pconfig, chat_id, chunk, thread_id) -> result.
_TEXT_SENDERS = {
    "whatsapp": lambda pc, cid, chunk, tid: _registry_standalone_send("whatsapp", pc, cid, chunk, tid),
    "signal": lambda pc, cid, chunk, tid: _send_signal(pc.extra, cid, chunk),
    "email": lambda pc, cid, chunk, tid: _registry_standalone_send("email", pc, cid, chunk, tid),
    "sms": lambda pc, cid, chunk, tid: _registry_standalone_send("sms", pc, cid, chunk, tid),
    "dingtalk": lambda pc, cid, chunk, tid: _registry_standalone_send("dingtalk", pc, cid, chunk, tid),
    "feishu": lambda pc, cid, chunk, tid: _registry_standalone_send("feishu", pc, cid, chunk, tid),
    "wecom": lambda pc, cid, chunk, tid: _registry_standalone_send("wecom", pc, cid, chunk, tid),
    "bluebubbles": lambda pc, cid, chunk, tid: _send_bluebubbles(pc.extra, cid, chunk),
    "qqbot": lambda pc, cid, chunk, tid: _send_qqbot(pc, cid, chunk),
    "yuanbao": lambda pc, cid, chunk, tid: _send_yuanbao(cid, chunk),
}

_MEDIA_PLATFORMS_NOTE = "telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack"


async def _send_to_platform(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
    """Route a message to the appropriate platform sender.

    Long messages are chunked with the adapters' smart splitter (code-block
    aware, part indicators). Branch order matters: Weixin first (its native
    helper must not be blocked by unrelated optional imports such as
    lark-oapi's heavy Feishu path), Telegram (chunks itself), plugin
    standalone media routes, native chunked routes, then the generic text
    path that drops media with a warning.
    """
    from gateway.config import Platform

    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    media_files = media_files or []

    if platform == Platform.WEIXIN:
        return await _send_weixin(pconfig, chat_id, message, media_files=media_files)

    # Telegram chunks internally on the *formatted* text in UTF-16 units
    # (MarkdownV2/HTML escaping inflates length), so it gets the whole
    # message; media attaches after all text chunks.
    if platform == Platform.TELEGRAM:
        disable_link_previews = bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews"))
        return await _send_telegram(
            pconfig.token,
            chat_id,
            message,
            media_files=media_files,
            thread_id=thread_id,
            disable_link_previews=disable_link_previews,
            force_document=force_document,
        )

    from gateway.platforms.base import BasePlatformAdapter

    max_len = _platform_max_length(platform)
    chunks = BasePlatformAdapter.truncate_message(message, max_len) if max_len else [message]

    if platform_name == "discord" or (media_files and platform_name in _PLUGIN_STANDALONE_MEDIA):
        return await _send_plugin_standalone(
            platform_name, pconfig, chat_id, message, chunks, media_files,
            thread_id=thread_id, max_len=max_len, force_document=force_document,
        )

    route = _CHUNKED_ROUTES.get(platform_name)
    if route is not None and (media_files or not route[0]):
        _, empty_media, sender = route
        return await _send_chunks(chunks, lambda chunk, is_last: sender(
            platform, pconfig, chat_id, chunk,
            media_files if is_last else empty_media, thread_id, force_document,
        ))

    # --- Generic path: text only. Buzz is a plugin platform with verified
    # native media delivery through _send_via_adapter (media-only sends
    # included), so it is exempt from the media error/warning.
    if media_files and platform_name != "buzz" and not message.strip():
        return {
            "error": (
                f"send_message MEDIA delivery is currently only supported for {_MEDIA_PLATFORMS_NOTE}; "
                f"target {platform_name} had only media attachments"
            )
        }
    warning = None
    if media_files and platform_name != "buzz":
        warning = (
            f"MEDIA attachments were omitted for {platform_name}; "
            f"native send_message media delivery is currently only supported for {_MEDIA_PLATFORMS_NOTE}"
        )

    text_sender = _TEXT_SENDERS.get(platform_name)
    if text_sender is not None:
        send_one = lambda chunk, is_last: text_sender(pconfig, chat_id, chunk, thread_id)  # noqa: E731
    else:
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get(platform_name)
        handler = entry.send_message_handler if entry is not None else None
        if handler is not None:
            # Custom handler receives the full typed request once (not per chunk).
            try:
                import inspect

                result = handler(args or {}, chat_id, platform_name, pconfig)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as e:
                return {"error": f"Plugin send_message handler failed: {e}"}
        # Plugin platform: live gateway adapter if available, else standalone_sender_fn.
        send_one = lambda chunk, is_last: _send_via_adapter(  # noqa: E731
            platform, pconfig, chat_id, chunk, thread_id=thread_id,
            media_files=media_files if is_last else [], force_document=force_document,
        )

    last_result = await _send_chunks(chunks, send_one)
    if (
        warning
        and isinstance(last_result, dict)
        and last_result.get("success")
        and not last_result.get("media_delivered")
    ):
        last_result["warnings"] = [*last_result.get("warnings", []), warning]
    return last_result


# --- Registry ---
from tools.registry import tool_error

# NOTE: ``send_message`` is intentionally NOT registered as an agent-callable
# model tool (the agent must not fire cross-platform messages on its own).
# The send engine here is the shared transport for cron delivery, the
# ``hermes send`` CLI, the gateway kanban notifier and the opt-in MCP server,
# which import the helpers directly.
