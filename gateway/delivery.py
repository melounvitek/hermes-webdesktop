"""
Delivery routing for cron job outputs and agent responses.

Routes messages by target: explicit ("telegram:123456789"), platform home
channel ("telegram"), origin (back to where the job was created), or local
(saved to files).
"""

import logging
import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

# Cap before gateway-level truncation of cron output for non-chunking platform
# delivery. Telegram's hard API limit is 4096; the headroom covers the "full
# output saved to …" footer. Adapters that split long messages natively
# (BasePlatformAdapter.splits_long_messages) bypass this entirely.
MAX_PLATFORM_OUTPUT = 4000

# Matches strings that are *only* a "silence" narration with optional markdown
# wrappers: *(silent)*, _silent_, `silent`, (silent), silent, 🔇, a bare ".",
# "…", and marker-padded variants. Anchored so substantive messages that merely
# *contain* the word "silent" never match.
_SILENCE_NARRATION = re.compile(
    r'^[\s*_~`]*\(?\s*(silent|silence|no\s+response|no\s+reply)\s*\.?\)?[\s*_~`]*$'
    r'|^[\s*_~`]*[\U0001F507\.\u2026]+[\s*_~`]*$',
    re.IGNORECASE,
)


def _is_silence_narration(content: Optional[str]) -> bool:
    """True when ``content`` is *only* a silence-narration token (length-guarded)."""
    if not content:
        return False
    stripped = content.strip()
    if not stripped or len(stripped) > 64:
        return False
    return bool(_SILENCE_NARRATION.match(stripped))

from .config import Platform, GatewayConfig, PlatformConfig
from .session import SessionSource
from .dead_targets import DeadTargetRegistry


@dataclass(frozen=True)
class DeliveryTransport:
    """Resolved live transport for one logical delivery platform."""

    adapter: Any
    config: Optional[PlatformConfig]
    transport_platform: Platform

    @property
    def is_relay(self) -> bool:
        return self.transport_platform == Platform.RELAY

    async def send(
        self,
        logical_platform: Platform,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        """Send through this transport while preserving the logical platform."""
        if self.is_relay:
            return await self.adapter.send_for_platform(
                logical_platform, chat_id, content, metadata=metadata,
            )
        return await self.adapter.send(chat_id, content, metadata=metadata)


def resolve_delivery_transport(
    platform: Platform,
    config: GatewayConfig,
    adapters: Optional[Dict[Platform, Any]],
) -> Optional[DeliveryTransport]:
    """Resolve a logical platform to its live delivery transport.

    A concrete native adapter always wins. Relay is eligible only when its
    authenticated transport explicitly advertises that it fronts the logical
    platform, so restart-time delivery is independent of per-chat caches without
    letting Relay hijack unrelated platform targets.
    """
    live_adapters = adapters or {}
    native = live_adapters.get(platform)
    native_config = config.platforms.get(platform)
    # Explicitly supplied live adapters with no config block are honored, but an
    # explicitly disabled native adapter never shadows an enabled Relay transport.
    if native is not None and (native_config is None or native_config.enabled):
        return DeliveryTransport(native, native_config, platform)

    relay = live_adapters.get(Platform.RELAY)
    relay_config = config.platforms.get(Platform.RELAY)
    fronts_platform = getattr(relay, "fronts_platform", None)
    if (
        relay is not None
        and (relay_config is None or relay_config.enabled)
        and callable(fronts_platform)
        and fronts_platform(platform)
    ):
        return DeliveryTransport(relay, relay_config, Platform.RELAY)
    return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def looks_like_telegram_private_chat_id(chat_id: Optional[str]) -> bool:
    """True when ``chat_id`` is a positive int — Telegram's private-chat shape.

    Groups/channels/supergroups use negative IDs. Single source of truth for the
    heuristic, reused by the handoff seed path in ``gateway/run.py`` so
    handoff-created DM topics key the same way as inbound DM-topic messages.
    """
    if chat_id is None:
        return False
    parsed = _as_int(chat_id)
    return parsed is not None and parsed > 0


def _looks_like_int(value: Optional[str]) -> bool:
    return value is not None and _as_int(value) is not None


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a SendResult object or a plain result dict."""
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _send_result_failed(result: Any) -> bool:
    return _result_field(result, "success", True) is False


def _send_result_error(result: Any) -> Optional[str]:
    error = _result_field(result, "error")
    return str(error) if error else None


def _is_thread_not_found_delivery_error(result: Any) -> bool:
    error = _send_result_error(result)
    return bool(error and "thread not found" in error.lower())


def _classify_dead_from_error_text(error_text: Optional[str]) -> Optional[str]:
    """Best-effort dead-target classification from a raised error's text.

    ``_deliver_to_platform`` raises on hard failure (no SendResult), so the
    ``deliver()`` loop only has the exception string; reuse the platform-neutral
    classifier to recover the error_kind from it.
    """
    if not error_text:
        return None
    try:
        from .platforms.base import classify_send_error, is_chat_level_not_found
    except Exception:  # pragma: no cover - import guard
        return None
    kind = classify_send_error(None, error_text=error_text)
    if not DeadTargetRegistry.is_dead_error_kind(kind):
        return None
    # ``not_found`` collapses chat-level and thread/topic/message-level failures.
    # Only a whole-chat not_found means the target is dead — a deleted forum
    # topic or an edited-away message must not mark the entire chat (and all its
    # future deliveries) dead (see gateway.dead_targets).
    if kind == "not_found" and not is_chat_level_not_found(error_text=error_text):
        return None
    return kind


@dataclass
class DeliveryTarget:
    """A single delivery target: "origin", "local", "telegram" (home channel),
    or "telegram:123456[:thread]" (specific chat)."""
    platform: Platform
    chat_id: Optional[str] = None  # None means use home channel
    thread_id: Optional[str] = None
    is_origin: bool = False
    is_explicit: bool = False  # True if chat_id was explicitly specified

    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource] = None) -> "DeliveryTarget":
        """Parse "origin" | "local" | "<platform>" | "<platform>:<chat_id>[:<thread_id>]"."""
        target_stripped = target.strip()
        target_lower = target_stripped.lower()

        if target_lower == "origin":
            if origin:
                return cls(
                    platform=origin.platform,
                    chat_id=origin.chat_id,
                    thread_id=origin.thread_id,
                    is_origin=True,
                )
            return cls(platform=Platform.LOCAL, is_origin=True)

        if target_lower == "local":
            return cls(platform=Platform.LOCAL)

        # Platform names are case-insensitive; chat/thread ids keep original case.
        # Unknown platforms are treated as local.
        if ":" in target_stripped:
            parts = target_stripped.split(":", 2)
            chat_id = parts[1] if len(parts) > 1 else None
            thread_id = parts[2] if len(parts) > 2 else None
            try:
                platform = Platform(parts[0].lower())
            except ValueError:
                return cls(platform=Platform.LOCAL)
            return cls(platform=platform, chat_id=chat_id, thread_id=thread_id, is_explicit=True)

        try:
            return cls(platform=Platform(target_lower))
        except ValueError:
            return cls(platform=Platform.LOCAL)

    def to_string(self) -> str:
        """Convert back to string format."""
        if self.is_origin:
            return "origin"
        if self.platform == Platform.LOCAL:
            return "local"
        if self.chat_id and self.thread_id:
            return f"{self.platform.value}:{self.chat_id}:{self.thread_id}"
        if self.chat_id:
            return f"{self.platform.value}:{self.chat_id}"
        return self.platform.value


async def _ensure_named_dm_topic(adapter: Any, chat_id: str, name: str, *, refresh: bool) -> str:
    """Create (or force-recreate) a named Telegram private DM topic; return its thread id."""
    verb = "refresh" if refresh else "create"
    ensure_dm_topic = getattr(adapter, "ensure_dm_topic", None)
    if ensure_dm_topic is None:
        raise RuntimeError(f"Telegram adapter cannot {verb} named private DM topics")
    if refresh:
        thread_id = await ensure_dm_topic(chat_id, name, force_create=True)
    else:
        thread_id = await ensure_dm_topic(chat_id, name)
    if not thread_id:
        raise RuntimeError(f"Failed to {verb} Telegram private DM topic '{name}'")
    return str(thread_id)


class DeliveryRouter:
    """Resolves delivery targets and dispatches messages to platform adapters."""

    def __init__(self, config: GatewayConfig, adapters: Dict[Platform, Any] = None,
                 dead_targets: Optional[DeadTargetRegistry] = None):
        """``dead_targets``: shared registry of confirmed-unreachable targets;
        a profile-local registry is created when omitted."""
        self.config = config
        self.adapters = adapters or {}
        self.output_dir = get_hermes_home() / "cron" / "output"
        self.dead_targets = dead_targets or DeadTargetRegistry()

    async def deliver(
        self,
        content: str,
        targets: List[DeliveryTarget],
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deliver content to all targets; returns per-target results keyed by target string."""
        results = {}

        for target in targets:
            # Skip targets proven permanently unreachable (deleted group, blocked
            # bot, deactivated user) — re-sending each tick wastes flood-control
            # budget. Self-healing: a later successful send clears the flag.
            # LOCAL/origin-without-chat targets are never dead-tracked.
            tracked = target.platform != Platform.LOCAL and target.chat_id
            if tracked and self.dead_targets.is_dead(target.platform.value, target.chat_id):
                logger.info(
                    "Skipping delivery to known-dead target %s:%s "
                    "(send to it again to clear)",
                    target.platform.value, target.chat_id,
                )
                results[target.to_string()] = {
                    "success": False,
                    "skipped": "dead_target",
                    "error": "target previously confirmed unreachable",
                }
                continue
            try:
                if target.platform == Platform.LOCAL:
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)
                    if target.chat_id and not _send_result_failed(result):
                        self.dead_targets.clear(target.platform.value, target.chat_id)
                results[target.to_string()] = {"success": True, "result": result}
            except Exception as e:
                # Hard failures raise. Record a whole-chat death so future
                # deliveries short-circuit.
                if tracked:
                    dead_kind = _classify_dead_from_error_text(str(e))
                    if dead_kind:
                        self.dead_targets.mark_dead(
                            target.platform.value, target.chat_id,
                            reason=f"{dead_kind}: {str(e)[:120]}",
                        )
                results[target.to_string()] = {"success": False, "error": str(e)}

        return results

    def _deliver_local(
        self,
        content: str,
        job_id: Optional[str],
        job_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save content to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / (job_id or "misc") / f"{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# {job_name}" if job_name else "# Delivery Output",
            "",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if job_id:
            lines.append(f"**Job ID:** {job_id}")
        for key, value in (metadata or {}).items():
            lines.append(f"**{key}:** {value}")
        lines += ["", "---", "", content]

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {"path": str(output_path), "timestamp": timestamp}

    def _save_full_output(self, content: str, job_id: str) -> Path:
        """Save full cron output to disk and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = get_hermes_home() / "cron" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{job_id}_{timestamp}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _filter_silence_narration_enabled(self) -> bool:
        """``HERMES_FILTER_SILENCE_NARRATION`` env overrides the
        ``gateway.filter_silence_narration`` config flag (default True)."""
        env = os.getenv("HERMES_FILTER_SILENCE_NARRATION")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(getattr(self.config, "filter_silence_narration", True))

    def _cap_oversized_output(self, adapter: Any, content: str, job_id: str) -> str:
        """Audit-save oversized cron output; truncate it for non-chunking adapters.

        Two independent decisions: (1) above MAX_PLATFORM_OUTPUT the full output
        is always written to disk as an audit trail, best-effort — a failed save
        (full disk, permissions) never blocks delivery; (2) non-chunking adapters
        get the content truncated with a footer pointing to the saved file, while
        ``splits_long_messages`` adapters receive the full payload.
        """
        if len(content) <= MAX_PLATFORM_OUTPUT:
            return content
        saved_path: Optional[Path] = None
        try:
            saved_path = self._save_full_output(content, job_id)
        except OSError as exc:
            logger.warning(
                "Audit save failed for cron output (%d chars, job=%s): %s — "
                "delivery proceeds without audit copy",
                len(content), job_id, exc,
            )

        if getattr(adapter, "splits_long_messages", False):
            if saved_path:
                logger.info(
                    "Cron output preserved for chunking adapter (%d chars) — "
                    "full output saved to %s",
                    len(content), saved_path,
                )
            return content

        # The footer needs a valid path: if the best-effort save failed, retry
        # (a failure now is a real delivery problem and propagates).
        if saved_path is None:
            saved_path = self._save_full_output(content, job_id)
        footer = f"\n\n... [truncated, full output saved to {saved_path}]"
        visible = max(0, MAX_PLATFORM_OUTPUT - len(footer))
        logger.info(
            "Cron output truncated (%d chars) — full output: %s",
            len(content), saved_path,
        )
        return content[:visible] + footer

    async def _deliver_to_platform(
        self,
        target: DeliveryTarget,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deliver content to a messaging platform."""
        transport = resolve_delivery_transport(target.platform, self.config, self.adapters)
        if transport is None:
            raise ValueError(f"No adapter configured for {target.platform.value}")
        adapter = transport.adapter

        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform.value} delivery")

        content = self._cap_oversized_output(
            adapter, content, (metadata or {}).get("job_id", "unknown")
        )

        # Substrate-level anti-loop guard: drop hallucinated "silence narration"
        # (*(silent)*, 🔇, a bare ".") before it reaches any adapter — in
        # bot-to-bot channels these mirror back and forth until a model crashes
        # with "no content after all retries"; prompt rules drift across
        # providers, so this single chokepoint covers every platform. Local/file
        # delivery is never filtered (saved silence has no loop risk).
        # Cron output is an ARTIFACT, not model chatter: a legitimately terse job
        # ("...", a single 🔇) has no mirror loop, and dropping it while returning
        # success is how a cron gets logged as delivered with nothing on the
        # wire. Cron sends carry job_id in metadata; everything else is filtered.
        is_cron_artifact = "job_id" in (metadata or {})
        if (
            self._filter_silence_narration_enabled()
            and not is_cron_artifact
            and _is_silence_narration(content)
        ):
            logger.warning(
                "Dropped silence-narration outbound to %s (chat=%s): %r",
                target.platform.value,
                target.chat_id,
                content[:40],
            )
            return {
                "success": True,
                "filtered": "silence_narration",
                "delivered": False,
            }

        send_metadata = dict(metadata or {})
        if transport.is_relay:
            home = self.config.get_home_channel(target.platform)
            if home is not None and home.chat_id == target.chat_id:
                if home.user_id:
                    send_metadata["user_id"] = home.user_id
                if home.scope_id:
                    send_metadata["scope_id"] = home.scope_id

        # Caller-supplied thread routing always wins over target.thread_id.
        named_telegram_private_topic_name: Optional[str] = None
        thread_id = target.thread_id
        thread_unrouted = thread_id and not any(
            key in send_metadata
            for key in ("thread_id", "message_thread_id",
                        "direct_messages_topic_id", "telegram_direct_messages_topic_id")
        )
        if thread_unrouted:
            is_telegram_private = (
                target.platform == Platform.TELEGRAM
                and looks_like_telegram_private_chat_id(target.chat_id)
            )
            if is_telegram_private and not _looks_like_int(thread_id):
                # Named topic: create via createForumTopic, use message_thread_id directly.
                named_telegram_private_topic_name = thread_id
                send_metadata["thread_id"] = await _ensure_named_dm_topic(
                    adapter, target.chat_id, thread_id, refresh=False
                )
                send_metadata["telegram_dm_topic_created_for_send"] = True
            elif is_telegram_private:
                # Legacy numeric private topic ids not created by this send path
                # need a reply anchor to stay visible in the requested lane.
                if send_metadata.get("telegram_reply_to_message_id") is None:
                    raise RuntimeError(
                        "Telegram private DM topic delivery requires telegram_reply_to_message_id; "
                        "send to the bare chat or provide a reply anchor"
                    )
                send_metadata["thread_id"] = thread_id
                send_metadata["telegram_dm_topic_reply_fallback"] = True
            else:
                send_metadata["thread_id"] = thread_id

        result = await transport.send(
            target.platform, target.chat_id, content, metadata=send_metadata or None,
        )
        if _send_result_failed(result):
            if named_telegram_private_topic_name and _is_thread_not_found_delivery_error(result):
                send_metadata["thread_id"] = await _ensure_named_dm_topic(
                    adapter, target.chat_id, named_telegram_private_topic_name, refresh=True
                )
                send_metadata["telegram_dm_topic_created_for_send"] = True
                result = await transport.send(
                    target.platform, target.chat_id, content, metadata=send_metadata or None,
                )
            if _send_result_failed(result):
                raise RuntimeError(_send_result_error(result) or f"{target.platform.value} delivery failed")
        return result
