"""RelayAdapter — one generic gateway adapter fronted by the connector. EXPERIMENTAL.

A single ``BasePlatformAdapter`` subclass that, at handshake, receives a
``CapabilityDescriptor`` telling it which platform it fronts and which
capabilities to advertise to the ``GatewayStreamConsumer``. It implements the
abstract methods (``connect`` / ``disconnect`` / ``send`` / ``get_chat_info``)
plus the capability surface by delegating wire I/O to an injected transport and
reading capabilities off the descriptor.

There is NO per-platform gateway code: only the connector knows "this chat_id
maps to a Discord channel". The gateway sees an ordinary ``MessageEvent`` in and
calls ``adapter.send`` out. The transport protocol and descriptor schema may
change without a deprecation cycle until >=2 Class-1 platforms validate them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.media import RelayMediaClient
from gateway.relay.transport import RelayTransport
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# The drain-path going-idle ACK budget must stay strictly under the runner's
# default adapter disconnect timeout (5s) or cancellation fires before
# transport.disconnect() and leaves the websocket open. With transport teardown
# budgets of 1s each for supervisor, reader and ws.close, the drain stays <5s.
_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S = 2.0
_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S = 1.0

# Link detection for the fresh-final unfurl route: raw URLs, Slack mrkdwn links
# and markdown links. Permissive on purpose — a false positive costs one fresh
# (non-edited) final; a false negative silently loses the preview.
_URL_RE = re.compile(r"https?://|<https?:|\]\(https?:")

# Already-answered prompt ids to remember so a duplicate answer (double tap or
# connector redelivery) reads as a repeat, not a stale prompt.
_RESOLVED_PROMPT_MEMORY = 256

# Connector promptCodec.decodePromptCallback id alphabet ([A-Za-z0-9_.-], <=32).
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_SLACK = Platform.SLACK.value


def _utf16_len(text: str) -> int:
    """Count UTF-16 code units (Telegram's length unit)."""
    return len(text.encode("utf-16-le")) // 2


_LEN_FNS: Dict[str, Callable[[str], int]] = {
    "chars": len,
    "utf16": _utf16_len,
}


def _event_ids(event) -> Tuple[Optional[str], Optional[str]]:
    """(message_id, chat_id) of an inbound event; message_id lives on the event, falls back to source."""
    message_id = getattr(event, "message_id", None) or getattr(
        event.source, "message_id", None
    )
    return message_id, getattr(event.source, "chat_id", None)


class RelayAdapter(BasePlatformAdapter):
    """Generic relay adapter advertising a connector-negotiated capability profile."""

    def __init__(
        self,
        config: PlatformConfig,
        descriptor: CapabilityDescriptor,
        transport: Optional[RelayTransport] = None,
    ) -> None:
        # Fronts many platforms but presents to the runner as Platform.RELAY.
        super().__init__(config, Platform.RELAY)
        self.descriptor = descriptor
        self._transport = transport
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        # Per-chat egress routing caches, learned from inbound events (send()
        # only receives a chat_id). The connector's egress guard resolves the
        # owning tenant from OUTBOUND metadata.scope_id / metadata.user_id, so
        # we re-attach them from what we saw inbound (see _capture_scope).
        self._scope_by_chat: Dict[str, str] = {}
        self._dm_user_by_chat: Dict[str, str] = {}
        # chat_id -> chat_type; needed to reproduce native Slack's synthetic
        # DM-thread suppression (a raw reply_to becomes a Slack thread_ts
        # connector-side, so a plain DM reply would thread under the user).
        self._chat_type_by_chat: Dict[str, str] = {}
        # chat_id -> last triggering Slack message ts (typing/status lane's
        # synthetic thread anchor in thread-per-message mode).
        self._last_inbound_ts_by_chat: Dict[str, str] = {}
        # chat_id -> UNDERLYING platform ("discord", ...): one relay adapter
        # fronts N platforms on one WS and a reply must egress through the
        # platform the inbound came from. Empty for a single-platform gateway
        # (the connector falls back to its session default).
        self._platform_by_chat: Dict[str, str] = {}
        # chat_id -> (thread_id, initial_name) of the auto-thread the CONNECTOR
        # created for our latest send (SendResult feedback); read by the
        # semantic thread-rename lane. Bounded like the sibling caches.
        self._auto_thread_by_chat: Dict[str, Tuple[str, str]] = {}
        # chat_id -> event fired when the entry above lands (wait_for_auto_thread_info).
        self._auto_thread_waiters: Dict[str, asyncio.Event] = {}
        # Bounded FIFO seen-set for inbound replay dedupe (insertion-ordered dict).
        self._seen_inbound: Dict[str, None] = {}
        # Live cards: draft_key -> draft_id of the OPEN native stream. Armed
        # by send_draft; consumed by send() to convert the turn-final into
        # draft(final=true) instead of a duplicate post. Keyed by _draft_key
        # (chat + per-turn identity), NOT bare chat: parallel turns in one DM
        # are distinct streams (per-chat keying merged three concurrent turns).
        self._open_draft_by_chat: Dict[str, int] = {}
        # draft_key -> draft_id of the most recently SEALED stream (mirror of
        # the connector's sealed-key tombstone): post-seal stragglers must
        # neither re-arm interception nor re-open a stream.
        self._sealed_draft_by_chat: Dict[str, int] = {}
        # Draft keys whose post-seal swallow has been logged once (bounded FIFO).
        self._tombstone_swallow_logged: Dict[str, int] = {}
        # Strong refs for fire-and-forget lifecycle acks (asyncio holds tasks weakly).
        self._lifecycle_ack_tasks: set = set()
        # Stream-is-the-message marker read by the stream consumer to keep ONE
        # draft stream per turn instead of bumping draft_id at tool boundaries.
        # SLACK-ONLY: the base send_draft contract is Telegram-shaped (draft
        # clears, final arrives as a separate real send); setting this for any
        # connector advertising "draft" intercepted the turn-final into
        # draft(final=true) and no history message was ever posted. A future
        # platform with this semantic should advertise it via the descriptor.
        self.draft_stream_is_message = (
            str(getattr(descriptor, "platform", "") or "").lower() == "slack"
        )
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        self.supports_inchannel_continuable = bool(
            getattr(descriptor, "supports_inchannel_continuable", False)
        )
        # Watches the transport for a terminal auth revocation (4401 after a
        # successful handshake = operator opted this instance out) and surfaces
        # a clean non-retryable "relay disabled" fatal instead of a retry spin.
        self._revocation_monitor: Optional[asyncio.Task[None]] = None
        # Lazily built client for the connector's /relay/media routes; None when
        # dial URL or creds are absent (media lanes degrade to text fallbacks).
        self._media_client: Optional["RelayMediaClient"] = None
        # prompt_id -> pending-prompt state for the interactive `prompt` op; the
        # user's pick comes back as a prompt_response naming this id and resolves
        # the waiting primitive exactly like native button callbacks. Entries
        # expire lazily (_pop_prompt).
        self._pending_prompts: Dict[str, Dict[str, Any]] = {}
        # Per-process marker prefixed onto every prompt id we mint. WHY: button
        # presses ride the passthrough plane, which the connector fans out to
        # EVERY live gateway session of the tenant, while _pending_prompts is
        # process-local. Without the marker a sibling cannot tell "someone
        # else owns this" from "my prompt expired", and the id-shaped text
        # ("/c1") falls through to chat dispatch as "Unknown command" — once
        # per sibling. Siblings are the common case in a DM.
        self._prompt_owner_nonce: str = secrets.token_hex(3)
        # Prompt ids this process already resolved, newest last (repeat
        # answers are consumed silently instead of treated as stale).
        self._resolved_prompts: "OrderedDict[str, float]" = OrderedDict()

    # ── capability surface (from descriptor) ─────────────────────────────
    @property
    def authorization_is_upstream(self) -> bool:
        """Authorization is enforced by the connector (owner-only author-binding
        resolution before delivery), so relay users must not be default-denied
        for lack of a local ``RELAY_ALLOWED_USERS`` allowlist."""
        return True

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        return _LEN_FNS.get(self.descriptor.len_unit, len)

    @property
    def supports_status_text(self) -> bool:  # type: ignore[override]
        """Whether the fronted platform renders a TEXT status line.

        Slack's typing surface is the assistant status line, so run.py's
        live-status lane may feed per-tool phrases (native SlackAdapter parity);
        other platforms have textless bubbles and must NOT receive phrases.
        Reflects the PRIMARY identity's platform, like the scalar ``descriptor``.
        """
        return self.descriptor.platform == _SLACK

    # ── per-chat capability resolution (multi-platform) ──────────────────
    def _negotiated_descriptor(self, platform: Optional[str]) -> Optional[CapabilityDescriptor]:
        """The transport's negotiated descriptor for ``platform``, or None
        (unknown platform, no transport, or a transport predating
        ``descriptor_for_platform``). Never raises — capability lookup must
        never break a send."""
        if not platform or self._transport is None:
            return None
        resolve = getattr(self._transport, "descriptor_for_platform", None)
        if not callable(resolve):
            return None
        try:
            return resolve(platform)
        except Exception:  # noqa: BLE001
            return None

    def _chat_platform(self, chat_id: str) -> Optional[str]:
        """The chat's underlying platform as seen inbound, else the primary's."""
        return self._platform_by_chat.get(str(chat_id)) or self.descriptor.platform

    def _descriptor_for_chat(self, chat_id: str) -> CapabilityDescriptor:
        """The capability descriptor governing a specific chat.

        Platform caps genuinely differ (Discord 2000 / Telegram 4096 / Slack
        39000), so the primary's scalar cap either fragments needlessly or
        over-sends into a platform 400. Falls back to the scalar descriptor
        when the chat's platform is unknown (never saw inbound).
        """
        per_platform = self._negotiated_descriptor(self._platform_by_chat.get(str(chat_id)))
        return per_platform if per_platform is not None else self.descriptor

    def max_message_length_for_chat(self, chat_id: str) -> int:
        return self._descriptor_for_chat(chat_id).max_message_length

    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        return _LEN_FNS.get(self._descriptor_for_chat(chat_id).len_unit, len)

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        # Needs BOTH the descriptor flag and an explicit "draft" op: supported_ops
        # is fail-open for legacy connectors, but "draft" did not exist
        # pre-contract, so it must NOT fail open. Resolved per chat when the
        # caller names one (a Telegram primary must not starve a Slack chat).
        desc = (
            self._descriptor_for_chat(str(chat_id))
            if chat_id is not None
            else self.descriptor
        )
        if not (
            desc.supports_draft_streaming
            and "draft" in (desc.supported_ops or ())
        ):
            return False
        # Slack chat.*Stream has no unfurl_links / unfurl_media; like native
        # SlackAdapter, refuse streaming when those knobs are set so
        # chat.postMessage can carry them.
        platform = self._chat_platform(chat_id) if chat_id is not None else desc.platform
        return not self._slack_unfurl_hints(platform)

    def prefers_fresh_final_streaming(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        """Deliver streamed finals as a FRESH send when Slack unfurl is forced on.

        Slack evaluates link previews exactly once, at ``chat.postMessage``;
        a ``chat.update`` that INTRODUCES the URL never unfurls. Edit-based
        streaming posts its first frame before any URL exists, so a configured
        ``unfurl_*: true`` can only surface via a fresh final that ``send()``
        stamps with the hints. ONLY when the hints contain an explicit True:
        false-only hints (fail-closed posture) ride the placeholder post fine.
        Only link-bearing finals qualify — the relay has no delete op in
        contract v1, so a linkless fresh final would just be a duplicate.
        """
        platform = None
        if chat_id is not None:
            platform = self._platform_by_chat.get(str(chat_id))
        # The stream consumer's hook passes (content, metadata=...) only.
        if platform is None and isinstance(metadata, dict):
            platform = metadata.get("platform")
        if platform is None:
            platform = self.descriptor.platform
        hints = self._slack_unfurl_hints(platform)
        if not hints or not any(v is True for v in hints.values()):
            return False
        return bool(_URL_RE.search(content or ""))

    def stream_is_message_for_chat(self, chat_id: str) -> bool:
        """Per-chat stream-is-the-message semantic (see ``draft_stream_is_message``).

        A Slack primary must not impose seal semantics on a Telegram chat (its
        turn-final would become draft(final=true) — no history message), nor a
        Telegram primary deny a Slack chat native streaming. Platform-name
        inference is deliberate; a descriptor field is the eventual contract.
        """
        return (
            str(self._descriptor_for_chat(str(chat_id)).platform or "").lower()
            == "slack"
        )

    # ── Live cards: native draft streaming + task cards ──────────────────
    #
    # Additive relay ops within contract v1. The gateway emits ops when the
    # negotiated descriptor advertises them; the connector owns the platform
    # API mechanics, feature-gate caching, and the send+edit fallback.
    # Semantic bridge: the base send_draft contract is Telegram-shaped (draft
    # clears, final arrives as a separate send()); Slack native streaming
    # makes the stream THE message. The adapter tracks the open draft per turn
    # and converts that turn's final send() into draft(final=true).

    def supports_native_task_cards(self) -> bool:
        """Explicit advertisement required — same no-fail-open rule as "draft"."""
        return "task_card" in (self.descriptor.supported_ops or ())

    def native_task_cards_enabled(self) -> bool:
        """TurnRunner opt-in probe (gateway/run.py calls THIS name, same contract as
        native Slack); without the alias the card lane silently stays text-mode."""
        return self.supports_native_task_cards()

    @staticmethod
    def _draft_key(chat_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Coordination key for one turn's stream.

        Prefers a PER-TURN identity (the triggering inbound message id, stamped
        as ``message_id`` or ``reply_to_message_id``) over the thread anchor:
        two parallel turns replying inside ONE thread share thread_ts (turn A's
        final sealed turn B's stream), and a flat DM with no anchor degraded to
        the bare chat id. The anchor remains the fallback for placement-only
        callers; the bare chat is the last resort.
        """
        md = metadata or {}
        turn_id = md.get("message_id") or md.get("reply_to_message_id")
        if turn_id:
            return f"{chat_id}:turn:{turn_id}"
        anchor = md.get("thread_ts") or md.get("thread_id") or ""
        return f"{chat_id}:{anchor}"

    # Cap for the draft/seal coordination dicts (per-turn keys); matches the
    # connector's tombstone store size.
    _DRAFT_STATE_CAP = 512

    @classmethod
    def _evict_oldest(cls, d: Dict[str, int]) -> None:
        """FIFO-bound a coordination dict in place."""
        while len(d) > cls._DRAFT_STATE_CAP:
            d.pop(next(iter(d)), None)

    @staticmethod
    def _card_key(
        reply_to: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Per-turn task-card identity — same precedence as ``_draft_key``.

        One derivation for send AND stop, so the stop always hits the stream
        the send opened.
        """
        md = metadata or {}
        anchor = (
            reply_to
            or md.get("message_id")
            or md.get("reply_to_message_id")
            or md.get("thread_ts")
            or md.get("thread_id")
            or "root"
        )
        return f"turn:{anchor}"

    def _match_open_draft(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Resolve which open stream (if any) a turn-final send belongs to.

        Exact key match first. Callers carrying a per-turn MESSAGE id never
        fall back — their identity is authoritative. Callers without one
        (placement-only metadata, or none) may absorb into the chat's single
        open stream; with several open the send stays plain: a duplicate
        message is recoverable, sealing someone else's stream is not.
        """
        key = self._draft_key(str(chat_id), metadata)
        if key in self._open_draft_by_chat:
            return key
        md = metadata or {}
        if md.get("message_id") or md.get("reply_to_message_id"):
            return None
        prefix = f"{chat_id}:"
        candidates = [
            k for k in self._open_draft_by_chat if k.startswith(prefix)
        ]
        if len(candidates) == 1:
            # Absorbing a send into a stream is a significant decision (the
            # prompt-ack-seals-own-stream bug); log it so the next mismatch is a grep.
            logger.info(
                "relay: absorbing identity-less send into the single open "
                "stream %s (single-open-stream fallback)",
                candidates[0],
            )
            return candidates[0]
        return None

    async def _outbound(self, chat_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send one outbound frame tagged with the chat's underlying platform."""
        return await self._transport.send_outbound(  # type: ignore[union-attr]
            action, platform=self._platform_by_chat.get(str(chat_id))
        )

    def _text_metadata(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Metadata for a text egress frame: format hints + tenant discriminators.

        Boundary rule (live relay testing): draft, seal, send and edit are all
        text lanes — a streamed final can only render blocks if every frame
        carries the hint (a hintless seal is the plain-code-block downgrade).
        """
        return self._with_scope(chat_id, self._with_format_hints_for_chat(chat_id, metadata))

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self.supports_draft_streaming(chat_id=str(chat_id)):
            raise NotImplementedError(
                "connector does not advertise the 'draft' relay op"
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # Arm optimistically BEFORE the transport call (a lossy ack often means
        # delivered), but NEVER for a draft_id already sealed on this key: a
        # straggler after the seal re-armed interception with no live stream,
        # and the next unrelated send was wrongly converted into a seal.
        chat_key = self._draft_key(str(chat_id), metadata)
        if self._sealed_draft_by_chat.get(chat_key) == draft_id:
            # Post-seal straggler: content is already in the sealed message;
            # report success, send nothing. Log the FIRST swallow per key —
            # one straggler is the normal race, but a burst means something
            # sealed a live stream mid-flight (silence here cost a forensic hunt).
            if chat_key not in self._tombstone_swallow_logged:
                self._tombstone_swallow_logged[chat_key] = draft_id
                self._evict_oldest(self._tombstone_swallow_logged)
                logger.warning(
                    "relay: draft frame for %s swallowed by post-seal "
                    "tombstone (draft_id=%s) — expected for a straggler; "
                    "a live stream freezing NOW means something sealed it "
                    "mid-flight",
                    chat_key,
                    draft_id,
                )
            return SendResult(success=True)
        # Arm seal-interception ONLY for stream-is-the-message chats: on a
        # Telegram-shaped connector the final MUST go out as a real send.
        if self.stream_is_message_for_chat(str(chat_id)):
            self._open_draft_by_chat[chat_key] = draft_id
            self._evict_oldest(self._open_draft_by_chat)
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "draft",
                    "chat_id": chat_id,
                    "draft_id": draft_id,
                    "content": content,
                    "final": False,
                    "metadata": self._text_metadata(chat_id, dict(metadata or {})),
                },
            )
        except Exception as e:
            # Ambiguous (stale socket, mid-write drop): may have been delivered;
            # keep interception armed.
            return SendResult(success=False, error=f"draft transport error: {e}")
        if result.get("success"):
            return SendResult(success=True)
        if result.get("ambiguous"):
            # Ack lost (transport timeout, returned rather than raised): same
            # contract as the except branch — keep interception armed.
            return SendResult(
                success=False, error=str(result.get("error") or "draft ack lost")
            )
        # DEFINITE connector rejection: disarm. The stream consumer falls back
        # to edit-based streaming and its turn-final must go out as a REAL
        # send, not a seal on a stream the connector just declared unusable.
        if self._open_draft_by_chat.get(chat_key) == draft_id:
            self._open_draft_by_chat.pop(chat_key, None)
        return SendResult(
            success=False, error=str(result.get("error") or "draft failed")
        )

    async def _seal_open_draft(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
        *,
        draft_key: Optional[str] = None,
    ) -> SendResult:
        """Convert the turn-final send into the sealing draft frame."""
        if draft_key is None:
            draft_key = self._draft_key(str(chat_id), metadata)
        draft_id = self._open_draft_by_chat.pop(draft_key)
        # Tombstone BEFORE the transport call: whatever the ack says, this
        # draft_id must never be re-armed by a straggler frame. Bounded FIFO —
        # the straggler window is seconds, and the key embeds a per-turn identity.
        self._sealed_draft_by_chat[draft_key] = draft_id
        self._evict_oldest(self._sealed_draft_by_chat)
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        seal_frame = {
            "op": "draft",
            "chat_id": chat_id,
            "draft_id": draft_id,
            "content": content,
            "final": True,
            "metadata": self._text_metadata(chat_id, dict(metadata or {})),
        }

        _seal_platform = self._platform_by_chat.get(str(chat_id))
        _transport = self._transport  # narrowed by the None-guard above

        async def _attempt() -> Optional[Dict[str, Any]]:
            """One seal attempt; None means ambiguous (exception or lost ack)."""
            try:
                r = await _transport.send_outbound(seal_frame, platform=_seal_platform)
            except Exception as e:
                logger.warning("relay seal transport error (ambiguous): %s", e)
                return None
            if r.get("ambiguous"):
                logger.warning(
                    "relay seal ack lost (ambiguous): %s", r.get("error")
                )
                return None
            return r

        # Ambiguous outcomes retry the SAME idempotent frame once: the
        # connector's sealed-key tombstone returns the original stream ts for
        # a repeated final and never opens a second stream. Two consecutive
        # ack losses on one socket almost always mean the transport is down.
        #
        # Cancellation safety: the open entry was popped and the tombstone
        # written BEFORE the await. Restore both before re-raising so the
        # later abandon pass can still seal the stream.
        try:
            result = await _attempt()
            if result is None:
                result = await _attempt()
        except asyncio.CancelledError:
            self._open_draft_by_chat[draft_key] = draft_id
            if self._sealed_draft_by_chat.get(draft_key) == draft_id:
                self._sealed_draft_by_chat.pop(draft_key, None)
            raise
        if result is None:
            return SendResult(
                success=False,
                error="draft seal ambiguous after retry (transport ack lost)",
            )
        if result.get("success"):
            # The connector returns the stream's ts as the message identity.
            return SendResult(
                success=True,
                message_id=str(result.get("message_id") or "") or None,
            )
        return SendResult(
            success=False, error=str(result.get("error") or "draft seal failed")
        )

    async def _absorb_into_open_draft(
        self, chat_id: str, content: str, metadata: Dict[str, Any], interim: bool
    ) -> Optional[SendResult]:
        """Seal an open native stream with this turn-final; None = do a plain send.

        An open stream absorbs the turn-final no matter which egress door it
        arrives through (send / send_for_platform) — otherwise the stream is
        left frozen mid-word AND the final posts as a duplicate. A failed seal
        must NOT swallow the final: the consumer already disabled the draft
        transport, so fall through to a plain send (the orphaned stream is
        sealed connector-side by recycling / eviction). Interim sends
        (commentary, tail flush, lifecycle acks) never seal.
        """
        if interim:
            return None
        key = self._match_open_draft(str(chat_id), metadata)
        if key is None:
            return None
        seal = await self._seal_open_draft(chat_id, content, metadata, draft_key=key)
        if seal.success:
            return seal
        logger.warning(
            "relay seal failed (%s); delivering turn-final as plain send",
            seal.error,
        )
        return None

    async def send_native_task_card_progress(
        self,
        chat_id: str,
        tasks: list,
        *,
        title: str = "Hermes is working",
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_text: Optional[str] = None,
    ) -> SendResult:
        """Relay leg of the task-card lane: emit one card frame.

        SIGNATURE CONTRACT: the TurnRunner calls this with the NATIVE Slack
        adapter's keyword contract, not a card_id. ``fallback_text``/``title``
        are accepted for parity but not forwarded (the connector's plan-mode
        stream renders task chunks; field limits are enforced connector-side).
        """
        if not self.supports_native_task_cards():
            return SendResult(
                success=False, error="connector does not advertise task_card"
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        card_id = self._card_key(reply_to, metadata)
        merged_meta = dict(metadata or {})
        if reply_to and "thread_ts" not in merged_meta:
            # Slack card streams are thread replies anchored on the trigger.
            merged_meta["thread_ts"] = str(reply_to)
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "task_card",
                    "chat_id": chat_id,
                    "card_id": card_id,
                    "chunks": [dict(t) for t in tasks],
                    "metadata": self._with_scope(chat_id, merged_meta),
                },
            )
        except Exception as e:
            # Progress is advisory: degrade to the TurnRunner's text fallback,
            # never raise into the progress loop / turn-cleanup path (an
            # escaping card exception in cleanup skipped final delivery).
            return SendResult(
                success=False, error=f"task_card transport error: {e}"
            )
        if result.get("success"):
            return SendResult(success=True)
        return SendResult(
            success=False, error=str(result.get("error") or "task_card failed")
        )

    async def stop_native_task_card_progress(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Seal the card stream at turn end (idempotent connector-side); same key derivation as send."""
        if not self.supports_native_task_cards():
            return SendResult(
                success=False, error="connector does not advertise task_card"
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "task_card_stop",
                    "chat_id": chat_id,
                    "card_id": self._card_key(reply_to, metadata),
                    "metadata": self._with_scope(chat_id, dict(metadata or {})),
                },
            )
        except Exception as e:
            # Runs in the progress loop's finally block: an escaping exception
            # there skipped final delivery. A lost stop is cosmetic (the
            # connector seals orphaned card streams on its own).
            return SendResult(
                success=False, error=f"task_card_stop transport error: {e}"
            )
        return SendResult(success=bool(result.get("success")))

    async def abandon_open_draft(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Seal an orphaned stream when its turn dies (/stop, /new, supersede).

        Seals in place with ``content`` (the text already on screen) so the
        seal adds and claims nothing; otherwise the live indicator stays
        forever and the NEXT turn could inherit the armed interception state.
        Best-effort: failure is reported, never raised.
        """
        draft_key = self._match_open_draft(str(chat_id), metadata)
        if draft_key is None:
            return SendResult(success=True)  # nothing armed — no-op
        try:
            return await self._seal_open_draft(
                chat_id, content, metadata, draft_key=draft_key
            )
        except Exception as e:
            return SendResult(
                success=False, error=f"abandon seal transport error: {e}"
            )

    # ── abstract methods (delegated to the transport) ────────────────────
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # ``is_reconnect`` is part of the BasePlatformAdapter.connect contract
        # (the reconnect watcher calls connect(is_reconnect=True); refusing the
        # kwarg would break that recovery path). Relay IGNORES it: messages
        # buffered during a gap live in the CONNECTOR's durable buffer and
        # replay on re-handshake; routine WS drops are handled by the
        # transport's own reconnect supervisor.
        if self._transport is None:
            raise RuntimeError("RelayAdapter has no transport configured")
        self._transport.set_inbound_handler(self._on_inbound)
        # Interrupts and passthrough-plane forwards (Discord interactions,
        # Twilio, …) ride the SAME outbound WS — there is no inbound HTTP
        # receiver, so a hosted gateway needs no public port.
        set_interrupt = getattr(self._transport, "set_interrupt_inbound_handler", None)
        if callable(set_interrupt):
            set_interrupt(self.on_interrupt)
        set_passthrough = getattr(self._transport, "set_passthrough_handler", None)
        if callable(set_passthrough):
            set_passthrough(self._on_passthrough)
        ok = await self._transport.connect()
        if not ok:
            return False
        # Adopt the connector-advertised descriptor in place of the
        # construction-time placeholder.
        try:
            descriptor = await self._transport.handshake()
        except Exception as exc:  # noqa: BLE001 - a failed handshake = a failed connect
            logger.warning("relay handshake failed: %s", exc)
            return False
        self._apply_descriptor(descriptor)
        # Only the production WebSocket transport exposes `auth_revoked`.
        if hasattr(self._transport, "auth_revoked"):
            self._start_revocation_monitor()
        return True

    def _start_revocation_monitor(self) -> None:
        """Spawn (once) the task turning a transport auth-revocation into a
        clean non-retryable 'relay disabled' fatal. Idempotent."""
        if self._revocation_monitor is not None and not self._revocation_monitor.done():
            return
        try:
            self._revocation_monitor = asyncio.create_task(
                self._watch_for_revocation(), name="relay-revocation-monitor"
            )
        except RuntimeError:
            # No running loop (a unit test calling connect() via a stub).
            self._revocation_monitor = None

    async def _watch_for_revocation(self, poll_interval_s: float = 1.0) -> None:
        """Poll for a terminal 4401 revocation (opt-out); then surface a
        non-retryable `relay_disabled` fatal so the adapter is cleanly removed
        rather than queued for reconnection (the credential is dead until the
        instance is recreated)."""
        transport = self._transport
        if transport is None:
            return
        while not getattr(transport, "auth_revoked", False):
            await asyncio.sleep(poll_interval_s)
        logger.warning(
            "relay credential revoked (opt-out) — marking the relay adapter disabled"
        )
        self._set_fatal_error(
            "relay_disabled",
            "Relay disabled (opted out — recreate the instance to re-enable)",
            retryable=False,
        )
        try:
            await self._notify_fatal_error()
        except Exception:  # noqa: BLE001 - notification is best-effort
            logger.debug("relay revocation fatal-error notify failed", exc_info=True)

    def _apply_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Adopt a (re)negotiated descriptor into the live capability surface."""
        self.descriptor = descriptor
        self.MAX_MESSAGE_LENGTH = descriptor.max_message_length
        self.supports_code_blocks = descriptor.markdown_dialect not in ("", "plain")
        # Cron in_channel continuable surface (D6 gate in cron/scheduler.py);
        # class default is False, so only an explicit descriptor bit turns it on.
        self.supports_inchannel_continuable = bool(
            getattr(descriptor, "supports_inchannel_continuable", False)
        )

    async def _on_inbound(self, event) -> None:
        """Bridge a connector-delivered MessageEvent into the normal adapter path."""
        # Inbound replay dedupe: the relay leg is at-least-once — on WS
        # re-handshake the connector replays its durable buffer, and a long
        # turn straddling a quiet socket drop got re-run (final answer 2-5x).
        # Platform message identity is stable across replays.
        dedupe_key = self._inbound_dedupe_key(event)
        if dedupe_key is not None:
            if dedupe_key in self._seen_inbound:
                logger.info(
                    "relay inbound dropped as replay (dedupe key=%s)", dedupe_key
                )
                return
            self._seen_inbound[dedupe_key] = None
            while len(self._seen_inbound) > self._SEEN_INBOUND_MAX:
                self._seen_inbound.pop(next(iter(self._seen_inbound)))
        self._capture_scope(event)
        self._stamp_slack_session_thread(event)
        # A structured prompt answer resolves its waiting primitive and is
        # CONSUMED — never also dispatched as chat. Unknown/expired ids fall
        # through (command-shaped text then behaves like a typed reply).
        if await self._consume_prompt_response(event):
            return
        await self._localize_inbound_media(event)
        await self.handle_message(event)

    _SEEN_INBOUND_MAX = 512

    def _inbound_dedupe_key(self, event) -> Optional[str]:
        """Stable replay identity: (platform, chat, platform message id).

        The platform joins the key because one relay socket can front several
        platforms whose numeric ids may collide. Returns None when the event
        carries no platform message id — those never dedupe (fail-open:
        dropping a real message is worse than rerunning one).
        """
        source = getattr(event, "source", None)
        message_id = getattr(event, "message_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not message_id or not chat_id:
            return None
        # Enum value when present, plain string otherwise: both spellings of
        # one platform must produce ONE key.
        raw_platform = getattr(source, "platform", None)
        platform = getattr(raw_platform, "value", raw_platform) or ""
        return f"{platform}:{chat_id}:{message_id}"

    def _relay_slack_extra(self) -> Dict[str, Any]:
        """The Slack-behavior subset of the RELAY platform config.

        ``platforms.relay.extra.slack.*`` (relay-namespaced mirror of the
        native Slack knobs; ``platforms.slack`` keeps meaning native settings).
        Legacy fallback: flat keys on the relay extra still win when no
        ``slack`` object exists, preserving existing staging configs.
        """
        extra = getattr(self.config, "extra", None) or {}
        sub = extra.get("slack")
        return sub if isinstance(sub, dict) else extra

    @staticmethod
    def _coerce_flag(raw: Any, default: bool) -> bool:
        """Coerce an operator-supplied boolean exactly as native Slack does.

        A YAML-quoted ``"false"`` must turn the flag OFF; a bare ``bool()``
        would read that non-empty string as True and silently ignore the switch.
        """
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in _TRUTHY

    def _slack_flag(self, knob: str, default: bool) -> bool:
        """A coerced boolean knob from the relay Slack extra; ``default`` on any config-shape error."""
        try:
            return self._coerce_flag(self._relay_slack_extra().get(knob), default)
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return default

    def _effective_reply_in_thread(self) -> bool:
        """Resolve the thread-per-message vs flat-DM mode for fronted Slack."""
        return self._slack_flag("reply_in_thread", True)

    def _dm_top_level_threads_as_sessions(self) -> bool:
        """Native-parity escape hatch: per-message DM sessions on/off.

        Default True: in thread-per-message mode each top-level DM message keys
        its own session. False keeps threaded PLACEMENT but ONE rolling DM
        session (the legacy steer/queue posture), decoupled from reply_in_thread.
        """
        return self._slack_flag("dm_top_level_threads_as_sessions", True)

    def _slack_unfurl_hints(self, platform: Optional[str]) -> Optional[Dict[str, bool]]:
        """Slack-only outbound link-preview knobs (``unfurl_links``/``unfurl_media``).

        Reads the relay namespace like ``reply_in_thread``. Only explicitly
        configured booleans are returned (omitted keys preserve Slack's
        default); YAML strings ("true"/"false") are coerced, junk is dropped.
        Non-Slack platforms return None so their metadata is never polluted.
        """
        if str(platform or "").lower() != _SLACK:
            return None
        extra = self._relay_slack_extra()
        hints: Dict[str, bool] = {}
        for knob in ("unfurl_links", "unfurl_media"):
            val = extra.get(knob)
            if isinstance(val, bool):
                hints[knob] = val
            elif isinstance(val, str) and val.strip().lower() in (_TRUTHY | _FALSY):
                hints[knob] = val.strip().lower() in _TRUTHY
        return hints or None

    def _stamp_slack_unfurl(self, platform: Optional[str], metadata: Dict[str, Any]) -> None:
        unfurl = self._slack_unfurl_hints(platform)
        if unfurl:
            metadata.update(unfurl)

    def _stamp_slack_session_thread(self, event) -> None:
        """Native session-keying parity for fronted Slack DMs.

        Native Slack stamps ``thread_ts = event.thread_ts or ts``, so each
        TOP-LEVEL message keys a FRESH session (parallel turns). The connector
        normalizes top-level messages with thread_id=null, so without this every
        top-level DM collapsed into ONE session and message 2 pre-empted
        message 1. Only in thread-per-message mode (flat mode keeps the shared
        rolling session on purpose); never overwrites a real thread_id.
        """
        try:
            src = getattr(event, "source", None)
            if not src:
                return
            platform = getattr(src, "platform", None)
            if getattr(platform, "value", platform) != _SLACK:
                return
            if getattr(src, "thread_id", None):
                return  # real thread — its session key is already correct
            message_id = getattr(event, "message_id", None) or getattr(
                src, "message_id", None
            )
            if not message_id:
                return
            if not self._effective_reply_in_thread():
                return
            if not self._dm_top_level_threads_as_sessions():
                return  # opt-out: threaded replies, one rolling session
            src.thread_id = str(message_id)
        except Exception:  # noqa: BLE001 - session stamping must never break inbound
            logger.debug("slack session-thread stamp failed", exc_info=True)

    async def _localize_inbound_media(self, event) -> None:
        """Download connector re-hosted attachments to local temp paths.

        Every NATIVE adapter presents inbound media as LOCAL FILE PATHS (the
        vision/file tools consume paths), so mirror that. Best-effort per
        entry: a failed download drops that entry, never the message; with no
        client only re-host URLs are dropped (they'd 401 downstream), public
        URLs stay.
        """
        try:
            urls = list(getattr(event, "media_urls", None) or [])
            if not urls:
                return
            # media_types is INDEXED IN PARALLEL with media_urls by every
            # downstream classifier: carry (url, mime) PAIRS through the loop
            # or surviving attachments inherit a neighbour's type.
            types = list(getattr(event, "media_types", None) or [])
            pairs = [
                (u, types[i] if i < len(types) else "") for i, u in enumerate(urls)
            ]
            client = self._get_media_client()
            localized: list[tuple[str, str]] = []
            for url, mime in pairs:
                if not isinstance(url, str) or not url:
                    continue
                if client is None:
                    if "/relay/media/" not in url:
                        localized.append((url, mime))
                    continue
                path = await client.download(url)
                if path:
                    localized.append((path, mime))
                elif "/relay/media/" not in url:
                    # A public URL still has value as a URL; a dead re-host does not.
                    localized.append((url, mime))
            event.media_urls = [u for u, _ in localized]
            event.media_types = [m for _, m in localized]
        except Exception:  # noqa: BLE001 - media localization must never break inbound
            logger.debug("relay inbound media localization failed", exc_info=True)

    def prime_routing_cache(self, event) -> None:
        """Warm the per-chat egress routing caches from a SYNTHETIC event.

        A synthetic completion turn injected right after a restart (durable
        async-delegation replay) reaches handle_message with the caches COLD,
        so its replies egress without scope_id/user_id and the connector's
        fail-closed tenant guard declines them. Never raises.
        """
        if event is None or getattr(event, "source", None) is None:
            return
        self._capture_scope(event)

    def _capture_scope(self, event) -> None:
        """Remember a chat's egress discriminators from an inbound event. Never raises.

        - scope_id: scoped (guild/channel) message → routing-table resolution.
        - user_id: authentic author id, captured for EVERY message. Sole
          discriminator for a DM AND the author-first fallback for a scoped
          reply whose guild has no route row (managed agents join guilds
          dynamically). Without a resolvable discriminator the connector
          declines egress as 'target not routed to an onboarded tenant'.
        """
        try:
            src = getattr(event, "source", None)
            if not src:
                return
            chat = getattr(src, "chat_id", None)
            if not chat:
                return
            # Underlying platform's string VALUE, skipping the generic RELAY
            # fallback (the connector's session default handles egress then).
            platform = getattr(src, "platform", None)
            platform_value = getattr(platform, "value", platform)
            if platform_value and platform_value != "relay":
                self._platform_by_chat[str(chat)] = str(platform_value)
            user_id = getattr(src, "user_id", None)
            if user_id:
                self._dm_user_by_chat[str(chat)] = str(user_id)
            scope = getattr(src, "scope_id", None)
            if scope:
                self._scope_by_chat[str(chat)] = str(scope)
            chat_type = getattr(src, "chat_type", None)
            if chat_type:
                self._chat_type_by_chat[str(chat)] = str(chat_type)
            # Triggering message ts for the typing/status lane's synthetic
            # thread anchor (message_id lives on the EVENT; source is a fallback).
            message_id = getattr(event, "message_id", None) or getattr(
                src, "message_id", None
            )
            if message_id:
                self._last_inbound_ts_by_chat[str(chat)] = str(message_id)
        except Exception:  # noqa: BLE001 - scope tracking must never break inbound
            pass

    def _with_scope(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Outbound metadata carrying the tenant discriminators (see _capture_scope).

        Both scope_id and user_id are attached when known and not already set;
        the connector tries scope_id first and only falls back to user_id on a
        route miss, so carrying both never overrides routing-table resolution.
        """
        meta: Dict[str, Any] = dict(metadata or {})
        if not meta.get("scope_id"):
            scope = self._scope_by_chat.get(str(chat_id))
            if scope:
                meta["scope_id"] = scope
        if not meta.get("user_id"):
            author = self._dm_user_by_chat.get(str(chat_id))
            if author:
                meta["user_id"] = author
        return meta

    def fronts_platform(self, platform: Any) -> bool:
        """Whether the authenticated relay transport advertises ``platform``.

        Restart-safe delivery ownership signal: comes from the identity set
        sent at handshake, not from an inbound chat cache.
        """
        platform_value = getattr(platform, "value", platform)
        if not platform_value:
            return False
        ids = getattr(self._transport, "_identities", None)
        if not ids:
            return False
        return any(p == str(platform_value) for p, _ in ids)

    def supports_inchannel_continuable_for_platform(self, platform: Any) -> bool:
        """Whether ONE fronted platform can host the flat continuable cron
        surface (D6 gate). The scalar bit is the PRIMARY's only, so resolve the
        platform's own negotiated descriptor; fall back to the scalar when
        unavailable."""
        per_platform = self._negotiated_descriptor(
            str(getattr(platform, "value", platform) or "")
        )
        if per_platform is not None:
            return bool(getattr(per_platform, "supports_inchannel_continuable", False))
        return bool(self.supports_inchannel_continuable)

    async def on_interrupt(self, session_key: str, chat_id: str) -> None:
        """Bridge a connector-delivered /stop into the per-session interrupt path."""
        await self.interrupt_session_activity(session_key, chat_id)

    async def _on_passthrough(self, forward, buffer_id: Optional[str] = None) -> None:
        """Handle a connector-forwarded passthrough request.

        The connector answered the provider's latency-critical ACK at the
        edge, verified the signature and stripped any shared-identity
        credential into its vault; the agent later acts via the token-less
        ``send_follow_up`` path. A Discord interaction becomes a normalized
        ``MessageEvent`` on the SAME agent path as chat; other forwards are
        logged and dropped. NEVER raises: a malformed forward must not kill
        the read loop.
        """
        try:
            platform = getattr(forward, "platform", "") or ""
            if platform == "discord":
                event = self._discord_interaction_to_event(forward)
                if event is not None:
                    self._capture_scope(event)
                    # A component press carrying a Hermes prompt token resolves
                    # its waiting primitive and is consumed (same gate as _on_inbound).
                    if await self._consume_prompt_response(event):
                        return
                    await self.handle_message(event)
                    return
            logger.info(
                "relay passthrough_forward dropped (no handler): platform=%s method=%s path=%s",
                platform,
                getattr(forward, "method", "?"),
                getattr(forward, "path", "?"),
            )
        except Exception:  # noqa: BLE001 - a bad forward must never break the reader
            logger.warning("relay passthrough_forward handling failed", exc_info=True)

    def _discord_interaction_to_event(self, forward):
        """Convert a forwarded Discord interaction body to a MessageEvent, or None.

        The session source is built the way the connector builds it for an
        interaction (``interactionSessionSource``) so the session key matches
        the one the follow-up capability was bound under. None for an unusable
        body (a PING is answered at the edge and never forwarded).
        """
        try:
            payload = json.loads(bytes(getattr(forward, "body", b"")).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        # type 2 = APPLICATION_COMMAND; 3 = MESSAGE_COMPONENT; 5 = MODAL_SUBMIT.
        itype = payload.get("type")
        data = payload.get("data") or {}
        message_type = MessageType.TEXT
        if itype == 2:
            # Normalize to a leading-slash command string ("/name arg…"), the
            # shape the dispatcher and the connector's Slack slash lane expect.
            text = ("/" + str(data.get("name") or "")).rstrip("/") or ""
            if text:
                parts = [text] + self._render_interaction_options(data.get("options"))
                text = " ".join(parts).strip()
                message_type = MessageType.COMMAND
        elif itype == 3:
            text = str(data.get("custom_id") or "")
        else:
            text = ""
        member = payload.get("member") or {}
        user = (
            (member.get("user") if isinstance(member, dict) else None)
            or payload.get("user")
            or {}
        )
        channel_id = str(payload.get("channel_id") or "")
        guild_id = payload.get("guild_id")
        source = SessionSource(
            # The LOGICAL platform, not RELAY: session keys must match the
            # connector's capability binding (platform="discord"), /sethome
            # must file under the logical platform, and _capture_scope skips
            # the generic "relay" for egress routing.
            platform=Platform.DISCORD,
            chat_id=channel_id,
            # "group", not "channel": both the connector's capability binding
            # and the native Discord adapter key guild channels as "group".
            chat_type="group" if guild_id else "dm",
            user_id=str(user.get("id"))
            if isinstance(user, dict) and user.get("id")
            else None,
            user_name=str(user.get("username"))
            if isinstance(user, dict) and user.get("username")
            else None,
            scope_id=str(guild_id) if guild_id else None,
            message_id=str(payload.get("id")) if payload.get("id") else None,
            # Same upstream-trust marker the relay text lane stamps: arrived
            # over the authenticated relay WS after edge verification. Set
            # locally, never read off the wire (engages /sethome's via_relay guard).
            delivered_via_upstream_relay=True,
            # Profile routing (multiplex mode), mirroring _event_from_wire.
            profile=getattr(forward, "profile", None),
        )
        event = MessageEvent(text=text, message_type=message_type, source=source)
        if itype == 3:
            # A component press whose custom_id is a Hermes prompt token
            # (hp1:<prompt_id>:<option_id>) becomes a STRUCTURED prompt answer;
            # foreign custom_ids keep the best-effort TEXT shape.
            decoded = self._decode_prompt_token(text)
            if decoded:
                prompt_id, option_id = decoded
                msg = payload.get("message") or {}
                prompt_message_id = (
                    str(msg.get("id"))
                    if isinstance(msg, dict) and msg.get("id")
                    else None
                )
                event.prompt_response = {
                    "prompt_id": prompt_id,
                    "option_id": option_id,
                    "prompt_message_id": prompt_message_id,
                }
                event.text = f"/{option_id}"
                event.message_type = MessageType.COMMAND
        return event

    @staticmethod
    def _decode_prompt_token(token: str):
        """Decode an hp1:<prompt_id>:<option_id> callback token, or None (mirrors the connector's promptCodec)."""
        if not token:
            return None
        parts = token.split(":")
        if len(parts) != 3 or parts[0] != "hp1":
            return None
        if not _PROMPT_ID_RE.match(parts[1]) or not _PROMPT_ID_RE.match(parts[2]):
            return None
        return parts[1], parts[2]

    @staticmethod
    def _render_interaction_options(options) -> list:
        """Render Discord interaction options to space-separated text parts.

        Scalar options contribute just their value (native ``f"/model {name}"``
        shape); SUB_COMMAND (1) / SUB_COMMAND_GROUP (2) contribute their name
        then recurse into their nested options.
        """
        parts: list = []
        if not isinstance(options, list):
            return parts
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if opt.get("type") in (1, 2):
                sub_name = str(opt.get("name") or "").strip()
                if sub_name:
                    parts.append(sub_name)
                parts.extend(
                    RelayAdapter._render_interaction_options(opt.get("options"))
                )
            else:
                value = opt.get("value")
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
        return parts

    async def disconnect(self) -> None:
        # The runner wraps this call in wait_for(adapter disconnect budget).
        # Monitor teardown and go_idle eat into the transport's drain time, so
        # measure from the top and thread the REMAINDER down — otherwise
        # teardown is cancelled mid-drain and the transport's fail-pending
        # loop is skipped (callers then block on _OUTBOUND_TIMEOUT_S).
        from gateway.relay.ws_transport import _env_disconnect_budget_s
        _started = time.monotonic()
        _budget = _env_disconnect_budget_s()
        # Stop the revocation monitor first so it can't fire a spurious fatal
        # during/after a deliberate teardown.
        if self._revocation_monitor is not None:
            self._revocation_monitor.cancel()
            try:
                await asyncio.wait_for(
                    self._revocation_monitor,
                    timeout=_RELAY_REVOCATION_MONITOR_TEARDOWN_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort teardown
                pass
            self._revocation_monitor = None
        if self._transport is not None:
            # Ask the connector to flip this instance to buffered-only BEFORE
            # tearing down the socket, so inbound arriving while asleep buffers
            # durably and replays on reconnect. Best-effort: a transport
            # without go_idle (the stub) or a failed ack must not block shutdown.
            #
            # transport.disconnect() runs in finally so an outer cancellation
            # during go_idle still closes the socket/supervisor; shield() keeps
            # the teardown await itself from being cancelled mid-flight.
            try:
                go_idle = getattr(self._transport, "go_idle", None)
                if callable(go_idle):
                    try:
                        result: Any = go_idle(
                            timeout_s=_RELAY_GO_IDLE_ON_DISCONNECT_TIMEOUT_S
                        )
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001 - going-idle is an optimization, never blocks drain
                        logger.debug(
                            "relay going_idle failed during drain", exc_info=True
                        )
            finally:
                try:
                    _remaining = max(0.0, _budget - (time.monotonic() - _started))
                    try:
                        _td = self._transport.disconnect(budget_s=_remaining)  # type: ignore[call-arg]
                    except TypeError:
                        # Transports without the budget_s keyword (stubs).
                        _td = self._transport.disconnect()
                    await asyncio.shield(_td)
                except Exception:  # noqa: BLE001 - teardown must not block outer cancel propagation
                    logger.debug(
                        "relay transport disconnect failed during drain",
                        exc_info=True,
                    )

    async def go_dormant(self) -> bool:
        """Quiesce the relay for a scale-to-zero suspend.

        Unlike ``disconnect()`` this keeps the reconnect path armed so the
        gateway re-dials and drains its backlog on wake. A transport without
        ``go_dormant`` (the stub) is a no-op returning False. Deliberately
        does NOT stop the revocation monitor — dormancy is not a teardown.
        """
        if self._transport is None:
            return False
        go_dormant = getattr(self._transport, "go_dormant", None)
        if not callable(go_dormant):
            return False
        try:
            result: Any = go_dormant()
            if asyncio.iscoroutine(result):
                return bool(await result)
            return bool(result)
        except Exception:  # noqa: BLE001 - dormancy is best-effort, never blocks the idle path
            logger.debug("relay go_dormant failed", exc_info=True)
            return False

    async def send_for_platform(
        self,
        logical_platform: Any,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send to an explicitly advertised logical platform over Relay.

        Scheduled and persisted-home deliveries have no fresh inbound event to
        populate ``_platform_by_chat``. The delivery resolver calls this only
        after ``fronts_platform`` succeeds; repeated here fail-closed.
        """
        platform_value = getattr(logical_platform, "value", logical_platform)
        if not self.fronts_platform(platform_value):
            return SendResult(
                success=False,
                error=f"relay does not front platform {platform_value}",
            )
        _sfp_metadata = dict(metadata or {})
        # Gateway-internal interim marker (see send()): strip before the wire.
        _interim = bool(_sfp_metadata.pop("_interim_send", False))
        # The delivery resolver calls THIS method directly, bypassing send()
        # — an open native stream must absorb the turn-final here too.
        seal = await self._absorb_into_open_draft(chat_id, content, _sfp_metadata, _interim)
        if seal is not None:
            return seal
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        self._stamp_slack_unfurl(str(platform_value), _sfp_metadata)
        result = await self._transport.send_outbound(
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                # format_hints on the explicit-platform lane too: the cron
                # brief must render blocks exactly like an interactive send.
                "metadata": self._with_scope(
                    chat_id,
                    self._with_format_hints_for_platform(
                        str(platform_value), _sfp_metadata
                    ),
                ),
            },
            platform=str(platform_value),
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
            raw_response=result,
        )

    def _format_hints(
        self, descriptor: Optional[CapabilityDescriptor], platform: Optional[str]
    ) -> Optional[Dict[str, bool]]:
        """Block-formatting hints for one outbound text frame, or None.

        On the relay lane the CONNECTOR owns the platform API call, so the
        gateway only signals intent. Stamped ONLY when (a) the DESTINATION
        platform's negotiated descriptor advertises ``supports_block_formatting``
        (an old connector never receives dead metadata) and (b) the operator
        enabled ``platforms.relay.extra.<platform>.rich_blocks`` /
        ``markdown_blocks`` (both default OFF, same ``_coerce_flag`` semantics
        as reply_in_thread). ``descriptor``/``platform`` are the DESTINATION's,
        never the scalar primary: gating on the primary both leaked hints onto
        platforms that never advertised the bit and suppressed them for ones that did.
        """
        if descriptor is None or not getattr(
            descriptor, "supports_block_formatting", False
        ):
            return None
        try:
            extra = getattr(self.config, "extra", None) or {}
            sub = extra.get(str(platform or "").lower())
            knob_src = sub if isinstance(sub, dict) else extra
        except Exception:  # noqa: BLE001 - config shape is operator-owned
            return None
        hints: Dict[str, bool] = {}
        for knob in ("rich_blocks", "markdown_blocks"):
            if self._coerce_flag(knob_src.get(knob), False):
                hints[knob] = True
        return hints or None

    @staticmethod
    def _stamp_format_hints(
        hints: Optional[Dict[str, bool]], metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not hints:
            return metadata
        merged = dict(metadata or {})
        merged.setdefault("format_hints", hints)
        return merged

    def _with_format_hints_for_chat(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Metadata with ``format_hints`` stamped for a chat-addressed send
        (chat's platform as seen inbound, falling back to the primary)."""
        hints = self._format_hints(
            self._descriptor_for_chat(chat_id), self._chat_platform(chat_id)
        )
        return self._stamp_format_hints(hints, metadata)

    def _with_format_hints_for_platform(
        self, platform_value: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Metadata with ``format_hints`` stamped for an explicit-platform send
        (the scheduled/persisted-home lane). Falls back to the scalar
        descriptor only when it IS that platform's — never stamp from another
        platform's capability bit."""
        descriptor = self._negotiated_descriptor(str(platform_value))
        if descriptor is None and self.descriptor.platform == str(platform_value):
            descriptor = self.descriptor
        hints = self._format_hints(descriptor, str(platform_value))
        return self._stamp_format_hints(hints, metadata)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        send_metadata = dict(metadata or {})
        explicit_platform = send_metadata.pop("_relay_logical_platform", None)
        # Consumer-declared interim send (commentary, tail flush): NOT the
        # turn-final, so it must never trigger seal-interception (sealing the
        # live stream with interim text orphans the true final into a plain
        # duplicate). Gateway-internal marker; strip before the wire.
        _interim = bool(send_metadata.pop("_interim_send", False))
        # Seal-interception is checked BEFORE the explicit-platform branch:
        # an open stream absorbs the turn-final whichever door it arrives through.
        seal = await self._absorb_into_open_draft(chat_id, content, send_metadata, _interim)
        if seal is not None:
            return seal
        if explicit_platform:
            return await self.send_for_platform(
                explicit_platform,
                chat_id,
                content,
                reply_to=reply_to,
                metadata=send_metadata or None,
            )
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # Slack DM replies post flat at the DM root (native _resolve_thread_ts
        # parity); one shared helper for the text and media lanes.
        effective_reply_to = self._apply_slack_thread_anchor(
            chat_id, reply_to, send_metadata
        )
        self._stamp_slack_unfurl(self._chat_platform(chat_id), send_metadata)
        result = await self._outbound(
            chat_id,
            {
                "op": "send",
                "chat_id": chat_id,
                "content": content,
                "reply_to": effective_reply_to,
                "metadata": self._text_metadata(chat_id, send_metadata),
            },
        )
        # Auto-thread routing feedback: when the connector's auto-thread policy
        # routed this send into a thread it just created, the result carries
        # thread_id (+ initial name). The conversation was keyed on the PARENT
        # channel, so this is the only place the gateway learns where the reply landed.
        try:
            _at_thread = result.get("thread_id")
            _at_name = result.get("auto_thread_name")
            if _at_thread and _at_name:
                self._auto_thread_by_chat[str(chat_id)] = (
                    str(_at_thread),
                    str(_at_name),
                )
                if len(self._auto_thread_by_chat) > 256:
                    self._auto_thread_by_chat.pop(
                        next(iter(self._auto_thread_by_chat)), None
                    )
        except Exception:  # noqa: BLE001 - feedback capture must never break send
            pass
        # Wake the rename lane on EVERY send into this chat: "nowhere new" is
        # an answer it should get now rather than by outlasting a timeout.
        waiter = self._auto_thread_waiters.get(str(chat_id))
        if waiter is not None:
            waiter.set()
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

    def auto_thread_info_for_chat(
        self, chat_id: str
    ) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) of the connector-created auto-thread for the
        most recent send into *chat_id*, if any (semantic thread-rename lane)."""
        return self._auto_thread_by_chat.get(str(chat_id))

    async def wait_for_auto_thread_info(
        self, chat_id: str, timeout: float
    ) -> Optional[Tuple[str, str]]:
        """``auto_thread_info_for_chat``, but willing to wait for the send.

        The rename lane asks as soon as the session is titled — a whole turn
        early. Waits for the next send into this chat, so a reply the connector
        didn't auto-thread reports its miss immediately; *timeout* is only a
        backstop for a turn that never sends.
        """
        info = self.auto_thread_info_for_chat(chat_id)
        if info is not None:
            return info
        key = str(chat_id)
        waiter = self._auto_thread_waiters.get(key)
        if waiter is None:
            waiter = asyncio.Event()
            self._auto_thread_waiters[key] = waiter
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            # Only the waiter we installed, and only if no later call replaced
            # it; a fired event must not make the next turn's wait return instantly.
            if self._auto_thread_waiters.get(key) is waiter:
                self._auto_thread_waiters.pop(key, None)
        return self.auto_thread_info_for_chat(chat_id)

    def _resolve_reply_to_for_send(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Suppress the synthetic-DM thread anchor for a Slack DM reply.

        The stream consumer sends a DM reply with ``reply_to`` = the triggering
        ts (its edit anchor); the connector maps a raw reply_to to a Slack
        thread_ts, so the reply would thread under the user's message and lose
        progressive edit streaming. Native ``_resolve_thread_ts`` drops that
        anchor only when ``reply_in_thread`` is off; mirror it:

          Slack DM + no real ``thread_id``/``thread_ts`` + flat mode ⇒ drop.

        In thread-per-message mode the triggering ts IS the thread anchor and
        the final reply's ONLY threading signal (dropping it unconditionally
        exiled finals to the DM root while progress stayed threaded). Removes
        an anchor, never adds one; real threads and channel autoThread carry
        ``thread_id`` and are left alone.
        """
        if reply_to is None:
            return None
        if self._platform_by_chat.get(str(chat_id)) != _SLACK:
            return reply_to
        if self._chat_type_by_chat.get(str(chat_id)) != "dm":
            return reply_to
        md = metadata or {}
        if md.get("thread_id") or md.get("thread_ts"):
            return reply_to
        return reply_to if self._effective_reply_in_thread() else None

    def _apply_slack_thread_anchor(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Dict[str, Any],
        *,
        mirror_key: str = "reply_to_message_id",
    ) -> Optional[str]:
        """Resolve the outbound Slack thread anchor for ONE egress frame.

        The single choke point for text (``send``) and media (``_send_media``):
          1. Mode gate: ``_resolve_reply_to_for_send``.
          2. Mirror strip: when the anchor is dropped, remove the mirrored
             ``metadata.reply_to_message_id`` too, or the connector threads on it.
          3. Anchor promotion: the connector's Slack sender THREADS ON METADATA
             ONLY (``threadTs()`` never reads the frame's ``reply_to``), so a
             surviving anchor is promoted into ``metadata.thread_id``.

        ``metadata`` is mutated in place; the effective ``reply_to`` is returned.
        """
        effective_reply_to = self._resolve_reply_to_for_send(
            chat_id, reply_to, metadata
        )
        if effective_reply_to is None and reply_to is not None:
            metadata.pop(mirror_key, None)
        if (
            effective_reply_to is not None
            and self._platform_by_chat.get(str(chat_id)) == _SLACK
            and not (metadata.get("thread_id") or metadata.get("thread_ts"))
        ):
            metadata["thread_id"] = str(effective_reply_to)
        return effective_reply_to

    def _with_status_thread_anchor(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Copy ``metadata`` with the typing/status thread anchor applied.

        Slack's status line is THREAD-scoped and the typing lane's metadata
        carries no anchor for a top-level DM, so synthesize it from the
        per-chat inbound-ts cache (native ``send_typing`` parity). Shared by
        ``send_typing`` and ``stop_typing`` — the clear MUST target the same
        thread the heartbeat set or the status sticks until Slack's timeout.
        """
        md = dict(metadata or {})
        if (
            not (md.get("thread_id") or md.get("thread_ts"))
            and self._platform_by_chat.get(str(chat_id)) == _SLACK
            and self._chat_type_by_chat.get(str(chat_id)) == "dm"
        ):
            anchor = self._last_inbound_ts_by_chat.get(str(chat_id))
            if anchor:
                md["thread_id"] = anchor
        return md

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a relayed message through the connector-owned platform API."""
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        result = await self._outbound(
            chat_id,
            {
                "op": "edit",
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": self._text_metadata(chat_id, metadata),
            },
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id") or message_id,
            error=result.get("error"),
        )

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> bool:
        """Delete a relayed message (the stream consumer's fresh-final cleanup).

        Gated on the descriptor advertising ``delete``: older connectors return
        False so cleanup degrades to leaving the preview in place.
        """
        if self._transport is None:
            return False
        desc = self._descriptor_for_chat(str(chat_id))
        if "delete" not in (desc.supported_ops or ()):
            return False
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "delete",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "metadata": self._with_scope(chat_id, {}),
                },
            )
        except Exception:
            logger.debug("relay delete_message failed", exc_info=True)
            return False
        return bool(result.get("success"))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Egress a typing indicator through the connector.

        Bridges the base ``_keep_typing`` tick onto the ``typing`` op. Carries
        ``_with_scope`` (the egress guard wraps ALL ops) and the per-frame
        platform tag. Best-effort and one-shot: Discord/Telegram indicators
        self-expire; Slack Assistant status persists, so ``stop_typing``
        sends an explicit clear for Slack only.
        """
        if self._transport is None:
            return
        md = self._with_status_thread_anchor(chat_id, metadata)
        # Rich status parity: carry run.py's per-tool phrase as the frame's
        # content (rendered on assistant.threads.setStatus). Absent => omit
        # content and the connector uses its default heartbeat. NEVER send
        # empty-string content here: on Slack that is the CLEAR request.
        frame: Dict[str, Any] = {
            "op": "typing",
            "chat_id": chat_id,
            "metadata": self._with_scope(chat_id, md),
        }
        phrase = getattr(self, "_status_text", {}).get(str(chat_id))
        if phrase:
            frame["content"] = str(phrase)
        try:
            await self._outbound(chat_id, frame)
        except Exception:  # noqa: BLE001 - typing is cosmetic, never breaks a turn
            logger.debug("relay send_typing failed for %s", chat_id, exc_info=True)

    async def stop_typing(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forward an explicit typing/status clear (empty ``content``) — Slack only.

        Other relay senders have one-shot heartbeats, where an empty heartbeat
        would re-trigger typing at completion. Deploy-order note: a connector
        older than gateway-gateway #154 hardcodes the typing status and would
        SET it on a clear frame — deploy the connector first.
        """
        if self._transport is None:
            return
        if self._platform_by_chat.get(str(chat_id)) != _SLACK:
            return
        md = self._with_status_thread_anchor(chat_id, metadata)
        try:
            await self._outbound(
                chat_id,
                {
                    "op": "typing",
                    "chat_id": chat_id,
                    "content": "",
                    "metadata": self._with_scope(chat_id, md),
                },
            )
        except Exception:  # noqa: BLE001 - status clear is cosmetic, never breaks a turn
            logger.debug("relay stop_typing failed for %s", chat_id, exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Proxied to the connector; op-gated so a legacy connector (which would
        # only answer "unsupported op") gets the same local fallback without a round trip.
        if self._transport is None or not self.descriptor.supports_op("get_chat_info"):
            return {"name": chat_id, "type": "dm"}
        return await self._transport.get_chat_info(chat_id)

    async def send_follow_up(
        self,
        session_key: str,
        kind: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send via a shared-identity capability bound to a session.

        The gateway never holds the credential: it names the session and the
        capability ``kind``; the connector resolves the value from its vault
        and egresses (enforcing the tenant match).
        """
        if self._transport is None:
            return SendResult(success=False, error="no transport")
        # `kind` is platform-prefixed ("discord.interaction_token"): tag the
        # frame with that platform when we front it; otherwise the connector's
        # session default routes it.
        follow_up_platform = None
        if kind and "." in kind:
            prefix = kind.split(".", 1)[0]
            if self.fronts_platform(prefix):
                follow_up_platform = prefix
        result = await self._transport.send_follow_up(
            {
                "op": "follow_up",
                "session_key": session_key,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            },
            platform=follow_up_platform,
        )
        return SendResult(
            success=bool(result.get("success")),
            message_id=result.get("message_id"),
            error=result.get("error"),
        )

    # ── Phase 2 media ─────────────────────────────────────────────────────

    def _get_media_client(self) -> Optional[RelayMediaClient]:
        """Lazily build the authenticated /relay/media client from the SAME
        dial URL and per-gateway creds the WS uses; None when unavailable
        (media lanes then degrade to their pre-media fallbacks)."""
        if self._media_client is not None:
            return self._media_client
        try:
            from gateway.relay import relay_connection_auth, relay_url
            from gateway.relay.media import media_base_url

            url = relay_url()
            gateway_id, secret = relay_connection_auth()
            if not url:
                return None
            client = RelayMediaClient(media_base_url(url), gateway_id, secret)
            if not client.enabled:
                return None
            self._media_client = client
            return client
        except Exception:  # noqa: BLE001 - media plumbing must never break the adapter
            logger.debug("relay media client init failed", exc_info=True)
            return None

    async def _send_media(
        self,
        chat_id: str,
        *,
        media_kind: str,
        source: str,
        source_is_path: bool,
        caption: Optional[str] = None,
        filename: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Egress one media object via the connector's ``send_media`` op.

        ``source`` is a LOCAL path (uploaded to /relay/media first — the
        connector cannot reach our filesystem) or an already-public URL
        (passed through). Returns None when the lane is unavailable (op not
        advertised, no transport, upload failed, connector decline) so each
        caller falls back to its pre-media behaviour.
        """
        if self._transport is None or not self.descriptor.supports_op("send_media"):
            return None
        source_url = source
        if source_is_path:
            client = self._get_media_client()
            if client is None:
                return None
            uploaded = await client.upload(source, filename=filename)
            if not uploaded:
                return None
            source_url = uploaded
        # Same Slack thread-anchor contract as the text lane: media frames go
        # through the connector's Slack sender too (threadTs() reads metadata only).
        media_metadata: Dict[str, Any] = dict(metadata or {})
        effective_reply_to = self._apply_slack_thread_anchor(
            chat_id, reply_to, media_metadata
        )
        self._stamp_slack_unfurl(self._chat_platform(chat_id), media_metadata)
        action: Dict[str, Any] = {
            "op": "send_media",
            "chat_id": chat_id,
            "media_kind": media_kind,
            "source_url": source_url,
            "content": caption or "",
            "reply_to": effective_reply_to,
            "metadata": self._with_scope(chat_id, media_metadata),
        }
        if filename:
            action["filename"] = filename
        try:
            result = await self._outbound(chat_id, action)
        except Exception:  # noqa: BLE001 - transport failure degrades to the caller's fallback
            logger.debug("relay send_media transport failure", exc_info=True)
            return None
        if not result.get("success"):
            # Structured connector decline (size cap, platform rejection): the
            # caller's fallback still delivers the caption/notice.
            logger.warning(
                "relay send_media declined for %s: %s",
                chat_id,
                result.get("error"),
            )
            return None
        return SendResult(
            success=True,
            message_id=result.get("message_id"),
            raw_response=result,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image (public URL) as a native attachment via the connector."""
        result = await self._send_media(
            chat_id, media_kind="image", source=image_url, source_is_path=False,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_image(
            chat_id, image_url, caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file natively (upload → send_media)."""
        result = await self._send_media(
            chat_id, media_kind="image", source=image_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_image_file(
            chat_id, image_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local audio file as a native voice message (upload → send_media)."""
        result = await self._send_media(
            chat_id, media_kind="voice", source=audio_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_voice(
            chat_id, audio_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local video file natively (upload → send_media)."""
        result = await self._send_media(
            chat_id, media_kind="video", source=video_path, source_is_path=True,
            caption=caption, reply_to=reply_to, metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_video(
            chat_id, video_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local file as a downloadable attachment (upload → send_media)."""
        result = await self._send_media(
            chat_id, media_kind="document", source=file_path, source_is_path=True,
            caption=caption, filename=file_name, reply_to=reply_to, metadata=metadata,
        )
        if result is not None:
            return result
        return await super().send_document(
            chat_id, file_path, caption=caption, file_name=file_name,
            reply_to=reply_to, metadata=metadata, **kwargs,
        )

    # ── Phase 3 interactive: prompt + react ──────────────────────────────

    def _mint_prompt(
        self, kind: str, state: Dict[str, Any], timeout_s: float = 3600.0
    ) -> str:
        """Register a pending prompt and return its id (``<owner nonce>.<8 hex>``).

        ``state`` carries what the resolver needs when the answer comes back.
        Expiry is enforced gateway-side on consumption (_pop_prompt); the
        wire's timeout_s is advisory. The nonce marks the minting process so a
        sibling gateway receiving the fanned-out answer stays quiet. Both
        segments use the connector codec's alphabet ([A-Za-z0-9_.-], <=32).
        """
        prompt_id = f"{self._prompt_owner_nonce}.{secrets.token_hex(4)}"
        self._pending_prompts[prompt_id] = {
            **state,
            "kind": kind,
            "expires_at": time.time() + timeout_s,
        }
        # Opportunistic sweep so abandoned prompts can't accumulate.
        now = time.time()
        for stale in [
            k for k, v in self._pending_prompts.items() if v.get("expires_at", 0) < now
        ]:
            self._pending_prompts.pop(stale, None)
        return prompt_id

    def _minted_here(self, prompt_id: str) -> bool:
        """True when this process minted ``prompt_id``. Ids without a ``.``
        segment predate the owner nonce (in-flight across an in-place upgrade)
        and are treated as ours."""
        head, sep, _ = str(prompt_id).partition(".")
        return head == self._prompt_owner_nonce if sep else True

    def _pop_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Consume a pending prompt: one answer wins, expired entries miss."""
        state = self._pending_prompts.pop(str(prompt_id), None)
        if not state:
            return None
        if state.get("expires_at", 0) < time.time():
            return None
        return state

    def _note_prompt_resolved(self, prompt_id: str) -> None:
        """Remember that this process answered ``prompt_id`` (bounded FIFO: a
        repeat is only interesting while a redelivery/double tap can arrive)."""
        self._resolved_prompts[str(prompt_id)] = time.time()
        while len(self._resolved_prompts) > _RESOLVED_PROMPT_MEMORY:
            self._resolved_prompts.popitem(last=False)

    async def _send_prompt(
        self,
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        prompt_id: str,
        options: list,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[SendResult]:
        """Egress one `prompt` op; None when the lane is unavailable (the
        caller falls back to its numbered-text base behaviour).

        Prompt metadata is forwarded VERBATIM: the threading mode is decided
        in exactly one place — run.py's _resolve_progress_thread_id (flat mode
        suppresses the synthetic self-anchor there; thread mode stamps the
        turn's thread). Boundary pinned by test_run_py_suppresses_self_anchor*.
        """
        if self._transport is None or not self.descriptor.supports_op("prompt"):
            return None
        action: Dict[str, Any] = {
            "op": "prompt",
            "chat_id": chat_id,
            "content": text,
            "prompt_kind": prompt_kind,
            "prompt_id": prompt_id,
            "options": options,
            "reply_to": self._resolve_reply_to_for_send(chat_id, reply_to, metadata),
            "metadata": self._with_scope(chat_id, metadata),
        }
        if timeout_s is not None:
            action["timeout_s"] = int(timeout_s)
        try:
            result = await self._outbound(chat_id, action)
        except Exception:  # noqa: BLE001 - transport failure degrades to fallback
            logger.debug("relay prompt transport failure", exc_info=True)
            return None
        if not result.get("success"):
            logger.warning(
                "relay prompt declined for %s: %s", chat_id, result.get("error")
            )
            return None
        return SendResult(
            success=True,
            message_id=result.get("message_id"),
            raw_response=result,
        )

    async def _mint_and_send_prompt(
        self,
        kind: str,
        state: Dict[str, Any],
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        options: list,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Register + egress a prompt; unregisters and returns None when the lane is unavailable."""
        prompt_id = self._mint_prompt(kind, {**state, "chat_id": str(chat_id)})
        result = await self._send_prompt(
            chat_id,
            prompt_kind=prompt_kind,
            text=text,
            prompt_id=prompt_id,
            options=options,
            metadata=metadata,
        )
        if result is None:
            self._pending_prompts.pop(prompt_id, None)
        return result

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Native-button exec approval over the relay.

        Same choice set as the native adapters; the press resolves via
        tools.approval.resolve_gateway_approval. When the lane is unavailable
        the send FAILS (success=False) so run.py's button→text fallback runs.
        """
        options: list = [{"id": "once", "label": "Allow Once", "style": "primary"}]
        if not smart_denied and allow_session:
            options.append({"id": "session", "label": "Allow Session"})
            if allow_permanent:
                options.append({"id": "always", "label": "Always Allow"})
        options.append({"id": "deny", "label": "Deny", "style": "danger"})

        cmd_preview = command if len(command) <= 1500 else command[:1500] + "..."
        text = (
            "⚠️ **Command Approval Required**\n\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {description}"
        )
        if smart_denied:
            text += (
                "\n\n**Smart DENY:** owner override applies to this one operation only."
            )
        result = await self._mint_and_send_prompt(
            "exec_approval",
            {"session_key": session_key},
            chat_id,
            prompt_kind="approval",
            text=text,
            options=options,
            metadata=metadata,
        )
        if result is not None:
            return result
        return SendResult(success=False, error="relay prompt op unavailable")

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Three-button slash-command confirmation over the relay (resolves via
        tools.slash_confirm.resolve; success=False falls back to text-intercept)."""
        options = [
            {"id": "once", "label": "Approve Once", "style": "primary"},
            {"id": "always", "label": "Always Approve"},
            {"id": "cancel", "label": "Cancel", "style": "danger"},
        ]
        result = await self._mint_and_send_prompt(
            "slash_confirm",
            {"session_key": session_key, "confirm_id": confirm_id},
            chat_id,
            prompt_kind="approval",
            text=f"**{title}**\n\n{message}" if title else message,
            options=options,
            metadata=metadata,
        )
        if result is not None:
            return result
        return SendResult(success=False, error="relay prompt op unavailable")

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Native-button clarify over the relay.

        A press resolves with the CHOICE TEXT (never the option id); "Other"
        flips to text-capture. Option ids are positional (c0..cN / other) —
        choice text is arbitrary UTF-8 and would blow the 64-byte callback
        budget. Open-ended clarifies and unavailable lanes fall back to base.
        """
        if choices and self.descriptor.supports_op("prompt"):
            options = [
                {"id": f"c{i}", "label": str(choice)[:75]}
                for i, choice in enumerate(choices)
            ]
            options.append({"id": "other", "label": "✏️ Other (type your answer)"})
            result = await self._mint_and_send_prompt(
                "clarify",
                {
                    "session_key": session_key,
                    "clarify_id": clarify_id,
                    "choices": [str(c) for c in choices],
                },
                chat_id,
                prompt_kind="clarify",
                text=f"❓ {question}",
                options=options,
                metadata=metadata,
            )
            if result is not None:
                return result
        return await super().send_clarify(
            chat_id, question, choices, clarify_id, session_key, metadata=metadata
        )

    async def _consume_prompt_response(self, event) -> bool:
        """Route an inbound prompt_response to its waiting primitive.

        Returns True when the event was a prompt answer (consumed — never
        dispatched as chat). Every prompt answer is consumed, whoever owns it:
        a sibling's prompt (the connector fans the press to every gateway of
        the tenant; falling through produced a wall of "Unknown command"), a
        repeat answer (first one won), or our own expired/unknown prompt
        (answered with a short expiry notice — option ids are not commands).
        """
        pr = getattr(event, "prompt_response", None)
        if not isinstance(pr, dict):
            return False
        prompt_id = str(pr.get("prompt_id") or "")
        option_id = str(pr.get("option_id") or "")
        if not prompt_id or not option_id:
            return False
        if not self._minted_here(prompt_id):
            logger.debug(
                "relay prompt_response %s (option=%s) belongs to another "
                "gateway instance — ignoring",
                prompt_id,
                option_id,
            )
            return True
        if prompt_id in self._resolved_prompts:
            logger.debug(
                "relay prompt_response %s (option=%s) already resolved — ignoring "
                "repeat",
                prompt_id,
                option_id,
            )
            return True
        state = self._pop_prompt(prompt_id)
        if state is None:
            logger.info(
                "relay prompt_response for unknown/expired prompt %s (option=%s)",
                prompt_id,
                option_id,
            )
            await self._notify_prompt_expired(event)
            return True
        self._note_prompt_resolved(prompt_id)

        kind = state.get("kind")
        chat_id = str(state.get("chat_id") or getattr(event.source, "chat_id", ""))
        handler = _PROMPT_RESOLVERS.get(kind)
        try:
            if handler is None:
                logger.warning("relay prompt_response with unknown kind %r", kind)
            else:
                # Acks are fire-and-forget: we are ON the read loop here (see
                # _send_lifecycle_ack) and awaiting a send would self-deadlock.
                await handler(self, state, option_id, chat_id, self._prompt_reply_metadata(event))
        except Exception:  # noqa: BLE001 - a resolver failure must not kill the reader
            logger.warning("relay prompt_response resolution failed", exc_info=True)
        return True

    async def _resolve_exec_approval(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.approval import resolve_gateway_approval

        choice = option_id if option_id in {"once", "session", "always", "deny"} else "deny"
        count = resolve_gateway_approval(str(state.get("session_key") or ""), choice)
        label = {
            "once": "✅ Approved once",
            "session": "✅ Approved for session",
            "always": "✅ Approved permanently",
            "deny": "❌ Denied",
        }.get(choice, "Resolved")
        if not count:
            label = "⌛ Approval expired — no command was waiting."
        # In-channel ack preserves the audit trail the native edit gives (the
        # connector's prompt message can't be edited cross-platform yet).
        self._send_lifecycle_ack(chat_id, label, ack_meta)
        if count:
            self.resume_typing_for_chat(chat_id)

    async def _resolve_slash_confirm(self, state, option_id, chat_id, ack_meta) -> None:
        from tools import slash_confirm as slash_confirm_mod

        choice = option_id if option_id in {"once", "always", "cancel"} else "cancel"
        result_text = await slash_confirm_mod.resolve(
            str(state.get("session_key") or ""), str(state.get("confirm_id") or ""), choice
        )
        label = {
            "once": "✅ Approved once",
            "always": "🔒 Always approve",
            "cancel": "❌ Cancelled",
        }.get(choice, "Resolved")
        self._send_lifecycle_ack(chat_id, label, ack_meta)
        if result_text:
            self._send_lifecycle_ack(chat_id, str(result_text), ack_meta)

    async def _resolve_clarify(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

        clarify_id = str(state.get("clarify_id") or "")
        if option_id == "other":
            mark_awaiting_text(clarify_id)
            self._send_lifecycle_ack(chat_id, "✏️ Type your answer:", ack_meta)
            return
        choices = state.get("choices") or []
        try:
            idx = int(option_id[1:]) if option_id.startswith("c") else -1
        except ValueError:
            idx = -1
        if 0 <= idx < len(choices):
            resolve_gateway_clarify(clarify_id, str(choices[idx]))
            self._send_lifecycle_ack(chat_id, f"✅ {choices[idx]}", ack_meta)
        else:
            # Unmappable option: flip to text capture (never dead-end a clarify).
            mark_awaiting_text(clarify_id)

    def _send_lifecycle_ack(
        self, chat_id: str, text: str, metadata: Dict[str, Any]
    ) -> None:
        """Fire-and-forget a prompt-lifecycle ack from read-loop context.

        _consume_prompt_response executes ON the transport read loop; an
        ``await self.send(...)`` there is a SELF-DEADLOCK (send() blocks on an
        outbound_result future only the read loop can resolve) — every button
        tap wedged the transport for the full outbound timeout. Acks are
        cosmetic, so they ride a background task; failures log at debug. The
        task ref is retained (asyncio only weakly references tasks).
        """

        async def _ack() -> None:
            try:
                await self.send(chat_id, text, metadata=metadata)
            except Exception:  # noqa: BLE001 - ack is best-effort
                logger.debug("relay lifecycle ack failed", exc_info=True)

        task = asyncio.create_task(_ack(), name="relay-lifecycle-ack")
        self._lifecycle_ack_tasks.add(task)
        task.add_done_callback(self._lifecycle_ack_tasks.discard)

    async def _notify_prompt_expired(self, event) -> None:
        """Tell the presser their prompt is no longer waiting (owning gateway only, best-effort)."""
        chat_id = str(getattr(event.source, "chat_id", "") or "")
        if not chat_id:
            return
        self._send_lifecycle_ack(
            chat_id,
            "⌛ That prompt is no longer waiting for an answer. "
            "Send your reply as a normal message.",
            self._prompt_reply_metadata(event),
        )

    def _prompt_reply_metadata(self, event) -> Dict[str, Any]:
        """Thread metadata so prompt acks land where the prompt lives.

        Marked INTERIM: acks fire while the approval turn's OWN draft stream
        is open and carry only placement metadata, so send()'s
        single-open-stream fallback sealed the live draft with the ack text
        (frozen stream + duplicate final on every approval turn).
        """
        meta: Dict[str, Any] = {"_interim_send": True}
        thread_id = getattr(event.source, "thread_id", None)
        if thread_id:
            meta["thread_id"] = str(thread_id)
        return meta

    # ── Phase 3 ack lifecycle (👀 → ✅/❌) ────────────────────────────────

    async def _react(
        self,
        chat_id: str,
        message_id: str,
        emoji: str,
        *,
        remove: bool = False,
    ) -> bool:
        """Egress one `react` op; best-effort (False on any failure, logged at debug)."""
        if self._transport is None or not self.descriptor.supports_op("react"):
            return False
        if not chat_id or not message_id:
            return False
        try:
            result = await self._outbound(
                chat_id,
                {
                    "op": "react",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "emoji": emoji,
                    "remove": remove,
                    "metadata": self._with_scope(chat_id, None),
                },
            )
            return bool(result.get("success"))
        except Exception:  # noqa: BLE001 - reactions are cosmetic
            logger.debug("relay react failed", exc_info=True)
            return False

    async def on_processing_start(self, event) -> None:
        """Add the 👀 in-progress reaction (op-gated; silent no-op otherwise)."""
        message_id, chat_id = _event_ids(event)
        if message_id and chat_id:
            await self._react(str(chat_id), str(message_id), "👀")

    async def on_processing_complete(self, event, outcome) -> None:
        """Swap 👀 for ✅/❌ per outcome (op-gated; silent no-op otherwise)."""
        message_id, chat_id = _event_ids(event)
        if not (message_id and chat_id):
            return
        await self._react(str(chat_id), str(message_id), "👀", remove=True)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._react(str(chat_id), str(message_id), "✅")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._react(str(chat_id), str(message_id), "❌")

    # ── Phase 4 thread lifecycle ──────────────────────────────────────────

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a thread/topic under ``parent_chat_id`` via the connector.

        One `thread_create` op covers Discord (channel thread), Telegram
        (forum topic) and Slack (named seed root message). None on any
        failure/unavailability so the handoff watcher falls back to the parent.
        """
        if self._transport is None or not self.descriptor.supports_op("thread_create"):
            return None
        thread_name = (str(name or "").strip() or "handoff")[:100]
        try:
            result = await self._outbound(
                str(parent_chat_id),
                {
                    "op": "thread_create",
                    "chat_id": str(parent_chat_id),
                    "thread_name": thread_name,
                    "metadata": self._with_scope(str(parent_chat_id), None),
                },
            )
        except Exception:  # noqa: BLE001 - handoff falls back to the parent channel
            logger.debug("relay thread_create transport failure", exc_info=True)
            return None
        if not result.get("success"):
            logger.info(
                "relay thread_create declined for %s: %s",
                parent_chat_id,
                result.get("error"),
            )
            return None
        thread_id = result.get("thread_id") or result.get("message_id")
        return str(thread_id) if thread_id else None

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: Optional[str] = None,
        prefer_connector_created: bool = False,
        parent_chat_id: Optional[str] = None,
    ) -> bool:
        """Best-effort thread rename via the connector's `thread_rename` op.

        Prefer ``prefer_connector_created=True``: the CONNECTOR enforces the
        no-clobber guard from its own created-name memory, so the gateway need
        not reproduce the initial name byte-for-byte (any normalization drift
        silently declined every rename). ``only_if_current_name`` is the legacy
        string guard for older connectors. ``parent_chat_id`` defaults to the
        thread id (Telegram needs the containing chat; Discord ignores it).
        """
        if self._transport is None or not self.descriptor.supports_op("thread_rename"):
            return False
        cleaned = " ".join(str(name or "").split()).strip()
        if not cleaned or not thread_id:
            return False
        chat_id = str(parent_chat_id or thread_id)
        action: Dict[str, Any] = {
            "op": "thread_rename",
            "chat_id": chat_id,
            "message_id": str(thread_id),
            "thread_name": cleaned[:100],
            "metadata": self._with_scope(chat_id, None),
        }
        if prefer_connector_created:
            action["only_if_connector_created"] = True
        elif only_if_current_name is not None:
            action["only_if_current_name"] = str(only_if_current_name)
        try:
            result = await self._transport.send_outbound(
                action,
                platform=self._platform_by_chat.get(chat_id)
                or self._platform_by_chat.get(str(thread_id)),
            )
        except Exception:  # noqa: BLE001 - renames are cosmetic
            logger.debug("relay thread_rename transport failure", exc_info=True)
            return False
        if not result.get("success"):
            logger.info(
                "relay thread_rename declined for %s: %s",
                thread_id,
                result.get("error"),
            )
            return False
        return True


# prompt kind -> resolver (order-independent: kinds are distinct keys).
_PROMPT_RESOLVERS = {
    "exec_approval": RelayAdapter._resolve_exec_approval,
    "slash_confirm": RelayAdapter._resolve_slash_confirm,
    "clarify": RelayAdapter._resolve_clarify,
}
