"""User-authorization methods for ``GatewayRunner``.

Inbound-message authorization cluster: whether a user/chat may talk to the
agent, the per-adapter DM policy, and the unauthorized-DM behavior.

``self.*`` calls resolve via the MRO. ``gateway.run`` is never imported at
module import time (cycle); the one method that logs imports its ``logger``
lazily so records keep the ``"gateway.run"`` logger name.
"""

from __future__ import annotations

import os
from typing import Optional

from gateway.config import Platform
from gateway.pairing import _PLATFORM_ALLOWLIST_ENV
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)

_GROUP_CHAT_TYPES = frozenset({"group", "forum", "channel"})
_TRUTHY = frozenset({"true", "1", "yes"})

# Platform -> ``<PLATFORM>_ALLOWED_USERS`` / ``<PLATFORM>_ALLOW_ALL_USERS``.
# Shared with the pairing store's allowlist mirror (single source of truth);
# plugin platforms are added per-call from the platform registry.
_ALLOWED_USERS_ENV = {Platform(k): v for k, v in _PLATFORM_ALLOWLIST_ENV.items()}
_ALLOW_ALL_ENV = {
    p: v.replace("_ALLOWED_USERS", "_ALLOW_ALL_USERS") for p, v in _ALLOWED_USERS_ENV.items()
}
_GROUP_USER_ENV = {Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_USERS"}
_GROUP_CHAT_ENV = {
    Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
    Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
}
_ALLOW_BOTS_ENV = {
    Platform.DISCORD: "DISCORD_ALLOW_BOTS",
    Platform.FEISHU: "FEISHU_ALLOW_BOTS",
    Platform.TELEGRAM: "TELEGRAM_ALLOW_BOTS",
    Platform.SLACK: "SLACK_ALLOW_BOTS",
}


def _platform_gate_env(name: str, default: str = "") -> str:
    """Read an allow/deny gate env var with per-profile isolation.

    When a profile secret scope is installed AND multiplexing is active, a
    scoped miss returns ``default`` instead of falling through to
    ``os.environ``: the process env may hold ANOTHER profile's first-writer
    bridged value, so a fallthrough would leak allowlists across profiles.
    Single-profile deployments behave exactly like ``os.getenv``.
    """
    if not name:
        return default
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        if scope is not None and is_multiplex_active():
            val = scope.get(name)
            return default if val is None else str(val).strip()
    except Exception:
        pass
    return (os.getenv(name) or default).strip()


_auth_env = _platform_gate_env


def _registry_entry(platform):
    """Platform-registry entry for a (plugin) platform, or None."""
    if platform is None:
        return None
    try:
        from gateway.platform_registry import platform_registry

        return platform_registry.get(platform.value)
    except Exception:
        return None


def _platform_declares_allowed_users_env(platform) -> bool:
    """Whether a plugin platform's registry entry declares ``allowed_users_env``.

    Such platforms (Buzz, DingTalk, ...) document ``PlatformConfig.extra
    .allowed_users`` as the config-file spelling of that env allowlist, so the
    live adapter's extra is a valid authorization source when the env var is
    absent. Built-in platforms and unknown entries return False.
    """
    entry = _registry_entry(platform)
    return bool(entry and entry.allowed_users_env)


def _coerce_allow_set(raw) -> set[str]:
    """Parse an allowlist (YAML list or comma-separated scalar) into a set of strings."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _allows(allowed: set[str], candidate: Optional[str]) -> bool:
    return "*" in allowed or candidate in allowed


# ---------------------------------------------------------------------------
# Nostr npub -> hex (Buzz). ``BUZZ_ALLOWED_USERS`` accepts hex or ``npub1…``
# but inbound pubkeys are always hex, so npub entries must be decoded.
# Pure stdlib; mirrors plugins/platforms/buzz/adapter.py.
# ---------------------------------------------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values):
    chk = 1
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data, frombits: int, tobits: int, pad: bool = True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def _npub_to_hex(npub: str) -> Optional[str]:
    """Decode an ``npub1…`` bech32 string to a 64-char hex pubkey, else None."""
    npub = npub.strip().lower()
    if not npub.startswith("npub1"):
        return None
    try:
        data = [_BECH32_CHARSET.index(c) for c in npub[len("npub1"):]]
    except ValueError:
        return None
    if _bech32_polymod(_bech32_hrp_expand("npub") + data) != 1:
        return None
    decoded = _convertbits(data[:-6], 5, 8, pad=False)
    if decoded is None or len(decoded) != 32:
        return None
    return bytes(decoded).hex()


def _normalize_nostr_allow_entries(entries: set) -> set:
    """Add the hex form of every valid ``npub1…`` entry; invalid entries are kept as-is."""
    expanded = set(entries)
    for entry in entries:
        if entry.lower().startswith("npub1"):
            hex_key = _npub_to_hex(entry)
            if hex_key:
                expanded.add(hex_key)
    return expanded


class GatewayAuthorizationMixin:
    """User/chat authorization methods for ``GatewayRunner``."""

    # ``getattr(self, ...)`` throughout: test helpers build bare runners via
    # ``object.__new__`` without ``adapters`` / ``config`` (pitfalls.md #17).

    def _authorization_adapter(
        self,
        platform: Optional[Platform],
        profile: Optional[str] = None,
    ):
        """Resolve the live adapter whose intake policy should gate authorization.

        Secondary-profile adapters live in ``_profile_adapters[profile]``; the
        launched (primary) profile owns ``self.adapters``. ``_profile_adapters``
        is consulted BEFORE comparing against the active profile name: multiplex
        turns override ``HERMES_HOME`` so ``_active_profile_name()`` reports the
        secondary profile mid-turn, and treating it as primary would hand it the
        default bot (or default-deny secondary-only platforms like A2A).
        """
        if not platform:
            return None
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            profile_adapters = getattr(self, "_profile_adapters", None) or {}
            if profile_name in profile_adapters:
                return profile_adapters[profile_name].get(platform)
            # Compare against the identity captured at construction, not the
            # per-turn HERMES_HOME-derived name.
            primary_profile = getattr(self, "_primary_profile_name", None)
            if not primary_profile:
                active_profile_fn = getattr(self, "_active_profile_name", None)
                if callable(active_profile_fn):
                    try:
                        primary_profile = active_profile_fn()
                    except Exception:
                        primary_profile = None
            if profile_name == primary_profile:
                return (getattr(self, "adapters", None) or {}).get(platform)
            # Fail closed: a stamped secondary profile with no registry entry
            # (adapter failed to connect) must NOT fall back to the default
            # profile's adapter -- that sends replies out the wrong bot.
            return None
        return (getattr(self, "adapters", None) or {}).get(platform)

    def _adapter_for_source(self, source: Optional[SessionSource]):
        """Resolve the live adapter for an inbound ``SessionSource``."""
        if source is None:
            return None
        transport_adapter = self._registered_transport_adapter(source)
        if transport_adapter is not None:
            return transport_adapter
        # Relay ingress keeps the underlying platform on the source (session
        # keys / display policy), but delivery must use the one process-level
        # RelayAdapter that owns the connector socket -- secondary profiles do
        # not register their own, so a profile-aware lookup would silently
        # disable streaming/typing/tool progress.
        if getattr(source, "delivered_via_upstream_relay", False) is True:
            return (getattr(self, "adapters", None) or {}).get(Platform.RELAY)
        # ``getattr``: test fixtures build bare SimpleNamespace sources without ``profile``.
        return self._authorization_adapter(
            getattr(source, "platform", None),
            getattr(source, "profile", None),
        )

    def _owning_profile(self, adapter, platform):
        """Return (registered, profile) for a live adapter: profile is None for primary."""
        if adapter is (getattr(self, "adapters", None) or {}).get(platform):
            return True, None
        for profile, profile_adapters in (getattr(self, "_profile_adapters", None) or {}).items():
            if adapter is profile_adapters.get(platform):
                return True, profile
        return False, None

    def _registered_transport_adapter(self, source: SessionSource):
        """Return the registered adapter that created *source*, if retained.

        ``source.profile`` is the runtime/session namespace and may differ from
        the adapter profile when one shared credential serves several routed
        runtimes; ``build_source`` keeps the receiving adapter as in-process
        provenance so replies and intake-policy checks stay on that transport.
        Restored or hand-built sources fall back (fail-closed) to profile lookup.
        """
        adapter_ref = getattr(source, "_transport_adapter_ref", None)
        adapter = adapter_ref() if callable(adapter_ref) else None
        platform = getattr(source, "platform", None)
        if adapter is None or platform is None:
            return None
        registered, _ = self._owning_profile(adapter, platform)
        return adapter if registered else None

    def _adapter_profile_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the transport-owning profile for adapter policy lookups."""
        adapter = self._registered_transport_adapter(source)
        if adapter is not None:
            registered, profile = self._owning_profile(adapter, getattr(source, "platform", None))
            if registered:
                return profile
        return getattr(source, "profile", None)

    def _adapter_flag(self, platform, name: str, profile) -> bool:
        if not platform:
            return False
        adapter = self._authorization_adapter(platform, profile)
        return adapter is not None and bool(getattr(adapter, name, False))

    def _adapter_authorization_is_upstream(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter delegates authz to a trusted authenticated upstream.

        Mirrors ``BasePlatformAdapter.authorization_is_upstream`` (True for the
        relay adapter: the connector authenticates the WS and resolves owner
        bindings before delivering). Unlike ``_adapter_enforces_own_access_policy``
        (a LOCAL policy only trusted when it is an allowlist) this UPSTREAM
        decision is honored directly. False when the adapter is unknown.
        """
        return self._adapter_flag(platform, "authorization_is_upstream", profile)

    def _adapter_enforces_own_access_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter gates access at intake itself.

        Mirrors ``BasePlatformAdapter.enforces_own_access_policy`` (WeCom, Weixin,
        Yuanbao, QQBot, WhatsApp). The flag alone is NOT "already authorized":
        these adapters default to ``open``, so ``_is_user_authorized`` only trusts
        them when the effective policy is an actual ``allowlist``.
        """
        return self._adapter_flag(platform, "enforces_own_access_policy", profile)

    def _config_extra(self, platform) -> dict:
        """``config.platforms[platform].extra`` as a dict ({} when absent)."""
        config = getattr(self, "config", None)
        platform_cfg = (
            config.platforms.get(platform)
            if config is not None and hasattr(config, "platforms")
            else None
        )
        extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
        return extra if isinstance(extra, dict) else {}

    def _adapter_setting(self, platform, attr: str, extra_key: str, profile):
        """Live adapter's resolved ``attr``, else ``config.extra[extra_key]``.

        The adapter value is preferred because it already folds in the
        ``<PLATFORM>_*`` env override, which is not always bridged back into
        ``config.extra``; the extra fallback serves bare runners with no adapter.
        """
        adapter = self._authorization_adapter(platform, profile)
        value = getattr(adapter, attr, None) if adapter is not None else None
        if value is None:
            value = self._config_extra(platform).get(extra_key)
        return value

    def _adapter_dm_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Lowercased effective ``dm_policy`` (open/allowlist/disabled/pairing), ``""`` if unknown.

        "Reached the gateway" only carries an authorization signal in the
        ``allowlist`` case; ``open`` forwards everyone and ``pairing`` forwards
        unpaired DMs for the handshake.
        """
        if not platform:
            return ""
        policy = self._adapter_setting(platform, "_dm_policy", "dm_policy", profile)
        return str(policy or "").strip().lower()

    def _adapter_group_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Lowercased effective ``group_policy`` (open/allowlist/disabled), ``""`` if unknown."""
        if not platform:
            return ""
        policy = self._adapter_setting(platform, "_group_policy", "group_policy", profile)
        return str(policy or "").strip().lower()

    def _adapter_group_has_sender_allowlist(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether a per-group sender allowlist (WeCom ``groups.<id>.allow_from``) gated this message.

        A group may be open at the chat level while restricting senders; if such
        a message reached the gateway the adapter already checked that sender
        allowlist, so it is a trustworthy intake decision.
        """
        if not platform or not chat_id:
            return False
        groups = self._adapter_setting(platform, "_groups", "groups", profile)
        if not isinstance(groups, dict):
            return False

        chat_id_str = str(chat_id)
        group_cfg = groups.get(chat_id_str)
        if not isinstance(group_cfg, dict):
            lowered = chat_id_str.lower()
            for key, value in groups.items():
                if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
                    group_cfg = value
                    break
        if not isinstance(group_cfg, dict):
            group_cfg = groups.get("*")
        if not isinstance(group_cfg, dict):
            return False

        sender_allow = group_cfg.get("allow_from") or group_cfg.get("allowFrom")
        if isinstance(sender_allow, str):
            return bool(sender_allow.strip())
        if isinstance(sender_allow, (list, tuple, set)):
            return any(str(item).strip() for item in sender_allow)
        return False

    def _pairing_store_for(self, source: "SessionSource"):
        """Per-profile PairingStore for a source, else the global ``self.pairing_store``."""
        per_profile = getattr(self, "pairing_stores", None) or {}
        profile = getattr(source, "profile", None)
        if profile and profile in per_profile:
            return per_profile[profile]
        return getattr(self, "pairing_store", None)

    def _adapter_extra_for_source(self, source) -> dict:
        adapter = self._adapter_for_source(source)
        if adapter is None:
            return {}
        return getattr(getattr(adapter, "config", None), "extra", None) or {}

    def _is_user_authorized(
        self,
        source: SessionSource,
        *,
        allow_adapter_delegation: bool = True,
    ) -> bool:
        """Check if a user is authorized to use the bot.

        Order: trusted-upstream delegation, chat-scoped group allowlists,
        ``{PLATFORM}_ALLOW_BOTS``, per-platform allow-all, adapter role auth,
        pairing store, env/config allowlists, ``GATEWAY_ALLOW_ALL_USERS``,
        default deny.
        """
        from gateway.run import logger
        # HA events are system-generated (HASS_TOKEN authenticates the
        # connection); webhook events are HMAC-verified in the adapter.
        if source.platform in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
            return True

        adapter_profile = self._adapter_profile_for_source(source)
        is_group = source.chat_type in _GROUP_CHAT_TYPES

        # Trusted-upstream delegation (relay): the Team Gateway connector
        # authenticates this gateway's WS and resolves owner-only author
        # bindings BEFORE delivering, so the event is already authorized and
        # there is no local RELAY_ALLOWED_USERS to consult. Not a fail-open: it
        # fires only for events actually delivered over the authenticated relay
        # WS (transport stamps ``delivered_via_upstream_relay``) or whose
        # adapter declares ``authorization_is_upstream=True``. The delivery
        # marker is PRIMARY because a relayed message carries the UNDERLYING
        # platform (discord/...), not ``Platform.RELAY``; the adapter-flag check
        # covers events whose platform IS RELAY (interaction passthrough).
        # ``is True``: a MagicMock stand-in must not auto-truthy into authz.
        if allow_adapter_delegation and (
            source.delivered_via_upstream_relay is True
            or self._adapter_authorization_is_upstream(source.platform, profile=adapter_profile)
        ):
            return True

        user_id = source.user_id

        # Chat-scoped group allowlists (TELEGRAM_GROUP_ALLOWED_CHATS /
        # QQ_GROUP_ALLOWED_USERS) must work with ``user_id is None`` (anonymous
        # admins, sender_chat posts, channel broadcasts), so they run before
        # the no-user-id guard.
        if is_group and source.chat_id:
            chat_allowlist_env = _GROUP_CHAT_ENV.get(source.platform, "")
            if chat_allowlist_env and _allows(
                _coerce_allow_set(_platform_gate_env(chat_allowlist_env)), source.chat_id
            ):
                return True
            # config.yaml fallback (``extra.group_allowed_chats``): Telegram
            # observe-unmentioned mode strips user_id from triggered group
            # messages, so the env-only check above misses config allowlists.
            try:
                adapter_group_allowed = self._adapter_extra_for_source(source).get(
                    "group_allowed_chats"
                )
                if adapter_group_allowed and _allows(
                    _coerce_allow_set(adapter_group_allowed), source.chat_id
                ):
                    return True
            except Exception:
                pass

        # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist.
        # Also before the no-user-id guard: Slack Workflow Builder posts arrive
        # as bot_message with user=None.
        if getattr(source, "is_bot", False):
            allow_bots_var = _ALLOW_BOTS_ENV.get(source.platform)
            if allow_bots_var and _platform_gate_env(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
                return True

        if not user_id:
            return False

        platform_env_map = dict(_ALLOWED_USERS_ENV)
        platform_allow_all_map = dict(_ALLOW_ALL_ENV)
        if source.platform not in platform_env_map:
            try:
                entry = _registry_entry(source.platform)
                if entry:
                    if entry.allowed_users_env:
                        platform_env_map[source.platform] = entry.allowed_users_env
                    if entry.allow_all_env:
                        platform_allow_all_map[source.platform] = entry.allow_all_env
            except Exception:
                pass

        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        if platform_allow_all_var and _auth_env(platform_allow_all_var).lower() in _TRUTHY:
            return True

        # Adapter-verified role auth (Discord confirmed DISCORD_ALLOWED_ROLES
        # before dispatch). ``is True`` so a MagicMock source cannot pass.
        if allow_adapter_delegation and getattr(source, "role_authorized", False) is True:
            return True

        # Pairing store: a first-class grant created only by a trusted operator
        # approving a code (inbound senders can never reach approve_code).
        # Honored as a UNION with the allowlist; approval also mirrors the user
        # into a configured allowlist (PairingStore._approve_user).
        platform_name = source.platform.value if source.platform else ""
        pairing_store = self._pairing_store_for(source)
        if pairing_store is not None and pairing_store.is_approved(platform_name, user_id):
            return True

        platform_allowlist = _auth_env(platform_env_map.get(source.platform, ""))
        group_user_allowlist = ""
        group_chat_allowlist = ""
        if source.chat_type in {"group", "forum"}:
            group_user_allowlist = _auth_env(_GROUP_USER_ENV.get(source.platform, ""))
            group_chat_allowlist = _auth_env(_GROUP_CHAT_ENV.get(source.platform, ""))
        global_allowlist = _auth_env("GATEWAY_ALLOWED_USERS")

        if not platform_allowlist and not group_user_allowlist and not group_chat_allowlist and not global_allowlist:
            # No env allowlist. Own-policy adapters gate at intake, but their
            # decision is only trusted when the effective policy for THIS chat
            # type is ``allowlist``: ``open`` forwards EVERY sender (trusting it
            # is the fail-open SECURITY.md §2.6 forbids), ``disabled`` never
            # forwards, ``pairing`` forwards unpaired DMs for the handshake
            # (already denied by the pairing-store check above). Anything else
            # falls through to default-deny; GATEWAY_ALLOW_ALL_USERS, the
            # per-platform ALLOW_ALL flag and pairing stay the explicit opt-ins.
            if allow_adapter_delegation and self._adapter_enforces_own_access_policy(
                source.platform, profile=adapter_profile
            ):
                if is_group:
                    effective_policy = self._adapter_group_policy(source.platform, profile=adapter_profile)
                    if self._adapter_group_has_sender_allowlist(
                        source.platform, source.chat_id, profile=adapter_profile
                    ):
                        return True
                else:
                    effective_policy = self._adapter_dm_policy(source.platform, profile=adapter_profile)
                if effective_policy == "allowlist":
                    # Re-check DMs against the live adapter when it exposes
                    # ``_is_dm_allowed``: pairing revoke can clear
                    # WHATSAPP_ALLOWED_USERS while a construction-time
                    # ``_allow_from`` snapshot would keep authorizing until
                    # restart. Adapters without the helper keep the historical
                    # "reached the gateway under allowlist policy" rubber-stamp.
                    if not is_group:
                        adapter = self._authorization_adapter(source.platform, profile=adapter_profile)
                        dm_check = getattr(adapter, "_is_dm_allowed", None) if adapter is not None else None
                        if callable(dm_check):
                            return bool(dm_check(user_id))
                    return True
            # Adapters (e.g. Telegram) that gate via config.extra.allow_from /
            # group_allow_from without setting enforces_own_access_policy.
            adapter = self._adapter_for_source(source)
            if adapter is not None:
                extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
                adapter_allow = extra.get("group_allow_from" if is_group else "allow_from")
                if not adapter_allow and _platform_declares_allowed_users_env(source.platform):
                    # Plugin platforms (Buzz) carry the env allowlist as
                    # ``extra.allowed_users``. Under multiplex the YAML->env
                    # bridge is first-writer-wins, so only the default profile's
                    # list reaches the env read above; consult the live
                    # profile-routed adapter's config instead.
                    adapter_allow = extra.get("allowed_users")
                if adapter_allow:
                    allowed = _coerce_allow_set(adapter_allow)
                    normalize = getattr(adapter, "normalize_user_id", None)
                    if callable(normalize):
                        # Ids and entries may spell the same principal
                        # differently (Buzz hex vs npub).
                        allowed = {normalize(entry) or entry for entry in allowed}
                    if _allows(allowed, user_id):
                        return True
            return _auth_env("GATEWAY_ALLOW_ALL_USERS").lower() in _TRUTHY

        # Telegram group traffic authorized by chat ID (separate from
        # TELEGRAM_GROUP_ALLOWED_USERS, which gates the sender).
        if group_chat_allowlist and source.chat_type in {"group", "forum"} and source.chat_id:
            if _allows(_coerce_allow_set(group_chat_allowlist), source.chat_id):
                return True

        # Backward-compat: TELEGRAM_GROUP_ALLOWED_USERS was once (mis)used as a
        # chat-ID allowlist. "-"-prefixed values are chat IDs, so honor them as
        # such and warn once; the correct var is TELEGRAM_GROUP_ALLOWED_CHATS.
        if (
            source.platform == Platform.TELEGRAM
            and group_user_allowlist
            and source.chat_type in {"group", "forum"}
            and source.chat_id
        ):
            legacy_chat_ids = {
                v.strip() for v in group_user_allowlist.split(",") if v.strip().startswith("-")
            }
            if legacy_chat_ids:
                if not getattr(self, "_warned_telegram_group_users_legacy", False):
                    logger.warning(
                        "TELEGRAM_GROUP_ALLOWED_USERS contains chat-ID-shaped values "
                        "(%s). Treating them as chat IDs for backward compatibility. "
                        "Move chat IDs to TELEGRAM_GROUP_ALLOWED_CHATS — the _USERS var "
                        "is now for sender user IDs.",
                        ",".join(sorted(legacy_chat_ids)),
                    )
                    self._warned_telegram_group_users_legacy = True
                if source.chat_id in legacy_chat_ids:
                    return True

        # In group/forum chats TELEGRAM_GROUP_ALLOWED_USERS is the scoped
        # allowlist and does not imply DM access; TELEGRAM_ALLOWED_USERS is
        # platform-wide and works everywhere.
        allowed_ids = (
            _coerce_allow_set(platform_allowlist)
            | _coerce_allow_set(group_user_allowlist)
            | _coerce_allow_set(global_allowlist)
        )

        # Adapters that resolve username-shaped entries to numeric IDs at
        # connect time (Discord) keep the resolved set in memory; the per-turn
        # .env hot-reload restores the RAW usernames into the env, so from the
        # second turn on ``platform_allowlist`` holds usernames while user_id is
        # numeric. Union in the adapter's resolved IDs. Never a widening: the
        # empty-allowlist branch already returned and adapters only resolve
        # operator-written entries. Guarded on ``platform_allowlist`` so
        # group/global-only configs never consult adapter memory; duck-typed +
        # type-checked so mock adapters cannot auto-truthy in.
        if platform_allowlist:
            try:
                adapter = self._adapter_for_source(source)
            except Exception:
                adapter = None
            resolver = getattr(adapter, "resolved_allowlist_user_ids", None)
            if callable(resolver):
                try:
                    resolved_ids = resolver()
                except Exception:
                    resolved_ids = None
                if isinstance(resolved_ids, (set, frozenset, list, tuple)):
                    allowed_ids.update(
                        str(entry).strip()
                        for entry in resolved_ids
                        if isinstance(entry, (str, int)) and str(entry).strip()
                    )

        if "*" in allowed_ids:
            return True

        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])

        # WhatsApp (Baileys + Cloud): resolve phone<->LID / JID aliases so
        # device-suffix and bare-phone entries match the same principal.
        if source.platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
            normalized_allowed_ids = set()
            for allowed_id in allowed_ids:
                normalized_allowed_ids.update(_expand_whatsapp_auth_aliases(allowed_id))
            if normalized_allowed_ids:
                allowed_ids = normalized_allowed_ids

            check_ids.update(_expand_whatsapp_auth_aliases(user_id))
            normalized_user_id = _normalize_whatsapp_identifier(user_id)
            if normalized_user_id:
                check_ids.add(normalized_user_id)

        platform_value = source.platform.value if source.platform is not None else None

        # SimpleX: user_id is the numeric contactId (stable across renames) but
        # the UI only surfaces display names, so match both. Plugin platform:
        # compare by value.
        if platform_value == "simplex" and source.user_name:
            check_ids.add(source.user_name)

        # Buzz: allowlist may hold npub or hex; inbound pubkeys are hex.
        if platform_value == "buzz":
            allowed_ids = _normalize_nostr_allow_entries(allowed_ids)
            if user_id.startswith("npub"):
                hex_user = _npub_to_hex(user_id)
                if hex_user:
                    check_ids.add(hex_user)

        return bool(check_ids & allowed_ids)

    def _get_unauthorized_dm_behavior(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Return how unauthorized DMs should be handled for a platform.

        Resolution order:
        1. Explicit per-platform ``unauthorized_dm_behavior`` in config.
        2. Email -> ``"ignore"`` (inboxes hold arbitrary human mail; pairing
           codes are not a safe default; matches GatewayConfig).
        3. Explicit (non-default) global ``unauthorized_dm_behavior``.
        4. Adapter dm_policy: ``pairing`` -> ``"pair"``; ``allowlist`` /
           ``disabled`` -> ``"ignore"``.
        5. Any configured allowlist -> ``"ignore"`` (the operator restricted
           access; spamming unknown contacts with codes is noisy and leaks).
        6. Otherwise ``"pair"`` (open-gateway default).
        """
        config = getattr(self, "config", None)

        if config and hasattr(config, "get_unauthorized_dm_behavior") and platform:
            if "unauthorized_dm_behavior" in self._config_extra(platform):
                return config.get_unauthorized_dm_behavior(platform)

        if platform == Platform.EMAIL:
            return "ignore"

        if config and hasattr(config, "unauthorized_dm_behavior"):
            if config.unauthorized_dm_behavior != "pair":
                return config.unauthorized_dm_behavior

        if platform:
            dm_policy = self._adapter_dm_policy(platform, profile=profile)
            if not dm_policy:
                dm_policy = str(self._config_extra(platform).get("dm_policy") or "").strip().lower()
            if dm_policy == "pairing":
                return "pair"
            if dm_policy in {"allowlist", "disabled"}:
                return "ignore"

            # Historical: Yuanbao is absent from this allowlist-aware default.
            env_key = "" if platform == Platform.YUANBAO else _ALLOWED_USERS_ENV.get(platform, "")
            group_keys = (_GROUP_USER_ENV.get(platform), _GROUP_CHAT_ENV.get(platform))
            for key in (env_key, *group_keys):
                if key and _platform_gate_env(key).strip():
                    return "ignore"

        if _platform_gate_env("GATEWAY_ALLOWED_USERS").strip():
            return "ignore"

        return "pair"
