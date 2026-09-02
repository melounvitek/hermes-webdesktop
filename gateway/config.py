"""
Gateway configuration management.

Handles loading and validating configuration for:
- Connected platforms (Telegram, Discord, WhatsApp, Weixin, and more)
- Home channels for each platform
- Session reset policies
- Delivery preferences
"""

import logging
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from hermes_cli.config import get_hermes_home
from agent.secret_scope import current_secret_scope, get_secret as _get_secret
from gateway.shutdown_watchdog import (
    DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
)
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return is_truthy_value(value, default=default)


def _normalize_multiplex_profile_allowlist(value: Any) -> Optional[List[str]]:
    """Normalize the optional named-profile allowlist.

    ``None`` preserves the historical serve-all behavior. A malformed outer
    value fails safe to an empty list (default profile only); malformed list
    entries are skipped with a warning.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        logger.warning(
            "Invalid gateway.multiplex_profile_allowlist (expected a list, got %s); "
            "serving only the default profile",
            type(value).__name__,
        )
        return []

    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    normalized: List[str] = []
    for entry in value:
        if not isinstance(entry, str):
            logger.warning(
                "Skipping invalid gateway.multiplex_profile_allowlist entry %r (expected a profile name)",
                entry,
            )
            continue
        try:
            name = normalize_profile_name(entry)
            validate_profile_name(name)
        except ValueError:
            logger.warning("Skipping invalid gateway.multiplex_profile_allowlist entry %r", entry)
            continue
        if name != "default" and name not in normalized:
            normalized.append(name)
    return normalized


# Recognized truthy / falsy tokens for the GATEWAY_MULTIPLEX_PROFILES operator
# override. Anything not in either set — and a blank/whitespace value — is
# treated as "unset" so it falls through to config.yaml rather than silently
# forcing the flag off.
_MULTIPLEX_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})
_MULTIPLEX_FALSY_STRINGS = frozenset({"0", "false", "no", "off"})


def _env_multiplex_profiles_override() -> "bool | None":
    """Resolve the GATEWAY_MULTIPLEX_PROFILES operator override.

    Returns ``True``/``False`` when the env var is set to a recognized truthy/
    falsy token, or ``None`` when it is unset, blank, or unrecognized — in which
    case the caller keeps the config.yaml value (env > config > default). Blank
    is deliberately ``None``, not ``False``: a provisioned-but-unpopulated Fly
    secret arrives as ``""`` and must NOT shadow a config.yaml opt-in.
    """
    raw = os.getenv("GATEWAY_MULTIPLEX_PROFILES")
    token = (raw or "").strip().lower()
    if not token:
        return None
    if token in _MULTIPLEX_TRUTHY_STRINGS:
        return True
    if token in _MULTIPLEX_FALSY_STRINGS:
        return False
    logger.warning(
        "Ignoring unrecognized GATEWAY_MULTIPLEX_PROFILES=%r "
        "(expected one of %s or %s); falling back to config.yaml.",
        raw,
        sorted(_MULTIPLEX_TRUTHY_STRINGS),
        sorted(_MULTIPLEX_FALSY_STRINGS),
    )
    return None


def _normalize_transport_token(value: Any) -> str:
    """Normalize a streaming transport/mode value to a canonical token.

    Handles the YAML 1.1 boolean quirk where bare ``on`` / ``off`` parse to
    Python ``True`` / ``False`` (see ``gateway/display_config.py`` ``_normalise``).
    Without this, ``mode: off`` arrives as boolean ``False`` and stringifying it
    yields ``"false"`` instead of the advertised ``"off"``, so streaming would be
    enabled instead of disabled. Booleans map to ``"auto"`` (True) / ``"off"``
    (False); anything else is lower-cased, defaulting to ``"auto"``.
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower() or "auto"


def _coerce_float(value: Any, default: float) -> float:
    """Coerce numeric config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce integer config values, falling back on malformed input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float("inf")) — a non-finite YAML value must
        # degrade to the default, not abort gateway config loading.
        return default


def _coerce_optional_positive_int(value: Any, key: str) -> Optional[int]:
    """Coerce an optional positive integer config value.

    ``None``/0/negative disable the setting. Malformed values are ignored with
    a warning so a typo never prevents the gateway from starting.
    """
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise ValueError(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(value)
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip(), 10)
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)", key, value
        )
        return None
    return parsed if parsed > 0 else None


_SYSTEMD_WATCHDOG_MAX_SECONDS = 2_147_483_647


def coerce_systemd_watchdog_seconds(
    value: Any, key: str = "gateway.systemd_watchdog_seconds"
) -> int:
    """Return a bounded positive watchdog interval or zero when disabled.

    Runtime and service generation share this normalization so a value can
    never enable ``Type=notify`` while disabling application heartbeats.
    """
    if value is None:
        return 0
    parsed: Optional[int] = None
    if isinstance(value, int) and not isinstance(value, bool):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw and raw.isascii() and raw.isdecimal():
            try:
                parsed = int(raw, 10)
            except (TypeError, ValueError, OverflowError):
                parsed = None
    if parsed is None:
        logger.warning("Ignoring invalid %s (expected a positive integer)", key)
        return 0
    if parsed == 0:
        return 0
    if not 0 < parsed <= _SYSTEMD_WATCHDOG_MAX_SECONDS:
        logger.warning(
            "Ignoring invalid %s (expected an integer from 1 to %d)",
            key,
            _SYSTEMD_WATCHDOG_MAX_SECONDS,
        )
        return 0
    return parsed


def _coerce_dict(value: Any) -> Dict[str, Any]:
    """Return *value* when it is a mapping, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _normalize_choice(value: Any, choices: set, default: str) -> str:
    """Lower-cased *value* when it is one of *choices*, else *default*."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in choices:
            return normalized
    return default


def _dict_slot(container: dict, key: str) -> dict:
    """Get-or-create ``container[key]`` as a dict, replacing a non-dict value with ``{}``."""
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        value = {}
        container[key] = value
    return value


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read env vars through the active profile secret scope when present.

    ``load_gateway_config()`` runs in many contexts, including multiplexed
    profile startup where ``_profile_runtime_scope`` installs per-profile
    secrets. In that scope we must prefer the scoped value; outside it we keep
    legacy ``os.getenv`` behavior for single-profile callers and unscoped
    gateway reads.
    """
    if current_secret_scope() is not None:
        scope_val = _get_secret(name, None)
        return scope_val if scope_val is not None else default
    return os.environ.get(name, default)


def _getenv_str(name: str, default: str = "") -> str:
    val = _getenv(name, default)
    return val if val is not None else default


# Module-level cache for bundled platform plugin names (lives outside the
# enum so it doesn't become an accidental enum member).
_Platform__bundled_plugin_names: Optional[set] = None


class Platform(Enum):
    """Supported messaging platforms.

    Built-in platforms have explicit members.  Plugin platforms use dynamic
    members created on-demand by ``_missing_()`` so that
    ``Platform("irc")`` works without modifying this enum.  Dynamic members
    are cached in ``_value2member_map_`` for identity-stable comparisons.
    """
    LOCAL = "local"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    WHATSAPP_CLOUD = "whatsapp_cloud"
    SLACK = "slack"
    SIGNAL = "signal"
    MATTERMOST = "mattermost"
    MATRIX = "matrix"
    HOMEASSISTANT = "homeassistant"
    EMAIL = "email"
    SMS = "sms"
    DINGTALK = "dingtalk"
    API_SERVER = "api_server"
    WEBHOOK = "webhook"
    MSGRAPH_WEBHOOK = "msgraph_webhook"
    FEISHU = "feishu"
    WECOM = "wecom"
    WECOM_CALLBACK = "wecom_callback"
    WEIXIN = "weixin"
    BLUEBUBBLES = "bluebubbles"
    QQBOT = "qqbot"
    YUANBAO = "yuanbao"
    RELAY = "relay"  # generic relay adapter fronted by the connector (EXPERIMENTAL)
    @classmethod
    def _missing_(cls, value):
        """Accept unknown platform names only for known plugin adapters.

        Creates a pseudo-member cached in ``_value2member_map_`` so that
        ``Platform("irc") is Platform("irc")`` holds True (identity-stable).
        Arbitrary strings are rejected to prevent enum pollution.
        """
        if not isinstance(value, str) or not value.strip():
            return None
        # Normalise to lowercase to avoid case mismatches in config
        value = value.strip().lower()
        # Check cache first (another call may have created it already)
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]

        # Only create pseudo-members for bundled plugin platforms (discovered
        # via filesystem scan) or runtime-registered plugin platforms.
        global _Platform__bundled_plugin_names
        if _Platform__bundled_plugin_names is None:
            _Platform__bundled_plugin_names = cls._scan_bundled_plugin_platforms()
        if value in _Platform__bundled_plugin_names:
            return cls._add_pseudo_member(value)

        # Runtime-registered plugins (e.g. user-installed, discovered after
        # the enum was defined).
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(value):
                return cls._add_pseudo_member(value)
        except Exception:
            pass

        return None

    @classmethod
    def _add_pseudo_member(cls, value: str) -> "Platform":
        pseudo = object.__new__(cls)
        pseudo._value_ = value
        pseudo._name_ = value.upper().replace("-", "_").replace(" ", "_")
        cls._value2member_map_[value] = pseudo
        cls._member_map_[pseudo._name_] = pseudo
        return pseudo

    @classmethod
    def _scan_bundled_plugin_platforms(cls) -> set:
        """Return names of bundled platform plugins under ``plugins/platforms/``."""
        names: set = set()
        try:
            platforms_dir = Path(__file__).parent.parent / "plugins" / "platforms"
            if platforms_dir.is_dir():
                for child in platforms_dir.iterdir():
                    if child.is_dir() and (child / "__init__.py").exists() and (
                        (child / "plugin.yaml").exists() or (child / "plugin.yml").exists()
                    ):
                        names.add(child.name.lower())
        except Exception:
            pass
        return names


# Snapshot of built-in platform values before any dynamic _missing_ lookups.
# Used to distinguish real platforms from arbitrary strings.
_BUILTIN_PLATFORM_VALUES = frozenset(m.value for m in Platform.__members__.values())


# Platforms that bind a host TCP port (HTTP/webhook listeners). In a profile
# multiplexer the default profile owns the single shared listener and serves
# every profile through the /p/<profile>/ URL prefix, so a SECONDARY profile
# enabling one of these is always a misconfiguration: it would try to bind a
# port already held by the default's listener. Single source of truth for
# both the gateway's fail-fast startup validation (gateway/run.py) and the
# dashboard's pre-write mutation validation (hermes_cli/web_server.py) so
# the two policies cannot drift. Stored as platform .value strings.
PORT_BINDING_PLATFORM_VALUES = frozenset({
    "webhook",
    "api_server",
    "msgraph_webhook",
    "feishu",
    "wecom_callback",
    "bluebubbles",
    "sms",
    "whatsapp_cloud",
    "line",
    "teams",
})

# Platforms whose port-binding status depends on connection mode. Feishu in
# websocket mode (its default) uses an outbound long connection — no listener.
# Only webhook/callback mode binds a port. Maps platform value → the mode
# value that actually binds (#52563).
PORT_BINDING_CONDITIONAL_MODES: dict[str, str] = {
    "feishu": "webhook",
}


def platform_binds_port(platform_value: str, extra: Optional[dict] = None) -> bool:
    """Return True when *platform_value* actually binds a port for *extra* config.

    Mode-conditional platforms (Feishu) only bind in their listener mode;
    everything else in ``PORT_BINDING_PLATFORM_VALUES`` always binds.
    """
    if platform_value not in PORT_BINDING_PLATFORM_VALUES:
        return False
    expected_mode = PORT_BINDING_CONDITIONAL_MODES.get(platform_value)
    if expected_mode is not None:
        actual = str((extra or {}).get("connection_mode", "websocket")).strip().lower()
        return actual == expected_mode
    return True


@dataclass
class HomeChannel:
    """
    Default destination for a platform.
    
    When a cron job specifies deliver="telegram" without a specific chat ID,
    messages are sent to this home channel. Thread-aware platforms may also
    store a thread/topic ID so the bare platform target routes to the exact
    conversation where /sethome was run.
    """
    platform: Platform
    chat_id: str
    name: str  # Human-readable name for display
    thread_id: Optional[str] = None
    # Authenticated logical-target provenance observed by a platform adapter.
    # Relay egress re-attaches these values, but the connector remains the
    # authorization boundary and resolves them against its authoritative stores.
    user_id: Optional[str] = None
    scope_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "name": self.name,
        }
        for key in ("thread_id", "user_id", "scope_id"):
            if getattr(self, key):
                result[key] = getattr(self, key)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HomeChannel":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            name=data.get("name", "Home"),
            thread_id=str(data["thread_id"]) if data.get("thread_id") else None,
            user_id=str(data["user_id"]) if data.get("user_id") else None,
            scope_id=str(data["scope_id"]) if data.get("scope_id") else None,
        )


def persist_home_channel(home: HomeChannel, *, enabled_if_new: bool = False) -> None:
    """Persist a logical home without falsely enabling a Relay-fronted adapter."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    platform_config = _dict_slot(_dict_slot(config, "platforms"), home.platform.value)
    if enabled_if_new:
        platform_config.setdefault("enabled", True)
    platform_config["home_channel"] = home.to_dict()
    save_config(config)


@dataclass
class SessionResetPolicy:
    """
    Controls when sessions reset (lose context).
    
    Modes:
    - "daily": Reset at a specific hour each day
    - "idle": Reset after N minutes of inactivity
    - "both": Whichever triggers first (daily boundary OR idle timeout)
    - "none": Never auto-reset (context managed only by compression)

    Default is "none" — sessions never auto-reset unless the user opts in
    via the `session_reset` section in config.yaml (or gateway.json
    overrides). Changed July 2026 from "both" (24h idle + daily 4am), which
    surprised users who expected their conversations to persist.
    """
    mode: str = "none"  # "daily", "idle", "both", or "none"
    at_hour: int = 4  # Hour for daily reset (0-23, local time)
    idle_minutes: int = 1440  # Minutes of inactivity before reset (24 hours)
    notify: bool = True  # Send a notification to the user when auto-reset occurs
    notify_exclude_platforms: tuple = ("api_server", "webhook")  # Platforms that don't get reset notifications
    # A background process this many hours old (or older) no longer blocks
    # session idle/daily reset. A forgotten preview server should not keep a
    # session alive forever (#29177). The process is NOT killed — only ignored
    # by the reset guard. Raise this if you run legitimate multi-day jobs whose
    # liveness should pin the conversation open.
    bg_process_max_age_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "at_hour": self.at_hour,
            "idle_minutes": self.idle_minutes,
            "notify": self.notify,
            "notify_exclude_platforms": list(self.notify_exclude_platforms),
            "bg_process_max_age_hours": self.bg_process_max_age_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        data = _coerce_dict(data)

        def val(key: str, default: Any) -> Any:
            # Handle both missing keys and explicit null values (YAML null → None)
            value = data.get(key)
            return default if value is None else value

        exclude = data.get("notify_exclude_platforms")
        return cls(
            mode=val("mode", "none"),
            at_hour=val("at_hour", 4),
            idle_minutes=val("idle_minutes", 1440),
            notify=_coerce_bool(data.get("notify"), True),
            notify_exclude_platforms=tuple(exclude) if exclude is not None else ("api_server", "webhook"),
            bg_process_max_age_hours=val("bg_process_max_age_hours", 24),
        )


@dataclass
class ChannelOverride:
    """
    Per-channel override for model, provider, and system prompt.

    Used in config under platforms.<name>.channel_overrides[channel_id].
    Enables different channels (e.g. Discord #daily vs #dev) to use different
    models and personas without running separate gateway instances.
    """
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelOverride":
        if not data:
            return cls()
        return cls(
            model=data.get("model"),
            provider=data.get("provider"),
            system_prompt=data.get("system_prompt"),
        )


# Canonical map of platforms whose primary credential is ``PlatformConfig.token``
# and the env var it loads from. Used for empty-token warnings at config
# validation and by the multiplex primary-startup credential gate in
# ``gateway.run`` (#64674). Platforms absent from this map authenticate some
# other way (session files, port-bound webhooks, api_key-only) and must never
# be skipped for a missing token.
PLATFORM_TOKEN_ENV_NAMES: dict["Platform", str] = {
    Platform.TELEGRAM: "TELEGRAM_BOT_TOKEN",
    Platform.DISCORD: "DISCORD_BOT_TOKEN",
    Platform.SLACK: "SLACK_BOT_TOKEN",
    Platform.MATTERMOST: "MATTERMOST_TOKEN",
    Platform.MATRIX: "MATRIX_ACCESS_TOKEN",
    Platform.WEIXIN: "WEIXIN_TOKEN",
}


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None  # Bot token (Telegram, Discord)
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None

    # Reply threading mode (Telegram/Slack)
    # - "off": Never thread replies to original message
    # - "first": Only first chunk threads to user's message (default)
    # - "all": All chunks in multi-part replies thread to user's message
    reply_to_mode: str = "first"

    # Whether the gateway is allowed to send "♻️ Gateway online" /
    # "♻ Gateway restarted" lifecycle notifications on this platform.
    # Default True preserves prior behavior. Set False on platforms used
    # by end users (e.g. Slack) where operator-flavored restart pings are
    # noise; keep True for back-channels where the operator wants them.
    gateway_restart_notification: bool = True

    # Whether the gateway shows a "typing…" / "is thinking…" status indicator
    # while the agent processes a message on this platform. Default True
    # preserves prior behavior. Set False on platforms where the indicator is
    # unwanted (e.g. Slack's assistant.threads.setStatus "is thinking…", which
    # disables the compose box, or any platform where users find the bubble
    # noisy). Drives the per-message _keep_typing refresh loop in
    # gateway/platforms/base.py.
    typing_indicator: bool = True

    # Custom text for the working-state line on platforms whose typing
    # indicator renders text rather than a native bubble: Slack's
    # assistant.threads.setStatus line (shown next to the bot name; needs the
    # assistant:write scope to render) and Google Chat's visible marker
    # message. None keeps each platform's built-in default ("is thinking..." /
    # "Hermes is thinking…"). Platforms with textless indicators (Discord,
    # Telegram, Matrix, …) ignore it.
    typing_status_text: Optional[str] = None

    # Per-channel model/provider/system_prompt overrides (channel_id -> ChannelOverride)
    channel_overrides: Dict[str, ChannelOverride] = field(default_factory=dict)

    # Platform-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "extra": self.extra,
            "reply_to_mode": self.reply_to_mode,
            "gateway_restart_notification": self.gateway_restart_notification,
            "typing_indicator": self.typing_indicator,
        }
        if self.typing_status_text is not None:
            result["typing_status_text"] = self.typing_status_text
        if self.token:
            result["token"] = self.token
        if self.api_key:
            result["api_key"] = self.api_key
        if self.home_channel:
            result["home_channel"] = self.home_channel.to_dict()
        if self.channel_overrides:
            result["channel_overrides"] = {
                cid: ov.to_dict() for cid, ov in self.channel_overrides.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        data = _coerce_dict(data)
        home_channel = None
        if isinstance(data.get("home_channel"), dict):
            home_channel = HomeChannel.from_dict(data["home_channel"])

        # gateway_restart_notification / typing_indicator / typing_status_text may
        # arrive top-level or bridged into ``extra`` by the shared-key loop in
        # load_gateway_config(), so YAML ``discord: gateway_restart_notification: false``
        # works without a separate platforms: block. Check both (top-level wins).
        extra = _coerce_dict(data.get("extra", {}))

        def toplevel_or_extra(key: str) -> Any:
            value = data.get(key)
            return extra.get(key) if value is None else value

        channel_overrides: Dict[str, ChannelOverride] = {}
        raw_overrides = data.get("channel_overrides") or {}
        if isinstance(raw_overrides, dict):
            for cid, ov_data in raw_overrides.items():
                if isinstance(ov_data, dict):
                    channel_overrides[str(cid)] = ChannelOverride.from_dict(ov_data)

        return cls(
            enabled=_coerce_bool(data.get("enabled"), False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_channel=home_channel,
            reply_to_mode=data.get("reply_to_mode", "first"),
            gateway_restart_notification=_coerce_bool(toplevel_or_extra("gateway_restart_notification"), True),
            typing_indicator=_coerce_bool(toplevel_or_extra("typing_indicator"), True),
            typing_status_text=toplevel_or_extra("typing_status_text"),  # string passthrough, no coercion
            channel_overrides=channel_overrides,
            extra=extra,
        )


# Streaming defaults — single source of truth so both StreamingConfig and
# StreamConsumerConfig agree on the out-of-the-box edit rhythm.  Tuned for
# Telegram's ~1 edit/s flood envelope: a touch under 1s lets the cadence
# breathe without bumping into rate limits, and a smaller buffer threshold
# makes short replies feel near-instant in DMs.
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24
DEFAULT_STREAMING_CURSOR: str = " ▉"


@dataclass
class StreamingConfig:
    """Configuration for real-time token streaming to messaging platforms."""
    enabled: bool = False
    # Transport selection:
    #   "auto"  — prefer native streaming-draft updates when the platform
    #             supports them (Telegram sendMessageDraft, Bot API 9.5+);
    #             fall back to edit-based when not.
    #   "draft" — explicitly request native drafts; falls back to edit when
    #             the platform/chat doesn't support them.
    #   "edit"  — progressive editMessageText only (legacy behaviour).
    #   "off"   — disable streaming entirely.
    #
    # Default is "auto": prefer native draft streaming on platforms that
    # support it (Telegram DMs via sendMessageDraft, Bot API 9.5+) and fall
    # back to edit-based streaming everywhere else.  This is safe as a global
    # default because adapters without draft support (Discord, Slack, Matrix,
    # …) report supports_draft_streaming() == False and transparently use the
    # edit path — so "auto" never regresses non-Telegram platforms, it only
    # upgrades the chats that can render the smoother native preview.
    transport: str = "auto"
    edit_interval: float = DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = DEFAULT_STREAMING_CURSOR
    # Ported from openclaw/openclaw#72038.  When >0, the final edit for
    # a long-running streamed response is delivered as a fresh message
    # if the original preview has been visible for at least this many
    # seconds, so the platform's visible timestamp reflects completion
    # time instead of the preview creation time.  Currently applied to
    # Telegram only (other platforms ignore the setting).  Default 0 disables
    # the fresh-message replacement path; set >0 to opt in.
    fresh_final_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "transport": self.transport,
            "edit_interval": self.edit_interval,
            "buffer_threshold": self.buffer_threshold,
            "cursor": self.cursor,
            "fresh_final_after_seconds": self.fresh_final_after_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        if not isinstance(data, dict) or not data:
            return cls()

        # ``mode`` is an ergonomic alias for the transport that ALSO implies
        # ``enabled``.  A config like ``streaming: {mode: auto}`` reads as
        # "turn streaming on, transport=auto" — matching the natural intent
        # of someone enabling streaming without also spelling out
        # ``enabled: true``.  Without this, ``mode`` was silently ignored and
        # streaming stayed disabled (``enabled`` defaults to False), which is
        # a surprising footgun: the whole reply buffers and sends at once.
        # ``mode: off`` disables streaming; an explicit ``enabled`` key always
        # wins so callers can force either state.
        #
        # ``transport`` alone does NOT imply ``enabled``: ``streaming.enabled``
        # is the documented master switch (see website/docs/user-guide/
        # configuration.md), so a bare ``transport`` only selects HOW to stream
        # once streaming is on. Only the ``mode`` alias flips ``enabled``.
        raw_transport = data.get("transport")
        raw_mode = data.get("mode")
        # Normalize both through the same helper so YAML's bare ``off``/``on``
        # (parsed as bool False/True) become canonical tokens rather than
        # ``"false"``/``"true"``.
        picked = raw_transport if raw_transport is not None else raw_mode
        transport = _normalize_transport_token(picked)

        if "enabled" in data:
            enabled = _coerce_bool(data.get("enabled"), False)
        elif raw_mode is not None:
            # The ``mode`` alias (and only ``mode``) infers enabled:
            # ``off`` disables, anything else enables.
            enabled = _normalize_transport_token(raw_mode) != "off"
        else:
            enabled = False

        return cls(
            enabled=enabled,
            transport=transport,
            edit_interval=_coerce_float(
                data.get("edit_interval"), DEFAULT_STREAMING_EDIT_INTERVAL,
            ),
            buffer_threshold=_coerce_int(
                data.get("buffer_threshold"), DEFAULT_STREAMING_BUFFER_THRESHOLD,
            ),
            cursor=data.get("cursor", DEFAULT_STREAMING_CURSOR),
            fresh_final_after_seconds=_coerce_float(
                data.get("fresh_final_after_seconds"), 0.0
            ),
        )


# -----------------------------------------------------------------------------
# Built-in platform connection checkers
# -----------------------------------------------------------------------------
# Each callable receives a ``PlatformConfig`` and returns ``True`` when the
# platform is sufficiently configured to be considered "connected".  Platforms
# that rely on the generic ``token or api_key`` check (Telegram, Discord,
# Slack, Matrix, Mattermost, HomeAssistant) do not need an entry here.
def _has_usable_api_server_key(key: object) -> bool:
    """True when API_SERVER_KEY is present and strong enough to be usable.

    Mirrors the startup guard in ``gateway/platforms/api_server.py``
    (``has_usable_secret`` with ``min_length=16``) so the platform is only
    enrolled at load time when the adapter would actually agree to start.
    """
    if not key:
        return False
    try:
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        return len(str(key).strip()) >= 16
    return has_usable_secret(key, min_length=16)


_PLATFORM_CONNECTED_CHECKERS: dict[Platform, Callable[[PlatformConfig], bool]] = {
    Platform.WEIXIN: lambda cfg: bool(
        cfg.extra.get("account_id") and (cfg.token or cfg.extra.get("token"))
    ),
    Platform.WHATSAPP_CLOUD: lambda cfg: bool(
        cfg.extra.get("phone_number_id") and cfg.extra.get("access_token")
    ),
    Platform.SIGNAL: lambda cfg: bool(cfg.extra.get("http_url")),
    Platform.API_SERVER: lambda cfg: _has_usable_api_server_key(
        cfg.extra.get("key") if cfg else None
    ),
    Platform.WEBHOOK: lambda cfg: True,
    Platform.MSGRAPH_WEBHOOK: lambda cfg: bool(
        str(cfg.extra.get("client_state") or "").strip()
    ),
    Platform.BLUEBUBBLES: lambda cfg: bool(
        cfg.extra.get("server_url") and cfg.extra.get("password")
    ),
    Platform.QQBOT: lambda cfg: bool(
        cfg.extra.get("app_id") and cfg.extra.get("client_secret")
    ),
    Platform.YUANBAO: lambda cfg: bool(
        cfg.extra.get("app_id") and cfg.extra.get("app_secret")
    ),
    # Relay dials OUT to a connector; it is "connected" once an endpoint URL is
    # configured (extra["relay_url"] or extra["url"]). The capability descriptor
    # is negotiated at handshake time, so the URL is the only config-level
    # signal in the experimental phase. EXPERIMENTAL — may change.
    Platform.RELAY: lambda cfg: bool(
        cfg.extra.get("relay_url") or cfg.extra.get("url")
    ),
}


@dataclass
class GatewayConfig:
    """
    Main gateway configuration.
    
    Manages all platform connections, session policies, and delivery settings.
    """
    # Platform configurations
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    
    # Session reset policies by type
    default_reset_policy: SessionResetPolicy = field(default_factory=SessionResetPolicy)
    reset_by_type: Dict[str, SessionResetPolicy] = field(default_factory=dict)
    reset_by_platform: Dict[Platform, SessionResetPolicy] = field(default_factory=dict)
    
    # Reset trigger commands
    reset_triggers: List[str] = field(default_factory=lambda: ["/new", "/reset"])

    # User-defined quick commands (slash commands that bypass the agent loop)
    quick_commands: Dict[str, Any] = field(default_factory=dict)
    
    # Storage paths
    sessions_dir: Path = field(default_factory=lambda: get_hermes_home() / "sessions")

    # Whether to keep writing the legacy sessions.json mirror of the gateway
    # routing index. The primary copy lives in state.db (gateway_routing
    # table, #9006). Default True for backward compatibility with external
    # tooling and downgrade safety; set gateway.write_sessions_json: false in
    # config.yaml to stop producing the file.
    write_sessions_json: bool = True
    
    # Delivery settings
    always_log_local: bool = True  # Always save cron outputs to local files
    # Drop outbound "silence narration" messages (e.g. *(silent)*, 🔇, a bare
    # ".") pre-send. These are model hallucinations emitted when a persona has
    # nothing actionable to say; in bot-to-bot channels they mirror back and
    # forth, burning tokens and crashing models. Substrate-level guard that
    # survives SOUL.md/prompt drift across providers. Opt out with False for
    # raw passthrough.
    filter_silence_narration: bool = True

    # STT settings
    stt_enabled: bool = True  # Whether to auto-transcribe inbound voice messages
    stt_echo_transcripts: bool = True  # Whether to echo raw STT transcripts back to the user

    # Session isolation in shared chats
    group_sessions_per_user: bool = True  # Isolate group/channel sessions per participant when user IDs are available
    thread_sessions_per_user: bool = False  # When False (default), threads are shared across all participants
    max_concurrent_sessions: Optional[int] = None  # Positive int caps simultaneous active chat sessions

    # Multi-profile multiplexing (opt-in; default off preserves one-gateway-per-profile).
    # When True, the default profile's gateway serves inbound messages for every
    # profile on the host: profiles are stamped into session keys and (in later
    # phases) per-profile adapters/credentials are resolved. When False, the
    # gateway behaves exactly as before — single HERMES_HOME, no profile stamping.
    multiplex_profiles: bool = False
    # Optional named-profile allowlist for multiplex mode. None preserves the
    # historical serve-all behavior; [] serves only the default profile.
    multiplex_profile_allowlist: Optional[List[str]] = None

    # Public HTTPS endpoint another gateway may use for scoped RoomLink calls.
    # Disabled by default: setting an API key alone must never expose or
    # advertise a route. HERMES_ROOM_LINK_URL remains the operator override.
    room_link_url: Optional[str] = None

    # Opt-in systemd event-loop watchdog. Zero preserves Type=simple and
    # disables sd_notify at runtime.
    systemd_watchdog_seconds: int = 0

    # In-process event-loop liveness watchdog (#69089). A daemon OS thread
    # probes the gateway loop with call_soon_threadsafe; after consecutive
    # missed probes it dumps all-thread stacks and hard-exits with the
    # service-restart code so the supervisor can revive the process. On by
    # default; set gateway.loop_watchdog: false in config.yaml to disable.
    #
    # Tuning knobs (all seconds unless noted) make the watchdog tolerate
    # *transient, self-recovering* event-loop stalls — e.g. Telegram/Discord
    # reconnect doing synchronous socket I/O during a network blip — so a
    # short block does not force exit code 75 and trigger a restart churn
    # that stalls cron dispatch (recurring fleet incidents on 2026-08-17,
    # kanban t_0f76430f/t_70483f23). A genuine wedge (event loop frozen for
    # the full tolerance window) still escalates to a supervised restart.
    loop_watchdog: bool = True
    # Seconds the watchdog waits between liveness probes.
    loop_watchdog_probe_interval_s: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S
    # Seconds a single probe may go unprocessed before it counts as a miss.
    loop_watchdog_probe_timeout_s: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
    # Consecutive missed probes allowed before the watchdog hard-exits.
    # Default stays at 3 (~90-120s of sustained loop block): the transient
    # false-positive class (the watchdog's own on-loop heartbeat fsync)
    # is fixed at the root by the off-loop write + two-witness probe, so
    # raising this fleet-wide would only delay genuine-wedge recovery.
    loop_watchdog_max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES

    # Unauthorized DM policy
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"

    # Streaming configuration
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # Session store pruning: drop SessionEntry records older than this many
    # days from the in-memory dict and sessions.json.  Keeps the store from
    # growing unbounded in gateways serving many chats/threads/users over
    # months.  Pruning is invisible to users — if they resume, they get a
    # fresh session exactly as if the reset policy had fired.  0 = disabled.
    session_store_max_age_days: int = 90

    # Profile-based routing: route specific guilds/channels/threads to
    # different profiles. See gateway/profile_routing.py. Each entry is a
    # dict with: name, platform, profile, and optional guild_id/chat_id/thread_id.
    profile_routes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.multiplex_profile_allowlist = _normalize_multiplex_profile_allowlist(
            self.multiplex_profile_allowlist
        )
        self.systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            self.systemd_watchdog_seconds
        )

    def get_connected_platforms(self) -> List[Platform]:
        """Return list of platforms that are enabled and configured.

        Sorted by platform value so the rendered "Connected Platforms" list
        (and the home-channel blocks derived from it) is byte-stable across
        gateway restarts and mid-process platform registration — dict
        insertion order is not a stable contract and a reorder busts the
        prompt cache without any semantic change.
        """
        connected = [
            platform
            for platform, config in self.platforms.items()
            if config.enabled and self._is_platform_connected(platform, config)
        ]
        return sorted(connected, key=lambda p: str(p.value))

    def _is_platform_connected(self, platform: Platform, config: PlatformConfig) -> bool:
        """Check whether a single platform is sufficiently configured."""
        checker = _PLATFORM_CONNECTED_CHECKERS.get(platform)
        # Weixin requires both a token and an account_id (checked first so
        # the generic token branch doesn't let it through without account_id).
        if platform == Platform.WEIXIN:
            return checker(config)

        # Generic token/api_key auth covers Telegram, Discord, Slack, etc.
        if config.token or config.api_key:
            return True

        # Platform-specific check
        if checker is not None:
            return checker(config)

        # Plugin-registered platforms.  Force plugin discovery first so this
        # works even when GatewayConfig is constructed directly (e.g. in tests
        # or callers that bypass load_gateway_config(), which is what triggers
        # discovery in the normal path).  discover_plugins() is idempotent.
        try:
            from gateway.platform_registry import platform_registry
            try:
                from hermes_cli.plugins import discover_plugins
                discover_plugins()
            except Exception:
                pass
            entry = platform_registry.get(platform.value)
            if entry:
                if entry.is_connected is not None:
                    return entry.is_connected(config)
                if entry.validate_config is not None:
                    return entry.validate_config(config)
                return True
        except Exception:
            pass  # Registry not yet initialised during early import

        return False
    
    def get_home_channel(self, platform: Platform) -> Optional[HomeChannel]:
        """Get the home channel for a platform."""
        config = self.platforms.get(platform)
        return config.home_channel if config else None
    
    def get_reset_policy(
        self, 
        platform: Optional[Platform] = None,
        session_type: Optional[str] = None
    ) -> SessionResetPolicy:
        """
        Get the appropriate reset policy for a session.
        
        Priority: platform override > type override > default
        """
        # Platform-specific override takes precedence
        if platform and platform in self.reset_by_platform:
            return self.reset_by_platform[platform]
        
        # Type-specific override (dm, group, thread)
        if session_type and session_type in self.reset_by_type:
            return self.reset_by_type[session_type]
        
        return self.default_reset_policy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {
                p.value: c.to_dict() for p, c in self.platforms.items()
            },
            "default_reset_policy": self.default_reset_policy.to_dict(),
            "reset_by_type": {
                k: v.to_dict() for k, v in self.reset_by_type.items()
            },
            "reset_by_platform": {
                p.value: v.to_dict() for p, v in self.reset_by_platform.items()
            },
            "reset_triggers": self.reset_triggers,
            "quick_commands": self.quick_commands,
            "sessions_dir": str(self.sessions_dir),
            "write_sessions_json": self.write_sessions_json,
            "always_log_local": self.always_log_local,
            "filter_silence_narration": self.filter_silence_narration,
            "stt_enabled": self.stt_enabled,
            "stt_echo_transcripts": self.stt_echo_transcripts,
            "group_sessions_per_user": self.group_sessions_per_user,
            "thread_sessions_per_user": self.thread_sessions_per_user,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "multiplex_profiles": self.multiplex_profiles,
            "multiplex_profile_allowlist": self.multiplex_profile_allowlist,
            "room_link_url": self.room_link_url,
            "systemd_watchdog_seconds": self.systemd_watchdog_seconds,
            "loop_watchdog": self.loop_watchdog,
            "loop_watchdog_probe_interval_s": self.loop_watchdog_probe_interval_s,
            "loop_watchdog_probe_timeout_s": self.loop_watchdog_probe_timeout_s,
            "loop_watchdog_max_strikes": self.loop_watchdog_max_strikes,
            "unauthorized_dm_behavior": self.unauthorized_dm_behavior,
            "streaming": self.streaming.to_dict(),
            "session_store_max_age_days": self.session_store_max_age_days,
            "profile_routes": [
                asdict(r) if is_dataclass(r) and not isinstance(r, type) else r
                for r in self.profile_routes
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        data = _coerce_dict(data)
        nested_gateway = _coerce_dict(data.get("gateway"))

        def pick(key: str) -> Any:
            """Top-level key wins by presence; else the nested ``gateway.<key>`` form."""
            return data[key] if key in data else nested_gateway.get(key)

        platforms = {}
        for platform_name, platform_data in _coerce_dict(data.get("platforms", {})).items():
            if not isinstance(platform_data, dict):
                continue
            try:
                platforms[Platform(platform_name)] = PlatformConfig.from_dict(platform_data)
            except ValueError:
                pass  # Skip unknown platforms

        reset_by_platform = {}
        for platform_name, policy_data in _coerce_dict(data.get("reset_by_platform", {})).items():
            try:
                reset_by_platform[Platform(platform_name)] = SessionResetPolicy.from_dict(policy_data)
            except ValueError:
                pass

        stt = _coerce_dict(data.get("stt"))
        stt_enabled = data.get("stt_enabled")
        if stt_enabled is None:
            stt_enabled = stt.get("enabled")
        stt_echo_transcripts = data.get("stt_echo_transcripts")
        if stt_echo_transcripts is None:
            stt_echo_transcripts = stt.get("echo_transcripts")

        room_link_url = data.get("room_link_url")
        if not isinstance(room_link_url, str):
            room_link_url = None

        # Key prefix for the warning: "gateway." when the nested form was the one consulted.
        def key_label(key: str) -> str:
            return key if key in data else f"gateway.{key}"

        # Watchdog knobs: out-of-range / non-finite values fall back to the shipped defaults.
        probe_interval = _coerce_float(pick("loop_watchdog_probe_interval_s"), DEFAULT_LOOP_WATCHDOG_INTERVAL_S)
        if not math.isfinite(probe_interval) or not 1.0 <= probe_interval <= 3600.0:
            probe_interval = DEFAULT_LOOP_WATCHDOG_INTERVAL_S
        probe_timeout = _coerce_float(pick("loop_watchdog_probe_timeout_s"), DEFAULT_LOOP_WATCHDOG_TIMEOUT_S)
        if not math.isfinite(probe_timeout) or not 1.0 <= probe_timeout <= 600.0:
            probe_timeout = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
        max_strikes = _coerce_int(pick("loop_watchdog_max_strikes"), DEFAULT_LOOP_WATCHDOG_MAX_STRIKES)
        if not 1 <= max_strikes <= 1000:
            max_strikes = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES

        systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            pick("systemd_watchdog_seconds"), key_label("systemd_watchdog_seconds")
        )

        # Multiplexing is a genuine 3-tier chain: env > config.yaml > default False. The
        # GATEWAY_MULTIPLEX_PROFILES operator override wins when set to a recognized value
        # (hosted deployments stamp it on the container so the single multiplexed gateway the
        # connector depends on is forced on at every boot regardless of the image's config.yaml);
        # a blank or unrecognized env value falls through to config — a provisioned-but-
        # unpopulated Fly secret must not shadow a config.yaml opt-in. Config side: the
        # top-level VALUE wins when not None, else ``gateway.multiplex_profiles`` (written by
        # ``hermes config set gateway.multiplex_profiles true``).
        multiplex_profiles = data.get("multiplex_profiles")
        if multiplex_profiles is None:
            multiplex_profiles = nested_gateway.get("multiplex_profiles")
        env_multiplex = _env_multiplex_profiles_override()
        if env_multiplex is not None:
            multiplex_profiles = env_multiplex

        max_concurrent_sessions = _coerce_optional_positive_int(
            pick("max_concurrent_sessions"), key_label("max_concurrent_sessions")
        )

        try:
            session_store_max_age_days = max(int(data.get("session_store_max_age_days", 90)), 0)
        except (TypeError, ValueError):
            session_store_max_age_days = 90

        # Parse profile routes (validated by gateway.profile_routing)
        from gateway.profile_routing import parse_profile_routes
        profile_routes = parse_profile_routes(data.get("profile_routes") or [])

        return cls(
            platforms=platforms,
            default_reset_policy=SessionResetPolicy.from_dict(data["default_reset_policy"])
            if "default_reset_policy" in data
            else SessionResetPolicy(),
            reset_by_type={
                type_name: SessionResetPolicy.from_dict(policy_data)
                for type_name, policy_data in _coerce_dict(data.get("reset_by_type", {})).items()
            },
            reset_by_platform=reset_by_platform,
            reset_triggers=data.get("reset_triggers", ["/new", "/reset"]),
            quick_commands=_coerce_dict(data.get("quick_commands", {})),
            sessions_dir=Path(data["sessions_dir"]) if "sessions_dir" in data else get_hermes_home() / "sessions",
            write_sessions_json=_coerce_bool(data.get("write_sessions_json"), True),
            always_log_local=_coerce_bool(data.get("always_log_local"), True),
            filter_silence_narration=_coerce_bool(data.get("filter_silence_narration"), True),
            stt_enabled=_coerce_bool(stt_enabled, True),
            stt_echo_transcripts=_coerce_bool(stt_echo_transcripts, True),
            group_sessions_per_user=_coerce_bool(data.get("group_sessions_per_user"), True),
            thread_sessions_per_user=_coerce_bool(data.get("thread_sessions_per_user"), False),
            multiplex_profiles=_coerce_bool(multiplex_profiles, False),
            multiplex_profile_allowlist=pick("multiplex_profile_allowlist"),
            room_link_url=room_link_url,
            systemd_watchdog_seconds=systemd_watchdog_seconds,
            loop_watchdog=_coerce_bool(pick("loop_watchdog"), True),
            loop_watchdog_probe_interval_s=probe_interval,
            loop_watchdog_probe_timeout_s=probe_timeout,
            loop_watchdog_max_strikes=max_strikes,
            max_concurrent_sessions=max_concurrent_sessions,
            unauthorized_dm_behavior=_normalize_choice(
                data.get("unauthorized_dm_behavior"), {"pair", "ignore"}, "pair"
            ),
            streaming=StreamingConfig.from_dict(data.get("streaming", {})),
            session_store_max_age_days=session_store_max_age_days,
            profile_routes=profile_routes,
        )

    def get_unauthorized_dm_behavior(self, platform: Optional[Platform] = None) -> str:
        """Return the effective unauthorized-DM behavior for a platform.

        Email is inbox-shaped, not chat-shaped, so it defaults to ``"ignore"``
        unless ``platforms.email.unauthorized_dm_behavior`` explicitly opts
        into pairing. A global default does not opt email into pairing.
        """
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "unauthorized_dm_behavior" in platform_cfg.extra:
                return _normalize_choice(
                    platform_cfg.extra.get("unauthorized_dm_behavior"),
                    {"pair", "ignore"},
                    self.unauthorized_dm_behavior,
                )
            if platform == Platform.EMAIL:
                return "ignore"
        return self.unauthorized_dm_behavior

    def get_notice_delivery(self, platform: Optional[Platform] = None) -> str:
        """Return the effective notice-delivery mode for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "notice_delivery" in platform_cfg.extra:
                return _normalize_choice(
                    platform_cfg.extra.get("notice_delivery"), {"public", "private"}, "public"
                )
        return "public"


def load_gateway_config() -> GatewayConfig:
    """
    Load gateway configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables
    2. ~/.hermes/config.yaml (primary user-facing config)
    3. ~/.hermes/gateway.json (legacy — provides defaults under config.yaml)
    4. Built-in defaults
    """
    from gateway import config_loader

    _home = get_hermes_home()
    gw_data = config_loader.load_legacy_gateway_json(_home)
    try:
        config_loader.load_yaml_layer(_home, gw_data)
    except Exception as e:
        logger.warning(
            "Failed to process config.yaml — falling back to .env / gateway.json values. "
            "Check %s for syntax errors. Error: %s",
            _home / "config.yaml",
            e,
        )

    config = GatewayConfig.from_dict(gw_data)
    _apply_env_overrides(config)
    _validate_gateway_config(config)
    return config


def _validate_gateway_config(config: "GatewayConfig") -> None:
    """Validate and sanitize a loaded GatewayConfig in place.

    Called by ``load_gateway_config()`` after all config sources are merged.
    Extracted as a separate function for testability.
    """
    policy = config.default_reset_policy

    if not (0 <= policy.at_hour <= 23):
        logger.warning(
            "Invalid at_hour=%s (must be 0-23). Using default 4.", policy.at_hour
        )
        policy.at_hour = 4

    if policy.idle_minutes is None or policy.idle_minutes <= 0:
        logger.warning(
            "Invalid idle_minutes=%s (must be positive). Using default 1440.",
            policy.idle_minutes,
        )
        policy.idle_minutes = 1440

    # Warn about empty bot tokens — platforms that loaded an empty string
    # won't connect and the cause can be confusing without a log line.
    for platform, pconfig in config.platforms.items():
        if not pconfig.enabled:
            continue
        env_name = PLATFORM_TOKEN_ENV_NAMES.get(platform)
        if env_name and pconfig.token is not None and not pconfig.token.strip():
            logger.warning(
                "%s is enabled but %s is empty. "
                "The adapter will likely fail to connect.",
                platform.value, env_name,
            )

    # Reject known-weak placeholder tokens.
    # Ported from openclaw/openclaw#64586: users who copy .env.example
    # without changing placeholder values get a clear startup error instead
    # of a confusing "auth failed" from the platform API.
    try:
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        return

    for platform, pconfig in config.platforms.items():
        env_name = PLATFORM_TOKEN_ENV_NAMES.get(platform)
        token = pconfig.token
        if not (pconfig.enabled and env_name and token and token.strip()):
            continue
        if not has_usable_secret(token, min_length=4):
            logger.error(
                "%s is enabled but %s is set to a placeholder value ('%s'). "
                "Set a real bot token before starting the gateway. "
                "The adapter will NOT be started.",
                platform.value, env_name, token.strip()[:6] + "...",
            )
            pconfig.enabled = False


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply environment variable overrides to config (see ``gateway.config_env``)."""
    from gateway.config_env import _apply_env_overrides as _impl

    _impl(config)
