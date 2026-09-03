"""Gateway configuration: connected platforms, home channels, session reset
policies and delivery preferences, loaded from config.yaml / gateway.json / env.
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

_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Coerce bool-ish config values, preserving a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY_STRINGS:
            return True
        if lowered in _FALSY_STRINGS:
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


def _env_multiplex_profiles_override() -> "bool | None":
    """GATEWAY_MULTIPLEX_PROFILES operator override: True/False for a recognized token.

    ``None`` when unset, blank, or unrecognized so the caller keeps the config.yaml
    value (env > config > default). Blank is deliberately ``None``, not ``False``:
    a provisioned-but-unpopulated Fly secret arrives as ``""`` and must NOT shadow
    a config.yaml opt-in.
    """
    raw = os.getenv("GATEWAY_MULTIPLEX_PROFILES")
    token = (raw or "").strip().lower()
    if not token:
        return None
    if token in _TRUTHY_STRINGS:
        return True
    if token in _FALSY_STRINGS:
        return False
    logger.warning(
        "Ignoring unrecognized GATEWAY_MULTIPLEX_PROFILES=%r "
        "(expected one of %s or %s); falling back to config.yaml.",
        raw,
        sorted(_TRUTHY_STRINGS),
        sorted(_FALSY_STRINGS),
    )
    return None


def _normalize_transport_token(value: Any) -> str:
    """Canonical streaming transport token.

    YAML 1.1 parses bare ``on``/``off`` as booleans, so ``mode: off`` arrives as
    ``False``; stringifying would yield ``"false"`` and enable streaming. Booleans
    map to ``"auto"`` / ``"off"``; anything else lower-cases, default ``"auto"``.
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "off"
    return str(value).strip().lower() or "auto"


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float("inf")) — non-finite YAML must degrade, not abort loading.
        return default


def _coerce_optional_positive_int(value: Any, key: str) -> Optional[int]:
    """``None``/0/negative disable; malformed values are ignored with a warning so a typo never blocks startup."""
    if value is None:
        return None
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(value)
        parsed = int(value.strip(), 10) if isinstance(value, str) else int(value)
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
    """Bounded positive watchdog interval, or zero when disabled/invalid.

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

    Multiplexed profile startup installs per-profile secrets, which must win;
    outside a scope keep legacy ``os.getenv`` behavior.
    """
    if current_secret_scope() is not None:
        scope_val = _get_secret(name, None)
        return scope_val if scope_val is not None else default
    return os.environ.get(name, default)


def _getenv_str(name: str, default: str = "") -> str:
    val = _getenv(name, default)
    return val if val is not None else default


# Bundled platform plugin names, cached outside the enum so it never becomes a member.
_Platform__bundled_plugin_names: Optional[set] = None


class Platform(Enum):
    """Supported messaging platforms.

    Built-ins are explicit members. Plugin platforms are dynamic members created
    on demand by ``_missing_`` and cached in ``_value2member_map_`` so
    ``Platform("irc") is Platform("irc")`` holds.
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
        """Accept unknown names only for bundled or runtime-registered plugin adapters (no enum pollution)."""
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip().lower()
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]

        global _Platform__bundled_plugin_names
        if _Platform__bundled_plugin_names is None:
            _Platform__bundled_plugin_names = cls._scan_bundled_plugin_platforms()
        if value in _Platform__bundled_plugin_names:
            return cls._add_pseudo_member(value)
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
        """Names of bundled platform plugins under ``plugins/platforms/``."""
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


# Built-in values snapshotted before any dynamic _missing_ lookup.
_BUILTIN_PLATFORM_VALUES = frozenset(m.value for m in Platform.__members__.values())


# Platforms that bind a host TCP port. In a profile multiplexer only the default
# profile owns the shared listener (served via /p/<profile>/), so a SECONDARY
# profile enabling one of these is always a misconfiguration. Single source of
# truth for gateway/run.py startup validation and the dashboard's pre-write
# validation (hermes_cli/web_server.py). Platform .value strings.
PORT_BINDING_PLATFORM_VALUES = frozenset({
    "webhook", "api_server", "msgraph_webhook", "feishu", "wecom_callback",
    "bluebubbles", "sms", "whatsapp_cloud", "line", "teams",
})

# Platforms that only bind in one connection mode: Feishu's default websocket
# mode is an outbound long connection. platform value → the mode that binds.
PORT_BINDING_CONDITIONAL_MODES: dict[str, str] = {
    "feishu": "webhook",
}


def platform_binds_port(platform_value: str, extra: Optional[dict] = None) -> bool:
    """True when *platform_value* actually binds a port for *extra* config."""
    if platform_value not in PORT_BINDING_PLATFORM_VALUES:
        return False
    expected_mode = PORT_BINDING_CONDITIONAL_MODES.get(platform_value)
    if expected_mode is not None:
        actual = str((extra or {}).get("connection_mode", "websocket")).strip().lower()
        return actual == expected_mode
    return True


@dataclass
class HomeChannel:
    """Default destination for a platform (``deliver="telegram"`` without a chat ID).

    Thread-aware platforms may store a thread/topic ID so the bare platform
    target routes to the conversation where /sethome was run.
    """
    platform: Platform
    chat_id: str
    name: str
    thread_id: Optional[str] = None
    # Authenticated logical-target provenance observed by a platform adapter.
    # Relay egress re-attaches these; the connector remains the authorization
    # boundary and resolves them against its authoritative stores.
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
        optional = {k: str(data[k]) if data.get(k) else None for k in ("thread_id", "user_id", "scope_id")}
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            name=data.get("name", "Home"),
            **optional,
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
    """Controls when sessions reset (lose context).

    Modes: "daily" (at ``at_hour``), "idle" (after ``idle_minutes``), "both"
    (whichever first), "none" (never; context managed only by compression).
    Default "none": sessions never auto-reset unless the user opts in via
    ``session_reset`` in config.yaml (or gateway.json).
    """
    mode: str = "none"
    at_hour: int = 4  # 0-23, local time
    idle_minutes: int = 1440
    notify: bool = True  # Notify the user when auto-reset occurs
    notify_exclude_platforms: tuple = ("api_server", "webhook")
    # A background process this many hours old no longer blocks idle/daily reset
    # (a forgotten preview server must not pin a session forever). The process
    # is NOT killed — only ignored by the reset guard.
    bg_process_max_age_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "notify_exclude_platforms": list(self.notify_exclude_platforms)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionResetPolicy":
        data = _coerce_dict(data)

        def val(key: str, default: Any) -> Any:
            # Missing keys and explicit YAML nulls both take the default.
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
    """Per-channel model/provider/system_prompt override (``platforms.<name>.channel_overrides[channel_id]``)."""
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


# Platforms whose primary credential is ``PlatformConfig.token`` and the env var it
# loads from: drives empty-token warnings at validation and the multiplex
# primary-startup credential gate in ``gateway.run``. Platforms absent here
# authenticate another way (session files, webhooks, api_key) and must never be
# skipped for a missing token.
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
    token: Optional[str] = None
    api_key: Optional[str] = None  # API key if different from token
    home_channel: Optional[HomeChannel] = None
    # Reply threading: "off" never threads, "first" threads only the first chunk, "all" every chunk.
    reply_to_mode: str = "first"
    # "♻️ Gateway online/restarted" lifecycle pings. Set False on end-user
    # platforms (e.g. Slack) where operator-flavored notices are noise.
    gateway_restart_notification: bool = True
    # "typing…" / "is thinking…" indicator while the agent works (drives the
    # _keep_typing loop in gateway/platforms/base.py). Set False where it is
    # unwanted, e.g. Slack's setStatus disables the compose box.
    typing_indicator: bool = True
    # Custom working-state text for platforms whose indicator renders text (Slack
    # assistant status — needs assistant:write; Google Chat marker message). None
    # keeps each platform's built-in default; textless indicators ignore it.
    typing_status_text: Optional[str] = None
    channel_overrides: Dict[str, ChannelOverride] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)  # Platform-specific settings

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

        # gateway_restart_notification / typing_indicator / typing_status_text may be
        # top-level or bridged into ``extra`` by the shared-key loop; top-level wins.
        extra = _coerce_dict(data.get("extra", {}))

        def toplevel_or_extra(key: str) -> Any:
            value = data.get(key)
            return extra.get(key) if value is None else value

        raw_overrides = data.get("channel_overrides") or {}
        channel_overrides = {
            str(cid): ChannelOverride.from_dict(ov_data)
            for cid, ov_data in raw_overrides.items()
            if isinstance(ov_data, dict)
        } if isinstance(raw_overrides, dict) else {}

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


# Streaming defaults shared by StreamingConfig and StreamConsumerConfig. Tuned for
# Telegram's ~1 edit/s flood envelope: a touch under 1s breathes without hitting
# rate limits; the small buffer threshold makes short DM replies feel instant.
DEFAULT_STREAMING_EDIT_INTERVAL: float = 0.8
DEFAULT_STREAMING_BUFFER_THRESHOLD: int = 24
DEFAULT_STREAMING_CURSOR: str = " ▉"


@dataclass
class StreamingConfig:
    """Real-time token streaming to messaging platforms."""
    enabled: bool = False
    # Transport: "auto" prefers native draft updates (Telegram sendMessageDraft,
    # Bot API 9.5+) and falls back to edit-based; "draft" requests drafts with
    # edit fallback; "edit" is progressive editMessageText only; "off" disables.
    # "auto" is safe globally: adapters without draft support report
    # supports_draft_streaming() == False and use the edit path unchanged.
    transport: str = "auto"
    edit_interval: float = DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = DEFAULT_STREAMING_CURSOR
    # When >0, the final edit of a long stream is delivered as a fresh message if
    # the preview has been visible at least this many seconds, so the visible
    # timestamp reflects completion. Telegram only; 0 disables.
    fresh_final_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingConfig":
        if not isinstance(data, dict) or not data:
            return cls()

        # ``mode`` is a transport alias that ALSO implies ``enabled`` (``mode: off``
        # disables); an explicit ``enabled`` key always wins. A bare ``transport``
        # does NOT imply enabled: ``streaming.enabled`` is the documented master
        # switch, so ``transport`` only selects HOW to stream once it is on.
        raw_transport = data.get("transport")
        raw_mode = data.get("mode")
        transport = _normalize_transport_token(raw_transport if raw_transport is not None else raw_mode)

        if "enabled" in data:
            enabled = _coerce_bool(data.get("enabled"), False)
        elif raw_mode is not None:
            enabled = _normalize_transport_token(raw_mode) != "off"
        else:
            enabled = False

        return cls(
            enabled=enabled,
            transport=transport,
            edit_interval=_coerce_float(data.get("edit_interval"), DEFAULT_STREAMING_EDIT_INTERVAL),
            buffer_threshold=_coerce_int(data.get("buffer_threshold"), DEFAULT_STREAMING_BUFFER_THRESHOLD),
            cursor=data.get("cursor", DEFAULT_STREAMING_CURSOR),
            fresh_final_after_seconds=_coerce_float(data.get("fresh_final_after_seconds"), 0.0),
        )


def _has_usable_api_server_key(key: object) -> bool:
    """True when API_SERVER_KEY is strong enough for the adapter to start.

    Mirrors the startup guard in ``gateway/platforms/api_server.py``
    (``has_usable_secret`` with ``min_length=16``).
    """
    if not key:
        return False
    try:
        from hermes_cli.auth import has_usable_secret
    except ImportError:
        return len(str(key).strip()) >= 16
    return has_usable_secret(key, min_length=16)


def _needs_extra(*keys: str) -> Callable[[PlatformConfig], bool]:
    return lambda cfg: all(cfg.extra.get(k) for k in keys)


# Built-in "is this platform sufficiently configured?" checks by PlatformConfig.
# Platforms covered by the generic ``token or api_key`` check (Telegram, Discord,
# Slack, Matrix, Mattermost, HomeAssistant) need no entry.
_PLATFORM_CONNECTED_CHECKERS: dict[Platform, Callable[[PlatformConfig], bool]] = {
    Platform.WEIXIN: lambda cfg: bool(cfg.extra.get("account_id") and (cfg.token or cfg.extra.get("token"))),
    Platform.WHATSAPP_CLOUD: _needs_extra("phone_number_id", "access_token"),
    Platform.SIGNAL: _needs_extra("http_url"),
    Platform.API_SERVER: lambda cfg: _has_usable_api_server_key(cfg.extra.get("key") if cfg else None),
    Platform.WEBHOOK: lambda cfg: True,
    Platform.MSGRAPH_WEBHOOK: lambda cfg: bool(str(cfg.extra.get("client_state") or "").strip()),
    Platform.BLUEBUBBLES: _needs_extra("server_url", "password"),
    Platform.QQBOT: _needs_extra("app_id", "client_secret"),
    Platform.YUANBAO: _needs_extra("app_id", "app_secret"),
    # Relay dials OUT to a connector: "connected" once an endpoint URL is configured
    # (capabilities are negotiated at handshake). EXPERIMENTAL.
    Platform.RELAY: lambda cfg: bool(cfg.extra.get("relay_url") or cfg.extra.get("url")),
}


@dataclass
class GatewayConfig:
    """Main gateway configuration: platform connections, session policies, delivery settings."""
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    default_reset_policy: SessionResetPolicy = field(default_factory=SessionResetPolicy)
    reset_by_type: Dict[str, SessionResetPolicy] = field(default_factory=dict)
    reset_by_platform: Dict[Platform, SessionResetPolicy] = field(default_factory=dict)
    reset_triggers: List[str] = field(default_factory=lambda: ["/new", "/reset"])
    # Slash commands that bypass the agent loop.
    quick_commands: Dict[str, Any] = field(default_factory=dict)
    sessions_dir: Path = field(default_factory=lambda: get_hermes_home() / "sessions")
    # Keep writing the legacy sessions.json mirror of the routing index (primary
    # copy: state.db gateway_routing). Default True for external tooling and
    # downgrade safety.
    write_sessions_json: bool = True
    always_log_local: bool = True  # Always save cron outputs to local files
    # Drop outbound "silence narration" (*(silent)*, 🔇, a bare ".") pre-send:
    # model hallucinations that ping-pong in bot-to-bot channels. Substrate-level
    # guard that survives SOUL.md/prompt drift; False = raw passthrough.
    filter_silence_narration: bool = True
    stt_enabled: bool = True  # Auto-transcribe inbound voice messages
    stt_echo_transcripts: bool = True  # Echo raw STT transcripts back to the user
    group_sessions_per_user: bool = True  # Isolate group sessions per participant when user IDs exist
    thread_sessions_per_user: bool = False  # False = threads shared across participants
    max_concurrent_sessions: Optional[int] = None  # Positive int caps simultaneous active sessions
    # Opt-in: the default profile's gateway serves inbound messages for every
    # profile on the host (profiles stamped into session keys, per-profile
    # adapters/credentials). False = single HERMES_HOME, no profile stamping.
    multiplex_profiles: bool = False
    # None = historical serve-all; [] = default profile only.
    multiplex_profile_allowlist: Optional[List[str]] = None
    # Public HTTPS endpoint for scoped RoomLink calls. Disabled by default: an API
    # key alone must never advertise a route. HERMES_ROOM_LINK_URL overrides.
    room_link_url: Optional[str] = None
    # Opt-in systemd event-loop watchdog; zero keeps Type=simple and disables sd_notify.
    systemd_watchdog_seconds: int = 0
    # In-process event-loop liveness watchdog: a daemon thread probes the loop
    # with call_soon_threadsafe; after consecutive misses it dumps all-thread
    # stacks and hard-exits with the service-restart code. The knobs (seconds)
    # tolerate transient self-recovering stalls (adapter reconnect doing sync
    # socket I/O) so a short block does not force exit code 75 and restart
    # churn that stalls cron; a genuine wedge still escalates.
    loop_watchdog: bool = True
    loop_watchdog_probe_interval_s: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S
    loop_watchdog_probe_timeout_s: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
    # Default 3 (~90-120s sustained block): the false-positive class (the
    # watchdog's own on-loop heartbeat fsync) is fixed at the root by the off-loop
    # write + two-witness probe, so raising this would only delay recovery.
    loop_watchdog_max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES
    unauthorized_dm_behavior: str = "pair"  # "pair" or "ignore"
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    # Drop SessionEntry records older than this from the store and sessions.json.
    # Invisible to users (a resumed chat gets a fresh session). 0 = disabled.
    session_store_max_age_days: int = 90
    # Route guilds/channels/threads to profiles (gateway/profile_routing.py).
    profile_routes: list = field(default_factory=list)

    # Scalar fields serialized verbatim by ``to_dict`` (in output order).
    _SCALAR_DICT_FIELDS = (
        "write_sessions_json", "always_log_local", "filter_silence_narration", "stt_enabled",
        "stt_echo_transcripts", "group_sessions_per_user", "thread_sessions_per_user",
        "max_concurrent_sessions", "multiplex_profiles", "multiplex_profile_allowlist",
        "room_link_url", "systemd_watchdog_seconds", "loop_watchdog",
        "loop_watchdog_probe_interval_s", "loop_watchdog_probe_timeout_s",
        "loop_watchdog_max_strikes", "unauthorized_dm_behavior",
    )

    def __post_init__(self) -> None:
        self.multiplex_profile_allowlist = _normalize_multiplex_profile_allowlist(
            self.multiplex_profile_allowlist
        )
        self.systemd_watchdog_seconds = coerce_systemd_watchdog_seconds(
            self.systemd_watchdog_seconds
        )

    def get_connected_platforms(self) -> List[Platform]:
        """Enabled + configured platforms, sorted by value.

        Sorted so the rendered "Connected Platforms" list is byte-stable across
        restarts and mid-process registration: a reorder busts the prompt cache.
        """
        connected = [
            platform
            for platform, config in self.platforms.items()
            if config.enabled and self._is_platform_connected(platform, config)
        ]
        return sorted(connected, key=lambda p: str(p.value))

    def _is_platform_connected(self, platform: Platform, config: PlatformConfig) -> bool:
        checker = _PLATFORM_CONNECTED_CHECKERS.get(platform)
        # Weixin needs token AND account_id, so it must bypass the generic token branch.
        if platform == Platform.WEIXIN:
            return checker(config)
        if config.token or config.api_key:
            return True
        if checker is not None:
            return checker(config)

        # Plugin-registered platforms. Force (idempotent) plugin discovery so this
        # works when GatewayConfig is constructed directly, bypassing load_gateway_config().
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
        config = self.platforms.get(platform)
        return config.home_channel if config else None

    def get_reset_policy(
        self,
        platform: Optional[Platform] = None,
        session_type: Optional[str] = None
    ) -> SessionResetPolicy:
        """Priority: platform override > type override > default."""
        if platform and platform in self.reset_by_platform:
            return self.reset_by_platform[platform]
        if session_type and session_type in self.reset_by_type:
            return self.reset_by_type[session_type]
        return self.default_reset_policy

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "platforms": {p.value: c.to_dict() for p, c in self.platforms.items()},
            "default_reset_policy": self.default_reset_policy.to_dict(),
            "reset_by_type": {k: v.to_dict() for k, v in self.reset_by_type.items()},
            "reset_by_platform": {p.value: v.to_dict() for p, v in self.reset_by_platform.items()},
            "reset_triggers": self.reset_triggers,
            "quick_commands": self.quick_commands,
            "sessions_dir": str(self.sessions_dir),
        }
        for name in self._SCALAR_DICT_FIELDS:
            result[name] = getattr(self, name)
        result["streaming"] = self.streaming.to_dict()
        result["session_store_max_age_days"] = self.session_store_max_age_days
        result["profile_routes"] = [
            asdict(r) if is_dataclass(r) and not isinstance(r, type) else r
            for r in self.profile_routes
        ]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        data = _coerce_dict(data)
        nested_gateway = _coerce_dict(data.get("gateway"))

        def pick(key: str) -> Any:
            """Top-level key wins by presence; else the nested ``gateway.<key>`` form."""
            return data[key] if key in data else nested_gateway.get(key)

        def key_label(key: str) -> str:
            """Warning key prefix: "gateway." when the nested form was the one consulted."""
            return key if key in data else f"gateway.{key}"

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

        # Multiplexing is env > config.yaml > default False. The GATEWAY_MULTIPLEX_PROFILES
        # operator override wins when set to a recognized value (hosted deployments stamp
        # it on the container); blank/unrecognized falls through to config. Config side:
        # the top-level VALUE wins when not None, else ``gateway.multiplex_profiles``.
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

        from gateway.profile_routing import parse_profile_routes

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
            profile_routes=parse_profile_routes(data.get("profile_routes") or []),
        )

    def get_unauthorized_dm_behavior(self, platform: Optional[Platform] = None) -> str:
        """Effective unauthorized-DM behavior for a platform.

        Email is inbox-shaped, not chat-shaped, so it defaults to ``"ignore"``
        unless ``platforms.email.unauthorized_dm_behavior`` explicitly opts in;
        a global default does not opt email into pairing.
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
        """Effective notice-delivery mode ("public"/"private") for a platform."""
        if platform:
            platform_cfg = self.platforms.get(platform)
            if platform_cfg and "notice_delivery" in platform_cfg.extra:
                return _normalize_choice(
                    platform_cfg.extra.get("notice_delivery"), {"public", "private"}, "public"
                )
        return "public"


def load_gateway_config() -> GatewayConfig:
    """Load gateway configuration. Priority: env > ~/.hermes/config.yaml > legacy gateway.json > defaults."""
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
    """Validate and sanitize a loaded GatewayConfig in place (after all sources are merged)."""
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

    # An empty token won't connect and the cause is confusing without a log line.
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

    # Reject known-weak placeholder tokens (copied .env.example) with a clear
    # startup error instead of a confusing "auth failed" from the platform API.
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
