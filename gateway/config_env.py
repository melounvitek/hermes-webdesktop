"""Environment-variable overrides for the gateway config (``_apply_env_overrides``).

Runs after ``GatewayConfig.from_dict`` so env always wins over config.yaml /
gateway.json. Most platforms follow one shape — "credentials present in env
⇒ enable the platform and copy the values into ``extra``" — declared as
``_Cred`` rows in ``_ENV_STEPS`` (source order = application order).
Platforms with unique gating are small functions in the same table.
"""

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    _getenv_str,
    _has_usable_api_server_key,
)
from utils import is_truthy_value

# Logger name parity with the origin module: records stay under "gateway.config".
logger = logging.getLogger("gateway.config")

getenv = _getenv_str

# Platforms already warned about "explicitly disabled in config.yaml, but
# credentials present in env". Config reloads every turn, so the notice is
# one-time per platform per process.
_EXPLICIT_DISABLE_WARNED: set = set()

# Env var(s) whose presence drives each platform's env-enable branch, named in
# the explicit-disable WARNING.
_ENV_ENABLE_CREDENTIALS: dict = {
    Platform.TELEGRAM: ("TELEGRAM_BOT_TOKEN",),
    Platform.DISCORD: ("DISCORD_BOT_TOKEN",),
    Platform.SLACK: ("SLACK_BOT_TOKEN",),
    Platform.WHATSAPP_CLOUD: ("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN"),
    Platform.SIGNAL: ("SIGNAL_HTTP_URL",),
    Platform.MATTERMOST: ("MATTERMOST_TOKEN",),
    Platform.MATRIX: ("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"),
    Platform.HOMEASSISTANT: ("HASS_TOKEN",),
    Platform.EMAIL: ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
    Platform.SMS: ("TWILIO_ACCOUNT_SID",),
    Platform.DINGTALK: ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
    Platform.FEISHU: ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
    Platform.WECOM: ("WECOM_BOT_ID", "WECOM_SECRET"),
    Platform.WECOM_CALLBACK: ("WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"),
    Platform.WEIXIN: ("WEIXIN_TOKEN", "WEIXIN_ACCOUNT_ID"),
    Platform.BLUEBUBBLES: ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
    Platform.QQBOT: ("QQ_APP_ID", "QQ_CLIENT_SECRET"),
    Platform.YUANBAO: ("YUANBAO_APP_ID", "YUANBAO_APP_SECRET"),
    Platform.RELAY: ("GATEWAY_RELAY_URL",),
}


def _warn_explicit_disable_beats_env(platform: Platform) -> None:
    """One-time WARNING: ``platforms.<x>.enabled: false`` wins over env creds.

    Credential presence used to force-enable platforms regardless of an explicit
    ``enabled: false``; users relying on "creds in .env = platform on" must be
    told why it went dark.
    """
    if platform in _EXPLICIT_DISABLE_WARNED:
        return
    _EXPLICIT_DISABLE_WARNED.add(platform)
    names = _ENV_ENABLE_CREDENTIALS.get(platform) or ()
    present = [n for n in names if (os.environ.get(n) or "").strip()]
    creds = ", ".join(present or names) or "its credentials"
    logger.warning(
        "Platform '%s' is explicitly disabled by platforms.%s.enabled: false in "
        "config.yaml, so the credentials found in the environment (%s) will NOT "
        "start its adapter. Environment credentials no longer override an "
        "explicit disable. Remove the key or set platforms.%s.enabled: true to "
        "turn it back on.",
        platform.value, platform.value, creds, platform.value,
    )


# --- small value parsers -----------------------------------------------------

def _csv_list(value: str) -> list:
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_or(default: int) -> Callable[[str], int]:
    """``int(raw.strip(), 10)`` with *default* on malformed/blank input."""
    def parse(raw: str) -> int:
        try:
            return int(str(raw).strip(), 10)
        except (TypeError, ValueError):
            return default
    return parse


def _strip_slash(value: str) -> str:
    return value.rstrip("/")


def _truthy_token(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "on"}


def _mention_patterns(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return _csv_list(value.replace("\n", ","))


def _env_first(envs) -> str:
    """First truthy value among *envs* (a single name or a tuple of alternatives)."""
    if isinstance(envs, str):
        return getenv(envs)
    return next((v for v in map(getenv, envs) if v), "")


# --- reusable steps -----------------------------------------------------------

_INT = object()  # spec marker: ``int(value)``, silently skipped when malformed


def _env_extras(extra: Dict[str, Any], spec, *, strip: bool = False) -> None:
    """``extra[key] = fn(value)`` for each ``(key, env[, fn])`` whose env value is truthy.

    ``strip=True`` strips BEFORE the truthiness check. ``fn=_INT`` parses an int
    and silently ignores a non-integer value.
    """
    for key, env, *fn in spec:
        value = getenv(env)
        if strip:
            value = value.strip()
        if not value:
            continue
        if fn and fn[0] is _INT:
            with contextlib.suppress(ValueError):
                extra[key] = int(value)
        else:
            extra[key] = fn[0](value) if fn else value


def _env_home_channel(config: GatewayConfig, platform: Platform, env_base: str, *, strip: bool = False) -> None:
    """Set ``home_channel`` from ``<env_base>`` (+``_NAME``/``_THREAD_ID``) when the platform is configured."""
    chat_id = getenv(env_base)
    if strip:
        chat_id = chat_id.strip()
    if chat_id and platform in config.platforms:
        config.platforms[platform].home_channel = HomeChannel(
            platform=platform,
            chat_id=chat_id,
            name=getenv(f"{env_base}_NAME", "Home"),
            thread_id=getenv(f"{env_base}_THREAD_ID") or None,
        )


def _env_reply_mode(config: GatewayConfig, platform: Platform, env: str) -> None:
    mode = getenv(env).lower()
    if mode in {"off", "first", "all"}:
        config.platforms.setdefault(platform, PlatformConfig()).reply_to_mode = mode


def _enable_from_env(config: GatewayConfig, platform: Platform) -> PlatformConfig:
    """Enable *platform* because its env credentials are present — unless config.yaml explicitly disabled it.

    READS (does not pop) the ``_enabled_explicit`` marker: the registry-driven
    plugin-enable pass also needs it; ``_scrub_explicit_markers`` removes it last.
    """
    if platform not in config.platforms:
        config.platforms[platform] = PlatformConfig(enabled=True)
        return config.platforms[platform]
    platform_config = config.platforms[platform]
    if not platform_config.enabled:
        if platform_config.extra.get("_enabled_explicit", False):
            _warn_explicit_disable_beats_env(platform)
        else:
            platform_config.enabled = True
    return platform_config


def _enable_port_bound_from_env(config: GatewayConfig, platform: Platform) -> PlatformConfig:
    """Enable a port-binding platform (api_server/webhook) unless config.yaml explicitly disabled it.

    In multiplex mode a secondary profile pins ``platforms.<x>.enabled: false`` so
    it shares the default profile's listener yet still inherits the process env;
    without this guard env presence would force-enable the listener and trip
    MultiplexConfigError. POPs the marker: these branches are terminal.
    """
    platform_config = config.platforms.setdefault(platform, PlatformConfig())
    explicit = platform_config.extra.pop("_enabled_explicit", False)
    if not explicit or platform_config.enabled:
        platform_config.enabled = True
    return platform_config


@dataclass(frozen=True)
class _Cred:
    """Credential-gated platform enable, applied as ``step(config)``.

    ``creds``: env names that must ALL be truthy; an inner tuple lists alternatives (ANY).
    ``token``: env whose truthy value becomes ``PlatformConfig.token`` (stored even when
    yaml disables the adapter, so sending skills can use it).
    ``fixed``: ``(extra_key, env[, default[, fn]])`` always written once enabled (``env`` may
    be a tuple of alternatives). ``optional`` / ``optional_stripped``: ``_env_extras`` specs.
    ``warn_missing``: ``(env, msg)`` logged BEFORE enabling when *env* is blank.
    ``then``: unique tail ``fn(config, platform_config)``. ``home``: ``_env_home_channel`` env
    base applied only when the gate passed.
    """
    platform: Platform
    creds: tuple
    token: Optional[str] = None
    fixed: tuple = ()
    optional: tuple = ()
    optional_stripped: tuple = ()
    warn_missing: Optional[tuple] = None
    then: Optional[Callable[[GatewayConfig, PlatformConfig], None]] = None
    home: Optional[str] = None
    home_strip: bool = False

    def __call__(self, config: GatewayConfig) -> None:
        if not all(_env_first(group) for group in self.creds):
            return
        if self.warn_missing and not getenv(self.warn_missing[0]):
            logger.warning(self.warn_missing[1])
        platform_config = _enable_from_env(config, self.platform)
        if self.token:
            token = getenv(self.token)
            if token:
                platform_config.token = token
        extra = platform_config.extra
        for key, env, *rest in self.fixed:
            default = rest[0] if rest else ""
            value = _env_first(env) or default if isinstance(env, tuple) else getenv(env, default)
            extra[key] = rest[1](value) if len(rest) > 1 else value
        _env_extras(extra, self.optional)
        _env_extras(extra, self.optional_stripped, strip=True)
        if self.then is not None:
            self.then(config, platform_config)
        if self.home:
            _env_home_channel(config, self.platform, self.home, strip=self.home_strip)


def _Home(platform: Platform, env_base: str, *, strip: bool = False):
    return partial(_env_home_channel, platform=platform, env_base=env_base, strip=strip)


def _ReplyMode(platform: Platform, env: str):
    return partial(_env_reply_mode, platform=platform, env=env)


# --- platform-unique branches ------------------------------------------------

def _telegram_fallback_ips(config: GatewayConfig) -> None:
    ips = getenv("TELEGRAM_FALLBACK_IPS")
    if ips:
        config.platforms.setdefault(Platform.TELEGRAM, PlatformConfig()).extra["fallback_ips"] = _csv_list(ips)


def _whatsapp(config: GatewayConfig) -> None:
    """WhatsApp (Baileys bridge) uses a flag, not credentials; an explicit false overrides YAML."""
    raw = getenv("WHATSAPP_ENABLED")
    enabled = is_truthy_value(raw)
    wa_cfg = config.platforms.get(Platform.WHATSAPP)
    if wa_cfg is not None:
        if raw.lower() in {"false", "0", "no"}:
            wa_cfg.enabled = False
        elif enabled:
            wa_cfg.enabled = True
    elif enabled:
        config.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True)


def _slack_home(config: GatewayConfig) -> None:
    """SLACK_HOME_CHANNEL creates a disabled Slack entry if needed and keeps
    user_id/scope_id provenance when the chat_id is unchanged."""
    slack_home = getenv("SLACK_HOME_CHANNEL")
    if not slack_home:
        return
    slack_config = config.platforms.setdefault(Platform.SLACK, PlatformConfig(enabled=False))
    existing_home = slack_config.home_channel
    same_home = existing_home is not None and existing_home.chat_id == slack_home
    slack_config.home_channel = HomeChannel(
        platform=Platform.SLACK,
        chat_id=slack_home,
        name=getenv("SLACK_HOME_CHANNEL_NAME"),
        thread_id=getenv("SLACK_HOME_CHANNEL_THREAD_ID") or None,
        user_id=existing_home.user_id if same_home else None,
        scope_id=existing_home.scope_id if same_home else None,
    )


def _matrix_e2ee(config: GatewayConfig, matrix_config: PlatformConfig) -> None:
    mode = getenv("MATRIX_E2EE_MODE").strip().lower()
    matrix_config.extra["encryption"] = (
        mode in ("required", "require", "optional", "prefer", "preferred")
        or is_truthy_value(getenv("MATRIX_ENCRYPTION"))
    )
    if mode:
        matrix_config.extra["e2ee_mode"] = mode
    _env_extras(matrix_config.extra, (("device_id", "MATRIX_DEVICE_ID"),))


def _sms_api_key(config: GatewayConfig, sms_config: PlatformConfig) -> None:
    sms_config.api_key = getenv("TWILIO_AUTH_TOKEN")


def _api_server(config: GatewayConfig) -> None:
    """Require a usable key: API_SERVER_ENABLED alone would load an unauthenticated
    platform whose adapter refuses to start, leaving the reconnect watcher spinning."""
    key = getenv("API_SERVER_KEY")
    if not _has_usable_api_server_key(key):
        return
    extra = _enable_port_bound_from_env(config, Platform.API_SERVER).extra
    extra["key"] = key
    origins = _csv_list(getenv("API_SERVER_CORS_ORIGINS"))
    if origins:
        extra["cors_origins"] = origins
    _env_extras(extra, (("port", "API_SERVER_PORT", _INT), ("host", "API_SERVER_HOST"), ("model_name", "API_SERVER_MODEL_NAME")))


def _webhook(config: GatewayConfig) -> None:
    if is_truthy_value(getenv("WEBHOOK_ENABLED")):
        extra = _enable_port_bound_from_env(config, Platform.WEBHOOK).extra
        _env_extras(extra, (("port", "WEBHOOK_PORT", _INT), ("secret", "WEBHOOK_SECRET")))


def _msgraph_webhook(config: GatewayConfig) -> None:
    enabled = is_truthy_value(getenv("MSGRAPH_WEBHOOK_ENABLED"))
    client_state = getenv("MSGRAPH_WEBHOOK_CLIENT_STATE")
    resources = getenv("MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES")
    allowed_cidrs = getenv("MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS")
    if not (
        enabled
        or Platform.MSGRAPH_WEBHOOK in config.platforms
        or getenv("MSGRAPH_WEBHOOK_PORT")
        or client_state
        or resources
        or allowed_cidrs
    ):
        return
    msgraph_cfg = config.platforms.setdefault(Platform.MSGRAPH_WEBHOOK, PlatformConfig())
    # Same explicit-disable guard as webhook, but READ (don't pop) the marker: the
    # relay-exclusive pass still consults it; the end scrub removes it.
    if enabled and (not msgraph_cfg.extra.get("_enabled_explicit", False) or msgraph_cfg.enabled):
        msgraph_cfg.enabled = True
    _env_extras(msgraph_cfg.extra, (("port", "MSGRAPH_WEBHOOK_PORT", _INT),))
    if client_state:
        msgraph_cfg.extra["client_state"] = client_state
    for key, raw in (("accepted_resources", resources), ("allowed_source_cidrs", allowed_cidrs)):
        items = _csv_list(raw)
        if items:
            msgraph_cfg.extra[key] = items


def _qq_home(config: GatewayConfig, qq_config: PlatformConfig) -> None:
    qq_home = getenv("QQBOT_HOME_CHANNEL").strip()
    name_env = "QQBOT_HOME_CHANNEL_NAME"
    if not qq_home:
        # Back-compat: accept the pre-rename name and log a one-time warning.
        qq_home = getenv("QQ_HOME_CHANNEL").strip()
        if qq_home:
            name_env = "QQ_HOME_CHANNEL_NAME"
            logger.warning(
                "QQ_HOME_CHANNEL is deprecated; rename to QQBOT_HOME_CHANNEL "
                "in your .env for consistency with the platform key."
            )
    if qq_home:
        qq_config.home_channel = HomeChannel(
            platform=Platform.QQBOT,
            chat_id=qq_home,
            name=getenv("QQBOT_HOME_CHANNEL_NAME") or getenv(name_env, "Home"),
            thread_id=getenv("QQBOT_HOME_CHANNEL_THREAD_ID") or getenv("QQ_HOME_CHANNEL_THREAD_ID") or None,
        )


def _session_settings(config: GatewayConfig) -> None:
    for env, attr in (("SESSION_IDLE_MINUTES", "idle_minutes"), ("SESSION_RESET_HOUR", "at_hour")):
        raw = getenv(env)
        if raw:
            with contextlib.suppress(ValueError):
                setattr(config.default_reset_policy, attr, int(raw))


def _enable_plugin_platforms_from_env(config: GatewayConfig) -> None:
    """Registry-driven enable for plugin platforms (built-ins have explicit rows in ``_ENV_STEPS``).

    A plugin platform is enabled when its credentials are configured (``is_connected``)
    and its deps are present (passive ``check_fn``) or installable on demand
    (``ensure_deps_fn`` — run later by ``create_adapter()``, never here: an active
    installer in this sweep pip-installed SDKs on every load and boot-looped the
    desktop app). ``is_connected`` MUST gate enablement: ``check_fn`` alone would
    enable unconfigured platforms that then retry-connect forever with no token.
    """
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            try:
                platform = Platform(entry.name)
            except Exception as e:
                logger.debug("unknown platform name %r: %s", entry.name, e)
                continue
            existing_cfg = config.platforms.get(platform)
            already_enabled = existing_cfg is not None and existing_cfg.enabled
            existing_extra = dict(existing_cfg.extra or {}) if existing_cfg is not None else {}
            # Never re-enable a platform the user explicitly disabled (marker set by the YAML loader).
            if existing_cfg is not None and not already_enabled and existing_extra.get("_enabled_explicit", False):
                continue
            # Seed candidate extras so plugins whose ``is_connected`` reads ``config.extra``
            # (Google Chat) see the same state they will after enablement.
            seed_for_probe = None
            if entry.env_enablement_fn is not None:
                try:
                    seed_for_probe = entry.env_enablement_fn()
                except Exception as e:
                    logger.debug("env_enablement_fn for %s raised: %s", entry.name, e)
            has_seed = isinstance(seed_for_probe, dict) and bool(seed_for_probe)

            # Only consult is_connected for platforms not already enabled by YAML/env.
            if not already_enabled and entry.is_connected is not None:
                try:
                    # Probe a transient ``enabled=True`` view (some ``is_connected``
                    # short-circuit on ``config.enabled``); never mutate ``existing_cfg``.
                    if has_seed:
                        for k, v in seed_for_probe.items():
                            if k != "home_channel":
                                existing_extra.setdefault(k, v)
                    configured = bool(entry.is_connected(PlatformConfig(enabled=True, extra=existing_extra)))
                except Exception as exc:
                    logger.debug("is_connected for %s raised: %s — skipping enablement", entry.name, exc)
                    configured = False
                if not configured:
                    logger.debug(
                        "Plugin platform '%s' available but not configured "
                        "(is_connected returned False) — skipping enable",
                        entry.name,
                    )
                    continue
            # Verify dependencies LAST — only for platforms already enabled or past the credential gate.
            try:
                deps_ok = bool(entry.check_fn())
            except Exception as e:
                logger.debug("check_fn for %s raised: %s", entry.name, e)
                deps_ok = False
            if not deps_ok and entry.ensure_deps_fn is None:
                continue
            platform_config = config.platforms.setdefault(platform, PlatformConfig())
            platform_config.enabled = True
            if has_seed:
                # Commit the env-seeded extras (reuse the probe result; don't call env_enablement_fn twice).
                seed = dict(seed_for_probe)
                home = seed.pop("home_channel", None)
                platform_config.extra.update(seed)
                if isinstance(home, dict) and home.get("chat_id"):
                    platform_config.home_channel = HomeChannel(
                        platform=platform,
                        chat_id=str(home["chat_id"]),
                        name=str(home.get("name") or "Home"),
                        thread_id=str(home["thread_id"]) if home.get("thread_id") else None,
                    )
    except Exception as e:
        logger.debug("Plugin platform enable pass failed: %s", e)


def _relay(config: GatewayConfig) -> None:
    """Relay (connector-fronted platform, EXPERIMENTAL): enabled by GATEWAY_RELAY_URL or
    gateway.relay_url. The adapter dials OUT (no inbound port); the connected-checker
    keys on extra["relay_url"], so the URL is mirrored into extra.

    Relay-exclusive: the GATEWAY_RELAY_URL env stamp marks a deployment where the
    connector owns every platform connection; a directly-connected adapter in the same
    process would be a second unmanaged ingress (duplicate deliveries, split sessions,
    a socket that disarms scale-to-zero). So the env stamp disables all other messaging
    platforms — even ones explicitly enabled in config.yaml. Non-messaging surfaces
    (local, api_server, webhook — same set as the scale-to-zero arm gate) are untouched;
    relay via gateway.relay_url only keeps the old additive behavior. Opt out with
    GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true (also read through the scoped getenv).
    """
    relay_url_env = getenv("GATEWAY_RELAY_URL").strip()
    existing_relay = config.platforms.get(Platform.RELAY)
    relay_url_yaml = str(existing_relay.extra.get("relay_url") or "").strip() if existing_relay else ""
    relay_url_val = relay_url_env or relay_url_yaml
    if relay_url_val:
        _enable_from_env(config, Platform.RELAY).extra["relay_url"] = relay_url_val.rstrip("/")

    if not relay_url_env or is_truthy_value(getenv("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS")):
        return
    non_messaging = {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}
    for platform, platform_config in config.platforms.items():
        if platform is Platform.RELAY or platform in non_messaging or not platform_config.enabled:
            continue
        if platform_config.extra.get("_enabled_explicit"):
            logger.warning(
                "Relay connector is configured via GATEWAY_RELAY_URL; "
                "disabling directly-connected platform '%s' even though "
                "it is explicitly enabled in this profile's configuration. "
                "All messaging goes through the connector on this "
                "deployment. Set GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true "
                "to keep direct platforms alongside the relay.",
                platform.value,
            )
        else:
            logger.info(
                "Relay connector is configured via GATEWAY_RELAY_URL; "
                "disabling directly-connected platform '%s'.",
                platform.value,
            )
        platform_config.enabled = False


def _scrub_explicit_markers(config: GatewayConfig) -> None:
    for platform_config in config.platforms.values():
        platform_config.extra.pop("_enabled_explicit", None)


# Application order is significant: e.g. Telegram's reply mode may create the (disabled)
# platform entry BEFORE its home channel is read, while Discord reads home first; a home
# channel is only attached to a platform that already exists. Relay-exclusive disabling
# runs after the plugin pass and before the marker scrub, which must be last.
_ENV_STEPS: tuple = (
    _Cred(Platform.TELEGRAM, ("TELEGRAM_BOT_TOKEN",), token="TELEGRAM_BOT_TOKEN"),
    _ReplyMode(Platform.TELEGRAM, "TELEGRAM_REPLY_TO_MODE"),
    _telegram_fallback_ips,
    _Home(Platform.TELEGRAM, "TELEGRAM_HOME_CHANNEL"),
    _Cred(Platform.DISCORD, ("DISCORD_BOT_TOKEN",), token="DISCORD_BOT_TOKEN"),
    _Home(Platform.DISCORD, "DISCORD_HOME_CHANNEL"),
    _ReplyMode(Platform.DISCORD, "DISCORD_REPLY_TO_MODE"),
    _whatsapp,
    _Home(Platform.WHATSAPP, "WHATSAPP_HOME_CHANNEL"),
    # WhatsApp Cloud API (Meta Business Platform). Distinct from the Baileys bridge;
    # both adapters can run in parallel against different phone numbers.
    _Cred(
        Platform.WHATSAPP_CLOUD, ("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "WHATSAPP_CLOUD_ACCESS_TOKEN"),
        fixed=(("phone_number_id", "WHATSAPP_CLOUD_PHONE_NUMBER_ID"), ("access_token", "WHATSAPP_CLOUD_ACCESS_TOKEN")),
        optional=(
            ("app_id", "WHATSAPP_CLOUD_APP_ID"),
            ("app_secret", "WHATSAPP_CLOUD_APP_SECRET"),
            ("waba_id", "WHATSAPP_CLOUD_WABA_ID"),
            ("verify_token", "WHATSAPP_CLOUD_VERIFY_TOKEN"),  # Meta hub.verify_token shared secret
            ("webhook_host", "WHATSAPP_CLOUD_WEBHOOK_HOST"),
            ("webhook_port", "WHATSAPP_CLOUD_WEBHOOK_PORT", _INT),
            ("webhook_path", "WHATSAPP_CLOUD_WEBHOOK_PATH"),
            ("api_version", "WHATSAPP_CLOUD_API_VERSION"),
        ),
    ),
    _Home(Platform.WHATSAPP_CLOUD, "WHATSAPP_CLOUD_HOME_CHANNEL"),
    _Cred(Platform.SLACK, ("SLACK_BOT_TOKEN",), token="SLACK_BOT_TOKEN"),
    _slack_home,
    _Cred(
        Platform.SIGNAL, ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
        fixed=(
            ("http_url", "SIGNAL_HTTP_URL"),
            ("account", "SIGNAL_ACCOUNT"),
            ("ignore_stories", "SIGNAL_IGNORE_STORIES", "true", is_truthy_value),
        ),
    ),
    _Home(Platform.SIGNAL, "SIGNAL_HOME_CHANNEL"),
    _Cred(
        Platform.MATTERMOST, ("MATTERMOST_TOKEN",), token="MATTERMOST_TOKEN",
        warn_missing=("MATTERMOST_URL", "MATTERMOST_TOKEN set but MATTERMOST_URL is missing"),
        fixed=(("url", "MATTERMOST_URL"),),
    ),
    _Home(Platform.MATTERMOST, "MATTERMOST_HOME_CHANNEL"),
    _Cred(
        Platform.MATRIX, (("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"),), token="MATRIX_ACCESS_TOKEN",
        warn_missing=("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN/MATRIX_PASSWORD set but MATRIX_HOMESERVER is missing"),
        fixed=(("homeserver", "MATRIX_HOMESERVER"),),
        optional=(("user_id", "MATRIX_USER_ID"), ("password", "MATRIX_PASSWORD")),
        then=_matrix_e2ee,
    ),
    _Home(Platform.MATRIX, "MATRIX_HOME_ROOM"),
    _Cred(Platform.HOMEASSISTANT, ("HASS_TOKEN",), token="HASS_TOKEN", optional=(("url", "HASS_URL"),)),
    _Cred(
        Platform.EMAIL, ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        fixed=(("address", "EMAIL_ADDRESS"), ("imap_host", "EMAIL_IMAP_HOST"), ("smtp_host", "EMAIL_SMTP_HOST")),
    ),
    _Home(Platform.EMAIL, "EMAIL_HOME_ADDRESS"),
    _Cred(Platform.SMS, ("TWILIO_ACCOUNT_SID",), then=_sms_api_key),
    _Home(Platform.SMS, "SMS_HOME_CHANNEL"),
    _api_server,
    _webhook,
    _msgraph_webhook,
    _Cred(
        Platform.DINGTALK, ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        fixed=(("client_id", "DINGTALK_CLIENT_ID"), ("client_secret", "DINGTALK_CLIENT_SECRET")),
        home="DINGTALK_HOME_CHANNEL",
    ),
    _Cred(
        Platform.FEISHU, ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
        fixed=(
            ("app_id", "FEISHU_APP_ID"),
            ("app_secret", "FEISHU_APP_SECRET"),
            ("domain", "FEISHU_DOMAIN", "feishu"),
            ("connection_mode", "FEISHU_CONNECTION_MODE", "websocket"),
        ),
        optional=(("encrypt_key", "FEISHU_ENCRYPT_KEY"), ("verification_token", "FEISHU_VERIFICATION_TOKEN")),
        home="FEISHU_HOME_CHANNEL",
    ),
    _Cred(
        Platform.WECOM, ("WECOM_BOT_ID", "WECOM_SECRET"),
        fixed=(("bot_id", "WECOM_BOT_ID"), ("secret", "WECOM_SECRET")),
        optional=(("websocket_url", "WECOM_WEBSOCKET_URL"),),
        home="WECOM_HOME_CHANNEL",
    ),
    _Cred(
        Platform.WECOM_CALLBACK, ("WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"),
        fixed=(
            ("corp_id", "WECOM_CALLBACK_CORP_ID"),
            ("corp_secret", "WECOM_CALLBACK_CORP_SECRET"),
            ("agent_id", "WECOM_CALLBACK_AGENT_ID"),
            ("token", "WECOM_CALLBACK_TOKEN"),
            ("encoding_aes_key", "WECOM_CALLBACK_ENCODING_AES_KEY"),
            # No default: an unset WECOM_CALLBACK_HOST leaves extra.host falsy so the adapter's
            # dual-stack DEFAULT_HOST=None applies (binds IPv4 + IPv6; "0.0.0.0" was IPv4-only).
            ("host", "WECOM_CALLBACK_HOST"),
            ("port", "WECOM_CALLBACK_PORT", "", _int_or(8645)),
        ),
    ),
    # Weixin (personal WeChat via iLink Bot API)
    _Cred(
        Platform.WEIXIN, (("WEIXIN_TOKEN", "WEIXIN_ACCOUNT_ID"),), token="WEIXIN_TOKEN",
        optional=(("account_id", "WEIXIN_ACCOUNT_ID"),),
        optional_stripped=(
            ("base_url", "WEIXIN_BASE_URL", _strip_slash),
            ("cdn_base_url", "WEIXIN_CDN_BASE_URL", _strip_slash),
            ("dm_policy", "WEIXIN_DM_POLICY", str.lower),
            ("group_policy", "WEIXIN_GROUP_POLICY", str.lower),
            ("allow_from", "WEIXIN_ALLOWED_USERS"),
            ("group_allow_from", "WEIXIN_GROUP_ALLOWED_USERS"),
            ("split_multiline_messages", "WEIXIN_SPLIT_MULTILINE_MESSAGES"),
        ),
        home="WEIXIN_HOME_CHANNEL", home_strip=True,
    ),
    # BlueBubbles (iMessage). ``require_mention`` is always written: an unset env reads as "" → False.
    _Cred(
        Platform.BLUEBUBBLES, ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
        fixed=(
            ("server_url", "BLUEBUBBLES_SERVER_URL", "", _strip_slash),
            ("password", "BLUEBUBBLES_PASSWORD"),
            ("webhook_host", "BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.1"),
            ("webhook_port", "BLUEBUBBLES_WEBHOOK_PORT", "", _int_or(8645)),
            ("webhook_path", "BLUEBUBBLES_WEBHOOK_PATH", "/bluebubbles-webhook"),
            ("send_read_receipts", "BLUEBUBBLES_SEND_READ_RECEIPTS", "true", is_truthy_value),
            ("require_mention", "BLUEBUBBLES_REQUIRE_MENTION", "", _truthy_token),
        ),
        optional=(("mention_patterns", "BLUEBUBBLES_MENTION_PATTERNS", _mention_patterns),),
    ),
    _Home(Platform.BLUEBUBBLES, "BLUEBUBBLES_HOME_CHANNEL"),
    # QQ (Official Bot API v2)
    _Cred(
        Platform.QQBOT, (("QQ_APP_ID", "QQ_CLIENT_SECRET"),),
        optional=(("app_id", "QQ_APP_ID"), ("client_secret", "QQ_CLIENT_SECRET")),
        optional_stripped=(("allow_from", "QQ_ALLOWED_USERS"), ("group_allow_from", "QQ_GROUP_ALLOWED_USERS")),
        then=_qq_home,
    ),
    # Yuanbao — YUANBAO_APP_ID preferred over the legacy YUANBAO_APP_KEY
    _Cred(
        Platform.YUANBAO, (("YUANBAO_APP_ID", "YUANBAO_APP_KEY"), "YUANBAO_APP_SECRET"),
        fixed=(("app_id", ("YUANBAO_APP_ID", "YUANBAO_APP_KEY")), ("app_secret", "YUANBAO_APP_SECRET")),
        optional=(
            ("bot_id", "YUANBAO_BOT_ID"),
            ("ws_url", "YUANBAO_WS_URL"),
            ("api_domain", "YUANBAO_API_DOMAIN"),
            ("route_env", "YUANBAO_ROUTE_ENV"),
            ("dm_policy", "YUANBAO_DM_POLICY", lambda v: v.strip().lower()),
            ("dm_allow_from", "YUANBAO_DM_ALLOW_FROM"),
            ("group_policy", "YUANBAO_GROUP_POLICY", lambda v: v.strip().lower()),
            ("group_allow_from", "YUANBAO_GROUP_ALLOW_FROM"),
        ),
        home="YUANBAO_HOME_CHANNEL",
    ),
    _session_settings,
    _enable_plugin_platforms_from_env,
    _relay,
    _scrub_explicit_markers,
)


def _apply_env_overrides(config: GatewayConfig) -> None:
    """Apply environment variable overrides to *config* (see ``_ENV_STEPS``)."""
    for step in _ENV_STEPS:
        step(config)
