"""Per-platform display/verbosity configuration resolver.

``resolve_display_setting()`` is the single entry-point.  Resolution order
(first non-None wins):
    1. ``display.platforms.<platform>.<key>``  — explicit per-platform override
    2. ``display.<key>``                       — global user setting
    3. ``_PLATFORM_DEFAULTS[<platform>][<key>]``  — built-in platform default
    4. ``_GLOBAL_DEFAULTS[<key>]``              — built-in global default

Exception: ``display.streaming`` is CLI-only.  Gateway streaming follows the
top-level ``streaming`` config unless ``display.platforms.<platform>.streaming``
sets an explicit per-platform override.

Backward compatibility: ``display.tool_progress_overrides`` is still read as a
fallback for ``tool_progress`` when no ``display.platforms`` entry exists (a
config migration moves the old format into ``display.platforms``).
"""

from __future__ import annotations

from typing import Any

# Settings configurable per-platform, with global defaults.  Other display
# settings (compact, personality, skin, ...) are CLI-only.
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "show_reasoning": False,
    # Reasoning summary rendering: "code" (💭 **Reasoning:** + fenced block),
    # "blockquote" ("> " lines), "subtext" ("-# " Discord small grey text).
    "reasoning_style": "code",
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter; default on for back-compat, mobile
    # platforms opt down to final-answer-first.
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    # busy_input_mode=steer confirmation echo ("Steered into current run").
    # Disabling only suppresses the echo; the text still lands in the run.
    "busy_steer_ack_enabled": True,
    # Delete tool-progress / "⏳ Working" / status bubbles after the final
    # response on platforms that support deletion (e.g. Telegram).  Off by
    # default; progress is still shown live, only cleaned up after success so
    # the chat doesn't fill with stale breadcrumbs.  Failed runs leave bubbles
    # in place as breadcrumbs.
    "cleanup_progress": False,
    # Live working-state text on platforms whose typing indicator renders text
    # (Slack assistant status): "full"/true = verb + argument preview,
    # "verb" = verb only (keeps paths/commands out of shared channels),
    # "off"/false = static text.  Independent of tool_progress and free: the
    # existing typing refresh cadence just renders different text.
    "live_status": "full",
}

# Per-platform defaults tiered by capability:
#   HIGH    — message editing, personal/team use
#   MEDIUM  — editing, but often workspace/customer-facing
#   LOW     — no edit support: each progress message is permanent
#   MINIMAL — batch / non-interactive delivery
_TIER_HIGH = {
    "tool_progress": "all",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,  # follow global
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
}
_TIER_MEDIUM = {**_TIER_HIGH, "tool_progress": "new"}
_TIER_LOW = {
    **_TIER_HIGH,
    "tool_progress": "off",
    "streaming": False,
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
}
_TIER_MINIMAL = {**_TIER_LOW, "tool_preview_length": 0}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Telegram is usually a mobile inbox: quiet tool_progress and no busy-ack
    # iteration counter, but DO surface interim assistant commentary and
    # heartbeats so it doesn't look like "typing..." for 30 minutes.
    "telegram": {**_TIER_HIGH, "tool_progress": "off", "busy_ack_detail": False},
    # Discord's "-# " subtext reads as metadata, so reasoning defaults to it.
    "discord": {**_TIER_HIGH, "reasoning_style": "subtext"},

    # Slack: Bolt posts cannot be edited like CLI; "new"/"all" spam permanent lines.
    "slack": {
        **_TIER_MEDIUM,
        "tool_progress": "off",
        "long_running_notifications": False,
        "busy_ack_detail": False,
    },
    "mattermost": _TIER_MEDIUM,
    "matrix": _TIER_MEDIUM,
    "feishu": _TIER_MEDIUM,
    # Buzz (Nostr via buzz-cli) can edit in place but channels are shared
    # community spaces; without an entry it inherited the verbose globals.
    "buzz": _TIER_MEDIUM,

    "signal": _TIER_LOW,
    "whatsapp": _TIER_MEDIUM,  # Baileys bridge supports /edit
    # Cloud API supports editing but our adapter lacks edit_message; promote
    # to MEDIUM once it lands.
    "whatsapp_cloud": _TIER_LOW,
    # Photon and BlueBubbles are permanent-message iMessage inboxes (no edit);
    # without an entry Photon inherited the noisy "all" globals and narrated
    # on nearly every turn.
    "photon": _TIER_LOW,
    "bluebubbles": _TIER_LOW,
    "weixin": _TIER_LOW,
    # WeCom is non-editable but has a native streaming transport (msgtype
    # "stream") the consumer routes mid-stream content through; streaming on
    # gives the client a typing animation + cumulative updates instead of a
    # single one-shot markdown drop.
    "wecom": {**_TIER_LOW, "streaming": True},
    "wecom_callback": _TIER_LOW,
    "dingtalk": _TIER_LOW,

    "email": _TIER_MINIMAL,
    "sms": _TIER_MINIMAL,
    "webhook": _TIER_MINIMAL,
    "homeassistant": _TIER_MINIMAL,
    "api_server": {**_TIER_HIGH, "tool_preview_length": 0},
}

# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def resolve_display_setting(
    user_config: dict,
    platform_key: str,
    setting: str,
    fallback: Any = None,
) -> Any:
    """Resolve a display setting with per-platform override support.

    ``platform_key`` is the platform config key (``"telegram"``, ``"slack"``;
    see ``_platform_config_key`` in gateway/run.py).  Returns *fallback* when
    nothing is configured.
    """
    display_cfg = user_config.get("display") or {}

    # 1. Explicit per-platform override
    plat_overrides = (display_cfg.get("platforms") or {}).get(platform_key)
    if isinstance(plat_overrides, dict) and plat_overrides.get(setting) is not None:
        return _normalise(setting, plat_overrides[setting])

    # 1b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return _normalise(setting, legacy[platform_key])

    # 2. Global user setting.  display.streaming controls only CLI terminal
    # streaming; gateway streaming follows the top-level config + platform overrides.
    if setting != "streaming" and display_cfg.get(setting) is not None:
        return _normalise(setting, display_cfg[setting])

    # 3. Built-in platform default, 4. built-in global default
    val = _PLATFORM_DEFAULTS.get(platform_key, {}).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


# ---------------------------------------------------------------------------
# Normalisation of YAML quirks (bare ``off`` → False in YAML 1.1, etc.)
# ---------------------------------------------------------------------------

_TRUTHY = {"true", "1", "yes", "on"}
_FALSY = {"false", "0", "no"}


def _norm_tool_progress(value: Any) -> str:
    if value is False:
        return "off"
    if value is True:
        return "all"
    val = str(value).strip().lower()
    if val in _FALSY:
        return "off"
    if val in _TRUTHY:
        return "all"
    return val if val in {"off", "new", "all", "verbose", "log"} else "all"


def _norm_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY | {"raw", "verbose"}
    return bool(value)


def _norm_long_running(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "generic":
        return "generic"
    return _norm_bool(value)


def _norm_cleanup_progress(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in _TRUTHY
    return bool(value)


def _norm_live_status(value: Any) -> str:
    """Tri-state: "full" (verb + preview), "verb" (verb only), "off"."""
    if value is True:
        return "full"
    if value is False:
        return "off"
    val = str(value).strip().lower()
    if val in _TRUTHY | {"all"}:
        return "full"
    if val in _FALSY:
        return "off"
    return val if val in {"full", "verb", "off"} else "full"


def _norm_choice(choices: tuple[str, ...]) -> Any:
    def norm(value: Any) -> str:
        val = str(value).lower()
        return val if val in choices else choices[0]

    return norm


def _norm_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_NORMALISERS: dict[str, Any] = {
    "tool_progress": _norm_tool_progress,
    "show_reasoning": _norm_bool,
    "streaming": _norm_bool,
    "interim_assistant_messages": _norm_bool,
    "long_running_notifications": _norm_long_running,
    "busy_ack_detail": _norm_bool,
    "busy_steer_ack_enabled": _norm_bool,
    "thinking_progress": _norm_bool,
    "cleanup_progress": _norm_cleanup_progress,
    "live_status": _norm_live_status,
    "tool_progress_grouping": _norm_choice(("accumulate", "separate")),
    "reasoning_style": _norm_choice(("code", "blockquote", "subtext")),
    "tool_preview_length": _norm_int,
}


def _normalise(setting: str, value: Any) -> Any:
    """Normalise a user-supplied value for *setting*; unknown settings pass through."""
    norm = _NORMALISERS.get(setting)
    return norm(value) if norm else value
