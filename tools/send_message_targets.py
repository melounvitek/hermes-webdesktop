"""Target parsing and resolution for send_message (platform:chat_id[:thread] → ids)."""

import logging
import re

logger = logging.getLogger("tools.send_message_tool")

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_FEISHU_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")
# Slack conversation IDs: C (public channel), G (private/group channel), D (DM).
# Must be uppercase alphanumeric, 9+ chars. User IDs (U...) are parsed as
# explicit user targets (``user:U...``) and are converted to D... conversations
# via conversations.open before chat.postMessage — posting directly to a U/W
# ID fails because the API requires a conversation ID. ``@handle`` targets are
# resolved through users.list first (``user_name:...``).
_SLACK_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,})\s*$")
_SLACK_USER_ID_RE = re.compile(r"^\s*(U[A-Z0-9]{8,})\s*$")
_SLACK_USER_NAME_RE = re.compile(r"^\s*@([A-Za-z0-9._-]{1,80})\s*$")
_SLACK_MENTION_RE = re.compile(r"^\s*<@(U[A-Z0-9]{8,})(?:\|[^>]+)?>\s*$")
# Session-derived Slack thread targets use "<conversation_id>:<thread_ts>".
_SLACK_THREAD_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,}):([^\s:]+)\s*$")
_WEIXIN_TARGET_RE = re.compile(r"^\s*((?:wxid|gh|v\d+|wm|wb)_[A-Za-z0-9_-]+|[A-Za-z0-9._-]+@chatroom|filehelper)\s*$")
_YUANBAO_TARGET_RE = re.compile(r"^\s*((?:group|direct):[^:]+)\s*$")
# Discord snowflake IDs are numeric, same regex pattern as Telegram topic targets.
_NUMERIC_TOPIC_RE = _TELEGRAM_TOPIC_TARGET_RE
# Platforms that address recipients by phone number and accept E.164 format
# (with a leading '+'). Without this, "+15551234567" fails the isdigit() check
# below and falls through to channel-name resolution, which has no way to
# resolve a raw phone number. Keeping the '+' preserves the E.164 form that
# downstream adapters (signal, etc.) expect.
_PHONE_PLATFORMS = frozenset({"photon", "signal", "sms", "whatsapp"})
_E164_TARGET_RE = re.compile(r"^\s*\+(\d{7,15})\s*$")
# Photon DM chat GUID (mirrors _DM_CHAT_GUID_RE in the photon adapter).
_PHOTON_DM_GUID_RE = re.compile(r"^any;-;\+\d{6,}$")
# WhatsApp JIDs: group chats (<digits>@g.us), individual users
# (<phone>@s.whatsapp.net), linked identities (<id>@lid), and broadcast /
# newsletter chats. These are explicit native targets the bridge accepts
# verbatim — they must NOT fall through to home-channel resolution.
_WHATSAPP_JID_RE = re.compile(
    r"^\s*[\w-]+@(?:g\.us|s\.whatsapp\.net|lid|broadcast|newsletter)\s*$",
    re.IGNORECASE,
)
# Buzz channels and DMs use native UUID identifiers. They are explicit
# targets and must never substitute the configured home channel.
_BUZZ_UUID_RE = re.compile(
    r"^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$",
    re.IGNORECASE,
)
# Email addresses — a valid email like "user@domain.com" should be treated as
# an explicit target for the email platform, not fall through to channel-name
# resolution which has no way to resolve a raw address.
_EMAIL_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")
# Most platforms read their home channel from "<PLATFORM>_HOME_CHANNEL", but a
# few diverge. Email reads EMAIL_HOME_ADDRESS (see gateway/config.py), so the
# generic "<PLATFORM>_HOME_CHANNEL" hint would point users at a variable that is
# never read. Map the exceptions so the error guidance is actually actionable.
_HOME_CHANNEL_ENV_OVERRIDES = {"email": "EMAIL_HOME_ADDRESS"}


# Per-platform explicit-target parsers: target_ref -> (chat_id, thread_id) or
# None to fall through to the generic rules below. Order inside each parser
# matters (e.g. Slack thread form before bare conversation id).
def _parse_telegram(ref):
    match = _TELEGRAM_TOPIC_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), match.group(2)
    from plugins.platforms.telegram.telegram_ids import parse_telegram_username_target

    username = parse_telegram_username_target(ref)
    return (username, None) if username else None


def _parse_topic(regex):
    def parse(ref):
        match = regex.fullmatch(ref)
        return (match.group(1), match.group(2)) if match else None
    return parse


def _parse_slack(ref):
    match = _SLACK_THREAD_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), match.group(2)
    match = _SLACK_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), None
    match = _SLACK_USER_ID_RE.fullmatch(ref) or _SLACK_MENTION_RE.fullmatch(ref)
    if match:
        return f"user:{match.group(1)}", None
    match = _SLACK_USER_NAME_RE.fullmatch(ref)
    return (f"user_name:{match.group(1)}", None) if match else None


def _parse_matrix(ref):
    # "<room>:$<event_id>" addresses a thread; bare "!room" / "@user" handled below.
    trimmed = ref.strip()
    split_idx = trimmed.rfind(":$")
    if split_idx > 0:
        return trimmed[:split_idx], trimmed[split_idx + 1 :]
    if trimmed.startswith(("!", "@")):
        return None  # deferred: numeric check must run first (generic rule)
    return None


def _parse_regex_stripped(regex):
    """Explicit when ``regex`` fully matches; returns the stripped ref verbatim."""
    def parse(ref):
        return (ref.strip(), None) if regex.fullmatch(ref) else None
    return parse


def _parse_group1(regex):
    def parse(ref):
        match = regex.fullmatch(ref)
        return (match.group(1), None) if match else None
    return parse


def _parse_yuanbao(ref):
    match = _YUANBAO_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), None
    if ref.strip().isdigit():
        return f"group:{ref.strip()}", None
    return _UNRESOLVED  # yuanbao never falls through to the generic rules


def _parse_nonempty(ref):
    # ntfy topics and WeCom ids (wr/wc groups, wo/bare users — the adapter
    # picks the send command) are explicit whenever non-empty.
    stripped = ref.strip()
    return (stripped, None) if stripped else None


def _parse_signal(ref):
    stripped = ref.strip()
    if stripped.startswith("group:"):
        group_id = stripped[len("group:"):].strip()
        return (f"group:{group_id}", None) if group_id else _UNRESOLVED
    return None


_UNRESOLVED = object()  # sentinel: stop parsing, target is not explicit

_PLATFORM_PARSERS = {
    "telegram": _parse_telegram,
    "feishu": _parse_topic(_FEISHU_TARGET_RE),
    "discord": _parse_topic(_NUMERIC_TOPIC_RE),
    "slack": _parse_slack,
    "matrix": _parse_matrix,
    "weixin": _parse_group1(_WEIXIN_TARGET_RE),
    "yuanbao": _parse_yuanbao,
    "ntfy": _parse_nonempty,
    "email": _parse_regex_stripped(_EMAIL_TARGET_RE),
    # Native WhatsApp JIDs pass through verbatim; E.164 numbers use the phone rule.
    "whatsapp": _parse_regex_stripped(_WHATSAPP_JID_RE),
    "buzz": _parse_regex_stripped(_BUZZ_UUID_RE),
    "signal": _parse_signal,
    "wecom": _parse_nonempty,
    # Photon DM GUIDs are adapter-native ids (mirrors the react handler).
    "photon": _parse_regex_stripped(_PHOTON_DM_GUID_RE),
}


def _parse_target_ref(platform_name: str, target_ref: str):
    """Parse a tool target into (chat_id, thread_id, explicit).

    Platform parser first, then the shared rules: E.164 phone numbers (with
    the '+' the signal/sms/whatsapp adapters expect), bare numeric ids,
    Matrix ``!room``/``@user`` and XMPP JIDs. Anything else is not explicit
    and goes to channel-directory resolution.
    """
    parser = _PLATFORM_PARSERS.get(platform_name)
    if parser is not None:
        parsed = parser(target_ref)
        if parsed is _UNRESOLVED:
            return None, None, False
        if parsed is not None:
            return parsed[0], parsed[1], True
    if platform_name in _PHONE_PLATFORMS and _E164_TARGET_RE.fullmatch(target_ref):
        return target_ref.strip(), None, True
    if target_ref.lstrip("-").isdigit():
        return target_ref, None, True
    if platform_name == "matrix" and target_ref.startswith(("!", "@")):
        return target_ref, None, True
    if platform_name == "xmpp" and "@" in target_ref:
        return target_ref, None, True
    return None, None, False


def resolve_send_target(
    platform_name: str, target_ref: str, *, pass_unresolved_references: bool = False
) -> tuple[str | None, str | None, str | None]:
    """Resolve one send target the same way for every caller (model tool, CLI, cron).

    Channel-directory IDs are trusted. Plugin platforms must explicitly parse
    native target syntax; for the model-facing send tool (the default), a
    target that can't be resolved is an error — the model can read the error
    and pick a listed target instead.

    ``pass_unresolved_references=True`` restores the old pass-through behavior for
    callers that have no model in the loop (cron delivering a stored job's
    output, react/unreact on platform-native message ids): if the target
    can't be resolved and the platform is built in, or is a plugin platform
    that declares no parser, the string is handed to the adapter exactly as
    written and the adapter decides whether it's valid. A plugin platform
    that DOES declare a parser stays strict for every caller — its parser is
    the authority on native syntax.

    The optional validator has the final say over parser-normalized,
    directory-resolved, and passed-through IDs alike.
    """
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry

    entry = platform_registry.get(platform_name)

    def _validate(candidate: str) -> str | None:
        if entry is None or entry.validate_target_ref_fn is None:
            return None
        try:
            verdict = entry.validate_target_ref_fn(candidate)
        except Exception:
            logger.debug(
                "Plugin target validator failed for %s", platform_name, exc_info=True
            )
            return f"Target validator failed for platform '{platform_name}'"
        if verdict is True:
            return None
        if isinstance(verdict, str) and verdict:
            return f"Invalid target '{target_ref}' on {platform_name}: {verdict}"
        return f"Invalid target '{target_ref}' on {platform_name}"

    if entry is not None and entry.parse_target_ref_fn is not None:
        try:
            parsed = entry.parse_target_ref_fn(target_ref)
        except Exception:
            logger.debug(
                "Plugin target parser failed for %s", platform_name, exc_info=True
            )
            return None, None, f"Target parser failed for platform '{platform_name}'"
        if parsed is not None:
            if (
                not isinstance(parsed, tuple)
                or len(parsed) != 2
                or not isinstance(parsed[0], str)
                or not parsed[0]
                or (parsed[1] is not None and not isinstance(parsed[1], str))
            ):
                return (
                    None,
                    None,
                    f"Target parser for platform '{platform_name}' returned an invalid result",
                )
            parsed_chat_id, parsed_thread_id = parsed
            error = _validate(parsed_chat_id)
            return (None, None, error) if error else (
                parsed_chat_id,
                parsed_thread_id,
                None,
            )

    parsed_chat_id, parsed_thread_id, explicit = _parse_target_ref(
        platform_name, target_ref
    )
    if explicit and parsed_chat_id is not None:
        error = _validate(parsed_chat_id)
        return (None, None, error) if error else (
            parsed_chat_id,
            parsed_thread_id,
            None,
        )

    resolution_failed = False
    try:
        from gateway.channel_directory import resolve_channel_name

        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        resolved = None
        resolution_failed = True
    if resolved:
        parsed_chat_id, parsed_thread_id, _ = _parse_target_ref(
            platform_name, resolved
        )
        chat_id = parsed_chat_id or resolved
        error = _validate(chat_id)
        return (None, None, error) if error else (
            chat_id,
            parsed_thread_id,
            None,
        )

    is_builtin = platform_name in {member.value for member in Platform}
    if entry is None and not is_builtin:
        return None, None, f"Unknown or unregistered plugin platform: {platform_name}"

    def _pass_through_unresolved():
        """Hand the raw target to the adapter unchanged (it validates)."""
        error = _validate(target_ref)
        if error:
            return None, None, error
        logger.debug(
            "Handing unresolved target '%s' to the %s adapter unchanged "
            "(the adapter validates it)",
            target_ref, platform_name,
        )
        return target_ref, None, None

    if entry is not None and entry.source == "plugin" and not is_builtin:
        if pass_unresolved_references and entry.parse_target_ref_fn is None:
            return _pass_through_unresolved()
        return (
            None,
            None,
            f"Could not resolve '{target_ref}' on {platform_name}. "
            "The plugin parser did not recognize it and no channel-directory entry matched.",
        )
    if pass_unresolved_references:
        return _pass_through_unresolved()
    hint = (
        "Try using a numeric channel ID instead."
        if resolution_failed
        else "Use send_message(action='list') to see available targets."
    )
    return None, None, f"Could not resolve '{target_ref}' on {platform_name}. {hint}"
