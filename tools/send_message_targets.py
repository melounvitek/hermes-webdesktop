"""Target parsing and resolution for send_message (platform:chat_id[:thread] → ids)."""

import logging
import re

logger = logging.getLogger("tools.send_message_tool")

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_FEISHU_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")
# Slack conversation IDs: C (public), G (private/group), D (DM); uppercase alnum, 9+ chars.
# User IDs (U...) become ``user:U...`` and are opened as D... conversations before
# chat.postMessage (posting straight to a U/W id fails); ``@handle`` -> ``user_name:...``
# is resolved through users.list first.
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
# Platforms addressing recipients by E.164 phone number ("+1555..."): without this the
# '+' fails the isdigit() rule and falls through to channel-name resolution, which cannot
# resolve a raw number. The '+' is kept because the adapters expect the E.164 form.
_PHONE_PLATFORMS = frozenset({"photon", "signal", "sms", "whatsapp"})
_E164_TARGET_RE = re.compile(r"^\s*\+(\d{7,15})\s*$")
# Photon DM chat GUID (mirrors _DM_CHAT_GUID_RE in the photon adapter).
_PHOTON_DM_GUID_RE = re.compile(r"^any;-;\+\d{6,}$")
# WhatsApp JIDs (groups <digits>@g.us, users <phone>@s.whatsapp.net, linked ids @lid,
# broadcast/newsletter): native targets the bridge accepts verbatim — never home-channel.
_WHATSAPP_JID_RE = re.compile(
    r"^\s*[\w-]+@(?:g\.us|s\.whatsapp\.net|lid|broadcast|newsletter)\s*$", re.IGNORECASE)
# Buzz channels/DMs are native UUIDs: explicit targets, never the home channel.
_BUZZ_UUID_RE = re.compile(
    r"^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$", re.IGNORECASE)
# A valid address is an explicit email target, not a channel name to resolve.
_EMAIL_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")
# Home-channel env var exceptions to "<PLATFORM>_HOME_CHANNEL" (email reads
# EMAIL_HOME_ADDRESS, see gateway/config.py) so the error guidance is actionable.
_HOME_CHANNEL_ENV_OVERRIDES = {"email": "EMAIL_HOME_ADDRESS"}

_UNRESOLVED = object()  # sentinel: stop parsing, target is NOT explicit (skip generic rules)


# Per-platform explicit-target parsers: target_ref -> (chat_id, thread_id), None to fall
# through to the generic rules in _parse_target_ref, or _UNRESOLVED. Order inside each
# parser matters (e.g. Slack thread form before bare conversation id).
def _parse_regex_groups(regex, *, thread_group=True):
    """Explicit when ``regex`` fully matches: chat_id = group 1, thread = group 2 (or None)."""
    def parse(ref):
        match = regex.fullmatch(ref)
        if not match:
            return None
        return match.group(1), (match.group(2) if thread_group else None)
    return parse


def _parse_regex_stripped(regex):
    """Explicit when ``regex`` fully matches; returns the stripped ref verbatim."""
    def parse(ref):
        return (ref.strip(), None) if regex.fullmatch(ref) else None
    return parse


def _parse_telegram(ref):
    # "<chat_id>[:<topic_id>]" or an @username (usernames must not be force-int'd).
    parsed = _parse_regex_groups(_TELEGRAM_TOPIC_TARGET_RE)(ref)
    if parsed:
        return parsed
    from plugins.platforms.telegram.telegram_ids import parse_telegram_username_target

    username = parse_telegram_username_target(ref)
    return (username, None) if username else None


# (regex, chat_id template, thread comes from group 2) — thread form before bare id.
_SLACK_FORMS = (
    (_SLACK_THREAD_TARGET_RE, "{}", True),
    (_SLACK_TARGET_RE, "{}", False),
    (_SLACK_USER_ID_RE, "user:{}", False),
    (_SLACK_MENTION_RE, "user:{}", False),
    (_SLACK_USER_NAME_RE, "user_name:{}", False),
)


def _parse_slack(ref):
    for regex, template, has_thread in _SLACK_FORMS:
        match = regex.fullmatch(ref)
        if match:
            return template.format(match.group(1)), (match.group(2) if has_thread else None)
    return None


def _parse_matrix(ref):
    # "<room>:$<event_id>" addresses a thread (rfind: room ids contain ':'). Bare "!room" /
    # "@user" are explicit too, but via the generic rule so the numeric check keeps precedence.
    trimmed = ref.strip()
    split_idx = trimmed.rfind(":$")
    if split_idx > 0:
        return trimmed[:split_idx], trimmed[split_idx + 1 :]
    return None


def _parse_yuanbao(ref):
    # "group:<code>" / "direct:<id>"; a bare number is a group code, never a generic
    # numeric chat id (yuanbao never falls through to the generic rules).
    match = _YUANBAO_TARGET_RE.fullmatch(ref)
    if match:
        return match.group(1), None
    if ref.strip().isdigit():
        return f"group:{ref.strip()}", None
    return _UNRESOLVED


def _parse_nonempty(ref):
    # ntfy topics and WeCom ids (wr/wc groups, wo/bare users — the adapter picks the
    # send command) are explicit whenever non-empty.
    stripped = ref.strip()
    return (stripped, None) if stripped else None


def _parse_signal(ref):
    # "group:<id>" is a native group target; an empty id is not explicit.
    stripped = ref.strip()
    if stripped.startswith("group:"):
        group_id = stripped[len("group:"):].strip()
        return (f"group:{group_id}", None) if group_id else _UNRESOLVED
    return None


_PLATFORM_PARSERS = {
    "telegram": _parse_telegram,
    "feishu": _parse_regex_groups(_FEISHU_TARGET_RE),
    "discord": _parse_regex_groups(_NUMERIC_TOPIC_RE),  # "<channel>[:<thread>]" snowflakes
    "slack": _parse_slack,
    "matrix": _parse_matrix,
    "weixin": _parse_regex_groups(_WEIXIN_TARGET_RE, thread_group=False),
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

    Platform parser first, then the shared rules in this order: E.164 phone numbers
    (keeping the '+' the signal/sms/whatsapp adapters expect), bare numeric ids, Matrix
    ``!room``/``@user``, XMPP JIDs. Anything else goes to channel-directory resolution.
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

    Channel-directory IDs are trusted. Plugin platforms must explicitly parse native
    target syntax; for the model-facing send tool (the default) an unresolvable target
    is an error the model can read and pick a listed target instead.

    ``pass_unresolved_references=True`` is for callers with no model in the loop (cron
    delivering a stored job's output, react/unreact on platform-native message ids): an
    unresolvable target on a built-in platform, or on a plugin platform that declares no
    parser, is handed to the adapter exactly as written and the adapter decides. A plugin
    platform that DOES declare a parser stays strict for every caller.

    The optional validator has the final say over parser-normalized, directory-resolved,
    and passed-through IDs alike.
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
            logger.debug("Plugin target validator failed for %s", platform_name, exc_info=True)
            return f"Target validator failed for platform '{platform_name}'"
        if verdict is True:
            return None
        if isinstance(verdict, str) and verdict:
            return f"Invalid target '{target_ref}' on {platform_name}: {verdict}"
        return f"Invalid target '{target_ref}' on {platform_name}"

    def _validated(chat_id, thread_id):
        error = _validate(chat_id)
        return (None, None, error) if error else (chat_id, thread_id, None)

    if entry is not None and entry.parse_target_ref_fn is not None:
        try:
            parsed = entry.parse_target_ref_fn(target_ref)
        except Exception:
            logger.debug("Plugin target parser failed for %s", platform_name, exc_info=True)
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
                    None, None,
                    f"Target parser for platform '{platform_name}' returned an invalid result",
                )
            return _validated(*parsed)

    parsed_chat_id, parsed_thread_id, explicit = _parse_target_ref(platform_name, target_ref)
    if explicit and parsed_chat_id is not None:
        return _validated(parsed_chat_id, parsed_thread_id)

    resolution_failed = False
    try:
        from gateway.channel_directory import resolve_channel_name

        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        resolved = None
        resolution_failed = True
    if resolved:
        parsed_chat_id, parsed_thread_id, _ = _parse_target_ref(platform_name, resolved)
        return _validated(parsed_chat_id or resolved, parsed_thread_id)

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
            None, None,
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
