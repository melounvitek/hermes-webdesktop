"""Discord server introspection and management tool (REST API + bot token).

Only in the hermes-discord toolset. The model-visible schema is filtered by two
gates: (1) privileged intents from GET /applications/@me — actions needing an
intent the bot lacks (search_members / member_info → GUILD_MEMBERS) are hidden,
and fetch_messages/list_pins are annotated when MESSAGE_CONTENT is missing;
(2) the ``discord.server_actions`` config allowlist (comma string or YAML list;
empty/unset = all). Per-guild permissions are NOT pre-checked — a call-time 403
is mapped to actionable guidance by :func:`_enrich_403`.
"""

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
_DISCORD_RESPONSE_BODY_MAX_BYTES = 4 * 1024 * 1024
_DISCORD_ERROR_BODY_MAX_BYTES = 64 * 1024

# Application flag bits (from GET /applications/@me → "flags").
# Source: https://discord.com/developers/docs/resources/application#application-object-application-flags
_FLAG_GATEWAY_GUILD_MEMBERS = 1 << 14
_FLAG_GATEWAY_GUILD_MEMBERS_LIMITED = 1 << 15
_FLAG_GATEWAY_MESSAGE_CONTENT = 1 << 18
_FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED = 1 << 19

class DiscordAPIError(Exception):
    """Raised when a Discord API call fails."""
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Discord API error {status}: {body}")


def _read_limited_response_body(source: Any, limit: int, *, label: str) -> bytes:
    body = source.read(limit + 1)
    if len(body) > limit:
        raise DiscordAPIError(502, f"Discord API {label} exceeded {limit} bytes.")
    return body


def _get_bot_token() -> Optional[str]:
    """Resolve the Discord bot token under the active profile secret scope."""
    return (get_secret("DISCORD_BOT_TOKEN", "") or "").strip() or None


def _discord_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    """Make a request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            response_body = _read_limited_response_body(
                resp,
                _DISCORD_RESPONSE_BODY_MAX_BYTES,
                label="response body",
            )
            return json.loads(response_body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = _read_limited_response_body(
                e, _DISCORD_ERROR_BODY_MAX_BYTES, label="error body"
            ).decode("utf-8", errors="replace")
        except DiscordAPIError as too_large:
            error_body = too_large.body
        except Exception:
            pass
        raise DiscordAPIError(e.code, error_body) from e


_CHANNEL_TYPE_NAMES = {
    0: "text",
    2: "voice",
    4: "category",
    5: "announcement",
    10: "announcement_thread",
    11: "public_thread",
    12: "private_thread",
    13: "stage",
    15: "forum",
    16: "media",
}


def _channel_type_name(type_id: int) -> str:
    return _CHANNEL_TYPE_NAMES.get(type_id, f"unknown({type_id})")


# ── capability detection (application intents) ──────────────────────────────

# Per-token in-process cache: the app/me endpoint is hit at most once per process.
_capability_cache: Dict[str, Dict[str, Any]] = {}

# Privileged intents change only when the user flips them in the Developer
# Portal, so 24h disk staleness is harmless: a hidden action re-appears on the
# next refresh; an exposed action the bot lost fails at call time with an
# enriched 403.
_CAPABILITY_DISK_TTL_SECONDS = 24 * 3600

# One background detection per (process, token) at most.
_capability_bg_started: set = set()
_capability_bg_lock = threading.Lock()

# Permissive default: all actions exposed, call-time 403s mapped to guidance.
_PERMISSIVE_CAPS = {"has_members_intent": True, "has_message_content": True, "detected": False}


def _capability_disk_cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "discord_capabilities.json"


def _token_cache_key(token: str) -> str:
    """Stable non-reversible cache key for a bot token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _load_caps_from_disk(token: str) -> Optional[Dict[str, Any]]:
    """Return fresh disk-cached capabilities for *token*, or None."""
    try:
        path = _capability_disk_cache_path()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(_token_cache_key(token))
        if not isinstance(entry, dict):
            return None
        if time.time() - float(entry.get("ts", 0)) > _CAPABILITY_DISK_TTL_SECONDS:
            return None
        caps = entry.get("caps")
        if isinstance(caps, dict) and "has_members_intent" in caps:
            return caps
    except Exception:
        pass
    return None


def _save_caps_to_disk(token: str, caps: Dict[str, Any]) -> None:
    try:
        path = _capability_disk_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data[_token_cache_key(token)] = {"caps": caps, "ts": time.time()}
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        logger.debug("discord capability disk-cache write failed", exc_info=True)


def _detect_capabilities_nonblocking(token: str) -> Dict[str, Any]:
    """Non-blocking capability lookup for schema builds.

    Order: in-process cache → fresh disk cache → permissive default plus a
    fire-and-forget background detection that fills the disk cache for the
    NEXT process. The blocking HTTPS call (~2-5s) used to sit on the first-token
    critical path of every cold process with DISCORD_BOT_TOKEN set.
    """
    cached = _capability_cache.get(token)
    if cached is not None:
        return cached

    disk = _load_caps_from_disk(token)
    if disk is not None:
        _capability_cache[token] = disk
        return disk

    # Cold start — pin the permissive default for THIS process: schemas must not
    # change between agent inits within a live process or the per-conversation
    # prompt cache breaks.
    caps_default = dict(_PERMISSIVE_CAPS)
    _capability_cache[token] = caps_default

    with _capability_bg_lock:
        if token not in _capability_bg_started:
            _capability_bg_started.add(token)

            def _bg_detect() -> None:
                try:
                    caps = _fetch_capabilities(token)
                    if caps.get("detected"):
                        _save_caps_to_disk(token, caps)
                except Exception:
                    logger.debug("background discord capability detection failed", exc_info=True)

            threading.Thread(
                target=_bg_detect, name="discord-caps-detect", daemon=True
            ).start()

    return caps_default


def _fetch_capabilities(token: str) -> Dict[str, Any]:
    """Fetch capabilities from GET /applications/@me. Pure network fetch — never
    touches the in-process cache (background detection must not mutate schemas
    mid-process). Detection failure is permissive."""
    caps: Dict[str, Any] = dict(_PERMISSIVE_CAPS)
    try:
        app = _discord_request("GET", "/applications/@me", token, timeout=5)
        flags = int(app.get("flags", 0) or 0)
        caps["has_members_intent"] = bool(
            flags & (_FLAG_GATEWAY_GUILD_MEMBERS | _FLAG_GATEWAY_GUILD_MEMBERS_LIMITED)
        )
        caps["has_message_content"] = bool(
            flags & (_FLAG_GATEWAY_MESSAGE_CONTENT | _FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED)
        )
        caps["detected"] = True
    except Exception as exc:  # nosec — detection is best-effort
        logger.info(
            "Discord capability detection failed (%s); exposing all actions.", exc,
        )

    return caps


def _detect_capabilities(token: str, *, force: bool = False) -> Dict[str, Any]:
    """Blocking detection via GET /applications/@me, cached per token (the
    warm-up path; schema builds use the non-blocking variant). ``force`` re-fetches."""
    if token in _capability_cache and not force:
        return _capability_cache[token]
    caps = _fetch_capabilities(token)
    _capability_cache[token] = caps
    return caps


def _reset_capability_cache() -> None:
    """Test hook: clear the detection cache."""
    global _capability_cache, _capability_bg_started
    _capability_cache = {}
    with _capability_bg_lock:
        _capability_bg_started = set()


# ── action implementations ───────────────────────────────────────────────────


def _listing(key: str, items: List[Dict[str, Any]]) -> str:
    return json.dumps({key: items, "count": len(items)})


def _ok(message: str) -> str:
    return json.dumps({"success": True, "message": message})


def _member_summary(m: Dict[str, Any], *, full: bool) -> Dict[str, Any]:
    user = m.get("user", {})
    out = {
        "user_id": user.get("id"),
        "username": user.get("username"),
        "display_name": user.get("global_name"),
        "nickname": m.get("nick"),
    }
    if full:
        out["avatar"] = user.get("avatar")
    out["bot"] = user.get("bot", False)
    out["roles"] = m.get("roles", [])
    if full:
        out["joined_at"] = m.get("joined_at")
        out["premium_since"] = m.get("premium_since")
    return out


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_guilds(token: str, **_kwargs: Any) -> str:
    """List all guilds the bot is a member of."""
    guilds = _discord_request("GET", "/users/@me/guilds", token)
    return _listing("guilds", [
        {
            "id": g["id"],
            "name": g["name"],
            "icon": g.get("icon"),
            "owner": g.get("owner", False),
            "permissions": g.get("permissions"),
        }
        for g in guilds
    ])


def _server_info(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get detailed information about a guild."""
    g = _discord_request("GET", f"/guilds/{guild_id}", token, params={"with_counts": "true"})
    return json.dumps({
        "id": g["id"],
        "name": g["name"],
        "description": g.get("description"),
        "icon": g.get("icon"),
        "owner_id": g.get("owner_id"),
        "member_count": g.get("approximate_member_count"),
        "online_count": g.get("approximate_presence_count"),
        "features": g.get("features", []),
        "premium_tier": g.get("premium_tier"),
        "premium_subscription_count": g.get("premium_subscription_count"),
        "verification_level": g.get("verification_level"),
    })


def _list_channels(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all channels in a guild, organized by category."""
    channels = _discord_request("GET", f"/guilds/{guild_id}/channels", token)
    categories: Dict[Optional[str], Dict[str, Any]] = {
        ch["id"]: {"id": ch["id"], "name": ch["name"], "position": ch.get("position", 0), "channels": []}
        for ch in channels
        if ch["type"] == 4  # category
    }
    uncategorized: List[Dict[str, Any]] = []
    for ch in channels:
        if ch["type"] == 4:
            continue
        entry = {
            "id": ch["id"],
            "name": ch.get("name", ""),
            "type": _channel_type_name(ch["type"]),
            "position": ch.get("position", 0),
            "topic": ch.get("topic"),
            "nsfw": ch.get("nsfw", False),
        }
        parent = ch.get("parent_id")
        if parent and parent in categories:
            categories[parent]["channels"].append(entry)
        else:
            uncategorized.append(entry)

    sorted_cats = sorted(categories.values(), key=lambda c: c["position"])
    for cat in sorted_cats:
        cat["channels"].sort(key=lambda c: c["position"])
    uncategorized.sort(key=lambda c: c["position"])

    result: List[Dict[str, Any]] = []
    if uncategorized:
        result.append({"category": None, "channels": uncategorized})
    result.extend(
        {"category": {"id": cat["id"], "name": cat["name"]}, "channels": cat["channels"]}
        for cat in sorted_cats
    )
    total = sum(len(group["channels"]) for group in result)
    return json.dumps({"channel_groups": result, "total_channels": total})


def _channel_info(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Get detailed info about a specific channel."""
    ch = _discord_request("GET", f"/channels/{channel_id}", token)
    return json.dumps({
        "id": ch["id"],
        "name": ch.get("name"),
        "type": _channel_type_name(ch["type"]),
        "guild_id": ch.get("guild_id"),
        "topic": ch.get("topic"),
        "nsfw": ch.get("nsfw", False),
        "position": ch.get("position"),
        "parent_id": ch.get("parent_id"),
        "rate_limit_per_user": ch.get("rate_limit_per_user", 0),
        "last_message_id": ch.get("last_message_id"),
    })


def _list_roles(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all roles in a guild."""
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    return _listing("roles", [
        {
            "id": r["id"],
            "name": r["name"],
            "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
            "position": r.get("position", 0),
            "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False),
            "member_count": r.get("member_count"),
            "hoist": r.get("hoist", False),
        }
        for r in sorted(roles, key=lambda r: r.get("position", 0), reverse=True)
    ])


def _member_info(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Get info about a specific guild member."""
    m = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    return json.dumps(_member_summary(m, full=True))


def _search_members(token: str, guild_id: str, query: str, limit: int = 20, **_kwargs: Any) -> str:
    """Search for guild members by name."""
    params = {"query": query, "limit": str(min(_int_or(limit, 20), 100))}
    members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params=params)
    return _listing("members", [_member_summary(m, full=False) for m in members])


def _fetch_messages(
    token: str, channel_id: str, limit: int = 50,
    before: Optional[str] = None, after: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Fetch recent messages from a channel."""
    params: Dict[str, str] = {"limit": str(min(_int_or(limit, 50), 100))}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    messages = _discord_request("GET", f"/channels/{channel_id}/messages", token, params=params)
    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", ""),
            "author": {
                "id": author.get("id"),
                "username": author.get("username"),
                "display_name": author.get("global_name"),
                "bot": author.get("bot", False),
            },
            "timestamp": msg.get("timestamp"),
            "edited_timestamp": msg.get("edited_timestamp"),
            "attachments": [
                {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
                for a in msg.get("attachments", [])
            ],
            "reactions": [
                {"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)}
                for r in msg.get("reactions", [])
            ] if msg.get("reactions") else [],
            "pinned": msg.get("pinned", False),
        })
    return _listing("messages", result)


def _list_pins(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List pinned messages in a channel (content truncated for overview)."""
    messages = _discord_request("GET", f"/channels/{channel_id}/pins", token)
    return _listing("pinned_messages", [
        {
            "id": msg["id"],
            "content": msg.get("content", "")[:200],
            "author": msg.get("author", {}).get("username"),
            "timestamp": msg.get("timestamp"),
        }
        for msg in messages
    ])


def _pin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    _discord_request("PUT", f"/channels/{channel_id}/pins/{message_id}", token)
    return _ok(f"Message {message_id} pinned.")


def _unpin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    _discord_request("DELETE", f"/channels/{channel_id}/pins/{message_id}", token)
    return _ok(f"Message {message_id} unpinned.")


def _delete_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}", token)
    return _ok(f"Message {message_id} deleted.")


def _create_thread(
    token: str, channel_id: str, name: str,
    message_id: Optional[str] = None,
    auto_archive_duration: int = 1440,
    **_kwargs: Any,
) -> str:
    """Create a thread — anchored to ``message_id`` when given, else standalone public."""
    body: Dict[str, Any] = {"name": name, "auto_archive_duration": auto_archive_duration}
    if message_id:
        path = f"/channels/{channel_id}/messages/{message_id}/threads"
    else:
        path = f"/channels/{channel_id}/threads"
        body["type"] = 11  # PUBLIC_THREAD
    thread = _discord_request("POST", path, token, body=body)
    return json.dumps({"success": True, "thread_id": thread["id"], "name": thread.get("name")})


def _add_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    _discord_request("PUT", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return _ok(f"Role {role_id} added to user {user_id}.")


def _remove_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    _discord_request("DELETE", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return _ok(f"Role {role_id} removed from user {user_id}.")


# ── action dispatch + metadata ───────────────────────────────────────────────

_ACTIONS = {
    "list_guilds": _list_guilds,
    "server_info": _server_info,
    "list_channels": _list_channels,
    "channel_info": _channel_info,
    "list_roles": _list_roles,
    "member_info": _member_info,
    "search_members": _search_members,
    "fetch_messages": _fetch_messages,
    "list_pins": _list_pins,
    "pin_message": _pin_message,
    "unpin_message": _unpin_message,
    "delete_message": _delete_message,
    "create_thread": _create_thread,
    "add_role": _add_role,
    "remove_role": _remove_role,
}

_CORE_ACTION_NAMES = frozenset({"fetch_messages", "search_members", "create_thread"})
_ADMIN_ACTION_NAMES = frozenset(_ACTIONS.keys()) - _CORE_ACTION_NAMES

_CORE_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _CORE_ACTION_NAMES}
_ADMIN_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _ADMIN_ACTION_NAMES}

# Single source of truth: action → (required-param signature, description).
# Drives the schema description AND runtime required-param validation.
_ACTION_MANIFEST: List[Tuple[str, str, str]] = [
    ("list_guilds", "()", "list servers the bot is in"),
    ("server_info", "(guild_id)", "server details + member counts"),
    ("list_channels", "(guild_id)", "all channels grouped by category"),
    ("channel_info", "(channel_id)", "single channel details"),
    ("list_roles", "(guild_id)", "roles sorted by position"),
    ("member_info", "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", "(guild_id, query)", "find members by name prefix"),
    ("fetch_messages", "(channel_id)", "recent messages; optional before/after snowflakes"),
    ("list_pins", "(channel_id)", "pinned messages in a channel"),
    ("pin_message", "(channel_id, message_id)", "pin a message"),
    ("unpin_message", "(channel_id, message_id)", "unpin a message"),
    ("delete_message", "(channel_id, message_id)", "delete a message"),
    ("create_thread", "(channel_id, name)", "create a public thread; optional message_id anchor"),
    ("add_role", "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", "(guild_id, user_id, role_id)", "remove a role"),
]

# Actions that require the GUILD_MEMBERS privileged intent.
_INTENT_GATED_MEMBERS = frozenset({"member_info", "search_members"})

# Per-action required params for runtime validation, parsed from the manifest.
_REQUIRED_PARAMS: Dict[str, List[str]] = {
    name: [p.strip() for p in sig.strip("()").split(",") if p.strip()]
    for name, sig, _desc in _ACTION_MANIFEST
}


def _load_allowed_actions_config() -> Optional[List[str]]:
    """``discord.server_actions`` allowlist (comma string or YAML list), or
    ``None`` when unrestricted. Unknown names are dropped with a warning."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); allowing all actions.", exc)
        return None

    raw = (cfg.get("discord") or {}).get("server_actions")
    if raw is None or raw == "":
        return None

    if isinstance(raw, str):
        names = [n.strip() for n in raw.split(",") if n.strip()]
    elif isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw if str(n).strip()]
    else:
        logger.warning(
            "discord.server_actions: unexpected type %s; ignoring.", type(raw).__name__,
        )
        return None

    valid = [n for n in names if n in _ACTIONS]
    invalid = [n for n in names if n not in _ACTIONS]
    if invalid:
        logger.warning(
            "discord.server_actions: unknown action(s) ignored: %s. "
            "Known: %s",
            ", ".join(invalid), ", ".join(_ACTIONS.keys()),
        )
    return valid


def _available_actions(
    caps: Dict[str, Any],
    allowlist: Optional[List[str]],
) -> List[str]:
    """Visible actions from intents + config allowlist, in :data:`_ACTIONS` order."""
    members_ok = caps.get("has_members_intent", True)
    return [
        name
        for name in _ACTIONS
        if (members_ok or name not in _INTENT_GATED_MEMBERS)
        and (allowlist is None or name in allowlist)
    ]


# ── schema construction ──────────────────────────────────────────────────────

_TOOL_DESCRIPTIONS = {
    "discord_admin": (
        "Manage a Discord server via the REST API.",
        "Call list_guilds first to discover guild_ids, then list_channels for "
        "channel_ids. Runtime errors will tell you if the bot lacks a specific "
        "per-guild permission (e.g. MANAGE_ROLES for add_role).",
    ),
    "discord": (
        "Read and participate in a Discord server.",
        "Use the channel_id from the current conversation context. "
        "Use search_members to look up user IDs by name prefix.",
    ),
}

_SCHEMA_PROPERTIES: Dict[str, Any] = {
    "guild_id": {"type": "string", "description": "Discord server (guild) ID."},
    "channel_id": {"type": "string", "description": "Discord channel ID."},
    "user_id": {"type": "string", "description": "Discord user ID."},
    "role_id": {"type": "string", "description": "Discord role ID."},
    "message_id": {"type": "string", "description": "Discord message ID."},
    "query": {"type": "string", "description": "Member name prefix to search for (search_members)."},
    "name": {"type": "string", "description": "New thread name (create_thread)."},
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Max results (default 50). Applies to fetch_messages, search_members.",
    },
    "before": {"type": "string", "description": "Snowflake ID for reverse pagination (fetch_messages)."},
    "after": {"type": "string", "description": "Snowflake ID for forward pagination (fetch_messages)."},
    "auto_archive_duration": {
        "type": "integer",
        "enum": [60, 1440, 4320, 10080],
        "description": "Thread archive duration in minutes (create_thread, default 1440).",
    },
}


def _build_schema(
    actions: List[str],
    caps: Optional[Dict[str, Any]] = None,
    tool_name: str = "discord",
) -> Optional[Dict[str, Any]]:
    """Tool schema for the filtered action list; ``None`` when empty (drop the tool)."""
    caps = caps or {}
    if not actions:
        return None

    manifest_block = "\n".join(
        f"  {name}{sig}  — {desc}" for name, sig, desc in _ACTION_MANIFEST if name in actions
    )

    content_note = ""
    affected_actions = {"fetch_messages", "list_pins"} & set(actions)
    if affected_actions and caps.get("detected") and caps.get("has_message_content") is False:
        names = " and ".join(sorted(affected_actions))
        content_note = (
            f"\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. "
            f"{names} will return message metadata (author, "
            "timestamps, attachments, reactions, pin state) but `content` will be "
            "empty for messages not sent as a direct mention to the bot or in DMs. "
            "Enable the intent in the Discord Developer Portal to see all content."
        )

    lead, guidance = _TOOL_DESCRIPTIONS["discord_admin" if tool_name == "discord_admin" else "discord"]
    description = f"{lead}\n\nAvailable actions:\n{manifest_block}\n\n{guidance}{content_note}"
    properties: Dict[str, Any] = {"action": {"type": "string", "enum": actions}, **_SCHEMA_PROPERTIES}
    return {
        "name": tool_name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": ["action"]},
    }


def _get_dynamic_schema(
    action_subset: Dict[str, Any],
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    """Build a dynamic schema for *action_subset* filtered by intents + config."""
    token = _get_bot_token()
    if not token:
        return None
    caps = _detect_capabilities_nonblocking(token)
    allowlist = _load_allowed_actions_config()
    actions = [a for a in _available_actions(caps, allowlist) if a in action_subset]
    if not actions:
        return None
    return _build_schema(actions, caps, tool_name=tool_name)


def get_dynamic_schema_core() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_CORE_ACTIONS, "discord")


def get_dynamic_schema_admin() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_ADMIN_ACTIONS, "discord_admin")


# ── 403 error enrichment ─────────────────────────────────────────────────────

_ACTION_403_HINT = {
    "pin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel. "
        "Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, "
        "or a per-channel overwrite."
    ),
    "unpin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel."
    ),
    "delete_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel, or cannot view the channel/message."
    ),
    "create_thread": (
        "Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it."
    ),
    "add_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role. Roles can only be assigned below the "
        "bot's own position in the role hierarchy."
    ),
    "remove_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role."
    ),
    "fetch_messages": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "list_pins": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "channel_info": (
        "Bot cannot view this channel (missing VIEW_CHANNEL)."
    ),
    "search_members": (
        "Likely missing the Server Members privileged intent — enable it in the "
        "Discord Developer Portal under your bot's settings."
    ),
    "member_info": (
        "Bot cannot see this guild member (missing Server Members intent or "
        "insufficient permissions)."
    ),
}


def _enrich_403(action: str, body: str) -> str:
    """Return a user-friendly guidance string for a 403 on ``action``."""
    hint = _ACTION_403_HINT.get(action)
    base = f"Discord API 403 (forbidden) on '{action}'."
    return f"{base} {hint} (Raw: {body})" if hint else f"{base} (Raw: {body})"


def check_discord_tool_requirements() -> bool:
    """Tool is available only when a Discord bot token is configured."""
    return bool(_get_bot_token())


# ── handlers ─────────────────────────────────────────────────────────────────

_HANDLER_DEFAULTS = {
    "action": "", "guild_id": "", "channel_id": "", "user_id": "",
    "role_id": "", "message_id": "", "query": "", "name": "",
    "limit": 50, "before": "", "after": "", "auto_archive_duration": 1440,
}


def _run_discord_action(
    action: str,
    valid_actions: Dict[str, Any],
    tool_label: str,
    **params: Any,
) -> str:
    """Shared handler logic for both discord tools (``params`` default per
    :data:`_HANDLER_DEFAULTS`)."""
    token = _get_bot_token()
    if not token:
        return tool_error("DISCORD_BOT_TOKEN not configured.")

    action_fn = valid_actions.get(action)
    if not action_fn:
        return tool_error(
            f"Unknown action: {action}",
            available_actions=list(valid_actions.keys()),
        )

    # Config-level allowlist gate (defense in depth): a stale cached schema from
    # a prior config must not let denied actions through.
    allowlist = _load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return tool_error(
            f"Action '{action}' is disabled by config (discord.server_actions). "
            f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}"
        )

    kwargs = {k: params.get(k, v) for k, v in _HANDLER_DEFAULTS.items() if k != "action"}
    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not kwargs.get(p)]
    if missing:
        return tool_error(
            f"Missing required parameters for '{action}': {', '.join(missing)}"
        )

    try:
        return action_fn(token=token, **kwargs)
    except DiscordAPIError as e:
        logger.warning("Discord API error in %s action '%s': %s", tool_label, action, e)
        if e.status == 403:
            return tool_error(_enrich_403(action, e.body))
        return tool_error(str(e))
    except Exception as e:
        logger.exception("Unexpected error in %s action '%s'", tool_label, action)
        return tool_error(f"Unexpected error: {e}")


def discord_core(action: str, **kwargs) -> str:
    """Execute a core Discord action (fetch_messages, search_members, create_thread)."""
    return _run_discord_action(action, _CORE_ACTIONS, "discord", **kwargs)


def discord_admin_handler(action: str, **kwargs) -> str:
    """Execute a Discord admin action (server management)."""
    return _run_discord_action(action, _ADMIN_ACTIONS, "discord_admin", **kwargs)


# ── tool registration ────────────────────────────────────────────────────────


def _make_handler(handler_fn):
    """Create a registry-compatible handler lambda for a discord handler."""
    return lambda args, **kw: handler_fn(
        **{k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()},
    )


_STATIC_CORE_SCHEMA = _build_schema(
    list(_CORE_ACTIONS.keys()), caps={"detected": False}, tool_name="discord",
)
_STATIC_ADMIN_SCHEMA = _build_schema(
    list(_ADMIN_ACTIONS.keys()), caps={"detected": False}, tool_name="discord_admin",
)

registry.register(
    name="discord",
    toolset="discord",
    schema=_STATIC_CORE_SCHEMA,
    handler=_make_handler(discord_core),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)

registry.register(
    name="discord_admin",
    toolset="discord_admin",
    schema=_STATIC_ADMIN_SCHEMA,
    handler=_make_handler(discord_admin_handler),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)
