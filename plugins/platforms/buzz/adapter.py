"""Buzz Platform Adapter for Hermes Agent.

Connects to a Buzz community relay (Block's Nostr-based human+agent platform).
Outbound and polling go through the ``buzz`` CLI ("JSON in, JSON out", never a
shell); inbound prefers a NIP-42-authenticated WebSocket subscription with a
CLI poll-loop fallback.

Configuration in config.yaml::

    gateway:
      platforms:
        buzz:
          enabled: true
          extra:
            relay_url: https://mycommunity.communities.buzz.xyz
            channels:                  # channel UUIDs to watch (empty = all joined)
              - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
            home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
            poll_interval: 4           # seconds between poll sweeps
            cli_path: ""               # path to the buzz binary (default: PATH, then ~/bin/buzz)
            credentials_file: ""       # JSON file holding the nsec (fallback for BUZZ_PRIVATE_KEY)
            allowed_users: []          # empty = allow all; entries are hex pubkeys or npubs
            reply_in_thread: true      # false = post replies flat to the channel timeline
            reaction_only_users: []    # acknowledge explicit tags without dispatching; allowed_users wins on overlap

Or via environment variables (override config.yaml): BUZZ_RELAY_URL, BUZZ_CHANNELS,
BUZZ_HOME_CHANNEL, BUZZ_POLL_INTERVAL, BUZZ_CLI_PATH, BUZZ_CREDENTIALS_FILE, BUZZ_ALLOWED_USERS,
BUZZ_REACTION_ONLY_USERS, BUZZ_ALLOW_ALL_USERS, BUZZ_REPLY_IN_THREAD, BUZZ_REPLY_TO_MODE.

The only secret is BUZZ_PRIVATE_KEY (nsec or hex) — it belongs in ``~/.hermes/.env``,
travels to the CLI via the subprocess environment and is never logged.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import (
    UnscopedSecretError as _UnscopedSecretError, current_secret_scope as _current_secret_scope,
    get_secret as _scoped_get_secret, is_multiplex_active as _is_multiplex_active,
)
from gateway.platforms._shared import profile_scoped as _profile_scoped


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read. An active profile scope is authoritative (a miss
    returns ``default``, never an env borrow); the unscoped default profile uses its own
    env. Extra unscoped rung (unlike ``_shared.get_scoped_secret``): the startup gate
    runs before any scope exists, so externally managed secrets are consulted via a
    one-shot profile-scope build. An ACTIVE scope shadows that rung."""
    try:
        val = _scoped_get_secret(name, None)
    except _UnscopedSecretError:
        val = os.getenv(name)
    if val is None and _current_secret_scope() is None:
        val = _unscoped_profile_secrets().get(name)
    return val if val is not None else default


_UNSCOPED_PROFILE_SECRETS: Optional[Dict[str, str]] = None


def _unscoped_profile_secrets() -> Dict[str, str]:
    """One-shot, process-cached profile secret mapping (external resolvers are slow).

    Any failure degrades to an empty mapping (platform reported as not configured).
    Startup-gate-only: it pins the ``get_hermes_home()`` seen on first build.
    """
    global _UNSCOPED_PROFILE_SECRETS
    if _UNSCOPED_PROFILE_SECRETS is None:
        try:
            from agent.secret_scope import build_profile_secret_scope
            from hermes_constants import get_hermes_home
            _UNSCOPED_PROFILE_SECRETS = dict(build_profile_secret_scope(get_hermes_home()))
        except Exception:
            logger.warning(
                "Buzz requirement probe could not build the profile secret "
                "scope; Bitwarden-managed credentials will not be visible "
                "to the startup gate (#95216)",
                exc_info=True,
            )
            _UNSCOPED_PROFILE_SECRETS = {}
    return _UNSCOPED_PROFILE_SECRETS


def _scoped_platform_setting(env_name, extra, key):
    """Raw non-secret setting read. Inside a secondary profile scope ``os.environ``
    holds the DEFAULT profile's bridge output, so ``extra`` is authoritative and a
    miss yields ``None``; elsewhere the legacy env-over-config read is preserved."""
    if _profile_scoped():
        return (extra or {}).get(key)
    return os.getenv(env_name)


logger = logging.getLogger(__name__)

from gateway.platforms.base import (
    BasePlatformAdapter, CachedMedia, SendResult, MessageEvent, MessageType, cache_media_bytes,
)
from gateway.config import Platform


# Buzz chat messages are Nostr kind 9; ``messages get`` also returns housekeeping
# kinds (joins, canvas updates, …) which are never dispatched.
_CHAT_KIND = 9
# Dispatched kinds: chat (9) plus forum post (45001) and forum comment (45003).
# Stream kinds (46010/40007/45002) are left out until their semantics are
# confirmed. ``_is_direct_message_event`` deliberately stays kind-9-only so a
# p-tagged forum post can't be reclassified as a DM and bypass mention gating.
_DISPATCH_KINDS = frozenset({_CHAT_KIND, 45001, 45003})
_UNRESOLVED_MENTION_ERROR_RE = re.compile(r"mention '@(?P<name>[^']+)' does not match a current channel member")
_BUZZ_PRESENTATION_MENTION_SEPARATOR = "\u200b"


def _escape_unresolved_presentation_mention(content: str, error: str) -> Optional[str]:
    """Make one CLI-rejected ``@name`` token presentation-only via an invisible
    separator after the ``@`` (Buzz p-tags whitespace-prefixed @tokens at publish,
    so prose like ``@session:...`` fails preflight). ``None`` if not applicable."""
    match = _UNRESOLVED_MENTION_ERROR_RE.search(error or "")
    if match is None:
        return None
    name = match.group("name")
    if not name:
        return None
    token = re.compile(rf"(?<!\S)@{re.escape(name)}(?=$|[^A-Za-z0-9._-])", re.IGNORECASE)
    escaped, count = token.subn(
        lambda found: "@" + _BUZZ_PRESENTATION_MENTION_SEPARATOR + found.group(0)[1:], content
    )
    return escaped if count else None


_FETCH_LIMIT = 50  # events per poll / seed call
_SEEN_CAP = 500  # per-channel de-dupe set bound (events)
# Per-channel cursors survive a restart here, relative to HERMES_HOME.
_CURSOR_STATE_SUBDIR = "buzz"
_CURSOR_STATE_FILENAME = "channel-cursors.json"
_DM_DISCOVERY_EVERY = 5  # re-run DM discovery every N poll sweeps

_DEFAULT_POLL_INTERVAL = 4.0
_MIN_POLL_INTERVAL = 1.0
_CLI_TIMEOUT = 30.0

# Mention-resolution caches: member lists are hit on every publish containing
# "@"; display names change rarely but must not survive a rename forever.
_MEMBER_CACHE_TTL = 60.0
_PROFILE_NAME_TTL = 300.0
# Inbound attachments download only after sender/mention/allow-list gates pass and
# must declare + match an exact size and SHA-256 in their NIP-94 ``imeta`` tag.
_MAX_INBOUND_ATTACHMENTS = 4
_MAX_INBOUND_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ATTACHMENT_DOWNLOAD_TIMEOUT = 30.0
_MAX_ATTACHMENT_FILENAME_BYTES = 120


def _safe_attachment_filename(value: str) -> str:
    """Return a basename that is safe for cache files and agent context."""
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if ord(c) >= 32 and c != "\x7f").strip()
    if name in {"", ".", ".."}:
        return "attachment.bin"
    suffix = Path(name).suffix
    if len(suffix.encode("utf-8")) > 20:
        suffix = ""
    stem = name[:-len(suffix)] if suffix else name
    byte_budget = _MAX_ATTACHMENT_FILENAME_BYTES - len(suffix.encode("utf-8"))
    safe_stem = stem.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{safe_stem or 'attachment'}{suffix}"


def _attachment_origin(value: str) -> Optional[tuple[str, int]]:
    """Normalize a configured host/URL to an exact HTTPS-equivalent origin."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port or 443
    except ValueError:
        return None
    if (parsed.scheme and parsed.scheme not in {"https", "wss"}) or not host:
        return None
    return host, port

# WebSocket transport (NIP-42 authenticated Nostr subscription).
_WS_AUTH_TIMEOUT = 20.0
# Last-resort read bound: the library keepalive should catch a dead relay first,
# but a relay-side close the transport never surfaces (CLOSE_WAIT socket, loop
# parked on recv) leaves the gateway "connected" while inbound stops; this
# timeout forces the normal reconnect path.
_WS_READ_IDLE_TIMEOUT = 300.0
_WS_MAX_MESSAGE_BYTES = 2_000_000
_WS_MEMBERSHIP_KIND = 44100  # Buzz channel-membership event — live DM discovery
_WS_MEMBERSHIP_SUB_ID = "hermes-buzz-membership"

# Credentials JSON (keys: nsec / private_key_hex) fallback when BUZZ_PRIVATE_KEY
# is not set. Module-level so tests can point it at a tmpdir.
_DEFAULT_CREDENTIALS_DIR = Path("~/.config/buzz").expanduser()

# Buzz-hosted Blossom media is private to the community: same-relay markdown or
# bare media URLs must be authenticated and localised before vision sees them.
_MEDIA_URL_PATTERN = (
    r"https?://[^\s<>\[\]()]+/media/"
    r"[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?(?:\?[^\s<>\[\]()]*)?"
)
_MARKDOWN_MEDIA_RE = re.compile(
    rf"!\[(?P<alt>[^\]]*)\]\(\s*(?P<url>{_MEDIA_URL_PATTERN})"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)",
    re.IGNORECASE,
)
_BARE_MEDIA_RE = re.compile(_MEDIA_URL_PATTERN, re.IGNORECASE)
_MEDIA_PATH_RE = re.compile(r"^/media/(?P<sha>[0-9a-f]{64})(?P<ext>\.[a-z0-9]{1,10})?/?$", re.IGNORECASE)


def _effective_port(parsed) -> Optional[int]:
    try:
        if parsed.port is not None:
            return parsed.port
    except ValueError:
        return None
    return {"https": 443, "wss": 443, "http": 80, "ws": 80}.get(parsed.scheme)


def _is_relay_media_url(url: str, relay_url: str) -> bool:
    """Return whether *url* is a Buzz media object on the configured relay."""
    candidate = urlsplit(url)
    relay = urlsplit(relay_url)
    return bool(
        candidate.scheme in ("http", "https")
        and candidate.hostname and relay.hostname
        and candidate.hostname.lower() == relay.hostname.lower()
        and _effective_port(candidate) == _effective_port(relay)
        and _MEDIA_PATH_RE.fullmatch(candidate.path)
    )


def _find_relay_media_refs(text: str, relay_url: str) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Find same-relay media URLs and their safe text replacements."""
    urls: List[str] = []
    replacements: List[Tuple[int, int, str]] = []
    markdown_spans: List[Tuple[int, int]] = []
    for match in _MARKDOWN_MEDIA_RE.finditer(text):
        url = match.group("url")
        if not _is_relay_media_url(url, relay_url):
            continue
        markdown_spans.append(match.span())
        replacements.append((*match.span(), match.group("alt").strip()))
        if url not in urls:
            urls.append(url)
    for match in _BARE_MEDIA_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in markdown_spans):
            continue
        url = match.group(0)
        if not _is_relay_media_url(url, relay_url):
            continue
        replacements.append((*match.span(), ""))
        if url not in urls:
            urls.append(url)
    return urls, replacements


def _replace_media_refs(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    cleaned = text
    for start, end, replacement in sorted(replacements, reverse=True):
        cleaned = f"{cleaned[:start]}{replacement}{cleaned[end:]}"
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _load_nostr_auth():
    """Import sibling nostr_auth loader-agnostically (the test plugin loader imports this
    file as a bare module, where relative imports have no parent package)."""
    try:
        from . import nostr_auth  # type: ignore[no-redef]
        return nostr_auth
    except ImportError:
        import importlib.util
        path = Path(__file__).with_name("nostr_auth.py")
        spec = importlib.util.spec_from_file_location("plugin_adapter_buzz_nostr_auth", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_nostr_auth = _load_nostr_auth()

# bech32 (BIP-173) helpers — npub <-> hex so mention detection and allow-lists
# accept either form. Same pure-stdlib implementation as nostr_auth / authz_mixin.
_BECH32_CHARSET = _nostr_auth.BECH32_CHARSET
_bech32_polymod = _nostr_auth._bech32_polymod
_bech32_hrp_expand = _nostr_auth._bech32_hrp_expand
from gateway.authz_mixin import _convertbits, _npub_to_hex as npub_to_hex  # noqa: E402


def hex_to_npub(pubkey_hex: str) -> Optional[str]:
    """Encode a 64-char hex pubkey as an ``npub1…`` bech32 string."""
    try:
        raw = bytes.fromhex(pubkey_hex)
    except ValueError:
        return None
    if len(raw) != 32:
        return None
    data = _convertbits(raw, 8, 5)
    if data is None:
        return None
    values = _bech32_hrp_expand("npub") + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return "npub1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def _split_csv(raw):
    return raw.split(",") if isinstance(raw, str) else raw


def _pubkey_set(raw) -> set:
    """Normalize a csv string / list of hex pubkeys or npubs to a set of hex pubkeys."""
    return {
        normalized
        for entry in _split_csv(raw)
        if isinstance(entry, str) and (normalized := _normalize_user_ref(entry))
    }


def _setting_or(env_name: str, extra: dict, key: str, default):
    """Scoped setting read with an explicit ``None`` -> ``extra`` fallback."""
    raw = _scoped_platform_setting(env_name, extra, key)
    return extra.get(key, default) if raw is None else raw


def _add_pubkey(bucket: List[str], raw) -> None:
    """Append the lowercased pubkey once (empty values are skipped)."""
    pk = str(raw or "").lower()
    if pk and pk not in bucket:
        bucket.append(pk)


def _normalize_user_ref(ref: str) -> Optional[str]:
    """Normalize a user reference (hex pubkey or npub) to lowercase hex."""
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    if ref.startswith("npub1"):
        return npub_to_hex(ref)
    if re.fullmatch(r"[0-9a-f]{64}", ref):
        return ref
    return None


# ── buzz-cli invocation helpers ──────────────────────────────────────────────

def _reply_to_mode(config, extra: dict) -> str:
    """Reply mode ("first"/"all" thread onto the parent, "off" posts flat). Env overrides
    the PlatformConfig field; Slack-convention ``reply_in_thread: false`` aliases "off"."""
    mode = os.getenv("BUZZ_REPLY_TO_MODE") or getattr(config, "reply_to_mode", "first") or "first"
    mode = str(mode).strip().lower()
    rit = os.getenv("BUZZ_REPLY_IN_THREAD")
    if rit is None:
        rit = extra.get("reply_in_thread")
    if rit is not None and str(rit).strip().lower() in ("false", "0", "no", "off"):
        return "off"
    return mode


def _configured_relay(extra: dict) -> str:
    raw = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
    return (raw or extra.get("relay_url", "")).strip()


def _configured_home_channel(extra: dict) -> str:
    raw = _scoped_platform_setting("BUZZ_HOME_CHANNEL", extra, "home_channel")
    return (raw or str(extra.get("home_channel", "") or "")).strip()


def _configured_cli_path(extra: dict) -> str:
    raw = _scoped_platform_setting("BUZZ_CLI_PATH", extra, "cli_path")
    return _resolve_cli_path(str(raw or "").strip() or str(extra.get("cli_path", "") or ""))


def _configured_credentials_file(extra: Optional[dict]) -> str:
    # Scope-aware read: inside a secondary profile scope a miss falls to the profile's
    # own config extra, never the default profile's os.environ; unscoped reads keep env
    # precedence plus the external-secret rung.
    return str(_get_scoped_secret("BUZZ_CREDENTIALS_FILE", "") or "").strip() or str(
        (extra or {}).get("credentials_file", "") or ""
    ).strip()


def _resolve_cli_path(configured: str = "") -> str:
    """Resolve the buzz binary: explicit config → ``buzz`` on PATH → ``~/bin/buzz``; "" if none."""
    if configured:
        p = Path(configured).expanduser()
        return str(p) if p.is_file() else ""
    found = shutil.which("buzz")
    if found:
        return found
    fallback = Path.home() / "bin" / "buzz"
    return str(fallback) if fallback.is_file() else ""


def _credentials_candidates(extra: Optional[dict] = None) -> List[Path]:
    configured = _configured_credentials_file(extra)
    if configured:
        return [Path(configured).expanduser()]
    if _is_multiplex_active():
        return []
    try:
        return sorted(_DEFAULT_CREDENTIALS_DIR.glob("*credentials*.json"))
    except OSError:
        return []


_KEY_FIELDS = ("nsec", "private_key_hex", "private_key")


def _credentials_key(data: dict) -> str:
    """First non-empty private-key field in a credentials record, stripped ("" if none)."""
    for field in _KEY_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_credentials_data(extra: Optional[dict] = None) -> dict:
    """Load the first credential record containing a private key."""
    for path in _credentials_candidates(extra):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and _credentials_key(data):
            return data
    return {}


def _resolve_private_key(extra: Optional[dict] = None) -> str:
    """Resolve the Nostr private key: scoped secret first, then credentials JSON. NEVER log it."""
    key = str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip()
    return key or _credentials_key(_resolve_credentials_data(extra))


def _resolve_auth_tag(extra: Optional[dict] = None) -> str:
    """Resolve and validate the optional NIP-OA owner-attestation tag."""
    configured = str(_get_scoped_secret("BUZZ_AUTH_TAG", "") or "").strip()
    if configured:
        raw: Any = configured
    else:
        credentials_file = _configured_credentials_file(extra)
        direct_key = str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip()
        if direct_key and not credentials_file:
            return ""
        data = _resolve_credentials_data(extra)
        if "auth_tag" not in data:
            return ""
        raw = data["auth_tag"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Buzz auth tag is not valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != 4 or raw[0] != "auth" or not all(isinstance(p, str) for p in raw):
        raise ValueError("Buzz auth tag must be a four-string auth tag")
    return json.dumps(raw, separators=(",", ":"))


async def _exec_buzz(
    cli_path: str, args: List[str], *, relay_url: str, private_key: str, auth_tag: str = "",
    input_text: Optional[str] = None, timeout: float = _CLI_TIMEOUT,
) -> Tuple[int, str, str]:
    """Run the buzz CLI (argv, never a shell) -> ``(rc, stdout, stderr)``. Key travels via env only."""
    env = os.environ.copy()
    env["BUZZ_RELAY_URL"] = relay_url
    env["BUZZ_PRIVATE_KEY"] = private_key
    env.pop("BUZZ_AUTH_TAG", None)
    if auth_tag:
        env["BUZZ_AUTH_TAG"] = auth_tag
    proc = await asyncio.create_subprocess_exec(
        cli_path, *args,
        stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    try:
        stdin_bytes = input_text.encode("utf-8") if input_text is not None else None
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", json.dumps({"error": "timeout", "message": f"buzz {args[0] if args else ''} timed out after {timeout}s"})
    return (
        proc.returncode if proc.returncode is not None else 4,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


_MAX_CLI_MESSAGE_CHARS = 900


def _bounded_cli_message(message: str, redact_path: Optional[Path] = None) -> str:
    """Keep untrusted CLI detail useful without exposing unbounded output."""
    if redact_path is not None:
        message = message.replace(str(redact_path), redact_path.name)
    if len(message) <= _MAX_CLI_MESSAGE_CHARS:
        return message
    return f"{message[: _MAX_CLI_MESSAGE_CHARS - 3]}..."


def _cli_error_message(stderr: str, returncode: int, *, redact_path: Optional[Path] = None) -> str:
    """Extract a bounded human-readable message from the CLI error contract."""
    text = (stderr or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            detail = data.get("message")
            category = data.get("error")
            if isinstance(detail, str) and detail.strip():
                label = category.strip() if isinstance(category, str) and category.strip() else "error"
                return _bounded_cli_message(f"{label}: {detail.strip()} (exit {returncode})", redact_path)
    except ValueError:
        pass
    return _bounded_cli_message(text or f"buzz CLI failed with exit code {returncode}", redact_path)


def _parse_send_receipt(stdout: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate the buzz-cli success receipt and return ``(event_id, error)``."""
    data = _json_or(stdout, None)
    if not isinstance(data, dict):
        return None, "invalid CLI response"
    if data.get("accepted") is False:
        detail = data.get("message")
        if not isinstance(detail, str) or not detail.strip():
            detail = "message was not accepted"
        return None, _bounded_cli_message(detail.strip())
    if data.get("accepted") is not True:
        return None, "invalid CLI response"
    event_id = data.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        return None, "invalid CLI response"
    return event_id.strip(), None


def _json_or(text: str, default):
    """``json.loads`` of CLI stdout, or *default* when empty/malformed."""
    try:
        return json.loads(text or json.dumps(default))
    except ValueError:
        return default


def _parse_json_list(stdout: str) -> List[dict]:
    """Parse CLI stdout expected to be a JSON array of objects."""
    data = _json_or(stdout, [])
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _event_reply_parent_id(event: dict) -> Optional[str]:
    """Direct parent id from NIP-10 ``e`` tags: ``reply`` marker, then ``root``, else last positional."""
    tags = event.get("tags")
    if not isinstance(tags, list):
        return None
    reply_id: Optional[str] = None
    root_id: Optional[str] = None
    last_e: Optional[str] = None
    for tag in tags:
        if not isinstance(tag, (list, tuple)) or len(tag) < 2 or tag[0] != "e":
            continue
        target = str(tag[1] or "").strip()
        if not target:
            continue
        marker = str(tag[3] or "") if len(tag) > 3 else ""
        last_e = target
        if marker == "reply":
            reply_id = target
        elif marker == "root":
            root_id = target
    return reply_id or root_id or last_e


# Cap stored parent content snippets (gateway reply injection also clips).
_EVENT_META_CONTENT_CAP = 500
_MEDIA_KIND_PRIORITY = (("image", MessageType.PHOTO), ("audio", MessageType.AUDIO), ("video", MessageType.VIDEO))
_ATTACHMENT_KIND_TYPES = {
    "image": MessageType.PHOTO,
    "video": MessageType.VIDEO,
    "audio": MessageType.AUDIO,
    "document": MessageType.DOCUMENT,
}


class BuzzAdapter(BasePlatformAdapter):
    """Buzz adapter (WebSocket push with poll fallback) for the BasePlatformAdapter interface."""

    def __init__(self, config, **kwargs):
        platform = Platform("buzz")
        super().__init__(config=config, platform=platform)
        extra = getattr(config, "extra", {}) or {}
        self._extra = extra
        # Connection settings: env overrides config.yaml, except under a secondary
        # multiplex profile scope where the profile's extra wins (see _scoped_platform_setting).
        self.relay_url = _configured_relay(extra)
        hosts = _split_csv(extra.get("attachment_hosts", []))
        origins = (_attachment_origin(h) for h in hosts if isinstance(h, str))
        self._attachment_origins = {o for o in origins if o is not None}
        relay_origin = _attachment_origin(self.relay_url)
        if relay_origin is not None:
            self._attachment_origins.add(relay_origin)
        self.cli_path = _configured_cli_path(extra)
        # Channels to watch: env csv > extra list/csv; empty = all joined channels
        raw_channels = _split_csv(_setting_or("BUZZ_CHANNELS", extra, "channels", []))
        self.channels: List[str] = [c.strip() for c in raw_channels if isinstance(c, str) and c.strip()]
        self.home_channel = _configured_home_channel(extra)
        _pi_raw = _scoped_platform_setting("BUZZ_POLL_INTERVAL", extra, "poll_interval")
        try:
            interval = float(_pi_raw or extra.get("poll_interval", _DEFAULT_POLL_INTERVAL))
        except (TypeError, ValueError):
            interval = _DEFAULT_POLL_INTERVAL
        self.poll_interval = max(_MIN_POLL_INTERVAL, interval)
        # Channel messages must @mention the agent unless disabled; DMs always dispatch.
        _rm_cfg = _setting_or("BUZZ_REQUIRE_MENTION", extra, "require_mention", True)
        self.require_mention = str(_rm_cfg).strip().lower() not in ("false", "0", "no", "off")
        self._reply_to_mode: str = _reply_to_mode(config, extra)
        # Inbound transport: "auto" (WebSocket with poll fallback), "websocket"
        # (require WS; fail connect when it can't authenticate), or "poll".
        _transport_raw = _scoped_platform_setting("BUZZ_TRANSPORT", extra, "transport")
        _transport = (_transport_raw or str(extra.get("transport", "auto") or "auto")).strip().lower()
        self.transport = _transport if _transport in ("auto", "websocket", "poll") else "auto"
        # Auth: entries may be hex pubkeys or npubs; normalized to hex.
        self._allowed_pubkeys: set = _pubkey_set(_setting_or("BUZZ_ALLOWED_USERS", extra, "allowed_users", []))
        # Reaction-only identities get a 👀 receipt on explicit tags but never
        # dispatch; allowed_users wins on overlap (normal dispatch path runs).
        self._reaction_only_pubkeys: set = _pubkey_set(
            os.getenv("BUZZ_REACTION_ONLY_USERS") or extra.get("reaction_only_users", [])
        )
        # Secret — resolved lazily (never at import time, never logged); connect() re-resolves.
        self._private_key: str = ""
        self._auth_tag: str = ""
        # Identity — filled in by connect() from ``buzz users get``
        self._self_pubkey: str = ""
        self._self_npub: str = ""
        self._display_name: str = ""
        # Runtime state
        self._poll_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_ready: Optional[asyncio.Event] = None
        self._membership_since = 0
        self._lock_key: Optional[str] = None
        # Channels the relay permanently rejected ("restricted: not a channel
        # member"); persists across reconnects so we never re-subscribe.
        self._restricted_channels: set = set()
        # channel_id -> {"chat_type", "last_ts", "seen": OrderedDict[event_id, None],
        #   "event_meta": OrderedDict[event_id, (author_pubkey, snippet)]}; event_meta
        # backs NIP-10 reply-parent resolution (replies to our messages count as addressed).
        self._channel_state: Dict[str, dict] = {}
        # Cursors read from disk at connect(), consumed by each channel's first seed.
        self._restored_cursors: Dict[str, dict] = {}
        self._channel_names: Dict[str, str] = {}
        # channel_id -> raw ``channels list`` entry; drives DM-vs-channel classification.
        self._channel_meta: Dict[str, dict] = {}
        self._user_names: Dict[str, str] = {}
        self._member_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._profile_name_cache: Dict[str, Tuple[float, str]] = {}
        self._poll_count = 0
        # inbound event_id -> thread root id (None when top-level), so send() joins
        # the user's thread instead of nesting a new one under every reply.
        self._thread_roots: "OrderedDict[str, Optional[str]]" = OrderedDict()

    @property
    def name(self) -> str:
        return "Buzz"

    @staticmethod
    def normalize_user_id(user_id: str) -> Optional[str]:
        """Normalize a user reference (hex or npub) to hex — authz_mixin allowlist hook."""
        return _normalize_user_ref(user_id)

    # ── buzz-cli plumbing ─────────────────────────────────────────────────

    async def _run_cli(self, args: List[str], *, input_text: Optional[str] = None) -> Tuple[int, str, str]:
        if not self._private_key:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        return await _exec_buzz(
            self.cli_path, args, relay_url=self.relay_url, private_key=self._private_key,
            auth_tag=self._auth_tag, input_text=input_text,
        )

    # ── Connection lifecycle ──────────────────────────────────────────────

    def _connect_failed(self, code: str, detail: str, log: str, *log_args, retryable: bool = False) -> bool:
        """Log and record a fatal connect() error; always returns False."""
        logger.error(log, *log_args)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Verify relay credentials, seed high-water marks, start polling."""
        if not self.relay_url:
            return self._connect_failed(
                "config_missing", "BUZZ_RELAY_URL must be set", "Buzz: relay URL must be configured"
            )
        if not self.cli_path:
            return self._connect_failed(
                "cli_missing", "buzz CLI binary not found",
                "Buzz: buzz CLI binary not found (set BUZZ_CLI_PATH or put 'buzz' on PATH)",
            )
        try:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        except ValueError as exc:
            return self._connect_failed(
                "config_invalid", str(exc), "Buzz: invalid owner-auth configuration — %s", exc
            )
        if not self._private_key:
            return self._connect_failed(
                "config_missing", "BUZZ_PRIVATE_KEY must be set",
                "Buzz: no private key (set BUZZ_PRIVATE_KEY or a credentials file)",
            )
        # Own identity: pubkey drives self-echo suppression, display name drives mention gating.
        code, out, err = await self._run_cli(["users", "get"])
        if code != 0:
            message = _cli_error_message(err, code)
            return self._connect_failed(
                "connect_failed", message, "Buzz: failed to fetch own profile from %s — %s",
                self.relay_url, message, retryable=code == 2,
            )
        profiles = _parse_json_list(out)
        if not profiles or not profiles[0].get("pubkey"):
            return self._connect_failed(
                "connect_failed", "buzz users get returned no profile",
                "Buzz: 'users get' returned no profile — is the key a member of this community?",
                retryable=True,
            )
        self._self_pubkey = str(profiles[0]["pubkey"]).lower()
        self._display_name = str(profiles[0].get("display_name") or "").strip()
        self._self_npub = hex_to_npub(self._self_pubkey) or ""
        # Two profiles must not drive the same identity on one relay (duplicate
        # replies, split de-dupe state).
        try:
            from gateway.status import acquire_scoped_lock
            lock_key = f"{self.relay_url}:{self._self_pubkey}"
            if not acquire_scoped_lock("buzz", lock_key):
                return self._connect_failed(
                    "lock_conflict", "Buzz identity in use by another profile",
                    "Buzz: identity %s… on %s already in use by another profile",
                    self._self_pubkey[:8], self.relay_url,
                )
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module not available (e.g. tests)
        # Map channel ids to names and pick the watch set.
        code, out, err = await self._run_cli(["channels", "list"])
        if code != 0:
            message = _cli_error_message(err, code)
            return self._connect_failed(
                "connect_failed", message, "Buzz: failed to list channels — %s", message, retryable=code == 2
            )
        self._channel_names = {}
        for ch in _parse_json_list(out):
            if ch.get("channel_id"):
                self._channel_names[str(ch["channel_id"])] = str(ch.get("name") or ch["channel_id"])
                self._channel_meta[str(ch["channel_id"])] = ch
        watch = self.channels or list(self._channel_names)
        if not watch:
            return self._connect_failed(
                "config_missing", "no Buzz channels to watch",
                "Buzz: no channels to watch (configure BUZZ_CHANNELS or join a channel)",
            )
        # Seed high-water marks so a (re)start never replays history — except
        # where a persisted cursor is restored so events that landed while we
        # were down still dispatch. Relay-rejected channels are skipped.
        self._load_cursors()
        for channel_id in watch:
            if channel_id in self._restricted_channels:
                logger.debug("Buzz: skipping restricted channel %s (relay rejected subscription)", channel_id)
                continue
            await self._seed_channel(channel_id, chat_type="group")
        await self._discover_dms(seed=True)
        self._save_cursors()
        # Prefer the NIP-42 WebSocket push; fall back to CLI polling when it can't
        # be established (transport="auto") or the user pinned transport="poll".
        transport_used = "poll"
        if self.transport in ("auto", "websocket"):
            if await self._start_websocket():
                transport_used = "websocket"
            elif self.transport == "websocket":
                self._set_fatal_error(
                    "ws_auth_failed", "Buzz WebSocket transport did not authenticate (transport=websocket)",
                    retryable=True,
                )
                await self.disconnect()
                return False
        if transport_used == "poll":
            self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info(
            "Buzz: connected to %s as %s, watching %d channel(s) via %s%s",
            self.relay_url, self._display_name or self._self_npub[:16], len(self._channel_state),
            transport_used, "" if transport_used == "websocket" else f", poll interval {self.poll_interval:.1f}s",
        )
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Stop the inbound transport and drop runtime state."""
        self._mark_disconnected()
        lock_key = getattr(self, "_lock_key", None)
        if lock_key:
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("buzz", lock_key)
            except Exception:
                pass
            self._lock_key = None
        await self._cancel_task(self._ws_task)
        self._ws_task = None
        await self._cancel_task(self._poll_task)
        self._poll_task = None
        self._channel_state = {}
        self._poll_count = 0

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ── Sending ───────────────────────────────────────────────────────────

    async def _channel_member_pubkeys(self, chat_id: str) -> List[str]:
        """Candidate pubkeys for mention resolution: ``channels members`` (a non-member
        ``--mention`` makes the CLI reject the publish), else recent traffic which can
        over-approximate — ``send()`` recovers by retrying without mentions."""
        cache = self._member_cache
        cached = cache.get(str(chat_id))
        if cached is not None and (time.monotonic() - cached[0]) < _MEMBER_CACHE_TTL:
            return list(cached[1])
        code, out, _err = await self._run_cli(["channels", "members", "--channel", str(chat_id)])
        if code == 0:
            pks: List[str] = []
            for row in _json_or(out, []):
                _add_pubkey(pks, row.get("pubkey") if isinstance(row, dict) else row)
            if pks:
                cache[str(chat_id)] = (time.monotonic(), list(pks))
                return pks
        candidates: List[str] = []
        code, out, _err = await self._run_cli(["messages", "get", "--channel", str(chat_id), "--limit", "50"])
        if code == 0:
            try:
                for msg in json.loads(out or "[]"):
                    _add_pubkey(candidates, msg.get("pubkey"))
                    for t in msg.get("tags") or []:
                        if isinstance(t, list) and len(t) > 1 and t[0] == "p":
                            _add_pubkey(candidates, str(t[1]))
            except ValueError:
                pass
        if candidates:
            cache[str(chat_id)] = (time.monotonic(), list(candidates))
        return candidates

    async def _profile_display_name(self, pubkey: str) -> str:
        """Display name for *pubkey* via ``users get --pubkey`` (bare ``users get`` may
        return only our own profile), cached for ``_PROFILE_NAME_TTL`` so renames resolve."""
        cache = self._profile_name_cache
        cached = cache.get(pubkey)
        if cached is not None and (time.monotonic() - cached[0]) < _PROFILE_NAME_TTL:
            return cached[1]
        name = ""
        code, out, _err = await self._run_cli(["users", "get", "--pubkey", pubkey])
        if code == 0:
            profiles = _json_or(out, [])
            if profiles and isinstance(profiles[0], dict):
                p0 = profiles[0]
                name = str(p0.get("display_name") or p0.get("name") or "").strip()
                if not name and p0.get("content"):
                    try:
                        prof = json.loads(p0["content"])
                        name = str(prof.get("display_name") or prof.get("name") or "").strip()
                    except ValueError:
                        pass
        cache[pubkey] = (time.monotonic(), name)
        return name

    async def _mention_pubkeys_for(self, chat_id: str, content: str) -> List[str]:
        """Resolve ``@Name`` tokens to member pubkeys (explicit ``--mention`` keeps genuine
        mentions notifying while unresolvable @-prose becomes plain text). Token semantics,
        Unicode word-bounded: "email@Fizz", "@@Fizz", "@FizzBuzz" do NOT wake Fizz; "@Riley!!"
        does. Longer names match first and consume their span; duplicates tag nobody."""
        if "@" not in content:
            return []
        by_name: Dict[str, List[str]] = {}
        display: Dict[str, str] = {}
        self_pk = getattr(self, "_self_pubkey", None)
        for pk in await self._channel_member_pubkeys(chat_id):
            if pk == self_pk:
                continue
            name = await self._profile_display_name(pk)
            if not name:
                continue
            key = name.lower()
            by_name.setdefault(key, [])
            if pk not in by_name[key]:
                by_name[key].append(pk)
            display.setdefault(key, name)
        found: List[str] = []
        text = content
        for key in sorted(by_name, key=len, reverse=True):
            pattern = re.compile(r"(?<![\w@])@" + re.escape(display[key]) + r"(?!\w)", re.IGNORECASE)
            if pattern.search(text):
                pks = by_name[key]
                if len(pks) == 1 and pks[0] not in found:
                    found.append(pks[0])
                # Consume the span either way so a shorter prefix name can't
                # double-match and an ambiguous name stays presentation-only.
                text = pattern.sub("\x00", text)
        return found

    async def _run_message_send(self, args: List[str], content: str, mention_pubkeys: Optional[List[str]] = None):
        """Send with bounded mention-failure recovery (each rung at most once):
        explicit ``--mention`` pubkeys; on "not channel members" retry without them;
        on an unresolvable presentation ``@token`` escape it and retry; finally
        ``--mention <self>`` (any explicit identity downgrades @names to plain text)."""
        mention_args: List[str] = []
        for pk in mention_pubkeys or []:
            mention_args += ["--mention", pk]
        code, out, err = await self._run_cli(args + mention_args, input_text=content)
        if code == 0:
            return code, out, err
        if mention_args and "not channel members" in (err or ""):
            code, out, err = await self._run_cli(args, input_text=content)
            if code == 0:
                return code, out, err
        escaped = _escape_unresolved_presentation_mention(content, err)
        if escaped is not None:
            logger.info("Buzz: retrying message after unresolved presentation-mention preflight")
            code, out, err = await self._run_cli(args, input_text=escaped)
            if code == 0:
                return code, out, err
        if (
            code != 0
            and "does not match a current channel member" in (err or "")
            and getattr(self, "_self_pubkey", None)
        ):
            code, out, err = await self._run_cli(args + ["--mention", self._self_pubkey], input_text=content)
        return code, out, err

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        if not content:
            return SendResult(success=False, error="Empty message")
        # Anchor preference: metadata.thread_id (Slack-style), then
        # metadata.reply_to_message_id (stream consumer / progress sends), then
        # the explicit reply_to; without any, interim commentary posts flat.
        meta = metadata or {}
        args = ["messages", "send", "--channel", str(chat_id), "--content", "-"]
        args += self._reply_args(meta.get("thread_id") or meta.get("reply_to_message_id") or reply_to)
        mention_pubkeys = await self._mention_pubkeys_for(chat_id, content)
        code, out, err = await self._run_message_send(args, content, mention_pubkeys)
        result = self._send_result(chat_id, code, out, err)
        if result.success:
            # Record event_meta so a thread reply to this send matches even if
            # the WS/poll echo never arrives.
            self._remember_event_meta(str(chat_id), result.message_id, self._self_pubkey, content)
        return result

    def _reply_args(self, anchor: Optional[str]) -> List[str]:
        """``--reply-to`` CLI args for *anchor*, honoring ``reply_to_mode``."""
        reply_target = self._resolve_reply_anchor(anchor)
        if reply_target and self._reply_to_mode != "off":
            return ["--reply-to", str(reply_target)]
        return []

    def _send_result(
        self, chat_id: str, code: int, out: str, err: str, *, redact_path: Optional[Path] = None
    ) -> SendResult:
        """``messages send`` CLI result -> SendResult; marks the verified id seen
        (belt-and-braces echo suppression on top of the inbound self-pubkey skip)."""
        if code != 0:
            return SendResult(
                success=False, error=_cli_error_message(err, code, redact_path=redact_path),
                retryable=code == 2,
            )
        event_id, receipt_error = _parse_send_receipt(out)
        if receipt_error:
            return SendResult(success=False, error=receipt_error)
        assert event_id is not None
        self._mark_seen(str(chat_id), event_id)
        return SendResult(success=True, message_id=event_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Buzz has no typing indicator API — no-op."""
        pass

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Best-effort reaction via buzz-cli; failures are logged, never raised."""
        if not self.cli_path or not emoji or not message_id:
            return False
        # The event id IS the dispatched message_id; channel is not a parameter here.
        code, _out, err = await self._run_cli(["reactions", "add", "--event", str(message_id), "--emoji", emoji])
        if code != 0:
            logger.debug(
                "Buzz: reaction add failed for message %s in %s — %s",
                message_id[:12], chat_id, _cli_error_message(err, code),
            )
            return False
        return True

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False
    ) -> SendResult:
        """Edit a sent message (enables streamed replies). buzz-cli reports a NEW event
        id for the edit but the TARGET stays the original, which the stream consumer
        holds across the stream — so return the given id, never the CLI's."""
        if not message_id:
            return SendResult(success=False, error="Buzz edit needs a message id")
        if not content:
            return SendResult(success=False, error="Empty message")
        args = ["messages", "edit", "--event", str(message_id), "--content", "-"]
        code, out, err = await self._run_cli(args, input_text=content)
        if code != 0:
            return SendResult(
                success=False, error=_cli_error_message(err, code), retryable=code == 2,
            )
        data = _json_or(out, {})
        edit_event_id = data.get("event_id")
        if edit_event_id:
            # The edit is itself a relay event that echoes back on our subscription.
            self._mark_seen(str(chat_id), str(edit_event_id))
        return SendResult(success=bool(data.get("accepted", True)), message_id=str(message_id), raw_response=data)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a sent message (stream consumer's fresh-final cleanup path)."""
        if not message_id:
            return False
        code, out, _err = await self._run_cli(["messages", "delete", "--event", str(message_id)])
        if code != 0:
            return False
        try:
            data = json.loads(out or "{}")
        except ValueError:
            return True
        event_id = data.get("event_id")
        if event_id:
            self._mark_seen(str(chat_id), str(event_id))
        return bool(data.get("accepted", True))

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image: local files upload via --file, URLs go as a link."""
        local = Path(image_url).expanduser() if not image_url.startswith(("http://", "https://")) else None
        if local is not None and local.is_file():
            return await self._send_file_attachment(
                chat_id, local, caption=caption, reply_to=reply_to, metadata=metadata, probe=False
            )
        # Markdown renders in Buzz, so a URL arrives as a clickable image link.
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def _send_file_attachment(
        self, chat_id: str, file_path: Path, *, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        probe: bool = True,
    ) -> SendResult:
        """Upload a local file as a native attachment. ``probe=False`` skips the
        existence re-check when the caller verified it (a second probe could race)."""
        local = Path(file_path).expanduser()
        if probe and not local.is_file():
            # Never leak host filesystem paths into chat-visible errors.
            return SendResult(success=False, error="Media file not found")
        args = ["messages", "send", "--channel", str(chat_id), "--file", str(local), "--content", "-"]
        args += self._reply_args((metadata or {}).get("thread_id") or reply_to)
        code, out, err = await self._run_message_send(args, caption or "")
        return self._send_result(chat_id, code, out, err, redact_path=local)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Upload a local image via ``--file``; missing paths keep the Base fallback
        so host filesystem paths are never echoed into chat."""
        local = Path(image_path).expanduser()
        if local.is_file():
            return await self._send_file_attachment(
                chat_id, local, caption=caption, reply_to=reply_to, metadata=metadata, probe=False
            )
        return await super().send_image_file(
            chat_id=chat_id, image_path=image_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs
        )

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Upload a local document through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id, Path(file_path), caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Upload a local video through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id, Path(video_path), caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Upload a local audio file through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id, Path(audio_path), caption=caption, reply_to=reply_to, metadata=metadata
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        state = self._channel_state.get(chat_id)
        chat_type = state["chat_type"] if state else "group"
        name = self._channel_names.get(chat_id)
        if name is None and self.cli_path:
            code, out, _err = await self._run_cli(["channels", "get", "--channel", chat_id])
            if code == 0:
                data = _json_or(out, {})
                if isinstance(data, dict) and data.get("name"):
                    name = str(data["name"])
                    self._channel_names[chat_id] = name
        return {"name": name or chat_id, "type": chat_type, "chat_id": chat_id}

    # ── Inbound: WebSocket transport (NIP-42 authenticated) ──────────────
    # Dispatches through the same _handle_event() as the poll loop so de-dupe,
    # mention gating, DM latching and the allow-list behave identically.

    def _websocket_url(self) -> str:
        parsed = urlsplit(self.relay_url.strip())
        scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
        if scheme not in ("ws", "wss") or not parsed.netloc:
            raise ValueError("Buzz relay URL must use http(s) or ws(s)")
        return urlunsplit((scheme, parsed.netloc, parsed.path or "", parsed.query, ""))

    async def _start_websocket(self) -> bool:
        """Start the WS loop; True when it authenticates within the timeout."""
        try:
            import websockets  # noqa: F401  (availability probe)
            self._websocket_url()
        except Exception as e:
            logger.info("Buzz: WebSocket transport unavailable (%s); falling back to polling", e)
            return False
        self._ws_ready = asyncio.Event()
        self._membership_since = int(time.time())
        self._ws_task = asyncio.create_task(self._websocket_loop())
        try:
            await asyncio.wait_for(self._ws_ready.wait(), timeout=_WS_AUTH_TIMEOUT + 5)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Buzz: WebSocket did not authenticate in time")
            await self._cancel_task(self._ws_task)
            self._ws_task = None
            return False
        return True

    async def _authenticate_websocket(self, websocket) -> None:
        """NIP-42: await the AUTH challenge, answer with a signed kind-22242 event
        (plus the optional NIP-OA owner-attestation tag), await the OK."""
        raw = await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT)
        message = json.loads(raw)
        if not isinstance(message, list) or len(message) < 2 or message[0] != "AUTH":
            raise ConnectionError("Buzz relay did not send a NIP-42 AUTH challenge")
        # BUZZ_AUTH_TAG is per-identity, so it resolves through the profile secret
        # scope (a scoped profile without one fails closed to "" rather than
        # borrowing the default profile's env tag). Resolved lazily here too so a
        # re-auth on a bare adapter (no connect()) stays scope-correct.
        auth_tag = getattr(self, "_auth_tag", "") or ""
        if not auth_tag:
            try:
                auth_tag = _resolve_auth_tag(getattr(self, "_extra", None))
            except ValueError:
                auth_tag = ""
        event = _nostr_auth.build_auth_event(
            private_key=self._private_key, challenge=str(message[1]),
            relay_url=self._websocket_url(), auth_tag_json=auth_tag,
        )
        await websocket.send(json.dumps(["AUTH", event], separators=(",", ":")))
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT)
            response = json.loads(raw)
            if not isinstance(response, list) or not response:
                continue
            if response[0] == "OK" and len(response) >= 4 and response[1] == event["id"]:
                if response[2] is True:
                    return
                raise ConnectionError(f"Buzz WebSocket AUTH rejected: {response[3]}")
            if response[0] in ("NOTICE", "CLOSED"):
                detail = response[-1] if len(response) > 1 else "authentication failed"
                raise ConnectionError(f"Buzz WebSocket AUTH failed: {detail}")

    @staticmethod
    async def _send_req(websocket, subscription_id: str, request_filter: dict) -> None:
        await websocket.send(json.dumps(["REQ", subscription_id, request_filter], separators=(",", ":")))

    async def _send_channel_subscription(self, websocket, subscription_id: str, channel_id: str) -> None:
        state = self._channel_state.get(channel_id) or {}
        last_ts = int(state.get("last_ts") or 0)
        request_filter = {"kinds": sorted(_DISPATCH_KINDS), "#h": [channel_id]}
        if last_ts:
            # Resume from the high-water mark (same-second overlap de-duped by id).
            request_filter["since"] = max(last_ts - 1, 0)
        else:
            # A conversation adopted mid-run with no high-water mark is fresh: its
            # history IS the conversation, so subscribe from the beginning or the
            # message that *created* it (created_at fractionally before this REQ)
            # is silently dropped. Seeded channels always have last_ts != 0.
            request_filter["limit"] = _FETCH_LIMIT
        await self._send_req(websocket, subscription_id, request_filter)

    async def _subscribe_websocket(self, websocket) -> Dict[str, Optional[str]]:
        """Subscribe to every watched conversation plus membership events
        (kind 44100 p-tagged to us) for live DM discovery."""
        subscriptions: Dict[str, Optional[str]] = {}
        for index, channel_id in enumerate(list(self._channel_state)):
            if channel_id in self._restricted_channels:
                continue
            subscription_id = f"hermes-buzz-{index}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
        if self._self_pubkey:
            await self._send_req(websocket, _WS_MEMBERSHIP_SUB_ID, {
                "kinds": [_WS_MEMBERSHIP_KIND],
                "#p": [self._self_pubkey],
                "since": max(self._membership_since - 1, 0),
            })
            subscriptions[_WS_MEMBERSHIP_SUB_ID] = None
        return subscriptions

    async def _rediscover_and_subscribe(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Rediscover conversations and subscribe to any adopted since (fresh DMs dispatch from their start)."""
        before = set(self._channel_state)
        await self._discover_dms(seed=False)
        for channel_id in list(self._channel_state):
            if channel_id in before:
                continue
            subscription_id = f"hermes-buzz-dm-{len(subscriptions)}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
            logger.info("Buzz: subscribed to new conversation %s", channel_id)

    async def _handle_membership_event(self, websocket, subscriptions: Dict[str, Optional[str]], event: dict) -> None:
        """A membership event p-tagged to us: rediscover and subscribe to new conversations."""
        self._membership_since = max(self._membership_since, int(event.get("created_at") or 0))
        await self._rediscover_and_subscribe(websocket, subscriptions)

    async def _ws_discovery_loop(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Periodic discovery on the poll transport's cadence: relays don't
        guarantee a kind-44100 membership event for every conversation that
        materializes mid-session. Failures retry next tick; the read loop alone
        owns connection health."""
        interval = max(self.poll_interval * _DM_DISCOVERY_EVERY, _MIN_POLL_INTERVAL)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._rediscover_and_subscribe(websocket, subscriptions)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Buzz: WebSocket discovery sweep failed", exc_info=True)

    async def _websocket_loop(self) -> None:
        """Persistent authenticated subscription with bounded reconnect backoff;
        on reconnect per-channel `since` filters resume from the last timestamps."""
        import websockets
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self._websocket_url(),
                    open_timeout=_WS_AUTH_TIMEOUT,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=_WS_MAX_MESSAGE_BYTES,
                ) as websocket:
                    await self._authenticate_websocket(websocket)
                    subscriptions = await self._subscribe_websocket(websocket)
                    if self._ws_ready is not None:
                        self._ws_ready.set()
                    backoff = 1.0
                    discovery_task = asyncio.create_task(self._ws_discovery_loop(websocket, subscriptions))
                    try:
                        frame_iter = websocket.__aiter__()
                        while True:
                            try:
                                raw = await asyncio.wait_for(frame_iter.__anext__(), timeout=_WS_READ_IDLE_TIMEOUT)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                raise ConnectionError(
                                    f"no WebSocket frame for {_WS_READ_IDLE_TIMEOUT:.0f}s; "
                                    "assuming the connection went silent"
                                ) from None
                            try:
                                message = json.loads(raw)
                            except (ValueError, TypeError):
                                logger.warning("Buzz: ignoring malformed WebSocket frame")
                                continue
                            if isinstance(message, list) and message:
                                await self._handle_ws_message(websocket, subscriptions, message)
                    finally:
                        discovery_task.cancel()
                        try:
                            await discovery_task
                        except (asyncio.CancelledError, Exception):
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Buzz: WebSocket disconnected; retrying in %.1fs: %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle_ws_message(self, websocket, subscriptions: Dict[str, Optional[str]], message: list) -> None:
        """Route one parsed relay frame (EVENT / CLOSED / NOTICE)."""
        if message[0] == "EVENT" and len(message) >= 3:
            subscription_id = str(message[1])
            event = message[2]
            if not isinstance(event, dict):
                return
            if subscription_id == _WS_MEMBERSHIP_SUB_ID:
                await self._handle_membership_event(websocket, subscriptions, event)
                return
            channel_id = subscriptions.get(subscription_id)
            state = self._channel_state.get(channel_id or "")
            if channel_id and state is not None:
                before = self._cursor_mark(state)
                await self._handle_event(channel_id, state, event)
                self._trim_seen(state)
                if self._cursor_mark(state) != before:
                    self._save_cursors()
        elif message[0] == "CLOSED":
            detail = message[-1] if len(message) > 2 else "subscription closed"
            sub_id = str(message[1]) if len(message) > 1 else ""
            closed_channel = subscriptions.get(sub_id)
            detail_l = str(detail).lower()
            # A membership rejection means the relay will never serve this
            # subscription — drop it permanently instead of reconnect-looping.
            is_membership_rejection = any(m in detail_l for m in ("restricted", "not a channel member", "auth-required"))
            if is_membership_rejection and closed_channel:
                logger.warning(
                    "Buzz: relay permanently rejected channel %s (%s) — removing from watch list",
                    closed_channel, detail,
                )
                self._restricted_channels.add(closed_channel)
                del subscriptions[sub_id]
                self._channel_state.pop(closed_channel, None)
            else:
                raise ConnectionError(str(detail))
        elif message[0] == "NOTICE":
            logger.warning("Buzz: relay notice: %s", message[-1])

    # ── Inbound polling ───────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll every watched channel for new events until cancelled."""
        while True:
            await asyncio.sleep(self.poll_interval)
            self._poll_count += 1
            try:
                if self._poll_count % _DM_DISCOVERY_EVERY == 0:
                    await self._discover_dms(seed=False)
                for channel_id in list(self._channel_state):
                    await self._poll_channel(channel_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Buzz: poll sweep failed", exc_info=True)

    def _new_channel_state(self, chat_type: str) -> dict:
        return {"chat_type": chat_type, "last_ts": 0, "seen": OrderedDict(), "event_meta": OrderedDict()}

    # ── Durable channel cursors ───────────────────────────────────────────

    @staticmethod
    def _cursor_path() -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / _CURSOR_STATE_SUBDIR / _CURSOR_STATE_FILENAME

    def _load_cursors(self) -> None:
        """Read persisted cursors. A file from another identity/relay is ignored (channel
        ids would collide); read/parse failures degrade to seed-from-history."""
        self._restored_cursors = {}
        try:
            path = self._cursor_path()
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Buzz: could not read channel cursors", exc_info=True)
            return
        if not isinstance(data, dict) or data.get("identity") != self._self_pubkey:
            return
        if data.get("relay") != self.relay_url:
            return
        channels = data.get("channels")
        if not isinstance(channels, dict):
            return
        for channel_id, entry in channels.items():
            if not isinstance(entry, dict):
                continue
            try:
                last_ts = int(entry.get("last_ts") or 0)
            except (TypeError, ValueError):
                continue
            raw_seen = entry.get("seen")
            seen = [str(event_id) for event_id in raw_seen][-_SEEN_CAP:] if isinstance(raw_seen, list) else []
            self._restored_cursors[str(channel_id)] = {
                "chat_type": str(entry.get("chat_type") or ""), "last_ts": last_ts, "seen": seen,
            }

    def _save_cursors(self) -> None:
        """Persist every watched channel's cursor.  Never raises."""
        channels = {
            channel_id: {
                "chat_type": state.get("chat_type") or "group",
                "last_ts": int(state.get("last_ts") or 0),
                "seen": list(state.get("seen") or ()),
            }
            for channel_id, state in self._channel_state.items()
        }
        payload = {"identity": self._self_pubkey, "relay": self.relay_url, "channels": channels}
        try:
            from utils import atomic_json_write
            atomic_json_write(self._cursor_path(), payload, indent=None)
        except Exception:
            logger.debug("Buzz: could not persist channel cursors", exc_info=True)

    @staticmethod
    def _cursor_mark(state: dict) -> tuple:
        """Cheap change detector for one channel's cursor."""
        seen = state.get("seen") or ()
        return int(state.get("last_ts") or 0), len(seen), (next(reversed(seen), None) if seen else None)

    def _restore_channel_state(self, channel_id: str, chat_type: str) -> bool:
        """Install a persisted cursor for *channel_id*; True when one existed. This
        closes the restart gap: seeding would mark everything that arrived while
        the gateway was down as already seen."""
        restored = self._restored_cursors.pop(channel_id, None)
        if restored is None:
            return False
        state = self._new_channel_state(restored["chat_type"] or chat_type)
        state["last_ts"] = restored["last_ts"]
        state["seen"] = OrderedDict((event_id, None) for event_id in restored["seen"])
        self._channel_state[channel_id] = state
        return True

    async def _seed_channel(self, channel_id: str, chat_type: str) -> None:
        """Initialize a channel's high-water mark from its newest events."""
        if self._restore_channel_state(channel_id, chat_type):
            return
        state = self._new_channel_state(chat_type)
        self._channel_state[channel_id] = state
        code, out, err = await self._run_cli(["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)])
        if code != 0:
            logger.warning("Buzz: could not seed channel %s — %s", channel_id, _cli_error_message(err, code))
            # "now" so a transiently unreadable channel never replays its history later.
            state["last_ts"] = int(time.time())
            return
        for event in _parse_json_list(out):
            event_id = event.get("id")
            created_at = int(event.get("created_at") or 0)
            if event_id:
                state["seen"][str(event_id)] = None
            state["last_ts"] = max(state["last_ts"], created_at)
            # History is never dispatched but still feeds event_meta (post-restart
            # thread replies to our earlier messages must match) and latches a DM
            # that leaked in via ``channels list`` before the first poll.
            self._remember_event(state, event)
            self._maybe_latch_dm(channel_id, state, event)
        self._trim_seen(state)

    async def _discover_dms(self, *, seed: bool) -> None:
        """Watch DM conversations: startup ones are seeded, mid-run ones dispatch from
        their beginning. ``dms list`` is best-effort (some relays return ``[]``); DMs
        also surface in ``channels list`` as name "DM" + empty description, which is
        the fallback shape. Named rooms and missing metadata fail closed as groups."""
        code, out, _err = await self._run_cli(["dms", "list"])
        if code == 0:
            for dm in _parse_json_list(out):
                dm_id = str(dm.get("dm_id") or "")
                if not dm_id or dm_id in self._channel_state or dm_id in self._restricted_channels:
                    continue
                await self._adopt_conversation(dm_id, seed)
                self._channel_names.setdefault(dm_id, "DM")
        code, out, _err = await self._run_cli(["channels", "list"])
        if code != 0:
            return
        for ch in _parse_json_list(out):
            ch_id = str(ch.get("channel_id") or "")
            if not ch_id:
                continue
            self._channel_meta[ch_id] = ch
            self._channel_names.setdefault(ch_id, str(ch.get("name") or ch_id))
            if ch_id in self._restricted_channels:
                continue
            if self._may_reclassify_as_dm(ch_id):
                # DM-shaped entries promote to DM — including ones already watched.
                if ch_id in self._channel_state:
                    self._channel_state[ch_id]["chat_type"] = "dm"
                else:
                    await self._adopt_conversation(ch_id, seed)
                continue
            if ch_id in self._channel_state:
                continue
            # Watch-all mode adopts channels joined mid-run without a restart.
            # Their history predates us, so seed from newest events (only later
            # messages dispatch). Explicit watch lists stay authoritative.
            if not seed and not self.channels:
                await self._seed_channel(ch_id, chat_type="group")
                logger.info("Buzz: adopted newly joined channel %s (%s)", ch_id, self._channel_names.get(ch_id, ch_id))

    async def _adopt_conversation(self, channel_id: str, seed: bool) -> None:
        """Start watching a DM conversation: seed at startup, else restore or start fresh."""
        if seed:
            await self._seed_channel(channel_id, chat_type="dm")
        elif not self._restore_channel_state(channel_id, "dm"):
            self._channel_state[channel_id] = self._new_channel_state("dm")

    async def _poll_channel(self, channel_id: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is None:
            return
        args = ["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)]
        if state["last_ts"]:
            # Nostr `since` is inclusive: same-second events re-fetch and de-dupe by id.
            args += ["--since", str(state["last_ts"])]
        code, out, err = await self._run_cli(args)
        if code != 0:
            logger.debug("Buzz: poll of channel %s failed — %s", channel_id, _cli_error_message(err, code))
            return
        before = self._cursor_mark(state)
        for event in _parse_json_list(out):
            await self._handle_event(channel_id, state, event)
        self._trim_seen(state)
        # Persist only when the cursor moved so idle channels don't rewrite the file.
        if self._cursor_mark(state) != before:
            self._save_cursors()

    @staticmethod
    def _parse_imeta_attachments(event: dict) -> Tuple[List[dict], int]:
        """Return accepted NIP-94 metadata and the rejected ``imeta`` count."""
        tags = event.get("tags")
        if not isinstance(tags, list):
            return [], 0
        attachments: List[dict] = []
        rejected = 0
        total_declared_bytes = 0
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or not tag or tag[0] != "imeta":
                continue
            if len(attachments) >= _MAX_INBOUND_ATTACHMENTS:
                rejected += 1
                continue
            fields: Dict[str, str] = {}
            for raw_field in tag[1:]:
                if not isinstance(raw_field, str):
                    continue
                key, separator, value = raw_field.partition(" ")
                if separator and key not in fields:
                    fields[key] = value.strip()
            url = fields.get("url", "")
            digest = fields.get("x", "").lower()
            filename = fields.get("filename", "")
            mime_type = fields.get("m", "")
            try:
                size = int(fields.get("size", ""))
                parsed = urlsplit(url)
                parsed_hostname = parsed.hostname
                parsed.port  # access validates malformed/non-numeric ports
            except (TypeError, ValueError):
                rejected += 1
                continue
            if (
                parsed.scheme != "https"
                or not parsed_hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not 0 < size <= _MAX_INBOUND_ATTACHMENT_BYTES
                or total_declared_bytes + size > _MAX_INBOUND_ATTACHMENT_BYTES
            ):
                rejected += 1
                continue
            total_declared_bytes += size
            attachments.append({
                "url": url, "sha256": digest, "size": size,
                "filename": _safe_attachment_filename(filename), "mime_type": mime_type[:255],
            })
        return attachments, rejected

    @staticmethod
    def _imeta_attachments(event: dict) -> List[dict]:
        """Return bounded, structurally valid NIP-94 attachment metadata."""
        attachments, _rejected = BuzzAdapter._parse_imeta_attachments(event)
        return attachments

    @staticmethod
    def _attachment_rejection_note(rejected: int) -> str:
        """Return a fixed-width diagnostic for malformed or excess metadata."""
        return f"[{rejected if rejected <= 999 else '999+'} Buzz attachment(s) rejected as malformed or over limits.]"

    async def _download_attachment(self, metadata: dict) -> Optional[CachedMedia]:
        """Download, integrity-check, and cache one authorized Buzz attachment."""
        url = metadata["url"]
        try:
            parsed_url = urlsplit(url)
            host = (parsed_url.hostname or "").lower().rstrip(".")
            origin = (host, parsed_url.port or 443)
        except ValueError:
            parsed_url = None
            origin = ("", 0)
        if parsed_url is None or parsed_url.scheme != "https" or origin not in self._attachment_origins:
            logger.warning(
                "Buzz: refusing attachment from untrusted origin %s:%s", origin[0] or "<missing>", origin[1]
            )
            return None
        import httpx
        try:
            timeout = httpx.Timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT)
            async with (
                asyncio.timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT),
                httpx.AsyncClient(
                    follow_redirects=False, timeout=timeout, headers={"Accept-Encoding": "identity"}
                ) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code != 200:
                    logger.warning("Buzz: attachment download returned HTTP %s", response.status_code)
                    return None
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_response_size = int(content_length)
                    except ValueError:
                        return None
                    if declared_response_size != metadata["size"]:
                        logger.warning("Buzz: attachment Content-Length does not match imeta size")
                        return None
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > metadata["size"]:
                        logger.warning("Buzz: attachment exceeded its declared size")
                        return None
        except (TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning("Buzz: attachment download failed: %s", exc)
            return None
        if len(data) != metadata["size"]:
            logger.warning("Buzz: attachment size does not match imeta")
            return None
        if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            logger.warning("Buzz: attachment SHA-256 does not match imeta")
            return None
        try:
            return cache_media_bytes(bytes(data), filename=metadata["filename"], mime_type=metadata["mime_type"])
        except (OSError, ValueError) as exc:
            logger.warning("Buzz: attachment cache write failed: %s", exc)
            return None

    async def _cache_inbound_attachments(self, metadata_items: List[dict]) -> List[CachedMedia]:
        return [a for m in metadata_items if (a := await self._download_attachment(m)) is not None]

    async def _handle_event(self, channel_id: str, state: dict, event: dict) -> None:
        """De-dupe, filter, and dispatch a single ``messages get`` event."""
        event_id = str(event.get("id") or "")
        created_at = int(event.get("created_at") or 0)
        if not event_id or event_id in state["seen"]:
            return
        state["seen"][event_id] = None
        state["last_ts"] = max(state["last_ts"], created_at)
        if int(event.get("kind") or 0) not in _DISPATCH_KINDS:
            return
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        attachment_metadata, rejected_attachments = self._parse_imeta_attachments(event)
        has_imeta = bool(attachment_metadata or rejected_attachments)
        if not pubkey or not isinstance(content, str) or (not content.strip() and not has_imeta):
            return
        # Feed the event cache before any early return so self-echo and
        # concurrent-author traffic can still be reply parents.
        self._remember_event(state, event)
        if pubkey == self._self_pubkey:
            return
        # Reclassify a leaked DM before gating so its first un-mentioned
        # message both latches the conversation and dispatches.
        self._maybe_latch_dm(channel_id, state, event)
        is_dm = state["chat_type"] == "dm"
        reply_parent_id = _event_reply_parent_id(event)
        reply_meta = self._lookup_event_meta(state, reply_parent_id) if reply_parent_id else None
        reply_to_is_own = bool(reply_meta is not None and reply_meta[0] == self._self_pubkey)
        # Shared channels dispatch only when addressed (text @mention OR signed
        # recipient p-tag) or when the NIP-10 parent is one of our messages
        # (parity with Signal/WhatsApp), unless require_mention is off. DMs always dispatch.
        if not is_dm and self.require_mention and not self._is_addressed(event) and not reply_to_is_own:
            return
        # Adapter-level allow-list (gateway also applies it centrally); empty = no filter.
        if self._allowed_pubkeys and pubkey not in self._allowed_pubkeys:
            explicitly_tagged = any(
                isinstance(tag, (list, tuple)) and len(tag) > 1 and tag[0] == "p"
                and str(tag[1]).lower() == self._self_pubkey
                for tag in event.get("tags") or []
            )
            if pubkey in self._reaction_only_pubkeys and explicitly_tagged and self._is_mentioned(content):
                await self.send_reaction(channel_id, event_id, "👀")
            logger.debug("Buzz: ignoring message from unauthorized pubkey %s…", pubkey[:8])
            return
        # Strip a leading @mention (both chat types — DMs often open with it too)
        # so slash commands like "@Chip /whoami" are recognized.
        dispatch_text = self._strip_mention(content)
        # NIP-10 thread root scopes the session; remember this message's root so
        # our reply joins the SAME thread instead of nesting a new one.
        thread_id = self._extract_thread_root(event)
        self._record_thread_root(event_id, event)
        # Attachment fetch is a credentialed side effect: only the gateway's
        # explicit ``True`` permits it (false/absent/failed all fail closed). The
        # message still dispatches so GatewayRunner can apply denial/pairing.
        chat_type = "dm" if is_dm else "group"
        attachment_fetch_allowed = bool(attachment_metadata) and (
            self._is_sender_authorized(pubkey, chat_type, channel_id) is True
        )
        attachments = await self._cache_inbound_attachments(attachment_metadata) if attachment_fetch_allowed else []
        if rejected_attachments:
            dispatch_text = f"{dispatch_text}\n{self._attachment_rejection_note(rejected_attachments)}".strip()
        if attachment_fetch_allowed and len(attachments) < len(attachment_metadata):
            failed = len(attachment_metadata) - len(attachments)
            dispatch_text = (
                f"{dispatch_text}\n"
                f"[{failed} Buzz attachment(s) could not be downloaded or failed integrity checks.]"
            ).strip()
        message_type = MessageType.TEXT
        if attachments:
            # Mixed kinds use document semantics so an audio member is not
            # mistaken for a voice note and sent through STT.
            attachment_kinds = {attachment.kind for attachment in attachments}
            message_type = MessageType.DOCUMENT
            if len(attachment_kinds) == 1:
                message_type = _ATTACHMENT_KIND_TYPES.get(next(iter(attachment_kinds)), MessageType.DOCUMENT)
        await self._dispatch_message(
            text=dispatch_text, chat_id=channel_id, chat_type=chat_type, user_id=pubkey,
            user_name=await self._resolve_user_name(pubkey), message_id=event_id,
            created_at=created_at, thread_id=thread_id, reply_to_message_id=reply_parent_id,
            reply_to_text=reply_meta[1] if reply_meta else None,
            reply_to_author_id=reply_meta[0] if reply_meta else None,
            reply_to_is_own_message=reply_to_is_own,
            media_urls=[attachment.path for attachment in attachments],
            media_types=[attachment.media_type for attachment in attachments],
            message_type=message_type, raw_message=event,
        )

    # ── DM classification ─────────────────────────────────────────────────
    # DMs can leak in through ``channels list`` as chat_type="group" (see
    # _discover_dms). In a real channel a p-tag is only an addressing signal
    # and must wake the agent without changing the conversation type.

    def _may_reclassify_as_dm(self, channel_id: str) -> bool:
        """True when metadata does not rule out a DM: named "DM" with empty description.
        Real channels never become DMs from a p-tag; missing metadata fails closed."""
        meta = self._channel_meta.get(channel_id)
        if meta is None:
            return False
        return str(meta.get("name") or "").strip() == "DM" and not str(meta.get("description") or "").strip()

    def _p_tagged_to_self(self, event: dict) -> bool:
        """True when the signed event addresses this identity by pubkey."""
        tags = event.get("tags")
        return bool(self._self_pubkey) and isinstance(tags, list) and any(
            isinstance(tag, (list, tuple))
            and len(tag) > 1
            and tag[0] == "p"
            and str(tag[1]).lower() == self._self_pubkey
            for tag in tags
        )

    def _is_direct_message_event(self, channel_id: str, event: dict) -> bool:
        """True for a kind-9 message from another user, p-tagged to us, whose text does
        NOT mention us — structural DM addressing, not the artifact of a typed @mention."""
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        return bool(
            self._self_pubkey and self._may_reclassify_as_dm(channel_id)
            and int(event.get("kind") or 0) == _CHAT_KIND
            and pubkey and pubkey != self._self_pubkey
            and self._p_tagged_to_self(event)
            and isinstance(content, str) and not self._is_mentioned(content)
        )

    def _maybe_latch_dm(self, channel_id: str, state: dict, event: dict) -> None:
        """Latch a group conversation to "dm" once a direct message is seen; it sticks."""
        if state["chat_type"] == "dm" or not self._is_direct_message_event(channel_id, event):
            return
        state["chat_type"] = "dm"
        self._channel_names.setdefault(channel_id, "DM")
        logger.info("Buzz: conversation %s reclassified as DM (message p-tagged to self)", channel_id)

    def _is_mentioned(self, content: str) -> bool:
        """True when text explicitly addresses this agent (npub, hex, or @name)."""
        lowered = content.lower()
        patterns = []
        if self._self_pubkey and re.fullmatch(r"[0-9a-f]{64}", self._self_pubkey):
            patterns.append(rf"(?<![0-9a-f]){re.escape(self._self_pubkey)}(?![0-9a-f])")
        if self._self_npub:
            patterns.append(rf"(?<![a-z0-9]){re.escape(self._self_npub.lower())}(?![a-z0-9])")
        if self._display_name:
            patterns.append(rf"(?<![\w@])@{re.escape(self._display_name.lower())}" + r"(?=$|[\s,;.!?:)\]}])")
        return any(re.search(p, lowered) for p in patterns)

    def _is_addressed(self, event: dict) -> bool:
        """True when a group event carries an explicit text or p-tag address."""
        content = event.get("content")
        return isinstance(content, str) and (self._is_mentioned(content) or self._p_tagged_to_self(event))

    def _strip_mention(self, content: str) -> str:
        """Strip a LEADING @mention of this agent (case-insensitive) so the gateway's
        ``is_command()`` sees "/whoami"; mid-sentence mentions are left intact."""
        text = content.strip()
        candidates = []
        if self._display_name:
            candidates.append(rf"@{re.escape(self._display_name)}" + r"(?=$|[\s,;.!?:)\]}])")
        if self._self_npub:
            candidates.append(rf"@?{re.escape(self._self_npub)}(?![a-z0-9])")
        if self._self_pubkey:
            candidates.append(rf"@?{re.escape(self._self_pubkey)}(?![0-9a-f])")
        if not candidates:
            return text
        # Display names require '@'; npub/hex identities are unambiguous without it.
        pattern = rf"^(?:{'|'.join(candidates)})[\s:,]*"
        stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        return stripped.strip()

    async def _resolve_user_name(self, pubkey: str) -> str:
        """Pubkey -> display name, cached (negatively too, so a profile-less pubkey
        doesn't re-run ``users get`` every sweep); falls back to the npub prefix."""
        cached = self._user_names.get(pubkey)
        if cached is not None:
            return cached
        name = ""
        code, out, _err = await self._run_cli(["users", "get", "--pubkey", pubkey])
        if code == 0:
            profiles = _parse_json_list(out)
            if profiles:
                name = str(profiles[0].get("display_name") or "").strip()
        if not name:
            name = (hex_to_npub(pubkey) or pubkey)[:16]
        self._user_names[pubkey] = name
        return name

    @staticmethod
    def _trim_seen(state: dict) -> None:
        seen = state["seen"]
        while len(seen) > _SEEN_CAP:
            seen.popitem(last=False)
        meta = state.get("event_meta")
        if isinstance(meta, OrderedDict):
            while len(meta) > _SEEN_CAP:
                meta.popitem(last=False)

    def _mark_seen(self, channel_id: str, event_id: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is not None:
            state["seen"][event_id] = None
            self._trim_seen(state)

    # ── Thread anchoring ──────────────────────────────────────────────────
    # NIP-10: a reply carries ["e", <root>, "", "root"] + ["e", <parent>, "", "reply"];
    # a thread STARTER carries a single ["e", <parent>, "", "reply"]. The gateway
    # anchors replies to the triggering message id, which inside a thread would
    # nest a new sub-thread under every answer — so remember each inbound
    # message's ROOT and reply against that when the trigger was in a thread.

    _THREAD_ROOT_CACHE = 512

    @staticmethod
    def _extract_thread_root(event: dict) -> Optional[str]:
        """Return the NIP-10 thread root of ``event``, or None if top-level."""
        tags = event.get("tags")
        if not isinstance(tags, list):
            return None
        root = None
        reply = None
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or len(tag) < 2 or str(tag[0]) != "e":
                continue
            marker = str(tag[3]).lower() if len(tag) > 3 else ""
            if marker == "root":
                root = str(tag[1])
            elif marker == "reply":
                reply = str(tag[1])
            elif not marker and reply is None:
                reply = str(tag[1])  # unmarked (deprecated positional) e-tag = parent
        # A lone "reply" e-tag started a thread off <reply>; that parent IS the root.
        return root or reply

    def _record_thread_root(self, event_id: str, event: dict) -> None:
        """Cache the thread root for an inbound message id."""
        if not event_id:
            return
        roots = self._thread_roots
        roots[event_id] = self._extract_thread_root(event)
        roots.move_to_end(event_id)
        while len(roots) > self._THREAD_ROOT_CACHE:
            roots.popitem(last=False)

    def _resolve_reply_anchor(self, anchor: Optional[str]) -> Optional[str]:
        """Thread root when the trigger was inside a thread (reply joins it), else the anchor unchanged."""
        if not anchor:
            return anchor
        return self._thread_roots.get(str(anchor)) or anchor

    def _remember_event(self, state: dict, event: dict) -> None:
        """Record author + content snippet for later NIP-10 parent lookup."""
        event_id = str(event.get("id") or "")
        content = event.get("content")
        if event_id:
            self._store_event_meta(
                state, event_id, str(event.get("pubkey") or "").lower(),
                content[:_EVENT_META_CONTENT_CAP] if isinstance(content, str) else "",
            )

    def _remember_event_meta(self, channel_id: str, event_id: str, pubkey: str, content: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is not None and event_id:
            self._store_event_meta(state, event_id, (pubkey or "").lower(), (content or "")[:_EVENT_META_CONTENT_CAP])

    @staticmethod
    def _store_event_meta(state: dict, event_id: str, pubkey: str, snippet: str) -> None:
        cache = state.setdefault("event_meta", OrderedDict())
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict(cache)
            state["event_meta"] = cache
        cache[event_id] = (pubkey, snippet)
        cache.move_to_end(event_id)
        while len(cache) > _SEEN_CAP:
            cache.popitem(last=False)

    @staticmethod
    def _lookup_event_meta(state: dict, event_id: Optional[str]) -> Optional[Tuple[str, str]]:
        if not event_id:
            return None
        cache = state.get("event_meta") or {}
        entry = cache.get(event_id)
        if not entry or not isinstance(entry, tuple) or len(entry) < 2:
            return None
        return str(entry[0] or ""), str(entry[1] or "")

    async def _localize_inbound_media(
        self, text: str, message_id: str, *, user_id: str = "",
        chat_type: Optional[str] = None, chat_id: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], MessageType]:
        """Authenticate and cache same-relay media references in *text* (each object
        independent; failures are skipped). Spends this agent's credentials on a
        sender-chosen URL, so it runs only on the gateway's explicit ``True``."""
        urls, replacements = _find_relay_media_refs(text, self.relay_url)
        if not urls:
            return text, [], [], MessageType.TEXT
        if self._is_sender_authorized(user_id, chat_type, chat_id) is not True:
            logger.warning(
                "Buzz: not localizing %d media reference(s) in message %s — "
                "sender %s… is not explicitly authorized",
                len(urls), message_id[:12], (user_id or "?")[:8],
            )
            return text, [], [], MessageType.TEXT
        cleaned_text = _replace_media_refs(text, replacements)
        media_urls: List[str] = []
        media_types: List[str] = []
        media_kinds: List[str] = []
        from gateway.platforms.base import cache_media_bytes, validate_inbound_media_size
        for url in urls:
            path_match = _MEDIA_PATH_RE.fullmatch(urlsplit(url).path)
            if path_match is None:
                continue
            ext = (path_match.group("ext") or ".bin").lower()
            label = f"{path_match.group('sha')[:12]}{ext}"
            try:
                with tempfile.TemporaryDirectory(prefix="hermes-buzz-media-") as temp_dir:
                    download_path = Path(temp_dir) / f"buzz_{label}"
                    code, _out, _err = await self._run_cli(["media", "get", "-o", str(download_path), url])
                    if code != 0 or not download_path.is_file():
                        logger.warning("Buzz: failed to localize inbound media %s (exit %d)", label, code)
                        continue
                    validate_inbound_media_size(download_path.stat().st_size, media_type="Buzz media")
                    mime_type = mimetypes.guess_type(download_path.name)[0] or "application/octet-stream"
                    cached = cache_media_bytes(
                        download_path.read_bytes(), filename=download_path.name, mime_type=mime_type
                    )
            except Exception as exc:
                logger.warning("Buzz: failed to localize inbound media %s (%s)", label, type(exc).__name__)
                continue
            if cached is None:
                logger.warning("Buzz: rejected invalid inbound media %s", label)
                continue
            media_urls.append(cached.path)
            media_types.append(cached.media_type)
            media_kinds.append(cached.kind)
        if media_urls:
            logger.info("Buzz: localized %d inbound media attachment(s) for message %s", len(media_urls), message_id[:12])
        # Priority order: image > audio > video > any other kind.
        message_type = MessageType.TEXT if not media_kinds else next(
            (mt for kind, mt in _MEDIA_KIND_PRIORITY if kind in media_kinds), MessageType.DOCUMENT
        )
        if not cleaned_text:
            cleaned_text = "(attachment)" if media_urls else "(Buzz media attachment unavailable)"
        return cleaned_text, media_urls, media_types, message_type

    async def _dispatch_message(
        self, text: str, chat_id: str, chat_type: str, user_id: str, user_name: str,
        message_id: str, created_at: int, thread_id: Optional[str] = None,
        reply_to_message_id: Optional[str] = None, reply_to_text: Optional[str] = None,
        reply_to_author_id: Optional[str] = None, reply_to_is_own_message: bool = False,
        media_urls: Optional[List[str]] = None, media_types: Optional[List[str]] = None,
        message_type: MessageType = MessageType.TEXT, raw_message: Any = None,
    ) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return
        media_urls = list(media_urls or [])
        media_types = list(media_types or [])
        # Same-relay URL references in the text are localized in addition to the
        # native imeta attachments already cached by the caller (both gated on
        # the gateway's explicit-True authorization).
        text, localized_urls, localized_types, localized_type = await self._localize_inbound_media(
            text, message_id, user_id=user_id, chat_type=chat_type, chat_id=chat_id
        )
        for path, mime in zip(localized_urls, localized_types):
            if path not in media_urls:
                media_urls.append(path)
                media_types.append(mime)
        if message_type == MessageType.TEXT:
            message_type = localized_type
        elif localized_urls and localized_type not in (message_type, MessageType.TEXT):
            # Mixed sources use document semantics so audio isn't routed to STT.
            message_type = MessageType.DOCUMENT
        source = self.build_source(
            chat_id=chat_id, chat_name=self._channel_names.get(chat_id, chat_id), chat_type=chat_type,
            user_id=user_id, user_name=user_name, thread_id=thread_id,
        )
        event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=raw_message,
            message_id=message_id, media_urls=list(media_urls or []),
            media_types=list(media_types or []), media_text_inlined=[False] * len(media_urls or []),
            timestamp=datetime.fromtimestamp(created_at) if created_at else datetime.now(),
            reply_to_message_id=reply_to_message_id, reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author_id, reply_to_is_own_message=reply_to_is_own_message,
        )
        await self.handle_message(event)
        # "Seen" reaction: signals the message was received and is being processed.
        try:
            await self.send_reaction(chat_id, message_id, "👀")
        except Exception:
            logger.debug("Buzz: reaction failed for message %s", message_id[:12], exc_info=True)


# ── Plugin registration ──────────────────────────────────────────────────────

def _profile_buzz_extra() -> dict:
    """``buzz.extra`` from the scoped profile's config.yaml, for ``check_requirements``
    (which has no PlatformConfig). Best-effort: failures yield {} and callers fail closed."""
    if not _profile_scoped():
        return {}
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import read_user_config_raw
        cfg = read_user_config_raw(Path(get_hermes_home()) / "config.yaml")
    except Exception:
        return {}
    buzz = ((cfg.get("gateway") or {}).get("platforms") or {}).get("buzz") if isinstance(cfg, dict) else None
    extra = buzz.get("extra", buzz) if isinstance(buzz, dict) else None
    return extra if isinstance(extra, dict) else {}


def check_requirements() -> bool:
    """Check if Buzz is configured: a relay URL plus a resolvable key."""
    if _profile_scoped():
        # Secondary profile: os.environ's BUZZ_* are the default profile's bridge
        # output and must not satisfy the gate; an unconfigured profile fails closed.
        extra = _profile_buzz_extra()
        relay = str(extra.get("relay_url") or "").strip()
        return bool(relay and _resolve_private_key(extra))
    # The gate runs before per-profile scopes install; the relay can be externally managed too.
    if not (_get_scoped_secret("BUZZ_RELAY_URL", "") or "").strip():
        return False
    return bool(_resolve_private_key())


def validate_config(config) -> bool:
    """Validate that the platform config has enough information to connect."""
    extra = getattr(config, "extra", {}) or {}
    # Scoped: extra is authoritative; unscoped: env read gains the external-secret rung.
    if _profile_scoped():
        relay = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
        relay = relay if relay is not None else extra.get("relay_url", "")
    else:
        relay = _get_scoped_secret("BUZZ_RELAY_URL", "") or extra.get("relay_url", "")
    return bool(relay and _resolve_private_key(extra))


def is_connected(config) -> bool:
    """Check whether Buzz is configured (env or config.yaml)."""
    return validate_config(config)


# (extra key, env var, kind): "str" bridges truthy values as-is, "csv" joins lists,
# "flag" lowercases when present, "thread" lowercases and ignores profile scope.
_YAML_BRIDGE = (
    ("relay_url", "BUZZ_RELAY_URL", "str"),
    ("cli_path", "BUZZ_CLI_PATH", "str"),
    ("home_channel", "BUZZ_HOME_CHANNEL", "str"),
    ("transport", "BUZZ_TRANSPORT", "str"),
    ("channels", "BUZZ_CHANNELS", "csv"),
    ("allowed_users", "BUZZ_ALLOWED_USERS", "csv"),
    ("reaction_only_users", "BUZZ_REACTION_ONLY_USERS", "csv"),
    ("allow_all_users", "BUZZ_ALLOW_ALL_USERS", "flag"),
    ("require_mention", "BUZZ_REQUIRE_MENTION", "flag"),
    ("reply_in_thread", "BUZZ_REPLY_IN_THREAD", "thread"),
    ("reply_to_mode", "BUZZ_REPLY_TO_MODE", "thread"),
)


def _apply_yaml_config(yaml_cfg: dict, buzz_cfg: dict) -> Optional[dict]:
    """Bridge ``buzz.extra`` into ``BUZZ_*`` env (``apply_yaml_config_fn``) so a
    config.yaml-only setup passes the env-reading gate. Env wins over YAML;
    ``BUZZ_PRIVATE_KEY`` is never sourced from config.yaml."""
    extra = buzz_cfg.get("extra", buzz_cfg) or {}
    if not isinstance(extra, dict):
        return None
    # A secondary multiplex profile must NOT write to the process-global env
    # (first-writer-wins would pin its values for every profile); its adapter
    # reads PlatformConfig.extra directly instead.
    skip_env_bridge = _profile_scoped()
    interval = extra.get("poll_interval")
    if interval is not None and not skip_env_bridge and not os.getenv("BUZZ_POLL_INTERVAL"):
        os.environ["BUZZ_POLL_INTERVAL"] = str(interval)
    for src, env, kind in _YAML_BRIDGE:
        val = extra.get(src)
        missing = {"str": not val, "csv": val is None}.get(kind, src not in extra)
        if missing or (kind != "thread" and skip_env_bridge) or os.getenv(env):
            continue
        if kind == "csv" and isinstance(val, (list, tuple)):
            val = ",".join(str(v) for v in val)
        os.environ[env] = str(val).lower() if kind in ("flag", "thread") else str(val)
    return None


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env so env-only setups show in gateway
    status; ``None`` when not minimally configured."""
    if _profile_scoped():
        # Process env holds the default profile's BUZZ_*; never fabricate a
        # Buzz platform for a secondary profile that did not configure one.
        return None
    relay = os.getenv("BUZZ_RELAY_URL", "").strip()
    if not relay or not _resolve_private_key():
        return None
    seed: dict = {"relay_url": relay}
    channels = os.getenv("BUZZ_CHANNELS", "").strip()
    if channels:
        seed["channels"] = [c.strip() for c in channels.split(",") if c.strip()]
    interval = os.getenv("BUZZ_POLL_INTERVAL", "").strip()
    if interval:
        try:
            seed["poll_interval"] = float(interval)
        except ValueError:
            pass
    cli_path = os.getenv("BUZZ_CLI_PATH", "").strip()
    if cli_path:
        seed["cli_path"] = cli_path
    # Cron delivery target; defaults to the first watched channel.
    home = os.getenv("BUZZ_HOME_CHANNEL", "").strip() or (seed.get("channels") or [""])[0]
    if home:
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("BUZZ_HOME_CHANNEL_NAME", home)}
    return seed


async def _standalone_send(
    pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """One-shot send without a live adapter (out-of-process ``deliver=buzz`` cron)."""
    extra = getattr(pconfig, "extra", {}) or {}
    relay = _configured_relay(extra)
    private_key = _resolve_private_key(extra)
    try:
        auth_tag = _resolve_auth_tag(extra)
    except ValueError as exc:
        return {"error": f"Buzz standalone send: {exc}"}
    cli_path = _configured_cli_path(extra)
    if not relay or not private_key:
        return {"error": "Buzz standalone send: BUZZ_RELAY_URL and BUZZ_PRIVATE_KEY must be configured"}
    if not cli_path:
        return {"error": "Buzz standalone send: buzz CLI binary not found"}
    target = (chat_id or "").strip() or _configured_home_channel(extra)
    if not target:
        return {"error": "Buzz standalone send: no target channel (set BUZZ_HOME_CHANNEL)"}
    args = ["messages", "send", "--channel", target, "--content", "-"]
    # Same reply_to_mode / reply_in_thread gate as the live adapter.
    if thread_id and _reply_to_mode(pconfig, extra) != "off":
        args += ["--reply-to", str(thread_id)]
    for media in media_files or []:
        path = media[0] if isinstance(media, (list, tuple)) and media else media
        args += ["--file", str(path)]
    try:
        code, out, err = await _exec_buzz(
            cli_path, args, relay_url=relay, private_key=private_key, auth_tag=auth_tag, input_text=message
        )
        if code != 0:
            escaped = _escape_unresolved_presentation_mention(message, err)
            if escaped is not None:
                logger.info("Buzz: retrying standalone message after unresolved presentation-mention preflight")
                # Retry intentionally omits auth_tag (legacy behavior).
                code, out, err = await _exec_buzz(
                    cli_path, args, relay_url=relay, private_key=private_key, input_text=escaped
                )
    except asyncio.CancelledError:
        raise
    except OSError as e:
        detail = _bounded_cli_message(str(e))
        return {"error": f"Buzz standalone send failed to launch CLI: {detail}"}
    if code != 0:
        return {"error": f"Buzz standalone send failed: {_cli_error_message(err, code)}"}
    event_id, receipt_error = _parse_send_receipt(out)
    if receipt_error:
        return {"error": f"Buzz standalone send failed: {receipt_error}"}
    result = {"success": True, "message_id": event_id}
    if media_files:
        result["media_delivered"] = True
    return result


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow (lazy CLI imports keep the plugin importable elsewhere)."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value, print_header, print_info, print_warning, print_success,
    )
    def ask(label: str, env: str) -> str:
        return prompt(label, default=get_env_value(env) or "")

    print_header("Buzz")
    existing_relay = get_env_value("BUZZ_RELAY_URL")
    if existing_relay:
        print_info(f"Buzz: already configured (relay: {existing_relay})")
        if not prompt_yes_no("Reconfigure Buzz?", False):
            return
    print_info("Connect Hermes to a Buzz community (Block's Nostr-based human+agent platform).")
    print_info("   Requires the buzz CLI binary and a Nostr key that is a community member.")
    print()
    relay = prompt("Relay URL (e.g. https://mycommunity.communities.buzz.xyz)", default=existing_relay or "")
    if not relay:
        print_warning("Relay URL is required — skipping Buzz setup")
        return
    save_env_value("BUZZ_RELAY_URL", relay.strip())
    key = prompt("Nostr private key (nsec or hex; leave blank to keep current)", password=True)
    if key:
        save_env_value("BUZZ_PRIVATE_KEY", key.strip())
    elif not _resolve_private_key():
        print_warning("No private key configured — set BUZZ_PRIVATE_KEY before starting the gateway")
    channels = ask("Channel UUIDs to watch (comma-separated, empty = all joined channels)", "BUZZ_CHANNELS")
    if channels:
        save_env_value("BUZZ_CHANNELS", channels.replace(" ", ""))
    home = ask("Home channel UUID for cron/notification delivery (optional)", "BUZZ_HOME_CHANNEL")
    if home:
        save_env_value("BUZZ_HOME_CHANNEL", home.strip())
    print()
    print_info("🔒 Access control: restrict who can talk to the agent")
    allow_all = prompt_yes_no("Allow all community members to talk to the agent?", False)
    if allow_all:
        save_env_value("BUZZ_ALLOW_ALL_USERS", "true")
        save_env_value("BUZZ_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — anyone in the community can command the agent.")
    else:
        save_env_value("BUZZ_ALLOW_ALL_USERS", "false")
        allowed = ask("Allowed users (comma-separated npubs or hex pubkeys, empty to deny everyone)", "BUZZ_ALLOWED_USERS")
        save_env_value("BUZZ_ALLOWED_USERS", allowed.replace(" ", "") if allowed else "")
    print()
    print_success("Buzz configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="buzz",
        label="Buzz",
        adapter_factory=lambda cfg: BuzzAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"],
        install_hint="Requires the buzz CLI binary (https://github.com/block/buzz) on PATH or at BUZZ_CLI_PATH",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="BUZZ_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
        emoji="🐝",
        pii_safe=False,  # identities are pubkeys, not phone numbers
        allow_update_command=True,
        platform_hint=(
            "You are collaborating in a Buzz workspace (Block's Nostr-based "
            "human+agent platform). Markdown IS supported. Users address you "
            "by @-mentioning your name or npub in channels; direct messages "
            "reach you without a mention. Keep responses conversational."
        ),
    )
