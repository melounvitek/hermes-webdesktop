#!/usr/bin/env python3
"""MCP OAuth 2.1 client support: browser authorization-code flow with PKCE.

The MCP SDK's ``OAuthClientProvider`` (an ``httpx.Auth``) does discovery, client
identification, PKCE, exchange, refresh and step-up; this module supplies
``HermesTokenStorage`` (on-disk persistence), the localhost callback listener,
and ``build_oauth_auth()`` (legacy entry point). Per the MCP 2026-07-28 spec the
client_id is Hermes' published Client ID Metadata Document URL (CIMD) when the
server advertises support, else RFC 7591 dynamic client registration (DCR).

``mcp_servers.<name>.oauth`` keys (all optional): ``client_id`` (skip DCR),
``client_secret`` (confidential clients), ``scope``, ``redirect_port`` (0 =
auto), ``redirect_uri`` (proxy callback), ``redirect_host`` (loopback hostname,
WAF-safe), ``client_name`` (default "Hermes Agent"), ``client_metadata_url``
(self-hosted CIMD), ``cimd: false`` (force DCR), ``user_agent``, ``timeout``.
"""

import asyncio
import contextlib
import contextvars
import importlib.util as _importlib_util
import json
import logging
import os
import re
import secrets
import socket
import stat
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from hermes_constants import secure_parent_dir
from tools.mcp_dashboard_oauth import contextvar_set as _contextvar_set, get_dashboard_oauth_flow

logger = logging.getLogger(__name__)

# Lazy SDK imports: availability is detected WITHOUT importing mcp (~170 ms).
# Module-level names stay None placeholders so tests can patch.object them;
# _ensure_sdk_loaded() binds the real classes on first use. _SDK_CLASSES caches
# them so a test that patches a name and restores it to None afterwards doesn't
# strand the module in a broken state.
_OAUTH_AVAILABLE = _importlib_util.find_spec("mcp") is not None
if not _OAUTH_AVAILABLE:
    logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")

OAuthClientProvider: Any = None
OAuthClientInformationFull: Any = None
OAuthClientMetadata: Any = None
OAuthMetadata: Any = None
OAuthToken: Any = None

_SDK_CLASSES: dict[str, Any] = {}
_SDK_LOAD_FAILED = False


def _ensure_sdk_loaded() -> bool:
    """Bind the SDK OAuth classes into module globals; True when available.
    Only names currently ``None`` are (re)bound, so test patches stay intact."""
    global _SDK_LOAD_FAILED, _OAUTH_AVAILABLE
    if _SDK_LOAD_FAILED:
        return False
    if not _SDK_CLASSES:
        try:
            from mcp.client import auth as _client_auth
            from mcp.shared import auth as _shared_auth

            _SDK_CLASSES["OAuthClientProvider"] = _client_auth.OAuthClientProvider
            for _name in ("OAuthClientInformationFull", "OAuthClientMetadata", "OAuthMetadata", "OAuthToken"):
                _SDK_CLASSES[_name] = getattr(_shared_auth, _name)
        except (ImportError, AttributeError):
            _SDK_CLASSES.clear()
            _SDK_LOAD_FAILED = True
            _OAUTH_AVAILABLE = False
            logger.debug("MCP OAuth types not available -- OAuth MCP auth disabled")
            return False
    g = globals()
    for _name, _cls in _SDK_CLASSES.items():
        if g.get(_name) is None:
            g[_name] = _cls
    return True


def _sdk_class(name: str) -> Any:
    """Return the (possibly test-patched) SDK class bound under *name*, or None."""
    if globals().get(name) is None and not _ensure_sdk_loaded():
        return None
    return globals().get(name)


try:
    from pydantic import AnyUrl
except ImportError:
    AnyUrl = None  # type: ignore[assignment, misc]


class OAuthNonInteractiveError(RuntimeError):
    """Raised when OAuth requires browser interaction in a non-interactive env."""


# Port used by the most recent callback-port resolution. Legacy global; the
# per-flow closures are the real mechanism (concurrent flows must not share it).
_oauth_port: int | None = None

# Interactivity gates for OAuth stdin prompts. ContextVars (NOT threading.local):
# background MCP discovery sets them on the discovery thread while connect+OAuth
# runs on the `mcp-event-loop` thread via run_coroutine_threadsafe, which copies
# the calling context into the coroutine — a threading.local would not cross.
_oauth_interactive_enabled = contextvars.ContextVar("_oauth_interactive_enabled", default=True)
# Forces _is_interactive() past the stdin-TTY check for GUI-driven flows
# (dashboard/desktop REST): the browser + callback server do the work and the
# stdin paste fallback degrades harmlessly (EOF swallowed). Suppression wins —
# background discovery must never start a browser flow.
_oauth_interactive_forced = contextvars.ContextVar("_oauth_interactive_forced", default=False)

# Skip tokens accepted at the paste prompt — exit OAuth without auth.
_SKIP_TOKENS = frozenset({"skip", "cancel", "s", "n", "no", "q", "quit"})
# Written to result["error"] on stdin skip; the waiter maps it to
# OAuthNonInteractiveError("user_skipped") so MCP setup treats it as a
# non-fatal "continue without this server".
_USER_SKIPPED_SENTINEL = "__hermes_user_skipped__"


def _get_token_dir(hermes_home: str | Path | None = None) -> Path:
    """``HERMES_HOME/mcp-tokens/`` — per-profile token directory."""
    from hermes_constants import get_hermes_home

    return Path(hermes_home if hermes_home is not None else get_hermes_home()) / "mcp-tokens"


def _safe_filename(name: str) -> str:
    """Sanitize a server name for use as a filename (no path separators)."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")[:128] or "default"


# -- Callback-port reservation ---------------------------------------------
# Bound-but-not-listening sockets for pending callback flows, keyed by port.
# Holding the socket from port selection until the waiter adopts it closes the
# TOCTOU window where another process grabs the port in between. Bounded FIFO
# so reconnect loops cannot leak fds.
_reserved_sockets: "dict[int, socket.socket]" = {}
_MAX_RESERVED_SOCKETS = 8


def _park_reserved_socket(port: int, sock: socket.socket) -> None:
    """Hold *sock* bound to *port* until the callback waiter adopts it.

    Pinned CIMD sockets are never evicted: the published document only declares
    the pinned ports, so losing one mid-flow would reopen the exact race the
    parking prevents. The FIFO cap applies to ephemeral ports only.
    """
    while len(_reserved_sockets) >= _MAX_RESERVED_SOCKETS:
        stale_port = next((p for p in _reserved_sockets if p not in _CIMD_PORTS), None)
        if stale_port is None:
            break  # only pinned sockets remain — never evict those
        try:
            _reserved_sockets.pop(stale_port).close()
        except OSError:
            pass
    _reserved_sockets[port] = sock


def _bind_reserved(port: int) -> int | None:
    """Bind ``127.0.0.1:port`` (0 = ephemeral) and park it; None if taken."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        if port:
            return None
        raise
    bound = sock.getsockname()[1]
    _park_reserved_socket(bound, sock)
    return bound


def _reserve_callback_port() -> int:
    """Pick an ephemeral callback port and keep its socket bound (parked)."""
    return _bind_reserved(0)  # type: ignore[return-value]  # port 0 never returns None


# -- Cached registration lookups ---------------------------------------------
def _cached_client_info(storage: "HermesTokenStorage | None") -> dict | None:
    """The on-disk client registration for *storage*, or None."""
    try:
        return _read_json(storage._client_info_path()) if storage is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _cached_redirect_uris(storage: "HermesTokenStorage | None"):
    """Yield ``(raw_uri, parsed)`` for each redirect URI in the cached registration."""
    for uri in (_cached_client_info(storage) or {}).get("redirect_uris") or []:
        with contextlib.suppress(TypeError, ValueError):
            yield str(uri), urlparse(str(uri))


def _cached_redirect_port(storage: "HermesTokenStorage | None") -> int | None:
    """Loopback callback port from the cached client registration.

    Providers bind a dynamically-registered ``client_id`` to the exact redirect
    URI registered with it; a new random port on restart under the stored
    ``client_id`` gets ``redirect_uri does not match any registered URIs``.
    """
    for _uri, parsed in _cached_redirect_uris(storage):
        is_loopback_callback = parsed.scheme == "http" and parsed.path == "/callback" and parsed.hostname in {"127.0.0.1", "localhost"}
        if is_loopback_callback and parsed.port is not None:
            return int(parsed.port)
    return None


def _cached_redirect_uri(storage: "HermesTokenStorage | None") -> str | None:
    """A cached non-loopback (https) redirect URI, if one was registered."""
    for uri, parsed in _cached_redirect_uris(storage):
        if parsed.scheme == "https" and parsed.netloc:
            return uri
    return None


# -- Interactivity -----------------------------------------------------------
def _is_interactive() -> bool:
    """True if we can reasonably expect to interact with a user."""
    if not _oauth_interactive_enabled.get():
        return False
    if _oauth_interactive_forced.get():
        return True
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _raise_if_non_interactive(lead: str) -> None:
    """Raise ``OAuthNonInteractiveError`` unless interactive; *lead* is the
    boundary-specific first sentence, the ``hermes mcp login`` next step is shared."""
    if not _is_interactive():
        raise OAuthNonInteractiveError(
            f"{lead} Run `hermes mcp login <server>` interactively to (re)authorize, "
            "then restart or reload the gateway."
        )


def force_interactive_oauth():
    """Treat the current context as interactive despite no TTY (GUI-driven auth):
    the user IS present, just not on stdin. Crosses the MCP event-loop thread
    like ``suppress_interactive_oauth``."""
    return _contextvar_set(_oauth_interactive_forced, True)


def suppress_interactive_oauth():
    """Disable stdin-based OAuth prompts for the current execution context;
    ContextVar-based so a background-discovery thread's suppression reaches the
    coroutine scheduled on the MCP event-loop thread."""
    return _contextvar_set(_oauth_interactive_enabled, False)


def _can_open_browser() -> bool:
    """True if opening a browser is likely to work."""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False  # explicit SSH session → no local display
    if os.name == "nt" or (hasattr(os, "uname") and os.uname().sysname == "Darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# -- JSON file I/O ------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _write_json(path: Path, data: dict) -> None:
    """Atomically write *data* as JSON created at 0o600.

    ``os.open`` with ``O_EXCL`` + explicit mode avoids the write-then-chmod TOCTOU
    window where the file briefly inherits the umask (often world-readable). The
    parent dir is tightened to 0o700 (no-op on Windows; ``secure_parent_dir``
    refuses /, top-level dirs and the install tree). The per-process random tmp
    suffix avoids collisions between concurrent writers and stale crash leftovers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


# -- HermesTokenStorage -- persistent token/client-info on disk --------------
class HermesTokenStorage:
    """Persist OAuth tokens and client registration to JSON files.

    File layout::

        HERMES_HOME/mcp-tokens/<server_name>.json         -- tokens
        HERMES_HOME/mcp-tokens/<server_name>.client.json   -- client info
        HERMES_HOME/mcp-tokens/<server_name>.meta.json     -- oauth server metadata
        HERMES_HOME/mcp-tokens/<server_name>.cimd-off      -- CIMD refused here
    """

    def __init__(self, server_name: str, *, hermes_home: str | Path | None = None):
        self._server_name = _safe_filename(server_name)
        self._hermes_home = Path(hermes_home) if hermes_home is not None else None

    def _path(self, suffix: str) -> Path:
        return _get_token_dir(self._hermes_home) / f"{self._server_name}{suffix}"

    def _tokens_path(self) -> Path:
        return self._path(".json")

    def _client_info_path(self) -> Path:
        return self._path(".client.json")

    def _meta_path(self) -> Path:
        return self._path(".meta.json")

    def _cimd_rejected_path(self) -> Path:
        return self._path(".cimd-off")

    def _state_paths(self) -> tuple[Path, Path, Path]:
        return self._tokens_path(), self._client_info_path(), self._meta_path()

    @staticmethod
    def _load_model(path: Path, sdk_name: str, label: str, fixup=None):
        """Read *path* into SDK model *sdk_name*; None if absent, no SDK, or corrupt.
        ``fixup(data)`` may rewrite the raw dict before validation."""
        data = _read_json(path)
        cls = _sdk_class(sdk_name) if data is not None else None
        if cls is None:
            return None
        if fixup is not None:
            fixup(data)
        try:
            return cls.model_validate(data)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Corrupt %s at %s -- ignoring: %s", label, path, exc)
            return None

    # -- tokens ------------------------------------------------------------

    def _rebase_expires_in(self, data: dict) -> None:
        """Rewrite ``expires_in`` to the seconds remaining now.

        ``set_tokens`` stores an absolute ``expires_at`` (not an SDK field, so it
        is stripped here); a relative value reloaded after restart would make
        ``is_token_valid()`` True for tokens that expired while down. Legacy files
        without it use the file mtime as a best-effort write time, clamped to
        zero (self-heals on the next ``set_tokens``).
        """
        absolute_expiry = data.pop("expires_at", None)
        if absolute_expiry is not None:
            data["expires_in"] = int(max(absolute_expiry - time.time(), 0))
        elif data.get("expires_in") is not None:
            with contextlib.suppress(OSError, TypeError, ValueError):
                implied_expiry = self._tokens_path().stat().st_mtime + int(data["expires_in"])
                data["expires_in"] = int(max(implied_expiry - time.time(), 0))

    async def get_tokens(self) -> "OAuthToken | None":
        return self._load_model(self._tokens_path(), "OAuthToken", "tokens", self._rebase_expires_in)

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        payload = tokens.model_dump(mode="json", exclude_none=True)
        # Persist an absolute ``expires_at``: a relative ``expires_in`` reloaded
        # after restart has no wall-clock reference, leaving the SDK's
        # ``token_expiry_time=None`` and ``is_token_valid()`` falsely True.
        if payload.get("expires_in") is not None:
            with contextlib.suppress(TypeError, ValueError):  # mock tokens / odd shapes: skip, don't fail persistence
                payload["expires_at"] = time.time() + int(payload["expires_in"])
        _write_json(self._tokens_path(), payload)
        logger.debug("OAuth tokens saved for %s", self._server_name)

    # -- client info -------------------------------------------------------

    @staticmethod
    def _coerce_secret_auth_method(data: dict) -> bool:
        """Set ``client_secret_post`` when a secret is present but no method is.

        Some DCR providers (notably Supabase) return a ``client_secret`` but omit
        ``token_endpoint_auth_method``; the SDK defaults it to ``none``, omits the
        secret at the token endpoint and the exchange fails.
        """
        if data.get("client_secret") and data.get("token_endpoint_auth_method") in (None, "none", ""):
            data["token_endpoint_auth_method"] = "client_secret_post"
            return True
        return False

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        coerced: list[bool] = []
        info = self._load_model(
            self._client_info_path(), "OAuthClientInformationFull", "client info",
            lambda data: coerced.append(self._coerce_secret_auth_method(data)),
        )
        if info is not None and coerced and coerced[0]:
            # Persist the effective method so later flows skip the coercion.
            _write_json(self._client_info_path(), info.model_dump(mode="json", exclude_none=True))
        return info

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        data = client_info.model_dump(mode="json", exclude_none=True)
        self._coerce_secret_auth_method(data)
        _write_json(self._client_info_path(), data)
        logger.debug("OAuth client info saved for %s", self._server_name)

    # -- oauth server metadata --------------------------------------------
    # The SDK keeps discovered ``OAuthMetadata`` in memory only. Persisting it
    # lets a restarted process refresh without re-discovery; otherwise the SDK
    # guesses ``{server_url}/token`` (404 on most providers) and forces a full
    # browser re-authorization.

    def save_oauth_metadata(self, metadata: "OAuthMetadata") -> None:
        _write_json(self._meta_path(), metadata.model_dump(mode="json", exclude_none=True))
        logger.debug("OAuth metadata saved for %s", self._server_name)

    def load_oauth_metadata(self) -> "OAuthMetadata | None":
        return self._load_model(self._meta_path(), "OAuthMetadata", "OAuth metadata")

    # -- CIMD refusal ------------------------------------------------------

    def mark_cimd_rejected(self) -> None:
        """Durably record that this server refused our Client ID Metadata Document.

        The in-memory fallback only holds for one process; without a marker every
        restart re-presents a refused client_id. Cleared by ``remove()``
        (``hermes mcp login`` / ``remove``) so a fixed document gets a retry.
        """
        path = self._cimd_rejected_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        except OSError as exc:  # non-fatal — worst case we retry CIMD later
            logger.debug("Could not record CIMD rejection at %s: %s", path, exc)

    def cimd_rejected(self) -> bool:
        """True when this server has refused our metadata document before."""
        return self._cimd_rejected_path().exists()

    # -- cleanup -----------------------------------------------------------

    def remove(self) -> None:
        """Delete all stored OAuth state for this server."""
        for p in (*self._state_paths(), self._cimd_rejected_path()):
            p.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, bytes]:
        """filename -> bytes for the existing state files; feed to ``restore()``
        to undo a ``remove()`` after a failed re-auth so a valid token survives."""
        snap: dict[str, bytes] = {}
        for p in self._state_paths():
            with contextlib.suppress(OSError):
                snap[p.name] = p.read_bytes()
        return snap

    def restore(self, snapshot: dict[str, bytes], *, only_if_absent: bool = False) -> None:
        """Revert to a snapshot without overwriting a concurrent successful write."""
        if only_if_absent and any(path.exists() for path in self._state_paths()):
            logger.info("Skipping OAuth rollback for %s because newer state exists", self._server_name)
            return
        self.remove()
        if not snapshot:
            return
        token_dir = _get_token_dir(self._hermes_home)
        token_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in snapshot.items():
            try:
                fd = os.open(str(token_dir / fname), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.warning("Failed to restore OAuth state %s: %s", fname, exc)

    def poison_client_registration(self) -> bool:
        """Discard a dead dynamically-registered client (``invalid_client`` at the
        token endpoint) so the SDK re-runs DCR next flow; stale ``meta.json`` goes
        too. Tokens are kept: re-auth overwrites them and a valid refresh token
        survives if re-registration never completes. Keeps one ``.bak``.
        Returns True if a client file was present and removed.
        """
        client_path = self._client_info_path()
        if not client_path.exists():
            return False
        backup = client_path.with_name(client_path.name + ".bak")
        try:
            backup.write_bytes(client_path.read_bytes())
        except OSError as exc:  # non-fatal — proceed with the removal anyway
            logger.warning("Could not back up client info at %s: %s", client_path, exc)
        client_path.unlink(missing_ok=True)
        self._meta_path().unlink(missing_ok=True)
        logger.warning(
            "MCP OAuth '%s': cached client registration rejected as invalid_client; "
            "removed client.json + meta.json (backup at %s) to force re-registration",
            self._server_name, backup.name,
        )
        return True

    def has_cached_tokens(self) -> bool:
        """True if we have tokens on disk (may be expired)."""
        return self._tokens_path().exists()


# -- Callback capture -- HTTP listener and stdin paste share one result dict --
def _authorization_code_result(code: str, state: "str | None", iss: "str | None" = None):
    """Package redirect parameters in the shape the installed SDK expects: mcp 2.0's
    ``callback_handler`` returns an ``AuthorizationCodeResult`` (the SDK reads
    ``.state`` / ``.iss`` off it); older SDKs take a tuple."""
    try:
        from mcp.shared.auth import AuthorizationCodeResult
    except ImportError:  # mcp < 2.0
        return code, state
    return AuthorizationCodeResult(code=code, state=state, iss=iss)


def _parse_redirect_query(query: str) -> dict[str, Any]:
    """Extract code/state/error/iss from a redirect query string. ``iss`` is the
    RFC 9207 issuer: mcp 2.0 rejects a response that omits it when the server
    advertised ``authorization_response_iss_parameter_supported``, so keep it."""
    params = parse_qs(query)
    return {k: params.get(k, [None])[0] for k in ("code", "state", "error", "iss")}


def _result_taken(result: dict) -> bool:
    return result.get("auth_code") is not None or result.get("error") is not None


def _fill_result(result: dict, parsed: dict[str, Any]) -> None:
    result.update(auth_code=parsed["code"], state=parsed["state"], error=parsed["error"], iss=parsed["iss"])


def _make_callback_handler() -> tuple[type, dict]:
    """Fresh ``(HandlerClass, result_dict)`` per flow so concurrent flows don't
    stomp on each other."""
    result: dict[str, Any] = {"auth_code": None, "state": None, "error": None, "iss": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = _parse_redirect_query(urlparse(self.path).query)
            _fill_result(result, parsed)
            if parsed["code"]:
                body = "<h2>Authorization Successful</h2><p>You can close this tab and return to Hermes.</p>"
            else:
                body = f"<h2>Authorization Failed</h2><p>Error: {parsed['error'] or 'unknown'}</p>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<html><body>{body}</body></html>".encode())

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("OAuth callback: %s", fmt % args)

    return _Handler, result


def _paste_callback_reader(result: dict) -> None:
    """Read one stdin line, parse it as an OAuth redirect, write to *result*.

    Accepts a full redirect URL, the provider's own callback URL, just the query
    string (``?code=...&state=...`` or ``code=...``), or a skip token
    (``skip``/``cancel``/``s``/``n``/``no``/``q``/``quit``) which exits the flow
    without auth. Parse failures, EOF and interrupts are swallowed — this is a
    best-effort fallback racing the HTTP listener, which stays primary.
    """
    try:
        line = sys.stdin.readline()
    except (KeyboardInterrupt, OSError, ValueError):
        return
    line = line.strip() if line else ""
    if not line or _result_taken(result):
        return  # EOF / blank, or the HTTP listener already won

    if line.lower() in _SKIP_TOKENS:
        result["error"] = _USER_SKIPPED_SENTINEL
        print(
            "  OAuth skipped. Run `hermes mcp login <server>` later to authenticate, "
            "or set ``enabled: false`` on that server in config.yaml to disable persistently.",
            file=sys.stderr,
        )
        return

    # Full URL or "?code=...": take everything after the first "?".
    query = line.split("?", 1)[1] if "?" in line else line
    try:
        parsed = _parse_redirect_query(query.removeprefix("?"))
    except (ValueError, TypeError):
        print("  Could not parse pasted input as an OAuth redirect — ignoring.", file=sys.stderr)
        return
    if not parsed["code"] and not parsed["error"]:
        print("  Pasted input did not contain ``code=`` or ``error=`` — ignoring.", file=sys.stderr)
        return
    if _result_taken(result):  # one more race-check before writing
        return
    _fill_result(result, parsed)
    if parsed["code"]:
        print("  Got authorization code from paste — completing flow.", file=sys.stderr)


# -- Async redirect + callback handlers for OAuthClientProvider --------------

_SSH_HINT_PROXY = (
    "  Remote session detected. After you authorize, the provider redirects to\n"
    "    {redirect_uri}\n"
    "  which forwards to the callback listener on this machine — no SSH tunnel needed.\n"
)
_SSH_HINT_LOOPBACK = (
    "  Remote session detected. After you authorize, the provider redirects to\n"
    "    http://127.0.0.1:{port}/callback\n"
    "  which only the listener on THIS machine can receive. Two options:\n"
    "\n"
    "    1. Easiest — when your browser shows a connection error after\n"
    "       authorizing, copy the full URL from the address bar and paste\n"
    "       it at the prompt below. The pasted ``code=...&state=...`` is\n"
    "       enough to complete the flow.\n"
    "\n"
    "    2. Or forward the port first in a separate terminal:\n"
    "         ssh -N -L {port}:127.0.0.1:{port} <user>@<this-host>\n"
    "       then open the URL above and let it redirect normally.\n"
    "\n"
    "  See: https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh\n"
)


def _print_ssh_hint(port: int, redirect_uri: str | None) -> None:
    """Remote-session guidance printed under the authorization URL.

    With a proxy callback (e.g. Tailscale Funnel) the redirect is forwarded to
    this machine's listener, so no tunnel/paste is needed. On the loopback default
    the redirect reaches the *remote* machine's listener, not the browser's, so
    the user must paste the redirect URL back or SSH-forward the port.
    """
    if redirect_uri:
        print(_SSH_HINT_PROXY.format(redirect_uri=redirect_uri), file=sys.stderr)
    elif port:
        print(_SSH_HINT_LOOPBACK.format(port=port), file=sys.stderr)


def _announce_authorization_url(authorization_url: str, port: int, redirect_uri: str | None) -> None:
    """Print the URL (always, as the fallback) and open the browser when possible."""
    print(f"\n  MCP OAuth: authorization required.\n  Open this URL in your browser:\n\n    {authorization_url}\n", file=sys.stderr)
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        _print_ssh_hint(port, redirect_uri)

    if not _can_open_browser():
        note = "Headless environment detected — open the URL manually."
    else:
        try:
            opened = webbrowser.open(authorization_url)
        except Exception:
            opened = False
        note = "Browser opened automatically." if opened else "Could not open browser — please open the URL manually."
    print(f"  ({note})\n", file=sys.stderr)


def _make_redirect_handler(port: int, redirect_uri: str | None = None):
    """Return a redirect handler closing over this flow's port (a closure, not the
    module-level ``_oauth_port``, keeps concurrent server flows isolated).
    ``redirect_uri`` is a configured proxy callback (None for loopback) and only
    tailors the remote-session hint."""
    async def _redirect_handler(authorization_url: str) -> None:
        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            await dashboard_flow.publish_authorization_url(authorization_url)
            return
        # Fail fast in non-interactive contexts (systemd gateway, cron, background
        # discovery): a cached-but-unusable token makes the SDK fall through to
        # the authorization-code flow even though the token-file guard passed, and
        # we would otherwise block in the waiter for the full timeout.
        # Deliberately re-checks interactivity here.
        _raise_if_non_interactive(
            "MCP OAuth requires browser authorization but no interactive "
            "session is available (non-interactive/background context)."
        )
        _announce_authorization_url(authorization_url, port, redirect_uri)

    return _redirect_handler


def _start_callback_server(port: int, handler_cls: type) -> HTTPServer:
    """Bind the callback listener on *port*, adopting a parked reserved socket.

    Adopting the bound socket closes the select→bind TOCTOU window.
    ``allow_reuse_address`` is set BEFORE binding (a no-op afterwards) so a
    lingering TIME_WAIT socket from a previous flow cannot block the next.
    """
    try:
        server = HTTPServer(("127.0.0.1", port), handler_cls, bind_and_activate=False)
        reserved = _reserved_sockets.pop(port, None)
        if reserved is not None:
            server.socket.close()
            server.socket, server.server_address = reserved, reserved.getsockname()
        else:
            server.allow_reuse_address = True
            server.server_bind()
        server.server_activate()
    except OSError as exc:
        # Genuinely in use: a concurrent flow, leftover listener, or colliding
        # fixed `oauth.redirect_port`. Nothing to poll — say so, not "timed out".
        raise OAuthNonInteractiveError(
            f"OAuth callback port {port} is already in use ({exc}). "
            "Close any other in-progress login, or set a free `oauth.redirect_port` "
            "in the server config, then retry."
        ) from exc
    return server


async def _poll_callback_result(server: HTTPServer, result: dict, timeout: float) -> None:
    """Poll *result* until filled or *timeout* elapses; always closes the listener."""
    elapsed = 0.0
    try:
        while elapsed < timeout and not _result_taken(result):
            await asyncio.sleep(0.5)
            elapsed += 0.5
    finally:
        server.server_close()


def _callback_outcome(result: dict, cimd_url: str | None):
    """Turn a filled/empty result dict into the SDK's callback value, or raise."""
    if result["error"] == _USER_SKIPPED_SENTINEL:
        raise OAuthNonInteractiveError("user_skipped")
    if result["error"]:
        raise RuntimeError(f"OAuth authorization failed: {result['error']}")
    if result["auth_code"] is None:
        hint = (
            " If the browser showed an invalid-client error instead of "
            "an approval prompt, the authorization server rejected "
            f"Hermes' Client ID Metadata Document ({cimd_url}); set "
            "``cimd: false`` under that server's ``oauth:`` block in "
            "config.yaml to authorize via dynamic client registration "
            "instead."
        ) if cimd_url else ""
        raise OAuthNonInteractiveError(
            "OAuth callback timed out — no authorization code received. "
            "Ensure you completed the browser authorization flow." + hint
        )
    return _authorization_code_result(result["auth_code"], result["state"], result.get("iss"))


def _make_callback_waiter(port: int, cimd_url: str | None = None, timeout: float = 300.0):
    """Return a callback waiter bound to one flow's port (isolating concurrent flows).

    ``timeout`` is the only place ``oauth.timeout`` applies (mcp 2.0 dropped the
    provider's own). ``cimd_url`` only tailors the timeout message: a server that
    refuses the document aborts at the *authorization* endpoint, so no redirect
    arrives and a bare "timed out" would hide the cause. On a TTY the HTTP
    listener races a stdin paste fallback. Raises ``OAuthNonInteractiveError`` on
    timeout or when non-interactive.
    """

    async def _wait():
        dashboard_flow = get_dashboard_oauth_flow()
        if dashboard_flow is not None:
            # Dashboard flow speaks the legacy tuple; normalize to one shape.
            dash_code, dash_state = await dashboard_flow.wait_for_callback()
            return _authorization_code_result(dash_code, dash_state)

        # Reaching here means the SDK entered the authorization-code flow, so any
        # cached token is unusable. Reject BEFORE binding: binding would block for
        # the full timeout and collide with the TIME_WAIT port on retry
        # (``Address already in use``). Holds regardless of token files.
        _raise_if_non_interactive(
            "OAuth callback requires an interactive session but none is "
            "available (non-interactive/background context); skipping browser "
            "authorization without binding a callback listener."
        )

        handler_cls, result = _make_callback_handler()
        server = _start_callback_server(port, handler_cls)
        threading.Thread(target=server.handle_request, daemon=True).start()

        # Paste fallback races the HTTP listener; whichever fills result first wins.
        if _is_interactive():
            print(
                "\n  Or paste the redirect URL here (or the ``?code=...&state=...`` "
                "portion) and press Enter. Type ``skip`` + Enter to continue "
                "without this server:",
                file=sys.stderr, flush=True,
            )
            threading.Thread(target=_paste_callback_reader, args=(result,), daemon=True).start()

        await _poll_callback_result(server, result, timeout)
        return _callback_outcome(result, cimd_url)

    return _wait


# -- OAuth provider class (legacy build_oauth_auth path) ---------------------
HermesOAuthClientProvider: Any = None


def _get_hermes_oauth_provider_class() -> type | None:
    """Build (once) and cache ``HermesOAuthClientProvider``; None without the SDK."""
    global HermesOAuthClientProvider
    if HermesOAuthClientProvider is None and _ensure_sdk_loaded():
        from tools.mcp_oauth_provider import HermesProviderMixin

        HermesOAuthClientProvider = type(
            "HermesOAuthClientProvider",
            (HermesProviderMixin, OAuthClientProvider),
            {
                "__doc__": "SDK provider plus Hermes' token-endpoint fixes (see ``HermesProviderMixin``).",
                "__module__": __name__,
                "_hermes_logger": logger,
            },
        )
    return HermesOAuthClientProvider


def remove_oauth_tokens(server_name: str, *, hermes_home: str | Path | None = None) -> None:
    """Delete stored OAuth tokens and client info for a server."""
    HermesTokenStorage(server_name, hermes_home=hermes_home).remove()
    logger.info("OAuth tokens removed for '%s'", server_name)


# -- CIMD -- OAuth Client ID Metadata Documents -------------------------------
# Under CIMD the client_id IS an HTTPS URL the authorization server fetches to
# learn our name, logo and permitted redirect URIs, replacing per-install DCR.
# The SDK does the protocol work; Hermes only decides whether a flow is
# eligible and hands the URL to ``OAuthClientProvider``.

# Published from ``website/static/oauth/client-metadata.json`` by the docs
# deploy. The github.io origin is deliberate: an authorization server MUST NOT
# follow HTTP redirects when fetching the document, and
# hermes-agent.nousresearch.com/docs/* 301s here.
_CIMD_CLIENT_METADATA_URL = "https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json"
# Loopback callback ports declared in that document (exact string match, so a
# CIMD flow cannot use an ephemeral port). Below Linux's 32768 ephemeral floor so
# the kernel never hands one to another process. Keep in sync with the document
# (tests/tools/test_mcp_cimd.py enforces it).
_CIMD_PORTS = (27890, 27891, 27892, 27893, 27894)
# Loopback hostnames the document lists alongside each port, so the
# ``oauth.redirect_host: localhost`` WAF workaround still works under CIMD.
_CIMD_REDIRECT_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _is_valid_cimd_url(url: str) -> bool:
    """True when *url* is usable as a CIMD client_id on the installed SDK.

    Delegates to the SDK's validator (ImportError = SDK predates CIMD → DCR
    only). The SDK checks only https-scheme and non-root-path; userinfo,
    fragments and dot segments are rejected here because they fail at the
    authorization server mid-browser-flow as an opaque invalid-client page.
    """
    try:
        from mcp.client.auth.utils import is_valid_client_metadata_url
    except ImportError:
        return False
    if not is_valid_client_metadata_url(url):
        return False
    try:
        parsed = urlparse(url)
        has_userinfo = bool(parsed.username or parsed.password)  # netloc parse can raise
    except ValueError:
        return False
    return not (has_userinfo or parsed.fragment or any(seg in {".", ".."} for seg in parsed.path.split("/")))


# Pinned ports this process has committed to, in order taken. A provider is
# built once per server and keeps its port for the process lifetime, so
# assignments are never released. Includes ports restored from a cached
# registration so a sibling server is never handed one already in use.
_assigned_cimd_ports: "list[int]" = []


def _note_assigned_cimd_port(port: int) -> None:
    """Claim *port* for this process when it belongs to the pinned range."""
    if port in _CIMD_PORTS and port not in _assigned_cimd_ports:
        _assigned_cimd_ports.append(port)


def _pick_cimd_port() -> int | None:
    """Reserve a pinned CIMD callback port, or None when none is usable.

    Holding the bound socket until adoption prevents the same steal window as
    for ephemeral ports, and makes contention cooperative: another profile or
    sibling server finds the bind refused and moves down the range. Once every
    pinned port belongs to this process the range wraps rather than falling
    back to DCR — a reused port only bites if both servers authorize at the
    same moment (reported clearly by the waiter), whereas DCR may be
    unsupported by the server entirely.
    """
    for port in _CIMD_PORTS:
        if port not in _assigned_cimd_ports and _bind_reserved(port) is not None:
            _assigned_cimd_ports.append(port)
            return port
    return _assigned_cimd_ports[0] if _assigned_cimd_ports else None


def _server_declined_cimd(storage: "HermesTokenStorage | None") -> bool:
    """True when cached metadata shows this server doesn't advertise CIMD.

    The SDK decides CIMD vs DCR in its 401 branch — after Hermes must fix the
    redirect URI. Cached authorization-server metadata closes the gap for every
    server already reached: only a genuinely unknown server pays the optimistic pin.
    """
    try:
        metadata = storage.load_oauth_metadata() if storage is not None else None
    except (AttributeError, TypeError, ValueError):
        return False
    return metadata is not None and getattr(metadata, "client_id_metadata_document_supported", None) is not True


def _maybe_use_cimd(cfg: dict, storage: "HermesTokenStorage | None" = None) -> "tuple[str, int] | None":
    """Return ``(client_id URL, pinned callback port)``, or None to use DCR.

    Every ineligibility case below is one where the redirect URI Hermes would
    send is not one the document declares, where the client identity is already
    settled, or where the server is known not to want a document — passing a
    metadata URL anyway would make the authorization server reject the flow.
    """
    url = cfg.get("client_metadata_url") or _CIMD_CLIENT_METADATA_URL
    ineligible = (
        cfg.get("cimd") is False
        or not _is_valid_cimd_url(url)
        # A pinned client is the user's explicit choice; a secret means a
        # confidential client, which the document forbids.
        or cfg.get("client_id") or cfg.get("client_secret")
        # The document supplies name and auth method; a caller setting either
        # asks for an identity CIMD cannot present (Figma's DCR name allowlist).
        or cfg.get("client_name") or (cfg.get("token_endpoint_auth_method") or "none") != "none"
        # Dashboard/desktop flows redirect to a deployment-specific server URL
        # that can never appear in a static document.
        or get_dashboard_oauth_flow() is not None
        or cfg.get("redirect_uri") or cfg.get("redirect_port")
        or (cfg.get("redirect_host") or "127.0.0.1") not in _CIMD_REDIRECT_HOSTS
        # An existing registration is bound to its redirect URI; swapping in a
        # CIMD client_id now would invalidate stored tokens.
        or _cached_client_info(storage) is not None
        or (storage is not None and storage.cimd_rejected())
        or _server_declined_cimd(storage)
    )
    if ineligible:
        return None
    port = _pick_cimd_port()
    return None if port is None else (url, port)


def cimd_provider_kwargs(cfg: dict) -> dict[str, Any]:
    """``client_metadata_url=`` for ``OAuthClientProvider``, when CIMD applies.
    Returned as kwargs so the argument is omitted entirely on a DCR flow: an SDK
    too old for CIMD rejects the keyword outright."""
    url = cfg.get("_cimd_url")
    return {"client_metadata_url": url} if url else {}


def token_request_user_agent(cfg: dict) -> str | None:
    """Configured ``oauth.user_agent`` for token-endpoint requests, or None.

    Opt-in and per-server; anything but a non-empty string is unset so a
    null/empty YAML value never sends a blank header. Applied ONLY to
    authorization-code exchange and refresh — never MCP traffic or discovery;
    no other headers are configurable (secrets would land in config.yaml).
    """
    ua = cfg.get("user_agent")
    return ua.strip() if isinstance(ua, str) and ua.strip() else None


def _configure_callback_port(cfg: dict, storage: "HermesTokenStorage | None" = None) -> int:
    """Resolve the callback port into ``cfg['_resolved_port']`` (0 = non-loopback URI).

    Precedence: dashboard flow / cached https redirect URI → CIMD pinned port
    (also sets ``cfg['_cimd_url']``) → ``oauth.redirect_port`` → cached
    registration port → fresh ephemeral port. Only the fresh pick is parked;
    fixed ports bind via reuse_address. Also sets the legacy ``_oauth_port``.
    """
    global _oauth_port
    dashboard_flow = get_dashboard_oauth_flow()
    if dashboard_flow is not None:
        cfg["_resolved_port"] = 0
        cfg["redirect_uri"] = cfg.get("redirect_uri") or dashboard_flow.redirect_uri
        return 0
    cached_redirect_uri = None if cfg.get("redirect_uri") else _cached_redirect_uri(storage)
    if cached_redirect_uri:
        cfg["redirect_uri"] = cached_redirect_uri
        cfg["_resolved_port"] = 0
        return 0
    cimd = _maybe_use_cimd(cfg, storage)
    if cimd is not None:
        cfg["_cimd_url"], port = cimd
    else:
        port = int(cfg.get("redirect_port", 0)) or _cached_redirect_port(storage) or _reserve_callback_port()
        # A cached port may be a pinned CIMD port left by an earlier CIMD
        # login; claim it so a sibling's _pick_cimd_port doesn't reuse it.
        _note_assigned_cimd_port(port)
    cfg["_resolved_port"] = port
    _oauth_port = port
    return port


def _resolve_redirect_uri(cfg: dict, port: int) -> str:
    """Configured ``redirect_uri`` (proxy, e.g. Tailscale Funnel) or
    ``http://<redirect_host>:<port>/callback``.

    Client metadata and pre-registered client info must both derive the URI here
    so they stay identical. ``redirect_host`` (default ``127.0.0.1``) only
    changes the hostname: some WAFs reject a literal ``127.0.0.1`` in the
    authorize query, and ``localhost`` works around it. The listener still binds
    ``127.0.0.1``.
    """
    return cfg.get("redirect_uri") or f"http://{cfg.get('redirect_host') or '127.0.0.1'}:{port}/callback"


# Figma's remote MCP implements DCR as a client_name *allowlist*: "Claude Code"
# and "Codex" register (200); "Hermes Agent"/"Cursor"/… get 403. Register under
# an allowlisted name so the browser flow can start; oauth.client_name overrides.
_FIGMA_DCR_CLIENT_NAME = "Claude Code"
_FIGMA_DEFAULT_SCOPE = "mcp:connect"


def _is_figma_remote_mcp(server_name: str | None = None, server_url: str | None = None) -> bool:
    """True when this MCP server is Figma's hosted remote endpoint."""
    url = (server_url or "").lower()
    name = (server_name or "").lower()
    from utils import base_url_host_matches, base_url_hostname
    if base_url_host_matches(url, "mcp.figma.com") or (base_url_host_matches(url, "figma.com") and "/mcp" in url):
        return True
    # Name-only match only when the URL isn't some other host called figma-*.
    return "figma" in name and (not url or "figma" in base_url_hostname(url))


def apply_oauth_provider_defaults(cfg: dict, *, server_name: str = "", server_url: str | None = None) -> dict:
    """Mutate *cfg* with provider-specific OAuth workarounds. Returns *cfg*.

    Call before building client metadata / pre-registering. Only fills keys
    the user left unset — explicit ``oauth.client_name`` / ``oauth.scope`` win.
    """
    if _is_figma_remote_mcp(server_name, server_url):
        if not cfg.get("client_name"):
            cfg["client_name"] = _FIGMA_DCR_CLIENT_NAME
            logger.info(
                "MCP OAuth '%s': Figma DCR allowlist — registering as "
                "client_name=%r (override via oauth.client_name)",
                server_name or server_url, _FIGMA_DCR_CLIENT_NAME,
            )
        if not cfg.get("scope"):
            cfg["scope"] = _FIGMA_DEFAULT_SCOPE
        # Figma advertises token_endpoint_auth_method=none yet returns a
        # client_secret and then demands it at the token endpoint; request a
        # confidential-client registration so the SDK posts the secret.
        cfg["token_endpoint_auth_method"] = cfg.get("token_endpoint_auth_method") or "client_secret_post"
    return cfg


def _build_client_metadata(cfg: dict) -> "OAuthClientMetadata":
    """Build OAuthClientMetadata; requires ``_configure_callback_port`` first."""
    port = cfg.get("_resolved_port")
    if port is None:
        raise ValueError("_configure_callback_port() must be called before _build_client_metadata()")
    if OAuthClientMetadata is None:
        _ensure_sdk_loaded()
    # Public client by default; confidential only with a known secret or a
    # provider (e.g. Figma) that needs confidential-style token posts.
    auth_method = cfg.get("token_endpoint_auth_method") or ("client_secret_post" if cfg.get("client_secret") else "none")
    metadata_kwargs: dict[str, Any] = {
        "client_name": cfg.get("client_name", "Hermes Agent"),
        "redirect_uris": [AnyUrl(_resolve_redirect_uri(cfg, port))],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": auth_method,
        # SEP-837: clients MUST declare application_type so OIDC-strict servers
        # accept loopback redirect_uris; Hermes is a CLI/desktop app → "native".
        # Overridable for a hosted dashboard fronting a real https redirect.
        "application_type": cfg.get("application_type", "native"),
    }
    if cfg.get("scope"):
        metadata_kwargs["scope"] = cfg["scope"]
    try:
        return OAuthClientMetadata.model_validate(metadata_kwargs)
    except Exception:
        # mcp 1.x metadata models predate SEP-837 and reject the unknown field.
        metadata_kwargs.pop("application_type", None)
        return OAuthClientMetadata.model_validate(metadata_kwargs)


def _invalidate_tokens_on_client_change(
    storage: "HermesTokenStorage", new_client_id: str, new_client_secret: str | None
) -> None:
    """Drop cached tokens when the configured OAuth client identity changes.

    Tokens are minted for a specific ``client_id``; after editing
    ``oauth.client_id`` / ``oauth.client_secret`` (or switching from DCR to a
    pre-registered client) the old tokens fail refresh with ``invalid_client``.
    Pre-registered clients are exempt from the auto-poison path, so without this
    check stale tokens wedge every request until a manual wipe. Compares on-disk
    ``client.json`` against the incoming identity BEFORE it is overwritten; a
    matching identity is a no-op.
    """
    existing = _read_json(storage._client_info_path())
    old_client_id = existing.get("client_id") if isinstance(existing, dict) else None
    if not old_client_id:
        return
    if old_client_id == new_client_id and (existing.get("client_secret") or None) == (new_client_secret or None):
        return
    removed = False
    for path in (storage._tokens_path(), storage._meta_path()):
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:  # non-fatal — stale tokens fail later anyway
            logger.warning(
                "MCP OAuth '%s': could not remove stale %s after client "
                "change: %s", storage._server_name, path.name, exc,
            )
    if removed:
        logger.warning(
            "MCP OAuth '%s': configured OAuth client changed (client_id %r "
            "-> %r); discarded tokens minted under the previous client. "
            "Re-authorize with: hermes mcp login %s",
            storage._server_name, old_client_id, new_client_id,
            storage._server_name,
        )


def _maybe_preregister_client(storage: "HermesTokenStorage", cfg: dict, client_metadata: "OAuthClientMetadata") -> None:
    """If cfg has a pre-registered client_id, persist it to storage."""
    client_id = cfg.get("client_id")
    if not client_id:
        return
    if OAuthClientInformationFull is None:
        _ensure_sdk_loaded()
    _invalidate_tokens_on_client_change(storage, client_id, cfg.get("client_secret"))
    info_dict: dict[str, Any] = {
        "client_id": client_id,
        "redirect_uris": [_resolve_redirect_uri(cfg, cfg["_resolved_port"])],
        "grant_types": client_metadata.grant_types,
        "response_types": client_metadata.response_types,
        "token_endpoint_auth_method": client_metadata.token_endpoint_auth_method,
        **{key: cfg[key] for key in ("client_secret", "client_name", "scope") if cfg.get(key)},
    }
    client_info = OAuthClientInformationFull.model_validate(info_dict)
    _write_json(storage._client_info_path(), client_info.model_dump(mode="json", exclude_none=True))
    logger.debug("Pre-registered client_id=%s for '%s'", client_id, storage._server_name)


def humanize_oauth_registration_error(
    server_name: str, exc: BaseException | str, *, server_url: str | None = None
) -> str | None:
    """Turn a Dynamic Client Registration 403/Forbidden into a useful next step.

    Returns None for anything else so the caller keeps the original text.
    Figma gates DCR on exact ``client_name``; Hermes auto-sets ``Claude Code``,
    so this fires when the user overrode it or an older Hermes is running.
    """
    msg = str(exc)
    lowered = msg.lower()
    if "403" not in msg and "forbidden" not in lowered:
        return None
    looks_like_registration = (
        any(k in lowered for k in ("regist", "dcr", "dynamic client"))
        or lowered.strip() in {"forbidden", "403 forbidden", "http 403: forbidden"}
        or ("403" in msg and "forbidden" in lowered)
    )
    if not looks_like_registration:
        return None

    if _is_figma_remote_mcp(server_name, server_url):
        return (
            f"'{server_name}' is Figma's remote MCP — DCR is allowlisted by "
            f"exact client_name (\"{_FIGMA_DCR_CLIENT_NAME}\" and \"Codex\" "
            "work; most other names 403). Hermes defaults to "
            f"client_name: {_FIGMA_DCR_CLIENT_NAME!r} automatically. If you "
            "set oauth.client_name yourself, change it to one of those, or "
            "clear it and re-run:\n"
            f"  hermes mcp login {server_name}"
        )

    return (
        f"'{server_name}' only allows pre-approved OAuth clients — it rejected "
        "client registration (403), so no browser flow can start. Options: "
        "set oauth.client_name to a name the provider allowlists, add a "
        "pre-registered client (oauth: {client_id: ..., client_secret: ...}), "
        "or use the provider's stdio / API-key / local server instead."
    )


def build_oauth_auth(server_name: str, server_url: str, oauth_config: dict | None = None) -> "OAuthClientProvider | None":
    """Build an ``httpx.Auth``-compatible OAuth handler for an MCP server.

    Legacy public API; new code should use
    :func:`tools.mcp_oauth_manager.get_manager` so OAuth state is shared across
    config-time, runtime, and reconnect paths. Returns None if the MCP SDK lacks
    OAuth support.
    """
    if not _OAUTH_AVAILABLE or (OAuthClientProvider is None and not _ensure_sdk_loaded()):
        logger.warning(
            "MCP OAuth requested for '%s' but SDK auth types are not available. "
            "Install with: pip install 'mcp>=1.26.0'",
            server_name,
        )
        return None

    from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config

    cfg, storage = prepare_oauth_config(server_name, server_url, oauth_config)
    if not _is_interactive() and not storage.has_cached_tokens():
        raise OAuthNonInteractiveError(
            "MCP OAuth for "
            f"'{server_name}': non-interactive environment and no cached tokens "
            "found. The OAuth flow requires browser authorization. Run "
            f"`hermes mcp login {server_name}` interactively first to complete "
            "initial authorization, then cached tokens will be reused."
        )

    kwargs = build_provider_kwargs(cfg, storage, ssh_proxy_hint=True)
    provider_class = _get_hermes_oauth_provider_class()
    if provider_class is None:
        logger.warning("MCP OAuth requested for '%s' but the provider class is unavailable", server_name)
        return None
    return provider_class(server_url=server_url, **kwargs)
