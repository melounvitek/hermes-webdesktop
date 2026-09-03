"""MCP connection/transport error classification: URL validation, TLS client certs, identity headers, redirect header stripping, exception-group unwrapping, auth/session-expired/method-not-found detection and connect-error formatting. Split from tools/mcp_tool.py."""

import asyncio
import errno
import importlib
import logging
import os
import re
from typing import Any, List, Optional
from urllib.parse import urlparse
from tools.mcp_tool_common import _sanitize_error, _core

logger = logging.getLogger("tools.mcp_tool")


# Stateless (2026-07-28) servers reject a legacy ``initialize`` with
# UnsupportedProtocolVersion (-32022) or plain method-not-found.
_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION = -32022


def _jsonrpc_code(exc: BaseException):
    """Structural ``MCPError.error.code`` (None when absent)."""
    return getattr(getattr(exc, "error", None), "code", None)


def _jsonrpc_matches(exc: BaseException, code, codes: tuple, markers: tuple) -> bool:
    """Structural *code* in *codes*, else any lowercased *marker* in ``str(exc)``. Never
    ``isinstance`` on SDK exception types: they arrive wrapped in ExceptionGroups and drift
    across generations."""
    if code in codes:
        return True
    msg = str(exc).lower()
    return bool(msg) and any(marker in msg for marker in markers)


def _handshake_rejected_as_modern(exc: BaseException) -> bool:
    """True when a failed ``initialize`` signals a stateless-only (2026-07-28) server."""
    return _jsonrpc_matches(
        exc, _jsonrpc_code(exc) or getattr(exc, "code", None),
        (_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION, _core._JSONRPC_METHOD_NOT_FOUND),
        ("unsupported protocol version", str(_JSONRPC_UNSUPPORTED_PROTOCOL_VERSION)),
    ) or _is_method_not_found_error(exc)


def _is_method_not_found_error(exc: BaseException) -> bool:
    """True if *exc* is a JSON-RPC ``method not found`` (-32601). ``ping`` is optional in MCP;
    servers lacking it answer -32601. The substring fallback includes "Unknown method: <name>",
    which some servers use; without it the ping→list_tools keepalive fallback never latches
    and reconnect-loops."""
    return _jsonrpc_matches(
        exc, _jsonrpc_code(exc), (_core._JSONRPC_METHOD_NOT_FOUND,),
        (str(_core._JSONRPC_METHOD_NOT_FOUND), "method not found", "unknown method", "not found: ping"),
    )


class InvalidMcpUrlError(ValueError):
    """A remote MCP server's ``url`` is not parseable http(s)://. Validated once at startup so
    we fail fast instead of burning the reconnect-backoff loop on every attempt."""


class NonMcpEndpointError(ConnectionError):
    """An HTTP MCP URL served a non-MCP 2xx response (e.g. ``text/html``); real Streamable-HTTP
    endpoints answer ``application/json`` or ``text/event-stream``. Non-retryable: every attempt
    gets the same page, so the backoff loop is skipped and the server is failed immediately.
    Subclasses ConnectionError so broad catches still see a connection problem."""


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    """Extract the root-cause leaf from anyio ``(Base)ExceptionGroup`` wrappers (group ``str()``
    is opaque, so log sites must unwrap). Two rules: a ``KeyboardInterrupt``/``SystemExit`` leaf
    anywhere is re-raised (never flattened into a loggable error); a non-cancellation leaf is
    preferred over the ``CancelledError`` noise anyio sprays on siblings."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        fatal, _rest = exc.split((KeyboardInterrupt, SystemExit))
        if fatal is not None:
            leaf: BaseException = fatal
            while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
                leaf = leaf.exceptions[0]
            raise leaf
        exc = next((sub for sub in exc.exceptions if not _contains_only_cancellation(sub)), exc.exceptions[0])
    return exc


def _contains_only_cancellation(exc: BaseException) -> bool:
    """True if ``exc`` is (or a group containing only) CancelledError."""
    if isinstance(exc, BaseExceptionGroup):
        return all(_contains_only_cancellation(sub) for sub in exc.exceptions)
    return isinstance(exc, asyncio.CancelledError)


def _classify_mcp_failure(exc: BaseException) -> str:
    """``'permanent'`` (deterministic — ``run()`` parks immediately instead of burning the retry
    ladder: auth 401/403, NonMcpEndpointError, InvalidMcpUrlError, missing stdio command
    FileNotFoundError / ENOENT) or ``'transient'`` (keeps backoff retry)."""
    root = _unwrap_exception_group(exc)
    permanent = (
        _core._is_auth_error(root)
        or isinstance(root, (NonMcpEndpointError, InvalidMcpUrlError, FileNotFoundError))
        or (isinstance(root, OSError) and getattr(root, "errno", None) == errno.ENOENT)
        # 401/403 HTTPStatusError that _is_auth_error's type-gate missed (auth types not importable here).
        or _response_status(root) in (401, 403)
    )
    return "permanent" if permanent else "transient"


def _response_status(exc: BaseException):
    """``exc.response.status_code`` for httpx-shaped errors, else None."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def _validate_remote_mcp_url(server_name: str, url: Any) -> str:
    """The stripped URL if it is a valid http(s) remote MCP URL. Raises InvalidMcpUrlError naming
    the server for non-strings, missing/other schemes (stdio servers use ``command``, not ``url``),
    and empty hosts."""
    def _bad(detail: str) -> InvalidMcpUrlError:
        return InvalidMcpUrlError(f"Invalid MCP URL for '{server_name}': {detail}")

    if not isinstance(url, str):
        raise _bad(f"expected a string, got {type(url).__name__}")
    stripped = url.strip()
    if not stripped:
        raise _bad("empty url")
    try:
        parsed = urlparse(stripped)
    except Exception as exc:  # urlparse is very permissive — belt and braces
        raise _bad(f"{stripped!r} ({exc})") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise _bad(f"scheme must be http or https, got {parsed.scheme!r} ({stripped!r})")
    if not parsed.netloc:
        raise _bad(f"missing host ({stripped!r})")
    if not parsed.hostname:  # ``urlparse`` accepts ``http://:8080`` (empty host, explicit port)
        raise _bad(f"missing hostname ({stripped!r})")
    return stripped


def _resolve_client_cert(server_name: str, config: dict):
    """``client_cert`` / ``client_key`` in httpx's ``cert=`` shape: None when neither is set; a
    single path for a combined PEM; ``(cert, key)`` or ``(cert, key, password)`` for the
    pair/list forms. ``~`` is expanded and missing files raise a server-scoped
    FileNotFoundError instead of an opaque TLS handshake error."""
    raw_cert = config.get("client_cert")
    raw_key = config.get("client_key")
    if raw_cert is None and raw_key is None:
        return None
    prefix = f"MCP server '{server_name}': "

    def _expand(path: Any, label: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{prefix}{label} must be a non-empty string path (got {type(path).__name__})")
        expanded = os.path.expanduser(path.strip())
        if not os.path.isfile(expanded):
            raise FileNotFoundError(f"{prefix}{label} not found at {expanded!r}")
        return expanded

    if isinstance(raw_cert, (list, tuple)):
        if raw_key is not None:
            raise ValueError(f"{prefix}specify either client_cert as a list [cert, key] OR "
                             f"client_cert + client_key, not both")
        if len(raw_cert) not in (2, 3):
            raise ValueError(f"{prefix}client_cert list form must have 2 or 3 elements (got {len(raw_cert)})")
        pair = (_expand(raw_cert[0], "client_cert[0]"), _expand(raw_cert[1], "client_cert[1]"))
        if len(raw_cert) == 2:
            return pair
        if not isinstance(raw_cert[2], str):
            raise ValueError(f"{prefix}client_cert[2] (key passphrase) must be a string")
        return (*pair, raw_cert[2])
    cert_path = _expand(raw_cert, "client_cert")
    if raw_key is not None:
        return (cert_path, _expand(raw_key, "client_key"))
    return cert_path  # single combined PEM (cert + key)


def _resolve_identity_header(server_name: str, config: dict):
    """Optional per-server ``identity_header`` ``{name, value_from: "static"|"profile", value}``
    (``value`` required for static) → ``(name, value)`` or None. Invalid configs warn and are
    ignored — an identity header must never break the connection. ``profile`` resolves once at
    connect time; no per-call mutation."""
    raw = config.get("identity_header")
    if raw is None:
        return None

    def _ignore(detail: str, *args):
        logger.warning("MCP server '%s': identity_header " + detail + " — ignoring", server_name, *args)
        return None

    if not isinstance(raw, dict):
        return _ignore("must be a mapping with 'name' and 'value'/'value_from' keys (got %s)", type(raw).__name__)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return _ignore("requires a non-empty 'name'")
    value_from = (raw.get("value_from") or "static").strip().lower()
    if value_from == "static":
        value = raw.get("value")
        if not isinstance(value, str) or not value.strip():
            return _ignore("with value_from: static requires a non-empty string 'value'")
        return (name.strip(), value)
    if value_from == "profile":
        from hermes_cli.profiles import get_active_profile_name
        return (name.strip(), get_active_profile_name())
    return _ignore("value_from must be 'static' or 'profile' (got %r)", value_from)


def _apply_identity_header(server_name: str, config: dict, headers: dict) -> dict:
    """Merge the resolved identity header into ``headers`` in place. An explicit per-server
    ``headers`` entry with the same name (any casing) wins — the identity header never silently
    overrides user config."""
    resolved = _resolve_identity_header(server_name, config)
    if resolved is None:
        return headers
    name, value = resolved
    if any(key.lower() == name.lower() for key in headers):
        logger.debug("MCP server '%s': identity_header '%s' already set via explicit "
                     "headers config — keeping the explicit value", server_name, name)
        return headers
    headers[name] = value
    return headers


def _make_redirect_header_stripper(original_url, *, strict: bool = False,
                                   configured_header_names: "set[str] | frozenset[str]" = frozenset()):
    """httpx response hook guarding cross-origin redirects: always strips ``Authorization`` when
    a redirect leaves the original origin; with *strict* (Agent Plugins v1
    ``strict_redirect_headers``) every configured header (lowercase names in
    *configured_header_names*) is stripped too — the v1 spec forbids forwarding
    package-configured headers cross-origin."""
    origin = (original_url.scheme, original_url.host, original_url.port)

    async def _strip_on_cross_origin_redirect(response):
        if not (response.is_redirect and response.next_request):
            return
        target = response.next_request.url
        if (target.scheme, target.host, target.port) == origin:
            return
        headers = response.next_request.headers
        headers.pop("authorization", None)
        headers.pop("Authorization", None)
        if strict:
            for _name in configured_header_names:
                while _name in headers:
                    del headers[_name]

    return _strip_on_cross_origin_redirect


def _exc_children(exc: BaseException) -> List[BaseException]:
    """Sub-exceptions of a group, else ``__cause__``/``__context__`` when they are exceptions."""
    nested = getattr(exc, "exceptions", None)
    if nested:
        return list(nested)
    return [c for c in (exc.__cause__, exc.__context__) if isinstance(c, BaseException)]


def _format_connect_error(exc: BaseException) -> str:
    """Render nested MCP connection errors into an actionable short message."""

    def _find_missing(current: BaseException) -> Optional[str]:
        if isinstance(current, FileNotFoundError):
            if getattr(current, "filename", None):
                return str(current.filename)
            match = re.search(r"No such file or directory: '([^']+)'", str(current))
            if match:
                return match.group(1)
        return next(filter(None, map(_find_missing, _exc_children(current))), None)

    def _flatten_messages(current: BaseException) -> List[str]:
        # A group's own str() is opaque — only its children speak.
        text = "" if getattr(current, "exceptions", None) else str(current).strip()
        messages = [text] if text else []
        for child in _exc_children(current):
            messages.extend(_flatten_messages(child))
        return messages or [current.__class__.__name__]

    missing = _find_missing(exc)
    if missing:
        message = f"missing executable '{missing}'"
        if os.path.basename(missing) in {"npx", "npm", "node"}:
            message += (" (ensure Node.js is installed and PATH includes its bin directory, "
                        "or set mcp_servers.<name>.command to an absolute path and include "
                        "that directory in mcp_servers.<name>.env.PATH)")
        return _sanitize_error(message)
    deduped = list(dict.fromkeys(_flatten_messages(exc)))
    return _sanitize_error("; ".join(deduped[:3]))


# Lazily-built caches so this module imports even without the SDK OAuth module.
_AUTH_ERROR_TYPES: tuple = ()
_HTTP_STATUS_ERROR_TYPES: Optional[tuple] = None


def _optional_types(module: str, *names: str) -> list:
    """``[module.name, ...]`` or ``[]`` when the module/attribute is unavailable."""
    try:
        mod = importlib.import_module(module)
        return [getattr(mod, name) for name in names]
    except (ImportError, AttributeError):
        return []


def _http_status_error_types() -> tuple:
    """``HTTPStatusError`` classes from both httpx flavours: a 401 may come from the SDK's own
    stack (``httpx2`` on mcp >= 2.0) or from Hermes' pinned ``httpx``; the classes are unrelated."""
    global _HTTP_STATUS_ERROR_TYPES
    if _HTTP_STATUS_ERROR_TYPES is None:
        sdk_mod = _core.sdk_httpx()
        found: list = [sdk_mod.HTTPStatusError] if sdk_mod is not None else []
        found += [cls for cls in _optional_types("httpx", "HTTPStatusError") if cls not in found]
        _HTTP_STATUS_ERROR_TYPES = tuple(found)
    return _HTTP_STATUS_ERROR_TYPES


def _get_auth_error_types() -> tuple:
    """Cached exception types indicating MCP OAuth failure: SDK ``OAuthFlowError``/``OAuthTokenError``
    (+ legacy ``UnauthorizedError``), our ``OAuthNonInteractiveError``, and ``HTTPStatusError`` from
    both httpx flavours — the latter needs the 401 status check in :func:`_is_auth_error`."""
    global _AUTH_ERROR_TYPES
    if not _AUTH_ERROR_TYPES:
        _AUTH_ERROR_TYPES = tuple(
            _optional_types("mcp.client.auth", "OAuthFlowError", "OAuthTokenError")
            + _optional_types("mcp.client.auth", "UnauthorizedError")  # older SDKs
            + _optional_types("tools.mcp_oauth", "OAuthNonInteractiveError")
            + list(_http_status_error_types())
        )
    return _AUTH_ERROR_TYPES


def _is_auth_error(exc: BaseException) -> bool:
    """True if ``exc`` indicates an MCP OAuth failure. ``HTTPStatusError`` counts only with
    status 401; other HTTP errors fall through to the generic error path."""
    types = _get_auth_error_types()
    if not types or not isinstance(exc, types):
        return False
    status_error_types = _http_status_error_types()
    if status_error_types and isinstance(exc, status_error_types):
        return getattr(exc.response, "status_code", None) == 401
    return True


# Lower-cased substrings meaning the server-side transport session expired / was GC'd.
# The OAuth token is still valid — only the transport needs rebuilding.
_SESSION_EXPIRED_MARKERS: tuple = (
    "invalid or expired session", "expired session", "session expired", "session not found",
    "unknown session", "session terminated", "closedresourceerror", "closed resource",
    "transport is closed", "connection closed", "broken pipe", "end of file",
)

# Node budget for ``_is_session_expired_error``. The visited set breaks cycles; the budget
# bounds pathological acyclic graphs. Kept well above ``sys.getrecursionlimit()`` so deep
# task-group nesting is still fully scanned.
_EXC_TRAVERSAL_MAX_NODES = 10_000


def _is_session_expired_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like an MCP transport session expiry. Streamable-HTTP servers GC
    session state (idle TTL, restart, pod rotation) while the OAuth token stays valid, so unlike
    :func:`_is_auth_error` the fix is a transport reconnect (``_reconnect_event``), not an OAuth refresh."""
    # AnyIO stream exceptions are often message-less (``str(ClosedResourceError()) == ""``),
    # so type checks are needed in addition to marker matching.
    transport_error_types = tuple(_optional_types("anyio", "BrokenResourceError", "ClosedResourceError", "EndOfStream"))

    # Iterative traversal over ``exceptions`` / ``__cause__`` / ``__context__`` with an
    # identity-visited set AND a node budget (graphs can be deep or cyclic). Every reachable
    # node is inspected so an InterruptedError anywhere overrides transport markers; the chain
    # walk matters because SDK wrappers often raise a generic RuntimeError *from* the
    # message-less ClosedResourceError.
    stack: "list[BaseException | None]" = [exc]
    seen: set[int] = set()
    transport_error_found = False
    budget = _EXC_TRAVERSAL_MAX_NODES
    while stack and budget > 0:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        budget -= 1
        if isinstance(current, InterruptedError):
            return False
        # Messages vary across SDK versions and servers: match a narrow allow-list of stable
        # substrings, not exception type, to avoid false positives.
        msg = str(current).lower()
        if isinstance(current, transport_error_types) or (msg and any(marker in msg for marker in _SESSION_EXPIRED_MARKERS)):
            transport_error_found = True
        stack.extend(getattr(current, "exceptions", ()))
        stack.extend((getattr(current, "__cause__", None), getattr(current, "__context__", None)))
    return transport_error_found
