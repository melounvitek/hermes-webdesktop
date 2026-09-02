#!/usr/bin/env python3
"""
MCP (Model Context Protocol) client: connects to configured MCP servers over
stdio, Streamable HTTP or SSE, discovers their tools and registers them into
the hermes tool registry. The ``mcp`` package is optional; without it this
module is a no-op.

Config lives under ``mcp_servers`` in ~/.hermes/config.yaml::

    mcp_servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
        timeout: 120                 # per tool call (default 300)
        connect_timeout: 60          # initial connect (default 60)
        keepalive_interval: 10       # liveness ping; keep below the server's
                                     # session TTL (default 180, floor 5)
        idle_timeout_seconds: 3600   # optional stdio recycle (0 = off); may
        max_lifetime_seconds: 86400  # also live under lifecycle: {...}
        supports_parallel_tool_calls: true
      remote_api:
        url: "https://my-mcp-server.example.com/mcp"
        headers: {Authorization: "Bearer sk-..."}
        identity_header: {name: "X-User-Id", value_from: "static", value: "alice"}
        skip_preflight: true         # endpoint answers HEAD/GET with a non-MCP
                                     # content type but serves MCP over POST
      searxng:
        url: "http://localhost:8000/sse"
        transport: sse
        sampling: {enabled: true, model: "gemini-3-flash", max_tokens_cap: 4096,
                   timeout: 30, max_rpm: 10, allowed_models: [], max_tool_rounds: 5}

Architecture: one background event loop (``_mcp_loop``) in a daemon thread;
each server is a long-lived Task on it (``MCPServerTask``) so the transport's
anyio cancel scopes are entered and exited in the same Task. Tool calls are
scheduled onto the loop via ``run_coroutine_threadsafe``. ``_servers`` and the
loop handles are shared with caller threads; every mutation holds ``_lock``.

Module map (all names re-exported here): ``mcp_tool_common`` (pure helpers),
``mcp_tool_schema`` (schema conversion / naming), ``mcp_tool_content`` (result
block rendering), ``mcp_tool_errors`` (failure classification, URL/cert/header
resolution), ``mcp_tool_config`` (config loading, stdio env), ``mcp_tool_sampling``
(sampling + elicitation handlers), ``mcp_tool_handlers`` (registry handlers and
per-call recovery), ``mcp_tool_registration`` (registry writes), ``mcp_tool_transport``
/ ``mcp_tool_health`` (MCPServerTask mixins), ``mcp_tool_lifecycle`` (shutdown,
orphan reaping), ``mcp_tool_agent`` (live-agent tool list refresh).
"""

import asyncio
import contextvars
import concurrent.futures
import errno
import inspect
import logging
import os
import shutil  # noqa: F401  — tests patch ``tools.mcp_tool.shutil.which``
import threading
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Split modules. Every name is re-exported here so ``from tools.mcp_tool import X``
# and ``mock.patch("tools.mcp_tool.X")`` keep working; the siblings read origin
# state back through ``tools.mcp_tool`` at call time (never by value).
from tools.mcp_tool_common import (  # noqa: F401
    _BACKOFF_JITTER,
    _CREDENTIAL_PATTERN,
    _DEFAULT_TOOL_TIMEOUT,
    _MISSING,
    _env_ref_name,
    _exc_str,
    _get_lifecycle_seconds,
    _jittered,
    _parse_boolish,
    _prepend_path,
    _resolve_tool_timeout,
    _safe_numeric,
    _sanitize_error,
    mcp_field,
)
from tools.mcp_tool_schema import (  # noqa: F401
    MCP_TOOL_NAME_PREFIX,
    _MCP_INJECTION_PATTERNS,
    _MCP_NAME_DELIM,
    _UTILITY_CAPABILITY_ATTRS,
    _UTILITY_CAPABILITY_METHODS,
    _build_utility_schemas,
    _convert_mcp_schema,
    _normalize_mcp_input_schema,
    _normalize_name_filter,
    _scan_mcp_description,
    matches_name_filter,
    mcp_prefixed_tool_name,
    sanitize_mcp_name_component,
)
from tools.mcp_tool_content import (  # noqa: F401
    _MCP_HARD_RESULT_CAP_CHARS,
    _MCP_RESOURCE_MAX_B64_CHARS,
    _MCP_RESOURCE_MAX_BYTES,
    _cache_mcp_audio_block,
    _cache_mcp_image_block,
    _is_reserved_mcp_meta_key,
    _mcp_image_extension_for_mime_type,
    _mcp_resource_filename,
    _render_mcp_resource_block,
    _strip_reserved_meta_keys,
    _truncate_mcp_text_result,
)
from tools.mcp_tool_errors import (  # noqa: F401
    InvalidMcpUrlError,
    NonMcpEndpointError,
    _AUTH_ERROR_TYPES,
    _EXC_TRAVERSAL_MAX_NODES,
    _HTTP_STATUS_ERROR_TYPES,
    _JSONRPC_UNSUPPORTED_PROTOCOL_VERSION,
    _SESSION_EXPIRED_MARKERS,
    _apply_identity_header,
    _classify_mcp_failure,
    _contains_only_cancellation,
    _format_connect_error,
    _get_auth_error_types,
    _handshake_rejected_as_modern,
    _http_status_error_types,
    _is_auth_error,
    _is_method_not_found_error,
    _is_session_expired_error,
    _make_redirect_header_stripper,
    _resolve_client_cert,
    _resolve_identity_header,
    _unwrap_exception_group,
    _validate_remote_mcp_url,
)
from tools.mcp_tool_config import (  # noqa: F401
    _ENV_VAR_PATTERN,
    _SAFE_ENV_KEYS,
    _SAFE_ENV_KEYS_CASE_INSENSITIVE,
    _build_safe_env,
    _context_var_value,
    _filter_suspicious_mcp_servers,
    _get_mcp_stderr_log,
    _interpolate_env_vars,
    _load_mcp_config,
    _mcp_stderr_log_fh,
    _mcp_stderr_log_lock,
    _resolve_stdio_command,
    _warn_hidden_whitespace,
    _whitespace_warned,
    _workspace_folder,
    _wrap_command_with_watchdog,
    _write_stderr_log_header,
)
from tools.mcp_tool_sampling import (  # noqa: F401
    ElicitationHandler,
    SamplingHandler,
    _format_elicitation_schema_summary,
)
from tools.mcp_tool_handlers import (  # noqa: F401
    _StdioChildExited,
    _handle_auth_error_and_retry,
    _handle_session_expired_and_retry,
    _handle_stdio_child_exited_and_retry,
    _interrupted_call_result,
    _make_check_fn,
    _make_get_prompt_handler,
    _make_list_prompts_handler,
    _make_list_resources_handler,
    _make_read_resource_handler,
    _make_tool_handler,
    _mark_server_call_started,
    _track_inflight_rpc,
    _trust_gate_check,
)
from tools.mcp_tool_registration import (  # noqa: F401
    _CachedMCPTool,
    _annotation_read_only_hint,
    _existing_tool_names,
    _forget_mcp_tool_server,
    _normalize_server_trust,
    _record_tool_trust_metadata,
    _register_from_cache_sync,
    _register_server_tools,
    _select_utility_schemas,
    _track_mcp_tool_server,
)
from tools.mcp_tool_lifecycle import (  # noqa: F401
    _NON_MCP_CHILD_CMDLINE_MARKERS,
    _drain_and_stop_mcp_loop,
    _drain_mcp_loop_tasks,
    _filter_mcp_children,
    _kill_orphaned_mcp_children,
    _orphan_stdio_pid_servers,
    _orphan_stdio_pids,
    _snapshot_child_pids,
    _stdio_pgids,
    _stdio_pids,
    _stop_mcp_loop_if_idle,
    shutdown_mcp_servers,
)
from tools.mcp_tool_agent import (  # noqa: F401
    _agent_tools_lock,
    _merge_preserving_prefix,
    _reinject_post_build_tools,
    persist_agent_tool_names,
    refresh_agent_mcp_tools,
    reprobe_tool_availability,
    restore_agent_tool_prefix,
)
from tools.mcp_tool_transport import MCPServerTransportMixin
from tools.mcp_tool_health import MCPServerHealthMixin


# Wall-clock bound on the (fail-open) OSV malware preflight run off the loop
# before a stdio spawn. Kept just ABOVE osv_check._TIMEOUT (10s) so the inner
# socket timeout normally fires first; this only bites when a stalled SSL
# handshake defeats it (which used to freeze the event loop at startup).
_OSV_MALWARE_CHECK_TIMEOUT_S = 12.0


# ---------------------------------------------------------------------------
# Optional MCP SDK: availability probe now, symbol import on first use
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_HTTP_AVAILABLE = False
_MCP_NEW_HTTP = False
_MCP_LEGACY_HTTP = False
_MCP_SAMPLING_TYPES = False
_MCP_NOTIFICATION_TYPES = False
_MCP_ELICITATION_TYPES = False
_MCP_MESSAGE_HANDLER_SUPPORTED = False
_MCP_LOGGING_CALLBACK_SUPPORTED = False
_MCP_NEW_HTTP = False
sse_client = None
# Fallback for SDKs that don't export LATEST_PROTOCOL_VERSION (Streamable HTTP
# arrived with 2025-03-26, so this stays valid for the HTTP path).
LATEST_PROTOCOL_VERSION = "2025-03-26"
# Newest revision `ClientSession.initialize()` actually speaks. From 2026-07-28
# the handshake is replaced by a per-request envelope, so this can be OLDER
# than LATEST_PROTOCOL_VERSION; the MCP-Protocol-Version header must be seeded
# from this one or it advertises a revision the body does not speak.
LATEST_HANDSHAKE_VERSION = LATEST_PROTOCOL_VERSION

# Importing `mcp` costs ~260ms, so it is deferred to first real use
# (_ensure_mcp_sdk). Availability is decided here with a metadata-only
# find_spec probe so every `if not _MCP_AVAILABLE` gate / test patch / skipif
# keeps its exact semantics.
try:
    import importlib.util as _importlib_util
    _MCP_AVAILABLE = _importlib_util.find_spec("mcp") is not None
except Exception:
    _MCP_AVAILABLE = False
if not _MCP_AVAILABLE:
    logger.debug("mcp package not installed -- MCP tool support disabled")

ClientSession: Any = None
_MCP_SDK_IMPORT_ATTEMPTED = False
_MCP_SDK_IMPORT_LOCK = threading.Lock()

# SDK symbols bound by _ensure_mcp_sdk(). Module __getattr__ (PEP 562) imports
# the SDK on first external access, so mock.patch("tools.mcp_tool.stdio_client")
# sees a real original and the mock is never clobbered (_ensure is idempotent).
_MCP_SDK_LAZY_SYMBOLS = frozenset({
    "StdioServerParameters", "stdio_client",
    "streamablehttp_client", "streamable_http_client",
    "CreateMessageResult", "CreateMessageResultWithTools", "ErrorData",
    "SamplingCapability", "SamplingToolsCapability", "TextContent",
    "ToolUseContent", "ElicitRequestParams", "ElicitResult",
    "ServerNotification", "ToolListChangedNotification",
    "PromptListChangedNotification", "ResourceListChangedNotification",
})


def __getattr__(name: str):
    if name in _MCP_SDK_LAZY_SYMBOLS:
        _ensure_mcp_sdk()
        try:
            return globals()[name]
        except KeyError:
            pass  # SDK missing or symbol absent on this SDK build
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_mcp_sdk() -> bool:
    """Import the optional ``mcp`` SDK on first use; return availability.

    Idempotent and thread-safe. Honors a test-patched ``_MCP_AVAILABLE=False``
    (no import) and pre-installed mock symbols (``ClientSession`` already set
    means no re-import, so mocks are never clobbered).
    """
    global _MCP_SDK_IMPORT_ATTEMPTED, _MCP_AVAILABLE, _MCP_HTTP_AVAILABLE
    global _MCP_SAMPLING_TYPES, _MCP_NOTIFICATION_TYPES, _MCP_ELICITATION_TYPES
    global _MCP_MESSAGE_HANDLER_SUPPORTED, _MCP_LOGGING_CALLBACK_SUPPORTED
    global _MCP_NEW_HTTP, _MCP_LEGACY_HTTP, LATEST_PROTOCOL_VERSION, LATEST_HANDSHAKE_VERSION, sse_client
    global ClientSession, StdioServerParameters, stdio_client
    global streamablehttp_client, streamable_http_client
    global CreateMessageResult, CreateMessageResultWithTools, ErrorData
    global SamplingCapability, SamplingToolsCapability, TextContent, ToolUseContent
    global ElicitRequestParams, ElicitResult
    global ServerNotification, ToolListChangedNotification
    global PromptListChangedNotification, ResourceListChangedNotification

    if not _MCP_AVAILABLE:
        return False
    if _MCP_SDK_IMPORT_ATTEMPTED or ClientSession is not None:
        return _MCP_AVAILABLE
    with _MCP_SDK_IMPORT_LOCK:
        if _MCP_SDK_IMPORT_ATTEMPTED or ClientSession is not None:
            return _MCP_AVAILABLE
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            _MCP_AVAILABLE = True
            # mcp >= 1.24 ships streamable_http_client; 2.0 dropped the
            # deprecated streamablehttp_client alias. Either one gives HTTP.
            try:
                from mcp.client.streamable_http import streamable_http_client
                _MCP_NEW_HTTP = True
            except ImportError:
                _MCP_NEW_HTTP = False
            try:
                from mcp.client.streamable_http import streamablehttp_client
                _MCP_LEGACY_HTTP = True
            except ImportError:
                _MCP_LEGACY_HTTP = False
            _MCP_HTTP_AVAILABLE = _MCP_NEW_HTTP or _MCP_LEGACY_HTTP
            try:
                from mcp.types import LATEST_PROTOCOL_VERSION
            except ImportError:
                logger.debug("mcp.types.LATEST_PROTOCOL_VERSION not available -- using fallback protocol version")
            try:
                from mcp.client.session import LATEST_HANDSHAKE_VERSION
            except ImportError:
                # Pre-2.x SDKs: newest revision IS the handshake revision.
                LATEST_HANDSHAKE_VERSION = LATEST_PROTOCOL_VERSION
            try:
                from mcp.client.sse import sse_client
            except ImportError:
                sse_client = None
                logger.debug("mcp.client.sse.sse_client not available -- SSE transport disabled")
            # Optional type families are gated separately so an older SDK
            # only loses that feature, not MCP support.
            try:
                from mcp.types import (
                    CreateMessageResult,
                    CreateMessageResultWithTools,
                    ErrorData,
                    SamplingCapability,
                    SamplingToolsCapability,
                    TextContent,
                    ToolUseContent,
                )
                _MCP_SAMPLING_TYPES = True
            except ImportError:
                logger.debug("MCP sampling types not available -- sampling disabled")
            try:
                from mcp.types import ElicitRequestParams, ElicitResult
                _MCP_ELICITATION_TYPES = True
            except ImportError:
                logger.debug("MCP elicitation types not available -- elicitation disabled")
            try:
                from mcp.types import (
                    ServerNotification,
                    ToolListChangedNotification,
                    PromptListChangedNotification,
                    ResourceListChangedNotification,
                )
                _MCP_NOTIFICATION_TYPES = True
            except ImportError:
                logger.debug("MCP notification types not available -- dynamic tool discovery disabled")
        except ImportError:
            logger.debug("mcp package not installed -- MCP tool support disabled")

        if _MCP_AVAILABLE:
            try:
                from mcp.types import METHOD_NOT_FOUND as _mnf
                global _JSONRPC_METHOD_NOT_FOUND
                _JSONRPC_METHOD_NOT_FOUND = _mnf
            except Exception:  # pragma: no cover — SDK without the constant
                pass

        _MCP_MESSAGE_HANDLER_SUPPORTED = _check_message_handler_support()
        if _MCP_AVAILABLE and not _MCP_MESSAGE_HANDLER_SUPPORTED:
            logger.debug("MCP SDK does not support message_handler -- dynamic tool discovery disabled")
        _MCP_LOGGING_CALLBACK_SUPPORTED = _check_logging_callback_support()
        _MCP_SDK_IMPORT_ATTEMPTED = True
        return _MCP_AVAILABLE


_SDK_HTTPX_MOD = None


def sdk_httpx():
    """Return the httpx module the *installed* MCP SDK is built against.

    mcp 2.0 moved to ``httpx2`` (same API, separate distribution). Every
    object crossing the SDK boundary — the ``AsyncClient`` passed to the
    transport, OAuth ``Request`` objects, the exception classes — must come
    from the module the SDK itself imports, or it fails at the transport
    layer rather than at import. Resolved from the SDK's transport module, not
    a version number. ``None`` only when neither module is importable.
    """
    global _SDK_HTTPX_MOD
    if _SDK_HTTPX_MOD is not None:
        return _SDK_HTTPX_MOD
    try:
        from mcp.client import streamable_http as _transport
        _SDK_HTTPX_MOD = getattr(_transport, "httpx2", None) or getattr(
            _transport, "httpx", None
        )
    except ImportError:
        _SDK_HTTPX_MOD = None
    if _SDK_HTTPX_MOD is None:
        # Transport module missing / renamed its import: newest present wins.
        try:
            import httpx2 as _fallback
        except ImportError:
            try:
                import httpx as _fallback  # type: ignore[no-redef]
            except ImportError:
                return None
        _SDK_HTTPX_MOD = _fallback
    return _SDK_HTTPX_MOD


def _client_session_accepts(kwarg: str) -> bool:
    """Whether this SDK's ``ClientSession.__init__`` takes ``kwarg``.

    Older SDKs lack ``message_handler`` (no list_changed notifications) and
    ``logging_callback`` (server ``notifications/message`` silently dropped).
    """
    if not _MCP_AVAILABLE:
        return False
    try:
        return kwarg in inspect.signature(ClientSession).parameters
    except (TypeError, ValueError):
        return False


def _check_message_handler_support() -> bool:
    return _client_session_accepts("message_handler")


def _check_logging_callback_support() -> bool:
    return _client_session_accepts("logging_callback")


# MCP logging levels (RFC 5424 syslog severities) -> Python logging levels.
_MCP_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.ERROR,
    "alert": logging.ERROR,
    "emergency": logging.ERROR,
}

# ---------------------------------------------------------------------------
# Reconnect / keepalive tuning
# ---------------------------------------------------------------------------


_DEFAULT_CONNECT_TIMEOUT = 60    # seconds for initial connection per server
_MAX_RECONNECT_RETRIES = 5
_MAX_INITIAL_CONNECT_RETRIES = 3 # retries for the very first connection attempt
_MAX_BACKOFF_SECONDS = 60
# Parked servers (budget exhausted, tools deregistered) self-probe on this
# cadence: with no tools registered nothing else can ever revive them.
_PARKED_RETRY_INTERVAL = 300     # seconds between parked self-probes
_RECYCLED_RECONNECT_TIMEOUT = 15.0
# Bounded wait for a respawned stdio child when a call finds it dead (gateway
# restarts kill every MCP child). Bounded so a broken server still parks via
# run()'s rapid-drop budget instead of hot-cycling respawns.
_STDIO_RESPAWN_WAIT_SEC = 15.0


# Servers may expire idle sessions on any TTL, so the client MUST ping faster
# than that TTL; servers with short TTLs (~15s) need a smaller configured
# ``keepalive_interval``. The floor stops a tiny interval from busy-looping.
_DEFAULT_KEEPALIVE_INTERVAL = 180  # seconds between liveness pings
_MIN_KEEPALIVE_INTERVAL = 5        # clamp floor for configured intervals

# One bounded cancellation cycle for pending loop tasks at final shutdown, so
# cancellation-resistant tasks cannot hang process exit.
_MCP_LOOP_DRAIN_TIMEOUT = 3.0

# JSON-RPC 2.0 "method not found" (e.g. a server without the optional ``ping``).
# _ensure_mcp_sdk() overrides it from mcp.types once the SDK is loaded.
_JSONRPC_METHOD_NOT_FOUND = -32601

# Cap on nextCursor pagination so a server returning a cursor forever cannot
# spin discovery; 50 pages at 50-100 items/page covers thousands of entries.
_MCP_LIST_MAX_PAGES = 50


async def _paginate_full_list(list_method, items_attr: str, server_name: str,
                              cache_meta_out: Optional[dict] = None):
    """Drain a paginated ``list_*`` call by following ``nextCursor``.

    The SDK fetches one page per call, so without this every entry past page
    1 would be invisible. ``cache_meta_out`` receives the first page's
    SEP-2549 hints (``ttl_ms``, ``cache_scope``) when present. Callers must
    hold the server's ``_rpc_lock`` so pages come from a consistent snapshot.
    """
    items: list = []
    cursor = None
    for _ in range(_MCP_LIST_MAX_PAGES):
        if not cursor:
            result = await list_method()
        else:
            # mcp 2.0 takes params=PaginatedRequestParams, 1.x takes cursor=.
            try:
                _params_cls = getattr(_mcp_types(), "PaginatedRequestParams", None)
                if _params_cls is not None:
                    result = await list_method(params=_params_cls(cursor=cursor))
                else:
                    result = await list_method(cursor=cursor)
            except TypeError:
                result = await list_method(cursor=cursor)
        if cache_meta_out is not None and not items:
            _ttl = mcp_field(result, "ttl_ms", "ttlMs")
            _scope = mcp_field(result, "cache_scope", "cacheScope")
            if _ttl is not None:
                cache_meta_out["ttl_ms"] = _ttl
            if _scope is not None:
                cache_meta_out["cache_scope"] = _scope
        items.extend(getattr(result, items_attr, None) or [])
        cursor = mcp_field(result, "next_cursor", "nextCursor")
        # Cursor is an opaque string; anything else (incl. mocks) = last page.
        if not isinstance(cursor, str) or not cursor:
            break
    else:
        logger.warning(
            "MCP server '%s': %s pagination exceeded %d pages; "
            "truncating at %d items",
            server_name, items_attr, _MCP_LIST_MAX_PAGES, len(items),
        )
    return items


def _mcp_types():
    """Late import of ``mcp.types`` (module keeps the SDK import lazy)."""
    import mcp.types as _t
    return _t


# ---------------------------------------------------------------------------
# Server task -- each MCP server lives in one long-lived asyncio Task
# ---------------------------------------------------------------------------

class MCPServerTask(MCPServerTransportMixin, MCPServerHealthMixin):
    """One MCP server connection living in one long-lived asyncio Task.

    Connect, discover, serve and disconnect all run in that Task so the
    transport's anyio cancel scopes are entered and exited in the same Task.
    Transport bring-up lives in ``MCPServerTransportMixin``; keepalive,
    refresh and liveness in ``MCPServerHealthMixin``.
    """

    __slots__ = (
        "name", "session", "tool_timeout",
        "_task", "_ready", "_shutdown_event", "_reconnect_event",
        "_tools", "_error", "_config",
        "_sampling", "_elicitation",
        "_registered_tool_names", "_auth_type", "_refresh_lock",
        "_rpc_lock", "_pending_refresh_tasks",
        "_pending_call_context",
        "_lifecycle_started_at", "_last_tool_call_at",
        "_idle_timeout_seconds", "_max_lifetime_seconds", "_recycled_reason",
        "initialize_result", "_ping_unsupported", "_list_cache_meta",
        "_reconnect_retries", "_session_proven", "_was_parked",
        "_inflight_tasks", "_reconnecting", "_suspect_reason",
        "_teardown_race", "_permanent_grace_used", "_stdio_child_pids",
        "_ever_connected",
    )

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        # When set, _run_http/_run_stdio exit their async-with cleanly and
        # run() re-enters the transport (auth recovery, manual refresh, ...).
        self._reconnect_event = asyncio.Event()
        self._tools: list = []
        self._error: Optional[Exception] = None
        self._config: dict = {}
        self._sampling: Optional[SamplingHandler] = None
        self._elicitation: Optional[ElicitationHandler] = None
        self._registered_tool_names: list[str] = []
        self._reconnect_retries: int = 0
        # Rapid-drop budget: a (re)established session is UNPROVEN until it
        # survives a full keepalive interval or serves a successful call. Only
        # a proven session clears the reconnect budget, so a transport that
        # flaps right after the handshake still reaches the park.
        self._session_proven: bool = False
        # Set once tools were ever registered, never cleared (unlike _ready,
        # which clears every reconnect cycle): separates a first-connect
        # failure from a later reconnect failure in run()'s retry ladders.
        self._ever_connected: bool = False
        # True from park until the session proves healthy again; logs the
        # parked->revived transition exactly once.
        self._was_parked: bool = False
        # In-flight RPC tasks, so a reconnect/shutdown teardown can fail them
        # fast instead of orphaning them on a dying transport.
        self._inflight_tasks: set = set()
        # True while a deliberate teardown fails in-flight calls; lets
        # _track_inflight_rpc turn the cancel into a retryable error.
        self._reconnecting: bool = False
        # Latched by races (teardown-vs-keepalive, auth-lock corruption);
        # verified lazily by ensure_healthy() before the next call.
        self._suspect_reason: Optional[str] = None
        # A teardown that failed >=1 in-flight call makes the next reconnect a
        # RACE RECOVERY: it must not charge the rapid-drop budget.
        self._teardown_race: bool = False
        # One-time grace: an auth/permanent-classified failure on a previously
        # PROVEN session gets one suspect+reconnect cycle before the park
        # ladder applies (single auth-lock corruption must not park).
        self._permanent_grace_used: bool = False
        # Children of the current stdio transport: lets in-flight calls fail
        # FAST when the child dies instead of riding out the tool timeout.
        self._stdio_child_pids: Set[int] = set()
        self._auth_type: str = ""
        self._refresh_lock = asyncio.Lock()
        # A stdio session is one JSON-RPC stream: a list_tools issued by the
        # notification handler while a tool call is in flight can wedge it.
        # Serialize client-initiated RPCs per server (HTTP too, for ordering).
        self._rpc_lock = asyncio.Lock()
        self._pending_refresh_tasks: set[asyncio.Task] = set()
        # contextvars snapshot of the agent task inside session.call_tool().
        # The SDK dispatches elicitation/create on a separate task that does
        # not inherit HERMES_SESSION_PLATFORM; replaying this context in the
        # elicitation callback routes the approval prompt to the right surface.
        self._pending_call_context: Optional[contextvars.Context] = None
        now = time.monotonic()
        self._lifecycle_started_at: float = now
        self._last_tool_call_at: float = now
        self._idle_timeout_seconds: Optional[float] = None
        self._max_lifetime_seconds: Optional[float] = None
        self._recycled_reason: Optional[str] = None
        # InitializeResult from the handshake: the server's REAL advertised
        # capabilities, used instead of assuming every ClientSession method
        # maps to a supported server method.
        self.initialize_result: Optional[Any] = None
        # SEP-2549 cache hints from the last tools/list (ttl_ms, cache_scope).
        self._list_cache_meta: dict = {}
        # Latched when keepalive ``ping`` returns -32601 (optional utility not
        # implemented); later keepalives use list_tools instead of
        # reconnect-looping. Reset on every fresh transport connection.
        self._ping_unsupported: bool = False

    # Content types a real Streamable-HTTP endpoint may return on the initial
    # POST/GET; anything else on a 2xx means the URL is not an MCP endpoint.
    _MCP_CONTENT_TYPES = ("application/json", "text/event-stream")

    @staticmethod
    async def _cancel_waiters(*tasks: asyncio.Task) -> None:
        for t in tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _wait_for_lifecycle_event(self) -> str:
        """Serve the connection until a lifecycle event; return its kind.

        ``"shutdown"`` exits the run loop; ``"reconnect"`` tears the session
        down and re-enters the transport (event cleared before return);
        ``"recycle"`` means a stdio idle/lifetime limit elapsed and the
        transport restarts lazily on the next call. Shutdown wins a tie.

        Between events a keepalive (``ping``, list_tools fallback) runs every
        ``keepalive_interval`` — which must stay below the server's session
        TTL — and a failure triggers a reconnect. ``ping`` is a few bytes
        regardless of tool count; list_changed notifications still arrive
        out-of-band.
        """
        keepalive_interval = max(
            _MIN_KEEPALIVE_INTERVAL,
            float(self._config.get("keepalive_interval", _DEFAULT_KEEPALIVE_INTERVAL)),
        )

        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            while True:
                recycle_reason = self._stdio_recycle_reason()
                if recycle_reason is not None:
                    self._mark_stdio_recycled(recycle_reason)
                    return "recycle"

                timeout = keepalive_interval
                recycle_deadline = self._next_stdio_recycle_deadline()
                if recycle_deadline is not None:
                    timeout = max(0.0, min(timeout, recycle_deadline - time.monotonic()))

                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    break

                recycle_reason = self._stdio_recycle_reason()
                if recycle_reason is not None:
                    self._mark_stdio_recycled(recycle_reason)
                    return "recycle"

                # Timeout: probe for a stale session — but NEVER while an RPC
                # is in flight (a concurrent ping can wedge the single stdio
                # stream, and a busy server is provably alive anyway).
                if self.session:
                    if self._rpc_lock.locked() or any(
                        not t.done() for t in self._inflight_tasks
                    ):
                        continue
                    try:
                        async def _probe_under_lock():
                            async with self._rpc_lock:
                                await self._keepalive_probe()

                        await _probe_under_lock()
                    except Exception as exc:
                        root = _unwrap_exception_group(exc)
                        logger.warning(
                            "MCP server '%s' keepalive failed, triggering "
                            "reconnect (state: connected → degraded): %s: %s",
                            self.name, type(root).__name__, root,
                        )
                        self.mark_suspect(
                            f"keepalive failed: {type(root).__name__}: {root}"
                        )
                        self._reconnect_event.set()
                        break
                    # Survived a full keepalive interval: real proof of health.
                    self._mark_session_proven()
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)

        if self._shutdown_event.is_set():
            self._fail_inflight_calls("shutdown")
            return "shutdown"
        # Deliberate teardown: fail in-flight RPCs NOW rather than letting
        # them ride the dying transport to the full tool timeout.
        self._fail_inflight_calls("reconnect")
        self._reconnect_event.clear()
        return "reconnect"

    async def _wait_for_reconnect_or_shutdown(
        self, timeout: Optional[float] = None
    ) -> str:
        """Wait, while parked, for a reconnect request or shutdown.

        Returns ``"shutdown"`` or ``"reconnect"`` (explicit request or, with
        ``timeout``, the periodic self-probe); the reconnect event is cleared
        first. Shutdown wins a tie.
        """
        shutdown_task = asyncio.ensure_future(self._shutdown_event.wait())
        reconnect_task = asyncio.ensure_future(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)
        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _park(self, revival_reason: str) -> bool:
        """Drop this server's tools and wait for a reconnect request.

        The run task must NOT exit: it is the only listener on
        ``_reconnect_event``, so returning leaves the server unrevivable for
        the life of the process. Parking deregisters the tools, so no call
        can reach the breaker probe or ``_signal_reconnect``; the wait is
        therefore TIMED (one self-probe per ``_PARKED_RETRY_INTERVAL``), and
        an explicit ``_reconnect_event.set()`` wakes it immediately. Returns
        True when shutdown was requested instead.
        """
        self._was_parked = True
        self._deregister_tools()
        self._reconnect_event.clear()
        parked = await self._wait_for_reconnect_or_shutdown(
            timeout=_PARKED_RETRY_INTERVAL
        )
        if parked == "shutdown":
            return True
        logger.debug(
            "MCP server '%s': attempting revival %s (self-probe or explicit "
            "reconnect request); rebuilding transport.",
            self.name, revival_reason,
        )
        return False

    async def _prepare_run(self, config: dict) -> bool:
        """Bind config, build sampling/elicitation handlers, validate HTTP.

        Returns False when the server must not start: a bad remote URL or a
        non-MCP endpoint (both fail fast, non-retryably, with ``_error`` set
        and ``_ready`` fired) instead of burning the reconnect ladder inside
        the SDK's httpx layer on every retry.
        """
        self._config = config
        self.tool_timeout = _resolve_tool_timeout(config)
        self._auth_type = (config.get("auth") or "").lower().strip()
        self._idle_timeout_seconds = _get_lifecycle_seconds(config, "idle_timeout_seconds")
        self._max_lifetime_seconds = _get_lifecycle_seconds(config, "max_lifetime_seconds")

        # The _MCP_*_TYPES flags are False until the lazy SDK import runs.
        _ensure_mcp_sdk()

        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            self._sampling = SamplingHandler(self.name, sampling_config)
        else:
            self._sampling = None

        # elicitation/create lets a server ask for structured input mid-call;
        # the handler routes it through Hermes' approval system.
        elicitation_config = config.get("elicitation", {})
        if elicitation_config.get("enabled", True) and _MCP_ELICITATION_TYPES:
            self._elicitation = ElicitationHandler(self.name, elicitation_config, owner=self)
        else:
            self._elicitation = None

        if "url" in config and "command" in config:
            logger.warning(
                "MCP server '%s' has both 'url' and 'command' in config. "
                "Using HTTP transport ('url'). Remove 'command' to silence "
                "this warning.",
                self.name,
            )

        if not self._is_http():
            return True
        try:
            _validate_remote_mcp_url(self.name, config.get("url"))
        except InvalidMcpUrlError as exc:
            logger.warning("%s", exc)
            self._error = exc
            self._ready.set()
            return False

        # Content-type preflight (Streamable HTTP only; SSE legitimately serves
        # text/event-stream): a URL at a web-app root returns HTML and would
        # make the SDK hang for the full connect_timeout. Skipped once _ready
        # was ever set (endpoint already validated) and for OAuth servers,
        # where a token-less probe sees HTML/401 and would block the flow.
        if config.get("transport") != "sse" and not config.get("skip_preflight") and not self._ready.is_set() and self._auth_type != "oauth":
            try:
                _probe_headers = dict(config.get("headers") or {})
                await self._preflight_content_type(
                    config["url"],
                    headers=_probe_headers,
                    ssl_verify=config.get("ssl_verify", True),
                    client_cert=_resolve_client_cert(self.name, config),
                )
            except NonMcpEndpointError as exc:
                logger.warning("%s", exc)
                self._error = exc
                self._ready.set()
                return False
        return True

    async def run(self, config: dict):
        """Long-lived coroutine: connect, discover, serve, reconnect.

        State machine: connecting -> connected -> (degraded -> parked ->
        revived)*. Unproven drops and transport errors charge a rapid-drop
        budget with jittered exponential backoff; exhausting it (or a
        permanent error) parks the server via :meth:`_park` rather than
        exiting, so it stays revivable.
        """
        if not await self._prepare_run(config):
            return

        self._reconnect_retries = 0
        initial_retries = 0
        backoff = 1.0

        while True:
            try:
                if self._is_http():
                    lifecycle_reason = await self._run_http(config)
                else:
                    lifecycle_reason = await self._run_stdio(config)
                # Clean transport return: shutdown, stdio recycle, or a
                # requested rebuild (auth recovery / manual refresh / keepalive
                # failure). A rebuild is not a failure for the retry counters.
                if self._shutdown_event.is_set():
                    break
                if lifecycle_reason == "recycle":
                    logger.info(
                        "MCP server '%s': stdio session recycled after %s; "
                        "waiting for lazy reconnect",
                        self.name, self._recycled_reason,
                    )
                    self.session = None
                    await self._wait_for_lazy_reconnect()
                    if self._shutdown_event.is_set():
                        break
                    self._reconnect_event.clear()
                    continue
                # Per-cycle chatter stays DEBUG; WARNINGs mark state transitions.
                logger.debug(
                    "MCP server '%s': reconnecting (OAuth recovery or "
                    "manual refresh)",
                    self.name,
                )
                # A clean return is NOT proof of health (a flapping transport
                # handshakes fine and drops moments later). Only a PROVEN
                # session clears the budget; a teardown race is recovery, not
                # a failure, and must never reach the park on its own.
                if self._teardown_race and not self._session_proven:
                    logger.info(
                        "MCP server '%s': reconnect after teardown race "
                        "(in-flight calls were failed); not charging the "
                        "rapid-drop budget",
                        self.name,
                    )
                    self._teardown_race = False
                    backoff = 1.0
                elif self._session_proven:
                    self._reconnect_retries = 0
                    backoff = 1.0
                else:
                    self._reconnect_retries += 1
                    if self._reconnect_retries > _MAX_RECONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s': %d consecutive reconnects "
                            "without a healthy session (rapid-drop budget "
                            "exhausted), parking; will self-probe every %ds "
                            "until it recovers (state: degraded → parked)",
                            self.name, _MAX_RECONNECT_RETRIES,
                            _PARKED_RETRY_INTERVAL,
                        )
                        if await self._park("from parked state"):
                            break
                        # Budget of one probe per wake, so a still-dead server
                        # parks again instead of burning 5 rapid retries.
                        self._reconnect_retries = _MAX_RECONNECT_RETRIES
                        backoff = 1.0
                # Clear readiness too: a stale _ready lets handler-side
                # recovery mistake the old session for a fresh one.
                self._ready.clear()
                self.session = None
                continue
            except asyncio.CancelledError:
                # Not a connection failure: re-raise so cancellation reaches
                # asyncio and shutdown()'s ``await self._task`` completes.
                self.session = None
                raise
            except Exception as exc:
                self.session = None
                # Unwrap anyio TaskGroup wrappers: the group's str() is useless
                # and hides the root cause from the classification below.
                root = _unwrap_exception_group(exc)
                failure_class = _classify_mcp_failure(root)
                if self._is_recycled_stdio():
                    logger.warning(
                        "MCP server '%s': lazy reconnect after stdio recycle "
                        "failed, marking unavailable while retrying: %s: %s",
                        self.name, type(root).__name__, root,
                    )
                    self._recycled_reason = None

                # Initial-connect ladder: a transient blip at startup must not
                # kill the server. Gated on _ever_connected (never cleared),
                # not _ready (cleared every reconnect cycle).
                if not self._ever_connected:
                    if failure_class == "permanent":
                        # Deterministic failure (bad command, non-MCP URL,
                        # 401/403): park at once instead of burning the ladder.
                        # Auth failures park rather than return so the task
                        # stays alive to pick up fresh tokens later.
                        if _is_auth_error(root):
                            logger.warning(
                                "MCP server '%s' failed initial authentication, "
                                "parking until credentials change; re-authenticate "
                                "with `hermes mcp login %s` "
                                "(state: connecting → parked): %s: %s",
                                self.name, self.name,
                                type(root).__name__, root,
                            )
                        else:
                            logger.warning(
                                "MCP server '%s' failed initial connection with a "
                                "permanent error, parking without retries "
                                "(state: connecting → parked): %s: %s",
                                self.name, type(root).__name__, root,
                            )
                        self._error = exc
                        self._ready.set()
                        if await self._park("after permanent initial failure"):
                            return
                        initial_retries = 0
                        self._reconnect_retries = 0
                        backoff = 1.0
                        self._error = None
                        self._ready.clear()
                        continue

                    initial_retries += 1
                    if initial_retries > _MAX_INITIAL_CONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s' failed initial connection after "
                            "%d attempts, parking until a reconnect is "
                            "requested (state: connecting → parked): %s: %s",
                            self.name, _MAX_INITIAL_CONNECT_RETRIES,
                            type(root).__name__, root,
                        )
                        self._error = exc
                        self._ready.set()
                        if await self._park("after initial connection failures"):
                            return
                        initial_retries = 0
                        self._reconnect_retries = 0
                        backoff = 1.0
                        self._error = None
                        self._ready.clear()
                        continue

                    logger.debug(
                        "MCP server '%s' initial connection failed "
                        "(attempt %d/%d), retrying in %.0fs: %s: %s",
                        self.name, initial_retries,
                        _MAX_INITIAL_CONNECT_RETRIES, backoff,
                        type(root).__name__, root,
                    )
                    await asyncio.sleep(_jittered(backoff))
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                    # Check if shutdown was requested during the sleep
                    if self._shutdown_event.is_set():
                        self._error = exc
                        self._ready.set()
                        return
                    continue

                # If shutdown was requested, don't reconnect
                if self._shutdown_event.is_set():
                    logger.debug(
                        "MCP server '%s' disconnected during shutdown: %s: %s",
                        self.name, type(root).__name__, root,
                    )
                    return

                if failure_class == "permanent":
                    # An auth failure on a PROVEN session is often a corrupt
                    # OAuth lock from a raced teardown, not revoked
                    # credentials: grant ONE suspect+reconnect cycle first.
                    if (
                        _is_auth_error(root)
                        and self._session_proven
                        and not self._permanent_grace_used
                    ):
                        self._permanent_grace_used = True
                        self.mark_suspect(
                            f"auth error on proven session: {root}"
                        )
                        logger.warning(
                            "MCP server '%s': auth error on a previously "
                            "healthy session — marking suspect and forcing "
                            "one reconnect instead of parking (state: "
                            "connected → suspect): %s: %s",
                            self.name, type(root).__name__, root,
                        )
                        self._reconnect_retries = 0
                        backoff = 1.0
                        await asyncio.sleep(_jittered(1.0))
                        if self._shutdown_event.is_set():
                            return
                        continue
                    # Deterministic failure on a working server: park now.
                    logger.warning(
                        "MCP server '%s' hit a permanent error, parking "
                        "without retries; will self-probe every %ds "
                        "(state: connected → parked): %s: %s",
                        self.name, _PARKED_RETRY_INTERVAL,
                        type(root).__name__, root,
                    )
                    if await self._park("from parked state (permanent error)"):
                        return
                    self._reconnect_retries = _MAX_RECONNECT_RETRIES
                    backoff = 1.0
                    continue

                self._reconnect_retries += 1
                if self._reconnect_retries > _MAX_RECONNECT_RETRIES:
                    logger.warning(
                        "MCP server '%s' failed after %d reconnection attempts, "
                        "parking; will self-probe every %ds until it recovers "
                        "(state: degraded → parked): %s: %s",
                        self.name, _MAX_RECONNECT_RETRIES,
                        _PARKED_RETRY_INTERVAL,
                        type(root).__name__, root,
                    )
                    if await self._park("from parked state"):
                        return
                    self._reconnect_retries = _MAX_RECONNECT_RETRIES
                    backoff = 1.0
                    continue

                logger.debug(
                    "MCP server '%s' connection lost (attempt %d/%d), "
                    "reconnecting in %.0fs: %s: %s",
                    self.name, self._reconnect_retries, _MAX_RECONNECT_RETRIES,
                    backoff, type(root).__name__, root,
                )
                await asyncio.sleep(_jittered(backoff))
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                # Check again after sleeping
                if self._shutdown_event.is_set():
                    return
            finally:
                self.session = None
                # Stale PIDs must never fast-fail the NEXT transport's calls.
                self._stdio_child_pids = set()

    async def start(self, config: dict):
        """Create the background Task and wait until ready (or failed)."""
        self._task = asyncio.ensure_future(self.run(config))
        try:
            await self._ready.wait()
        except asyncio.CancelledError:
            # The caller's connect timeout (discover_mcp_tools wraps start()
            # in asyncio.wait_for) cancels *this* coroutine, but the
            # ensure_future'd run() task is independent and would otherwise
            # keep running detached — parked on a hung transport with no
            # owner to reap it (#59349). Propagate the cancellation so the
            # transport context managers unwind and their finally blocks
            # release the child process / FDs.
            if self._task and not self._task.done():
                self._task.cancel()
            raise
        if self._error:
            raise self._error

    async def shutdown(self):
        """Signal the Task to exit and wait for clean resource teardown."""
        self._shutdown_event.set()
        # Defensive: if _wait_for_lifecycle_event is blocking, we need ANY
        # event to unblock it. _shutdown_event alone is sufficient (the
        # helper checks shutdown first), but setting reconnect too ensures
        # there's no race where the helper misses the shutdown flag after
        # returning "reconnect".
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP server '%s' shutdown timed out, cancelling task",
                    self.name,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._pending_refresh_tasks:
            for task in list(self._pending_refresh_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_refresh_tasks, return_exceptions=True)
            self._pending_refresh_tasks.clear()
        self._deregister_tools()
        self.session = None

    def _deregister_tools(self) -> None:
        """Drop this server's tools from the global registry (idempotent).

        Pulls the server's tool schemas out of the registry so the agent
        stops advertising them to the model. Called on shutdown AND when the
        reconnect budget is exhausted, so a dead server never leaves phantom
        tool definitions bloating the prompt cache and producing "not
        connected" errors on every turn.
        """
        from tools.registry import registry

        for tool_name in list(getattr(self, "_registered_tool_names", [])):
            registry.deregister(tool_name, scope=_server_registry_scope(self.name))
            _forget_mcp_tool_server(tool_name)
        self._registered_tool_names = []

    async def _wait_for_lazy_reconnect(self) -> None:
        """Wait while an intentionally recycled stdio server is dormant."""
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)


# ---------------------------------------------------------------------------
# Module-level state (every mutation under ``_lock``)
# ---------------------------------------------------------------------------

_servers: Dict[str, MCPServerTask] = {}
# Profile registry scope owning each live connection (None outside multiplex):
# a multiplexed /reload-mcp tears down only its own profile's servers.
_server_scope_keys: Dict[str, Optional[str]] = {}
_server_connecting: set[str] = set()
_server_connect_errors: Dict[str, str] = {}
# Lazy startup: servers registered from the on-disk schema cache without
# connecting; popped once a real connection is established on first use.
_lazy_server_configs: Dict[str, dict] = {}
_lazy_server_fingerprints: Dict[str, str] = {}
_lazy_server_tool_names: Dict[str, List[str]] = {}
# Discovery installs a task-local claim around ``_connect_server`` so it can
# retain a recoverable parked task without standalone probe calls publishing
# failed servers into module-global ownership.
_connect_server_claim: contextvars.ContextVar[
    Optional[Callable[[MCPServerTask], None]]
] = contextvars.ContextVar("mcp_connect_server_claim", default=None)

# Per-server connect cooldown. A server that fails to spawn never reaches
# ``_servers``, so without this every ``discover_mcp_tools()`` (one per worker
# session) would respawn it from scratch — a restart storm whose unreaped
# subprocesses destabilise the healthy co-located servers. Failed attempts
# stamp an exponential-backoff deadline that ``register_mcp_servers`` honours;
# a successful connection clears it.
_server_connect_retry_after: Dict[str, float] = {}   # name -> monotonic deadline
_server_connect_failures: Dict[str, int] = {}        # name -> consecutive failures
_CONNECT_RETRY_BASE_BACKOFF_SEC = 30.0
_CONNECT_RETRY_MAX_BACKOFF_SEC = 600.0


def _record_connect_failure(server_name: str) -> None:
    """Stamp a geometric, capped retry cooldown after a failed connect (under ``_lock``)."""
    n = _server_connect_failures.get(server_name, 0) + 1
    _server_connect_failures[server_name] = n
    backoff = min(
        _CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)),
        _CONNECT_RETRY_MAX_BACKOFF_SEC,
    )
    _server_connect_retry_after[server_name] = time.monotonic() + backoff


def _clear_connect_failure(server_name: str) -> None:
    """Clear the connect-cooldown state after a successful connection."""
    _server_connect_failures.pop(server_name, None)
    _server_connect_retry_after.pop(server_name, None)


def _connect_cooldown_active(server_name: str) -> bool:
    """Return True if ``server_name`` is still within its retry cooldown."""
    deadline = _server_connect_retry_after.get(server_name)
    return deadline is not None and time.monotonic() < deadline

# Circuit breaker per server: closed (count < threshold) -> open (calls
# short-circuit with a "stop retrying" message until the cooldown elapses) ->
# half-open (next call is a probe; success closes, failure re-arms). Mutate
# only via _bump_server_error / _reset_server_error, which keep the count and
# the open timestamp in sync.
_server_error_counts: Dict[str, int] = {}
_server_breaker_opened_at: Dict[str, float] = {}
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0

# Trust-tier gating (``mcp_servers.<name>.trust: full | untrusted``). On an
# untrusted server every write-capable call needs user approval before the
# RPC fires; a tool is write-capable unless its discovery-time
# ``annotations.readOnlyHint`` is exactly True (malformed fails closed).
# Security model: readOnlyHint is a server-supplied HINT and a hostile server
# can lie, but on an untrusted server a lie can only skip approval for calls
# the operator was already warned about — never widen access. Missing
# ``trust`` defaults to full (backward compatible); any unrecognized value
# normalizes to untrusted (a typo must never disable the gate). Classified
# at CALL time from DISCOVERY data: no schema mutation, prompt cache intact.
_server_trust_levels: Dict[str, str] = {}
_tool_read_only_hints: Dict[str, Dict[str, bool]] = {}

_TRUST_FULL = "full"
_TRUST_UNTRUSTED = "untrusted"


def _bump_server_error(server_name: str) -> None:
    """Count a failure; at the threshold (re)stamp the breaker-open time."""
    n = _server_error_counts.get(server_name, 0) + 1
    _server_error_counts[server_name] = n
    if n >= _CIRCUIT_BREAKER_THRESHOLD:
        _server_breaker_opened_at[server_name] = time.monotonic()


def _reset_server_error(server_name: str) -> None:
    """Close the breaker on any unambiguous success signal."""
    _server_error_counts[server_name] = 0
    _server_breaker_opened_at.pop(server_name, None)


def _signal_reconnect(server: Any) -> bool:
    """Ask a server task to rebuild its transport, thread-safely.

    Handlers run on caller threads while the event lives on the MCP loop, so
    it is set via ``call_soon_threadsafe`` when the loop runs (direct
    ``.set()`` otherwise). False when the server has no reconnect machinery.
    """
    event = getattr(server, "_reconnect_event", None)
    if event is None:
        return False
    loop = _mcp_loop
    if (
        isinstance(event, asyncio.Event)
        and loop is not None
        and loop.is_running()
    ):
        loop.call_soon_threadsafe(event.set)
    else:
        event.set()
    return True


def reconnect_mcp_server(server_name: str) -> bool:
    """Ask a currently-live MCP server to rebuild after external re-auth."""
    with _lock:
        server = _servers.get(server_name)
    if server is None:
        return False
    return _signal_reconnect(server)


def _wait_for_server_session_ready(
    srv: "MCPServerTask",
    *,
    old_session: Any = None,
    timeout: float = 15.0,
) -> bool:
    """Poll until the server exposes a usable, ready session.

    During a reconnect ``srv.session`` is briefly None or still the stale
    object; retrying blindly there burns breaker strikes. With
    ``old_session`` the observed session must differ from it. Iteration-
    bounded, not deadline-bounded: tests freeze ``time.monotonic``.
    """
    poll_interval = 0.25
    iterations = max(1, int(max(float(timeout), 0.0) / poll_interval))
    for i in range(iterations):
        session = getattr(srv, "session", None)
        ready = getattr(srv, "_ready", None)
        is_ready = True
        if ready is not None and hasattr(ready, "is_set"):
            try:
                is_ready = bool(ready.is_set())
            except Exception:
                is_ready = True
        if session is not None and session is not old_session and is_ready:
            return True
        if i < iterations - 1:
            time.sleep(poll_interval)
    return False


def _signal_reconnect_and_wait(
    server_name: str,
    srv: "MCPServerTask",
    *,
    op_description: str,
    timeout: float = 15.0,
) -> bool:
    """Request a transport rebuild and wait for the fresh session.

    ``_ready`` is cleared on the loop BEFORE ``_reconnect_event`` is set;
    otherwise the readiness poll returns immediately and retries against the
    same dead session.
    """
    loop = _mcp_loop
    if loop is None or not loop.is_running():
        return False

    old_session = getattr(srv, "session", None)

    def _request_reconnect() -> None:
        ready = getattr(srv, "_ready", None)
        if ready is not None and hasattr(ready, "clear"):
            ready.clear()
        reconnect_event = getattr(srv, "_reconnect_event", None)
        if reconnect_event is not None and hasattr(reconnect_event, "set"):
            reconnect_event.set()

    logger.info(
        "MCP server '%s': %s requesting transport reconnect",
        server_name, op_description,
    )
    loop.call_soon_threadsafe(_request_reconnect)
    return _wait_for_server_session_ready(
        srv,
        old_session=old_session,
        timeout=timeout,
    )

# Raw server names opted into parallel tool calls. Raw identity matters:
# ``foo-bar`` and ``foo_bar`` both sanitize to ``foo_bar`` but must not share
# policy.
_parallel_safe_servers: set = set()
# registry tool name -> raw server name, captured at registration. The
# generated name is lossy (punctuation -> ``_``), so never re-parse it.
_mcp_tool_server_names: Dict[str, str] = {}

# Dedicated event loop running in a background daemon thread.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None
# Guards the loop handles, _servers, the status maps and the PID ledgers.
_lock = threading.Lock()


def _mcp_registry_scope() -> Optional[str]:
    """Registry scope for MCP registrations from the current context.

    Under a profile multiplexer each profile's MCP tools live in its own
    registry overlay; single-profile processes stay process-global (None).
    """
    from agent.secret_scope import is_multiplex_active

    if not is_multiplex_active():
        return None
    from tools.registry import registry

    return registry.current_scope_key()


def _server_registry_scope(name: str) -> Optional[str]:
    """Scope owning server *name*'s tools: recorded at connect, else current.

    Teardown runs on the MCP loop without the discovering profile's context,
    so the scope captured at adoption into ``_servers`` is authoritative.
    """
    if name in _server_scope_keys:
        return _server_scope_keys[name]
    return _mcp_registry_scope()


# ---------------------------------------------------------------------------
# Cross-process MCP discovery guard: advisory file lock so gateway + CLI + TUI
# don't all run discovery at once.
# ---------------------------------------------------------------------------
_LOCK_UNAVAILABLE: Any = object()  # sentinel: locking broken/unavailable
_MCP_DISCOVERY_LOCK_PATH: Optional[str] = None  # resolved lazily
# Bounded wait when another process holds the lock.
_MCP_DISCOVERY_LOCK_MAX_RETRIES: int = 240
_MCP_DISCOVERY_LOCK_RETRY_DELAY_S: float = 0.5


class _LockCookie:
    """Holds a cross-process file lock; ``release()`` drops it.

    The file object MUST stay open while the lock is held: both the fcntl and
    the portalocker lock are tied to the descriptor's lifetime.
    """

    def __init__(self, fh: Any) -> None:
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fd = self._fh.fileno()
                if os.name == "posix":
                    import fcntl
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
                else:
                    import portalocker
                    try:
                        portalocker.unlock(self._fh)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def _acquire_lock_on_fh(fh: Any) -> bool:
    """Non-blocking exclusive lock (fcntl on POSIX, portalocker elsewhere).

    False when another process holds it; unexpected errors propagate so the
    caller can treat locking as unavailable.
    """
    fd = fh.fileno()
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
    else:
        import portalocker
        try:
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            return True
        except portalocker.LockException:
            return False


def _try_acquire_mcp_discovery_lock() -> Any:
    """Return a ``_LockCookie`` (acquired), ``None`` (held by another process)
    or ``_LOCK_UNAVAILABLE`` (locking broken: run discovery unguarded)."""
    global _MCP_DISCOVERY_LOCK_PATH
    try:
        from hermes_constants import get_hermes_home
        if _MCP_DISCOVERY_LOCK_PATH is None:
            _MCP_DISCOVERY_LOCK_PATH = str(
                get_hermes_home() / ".mcp-discovery.lock"
            )
        lock_path = _MCP_DISCOVERY_LOCK_PATH
    except Exception:
        return _LOCK_UNAVAILABLE

    try:
        fh = open(lock_path, "w", encoding="utf-8")
    except Exception:
        return _LOCK_UNAVAILABLE

    try:
        acquired = _acquire_lock_on_fh(fh)
    except Exception:
        fh.close()
        return _LOCK_UNAVAILABLE

    if acquired:
        return _LockCookie(fh)
    fh.close()
    return None


def _mcp_loop_exception_handler(loop, context):
    """Suppress the benign 'Event loop is closed' RuntimeError that httpx
    finalizers raise against the dead loop during shutdown; forward the rest."""
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return
    loop.default_exception_handler(context)


def _ensure_mcp_loop():
    """Start the background event loop thread if not already running."""
    global _mcp_loop, _mcp_thread
    with _lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            return
        _mcp_loop = asyncio.new_event_loop()
        _mcp_loop.set_exception_handler(_mcp_loop_exception_handler)
        _mcp_thread = threading.Thread(
            target=_mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _mcp_thread.start()


def _wrap_with_home_override(coro: "Coroutine") -> "Coroutine":
    """Carry the caller's context-local HERMES_HOME override into ``coro``
    (task-local on the MCP loop, so concurrent scopes don't interfere)."""
    try:
        from hermes_constants import (
            get_hermes_home_override,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_override = get_hermes_home_override()
    except Exception:
        return coro
    if not home_override:
        return coro

    async def _scoped():
        token = set_hermes_home_override(home_override)
        try:
            return await coro
        finally:
            reset_hermes_home_override(token)

    return _scoped()


def _wrap_with_dashboard_oauth_flow(coro):
    """Propagate a dashboard OAuth flow onto the dedicated MCP loop task."""
    try:
        from tools.mcp_dashboard_oauth import (
            dashboard_oauth_flow,
            get_dashboard_oauth_flow,
        )

        flow = get_dashboard_oauth_flow()
    except Exception:
        return coro
    if flow is None:
        return coro

    async def _scoped():
        with dashboard_oauth_flow(flow):
            return await coro

    return _scoped()


def _run_on_mcp_loop(coro_or_factory, timeout: float = 30):
    """Schedule a coroutine on the MCP loop and block until done.

    Accepts a coroutine or a zero-arg factory (a factory avoids leaking a
    never-awaited coroutine when the loop is down). Polls in short intervals
    so the calling thread can honor user interrupts.
    """
    from tools.interrupt import is_interrupted
    from agent.async_utils import safe_schedule_threadsafe

    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        if asyncio.iscoroutine(coro_or_factory):
            coro_or_factory.close()
        raise RuntimeError("MCP event loop is not running")

    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

    # Tasks created via run_coroutine_threadsafe copy the LOOP thread's
    # context, so a per-request profile scope would vanish here; re-establish
    # it inside the task's own context.
    coro = _wrap_with_home_override(coro)
    coro = _wrap_with_dashboard_oauth_flow(coro)

    future = safe_schedule_threadsafe(
        coro, loop,
        logger=logger,
        log_message="MCP scheduling failed",
    )
    if future is None:
        raise RuntimeError("MCP event loop unavailable (failed to schedule)")
    start_time = time.monotonic()
    deadline = None if timeout is None else start_time + timeout

    while True:
        if is_interrupted():
            future.cancel()
            raise InterruptedError("User sent a new message")

        wait_timeout = 0.1
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                elapsed = time.monotonic() - start_time
                raise TimeoutError(
                    f"MCP call timed out after {elapsed:.1f}s "
                    f"(configured timeout: {float(timeout):.1f}s)"
                )
            wait_timeout = min(wait_timeout, remaining)

        try:
            return future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            # Aliases builtin TimeoutError, so this also fires for the
            # coroutine's own timeout: a done future must yield its outcome.
            if future.done():
                return future.result()
            continue


# ---------------------------------------------------------------------------
# Connecting, lazy start, discovery
# ---------------------------------------------------------------------------

async def _connect_server(name: str, config: dict) -> MCPServerTask:
    """Create an MCPServerTask, start it and return once ready.

    Tear it down with ``server.shutdown()`` on the same loop. Raises on bad
    config, missing HTTP support, or connect/initialize failure.
    """
    server = MCPServerTask(name)
    claim = _connect_server_claim.get()
    claim_token = None
    if claim is not None:
        claim(server)
        # The run task copies this context; the claim is for this attempt
        # only, so don't retain the discovery closure for the server's life.
        claim_token = _connect_server_claim.set(None)
    try:
        await server.start(config)
    except asyncio.CancelledError:
        # start() already reaps server._task; a shutdown() here could
        # swallow the cancellation.
        raise
    except BaseException:
        # Discovery owns claimed tasks (recoverable park vs terminal failure);
        # standalone probes have no revival owner and must reap locally.
        if claim is None:
            try:
                await server.shutdown()
            except Exception as shutdown_exc:  # noqa: BLE001 -- best-effort reap, don't mask the real error
                logger.debug(
                    "MCP server '%s' shutdown during orphan-reap failed: %s",
                    name, shutdown_exc,
                )
        raise
    finally:
        if claim_token is not None:
            _connect_server_claim.reset(claim_token)
    return server


def _request_lazy_reconnect(server_name: str, server: MCPServerTask) -> bool:
    """Wake a recycled stdio server and wait briefly for a fresh session."""
    if not server._is_recycled_stdio():
        return False

    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        return False

    def _signal_reconnect() -> None:
        server._ready.clear()
        server._reconnect_event.set()

    loop.call_soon_threadsafe(_signal_reconnect)

    async def _await_ready() -> bool:
        deadline = time.monotonic() + _RECYCLED_RECONNECT_TIMEOUT
        while time.monotonic() < deadline:
            if server.session is not None and server._ready.is_set():
                return True
            await asyncio.sleep(0.05)
        return False

    try:
        return bool(_run_on_mcp_loop(_await_ready, timeout=_RECYCLED_RECONNECT_TIMEOUT))
    except Exception as exc:
        logger.warning(
            "MCP server '%s': lazy reconnect after stdio recycle failed: %s",
            server_name, exc,
        )
        return False


def _resolve_server_lazy(name: str, config: dict) -> bool:
    """True when ``mcp_servers.<name>.lazy`` defers connect to first tool use (default off)."""
    return _parse_boolish(config.get("lazy", False), default=False)


def _ensure_lazy_server_connected(server_name: str) -> bool:
    """Connect a lazily-registered server on demand (sync; blocks the caller).

    Honours the connect cooldown and the ``_server_connecting`` dedup set and
    routes through ``_discover_and_register_server`` so park/recycle/cooldown
    bookkeeping stays in one place. True when a live session exists after.
    """
    with _lock:
        server = _servers.get(server_name)
        if server is not None and server.session is not None:
            return True
        config = _lazy_server_configs.get(server_name)
        if not config:
            return False
        if _connect_cooldown_active(server_name):
            return False
        if server_name in _server_connecting:
            return False
        _server_connecting.add(server_name)
        _server_connect_errors.pop(server_name, None)

    logger.info("MCP server '%s': lazy start on first use", server_name)
    _ensure_mcp_loop()
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)

    async def _connect():
        return await _discover_and_register_server(server_name, config)

    try:
        _run_on_mcp_loop(_connect, timeout=float(connect_timeout) + 30.0)
    except BaseException as exc:
        message = _format_connect_error(exc)
        with _lock:
            _server_connecting.discard(server_name)
            _server_connect_errors[server_name] = message
            _record_connect_failure(server_name)
        logger.warning(
            "Lazy MCP connect failed for '%s': %s", server_name, message,
        )
        return False

    with _lock:
        _server_connecting.discard(server_name)
        _clear_connect_failure(server_name)
        _lazy_server_configs.pop(server_name, None)
        stale_fingerprint = _lazy_server_fingerprints.pop(server_name, None)
        cached_names = _lazy_server_tool_names.pop(server_name, None) or []
        server = _servers.get(server_name)
        live_names = set(
            getattr(server, "_registered_tool_names", []) or []
        )
    # The cached manifest may advertise tools the live server no longer
    # serves; deregister those phantoms.
    phantom_names = [n for n in cached_names if n not in live_names]
    if phantom_names:
        from tools.registry import registry

        for tool_name in phantom_names:
            registry.deregister(tool_name, scope=_server_registry_scope(server_name))
            _forget_mcp_tool_server(tool_name)
        logger.info(
            "MCP server '%s': deregistered %d phantom cached tool(s) not "
            "served live (stale schema-cache fingerprint %s): %s",
            server_name, len(phantom_names), stale_fingerprint,
            ", ".join(phantom_names),
        )
    return server is not None and server.session is not None


def _get_connected_server_for_call(server_name: str) -> Optional[MCPServerTask]:
    """Return a connected server; the single first-use connect point for lazy
    servers and the wake-up point for recycled stdio ones."""
    with _lock:
        server = _servers.get(server_name)
        is_lazy = server_name in _lazy_server_configs
    if is_lazy and (server is None or server.session is None):
        _ensure_lazy_server_connected(server_name)
        with _lock:
            server = _servers.get(server_name)
        return server
    if server is not None and server.session is None and server._is_recycled_stdio():
        _request_lazy_reconnect(server_name, server)
        with _lock:
            server = _servers.get(server_name)
    return server


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """Connect one server, register its tools; return the registered names."""
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    # The claim callback runs inside _connect_server while this frame is
    # suspended; a list append avoids a nonlocal rebind.
    claimed: List[MCPServerTask] = []

    def _claim_server(created: MCPServerTask) -> None:
        claimed.append(created)

    claim_token = _connect_server_claim.set(_claim_server)
    try:
        server = await asyncio.wait_for(
            _connect_server(name, config),
            timeout=connect_timeout,
        )
    except BaseException:
        server = claimed[0] if claimed else None
        task = server._task if server is not None else None
        task_cancelling = (
            task.cancelling()
            if task is not None and hasattr(task, "cancelling")
            else 0
        )
        if (
            server is not None
            and server._error is not None
            and task is not None
            and not task.done()
            and not task_cancelling
        ):
            # Recoverable park: the run task stays alive to self-probe, so
            # adopt it for shutdown/revival.
            with _lock:
                _servers[name] = server
                _server_scope_keys[name] = _mcp_registry_scope()
        elif server is not None:
            await server.shutdown()
        raise
    finally:
        _connect_server_claim.reset(claim_token)

    with _lock:
        _server_connecting.discard(name)
        _server_connect_errors.pop(name, None)
        _servers[name] = server
        _server_scope_keys[name] = _mcp_registry_scope()

    registered_names = _register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)

    transport_type = "HTTP" if "url" in config else "stdio"
    logger.info(
        "MCP server '%s' (%s): registered %d tool(s): %s",
        name, transport_type, len(registered_names),
        ", ".join(registered_names),
    )
    return registered_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """Connect the given ``{name: config}`` servers and register their tools.

    Idempotent for connected names; ``enabled: false`` servers are skipped
    without disconnecting existing sessions. Returns every registered MCP
    tool name.
    """
    if not _ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping explicit MCP registration")
        return []

    servers = _filter_suspicious_mcp_servers(servers)
    if not servers:
        logger.debug("No explicit MCP servers provided")
        return []

    # Candidates: enabled, not connected, not connecting (dedups concurrent
    # discovery entry points), not lazily registered, not in backoff.
    with _lock:
        connecting = set(_server_connecting)
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _servers
            and k not in connecting
            and k not in _lazy_server_configs
            and _parse_boolish(v.get("enabled", True), default=True)
            and not _connect_cooldown_active(k)
        }
        # Known servers without a live session are parked or mid-reconnect;
        # their tools are deregistered so nothing else can nudge them.
        stale_cached = [
            _servers[k]
            for k in servers
            if k in _servers and getattr(_servers[k], "session", None) is None
        ]
        _server_connecting.update(new_servers)
        for srv_name in new_servers:
            _server_connect_errors.pop(srv_name, None)
        # Track which servers opt-in to parallel tool calls (idempotent).
        for srv_name, srv_cfg in servers.items():
            if _parse_boolish(srv_cfg.get("supports_parallel_tool_calls", False), default=False):
                _parallel_safe_servers.add(srv_name)
            else:
                _parallel_safe_servers.discard(srv_name)

    for srv in stale_cached:
        _signal_reconnect(srv)

    if not new_servers:
        return _existing_tool_names()

    # ``lazy: true`` servers with a valid schema-cache entry register from
    # cache without connecting; a missing/stale entry falls back to eager.
    eager_servers: Dict[str, dict] = dict(new_servers)
    lazy_registered = 0
    lazy_server_count = 0
    try:
        from tools.mcp_schema_cache import config_fingerprint, get_cached_entry
    except Exception:  # pragma: no cover - cache module missing
        config_fingerprint = None  # type: ignore[assignment]
        get_cached_entry = None  # type: ignore[assignment]
    if config_fingerprint is not None and get_cached_entry is not None:
        for name, cfg in new_servers.items():
            if not _resolve_server_lazy(name, cfg):
                continue
            entry = get_cached_entry(name, config_fingerprint(cfg))
            if not entry:
                continue
            with _lock:
                _server_connecting.discard(name)
            try:
                names = _register_from_cache_sync(name, cfg, entry)
            except Exception as exc:
                logger.warning(
                    "Failed lazy MCP registration for '%s': %s", name, exc,
                )
                with _lock:
                    _server_connecting.add(name)
                continue
            eager_servers.pop(name, None)
            lazy_registered += len(names)
            lazy_server_count += 1
    new_servers = eager_servers

    if not new_servers:
        if lazy_registered:
            logger.info(
                "MCP: registered %d lazy tool(s) from schema cache "
                "(no processes spawned)",
                lazy_registered,
            )
        return _existing_tool_names()

    _ensure_mcp_loop()

    async def _discover_all():
        server_names = list(new_servers.keys())
        results = await asyncio.gather(
            *(_discover_and_register_server(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, BaseException):
                command = new_servers.get(name, {}).get("command")
                message = _format_connect_error(result)
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors[name] = message
                    _record_connect_failure(name)
                logger.warning(
                    "Failed to connect to MCP server '%s'%s: %s",
                    name,
                    f" (command={command})" if command else "",
                    message,
                )
            else:
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors.pop(name, None)
                    _clear_connect_failure(name)

    # Clear a stale interrupt flag (executor threads are reused) so a prior
    # session's interrupt cannot cancel this discovery pass.
    from tools.interrupt import is_interrupted as _is_interrupted, set_interrupt as _set_interrupt
    _was_interrupted = _is_interrupted()
    if _was_interrupted:
        _set_interrupt(False)
    try:
        _run_on_mcp_loop(_discover_all, timeout=120)
    except (TimeoutError, InterruptedError) as _e:
        # Entries stranded in _server_connecting would block future
        # reconnect attempts.
        with _lock:
            stale = [n for n in new_servers if n in _server_connecting]
            if stale:
                logger.warning(
                    "MCP discovery %s while %d server(s) were still "
                    "connecting; clearing stale connecting set: %s",
                    "timed out" if isinstance(_e, TimeoutError) else "interrupted",
                    len(stale),
                    ", ".join(stale),
                )
                _server_connecting.difference_update(stale)
                for _sn in stale:
                    _server_connect_errors.setdefault(
                        _sn,
                        f"Connection attempt {'timed out' if isinstance(_e, TimeoutError) else 'interrupted'} during discovery",
                    )
        raise
    finally:
        if _was_interrupted:
            _set_interrupt(True)

    with _lock:
        connected = [
            n
            for n in new_servers
            if n in _servers and n not in _server_connect_errors
        ]
        new_tool_count = sum(
            len(getattr(_servers[n], "_registered_tool_names", []))
            for n in connected
        )
    failed = len(new_servers) - len(connected)
    new_tool_count += lazy_registered
    connected_count = len(connected) + lazy_server_count
    if new_tool_count or failed:
        summary = f"MCP: registered {new_tool_count} tool(s) from {connected_count} server(s)"
        if failed:
            summary += f" ({failed} failed)"
        logger.info(summary)

    return _existing_tool_names()


def discover_mcp_tools() -> List[str]:
    """Entry point: load config, connect servers, register tools.

    Safe without the ``mcp`` package (returns []). Idempotent: only servers
    missing from a previous call are retried. Returns all MCP tool names.
    """
    servers = _load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []

    # SDK import deferred to here so a config without servers never pays it.
    if not _ensure_mcp_sdk():
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []

    # Cross-process guard: a lock loser waits for the holder, then runs its
    # own discovery; if locking is unavailable or the wait expires, run
    # unguarded (fail-soft).
    cookie = _try_acquire_mcp_discovery_lock()
    if cookie is None:
        logger.debug(
            "Another process holds MCP discovery lock -- retrying with backoff"
        )
        for _ in range(_MCP_DISCOVERY_LOCK_MAX_RETRIES):
            time.sleep(_MCP_DISCOVERY_LOCK_RETRY_DELAY_S)
            cookie = _try_acquire_mcp_discovery_lock()
            if cookie is not None:
                break

        if cookie is None:
            logger.warning(
                "MCP discovery lock still held after %d retries -- "
                "running discovery unguarded",
                _MCP_DISCOVERY_LOCK_MAX_RETRIES,
            )
        elif cookie is not _LOCK_UNAVAILABLE:
            logger.debug("Retry succeeded -- acquired MCP discovery lock")

    try:
        with _lock:
            connecting = set(_server_connecting)
            new_server_names = [
                name
                for name, cfg in servers.items()
                if name not in _servers
                and name not in connecting
                and _parse_boolish(cfg.get("enabled", True), default=True)
            ]

        tool_names = register_mcp_servers(servers)
        if not new_server_names:
            return tool_names

        with _lock:
            connected_server_names = [
                name
                for name in new_server_names
                if name in _servers and name not in _server_connect_errors
            ]
            new_tool_count = sum(
                len(getattr(_servers[name], "_registered_tool_names", []))
                for name in connected_server_names
            )

        failed_count = len(new_server_names) - len(connected_server_names)
        if new_tool_count or failed_count:
            summary = f"  MCP: {new_tool_count} tool(s) from {len(connected_server_names)} server(s)"
            if failed_count:
                summary += f" ({failed_count} failed)"
            logger.info(summary)

        return tool_names

    finally:
        if cookie not in (None, _LOCK_UNAVAILABLE):
            cookie.release()

def is_mcp_tool_parallel_safe(tool_name: str) -> bool:
    """True when the tool's server opted into ``supports_parallel_tool_calls``.

    Uses the provenance captured at registration, never the (ambiguous)
    ``mcp__{server}__{tool}`` string shape.
    """
    if not tool_name.startswith(MCP_TOOL_NAME_PREFIX):
        return False
    with _lock:
        server_name = _mcp_tool_server_names.get(tool_name)
        return bool(server_name and server_name in _parallel_safe_servers)


def get_mcp_status() -> List[dict]:
    """Status of every configured server for banner/TUI display.

    Each dict has name, transport, tools, connected, disabled, status (one of
    connected / disabled / connecting / failed / configured) and, for failed,
    error. ``enabled: false`` is reported as disabled, not failed.
    """
    configured = _load_mcp_config()
    if not configured:
        return []

    with _lock:
        active_servers = dict(_servers)
        connecting = set(_server_connecting)
        connect_errors = dict(_server_connect_errors)

    def _entry(name: str, transport: str, status: str, **extra) -> dict:
        return {
            "name": name,
            "transport": transport,
            "tools": 0,
            "connected": False,
            "disabled": status == "disabled",
            "status": status,
            **extra,
        }

    result: List[dict] = []
    for name, cfg in configured.items():
        transport = cfg.get("transport", "http") if "url" in cfg else "stdio"
        enabled = _parse_boolish(cfg.get("enabled", True), default=True)
        server = active_servers.get(name)
        if server and server.session is not None:
            entry = _entry(name, transport, "connected", connected=True)
            entry["tools"] = (
                len(server._registered_tool_names)
                if hasattr(server, "_registered_tool_names")
                else len(server._tools)
            )
            if server._sampling:
                entry["sampling"] = dict(server._sampling.metrics)
        elif not enabled:
            entry = _entry(name, transport, "disabled")
        elif name in connecting:
            entry = _entry(name, transport, "connecting")
        elif name in connect_errors:
            entry = _entry(name, transport, "failed", error=connect_errors[name])
        else:
            entry = _entry(name, transport, "configured")
        result.append(entry)

    return result


def probe_mcp_server_tools() -> Dict[str, List[tuple]]:
    """Connect to each enabled server, list ``(tool_name, description)`` and
    disconnect, without registering anything. Failed servers are omitted."""
    if not _ensure_mcp_sdk():
        return {}

    servers_config = _load_mcp_config()
    if not servers_config:
        return {}

    enabled = {
        k: v for k, v in servers_config.items()
        if _parse_boolish(v.get("enabled", True), default=True)
    }
    if not enabled:
        return {}

    _ensure_mcp_loop()

    result: Dict[str, List[tuple]] = {}
    probed_servers: List[MCPServerTask] = []

    async def _probe_all():
        names = list(enabled.keys())
        coros = []
        for name, cfg in enabled.items():
            ct = cfg.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
            coros.append(asyncio.wait_for(_connect_server(name, cfg), timeout=ct))

        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, Exception):
                logger.debug("Probe: failed to connect to '%s': %s", name, outcome)
                continue
            probed_servers.append(outcome)
            tools = []
            for t in outcome._tools:
                desc = getattr(t, "description", "") or ""
                tools.append((t.name, desc))
            result[name] = tools

        await asyncio.gather(
            *(s.shutdown() for s in probed_servers),
            return_exceptions=True,
        )

    try:
        _run_on_mcp_loop(_probe_all, timeout=120)
    except Exception as exc:
        logger.debug("MCP probe failed: %s", exc)
    finally:
        _stop_mcp_loop_if_idle()

    return result


def has_registered_mcp_tools() -> bool:
    """True if any MCP server has registered tools (cheap; no registry walk).

    Checks registered TOOLS, not connected servers, so the per-turn refresh
    hook stays idle for zero-tool servers.
    """
    with _lock:
        return bool(_mcp_tool_server_names)


def get_registered_mcp_server_names() -> set:
    """Server names that registered at least one tool (the live, filtered
    signal — not merely what config.yaml lists)."""
    with _lock:
        return set(_mcp_tool_server_names.values())


def _stop_mcp_loop(*, only_if_idle: bool = False) -> bool:
    """Stop the background event loop and join its thread."""
    global _mcp_loop, _mcp_thread
    with _lock:
        if only_if_idle and (_servers or _server_connecting):
            logger.debug("Leaving MCP event loop running; active servers are registered or connecting")
            return False
        loop = _mcp_loop
        thread = _mcp_thread
        _mcp_loop = None
        _mcp_thread = None
    if loop is not None:
        # Drain before stopping: tasks still suspended when the loop closes
        # get resumed by the GC against a closed loop. shutdown_mcp_servers
        # only reaps servers held in _servers; everything else ends up here.
        stop_owned_by_loop = False
        if loop.is_running():
            from agent.async_utils import safe_schedule_threadsafe

            future = safe_schedule_threadsafe(
                _drain_and_stop_mcp_loop(), loop,
                logger=logger,
                log_message="MCP loop drain: failed to schedule",
                log_level=logging.WARNING,
            )
            if future is not None:
                stop_owned_by_loop = True
                try:
                    future.result(timeout=_MCP_LOOP_DRAIN_TIMEOUT + 1)
                except TimeoutError:
                    logger.warning(
                        "Timed out waiting for MCP loop drain after %.1fs",
                        _MCP_LOOP_DRAIN_TIMEOUT + 1,
                    )
                except BaseException as exc:
                    logger.warning("Error draining MCP loop tasks: %s", exc)
        elif not loop.is_closed():
            try:
                loop.run_until_complete(
                    _drain_mcp_loop_tasks(timeout=_MCP_LOOP_DRAIN_TIMEOUT)
                )
            except BaseException as exc:
                logger.warning("Error draining stopped MCP loop tasks: %s", exc)

        if not stop_owned_by_loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("MCP event loop thread did not stop within 5.0s")
        try:
            loop.close()
        except Exception as exc:
            logger.warning("Unable to close MCP event loop cleanly: %s", exc)
        # The loop is gone, so no session can be in flight: reap active too.
        _kill_orphaned_mcp_children(include_active=True)
    return True
