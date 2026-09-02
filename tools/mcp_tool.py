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
import inspect
import logging
import os  # noqa: F401  — tests patch ``tools.mcp_tool.os.*``
import shutil  # noqa: F401  — tests patch ``tools.mcp_tool.shutil.which``
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Split modules. Every name is re-exported here so ``from tools.mcp_tool import X``
# and ``mock.patch("tools.mcp_tool.X")`` keep working; the siblings read origin
# state back through ``tools.mcp_tool`` at call time (never by value).
from tools.mcp_tool_common import (  # noqa: F401
    _DEFAULT_TOOL_TIMEOUT,
    _env_ref_name,
    _exc_str,
    _get_lifecycle_seconds,
    _jittered,
    _parse_boolish,
    _resolve_tool_timeout,
    _safe_numeric,
    _sanitize_error,
    mcp_field,
)
from tools.mcp_tool_schema import (  # noqa: F401
    MCP_TOOL_NAME_PREFIX,
    _build_utility_schemas,
    _convert_mcp_schema,
    _normalize_mcp_input_schema,
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
    _truncate_mcp_text_result,
)
from tools.mcp_tool_errors import (  # noqa: F401
    InvalidMcpUrlError,
    NonMcpEndpointError,
    _EXC_TRAVERSAL_MAX_NODES,
    _JSONRPC_UNSUPPORTED_PROTOCOL_VERSION,
    _classify_mcp_failure,
    _format_connect_error,
    _handshake_rejected_as_modern,
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
    _build_safe_env,
    _filter_suspicious_mcp_servers,
    _get_mcp_stderr_log,
    _interpolate_env_vars,
    _load_mcp_config,
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
    _handle_auth_error_and_retry,
    _handle_session_expired_and_retry,
    _make_check_fn,
    _make_get_prompt_handler,
    _make_list_prompts_handler,
    _make_list_resources_handler,
    _make_read_resource_handler,
    _make_tool_handler,
)
from tools.mcp_tool_registration import (  # noqa: F401
    _annotation_read_only_hint,
    _existing_tool_names,
    _forget_mcp_tool_server,
    _normalize_server_trust,
    _register_from_cache_sync,
    _register_server_tools,
    _select_utility_schemas,
    _track_mcp_tool_server,
)
from tools.mcp_tool_lifecycle import (  # noqa: F401
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
    _reinject_post_build_tools,
    persist_agent_tool_names,
    refresh_agent_mcp_tools,
    reprobe_tool_availability,
    restore_agent_tool_prefix,
)
from tools.mcp_tool_transport import MCPServerTransportMixin
from tools.mcp_tool_server_run import MCPServerRunMixin
from tools.mcp_tool_health import MCPServerHealthMixin
from tools.mcp_tool_loop import (  # noqa: F401 -- re-exported for callers and test patches
    _LockCookie,
    _acquire_lock_on_fh,
    _try_acquire_mcp_discovery_lock,
    _mcp_loop_exception_handler,
    _wrap_with_home_override,
    _wrap_with_dashboard_oauth_flow,
    _run_on_mcp_loop,
)
from tools.mcp_tool_discovery import (  # noqa: F401 -- re-exported for callers and test patches
    _record_connect_failure,
    _clear_connect_failure,
    _connect_cooldown_active,
    _connect_server,
    _request_lazy_reconnect,
    _resolve_server_lazy,
    _ensure_lazy_server_connected,
    _get_connected_server_for_call,
    _discover_and_register_server,
    register_mcp_servers,
    discover_mcp_tools,
    is_mcp_tool_parallel_safe,
    get_mcp_status,
    probe_mcp_server_tools,
    has_registered_mcp_tools,
    get_registered_mcp_server_names,
)


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

class MCPServerTask(MCPServerRunMixin, MCPServerTransportMixin, MCPServerHealthMixin):
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


# ---------------------------------------------------------------------------
# Connecting, lazy start, discovery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
