#!/usr/bin/env python3
"""Browser automation tools driven by the agent-browser CLI.

Backends — local headless Chromium (``agent-browser install [--with-deps]``),
Browser Use / Browserbase / Firecrawl cloud (auto-detected from config +
credentials), a user-supplied CDP endpoint, or Camofox — share one agent-facing
behaviour: per-task sessions, accessibility-tree snapshots with ``@eN`` refs,
automatic cleanup. Cloud credentials come from BROWSERBASE_API_KEY /
BROWSERBASE_PROJECT_ID / BROWSER_USE_API_KEY; behavioural settings live under
``browser.*`` in config.yaml. Sibling ``browser_tool_*`` modules hold extracted
clusters; their names are re-imported here so ``patch("tools.browser_tool.X")``
keeps working.
"""

import atexit
import json
import logging
import os
import subprocess  # noqa: F401  (tests patch tools.browser_tool.subprocess.Popen)
import shutil  # noqa: F401  (tests patch tools.browser_tool.shutil.which)
import sys
import tempfile
import threading
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from agent.redact import redact_cdp_url
from hermes_constants import (  # noqa: F401  (test-patchable surface, read via origin by sibling modules)
    agent_browser_runnable,
    get_hermes_home,
    get_hermes_home_override,
    hermes_home_key,
    node_tool_runnable,
)
from utils import env_int, is_truthy_value  # noqa: F401  (read via origin by sibling modules)
from hermes_cli.config import DEFAULT_CONFIG, cfg_get
from hermes_cli._subprocess_compat import windows_hide_flags  # noqa: F401  (test-patchable; read via origin)


def __getattr__(name: str):
    """PEP 562 lazy attributes: ``requests`` / ``call_llm`` load on first use and are
    bound into module globals so ``patch("tools.browser_tool.requests.get")`` works."""
    if name == "requests":
        import requests as _requests

        globals()["requests"] = _requests
        return _requests
    if name == "call_llm":
        from agent.auxiliary_client import call_llm as _call_llm

        globals()["call_llm"] = _call_llm
        return _call_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Env keys re-added to the agent-browser subprocess AFTER credential stripping.
# agent-browser is a Node process loading npm deps: a compromised transitive
# dependency could read every Hermes secret from process.env.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess (deferred import: test
    harnesses stub the ``tools`` package)."""
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for _key in _BROWSER_PASSTHROUGH_KEYS:
        if _key in os.environ:
            env[_key] = os.environ[_key]
    return env


try:
    from tools.website_policy import check_website_access
except Exception:
    check_website_access = lambda url: None  # noqa: E731 — fail-open if policy module unavailable

try:
    from tools.url_safety import (
        is_safe_url as _is_safe_url,
        is_always_blocked_url as _is_always_blocked_url,
        normalize_url_for_request as _normalize_url_for_request,
        sensitive_query_param_name as _sensitive_query_param_name,
    )
except Exception:
    _is_safe_url = lambda url: False  # noqa: E731 — fail-closed: block all if safety module unavailable
    _is_always_blocked_url = lambda url: True  # noqa: E731 — fail-closed on the floor too
    _normalize_url_for_request = lambda url: url  # noqa: E731 — best-effort fallback
    _sensitive_query_param_name = lambda url: None  # noqa: E731 — best-effort fallback
# Browser-provider ABC + registry; per-vendor providers live under
# ``plugins/browser/<vendor>/``. Legacy class names are re-exported as shims.
from agent.browser_provider import BrowserProvider as CloudBrowserProvider  # noqa: F401  (legacy alias)
from agent.browser_registry import (  # noqa: F401  (test-patchable surface)
    get_provider as _registry_get_browser_provider,
)
try:
    from agent.browser_registry import (
        registry_generation as _browser_registry_generation,
    )
except ImportError:
    # Isolated compat tests install a minimal ``agent.browser_registry`` stub
    # with only ``get_provider``; no mutable registry → constant generation.
    def _browser_registry_generation(*, scope=None):
        return (0, 0)
from plugins.browser.browserbase.provider import (  # noqa: F401  (legacy import surface)
    BrowserbaseBrowserProvider as BrowserbaseProvider,
)
from plugins.browser.browser_use.provider import (  # noqa: F401
    BrowserUseBrowserProvider as BrowserUseProvider,
)
from plugins.browser.firecrawl.provider import (  # noqa: F401
    FirecrawlBrowserProvider as FirecrawlProvider,
)
from tools.tool_backend_helpers import normalize_browser_cloud_provider  # noqa: F401  (read via origin)
# Optional backends: Camofox (CAMOFOX_URL routes everything through its REST API)
# and the Browser Use CLI.
try:
    from tools.browser_camofox import is_camofox_mode as _is_camofox_mode
except ImportError:
    _is_camofox_mode = lambda: False  # noqa: E731
try:
    from tools.browser_use_cli import is_browser_use_cli_mode as _is_browser_use_cli_mode
except ImportError:
    _is_browser_use_cli_mode = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)

# PATH fallbacks for minimal-PATH environments (systemd services): Termux,
# macOS Homebrew, and the usual system dirs — needed for agent-browser/npx/node.
_SANE_PATH_DIRS = (
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_SANE_PATH = os.pathsep.join(_SANE_PATH_DIRS)


from tools.browser_tool_install import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _discover_homebrew_node_dirs,
    _browser_candidate_path_dirs,
    _merge_browser_path,
    _browser_install_hint,
    _is_npx_agent_browser_sentinel,
    _requires_real_termux_browser_install,
    _termux_browser_install_error,
    _agent_browser_candidate_present,
    _resolve_npx_bin,
    _agent_browser_candidates,
    _find_agent_browser,
    warm_agent_browser_npx_cache,
    _chromium_search_roots,
    _chromium_installed,
    _maybe_autoinstall_chromium,
    _running_in_docker,
    check_browser_requirements,
    check_browser_vision_requirements,
)

# Throttle screenshot cleanup to avoid repeated full directory scans.
_last_screenshot_cleanup_by_dir: dict[str, float] = {}

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_COMMAND_TIMEOUT = 30  # seconds

# Floors for ``open``: cold daemon + first Chromium launch can exceed the
# generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Snapshot truncation budget — aligned with web_tools.DEFAULT_EXTRACT_CHAR_LIMIT
# so the model gets the same per-page budget from both paths. Configurable via
# ``browser.snapshot_threshold``.
DEFAULT_SNAPSHOT_THRESHOLD = 15000
MIN_SNAPSHOT_THRESHOLD = 1000
SNAPSHOT_SUMMARIZE_THRESHOLD = DEFAULT_SNAPSHOT_THRESHOLD  # legacy import surface

# Ceiling on the stored full-snapshot file (mirrors web_tools.MAX_STORED_TEXT_CHARS):
# the stored copy exists for read_file paging and must not be unbounded.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000

# Commands that legitimately return empty stdout.
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})

_cached_command_timeout: Optional[int] = None
_command_timeout_resolved = False
_cached_snapshot_threshold: Optional[int] = None
_snapshot_threshold_resolved = False

# Mask secrets in logged CDP URLs; agent.redact.redact_cdp_url is the single policy.
_sanitize_url_for_logs = redact_cdp_url


def _browser_cfg(key: str, default, parse, log_label: str):
    """``parse(browser.<key>)`` from the RAW profile config (loader warnings must not
    leak into tool JSON); ``default`` when absent, not a mapping, or on any error."""
    try:
        from hermes_cli.config import read_raw_config
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict) and key in browser_cfg:
            return parse(browser_cfg[key])
    except Exception as e:
        logger.debug("Could not read %s: %s", log_label, e)
    return default


def _cached_browser_cfg(cache_name: str, flag_name: str, key: str, default, parse, log_label: str):
    """Process-cached ``_browser_cfg`` read (cache cleared by ``cleanup_all_browsers``).

    The value is stored BEFORE the resolved flag flips so a concurrent reader can
    never observe ``resolved=True`` with a ``None`` cache.
    """
    g = globals()
    if g[flag_name] and g[cache_name] is not None:
        return g[cache_name]
    result = _browser_cfg(key, default, parse, log_label)
    g[cache_name] = result
    g[flag_name] = True
    return result


def _get_command_timeout() -> int:
    """``browser.command_timeout`` (floored at 5s; default 30s)."""
    return _cached_browser_cfg(
        "_cached_command_timeout", "_command_timeout_resolved",
        "command_timeout", DEFAULT_COMMAND_TIMEOUT,
        lambda v: DEFAULT_COMMAND_TIMEOUT if v is None else max(int(v), 5),
        "command_timeout from config",
    )


def _safe_command_timeout() -> int:
    """``_get_command_timeout`` guaranteed non-None (cache reset mid-flight); ``is not
    None`` rather than ``or`` so a configured ``0`` is preserved."""
    val = _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


def get_browser_snapshot_threshold() -> int:
    """``browser.snapshot_threshold`` (floored at MIN_SNAPSHOT_THRESHOLD)."""
    return _cached_browser_cfg(
        "_cached_snapshot_threshold", "_snapshot_threshold_resolved",
        "snapshot_threshold", DEFAULT_SNAPSHOT_THRESHOLD,
        lambda v: DEFAULT_SNAPSHOT_THRESHOLD if v is None else max(int(v), MIN_SNAPSHOT_THRESHOLD),
        "browser.snapshot_threshold",
    )


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    return max(_safe_command_timeout(), MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT)


from tools.browser_tool_session import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _needs_chromium_sandbox_bypass,
    _apply_chromium_sandbox_args,
    _read_command_output_files,
    _unlink_command_output_files,
    _format_browser_timeout_error,
    _agent_browser_argv,
    _prepare_session_socket_dir,
    _agent_browser_command_env,
    _popen_agent_browser,
    _create_local_session,
    _create_lightpanda_session,
    _local_backend_process_dead,
    _create_cdp_session,
    _create_cloud_session_or_fallback,
    _create_session_for_key,
    _get_session_info,
    _discard_timed_out_browser_session,
    _read_browser_daemon_pid,
    _browser_daemon_responsive,
    _handle_browser_command_timeout,
    _interpret_browser_command_output,
    _run_browser_command,
)


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


from tools.browser_tool_cdp import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _resolve_cdp_override,
    _get_cdp_override_raw,
    _get_cdp_override,
    _get_dialog_policy_config,
    _ensure_cdp_supervisor,
    _stop_cdp_supervisor,
)

# ----------------------------------------------------------------------------
# Cloud provider registry — legacy class-name dict is a backward-compat shim:
# honoured when a test monkeypatches it, otherwise agent.browser_registry wins.
# ----------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, type] = {
    "browserbase": BrowserbaseProvider,
    "browser-use": BrowserUseProvider,
    "firecrawl": FirecrawlProvider,
}
# Frozen import-time copy used to detect test-time monkeypatching. NEVER mutate.
_DEFAULT_PROVIDER_REGISTRY: Dict[str, type] = dict(_PROVIDER_REGISTRY)

_cached_cloud_provider: Optional[CloudBrowserProvider] = None
_cloud_provider_resolved = False
_cached_cloud_provider_scope: Optional[str] = None
_cached_cloud_providers: Dict[
    tuple[str, tuple[int, int]], Optional[CloudBrowserProvider]
] = {}
_cloud_provider_cache_lock = threading.RLock()
_allow_private_urls_resolved = False
_cached_allow_private_urls: Optional[bool] = None
_cached_agent_browser: Optional[str] = None
_agent_browser_resolved = False
# Lightpanda engine (agent-browser v0.25.3+ ``--engine lightpanda``), cached like the provider.
_cached_browser_engine: Optional[str] = None
_browser_engine_resolved = False


from tools.browser_tool_cloud import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _is_legacy_provider_registry_overridden,
    _ensure_browser_plugins_loaded,
    _get_cloud_provider,
    _instantiate_explicit_cloud_provider,
    _autodetect_cloud_provider,
    _resolve_cloud_provider_uncached,
    _is_local_mode,
    _is_local_backend,
    _get_browser_engine,
    _is_headed_mode,
    _should_inject_engine,
    _auto_local_for_private_urls,
    _use_real_profile,
    _allow_private_urls,
    _resolve_allow_private_urls,
)

from hermes_constants import is_termux as _is_termux_environment  # noqa: F401  (read via origin)


# Sentinel _find_agent_browser returns/caches to mean "resolve via npx" rather
# than a concrete path (also compared in hermes_cli/tools_config.py and doctor.py).
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"

# Pinned to match scripts/install.sh / install.ps1's managed install so a bare-npx
# resolution gets the same version instead of floating latest. Update together.
AGENT_BROWSER_NPX_SPEC = "agent-browser@^0.26.0"


_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True
_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False


from tools.browser_tool_lightpanda_fallback import (  # noqa: F401
    _using_lightpanda_engine,
    lightpanda_engine_status,
    _lightpanda_fallback_reason,
    _needs_lightpanda_fallback,
    _annotate_lightpanda_fallback,
    _copy_fallback_warning,
    _run_chrome_fallback_command,
    _chrome_fallback_screenshot,
)


# Single shared real-profile copy-browser session: concurrent tasks reuse it
# instead of each launching a rival Chromium on the same copied user-data-dir.
_REAL_PROFILE_SESSION = "hermes-real-profile"
_real_profile_cdp_lock = threading.Lock()
_real_profile_cdp_cache: dict = {}
_real_profile_chrome_procs: list = []  # Popen handles of directly-launched real browsers


from tools.browser_tool_real_profile import (  # noqa: F401
    _terminate_real_profile_chrome,
    _cdp_http_ready,
    _agent_browser_get_cdp,
    _cdp_on_data_dir,
    _agent_browser_close_session,
    _REAL_PROFILE_CHROME_FLAGS,
    _real_profile_unsupported_reason,
    _real_profile_snapshot_error,
    _launch_real_profile_chrome,
    _attach_agent_browser_to_real_profile,
    _real_profile_cdp,
)


_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".lan", ".internal")


def _url_is_private(url: str) -> bool:
    """True when the URL's host is (or resolves to) a private/LAN/loopback/CGNAT address.

    Routing oracle only: DNS failures are NOT private (fall through to the
    configured backend, which surfaces the DNS error). Obvious names short-circuit
    the DNS hop; bare ``localhost`` resolves via /etc/hosts otherwise.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    def private(ip) -> bool:
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip in ipaddress.ip_network("100.64.0.0/10")
        )

    try:
        hostname = (urlparse(url).hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        try:
            return private(ipaddress.ip_address(hostname))
        except ValueError:
            pass
        if hostname == "localhost" or hostname.endswith(_PRIVATE_HOST_SUFFIXES):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        for *_, sockaddr in addr_info:
            try:
                if private(ipaddress.ip_address(sockaddr[0])):
                    return True
            except ValueError:
                continue
        return False
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def _navigation_session_key(task_id: str, url: str) -> str:
    """Session key that should handle ``url`` for ``task_id``.

    ``f"{task_id}::local"`` (hybrid routing: local Chromium sidecar while the cloud
    session keeps serving public URLs) only when ALL hold: cloud provider
    configured, ``browser.auto_local_for_private_urls`` on, private URL, no CDP
    override (it owns the whole session), Camofox off (already local-only).
    """
    if task_id is None:
        task_id = "default"
    hybrid = (
        not _get_cdp_override_raw()
        and not _is_camofox_mode()
        and _get_cloud_provider() is not None
        and _auto_local_for_private_urls()
        and _url_is_private(url)
    )
    return f"{task_id}{_LOCAL_SUFFIX}" if hybrid else task_id


def _is_local_sidecar_key(session_key: str) -> bool:
    """True when ``session_key`` is a hybrid-routing local sidecar."""
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    """Owning bare task id for an opaque browser session key."""
    return session_key[: -len(_LOCAL_SUFFIX)] if _is_local_sidecar_key(session_key) else session_key


def _session_info_owned_by_task(session_info: Dict[str, Any], task_id: str, session_key: str) -> bool:
    """Ownership check; entries without metadata (older in-memory / hot-reload) pass,
    any explicit mismatch fails before a non-nav tool can act on the wrong session."""
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    return (owner is None or owner == task_id) and (key is None or key == session_key)


def _last_session_key(task_id: str) -> str:
    """Session key a non-nav tool must use: the one that served the task's last navigation.

    If that session was cleaned up or its ownership no longer matches, fail closed by
    dropping the stale binding rather than recreating or mutating the wrong browser.
    """
    if task_id is None:
        task_id = "default"
    recorded_key = _last_active_session_key.get(task_id)
    if not recorded_key:
        return task_id
    with _cleanup_lock:
        session_info = _active_sessions.get(recorded_key)
        if session_info and _session_info_owned_by_task(session_info, task_id, recorded_key):
            return recorded_key
        _last_active_session_key.pop(task_id, None)
    logger.debug(
        "browser session ownership: dropping stale/mismatched last-active binding %s -> %s",
        task_id,
        recorded_key,
    )
    return task_id


def _socket_safe_tmpdir() -> str:
    """Short temp dir for Unix sockets: macOS ``TMPDIR`` + ``agent-browser-hermes_…``
    exceeds the 104-byte AF_UNIX limit (silent screenshot failures), so use /tmp there."""
    return "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()


# Active sessions keyed by "session key": the bare task_id, or f"{task_id}::local"
# for a hybrid-routing local sidecar (opaque to _run_browser_command / cleanup_browser).
# Values: session_name (always), bb_session_id + cdp_url (cloud).
_active_sessions: Dict[str, Dict[str, Any]] = {}
_recording_sessions: set = set()  # session_keys with active recordings

# Most recent session_key per task_id (set by browser_navigate, read by every non-nav
# tool) so click/snapshot land in the session that served the last navigation.
_last_active_session_key: Dict[str, str] = {}
_LOCAL_SUFFIX = "::local"

_cleanup_done = False

# Inactivity timeout: config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT
# remains a legacy env fallback for unmigrated deployments.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(
    DEFAULT_CONFIG.get("browser", {}).get("inactivity_timeout", 120)
)


def _get_session_inactivity_timeout() -> int:
    env_default = env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT)
    return _browser_cfg(
        "inactivity_timeout", env_default,
        lambda v: env_default if v is None else max(int(v), 30),  # 30s floor: no instant reaping
        "inactivity_timeout from config",
    )


BROWSER_SESSION_INACTIVITY_TIMEOUT = _get_session_inactivity_timeout()

# Orphan reaper cadence: a startup-only reap can never recover from a leak that
# appears after boot in a long-lived process.
BROWSER_ORPHAN_REAP_INTERVAL = 300  # seconds

# Idle ceiling for a daemon whose owner is alive but which fell out of in-memory
# tracking (owner-alive alone would make it immortal); large multiple so a busy
# session is never touched.
BROWSER_ORPHAN_GRACE_SECONDS = max(3600, BROWSER_SESSION_INACTIVITY_TIMEOUT * 20)

_session_last_activity: Dict[str, float] = {}
# Owner Hermes home per session: the janitor is one process-global thread, so each
# teardown must re-enter the OWNING profile's scope (copy_context at spawn would
# pin the first profile's secrets onto every other profile's teardown).
_session_owner_homes: Dict[str, str] = {}
# Consecutive janitor failures per session; force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES.
_cleanup_failures: Dict[str, int] = {}
MAX_INACTIVITY_CLEANUP_FAILURES = 3

# Session keys flagged suspect after a command timeout (written lock-free by
# mark_suspect; consumed by ensure_healthy() at next use, which recycles).
_suspect_browser_sessions: Dict[str, str] = {}


class _BrowserSessionBackend:
    """``agent.deadline.SuspectableBackend`` adapter for one cached session key.

    Stateless view over ``_active_sessions[key]``. The timeout path calls
    ``mark_suspect`` inline; ``ensure_healthy`` runs at the top of
    ``_get_session_info`` — the choke point every command passes through.
    """

    __slots__ = ("_session_key",)

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    def mark_suspect(self, reason: str) -> None:
        """MUST stay cheap, non-blocking and lock-free (runs inline on the timed-out
        caller's thread); all recycle work is deferred to ``ensure_healthy``."""
        _suspect_browser_sessions[self._session_key] = reason

    def ensure_healthy(self) -> bool:
        """Recycle the session when a prior timeout marked it suspect; False after teardown.

        The flag is popped BEFORE teardown: the ``close`` re-enters
        ``_get_session_info`` and must not recurse into another recycle.
        """
        reason = _suspect_browser_sessions.pop(self._session_key, None)
        if reason is None:
            return True
        logger.info(
            "Recycling suspect browser session %s before reuse (%s)", self._session_key, reason
        )
        try:
            _cleanup_single_browser_session(self._session_key)
        except Exception:
            logger.warning(
                "Teardown of suspect browser session %s failed; a fresh "
                "session will be created anyway", self._session_key,
                exc_info=True,
            )
        return False


def _browser_session_backend(session_key: str) -> _BrowserSessionBackend:
    return _BrowserSessionBackend(session_key)


_cleanup_thread = None
_cleanup_running = False
# Protects _session_last_activity AND _active_sessions (subagents run concurrently).
_cleanup_lock = threading.Lock()


from tools.browser_tool_lifecycle import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _session_expiry_timestamp,
    _session_has_expired,
    _emergency_cleanup_all_sessions,
    _session_owner_scope,
    _cleanup_inactive_browser_sessions,
    _write_owner_pid,
    _verify_reapable_browser_daemon,
    _socket_dir_idle_seconds,
    _owner_pid_alive,
    _reap_socket_dir,
    _reap_orphaned_browser_sessions,
    _browser_cleanup_thread_worker,
    _start_browser_cleanup_thread,
    _stop_browser_cleanup_thread,
    _update_session_activity,
    _kill_process_tree,
    _legacy_kill_process_tree,
    _pid_exists,
    _cleanup_old_screenshots,
    _cleanup_old_recordings,
    _drop_last_active_binding,
    cleanup_browser,
    _kill_verified_daemon,
    _release_session_resources,
    _force_reap_browser_session,
    _cleanup_single_browser_session,
    cleanup_all_browsers,
)

# atexit only — NO SIGINT/SIGTERM handlers calling sys.exit(): a SystemExit raised
# inside a prompt_toolkit key-binding callback corrupts the coroutine state and
# makes the process unkillable.
atexit.register(_emergency_cleanup_all_sessions)
atexit.register(_stop_browser_cleanup_thread)


# ----------------------------------------------------------------------------
# Tool Schemas
# ----------------------------------------------------------------------------

BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'https://example.com')"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_snapshot",
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 15000 chars are truncated or LLM-summarized; when that happens the complete snapshot is saved to a file and the output includes its path so you can page through the rest with read_file. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_click",
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e3')"
                },
                "text": {
                    "type": "string", "description": "The text to type into the field"
                }
            },
            "required": ["ref", "text"]
        }
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string", "enum": ["up", "down"], "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object", "properties": {}, "required": []
        }
    },
    {
        "name": "browser_press",
        "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "browser_get_images",
        "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object", "properties": {}, "required": []
        }
    },
    {
        "name": "browser_vision",
        "description": "Take a screenshot of the current page so you can inspect it visually. Use this when you need to understand what the page looks like - especially for CAPTCHAs, visual verification challenges, complex layouts, or cases where the text snapshot misses important visual information. When your active model has native vision, the screenshot is attached to your context directly and you inspect it on the next turn; otherwise Hermes falls back to an auxiliary vision model and returns a text analysis. Includes a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually. Be specific about what you're looking for."
                },
                "annotate": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, overlay numbered [N] labels on interactive elements. Each [N] maps to ref @eN for subsequent browser commands. Useful for QA and spatial reasoning about page layout."
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "browser_console",
        "description": "Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, clear the message buffers after reading"
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context. Runs in the browser like DevTools console — full access to DOM, window, document. Return values are serialized to JSON. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'"
                }
            },
            "required": []
        }
    },
]


from tools.browser_tool_snapshot import (  # noqa: F401
    _store_full_snapshot,
    _truncate_snapshot,
    _redact_browser_output,
    _extract_screenshot_path_from_text,
)


# ----------------------------------------------------------------------------
# Browser Tool Functions
# ----------------------------------------------------------------------------

def _err(error: str, **extra) -> dict:
    return {"success": False, "error": error, **extra}


def _dumps(payload: Dict[str, Any], **kw) -> str:
    return json.dumps(payload, ensure_ascii=False, **kw)


def _secret_url_error(url: str) -> Optional[dict]:
    """Refuse URLs embedding an API key/token (raw and URL-decoded, catching ``%2D``
    tricks) — a prompt injection could otherwise exfiltrate secrets via the URL."""
    import urllib.parse
    from agent.redact import _PREFIX_RE

    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _err("Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.")
    return None


def _url_policy_error(url: str, *, auto_local: bool = False) -> Optional[dict]:
    """Backend-aware URL checks on an already-normalized URL; None if allowed.

    Ordered floors: (1) credential-like query params refused for cloud backends
    (third-party readers), allowed for local and the hybrid sidecar; (2) cloud
    metadata / IMDS refused UNCONDITIONALLY for every backend (a local Chromium on
    a cloud VM still reaches the host IMDS); (3) private addresses refused unless
    local, auto-routed to the sidecar, or ``browser.allow_private_urls``;
    (4) website policy allow/deny lists.
    """
    local = _is_local_backend()
    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not local and not auto_local:
        return _err(
            "Blocked: URL contains a credential-like query parameter "
            f"({sensitive_query_key}). Cloud browser backends are third-party "
            "readers; use a local browser/CDP session or remove the sensitive "
            "query parameter before navigating.")
    if _is_always_blocked_url(url):
        return _err("Blocked: URL targets a cloud metadata endpoint")
    if not local and not auto_local and not _allow_private_urls() and not _is_safe_url(url):
        return _err("Blocked: URL targets a private or internal address")
    blocked = check_website_access(url)
    if blocked:
        return _err(blocked["message"],
                    blocked_by_policy={"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]})
    return None


def _secret_url_error_normalized(url: str) -> tuple[str, Optional[dict]]:
    """Secret check on the raw URL, then again on the normalized one; returns ``(url, error)``."""
    err = _secret_url_error(url)
    if err is None:
        url = _normalize_url_for_request(url)
        err = _secret_url_error(url)
    return url, err


def evaluate_url_safety(url: str) -> Optional[dict]:
    """Run URL safety checks; None if safe, else an error dict"""
    url, err = _secret_url_error_normalized(url)
    return err or _url_policy_error(url)


_BOT_DETECTION_TITLE_PATTERNS = (
    "access denied", "access to this page has been denied",
    "blocked", "bot detected", "verification required",
    "please verify", "are you a robot", "captcha",
    "cloudflare", "ddos protection", "checking your browser",
    "just a moment", "attention required",
)


def _post_redirect_block(nav_session_key: str, url: str, final_url: str, auto_local_this_nav: bool) -> Optional[str]:
    """Post-redirect SSRF check; blocked JSON payload or None.

    A redirect onto a private/internal address would let later snapshots read
    internal content, so the page is navigated to about:blank first. The metadata
    floor fires for every backend; the private-address check is skipped for local
    backends, the hybrid sidecar, and ``browser.allow_private_urls``.
    """
    if not final_url or final_url == url:
        return None
    if _is_always_blocked_url(final_url):
        what = "a cloud metadata endpoint"
    elif (
        not _is_local_backend()
        and not auto_local_this_nav
        and not _allow_private_urls()
        and not _is_safe_url(final_url)
    ):
        what = "a private/internal address"
    else:
        return None
    _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
    return json.dumps(_err(f"Blocked: redirect landed on {what}"))


def _snapshot_fields(snap_result: Dict[str, Any]) -> Dict[str, Any]:
    """``snapshot`` + ``element_count`` response fields from a successful snapshot result.

    Oversized snapshots truncate at line boundaries; the full tree is stored to
    cache/web with a read_file paging note (same pattern as web_extract — no LLM).
    """
    data = snap_result.get("data", {})
    snapshot_text = data.get("snapshot", "")
    refs = data.get("refs", {})
    threshold = get_browser_snapshot_threshold()
    if len(snapshot_text) > threshold:
        snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)
    return {"snapshot": _redact_browser_output(snapshot_text), "element_count": len(refs) if refs else 0}


def _merge_fallback_warning(response: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Copy a secondary result's fallback warning only if the response has none yet."""
    if result.get("fallback_warning") and not response.get("fallback_warning"):
        _copy_fallback_warning(response, result)


def _attach_auto_snapshot(response: Dict[str, Any], nav_session_key: str) -> None:
    """Add a compact snapshot to a navigate response so the model can act without browser_snapshot."""
    try:
        snap_result = _run_browser_command(nav_session_key, "snapshot", ["-c"])
        if snap_result.get("success"):
            response.update(_snapshot_fields(snap_result))
            _merge_fallback_warning(response, snap_result)
    except Exception as e:
        logger.debug("Auto-snapshot after navigate failed: %s", e)


def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to ``url``; JSON with title, compact snapshot and, on first nav, stealth features.

    Hybrid routing decides BEFORE the safety checks whether this URL goes to a local
    Chromium sidecar; the cloud provider never sees the URL then, so the
    private-address checks are relaxed for it.
    """
    url, safety_error = _secret_url_error_normalized(url)
    if safety_error is not None:
        return json.dumps(safety_error)

    effective_task_id = task_id or "default"
    nav_session_key = _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    safety_error = _url_policy_error(url, auto_local=auto_local_this_nav)
    if safety_error is not None:
        return json.dumps(safety_error)

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_navigate
        return camofox_navigate(url, task_id)

    if auto_local_this_nav:
        logger.info(
            "browser_navigate: auto-routing %s to local Chromium sidecar "
            "(cloud provider %s stays on cloud for public URLs; "
            "set browser.auto_local_for_private_urls: false to disable)",
            url,
            type(_get_cloud_provider()).__name__ if _get_cloud_provider() else "none",
        )

    session_info = _get_session_info(nav_session_key)
    is_first_nav = session_info.get("_first_nav", True)
    if is_first_nav:
        session_info["_first_nav"] = False
        _maybe_start_recording(nav_session_key)

    result = _run_browser_command(
        nav_session_key, "open", [url], timeout=_get_open_command_timeout(first_open=is_first_nav)
    )
    if not result.get("success"):
        return _dumps(_err(result.get("error", "Navigation failed")))

    data = result.get("data", {})
    title = data.get("title", "")
    final_url = data.get("url", url)

    blocked = _post_redirect_block(nav_session_key, url, final_url, auto_local_this_nav)
    if blocked is not None:
        return blocked

    response = {"success": True, "url": final_url, "title": title}
    # Auditability: stamp navigations that ran on the user's real-profile copy-browser.
    try:
        if (session_info.get("features") or {}).get("real_profile"):
            response["used_real_profile"] = True
    except Exception:
        pass
    # Only a successful, non-blocked navigation becomes the task owner: failed opens
    # and blocked redirects must not retarget follow-up clicks to an irrelevant session.
    _last_active_session_key[effective_task_id] = nav_session_key
    _copy_fallback_warning(response, result)

    title_lower = title.lower()
    if any(pattern in title_lower for pattern in _BOT_DETECTION_TITLE_PATTERNS):
        response["bot_detection_warning"] = (
            f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
            "Options: 1) Try adding delays between actions, 2) Access different pages first, "
            "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
            "4) Some sites have very aggressive bot detection that may be unavoidable."
        )

    if is_first_nav and "features" in session_info:
        features = session_info["features"]
        if not features.get("proxies"):
            response["stealth_warning"] = (
                "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                "Consider upgrading Browserbase plan for proxy support."
            )
        response["stealth_features"] = [k for k, v in features.items() if v]

    _attach_auto_snapshot(response, nav_session_key)
    return _dumps(response)


def browser_snapshot(
    full: bool = False, task_id: Optional[str] = None, user_task: Optional[str] = None
) -> str:
    """Text snapshot of the page's accessibility tree (compact unless ``full``).
    ``user_task`` is deprecated and unused (oversized snapshots always truncate-and-store)."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot
        return camofox_snapshot(full, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "snapshot", [] if full else ["-c"])
    if not result.get("success"):
        return _failed_response(result, "Failed to get snapshot")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    response = {"success": True, **_snapshot_fields(result)}
    _copy_fallback_warning(response, result)

    # Merge supervisor state (pending dialogs + frame tree) when a CDP supervisor is
    # attached. See website/docs/developer-guide/browser-supervisor.md.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if _supervisor is not None:
            _sv_snap = _supervisor.snapshot()
            if _sv_snap.active:
                response.update(_redact_browser_output(_sv_snap.to_dict()))
    except Exception as _sv_exc:
        logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

    return _dumps(response)


def _json_with_fallback(response: Dict[str, Any], result: Dict[str, Any]) -> str:
    """``json.dumps`` of ``response`` with the Lightpanda fallback metadata copied from ``result``."""
    return _dumps(_copy_fallback_warning(response, result))


def _failed_response(result: Dict[str, Any], default_error: str) -> str:
    return _json_with_fallback(_err(result.get("error", default_error)), result)


def _tool_response(result: Dict[str, Any], ok: Dict[str, Any], default_error: str) -> str:
    """Standard tool JSON: success → ``{"success": True, **ok}``, failure →
    ``{"success": False, "error": result.error or default_error}``; fallback metadata on either."""
    if not result.get("success"):
        return _failed_response(result, default_error)
    return _json_with_fallback({"success": True, **ok}, result)


def _camofox(func_name: str, *args):
    """Call ``tools.browser_camofox.<func_name>(*args)`` (Camofox mode delegation)."""
    import importlib
    return getattr(importlib.import_module("tools.browser_camofox"), func_name)(*args)


def _guarded_action_session(task_id: Optional[str], action: str) -> tuple[str, Optional[str]]:
    """``(session_key, blocked_payload)`` for an input action on the task's current page."""
    effective_task_id = _last_session_key(task_id or "default")
    return effective_task_id, _blocked_private_page_action(effective_task_id, action)


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click the element ``ref`` (e.g. "@e5")."""
    if _is_camofox_mode():
        return _camofox("camofox_click", ref, task_id)
    effective_task_id, blocked = _guarded_action_session(task_id, "click")
    if blocked is not None:
        return blocked
    if not ref.startswith("@"):
        ref = f"@{ref}"
    result = _run_browser_command(effective_task_id, "click", [ref])
    return _tool_response(result, {"clicked": ref}, f"Failed to click {ref}")


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type ``text`` into the element ``ref`` (fill: clears, then types)."""
    if _is_camofox_mode():
        return _camofox("camofox_type", ref, text, task_id)
    effective_task_id, blocked = _guarded_action_session(task_id, "type")
    if blocked is not None:
        return blocked
    if not ref.startswith("@"):
        ref = f"@{ref}"
    result = _run_browser_command(effective_task_id, "fill", [ref, text])

    from agent.display import (
        redact_browser_typed_text_for_display, redact_tool_args_for_display
    )
    # Typed text goes through the secret-pattern redactor so API keys / tokens don't
    # leak into tool progress or chat history (the raw value already went to the browser).
    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]
    if result.get("success"):
        response = {"success": True, "typed": display_text, "element": ref}
    else:
        response = _err(result.get("error", f"Failed to type into {ref}"))
    response = _copy_fallback_warning(response, result)
    return _dumps(redact_browser_typed_text_for_display(response, text))


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page ``direction`` ("up"/"down") by about half a viewport."""
    if direction not in {"up", "down"}:
        return _dumps(_err(f"Invalid direction '{direction}'. Use 'up' or 'down'."))
    _SCROLL_PIXELS = 500  # ~half a viewport in one call instead of 5x subprocess calls
    if _is_camofox_mode():
        # Camofox REST API has no pixel argument; use repeated calls.
        result = None
        for _ in range(5):
            result = _camofox("camofox_scroll", direction, task_id)
        return result
    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)])
    return _tool_response(result, {"scrolled": direction}, f"Failed to scroll {direction}")


def browser_back(task_id: Optional[str] = None) -> str:
    """Navigate back in browser history."""
    if _is_camofox_mode():
        return _camofox("camofox_back", task_id)
    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "back", [])
    if result.get("success"):
        # History can land on a private/internal/metadata address the navigate
        # preflight never saw (earlier redirect chain, manipulated client-side history).
        blocked = _blocked_private_page(
            effective_task_id, "Browser history navigation (back) landed on this address.")
        if blocked is not None:
            return blocked
    return _tool_response(result, {"url": result.get("data", {}).get("url", "")}, "Failed to go back")


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key (e.g. "Enter", "Tab")."""
    if _is_camofox_mode():
        return _camofox("camofox_press", key, task_id)
    effective_task_id, blocked = _guarded_action_session(task_id, "press")
    if blocked is not None:
        return blocked
    result = _run_browser_command(effective_task_id, "press", [key])
    return _tool_response(result, {"pressed": key}, f"Failed to press {key}")


def _blocked_private_page_json(blocked_url: str, why: str) -> str:
    """Refusal payload for a page whose URL targets a private/internal address."""
    return _dumps(_err(f"Blocked: page URL targets a private or internal address ({blocked_url}). {why}"))


def _blocked_private_page(effective_task_id: str, why: str) -> Optional[str]:
    """Blocked payload when the SSRF guard is active and the current page is private, else
    None. Fail-open on probe failure (see ``_current_page_private_url``)."""
    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _current_page_private_url(effective_task_id)
    return _blocked_private_page_json(blocked_url, why) if blocked_url else None


def _blocked_private_page_action(effective_task_id: str, action: str) -> Optional[str]:
    """Blocked payload when an unsafe cloud page would receive input."""
    return _blocked_private_page(
        effective_task_id, f"Refusing to {action} on this page in this browser mode.")


_EVAL_NAVIGATED_WHY = "This may have been caused by a JavaScript navigation via browser_console."


def _blocked_private_page_content(effective_task_id: str) -> Optional[str]:
    """Content-returning tools (snapshot/vision/eval/get_images): after an eval that may
    have moved ``location.href`` to a private address, returning content would expose it."""
    return _blocked_private_page(effective_task_id, _EVAL_NAVIGATED_WHY)


def browser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Console messages + uncaught JS errors (optionally ``clear``ing the buffers),
    or — when ``expression`` is given — evaluate JS in the page like the DevTools console."""
    if expression is not None:
        policy_error = _enforce_browser_eval_policy(expression)
        if policy_error:
            return _dumps(_err(policy_error))
        return _browser_eval(expression, task_id)

    if _is_camofox_mode():
        return _camofox("camofox_console", clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    clear_args = ["--clear"] if clear else []
    console_result = _run_browser_command(effective_task_id, "console", clear_args)
    errors_result = _run_browser_command(effective_task_id, "errors", clear_args)

    messages = []
    if console_result.get("success"):
        for msg in console_result.get("data", {}).get("messages", []):
            messages.append({
                "type": msg.get("type", "log"),
                "text": _redact_browser_output(msg.get("text", "")),
                "source": "console",
            })
    errors = []
    if errors_result.get("success"):
        for err in errors_result.get("data", {}).get("errors", []):
            errors.append({
                "message": _redact_browser_output(err.get("message", "")), "source": "exception"
            })

    response = {
        "success": True,
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": len(messages),
        "total_errors": len(errors),
    }
    _copy_fallback_warning(response, console_result)
    _merge_fallback_warning(response, errors_result)
    return _dumps(response)


from tools.browser_tool_eval_policy import (  # noqa: F401
    _eval_ssrf_guard_active,
    _JS_URL_LITERAL_RE,
    _expression_targets_private_url,
    _current_page_private_url,
    _RISKY_BROWSER_EVAL_PATTERNS,
    _JS_STRING_LITERAL_RE,
    _SENSITIVE_BROWSER_EVAL_TOKENS,
    _allow_unsafe_browser_evaluate,
    _restrict_browser_evaluate,
    _decode_js_string_literal,
    _decoded_js_string_literals,
    _sensitive_browser_eval_token_reason,
    _risky_browser_eval_reason,
    _enforce_browser_eval_policy,
    _camofox_current_page_private_url,
)


def _parse_eval_value(raw_result: Any) -> Any:
    """Eval returns the JS value as a string; parse valid JSON so the model gets structured data."""
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string
    return raw_result


def _eval_ok_response(parsed: Any, **extra) -> Dict[str, Any]:
    return {
        "success": True,
        "result": _redact_browser_output(parsed),
        "result_type": type(parsed).__name__,
        **extra,
    }


def _eval_supervisor_fast_path(effective_task_id: str, expression: str) -> Optional[str]:
    """``Runtime.evaluate`` on the CDP supervisor's persistent WebSocket (no subprocess cost).

    Returns tool JSON when the supervisor gave a definitive answer (a value, a
    blocked private page, or a real JS-side exception — NOT retried through the
    subprocess, that would just reproduce it slower), or None to fall through to the
    subprocess path (no supervisor, supervisor-side failure, import error).
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if supervisor is None:
            return None
        sup_result = supervisor.evaluate_runtime(expression)
        if sup_result.get("ok"):
            parsed = _parse_eval_value(sup_result.get("result"))
            # Post-eval page-URL recheck: withhold the result if an eval navigated to a private address.
            blocked = _blocked_private_page_content(effective_task_id)
            if blocked is not None:
                return blocked
            return _dumps(_eval_ok_response(parsed, method="cdp_supervisor"), default=str)
        err = sup_result.get("error") or "evaluate_runtime failed"
        if "supervisor" not in err.lower():
            return _dumps(_err(err))
        logger.debug(
            "browser_eval: supervisor path unavailable (%s), falling back to subprocess", err
        )
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_eval: supervisor path errored (%s), falling back", exc)
    return None


def _eval_failure_response(result: Dict[str, Any]) -> str:
    """Tool JSON for a failed ``agent-browser eval``, with actionable rewrites of known errors."""
    err = result.get("error", "eval failed")
    if any(hint in err.lower() for hint in ("unknown command", "not supported", "not found", "no such command")):
        err = f"JavaScript evaluation is not supported by this browser backend. {err}"
    elif "reference chain is too long" in err.lower():
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP. The
        # supervisor path retries with returnByValue=false; the CLI can't.
        err = (
            "Expression returned a live DOM node / NodeList / Window, "
            "which can't be serialized. Extract a primitive value "
            "(e.g. .innerText, .href, .src, .value) or use "
            "JSON.stringify() / a snapshot tool instead."
        )
    return json.dumps(_copy_fallback_warning(_err(err), result))


def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context and return the result.

    Private-network guard, both halves gated on the same condition: the literal
    pre-scan closes direct fetches (``fetch('http://127.0.0.1/...')`` never updates
    ``location.href``); the post-eval page-URL recheck closes navigate-then-read.
    """
    effective_task_id = _last_session_key(task_id or "default")

    if _eval_ssrf_guard_active(effective_task_id):
        blocked_literal = _expression_targets_private_url(expression)
        if blocked_literal:
            return _dumps(_err(
                "Blocked: JavaScript expression targets a private or "
                f"internal address ({blocked_literal}). Reading internal "
                "endpoints via browser_console is not permitted in this "
                "browser mode."
            ))

    # Camofox keeps its own raw-task_id-keyed session map, so pass the raw id.
    if _is_camofox_mode():
        return _camofox_eval(expression, task_id)

    fast = _eval_supervisor_fast_path(effective_task_id, expression)
    if fast is not None:
        return fast

    result = _run_browser_command(effective_task_id, "eval", [expression])
    if not result.get("success"):
        return _eval_failure_response(result)

    response = _eval_ok_response(_parse_eval_value(result.get("data", {}).get("result")))
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked
    return _dumps(_copy_fallback_warning(response, result), default=str)


def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post
    try:
        tab_info = _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        resp = _post(f"/tabs/{tab_id}/evaluate", body={"expression": expression, "userId": user_id})
        parsed = _parse_eval_value(resp.get("result") if isinstance(resp, dict) else resp)

        if _eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = _camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return _blocked_private_page_json(_blocked_url, _EVAL_NAVIGATED_WHY)

        return _dumps(_eval_ok_response(parsed), default=str)
    except Exception as e:
        error_msg = str(e)
        if any(code in error_msg for code in ("404", "405", "501")):  # server without eval support
            return json.dumps(_err(
                "JavaScript evaluation is not supported by this Camofox server. "
                "Use browser_snapshot or browser_vision to inspect page state."))
        return tool_error(error_msg, success=False)


def _maybe_start_recording(task_id: str):
    """Start recording if browser.record_sessions is enabled in config."""
    with _cleanup_lock:
        if task_id in _recording_sessions:
            return
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        if not cfg_get(read_raw_config(), "browser", "record_sessions", default=False):
            return

        recordings_dir = hermes_home / "browser_recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_recordings(max_age_hours=72)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recording_path = recordings_dir / f"session_{timestamp}_{task_id[:16]}.webm"

        result = _run_browser_command(task_id, "record", ["start", str(recording_path)])
        if result.get("success"):
            with _cleanup_lock:
                _recording_sessions.add(task_id)
            logger.info("Auto-recording browser session %s to %s", task_id, recording_path)
        else:
            logger.debug("Could not start auto-recording: %s", result.get("error"))
    except Exception as e:
        logger.debug("Auto-recording setup failed: %s", e)


def _maybe_stop_recording(task_id: str):
    """Stop recording if one is active for this session."""
    with _cleanup_lock:
        if task_id not in _recording_sessions:
            return
    try:
        result = _run_browser_command(task_id, "record", ["stop"])
        if result.get("success"):
            path = result.get("data", {}).get("path", "")
            logger.info("Saved browser recording for session %s: %s", task_id, path)
    except Exception as e:
        logger.debug("Could not stop recording for %s: %s", task_id, e)
    finally:
        with _cleanup_lock:
            _recording_sessions.discard(task_id)


_GET_IMAGES_JS = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""


def browser_get_images(task_id: Optional[str] = None) -> str:
    """List the page's images (src, alt, natural size), excluding data: URIs."""
    if _is_camofox_mode():
        return _camofox("camofox_get_images", task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "eval", [_GET_IMAGES_JS])
    if not result.get("success"):
        return _failed_response(result, "Failed to get images")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    raw_result = result.get("data", {}).get("result", "[]")
    try:
        images = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        return _json_with_fallback(
            {"success": True, "images": _redact_browser_output(images), "count": len(images)}, result)
    except json.JSONDecodeError:
        return _json_with_fallback(
            {"success": True, "images": [], "count": 0, "warning": "Could not parse image data"}, result)


_LP_VISION_FALLBACK_REASON = (
    "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
)


from tools.browser_tool_vision import (  # noqa: F401  (re-exported; tests patch tools.browser_tool.<name>)
    _vision_mode_label,
    _lightpanda_vision_preroute,
    _native_vision_result,
    _analyze_screenshot_with_aux_llm,
)


def browser_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """Screenshot the current page for visual inspection (CAPTCHAs, images, layouts).

    Native-vision models get the screenshot attached to the conversation (multimodal
    tool-result envelope); otherwise the auxiliary vision model returns a text
    analysis. The file is saved persistently and its path returned (MEDIA:<path>).
    ``annotate`` overlays numbered [N] labels on interactive elements.
    """
    if _is_camofox_mode():
        return _camofox("camofox_vision", question, annotate, task_id)

    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir
    screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    _lp_prerouted, _lp_fallback_warning, screenshot_path = _lightpanda_vision_preroute(
        effective_task_id, annotate, screenshot_path,
    )
    result: Dict[str, Any] = {}
    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_screenshots(screenshots_dir, max_age_hours=24)

        if _lp_prerouted and screenshot_path.exists():
            result = _annotate_lightpanda_fallback(
                {"success": True, "data": {"path": str(screenshot_path)}},
                _LP_VISION_FALLBACK_REASON,
            )
        else:
            screenshot_args = ["--annotate"] if annotate else []
            screenshot_args += ["--full", str(screenshot_path)]
            result = _run_browser_command(
                effective_task_id,
                "screenshot",
                screenshot_args,
                # A failed Lightpanda pre-route forces Chrome so _run_browser_command
                # doesn't trigger a redundant LP fallback.
                _engine_override="auto" if _lp_prerouted else None,
            )

        if not result.get("success"):
            error_detail = result.get("error", "Unknown error")
            return _json_with_fallback(
                _err(f"Failed to take screenshot ({_vision_mode_label()} mode): {error_detail}"), result)

        actual_screenshot_path = result.get("data", {}).get("path")
        if actual_screenshot_path:
            screenshot_path = Path(actual_screenshot_path)

        if not screenshot_path.exists():
            return _dumps(_err(
                f"Screenshot file was not created at {screenshot_path} ({_vision_mode_label()} mode). "
                f"This may indicate a socket path issue (macOS /var/folders/), "
                f"a missing Chromium install ('agent-browser install'), "
                f"or a stale daemon process."
            ))

        # Native image routing for the active main model: attach the screenshot
        # directly instead of describing it through an aux vision LLM (no information loss).
        from tools.vision_tools import _should_use_native_vision_fast_path

        if _should_use_native_vision_fast_path():
            return _native_vision_result(screenshot_path, question, annotate, result, _lp_fallback_warning)

        analysis = _analyze_screenshot_with_aux_llm(screenshot_path, question)
        response_data = {
            "success": True,
            "analysis": analysis or "Vision analysis returned no content.",
            "screenshot_path": str(screenshot_path),
        }
        _copy_fallback_warning(response_data, result)
        if annotate and result.get("data", {}).get("annotations"):
            response_data["annotations"] = result["data"]["annotations"]
        return _dumps(response_data)

    except Exception as e:
        # Keep a captured screenshot — the failure is in the analysis, not the
        # capture, and deleting it loses evidence. The 24-hour cleanup bounds disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = _err(f"Error during vision analysis: {str(e)}")
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
        _copy_fallback_warning(error_info, result)
        return _dumps(error_info)


# Chromium discovery cache / one-shot autoinstall flag (a failed 170MB download must
# not retry on every call). Both reset by cleanup_all_browsers().
_cached_chromium_installed: Optional[bool] = None
_chromium_autoinstall_attempted = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error
from tools.browser_extension_router import (
    extension_controller_available, routed_browser_handler
)

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


def check_browser_routed_requirements(action: str = "browser_snapshot") -> bool:
    """Availability gate for tools that can use either browser backend."""
    return check_browser_requirements() or extension_controller_available(action)


# (tool name, emoji, availability gate, fallback call) — routed-through-extension
# tools (gate None) use the per-action gate; get_images/console/vision keep the plain
# requirement checks. ``_fallback`` receives (args, kw).
_BROWSER_TOOL_TABLE = (
    ("browser_navigate", "🌐", None,
     lambda args, kw: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id"))),
    ("browser_snapshot", "📸", None,
     lambda args, kw: browser_snapshot(
         full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task"))),
    ("browser_click", "👆", None,
     lambda args, kw: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id"))),
    ("browser_type", "⌨️", None,
     lambda args, kw: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id"))),
    ("browser_scroll", "📜", None,
     lambda args, kw: browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id"))),
    ("browser_back", "◀️", None,
     lambda args, kw: browser_back(task_id=kw.get("task_id"))),
    ("browser_press", "⌨️", None,
     lambda args, kw: browser_press(key=args.get("key", ""), task_id=kw.get("task_id"))),
    ("browser_get_images", "🖼️", check_browser_requirements,
     lambda args, kw: browser_get_images(task_id=kw.get("task_id"))),
    ("browser_vision", "👁️", check_browser_vision_requirements,
     lambda args, kw: browser_vision(
         question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id"))),
    ("browser_console", "🖥️", check_browser_requirements,
     lambda args, kw: browser_console(
         clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id"))),
)


def _routed_check_fn(name: str):
    """Per-action availability gate (a named function, as the registry expects)."""
    def check() -> bool:
        return check_browser_routed_requirements(name)
    check.__name__ = check.__qualname__ = f"check_{name}_requirements"
    return check


def _routed_handler(name: str, fallback):
    def handler(args, **kw):
        return routed_browser_handler(
            name, args, fallback=lambda: fallback(args, kw),
            task_id=kw.get("task_id"), session_id=kw.get("session_id"),
        )
    return handler


# Legacy per-tool gate names (tests + external callers); also looked up by the
# registration loop below via globals().
for _name, _emoji, _check_fn, _fallback in _BROWSER_TOOL_TABLE:
    if _check_fn is None:
        _check_fn = globals()[f"check_{_name}_requirements"] = _routed_check_fn(_name)
    registry.register(
        name=_name,
        toolset="browser",
        schema=_BROWSER_SCHEMA_MAP[_name],
        handler=_routed_handler(_name, _fallback),
        check_fn=_check_fn,
        emoji=_emoji,
    )
