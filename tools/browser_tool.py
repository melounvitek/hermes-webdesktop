#!/usr/bin/env python3
"""Browser automation tools driven by the agent-browser CLI.

Backends — local headless Chromium (default; ``agent-browser install
[--with-deps]`` one-time setup), Browser Use / Browserbase / Firecrawl cloud
(auto-detected from config + credentials), a user-supplied CDP endpoint, or
Camofox — share one agent-facing behaviour: per-task sessions, text snapshots
of the accessibility tree with ``@eN`` element refs, and automatic cleanup.

Env: BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID / BROWSER_USE_API_KEY select
direct cloud credentials; BROWSERBASE_PROXIES (default "true"),
BROWSERBASE_ADVANCED_STEALTH ("false", Scale plan), BROWSERBASE_KEEP_ALIVE
("true", paid plan) and BROWSERBASE_SESSION_TIMEOUT (seconds, max 21600) tune
Browserbase sessions. Behavioural settings live under ``browser.*`` in config.yaml.

Sibling modules hold extracted clusters (eval policy, lightpanda fallback,
real-profile CDP, snapshot store); their names are re-imported here so
``patch("tools.browser_tool.X")`` keeps working.
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
    """Lazy module attributes (PEP 562): ``requests`` and ``call_llm`` load on first use.

    First access binds the real object into module globals so the test-patch
    surface (``patch("tools.browser_tool.requests.get")`` / ``.call_llm``) works.
    """
    if name == "requests":
        import requests as _requests

        globals()["requests"] = _requests
        return _requests
    if name == "call_llm":
        from agent.auxiliary_client import call_llm as _call_llm

        globals()["call_llm"] = _call_llm
        return _call_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lazy_call_llm(*args, **kwargs):
    """Invoke ``call_llm`` through module globals so test patches of
    ``tools.browser_tool.call_llm`` are honored, importing lazily otherwise."""
    fn = globals().get("call_llm")
    if fn is None:
        fn = __getattr__("call_llm")
    return fn(*args, **kwargs)

# Keys re-added to the agent-browser subprocess env AFTER credential stripping.
# agent-browser is a Node process loading npm deps: a compromised transitive
# dependency could read every Hermes secret from process.env, so only the
# browser-backend keys the worker legitimately needs pass through.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess (only browser-backend keys re-added).

    The ``hermes_subprocess_env`` import is deferred so the module imports under
    test harnesses that stub the ``tools`` package.
    """
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
# Browser-provider ABC + registry. Per-vendor providers live under
# ``plugins/browser/<vendor>/``; the legacy class names are re-exported below as
# backward-compat shims for callers that import them from this module.
from agent.browser_provider import BrowserProvider as CloudBrowserProvider  # noqa: F401  (legacy alias)
from agent.browser_registry import (  # noqa: F401  (test-patchable surface)
    get_provider as _registry_get_browser_provider,
)
try:
    from agent.browser_registry import (
        registry_generation as _browser_registry_generation,
    )
except ImportError:
    # A few isolated compatibility tests intentionally install a minimal
    # ``agent.browser_registry`` stub exposing only ``get_provider``. Those
    # harnesses have no mutable registry, so a constant generation is exact.
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
# Camofox local anti-detection browser backend (optional).
# When CAMOFOX_URL is set, all browser operations route through the
# camofox REST API instead of the agent-browser CLI.
try:
    from tools.browser_camofox import is_camofox_mode as _is_camofox_mode
except ImportError:
    _is_camofox_mode = lambda: False  # noqa: E731
# Browser Use CLI (optional)
try:
    from tools.browser_use_cli import is_browser_use_cli_mode as _is_browser_use_cli_mode
except ImportError:
    _is_browser_use_cli_mode = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)

# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes Android/Termux and macOS Homebrew locations needed for agent-browser,
# npx, node, and Android's glibc runner (grun).
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

# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Floor for ``open`` (navigate) — cold daemon + first Chromium launch can exceed
# the generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Default max chars for snapshot content before truncation. Aligned with
# web_tools.DEFAULT_EXTRACT_CHAR_LIMIT (15000) — the snapshot and
# web_extract paths share the same truncate-and-store pattern, so the model
# gets the same per-page budget from both. Configurable via
# ``browser.snapshot_threshold`` in config.yaml.
DEFAULT_SNAPSHOT_THRESHOLD = 15000
MIN_SNAPSHOT_THRESHOLD = 1000

# Backwards-compatible import surface. Runtime call sites use
# ``get_browser_snapshot_threshold()`` so config overrides take effect.
SNAPSHOT_SUMMARIZE_THRESHOLD = DEFAULT_SNAPSHOT_THRESHOLD

# Hard ceiling on the full-snapshot file written to cache/web when a snapshot
# is truncated. Mirrors web_tools.MAX_STORED_TEXT_CHARS —
# the model only ever sees the truncated view; the stored copy exists for
# read_file paging and must not write unbounded bytes to disk.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000

# Commands that legitimately return empty stdout (e.g. close, record).
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})

_cached_command_timeout: Optional[int] = None
_command_timeout_resolved = False
_cached_snapshot_threshold: Optional[int] = None
_snapshot_threshold_resolved = False


def _sanitize_url_for_logs(value: object) -> str:
    """Mask secrets in logged CDP URLs; :func:`agent.redact.redact_cdp_url` is the single policy."""
    return redact_cdp_url(value)


def _browser_cfg(key: str, default, parse, log_label: str):
    """Read ``browser.<key>`` from the raw profile config and ``parse`` it.

    Returns ``default`` when the key is absent, the section is not a mapping,
    or reading/parsing raises (logged at debug as "Could not read <log_label>").
    Raw config is used so tool JSON output is not affected by loader warnings.
    """
    try:
        from hermes_cli.config import read_raw_config
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict) and key in browser_cfg:
            return parse(browser_cfg[key])
    except Exception as e:
        logger.debug("Could not read %s: %s", log_label, e)
    return default


def _get_command_timeout() -> int:
    """Return ``browser.command_timeout`` (floored at 5s; default 30s).

    Cached after the first call and cleared by ``cleanup_all_browsers()``.
    """
    global _cached_command_timeout, _command_timeout_resolved
    if _command_timeout_resolved and _cached_command_timeout is not None:
        return _cached_command_timeout

    result = _browser_cfg(
        "command_timeout", DEFAULT_COMMAND_TIMEOUT,
        lambda v: DEFAULT_COMMAND_TIMEOUT if v is None else max(int(v), 5),
        "command_timeout from config",
    )
    # Assign the cached value BEFORE flipping the resolved flag so a
    # concurrent reader cannot observe ``resolved=True`` with a ``None`` cache.
    _cached_command_timeout = result
    _command_timeout_resolved = True
    return result


def _safe_command_timeout() -> int:
    """``_get_command_timeout`` guaranteed non-None (cache reset mid-flight).

    Uses ``is not None`` rather than ``or`` so a configured ``0`` is preserved.
    """
    val = _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


def get_browser_snapshot_threshold() -> int:
    """Return ``browser.snapshot_threshold`` (floored at MIN_SNAPSHOT_THRESHOLD).

    Cached for the browser lifecycle and reset by :func:`cleanup_all_browsers`.
    """
    global _cached_snapshot_threshold, _snapshot_threshold_resolved
    if _snapshot_threshold_resolved and _cached_snapshot_threshold is not None:
        return _cached_snapshot_threshold

    result = _browser_cfg(
        "snapshot_threshold", DEFAULT_SNAPSHOT_THRESHOLD,
        lambda v: DEFAULT_SNAPSHOT_THRESHOLD if v is None else max(int(v), MIN_SNAPSHOT_THRESHOLD),
        "browser.snapshot_threshold",
    )
    # Same race-safety invariant as the command-timeout cache.
    _cached_snapshot_threshold = result
    _snapshot_threshold_resolved = True
    return result


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    base = _safe_command_timeout()
    floor = MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT
    return max(base, floor)


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

# ============================================================================
# Cloud Provider Registry
# ============================================================================
#
# Per-vendor providers live as plugins under ``plugins/browser/<vendor>/`` and
# self-register with :mod:`agent.browser_registry`, which is what
# ``_get_cloud_provider()`` consults. The legacy class-name dict below is a
# backward-compat shim: when a test monkeypatches it, it is honoured;
# otherwise the registry-backed path wins.

_PROVIDER_REGISTRY: Dict[str, type] = {
    "browserbase": BrowserbaseProvider,
    "browser-use": BrowserUseProvider,
    "firecrawl": FirecrawlProvider,
}
# Frozen copy of the import-time _PROVIDER_REGISTRY, used by
# ``_is_legacy_provider_registry_overridden`` to detect test-time
# monkeypatching. NEVER mutate this dict.
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

# Lightpanda engine support — cached like _get_cloud_provider().
# agent-browser v0.25.3+ supports ``--engine lightpanda`` natively.
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
# than a concrete executable path. A named constant + predicate keep the six
# comparison sites (four here, plus hermes_cli/tools_config.py and
# hermes_cli/doctor.py) from drifting if the sentinel's exact spelling ever
# changes.
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"

# Pinned to match scripts/install.sh / scripts/install.ps1's
# "agent-browser@^0.26.0" managed install so a git-clone install resolving
# agent-browser via bare npx gets the same version as a managed install,
# instead of floating latest with no integrity check. Update both together.
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


# Session name for the single shared real-profile copy-browser. All consented
# local browsing attaches to this one agent-browser session so concurrent
# tasks reuse the same copy-browser instead of each launching a rival Chromium
# on the same copied user-data-dir.
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


def _url_is_private(url: str) -> bool:
    """Return True when the URL's host resolves to a private/LAN/loopback address.

    Reuses ``tools.url_safety.is_safe_url`` as the oracle — if the SSRF check
    would reject the URL, we treat it as "private" for routing purposes.  DNS
    resolution failures are treated as NOT private (fall through to whatever
    backend is configured, which will surface the DNS error naturally).
    """
    try:
        # is_safe_url returns False for private/loopback/link-local/CGNAT AND
        # for DNS failures.  We only want the private-network case here, so
        # we parse + check the host shape as a DNS-failure sieve first.
        from urllib.parse import urlparse
        import ipaddress
        import socket
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        # Literal IP → check directly
        try:
            ip = ipaddress.ip_address(hostname)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                # 172.16.0.0/12: only covered by ip.is_private on Python
                # ≥3.11 (bpo-40791).  Explicit check keeps 3.10 runtimes
                # routing these to the local sidecar correctly.
                or ip in ipaddress.ip_network("172.16.0.0/12")
                or ip in ipaddress.ip_network("100.64.0.0/10")
            )
        except ValueError:
            pass
        # Hostname — must resolve to confirm it's private (bare "localhost"
        # resolves to 127.0.0.1 via /etc/hosts).  Short-circuit on obvious
        # names to avoid a DNS hop.
        if hostname in {"localhost",} or hostname.endswith(".localhost"):
            return True
        if hostname.endswith(".local") or hostname.endswith(".lan") or hostname.endswith(".internal"):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False  # DNS fail → not private, let the normal path fail
        for _, _, _, _, sockaddr in addr_info:
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip in ipaddress.ip_network("100.64.0.0/10")
            ):
                return True
        return False
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def _navigation_session_key(task_id: str, url: str) -> str:
    """Pick the session key that should handle ``url`` for ``task_id``.

    Returns ``f"{task_id}::local"`` (hybrid routing: a local Chromium sidecar
    while the cloud session keeps serving public URLs) only when ALL hold: a
    cloud provider is configured, ``browser.auto_local_for_private_urls`` is
    on (default), the URL resolves to a private/LAN/loopback address, no CDP
    override is active (it owns the whole session), and Camofox is off (already
    local-only). Otherwise the bare task_id.
    """
    if task_id is None:
        task_id = "default"
    if _get_cdp_override_raw():
        return task_id
    if _is_camofox_mode():
        return task_id
    if _get_cloud_provider() is None:
        return task_id
    if not _auto_local_for_private_urls():
        return task_id
    if not _url_is_private(url):
        return task_id
    return f"{task_id}{_LOCAL_SUFFIX}"


def _is_local_sidecar_key(session_key: str) -> bool:
    """Return True when ``session_key`` is a hybrid-routing local sidecar."""
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    """Return the owning bare task id for an opaque browser session key."""
    if _is_local_sidecar_key(session_key):
        return session_key[: -len(_LOCAL_SUFFIX)]
    return session_key


def _session_info_owned_by_task(session_info: Dict[str, Any], task_id: str, session_key: str) -> bool:
    """Return whether ``session_info`` still belongs to ``task_id``/``session_key``.

    Sessions created by current code carry explicit ownership metadata. Treat
    older in-memory entries without those fields as valid for hot-reload/test
    compatibility, but reject any explicit mismatch before a non-navigation
    tool can act on the wrong tab/session.
    """
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    return (owner is None or owner == task_id) and (key is None or key == session_key)


def _last_session_key(task_id: str) -> str:
    """Session key a non-nav tool must use: the one that served the task's last navigation.

    If that session was cleaned up or its ownership metadata no longer matches,
    fail closed by dropping the stale binding rather than recreating or mutating
    the wrong browser.
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
    """Short temp dir for Unix domain sockets.

    macOS ``TMPDIR`` (``/var/folders/.../T/``) plus ``agent-browser-hermes_…``
    exceeds the 104-byte ``AF_UNIX`` path limit ("Failed to create socket
    directory", silent screenshot failures), so ``/tmp`` is used there.
    """
    if sys.platform == "darwin":
        return "/tmp"
    return tempfile.gettempdir()


# Active sessions keyed by "session key": the bare task_id, or f"{task_id}::local"
# for a hybrid-routing local sidecar. The key is opaque to _run_browser_command /
# cleanup_browser. Values: session_name (always), bb_session_id + cdp_url (cloud).
_active_sessions: Dict[str, Dict[str, Any]] = {}
_recording_sessions: set = set()  # session_keys with active recordings

# Most recent session_key per task_id, set by browser_navigate() and read by every
# non-nav tool so click/snapshot land in the session that served the last
# navigation (otherwise a localhost sidecar task would fall back to the cloud session).
_last_active_session_key: Dict[str, str] = {}
_LOCAL_SUFFIX = "::local"

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long.
# config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT remains a legacy
# fallback so old deployments keep working if they have not migrated yet.
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

# How often the cleanup thread re-runs the orphan reaper (a startup-only reap
# can never recover from a leak that appears after boot in a long-lived process).
BROWSER_ORPHAN_REAP_INTERVAL = 300  # seconds

# Idle ceiling for a daemon whose owner process is alive but which fell out of
# its in-memory tracking — owner-alive alone would make it immortal. A large
# multiple of the inactivity timeout so a legitimately busy session is never touched.
BROWSER_ORPHAN_GRACE_SECONDS = max(3600, BROWSER_SESSION_INACTIVITY_TIMEOUT * 20)

_session_last_activity: Dict[str, float] = {}
# Owner Hermes home per session: the janitor is one process-global thread with
# no profile scope of its own, so each teardown must re-enter the OWNING
# profile's scope (copy_context at spawn would pin the first profile's secrets
# onto every other profile's teardown).
_session_owner_homes: Dict[str, str] = {}
# Consecutive janitor cleanup failures per session; force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES.
_cleanup_failures: Dict[str, int] = {}
MAX_INACTIVITY_CLEANUP_FAILURES = 3

# Session keys flagged suspect after a command timeout. Written by
# _BrowserSessionBackend.mark_suspect (a single GIL-atomic dict write — must stay
# cheap and lock-free per the SuspectableBackend contract); consumed by
# ensure_healthy() at next use, which recycles the session.
_suspect_browser_sessions: Dict[str, str] = {}


class _BrowserSessionBackend:
    """``agent.deadline.SuspectableBackend`` adapter for one cached session key.

    A thin stateless view over ``_active_sessions[key]`` + its daemon. The
    timeout path calls ``mark_suspect`` inline; ``ensure_healthy`` runs at the
    top of ``_get_session_info`` — the single choke point every command passes
    through before reusing a cached session.
    """

    __slots__ = ("_session_key",)

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    def mark_suspect(self, reason: str) -> None:
        """Flag the cached session as possibly poisoned.

        MUST stay cheap, non-blocking and lock-free (it runs inline on the
        timed-out caller's thread); all recycle work is deferred to ``ensure_healthy``.
        """
        _suspect_browser_sessions[self._session_key] = reason

    def ensure_healthy(self) -> bool:
        """Recycle the session when a prior timeout marked it suspect.

        True when safe to reuse; False after tearing down a suspect session
        (caller creates a fresh one). The flag is popped BEFORE teardown: the
        ``close`` re-enters ``_get_session_info`` and must not recurse into
        another recycle.
        """
        reason = _suspect_browser_sessions.pop(self._session_key, None)
        if reason is None:
            return True
        logger.info(
            "Recycling suspect browser session %s before reuse (%s)",
            self._session_key, reason,
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
    """Return the SuspectableBackend adapter for ``session_key``."""
    return _BrowserSessionBackend(session_key)

# Background cleanup thread state
_cleanup_thread = None
_cleanup_running = False
# Protects _session_last_activity AND _active_sessions for thread safety
# (subagents run concurrently via ThreadPoolExecutor)
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

# atexit only — NO SIGINT/SIGTERM handlers calling sys.exit(): a SystemExit
# raised inside a prompt_toolkit key-binding callback corrupts the coroutine
# state and makes the process unkillable. atexit runs on any normal exit.
atexit.register(_emergency_cleanup_all_sessions)


# =============================================================================
# Inactivity Cleanup Functions
# =============================================================================


# Register cleanup thread stop on exit
atexit.register(_stop_browser_cleanup_thread)


# ============================================================================
# Tool Schemas
# ============================================================================

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
                    "type": "string",
                    "description": "The text to type into the field"
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
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
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
            "type": "object",
            "properties": {},
            "required": []
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


# ============================================================================
# Utility Functions
# ============================================================================


from tools.browser_tool_snapshot import (  # noqa: F401
    _store_full_snapshot,
    _truncate_snapshot,
    _redact_browser_output,
    _extract_screenshot_path_from_text,
)


# ============================================================================
# Browser Tool Functions
# ============================================================================

def _secret_url_error(url: str) -> Optional[dict]:
    """Refuse URLs that embed an API key/token (raw and URL-decoded, catching ``%2D`` tricks).

    A prompt injection could otherwise make the agent navigate to
    ``https://evil.com/steal?key=sk-ant-...`` to exfiltrate secrets.
    """
    import urllib.parse
    from agent.redact import _PREFIX_RE

    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return {"success": False, "error": "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs."}
    return None


def _url_policy_error(url: str, *, auto_local: bool = False) -> Optional[dict]:
    """Backend-aware URL checks on an already-normalized URL; None if allowed.

    Order matters and every step is a floor for the next:
      1. Credential-like query params are refused for cloud backends (third-party
         readers) — allowed for local backends and for the hybrid local sidecar.
      2. Cloud metadata / IMDS endpoints are refused UNCONDITIONALLY, for every
         backend including pure-local Chromium and off-host CDP (a local Chromium
         on a cloud VM still reaches the host IMDS).
      3. Private/internal addresses are refused unless the backend is local, the
         URL is being auto-routed to the local sidecar (``auto_local``), or
         ``browser.allow_private_urls`` opts out.
      4. Website policy (config allow/deny lists).
    """
    local = _is_local_backend()
    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not local and not auto_local:
        return {"success": False, "error": (
            "Blocked: URL contains a credential-like query parameter "
            f"({sensitive_query_key}). Cloud browser backends are third-party "
            "readers; use a local browser/CDP session or remove the sensitive "
            "query parameter before navigating.")}
    if _is_always_blocked_url(url):
        return {"success": False, "error": "Blocked: URL targets a cloud metadata endpoint"}
    if not local and not auto_local and not _allow_private_urls() and not _is_safe_url(url):
        return {"success": False, "error": "Blocked: URL targets a private or internal address"}
    blocked = check_website_access(url)
    if blocked:
        return {"success": False, "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}
    return None


def evaluate_url_safety(url: str) -> Optional[dict]:
    """Run URL safety checks; None if safe, else an error dict"""
    err = _secret_url_error(url)
    if err:
        return err
    url = _normalize_url_for_request(url)
    return _secret_url_error(url) or _url_policy_error(url)


_BOT_DETECTION_TITLE_PATTERNS = (
    "access denied", "access to this page has been denied",
    "blocked", "bot detected", "verification required",
    "please verify", "are you a robot", "captcha",
    "cloudflare", "ddos protection", "checking your browser",
    "just a moment", "attention required",
)


def _post_redirect_block(nav_session_key: str, url: str, final_url: str, auto_local_this_nav: bool) -> Optional[str]:
    """Post-redirect SSRF check; returns a blocked JSON payload or None.

    If the browser followed a redirect to a private/internal address the model
    could read internal content via later snapshots, so the page is navigated to
    about:blank first. The cloud-metadata floor fires for every backend (even the
    local sidecar); the private-address check is skipped for local backends and
    the hybrid sidecar, and when ``browser.allow_private_urls`` opts out.
    """
    if not final_url or final_url == url:
        return None
    if _is_always_blocked_url(final_url):
        _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
        return json.dumps({
            "success": False,
            "error": "Blocked: redirect landed on a cloud metadata endpoint",
        })
    if (
        not _is_local_backend()
        and not auto_local_this_nav
        and not _allow_private_urls()
        and not _is_safe_url(final_url)
    ):
        _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
        return json.dumps({
            "success": False,
            "error": "Blocked: redirect landed on a private/internal address",
        })
    return None


def _attach_auto_snapshot(response: Dict[str, Any], nav_session_key: str) -> None:
    """Add a compact snapshot to a navigate response so the model can act without browser_snapshot."""
    try:
        snap_result = _run_browser_command(nav_session_key, "snapshot", ["-c"])
        if snap_result.get("success"):
            snap_data = snap_result.get("data", {})
            snapshot_text = snap_data.get("snapshot", "")
            refs = snap_data.get("refs", {})
            threshold = get_browser_snapshot_threshold()
            if len(snapshot_text) > threshold:
                snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)
            response["snapshot"] = _redact_browser_output(snapshot_text)
            response["element_count"] = len(refs) if refs else 0
            if snap_result.get("fallback_warning") and not response.get("fallback_warning"):
                _copy_fallback_warning(response, snap_result)
    except Exception as e:
        logger.debug("Auto-snapshot after navigate failed: %s", e)


def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to ``url``; returns JSON with title, compact snapshot and, on first nav, stealth features."""
    # Hybrid routing decides BEFORE the safety checks whether this URL goes to a
    # local Chromium sidecar (cloud provider configured + private URL +
    # ``browser.auto_local_for_private_urls``); the cloud provider never sees
    # the URL in that case, so the private-address checks are relaxed for it.
    safety_error = _secret_url_error(url)
    if safety_error is None:
        url = _normalize_url_for_request(url)
        safety_error = _secret_url_error(url)
    if safety_error is not None:
        return json.dumps(safety_error)

    effective_task_id = task_id or "default"
    nav_session_key = _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    safety_error = _url_policy_error(url, auto_local=auto_local_this_nav)
    if safety_error is not None:
        return json.dumps(safety_error)

    # Camofox backend — delegate after safety checks pass
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

    # Get session info to check if this is a new session
    # (will create one with features logged if not exists)
    session_info = _get_session_info(nav_session_key)
    is_first_nav = session_info.get("_first_nav", True)

    # Auto-start recording if configured and this is first navigation
    if is_first_nav:
        session_info["_first_nav"] = False
        _maybe_start_recording(nav_session_key)

    result = _run_browser_command(
        nav_session_key,
        "open",
        [url],
        timeout=_get_open_command_timeout(first_open=is_first_nav),
    )

    if not result.get("success"):
        return json.dumps({
            "success": False,
            "error": result.get("error", "Navigation failed")
        }, ensure_ascii=False)

    data = result.get("data", {})
    title = data.get("title", "")
    final_url = data.get("url", url)

    blocked = _post_redirect_block(nav_session_key, url, final_url, auto_local_this_nav)
    if blocked is not None:
        return blocked

    response = {
        "success": True,
        "url": final_url,
        "title": title
    }
    # Auditability: stamp navigations that ran on the user's real-profile
    # copy-browser so usage is visible in the tool result.
    try:
        if (session_info.get("features") or {}).get("real_profile"):
            response["used_real_profile"] = True
    except Exception:
        pass
    # Remember only a successful, non-blocked navigation as the task owner.
    # Failed opens and blocked redirects must not retarget follow-up clicks
    # or snapshots to a newly-created but irrelevant session.
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

    # Include feature info on first navigation so model knows what's active
    if is_first_nav and "features" in session_info:
        features = session_info["features"]
        active_features = [k for k, v in features.items() if v]
        if not features.get("proxies"):
            response["stealth_warning"] = (
                "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                "Consider upgrading Browserbase plan for proxy support."
            )
        response["stealth_features"] = active_features

    _attach_auto_snapshot(response, nav_session_key)
    return json.dumps(response, ensure_ascii=False)


def browser_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """Text snapshot of the page's accessibility tree (compact unless ``full``).

    ``user_task`` is deprecated and unused: oversized snapshots always
    truncate-and-store (no LLM pass).
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot
        return camofox_snapshot(full, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Build command args based on full flag
    args = []
    if not full:
        args.extend(["-c"])  # Compact mode

    result = _run_browser_command(effective_task_id, "snapshot", args)

    if result.get("success"):
        data = result.get("data", {})
        snapshot_text = data.get("snapshot", "")
        refs = data.get("refs", {})

        # ── Private-network guard: block snapshots from eval-navigated private pages ──
        blocked = _blocked_private_page_content(effective_task_id)
        if blocked is not None:
            return blocked

        # Oversized snapshots truncate at line boundaries; the full
        # accessibility tree is stored to cache/web and the appended note
        # tells the agent how to page through it with read_file (same
        # pattern as web_extract — no LLM summarization). Threshold is
        # configurable via browser.snapshot_threshold.
        threshold = get_browser_snapshot_threshold()
        if len(snapshot_text) > threshold:
            snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)

        response = {
            "success": True,
            "snapshot": _redact_browser_output(snapshot_text),
            "element_count": len(refs) if refs else 0
        }
        _copy_fallback_warning(response, result)

        # Merge supervisor state (pending dialogs + frame tree) when a CDP
        # supervisor is attached to this task. No-op otherwise. See
        # website/docs/developer-guide/browser-supervisor.md.
        try:
            from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
            _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
            if _supervisor is not None:
                _sv_snap = _supervisor.snapshot()
                if _sv_snap.active:
                    response.update(_redact_browser_output(_sv_snap.to_dict()))
        except Exception as _sv_exc:
            logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get snapshot")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def _tool_response(result: Dict[str, Any], ok: Dict[str, Any], default_error: str) -> str:
    """Standard tool JSON for a ``_run_browser_command`` result.

    Success → ``{"success": True, **ok}``; failure → ``{"success": False,
    "error": result.error or default_error}``. Lightpanda fallback metadata
    is copied onto either shape.
    """
    if result.get("success"):
        response = {"success": True, **ok}
    else:
        response = {"success": False, "error": result.get("error", default_error)}
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click the element ``ref`` (e.g. "@e5")."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_click
        return camofox_click(ref, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "click")
    if blocked is not None:
        return blocked

    if not ref.startswith("@"):
        ref = f"@{ref}"
    result = _run_browser_command(effective_task_id, "click", [ref])
    return _tool_response(result, {"clicked": ref}, f"Failed to click {ref}")


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type ``text`` into the element ``ref`` (fill: clears, then types)."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_type
        return camofox_type(ref, text, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "type")
    if blocked is not None:
        return blocked

    if not ref.startswith("@"):
        ref = f"@{ref}"
    # fill clears then types
    result = _run_browser_command(effective_task_id, "fill", [ref, text])

    from agent.display import (
        redact_browser_typed_text_for_display,
        redact_tool_args_for_display,
    )

    # Typed text goes through the secret-pattern redactor so API keys / tokens
    # don't leak into tool progress or chat history (the raw value was already
    # sent to the browser above); normal text passes through unchanged.
    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]
    if result.get("success"):
        response = {"success": True, "typed": display_text, "element": ref}
    else:
        response = {"success": False, "error": result.get("error", f"Failed to type into {ref}")}
    response = _copy_fallback_warning(response, result)
    response = redact_browser_typed_text_for_display(response, text)
    return json.dumps(response, ensure_ascii=False)


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page ``direction`` ("up"/"down") by about half a viewport."""
    if direction not in {"up", "down"}:
        return json.dumps({
            "success": False,
            "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."
        }, ensure_ascii=False)

    # Single scroll with a pixel amount (~half a viewport) instead of 5x subprocess calls.
    _SCROLL_PIXELS = 500

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll
        # Camofox REST API doesn't support pixel args; use repeated calls
        _SCROLL_REPEATS = 5
        result = None
        for _ in range(_SCROLL_REPEATS):
            result = camofox_scroll(direction, task_id)
        return result

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)])
    return _tool_response(result, {"scrolled": direction}, f"Failed to scroll {direction}")


def browser_back(task_id: Optional[str] = None) -> str:
    """Navigate back in browser history."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_back
        return camofox_back(task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "back", [])

    if result.get("success") and _eval_ssrf_guard_active(effective_task_id):
        # History can land on a private/internal/cloud-metadata address the
        # navigate preflight never saw (earlier redirect chain, manipulated
        # client-side history). Re-check post-navigation like every other
        # content-returning entry point — the floor fires for every backend.
        _blocked_url = _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({_blocked_url}). Browser history navigation (back) "
                    "landed on this address."
                ),
            }, ensure_ascii=False)
    return _tool_response(result, {"url": result.get("data", {}).get("url", "")}, "Failed to go back")


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key (e.g. "Enter", "Tab")."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_press
        return camofox_press(key, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "press")
    if blocked is not None:
        return blocked
    result = _run_browser_command(effective_task_id, "press", [key])
    return _tool_response(result, {"pressed": key}, f"Failed to press {key}")


def _blocked_private_page_action(effective_task_id: str, action: str) -> Optional[str]:
    """Return a blocked payload when an unsafe cloud page would receive input."""
    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _current_page_private_url(effective_task_id)
    if not blocked_url:
        return None
    return json.dumps({
        "success": False,
        "error": (
            "Blocked: page URL targets a private or internal address "
            f"({blocked_url}). Refusing to {action} on this page in this "
            "browser mode."
        ),
    }, ensure_ascii=False)


def _blocked_private_page_json(blocked_url: str) -> str:
    """Blocked payload for content-returning tools whose page was eval-navigated private."""
    return json.dumps({
        "success": False,
        "error": (
            "Blocked: page URL targets a private or internal address "
            f"({blocked_url}). This may have been caused by a "
            "JavaScript navigation via browser_console."
        ),
    }, ensure_ascii=False)


def _blocked_private_page_content(effective_task_id: str) -> Optional[str]:
    """Blocked payload when the SSRF guard is active and the current page is private, else None.

    Sibling of the snapshot/vision/eval/get_images guards: after any eval that
    may have changed ``location.href`` to a private address, returning page
    content would expose it. Fail-open on probe failure (see
    ``_current_page_private_url``).
    """
    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _current_page_private_url(effective_task_id)
    return _blocked_private_page_json(blocked_url) if blocked_url else None


def browser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Console messages + uncaught JS errors (optionally ``clear``ing the buffers),
    or — when ``expression`` is given — evaluate JS in the page like the DevTools console."""
    # --- JS evaluation mode ---
    if expression is not None:
        policy_error = _enforce_browser_eval_policy(expression)
        if policy_error:
            return json.dumps({"success": False, "error": policy_error}, ensure_ascii=False)
        return _browser_eval(expression, task_id)

    # --- Console output mode (original behaviour) ---
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_console
        return camofox_console(clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    console_args = ["--clear"] if clear else []
    error_args = ["--clear"] if clear else []

    console_result = _run_browser_command(effective_task_id, "console", console_args)
    errors_result = _run_browser_command(effective_task_id, "errors", error_args)

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
                "message": _redact_browser_output(err.get("message", "")),
                "source": "exception",
            })

    response = {
        "success": True,
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": len(messages),
        "total_errors": len(errors),
    }
    _copy_fallback_warning(response, console_result)
    if errors_result.get("fallback_warning") and not response.get("fallback_warning"):
        _copy_fallback_warning(response, errors_result)
    return json.dumps(response, ensure_ascii=False)


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


def _eval_supervisor_fast_path(effective_task_id: str, expression: str) -> Optional[str]:
    """Run ``Runtime.evaluate`` on the CDP supervisor's persistent WebSocket.

    Zero subprocess startup cost vs spawning ``agent-browser eval``. Returns a
    tool JSON string when the supervisor produced a definitive answer (a value,
    a blocked private page, or a real JS-side exception — which is NOT retried
    through the subprocess, that would just reproduce it slower), or None to
    fall through to the subprocess path (no supervisor, supervisor-side failure,
    import error), so behaviour is unchanged when no supervisor is running.
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if supervisor is None:
            return None
        sup_result = supervisor.evaluate_runtime(expression)
        if sup_result.get("ok"):
            parsed = _parse_eval_value(sup_result.get("result"))
            # Post-eval page-URL recheck: if this (or a prior) eval navigated
            # the page to a private address, withhold the result.
            blocked = _blocked_private_page_content(effective_task_id)
            if blocked is not None:
                return blocked
            response = {
                "success": True,
                "result": _redact_browser_output(parsed),
                "result_type": type(parsed).__name__,
                "method": "cdp_supervisor",
            }
            return json.dumps(response, ensure_ascii=False, default=str)
        err = sup_result.get("error") or "evaluate_runtime failed"
        if "supervisor" not in err.lower():
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        logger.debug(
            "browser_eval: supervisor path unavailable (%s), falling back to subprocess",
            err,
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
        # Backend capability gap — give the model a clear signal.
        err = f"JavaScript evaluation is not supported by this browser backend. {err}"
    elif "reference chain is too long" in err.lower():
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP.
        # The supervisor fast path retries with returnByValue=false; the CLI
        # subprocess can't, so replace the cryptic protocol error with guidance.
        err = (
            "Expression returned a live DOM node / NodeList / Window, "
            "which can't be serialized. Extract a primitive value "
            "(e.g. .innerText, .href, .src, .value) or use "
            "JSON.stringify() / a snapshot tool instead."
        )
    return json.dumps(_copy_fallback_warning({"success": False, "error": err}, result))


def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context and return the result.

    Private-network guard, both sub-paths gated on the same condition: the
    literal pre-scan closes direct fetches (``fetch('http://127.0.0.1/...')``,
    which never update ``location.href``); the post-eval page-URL recheck
    closes navigate-then-read (``location.href = ...`` then read the DOM) —
    eval returns arbitrary JS results directly, never via snapshot/vision.
    """
    effective_task_id = _last_session_key(task_id or "default")

    if _eval_ssrf_guard_active(effective_task_id):
        blocked_literal = _expression_targets_private_url(expression)
        if blocked_literal:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: JavaScript expression targets a private or "
                    f"internal address ({blocked_literal}). Reading internal "
                    "endpoints via browser_console is not permitted in this "
                    "browser mode."
                ),
            }, ensure_ascii=False)

    # Camofox keeps its own raw-``task_id``-keyed session map, so pass the raw
    # id (matching every other Camofox tool) rather than the resolved
    # agent-browser session key.  The literal pre-scan above already ran.
    if _is_camofox_mode():
        return _camofox_eval(expression, task_id)

    fast = _eval_supervisor_fast_path(effective_task_id, expression)
    if fast is not None:
        return fast

    result = _run_browser_command(effective_task_id, "eval", [expression])
    if not result.get("success"):
        return _eval_failure_response(result)

    parsed = _parse_eval_value(result.get("data", {}).get("result"))
    response = {
        "success": True,
        "result": _redact_browser_output(parsed),
        "result_type": type(parsed).__name__,
    }
    # Post-eval page-URL recheck (mirrors the supervisor path).
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False, default=str)


def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post
    try:
        tab_info = _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        resp = _post(f"/tabs/{tab_id}/evaluate", body={"expression": expression, "userId": user_id})

        # Camofox returns the result in a JSON envelope
        raw_result = resp.get("result") if isinstance(resp, dict) else resp
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass

        if _eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = _camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return _blocked_private_page_json(_blocked_url)

        return json.dumps({
            "success": True,
            "result": _redact_browser_output(parsed),
            "result_type": type(parsed).__name__,
        }, ensure_ascii=False, default=str)
    except Exception as e:
        error_msg = str(e)
        # Graceful degradation — server may not support eval
        if any(code in error_msg for code in ("404", "405", "501")):
            return json.dumps({
                "success": False,
                "error": "JavaScript evaluation is not supported by this Camofox server. "
                         "Use browser_snapshot or browser_vision to inspect page state.",
            })
        return tool_error(error_msg, success=False)


def _maybe_start_recording(task_id: str):
    """Start recording if browser.record_sessions is enabled in config."""
    with _cleanup_lock:
        if task_id in _recording_sessions:
            return
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        cfg = read_raw_config()
        record_enabled = cfg_get(cfg, "browser", "record_sessions", default=False)

        if not record_enabled:
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


def browser_get_images(task_id: Optional[str] = None) -> str:
    """List the page's images (src, alt, natural size), excluding data: URIs."""
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_get_images
        return camofox_get_images(task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Use eval to run JavaScript that extracts images
    js_code = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""

    result = _run_browser_command(effective_task_id, "eval", [js_code])

    if result.get("success"):
        # ── Private-network guard (sibling of snapshot/vision/eval guards) ──
        blocked = _blocked_private_page_content(effective_task_id)
        if blocked is not None:
            return blocked

        data = result.get("data", {})
        raw_result = data.get("result", "[]")

        try:
            # Parse the JSON string returned by JavaScript
            if isinstance(raw_result, str):
                images = json.loads(raw_result)
            else:
                images = raw_result

            response = {
                "success": True,
                "images": _redact_browser_output(images),
                "count": len(images)
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
        except json.JSONDecodeError:
            response = {
                "success": True,
                "images": [],
                "count": 0,
                "warning": "Could not parse image data"
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get images")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


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

    Native-vision models get the screenshot attached to the conversation (a
    multimodal tool-result envelope); otherwise the auxiliary vision model
    returns a text analysis as JSON. Either way the file is saved persistently
    and its path returned so it can be shared via MEDIA:<path>.
    ``annotate`` overlays numbered [N] labels on interactive elements.
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_vision
        return camofox_vision(question, annotate, task_id)

    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir
    screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")

    # ── Private-network guard: block vision from eval-navigated private pages ──
    blocked = _blocked_private_page_content(effective_task_id)
    if blocked is not None:
        return blocked

    _lp_prerouted, _lp_fallback_warning, screenshot_path = _lightpanda_vision_preroute(
        effective_task_id, annotate, screenshot_path,
    )

    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        # Prune old screenshots (older than 24 hours) to prevent unbounded disk growth
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
                # If the Lightpanda pre-route already failed, force Chrome so
                # _run_browser_command doesn't trigger a redundant LP fallback.
                _engine_override="auto" if _lp_prerouted else None,
            )

        if not result.get("success"):
            error_detail = result.get("error", "Unknown error")
            error_response = {
                "success": False,
                "error": f"Failed to take screenshot ({_vision_mode_label()} mode): {error_detail}"
            }
            return json.dumps(_copy_fallback_warning(error_response, result), ensure_ascii=False)

        actual_screenshot_path = result.get("data", {}).get("path")
        if actual_screenshot_path:
            screenshot_path = Path(actual_screenshot_path)

        if not screenshot_path.exists():
            return json.dumps({
                "success": False,
                "error": (
                    f"Screenshot file was not created at {screenshot_path} ({_vision_mode_label()} mode). "
                    f"This may indicate a socket path issue (macOS /var/folders/), "
                    f"a missing Chromium install ('agent-browser install'), "
                    f"or a stale daemon process."
                ),
            }, ensure_ascii=False)

        # Fast path: native image routing for the active main model — attach the
        # screenshot directly instead of describing it through an aux vision LLM
        # (no aux call, no information loss; consistent with vision_analyze).
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
        return json.dumps(response_data, ensure_ascii=False)

    except Exception as e:
        # Keep the screenshot if it was captured — the failure is in the vision
        # analysis, not the capture, and deleting it loses evidence the user may
        # need. The 24-hour cleanup bounds disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = {"success": False, "error": f"Error during vision analysis: {str(e)}"}
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
        _copy_fallback_warning(error_info, result if 'result' in locals() else {})
        return json.dumps(error_info, ensure_ascii=False)


# ============================================================================
# Cleanup and Management Functions
# ============================================================================


# ============================================================================
# Requirements Check
# ============================================================================


# Cache for Chromium discovery. Invalidated by _reset_browser_caches.
_cached_chromium_installed: Optional[bool] = None


# One-shot per process: a 170MB download that fails (or is slow) must not be
# retried on every browser call. Reset by _reset_browser_caches() for tests.
_chromium_autoinstall_attempted = False


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Browser Tool Module")
    print("=" * 40)

    _cp = _get_cloud_provider()
    mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
    print(f"   Mode: {mode}")

    # Check requirements
    if check_browser_requirements():
        print("✅ All requirements met")
    else:
        print("❌ Missing requirements:")
        try:
            browser_cmd = _find_agent_browser()
            if _requires_real_termux_browser_install(browser_cmd):
                print("   - bare npx fallback found (insufficient on Termux local mode)")
                print(f"     Install: {_browser_install_hint()}")
            elif _cp is None and not _chromium_installed():
                print("   - Chromium browser binary not found")
                searched = ", ".join(_chromium_search_roots()) or "(no candidate paths)"
                print(f"     Searched: {searched}")
                if _running_in_docker():
                    print(
                        "     Docker: pull the latest image — the current one "
                        "predates the bundled Chromium install"
                    )
                    print("       docker pull ghcr.io/nousresearch/hermes-agent:latest")
                else:
                    print("     Install it with:")
                    print("       npx agent-browser install --with-deps")
                    print("     Or:  npx playwright install --with-deps chromium")
        except FileNotFoundError:
            print("   - agent-browser CLI not found")
            print(f"     Install: {_browser_install_hint()}")
        if _cp is not None and not _cp.is_configured():
            print(f"   - {_cp.provider_name()} credentials not configured")
            print("   Tip: set browser.cloud_provider to 'local' to use free local mode instead")

    print("\n📋 Available Browser Tools:")
    for schema in BROWSER_TOOL_SCHEMAS:
        print(f"  🔹 {schema['name']}: {schema['description'][:60]}...")

    print("\n💡 Usage:")
    print("  from tools.browser_tool import browser_navigate, browser_snapshot")
    print("  result = browser_navigate('https://example.com', task_id='my_task')")
    print("  snapshot = browser_snapshot(task_id='my_task')")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error
from tools.browser_extension_router import (
    extension_controller_available,
    routed_browser_handler,
)

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


def _browser_router_kw(kw: dict) -> dict:
    """Identity kwargs forwarded to the extension router wrapper."""
    return {
        "task_id": kw.get("task_id"),
        "session_id": kw.get("session_id"),
    }


def check_browser_routed_requirements(action: str = "browser_snapshot") -> bool:
    """Availability gate for tools that can use either browser backend."""
    return check_browser_requirements() or extension_controller_available(action)


def check_browser_navigate_requirements() -> bool:
    return check_browser_routed_requirements("browser_navigate")


def check_browser_snapshot_requirements() -> bool:
    return check_browser_routed_requirements("browser_snapshot")


def check_browser_click_requirements() -> bool:
    return check_browser_routed_requirements("browser_click")


def check_browser_type_requirements() -> bool:
    return check_browser_routed_requirements("browser_type")


def check_browser_scroll_requirements() -> bool:
    return check_browser_routed_requirements("browser_scroll")


def check_browser_back_requirements() -> bool:
    return check_browser_routed_requirements("browser_back")


def check_browser_press_requirements() -> bool:
    return check_browser_routed_requirements("browser_press")


registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_navigate",
        args,
        fallback=lambda: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_navigate_requirements,
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_snapshot",
        args,
        fallback=lambda: browser_snapshot(
            full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_snapshot_requirements,
    emoji="📸",
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_click",
        args,
        fallback=lambda: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_click_requirements,
    emoji="👆",
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_type",
        args,
        fallback=lambda: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_type_requirements,
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_scroll",
        args,
        fallback=lambda: browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_scroll_requirements,
    emoji="📜",
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_back",
        args,
        fallback=lambda: browser_back(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_back_requirements,
    emoji="◀️",
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_press",
        args,
        fallback=lambda: browser_press(key=args.get("key", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_press_requirements,
    emoji="⌨️",
)

registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_get_images",
        args,
        fallback=lambda: browser_get_images(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_vision",
        args,
        fallback=lambda: browser_vision(question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_vision_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_console",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_console"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_console",
        args,
        fallback=lambda: browser_console(clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖥️",
)
