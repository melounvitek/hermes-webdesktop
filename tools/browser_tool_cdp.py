"""User-supplied CDP endpoint resolution (browser.cdp_url / real-profile), dialog-policy config and the per-task CDP supervisor lifecycle.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import os
from typing import Tuple

from tools.browser_tool_origin import origin_module as _origin


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete websocket URL.

    Full ``ws://.../devtools/browser/...`` endpoints pass through; HTTP
    discovery roots and bare ``ws://host:port`` are resolved via
    ``/json/version`` → ``webSocketDebuggerUrl`` (falls back to the raw value
    with a warning if discovery fails).
    """
    _bt = _origin()
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
        _bt.logger.warning(
            "Failed to resolve CDP endpoint %s via %s: %s",
            _bt._sanitize_url_for_logs(raw),
            _bt._sanitize_url_for_logs(version_url),
            _bt._sanitize_url_for_logs(exc),
        )
        return raw

    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        _bt.logger.info(
            "Resolved CDP endpoint %s -> %s",
            _bt._sanitize_url_for_logs(raw),
            _bt._sanitize_url_for_logs(ws_url),
        )
        return ws_url

    _bt.logger.warning(
        "CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
        _bt._sanitize_url_for_logs(version_url),
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
    _bt = _origin()
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override
    return _bt._browser_cfg(
        "cdp_url", "", lambda v: str(v or "").strip(), "browser.cdp_url from config"
    )


def _get_cdp_override() -> str:
    """Return the resolved CDP URL override, or "" (skips cloud AND local launch).

    May perform an HTTP ``/json/version`` discovery request — only call on
    paths about to *connect* (session creation, supervisor attach); pure
    is-it-configured gates must use :func:`_get_cdp_override_raw`.
    """
    _bt = _origin()
    raw = _bt._get_cdp_override_raw()
    if not raw:
        return ""
    return _bt._resolve_cdp_override(raw)


def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s`` from config.

    Returns a ``(policy, timeout_s)`` tuple, falling back to the supervisor's
    defaults when keys are absent or invalid.
    """
    # Defer imports so browser_tool can be imported in minimal environments.
    _bt = _origin()
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
            _bt.logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
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
    _bt = _origin()
    cdp_url = _bt._get_cdp_override()
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        with _bt._cleanup_lock:
            session_info = _bt._active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _bt._resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        policy, timeout_s = _bt._get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
        )
    except Exception as exc:
        _bt.logger.debug(
            "CDP supervisor attach for task=%s failed (non-fatal): %s",
            task_id,
            exc,
        )


def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    _bt = _origin()
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        _bt.logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)
