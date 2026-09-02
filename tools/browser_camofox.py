"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser (https://github.com/jo-inc/camofox-browser) is a self-hosted
Node.js server wrapping Camoufox (Firefox fork with C++ fingerprint spoofing).
Its REST API maps 1:1 to our browser tool interface: accessibility snapshots
with element refs, click/type/scroll by ref, screenshots.

Setup: ``npm install && npm start`` in a camofox-browser checkout, or
``docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser``; then set
``CAMOFOX_URL=http://localhost:9377`` in ``~/.hermes/.env``. For Docker Camofox,
``CAMOFOX_REWRITE_LOOPBACK_URLS=true`` opens page URLs like ``http://127.0.0.1:3000``
inside the container as ``http://host.docker.internal:3000``.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from hermes_cli.config import cfg_get, load_config, read_raw_config
from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_NO_SESSION_ERROR = "No browser session. Call browser_navigate first."
_vnc_url: Optional[str] = None  # cached from /health response
_vnc_url_checked = False  # only probe once per process

# Cached command timeout from config (resolved lazily, like browser_tool)
_cached_cmd_timeout: Optional[int] = None
_cmd_timeout_resolved = False


def _get_command_timeout() -> int:
    """Return ``browser.command_timeout`` (floored at 5s, default 30s), cached after first read.

    Mirrors :func:`tools.browser_tool._get_command_timeout` so both backends honour
    the same config knob.
    """
    global _cached_cmd_timeout, _cmd_timeout_resolved
    if _cmd_timeout_resolved:
        return _cached_cmd_timeout  # type: ignore[return-value]

    _cmd_timeout_resolved = True
    result = _DEFAULT_TIMEOUT
    try:
        val = cfg_get(read_raw_config(), "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)
    except Exception as exc:
        logger.debug("Could not read browser.command_timeout: %s", exc)
    _cached_cmd_timeout = result
    return result


def _auth_headers() -> Dict[str, str]:
    """Return Authorization header when CAMOFOX_API_KEY is set."""
    key = (get_secret("CAMOFOX_API_KEY", "") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return (get_secret("CAMOFOX_URL", "") or "").rstrip("/")


def _config_cdp_url() -> str:
    """Persistent ``browser.cdp_url`` from config.yaml, or empty string.

    Read here rather than via ``browser_tool._get_cdp_override`` (circular import)
    so Camofox yields to a config CDP override like it yields to ``BROWSER_CDP_URL``.
    """
    try:
        from hermes_cli.config import read_raw_config

        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def is_camofox_mode() -> bool:
    """True when the Camofox backend is selected and no CDP override is active.

    Selection is ``browser.cloud_provider: camofox``; ``CAMOFOX_URL`` is only the
    server address and does not override a different stored selection. Legacy: when
    no selection was ever written, a set ``CAMOFOX_URL`` still activates Camofox.
    A CDP override (``BROWSER_CDP_URL`` env or ``browser.cdp_url`` config, matching
    ``browser_tool._get_cdp_override()`` precedence) wins so tools drive the real
    CDP browser instead of being silently routed to Camofox.
    """
    if os.getenv("BROWSER_CDP_URL", "").strip() or _config_cdp_url():
        return False
    try:
        from tools.tool_backend_helpers import read_selection

        selected = read_selection("browser")
    except Exception:  # pragma: no cover — helpers are in-repo
        selected = None
    if selected is not None:
        return selected == "camofox"
    return bool(get_camofox_url())


def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable (and cache its VNC URL once)."""
    global _vnc_url, _vnc_url_checked
    url = get_camofox_url()
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200 and not _vnc_url_checked:
            try:
                vnc_port = resp.json().get("vncPort")
                if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                    host = urlsplit(url).hostname or "localhost"
                    _vnc_url = f"http://{host}:{vnc_port}"
            except (ValueError, KeyError):
                pass
            _vnc_url_checked = True
        return resp.status_code == 200
    except Exception:
        return False


def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    if not _vnc_url_checked:
        check_camofox_available()
    return _vnc_url


def _get_camofox_config() -> Dict[str, Any]:
    """Return the ``browser.camofox`` config block, or an empty dict."""
    try:
        camofox_cfg = load_config().get("browser", {}).get("camofox", {})
    except Exception as exc:
        logger.warning("camofox config check failed, defaulting to disabled: %s", exc)
        return {}
    return camofox_cfg if isinstance(camofox_cfg, dict) else {}


def _managed_persistence_enabled(camofox_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """``browser.camofox.managed_persistence``: stable profile-scoped userId vs random per session."""
    if camofox_cfg is None:
        camofox_cfg = _get_camofox_config()
    return bool(camofox_cfg.get("managed_persistence"))


def _secret_or_cfg(secret_name: str, camofox_cfg: Dict[str, Any], cfg_key: str) -> str:
    return (
        (get_secret(secret_name, "") or "").strip()
        or str(camofox_cfg.get(cfg_key) or "").strip()
    )


def _camofox_identity_override(task_id: Optional[str], camofox_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return an externally configured Camofox identity, if one is set.

    Integrations that own the visible Camofox browser share a user ID so Hermes
    operates in the same profile instead of a separate private session.
    """
    user_id = _secret_or_cfg("CAMOFOX_USER_ID", camofox_cfg, "user_id")
    if not user_id:
        return None
    session_key = (
        _secret_or_cfg("CAMOFOX_SESSION_KEY", camofox_cfg, "session_key")
        or f"task_{(task_id or 'default')[:16]}"
    )
    return {"user_id": user_id, "session_key": session_key}


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean env %s=%r", name, raw)
    return None


def _flag(env_name: str, camofox_cfg: Dict[str, Any], cfg_key: str) -> bool:
    """Boolean toggle: env var wins when set to a valid value, else config key."""
    env_value = _env_flag(env_name)
    return env_value if env_value is not None else bool(camofox_cfg.get(cfg_key))


def _adopt_existing_tab_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether Hermes should recover an existing Camofox tab ID."""
    return _flag("CAMOFOX_ADOPT_EXISTING_TAB", camofox_cfg, "adopt_existing_tab")


def _loopback_rewrite_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether loopback page URLs should be rewritten for Docker-hosted Camofox.

    ``CAMOFOX_URL`` may point at a host-published Docker port, but page URLs are
    opened by the browser *inside* the container, where loopback is the container,
    not the host. Opt-in because non-Docker installs run the browser on the host.
    """
    return _flag("CAMOFOX_REWRITE_LOOPBACK_URLS", camofox_cfg, "rewrite_loopback_urls")


def _loopback_rewrite_host(camofox_cfg: Dict[str, Any]) -> str:
    """Return the host alias used when rewriting loopback page URLs."""
    return (
        os.getenv("CAMOFOX_LOOPBACK_HOST_ALIAS", "").strip()
        or str(camofox_cfg.get("loopback_host_alias") or "").strip()
        or "host.docker.internal"
    )


def _is_loopback_hostname(hostname: Optional[str]) -> bool:
    """Return True for localhost/127.0.0.0/8/::1-style hostnames."""
    if not hostname:
        return False
    host = hostname.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _rewrite_loopback_url_for_camofox(url: str) -> tuple[str, Optional[Dict[str, str]]]:
    """Rewrite loopback page URLs for Docker-hosted Camofox, if configured.

    Returns ``(rewritten_url, metadata)``; ``metadata`` is present only when a
    rewrite happened so the tool result can disclose the change to the model.
    """
    camofox_cfg = _get_camofox_config()
    if not _loopback_rewrite_enabled(camofox_cfg):
        return url, None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None

    if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(parsed.hostname):
        return url, None

    alias = _loopback_rewrite_host(camofox_cfg)
    if not alias:
        return url, None

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    host_part = f"[{alias}]" if ":" in alias and not alias.startswith("[") else alias
    port_part = f":{parsed.port}" if parsed.port else ""
    rewritten = urlunsplit(
        SplitResult(parsed.scheme, f"{userinfo}{host_part}{port_part}", parsed.path, parsed.query, parsed.fragment)
    )
    return rewritten, {
        "from": parsed.hostname or "",
        "to": alias,
        "original_url": url,
        "rewritten_url": rewritten,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# Maps task_id -> {"user_id": str, "tab_id": str|None, ...}
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _adopt_existing_tab(session: Dict[str, Any]) -> Dict[str, Any]:
    """Attach process-local state to an already-open managed Camofox tab.

    Gateway restarts empty this module's in-memory cache while Camofox still has
    the integration-owned tab, so rehydrate tab_id before creating a new one.
    """
    if session.get("tab_id") or not session.get("adopt_existing_tab") or not get_camofox_url():
        return session

    try:
        tabs = _get("/tabs", params={"userId": session["user_id"]}, timeout=5).get("tabs", [])
    except Exception as exc:
        logger.debug("Camofox tab adoption failed for %s: %s", session.get("user_id"), exc)
        return session

    if not isinstance(tabs, list) or not tabs:
        return session

    session_key = session.get("session_key")
    dict_tabs = [tab for tab in tabs if isinstance(tab, dict)]
    candidates = [tab for tab in dict_tabs if tab.get("listItemId") == session_key] or dict_tabs
    tab_id = candidates[-1].get("tabId") if candidates else None
    if isinstance(tab_id, str) and tab_id:
        session["tab_id"] = tab_id
        logger.debug("Adopted existing Camofox tab %s for %s", tab_id, session.get("user_id"))

    return session


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    """Get or create a camofox session for the given task.

    Identity precedence: external override (CAMOFOX_USER_ID / config), then the
    deterministic profile-scoped identity when managed persistence is on, else a
    random ephemeral userId.
    """
    task_id = task_id or "default"
    with _sessions_lock:
        if task_id in _sessions:
            return _adopt_existing_tab(_sessions[task_id])

        camofox_cfg = _get_camofox_config()
        identity = _camofox_identity_override(task_id, camofox_cfg)
        if identity is None and _managed_persistence_enabled(camofox_cfg):
            identity = get_camofox_identity(task_id)
        if identity is not None:
            session = {
                "user_id": identity["user_id"],
                "tab_id": None,
                "session_key": identity["session_key"],
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
            }
        else:
            session = {
                "user_id": f"hermes_{uuid.uuid4().hex[:10]}",
                "tab_id": None,
                "session_key": f"task_{task_id[:16]}",
                "managed": False,
                "adopt_existing_tab": False,
            }
        _sessions[task_id] = session
        return _adopt_existing_tab(session)


def _ensure_tab(task_id: Optional[str], url: str = "about:blank") -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = _get_session(task_id)
    if session["tab_id"]:
        return session
    data = _request(
        "post",
        "/tabs",
        json={
            "userId": session["user_id"],
            "listItemId": session["session_key"],
            "url": url,
        },
    ).json()
    session["tab_id"] = data.get("tabId")
    return session


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info."""
    with _sessions_lock:
        return _sessions.pop(task_id or "default", None)


def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Release the in-memory session without destroying the server-side context.

    Managed (persistent or externally-owned) profiles must survive across agent
    tasks, so only the local tracking entry is dropped (returns ``True``). For
    ephemeral sessions returns ``False`` so the caller falls back to :func:`camofox_close`.
    """
    camofox_cfg = _get_camofox_config()
    if _managed_persistence_enabled(camofox_cfg) or _camofox_identity_override(task_id, camofox_cfg):
        _drop_session(task_id)
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
    """Issue an authenticated request to camofox and return the raised-for-status response."""
    if timeout is None:
        timeout = _get_command_timeout()
    resp = getattr(requests, method)(
        f"{get_camofox_url()}{path}", timeout=timeout, headers=_auth_headers(), **kwargs
    )
    resp.raise_for_status()
    return resp


def _post(path: str, body: dict, timeout: Optional[int] = None) -> dict:
    """POST JSON to camofox and return parsed response."""
    return _request("post", path, timeout, json=body).json()


def _get(path: str, params: dict = None, timeout: Optional[int] = None) -> dict:
    """GET from camofox and return parsed response."""
    return _request("get", path, timeout, params=params).json()


def _get_raw(path: str, params: dict = None, timeout: Optional[int] = None) -> requests.Response:
    """GET from camofox and return raw response (for binary data)."""
    return _request("get", path, timeout, params=params)


def _delete(path: str, body: dict = None, timeout: Optional[int] = None) -> dict:
    """DELETE to camofox and return parsed response."""
    return _request("delete", path, timeout, json=body).json()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tab_path(session: Dict[str, Any], suffix: str) -> str:
    return f"/tabs/{session['tab_id']}/{suffix}"


def _user_params(session: Dict[str, Any]) -> Dict[str, str]:
    return {"userId": session["user_id"]}


def _fetch_snapshot(session: Dict[str, Any]) -> tuple[str, int]:
    """Return ``(snapshot_text, refs_count)`` truncated like the main browser tool.

    Cuts at line boundaries, stores the full tree to cache/web, and appends a
    read_file pointer. ``browser_tool`` imports this module, so import lazily.
    """
    data = _get(_tab_path(session, "snapshot"), params=_user_params(session))
    snapshot = data.get("snapshot", "")
    from tools.browser_tool import (
        get_browser_snapshot_threshold,
        _truncate_snapshot,
    )

    threshold = get_browser_snapshot_threshold()
    if len(snapshot) > threshold:
        snapshot = _truncate_snapshot(snapshot, max_chars=threshold)
    return snapshot, data.get("refsCount", 0)


def camofox_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    try:
        browser_url, rewrite_info = _rewrite_loopback_url_for_camofox(url)
        session = _get_session(task_id)
        if not session["tab_id"]:
            session = _ensure_tab(task_id, browser_url)
            data = {"ok": True, "url": browser_url}
        else:
            try:
                data = _post(
                    _tab_path(session, "navigate"),
                    {"userId": session["user_id"], "url": browser_url},
                    timeout=60,
                )
            except requests.HTTPError as e:
                # Stale tab (garbage collected server-side) — recreate it.
                if e.response is not None and e.response.status_code == 404:
                    logger.warning(
                        "Camofox tab %s returned 404 — tab was garbage collected. "
                        "Creating a fresh tab.",
                        session["tab_id"],
                    )
                    session["tab_id"] = None
                    session = _ensure_tab(task_id, browser_url)
                    data = {"ok": True, "url": browser_url}
                else:
                    raise
        result = {
            "success": True,
            "url": data.get("url", browser_url),
            "title": data.get("title", ""),
        }
        if rewrite_info:
            result["requested_url"] = url
            result["url_rewrite"] = rewrite_info
            result["warning"] = (
                "Rewrote loopback URL for Docker-hosted Camofox: "
                f"{rewrite_info['from']} -> {rewrite_info['to']}"
            )
        vnc = get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = (
                "Browser is visible via VNC. "
                "Share this link with the user so they can watch the browser live."
            )

        # Auto-take a compact snapshot so the model can act immediately.
        try:
            result["snapshot"], result["element_count"] = _fetch_snapshot(session)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except requests.HTTPError as e:
        return tool_error(f"Navigation failed: {e}", success=False)
    except requests.ConnectionError:
        return json.dumps({
            "success": False,
            "error": f"Cannot connect to Camofox at {get_camofox_url()}. "
                     "Is the server running? Start with: npm start (in camofox-browser dir) "
                     "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser",
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def _camofox_private_page_block(session: Dict[str, Any], task_id: Optional[str], action: str) -> Optional[str]:
    """Return a blocked payload when the current Camofox page is private/internal.

    Mirrors the ``_camofox_eval`` guard in browser_tool.py: snapshot / vision /
    image-extraction read current page state, so on a non-local backend they can
    leak an intranet/metadata page the terminal itself can't reach. Only active
    when the SSRF guard applies (non-local backend, not a local sidecar,
    ``allow_private_urls`` unset); fail-open on probe failure like sibling guards.
    ``browser_tool`` imports this module, so import lazily.
    """
    from tools.browser_tool import (
        _camofox_current_page_private_url,
        _eval_ssrf_guard_active,
    )

    if not _eval_ssrf_guard_active(task_id or "default"):
        return None
    blocked_url = _camofox_current_page_private_url(session["tab_id"], session["user_id"])
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


def _require_tab(task_id: Optional[str], action: Optional[str] = None) -> tuple[Dict[str, Any], Optional[str]]:
    """Return ``(session, error_payload)``: error when no tab exists or, if ``action`` given, the page is private."""
    session = _get_session(task_id)
    if not session["tab_id"]:
        return session, tool_error(_NO_SESSION_ERROR, success=False)
    if action is not None:
        return session, _camofox_private_page_block(session, task_id, action)
    return session, None


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None,
                     user_task: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from Camofox.

    ``user_task`` is deprecated and ignored — oversized snapshots always
    truncate-and-store (no LLM summarization), same as the main browser tool.
    """
    try:
        session, blocked = _require_tab(task_id, "read a page snapshot")
        if blocked:
            return blocked
        snapshot, refs_count = _fetch_snapshot(session)
        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def _tab_action(task_id: Optional[str], guard_action: Optional[str], suffix: str,
                body: Dict[str, Any], result: Callable[[dict], dict]) -> str:
    """Shared shape of the simple tab actions: require a tab (+ private-page guard
    when ``guard_action`` is set), POST ``body`` to ``/tabs/<id>/<suffix>``, build result."""
    try:
        session, blocked = _require_tab(task_id, guard_action)
        if blocked:
            return blocked
        data = _post(_tab_path(session, suffix), {"userId": session["user_id"], **body})
        return json.dumps(result(data))
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    try:
        session, blocked = _require_tab(task_id, "click")
        if blocked:
            return blocked
        clean_ref = ref.lstrip("@")  # our tool convention prefixes refs with @
        data = _post(_tab_path(session, "click"), {"userId": session["user_id"], "ref": clean_ref})
        return json.dumps({"success": True, "clicked": clean_ref, "url": data.get("url", "")})
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session, blocked = _require_tab(task_id, "type")
        if blocked:
            return blocked
        clean_ref = ref.lstrip("@")
        _post(
            _tab_path(session, "type"),
            {"userId": session["user_id"], "ref": clean_ref, "text": text},
        )
        from agent.display import (
            redact_browser_typed_text_for_display,
            redact_tool_args_for_display,
        )

        # Match browser_tool.browser_type: the raw text is typed into the page, but
        # the returned display value is run through the secret-pattern redactor so
        # API keys / tokens don't leak into tool progress or chat history.
        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]
        response = {
            "success": True,
            "typed": display_text,
            "element": clean_ref,
        }
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response)
    except Exception as e:
        from agent.display import redact_browser_typed_text_for_display

        return tool_error(redact_browser_typed_text_for_display(str(e), text), success=False)


def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    return _tab_action(
        task_id, None, "scroll", {"direction": direction},
        lambda data: {"success": True, "scrolled": direction},
    )


def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    return _tab_action(
        task_id, None, "back", {},
        lambda data: {"success": True, "url": data.get("url", "")},
    )


def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    return _tab_action(
        task_id, "press", "press", {"key": key},
        lambda data: {"success": True, "pressed": key},
    )


def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = _drop_session(task_id)
        if session:
            _delete(f"/sessions/{session['user_id']}")
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        return json.dumps({"success": True, "closed": True, "warning": str(e)})


def camofox_get_images(task_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox.

    Parsed from the accessibility tree snapshot (``img "alt" [eN]`` entries with
    the URL on the following ``/url:`` line) — Camofox has no /images endpoint.
    """
    try:
        session, blocked = _require_tab(task_id, "extract page images")
        if blocked:
            return blocked

        data = _get(_tab_path(session, "snapshot"), params=_user_params(session))
        snapshot = data.get("snapshot", "")

        images = []
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("- img ", "img ")):
                alt_match = re.search(r'img\s+"([^"]*)"', stripped)
                alt = alt_match.group(1) if alt_match else ""
                src = ""
                if i + 1 < len(lines):
                    url_match = re.search(r'/url:\s*(\S+)', lines[i + 1].strip())
                    if url_match:
                        src = url_match.group(1)
                if alt or src:
                    images.append({"src": src, "alt": alt})

        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images),
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_vision(question: str, annotate: bool = False,
                   task_id: Optional[str] = None) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    try:
        session, blocked = _require_tab(task_id, "capture a screenshot")
        if blocked:
            return blocked

        resp = _get_raw(_tab_path(session, "screenshot"), params=_user_params(session))

        from hermes_constants import get_hermes_home
        screenshots_dir = get_hermes_home() / "browser_screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png")

        with open(screenshot_path, "wb") as f:
            f.write(resp.content)

        img_b64 = base64.b64encode(resp.content).decode("utf-8")

        annotation_context = ""
        if annotate:
            try:
                snap_data = _get(_tab_path(session, "snapshot"), params=_user_params(session))
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snap_data.get('snapshot', '')[:3000]}"
            except Exception:
                pass

        # The screenshot itself cannot be redacted, but the text-based accessibility
        # snippet sent alongside it must not leak secret values.
        from agent.redact import redact_sensitive_text
        annotation_context = redact_sensitive_text(annotation_context)

        from agent.auxiliary_client import call_llm

        vision_prompt = (
            f"Analyze this browser screenshot and answer: {question}"
            f"{annotation_context}"
        )

        try:
            _vision_cfg = cfg_get(load_config(), "auxiliary", "vision", default={})
            _vision_timeout = float(_vision_cfg.get("timeout", 120))
            _vision_temperature = float(_vision_cfg.get("temperature", 0.1))
        except Exception:
            _vision_timeout = 120.0
            _vision_temperature = 0.1

        response = call_llm(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }],
            task="vision",
            temperature=_vision_temperature,
            timeout=_vision_timeout,
        )
        analysis = (response.choices[0].message.content or "").strip() if response.choices else ""

        # Redact secrets the vision LLM may have read from the screenshot.
        analysis = redact_sensitive_text(analysis)

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "screenshot_path": screenshot_path,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


def camofox_console(clear: bool = False, task_id: Optional[str] = None) -> str:
    """Console output is not exposed by the Camofox REST API; return an empty result with a note."""
    return json.dumps({
        "success": True,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
                "Use browser_snapshot or browser_vision to inspect page state.",
    })
