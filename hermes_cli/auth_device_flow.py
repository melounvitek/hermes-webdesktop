"""Shared device-code / browser / TLS helpers for interactive OAuth logins.

Split out of ``hermes_cli/auth.py``; every moved name is re-imported there, so
``hermes_cli.auth.<name>`` keeps resolving (and monkeypatching) as before. Origin-internal
helpers are imported lazily inside each function (no import cycle; patches on
``hermes_cli.auth.<helper>`` still intercept).
"""

from __future__ import annotations

import logging
import os
import ssl
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse
from hermes_cli.auth_constants import (
    AuthError,
    DEFAULT_NOUS_PORTAL_URL,
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    DEVICE_CODE_GRANT_TYPE,
    OAUTH_OVER_SSH_DOCS_URL,
    httpx,
)
from utils import is_truthy_value

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")


def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    These environments typically don't set ``SSH_CLIENT`` / ``SSH_TTY``, so the SSH-only check left
    them with no guidance and no fallback.
    """
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        return True
    # Browser-only remote IDEs / cloud shells.  Keep this list narrow
    # (well-known, documented env vars set by the host platform) so
    # we don't falsely trip on a developer's local shell.
    for var in (
        "CLOUD_SHELL",         # GCP Cloud Shell
        "CODESPACES",          # GitHub Codespaces
        "CODESPACE_NAME",      # GitHub Codespaces (alt)
        "GITPOD_WORKSPACE_ID", # Gitpod
        "REPL_ID",             # Replit
        "STACKBLITZ",          # StackBlitz
    ):
        if os.getenv(var):
            return True
    return False


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    ``webbrowser.open()`` resolves to whatever the platform offers, and on a headless / CLI-only
    Linux box with no GUI browser installed that is often a text-mode browser (w3m/lynx/links) which
    launches inside the terminal and takes over the user's session.

    Heuristics: * Respect ``$BROWSER`` — if it names a known console browser, refuse. * On Linux,
    require a display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``) unless ``$BROWSER`` points at
    something graphical; no display server almost always means no GUI browser.
    """
    from hermes_cli.auth import _CONSOLE_BROWSER_NAMES
    import webbrowser as _webbrowser

    def _names_console_browser(value: str) -> bool:
        token = value.strip().split()[0] if value.strip() else ""
        base = os.path.basename(token).lower()
        return base in _CONSOLE_BROWSER_NAMES

    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False

    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        # An explicit graphical $BROWSER can work without $DISPLAY in odd
        # setups, but a console $BROWSER already returned False above, so the
        # only way to reach here with a $BROWSER set is a graphical one.
        if not has_display and not browser_env:
            return False

    try:
        controller = _webbrowser.get()
    except Exception:
        # No browser resolvable at all → definitely don't auto-open.
        return False

    candidate = (
        getattr(controller, "name", "")
        or getattr(controller, "basename", "")
        or ""
    )
    return not (candidate and _names_console_browser(candidate))


def _ssh_user_at_host() -> str:
    """Return best-effort 'user@hostname' for the SSH tunnel hint command.

    Falls back to placeholder tokens when the values cannot be determined so the hint is always
    syntactically valid even if not copy-pasteable.
    """
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when running a loopback-redirect OAuth flow on a remote host. The auth
    server (Spotify, MCP servers, ...) will redirect the user's browser to
    ``127.0.0.1:<port>/callback``. If the browser is on a different machine than the loopback
    listener (the usual SSH case), the redirect can't reach the listener without a local port
    forward.
    """
    from hermes_cli.auth import _is_remote_session
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host = parsed.hostname or ""
    port = parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print()
    print(divider)
    print("Remote session detected — SSH tunnel required")
    print(divider)
    print(f"Hermes is waiting for the OAuth callback on {redirect_uri}")
    print("but your browser is on a different machine. Run this command")
    print("in a NEW terminal on your local machine BEFORE opening the URL:")
    print()
    print(f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}")
    print()
    print("Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}")
    print(divider)
    print()


def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python, the system OpenSSL cannot locate the system trust store and valid
    public certs fail verification. When certifi is importable we pin its bundle explicitly;
    elsewhere we defer to httpx's built-in default (certifi via its own dependency). Mirrors the
    weixin fix in 3a0ec1d93.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    from hermes_cli.auth import _default_verify
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path,
            )
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code", "user_code", "verification_uri",
        "verification_uri_complete", "expires_in", "interval",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _nous_device_auth_timeout_message(portal_base_url: str) -> str:
    """Actionable timeout text for Nous device-code login failures.

    A bare "timed out" gives the user nothing to act on; the usual cause is Portal sign-in failing
    in the opened browser tab, so point at the Portal login page and the retry command.
    """
    portal = (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    return (
        "Timed out waiting for device authorization.\n"
        "  Portal sign-in is required before the device code can be approved.\n"
        "  If the browser showed a CAPTCHA / 'You did not pass CAPTCHA' error,\n"
        "  finish signing in at the Portal in a normal browser tab, then retry:\n"
        "    hermes portal\n"
        f"  Portal login: {portal}/login"
    )


def _print_device_code_instructions(
    verification_url: str,
    user_code: str,
    *,
    open_browser: bool,
    failure_dash: str = "--",
    swallow_open_errors: bool = False,
) -> None:
    """Print the shared "To continue" device-code block and optionally open the browser.

    Callers decide *whether* to open (remote-session / graphical-browser gating differs per
    provider); the wording of the fallback hint is parameterized so each provider keeps its
    historical dash style.
    """
    print()
    print("To continue:")
    print(f"  1. Open: {verification_url}")
    print(f"  2. If prompted, enter code: {user_code}")
    if not open_browser:
        return
    if swallow_open_errors:
        try:
            opened = webbrowser.open(verification_url)
        except Exception:
            opened = False
    else:
        opened = webbrowser.open(verification_url)
    if opened:
        print("  (Opened browser for verification)")
    else:
        print(f"  Could not open browser automatically {failure_dash} use the URL above.")


def _poll_device_token_generic(
    post: Callable[[], "httpx.Response"],
    *,
    expires_in: int,
    poll_interval: int,
    validate_success: Callable[[Dict[str, Any]], None],
    on_non_json_error: Callable[["httpx.Response"], Exception],
    on_error: Callable[["httpx.Response", Dict[str, Any]], Exception],
    on_timeout: Callable[[], Exception],
) -> Dict[str, Any]:
    """RFC 8628 device-code polling loop shared by the Nous and xAI flows.

    ``authorization_pending`` sleeps and retries; ``slow_down`` grows the interval by 1s (cap 30s).
    Every other error, a non-JSON error body, and the deadline are turned into provider-specific
    exceptions by the supplied factories so each caller keeps its exact error contract.
    """
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = poll_interval
    while time.monotonic() < deadline:
        response = post()
        if response.status_code == 200:
            payload = response.json()
            validate_success(payload)
            return payload
        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise on_non_json_error(response)
        error_code = str(error_payload.get("error") or "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue
        raise on_error(response, error_payload)
    raise on_timeout()


def _poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the Nous token endpoint until the user approves or the code expires."""
    def _validate(payload: Dict[str, Any]) -> None:
        if "access_token" not in payload:
            raise ValueError("Token response did not include access_token")

    def _error(_response, error_payload) -> Exception:
        error_code = error_payload.get("error", "")
        description = error_payload.get("error_description") or "Unknown authentication error"
        return RuntimeError(f"{error_code}: {description}")

    return _poll_device_token_generic(
        lambda: client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
            },
        ),
        expires_in=expires_in,
        poll_interval=max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)),
        validate_success=_validate,
        on_non_json_error=lambda _r: RuntimeError("Token endpoint returned a non-JSON error response"),
        on_error=_error,
        # Enriched at the SOURCE so every caller inherits the guidance:
        # the CLI login (_nous_device_code_login) and the dashboard/desktop
        # poller (web_server._nous_poller, which surfaces str(e) to the UI).
        on_timeout=lambda: TimeoutError(_nous_device_auth_timeout_message(portal_base_url)),
    )


def _prompt_yes_no(prompt: str, *, default: str) -> bool:
    """``input()`` a [Y/n]-style question; EOF/Ctrl-C count as *default*."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = default
    return answer in {"", "y", "yes"} if default == "y" else answer in {"y", "yes"}


def _print_login_success(provider_id: str, config_path: Path, *, show_auth_state: bool = False) -> None:
    print()
    print("Login successful!")
    if show_auth_state:
        from hermes_constants import display_hermes_home as _dhh
        print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider={provider_id})")


def _offer_existing_oauth_credentials(
    provider_id: str,
    *,
    resolve: Callable[[], Dict[str, Any]],
    is_expiring: Callable[[str, int], bool],
    display_name: str,
    default_base_url: str,
    expired_notice: Optional[str] = None,
) -> bool:
    """Offer to reuse still-valid stored OAuth credentials. Returns True when the user accepted.

    *resolve* attempts a refresh, so a resolved token should be valid — but double-check the
    expiry before telling the user "Login successful!".
    """
    from hermes_cli.auth import _update_config_for_provider
    try:
        existing = resolve()
        api_key = existing.get("api_key", "")
        if isinstance(api_key, str) and api_key and not is_expiring(api_key, 60):
            print(f"Existing {display_name} credentials found in Hermes auth store.")
            if _prompt_yes_no("Use existing credentials? [Y/n]: ", default="y"):
                config_path = _update_config_for_provider(
                    provider_id, existing.get("base_url", default_base_url),
                )
                _print_login_success(provider_id, config_path)
                return True
        elif expired_notice:
            print(expired_notice)
    except AuthError:
        pass
    return False
