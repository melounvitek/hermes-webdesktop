"""Status command for hermes CLI."""

import json
import os
import sys
import time
import importlib.util
import subprocess  # noqa: F401 — re-exported for tests that monkeypatch status.subprocess to guard against regressions
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.auth import AuthError, resolve_provider
from hermes_cli.colors import Colors, color
from hermes_cli.config import get_env_path, get_env_value, get_hermes_home, load_config, redact_key  # noqa: F401  (redact_key: status_auth resolves it here)
from hermes_cli.models import provider_label
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_cli.status_auth import (  # renderers wired into _SECTIONS below
    _render_api_keys,
    _render_apikey_providers,
    _render_auth_providers,
    _render_nous_gateway,
)
from hermes_constants import OPENROUTER_MODELS_URL
from hermes_constants import is_termux as _is_termux


def check_mark(ok: bool) -> str:
    return color("✓", Colors.GREEN) if ok else color("✗", Colors.RED)


def _section(title: str) -> None:
    """Print a blank line followed by a bold cyan ``◆`` section heading."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _row(name: str, ok: bool, text: str, width: int = 12) -> None:
    """Print one ``name  ✓/✗ text`` status row."""
    print(f"  {name:<{width}}  {check_mark(ok)} {text}")


def _detail(label: str, value) -> None:
    """Print an indented ``label: value`` detail line under a status row."""
    print(f"    {label:<12}{value}")


def _first_env_value(names) -> str:
    """Return the first non-empty env value among ``names`` (a str or tuple of names)."""
    return next((v for v in (get_env_value(n) or "" for n in ((names,) if isinstance(names, str) else names)) if v), "")


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_cfg = model_cfg.get("default") or model_cfg.get("name") or ""
    return (model_cfg.strip() if isinstance(model_cfg, str) else "") or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter":
        # A custom endpoint may live in config.yaml (model.base_url, the canonical
        # location) or the legacy OPENAI_BASE_URL env var; either way labeling it
        # "OpenRouter" is misleading.
        try:
            model_cfg = load_config().get("model")
            config_base_url = (model_cfg.get("base_url") or "").strip() if isinstance(model_cfg, dict) else ""
        except Exception:
            config_base_url = ""
        if config_base_url or get_env_value("OPENAI_BASE_URL"):
            effective = "custom"
    return provider_label(effective)


def _estop_status_line():
    """One-line pause banner for `hermes status`, or None when not paused."""
    try:
        from agent.estop import get_state
    except ImportError:
        return None
    state = get_state()
    if state is None:
        return None
    reason = state.get("reason")
    return f"⏸️  PAUSED (global emergency stop{f' — reason: {reason}' if reason else ''}; `hermes resume` to lift)"


# --- Data tables driving the per-section renderers -------------------------


# Simple env-driven terminal backends: (label, env var, default, empty-counts-as-unset).
_TERMINAL_ENV_ROWS = {
    "ssh": (("SSH Host:", "TERMINAL_SSH_HOST", "(not set)", True), ("SSH User:", "TERMINAL_SSH_USER", "(not set)", True)),
    "docker": (("Docker Image:", "TERMINAL_DOCKER_IMAGE", "python:3.11-slim", False),),
    "daytona": (("Daytona Image:", "TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20", False),),
}

_PLATFORMS = {  # name -> (token env var, home-channel env var or None)
    "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
    "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"),
    "WhatsApp": ("WHATSAPP_ENABLED", None),
    "Signal": ("SIGNAL_HTTP_URL", "SIGNAL_HOME_CHANNEL"),
    "Slack": ("SLACK_BOT_TOKEN", None),
    "Email": ("EMAIL_ADDRESS", "EMAIL_HOME_ADDRESS"),
    "SMS": ("TWILIO_ACCOUNT_SID", "SMS_HOME_CHANNEL"),
    "DingTalk": ("DINGTALK_CLIENT_ID", None),
    "Feishu": ("FEISHU_APP_ID", "FEISHU_HOME_CHANNEL"),
    "WeCom": ("WECOM_BOT_ID", "WECOM_HOME_CHANNEL"),
    "WeCom Callback": ("WECOM_CALLBACK_CORP_ID", None),
    "Weixin": ("WEIXIN_ACCOUNT_ID", "WEIXIN_HOME_CHANNEL"),
    "BlueBubbles": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_HOME_CHANNEL"),
    "QQBot": ("QQ_APP_ID", "QQ_HOME_CHANNEL"),
    "Yuanbao": ("YUANBAO_APP_ID", "YUANBAO_HOME_CHANNEL"),
}

# Gateway manager label when the runtime snapshot is unavailable, keyed by platform.
_GATEWAY_FALLBACK = {"linux": ("unknown", "systemd/manual"), "darwin": ("unknown", "launchd")}


class _StatusContext:
    """State shared across section renderers: config, --deep, and the Nous login facts
    the Auth Providers section derives that the Nous Tool Gateway section needs later."""

    def __init__(self, deep: bool):
        self.deep, self.config = deep, {}
        self.nous_logged_in = self.nous_inference_present = False
        self.nous_account_info = None


def _render_header(ctx):
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Hermes Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    paused = _estop_status_line()
    if paused:
        print()
        print(color(paused, Colors.YELLOW, Colors.BOLD))


def _render_environment(ctx):
    _section("Environment")
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    env_exists = get_env_path().exists()
    print(f"  .env file:    {check_mark(env_exists)} {'exists' if env_exists else 'not found'}")
    try:
        ctx.config = load_config()
    except Exception:
        ctx.config = {}
    print(f"  Model:        {_configured_model_label(ctx.config)}")
    print(f"  Provider:     {_effective_provider_label()}")


def _render_terminal(ctx):
    _section("Terminal Backend")
    terminal_cfg = ctx.config.get("terminal", {}) if isinstance(ctx.config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "") or terminal_cfg.get("backend", "local")
    print(f"  Backend:      {terminal_env}")

    if terminal_env in _TERMINAL_ENV_ROWS:
        for label, var, default, empty_is_unset in _TERMINAL_ENV_ROWS[terminal_env]:
            value = (os.getenv(var, "") or default) if empty_is_unset else os.getenv(var, default)
            print(f"  {label:<13} {value}")
    elif terminal_env == "vercel_sandbox":
        persist = os.getenv("TERMINAL_CONTAINER_PERSISTENT")
        persist_enabled = (bool(terminal_cfg.get("container_persistent", True)) if persist is None
                           else persist.lower() in {"1", "true", "yes", "on"})
        auth_status = describe_vercel_auth()
        sdk_ok = importlib.util.find_spec("vercel") is not None
        sdk_label = "installed" if sdk_ok else "missing (install: pip install 'hermes-agent[vercel]')"
        print(f"  Runtime:      {os.getenv('TERMINAL_VERCEL_RUNTIME') or terminal_cfg.get('vercel_runtime') or 'node24'}")
        print(f"  SDK:          {check_mark(sdk_ok)} {sdk_label}")
        print(f"  Auth:         {check_mark(auth_status.ok)} {auth_status.label}")
        for line in auth_status.detail_lines:
            print(f"  Auth detail:  {line}")
        print(f"  Persistence:  {'snapshot filesystem' if persist_enabled else 'ephemeral filesystem'}")
        print("  Processes:    live processes do not survive cleanup, snapshots, or sandbox recreation")
    else:
        # Plugin-registered terminal backends: show availability via the
        # provider's doctor rows (fail-soft — never break `hermes status`).
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            from agent.terminal_env_registry import get_provider
            provider = get_provider(terminal_env)
            if provider is not None:
                for ok, label, text in provider.doctor_checks():
                    print(f"  {label}: {check_mark(bool(ok))} {text}")
        except Exception:
            pass

    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")


def _render_platforms(ctx):
    _section("Messaging Platforms")
    for name, (token_var, home_var) in _PLATFORMS.items():
        has_token = bool(os.getenv(token_var, ""))
        home_channel = os.getenv(home_var, "") if home_var else ""
        _row(name, has_token, ("configured" if has_token else "not configured") + (f" (home: {home_channel})" if home_channel else ""))

    try:  # Plugin-registered platforms
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            # Per-entry guard: one raising probe must not abort the listing of
            # every remaining plugin platform (matches the other check_fn sites).
            try:
                configured = bool(entry.check_fn())
            except Exception:
                configured = False
            _row(entry.label, configured, f"{'configured' if configured else 'not configured'} (plugin)")
    except Exception:
        pass


def _render_gateway(ctx):
    _section("Gateway Service")
    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot, _format_gateway_pids

        snapshot = get_gateway_runtime_snapshot()
        print(f"  Status:       {check_mark(snapshot.running)} {'running' if snapshot.running else 'stopped'}")
        print(f"  Manager:      {snapshot.manager}")
        if snapshot.gateway_pids:
            print(f"  PID(s):       {_format_gateway_pids(snapshot.gateway_pids)}")
        if snapshot.has_process_service_mismatch:
            print("  Service:      installed but not managing the current running gateway")
        elif _is_termux() and not snapshot.gateway_pids:
            print("  Start with:   hermes gateway")
            print("  Note:         Android may stop background jobs when Termux is suspended")
        elif snapshot.service_installed and not snapshot.service_running:
            print("  Service:      installed but stopped")
    except Exception:
        if _is_termux():
            status_text, manager = "unknown", "Termux / manual process"
        else:
            platform = "linux" if sys.platform.startswith("linux") else sys.platform
            status_text, manager = _GATEWAY_FALLBACK.get(platform, ("N/A", "(not supported on this platform)"))
        print(f"  Status:       {color(status_text, Colors.DIM)}")
        print(f"  Manager:      {manager}")


def _render_cron(ctx):
    _section("Scheduled Jobs")
    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    if not jobs_file.exists():
        print("  Jobs:         0")
        return
    try:
        # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors
        # may leave a UTF-8 BOM that plain utf-8 json.load rejects.
        with open(jobs_file, encoding="utf-8-sig") as f:
            jobs = json.load(f).get("jobs", [])
        print(f"  Jobs:         {sum(1 for j in jobs if j.get('enabled', True))} active, {len(jobs)} total")
    except Exception:
        print("  Jobs:         (error reading jobs file)")


def _render_sessions(ctx):
    _section("Sessions")
    # Gateway session count: state.db is the source of truth; fall back to
    # sessions.json for pre-migration installs.
    gateway_rows = []
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            lister = getattr(db, "list_gateway_sessions", None)
            if callable(lister):
                gateway_rows = lister(active_only=True) or []
        finally:
            db.close()
    except Exception:
        gateway_rows = []

    if gateway_rows:
        print(f"  Active:       {len(gateway_rows)} session(s)")
        freshest = max((float(r.get("last_active") or 0) for r in gateway_rows), default=0.0)
        if freshest > 0:
            from hermes_cli.timefmt import relative_time
            print(f"  Last activity:{relative_time(freshest):>13}")
    else:
        sessions_file = get_hermes_home() / "sessions" / "sessions.json"
        if sessions_file.exists():
            try:
                with open(sessions_file, encoding="utf-8") as f:
                    data = json.load(f)
                entries = [k for k in data if not str(k).startswith("_")] if isinstance(data, dict) else []
                print(f"  Active:       {len(entries)} session(s)")
            except Exception:
                print("  Active:       (error reading sessions file)")
        else:
            print("  Active:       0")

    # Slot usage, only when max_concurrent_sessions is set. The cap is shared
    # across CLI, desktop/TUI and the messaging gateway, so the surface that
    # gets rejected is rarely the one holding the slots — without this the only
    # way to find out is reading runtime/active_sessions.json by hand.
    try:
        from hermes_cli.active_sessions import (
            active_session_registry_snapshot, format_age, resolve_max_concurrent_sessions,
        )
        cap = resolve_max_concurrent_sessions(ctx.config)
    except Exception:
        cap = None
    if cap:
        try:
            held = active_session_registry_snapshot()
        except Exception:
            held = []
        print("  Slots:        " + color(f"{len(held)}/{cap} in use", Colors.YELLOW if len(held) >= cap else Colors.GREEN))
        now = time.time()
        for entry in sorted(held, key=lambda e: e.get("started_at") or 0):
            age = format_age(now - float(entry.get("started_at") or now))
            print(f"                {entry.get('surface') or 'unknown':<17} {entry.get('session_id') or '?':<24} {age}")


def _render_deep(ctx):
    if not ctx.deep:
        return
    _section("Deep Checks")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        try:
            import httpx
            response = httpx.get(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {openrouter_key}"}, timeout=10)
            ok = response.status_code == 200
            print(f"  OpenRouter:   {check_mark(ok)} {'reachable' if ok else f'error ({response.status_code})'}")
        except Exception as e:
            print(f"  OpenRouter:   {check_mark(False)} error: {e}")
    try:  # gateway port, informational: in use == gateway likely running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        port_in_use = sock.connect_ex(('127.0.0.1', 18789)) == 0
        sock.close()
        print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
    except OSError:
        pass


def _render_footer(ctx):
    print()
    for line in ("─" * 60, "  Run 'hermes doctor' for detailed diagnostics", "  Run 'hermes setup' to configure"):
        print(color(line, Colors.DIM))
    print()


# Print order of `hermes status`; each renderer takes the shared _StatusContext.
_SECTIONS = (
    _render_header, _render_environment, _render_api_keys, _render_auth_providers, _render_nous_gateway,
    _render_apikey_providers, _render_terminal, _render_platforms, _render_gateway, _render_cron,
    _render_sessions, _render_deep, _render_footer,
)


def show_status(args):
    """Show status of all Hermes Agent components."""
    ctx = _StatusContext(deep=getattr(args, 'deep', False))
    for render in _SECTIONS:
        render(ctx)
