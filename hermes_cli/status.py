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
from hermes_cli.config import get_env_path, get_env_value, get_hermes_home, load_config
from hermes_cli.models import provider_label
from hermes_cli.nous_account import (
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
)
from hermes_cli.nous_subscription import get_nous_subscription_features
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_constants import OPENROUTER_MODELS_URL
from hermes_constants import is_termux as _is_termux
from tools.tool_backend_helpers import managed_nous_tools_enabled


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


def _oauth_block(name: str, status: dict, hint: str, rows) -> None:
    """Print an OAuth provider row plus its conditional detail lines.

    ``rows`` are ``(label, status_key, formatter, gate)``: a detail prints when the raw value is
    truthy and ``gate`` is None or equals the logged-in state (False = only while logged out).
    """
    logged_in = bool(status.get("logged_in"))
    _row(name, logged_in, "logged in" if logged_in else f"not logged in (run: {hint})")
    for label, key, fmt, gate in rows:
        raw = status.get(key)
        if raw and (gate is None or gate == logged_in):
            _detail(label, fmt(raw) if fmt else raw)


def _first_env_value(names) -> str:
    """Return the first non-empty env value among ``names`` (a str or tuple of names)."""
    if isinstance(names, str):
        names = (names,)
    for candidate in names:
        v = get_env_value(candidate) or ""
        if v:
            return v
    return ""


def redact_key(key: str) -> str:
    """Redact an API key for display.

    Thin wrapper over :func:`agent.redact.mask_secret` that keeps the dim "(not set)" placeholder
    consistent with ``hermes config`` output.
    """
    from agent.redact import mask_secret
    return mask_secret(key, empty=color("(not set)", Colors.DIM))


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return "(unknown)"
    from datetime import datetime, timezone
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _qwen_expiry(expires_at_ms) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(int(expires_at_ms) / 1000, tz=timezone.utc).isoformat()


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_cfg = model_cfg.get("default") or model_cfg.get("name") or ""
    elif not isinstance(model_cfg, str):
        model_cfg = ""
    return model_cfg.strip() or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter":
        # A custom endpoint may be configured either in config.yaml
        # (model.base_url — the canonical location; the runtime treats
        # config.yaml as the single source of truth) or via the legacy
        # OPENAI_BASE_URL env var. Either way, labeling it "OpenRouter"
        # is misleading (#3296).
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


# ---------------------------------------------------------------------------
# Data tables driving the per-section renderers below.
# ---------------------------------------------------------------------------

# Values may be a single env var name (str) or a tuple of alternates (first found wins).
_API_KEYS: dict[str, str | tuple[str, ...]] = {
    "OpenRouter": "OPENROUTER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Google / Gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "DeepSeek": "DEEPSEEK_API_KEY",
    "xAI / Grok": "XAI_API_KEY",
    "NVIDIA NIM": "NVIDIA_API_KEY",
    "Z.AI / GLM": "GLM_API_KEY",
    "Kimi": "KIMI_API_KEY",
    "StepFun Step Plan": "STEPFUN_API_KEY",
    "MiniMax": "MINIMAX_API_KEY",
    "MiniMax-CN": "MINIMAX_CN_API_KEY",
    "DeepInfra": "DEEPINFRA_API_KEY",
    "Firecrawl": "FIRECRAWL_API_KEY",
    "Tavily": "TAVILY_API_KEY",
    "Keenable": "KEENABLE_API_KEY",
    "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
    "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
    "FAL": "FAL_KEY",
    "ElevenLabs": "ELEVENLABS_API_KEY",
    "GitHub": "GITHUB_TOKEN",
}

# OAuth detail rows: (label, status key, formatter, gate) — see _oauth_block.
_FILE_REFRESH_ROWS = (
    ("Auth file:", "auth_store", None, None),
    ("Refreshed:", "last_refresh", _format_iso_timestamp, None),
    ("Error:", "error", None, False),
)
_OAUTH_BLOCKS = (
    # (row name, auth getter, login hint, detail rows)
    ("OpenAI Codex", "get_codex_auth_status", "hermes model", _FILE_REFRESH_ROWS),
    ("Qwen OAuth", "get_qwen_auth_status", "qwen auth qwen-oauth", (
        ("Auth file:", "auth_file", None, None),
        ("Access exp:", "expires_at_ms", _qwen_expiry, None),
        ("Error:", "error", None, False),
    )),
    ("MiniMax OAuth", "get_minimax_oauth_auth_status", "hermes auth add minimax-oauth", (
        ("Region:", "region", None, True),
        ("Access exp:", "expires_at", None, None),
        ("Error:", "error", None, False),
    )),
    ("xAI OAuth", "get_xai_oauth_auth_status", "hermes auth add xai-oauth", _FILE_REFRESH_ROWS),
)

_APIKEY_PROVIDERS = {
    "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    "Kimi / Moonshot":  ("KIMI_API_KEY",),
    "StepFun Step Plan": ("STEPFUN_API_KEY",),
    "MiniMax":          ("MINIMAX_API_KEY",),
    "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
    "DeepInfra":        ("DEEPINFRA_API_KEY",),
}

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
        self.deep = deep
        self.config: dict = {}
        self.nous_logged_in = False
        self.nous_inference_present = False
        self.nous_account_info = None


def _render_header(ctx):
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Hermes Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    _paused_line = _estop_status_line()
    if _paused_line:
        print()
        print(color(_paused_line, Colors.YELLOW, Colors.BOLD))


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


def _render_api_keys(ctx):
    _section("API Keys")
    for name, env_ref in _API_KEYS.items():
        value = _first_env_value(env_ref)
        _row(name, bool(value), redact_key(value))
    # Anthropic uses the dedicated lookup (it also resolves OAuth tokens).
    from hermes_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    _row("Anthropic", bool(anthropic_value), redact_key(anthropic_value))


def _render_auth_providers(ctx):
    _section("Auth Providers")
    import hermes_cli.auth as auth
    try:
        # Read-only display: use the refresh-free snapshot so `hermes status`
        # never performs an OAuth refresh or burns a single-use refresh token.
        nous_status = auth.get_nous_auth_status_local()
        statuses = {getter: getattr(auth, getter)() for _, getter, _, _ in _OAUTH_BLOCKS[:3]}
    except Exception:
        nous_status, statuses = {}, {}
    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the Nous/Codex/Qwen/MiniMax rows.
    try:
        statuses["get_xai_oauth_auth_status"] = auth.get_xai_oauth_auth_status() or {}
    except Exception:
        statuses["get_xai_oauth_auth_status"] = {}

    info = None
    if any(nous_status.get(k) for k in (
        "logged_in", "access_token", "portal_base_url", "inference_credential_present", "error_code"
    )):
        try:
            info = get_nous_portal_account_info()
        except Exception:
            info = None
    ctx.nous_account_info = info
    ctx.nous_logged_in = logged_in = bool(nous_status.get("logged_in") or (info and info.logged_in))
    ctx.nous_inference_present = inference = bool(
        nous_status.get("inference_credential_present") or (info and info.inference_credential_present)
    )
    nous_error = nous_status.get("error")
    _row(
        "Nous Portal", logged_in,
        "logged in" if logged_in
        else "not logged in (Nous inference key configured)" if inference
        else "not logged in (run: hermes portal)",
    )
    portal_url = nous_status.get("portal_base_url") or "(unknown)"
    inference_url = nous_status.get("inference_base_url") or (info.inference_base_url if info else None)
    for label, value, show in (
        ("Portal URL:", portal_url, logged_in or portal_url != "(unknown)" or nous_error),
        ("Inference:", inference_url, inference and inference_url),
        ("Access exp:", _format_iso_timestamp(nous_status.get("access_expires_at")),
         logged_in or nous_status.get("access_expires_at")),
        ("Key exp:", _format_iso_timestamp(nous_status.get("agent_key_expires_at")),
         logged_in or inference or nous_status.get("agent_key_expires_at")),
        ("Refresh:", "yes" if nous_status.get("has_refresh_token") else "no",
         logged_in or nous_status.get("has_refresh_token")),
        ("Error:", nous_error, nous_error),
    ):
        if show:
            _detail(label, value)
    for name, getter, hint, rows in _OAUTH_BLOCKS:
        _oauth_block(name, statuses.get(getter, {}), hint, rows)


def _render_nous_gateway(ctx):
    if managed_nous_tools_enabled():
        features = get_nous_subscription_features(ctx.config)
        _section("Nous Tool Gateway")
        print("  Nous Portal   ✓ managed tools available" if features.nous_auth_present else "  Nous Portal   ✗ not logged in")
        for feature in features.items():
            if feature.managed_by_nous:
                state = "active via Nous subscription"
            elif feature.active:
                state = f"active via {feature.current_provider or 'configured provider'}"
            elif feature.included_by_default and features.nous_auth_present:
                state = "included by subscription, not currently selected"
            elif feature.key == "modal" and features.nous_auth_present:
                state = "available via subscription (optional)"
            else:
                state = "not configured"
            print(f"  {feature.label:<15} {check_mark(feature.available or feature.active or feature.managed_by_nous)} {state}")
    elif ctx.nous_logged_in or ctx.nous_inference_present:
        # Nous OAuth without entitlement, or an opaque inference key without
        # Portal account information, cannot enable the Tool Gateway.
        _section("Nous Tool Gateway")
        message = format_nous_portal_entitlement_message(
            ctx.nous_account_info, capability="managed web, image, TTS, STT, browser, and Modal tools"
        )
        for line in (message or "").splitlines():
            print(f"  {line}")


def _render_apikey_providers(ctx):
    _section("API-Key Providers")
    for pname, env_vars in _APIKEY_PROVIDERS.items():
        configured = bool(_first_env_value(env_vars))
        label = "configured" if configured else "not configured (run: hermes model)"
        print(f"  {pname:<16} {check_mark(configured)} {label}")

    # LM Studio reachability — only probe when it's the active provider so
    # users with foreign configs don't see noise. Auth rejection vs. silent
    # empty list is the most common LM Studio support case.
    if _effective_provider_label() == "LM Studio":
        from hermes_cli.models import probe_lmstudio_models
        model_cfg = ctx.config.get("model")
        base = (model_cfg.get("base_url") if isinstance(model_cfg, dict) else None) or get_env_value("LM_BASE_URL") or "http://127.0.0.1:1234/v1"
        try:
            models = probe_lmstudio_models(api_key=get_env_value("LM_API_KEY") or "", base_url=base, timeout=1.5)
            ok = models is not None
            msg = f"reachable ({len(models)} model(s)) at {base}" if ok else f"unreachable at {base}"
        except AuthError:
            ok, msg = False, "auth rejected — set LM_API_KEY"
        print(f"  {'LM Studio':<16} {check_mark(ok)} {msg}")


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
        runtime = os.getenv("TERMINAL_VERCEL_RUNTIME") or terminal_cfg.get("vercel_runtime") or "node24"
        persist = os.getenv("TERMINAL_CONTAINER_PERSISTENT")
        persist_enabled = (
            bool(terminal_cfg.get("container_persistent", True))
            if persist is None
            else persist.lower() in {"1", "true", "yes", "on"}
        )
        auth_status = describe_vercel_auth()
        sdk_ok = importlib.util.find_spec("vercel") is not None
        sdk_label = "installed" if sdk_ok else "missing (install: pip install 'hermes-agent[vercel]')"
        print(f"  Runtime:      {runtime}")
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
            _provider = get_provider(terminal_env)
            if _provider is not None:
                for _ok, _label, _text in _provider.doctor_checks():
                    print(f"  {_label}: {check_mark(bool(_ok))} {_text}")
        except Exception:
            pass

    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")


def _render_platforms(ctx):
    _section("Messaging Platforms")
    for name, (token_var, home_var) in _PLATFORMS.items():
        has_token = bool(os.getenv(token_var, ""))
        home_channel = os.getenv(home_var, "") if home_var else ""
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        _row(name, has_token, status)

    # Plugin-registered platforms
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            # Per-entry guard: one raising probe must not abort the listing
            # of every remaining plugin platform (matches the other three
            # check_fn call sites).
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
        is_running = snapshot.running
        print(f"  Status:       {check_mark(is_running)} {'running' if is_running else 'stopped'}")
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
    if jobs_file.exists():
        try:
            # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors
            # may leave a UTF-8 BOM that plain utf-8 json.load rejects.
            with open(jobs_file, encoding="utf-8-sig") as f:
                jobs = json.load(f).get("jobs", [])
            enabled = sum(1 for j in jobs if j.get("enabled", True))
            print(f"  Jobs:         {enabled} active, {len(jobs)} total")
        except Exception:
            print("  Jobs:         (error reading jobs file)")
    else:
        print("  Jobs:         0")


def _render_sessions(ctx):
    _section("Sessions")
    # Gateway session count: state.db is the source of truth (#9006);
    # fall back to sessions.json for pre-migration installs.
    _session_count, _gateway_rows = None, []
    try:
        from hermes_state import SessionDB
        _db = SessionDB()
        try:
            _lister = getattr(_db, "list_gateway_sessions", None)
            if callable(_lister):
                _gateway_rows = _lister(active_only=True) or []
                _session_count = len(_gateway_rows)
        finally:
            _db.close()
    except Exception:
        _session_count, _gateway_rows = None, []

    if _session_count:
        print(f"  Active:       {_session_count} session(s)")
        freshest = max((float(r.get("last_active") or 0) for r in _gateway_rows), default=0.0)
        if freshest > 0:
            from hermes_cli.timefmt import relative_time
            print(f"  Last activity:{relative_time(freshest):>13}")
    else:
        sessions_file = get_hermes_home() / "sessions" / "sessions.json"
        if sessions_file.exists():
            try:
                with open(sessions_file, encoding="utf-8") as f:
                    data = json.load(f)
                _entries = [k for k in data if not str(k).startswith("_")] if isinstance(data, dict) else []
                print(f"  Active:       {len(_entries)} session(s)")
            except Exception:
                print("  Active:       (error reading sessions file)")
        else:
            print(f"  Active:       {_session_count or 0}")

    # Slot usage, only when max_concurrent_sessions is set. The cap is shared
    # across CLI, desktop/TUI and the messaging gateway, so the surface that
    # gets rejected is rarely the one holding the slots — without this the only
    # way to find out is reading runtime/active_sessions.json by hand.
    try:
        from hermes_cli.active_sessions import (
            active_session_registry_snapshot, format_age, resolve_max_concurrent_sessions,
        )
        _cap = resolve_max_concurrent_sessions(ctx.config)
    except Exception:
        _cap = None
    if _cap:
        try:
            _held = active_session_registry_snapshot()
        except Exception:
            _held = []
        _full = len(_held) >= _cap
        print("  Slots:        " + color(f"{len(_held)}/{_cap} in use", Colors.YELLOW if _full else Colors.GREEN))
        _now = time.time()
        for _entry in sorted(_held, key=lambda e: e.get("started_at") or 0):
            _age = format_age(_now - float(_entry.get("started_at") or _now))
            print(
                f"                {_entry.get('surface') or 'unknown':<17} "
                f"{_entry.get('session_id') or '?':<24} {_age}"
            )


def _render_deep(ctx):
    if not ctx.deep:
        return
    _section("Deep Checks")
    # Check OpenRouter connectivity
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        try:
            import httpx
            response = httpx.get(OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {openrouter_key}"}, timeout=10)
            ok = response.status_code == 200
            print(f"  OpenRouter:   {check_mark(ok)} {'reachable' if ok else f'error ({response.status_code})'}")
        except Exception as e:
            print(f"  OpenRouter:   {check_mark(False)} error: {e}")
    # Check gateway port
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        port_in_use = sock.connect_ex(('127.0.0.1', 18789)) == 0  # informational: gateway likely running
        sock.close()
        print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
    except OSError:
        pass


def _render_footer(ctx):
    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'hermes doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'hermes setup' to configure", Colors.DIM))
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
