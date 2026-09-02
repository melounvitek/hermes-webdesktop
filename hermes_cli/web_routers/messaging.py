"""Messaging dashboard routes: WhatsApp/Telegram onboarding and per-platform enable/config/test.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
import re
import subprocess
import threading
import asyncio
import json
import secrets
import time
import urllib.parse
import os
from datetime import datetime, timezone
from fastapi import APIRouter
from hermes_cli.web_deps import late
from fastapi import HTTPException
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import get_env_path
from hermes_cli.web_models import MessagingPlatformUpdate, TelegramOnboardingStart, TelegramOnboardingApply, WhatsAppOnboardingStart, WhatsAppOnboardingApply
from pathlib import Path
from typing import Any, Optional
from hermes_cli.config import redact_key
from gateway.status import resolve_gateway_liveness
from hermes_cli.config import OPTIONAL_ENV_VARS

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# web_server helpers, late-bound so monkeypatch.setattr(web_server, ...) stays authoritative.
_config_profile_scope = late("_config_profile_scope")
_messaging_platform_catalog = late("_messaging_platform_catalog")
_profile_scope = late("_profile_scope")
_resolve_profile_dir = late("_resolve_profile_dir")
_restart_gateway_after_whatsapp_onboarding = late("_restart_gateway_after_whatsapp_onboarding")
_restart_gateway_after = late("_restart_gateway_after")
_telegram_onboarding_error_message = late("_telegram_onboarding_error_message")
_telegram_onboarding_request_sync = late("_telegram_onboarding_request_sync")
_whatsapp_onboarding_payload = late("_whatsapp_onboarding_payload")
_whatsapp_session_path = late("_whatsapp_session_path")
_write_platform_enabled = late("_write_platform_enabled")
load_env = late("load_env")
read_runtime_status = late("read_runtime_status")
remove_env_value = late("remove_env_value")
save_env_value = late("save_env_value")
_gateway_subcommand = late("_gateway_subcommand")
_probe_gateway_health = late("_probe_gateway_health")


# Display labels for env vars not in OPTIONAL_ENV_VARS (HOME_CHANNEL_*, bridge
# toggles, Twilio, HASS, Email, etc.). Anything missing from OPTIONAL_ENV_VARS
# falls back here so the UI can still render a friendly label.
_MESSAGING_ENV_FALLBACKS: dict[str, dict[str, Any]] = {
    "SIGNAL_HTTP_URL": {
        "description": "signal-cli REST API base URL, e.g. http://127.0.0.1:8080",
        "prompt": "Signal bridge URL",
        "url": "https://github.com/bbernhard/signal-cli-rest-api",
    },
    "SIGNAL_ACCOUNT": {
        "description": "Signal account phone number registered with the bridge",
        "prompt": "Signal account",
    },
    "SIGNAL_ALLOWED_USERS": {
        "description": "Comma-separated Signal users allowed to use the bot",
        "prompt": "Allowed Signal users",
    },
    "WHATSAPP_ENABLED": {
        "description": "Enable the WhatsApp gateway adapter",
        "prompt": "Enable WhatsApp",
        "advanced": True,
    },
    "WHATSAPP_MODE": {
        "description": "WhatsApp bridge mode",
        "prompt": "WhatsApp mode",
        "advanced": True,
    },
    "WHATSAPP_DM_POLICY": {
        "description": "How WhatsApp direct messages are authorized",
        "prompt": "WhatsApp DM policy",
        "advanced": True,
    },
    "WHATSAPP_ALLOWED_USERS": {
        "description": "Comma-separated WhatsApp users allowed to use the bot",
        "prompt": "Allowed WhatsApp users",
    },
    "HASS_URL": {
        "description": "Home Assistant base URL, e.g. https://homeassistant.local:8123",
        "prompt": "Home Assistant URL",
    },
    "HASS_TOKEN": {
        "description": "Long-lived access token from Home Assistant (Profile → Security)",
        "prompt": "Home Assistant access token",
        "password": True,
    },
    "EMAIL_ADDRESS": {
        "description": "Email address to send and receive from",
        "prompt": "Email address",
    },
    "EMAIL_PASSWORD": {
        "description": "Email account password or app password",
        "prompt": "Email password",
        "password": True,
    },
    "EMAIL_IMAP_HOST": {
        "description": "IMAP server host (e.g. imap.gmail.com)",
        "prompt": "IMAP host",
    },
    "EMAIL_SMTP_HOST": {
        "description": "SMTP server host (e.g. smtp.gmail.com)",
        "prompt": "SMTP host",
    },
    "TWILIO_ACCOUNT_SID": {
        "description": "Twilio Account SID",
        "prompt": "Twilio Account SID",
        "url": "https://www.twilio.com/console",
    },
    "TWILIO_AUTH_TOKEN": {
        "description": "Twilio Auth Token",
        "prompt": "Twilio Auth Token",
        "password": True,
    },
    "WECOM_BOT_ID": {"description": "WeCom group bot ID", "prompt": "WeCom Bot ID"},
    "WECOM_SECRET": {
        "description": "WeCom group bot secret",
        "prompt": "WeCom Secret",
        "password": True,
    },
    "WECOM_CALLBACK_CORP_ID": {
        "description": "WeCom corp ID",
        "prompt": "WeCom Corp ID",
    },
    "WECOM_CALLBACK_CORP_SECRET": {
        "description": "WeCom app corp secret",
        "prompt": "WeCom Corp Secret",
        "password": True,
    },
    "WECOM_CALLBACK_AGENT_ID": {
        "description": "WeCom app agent ID",
        "prompt": "WeCom Agent ID",
    },
    "WECOM_CALLBACK_TOKEN": {
        "description": "WeCom callback verification token",
        "prompt": "WeCom Token",
    },
    "WECOM_CALLBACK_ENCODING_AES_KEY": {
        "description": "WeCom callback AES encoding key",
        "prompt": "WeCom AES Key",
        "password": True,
    },
    "WEIXIN_ACCOUNT_ID": {
        "description": "iLink Bot account ID obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot account ID",
    },
    "WEIXIN_TOKEN": {
        "description": "iLink Bot token obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot token",
        "password": True,
    },
    "WEIXIN_BASE_URL": {
        "description": "iLink API base URL saved by QR login (default: https://ilinkai.weixin.qq.com)",
        "prompt": "iLink API base URL",
    },
    "FEISHU_APP_ID": {"description": "Feishu / Lark app ID", "prompt": "App ID"},
    "FEISHU_APP_SECRET": {
        "description": "Feishu / Lark app secret",
        "prompt": "App secret",
        "password": True,
    },
    "FEISHU_ENCRYPT_KEY": {
        "description": "Feishu / Lark encrypt key",
        "prompt": "Encrypt key",
        "password": True,
    },
    "FEISHU_VERIFICATION_TOKEN": {
        "description": "Feishu / Lark verification token",
        "prompt": "Verification token",
        "password": True,
    },
    "DINGTALK_CLIENT_ID": {
        "description": "DingTalk client ID (App key)",
        "prompt": "Client ID",
    },
    "DINGTALK_CLIENT_SECRET": {
        "description": "DingTalk client secret (App secret)",
        "prompt": "Client secret",
        "password": True,
    },
}


# Kept in sync with the corresponding frontend validation in ChannelsPage.tsx.
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\d+:[A-Za-z0-9_-]{30,}")


_TELEGRAM_USER_ID_RE = re.compile(r"\d+")


_SLACK_MEMBER_ID_RE = re.compile(r"[UW][A-Z0-9]{2,}")


def _messaging_env_info(key: str) -> dict[str, Any]:
    info = OPTIONAL_ENV_VARS.get(key) or _MESSAGING_ENV_FALLBACKS.get(key) or {}
    return {
        "description": info.get("description", ""),
        "prompt": info.get("prompt", key),
        "help": info.get("help", ""),
        "url": info.get("url"),
        "is_password": info.get("password", False),
        "advanced": info.get("advanced", False),
    }


def _gateway_platform_config(platform_id: str):
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform = Platform(platform_id)
    platform_config = config.platforms.get(platform)
    return config, platform, platform_config


def _gateway_display_command(profile: Optional[str], verb: str) -> str:
    return " ".join(["hermes", *_gateway_subcommand(profile, verb)])


def _validate_messaging_env_value(platform_id: str, key: str, value: str) -> None:
    """Reject platform credentials that are clearly in the wrong field."""
    if not value:
        return

    if platform_id == "telegram":
        if key == "TELEGRAM_BOT_TOKEN" and not _TELEGRAM_BOT_TOKEN_RE.fullmatch(value):
            raise HTTPException(
                status_code=400,
                detail="Telegram bot token must be the complete token from @BotFather, such as 123456789:ABC…",
            )
        if key == "TELEGRAM_ALLOWED_USERS":
            user_ids = [part.strip() for part in value.split(",") if part.strip()]
            if any(not _TELEGRAM_USER_ID_RE.fullmatch(user_id) for user_id in user_ids):
                raise HTTPException(
                    status_code=400,
                    detail="Telegram allowed users must be comma-separated numeric user IDs.",
                )
        return

    if platform_id != "slack":
        return

    if key == "SLACK_BOT_TOKEN" and not value.startswith("xoxb-"):
        raise HTTPException(
            status_code=400,
            detail="Slack Bot Token must start with xoxb-. Paste the bot token from OAuth & Permissions.",
        )
    if key == "SLACK_APP_TOKEN" and not value.startswith("xapp-"):
        raise HTTPException(
            status_code=400,
            detail="Slack App Token must start with xapp-. Paste the app-level token from Basic Information > App-Level Tokens.",
        )
    if key == "SLACK_ALLOWED_USERS":
        # Mirror the gateway's parse (gateway/platforms/slack.py): split on comma,
        # strip, and drop empty entries so a trailing/interior comma isn't rejected
        # here when the runtime would accept it. "*" is the allow-all wildcard.
        user_ids = [part.strip() for part in value.split(",") if part.strip()]
        invalid = [
            user_id
            for user_id in user_ids
            if user_id != "*" and not _SLACK_MEMBER_ID_RE.fullmatch(user_id)
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail="Slack allowed user IDs must be comma-separated member IDs like U01ABC2DEF3.",
            )


def _catalog_lookup(platform_id: str) -> dict[str, Any] | None:
    for entry in _messaging_platform_catalog():
        if entry["id"] == platform_id:
            return entry
    return None


def _messaging_platform_payload(
    entry: dict[str, Any],
    env_on_disk: dict[str, str],
    runtime: dict | None,
    scoped: bool = False,
    profile_home: Optional[Path] = None,
) -> dict[str, Any]:
    from hermes_cli.web_server import (
        _GATEWAY_HEALTH_URL,
        get_running_pid_cached,
        get_runtime_status_running_pid,
        load_config,
    )
    platform_id = entry["id"]
    runtime_platforms = runtime.get("platforms") if runtime else {}
    runtime_platform = (
        runtime_platforms.get(platform_id, {})
        if isinstance(runtime_platforms, dict)
        else {}
    )
    # Same shared ladder /api/status uses. Before this was unified, the two
    # endpoints disagreed on the same page load — the sidebar strip read
    # "running" (it probed GATEWAY_HEALTH_URL and scoped to the requested
    # profile) while the Channels page rendered "The gateway is not running"
    # (it did neither). Cross-container, profile-scoped, and
    # launch-service-managed deployments each hit that split.
    #
    # profile_home is passed when the request was scoped to a named profile:
    # gateway/status readers resolve process-level paths and do NOT follow the
    # HERMES_HOME contextvar override (#56986 / #69143), so the profile's
    # directory has to be handed over explicitly or messaging silently reports
    # another profile's gateway (#71211).
    liveness = resolve_gateway_liveness(
        profile_dir=profile_home,
        runtime=runtime,
        health_probe=(
            _probe_gateway_health if _GATEWAY_HEALTH_URL else None
        ),
        pid_probe=get_running_pid_cached,
        runtime_reader=read_runtime_status,
        runtime_pid_probe=get_runtime_status_running_pid,
    )
    gateway_running = liveness.running
    env_vars = []

    for key in entry["env_vars"]:
        # When profile-scoped, judge only the profile's own .env — the
        # dashboard process's os.environ carries the ROOT install's .env
        # (loaded at startup) and would falsely report the root credentials
        # as the profile's.
        value = env_on_disk.get(key) or ("" if scoped else os.getenv(key, ""))
        env_vars.append(
            {
                "key": key,
                "required": key in entry["required_env"],
                "is_set": bool(value),
                "redacted_value": redact_key(value) if value else None,
                **_messaging_env_info(key),
            }
        )

    if scoped:
        # Profile-scoped view: derive enablement/configuration from the
        # profile's config.yaml + .env only. load_gateway_config()'s
        # env-override layer reads os.environ and would leak the root
        # install's tokens into the profile's reported state.
        try:
            cfg = load_config()
            platforms_cfg = cfg.get("platforms") or {}
            plat_cfg = platforms_cfg.get(platform_id)
            if not isinstance(plat_cfg, dict):
                plat_cfg = {}
            enabled = bool(plat_cfg.get("enabled"))
            hc = plat_cfg.get("home_channel")
            home_channel = hc if isinstance(hc, dict) else None
        except Exception:
            enabled = False
            home_channel = None
        configured = all(env_on_disk.get(key) for key in entry["required_env"])
    else:
        try:
            gateway_config, platform, platform_config = _gateway_platform_config(
                platform_id
            )
            enabled = bool(platform_config and platform_config.enabled)
            configured = bool(
                platform_config
                and gateway_config._is_platform_connected(platform, platform_config)
            )
            home_channel = (
                platform_config.home_channel.to_dict()
                if platform_config and platform_config.home_channel
                else None
            )
        except Exception:
            enabled = False
            configured = all(
                env_on_disk.get(key) or os.getenv(key, "")
                for key in entry["required_env"]
            )
            home_channel = None

    state = (
        runtime_platform.get("state") if isinstance(runtime_platform, dict) else None
    )
    runtime_gateway_state = runtime.get("gateway_state") if isinstance(runtime, dict) else None
    runtime_gateway_error = runtime.get("exit_reason") if isinstance(runtime, dict) else None
    if not enabled:
        state = "disabled"
    elif not configured:
        state = "not_configured"
    elif gateway_running and not state:
        state = "pending_restart"
    elif (
        not gateway_running
        and not state
        and runtime_gateway_state == "startup_failed"
    ):
        state = "startup_failed"
    elif not gateway_running and not state:
        state = "gateway_stopped"

    error_code = (
        runtime_platform.get("error_code")
        if isinstance(runtime_platform, dict)
        else None
    )
    error_message = (
        runtime_platform.get("error_message")
        if isinstance(runtime_platform, dict)
        else None
    )
    if state == "startup_failed":
        error_code = error_code or "startup_failed"
        error_message = error_message or runtime_gateway_error

    whatsapp_setup = None
    if platform_id == "whatsapp":
        whatsapp_mode = (
            env_on_disk.get("WHATSAPP_MODE")
            or ("" if scoped else os.getenv("WHATSAPP_MODE", ""))
        ).strip()
        allowed_users_value = (
            env_on_disk.get("WHATSAPP_ALLOWED_USERS")
            or ("" if scoped else os.getenv("WHATSAPP_ALLOWED_USERS", ""))
        ).strip()
        whatsapp_setup = {
            "mode": whatsapp_mode if whatsapp_mode in {"bot", "self-chat"} else "",
            "allowed_users_set": bool(allowed_users_value),
            "home_channel_set": bool(home_channel),
        }

    payload = {
        "id": platform_id,
        "name": entry["name"],
        "description": entry["description"],
        "docs_url": entry["docs_url"],
        "enabled": enabled,
        "configured": configured,
        "gateway_running": gateway_running,
        "state": state,
        "error_code": error_code,
        "error_message": error_message,
        "updated_at": (
            runtime_platform.get("updated_at")
            if isinstance(runtime_platform, dict)
            else None
        ),
        "home_channel": home_channel,
        "env_vars": env_vars,
    }
    if whatsapp_setup is not None:
        payload["whatsapp_setup"] = whatsapp_setup
    return payload


_WHATSAPP_ONBOARDING_TTL_SECONDS = 600


_WHATSAPP_ONBOARDING_TERMINAL_STATUSES = {"connected", "error", "expired", "cancelled"}


_whatsapp_onboarding_lock = threading.RLock()


def _utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_whatsapp_onboarding_mode(value: Any) -> str:
    mode = str(value or "bot").strip().lower()
    if mode not in {"bot", "self-chat"}:
        raise HTTPException(status_code=400, detail="WhatsApp mode must be 'bot' or 'self-chat'.")
    return mode


def _normalize_whatsapp_allowed_users(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return ",".join(part.replace(" ", "") for part in raw.split(",") if part.strip())


def _whatsapp_phone_from_identifier(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = raw.split("@", 1)[0].split(":", 1)[0]
    digits = re.sub(r"\D+", "", candidate)
    return digits or None


def _whatsapp_linked_account_from_session(session_path: Path) -> tuple[str | None, str | None, str | None]:
    creds_path = session_path / "creds.json"
    try:
        payload = json.loads(creds_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    account_id: str | None = None
    account_name: str | None = None

    def collect(candidate: Any) -> None:
        nonlocal account_id, account_name
        if not isinstance(candidate, dict):
            return
        if account_id is None:
            for key in ("id", "jid", "lid"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_id = value
                    break
        if account_name is None:
            for key in ("name", "verifiedName", "notify", "pushName"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_name = value
                    break

    collect(payload.get("me"))
    collect(payload.get("account"))
    collect(payload)
    return account_id, account_name, _whatsapp_phone_from_identifier(account_id)


def _ensure_whatsapp_bridge_dependencies(bridge_dir: Path) -> None:
    """Install bridge dependencies when the dashboard is the setup surface."""
    if (bridge_dir / "node_modules").exists():
        return

    from hermes_constants import find_node_executable, with_hermes_node_path
    from utils import env_int

    npm = find_node_executable("npm")
    if not npm:
        raise HTTPException(
            status_code=500,
            detail="npm was not found. WhatsApp setup needs Node.js and npm.",
        )

    timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
    try:
        result = subprocess.run(
            [npm, "install", "--silent"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            # npm output is UTF-8; guard the Windows ANSI-code-page default
            # against undefined bytes crashing the reader thread (#52649).
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=with_hermes_node_path(),
            creationflags=windows_hide_flags(),
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500,
            detail="Installing WhatsApp bridge dependencies timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to install WhatsApp bridge dependencies: {exc}",
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-10:])
        raise HTTPException(
            status_code=500,
            detail=f"npm install failed for WhatsApp bridge: {detail or 'no output'}",
        )


def _spawn_whatsapp_pairing_process(session_path: Path, mode: str) -> subprocess.Popen:
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    from hermes_constants import find_node_executable, with_hermes_node_path

    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"WhatsApp bridge script was not found at {bridge_script}.",
        )
    node = find_node_executable("node")
    if not node:
        raise HTTPException(
            status_code=500,
            detail="Node.js was not found. WhatsApp setup needs Node.js.",
        )

    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    env = with_hermes_node_path()
    env["WHATSAPP_MODE"] = mode
    env["WHATSAPP_DM_POLICY"] = "pairing"
    return subprocess.Popen(
        [
            node,
            str(bridge_script),
            "--pair-only",
            "--pair-json",
            "--session",
            str(session_path),
        ],
        cwd=str(bridge_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        env=env,
        creationflags=windows_hide_flags(),
    )


def _terminate_whatsapp_pairing(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _watch_whatsapp_pairing(pairing_id: str, proc: subprocess.Popen) -> None:
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    try:
        stream = proc.stdout
        if stream is not None:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = str(payload.get("event") or "").strip()
                with _whatsapp_onboarding_lock:
                    record = _whatsapp_onboarding_sessions.get(pairing_id)
                    if not record or record.proc is not proc:
                        return
                    if event == "qr":
                        qr = str(payload.get("qr") or "").strip()
                        if qr:
                            record.qr_payload = qr
                            record.status = "waiting"
                            record.error = None
                    elif event == "connected":
                        user = payload.get("user")
                        if isinstance(user, dict):
                            account_id = str(user.get("id") or "").strip()
                            account_name = str(user.get("name") or "").strip()
                            record.account_id = account_id or None
                            record.account_name = account_name or None
                            record.account_phone = _whatsapp_phone_from_identifier(account_id)
                        record.status = "connected"
                        record.error = None
                    elif event == "error":
                        record.status = "error"
                        record.error = str(payload.get("error") or "WhatsApp pairing failed.")
                    elif event == "disconnected" and record.status == "starting":
                        record.status = "waiting"
        returncode = proc.wait()
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.proc is proc and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.proc is not proc:
            return
        if record.status in {"connected", "cancelled", "expired"}:
            return
        record.status = "error"
        record.error = (
            "WhatsApp pairing process exited before pairing completed."
            if returncode == 0
            else f"WhatsApp pairing process exited with code {returncode}."
        )


def _run_whatsapp_pairing(pairing_id: str, session_path: Path, mode: str) -> None:
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            return
        record.status = "installing"

    try:
        proc = _spawn_whatsapp_pairing_process(session_path, mode)
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(proc)
            return
        record.proc = proc
        record.status = "starting"

    _watch_whatsapp_pairing(pairing_id, proc)


def _prune_whatsapp_onboarding_sessions() -> None:
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    now = time.time()
    remove_ids: list[str] = []
    for pairing_id, record in _whatsapp_onboarding_sessions.items():
        if (
            record.proc is not None
            and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES
            and record.proc.poll() is not None
        ):
            record.status = "error"
            record.error = "WhatsApp pairing process exited before pairing completed."
        if record.expires_at_ts <= now and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(record.proc)
            record.status = "expired"
            record.error = "WhatsApp QR setup expired. Start a new setup."
        if record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES and record.expires_at_ts + 300 <= now:
            remove_ids.append(pairing_id)
    for pairing_id in remove_ids:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)


def _supersede_whatsapp_onboarding_sessions(session_path: Path) -> None:
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    for existing in _whatsapp_onboarding_sessions.values():
        if existing.session_path == str(session_path) and existing.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            existing.status = "cancelled"
            existing.error = "Superseded by a newer WhatsApp setup session."
            _terminate_whatsapp_pairing(existing.proc)


@router.post("/api/messaging/whatsapp/onboarding/start")
async def start_whatsapp_onboarding(body: WhatsAppOnboardingStart):
    from hermes_cli.web_server import _WhatsAppOnboardingSession, _whatsapp_onboarding_sessions
    mode = _normalize_whatsapp_onboarding_mode(body.mode)
    allowed_users = _normalize_whatsapp_allowed_users(body.allowed_users)
    effective_profile = body.profile

    with _config_profile_scope(effective_profile):
        session_path = _whatsapp_session_path()
        expires_at_ts = time.time() + _WHATSAPP_ONBOARDING_TTL_SECONDS
        expires_at = _utc_iso_from_ts(expires_at_ts)
        if (session_path / "creds.json").exists():
            pairing_id = secrets.token_urlsafe(16)
            account_id, account_name, account_phone = _whatsapp_linked_account_from_session(session_path)
            record = _WhatsAppOnboardingSession(
                proc=None,
                mode=mode,
                allowed_users=allowed_users,
                session_path=str(session_path),
                expires_at=expires_at,
                expires_at_ts=expires_at_ts,
                profile=effective_profile,
                status="connected",
                account_id=account_id,
                account_name=account_name,
                account_phone=account_phone,
            )
            with _whatsapp_onboarding_lock:
                _prune_whatsapp_onboarding_sessions()
                _supersede_whatsapp_onboarding_sessions(session_path)
                _whatsapp_onboarding_sessions[pairing_id] = record
            return _whatsapp_onboarding_payload(pairing_id, record)

    pairing_id = secrets.token_urlsafe(16)
    record = _WhatsAppOnboardingSession(
        proc=None,
        mode=mode,
        allowed_users=allowed_users,
        session_path=str(session_path),
        expires_at=expires_at,
        expires_at_ts=expires_at_ts,
        profile=effective_profile,
    )

    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        _supersede_whatsapp_onboarding_sessions(session_path)
        _whatsapp_onboarding_sessions[pairing_id] = record

    threading.Thread(
        target=_run_whatsapp_pairing,
        args=(pairing_id, session_path, mode),
        daemon=True,
    ).start()

    return _whatsapp_onboarding_payload(pairing_id, record)


@router.get("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def get_whatsapp_onboarding_status(pairing_id: str):
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status == "expired":
            raise HTTPException(status_code=410, detail=record.error or "WhatsApp setup expired.")
        return _whatsapp_onboarding_payload(pairing_id, record)


@router.post("/api/messaging/whatsapp/onboarding/{pairing_id}/apply")
async def apply_whatsapp_onboarding(
    pairing_id: str, body: WhatsAppOnboardingApply, profile: Optional[str] = None
):
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status != "connected":
            raise HTTPException(status_code=409, detail="WhatsApp setup is not connected yet.")
        mode = _normalize_whatsapp_onboarding_mode(body.mode or record.mode)
        allowed_users = _normalize_whatsapp_allowed_users(
            record.allowed_users if body.allowed_users is None else body.allowed_users
        )
        if mode == "self-chat" and not allowed_users:
            allowed_users = record.account_phone or record.account_id or ""
        record_profile = record.profile

    effective_profile = body.profile or profile or record_profile
    try:
        with _config_profile_scope(effective_profile):
            save_env_value("WHATSAPP_MODE", mode)
            save_env_value("WHATSAPP_DM_POLICY", "pairing")
            if allowed_users:
                save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users)
            # Blank means "keep the existing allowlist"; explicit clearing
            # still lives in the normal config editor where the field is visible.
            save_env_value("WHATSAPP_ENABLED", "true")
            _write_platform_enabled("whatsapp", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("WhatsApp onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save WhatsApp setup.",
        ) from exc

    with _whatsapp_onboarding_lock:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)

    restart_result = _restart_gateway_after_whatsapp_onboarding(effective_profile)
    return {
        "ok": True,
        "platform": "whatsapp",
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@router.delete("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def cancel_whatsapp_onboarding(pairing_id: str):
    from hermes_cli.web_server import _whatsapp_onboarding_sessions
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.pop(pairing_id, None)
    if record:
        record.status = "cancelled"
        _terminate_whatsapp_pairing(record.proc)
    return {"ok": True}


def _parse_expiry_ts(value: str) -> float:
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return time.time() + 600


def _prune_telegram_onboarding_pairings() -> None:
    from hermes_cli.web_server import _telegram_onboarding_pairings
    now = time.time()
    expired = [
        pairing_id
        for pairing_id, record in _telegram_onboarding_pairings.items()
        if record.expires_at_ts <= now
    ]
    for pairing_id in expired:
        _telegram_onboarding_pairings.pop(pairing_id, None)


def _normalize_telegram_user_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if _TELEGRAM_USER_ID_RE.fullmatch(normalized):
        return normalized
    return None


async def _telegram_onboarding_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _telegram_onboarding_request_sync,
        method,
        path,
        body=body,
        bearer_token=bearer_token,
    )


@router.post("/api/messaging/telegram/onboarding/start")
async def start_telegram_onboarding(body: TelegramOnboardingStart):
    from hermes_cli.web_server import (
        _TelegramOnboardingPairing,
        _telegram_onboarding_lock,
        _telegram_onboarding_pairings,
    )
    bot_name = (body.bot_name or "Hermes Agent").strip() or "Hermes Agent"
    payload = await _telegram_onboarding_request(
        "POST",
        "/v1/telegram/pairings",
        body={"bot_name": bot_name},
    )

    pairing_id = str(payload.get("pairing_id") or "").strip()
    poll_token = str(payload.get("poll_token") or "").strip()
    expires_at = str(payload.get("expires_at") or "").strip()
    deep_link = str(payload.get("deep_link") or "").strip()
    qr_payload = str(payload.get("qr_payload") or deep_link).strip()
    suggested_username = str(payload.get("suggested_username") or "").strip()
    if not pairing_id or not poll_token or not expires_at or not deep_link:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an incomplete response.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        _telegram_onboarding_pairings[pairing_id] = _TelegramOnboardingPairing(
            poll_token=poll_token,
            expires_at=expires_at,
            expires_at_ts=_parse_expiry_ts(expires_at),
        )

    return {
        "pairing_id": pairing_id,
        "suggested_username": suggested_username,
        "deep_link": deep_link,
        "qr_payload": qr_payload,
        "expires_at": expires_at,
    }


@router.get("/api/messaging/telegram/onboarding/{pairing_id}")
async def get_telegram_onboarding_status(pairing_id: str):
    from hermes_cli.web_server import _telegram_onboarding_lock, _telegram_onboarding_pairings
    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        if record.bot_token:
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }
        poll_token = record.poll_token

    payload = await _telegram_onboarding_request(
        "GET",
        f"/v1/telegram/pairings/{urllib.parse.quote(pairing_id, safe='')}",
        bearer_token=poll_token,
    )
    status = str(payload.get("status") or "").strip()
    if status == "waiting":
        with _telegram_onboarding_lock:
            current = _telegram_onboarding_pairings.get(pairing_id)
            expires_at = current.expires_at if current else ""
        return {"status": "waiting", "expires_at": expires_at}

    if status == "ready":
        bot_token = str(payload.get("token") or "").strip()
        bot_username = str(payload.get("bot_username") or "").strip()
        if not bot_token:
            raise HTTPException(
                status_code=502,
                detail="Telegram setup service returned an incomplete response.",
            )
        owner_user_id = _normalize_telegram_user_id(payload.get("owner_user_id"))
        with _telegram_onboarding_lock:
            record = _telegram_onboarding_pairings.get(pairing_id)
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail="Telegram setup session was not found. Start a new setup.",
                )
            record.bot_token = bot_token
            record.bot_username = bot_username or None
            record.owner_user_id = owner_user_id
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }

    if status in {"expired", "claimed"}:
        with _telegram_onboarding_lock:
            _telegram_onboarding_pairings.pop(pairing_id, None)
        raise HTTPException(
            status_code=410,
            detail=_telegram_onboarding_error_message(
                status,
                "Telegram setup is no longer available. Start a new setup.",
            ),
        )

    raise HTTPException(
        status_code=502,
        detail="Telegram setup service returned an unknown status.",
    )


def _restart_gateway_after_telegram_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    """Best-effort gateway restart after saving Telegram QR onboarding.

    The QR flow naturally pulls users into Telegram on another device. If the
    saved token waits on a separate dashboard restart click, Hermes appears
    broken from the chat side. Keep the config save authoritative, but report
    restart failures so the UI can fall back to the existing manual banner.
    """
    return _restart_gateway_after(profile, what="Telegram onboarding", label="Telegram onboarding")


@router.post("/api/messaging/telegram/onboarding/{pairing_id}/apply")
async def apply_telegram_onboarding(
    pairing_id: str, body: TelegramOnboardingApply, profile: Optional[str] = None
):
    from hermes_cli.web_server import _telegram_onboarding_lock, _telegram_onboarding_pairings
    allowed_user_ids = []
    seen = set()
    for raw_id in body.allowed_user_ids:
        normalized = _normalize_telegram_user_id(raw_id)
        if not normalized:
            raise HTTPException(
                status_code=400,
                detail="Allowed Telegram user IDs must be numeric.",
            )
        if normalized not in seen:
            seen.add(normalized)
            allowed_user_ids.append(normalized)
    if not allowed_user_ids:
        raise HTTPException(
            status_code=400,
            detail="Add at least one allowed Telegram user ID.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        bot_token = record.bot_token
        bot_username = record.bot_username
        if not bot_token:
            raise HTTPException(
                status_code=409,
                detail="Telegram setup is not ready yet.",
            )

    effective_profile = body.profile or profile

    def _apply():
        with _profile_scope(effective_profile):
            save_env_value("TELEGRAM_BOT_TOKEN", bot_token)
            save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(allowed_user_ids))
            _write_platform_enabled("telegram", True)

    try:
        await asyncio.to_thread(_apply)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Telegram onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save Telegram setup.",
        ) from exc

    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)

    restart_result = _restart_gateway_after_telegram_onboarding(effective_profile)

    return {
        "ok": True,
        "platform": "telegram",
        "bot_username": bot_username,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@router.delete("/api/messaging/telegram/onboarding/{pairing_id}")
async def cancel_telegram_onboarding(pairing_id: str):
    from hermes_cli.web_server import _telegram_onboarding_lock, _telegram_onboarding_pairings
    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)
    return {"ok": True}


@router.get("/api/messaging/platforms")
async def get_messaging_platforms(profile: Optional[str] = None):
    # Profile-scoped so the dashboard's global profile switcher shows the
    # TARGET profile's channel credentials/state, not the root install's.
    # load_env() honors the HERMES_HOME contextvar override; the gateway
    # status readers do NOT (they resolve process-level paths), so the
    # profile directory is passed explicitly for those (#71211).
    def _run():
        with _profile_scope(profile) as scoped_dir:
            env_on_disk = load_env()
            runtime = (
                read_runtime_status(path=scoped_dir / "gateway_state.json")
                if scoped_dir is not None
                else read_runtime_status()
            )
            return {
                "env_path": str(get_env_path()),
                "gateway_start_command": _gateway_display_command(profile, "start"),
                "platforms": [
                    _messaging_platform_payload(
                        entry,
                        env_on_disk,
                        runtime,
                        scoped=scoped_dir is not None,
                        profile_home=scoped_dir,
                    )
                    for entry in _messaging_platform_catalog()
                ]
            }

    return await asyncio.to_thread(_run)


def _multiplex_port_binding_conflict(
    platform_id: str, requested_profile: Optional[str]
) -> Optional[str]:
    """Reason enabling ``platform_id`` on the target profile would break a
    multiplexed gateway, or ``None`` when the change is allowed.

    Mirrors the gateway's startup rule (``_start_one_profile_adapters`` in
    gateway/run.py): with ``gateway.multiplex_profiles`` on, the default
    profile owns the single shared HTTP listener and serves every profile via
    the ``/p/<profile>/`` prefix, so a SECONDARY profile must never enable a
    port-binding platform. Without this pre-write check the dashboard happily
    persisted the invalid config and the shared gateway died with
    ``MultiplexConfigError`` on its next start — for ALL profiles. Only
    *enabling* is blocked; disabling/clearing stays allowed so users can
    repair an already-invalid profile.
    """
    from gateway.config import PORT_BINDING_PLATFORM_VALUES, load_gateway_config

    if platform_id not in PORT_BINDING_PLATFORM_VALUES:
        return None

    requested = (requested_profile or "").strip()
    if not requested or requested.lower() == "current":
        from hermes_cli.profiles import get_active_profile_name

        # The dashboard's own profile. "custom" (an unrecognized HERMES_HOME)
        # is outside the profiles tree, so a multiplexed gateway never serves
        # it — nothing to guard.
        target = get_active_profile_name()
    else:
        _resolve_profile_dir(requested)  # same 400/404 as _profile_scope
        target = requested
    if target in ("default", "custom"):
        return None

    # The multiplex flag that matters is the one the shared gateway reads at
    # startup: the DEFAULT profile's gateway config (plus the process-wide
    # GATEWAY_MULTIPLEX_PROFILES override, which load_gateway_config applies).
    with _config_profile_scope("default"):
        if not load_gateway_config().multiplex_profiles:
            return None

    return (
        f"Cannot enable '{platform_id}' on profile '{target}': it binds its "
        "own listener port, and gateway.multiplex_profiles is on, so the "
        "default profile owns the single shared HTTP listener for every "
        "profile. Configure this channel on the default profile instead "
        "(disabling or clearing it here is still allowed)."
    )


@router.put("/api/messaging/platforms/{platform_id}")
async def update_messaging_platform(
    platform_id: str, body: MessagingPlatformUpdate, profile: Optional[str] = None
):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    target_profile = body.profile or profile
    if body.enabled:
        conflict = _multiplex_port_binding_conflict(platform_id, target_profile)
        if conflict:
            # Reject BEFORE any .env/config.yaml write so the profile stays
            # loadable by the multiplexed gateway.
            _log.info(
                "Rejected messaging platform update: platform=%s profile=%s "
                "(multiplex port-binding conflict)",
                platform_id,
                target_profile or "current",
            )
            raise HTTPException(status_code=409, detail=conflict)

    allowed_env = set(entry["env_vars"])

    def _apply():
        with _profile_scope(body.profile or profile):
            for key in body.clear_env:
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                remove_env_value(key)

            for key, value in body.env.items():
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                trimmed = value.strip()
                if trimmed:
                    _validate_messaging_env_value(platform_id, key, trimmed)
                    save_env_value(key, trimmed)

            if body.enabled is not None:
                _write_platform_enabled(platform_id, body.enabled)

    try:
        await asyncio.to_thread(_apply)

        # Audit trail for channel config mutations: names only, never values.
        _log.info(
            "Messaging platform updated: platform=%s profile=%s enabled=%s "
            "env_keys=%s cleared_keys=%s",
            platform_id,
            target_profile or "current",
            body.enabled,
            sorted(body.env),
            sorted(body.clear_env),
        )
        return {"ok": True, "platform": platform_id}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/messaging/platforms/%s failed", platform_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/api/messaging/platforms/{platform_id}/test")
async def test_messaging_platform(platform_id: str, profile: Optional[str] = None):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    def _run():
        with _profile_scope(profile) as scoped_dir:
            env_on_disk = load_env()
            runtime = (
                read_runtime_status(path=scoped_dir / "gateway_state.json")
                if scoped_dir is not None
                else read_runtime_status()
            )
            return _messaging_platform_payload(
                entry,
                env_on_disk,
                runtime,
                scoped=scoped_dir is not None,
                profile_home=scoped_dir,
            )

    payload = await asyncio.to_thread(_run)
    if not payload["enabled"]:
        message = f"{entry['name']} is disabled. Enable it, then restart the gateway."
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["configured"]:
        missing = [
            field["key"]
            for field in payload["env_vars"]
            if field["required"] and not field["is_set"]
        ]
        message = (
            f"Missing required setup: {', '.join(missing)}"
            if missing
            else "Platform setup is incomplete."
        )
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["gateway_running"]:
        return {
            "ok": False,
            "state": payload["state"],
            "message": "Gateway is not running. Restart the gateway to connect this platform.",
        }
    if payload["state"] == "connected":
        return {
            "ok": True,
            "state": payload["state"],
            "message": f"{entry['name']} is connected.",
        }
    if payload.get("error_message"):
        return {
            "ok": False,
            "state": payload["state"],
            "message": payload["error_message"],
        }
    return {
        "ok": False,
        "state": payload["state"],
        "message": "Setup looks complete, but the gateway has not reported a connection yet. Restart the gateway.",
    }
