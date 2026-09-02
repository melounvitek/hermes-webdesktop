"""Shared helpers for tool backend selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)


_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct", "managed"}


def managed_nous_tools_enabled(*, force_fresh: bool = False) -> bool:
    """True when the user is entitled to the Nous Tool Gateway (coarse gate).

    Fails closed on unknown/error entitlement — never blocks startup.
    Per-category coverage is narrowed by callers via ``tool_gateway_entitled_for``.
    ``force_fresh=True`` is for interactive flows that must see a just-purchased grant.
    """
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        if force_fresh:
            account_info = get_nous_portal_account_info(force_fresh=True)
        else:
            account_info = get_nous_portal_account_info()
        return bool(account_info.logged_in) and account_info.tool_gateway_entitled
    except Exception:
        return False


def nous_tool_gateway_unavailable_message(
    capability: str = "the Nous Tool Gateway",
    *,
    force_fresh: bool = False,
) -> str:
    """Return account-aware guidance for an unavailable Nous Tool Gateway path."""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=force_fresh)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        if message:
            return message
    except Exception:
        pass
    return (
        f"{capability} is unavailable. Run `hermes model` to refresh your "
        "Nous Portal login and billing status."
    )


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def coerce_modal_mode(value: object | None) -> str:
    """Return the requested modal mode when valid, else the default."""
    mode = str(value or _DEFAULT_MODAL_MODE).strip().lower()
    return mode if mode in _VALID_MODAL_MODES else _DEFAULT_MODAL_MODE


def normalize_modal_mode(value: object | None) -> str:
    """Return a normalized modal execution mode."""
    return coerce_modal_mode(value)


def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    try:
        modal_file_exists = (Path.home() / ".modal.toml").exists()
    except (PermissionError, OSError):
        modal_file_exists = False
    return bool(
        (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
        or modal_file_exists
    )


def resolve_modal_backend_state(
    modal_mode: object | None,
    *,
    has_direct: bool,
    managed_ready: bool,
    managed_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend: ``direct``/``managed`` are
    exclusive; ``auto`` prefers managed when available, else direct."""
    requested_mode = coerce_modal_mode(modal_mode)
    if managed_enabled is None:
        managed_enabled = managed_nous_tools_enabled()
    managed_mode_blocked = requested_mode == "managed" and not managed_enabled

    managed_ok = managed_enabled and managed_ready
    if requested_mode == "managed":
        selected_backend = "managed" if managed_ok else None
    elif requested_mode == "direct":
        selected_backend = "direct" if has_direct else None
    else:
        selected_backend = "managed" if managed_ok else "direct" if has_direct else None

    return {
        "requested_mode": requested_mode,
        "mode": requested_mode,
        "has_direct": has_direct,
        "managed_ready": managed_ready,
        "managed_mode_blocked": managed_mode_blocked,
        "selected_backend": selected_backend,
    }


def _scoped_credential(name: str) -> str:
    """Read a credential env var under the active profile secret scope."""
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name, "") or "").strip()
    except Exception:  # pragma: no cover — secret_scope is in-repo
        return (os.getenv(name, "") or "").strip()


def resolve_provider_secret(
    env_var: str,
    provider_id: str,
    config_value: str = "",
    env_getter=None,
) -> str:
    """Resolve a voice-provider API key (single owner for STT/TTS lookup).

    Order: explicit ``config_value`` -> profile secret scope / env -> ``.env``
    via ``env_getter`` (or ``hermes_cli.config.get_env_value``) -> credential
    pool for ``provider_id``. Under an active multiplex turn the profile scope
    is authoritative: a miss returns ``""`` rather than borrowing another
    profile's env or pool. Never raises.
    """
    value = str(config_value or "").strip()
    if value:
        return value

    key = _scoped_credential(env_var)
    if key:
        return key

    try:
        from agent.secret_scope import is_multiplex_active

        if is_multiplex_active():
            return ""
    except Exception:  # pragma: no cover — secret_scope is in-repo
        pass

    if env_getter is not None:
        key = str(env_getter(env_var) or "").strip()
    else:
        try:
            from hermes_cli.config import get_env_value

            key = str(get_env_value(env_var) or "").strip()
        except ImportError:  # pragma: no cover — config is in-repo
            key = ""
    if key:
        return key

    if not provider_id:
        return ""
    try:
        from agent.credential_pool import load_pool

        # config.yaml ``providers.<name>`` entries are pooled under ``custom:<name>``.
        for pool_key in (provider_id, f"custom:{provider_id}"):
            pool = load_pool(pool_key)
            if pool is None or not pool.has_credentials():
                continue
            entry = pool.peek()
            if entry is None:
                continue
            key = str(
                getattr(entry, "runtime_api_key", "")
                or getattr(entry, "access_token", "")
                or ""
            ).strip()
            if key:
                return key
    except Exception as exc:
        logger.debug("Could not read %s credential pool for %s: %s", provider_id, env_var, exc)
    return ""


def resolve_openai_audio_api_key() -> str:
    """Prefer VOICE_TOOLS_OPENAI_KEY, else OPENAI_API_KEY (scope-aware, with
    credential-pool fallback for the latter). Must go through the secret scope:
    a raw ``os.environ`` read could bill another profile's account under multiplex.
    """
    return (
        resolve_provider_secret("VOICE_TOOLS_OPENAI_KEY", "")
        or resolve_provider_secret("OPENAI_API_KEY", "openai-api")
    )


def prefers_gateway(config_section: str) -> bool:
    """Return True when the user opted into the Tool Gateway for this tool.

    Reads ``<section>.use_gateway`` from config.yaml.  Never raises.
    """
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get(config_section)
        if isinstance(section, dict):
            return is_truthy_value(section.get("use_gateway"), default=False)
    except Exception:
        pass
    return False


# Provider value the managed "Nous Subscription" picker rows write for every
# category; any other name = that vendor direct; no key = legacy autodetect.
NOUS_MANAGED_PROVIDER = "nous"

# Per-capability keys that also count as "this category has been configured".
_EXTRA_SELECTION_KEYS = {
    "web": ("search_backend", "extract_backend"),
}

# Key(s) carrying the category's provider selection. ``browser.backend`` is the
# DRIVER choice (browser-use CLI vs built-in), not the cloud provider — excluded.
_SELECTION_NAME_KEYS = {
    "browser": ("cloud_provider",),
    "web": ("backend",),
}
_DEFAULT_NAME_KEYS = ("provider", "backend", "cloud_provider")


def _raw_section(section: str) -> Dict[str, Any] | None:
    """The RAW (unmerged) config.yaml mapping for ``section``, or None."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        cfg = read_raw_config_readonly() or {}
        raw = cfg.get(section) if isinstance(cfg, dict) else None
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def read_selection(section: str) -> str | None:
    """THE single runtime read of the persisted `hermes tools` selection.

    Returns ``"nous"`` (managed gateway row), a vendor name (direct, own
    credentials), or ``None`` (never configured -> legacy autodetect allowed).
    Reads the RAW config.yaml so key presence means "actually written", not
    "schema default"; a raw ``local`` is therefore a real user selection.
    Legacy shim: ``use_gateway: true`` was only ever written by the managed
    row, so it maps to ``"nous"`` regardless of the name key. Never raises.
    """
    raw = _raw_section(section)
    if raw is None:
        return None

    if "use_gateway" in raw and is_truthy_value(raw.get("use_gateway"), default=False):
        return NOUS_MANAGED_PROVIDER

    for key in _SELECTION_NAME_KEYS.get(section, _DEFAULT_NAME_KEYS):
        value = raw.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    return None


def selection_exists(section: str) -> bool:
    """True when ANY selection signal was ever written for the section
    (wider than read_selection: per-capability web keys count too)."""
    if read_selection(section) is not None:
        return True
    extra = _EXTRA_SELECTION_KEYS.get(section, ())
    raw = _raw_section(section) if extra else None
    if raw is None:
        return False
    return any(str(raw.get(key) or "").strip() for key in extra)


# Backends that once shipped in-tree but were removed; consulted by the startup
# config check and selection_error() so a stale selection gets a real message.
# Add removals here, never as one-off string checks at call sites, e.g.
#   "web": {"<name>": "the <Name> backend was removed in vX.Y.Z (...)"},
REMOVED_BACKENDS: Dict[str, Dict[str, str]] = {}


def removed_backend_note(section: str, name: str) -> Optional[str]:
    """Explanation for a backend that used to ship in-tree, or None.

    ``name`` tolerates the quoted form callers pass to selection_error().
    """
    normalized = (name or "").strip().strip("'\"").lower()
    return REMOVED_BACKENDS.get(section, {}).get(normalized)


def selection_error(section: str, selection_name: str, failure: str) -> str:
    """The uniform honest-error contract for a selected-but-broken provider."""
    note = removed_backend_note(section, selection_name)
    if note:
        failure = note
    return (
        f"{section} is configured to use {selection_name} (set via hermes "
        f"tools), but {failure}. Run 'hermes tools' to change it."
    )


def fal_key_is_configured() -> bool:
    """True when FAL_KEY is set (scope/env, else ``.env`` for CLI paths that
    run before dotenv loads) to a non-whitespace value."""
    value = _scoped_credential("FAL_KEY") or None
    if value is None:
        try:
            from hermes_cli.config import get_env_value

            value = get_env_value("FAL_KEY")
        except Exception:
            value = None
    return bool(value and value.strip())
