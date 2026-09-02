"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process.  Each profile
has its own ``.env`` with its own keys, so they **cannot** be unioned into the
process-global ``os.environ`` (profile A's keys would leak into profile B's
turns and into every subprocess spawned with ``env=dict(os.environ)``).

This module is a fail-closed, context-local secret scope:

- ``set_secret_scope(mapping)`` installs the active profile's secrets for the
  current task (a contextvar, so it propagates into the agent's worker thread
  via ``copy_context()`` exactly like the HERMES_HOME override).
- ``get_secret(name)`` reads from that scope.  When multiplexing is **active**
  and no scope is set it RAISES rather than falling back to ``os.environ`` —
  an un-migrated call site fails loud at that line instead of leaking another
  profile's value.  When multiplexing is **off** (default) it reads
  ``os.environ`` so every non-multiplex caller behaves exactly as before.

Design rationale: ``docs/design/multiplexing-gateway.md`` (Workstream A).
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Dict, Mapping, Optional


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed)."""
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    return _MULTIPLEX_ACTIVE


_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar(
    "_SECRET_SCOPE", default=None
)


class UnscopedSecretError(RuntimeError):
    """A secret was read in multiplex mode with no scope installed.

    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the global allowlist.
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """Install the active profile's secret mapping; ``None`` clears.  Returns a reset token."""
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """The active secret mapping, or None when no scope is installed."""
    return _SECRET_SCOPE.get()


# Genuinely-global env vars: process/deployment settings, NOT profile secrets.
# They keep reading os.environ even in multiplex mode (routing them through the
# fail-closed path would wrongly crash).  Keep this tight — when in doubt a
# value is a profile secret.  Membership is exact name OR prefix.
_GLOBAL_ENV_EXACT = frozenset({
    # Hermes runtime / deployment
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_GATEWAY_LOCK_DIR",
    "HERMES_MAX_ITERATIONS", "HERMES_MAX_TOKENS", "HERMES_API_TIMEOUT",
    "HERMES_REDACT_SECRETS", "HERMES_NOUS_TIMEOUT_SECONDS",
    "_HERMES_GATEWAY",
    # OS / interpreter
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "PWD", "SHELL", "TMPDIR",
    "VIRTUAL_ENV", "PYTHONPATH", "SSL_CERT_FILE",
    # Kanban paths (per-board, not per-profile-secret)
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD",
    # API-server LISTENER settings — deployment config (compose/systemd env),
    # which the scoped runner reload must keep seeing or containers silently
    # lose the api_server platform.  API_SERVER_KEY is a credential: NOT here.
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
    # Relay-connector ROUTING stamps injected by managed deploys.  Every reader
    # (gateway.config, relay_url()/registration/self-provision) must resolve
    # the SAME value or the adapter registers while the platform is absent
    # from config.  GATEWAY_RELAY_SECRET/_ID/_DELIVERY_KEY and IDP_* are auth
    # material and deliberately stay profile-scoped.
    "GATEWAY_RELAY_URL", "GATEWAY_RELAY_ENDPOINT",
    "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS",
    "GATEWAY_RELAY_PLATFORMS", "GATEWAY_RELAY_BOT_IDS",
    "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
    "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_DISPLAY_NAME",
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # tuning knobs (batch delays, fallback toggles) — NOT the token
    "TERMINAL_",          # terminal/sandbox backend settings
)


def _is_global_env(name: str) -> bool:
    """True for genuinely process-global (non-profile-secret) env vars."""
    return name in _GLOBAL_ENV_EXACT or name.startswith(_GLOBAL_ENV_PREFIXES)


def _environ_or(name: str, default: Optional[str]) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    1. Global vars (``_is_global_env``) always read ``os.environ``.
    2. Scope installed: read from it.  Under multiplexing the scope is
       authoritative (a miss returns ``default``, never ``os.environ``, which
       may hold another profile's value).  With multiplexing OFF a miss falls
       through to ``os.environ``: single-profile deployments legitimately
       inject credentials via the process env (systemd, ``op run``, shell
       exports) and the scope — installed around e.g. every cron job — must
       stay a ``.env`` overlay, not a blindfold (otherwise cron 401s).
    3. No scope: multiplex INACTIVE reads ``os.environ`` (legacy behavior);
       ACTIVE raises ``UnscopedSecretError`` (fail closed).
    """
    if _is_global_env(name):
        return _environ_or(name, default)

    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        if val is not None:
            return val
        if _MULTIPLEX_ACTIVE:
            return default
        return _environ_or(name, default)

    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )

    return _environ_or(name, default)


def _strip_inline_comment(value: str) -> str:
    """Strip a dotenv-style inline comment from a raw ``.env`` value.

    Mirrors python-dotenv semantics: for quoted values scan to the matching
    close quote (backslash-escape-aware for double quotes, since
    ``save_env_value`` writes ``\\"``/``\\\\``) and drop a trailing ``# ...``;
    other trailing junk (or an unterminated quote) leaves the value untouched.
    Unquoted values truncate only at a ``#`` PRECEDED BY WHITESPACE, so
    ``foo#bar`` and a leading ``#`` survive while ``value # comment`` → ``value``.
    """
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2  # skip the escaped character
                continue
            if ch == quote:
                remainder = value[i + 1:].lstrip()
                if remainder.startswith("#"):
                    return value[: i + 1]
                return value
            i += 1
        return value  # unterminated quote: leave as-is
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a dict WITHOUT touching ``os.environ``.

    Handles the subset Hermes writes (``export`` prefix, full-line and inline
    ``#`` comments, quotes with the writer's escapes reversed via the canonical
    ``_parse_env_value`` — stripping only outer quotes would corrupt credentials
    containing ``"`` or ``\\``).  ``utf-8-sig`` so a Windows BOM doesn't prefix
    the first key as ``\\ufeffNAME``.
    """
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    from hermes_cli.config import _parse_env_value

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        secrets[key] = _parse_env_value(_strip_inline_comment(value))

    return secrets


def build_profile_secret_scope(hermes_home: Path) -> Dict[str, str]:
    """Build a profile's secret mapping from ``<home>/.env`` plus its external
    secret sources.  Global vars are NOT copied in — ``get_secret`` reads those
    from ``os.environ`` — so the scope holds only profile secrets."""
    home = Path(hermes_home)
    secrets = load_env_file(home / ".env")

    try:
        from hermes_cli.env_loader import get_secret_source_values
        external_secrets = get_secret_source_values(home)
    except Exception:
        external_secrets = {}

    for key, value in external_secrets.items():
        if not _is_global_env(key):
            secrets[key] = value

    return secrets
