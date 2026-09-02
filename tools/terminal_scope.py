"""Per-turn terminal scope: profile-scoped TERMINAL_* policy.

Multiplexed surfaces (gateway, dashboard/TUI, cron) serve several profiles
from one process; mirroring terminal settings into ``os.environ`` let the
first profile pin its backend onto everyone else (sandbox escape). Like
``agent/secret_scope.py`` for credentials, a ContextVar holds the active
profile's COMPLETE effective ``TERMINAL_*`` policy, installed at each
in-process profile boundary.

- **Authoritative projection.** While a scope is bound, ``terminal_env``
  resolves ONLY from the policy (defaults + profile ``.env`` + its
  ``config.yaml``); omitted keys yield the defined default, never ambient
  ``os.environ`` (#68559).
- **Fail closed.** If the policy cannot be resolved, callers install a
  *refusal* scope and terminal execution under it raises
  :class:`TerminalPolicyUnavailable` instead of falling back to ambient
  authority.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# None = no scope bound (historical process-env behavior); dict = the active
# profile's complete policy; TerminalPolicyRefusal = resolution failed.
_terminal_scope_var: ContextVar = ContextVar("hermes_terminal_scope", default=None)

# Terminal keys whose config default lives in the consuming tool
# (terminal_tool.py) rather than DEFAULT_CONFIG; DEFAULT_CONFIG wins on overlap.
_TOOL_LEVEL_DEFAULTS: Dict[str, Any] = {
    "cwd": ".",
    "ssh_host": "",
    "ssh_user": "",
    "ssh_port": 22,
    "ssh_key": "",
    "docker_orphan_reaper": True,
    "docker_persist_across_processes": True,
    "sandbox_dir": "",
    "lifetime_seconds": 300,
    "docker_shared_container_key": "",
    "home_mode": "auto",
}


class TerminalPolicyUnavailable(Exception):
    """The routed profile's ``.env``/``config.yaml`` exists but cannot be read/parsed."""


class TerminalPolicyRefusal(Dict[str, str]):
    """Marker scope (empty dict subclass) installed when policy resolution failed."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


def set_terminal_scope(mapping: Optional[Dict[str, str]]) -> Token:
    """Install *mapping* as the current context's terminal policy."""
    return _terminal_scope_var.set(mapping)


def install_refusal_scope(reason: str) -> Token:
    """Install a refusal scope; terminal execution under it is rejected."""
    return _terminal_scope_var.set(TerminalPolicyRefusal(reason))


def reset_terminal_scope(token: Token) -> None:
    _terminal_scope_var.reset(token)


def get_terminal_scope() -> Optional[Dict[str, str]]:
    """The active scope mapping/refusal, or ``None`` when no scope is bound."""
    return _terminal_scope_var.get()


def _raise_if_refusal(scope: Any) -> None:
    if isinstance(scope, TerminalPolicyRefusal):
        raise TerminalPolicyUnavailable(
            f"terminal policy unavailable for this profile: {scope.reason}"
        )


def terminal_env(name: str, default: str = "") -> str:
    """Authoritative read of a ``TERMINAL_*`` variable.

    No scope: process env, then *default*. Refusal scope: raise. Policy
    scope: ONLY the policy; a missing key yields *default*, never os.environ.
    """
    scope = _terminal_scope_var.get()
    if scope is None:
        return os.environ.get(name, default)
    _raise_if_refusal(scope)
    value = scope.get(name)
    return default if value is None else str(value)


def build_profile_terminal_scope(hermes_home: "Any") -> Dict[str, str]:
    """Build the COMPLETE effective ``TERMINAL_*`` policy for a profile home.

    Projection: ``DEFAULT_CONFIG['terminal']`` <- profile ``.env`` TERMINAL_*
    <- profile ``config.yaml`` ``terminal:`` keys. Total by construction, so a
    bound scope never widens back to ambient process authority. Raises
    :class:`TerminalPolicyUnavailable` when either file exists but cannot be
    read/parsed.
    """
    home = Path(hermes_home)

    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG.get("terminal") if isinstance(DEFAULT_CONFIG, dict) else None
    # Without the tool-level defaults the projection is not total.
    defaults = {**_TOOL_LEVEL_DEFAULTS, **(defaults if isinstance(defaults, dict) else {})}

    scope: Dict[str, str] = {}

    def _apply(cfg_key: str, value: Any) -> None:
        # cwd placeholders are resolved per-surface later; not a policy value.
        if value is None or (cfg_key == "cwd" and str(value).strip() in {".", "auto", "cwd"}):
            return
        env_var = TERMINAL_CONFIG_ENV_MAP.get(cfg_key)
        if env_var:
            scope[env_var] = str(value)

    for cfg_key, value in defaults.items():
        _apply(cfg_key, value)

    env_path = home / ".env"
    if env_path.exists():
        # load_env_file swallows OSError by design (secret scope fails soft);
        # an unreadable profile .env must fail closed here.
        try:
            env_path.read_bytes()
        except Exception as exc:
            raise TerminalPolicyUnavailable(f"cannot read {env_path}: {exc}") from exc
        from agent.secret_scope import load_env_file

        for key, value in load_env_file(env_path).items():
            if key.startswith("TERMINAL_"):
                scope[key] = str(value)

    # Read config.yaml through the HERMES_HOME override so the profile's own
    # file is consulted; a present-but-unparseable file fails closed.
    from hermes_constants import (
        get_hermes_home_override,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    override_token = None
    if get_hermes_home_override() != str(home):
        override_token = set_hermes_home_override(home)
    try:
        config_path = home / "config.yaml"
        if config_path.exists():
            # Not read_raw_config(): it collapses "missing" and "unparseable"
            # into {}; the file exists, so a parse failure must fail closed.
            from hermes_cli.config import fast_safe_load

            try:
                with open(config_path, encoding="utf-8") as f:
                    raw = fast_safe_load(f)
            except Exception as exc:
                raise TerminalPolicyUnavailable(f"cannot parse {config_path}: {exc}") from exc
            raw_terminal = raw.get("terminal") if isinstance(raw, dict) else None
            if isinstance(raw_terminal, dict):
                for cfg_key, value in raw_terminal.items():
                    _apply(cfg_key, value)
    except TerminalPolicyUnavailable:
        raise
    except Exception as exc:
        raise TerminalPolicyUnavailable(f"cannot resolve terminal config in {home}: {exc}") from exc
    finally:
        if override_token is not None:
            reset_hermes_home_override(override_token)

    return scope


def install_profile_terminal_scope(hermes_home: "Any") -> Token:
    """Build AND install a profile's policy; on failure install the refusal scope.

    Never raises. Returns the token for ``reset_terminal_scope``.
    """
    try:
        return set_terminal_scope(build_profile_terminal_scope(hermes_home))
    except TerminalPolicyUnavailable as exc:
        logger.warning("terminal policy unavailable: %s", exc)
        return install_refusal_scope(str(exc))


def enforce_no_refusal() -> None:
    """Raise when the active scope is a refusal scope (fail closed, #68559)."""
    _raise_if_refusal(_terminal_scope_var.get())


@contextmanager
def install_and_reset_profile_terminal_scope(
    hermes_home: "Any",
) -> Iterator[None]:
    """Install the profile's terminal policy for a bounded turn/fire. Never raises."""
    token = install_profile_terminal_scope(hermes_home)
    try:
        yield
    finally:
        reset_terminal_scope(token)
