"""Environment variable passthrough registry.

Skills that declare ``required_environment_variables`` need those vars in
sandboxed execution environments (execute_code, terminal), which strip secrets
from the child process environment by default.  This module is the
session-scoped allowlist, fed by two sources: skill declarations (registered
automatically by ``skill_view``) and ``terminal.env_passthrough`` in
config.yaml.

``code_execution_tool.py`` and ``tools/environments/local.py`` consult
:func:`is_env_passthrough` before stripping a variable.  When profile
multiplexing is active, forwarded values are resolved through the current
profile's secret scope rather than the process environment.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Iterable
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

# Session-scoped allowlist; ContextVar-backed to prevent cross-session bleed
# in the gateway pipeline.
_allowed_env_vars_var: ContextVar[set[str]] = ContextVar("_allowed_env_vars")


def _get_allowed() -> set[str]:
    """Get or create the allowed env vars set for the current context/session."""
    try:
        return _allowed_env_vars_var.get()
    except LookupError:
        val: set[str] = set()
        _allowed_env_vars_var.set(val)
        return val


# Cache for the config-based allowlist (loaded once per process).
_config_passthrough: frozenset[str] | None = None


def _is_hermes_provider_credential(name: str) -> bool:
    """True if ``name`` is a Hermes-managed provider credential per
    ``_HERMES_PROVIDER_ENV_BLOCKLIST`` (or a dynamic Hermes-internal secret).

    Skill-declared ``required_environment_variables`` must not override this
    list — that was the GHSA-rhgp-j443-p4rf bypass, where a malicious skill
    registered ``OPENAI_API_KEY`` as passthrough and received it in the
    ``execute_code`` child, defeating the sandbox's scrubbing guarantee.
    Non-Hermes API keys (TENOR_API_KEY, NOTION_TOKEN, …) are not in the
    blocklist and remain registerable.

    Fail closed: if the authoritative blocklist cannot be imported (partial
    install, import-time error), treat the name as protected and refuse
    passthrough rather than fall open.
    """
    try:
        from tools.environments.local import (
            _HERMES_PROVIDER_ENV_BLOCKLIST,
            _is_hermes_internal_secret,
        )
    except Exception as e:
        logger.warning(
            "env passthrough: provider credential blocklist import failed; "
            "failing closed and refusing passthrough registration for %r: %s",
            name,
            e,
        )
        return True
    # Dynamically-generated Hermes-internal secrets (AUXILIARY_*_API_KEY /
    # _BASE_URL, GATEWAY_RELAY_*) are injected per task/relay at gateway
    # startup, so the static blocklist can't enumerate them.
    if _is_hermes_internal_secret(name):
        return True
    return name in _HERMES_PROVIDER_ENV_BLOCKLIST


def register_env_passthrough(var_names: Iterable[str]) -> None:
    """Register env var names as allowed in sandboxed environments (typically
    from a skill's ``required_environment_variables``).

    Hermes-managed provider credentials are rejected to preserve the
    ``execute_code`` sandbox's credential-scrubbing guarantee
    (GHSA-rhgp-j443-p4rf); a skill needing a Hermes-managed provider should
    use the main-process tools (web_search, web_extract, …) where the
    credential stays in the main process.  Third-party keys pass normally.
    """
    for name in var_names:
        name = name.strip()
        if not name:
            continue
        if _is_hermes_provider_credential(name):
            logger.warning(
                "env passthrough: refusing to register Hermes provider "
                "credential %r (blocked by _HERMES_PROVIDER_ENV_BLOCKLIST). "
                "Skills must not override the execute_code sandbox's "
                "credential scrubbing; see GHSA-rhgp-j443-p4rf.",
                name,
            )
            continue
        _get_allowed().add(name)
        logger.debug("env passthrough: registered %s", name)


def _load_config_passthrough() -> frozenset[str]:
    """Load ``tools.env_passthrough`` from config.yaml (cached)."""
    global _config_passthrough
    if _config_passthrough is not None:
        return _config_passthrough

    result: set[str] = set()
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        passthrough = cfg_get(cfg, "terminal", "env_passthrough")
        if isinstance(passthrough, list):
            for item in passthrough:
                if not isinstance(item, str) or not item.strip():
                    continue
                name = item.strip()
                # Same filter as register_env_passthrough: provider credentials
                # must not reach sandbox children whether the request came from
                # a skill or from config.yaml (GHSA-rhgp-j443-p4rf).
                if _is_hermes_provider_credential(name):
                    logger.warning(
                        "env passthrough: refusing to register Hermes "
                        "provider credential %r from config.yaml (blocked "
                        "by _HERMES_PROVIDER_ENV_BLOCKLIST). Operator "
                        "configuration must not override the execute_code "
                        "sandbox's credential scrubbing; see "
                        "GHSA-rhgp-j443-p4rf.",
                        name,
                    )
                    continue
                result.add(name)
    except Exception as e:
        logger.debug("Could not read tools.env_passthrough from config: %s", e)

    _config_passthrough = frozenset(result)
    return _config_passthrough


def is_env_passthrough(var_name: str) -> bool:
    """True if *var_name* was registered by a skill or listed in config."""
    if var_name in _get_allowed():
        return True
    return var_name in _load_config_passthrough()


def get_all_passthrough() -> frozenset[str]:
    """Return the union of skill-registered and config-based passthrough vars."""
    return frozenset(_get_allowed()) | _load_config_passthrough()


def resolve_passthrough_value(
    name: str,
    fallback: str | None = None,
) -> str | None:
    """Resolve an allowlisted variable without crossing profile boundaries.

    ``fallback`` is the value the caller would have forwarded before profile
    secret scopes existed (typically a snapshot of ``os.environ`` or the
    current profile's ``.env``).  An active multiplex scope is authoritative:
    a missing key returns ``None`` and never falls back to the process-global
    environment; an unscoped read while multiplexing is active raises the
    fail-closed ``UnscopedSecretError`` from :mod:`agent.secret_scope`.
    Outside multiplexing, an installed scope keeps the overlay semantics and
    an unscoped caller keeps its already-resolved fallback.
    """
    from agent.secret_scope import (
        _is_global_env,
        current_secret_scope,
        get_secret,
        is_multiplex_active,
    )

    # Global terminal/runtime settings are not profile secrets.  ``fallback``
    # is already the caller's effective value (including an explicit per-call
    # override), so preserve it rather than replacing it with the process-wide
    # value while a multiplex scope is active.
    if _is_global_env(name) and fallback is not None:
        return fallback

    scope = current_secret_scope()
    multiplex_active = is_multiplex_active()
    if scope is None:
        if multiplex_active:
            return get_secret(name)
        return fallback
    return get_secret(name, None if multiplex_active else fallback)


def clear_env_passthrough() -> None:
    """Reset the skill-scoped allowlist (e.g. on session reset)."""
    _get_allowed().clear()
