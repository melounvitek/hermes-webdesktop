"""Secret-scrub policy for Hermes child processes: pure data + predicates for
which env names are Hermes-managed credentials. The env *builders* applying it
(``_sanitize_subprocess_env``, ``_make_run_env``, ``hermes_subprocess_env``,
``build_subprocess_env``) live in ``tools.environments.local``."""

import os

# Prefix a caller uses in ``extra_env`` to force a blocklisted var through.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

# Hermes-managed AWS *inference* credentials for ``auth_type="aws_sdk"`` (Bedrock).
# Deliberately only the Bedrock bearer token — an inference secret like
# OPENAI_API_KEY that no aws/terraform/boto3 toolchain uses. The general AWS
# credential chain stays inheritable on purpose: the local terminal is the user's
# trusted operator shell (SECURITY.md §3.2), and env_passthrough can never
# re-allow a blocklisted name (GHSA-rhgp-j443-p4rf), so blocking would be
# unrecoverable for every aws/terraform user.
_AWS_SDK_CREDENTIAL_ENV_VARS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
})

_STATIC_PROVIDER_ENV_BLOCKLIST = frozenset({
    "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION", "OPENROUTER_API_KEY", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "LLM_MODEL", "GOOGLE_API_KEY",
    # Path to a GCP service-account JSON, not a bare key, so OPTIONAL_ENV_VARS
    # marks it password=False and the registry loop skips it.
    "VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS", "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY", "PERPLEXITY_API_KEY",
    "COHERE_API_KEY", "FIREWORKS_API_KEY", "XAI_API_KEY", "HELICONE_API_KEY",
    "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
    "TELEGRAM_HOME_CHANNEL", "TELEGRAM_HOME_CHANNEL_NAME", "DISCORD_HOME_CHANNEL",
    "DISCORD_HOME_CHANNEL_NAME", "DISCORD_REQUIRE_MENTION",
    "DISCORD_FREE_RESPONSE_CHANNELS", "DISCORD_AUTO_THREAD", "SLACK_HOME_CHANNEL",
    "SLACK_HOME_CHANNEL_NAME", "SLACK_ALLOWED_USERS", "WHATSAPP_ENABLED",
    "WHATSAPP_MODE", "WHATSAPP_ALLOWED_USERS", "SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT",
    "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS", "SIGNAL_HOME_CHANNEL",
    "SIGNAL_HOME_CHANNEL_NAME", "SIGNAL_IGNORE_STORIES", "HASS_TOKEN", "HASS_URL",
    "EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST",
    "EMAIL_HOME_ADDRESS", "EMAIL_HOME_ADDRESS_NAME", "HERMES_DASHBOARD_SESSION_TOKEN",
    "GATEWAY_ALLOWED_USERS", "GH_TOKEN", "GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
    "DAYTONA_API_KEY", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET",
    "GATEWAY_RELAY_DELIVERY_KEY", "VERCEL_OIDC_TOKEN", "VERCEL_TOKEN",
    "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID",
})


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config."""
    blocked: set[str] = set(_STATIC_PROVIDER_ENV_BLOCKLIST)

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.auth_type == "aws_sdk":
                blocked.update(_AWS_SDK_CREDENTIAL_ENV_VARS)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"} or (
                category == "setting" and metadata.get("password")
            ):
                blocked.add(name)
    except ImportError:
        pass

    # CLAUDE_CODE_OAUTH_TOKEN is owned by the user's Claude Code install, not a
    # Hermes credential (subscription auth is not a Hermes provider path).
    # Stripping it made agent-spawned ``claude`` CLIs fall through to the shared
    # Keychain / ~/.claude credentials store and, on auth failure, wipe it —
    # logging the user out. It arrives via the anthropic registry entry above.
    blocked.discard("CLAUDE_CODE_OAUTH_TOKEN")
    # BUZZ_* is deliberately NOT discarded, even for Buzz-managed agents: this
    # blocklist feeds every scrub surface (terminal, execute_code, the
    # hermes_subprocess_env Tier-2 strip), so an import-time discard would leak
    # BUZZ_PRIVATE_KEY into non-terminal children. The Buzz carve-out is a
    # terminal-only, context-gated scrub-path exemption — see
    # ``_is_terminal_first_party_env``.
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()

# First-party platform credentials (``BUZZ_*``, driving the platform-mandated
# ``buzz`` CLI) carved out of the TERMINAL scrub only (``_make_run_env``,
# ``_sanitize_subprocess_env``); execute_code, hermes_subprocess_env, docker and
# env_passthrough registration stay sealed, so GHSA-rhgp-j443-p4rf holds.
# CONTEXT-GATED (``_buzz_terminal_context_active``): a Telegram/CLI/cron session
# on a host that also runs a Buzz gateway must not get the signing key. Values
# are used directly, never scope-resolved (UnscopedSecretError under multiplex),
# and the snapshot treats them as profile-scoped so they never persist across
# profiles. Prefix-based so future BUZZ_* names need no code change.
_TERMINAL_FIRST_PARTY_ENV_PREFIXES = ("BUZZ_",)


def _matches_terminal_first_party_prefix(name: str) -> bool:
    """Pure name check (``BUZZ_*``), regardless of session context — the
    snapshot exclusion must stay conservative even when the carve-out is inactive."""
    return name.startswith(_TERMINAL_FIRST_PARTY_ENV_PREFIXES)


def _buzz_terminal_context_active() -> bool:
    """True when this process/session operates as a Buzz agent.

    Either signal suffices: ``BUZZ_MANAGED_AGENT`` in the process env (set only
    by Buzz Desktop's buzz-acp harness), or the live session's platform is
    ``buzz`` via the gateway ContextVar — authoritative under a concurrent
    multi-session host, so a sibling Telegram session resolves its OWN platform.
    """
    if os.environ.get("BUZZ_MANAGED_AGENT"):
        return True
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower() == "buzz"
    except Exception:
        return False


def _is_terminal_first_party_env(name: str) -> bool:
    """``name`` is a first-party platform credential (``BUZZ_*``) AND the
    current process/session context entitles it to reach terminal children."""
    return _matches_terminal_first_party_prefix(name) and _buzz_terminal_context_active()


# Active-venv markers that must NOT leak: a leaked VIRTUAL_ENV/CONDA_PREFIX makes
# uv/poetry sync ANOTHER project's deps into the Hermes venv (clobbering it; the
# venv stays reachable via PATH so stripping is safe), and PYTHONHOME redirects
# any child interpreter's stdlib to the Hermes venv (version-mismatch crashes).
# PYTHONPATH is handled separately — only Hermes-owned entries are removed.
_ACTIVE_VENV_MARKER_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME")


def _is_hermes_internal_secret(key: str) -> bool:
    """True for Hermes-internal secrets injected under *dynamic* names the
    static blocklist cannot enumerate: ``AUXILIARY_<TASK>_API_KEY``/``_BASE_URL``
    (per-task side-LLM credentials) and ``GATEWAY_RELAY_*_SECRET``/``_KEY``/
    ``_TOKEN`` (relay auth; non-secret routing hints stay visible). Single source
    of truth for every spawn path, stripped regardless of env_passthrough
    registration or ``inherit_credentials``."""
    upper = key.upper()
    if upper.startswith("AUXILIARY_") and upper.endswith(("_API_KEY", "_BASE_URL")):
        return True
    return upper.startswith("GATEWAY_RELAY_") and upper.endswith(("_SECRET", "_KEY", "_TOKEN"))


def _plugin_terminal_env_strip_keys() -> frozenset:
    """Credential env keys owned by plugin-registered terminal backends.

    Computed at call time because plugins register after import. Tier-1:
    stripped from every spawned subprocess unconditionally. Fail-soft to empty.
    """
    try:
        from agent.terminal_env_registry import plugin_strip_env_keys

        return plugin_strip_env_keys()
    except Exception:
        return frozenset()


# Tier-1 secrets: stripped from EVERY spawned subprocess even under
# inherit_credentials (claude/codex/gemini). Not provider credentials — no child
# needs them and they are the highest-value secrets to keep from a compromised
# dependency. Provider keys are the conditional Tier-2 strip.
_ALWAYS_STRIP_KEYS: frozenset[str] = frozenset({
    # GitHub auth
    "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID",
    # Gateway / messaging bot tokens and access control
    "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN",
    "SLACK_SIGNING_SECRET", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
    # Gateway relay auth triplet. _SECRET/_DELIVERY_KEY are also matched by
    # _is_hermes_internal_secret, but _ID has no secret suffix, so it must be
    # enumerated here to stay stripped on the inherit_credentials=True path.
    "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY",
    "HASS_TOKEN", "EMAIL_PASSWORD", "HERMES_DASHBOARD_SESSION_TOKEN",
    # Remote-compute / infrastructure secrets
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "DAYTONA_API_KEY",
})
