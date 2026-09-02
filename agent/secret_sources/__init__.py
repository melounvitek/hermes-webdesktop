"""External secret source integrations.

A secret source supplies environment-variable-shaped credentials at process
startup, _after_ ~/.hermes/.env has loaded.  The contract is
:class:`agent.secret_sources.base.SecretSource`; the orchestrator (ordering,
mapped-beats-bulk precedence, first-claim-wins, ``override_existing``,
provenance) is :func:`agent.secret_sources.registry.apply_all`.  The
atomic-write / 0600 / TTL disk cache is shared in ``_cache``.

Bundled: ``bitwarden`` (bws CLI), ``onepassword`` (op CLI), ``command`` (user
helper).  The set is deliberately closed — new third-party managers ship as
standalone plugin repos that subclass ``SecretSource`` and register through
``PluginContext.register_secret_source()``.
"""

from agent.secret_sources.base import (  # noqa: F401
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    is_valid_env_name,
    run_secret_cli,
    scrub_ansi,
)
