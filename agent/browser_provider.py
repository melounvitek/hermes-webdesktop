"""
Browser Provider ABC
====================

Pluggable-backend interface for cloud browser providers (Browserbase, Browser
Use, Firecrawl, …). Providers register via
:meth:`PluginContext.register_browser_provider`; the active one (selected by
``browser.cloud_provider`` in ``config.yaml``) services every cloud-mode
``browser_*`` tool call. Providers live in ``<repo>/plugins/browser/<name>/``
(built-in) or ``~/.hermes/plugins/browser/<name>/`` (user, opt-in).

Session metadata contract (preserved from the legacy ``CloudBrowserProvider``
so :mod:`tools.browser_tool` needs no translation)::

    {
        "session_name": str,        # unique name for agent-browser --session
        "bb_session_id": str,       # provider session ID (for close/cleanup)
        "cdp_url": str,             # CDP websocket URL
        "expires_at": str,          # optional provider-authoritative ISO timestamp
        "features": dict,           # feature flags that were enabled
        "external_call_id": str,    # optional, managed-gateway billing key
    }

``bb_session_id`` is a legacy key name kept verbatim for backward compat — it
holds the provider's session ID regardless of which provider is in use.
"""

from __future__ import annotations

import abc
from typing import Dict

from agent.provider_base import ProviderBase


class BrowserProvider(ProviderBase):
    """Abstract base class for a cloud browser backend.

    Subclasses implement :attr:`name` (the ``browser.cloud_provider`` value,
    e.g. ``browserbase``, ``browser-use``, ``firecrawl``), :meth:`is_available`,
    and the lifecycle trio :meth:`create_session` / :meth:`close_session` /
    :meth:`emergency_cleanup`. ``get_setup_schema`` may add ``"post_setup"``
    (e.g. ``"agent_browser"``) to trigger the install hook.
    """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls.

        Cheap check only (env var present, managed-gateway token readable, dep
        importable) — must NOT make network calls; runs at tool-registration
        time and on every ``hermes tools`` paint.
        """

    @abc.abstractmethod
    def create_session(self, task_id: str) -> Dict[str, object]:
        """Create a cloud browser session and return the session metadata dict
        described in the module docstring.

        May raise ``ValueError`` (missing credentials) or ``RuntimeError``
        (network / API failure); the dispatcher surfaces these to the user.
        """

    @abc.abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Release a cloud session by provider session ID.

        Returns True on success, False on failure. Should not raise — log and
        return False so the dispatcher's cleanup loop keeps moving.
        """

    @abc.abstractmethod
    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit / signal handlers. Must tolerate
        missing credentials and network errors; must not raise."""

    # Legacy ``CloudBrowserProvider`` names still used by ``tools.browser_tool``
    # and out-of-tree subclasses; thin delegations to the current API.

    def is_configured(self) -> bool:
        """Backward-compat alias for :meth:`is_available`."""
        return self.is_available()

    def provider_name(self) -> str:
        """Backward-compat alias returning :attr:`display_name`."""
        return self.display_name
