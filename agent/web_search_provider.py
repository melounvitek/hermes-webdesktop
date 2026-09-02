"""
Web Search Provider ABC
=======================

Pluggable-backend interface for web search and content extraction — the SINGLE
plugin-facing surface every in-tree web provider (brave-free, ddgs, searxng,
exa, parallel, tavily, keenable, firecrawl) implements. Providers register via
``PluginContext.register_web_search_provider()``; the active one (selected by
``web.search_backend`` / ``web.extract_backend`` / ``web.backend``) services
every ``web_search`` / ``web_extract`` call.

Response shape (preserved from the legacy contract so the tool wrapper does not
translate). Search::

    {"success": True, "data": {"web": [
        {"title": str, "url": str, "description": str, "position": int}, ...]}}

Extract::

    {"success": True, "data": [
        {"url": str, "title": str, "content": str, "raw_content": str, "metadata": dict}, ...]}

On failure (either capability): ``{"success": False, "error": str}``.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, List, Optional

from agent.provider_base import ProviderBase


def get_provider_env(name: str) -> str:
    """Config-aware env lookup: ``os.environ`` first, then ``~/.hermes/.env``.

    Credentials set through Hermes' config layer must be visible even when never
    exported into the process environment (gateway sessions, delegate children,
    subprocess agent runs). Falls back to bare ``os.getenv`` when the config
    module is unavailable. Returns the stripped value, or ``""`` when unset.
    """
    val: Optional[str] = None
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:  # noqa: BLE001 — config layer optional here
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


class WebSearchProvider(ProviderBase):
    """Abstract base class for a web search/extract backend.

    Subclasses implement :meth:`is_available` and at least one of :meth:`search`
    / :meth:`extract`; the :meth:`supports_search` / :meth:`supports_extract`
    flags let the registry route each capability, so one class can serve both.
    """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls.

        Cheap check only (env var present, dep importable, instance URL set) —
        must NOT make network calls; runs at tool-registration time and on every
        ``hermes tools`` paint.
        """

    def supports_search(self) -> bool:
        """True if this provider implements :meth:`search`."""
        return True

    def is_keyless_available(self) -> bool:
        """True when this provider can serve calls WITHOUT credentials.

        A weaker tier than :meth:`is_available`, used only when NO provider is
        configured or keyed (public anonymous free tiers such as Exa / Parallel
        MCP). It must never make :meth:`is_available` True, or the legacy
        preference walk would route users holding real credentials for a
        lower-priority backend onto a higher-priority backend's free tier.
        Cheap, no network. Default False.
        """
        return False

    def supports_extract(self) -> bool:
        """True if this provider implements :meth:`extract` (sync or ``async def`` —
        the dispatcher awaits coroutine functions)."""
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search. Callers gate on :meth:`supports_search`."""
        raise NotImplementedError(
            f"{self.name} does not support search (override supports_search)"
        )

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        """Extract content from URLs. Callers gate on :meth:`supports_extract`.

        Returns a list of ``{"url", "title", "content", "raw_content",
        "metadata"?, "error"?}`` dicts (``error`` only on per-URL failure).
        May be ``async def``. ``kwargs`` may carry forward-compat fields
        (``format``, ``include_raw``, ``max_chars``) — ignore unknown keys.
        """
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )
