"""Firecrawl web search + extract provider (direct SDK, keyless cloud, or Nous tool-gateway).

Config: ``web.backend`` / ``web.search_backend`` / ``web.extract_backend: firecrawl``.
Env: FIRECRAWL_API_KEY, FIRECRAWL_API_URL (self-hosted), FIRECRAWL_GATEWAY_URL / TOOL_GATEWAY_*.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from plugins.web._common import BaseWebSearchProvider, search_fail, search_ok
from tools.url_safety import is_safe_url
# Module-level (cheap import) so tests can monkeypatch the policy gate on this module.
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_FIRECRAWL_CLOUD_API_URL = "https://api.firecrawl.dev"
_SELECTION_KEYS = ("backend", "search_backend", "extract_backend")


# --- Lazy Firecrawl SDK proxy -------------------------------------------------
# The SDK costs ~200ms of imports on a cold CLI; defer to first use. tools.web_tools
# re-exports ``Firecrawl`` so ``patch("tools.web_tools.Firecrawl")`` keeps working.

_FIRECRAWL_CLS_CACHE: Optional[type] = None


def _load_firecrawl_cls() -> type:
    """Import and cache ``firecrawl.Firecrawl`` (lazy_deps install hint → ImportError)."""
    global _FIRECRAWL_CLS_CACHE
    if _FIRECRAWL_CLS_CACHE is None:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("search.firecrawl", prompt=False)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface install hint
            raise ImportError(str(exc))
        from firecrawl import Firecrawl as _cls

        _FIRECRAWL_CLS_CACHE = _cls
    return _FIRECRAWL_CLS_CACHE


class _FirecrawlProxy:
    """Callable proxy that looks like ``firecrawl.Firecrawl`` but imports lazily."""

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _load_firecrawl_cls()(*args, **kwargs)

    def __instancecheck__(self, obj: Any) -> bool:
        return isinstance(obj, _load_firecrawl_cls())

    def __repr__(self) -> str:
        return "<lazy firecrawl.Firecrawl proxy>"


Firecrawl = _FirecrawlProxy()


# --- Client construction (direct vs managed-gateway) ---------------------------
# Client cache slots and gateway/token helpers are read through tools.web_tools so
# tests that reset ``tools.web_tools._firecrawl_client`` or patch
# ``tools.web_tools._peek_nous_access_token`` see their changes.


def _env(name: str) -> str:
    from hermes_cli.config import get_env_value

    return (get_env_value(name) or "").strip()


def _get_direct_firecrawl_config() -> Optional[tuple]:
    """Return direct Firecrawl ``(mode, kwargs, cache_key)`` or None.

    ``mode`` is ``"sdk"`` (keyed / self-hosted) or ``"keyless"`` (explicit
    Firecrawl selection with no credentials — public cloud API, anonymous
    rate-limited). Keyless requires the explicit selection so an unconfigured
    install never silently routes to it.
    """
    api_key = _env("FIRECRAWL_API_KEY")
    api_url = _env("FIRECRAWL_API_URL").rstrip("/")

    if not api_key and not api_url:
        if _is_explicit_firecrawl_selection():
            return "keyless", {"api_url": _FIRECRAWL_CLOUD_API_URL}, ("direct-keyless", _FIRECRAWL_CLOUD_API_URL, None)
        return None

    kwargs = {k: v for k, v in (("api_key", api_key), ("api_url", api_url)) if v}
    return "sdk", kwargs, ("direct", api_url or None, api_key or None)


def _is_explicit_firecrawl_selection() -> bool:
    """True when config explicitly selects Firecrawl for web tools."""
    import tools.web_tools as _wt

    cfg = _wt._load_web_config()
    return any((cfg.get(key) or "").lower().strip() == "firecrawl" for key in _SELECTION_KEYS)


def _use_keyless_ring() -> bool:
    """True when Firecrawl calls should route via the keyless ring.

    Only when there are no direct credentials, the managed Nous gateway isn't
    the selected path, and the keyless tier isn't disabled or pinned paid.
    """
    if _env("FIRECRAWL_API_KEY") or _env("FIRECRAWL_API_URL"):
        return False
    import tools.web_tools as _wt
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection

    try:
        if read_selection("web") == NOUS_MANAGED_PROVIDER:
            return False
    except Exception:  # noqa: BLE001 — selection helpers optional
        pass
    try:
        if _wt._is_tool_gateway_ready() and not _is_explicit_firecrawl_selection():
            return False
    except Exception:  # noqa: BLE001 — probe optional
        pass
    from plugins.web.keyless_mcp import use_keyless

    return use_keyless("firecrawl", "")


class _KeylessFirecrawlClient:
    """Minimal REST client for Firecrawl's keyless cloud mode.

    Duck-types the SDK's ``search`` / ``scrape``; never sends an Authorization header.
    """

    def __init__(self, api_url: str = _FIRECRAWL_CLOUD_API_URL):
        self.api_url = api_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = httpx.post(
            f"{self.api_url}{path}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    def search(self, *, query: str, limit: int = 5) -> Dict[str, Any]:
        return self._post("/v2/search", {"query": query, "limit": limit})

    def scrape(self, *, url: str, formats: List[str]) -> Dict[str, Any]:
        return self._post("/v2/scrape", {"url": url, "formats": formats})


def _get_firecrawl_gateway_url() -> str:
    """Return the configured Firecrawl gateway URL."""
    import tools.web_tools as _wt

    return _wt.build_vendor_gateway_url("firecrawl")


def _is_tool_gateway_ready() -> bool:
    """True when gateway URL + Nous Subscriber token are available."""
    import tools.web_tools as _wt

    return _wt.resolve_managed_tool_gateway("firecrawl", token_reader=_wt._peek_nous_access_token) is not None


def check_firecrawl_api_key() -> bool:
    """True when the Firecrawl route selected via ``hermes tools`` (or, on a
    never-configured install, either route) is usable. Re-exported by tools.web_tools."""
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection

    selected = read_selection("web")
    if selected == NOUS_MANAGED_PROVIDER:
        return _is_tool_gateway_ready()
    has_direct = _get_direct_firecrawl_config() is not None
    if selected is not None:
        return has_direct
    return has_direct or _is_tool_gateway_ready()


def _firecrawl_backend_help_suffix() -> str:
    """Return optional managed-gateway guidance for Firecrawl help text."""
    import tools.web_tools as _wt

    if not _wt.managed_nous_tools_enabled():
        return ""
    return ", or use the Nous Tool Gateway via your subscription (FIRECRAWL_GATEWAY_URL or TOOL_GATEWAY_DOMAIN)"


def _get_firecrawl_client() -> Any:
    """Get or create the cached Firecrawl client.

    Strict selection semantics on the stored ``web`` selection: ``"nous"`` →
    managed Tool Gateway ONLY; any other stored backend → direct Firecrawl ONLY
    (never a silent managed fallback billed to Nous); never-configured → direct
    when present, else managed. Raises ValueError when the resolved path is unusable.
    """
    import tools.web_tools as _wt
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        read_selection,
        selection_error,
        selection_exists,
    )

    selected = read_selection("web")
    direct_config = _get_direct_firecrawl_config()

    def _managed_kwargs():
        gw = _wt.resolve_managed_tool_gateway("firecrawl", token_reader=_wt._read_nous_access_token)
        if gw is None:
            return None
        kwargs = {"api_key": gw.nous_user_token, "api_url": gw.gateway_origin}
        return kwargs, ("tool-gateway", kwargs["api_url"], gw.nous_user_token)

    client_mode = "sdk"
    if selected == NOUS_MANAGED_PROVIDER:
        managed = _managed_kwargs()
        if managed is None:
            logger.error(
                "Firecrawl client initialization failed: the Nous "
                "Subscription web selection is stored but the tool gateway "
                "is unavailable."
            )
            raise ValueError(selection_error(
                "web", NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or unreachable)",
            ))
        kwargs, client_config = managed
    elif selected is not None or selection_exists("web"):
        # Stored vendor selection: direct Firecrawl only. With no credentials the
        # explicit selection unlocks keyless cloud mode instead of erroring.
        if direct_config is None:
            logger.error(
                "Firecrawl client initialization failed: direct Firecrawl "
                "selected but FIRECRAWL_API_KEY/FIRECRAWL_API_URL is not set."
            )
            raise ValueError(selection_error(
                "web", selected or "firecrawl", "neither FIRECRAWL_API_KEY nor FIRECRAWL_API_URL is set",
            ))
        client_mode, kwargs, client_config = direct_config
    elif direct_config is not None:
        client_mode, kwargs, client_config = direct_config
    else:
        # Never-configured web section: legacy managed fallback.
        managed = _managed_kwargs()
        if managed is None:
            logger.error("Firecrawl client initialization failed: missing direct config and tool-gateway auth.")
            message = (
                "Web tools are not configured. "
                "Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL "
                "for a self-hosted Firecrawl instance."
            )
            if _wt.managed_nous_tools_enabled():
                message += (
                    " With your Nous subscription you can also use the Tool Gateway. "
                    "run `hermes tools` and select Nous Subscription as the web provider."
                )
            else:
                message += " " + _wt.nous_tool_gateway_unavailable_message("managed Firecrawl web tools")
            raise ValueError(message)
        kwargs, client_config = managed

    cached = getattr(_wt, "_firecrawl_client", None)
    if cached is not None and getattr(_wt, "_firecrawl_client_config", None) == client_config:
        return cached

    if client_mode == "keyless":
        _wt._firecrawl_client = _KeylessFirecrawlClient(api_url=kwargs["api_url"])
    else:
        _wt._firecrawl_client = _wt.Firecrawl(**kwargs)
    _wt._firecrawl_client_config = client_config
    return _wt._firecrawl_client


# --- Response shape normalization (SDK / direct / gateway differ) --------------


def _to_plain_object(value: Any) -> Any:
    """Convert SDK objects (pydantic ``model_dump`` / ``__dict__``) to plain data when possible."""
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            pass
    return value


def _normalize_result_list(values: Any) -> List[Dict[str, Any]]:
    """Normalize mixed SDK/list payloads into a list of dicts."""
    if not isinstance(values, list):
        return []
    plain = (_to_plain_object(item) for item in values)
    return [p for p in plain if isinstance(p, dict)]


def _extract_web_search_results(response: Any) -> List[Dict[str, Any]]:
    """Extract Firecrawl search results across SDK/direct/gateway response shapes."""
    response_plain = _to_plain_object(response)

    if isinstance(response_plain, dict):
        data = response_plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)
        candidates = []
        if isinstance(data, dict):
            candidates += [data.get("web"), data.get("results")]
        candidates += [response_plain.get("web"), response_plain.get("results")]
        for candidate in candidates:
            normalized = _normalize_result_list(candidate)
            if normalized:
                return normalized

    if hasattr(response, "web"):
        return _normalize_result_list(getattr(response, "web", []))
    return []


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    """Normalize Firecrawl scrape payload shape across SDK and gateway variants."""
    result_plain = _to_plain_object(scrape_result)
    if not isinstance(result_plain, dict):
        return {}
    nested = result_plain.get("data")
    return nested if isinstance(nested, dict) else result_plain


def _error_entry(
    url: str,
    error: str,
    *,
    title: str = "",
    raw: bool = False,
    blocked: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-URL extract failure item. ``raw`` adds the ``raw_content`` key (post-scrape
    failures carry it, pre-scrape ones don't); ``blocked`` adds ``blocked_by_policy``."""
    entry: Dict[str, Any] = {"url": url, "title": title, "content": ""}
    if raw:
        entry["raw_content"] = ""
    entry["error"] = error
    if blocked:
        entry["blocked_by_policy"] = {k: blocked[k] for k in ("host", "rule", "source")}
    return entry


_SCRAPE_TIMEOUT_MSG = (
    "Scrape timed out after 60s — page may be too large "
    "or unresponsive. Try browser_navigate instead."
)
_UNSAFE_REDIRECT_MSG = "Blocked: URL targets a private or internal network address"


async def _scrape_one(url: str, formats: List[str], format: Optional[str]) -> Dict[str, Any]:
    """Scrape one URL (60s timeout) and re-check SSRF + website policy against the
    post-redirect URL. Never raises for scrape errors; returns an error entry instead."""
    blocked = check_website_access(url)
    if blocked:
        logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
        return _error_entry(url, blocked["message"], blocked=blocked)

    try:
        logger.info("Firecrawl scraping: %s", url)
        try:
            scrape_result = await asyncio.wait_for(
                asyncio.to_thread(_get_firecrawl_client().scrape, url=url, formats=formats),
                timeout=60,
            )
        except asyncio.TimeoutError:
            logger.warning("Firecrawl scrape timed out for %s", url)
            return _error_entry(url, _SCRAPE_TIMEOUT_MSG)

        scrape_payload = _extract_scrape_payload(scrape_result)
        metadata = scrape_payload.get("metadata", {})
        content_markdown = scrape_payload.get("markdown")
        content_html = scrape_payload.get("html")

        # SDK may return a typed object for metadata (raw __dict__ here, unlike _to_plain_object).
        if not isinstance(metadata, dict):
            if hasattr(metadata, "model_dump"):
                metadata = metadata.model_dump()
            elif hasattr(metadata, "__dict__"):
                metadata = metadata.__dict__
            else:
                metadata = {}

        title = metadata.get("title", "")
        final_url = metadata.get("sourceURL", url)

        if not is_safe_url(final_url):
            logger.info("Blocked redirected web_extract for unsafe final URL: %s", final_url)
            return _error_entry(final_url, _UNSAFE_REDIRECT_MSG, title=title, raw=True)

        final_blocked = check_website_access(final_url)
        if final_blocked:
            logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
            return _error_entry(final_url, final_blocked["message"], title=title, raw=True, blocked=final_blocked)

        if format == "markdown" or (format is None and content_markdown):
            chosen_content = content_markdown
        else:
            chosen_content = content_html or content_markdown or ""
        return {"url": final_url, "title": title, "content": chosen_content, "raw_content": chosen_content, "metadata": metadata}
    except Exception as scrape_err:  # noqa: BLE001
        logger.debug("Firecrawl scrape failed for %s: %s", url, scrape_err)
        return _error_entry(url, str(scrape_err), raw=True)


# --- Provider class ------------------------------------------------------------


class FirecrawlWebSearchProvider(BaseWebSearchProvider):
    """Firecrawl search + extract provider with dual auth paths."""

    NAME = "firecrawl"
    DISPLAY_NAME = "Firecrawl"
    EXTRACT = True
    KEYLESS = True  # default-on ring member unless pinned ``paid``

    def is_available(self) -> bool:
        return check_firecrawl_api_key()

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Sync search. Pre-flight errors (ValueError / ImportError) propagate so the
        dispatcher emits the legacy ``tool_error`` envelope; in-flight errors are
        returned as ``{"success": False, "error": ...}``."""
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return search_fail("Interrupted")

        if _use_keyless_ring():
            from plugins.web.keyless_mcp import search_with_failover

            logger.info("Firecrawl keyless search: '%s' (limit=%d)", query, limit)
            return search_with_failover("firecrawl", query, limit)

        logger.info("Firecrawl search: '%s' (limit=%d)", query, limit)
        client = _get_firecrawl_client()
        try:
            response = client.search(query=query, limit=limit)
            web_results = _extract_web_search_results(response)
            logger.info("Firecrawl: found %d search results", len(web_results))
            return search_ok(web_results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl search error: %s", exc)
            return search_fail(f"Firecrawl search failed: {exc}")

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Per-URL scrape via :func:`_scrape_one`; failures become items with an
        ``error`` field. ``format``: "markdown" | "html" | both (markdown preferred)."""
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        if _use_keyless_ring():
            from plugins.web.keyless_mcp import extract_with_failover

            logger.info("Firecrawl keyless extract: %d URL(s)", len(urls))
            return await asyncio.to_thread(extract_with_failover, "firecrawl", list(urls))

        format = kwargs.get("format")
        formats = [format] if format in ("markdown", "html") else ["markdown", "html"]

        return [
            {"url": url, "error": "Interrupted", "title": ""} if _is_interrupted() else await _scrape_one(url, formats, format)
            for url in urls
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Firecrawl",
            "badge": "keyless/paid · optional gateway",
            "tag": "Full search + extract; supports keyless cloud, direct API, and Nous tool-gateway routing.",
            "env_vars": [
                {
                    "key": "FIRECRAWL_API_KEY",
                    "prompt": "Firecrawl API key (optional; blank = keyless cloud or self-hosted)",
                    "url": "https://docs.firecrawl.dev/introduction",
                },
            ],
        }
