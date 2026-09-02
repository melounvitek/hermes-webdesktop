"""Tavily web search + content extraction (``/search``, ``/extract``; sync httpx).

Env: ``TAVILY_API_KEY`` (https://app.tavily.com/home, optional), ``TAVILY_BASE_URL``.
Keyed requests use ``Authorization: Bearer``; without a key the request is
keyless (``X-Tavily-Access-Mode: keyless``). Tavily is NOT in the zero-config
keyless ring — keyless access is opt-in by selecting Tavily in ``hermes tools``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP,
    BaseWebSearchProvider,
    document,
    extract_fail,
    http_status_detail,
    provider_env,
    run_extract,
    run_search,
    search_fail,
    search_ok,
    use_keyless,
)

logger = logging.getLogger(__name__)

_CLIENT_NAME = "hermes-agent"

_SEARCH_PAYLOAD = {"include_raw_content": False, "include_images": False}


def _tavily_headers(api_key: str) -> Dict[str, str]:
    headers = {"X-Client-Name": _CLIENT_NAME}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    return headers


def _tavily_request(endpoint: str, payload: Dict[str, Any], *, api_key: Optional[str] = None) -> Dict[str, Any]:
    """POST to Tavily and return parsed JSON.

    ``api_key=None`` reads ``TAVILY_API_KEY``; pass ``""`` to force the keyless
    header even when a key exists (``web.provider_tier.tavily: free``). Non-2xx
    raises ValueError with the body so Tavily's rate-limit/upgrade text reaches the model.
    """
    if api_key is None:
        api_key = provider_env("TAVILY_API_KEY")
    base_url = provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)

    response = httpx.post(url, json=payload, timeout=60, headers=_tavily_headers(api_key))
    if response.status_code >= 400:
        raise ValueError(http_status_detail(response))
    return response.json()


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    # title-first key order is Tavily's historical wire shape (differs from web_hit).
    return search_ok([
        {"title": r.get("title", ""), "url": r.get("url", ""), "description": r.get("content", ""), "position": i + 1}
        for i, r in enumerate(response.get("results", []))
    ])


def _normalize_tavily_documents(response: Dict[str, Any], fallback_url: str = "") -> List[Dict[str, Any]]:
    """Map ``/extract`` to documents; ``failed_results`` / ``failed_urls`` become ``error`` entries."""
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append(document(url, result.get("title", ""), raw))
    for fail in response.get("failed_results", []):
        url = fail.get("url", fallback_url)
        documents.append(_failed_document(url, fail.get("error", "extraction failed")))
    for fail_url in response.get("failed_urls", []):
        documents.append(_failed_document(str(fail_url), "extraction failed"))
    return documents


def _failed_document(url: str, error: str) -> Dict[str, Any]:
    return {"url": url, "title": "", "content": "", "raw_content": "", "error": error, "metadata": {"sourceURL": url}}


def _missing_key_error(action: str) -> str:
    return (
        "TAVILY_API_KEY is not set. Get a key at https://app.tavily.com/home "
        f"or select Tavily in `hermes tools` for opt-in keyless {action}."
    )


class TavilyWebSearchProvider(BaseWebSearchProvider):
    """Tavily search + extract provider (keyed, or opt-in keyless)."""

    NAME = "tavily"
    DISPLAY_NAME = "Tavily"
    KEY_ENV = "TAVILY_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            api_key = provider_env("TAVILY_API_KEY")
            force_keyless = use_keyless("tavily", api_key)
            if not force_keyless and not api_key:
                return search_fail(_missing_key_error("search"))

            logger.info("Tavily %ssearch: '%s' (limit=%d)", "keyless " if force_keyless else "", query, limit)
            payload = {"query": query, "max_results": min(limit, SEARCH_LIMIT_CAP), **_SEARCH_PAYLOAD}
            return _normalize_tavily_search_results(
                _tavily_request("search", payload, api_key="" if force_keyless else api_key)
            )

        return run_search("Tavily", logger, _body)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            api_key = provider_env("TAVILY_API_KEY")
            force_keyless = use_keyless("tavily", api_key)
            if not force_keyless and not api_key:
                return extract_fail(urls, _missing_key_error("extract"))

            logger.info("Tavily %sextract: %d URL(s)", "keyless " if force_keyless else "", len(urls))
            raw = _tavily_request("extract", {"urls": urls, "include_images": False}, api_key="" if force_keyless else api_key)
            return _normalize_tavily_documents(raw, fallback_url=urls[0] if urls else "")

        return run_extract("Tavily", logger, urls, _body)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Tavily",
            "badge": "free · key optional",
            "tag": "Search + extract. Opt-in keyless; set TAVILY_API_KEY for higher limits.",
            "env_vars": [
                {
                    "key": "TAVILY_API_KEY",
                    "prompt": "Tavily API key (optional — keyless works when Tavily is selected)",
                    "url": "https://app.tavily.com/home",
                },
            ],
        }
