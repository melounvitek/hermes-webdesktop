"""Shared plumbing for the bundled web-search provider plugins.

Every helper resolves patched-in-tests collaborators (``get_provider_env``,
``tools.interrupt``, ``plugins.web.keyless_mcp``, ``tools.web_tools`` client
slots) lazily at call time so monkeypatching the source module keeps working.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

SEARCH_LIMIT_CAP = 20  # every vendor here caps max_results at 20 server-side


def provider_env(name: str) -> str:
    """Config-aware env lookup (os.environ, then ~/.hermes/.env)."""
    from agent.web_search_provider import get_provider_env

    return get_provider_env(name)


def use_keyless(name: str, api_key: str) -> bool:
    from plugins.web.keyless_mcp import use_keyless as _use_keyless

    return _use_keyless(name, api_key)


def _interrupted() -> bool:
    from tools.interrupt import is_interrupted

    return is_interrupted()


# --- Result shapes (key order is part of the contract — it reaches the model as JSON) ---


def search_ok(web_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"success": True, "data": {"web": web_results}}


def search_fail(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def web_hit(url: str, title: str, description: str, position: int) -> Dict[str, Any]:
    return {"url": url, "title": title, "description": description, "position": position}


def document(url: str, title: str, content: str, *, source_url: Optional[str] = None) -> Dict[str, Any]:
    """Successful extract entry; ``raw_content`` mirrors ``content`` for the legacy pipeline."""
    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": {"sourceURL": url if source_url is None else source_url, "title": title},
    }


def extract_fail(urls: List[str], error: str) -> List[Dict[str, Any]]:
    return [{"url": u, "title": "", "content": "", "error": error} for u in urls]


# --- Guarded execution: interrupt check + uniform failure classification ---


def _failure_message(
    vendor: str, kind: str, exc: Exception, logger: logging.Logger, *, sdk: bool, verbatim_value_error: bool
) -> str:
    """Map an exception to the user-facing error string.

    ``verbatim_value_error``: ValueError carries a pre-formatted message (missing
    key, HTTP body) and is returned as-is. ``sdk``: ImportError means the lazily
    installed vendor SDK is missing. Anything else is logged and wrapped.
    """
    if verbatim_value_error and isinstance(exc, ValueError):
        return str(exc)
    if sdk and isinstance(exc, ImportError):
        return f"{vendor} SDK not installed: {exc}"
    logger.warning("%s %s error: %s", vendor, kind, exc)
    return f"{vendor} {kind} failed: {exc}"


def run_search(
    vendor: str, logger: logging.Logger, body: Callable[[], Dict[str, Any]], *, sdk: bool = False, verbatim_value_error: bool = True
) -> Dict[str, Any]:
    try:
        if _interrupted():
            return search_fail("Interrupted")
        return body()
    except Exception as exc:  # noqa: BLE001 — surface as failure dict
        return search_fail(_failure_message(vendor, "search", exc, logger, sdk=sdk, verbatim_value_error=verbatim_value_error))


def _extract_interrupted(urls: List[str]) -> List[Dict[str, Any]]:
    return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]


def run_extract(
    vendor: str, logger: logging.Logger, urls: List[str], body: Callable[[], List[Dict[str, Any]]],
    *, sdk: bool = False, verbatim_value_error: bool = True,
) -> List[Dict[str, Any]]:
    """Per-URL failures are returned as entries with ``error`` — never raised."""
    try:
        if _interrupted():
            return _extract_interrupted(urls)
        return body()
    except Exception as exc:  # noqa: BLE001
        return extract_fail(urls, _failure_message(vendor, "extract", exc, logger, sdk=sdk, verbatim_value_error=verbatim_value_error))


async def run_extract_async(
    vendor: str, logger: logging.Logger, urls: List[str], body: Callable[[], Awaitable[List[Dict[str, Any]]]],
    *, sdk: bool = False, verbatim_value_error: bool = True,
) -> List[Dict[str, Any]]:
    try:
        if _interrupted():
            return _extract_interrupted(urls)
        return await body()
    except Exception as exc:  # noqa: BLE001
        return extract_fail(urls, _failure_message(vendor, "extract", exc, logger, sdk=sdk, verbatim_value_error=verbatim_value_error))


# --- HTTP + SDK client helpers ---


def http_status_detail(response: Any) -> str:
    """Response body text for a >=400 reply, or ``HTTP <code>`` when the body is empty."""
    return (response.text or "").strip() or f"HTTP {response.status_code}"


def http_get_json(
    label: str, url: str, *, params: Dict[str, Any], headers: Dict[str, str], timeout: int,
    logger: logging.Logger, reach_target: Optional[str] = None,
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """GET ``url`` and parse JSON. Returns ``(data, None)`` or ``(None, failure_dict)``.

    ``label`` names the vendor in error strings; ``reach_target`` overrides the
    "Could not reach ..." subject (SearXNG includes its instance URL).
    """
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("%s HTTP error: %s", label, exc)
        return None, search_fail(f"{label} returned HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        logger.warning("%s request error: %s", label, exc)
        return None, search_fail(f"Could not reach {reach_target or label}: {exc}")
    try:
        return resp.json(), None
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s response parse error: %s", label, exc)
        return None, search_fail(f"Could not parse {label} response as JSON")


def cached_sdk_client(slot: str, env_var: str, missing_key_error: str, feature: str, factory: Callable[[str], Any]) -> Any:
    """Lazy-build + cache a vendor SDK client on ``tools.web_tools.<slot>``.

    The cache slot lives on ``tools.web_tools`` so tests that reset
    ``tools.web_tools._<vendor>_client = None`` between cases see fresh state.
    Raises ValueError when the key is unset; lazy_deps install hints are
    re-raised as ImportError (its own ImportError is benign and swallowed).
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, slot, None)
    if cached is not None:
        return cached

    api_key = provider_env(env_var)
    if not api_key:
        raise ValueError(missing_key_error)

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure(feature, prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise ImportError(str(exc))

    client = factory(api_key)
    setattr(_wt, slot, client)
    return client


# --- Provider base ---


class BaseWebSearchProvider(WebSearchProvider):
    """Common surface for the bundled providers.

    Subclasses set ``NAME`` / ``DISPLAY_NAME`` / ``KEY_ENV`` and flip
    ``EXTRACT`` / ``KEYLESS``. ``is_available`` deliberately ignores the
    keyless tier: otherwise the legacy preference walk would route users
    holding a key for a lower-priority backend onto this vendor's free tier.
    ``is_keyless_available`` is True for ring members and opt-in keyless
    vendors unless the user pinned ``web.provider_tier.<name>: paid``.
    """

    NAME: str = ""
    DISPLAY_NAME: str = ""
    KEY_ENV: str = ""
    EXTRACT: bool = False
    KEYLESS: bool = False

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    def is_available(self) -> bool:
        return bool(provider_env(self.KEY_ENV))

    def is_keyless_available(self) -> bool:
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return self.KEYLESS and keyless_enabled() and provider_tier(self.NAME) != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return self.EXTRACT


def keyless_variant_schema(display: str, key_env: str, key_url: str, *, free_tag: str, paid_tag: str) -> Dict[str, Any]:
    """``hermes tools`` picker entry for a keyless-ring vendor with a paid variant."""
    return {
        "name": f"{display} · Free (keyless)",
        "badge": "free · no key",
        "tag": free_tag,
        "env_vars": [],
        "web_tier": "free",
        "variants": [
            {
                "name": f"{display} · Paid (API key)",
                "badge": "paid",
                "tag": paid_tag,
                "env_vars": [{"key": key_env, "prompt": f"{display} API key", "url": key_url}],
                "web_tier": "paid",
            },
        ],
    }
