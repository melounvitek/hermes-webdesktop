#!/usr/bin/env python3
"""Generic web_search / web_extract tools over pluggable backends.

Backend is selected during ``hermes tools`` (``web.backend`` in config.yaml; per
capability via ``web.search_backend`` / ``web.extract_backend``). Every vendor
implementation lives in ``plugins/web/<vendor>/provider.py`` and registers with
``agent.web_search_registry``; this module owns selection, safety gates,
caching, keyless rescue, and the truncate-and-store result pipeline.

Debug: ``WEB_TOOLS_DEBUG=true`` writes ``logs/web_tools_debug_<UUID>.json``.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import httpx  # noqa: F401 — kept at module top so tests can patch tools.web_tools.httpx

# Vendor helpers re-exported so external code and unit-test patches of
# ``tools.web_tools.<name>`` keep working after the plugin migration.
if TYPE_CHECKING:
    from firecrawl import Firecrawl  # noqa: F401 — type hints only
from plugins.web.firecrawl.provider import (  # noqa: F401 — backward-compat names
    Firecrawl,
    _firecrawl_backend_help_suffix,
    _get_firecrawl_client,
    _get_firecrawl_gateway_url,
    _is_tool_gateway_ready,
    check_firecrawl_api_key,
)
from plugins.web.tavily.provider import (  # noqa: F401 — backward-compat names
    _normalize_tavily_documents,
    _normalize_tavily_search_results,
    _tavily_request,
)
from plugins.web.parallel.provider import (  # noqa: F401 — backward-compat names
    _get_async_parallel_client,
    _get_parallel_client,
)
from plugins.web.exa.provider import _get_exa_client  # noqa: F401

# Per-vendor client cache slots. Plugins read/write these via tools.web_tools so
# tests that reset ``tools.web_tools._<vendor>_client = None`` keep working.
_firecrawl_client: Optional[Any] = None
_firecrawl_client_config: Optional[Any] = None
_parallel_client: Optional[Any] = None
_async_parallel_client: Optional[Any] = None
_exa_client: Optional[Any] = None

from tools.debug_helpers import DebugSession
from tools.managed_tool_gateway import (  # noqa: F401 — backward-compat names for tests
    build_vendor_gateway_url,
    peek_nous_access_token as _peek_nous_access_token,
    read_nous_access_token as _read_nous_access_token,
    resolve_managed_tool_gateway,
)
from tools.tool_backend_helpers import (  # noqa: F401
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    prefers_gateway,
)
from tools.url_safety import async_is_safe_url
from tools.web_tools_rescue import (  # noqa: F401 — re-exported (tests patch tools.web_tools.<name>)
    _keyless_rescue_enabled,
    _policy_blocked_result,
    _rescue_eligible,
    _rescue_extract,
    _rescue_search,
)
from tools.web_tools_truncate import (  # noqa: F401 — re-exported (tests + web_result_cache import via tools.web_tools)
    DEFAULT_EXTRACT_CHAR_LIMIT,
    MAX_STORED_TEXT_CHARS,
    _clamp_char_limit,
    _effective_char_limit,
    _get_extract_char_limit,
    _store_full_text,
    _trim_results,
    _truncate_results,
    _truncate_with_footer,
    convert_base64_images_to_links,
)
from tools.web_tools_extract import (  # noqa: F401 — re-exported
    _EXTRACT_BACKENDS_HINT,
    _NO_RESULT_ERROR,
    _disabled_plugin_error,
    _extract_error_json,
    _extract_safe_urls,
    _no_provider_error,
    _resolve_extract_provider,
    _result_entry,
    _strict_selection_error,
    _validate_extract_urls,
    _web_extract_url,
)
import sys

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _env_value(name: str) -> str:
    """Resolve ``name`` via the Hermes config-aware env layer, then process env.

    Values set through ``hermes config set`` / ``hermes tools`` must be honored
    by autodetect and ``check_web_api_key()``, not just raw exports.
    """
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


def _has_env(name: str) -> bool:
    return bool(_env_value(name))


def _load_web_config() -> dict:
    """Load the ``web:`` section from config.yaml; always a dict (a null section yields ``{}``)."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web") or {}
    except Exception:
        return {}


def _configured_backend(key: str = "backend") -> str:
    """Lower-cased, stripped ``web.<key>`` value ("" when unset/null)."""
    return (_load_web_config().get(key) or "").lower().strip()


# Built-in backends probed by the hardcoded checks in _BUILTIN_AVAILABILITY.
# Any other name is a plugin-registered provider resolved via the registry's
# ``is_available()``. Includes ``xai`` (probed via has_xai_credentials(), not a
# registered provider) even though the registry's _LEGACY_PREFERENCE omits it —
# if xai ever ships as a registered provider, drop it here.
_LEGACY_WEB_BACKENDS = frozenset(
    {"parallel", "firecrawl", "tavily", "exa", "searxng", "brave-free", "ddgs", "xai", "keenable"}
)


def _registered_web_provider(backend: str):
    """Plugin-registered web provider by name, or ``None`` (registry lookups are never fatal)."""
    if not backend:
        return None
    try:
        from agent.web_search_registry import get_provider

        return get_provider(backend)
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry lookup failed for %r: %s", backend, exc)
        return None


def _probe(provider, method: str, context: str = "") -> Optional[bool]:
    """``bool(provider.<method>())``, or ``None`` if it raised (logged; a broken provider is unavailable).

    ``context`` is appended to the debug log line (e.g. " during readiness check").
    """
    try:
        return bool(getattr(provider, method)())
    except Exception as exc:  # noqa: BLE001 — a broken provider is "unavailable"
        logger.debug(
            "web provider %r.%s() raised%s: %s",
            getattr(provider, "name", provider), method, context, exc,
        )
        return None


def _registered_web_provider_available(backend: str):
    """``is_available()`` of a registered provider, or ``None`` when unregistered (caller falls through)."""
    provider = _registered_web_provider(backend)
    if provider is None:
        return None
    return _probe(provider, "is_available") or False


def _list_registered_web_providers():
    """All plugin-registered web providers (empty list on failure)."""
    try:
        from agent.web_search_registry import list_providers

        return list_providers()
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry list failed: %s", exc)
        return []


def _get_backend() -> str:
    """Shared web backend name.

    A stored ``web.backend`` is returned as-is — no availability probe, no
    fallback — so a broken selection surfaces the vendor's honest error rather
    than silently rerouting. The autodetect ladder runs ONLY when no web
    selection has ever been stored.
    """
    configured = _configured_backend()
    if configured:
        # "nous" (managed subscription) is serviced by the firecrawl provider,
        # whose client resolver routes it through the managed Tool Gateway.
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER

        return "firecrawl" if configured == NOUS_MANAGED_PROVIDER else configured

    from tools.tool_backend_helpers import selection_exists

    if selection_exists("web"):
        # Selection exists (use_gateway / per-capability keys) but no shared
        # name: keep the firecrawl default rather than credential-laddering.
        return "firecrawl"

    # Never-configured install. Explicit user credentials beat the managed-
    # gateway probe (a Nous OAuth token's tier may not grant web access, and the
    # gateway then fails at runtime with no fallback). Free tiers trail paid.
    backend_candidates = (
        ("tavily", _has_env("TAVILY_API_KEY")),
        ("exa", _has_env("EXA_API_KEY")),
        ("parallel", _has_env("PARALLEL_API_KEY")),
        ("keenable", _has_env("KEENABLE_API_KEY")),
        ("firecrawl", _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL")),
        ("firecrawl", _is_tool_gateway_ready()),
        ("searxng", _has_env("SEARXNG_URL")),
        ("brave-free", _has_env("BRAVE_SEARCH_API_KEY")),
        ("ddgs", _ddgs_package_importable()),
    )
    for backend, available in backend_candidates:
        if available:
            return backend

    # Plugin-contributed providers (built-ins are covered above). We already
    # hold the provider object, so probe it directly instead of re-looking-up.
    for provider in _list_registered_web_providers():
        if provider.name not in _LEGACY_WEB_BACKENDS and _probe(provider, "is_available"):
            return provider.name

    # Keyless free tier — strictly last so it never pre-empts a keyed backend.
    # Discovery must run first: reachable from contexts that haven't loaded
    # plugins (subprocess agent runs, delegate children, scripts).
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import _keyless_preference, _keyless_tier_enabled

        if _keyless_tier_enabled():
            for name in _keyless_preference():
                provider = _registered_web_provider(name)
                if provider is not None and _probe(provider, "is_keyless_available"):
                    return name
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("keyless fallback walk failed: %s", exc)

    return "firecrawl"  # default (backward compat)


def _get_capability_backend(capability: str) -> str:
    """``web.{capability}_backend`` if stored (strict, no probe), else ``_get_backend()``."""
    return _configured_backend(f"{capability}_backend") or _get_backend()


def _get_search_backend() -> str:
    """Backend for web_search: ``web.search_backend`` > ``web.backend`` > autodetect."""
    return _get_capability_backend("search")


def _get_extract_backend() -> str:
    """Backend for web_extract: ``web.extract_backend`` > ``web.backend`` > autodetect."""
    return _get_capability_backend("extract")


def _tavily_explicitly_configured() -> bool:
    return any(
        _configured_backend(key) == "tavily"
        for key in ("backend", "search_backend", "extract_backend")
    )


def _xai_available() -> bool:
    # Cheap probe only (env var OR auth.json OAuth). resolve_xai_http_credentials()
    # can trigger a network token refresh and this runs on every dispatch.
    try:
        from tools.xai_http import has_xai_credentials
        return has_xai_credentials()
    except Exception:
        return False


def _ddgs_package_importable() -> bool:
    """ddgs is the only backend gated on package presence; single symbol so tests can patch it."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False


# Availability probes for the built-in backends (see _LEGACY_WEB_BACKENDS).
# Lambdas so tests patching module-level helpers (e.g. _ddgs_package_importable,
# check_firecrawl_api_key) are honored at call time.
_BUILTIN_AVAILABILITY = {
    "exa": lambda: _has_env("EXA_API_KEY"),
    "parallel": lambda: _has_env("PARALLEL_API_KEY"),
    "keenable": lambda: _has_env("KEENABLE_API_KEY"),
    "firecrawl": lambda: check_firecrawl_api_key(),
    "tavily": lambda: _has_env("TAVILY_API_KEY") or _tavily_explicitly_configured(),
    "searxng": lambda: _has_env("SEARXNG_URL"),
    "brave-free": lambda: _has_env("BRAVE_SEARCH_API_KEY"),
    "ddgs": lambda: _ddgs_package_importable(),
    "xai": _xai_available,
}


def _is_backend_available(backend: str) -> bool:
    """True when *backend* is usable — the single availability chokepoint.

    Non-legacy names delegate to the registered provider's ``is_available()``;
    built-ins use the cheap hardcoded probes.
    """
    backend = (backend or "").lower().strip()
    if backend not in _LEGACY_WEB_BACKENDS:
        registered = _registered_web_provider_available(backend)
        if registered is not None:
            return registered
    probe = _BUILTIN_AVAILABILITY.get(backend)
    return probe() if probe else False


def _web_requires_env() -> list[str]:
    """Tool-registry metadata env vars for the web backends.

    Gateway vars are always listed: gating them on ``managed_nous_tools_enabled()``
    cost a synchronous portal HTTP refresh at every CLI startup. Contract: set var
    -> tool sees it; not-logged-in users simply lack the vars, so extras are harmless.
    """
    return [
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "KEENABLE_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
    ]


# Truncate-and-store pipeline lives in tools/web_tools_truncate.py (re-imported above).
_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def _ensure_web_plugins_loaded() -> None:
    """Idempotently run plugin discovery so the web registry is populated.

    Dispatch is reachable from contexts that never triggered discovery
    (subprocess agent runs, delegate children, scripts); without it the
    registry is empty and a configured backend yields a misleading
    "No web ... provider configured" error.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        # Warning, not debug: a broken plugin import is otherwise invisible.
        logger.warning("Web plugin discovery failed (non-fatal): %s", exc)


def _finish_debug(call_name: str, debug_call_data: dict) -> None:
    _debug.log_call(call_name, debug_call_data)
    _debug.save()


def web_search_tool(query: str, limit: int = 5) -> str:
    """Search the web via the configured backend.

    Returns a JSON string ``{"success": bool, "data": {"web": [{"title", "url",
    "description", "position"}, ...]}}`` (metadata only — use web_extract_tool
    for page content) or ``{"success": false, "error": ...}``.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 100)

    debug_call_data = {
        "parameters": {"query": query, "limit": limit},
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0,
    }

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Sync only — every provider's search() is sync.
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_search_provider,
            get_provider as _wsp_get_provider,
        )

        backend = _get_search_backend()
        provider = _wsp_get_provider(backend) if backend else None
        if provider is None or not provider.supports_search():
            from tools.tool_backend_helpers import selection_exists

            if provider is None and backend and selection_exists("web"):
                error_text = _strict_selection_error("search", backend)
                debug_call_data["error"] = error_text
                _finish_debug("web_search_tool", debug_call_data)
                return json.dumps(
                    {"success": False, "error": error_text}, indent=2, ensure_ascii=False
                )
            # Never-configured install: legacy availability-walked autodetect.
            provider = get_active_search_provider()

        if provider is None:
            response_data = {
                "success": False,
                "error": _no_provider_error(
                    "search",
                    "No web search provider configured. Run `hermes tools` to set one up.",
                ),
            }
        else:
            logger.info(
                "Web search via %s: '%s' (limit: %d)",
                provider.name, query, limit,
            )
            response_data = _memoized_search(provider, query, limit)

        debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        debug_call_data["final_response_size"] = len(result_json)
        _finish_debug("web_search_tool", debug_call_data)
        return result_json

    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)
        debug_call_data["error"] = error_msg
        _finish_debug("web_search_tool", debug_call_data)
        return tool_error(error_msg)


def _memoized_search(provider, query: str, limit: int) -> dict:
    """TTL memo + single-flight around the paid vendor call (tools/web_result_cache.py).

    Sits after every safety/config check. The provider is asked for the
    BUCKETED count so near-identical limits share an entry; the caller's count
    is sliced out. Only successful, non-rescued responses are cached — caching
    a rescue would make the one-shot ring fallback sticky for a whole TTL.
    """
    from tools.web_result_cache import bucket_limit, search_memo, slice_search_response

    def _paid_search() -> tuple[dict, bool]:
        fetch_limit = bucket_limit(limit)
        try:
            resp = provider.search(query, fetch_limit)
        except Exception as exc:  # noqa: BLE001 — candidate for rescue
            if not _rescue_eligible(provider):
                raise
            return _rescue_search(provider.name, str(exc), query, fetch_limit), True
        if not resp.get("success") and _rescue_eligible(provider):
            return _rescue_search(
                provider.name, str(resp.get("error", "")), query, fetch_limit
            ), True
        return resp, False

    response_data = search_memo.lookup(provider.name, query, limit)
    if response_data is None:
        with search_memo.flight_lock(provider.name, query, limit):
            # Re-check inside the lock: a concurrent identical call may have stored.
            response_data = search_memo.lookup(provider.name, query, limit)
            if response_data is None:
                response_data, was_rescued = _paid_search()
                if not was_rescued:
                    search_memo.store(provider.name, query, limit, response_data)
    return slice_search_response(response_data, limit)


async def web_extract_tool(
    urls: List[Any],
    format: str = None,
    char_limit: Optional[int] = None,
) -> str:
    """Extract clean page content (no LLM) from URLs via the configured backend.

    Pages over ``char_limit`` (default web.extract_char_limit or 15000) are
    head+tail truncated with a footer pointing at the stored full text. Inline
    base64 images become ``[IMAGE: alt]`` placeholders. URLs carrying secrets
    are refused before any fetch; private-network URLs are blocked per entry.

    Returns a JSON string with a ``results`` list of ``url``/``title``/``content``/``error``.
    """
    normalized_urls, normalized_indices, invalid_urls, blocked = _validate_extract_urls(urls)
    if blocked is not None:
        return blocked

    debug_call_data = {
        "parameters": {"urls": normalized_urls, "format": format, "char_limit": char_limit},
        "error": None,
        "pages_extracted": 0,
        "pages_truncated": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "truncation_metrics": [],
        "processing_applied": [],
    }

    try:
        logger.info("Extracting content from %d URL(s)", len(normalized_urls))

        # SSRF protection — filter private/internal URLs before any backend.
        safe_urls = []
        safe_indices = []
        ssrf_blocked: Dict[int, Dict[str, Any]] = {}
        for index, url in zip(normalized_indices, normalized_urls):
            if not await async_is_safe_url(url):
                ssrf_blocked[index] = _result_entry(
                    url, "Blocked: URL targets a private or internal network address"
                )
            else:
                safe_urls.append(url)
                safe_indices.append(index)

        if not safe_urls:
            results = []
        else:
            backend = _get_extract_backend()
            _ensure_web_plugins_loaded()
            provider, error_json = _resolve_extract_provider(backend)
            if error_json is not None:
                return error_json
            results = await _extract_safe_urls(provider, safe_urls, format)

        # Reconstruct input order across invalid, blocked, and provider entries
        # (providers preserve the order of the safe URL list they receive).
        if invalid_urls or ssrf_blocked:
            safe_results = {
                index: (
                    results[position]
                    if position < len(results)
                    else _result_entry(safe_urls[position], _NO_RESULT_ERROR)
                )
                for position, index in enumerate(safe_indices)
            }
            by_index = {**safe_results, **ssrf_blocked, **invalid_urls}
            results = [by_index[index] for index in range(len(urls))]

        response = {"results": results}

        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)

        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))

        debug_call_data["processing_applied"].append("truncate_and_store")
        _truncate_results(response.get("results", []), _effective_char_limit(char_limit), debug_call_data)
        trimmed_response = {"results": _trim_results(response.get("results", []))}

        if trimmed_response.get("results") == []:
            result_json = tool_error("Content was inaccessible or not found")
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)

        # Belt-and-suspenders sweep over the serialized JSON in case a provider
        # tucked a base64 blob somewhere unexpected (e.g. metadata).
        cleaned_result = convert_base64_images_to_links(result_json)

        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_conversion")
        _finish_debug("web_extract_tool", debug_call_data)
        return cleaned_result

    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)
        debug_call_data["error"] = error_msg
        _finish_debug("web_extract_tool", debug_call_data)
        return tool_error(error_msg)


def _provider_is_ready(provider) -> bool:
    """True when *provider* is keyed-available OR keyless-capable, without raising.

    ``get_active_*_provider()`` returns an explicitly configured backend even
    when ``is_available()`` is False (so dispatch can emit a precise error), so
    readiness gates (tool check_fn, ``hermes doctor``) must probe for real.
    Keyless mode (Exa/Parallel free tier) is a working state, not a misconfig.
    """
    if provider is None:
        return False
    for method in ("is_available", "is_keyless_available"):
        ready = _probe(provider, method, " during readiness check")
        if ready is None:  # broken provider == not ready; don't try the next probe
            return False
        if ready:
            return True
    return False


def check_web_api_key() -> bool:
    """``check_fn`` gate for web_search / web_extract: is any web backend available?

    A plugin-registered provider reporting ``is_available()`` must light the
    tools up even with no built-in credentials; resolution funnels through
    :func:`_is_backend_available`.
    """
    configured = _configured_backend()
    if configured and _is_backend_available(configured):
        return True
    # Boolean OR over built-ins — probe order is irrelevant here.
    if any(_is_backend_available(backend) for backend in _LEGACY_WEB_BACKENDS):
        return True
    # Plugin path. Discovery must run first: check_fn fires at tool-registration
    # time, before any dispatch has populated the registry.
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_search_provider,
            get_active_extract_provider,
        )

        return (
            _provider_is_ready(get_active_search_provider())
            or _provider_is_ready(get_active_extract_provider())
        )
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry availability check failed: %s", exc)
        return False


_DEMO_BACKEND_LINES = {
    "exa": "   Using Exa API (https://exa.ai)",
    "parallel": "   Using Parallel API (https://parallel.ai)",
    "brave-free": "   Using Brave Search free tier (search only)",
    "ddgs": "   Using DuckDuckGo via ddgs package (search only)",
}

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)

    web_available = check_web_api_key()
    from hermes_cli.config import get_env_value as _gev

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend in _DEMO_BACKEND_LINES:
            print(_DEMO_BACKEND_LINES[backend])
        elif backend == "tavily":
            if _has_env("TAVILY_API_KEY"):
                print("   Using Tavily API (https://tavily.com)")
            else:
                print("   Using Tavily keyless (https://docs.tavily.com/documentation/keyless)")
        elif backend == "searxng":
            print(f"   Using SearXNG (search only): {_env_value('SEARXNG_URL')}")
        elif (_gev("FIRECRAWL_API_URL") or "").strip():
            print(f"   Using self-hosted Firecrawl: {(_gev('FIRECRAWL_API_URL') or '').strip().rstrip('/')}")
        elif (_gev("FIRECRAWL_API_KEY") or "").strip():
            print("   Using direct Firecrawl cloud API")
        elif _is_tool_gateway_ready():
            print(f"   Using Firecrawl tool-gateway: {_get_firecrawl_gateway_url()}")
        else:
            print("   Firecrawl backend selected but not configured")
    else:
        print("❌ No web search backend configured")
        print(
            "Set EXA_API_KEY, PARALLEL_API_KEY, TAVILY_API_KEY, KEENABLE_API_KEY, FIRECRAWL_API_KEY, FIRECRAWL_API_URL"
            f"{_firecrawl_backend_help_suffix()}"
        )
        sys.exit(1)

    print("🛠️  Web tools ready for use!")
    print(f"   Extract char limit: {_get_extract_char_limit()} chars "
          "(pages over this are truncated; full text stored in cache/web)")

    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: {_debug.log_dir}/web_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set WEB_TOOLS_DEBUG=true to enable)")

    print("\nBasic usage:")
    print("  from web_tools import web_search_tool, web_extract_tool")
    print("  import asyncio")
    print("")
    print("  # Search (synchronous)")
    print("  results = web_search_tool('Python tutorials')")
    print("")
    print("  # Extract (asynchronous, no LLM — truncate-and-store)")
    print("  async def main():")
    print("      content = await web_extract_tool(['https://example.com'])")
    print("      # bigger budget for one call:")
    print("      content = await web_extract_tool(['https://docs.python.org'], char_limit=40000)")
    print("  asyncio.run(main())")

    print("\nDebug mode:")
    print("  export WEB_TOOLS_DEBUG=true")
    print("  # Logs saved to: ./logs/web_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 100,
                "default": 5
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns clean page content in markdown/text (no LLM summarization — fast). Also works with PDF URLs (arxiv papers, documents) — pass the PDF link directly. Pages within the char budget (default 15000) return whole; larger pages return a head+tail window with a footer telling you the full text's saved file path and the read_file call to page through the omitted middle. Inline images appear as [IMAGE: alt] placeholders; real image URLs are kept as links. If a URL fails or times out, use the browser tool instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            },
            "char_limit": {
                "type": "integer",
                "description": "Optional per-page character budget sent back (default 15000). Pages larger than this are head+tail truncated with the full text stored to disk. Raise it when you need more of a long page inline.",
                "minimum": 2000
            }
        },
        "required": ["urls"]
    }
}

registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=args.get("limit", 5)),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [],
        "markdown",
        char_limit=args.get("char_limit"),
    ),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    is_async=True,
    emoji="📄",
    max_result_size_chars=100_000,
)
