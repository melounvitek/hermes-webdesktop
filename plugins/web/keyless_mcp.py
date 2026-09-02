"""Keyless web search/extract via public free-tier endpoints (Exa, Parallel, Firecrawl, Keenable).

Resolved strictly LAST — after every keyed backend, the managed gateway, ddgs and
custom plugin providers — so it never pre-empts a deliberate setup. Privacy: no user
identifiers are sent; Parallel gets a random per-process ``session_id`` (rate
limiting only) and its optional ``model_name`` analytics field is deliberately
omitted. Disable the tier with ``web.keyless_fallback: false``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from plugins.web._common import document as _page, search_fail, search_ok, web_hit as _row

logger = logging.getLogger(__name__)

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
KEENABLE_API_URL = "https://api.keenable.ai"
_KEENABLE_TITLE = "hermes-agent"

# Parallel free-tier rate-limit correlation id — random per process, never persisted.
_SESSION_ID = uuid.uuid4().hex

_TIMEOUT_SECONDS = 30


class KeylessMCPError(RuntimeError):
    """A keyless MCP call failed (transport, rate limit, or tool error)."""


_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "ratelimit", "too many requests", "429", "quota exceeded", "slow down")

# vendor -> (display label, env key, signup URL) for the standard failure hint.
_VENDOR_HINTS = {
    "exa": ("Exa", "EXA_API_KEY", "https://exa.ai"),
    "parallel": ("Parallel", "PARALLEL_API_KEY", "https://parallel.ai"),
    "firecrawl": ("Firecrawl", "FIRECRAWL_API_KEY", "https://firecrawl.dev"),
    "keenable": ("Keenable", "KEENABLE_API_KEY", "https://keenable.ai"),
}


def _is_rate_limitish(message: str) -> bool:
    """Heuristic: does an error message look like free-tier throttling?"""
    return any(marker in (message or "").lower() for marker in _RATE_LIMIT_MARKERS)


def _fail_msg(vendor: str, kind: str, exc: Any, *, other_backends: bool = True) -> str:
    label, env_key, site = _VENDOR_HINTS[vendor]
    alt = " or another web backend via `hermes tools`" if other_backends else ""
    return f"Keyless {label} {kind} failed: {exc}. Set {env_key} ({site}){alt} for reliable service."


def _page_error(url: str, message: str) -> Dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": message}


# --- Tier / config ------------------------------------------------------------


def keyless_enabled() -> bool:
    """True when the keyless tier is enabled. Delegates to the registry so the
    ``web.keyless_fallback`` (default on) chokepoint lives with backend resolution."""
    try:
        from agent.web_search_registry import _keyless_tier_enabled

        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — resolver optional in stripped envs
        logger.debug("keyless_enabled(): registry helper unavailable: %s", exc)
        return True


def provider_tier(name: str) -> str:
    """Return ``web.provider_tier.<name>`` (set by the ``hermes tools`` Free/Paid rows):
    ``free``, ``paid``, or ``auto`` for anything else including unset."""
    try:
        from hermes_cli.config import load_config

        web_cfg = load_config().get("web") or {}
        tiers = web_cfg.get("provider_tier") or {}
        value = str(tiers.get(name, "") or "").lower().strip()
        return value if value in ("free", "paid") else "auto"
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("provider_tier(%r) config read failed: %s", name, exc)
        return "auto"


def use_keyless(name: str, api_key: str) -> bool:
    """Single chokepoint for search + extract so tier semantics can't drift:
    ``free`` → keyless even with a key; ``paid`` → keyed even without one (the keyed
    path raises its usual missing-key error); ``auto`` → keyless only when no key
    and the tier is enabled."""
    tier = provider_tier(name)
    if tier == "free":
        return True
    if tier == "paid":
        return False
    return not api_key and keyless_enabled()


# --- MCP transport ------------------------------------------------------------


def _parse_mcp_body(body: str) -> str:
    """Extract the first text content item from an MCP tools/call response.

    Handles plain-JSON bodies (Parallel) and SSE ``data: {...}`` lines (Exa). Raises
    :class:`KeylessMCPError` for JSON-RPC errors and ``isError`` tool results (e.g.
    Exa's free-tier rate-limit message).
    """

    def _from_payload(payload: str) -> Optional[str]:
        payload = payload.strip()
        if not payload.startswith("{"):
            return None
        data = json.loads(payload)
        err = data.get("error")
        if err:
            raise KeylessMCPError(str(err.get("message") or err))
        result = data.get("result") or {}
        content = result.get("content") or []
        if result.get("isError"):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            raise KeylessMCPError(" ".join(t for t in texts if t) or "MCP tool call failed")
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                return str(item["text"])
        return None

    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            text = _from_payload(stripped)
            if text is not None:
                return text
        except json.JSONDecodeError:
            pass

    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            text = _from_payload(line[len("data: "):])
        except json.JSONDecodeError:
            continue
        if text is not None:
            return text

    raise KeylessMCPError("Unrecognized MCP response shape")


def mcp_call(url: str, tool: str, arguments: Dict[str, Any], timeout: int = _TIMEOUT_SECONDS) -> str:
    """POST a JSON-RPC ``tools/call`` to *url* and return the text payload.

    Raises :class:`KeylessMCPError` on transport failures, non-2xx statuses,
    JSON-RPC errors, and error-shaped tool results.
    """
    import requests

    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "hermes-agent",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise KeylessMCPError(f"HTTP {response.status_code}: {response.text[:300]}")
    return _parse_mcp_body(response.text)


# --- Parallel (search.parallel.ai) — JSON text payloads -----------------------


def parallel_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Parallel web search → legacy search response shape."""
    try:
        text = mcp_call(
            PARALLEL_MCP_URL, "web_search",
            {"objective": query, "search_queries": [query], "session_id": _SESSION_ID},
        )
        data = json.loads(text)
        web_results = []
        for i, result in enumerate(data.get("results") or []):
            if limit and i >= limit:
                break
            web_results.append(_row(
                result.get("url") or "", result.get("title") or "",
                " ".join(result.get("excerpts") or []), i + 1,
            ))
        return search_ok(web_results)
    except KeylessMCPError as exc:
        return search_fail(_fail_msg("parallel", "search", exc))
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        return search_fail(f"Keyless Parallel search returned an unexpected payload: {exc}")


def parallel_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Parallel web fetch → legacy extract result list."""
    try:
        text = mcp_call(
            PARALLEL_MCP_URL, "web_fetch",
            {"urls": list(urls), "objective": "Full page content", "session_id": _SESSION_ID},
        )
        data = json.loads(text)
    except (KeylessMCPError, json.JSONDecodeError, TypeError) as exc:
        message = _fail_msg("parallel", "extract", exc)
        return [_page_error(u, message) for u in urls]

    results: List[Dict[str, Any]] = []
    seen = set()
    for result in data.get("results") or []:
        url = result.get("url") or ""
        content = result.get("full_content") or result.get("content") or "\n\n".join(result.get("excerpts") or [])
        seen.add(url)
        results.append(_page(url, result.get("title") or "", content))
    for error in data.get("errors") or []:
        url = error.get("url") or ""
        seen.add(url)
        entry = _page_error(url, str(error.get("content") or error.get("error_type") or "extraction failed"))
        entry["metadata"] = {"sourceURL": url}
        results.append(entry)
    # URLs the endpoint silently dropped still get an error entry (per-URL contract).
    results.extend(_page_error(u, "no content returned") for u in urls if u not in seen)
    return results


# --- Exa (mcp.exa.ai) — formatted plain-text payloads -------------------------


def _parse_exa_search_text(text: str, limit: int) -> List[Dict[str, Any]]:
    """Parse Exa's ``---``-separated ``Title:/URL:/Published:/Author:/Highlights:`` blocks."""
    results: List[Dict[str, Any]] = []
    for block in text.split("\n---\n"):
        title = url = ""
        highlight_lines: List[str] = []
        in_highlights = False
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("Title:"):
                title = stripped[len("Title:"):].strip()
                in_highlights = False
            elif stripped.startswith("URL:"):
                url = stripped[len("URL:"):].strip()
                in_highlights = False
            elif stripped.startswith("Highlights:"):
                in_highlights = True
            elif stripped.startswith(("Published:", "Author:")):
                in_highlights = False
            elif in_highlights and stripped:
                highlight_lines.append(stripped)
        if url:
            results.append(_row(url, title, " ".join(highlight_lines), len(results) + 1))
        if limit and len(results) >= limit:
            break
    return results


def exa_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Exa web search → legacy search response shape."""
    try:
        text = mcp_call(EXA_MCP_URL, "web_search_exa", {"query": query, "numResults": max(1, int(limit))})
    except KeylessMCPError as exc:
        return search_fail(_fail_msg("exa", "search", exc))
    return search_ok(_parse_exa_search_text(text, limit))


def exa_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Exa web fetch → legacy extract result list (called per-URL; the
    tool returns one combined text payload)."""
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            text = mcp_call(EXA_MCP_URL, "web_fetch_exa", {"urls": [url]})
        except KeylessMCPError as exc:
            results.append(_page_error(url, _fail_msg("exa", "extract", exc)))
            continue
        title = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
            if stripped.startswith("Title:"):
                title = stripped[len("Title:"):].strip()
                break
        results.append(_page(url, title, text))
    return results


# --- Firecrawl keyless (public cloud API, no auth header) ---------------------


def firecrawl_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Firecrawl cloud search → legacy search response shape."""
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_web_search_results

    try:
        response = _KeylessFirecrawlClient().search(query=query, limit=limit)
        return search_ok(_extract_web_search_results(response))
    except Exception as exc:  # noqa: BLE001 — normalized below
        return search_fail(_fail_msg("firecrawl", "search", exc))


def firecrawl_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Firecrawl cloud scrape → legacy extract result list."""
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_scrape_payload

    client = _KeylessFirecrawlClient()
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            payload = _extract_scrape_payload(client.scrape(url=url, formats=["markdown"])) or {}
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            content = payload.get("markdown") or payload.get("html") or ""
            results.append(_page(url, metadata.get("title") or "", content))
        except Exception as exc:  # noqa: BLE001 — per-URL error entry
            results.append(_page_error(url, _fail_msg("firecrawl", "extract", exc, other_backends=False)))
    return results


# --- Keenable keyless (api.keenable.ai public endpoints) ----------------------


def _keenable_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    """Call a Keenable public endpoint with the mandatory X-Keenable-Title app id."""
    import requests

    headers = {"X-Keenable-Title": _KEENABLE_TITLE}
    if method == "post":
        headers["Content-Type"] = "application/json"
    response = getattr(requests, method)(f"{KEENABLE_API_URL}{path}", headers=headers, timeout=_TIMEOUT_SECONDS, **kwargs)
    if response.status_code >= 400:
        raise KeylessMCPError((response.text or "").strip() or f"HTTP {response.status_code}")
    return response.json()


def keenable_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Keenable search (POST /v1/search/public) → legacy search response shape."""
    try:
        data = _keenable_request("post", "/v1/search/public", json={"query": query, "max_results": max(1, int(limit))})
    except KeylessMCPError as exc:
        return search_fail(_fail_msg("keenable", "search", exc))
    except Exception as exc:  # noqa: BLE001 — transport/JSON errors
        return search_fail(f"Keyless Keenable search failed: {exc}.")
    return search_ok([
        _row(r.get("url") or "", r.get("title") or "", r.get("snippet") or r.get("description") or "", i + 1)
        for i, r in enumerate(data.get("results") or [])
    ])


def keenable_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Keenable page fetch (GET /v1/fetch/public, per-URL) → legacy extract list."""
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            data = _keenable_request("get", "/v1/fetch/public", params={"url": url})
            results.append(_page(data.get("url") or url, data.get("title") or "", data.get("content") or "", source_url=url))
        except Exception as exc:  # noqa: BLE001 — per-URL error entry
            results.append(_page_error(url, _fail_msg("keenable", "extract", exc, other_backends=False)))
    return results


# --- Round-robin ring + next-in-line failover (rate-limited free tiers) -------

_KEYLESS_RING = ("exa", "parallel", "firecrawl", "keenable")

# Late-bound lookups (not bare references) so ``patch.object(keyless_mcp, "<vendor>_search_keyless")``
# is honored at call time. Tests also ``setitem`` these dicts directly.
_KEYLESS_SEARCHERS: Dict[str, Callable[[str, int], Dict[str, Any]]] = {
    v: (lambda query, limit, _v=v: globals()[f"{_v}_search_keyless"](query, limit)) for v in _KEYLESS_RING
}
_KEYLESS_EXTRACTORS: Dict[str, Callable[[List[str]], List[Dict[str, Any]]]] = {
    v: (lambda urls, _v=v: globals()[f"{_v}_extract_keyless"](urls)) for v in _KEYLESS_RING
}

# Per-process round-robin cursor, seeded by the random session id so the fleet
# spreads across vendors; advances once per unpinned keyless request.
_ring_lock = threading.Lock()
_ring_cursor = int(_SESSION_ID, 16) % len(_KEYLESS_RING)


def _vendor_pinned(name: str) -> bool:
    """True when config explicitly routes web traffic to *name* (backend keys or a
    ``free`` tier pin). A pinned vendor starts every keyless request."""
    if provider_tier(name) == "free":
        return True
    try:
        import tools.web_tools as _wt

        web_cfg = _wt._load_web_config()
        return any(
            (web_cfg.get(key) or "").lower().strip() == name
            for key in ("backend", "search_backend", "extract_backend")
        )
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("_vendor_pinned(%r) config read failed: %s", name, exc)
        return False


def _ring_order(name: str) -> List[str]:
    """Vendor walk order: pinned → start at *name* (its ring position fixes the
    failover succession); else round-robin from the cursor, advancing it per request.
    Vendors pinned ``paid`` are excluded (explicit paid opts their free endpoint out)."""
    global _ring_cursor
    if _vendor_pinned(name):
        start = _KEYLESS_RING.index(name) if name in _KEYLESS_RING else 0
    else:
        with _ring_lock:
            start = _ring_cursor
            _ring_cursor = (_ring_cursor + 1) % len(_KEYLESS_RING)
    ordered = _KEYLESS_RING[start:] + _KEYLESS_RING[:start]
    return [v for v in ordered if provider_tier(v) != "paid"]


_ALL_PAID_MSG = "All keyless web providers are pinned to paid tiers."


def _walk_ring(name: str, kind: str, call, throttled) -> tuple:
    """Shared ring walk: call each vendor from :func:`_ring_order` until a result is
    not ``throttled``. Returns ``(order, vendor, result, exhausted)``; ``order`` is
    empty (and result None) when every vendor is pinned paid."""
    order = _ring_order(name)
    vendor = None
    result: Any = None
    for i, vendor in enumerate(order):
        result = call(vendor)
        if not throttled(result):
            return order, vendor, result, False
        if i + 1 < len(order):
            logger.info("keyless %s %s throttled; failing over to %s", vendor, kind, order[i + 1])
    return order, vendor, result, True


def search_with_failover(name: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless search across the ring; rate-limit-shaped errors advance to the
    next vendor, other errors stop the walk (a malformed query fails everywhere).
    ``data.served_by`` is set when the serving vendor differs from *name*."""

    def _throttled(result: Dict[str, Any]) -> bool:
        return not result.get("success") and _is_rate_limitish(result.get("error", ""))

    order, vendor, result, exhausted = _walk_ring(
        name, "search", lambda v: _KEYLESS_SEARCHERS[v](query, limit), _throttled
    )
    if not order:
        return search_fail(_ALL_PAID_MSG)
    if exhausted:
        result["error"] = f"{result.get('error', '')} (all keyless vendors throttled: {', '.join(order)})"
    elif result.get("success") and vendor != name:
        result.setdefault("data", {})["served_by"] = vendor
    return result


def extract_with_failover(name: str, urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless extract across the ring; fails over only when EVERY url in a batch
    is rate-limit-shaped (partial failures are page problems, returned as-is)."""

    def _all_throttled(results: List[Dict[str, Any]]) -> bool:
        errors = [r.get("error", "") for r in results]
        return bool(results) and all(e and _is_rate_limitish(e) for e in errors)

    order, _vendor, results, _exhausted = _walk_ring(
        name, "extract", lambda v: _KEYLESS_EXTRACTORS[v](list(urls)), _all_throttled
    )
    if not order:
        return [_page_error(u, _ALL_PAID_MSG) for u in urls]
    return results
