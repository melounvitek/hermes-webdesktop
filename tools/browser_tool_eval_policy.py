"""browser_console(expression=...) policy: SSRF guard gating, private-URL probes,
and the opt-in sensitive-primitive denylist.

Origin-module symbols are resolved lazily through ``tools.browser_tool`` (``_bt``)
so ``patch("tools.browser_tool.X")`` in tests keeps working; this module must not
import ``tools.browser_tool`` at import time (cycle).
"""

import re
from typing import Optional
from tools.browser_tool_origin import origin_module as _origin


def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Return True when eval-driven private-network access must be guarded.

    Matches the gating used by ``browser_navigate`` / ``browser_snapshot`` /
    ``browser_vision``: the SSRF guard is only meaningful for non-local
    backends (cloud browser, or a containerized terminal whose browser-on-host
    can reach internal networks the terminal can't), and is skipped for local
    sidecar sessions and when ``allow_private_urls`` is set.
    """
    _bt = _origin()
    return (
        not _bt._is_local_backend()
        and not _bt._is_local_sidecar_key(effective_task_id)
        and not _bt._allow_private_urls()
    )


# URL-shaped literals embedded in a JS expression (http/https only).  Used to
# pre-screen ``browser_console(expression=...)`` calls that fetch/XHR/navigate
# to a private host directly — that path never updates ``location.href`` so the
# post-eval page-URL recheck below can't see it.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


def _expression_targets_private_url(expression: str) -> Optional[str]:
    """Return the first private/always-blocked URL literal in a JS expression.

    Best-effort: scans for ``http(s)://...`` literals (fetch/XHR/navigation
    targets the agent may have embedded) and returns the first one that targets
    a private/internal address or the always-blocked cloud-metadata floor.
    Returns ``None`` when no such literal is found.
    """
    _bt = _origin()
    if not isinstance(expression, str):
        return None
    for match in _JS_URL_LITERAL_RE.findall(expression):
        candidate = match.rstrip(".,;")
        if _bt._is_always_blocked_url(candidate) or not _bt._is_safe_url(candidate):
            return candidate
    return None


def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Return the current page URL when it targets a private/internal address.

    Reads ``window.location.href`` via a low-cost eval and returns it when the
    page has been navigated (e.g. via ``location.href = '...'`` in a prior
    eval) to an address the SSRF guard would reject.  Returns ``None`` when the
    page is public, the URL can't be determined, or the check errors (fail-open
    on probe failure, matching the snapshot/vision guards).
    """
    _bt = _origin()
    try:
        url_result = _bt._run_browser_command(
            effective_task_id, "eval", ["window.location.href"],
            timeout=5, _engine_override="auto",
        )
        if url_result.get("success"):
            current_url = (
                url_result.get("data", {}).get("result", "")
                .strip().strip('"').strip("'")
            )
            if current_url and (
                _bt._is_always_blocked_url(current_url) or not _bt._is_safe_url(current_url)
            ):
                return current_url
    except Exception as exc:
        _bt.logger.debug("_current_page_private_url: probe failed (%s)", exc)
    return None


_RISKY_BROWSER_EVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocument\s*\.\s*cookie\b", re.I), "document.cookie"),
    (re.compile(r"\b(?:localStorage|sessionStorage)\b", re.I), "web storage"),
    (re.compile(r"\bindexedDB\b", re.I), "IndexedDB"),
    (re.compile(r"\bcaches\s*\.\s*(?:open|match|keys)\b", re.I), "Cache Storage"),
    (re.compile(r"\bnavigator\s*\.\s*(?:clipboard|credentials|serviceWorker)\b", re.I), "navigator sensitive API"),
    (re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", re.I), "network request"),
    (re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.I), "network beacon"),
    (re.compile(r"\bdocument\s*\.\s*forms\b.*\bvalue\b", re.I | re.S), "form value extraction"),
    (re.compile(r"\bquerySelector(?:All)?\s*\([^)]*(?:input|textarea|password)[^)]*\).*\bvalue\b", re.I | re.S), "form value extraction"),
)


_JS_STRING_LITERAL_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`""",
    re.S,
)


_SENSITIVE_BROWSER_EVAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("cookie", "document.cookie"),
    ("localStorage", "web storage"),
    ("sessionStorage", "web storage"),
    ("indexedDB", "IndexedDB"),
    ("caches", "Cache Storage"),
    ("clipboard", "navigator sensitive API"),
    ("credentials", "navigator sensitive API"),
    ("serviceWorker", "navigator sensitive API"),
    ("fetch", "network request"),
    ("XMLHttpRequest", "network request"),
    ("WebSocket", "network request"),
    ("EventSource", "network request"),
    ("sendBeacon", "network beacon"),
)


def _allow_unsafe_browser_evaluate() -> bool:
    """Return whether sensitive browser JS evaluation is explicitly allowed.

    When true, ``browser_console(expression=...)`` runs without the
    sensitive-primitive denylist even if ``browser.restrict_evaluate`` is set.
    """
    _bt = _origin()
    return _bt._browser_cfg(
        "allow_unsafe_evaluate", False,
        lambda v: _bt.is_truthy_value(v, default=False),
        "browser.allow_unsafe_evaluate from config",
    )


def _restrict_browser_evaluate() -> bool:
    """Return whether the sensitive-primitive eval denylist is enabled.

    Off by default. ``browser_console(expression=...)`` is the agent's only
    programmatic page-inspection path, and the denylist blocks the *names* of
    common primitives (``fetch``, ``cookie``, ``querySelector(...input...)``)
    rather than any actual exfiltration — which also blocks a large class of
    legitimate DOM extraction (any selector or page script text containing
    those words). Egress itself is still gated by the SSRF/private-URL guards
    in ``_browser_eval`` regardless of this setting. Users who want the
    strict vocabulary denylist (e.g. when browsing hostile pages with a
    logged-in profile) opt in with ``browser.restrict_evaluate: true``;
    ``browser.allow_unsafe_evaluate: true`` overrides it back off.
    """
    _bt = _origin()
    return _bt._browser_cfg(
        "restrict_evaluate", False,
        lambda v: _bt.is_truthy_value(v, default=False),
        "browser.restrict_evaluate from config",
    )


def _decode_js_string_literal(literal: str) -> str:
    """Best-effort decode of a JavaScript string literal for policy checks.

    This is not a JS parser.  It only normalizes common escaped property names
    such as ``document["co\\x6fkie"]`` before the fail-closed sensitive-token
    check below.
    """
    if len(literal) < 2:
        return literal
    body = literal[1:-1]
    try:
        return bytes(body, "utf-8").decode("unicode_escape")
    except Exception:
        return body


def _decoded_js_string_literals(expression: str) -> list[str]:
    return [_decode_js_string_literal(match.group(0)) for match in _JS_STRING_LITERAL_RE.finditer(expression)]


def _sensitive_browser_eval_token_reason(expression: str) -> Optional[str]:
    """Return a risk reason for direct or quoted sensitive browser primitives.

    ``browser_console(expression=...)`` executes in the page origin.  A denylist
    that only searches direct spellings like ``document.cookie`` and ``fetch(``
    misses equivalent JavaScript property access such as ``document["cookie"]``
    or ``globalThis["fetch"](...)``.  Treat sensitive primitive names as risky
    whether they appear as identifiers or decoded string-literal property names.
    Concatenating all string literals catches simple obfuscations like
    ``document["coo" + "kie"]`` while the config opt-in preserves the escape
    hatch for trusted pages.
    """
    string_literals = _decoded_js_string_literals(expression)
    concatenated_literals = "".join(string_literals).lower()
    for token, reason in _SENSITIVE_BROWSER_EVAL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", expression, re.I):
            return reason
        token_lower = token.lower()
        if any(token_lower in literal.lower() for literal in string_literals):
            return reason
        if token_lower in concatenated_literals:
            return reason
    return None


def _risky_browser_eval_reason(expression: str) -> Optional[str]:
    """Return a human-readable reason if a JS expression uses risky primitives."""
    if not expression:
        return None
    for pattern, reason in _RISKY_BROWSER_EVAL_PATTERNS:
        if pattern.search(expression):
            return reason
    return _sensitive_browser_eval_token_reason(expression)


def _enforce_browser_eval_policy(expression: str) -> Optional[str]:
    """Block sensitive browser JS evaluation when the opt-in denylist is on.

    The denylist is opt-in (``browser.restrict_evaluate: true``) because it
    gates on primitive *names*, which cripples legitimate DOM extraction —
    see ``_restrict_browser_evaluate``. Network egress to private/internal
    addresses is enforced separately in ``_browser_eval`` and does not depend
    on this policy.
    """
    _bt = _origin()
    if not _bt._restrict_browser_evaluate():
        return None
    if _bt._allow_unsafe_browser_evaluate():
        return None
    reason = _risky_browser_eval_reason(expression)
    if not reason:
        return None
    return (
        "Blocked: browser_console(expression=...) tried to use sensitive browser "
        f"JavaScript primitive ({reason}) while browser.restrict_evaluate is "
        "enabled. Use browser_snapshot/browser_get_images/browser_console "
        "without expression for normal inspection, or set "
        "browser.restrict_evaluate: false in config.yaml to allow "
        "programmatic evaluation."
    )


def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Return the Camofox page URL when it targets a private/internal address.

    Camofox analogue of ``_current_page_private_url`` (evaluate endpoint instead
    of the agent-browser CLI).  Returns ``None`` when the page is public, the URL
    can't be determined, or the probe errors (fail-open on probe failure,
    matching the snapshot/vision guards — do not change to fail-closed without
    also changing the sibling).
    """
    _bt = _origin()
    try:
        from tools.browser_camofox import _post

        data = _post(
            f"/tabs/{tab_id}/evaluate",
            body={"expression": "window.location.href", "userId": user_id},
        )
        current_url = str(data.get("result") if isinstance(data, dict) else data or "")
        current_url = current_url.strip().strip('"').strip("'")
        if current_url and (_bt._is_always_blocked_url(current_url) or not _bt._is_safe_url(current_url)):
            return current_url
    except Exception as exc:
        _bt.logger.debug("_camofox_current_page_private_url: probe failed (%s)", exc)
    return None
