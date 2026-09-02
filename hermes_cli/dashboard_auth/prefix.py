"""X-Forwarded-Prefix and public-URL resolution for reverse-proxied deploys.

Mission-control style deploys proxy the dashboard at a path prefix and inject
``X-Forwarded-Prefix: /hermes`` so the backend can build prefixed URLs
(Location headers, OAuth redirect_uri, cookie Path, SPA asset URLs). When the
operator declares a complete ``HERMES_DASHBOARD_PUBLIC_URL`` /
``dashboard.public_url`` instead, that is used verbatim for the OAuth
redirect_uri (relief valve for unreliable proxy header chains). Single source
of truth so the gate, routes, cookies and SPA mount agree on validation.
"""
from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Optional

_log = logging.getLogger(__name__)

# Home Assistant ingress prefixes are already 63 chars before a deployment adds
# its own sub-path; keep a bounded header budget with room for real mounts.
_MAX_PREFIX_LENGTH = 256

# Presence of any of these in a public_url / prefix means a typo or a
# header-injection attempt: reject the whole value, never sanitise.
_REJECT_CHARS = frozenset(('"', "'", "<", ">", " ", "\n", "\r", "\t"))

# ``resolve_public_url`` runs on every authenticated request, so warnings are
# de-duplicated per distinct (source, value) — a changed value warns afresh.
_warned_malformed_public_urls: set = set()
_warned_malformed_prefixes: set = set()


def _warn_if_malformed(source: str, raw: str) -> None:
    """Warn once when a non-empty public-url value was rejected.

    Almost always a missing scheme; without this the value is silently
    discarded and the OAuth callback falls back to header reconstruction,
    which behind a proxy can yield the wrong scheme.
    """
    cleaned = raw.strip() if raw else ""
    if not cleaned:
        return  # empty/unset is a legitimate "no override"
    key = (source, cleaned)
    if key in _warned_malformed_public_urls:
        return
    _warned_malformed_public_urls.add(key)
    _log.warning(
        "%s is set to %r but was ignored because it is not a valid "
        "absolute URL — it must include an http:// or https:// scheme "
        "(e.g. https://%s). Falling back to reconstructing the OAuth "
        "redirect URI from request headers, which may produce the wrong "
        "scheme behind a reverse proxy.",
        source,
        cleaned,
        cleaned.split("://")[-1] or "hermes.example.com",
    )


def _warn_if_malformed_prefix(raw: Optional[str], reason: str) -> None:
    """Warn once when a non-empty X-Forwarded-Prefix value is rejected."""
    cleaned = raw.strip() if raw else ""
    if not cleaned:
        return
    key = (cleaned, reason)
    if key in _warned_malformed_prefixes:
        return
    _warned_malformed_prefixes.add(key)
    _log.warning(
        "X-Forwarded-Prefix header %r was ignored because %s. "
        "Dashboard URLs will be generated without a reverse-proxy path prefix.",
        cleaned,
        reason,
    )


def normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header to ``"/hermes"`` form (no trailing
    slash) or ``""`` when unset/malformed. ``..``, ``//`` and injection
    characters are rejected so a hostile proxy cannot smuggle HTML or traversal.
    """
    p = raw.strip() if raw else ""
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")
    if "//" in p or ".." in p or any(c in p for c in _REJECT_CHARS):
        _warn_if_malformed_prefix(
            raw, "it contains a disallowed character or path sequence",
        )
        return ""
    if len(p) > _MAX_PREFIX_LENGTH:
        _warn_if_malformed_prefix(
            raw, f"it is longer than {_MAX_PREFIX_LENGTH} characters",
        )
        return ""
    return p


def prefix_from_request(request) -> str:
    """Normalised ``X-Forwarded-Prefix`` from a Starlette request, or ``""``."""
    return normalise_prefix(request.headers.get("x-forwarded-prefix"))


# ---------------------------------------------------------------------------
# HERMES_DASHBOARD_PUBLIC_URL / dashboard.public_url
# ---------------------------------------------------------------------------


def _normalise_public_url(raw: Optional[str]) -> str:
    """Cleaned ``scheme://netloc[/path]`` (trailing slash stripped so callers
    can append paths) or ``""`` when empty/malformed/injection-suspect. Callers
    must treat ``""`` as "fall back to request reconstruction" — an explicit
    empty value is indistinguishable from an unset env var.
    """
    url = raw.strip() if raw else ""
    if not url or any(c in url for c in _REJECT_CHARS):
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url.rstrip("/")


def _load_dashboard_section() -> dict:
    """``dashboard`` block of config.yaml as a dict, or ``{}`` when the config
    cannot be loaded or the block is absent/non-dict."""
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        _log.debug(
            "dashboard-auth.prefix: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg.get("dashboard") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def resolve_public_url() -> str:
    """Operator-declared dashboard public URL, or ``""`` (reconstruct from request).

    Precedence: ``HERMES_DASHBOARD_PUBLIC_URL`` env (empty-after-strip counts
    as unset so a provisioned-but-blank secret cannot shadow config.yaml), then
    ``dashboard.public_url``. A malformed value at either level warns and falls
    through to the next, so a typo in one surface never disables the other.
    """
    env_raw = os.environ.get("HERMES_DASHBOARD_PUBLIC_URL", "")
    env_clean = _normalise_public_url(env_raw)
    if env_clean:
        return env_clean
    _warn_if_malformed("HERMES_DASHBOARD_PUBLIC_URL env var", env_raw)
    cfg_raw = str(_load_dashboard_section().get("public_url", ""))
    cfg_clean = _normalise_public_url(cfg_raw)
    if not cfg_clean:
        _warn_if_malformed("dashboard.public_url in config.yaml", cfg_raw)
    return cfg_clean
