"""OSV malware check for MCP extension packages.

Before launching an MCP server via npx/uvx, queries Google's free public OSV API for
known malware advisories (MAL-* IDs). Regular CVEs are ignored — only confirmed malware
is blocked. Fail-open: network errors allow the package to proceed (~300ms typical).
Inspired by Block/goose's extension malware check.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_OSV_ENDPOINT = os.getenv("OSV_ENDPOINT", "https://api.osv.dev/v1/query")
_TIMEOUT = 10  # seconds

# Result cache: (ecosystem, package, version) -> (expiry_monotonic, result).
# MCP reconnect ladders and parked-server self-probes re-run the preflight for
# the SAME package on every spawn; uncached, a flapping server becomes a
# sustained OSV/DNS query stream. Advisories don't flip on second timescales,
# so clean AND blocked verdicts are reusable. Network failures are NOT cached:
# fail-open covers them and caching one could mask a real advisory later.
_CACHE_TTL_S = float(os.getenv("OSV_CHECK_CACHE_TTL", "3600"))
_CACHE_MAX_ENTRIES = 256
_cache: dict = {}
_cache_lock = threading.Lock()


def _cache_get(key) -> Tuple[bool, Optional[str]]:
    """Return (hit, result) for a fresh cache entry."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None and time.monotonic() < entry[0]:
            return True, entry[1]
        _cache.pop(key, None)  # absent or expired
        return False, None


def _cache_put(key, result: Optional[str]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            now = time.monotonic()
            for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
                del _cache[k]
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.clear()  # tiny working set in practice; safe reset
        _cache[key] = (time.monotonic() + _CACHE_TTL_S, result)


def check_package_for_malware(command: str, args: list) -> Optional[str]:
    """Check an MCP server package (inferred from ``command``/``args``) for MAL-* advisories.

    Returns a BLOCKED message when malware is found, else None — including on network
    errors and unrecognized commands (fail-open).
    """
    ecosystem = _infer_ecosystem(command)
    if not ecosystem:
        return None  # not npx/uvx — skip

    package, version = _parse_package_from_args(args, ecosystem)
    if not package:
        return None

    cache_key = (ecosystem, package, version)
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached

    try:
        malware = _query_osv(package, ecosystem, version)
    except Exception as exc:
        # Fail-open; deliberately NOT cached — see _CACHE_TTL_S comment.
        logger.debug("OSV check failed for %s/%s (allowing): %s", ecosystem, package, exc)
        return None

    result = None
    if malware:
        ids = ", ".join(m["id"] for m in malware[:3])
        summaries = "; ".join(m.get("summary", m["id"])[:100] for m in malware[:3])
        result = (f"BLOCKED: Package '{package}' ({ecosystem}) has known malware "
                  f"advisories: {ids}. Details: {summaries}")
    _cache_put(cache_key, result)
    return result


_ECOSYSTEM_BY_COMMAND = {
    "npx": "npm", "npx.cmd": "npm", "uvx": "PyPI", "uvx.cmd": "PyPI", "pipx": "PyPI"}


def _infer_ecosystem(command: str) -> Optional[str]:
    return _ECOSYSTEM_BY_COMMAND.get(os.path.basename(command).lower())


def _parse_package_from_args(args: list, ecosystem: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (package_name, version) from command args, or (None, None) if not parseable."""
    # Skip flags to find the package token. Honor npx's explicit install target
    # (--package=NAME / --package NAME / -p NAME), which names a package distinct
    # from the executed binary; otherwise the first bare positional is used.
    package_token = None
    take_next = False
    for arg in args or ():
        if not isinstance(arg, str):
            continue
        if take_next:
            package_token = arg
            break
        if arg in ("--package", "-p"):
            take_next = True
            continue
        if arg.startswith("--package="):
            package_token = arg[len("--package="):]
            break
        if arg.startswith("-"):
            continue
        package_token = arg
        break

    if not package_token:
        return None, None
    parser = _PACKAGE_PARSERS.get(ecosystem)
    return parser(package_token) if parser else (package_token, None)


def _parse_npm_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse npm package: @scope/name@version or name@version."""
    if token.startswith("@"):
        match = re.match(r"^(@[^/]+/[^@]+)(?:@(.+))?$", token)
        return (match.group(1), match.group(2)) if match else (token, None)
    if "@" in token:
        name, version = token.rsplit("@", 1)
        return name, version if version != "latest" else None
    return token, None


def _parse_pypi_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse PyPI package: name==version or name[extras]==version."""
    match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$", token)
    return (match.group(1), match.group(2)) if match else (token, None)


_PACKAGE_PARSERS = {"npm": _parse_npm_package, "PyPI": _parse_pypi_package}


def _query_osv(package: str, ecosystem: str, version: Optional[str] = None) -> list:
    """Query the OSV API; return only MAL-* advisories (regular CVEs ignored)."""
    payload = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version
    req = urllib.request.Request(
        _OSV_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hermes-agent-osv-check/1.0"},
        method="POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        result = json.loads(resp.read())
    return [v for v in result.get("vulns", []) if v.get("id", "").startswith("MAL-")]
