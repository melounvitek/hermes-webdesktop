"""On-demand supply-chain audit for Hermes Agent installs.

Vulnerabilities are looked up against OSV.dev (``api.osv.dev/v1/querybatch`` + ``/v1/vulns/{id}``).
Single-shot, on-demand, never daily — see ``references/security-disclosure-triage.md``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from hermes_constants import get_hermes_home

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vid}"
OSV_BATCH_MAX = 1000  # OSV documented hard cap per request
HTTP_TIMEOUT = 20
DETAIL_PARALLELISM = 8

# Severity ordering for --fail-on gating. UNKNOWN sits below LOW so it never blocks.
SEVERITY_ORDER = {"UNKNOWN": 0, "LOW": 1, "MODERATE": 2, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass(frozen=True)
class Component:
    """A single (name, version, ecosystem) tuple discovered on disk."""

    name: str
    version: str
    ecosystem: str  # "PyPI" | "npm" — exactly as OSV expects
    source: str    # human-readable origin, e.g. "venv", "plugin:foo", "mcp:bar"


@dataclass
class Vulnerability:
    osv_id: str
    severity: str = "UNKNOWN"
    summary: str = ""
    fixed_versions: list[str] = field(default_factory=list)


@dataclass
class Finding:
    component: Component
    vuln: Vulnerability


# ─── Component discovery ──────────────────────────────────────────────────────


def _discover_venv() -> list[Component]:
    """Every dist installed in the running Python's import path."""
    from importlib.metadata import distributions

    out: list[Component] = []
    seen: set[tuple[str, str]] = set()
    for dist in distributions():
        try:
            name = (dist.metadata["Name"] or "").strip()
        except Exception:
            continue
        version = (dist.version or "").strip()
        key = (name.lower(), version)
        if name and version and key not in seen:
            seen.add(key)
            out.append(Component(name=name, version=version, ecosystem="PyPI", source="venv"))
    return out


# requirements.txt line: drop comments, environment markers, options, extras
_REQ_LINE = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
        (?:\[[^\]]+\])?              # extras
        \s*==\s*
        (?P<version>[A-Za-z0-9._+!-]+)
        \s*(?:;.*)?$
    """,
    re.VERBOSE,
)


def _match_pins(specs: Iterable[str]) -> list[tuple[str, str]]:
    """``name==version`` pairs for every spec that is an exact pin; all others are skipped."""
    return [(m.group("name"), m.group("version")) for spec in specs if (m := _REQ_LINE.match(spec))]


def _parse_requirements(text: str) -> list[tuple[str, str]]:
    """Extract ``name==version`` pins. Loose specs (>=, ~=, no pin) are skipped: they can't map to
    a single OSV query, and false positives train users to ignore an audit tool's output.
    """
    lines = (raw.strip() for raw in text.splitlines())
    return _match_pins(line for line in lines if line and not line.startswith(("#", "-")))


def _parse_pyproject_pins(text: str) -> list[tuple[str, str]]:
    """Pull ``name==version`` pins from a ``pyproject.toml`` ``dependencies`` list."""
    try:
        import tomllib
        data = tomllib.loads(text)
    except Exception:
        return []
    project = data.get("project") or {}
    optional = project.get("optional-dependencies") or {}
    groups = [project.get("dependencies")] + (list(optional.values()) if isinstance(optional, dict) else [])
    return _match_pins(str(x) for group in groups if isinstance(group, list) for x in group)


_PLUGIN_PIN_FILES = (
    ("requirements.txt", _parse_requirements),
    ("requirements-dev.txt", _parse_requirements),
    ("pyproject.toml", _parse_pyproject_pins),
)


def _discover_plugins(hermes_home: Path) -> list[Component]:
    """Python deps declared by plugins under ``~/.hermes/plugins``. Plugins typically don't install
    into the venv, so their stated requirements are audit surface the venv scan misses.
    """
    plugins_dir = hermes_home / "plugins"
    if not plugins_dir.is_dir():
        return []

    out: list[Component] = []
    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
            continue
        for filename, parse in _PLUGIN_PIN_FILES:
            path = plugin_dir / filename
            try:
                pins = parse(path.read_text(encoding="utf-8", errors="replace")) if path.is_file() else []
            except OSError:
                continue
            out.extend(Component(name=n, version=v, ecosystem="PyPI", source=f"plugin:{plugin_dir.name}") for n, v in pins)
    return out


# Recognised pinned refs: ``npx [-y|--yes] [@scope/]pkg@1.2.3`` and ``uvx [--with] pkg==1.2.3``.
# Unversioned names map to "latest" at runtime and aren't a stable audit subject.
_NPX_PKG = re.compile(r"^(@[A-Za-z0-9._-]+/[A-Za-z0-9._-]+|[A-Za-z0-9._-]+)@([A-Za-z0-9._+-]+)$")
_UVX_PKG = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._+!-]+)$")
# launcher basename -> (package-ref regex, OSV ecosystem)
_MCP_LAUNCHERS = {"npx": (_NPX_PKG, "npm"), "uvx": (_UVX_PKG, "PyPI")}


def _extract_mcp_component(server_name: str, command: str, args: list[str]) -> Optional[Component]:
    """Parse `command/args` into a Component, or None when the entry doesn't pin an auditable
    version (local paths, Docker images, unversioned npx, ...) — stay silent rather than guess.
    """
    cmd = (command or "").strip().lower()
    launcher = next((k for k in _MCP_LAUNCHERS if cmd.endswith(k)), None)  # any prefix path
    # Skip flag tokens; the first non-flag token must be a pinned ref or we stay silent.
    ref = next((token for token in args if not token.startswith("-")), None)
    if launcher is None or ref is None:
        return None
    pattern, ecosystem = _MCP_LAUNCHERS[launcher]
    m = pattern.match(ref)
    return m and Component(name=m.group(1), version=m.group(2), ecosystem=ecosystem, source=f"mcp:{server_name}")


def _discover_mcp() -> list[Component]:
    """Pinned MCP server packages from ``config.yaml``."""
    try:
        from hermes_cli.mcp_config import _get_mcp_servers
    except Exception:
        return []

    servers = _get_mcp_servers()
    if not isinstance(servers, dict):
        return []
    out: list[Component] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict) or not isinstance(cfg.get("args") or [], list):
            continue
        comp = _extract_mcp_component(name, cfg.get("command", "") or "", [str(a) for a in cfg.get("args") or []])
        if comp:
            out.append(comp)
    return out


# ─── OSV client ───────────────────────────────────────────────────────────────


_HTTP_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


def _http_json(url: str, payload: Optional[dict] = None) -> dict:
    """GET ``url`` (or POST ``payload`` as JSON when given) and decode the JSON body."""
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
        )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _osv_query_batch(components: list[Component]) -> dict[Component, list[str]]:
    """Return {component -> [osv_id, ...]} for components with any vulns."""
    findings: dict[Component, list[str]] = {}
    for chunk_start in range(0, len(components), OSV_BATCH_MAX):
        chunk = components[chunk_start:chunk_start + OSV_BATCH_MAX]
        payload = {
            "queries": [{"package": {"name": c.name, "ecosystem": c.ecosystem}, "version": c.version} for c in chunk]
        }
        try:
            resp = _http_json(OSV_BATCH_URL, payload)
        except _HTTP_ERRORS as exc:
            raise RuntimeError(f"OSV batch query failed: {exc}") from exc
        for comp, result in zip(chunk, resp.get("results") or []):
            ids = [v.get("id") for v in (result or {}).get("vulns") or [] if v.get("id")]
            if ids:
                findings[comp] = ids
    return findings


def _osv_severity_from_record(record: dict) -> str:
    """CVSS-derived severity tier from an OSV vuln record.

    Top-level ``severity`` holds CVSS vector strings we can't tier without a lib, so use the GHSA
    ``database_specific`` bucket first, then the per-affected ``ecosystem_specific`` one.
    """
    candidates = [(record.get("database_specific") or {}).get("severity")] + [
        (entry.get("ecosystem_specific") or {}).get("severity") for entry in record.get("affected") or []
    ]
    for sev in candidates:
        if isinstance(sev, str) and sev.strip().upper() in SEVERITY_ORDER:
            return sev.strip().upper()
    return "UNKNOWN"


def _osv_fixed_versions(record: dict) -> list[str]:
    fixes = [
        str(event["fixed"])
        for entry in record.get("affected") or []
        for rng in entry.get("ranges") or []
        for event in rng.get("events") or []
        if "fixed" in event
    ]
    return list(dict.fromkeys(fixes))  # dedupe, preserve order


def _osv_fetch_details(vuln_ids: Iterable[str]) -> dict[str, Vulnerability]:
    """Fetch summary/severity for each unique vuln id, in parallel."""
    unique = sorted({vid for vid in vuln_ids if vid})
    if not unique:
        return {}

    def _fetch_one(vid: str) -> Vulnerability:
        try:
            rec = _http_json(OSV_VULN_URL.format(vid=vid))
        except _HTTP_ERRORS:
            return Vulnerability(osv_id=vid)
        return Vulnerability(
            osv_id=vid,
            severity=_osv_severity_from_record(rec),
            summary=(rec.get("summary") or "").strip(),
            fixed_versions=_osv_fixed_versions(rec),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=DETAIL_PARALLELISM) as pool:
        return {vuln.osv_id: vuln for vuln in pool.map(_fetch_one, unique)}


# ─── Orchestration ────────────────────────────────────────────────────────────


def _discover_components(
    *,
    skip_venv: bool = False,
    skip_plugins: bool = False,
    skip_mcp: bool = False,
    hermes_home: Optional[Path] = None,
) -> list[Component]:
    """Discover all scannable components across the enabled sources."""
    home = hermes_home or Path(get_hermes_home())
    components: list[Component] = []
    if not skip_venv:
        components.extend(_discover_venv())
    if not skip_plugins:
        components.extend(_discover_plugins(home))
    if not skip_mcp:
        components.extend(_discover_mcp())
    return components


def run_audit(
    *,
    skip_venv: bool = False,
    skip_plugins: bool = False,
    skip_mcp: bool = False,
    hermes_home: Optional[Path] = None,
    components: Optional[list[Component]] = None,
) -> list[Finding]:
    """Query OSV for the given (or freshly discovered) components; ``components`` lets callers
    that already ran discovery reuse it instead of scanning a second time.
    """
    if components is None:
        components = _discover_components(
            skip_venv=skip_venv, skip_plugins=skip_plugins, skip_mcp=skip_mcp, hermes_home=hermes_home
        )
    raw = _osv_query_batch(components) if components else {}
    if not raw:
        return []
    details = _osv_fetch_details(vid for ids in raw.values() for vid in ids)
    findings = [
        Finding(component=comp, vuln=details.get(vid) or Vulnerability(osv_id=vid))
        for comp, ids in raw.items()
        for vid in ids
    ]
    findings.sort(
        key=lambda f: (
            -SEVERITY_ORDER.get(f.vuln.severity, 0),
            f.component.source,
            f.component.name.lower(),
            f.vuln.osv_id,
        )
    )
    return findings


# ─── Rendering ────────────────────────────────────────────────────────────────


def _render_human(findings: list[Finding], total_components: int) -> str:
    if not findings:
        return f"No known vulnerabilities found across {total_components} component(s)."

    lines = [f"Found {len(findings)} known vulnerability finding(s) across {total_components} component(s):", ""]
    last_source = None
    for f in findings:
        if f.component.source != last_source:
            lines.append(f"[{f.component.source}]")
            last_source = f.component.source
        lines.append(f"  {f.vuln.severity.ljust(8)}  {f.component.name}=={f.component.version}  {f.vuln.osv_id}")
        if summary := f.vuln.summary:
            lines.append(f"           {summary if len(summary) <= 100 else summary[:97] + '...'}")
        if f.vuln.fixed_versions:
            lines.append(f"           fixed in: {', '.join(f.vuln.fixed_versions[:3])}")
    return "\n".join(lines)


def _render_json(findings: list[Finding], total_components: int) -> str:
    payload = {
        "total_components_scanned": total_components,
        "finding_count": len(findings),
        "findings": [
            {
                "package": f.component.name,
                "version": f.component.version,
                "ecosystem": f.component.ecosystem,
                "source": f.component.source,
                "vuln_id": f.vuln.osv_id,
                "severity": f.vuln.severity,
                "summary": f.vuln.summary,
                "fixed_versions": f.vuln.fixed_versions,
            }
            for f in findings
        ],
    }
    return json.dumps(payload, indent=2)


def cmd_security_audit(args: argparse.Namespace) -> int:
    """Implementation of `hermes security audit`."""
    home = Path(get_hermes_home())
    output_json = bool(getattr(args, "json", False))
    fail_on = (getattr(args, "fail_on", None) or "critical").upper()
    if fail_on not in SEVERITY_ORDER:
        print(
            f"unknown --fail-on value: {fail_on.lower()} "
            f"(choose from: low, moderate, high, critical)",
            file=sys.stderr,
        )
        return 2

    components = _discover_components(
        skip_venv=bool(getattr(args, "skip_venv", False)),
        skip_plugins=bool(getattr(args, "skip_plugins", False)),
        skip_mcp=bool(getattr(args, "skip_mcp", False)),
        hermes_home=home,
    )
    total = len(components)
    if total == 0:
        print(
            json.dumps({"total_components_scanned": 0, "finding_count": 0, "findings": []})
            if output_json
            else "No components discovered (everything skipped, or empty environment)."
        )
        return 0

    try:
        findings = run_audit(hermes_home=home, components=components)
    except RuntimeError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2

    print((_render_json if output_json else _render_human)(findings, total))
    # Exit code: 1 iff any finding meets or exceeds the --fail-on threshold.
    threshold = SEVERITY_ORDER[fail_on]
    return int(any(SEVERITY_ORDER.get(f.vuln.severity, 0) >= threshold for f in findings))
