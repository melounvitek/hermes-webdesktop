#!/usr/bin/env python3
"""Plugin Guard — security scanner for externally-installed plugins.

Extends the ``tools/skills_guard.py`` static-analysis engine to
``hermes plugins install`` / ``update``, which otherwise clone and execute
arbitrary Git repositories unscanned.

Plugins run Python in-process, so they are more dangerous than skills — but
they are also *expected* to read their own API keys from env vars, call
provider HTTP APIs with them, and spawn subprocesses. Reusing the skill
patterns naively would flag every legitimate provider plugin, so this scanner:

- Runs the full skills_guard pattern set on documentation/config files, where
  prompt-injection and social-engineering content lives.
- Exempts the "reads own env secret" / "HTTP call with key" pattern family on
  *code* files while keeping genuinely malicious signals (foreign credential
  stores, reverse shells, destructive commands, persistence, obfuscation,
  known exfiltration services).
- Applies plugin-sized structural limits and skips VCS/venv noise.

Verdict → install policy: ``safe`` installs; ``caution`` requires explicit
confirmation (prompt, ``--force``, or caller callback); ``dangerous`` is
blocked and ``--force`` does NOT override.

Usage:
    from tools.plugin_guard import scan_plugin, should_allow_plugin_install

    result = scan_plugin(Path("/tmp/clone/my-plugin"), source="owner/repo")
    allowed, reason = should_allow_plugin_install(result)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tools.skills_guard import (
    Finding,
    ScanResult,
    SUSPICIOUS_BINARY_EXTENSIONS,
    _determine_verdict,
    format_scan_report,
    scan_file,
)

PLUGIN_SCANNER_VERSION = "plugin-guard-v1"

# Directories that are never scanned (VCS internals, caches, vendored envs).
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
}

# Code file extensions where "reads an env secret" / "HTTP call with a key
# variable" is the NORMAL, documented plugin pattern (requires_env).
CODE_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".sh", ".bash", ".rb", ".pl", ".php",
}

# skills_guard pattern ids exempt on code files (every legitimate provider
# plugin exhibits them). They still apply in full to docs/config files, where
# such content is a strong injection/social-engineering signal.
CODE_EXEMPT_PATTERN_IDS = {
    "python_environ_get_secret",
    "python_getenv_secret",
    "python_os_environ",
    "node_process_env",
    "ruby_env_secret",
    "env_exfil_httpx",
    "env_exfil_requests",
    "env_exfil_fetch",
    "env_exfil_curl",
    "env_exfil_wget",
    # Agent-facing instruction patterns are meaningless inside code
    # (docstrings/comments about prompts trip them constantly).
    "context_exfil",
    "send_to_url",
    "fake_policy",
    # Plugins legitimately write their own settings into config.yaml during
    # post_setup, and encode credentials (e.g. HTTP Basic auth) with base64.
    "agent_config_mod",
    "agent_config_contract",
    "encoded_exfil",
}

# Severity remaps for plugins. A bundled binary is warn-tier (plugin repos
# occasionally vendor one legitimately; skills never should). A mere
# ``~/.hermes/.env`` reference is the DOCUMENTED way plugin READMEs tell users
# where keys go — informational; actually READING it still trips
# ``read_secrets_file`` (critical). ``curl | sh`` install instructions are
# common in READMEs: caution, not an unoverridable block.
SEVERITY_REMAP = {
    "binary_file": "high",
    "hermes_env_access": "medium",
    "curl_pipe_shell": "high",
}

# Structural limits — plugins are real codebases, far larger than skills.
MAX_PLUGIN_FILE_COUNT = 400
MAX_PLUGIN_TOTAL_SIZE_KB = 10 * 1024   # 10MB of scannable tree
MAX_PLUGIN_SINGLE_FILE_KB = 1024       # 1MB single file


def _is_excluded(rel_parts: Tuple[str, ...]) -> bool:
    return any(part in EXCLUDED_DIRS for part in rel_parts)


def _walk(plugin_dir: Path) -> Iterator[Tuple[Path, str]]:
    """Yield (path, "a/b/c" relative path) for every non-excluded entry under plugin_dir."""
    for f in plugin_dir.rglob("*"):
        try:
            rel_parts = f.relative_to(plugin_dir).parts
        except ValueError:
            continue
        if not _is_excluded(rel_parts):
            yield f, "/".join(rel_parts)


def _finding(pattern_id: str, severity: str, category: str, file: str, match: str, description: str) -> Finding:
    return Finding(pattern_id=pattern_id, severity=severity, category=category,
                   file=file, line=0, match=match, description=description)


def _filter_findings(findings: List[Finding], rel_path: str) -> List[Finding]:
    """Apply plugin-specific exemptions and severity remaps to raw findings."""
    is_code = Path(rel_path).suffix.lower() in CODE_FILE_EXTENSIONS
    out: List[Finding] = []
    for f in findings:
        if is_code and f.pattern_id in CODE_EXEMPT_PATTERN_IDS:
            continue
        f.severity = SEVERITY_REMAP.get(f.pattern_id) or f.severity
        out.append(f)
    return out


def _check_plugin_structure(plugin_dir: Path) -> List[Finding]:
    """Structural checks sized for plugin repositories."""
    findings: List[Finding] = []
    file_count = 0
    total_size = 0

    for f, rel in _walk(plugin_dir):
        if f.is_symlink():
            file_count += 1
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(plugin_dir.resolve()):
                    findings.append(_finding(
                        "symlink_escape", "critical", "traversal", rel,
                        f"symlink -> {resolved}", "symlink points outside the plugin directory",
                    ))
            except OSError:
                findings.append(_finding(
                    "broken_symlink", "medium", "traversal", rel,
                    "broken symlink", "broken or circular symlink",
                ))
            continue

        if not f.is_file():
            continue
        file_count += 1

        try:
            size = f.stat().st_size
        except OSError:
            continue
        total_size += size

        if size > MAX_PLUGIN_SINGLE_FILE_KB * 1024:
            findings.append(_finding(
                "oversized_file", "medium", "structural", rel, f"{size // 1024}KB",
                f"file is {size // 1024}KB (limit: {MAX_PLUGIN_SINGLE_FILE_KB}KB)",
            ))

        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(_finding(
                "binary_file", SEVERITY_REMAP.get("binary_file", "high"), "structural", rel,
                f"binary: {ext}", f"binary/executable file ({ext}) bundled in plugin (cannot be scanned)",
            ))

    if file_count > MAX_PLUGIN_FILE_COUNT:
        findings.append(_finding(
            "too_many_files", "medium", "structural", "(directory)", f"{file_count} files",
            f"plugin has {file_count} files (limit: {MAX_PLUGIN_FILE_COUNT})",
        ))
    if total_size > MAX_PLUGIN_TOTAL_SIZE_KB * 1024:
        findings.append(_finding(
            "oversized_bundle", "medium", "structural", "(directory)", f"{total_size // 1024}KB",
            f"plugin is {total_size // 1024}KB total (limit: {MAX_PLUGIN_TOTAL_SIZE_KB}KB)",
        ))

    return findings


def scan_plugin(plugin_dir: Path, source: str = "") -> ScanResult:
    """Scan a plugin directory (typically the temp clone) for security threats.

    Returns a ScanResult with verdict ``safe`` | ``caution`` | ``dangerous``;
    every externally installed plugin is ``community`` trust.
    """
    all_findings: List[Finding] = []

    if plugin_dir.is_dir():
        all_findings.extend(_check_plugin_structure(plugin_dir))
        for f, rel in sorted(_walk(plugin_dir)):
            if f.is_file() and not f.is_symlink():
                all_findings.extend(_filter_findings(scan_file(f, rel_path=rel), rel))

    verdict = _determine_verdict(all_findings)
    result = ScanResult(
        skill_name=plugin_dir.name,
        source=source or plugin_dir.name,
        trust_level="community",
        verdict=verdict,
        findings=all_findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )
    if all_findings:
        categories = {f.category for f in all_findings}
        result.summary = (
            f"{plugin_dir.name}: {verdict} — {len(all_findings)} finding(s) "
            f"in {', '.join(sorted(categories))}"
        )
    else:
        result.summary = f"{plugin_dir.name}: clean scan, no threats detected"
    result.scan_provenance = {
        "scanner_version": PLUGIN_SCANNER_VERSION,
        "verdict": verdict,
        "source": result.source,
    }
    return result


def should_allow_plugin_install(
    result: ScanResult,
    force: bool = False,
) -> Tuple[Optional[bool], str]:
    """Map a plugin scan verdict to ``(allowed, reason)``.

    ``True`` installs, ``None`` needs explicit confirmation (caution), ``False``
    is blocked — ``force`` never overrides ``dangerous``.
    """
    n = len(result.findings)
    if result.verdict == "safe":
        return True, "Allowed (clean scan)"
    if result.verdict == "caution":
        if force:
            return True, f"Force-installed despite caution verdict ({n} findings)"
        return None, f"Requires confirmation (caution verdict, {n} findings)"
    return False, (
        f"Blocked (dangerous verdict, {n} findings). "
        f"--force does not override a dangerous verdict."
    )


__all__ = [
    "scan_plugin",
    "should_allow_plugin_install",
    "format_scan_report",
    "PLUGIN_SCANNER_VERSION",
]
