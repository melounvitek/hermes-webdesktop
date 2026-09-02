#!/usr/bin/env python3
"""Advisory NVIDIA SkillEvaluator Tier 1 scan for skill installs.

Runs alongside (never instead of) ``tools/skills_guard.py``, which stays the
enforcement layer (trust levels, install policy, block verdicts). This adds a
second, advisory opinion: NVIDIA SkillEvaluator's keyless Tier 1 static checks.

Contract:
- Warn, don't block: PII-class findings are shown with file/line and the
  install continues (the upstream PII scanner has known false-positive classes
  such as ``git@github.com`` and ``op://`` references).
- Prompt only for secrets-class criticals (private keys, cloud keys, tokens,
  credentialed connection strings); ``--force`` / non-interactive installs
  proceed with a loud warning instead of wedging on a prompt.
- Never break installs: scanner missing, crashing, timing out or emitting
  unparseable output all degrade to a no-op.

Optional binary::

    uv tool install --python 3.13 \
        "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.1.0"

Toggle via ``skills.tier1_advisory`` in config.yaml (default on; a no-op
unless the binary is installed).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SCANNER_BIN = "skillevaluator"

# Keyless, deterministic Tier 1 checks. Schema/quality are excluded: they are
# hygiene signal for the index pipeline, not install-time signal. `security`
# invokes NVIDIA SkillSpector (second optional binary, static-rules mode, no
# LLM); when absent or inconsistent it reports status="incomplete" = no opinion.
TIER1_CHECKS = "pii,unicode,lint,license,security"
SCAN_TIMEOUT_SECONDS = 120

# check_name values (from SkillEvaluator's pii_patterns.yaml categories)
# that indicate a possible REAL credential rather than personal-info
# hygiene. These are the only findings that earn a confirmation prompt.
SECRETS_CLASS_CHECKS = frozenset({
    "database_credentials",
    "hardcoded_secrets",
    "jwt_tokens",
    "webhook_urls",
    "aws_identifiers",
    "github_tokens",
    "private_keys",
})


@dataclass
class Tier1Finding:
    check: str          # e.g. "emails", "database_credentials"
    validator: str      # e.g. "PII Scan"
    severity: str       # "critical" | "high" | "medium" | "low" | "info"
    message: str
    file: str = ""
    line: int = 0
    suggestion: str = ""

    @property
    def is_secrets_class(self) -> bool:
        return self.check in SECRETS_CLASS_CHECKS

    def location(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        return self.file or "?"


@dataclass
class Tier1Report:
    available: bool                 # scanner ran and produced a report
    passed: bool = True
    findings: List[Tier1Finding] = field(default_factory=list)
    incomplete_checks: List[str] = field(default_factory=list)
    error: str = ""                 # why the scan is unavailable (debug only)

    @property
    def advisory_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if not f.is_secrets_class]

    @property
    def secrets_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if f.is_secrets_class]


def scanner_available() -> bool:
    return shutil.which(SCANNER_BIN) is not None


def tier1_advisory_enabled() -> bool:
    """Read skills.tier1_advisory from config (default True; safe because the
    scan is a no-op without the optional binary on PATH)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        skills_cfg = cfg.get("skills") or {}
        if not isinstance(skills_cfg, dict):
            return True
        value = skills_cfg.get("tier1_advisory", True)
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off")
        return bool(value)
    except Exception:
        return True


def _parse_report(report: dict) -> Tier1Report:
    """Reduce a SkillEvaluator JSON report to install-relevant findings.

    Findings from ``status == "incomplete"`` validators are kept (partial
    evidence is still evidence) but those validators are excluded from the
    pass/fail signal so an evidence-free fail can't render as a failure.
    """
    findings: List[Tier1Finding] = []
    incomplete: List[str] = []
    any_complete_failed = False
    for res in report.get("results", []) or []:
        validator = str(res.get("validator", "unknown"))
        is_incomplete = str(res.get("status", "")).lower() == "incomplete"
        if is_incomplete:
            incomplete.append(validator)
        elif not res.get("passed", True):
            any_complete_failed = True
        for f in res.get("findings", []) or []:
            if not isinstance(f, dict):
                continue
            findings.append(Tier1Finding(
                check=str(f.get("check_name", "")),
                validator=validator,
                severity=str(f.get("severity", "info")).lower(),
                message=str(f.get("message", ""))[:200],
                file=str(f.get("file_path", "")),
                line=int(f.get("line_number") or 0),
                suggestion=str(f.get("suggestion", ""))[:200],
            ))
    return Tier1Report(
        available=True,
        passed=not any_complete_failed and not findings,
        findings=findings,
        incomplete_checks=incomplete,
    )


def run_tier1_scan(skill_dir: Path, timeout: int = SCAN_TIMEOUT_SECONDS) -> Tier1Report:
    """Run SkillEvaluator Tier 1 over one skill directory; any failure returns
    ``available=False`` (no advisory opinion), never raises."""
    if not scanner_available():
        return Tier1Report(available=False, error="scanner not on PATH")
    with tempfile.TemporaryDirectory(prefix="se-tier1-") as outdir:
        try:
            subprocess.run(
                [SCANNER_BIN, "validate", str(skill_dir),
                 "--checks", TIER1_CHECKS, "--no-dedup",
                 "-r", "json", "-o", outdir],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Tier1Report(available=False, error=f"scan timed out after {timeout}s")
        except OSError as exc:
            return Tier1Report(available=False, error=f"scanner failed to launch: {exc}")
        reports = sorted(Path(outdir).glob("skillevaluator-output-*.json"))
        if not reports:
            return Tier1Report(available=False, error="scanner produced no JSON report")
        try:
            parsed = json.loads(reports[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return Tier1Report(available=False, error=f"unparseable report: {exc}")
        if not isinstance(parsed, dict):
            return Tier1Report(available=False, error="unexpected report shape")
        return _parse_report(parsed)


def format_tier1_report(report: Tier1Report, limit: int = 10) -> str:
    """Plain-text advisory summary for console display."""
    if not report.available:
        return ""
    lines: List[str] = []
    if not report.findings:
        if report.incomplete_checks:
            lines.append("SkillEvaluator Tier 1: no findings from completed checks.")
        else:
            lines.append("SkillEvaluator Tier 1: no findings.")
    else:
        lines.append(
            f"SkillEvaluator Tier 1 (advisory): "
            f"{len(report.findings)} finding(s) — informational, verify before relying on this skill."
        )
        shown = report.secrets_findings + report.advisory_findings
        for f in shown[:limit]:
            tag = "SECRETS" if f.is_secrets_class else f.severity.upper()
            lines.append(f"  [{tag}] {f.location()} — {f.message}")
        if len(shown) > limit:
            lines.append(f"  … and {len(shown) - limit} more")
    if report.incomplete_checks:
        names = ", ".join(report.incomplete_checks)
        lines.append(f"  (not run: {names} — no opinion from these checks)")
    return "\n".join(lines)
