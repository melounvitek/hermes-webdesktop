"""Startup security posture audit (warn-on-load, never blocks).

Surfaces dangerous host / deployment posture at process start so operators get an at-a-glance
"you're exposed" signal. These checks are advisory: they emit ``logger.warning`` records and return
human-readable strings; they never raise or block startup.

Checks (each is independent and fail-safe — any internal error is swallowed and simply yields no
finding):
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hermes.security_audit")

# Sentinel so the audit only runs once per process even if both the CLI and
# gateway startup paths call it.
_AUDIT_RAN = False


def _is_root() -> bool:
    """True when the process runs as POSIX uid 0. Always False on Windows."""
    getuid = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    try:
        return getuid is not None and getuid() == 0
    except Exception:
        return False


def _running_as_root() -> Optional[str]:
    if not _is_root():
        return None
    return (
        "Running as ROOT. The agent's terminal/file tools execute with full "
        "root privileges — a single prompt-injection or exposed endpoint is a "
        "full host compromise. Run Hermes as an unprivileged user (or in a "
        "sandboxed terminal backend / container with a non-root user)."
    )


def _iter_sshd_config_lines() -> list[str]:
    """Yield non-comment lines from sshd_config + its drop-in directory."""
    lines: list[str] = []
    paths: list[Path] = [Path("/etc/ssh/sshd_config")]
    try:
        d = Path("/etc/ssh/sshd_config.d")
        if d.is_dir():
            paths.extend(sorted(d.glob("*.conf")))
    except Exception:
        pass
    for p in paths:
        try:
            raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        lines.extend(s for s in map(str.strip, raw_lines) if s and not s.startswith("#"))
    return lines


def _ssh_password_auth_enabled() -> Optional[str]:
    """Warn when an SSH daemon has password authentication enabled.

    Password auth on a public SSH daemon is the classic brute-force surface and pairs badly with a
    root-capable agent box. POSIX-only; returns None when there's no sshd config to read (e.g.
    Windows, or SSH not installed).
    """
    lines = _iter_sshd_config_lines()
    if not lines:
        return None
    # Last directive wins in sshd_config. Default (no directive) is "yes".
    directives = [
        m.group(1).lower()
        for m in (re.match(r"(?i)^PasswordAuthentication\s+(\w+)", line) for line in lines)
        if m
    ]
    if directives and directives[-1] == "no":
        return None
    qualifier = "" if directives else " (default — no explicit directive)"
    return (
        f"SSH password authentication is ENABLED{qualifier}. Password auth is "
        "brute-forceable and dangerous on an internet-facing box. Set "
        "'PasswordAuthentication no' in sshd_config and use key-based auth."
    )


def _in_container() -> bool:
    """Best-effort container detection (Docker / Podman / generic OCI)."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("HERMES_DESKTOP_CHILD_PID"):
        return False  # desktop child, not a server container
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(tok in cgroup for tok in ("docker", "containerd", "kubepods", "libpod"))


def _path_is_mounted(path: Path) -> bool:
    """True if *path* sits on (or under) a real mount point per /proc/mounts.

    Container overlay/root filesystems are ephemeral; a bind/volume mount over the data dir
    shows up as a distinct mount entry. The path counts as persisted when a mountpoint at or
    above it is NOT the container root overlay.
    """
    try:
        target = path.resolve()
    except Exception:
        target = path
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return True  # can't tell — fail safe (no warning)
    # (mountpoint, fstype) entries at or above target; the longest (most specific) wins,
    # first one on ties.
    covering = [
        (Path(parts[1]), parts[2])
        for parts in (line.split() for line in mounts)
        if len(parts) >= 3 and (Path(parts[1]) == target or Path(parts[1]) in target.parents)
    ]
    if not covering:
        return True
    best_fstype = max(covering, key=lambda entry: len(str(entry[0])))[1]
    # overlay / tmpfs over the data dir = ephemeral container storage.
    return best_fstype not in ("overlay", "tmpfs", "aufs")


def _container_no_volume_mount(hermes_home: Optional[Path]) -> Optional[str]:
    if not _in_container():
        return None
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    if _path_is_mounted(hermes_home):  # any error propagates to run_security_audit (= no finding)
        return None
    return (
        f"Running in a container but the data dir ({hermes_home}) is NOT on a "
        "persistent volume mount — sessions, memory, skills, and API keys are "
        "ephemeral and lost on container restart. Mount a host volume over the "
        "HERMES_HOME data directory."
    )


def _network_listener_without_auth(config: Optional[dict]) -> list[str]:
    """Warn about network-accessible gateway listeners with no auth.

    Covers the API server (no API_SERVER_KEY) and the dashboard (non-loopback bind with no auth
    provider). Read-only against config + env; overlaps the hard fail-closed guards but surfaces the
    posture proactively at startup.
    """
    # Any error (incl. an unimportable gateway package) propagates to run_security_audit,
    # which treats it as "no finding".
    from gateway.platforms.base import is_network_accessible

    plats = (config or {}).get("platforms") or {}
    api = plats.get("api_server") if isinstance(plats, dict) else None
    if not (isinstance(api, dict) and api.get("enabled")):
        return []
    extra = api.get("extra") or {}
    host = extra.get("host") or os.environ.get("API_SERVER_HOST", "127.0.0.1")
    key = extra.get("key") or os.environ.get("API_SERVER_KEY", "")
    if not is_network_accessible(str(host)) or str(key).strip():
        return []
    return [
        f"OpenAI-compatible API server is network-accessible ({host}) "
        "with NO API_SERVER_KEY. It dispatches terminal-capable agent "
        "work — an unauthenticated network endpoint is remote code "
        "execution. Set a strong API_SERVER_KEY."
    ]


def run_security_audit(
    *, hermes_home: Optional[Path] = None, config: Optional[dict] = None
) -> list[str]:
    """Run all checks and return a list of human-readable warning strings.

    Pure: no logging, no side effects. Each check is independently fail-safe. Used directly by
    tests; the logging wrapper is :func:`log_startup_security_warnings`.
    """
    findings: list[str] = []
    for check in (
        _running_as_root,
        _ssh_password_auth_enabled,
        lambda: _container_no_volume_mount(hermes_home),
        lambda: _network_listener_without_auth(config),
    ):
        try:
            r = check()
        except Exception:
            continue
        if isinstance(r, list):
            findings.extend(r)
        elif r:
            findings.append(r)
    return findings


def log_startup_security_warnings(
    *,
    hermes_home: Optional[Path] = None,
    config: Optional[dict] = None,
    force: bool = False,
) -> list[str]:
    """Run the audit once per process and emit each finding via logger.warning.

    Returns the findings (also for tests). Never raises. Idempotent unless ``force=True`` (used by
    tests).
    """
    global _AUDIT_RAN
    if _AUDIT_RAN and not force:
        return []
    _AUDIT_RAN = True
    findings = run_security_audit(hermes_home=hermes_home, config=config)
    if findings:
        logger.warning(
            "Security posture audit found %d issue(s) — review your deployment:", len(findings)
        )
        for i, f in enumerate(findings, 1):
            logger.warning("  [security %d/%d] %s", i, len(findings), f)
    return findings
