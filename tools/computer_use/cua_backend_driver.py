"""cua-driver binary resolution, MCP-invocation discovery, the 0.20 runtime
contract gate, and the update check.

Config-derived policy (``_cua_no_overlay``, ``_run_driver`` ...) is looked up
lazily through ``tools.computer_use.cua_backend`` so tests that patch it there
keep working; logger name parity is kept for the same reason.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.computer_use.cua_backend")

# No version *pin* knob on purpose: the upstream installer always fetches the
# latest release, so a pin var would only LOOK like it pinned. Point
# HERMES_CUA_DRIVER_CMD at a specific binary instead.
_CUA_DRIVER_CMD_ENV = "HERMES_CUA_DRIVER_CMD"
_CUA_DRIVER_DEFAULT_CMD = "cua-driver"
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP; fallback when the driver has no `manifest` verb

_CUA_DRIVER_RUNTIME_CONTRACT_MIN = (0, 20, 0)
_CUA_DRIVER_RUNTIME_CONTRACT_ARGS = {
    "mcp": {"--socket", "--grant"},
    "serve": {"--socket", "--permission-mode", "--capability-manifest",
              "--approve-capability-manifest", "--embedded"},
    "stop": {"--socket"},
}
_SEMVER_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?")

def _cb():
    """Origin module, looked up lazily so ``patch("tools.computer_use.cua_backend.X")`` applies."""
    from tools.computer_use import cua_backend

    return cua_backend

def _driver_json(driver_cmd: str, *args: str, timeout: float, require_ok: bool) -> Optional[Dict[str, Any]]:
    """Run a driver verb and parse its stdout as a JSON object; None on spawn
    failure, empty stdout (older drivers print usage to stderr), unparseable or
    non-object output — and, with ``require_ok``, on a non-zero exit."""
    try:
        proc = _cb()._run_driver(driver_cmd, *args, timeout=timeout)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out or (require_ok and proc.returncode != 0):
        return None
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def _has_path_separator(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value)

def _wsl_windows_path_to_posix(path: str) -> str:
    """Translate a Windows absolute manifest command to its DrvFS
    ``/mnt/<drive>/...`` form when Hermes runs in WSL (a Windows cua-driver
    manifest can report ``C:\\...`` while Hermes spawns via POSIX). Non-Windows
    paths and non-WSL hosts are returned unchanged."""
    if not re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    try:
        from hermes_constants import is_wsl

        if not is_wsl():
            return path
    except Exception:
        return path
    win = PureWindowsPath(path)
    drive = (win.drive or "").rstrip(":").lower()
    if not drive:
        return path
    return os.path.join("/mnt", drive, *(str(part) for part in win.parts[1:]))

def _candidate_cua_driver_commands(override: Optional[str] = None) -> List[str]:
    """Candidate commands in resolution order. ``override`` / a non-empty
    ``HERMES_CUA_DRIVER_CMD`` is authoritative (if wrong, report the driver
    missing rather than silently picking another binary). Otherwise PATH, then
    canonical installer locations — Finder/Dock-launched apps inherit a narrow
    PATH without ``~/.local/bin``; fresh Windows sessions inherit a stale one."""
    configured = (override if override is not None else os.environ.get(_CUA_DRIVER_CMD_ENV, "")).strip()
    if configured:
        return [configured]
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        installed = [
            os.path.join(local_app_data, "Programs", "Cua", "cua-driver", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver"),
        ]
    else:
        installed = [
            os.path.join(home, ".local", "bin", "cua-driver"),
            os.path.join(home, ".cargo", "bin", "cua-driver"),
            "/opt/homebrew/bin/cua-driver",
            "/usr/local/bin/cua-driver",
        ]
    return [_CUA_DRIVER_DEFAULT_CMD, *installed]

def resolve_cua_driver_cmd(override: Optional[str] = None) -> Optional[str]:
    """Resolve the cua-driver executable for every runtime/status surface.
    An override is never silently replaced by another binary."""
    for candidate in _candidate_cua_driver_commands(override):
        expanded = os.path.expanduser(candidate)
        resolved = shutil.which(expanded)
        if resolved:
            return expanded if _has_path_separator(expanded) else resolved
    return None

def cua_driver_binary_available() -> bool:
    """True if `cua-driver` resolves via env, PATH, or known install paths."""
    return _cb().resolve_cua_driver_cmd() is not None

def cua_driver_install_hint() -> str:
    scripts = "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts"
    if sys.platform == "win32":
        installer = f"  irm {scripts}/install.ps1 | iex"
    else:
        installer = f'  /bin/bash -c "$(curl -fsSL {scripts}/install.sh)"'
    return (
        "cua-driver is not installed. Install with one of:\n"
        "  hermes computer-use install\n"
        "Or run the upstream installer directly:\n"
        f"{installer}\n"
        "Or run `hermes tools` and enable the Computer Use toolset to install it automatically."
    )


# ---------------------------------------------------------------------------
# MCP invocation
# ---------------------------------------------------------------------------

def _mcp_args_with_overlay_flag(
    args: List[str],
    driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD,
) -> List[str]:
    """Return *args* with ``--no-overlay`` appended when configured and supported."""
    if _cb()._cua_no_overlay() and _cb()._cua_driver_supports_no_overlay(driver_cmd):
        return [*args, "--no-overlay"]
    return list(args)

@functools.lru_cache(maxsize=1)
def _cua_driver_supports_no_overlay(driver_cmd: str) -> bool:
    """True if ``<driver> --help`` mentions ``--no-overlay`` (probed once).
    Older drivers reject unknown flags, which would crash the MCP spawn."""
    try:
        proc = _cb()._run_driver(driver_cmd, "--help", timeout=3.0)
        return "--no-overlay" in (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return False

def _resolve_mcp_invocation(driver_cmd: str, *, timeout: float = 6.0) -> Tuple[str, List[str]]:
    """``(command, args)`` that spawn cua-driver's stdio MCP server, asked of
    the driver itself via ``cua-driver manifest`` (``mcp_invocation``) so a
    subcommand rename keeps working. Falls back to ``(driver_cmd, ["mcp"])`` on
    older drivers or any discovery failure — the wrapper must not refuse to
    start over a failed discovery hop. ``--no-overlay`` appended when allowed."""
    manifest = _driver_json(driver_cmd, "manifest", timeout=timeout, require_ok=True) or {}
    invocation = manifest.get("mcp_invocation")
    invocation = invocation if isinstance(invocation, dict) else {}
    args = invocation.get("args")
    valid_args = isinstance(args, list) and all(isinstance(a, str) for a in args)
    if not valid_args:
        args = list(_CUA_DRIVER_ARGS)
    command = invocation.get("command") if valid_args else None
    if isinstance(command, str) and command:
        # Translate a Windows ``C:\...`` command for WSL BEFORE the separator
        # check (backslash is not a separator on POSIX). A generic ``cua-driver``
        # name would lose the resolved user-local path under a GUI's thin PATH,
        # so only a concrete (path-bearing) command replaces the one we verified
        # — and THAT binary is probed for `--no-overlay`, not the system one.
        command = _wsl_windows_path_to_posix(command)
        if _has_path_separator(command):
            return command, _mcp_args_with_overlay_flag(args, driver_cmd=command)
    return driver_cmd, _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)


# ---------------------------------------------------------------------------
# Runtime contract + update checking
# ---------------------------------------------------------------------------
#
# cua-driver's native `check-update` verb compares the installed binary against
# the latest GitHub release (cached ~20h); we prefer it over a hardcoded floor.

def cua_driver_runtime_contract_status(binary: Optional[str] = None) -> Dict[str, Any]:
    """Report whether a local driver can host Hermes' 0.20 integration."""
    resolved = binary or _cb().resolve_cua_driver_cmd()

    def _not_ready(reason: str, version: Optional[str] = None) -> Dict[str, Any]:
        return {"ready": False, "binary": resolved, "version": version, "reason": reason}

    if not resolved:
        return _not_ready("cua-driver is not installed")
    try:
        result = _cb()._run_driver(resolved, "manifest", timeout=15.0 if sys.platform == "win32" else 5.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return _not_ready(f"manifest check failed: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "manifest command failed").strip()
        return _not_ready(detail.splitlines()[-1][:200])
    try:
        manifest = json.loads(result.stdout or "")
    except (TypeError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        return _not_ready("driver manifest is missing or invalid")

    raw_version = str(manifest.get("binary_version") or "").strip()
    match = _SEMVER_RE.fullmatch(raw_version)
    if not match:
        return _not_ready("driver manifest does not report a semantic version", raw_version or None)
    if tuple(int(part) for part in match.groups()) < _CUA_DRIVER_RUNTIME_CONTRACT_MIN:
        return _not_ready("Hermes computer use requires cua-driver 0.20.0 or newer", raw_version)

    invocation = manifest.get("mcp_invocation")
    invocation_args = invocation.get("args") if isinstance(invocation, dict) else None
    if not (invocation_args and isinstance(invocation_args, list)
            and all(isinstance(arg, str) for arg in invocation_args)):
        return _not_ready("driver manifest does not provide an MCP launch command", raw_version)

    advertised: Dict[str, set[str]] = {
        command["name"]: {
            arg["name"] for arg in command.get("args") or []
            if isinstance(arg, dict) and isinstance(arg.get("name"), str)
        }
        for command in manifest.get("subcommands") or []
        if isinstance(command, dict) and isinstance(command.get("name"), str)
    }
    missing = [
        f"{command} {arg}"
        for command, required_args in _CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items()
        for arg in sorted(required_args - advertised.get(command, set()))
    ]
    if missing:
        return _not_ready("driver manifest is missing: " + ", ".join(missing), raw_version)
    return {"ready": True, "binary": resolved, "version": raw_version, "reason": ""}

def cua_driver_update_check(*, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """``cua-driver check-update --json`` payload (``{current_version,
    latest_version, update_available, ...}``), or ``None`` when the binary is
    missing, the driver predates the verb, the GitHub check failed (``error``
    set) or the output didn't parse. Never raises. ``timeout`` defaults to 8s
    on POSIX / 25s on Windows: first spawn of the exe routinely eats seconds in
    Defender scanning, and callers treat ``None`` as indeterminate (the upgrade
    path used to fall through to a full reinstall on a false timeout)."""
    if timeout is None:
        timeout = 25.0 if sys.platform == "win32" else 8.0
    driver_cmd = _cb().resolve_cua_driver_cmd()
    if not driver_cmd:
        return None
    data = _driver_json(driver_cmd, "check-update", "--json", timeout=timeout, require_ok=False)
    return None if data is None or data.get("error") else data

def cua_driver_update_nudge() -> Optional[str]:
    """One-line "an update is available" message, or ``None`` when up to date,
    indeterminate, or the driver is too old to report."""
    state = _cb().cua_driver_update_check()
    if not state or not state.get("update_available"):
        return None
    latest = state.get("latest_version") or "?"
    current = state.get("current_version") or "?"
    return (
        f"cua-driver {latest} is available (you have {current}); "
        f"update with `hermes computer-use install --upgrade`."
    )
