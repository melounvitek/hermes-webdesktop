"""cua-driver binary resolution, MCP-invocation discovery, the 0.20 runtime
contract gate, and the update check / auto-repair path.

Config-derived policy (``_cua_no_overlay``, ``sanitized_cua_driver_env`` ...) is
looked up lazily through ``tools.computer_use.cua_backend`` so tests that patch
it there keep working; logger name parity is kept for the same reason.
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


def _cb():
    """Origin module, looked up lazily so ``patch("tools.computer_use.cua_backend.X")`` applies."""
    from tools.computer_use import cua_backend

    return cua_backend


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
    """Candidate cua-driver commands in resolution order.

    ``override`` / a non-empty ``HERMES_CUA_DRIVER_CMD`` is authoritative (if
    it is wrong, report the driver missing rather than silently picking
    another binary). Otherwise PATH, then canonical installer locations —
    Desktop apps launched from Finder/Dock inherit a narrow PATH that omits
    ``~/.local/bin``, and freshly installed Windows sessions inherit a stale one.
    """
    configured = (override if override is not None else os.environ.get(_CUA_DRIVER_CMD_ENV, "")).strip()
    if configured:
        return [configured]

    candidates = [_CUA_DRIVER_DEFAULT_CMD]
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        candidates.extend([
            os.path.join(local_app_data, "Programs", "Cua", "cua-driver", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver"),
        ])
    else:
        candidates.extend([
            os.path.join(home, ".local", "bin", "cua-driver"),
            os.path.join(home, ".cargo", "bin", "cua-driver"),
            "/opt/homebrew/bin/cua-driver",
            "/usr/local/bin/cua-driver",
        ])
    return candidates


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
    """Return ``(command, args)`` that spawn cua-driver's stdio MCP server.

    Asks the driver itself via ``cua-driver manifest`` (``mcp_invocation``
    carries ``command`` + ``args``) so a future rename of the subcommand keeps
    working. Falls back to ``(driver_cmd, ["mcp"])`` for older drivers or any
    discovery failure — the wrapper must not refuse to start over a failed
    discovery hop. ``--no-overlay`` is appended when policy + driver allow.
    """
    def _with_driver(args: List[str]) -> Tuple[str, List[str]]:
        return driver_cmd, _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)

    default = list(_CUA_DRIVER_ARGS)
    try:
        proc = _cb()._run_driver(driver_cmd, "manifest", timeout=timeout)
    except Exception:
        return _with_driver(default)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return _with_driver(default)
    try:
        manifest = json.loads(out)
    except (ValueError, TypeError):
        return _with_driver(default)
    invocation = manifest.get("mcp_invocation") if isinstance(manifest, dict) else None
    if not isinstance(invocation, dict):
        return _with_driver(default)
    args = invocation.get("args")
    command = invocation.get("command")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return _with_driver(default)
    if not isinstance(command, str) or not command:
        # Args are authoritative; keep our resolved driver_cmd as the binary.
        return _with_driver(args)
    # Translate a Windows ``C:\...`` command for WSL BEFORE the separator
    # check (backslash is not a separator on POSIX).
    command = _wsl_windows_path_to_posix(command)
    if not _has_path_separator(command):
        # A generic ``cua-driver`` name would lose the resolved user-local
        # path under a GUI's thin PATH; keep the concrete command we verified.
        return _with_driver(args)
    # Manifest surfaced a relocated executable — probe THAT binary for
    # `--no-overlay` support, not the system-resolved one.
    return command, _mcp_args_with_overlay_flag(args, driver_cmd=command)


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
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", raw_version)
    if not match:
        return _not_ready("driver manifest does not report a semantic version", raw_version or None)
    if tuple(int(part) for part in match.groups()) < _CUA_DRIVER_RUNTIME_CONTRACT_MIN:
        return _not_ready("Hermes computer use requires cua-driver 0.20.0 or newer", raw_version)

    invocation = manifest.get("mcp_invocation")
    invocation_args = invocation.get("args") if isinstance(invocation, dict) else None
    if not (
        isinstance(invocation_args, list)
        and invocation_args
        and all(isinstance(arg, str) for arg in invocation_args)
    ):
        return _not_ready("driver manifest does not provide an MCP launch command", raw_version)

    advertised: Dict[str, set[str]] = {}
    for command in manifest.get("subcommands") or []:
        if not isinstance(command, dict) or not isinstance(command.get("name"), str):
            continue
        advertised[command["name"]] = {
            arg["name"]
            for arg in command.get("args") or []
            if isinstance(arg, dict) and isinstance(arg.get("name"), str)
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
    """Run ``cua-driver check-update --json``; payload mirrors the
    ``check_for_update`` MCP tool (``{current_version, latest_version,
    update_available, ...}``).

    ``timeout`` defaults to 8s on POSIX and 25s on Windows (first spawn of the
    exe routinely eats seconds in Defender scanning, and a false timeout is
    expensive: callers treat ``None`` as indeterminate and the upgrade path
    used to fall through to a full reinstall on it). Returns ``None`` when the
    binary is missing, the driver predates the verb, the GitHub check failed
    (``error`` set), or the output didn't parse. Never raises.
    """
    if timeout is None:
        timeout = 25.0 if sys.platform == "win32" else 8.0
    driver_cmd = _cb().resolve_cua_driver_cmd()
    if not driver_cmd:
        return None
    try:
        proc = _cb()._run_driver(driver_cmd, "check-update", "--json", timeout=timeout)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out:  # older drivers: usage goes to stderr, stdout empty
        return None
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("error"):
        return None
    return data


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
