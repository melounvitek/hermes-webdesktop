"""Cua-driver backend (macOS, Windows, Linux): MCP over stdio to `cua-driver`.

The async `mcp` SDK runs on a background loop (``cua_backend_session``); the
same tool surface works on all three platforms, and per-host gaps (no DISPLAY,
missing AT-SPI, TCC) surface via `hermes computer-use doctor` instead of
failing silently. Install with `hermes computer-use install`. The macOS path
uses private SkyLight SPIs that can break on OS updates.

Siblings: ``cua_backend_parse`` (pure parsing), ``cua_backend_session``
(bridge + session + CLI fallback), ``cua_backend_daemon`` (private daemon +
macOS app identity). Moved names are re-imported here so
``patch("tools.computer_use.cua_backend.X")`` keeps working.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement
from tools.computer_use.cua_backend_daemon import (  # noqa: F401
    _CUA_DRIVER_BUNDLE_ID,
    _CUA_DRIVER_TEAM_IDS,
    _EmbeddedCuaDaemon,
    _embedded_daemon_spawn_command,
    _resolve_cua_driver_app_path,
    _validate_cua_driver_app_signature,
)
from tools.computer_use.cua_backend_parse import (  # noqa: F401
    _ELEMENT_LINE_RE,
    _MISSING,
    _NON_APP_WINDOW_TITLE_PREFIXES,
    _action_result_from,
    _apps_from_windows,
    _extract_tool_result,
    _image_dimensions_from_bytes,
    _image_from_tool_result,
    _ingest_windows,
    _is_placeholder_id,
    _is_real_app_window,
    _mcp_field,
    _parse_elements_from_structured,
    _parse_elements_from_tree,
    _parse_key_combo,
    _parse_xprop_net_active_window,
    _positive_int,
    _split_tree_text,
    _windows_from_tool_result,
    _z_index_uninformative,
)
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession  # noqa: F401

logger = logging.getLogger(__name__)


# No version *pin* knob on purpose: the upstream installer always fetches the
# latest release, so a pin var would only LOOK like it pinned. Point
# HERMES_CUA_DRIVER_CMD at a specific binary instead.
_CUA_DRIVER_CMD_ENV = "HERMES_CUA_DRIVER_CMD"
_CUA_DRIVER_DEFAULT_CMD = "cua-driver"
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP; fallback when the driver has no `manifest` verb

# Whole-screen intents: app="screen"/... -> composited `get_desktop_state`
# (pixels only); app="desktop" -> the OS shell window via list_windows, WITH
# interactable elements (desktop icons, taskbar).
_FULL_SCREEN_SENTINELS = {"screen", "fullscreen", "full screen", "all"}
_DESKTOP_SHELL_SENTINELS = {"desktop"}
# Shell window identifiers (substring of app_name + title, case-insensitive).
# Windows: Progman/WorkerW = desktop, Shell_TrayWnd = taskbar; macOS: Finder/Dock.
_DESKTOP_WINDOW_NAMES = (
    "progman", "workerw", "program manager", "shell_traywnd", "taskbar",
    "finder", "desktop", "dock",
)
# Backdrop subset preferred over the taskbar when both are present.
_DESKTOP_BACKDROP_NAMES = ("progman", "workerw", "program manager", "finder", "desktop")

# cua-driver's anonymous PostHog telemetry gate ("0" disables; absent => ON upstream).
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"

_CUA_DRIVER_RUNTIME_CONTRACT_MIN = (0, 20, 0)
_CUA_DRIVER_RUNTIME_CONTRACT_ARGS = {
    "mcp": {"--socket", "--grant"},
    "serve": {"--socket", "--permission-mode", "--capability-manifest",
              "--approve-capability-manifest", "--embedded"},
    "stop": {"--socket"},
}

_WINDOW_TITLE_RE = re.compile(r'AXWindow\s+"([^"]+)"')


# ---------------------------------------------------------------------------
# Config-derived policy
# ---------------------------------------------------------------------------

def _computer_use_cfg() -> Dict[str, Any]:
    """The ``computer_use`` config block, or ``{}`` when config is unreadable."""
    try:
        from hermes_cli.config import load_config

        return (load_config() or {}).get("computer_use") or {}
    except Exception:
        return {}


def _cua_no_overlay() -> bool:
    """True when Hermes should pass ``--no-overlay`` to cua-driver.

    ``computer_use.no_overlay`` overrides when set. Auto-detect otherwise:
    off on macOS (cursor-overlay redraw loop can peg a core after a session),
    headless Linux / WSL2 / containers, and Linux X11 (the overlay is a
    fullscreen always-on-top all-workspaces window with no compositor-owned
    lifecycle, so an unclean session end can leave it wedged over every app);
    on for Windows and Linux Wayland (compositor owns the surface).
    """
    val = _computer_use_cfg().get("no_overlay")
    if val is not None:
        return bool(val)
    if sys.platform == "darwin":
        return True
    if sys.platform != "linux":
        return False
    if not os.environ.get("DISPLAY"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            if "microsoft" in f.read().lower():
                return True
    except Exception:
        pass
    return os.environ.get("XDG_SESSION_TYPE") != "wayland" and not os.environ.get("WAYLAND_DISPLAY")


def _cua_telemetry_disabled() -> bool:
    """True unless ``computer_use.cua_telemetry`` opts in (unreadable config
    fails SAFE toward disabling telemetry)."""
    return not bool(_computer_use_cfg().get("cua_telemetry", False))


def _cua_configured_permission_mode() -> str:
    """``computer_use.permission_mode`` (default ``standard``).

    Only ``standard`` / ``bounded`` are honored: ``unrestricted`` is
    deliberately NOT a config value — it stays tied to the per-session YOLO
    toggle so a stale config line can never silently bypass approvals.
    Unknown values fall closed to ``standard``.
    """
    raw = str(_computer_use_cfg().get("permission_mode", "standard") or "").strip().lower()
    return raw if raw in {"standard", "bounded"} else "standard"


def _cua_capability_manifest() -> Optional[str]:
    """``computer_use.capability_manifest`` path, or None. Existence is
    validated by ``_EmbeddedCuaDaemon`` so a missing file fails loudly."""
    raw = _computer_use_cfg().get("capability_manifest")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _manifest_is_mode_independent(path: str) -> bool:
    """True when this capability manifest may accompany any permission mode.

    v1/v2 manifests must declare ``mode: bounded`` and abort startup under an
    unrestricted runtime; v3 must NOT declare a mode and is the mode-
    independent ceiling the driver accepts alongside any mode. Unreadable /
    unparseable -> False (forwarding one would turn a working session into a
    hard startup failure; bounded forwards unconditionally anyway).
    """
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception:
        logger.debug("could not read capability manifest %s", path, exc_info=True)
        return False
    if not isinstance(parsed, dict):
        return False
    version = parsed.get("version")
    return isinstance(version, int) and not isinstance(version, bool) and version >= 3


def _computer_use_max_image_dimension() -> Optional[int]:
    """``computer_use.max_image_dimension`` longest-edge cap (default 1456,
    matching the aux-vision downscale); ``0``/negative -> None (unset)."""
    try:
        dim = int(_computer_use_cfg().get("max_image_dimension", 1456))
    except (TypeError, ValueError):
        return 1456
    return dim if dim > 0 else None


def cua_driver_child_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Env for spawning cua-driver: ``base_env`` (default ``os.environ``) plus
    ``CUA_DRIVER_RS_TELEMETRY_ENABLED=0`` unless the user opted in. Used by
    every spawn site (MCP, status, doctor, install) so the policy is uniform."""
    env = dict(base_env if base_env is not None else os.environ)
    if _cua_telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    return env


def sanitized_cua_driver_env() -> Dict[str, str]:
    """``cua_driver_child_env()`` with Hermes provider secrets stripped —
    cua-driver is a third-party binary and must never inherit API keys.
    Falls back to the unsanitized telemetry env if the sanitizer can't import."""
    env = cua_driver_child_env()
    try:
        from tools.environments.local import _sanitize_subprocess_env

        return _sanitize_subprocess_env(env)
    except Exception:
        return env


def _run_driver(driver_cmd: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a short cua-driver verb with the sanitized env, hidden window and
    stdin=DEVNULL (older drivers fall into a stdin-reading mode on unknown
    verbs; EOF makes them exit fast instead of blocking until the timeout)."""
    return subprocess.run(
        [driver_cmd, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
        env=sanitized_cua_driver_env(),
    )


# ---------------------------------------------------------------------------
# Linux display diagnostics / capture-target selection
# ---------------------------------------------------------------------------

def _linux_session_locked() -> Optional[bool]:
    """Best-effort: is the graphical session locked? (Linux only.)

    A locked KDE/GNOME session freezes renderers and half-disables the AX
    tree, so window discovery legitimately returns nothing — which otherwise
    reads as a driver bug. True/False when loginctl answers, None when
    unavailable (non-Linux, no systemd-logind, probe failure).
    """
    if sys.platform != "linux":
        return None
    try:
        proc = subprocess.run(["loginctl", "list-sessions", "--no-legend"],
                              capture_output=True, text=True, timeout=2.0)
        if proc.returncode != 0:
            return None
        any_seat = False
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or "seat" not in line:
                continue
            any_seat = True
            probe = subprocess.run(["loginctl", "show-session", parts[0], "-p", "LockedHint"],
                                   capture_output=True, text=True, timeout=2.0)
            if "LockedHint=no" in probe.stdout:
                return False
        return True if any_seat else None
    except Exception:
        return None


def _empty_discovery_reason() -> str:
    """One-line diagnosis for 'window discovery found nothing'."""
    if _linux_session_locked() is True:
        return (
            "the desktop session is LOCKED (loginctl LockedHint=yes) — "
            "unlock the screen; a locked compositor hides windows and "
            "freezes app renderers"
        )
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return "no DISPLAY is set — X11/XWayland is not reachable from this process"
    if sys.platform == "darwin":
        # Headless Mac / asleep panel: ScreenCaptureKit has 0 shareable
        # displays while TCC grants look fine.
        return (
            "window discovery returned no windows; on macOS this usually "
            "means no shareable display (headless Mac or panel asleep) — "
            "wake the display or attach a monitor/HDMI dummy, then run "
            "`hermes computer-use doctor`"
        )
    return (
        "window discovery returned no windows; run `hermes computer-use "
        "doctor` (display reachability, AX capability)"
    )


def _linux_x11_active_window_id() -> Optional[int]:
    """Best-effort read of ``_NET_ACTIVE_WINDOW`` via xprop. Never raises."""
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=2, check=False)
    except Exception:
        return None
    return _parse_xprop_net_active_window(proc.stdout or "") if proc.returncode == 0 else None


def _select_capture_target(
    windows: List[Dict[str, Any]],
    *,
    app_requested: bool,
    exact_target: bool = False,
) -> Dict[str, Any]:
    """Select the best window for capture from z-sorted list_windows output.

    Windows arrive sorted by ``z_index`` descending (frontmost first). For
    unqualified default captures on Linux (no app filter, no exact target),
    desktop/shell helper windows are skipped first — they are targetable but
    capture as empty — and when every remaining candidate shares the same
    ``z_index`` (the common X11 case) ``_NET_ACTIVE_WINDOW`` beats list order.
    Exact-target captures never pay for the ``xprop`` probe.
    """
    candidates = [w for w in windows if not w["off_screen"]]
    pool = candidates
    if not exact_target and not app_requested and sys.platform == "linux":
        real_apps = [w for w in candidates if _is_real_app_window(w)]
        if real_apps:
            pool = real_apps
        if pool and _z_index_uninformative(pool):
            active_id = _linux_x11_active_window_id()
            if active_id is not None:
                for w in pool:
                    if w.get("window_id") == active_id:
                        return w
    return pool[0] if pool else windows[0]


# ---------------------------------------------------------------------------
# Driver resolution + MCP invocation
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
    return resolve_cua_driver_cmd() is not None


def _mcp_args_with_overlay_flag(
    args: List[str],
    driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD,
) -> List[str]:
    """Return *args* with ``--no-overlay`` appended when configured and supported."""
    if _cua_no_overlay() and _cua_driver_supports_no_overlay(driver_cmd):
        return [*args, "--no-overlay"]
    return list(args)


@functools.lru_cache(maxsize=1)
def _cua_driver_supports_no_overlay(driver_cmd: str) -> bool:
    """True if ``<driver> --help`` mentions ``--no-overlay`` (probed once).
    Older drivers reject unknown flags, which would crash the MCP spawn."""
    try:
        proc = _run_driver(driver_cmd, "--help", timeout=3.0)
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
        proc = _run_driver(driver_cmd, "manifest", timeout=timeout)
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


# ---------------------------------------------------------------------------
# Runtime contract + update checking
# ---------------------------------------------------------------------------
#
# cua-driver's native `check-update` verb compares the installed binary against
# the latest GitHub release (cached ~20h); we prefer it over a hardcoded floor.

def cua_driver_runtime_contract_status(binary: Optional[str] = None) -> Dict[str, Any]:
    """Report whether a local driver can host Hermes' 0.20 integration."""
    resolved = binary or resolve_cua_driver_cmd()

    def _not_ready(reason: str, version: Optional[str] = None) -> Dict[str, Any]:
        return {"ready": False, "binary": resolved, "version": version, "reason": reason}

    if not resolved:
        return _not_ready("cua-driver is not installed")
    try:
        result = _run_driver(resolved, "manifest", timeout=15.0 if sys.platform == "win32" else 5.0)
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
    driver_cmd = resolve_cua_driver_cmd()
    if not driver_cmd:
        return None
    try:
        proc = _run_driver(driver_cmd, "check-update", "--json", timeout=timeout)
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
    state = cua_driver_update_check()
    if not state or not state.get("update_available"):
        return None
    latest = state.get("latest_version") or "?"
    current = state.get("current_version") or "?"
    return (
        f"cua-driver {latest} is available (you have {current}); "
        f"update with `hermes computer-use install --upgrade`."
    )


_update_checked = False

# One auto-repair attempt per process: when the runtime-contract gate fails
# for something a reinstall fixes (old version, missing manifest verbs) run
# the standard install path once instead of telling the user to. Guarded so a
# failing installer can't loop — the second start() goes straight to the error.
_contract_repair_attempted = False


def _maybe_repair_runtime_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Try one automatic driver repair; return the post-repair contract (or the
    original when no repair was attempted / it failed). Never raises. An
    explicit ``HERMES_CUA_DRIVER_CMD`` override is authoritative even when
    broken, and a missing binary means installation was never requested."""
    global _contract_repair_attempted
    if (
        contract.get("ready")
        or _contract_repair_attempted
        or os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip()
        or not contract.get("binary")
    ):
        return contract
    _contract_repair_attempted = True
    logger.info(
        "computer_use: installed cua-driver is not usable (%s); "
        "attempting automatic repair",
        contract.get("reason") or "runtime contract is incomplete",
    )
    try:
        from hermes_cli.tools_config import install_cua_driver

        if not install_cua_driver(upgrade=False, show_installer_progress=False):
            return contract
    except Exception as exc:
        logger.warning("computer_use: automatic cua-driver repair failed: %s", exc)
        return contract
    try:
        return cua_driver_runtime_contract_status()
    except Exception:
        return contract


def _maybe_nudge_update() -> None:
    """Emit an update nudge at most once per process, off-thread so the
    (cached, ~20h) GitHub poll never blocks the first computer_use action."""
    global _update_checked
    if _update_checked:
        return
    _update_checked = True

    def _run() -> None:
        try:
            msg = cua_driver_update_nudge()
        except Exception:
            return
        if msg:
            logger.info("computer_use: %s", msg)

    threading.Thread(target=_run, name="cua-driver-update-check", daemon=True).start()


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
# Capture helpers
# ---------------------------------------------------------------------------

def _gws_is_empty(out: Dict[str, Any]) -> bool:
    """True when a get_window_state result carries neither a screenshot nor a
    parseable tree. Modern drivers put the payload in structuredContent with
    no markdown tree — that is NOT empty."""
    if out.get("images"):
        return False
    sc_ = out.get("structuredContent") or {}
    if sc_.get("elements") or sc_.get("screenshot_png_b64"):
        return False
    txt = out.get("data") if isinstance(out.get("data"), str) else ""
    _, tr = _split_tree_text(txt or "")
    return not (tr and tr.strip())


def _tree_text(out: Dict[str, Any]) -> str:
    return out["data"] if isinstance(out["data"], str) else ""


def _window_title_from_tree(tree: str) -> str:
    wt = _WINDOW_TITLE_RE.search(tree)
    return wt.group(1) if wt else ""


def _png_metrics(png_b64: str, width: int, height: int) -> Tuple[int, int, int]:
    """Return ``(png_bytes_len, width, height)``, replacing the given size with
    the sniffed one when the bytes decode to a readable PNG/JPEG header."""
    try:
        raw = base64.b64decode(png_b64, validate=False)
        png_bytes_len = len(raw)
        detected_width, detected_height = _image_dimensions_from_bytes(raw)
        if detected_width and detected_height:
            width, height = detected_width, detected_height
    except Exception:
        png_bytes_len = len(png_b64) * 3 // 4
    return png_bytes_len, width, height


def _is_desktop_window(w: Dict[str, Any], names: Tuple[str, ...] = _DESKTOP_WINDOW_NAMES) -> bool:
    haystack = f"{w.get('app_name', '')} {w.get('title', '')}".lower()
    return any(name in haystack for name in names)


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

_NO_TARGET_MSG = "No active window — call capture() first."
_BTF_UNSUPPORTED_MSG = "The connected cua-driver does not advertise the standalone bring_to_front tool."


class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP."""

    def __init__(self, permission_mode: str = "standard") -> None:
        if permission_mode not in {"standard", "bounded", "unrestricted"}:
            raise ValueError(f"unsupported cua-driver permission mode: {permission_mode}")
        self.permission_mode = permission_mode
        self._embedded_daemon: Optional[_EmbeddedCuaDaemon] = None
        if permission_mode != "standard":
            # The manifest is mandatory for bounded (the daemon validates it)
            # and optional for unrestricted, where it still caps what an
            # approval-bypassed run may touch.
            self._embedded_daemon = _EmbeddedCuaDaemon(
                resolve_cua_driver_cmd() or "",
                permission_mode,
                capability_manifest=_cua_capability_manifest(),
            )
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge, self._embedded_daemon)
        # Sticky context — updated by capture()/focus_app(), used by actions.
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._last_app: Optional[str] = None
        # Exact identity for capture_after: app names may be generic on Linux
        # (several unrelated Qt windows can all say Qt6Application).
        self._last_target: Optional[Dict[str, Optional[int]]] = None
        # Per-snapshot `element_index -> element_token` map from capture().
        # Actions attach the token so cua-driver detects "stale" explicitly
        # instead of silently re-resolving to a different element.
        self._snapshot_tokens: Dict[int, str] = {}
        # Public session label (one per backend = one per Hermes run) passed as
        # `session` on every call: owns the agent cursor color and gives config
        # / recording state a stable owner inside the transport-private
        # lifecycle. Part of the 0.20 runtime contract checked at start().
        self._session_id: str = f"hermes-{uuid.uuid4().hex[:12]}"
        self._session.set_transport_reset_callback(self._handle_transport_reset)

    def _handle_transport_reset(self) -> None:
        """Invalidate every capability minted by the replaced transport."""
        self._clear_active_target()

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        contract = cua_driver_runtime_contract_status()
        if not contract.get("ready"):
            contract = _maybe_repair_runtime_contract(contract)
        if not contract.get("ready"):
            reason = contract.get("reason") or "runtime contract is incomplete"
            repair = (
                "Update the binary selected by HERMES_CUA_DRIVER_CMD or remove that override."
                if os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip()
                else "Run `hermes computer-use install` to repair it."
            )
            raise RuntimeError(f"cua-driver is not ready: {reason}. {repair}")
        _maybe_nudge_update()
        # `mcp` is an optional extra: lazy-install on first use (gated by
        # `security.allow_lazy_installs`); on failure ensure() raises
        # FeatureUnavailable with the exact `uv pip install` hint.
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.computer_use", prompt=False)
        import importlib
        importlib.invalidate_caches()  # a just-installed package may not be importable yet
        try:
            if self._embedded_daemon is not None:
                self._embedded_daemon.start()
            self._session.start()
        except Exception:
            if self._embedded_daemon is not None:
                self._embedded_daemon.stop()
            raise

        # Declare this run's identity. Non-fatal: cua-driver accepts anonymous
        # calls (the cursor just won't render), so degrade rather than abort.
        try:
            self._session.call_tool("start_session", {"session": self._session_id})
        except Exception as e:
            logger.debug("cua-driver start_session failed (continuing anonymous): %s", e)

        # Post-handshake tuning guards on `_started`: before the handshake
        # flips it, call_tool would re-enter session.start() and tests that
        # stub start() would recurse.
        if self._session._started:
            # Cap screenshot size so every later capture pays less over the
            # daemon socket and in the model turn.
            max_dim = _computer_use_max_image_dimension()
            if max_dim:
                try:
                    self.set_config(max_image_dimension=max_dim)
                except Exception as e:
                    logger.debug("cua-driver set_config(max_image_dimension) failed: %s", e)
            # Belt-and-suspenders when --no-overlay is unsupported or ignored.
            if _cua_no_overlay():
                try:
                    self.set_agent_cursor_enabled(False, cursor_id=self._session_id)
                except Exception as e:
                    logger.debug("cua-driver set_agent_cursor_enabled failed: %s", e)

    def stop(self) -> None:
        # Best-effort end_session first so the driver cleans per-session state
        # (cursor overlay, recording ownership, config overrides); the
        # connection drop below releases daemon-side state regardless.
        if self._session._started:
            try:
                self._session.call_tool("end_session", {"session": self._session_id})
            except Exception as e:
                logger.debug("cua-driver end_session failed (continuing teardown): %s", e)
        try:
            self._session.stop()
        finally:
            try:
                self._bridge.stop()
            finally:
                if self._embedded_daemon is not None:
                    self._embedded_daemon.stop()

    def is_available(self) -> bool:
        # Other Unix-likes haven't been exercised end-to-end.
        if sys.platform not in ("darwin", "win32", "linux"):
            return False
        return cua_driver_binary_available()

    # ── Target state ───────────────────────────────────────────────
    def _clear_active_target(self) -> None:
        """Forget a capture/focus target so a failed lookup cannot misroute input."""
        self._active_pid = None
        self._active_window_id = None
        self._last_app = None
        self._last_target = None
        self._snapshot_tokens = {}

    def _set_active_target(self, target: Dict[str, Any]) -> None:
        self._active_pid = target["pid"]
        self._active_window_id = target["window_id"]
        # Tokens belong to the prior snapshot; disarm before any capture call
        # so an exception cannot pair old tokens with this target.
        self._snapshot_tokens = {}
        self._last_target = {"pid": self._active_pid, "window_id": self._active_window_id}

    def _no_target(self, action: str, *, need_window: bool = False) -> Optional[ActionResult]:
        if self._active_pid is None or (need_window and self._active_window_id is None):
            return ActionResult(ok=False, action=action, message=_NO_TARGET_MSG)
        return None

    def _failed_capture(self, mode: str, message: str = "") -> CaptureResult:
        """Return an empty capture after disarming any prior target context."""
        self._clear_active_target()
        return CaptureResult(mode=mode, width=0, height=0, png_b64=None, elements=[],
                             app="", window_title=message, png_bytes_len=0)

    def _call_capture_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a capture-stage tool and disarm state on transport or logical failure."""
        try:
            out = self._session.call_tool(name, args)
        except Exception:
            self._clear_active_target()
            raise
        if out.get("isError") is True:
            message = out.get("data")
            self._clear_active_target()
            raise RuntimeError(
                f"cua-driver {name} failed"
                + (f": {message}" if isinstance(message, str) and message else "")
            )
        return out

    # ── Window discovery ───────────────────────────────────────────
    def _list_windows_args(self) -> Dict[str, Any]:
        return {"on_screen_only": True, "session": self._session_id}

    def _load_windows(self) -> List[Dict[str, Any]]:
        """Load normalized visible windows sorted by ``z_index`` DESCENDING
        (frontmost at index 0 — the default target for capture()/focus_app()),
        re-fetching over the CLI transport when MCP returns nothing."""
        out = self._call_capture_tool("list_windows", self._list_windows_args())
        windows = _ingest_windows(_windows_from_tool_result(out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        if windows:
            return windows

        logger.warning(
            "cua-driver list_windows returned no windows over MCP; "
            "re-fetching via CLI transport",
        )
        try:
            cli_out = self._session._call_tool_via_cli("list_windows", self._list_windows_args(), 20.0)
        except Exception as exc:
            logger.error("cua-driver CLI re-fetch for list_windows failed: %s", exc)
            return []
        if cli_out.get("isError") is True:
            logger.error("cua-driver CLI re-fetch for list_windows returned an error")
            self._clear_active_target()
            return []
        windows = _ingest_windows(_windows_from_tool_result(cli_out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        return windows

    def _match_windows_for_app(
        self, windows: List[Dict[str, Any]], app: str
    ) -> List[Dict[str, Any]]:
        """Resolve ``app=`` through exact names before convenience substrings.

        Linux ``list_windows`` can omit an app name while ``list_apps`` keeps
        name/bundle-ID metadata. Exact direct names and exact metadata aliases
        win over substring matches: querying ``Code`` must not silently select
        ``Visual Studio Code`` because it is frontmost.
        """
        app_lower = app.strip().lower()
        if not app_lower:
            return []

        def _by_name(exact: bool) -> List[Dict[str, Any]]:
            if exact:
                return [w for w in windows if app_lower == str(w.get("app_name", "")).strip().lower()]
            return [w for w in windows if app_lower in str(w.get("app_name", "")).lower()]

        direct_exact = _by_name(exact=True)
        if direct_exact:
            return direct_exact

        try:
            running_apps = self.list_apps()
        except Exception as exc:
            # A title can still be the only usable identity on X11 when app
            # enumeration is unavailable, so keep the title fallback below.
            logger.debug("computer_use list_apps fallback failed for %r: %s", app, exc)
            running_apps = []

        exact_pids: set[int] = set()
        partial_pids: set[int] = set()
        for raw_app in running_apps:
            if not isinstance(raw_app, dict) or raw_app.get("running") is False:
                continue
            pid = _positive_int(raw_app.get("pid"))
            if pid is None:
                continue
            aliases = {
                value.strip().lower()
                for key in ("bundle_id", "bundleId", "name", "app_name", "display_name")
                if isinstance((value := raw_app.get(key)), str) and value.strip()
            }
            if app_lower in aliases:
                exact_pids.add(pid)
            elif any(app_lower in alias for alias in aliases):
                partial_pids.add(pid)

        for matched in (
            [w for w in windows if w.get("pid") in exact_pids],
            _by_name(exact=False),
            [w for w in windows if w.get("pid") in partial_pids],
        ):
            if matched:
                return matched

        # Some X11 backends expose a title but no app name. Restrict this final
        # fallback to nameless rows so a localized app name is not overridden
        # merely because its title happens to be in the caller's language.
        return [
            w for w in windows
            if not str(w.get("app_name", "")).strip()
            and app_lower in str(w.get("title", "")).lower()
        ]

    def _resolve_capture_windows(
        self,
        mode: str,
        app: Optional[str],
        pid: Optional[int],
        window_id: Optional[int],
    ) -> "List[Dict[str, Any]] | CaptureResult":
        """Candidate windows for capture(), or a failed CaptureResult."""
        if pid is not None or window_id is not None:
            # An exact pid/window pair is both the stable capture_after target
            # and the escape hatch when discovery is unavailable on X11.
            if pid is None or window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires both pid and window_id>",
                )
            target_pid = _positive_int(pid)
            target_window_id = _positive_int(window_id)
            if target_pid is None or target_window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires positive integer pid and window_id>",
                )
            return [{"app_name": app or "", "pid": target_pid, "window_id": target_window_id,
                     "off_screen": False, "title": "", "z_index": 0}]

        try:
            windows = self._load_windows()
        except Exception:
            self._clear_active_target()
            raise
        if not windows:
            # Diagnose instead of a bare 0x0: the dominant real-world cause on
            # Linux is a locked desktop session.
            return self._failed_capture(mode, _empty_discovery_reason())
        if not app:
            return windows

        if app.strip().lower() in _DESKTOP_SHELL_SENTINELS:
            # Desktop-shell request: the OS shell window WITH its interactable
            # elements (desktop icons), so "click the taskbar" works.
            desktop = [w for w in windows if _is_desktop_window(w)]
            if not desktop:
                return self._failed_capture(mode, (
                    f"<no desktop/shell window found for app={app!r}; "
                    f"cua-driver captures one window at a time and exposes "
                    f"no whole-virtual-desktop or per-monitor capture. "
                    f"Call list_apps / capture(app='<AppName>') to target a "
                    f"specific window instead. On Windows the taskbar is "
                    f"'Shell_TrayWnd' and the desktop is 'Progman'.>"
                ))
            # Prefer the backdrop (Progman/WorkerW/Finder) over the taskbar so
            # the capture shows the full desktop rather than the task strip.
            return sorted(
                desktop,
                key=lambda w: 0 if _is_desktop_window(w, _DESKTOP_BACKDROP_NAMES) else 1,
            )

        # When the filter matches nothing, say so instead of silently capturing
        # the frontmost window — on macOS list_windows returns the localized
        # app name (e.g. "計算機"), so `app="Calculator"` legitimately misses.
        filtered = self._match_windows_for_app(windows, app)
        if not filtered:
            return self._failed_capture(mode, (
                f"<no on-screen window matched app={app!r}; "
                f"call list_apps to see available app names or bundle IDs "
                f"(macOS reports localized names, e.g. '計算機' "
                f"instead of 'Calculator'; some Linux/Qt apps only "
                f"resolve via list_apps metadata)>"
            ))
        return filtered

    # ── Capture ────────────────────────────────────────────────────
    def _gws_args(self) -> Dict[str, Any]:
        return {
            "pid": self._active_pid,
            "window_id": self._active_window_id,
            "session": self._session_id,
        }

    def _cli_refetch_window_state(self, what: str) -> Optional[Dict[str, Any]]:
        """One-shot get_window_state over the CLI transport (different daemon
        socket) after MCP came back imageless/empty without raising."""
        try:
            cli_out = self._session._call_tool_via_cli("get_window_state", self._gws_args(), 30.0)
        except Exception as cli_exc:
            logger.error("cua-driver CLI re-fetch for %s failed: %s", what, cli_exc)
            return None
        if cli_out.get("isError") is True:
            self._clear_active_target()
            return None
        return cli_out

    def _capture_vision(self) -> Tuple[Optional[str], Optional[str], str]:
        """Pixels only, no elements. Returns ``(png_b64, mime, window_title)``.

        Drivers that advertise the (cheaper) standalone ``screenshot`` tool use
        it; current drivers folded PNG capture into ``get_window_state``, whose
        tree is DISCARDED here. When discovery hasn't run we still try
        ``screenshot`` first and fall back, so the path self-heals on any
        driver version.
        """
        png_b64: Optional[str] = None
        image_mime_type: Optional[str] = None
        window_title = ""
        if self._session._has_tool("screenshot") or not self._session.capabilities_discovered:
            sc_out = self._call_capture_tool("screenshot", {
                "window_id": self._active_window_id, "format": "jpeg", "quality": 85,
                "session": self._session_id,
            })
            png_b64, image_mime_type = _image_from_tool_result(sc_out)
        if not png_b64:
            # "Unknown tool: screenshot" or an empty image part -> get_window_state.
            gws_out = self._call_capture_tool("get_window_state", self._gws_args())
            png_b64, image_mime_type = _image_from_tool_result(gws_out)
            # The title is cheap and useful; `elements` stays empty by contract.
            _, tree = _split_tree_text(_tree_text(gws_out))
            window_title = _window_title_from_tree(tree)
        if not png_b64:
            logger.warning(
                "cua-driver vision capture returned no image over MCP "
                "(window_id=%s); re-fetching via CLI transport",
                self._active_window_id,
            )
            cli_out = self._cli_refetch_window_state("vision screenshot")
            if cli_out is not None and cli_out.get("images"):
                png_b64 = cli_out["images"][0]
                image_mime_type = "image/png"
        return png_b64, image_mime_type, window_title

    def _capture_window_state(self) -> Tuple[Optional[str], Optional[str], List[UIElement], str]:
        """AX tree + screenshot. Returns ``(png_b64, mime, elements, window_title)``."""
        gws_out = self._call_capture_tool("get_window_state", self._gws_args())
        # A flaky bridge can return a degenerate result (no screenshot AND no
        # parseable tree) WITHOUT raising — a silent 0x0 to the model. Distinct
        # from the EAGAIN path handled in call_tool: here MCP "succeeded".
        if _gws_is_empty(gws_out):
            logger.warning(
                "cua-driver get_window_state returned an empty result over MCP "
                "(pid=%s window_id=%s); re-fetching via CLI transport",
                self._active_pid, self._active_window_id,
            )
            cli_out = self._cli_refetch_window_state("get_window_state")
            if cli_out is not None and not _gws_is_empty(cli_out):
                gws_out = cli_out

        _, tree = _split_tree_text(_tree_text(gws_out))
        # Prefer the canonical structuredContent.elements (real frames); the
        # markdown regex fallback yields (0,0,0,0) bounds.
        sc_elements = (gws_out.get("structuredContent") or {}).get("elements")
        if isinstance(sc_elements, list) and sc_elements:
            elements = _parse_elements_from_structured(sc_elements)
        else:
            elements = _parse_elements_from_tree(tree) if tree else []
        # Tokens are tied to this snapshot: overwrite the whole map (and clear
        # it when the new capture carries none).
        self._snapshot_tokens = {e.index: e.element_token for e in elements if e.element_token}
        png_b64, image_mime_type = _image_from_tool_result(gws_out)
        return png_b64, image_mime_type, elements, _window_title_from_tree(tree)

    def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> CaptureResult:
        """Capture the frontmost on-screen window or an exact known target.

        Maps hermes `capture(mode, app)` -> cua-driver `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision). Only the
        structured ``structuredContent.windows`` shape is supported.
        """
        # Drop schema-filler ids (models that zero-fill every optional
        # property) before they read as a targeting request.
        if _is_placeholder_id(pid):
            pid = None
        if _is_placeholder_id(window_id):
            window_id = None
        exact_target = pid is not None or window_id is not None
        # Full-screen lane bypasses enumeration entirely (also keeps
        # screenshots working when Windows UIA enumeration hangs).
        # app='desktop' deliberately does NOT take it: desktop icons stay clickable.
        if not exact_target and app and app.strip().lower() in _FULL_SCREEN_SENTINELS:
            return self._capture_full_screen(mode)

        windows = self._resolve_capture_windows(mode, app, pid, window_id)
        if isinstance(windows, CaptureResult):
            return windows

        target = _select_capture_target(windows, app_requested=bool(app), exact_target=exact_target)
        self._set_active_target(target)
        app_name = target["app_name"]
        # Record the resolved app so capture_after= follow-ups re-target the
        # same app rather than falling back to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name or app or ""

        elements: List[UIElement] = []
        if mode == "vision":
            png_b64, image_mime_type, window_title = self._capture_vision()
        else:
            png_b64, image_mime_type, elements, window_title = self._capture_window_state()

        png_bytes_len = width = height = 0
        if png_b64:
            png_bytes_len, width, height = _png_metrics(png_b64, 0, 0)

        return CaptureResult(mode=mode, width=width, height=height, png_b64=png_b64,
                             elements=elements, app=app_name, window_title=window_title,
                             png_bytes_len=png_bytes_len, image_mime_type=image_mime_type)

    def _capture_full_screen(self, mode: str) -> CaptureResult:
        """Composited grab of everything on screen via `get_desktop_state`
        (like PrtScn) — the shell window would only show wallpaper + icons.
        Never enumerates, so it also works when Windows UIA hangs. Pixels only:
        `elements` is always empty; `note` tells the model how to reach the
        interactive lanes. ``capture_scope`` is switched to desktop for the
        call and restored afterwards.
        """
        self._clear_active_target()
        previous_scope: Optional[str] = None
        try:
            cfg = self._session.call_tool("get_config", {"session": self._session_id}, timeout=10.0)
            sc = cfg.get("structuredContent") or {}
            if isinstance(sc, dict) and isinstance(sc.get("capture_scope"), str):
                previous_scope = sc["capture_scope"]
        except Exception as e:
            logger.debug("cua-driver get_config before full-screen capture failed: %s", e)

        def _set_scope(value: str) -> None:
            self._session.call_tool(
                "set_config",
                {"key": "capture_scope", "value": value, "session": self._session_id},
                timeout=10.0,
            )

        try:
            if previous_scope != "desktop":
                _set_scope("desktop")
            out = self._call_capture_tool("get_desktop_state", {"session": self._session_id})
        finally:
            if previous_scope and previous_scope != "desktop":
                try:
                    _set_scope(previous_scope)
                except Exception as e:
                    logger.debug("cua-driver restore capture_scope failed: %s", e)

        png_b64, image_mime_type = _image_from_tool_result(out)
        if not png_b64:
            return self._failed_capture(mode, "<get_desktop_state returned no image; the driver may "
                                              "predate the desktop capture lane — try "
                                              "capture(app='<AppName>') for a specific window>")
        structured = out.get("structuredContent") or {}
        png_bytes_len, width, height = _png_metrics(
            png_b64,
            int(structured.get("screenshot_width") or structured.get("screen_width") or 0),
            int(structured.get("screenshot_height") or structured.get("screen_height") or 0),
        )
        return CaptureResult(
            mode="vision", width=width, height=height, png_b64=png_b64, elements=[],
            app="screen", window_title="Full screen (composited)",
            png_bytes_len=png_bytes_len, image_mime_type=image_mime_type,
            note=("full-screen capture has no interactable elements; to act on "
                  "what you see, call capture(app='<AppName>') for that app's "
                  "clickable element list, or capture(app='desktop') for the "
                  "desktop shell (wallpaper icons / taskbar) with elements"),
        )

    # ── Input delivery ─────────────────────────────────────────────
    def _apply_delivery(
        self,
        action: str,
        args: Dict[str, Any],
        delivery_mode: Optional[str],
    ) -> Optional[ActionResult]:
        """Attach delivery_mode to an input-action args dict.

        Background is the default and needs no flag. Foreground is only sent
        when the live action schema accepts it; on an older driver we refuse
        with ``foreground_unsupported`` instead of silently downgrading to
        background (which would land input where the model didn't expect).
        Returns an ActionResult to short-circuit on refusal, or None to proceed.
        """
        if not delivery_mode or delivery_mode == "background":
            return None
        if delivery_mode != "foreground":
            return ActionResult(ok=False, action=action, code="bad_delivery_mode",
                                message=f"unknown delivery_mode {delivery_mode!r} — use background|foreground.")
        if not self._session.supports_input_property(action, "delivery_mode"):
            return ActionResult(
                ok=False, action=action, code="foreground_unsupported", delivery_mode="foreground",
                message=("The connected cua-driver action schema does not accept "
                         "delivery_mode, so foreground delivery is unavailable. "
                         "Use another verified rung without assuming the reported "
                         "package version describes the live schema."),
            )
        args["delivery_mode"] = "foreground"
        return None

    def _run_input_action(
        self,
        action: str,
        args: Dict[str, Any],
        delivery_mode: Optional[str],
        bring_to_front: bool,
    ) -> ActionResult:
        """Apply one delivery rung, optionally focusing via its own tool.

        ``bring_to_front`` is never an input-action property: when requested,
        the separately approved standalone focus action runs first, then the
        original foreground input runs unchanged.
        """
        refusal = self._apply_delivery(action, args, delivery_mode)
        if refusal is not None:
            return refusal
        if bring_to_front:
            if delivery_mode != "foreground":
                return ActionResult(ok=False, action=action, code="bring_to_front_requires_foreground",
                                    message="bring_to_front requires delivery_mode='foreground'.")
            if not self._session._has_tool("bring_to_front"):
                return ActionResult(ok=False, action=action, code="bring_to_front_unsupported",
                                    delivery_mode="foreground", message=_BTF_UNSUPPORTED_MSG)
            if self._active_pid is None or self._active_window_id is None:
                return ActionResult(
                    ok=False, action=action, code="bring_to_front_target_required",
                    delivery_mode="foreground",
                    message="Capture an exact target before requesting persistent foreground focus.",
                )
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
        result = self._action(action, args)
        if bring_to_front:
            result.meta["foreground_focus"] = {"invoked": True, "tool": "bring_to_front"}
        return result

    # ── Pointer ────────────────────────────────────────────────────
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("click")
        if missing is not None:
            return missing
        # Tool is chosen by click_count only; `button` goes through click's
        # enum (the driver rejects unknown buttons). `right_click` /
        # `middle_click` MCP tools are deprecated aliases and never invoked here.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return ActionResult(ok=False, action="click",
                                message=f"unknown button {button!r} — expected left, right, middle.")
        tool = "double_click" if click_count == 2 else "click"

        args: Dict[str, Any] = {"pid": self._active_pid, "button": button_norm}
        if element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for element_index click.")
            args["element_index"] = element
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for coordinate click.")
            args["x"] = x
            args["y"] = y
        else:
            return ActionResult(ok=False, action=tool, message="click requires element= or x/y.")
        args["window_id"] = self._active_window_id
        if modifiers:
            args["modifier"] = modifiers
        return self._run_input_action(tool, args, delivery_mode, bring_to_front)

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("drag")
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid}
        if from_element is not None and to_element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for element-based drag.")
            args["from_element"] = from_element
            args["to_element"] = to_element
        elif from_xy is not None and to_xy is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for coordinate drag.")
            args["from_x"], args["from_y"] = int(from_xy[0]), int(from_xy[1])
            args["to_x"], args["to_y"] = int(to_xy[0]), int(to_xy[1])
        else:
            return ActionResult(ok=False, action="drag",
                                message="drag requires from_element/to_element or from_coordinate/to_coordinate.")
        args["window_id"] = self._active_window_id
        return self._run_input_action("drag", args, delivery_mode, bring_to_front)

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("scroll")
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid, "direction": direction,
                                "amount": max(1, min(50, amount))}
        if element is not None and self._active_window_id is not None:
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="scroll",
                                    message="No active window_id for coordinate scroll.")
            # Some driver schemas reject x/y on scroll: only send coordinates
            # when the driver advertises support; otherwise it scrolls the
            # targeted window (window_id is still sent for routing).
            if self._session.supports_capability("input.scroll.coordinates", tool="scroll"):
                args["x"] = x
                args["y"] = y
            args["window_id"] = self._active_window_id
        return self._run_input_action("scroll", args, delivery_mode, bring_to_front)

    # ── Keyboard ───────────────────────────────────────────────────
    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult:
        missing = self._no_target("type_text", need_window=True)
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id, "text": text}
        return self._run_input_action("type_text", args, delivery_mode, bring_to_front)

    def key(self, keys: str, *, delivery_mode: Optional[str] = None,
            bring_to_front: bool = False) -> ActionResult:
        missing = self._no_target("key", need_window=True)
        if missing is not None:
            return missing
        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return ActionResult(ok=False, action="key",
                                message=f"Could not parse key from '{keys}'.")
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id}
        if modifiers:  # hotkey requires at least one modifier + one key
            args["keys"] = modifiers + [key_name]
            return self._run_input_action("hotkey", args, delivery_mode, bring_to_front)
        args["key"] = key_name
        return self._run_input_action("press_key", args, delivery_mode, bring_to_front)

    # ── Value setter ────────────────────────────────────────────────
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        missing = self._no_target("set_value", need_window=True)
        if missing is not None:
            return missing
        if element is None:
            return ActionResult(ok=False, action="set_value",
                                message="set_value requires element= (element index).")
        return self._action("set_value", {"pid": self._active_pid, "window_id": self._active_window_id,
                                          "element_index": element, "value": value})

    # ── Introspection ──────────────────────────────────────────────
    def list_apps(self) -> List[Dict[str, Any]]:
        out = self._session.call_tool("list_apps", {"session": self._session_id})
        structured = out.get("structuredContent")
        data = out.get("data")

        # structuredContent is canonical; empty lists fall through so a
        # populated compatibility envelope (older drivers, CLI fallback) can
        # still recover.
        if isinstance(structured, dict):
            apps = structured.get("apps")
            if isinstance(apps, list) and apps:
                return apps
        if isinstance(data, list) and data:
            return data
        for container in (data, out):
            if isinstance(container, dict):
                apps = container.get("apps")
                if isinstance(apps, list) and apps:
                    return apps

        derived = _apps_from_windows(_windows_from_tool_result(out))
        if derived:
            return derived

        # Old text-only drivers retain a small, name/PID-only fallback.
        if isinstance(data, str):
            return [
                {"name": m.group(1).strip(), "pid": int(m.group(2))}
                for m in (re.search(r'(.+?)\s+\(pid\s+(\d+)\)', line) for line in data.splitlines())
                if m
            ]
        return []

    def list_windows(self) -> List[Dict[str, Any]]:
        return self._load_windows()

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Target an app: a pure window-selector (store pid/window_id so later
        input hits the right process) — background automation never needs to
        raise a window. ``raise_window=True`` is explicit, separately approved,
        and uses the standalone ``bring_to_front`` tool.
        """
        try:
            windows = self._load_windows()
        except Exception:
            self._clear_active_target()
            raise

        matched = self._match_windows_for_app(windows, app)
        # No silent fallback to the frontmost window: that hides the real
        # failure (often a localized macOS app-name mismatch).
        if not matched:
            self._clear_active_target()
            return ActionResult(ok=False, action="focus_app",
                                message=f"No on-screen window found for app '{app}'.")
        target = matched[0]
        self._set_active_target(target)
        self._last_app = target["app_name"] or app  # retained for back-compat diagnostics
        if raise_window:
            if not self._session._has_tool("bring_to_front"):
                return ActionResult(ok=False, action="focus_app", code="bring_to_front_unsupported",
                                    message=_BTF_UNSUPPORTED_MSG)
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
            focused.action = "focus_app"
            focused.meta["target_selected"] = True
            return focused
        return ActionResult(ok=True, action="focus_app",
                            message=f"Targeted {target['app_name']} (pid {self._active_pid}, "
                                    f"window {self._active_window_id}) without raising window.")

    # ── App lifecycle ────────────────────────────────────────────────
    def launch_app(
        self,
        *,
        bundle_id: Optional[str] = None,
        name: Optional[str] = None,
        urls: Optional[List[str]] = None,
        additional_arguments: Optional[List[str]] = None,
        creates_new_application_instance: bool = False,
    ) -> Dict[str, Any]:
        """Idempotent launch returning ``{pid, bundle_id, name, windows[]}``.
        ``creates_new_application_instance=True`` forces a fresh instance so
        concurrent runs touching the same app get isolated windows."""
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        args: Dict[str, Any] = {"session": self._session_id}
        for key, value in (("bundle_id", bundle_id), ("name", name), ("urls", urls and list(urls)),
                           ("additional_arguments", additional_arguments and list(additional_arguments)),
                           ("creates_new_application_instance", creates_new_application_instance or None)):
            if value:
                args[key] = value
        out = self._session.call_tool("launch_app", args)
        return out["structuredContent"] or {"data": out["data"]}

    def bring_to_front(self, *, pid: int, window_id: Optional[int] = None) -> ActionResult:
        """Activate a window so subsequent foreground-dispatched input lands on it."""
        args: Dict[str, Any] = {"pid": int(pid)}
        if window_id is not None:
            args["window_id"] = int(window_id)
        # The live schema is strict and has no session property: this is a
        # standalone native focus operation, not a session-scoped input action.
        return self._action("bring_to_front", args, inject_session=False)

    # ── Agent cursor / config ────────────────────────────────────────
    def set_agent_cursor_enabled(self, enabled: bool, *,
                                 cursor_id: Optional[str] = None) -> ActionResult:
        """Toggle the agent cursor overlay's visibility for this run."""
        args: Dict[str, Any] = {"enabled": bool(enabled)}
        if cursor_id:
            args["cursor_id"] = cursor_id
        return self._action("set_agent_cursor_enabled", args)

    def set_config(self, **config) -> ActionResult:
        """Set cua-driver config keys (e.g. ``max_image_dimension``). Unknown
        keys pass through verbatim — cua-driver validates its own schema."""
        return self._action("set_config", dict(config))

    # ── Generic escape hatch ────────────────────────────────────────
    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None,
                  *, timeout: float = 30.0) -> Dict[str, Any]:
        """Call any cua-driver MCP tool by name. ``session`` is injected via
        setdefault, so this is the supported path for tools the wrapper does
        not type-wrap (preferred over ``self._session.call_tool``)."""
        payload = dict(args) if args else {}
        payload.setdefault("session", self._session_id)
        return self._session.call_tool(name, payload, timeout=timeout)

    # ── Internal ───────────────────────────────────────────────────
    def _maybe_attach_element_token(self, tool: str, args: Dict[str, Any]) -> None:
        """Attach the last snapshot's ``element_token`` for an ``element_index``
        call. The token takes precedence and yields an explicit 'stale' error
        when the snapshot was superseded. Gated on the per-tool capability so
        older drivers (``additionalProperties: false``) never see the field."""
        idx = args.get("element_index")
        if not isinstance(idx, int):
            return
        token = self._snapshot_tokens.get(idx)
        if not token:
            return
        if not self._session.supports_capability("accessibility.element_tokens", tool=tool):
            return
        args["element_token"] = token

    def _action(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        inject_session: bool = True,
    ) -> ActionResult:
        self._maybe_attach_element_token(name, args)
        # setdefault preserves any explicit session a caller already supplied.
        if inject_session:
            args.setdefault("session", self._session_id)
        try:
            out = self._session.call_tool(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        ok = not out["isError"]
        data = out["data"]
        structured = out.get("structuredContent") or {}
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message", ""))
        elif isinstance(data, str):
            message = data
        if not message and isinstance(structured, dict):
            message = str(structured.get("message", ""))
        # Merge data + structuredContent into meta, structured winning on
        # overlap (it is the canonical verdict surface).
        meta: Dict[str, Any] = {}
        if isinstance(data, dict):
            meta.update(data)
        if isinstance(structured, dict):
            meta.update(structured)
        return _action_result_from(name, ok, message, meta, structured,
                                   requested_delivery=args.get("delivery_mode"))
