"""Lazy dependency bootstrapper for non-Python runtime deps.

Detection and prompting live here in Python — not in install.sh — because: 1. shutil.which() works
on every platform; install.sh needs bash. 2. Detection is instant; spawning bash for a "is node
installed?" check is waste. 3. Python controls the UX (rich prompts, non-interactive fallback, TTY
detection).

install.sh is still the *installation* backend because it has 1900 lines of battle-tested OS
detection and package-manager logic (apt/brew/pacman/dnf/ zypper/Termux/…). Reimplementing that in
Python would be huge duplication.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import agent_browser_runnable, find_node_executable
from tools.environments.local import hermes_subprocess_env

_IS_WINDOWS = platform.system() == "Windows"

_DEP_CHECKS = {
    # find_node_executable() rather than a bare which(): $HERMES_HOME/node is
    # not on PATH, so which() would report Node missing on an install that has
    # a managed one and trigger a redundant re-install.
    "node": lambda: find_node_executable("node") is not None,
    "browser": lambda: (
        agent_browser_runnable(shutil.which("agent-browser"))
        or _has_system_browser()
        or _has_hermes_agent_browser()
        or _has_npx_agent_browser()
    ),
    "ripgrep": lambda: shutil.which("rg") is not None,
    "ffmpeg": lambda: shutil.which("ffmpeg") is not None,
}

_DEP_DESCRIPTIONS = {
    "node": "Node.js (required for browser tools and TUI)",
    "browser": "Browser engine (Chromium, for web browsing tools)",
    "ripgrep": "ripgrep (fast file search)",
    "ffmpeg": "ffmpeg (TTS voice messages)",
}


def _has_system_browser() -> bool:
    names = (
        ("chrome", "msedge", "chromium") if _IS_WINDOWS
        else ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")
    )
    return any(shutil.which(name) for name in names)


def _has_npx_agent_browser() -> bool:
    """agent-browser resolves lazily via npx on the default install (#43564), invisible to the
    PATH/managed-dir probes above. Mirror tools.browser_tool.check_browser_requirements's Termux
    carve-out so this check can't diverge from what browser tools actually find.
    """
    try:
        from tools.browser_tool import (
            _find_agent_browser,
            _is_npx_agent_browser_sentinel,
            _requires_real_termux_browser_install,
        )
        browser_cmd = _find_agent_browser(validate=False)
    except Exception:
        return False
    return _is_npx_agent_browser_sentinel(browser_cmd) and not _requires_real_termux_browser_install(
        browser_cmd
    )


def _has_hermes_agent_browser() -> bool:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    if _IS_WINDOWS:
        # npm -g --prefix puts .cmd shims directly in the prefix dir on Windows
        return (home / "node" / "agent-browser.cmd").is_file()
    # install.sh installs globally into $HERMES_HOME/node/bin/ via npm -g --prefix
    # Also check legacy node_modules/.bin/ path for git-clone installs.
    return (
        (home / "node" / "bin" / "agent-browser").is_file()
        or (home / "node_modules" / ".bin" / "agent-browser").is_file()
    )


def _find_install_script(
    package_dir: Path | None = None, repo_root: Path | None = None
) -> tuple[Path | None, str | None]:
    """Locate the install script — bundled in wheel or in git checkout."""
    package_dir = package_dir or Path(__file__).parent
    repo_root = repo_root or package_dir.parent
    candidates = [("install.sh", "bash"), ("install.ps1", "powershell")]
    if _IS_WINDOWS:
        candidates.reverse()
    for script_name, shell in candidates:
        for base in (package_dir, repo_root):
            script = base / "scripts" / script_name
            if script.is_file():
                return script, shell
    return None, None


def ensure_dependency(dep: str, interactive: bool = True) -> bool:
    """Ensure a non-Python dependency is available. Returns True if available."""
    check = _DEP_CHECKS.get(dep)
    if check is None:
        # Unknown dep — don't silently forward to install script.
        return False
    if check():
        return True

    script, shell = _find_install_script()
    desc = _DEP_DESCRIPTIONS.get(dep, dep)
    if script is None:
        if interactive:
            print(f"  {desc} is not installed and no install script was found.")
            print(f"  Install {dep} manually and try again.")
        return False

    if interactive and sys.stdin.isatty():
        try:
            reply = input(f"{desc} is not installed. Install now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if reply not in ("", "y", "yes"):
            return False

    if shell == "powershell":
        from hermes_constants import get_hermes_home
        ps_bin = shutil.which("powershell") or shutil.which("pwsh")
        if not ps_bin:
            if interactive:
                print("  PowerShell not found. Install PowerShell or run install.ps1 manually.")
            return False
        cmd = [
            ps_bin, "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-Ensure", dep, "-HermesHome", str(get_hermes_home()),
        ]
    else:
        cmd = ["bash", str(script), "--ensure", dep]

    run_env = hermes_subprocess_env(inherit_credentials=False)
    run_env["IS_INTERACTIVE"] = "false"
    result = subprocess.run(cmd, env=run_env)
    return result.returncode == 0 and check()
