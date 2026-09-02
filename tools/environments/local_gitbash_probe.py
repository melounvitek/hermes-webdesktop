"""Git Bash health probe + Mandatory-ASLR diagnostics (Windows), used by
``local._find_bash`` to pick a bash.exe that can actually launch MSYS children
and to build the targeted remediation when none can."""

import logging
import ntpath
import platform
import shutil
import subprocess

from hermes_cli._subprocess_compat import windows_hide_flags

_IS_WINDOWS = platform.system() == "Windows"

# Same logger as the origin module so log routing is unchanged.
logger = logging.getLogger("tools.environments.local")

_bash_starts_cache: dict[str, bool] = {}
_bash_probe_details_cache: dict[str, str] = {}
_mandatory_aslr_enabled_cache: "bool | None" = None

# External ``true`` and ``cat`` are intentional: a builtin-only ``exit 0`` probe
# misses Git-for-Windows fork/spawn failures under system-wide Mandatory ASLR.
_BASH_EXTERNAL_PROGRAM_PROBE = "/usr/bin/true; /usr/bin/cat --version >/dev/null"

_MSYS_SPAWN_FAILURE_MARKERS = ("dofork:", "child_copy:", "0xc0000142", "0xc0000005")


def _looks_like_msys_spawn_failure(details: str) -> bool:
    """Match Git-for-Windows child-launch failures associated with ASLR."""
    lowered = details.lower()
    return any(marker in lowered for marker in _MSYS_SPAWN_FAILURE_MARKERS)


def _mandatory_aslr_enabled() -> "bool | None":
    """Return Windows' system-wide ForceRelocateImages state when available."""
    global _mandatory_aslr_enabled_cache
    if _mandatory_aslr_enabled_cache is not None:
        return _mandatory_aslr_enabled_cache

    try:
        powershell = shutil.which("powershell.exe") or "powershell.exe"
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-ProcessMitigation -System).Aslr.ForceRelocateImages.ToString()",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip().upper()
        if value == "ON":
            _mandatory_aslr_enabled_cache = True
            return True
        if value in {"OFF", "NOTSET"}:
            _mandatory_aslr_enabled_cache = False
            return False
    except Exception as exc:
        logger.debug("Could not query Windows Mandatory ASLR state: %s", exc)
    return None


def _git_root_from_bash(bash: str) -> str:
    """Resolve Git's root from either <root>/bin or <root>/usr/bin bash."""
    bin_dir = ntpath.dirname(ntpath.normpath(bash))
    if ntpath.basename(bin_dir).lower() != "bin":
        return ntpath.dirname(bin_dir)
    parent = ntpath.dirname(bin_dir)
    if ntpath.basename(parent).lower() == "usr":
        return ntpath.dirname(parent)
    return parent


def _git_bash_aslr_help(bash: str, details: str = "") -> str:
    """Build the targeted per-program Mandatory-ASLR remediation."""
    git_root = _git_root_from_bash(bash)
    escaped_root = git_root.replace("'", "''")
    detail_line = f"\nGit Bash probe output: {details[:500]}" if details else ""
    return (
        f"Git Bash at {bash} cannot launch required MSYS child processes while "
        "Windows Mandatory ASLR (ForceRelocateImages) is enabled, or its output "
        f"matches that Git-for-Windows failure class.{detail_line}\n"
        "Reinstalling Git will not change the Windows mitigation policy. Open "
        "PowerShell as Administrator and run:\n"
        f"$gitRoot = '{escaped_root}'\n"
        'Get-Item "$gitRoot\\bin\\bash.exe", "$gitRoot\\usr\\bin\\*.exe" '
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "Set-ProcessMitigation -Name $_.FullName -Disable ForceRelocateImages }\n"
        "Then restart Hermes. If the override is blocked or later re-applied, "
        "ask your Windows administrator to allow this per-program exception."
    )


def _bash_starts(bash: str) -> bool:
    """True if *bash* can launch external MSYS programs (cached per path).
    ``--noprofile --norc`` so a broken login post-install (``Directory
    \\drivers\\etc``) does not falsely condemn an otherwise usable bash."""
    cached = _bash_starts_cache.get(bash)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [bash, "--noprofile", "--norc", "-c", _BASH_EXTERNAL_PROGRAM_PROBE],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15,
            creationflags=windows_hide_flags() if _IS_WINDOWS else 0,
        )
        ok = result.returncode == 0
        if not ok:
            combined = f"{result.stdout or ''}{result.stderr or ''}"
            _bash_probe_details_cache[bash] = combined.strip()[:2000]
            logger.debug("bash probe failed for %s: %s", bash, combined.strip()[:200])
    except Exception as exc:
        _bash_probe_details_cache[bash] = str(exc)[:2000]
        logger.debug("bash probe error for %s: %s", bash, exc)
        ok = False

    _bash_starts_cache[bash] = ok
    return ok
