"""Desktop (Electron) app: build/stamp, stage-and-swap pack, exe integrity gate, macOS signing/TCC, Linux sandbox, launch (hermes gui/desktop).

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import logging
import argparse
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time as _time_mod

from pathlib import Path
from typing import Optional
from hermes_cli.main_tui_launch import _npm_lifecycle_env
from hermes_cli.main_web_build import _hash_source_tree, _nixos_build_env, _stamp_is_current, _write_build_stamp

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")

_PREVIOUS_APP_KEPT = "  ↩ The previous desktop app was left untouched and still works."


def _desktop_dist_exists(desktop_dir: Path) -> bool:
    """Return True when a local desktop renderer build is present."""
    return (desktop_dir / "dist" / "index.html").exists()


def _compute_desktop_content_hash(project_root: Path) -> str:
    """SHA-256 of ``apps/desktop/`` (minus .gitignore matches) plus root workspace config."""
    return _hash_source_tree(project_root, project_root / "apps" / "desktop")


def _desktop_stamp_path() -> Path:
    """Path of the desktop build stamp under $HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "desktop-build-stamp.json"


def _renderer_bundle_dir(desktop_dir: Path, *, source_mode: bool) -> Optional[Path]:
    """The renderer ``dist`` directory a launch loads, when it is inspectable.

    Source mode builds to ``apps/desktop/dist``. A packaged app ships the bundle
    inside ``app.asar`` and (``asarUnpack: dist/**``) beside it in
    ``app.asar.unpacked``; only the unpacked copy is a real directory, and it is
    the one an interrupted replace tears.
    """
    from hermes_cli.main import _desktop_packaged_executable
    if source_mode:
        return desktop_dir / "dist"

    executable = _desktop_packaged_executable(desktop_dir)
    if executable is None:
        return None

    # macOS: …/Hermes.app/Contents/MacOS/Hermes → …/Contents/Resources
    resources = (
        executable.parent.parent / "Resources" if sys.platform == "darwin" else executable.parent / "resources"
    )
    return resources / "app.asar.unpacked" / "dist"


# The module files the renderer fetches before any app code runs: Vite emits
# them as `<script type="module" src>` plus `<link rel="modulepreload" href>`.
_HTML_TAG_WITH_URL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["'][^>]*>""", re.IGNORECASE)

_MODULE_TAG = re.compile(r"""\btype=["']module["']|\brel=["']modulepreload["']""", re.IGNORECASE)


def _renderer_bundle_torn(dist_dir: Path) -> bool:
    """True when ``index.html`` names hashed module files that aren't there.

    ``index.html`` and the hashed chunks under ``assets/`` are ONE generation. An
    update that replaces the app while its files are locked can leave them from
    different generations; the app then dies on its first lazy import, and
    because the content stamp still matches the intact SOURCE tree the rebuild
    that would fix it is skipped. Conservative: an unreadable index, or one naming
    nothing checkable, is NOT torn — the missing-bundle guards own those cases.
    """
    try:
        html = (dist_dir / "index.html").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    for match in _HTML_TAG_WITH_URL.finditer(html):
        href = match.group(1)
        # Absolute/CDN URLs aren't part of this bundle's generation.
        if not _MODULE_TAG.search(match.group(0)) or re.match(r"^[a-z]+:|^//", href, re.IGNORECASE):
            continue
        rel = href.split("?", 1)[0].split("#", 1)[0].lstrip("./")
        if rel and not (dist_dir / rel).exists():
            return True

    return False


def _desktop_build_needed(desktop_dir: Path, project_root: Path, *, source_mode: bool) -> bool:
    """True when the desktop build output is stale, missing, torn, or built in the other mode."""
    from hermes_cli.main import _desktop_dist_exists, _desktop_packaged_executable, _desktop_stamp_path
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            return True
    elif _desktop_packaged_executable(desktop_dir) is None:
        return True

    # A torn bundle is stale no matter what the stamp says: the hash describes
    # the intact SOURCE tree, not the half-replaced output.
    dist_dir = _renderer_bundle_dir(desktop_dir, source_mode=source_mode)
    if dist_dir is not None and _renderer_bundle_torn(dist_dir):
        print(f"  ⚠ A previous update left the desktop bundle incomplete ({dist_dir}); rebuilding it")
        return True

    return not _stamp_is_current(
        _desktop_stamp_path(), lambda: _compute_desktop_content_hash(project_root), sourceMode=source_mode
    )


def _write_desktop_build_stamp(project_root: Path, *, source_mode: bool) -> None:
    """Write the desktop build stamp after a successful build."""
    from hermes_cli.main import _desktop_stamp_path
    _write_build_stamp(
        _desktop_stamp_path(), "desktop",
        lambda: _compute_desktop_content_hash(project_root), sourceMode=source_mode,
    )


def _desktop_packaged_executable(desktop_dir: Path) -> Optional[Path]:
    """Return the current platform's unpacked Electron app executable."""
    return _desktop_packaged_executable_in(desktop_dir / "release")


def _desktop_packaged_executable_in(release_dir: Path) -> Optional[Path]:
    """The unpacked Electron app executable under *release_dir* (live ``release`` or a staging dir)."""
    from hermes_cli.main import _expected_windows_pe_machines
    if sys.platform == "darwin":
        candidates = list(release_dir.glob("mac*/Hermes.app/Contents/MacOS/Hermes"))
    elif sys.platform == "win32":
        candidates = [
            release_dir / d / "Hermes.exe" for d in ("win-unpacked", "win-ia32-unpacked", "win-arm64-unpacked")
        ]
    else:
        candidates = [
            release_dir / d / n for d in ("linux-unpacked", "linux-arm64-unpacked") for n in ("hermes", "Hermes")
        ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    if sys.platform == "win32" and len(existing) > 1:
        # A stale win-arm64-unpacked next to the real win-unpacked: picking by
        # mtime can hand a wrong-architecture Hermes.exe to the launcher. Prefer
        # candidates whose PE machine matches the host; mtime when none parse.
        expected = _expected_windows_pe_machines()
        matching = [p for p in existing if _pe_machine_or_none(p) in expected]
        if matching:
            existing = matching
    return max(existing, key=lambda p: p.stat().st_mtime)


_DESKTOP_STAGING_PREFIX = ".staging-"

_DESKTOP_PREVIOUS_SUFFIX = ".previous"


def _desktop_staging_dir(desktop_dir: Path) -> Path:
    """Fresh, unique staging output dir: ``apps/desktop/.staging-<pid>-<ts>``.

    A sibling of ``release/`` (same filesystem → the swap is a rename) but NOT
    inside it, so nothing globbing ``release/*-unpacked`` can mistake the
    half-built tree for the live app. Stale leftovers are swept first.
    """
    for stale in desktop_dir.glob(f"{_DESKTOP_STAGING_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)
    return desktop_dir / f"{_DESKTOP_STAGING_PREFIX}{os.getpid()}-{int(_time_mod.time())}"


def _desktop_unpacked_root(exe: Path, release_dir: Path) -> Path:
    """The dir directly under *release_dir* holding *exe* (electron-builder's ``appOutDir``, swapped whole)."""
    unpacked = exe
    while unpacked.parent != release_dir:
        if unpacked.parent == unpacked:
            raise ValueError(f"{exe} is not under {release_dir}")
        unpacked = unpacked.parent
    return unpacked


def _swap_staged_desktop_app(desktop_dir: Path, staging_dir: Path) -> Optional[Path]:
    """Promote a VERIFIED staged pack over the live ``release/`` app.

    ``release/<unpacked>`` → ``.previous``, ``<staging>/<unpacked>`` →
    ``release/<unpacked>``, then drop ``.previous``. The only window with no live
    app is between the two renames, and a failure there rolls back. Returns the
    live executable, or None (live app untouched or restored). Never raises.
    """
    staged_exe = _desktop_packaged_executable_in(staging_dir)
    if staged_exe is None:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return None
    release_dir = desktop_dir / "release"
    try:
        staged_root = _desktop_unpacked_root(staged_exe, staging_dir)
        live_root = release_dir / staged_root.name
        previous = release_dir / (staged_root.name + _DESKTOP_PREVIOUS_SUFFIX)
        release_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(previous, ignore_errors=True)
        moved_aside = live_root.exists()
        if moved_aside:
            os.rename(live_root, previous)
        try:
            os.rename(staged_root, live_root)
        except OSError:
            if moved_aside:
                os.rename(previous, live_root)  # restore; live app back as it was
            raise
        if moved_aside:
            shutil.rmtree(previous, ignore_errors=True)
    except (OSError, ValueError) as exc:
        logger.warning("desktop stage-and-swap failed, live app kept: %s", exc)
        return None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return live_root / staged_exe.relative_to(staged_root)


def _discard_desktop_staging(staging_dir: Path) -> None:
    shutil.rmtree(staging_dir, ignore_errors=True)


_PE_MACHINE_I386 = 0x014C
_PE_MACHINE_AMD64 = 0x8664
_PE_MACHINE_ARM64 = 0xAA64

_PE_MACHINE_NAMES = {
    _PE_MACHINE_I386: "x86 (32-bit)", _PE_MACHINE_AMD64: "x64 (AMD64)", _PE_MACHINE_ARM64: "ARM64",
}

_PE_MACHINE_TO_NAME = {_PE_MACHINE_ARM64: "ARM64", _PE_MACHINE_AMD64: "AMD64", _PE_MACHINE_I386: "X86"}

# MACHINE_ATTRIBUTES bits (processthreadsapi.h). UserEnabled means the host
# can run user-mode code of that machine type — natively or under emulation.
_MACHINE_ATTRIBUTE_USER_ENABLED = 0x00000001


def _windows_native_machine_from_iswow64() -> Optional[str]:
    """Ask IsWow64Process2 for the OS-native machine (None if unavailable/fail).

    ``restype``/``argtypes`` are bound to ``wintypes.HANDLE``: ctypes' default
    ``c_int`` truncates the ``(HANDLE)-1`` pseudo-handle to ``0xFFFFFFFF`` and
    ``IsWow64Process2`` then fails with ``ERROR_INVALID_HANDLE`` on Win64.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.IsWow64Process2.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT),
    ]
    kernel32.IsWow64Process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not kernel32.IsWow64Process2(
        kernel32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine)
    ):
        return None
    return _PE_MACHINE_TO_NAME.get(native_machine.value)


def _windows_user_runnable_pe_machines() -> Optional[set]:
    """PE machines this host can run in user mode, via GetMachineTypeAttributes.

    The only documented API that reports AMD64-on-ARM64 emulation support. None
    when unavailable (pre-Windows-11 build 22000) or nothing runnable, so callers
    fall back to name-based detection.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetMachineTypeAttributes.argtypes = [wintypes.USHORT, ctypes.POINTER(ctypes.c_int)]
    kernel32.GetMachineTypeAttributes.restype = ctypes.c_long

    runnable = set()
    for machine in (_PE_MACHINE_ARM64, _PE_MACHINE_AMD64, _PE_MACHINE_I386):
        attributes = ctypes.c_int(0)
        # HRESULT: zero is success, any nonzero value is a failure.
        if kernel32.GetMachineTypeAttributes(machine, ctypes.byref(attributes)):
            continue
        if attributes.value & _MACHINE_ATTRIBUTE_USER_ENABLED:
            runnable.add(machine)
    return runnable or None


def _windows_native_machine() -> str:
    """The Windows host OS's NATIVE machine architecture, normalized upper.

    ``platform.machine()`` reports the PROCESS architecture, which lies under
    emulation (x64 Python on Windows-on-ARM reports AMD64). Probe order:
    ``IsWow64Process2`` (the only API that tells the truth from an emulated
    process), ``PROCESSOR_ARCHITEW6432`` / ``PROCESSOR_ARCHITECTURE``, then
    ``platform.machine()``. ``GetNativeSystemInfo`` is deliberately NOT used: it
    also returns emulated details under emulation.
    """
    if sys.platform == "win32":
        try:
            name = _windows_native_machine_from_iswow64()
        except (OSError, AttributeError, TypeError, ValueError):
            name = None  # API missing, DLL load failure in tests, mistyped binding
        if name:
            return name
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE")
        if env_arch:
            return env_arch.upper()
    import platform as _platform

    return (_platform.machine() or "").upper()


def _expected_windows_pe_machines() -> set:
    """PE machine values the current Windows host can natively load.

    ``GetMachineTypeAttributes`` first; else name-based: AMD64 runs x64 + x86
    (WOW64), ARM64 runs ARM64 + x64 (emulation), x86 runs only x86. Unknown
    machines return the permissive full set so the gate can never brick launch.
    """
    from hermes_cli.main import _windows_native_machine
    if sys.platform == "win32":
        try:
            runnable = _windows_user_runnable_pe_machines()
        except (OSError, AttributeError, TypeError, ValueError):
            runnable = None
        if runnable:
            return runnable
    machine = _windows_native_machine().upper()
    if machine in ("AMD64", "X86_64", "X64"):
        return {_PE_MACHINE_AMD64, _PE_MACHINE_I386}
    if machine in ("ARM64", "AARCH64"):
        return {_PE_MACHINE_ARM64, _PE_MACHINE_AMD64}
    if machine in ("X86", "I386", "I486", "I586", "I686"):
        return {_PE_MACHINE_I386}
    return {_PE_MACHINE_AMD64, _PE_MACHINE_ARM64, _PE_MACHINE_I386}


def _parse_pe_machine(path: Path) -> int:
    """Parse ``path`` as a PE executable and return its COFF machine field.

    Raises ``ValueError`` with a human-readable reason when the file is not a
    structurally complete PE (missing MZ/PE magic, truncated header, or section
    data past EOF — the truncated-download shape). Header walk only; cheap.
    """
    import struct

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unreadable: {exc}")
    if file_size < 512:
        raise ValueError(f"file is only {file_size} bytes — far too small to be a Windows executable")
    with path.open("rb") as fh:
        head = fh.read(64)
        if len(head) < 64 or head[:2] != b"MZ":
            raise ValueError(
                "missing MZ header — not a Windows executable (a truncated or non-binary file saved as .exe?)"
            )
        e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
        if e_lfanew <= 0 or e_lfanew + 24 > file_size:
            raise ValueError("corrupt DOS header: PE header offset points past end of file")
        fh.seek(e_lfanew)
        pe_head = fh.read(24)
        if len(pe_head) < 24 or pe_head[:4] != b"PE\x00\x00":
            raise ValueError("missing PE signature — corrupt executable header")
        machine, n_sections = struct.unpack_from("<HH", pe_head, 4)
        size_of_optional = struct.unpack_from("<H", pe_head, 20)[0]
        fh.seek(e_lfanew + 24 + size_of_optional)
        max_section_end = 0
        for _ in range(n_sections):
            section = fh.read(40)
            if len(section) < 40:
                raise ValueError("truncated PE section table")
            size_of_raw, pointer_to_raw = struct.unpack_from("<II", section, 16)
            max_section_end = max(max_section_end, pointer_to_raw + size_of_raw)
        if file_size < max_section_end:
            raise ValueError(
                f"truncated executable: file is {file_size} bytes but its PE sections extend to {max_section_end} bytes"
            )
    return machine


def _pe_machine_or_none(path: Path) -> Optional[int]:
    from hermes_cli.main import _parse_pe_machine
    try:
        return _parse_pe_machine(path)
    except ValueError:
        return None


def _desktop_exe_integrity_error(path: Path) -> Optional[str]:
    """Why ``path`` cannot run on this Windows host, or None when it parses as a loadable PE."""
    from hermes_cli.main import _expected_windows_pe_machines, _parse_pe_machine, _windows_native_machine
    try:
        machine = _parse_pe_machine(path)
    except ValueError as exc:
        return str(exc)
    if machine not in _expected_windows_pe_machines():
        got = _PE_MACHINE_NAMES.get(machine, f"unknown machine 0x{machine:04X}")
        return (
            f"architecture mismatch: built a {got} executable but this is a "
            f"{_windows_native_machine()} Windows host"
        )
    return None


def _desktop_backup_unpacked_dir(packaged_executable: Path) -> Path:
    """The rollback tree before-pack.mjs preserves: ``<unpacked-dir>.bak``."""
    unpacked = packaged_executable.parent
    return unpacked.parent / (unpacked.name + ".bak")


def _rollback_desktop_from_backup(packaged_executable: Path) -> Optional[Path]:
    """Restore the previous unpacked desktop app from its ``.bak`` tree.

    None when no usable backup exists (missing, or fails the same integrity
    probe). The corrupt tree is kept as ``<unpacked-dir>.corrupt``. Never raises.
    """
    unpacked = packaged_executable.parent
    backup_dir = _desktop_backup_unpacked_dir(packaged_executable)
    backup_exe = backup_dir / packaged_executable.name
    if not backup_exe.exists() or _desktop_exe_integrity_error(backup_exe) is not None:
        return None
    corrupt_dir = unpacked.parent / (unpacked.name + ".corrupt")
    try:
        shutil.rmtree(corrupt_dir, ignore_errors=True)
        try:
            unpacked.rename(corrupt_dir)
        except OSError:
            shutil.rmtree(unpacked, ignore_errors=True)
        backup_dir.rename(unpacked)
    except OSError:
        return None
    restored = unpacked / packaged_executable.name
    return restored if restored.exists() else None


def _ensure_desktop_exe_launchable(desktop_dir: Path, packaged_executable: Optional[Path]) -> tuple:
    """Windows post-build integrity gate for the self-update rebuild.

    Returns ``(verified_exe_or_None, rolled_back)``: probe passed → ``(exe, False)``;
    corrupt/wrong-arch with backup restored → ``(old_exe, True)``; nothing
    restorable → ``(None, False)``. On failure the cached Electron zip is purged
    and the stamp invalidated so the retry-once rebuild pulls a fresh,
    SHASUM-verified download. No-op off Windows / with no executable.
    """
    from hermes_cli.main import _desktop_stamp_path, _purge_electron_build_cache, _rollback_desktop_from_backup
    if packaged_executable is None or sys.platform != "win32":
        return packaged_executable, False

    error = _desktop_exe_integrity_error(packaged_executable)
    if error is None:
        return packaged_executable, False

    print(f"✗ The built Hermes.exe failed its integrity check: {error}")
    print(f"    at: {packaged_executable}")

    # Only the exe's OWN output dir is purged (a staging dir), never the live
    # release/ tree that still holds the last working app.
    _purge_electron_build_cache(desktop_dir, release_dir=packaged_executable.parent.parent)
    try:
        _desktop_stamp_path().unlink()
    except OSError:
        pass

    restored = _rollback_desktop_from_backup(packaged_executable)
    if restored is not None:
        print("  ↩ Update aborted — restored the previous working Hermes.exe from backup.")
        print("    Your existing version was kept and still works. Run `hermes desktop`")
        print("    (or the in-app update) again to retry with a fresh Electron download.")
        return restored, True

    print("  ✗ No usable backup was found to restore.")
    print("    Run `hermes desktop --force-build` to rebuild, or re-run the Hermes")
    print("    installer to repair the install.")
    return None, False


def _electron_download_cache_dirs() -> list[Path]:
    """Per-user Electron download cache directories for this OS (deduped, in priority order).

    electron-builder's ``app-builder unpack-electron`` extracts Electron from a
    zip stored here (NOT from node_modules), so a corrupt zip here poisons the
    build. Honors the ``electron_config_cache`` / ``ELECTRON_CACHE`` overrides
    ``@electron/get`` respects.
    """
    home = Path.home()
    candidates: list[Path] = []
    override = os.environ.get("electron_config_cache") or os.environ.get("ELECTRON_CACHE")
    if override:
        candidates.append(Path(override))
    if sys.platform == "darwin":
        candidates.append(home / "Library" / "Caches" / "electron")
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "electron" / "Cache")
        candidates.append(home / "AppData" / "Local" / "electron" / "Cache")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            candidates.append(Path(xdg) / "electron")
        candidates.append(home / ".cache" / "electron")

    return list(dict.fromkeys(c.expanduser() for c in candidates))


def _purge_electron_build_cache(desktop_dir: Path, release_dir: Optional[Path] = None) -> list[Path]:
    """Clear the cached Electron download + half-written unpacked dir so the next pack restarts from scratch.

    Root cause of ``ENOENT … rename 'linux-unpacked/electron'``: a corrupt zip in
    the per-user cache (resumed partial download with prepended junk, or a
    truncated write) unpacks to a tree MISSING the ``electron`` binary, and
    re-running repeats the same extraction forever. We deliberately do NOT
    validate the zip ourselves: stdlib ``zipfile`` tolerates exactly the
    concat-junk that ``@electron/get`` rejects, so a gate would never self-heal.
    Instead purge unconditionally and let the caller retry once; ``@electron/get``
    re-downloads with SHASUM verification (the real source of truth).

    ``release_dir`` lets a stage-and-swap caller point at its STAGING output so
    the purge never touches the live app. Never raises; returns removed paths
    (empty ⇒ nothing to clear, so no point retrying).
    """
    from hermes_cli.main import _electron_download_cache_dirs
    removed: list[Path] = []

    for cache_dir in _electron_download_cache_dirs():
        if not cache_dir.is_dir():
            continue
        for zip_path in sorted(cache_dir.rglob("electron-*.zip")):
            try:
                zip_path.unlink()
                removed.append(zip_path)
            except OSError:
                pass  # locked/permission-denied: let the build report its own error

    if release_dir is None:
        release_dir = desktop_dir / "release"
    if release_dir.is_dir():
        for unpacked in release_dir.glob("*-unpacked"):
            try:
                shutil.rmtree(unpacked, ignore_errors=True)
                removed.append(unpacked)
            except OSError:
                pass

    return removed


# Last-resort Electron mirror after GitHub download fails. Only used when the
# user hasn't pinned ELECTRON_MIRROR.
_ELECTRON_FALLBACK_MIRROR = "https://npmmirror.com/mirrors/electron/"


def _electron_dir(project_root: Path) -> Path:
    """The Electron package directory the desktop workspace installs.

    npm may keep workspace-only dev deps under ``apps/desktop/node_modules`` or
    hoist them to the root depending on version; ``apps/desktop/package.json``
    points ``electronDist`` at the workspace-local path, so prefer that.
    """
    desktop_local = project_root / "apps" / "desktop" / "node_modules" / "electron"
    if desktop_local.exists():
        return desktop_local
    return project_root / "node_modules" / "electron"


def _electron_dist_binary(project_root: Path) -> Path:
    """The Electron main binary inside the installed package — the exact file ``electronDist`` needs."""
    dist = _electron_dir(project_root) / "dist"
    if sys.platform == "darwin":
        return dist / "Electron.app" / "Contents" / "MacOS" / "Electron"
    if sys.platform == "win32":
        return dist / "electron.exe"
    return dist / "electron"


def _electron_dist_ok(project_root: Path) -> bool:
    """True when ``node_modules/electron/dist`` holds a usable binary (a partial dir counts as NOT ok)."""
    from hermes_cli.main import _electron_dist_binary
    try:
        return _electron_dist_binary(project_root).exists()
    except OSError:
        return False


def _electron_pkg_staged_missing_dist(project_root: Path) -> bool:
    """electron staged (package.json + install.js) but dist missing — blocked postinstall."""
    from hermes_cli.main import _electron_dist_ok
    electron_dir = _electron_dir(project_root)
    return (
        (electron_dir / "package.json").is_file()
        and (electron_dir / "install.js").is_file()
        and not _electron_dist_ok(project_root)
    )


def _redownload_electron_dist(project_root: Path, env: dict, *, mirror: Optional[str] = None) -> bool:
    """Best-effort: run electron's install.js to populate dist/ (optional mirror)."""
    from hermes_cli.main import _electron_dist_ok
    if _electron_dist_ok(project_root):
        return True

    electron_dir = _electron_dir(project_root)
    installer = electron_dir / "install.js"
    if not installer.is_file():
        return False
    from hermes_constants import find_node_executable, with_hermes_node_path

    node = find_node_executable("node")
    if not node:
        return False

    shutil.rmtree(electron_dir / "dist", ignore_errors=True)
    try:
        (electron_dir / "path.txt").unlink()
    except OSError:
        pass

    dl_env = with_hermes_node_path(env)
    if mirror:
        dl_env["ELECTRON_MIRROR"] = mirror
    try:
        subprocess.run([node, str(installer)], cwd=str(electron_dir), env=dl_env, check=False)
    except OSError:
        return False
    return _electron_dist_ok(project_root)


def _try_redownload_electron_dist(project_root: Path, env: dict) -> bool:
    """Canonical download, then fallback mirror unless the user pinned one."""
    from hermes_cli.main import _redownload_electron_dist
    if _redownload_electron_dist(project_root, env):
        return True
    if env.get("ELECTRON_MIRROR"):
        return False
    return _redownload_electron_dist(project_root, env, mirror=_ELECTRON_FALLBACK_MIRROR)


def _stop_desktop_processes_locking_build(desktop_dir: Path) -> list[int]:
    """Terminate a running desktop app executing from this build's ``release`` dir (Windows only).

    A running ``Hermes.exe`` holds an exclusive lock, so electron-builder's pack
    dies with ``Access is denied`` and the retry repeats it. POSIX lets you
    unlink a running binary, so this is a no-op off Windows. Only processes whose
    exe lives INSIDE this desktop's ``release`` tree are stopped. Never raises;
    returns the PIDs asked to stop.
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
        release_dir = (desktop_dir / "release").resolve()
    except Exception:
        return []
    if not release_dir.is_dir():
        return []

    me = os.getpid()
    victims = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
            pid = info.get("pid")
            exe = info.get("exe")
            if not exe or pid is None or pid == me:
                continue
            exe_path = Path(exe).resolve()
        except Exception:
            continue
        if release_dir in exe_path.parents:
            victims.append(proc)

    stopped: list[int] = []
    for proc in victims:
        try:
            proc.terminate()
            stopped.append(int(proc.pid))
        except Exception:
            continue
    if stopped:
        # Wait for the handles (and thus the file locks) to actually release.
        try:
            _, alive = psutil.wait_procs(victims, timeout=5)
            for proc in alive:
                try:
                    proc.kill()
                except Exception:
                    continue
        except Exception:
            pass
    return stopped


def _desktop_macos_bundle_id(bundle: Path) -> Optional[str]:
    """Return a bundle/framework CFBundleIdentifier for local macOS signing."""
    import plistlib

    info = bundle / "Contents" / "Info.plist"
    if not info.exists() and bundle.suffix == ".framework":
        candidates = list(bundle.glob("Versions/*/Resources/Info.plist")) + list(
            bundle.glob("Resources/Info.plist")
        )
        if candidates:
            info = candidates[0]
    if not info.exists():
        return None
    try:
        data = plistlib.loads(info.read_bytes())
    except Exception:
        return None
    ident = data.get("CFBundleIdentifier")
    return str(ident) if ident else None


def _desktop_macos_local_signing_identity() -> Optional[str]:
    """The opt-in keychain identity for local macOS desktop signing (``desktop.macos_signing_identity``).

    A persistent code-signing cert (a self-signed one from Keychain Access is
    enough) gives the app a certificate-anchored Designated Requirement — the
    strongest way to keep TCC grants stable across rebuilds. Empty/unset keeps
    identifier-pinned ad-hoc signing.
    """
    if sys.platform != "darwin":
        return None
    try:
        from hermes_cli.config import load_config

        desktop = load_config().get("desktop", {})
        if not isinstance(desktop, dict):
            return None
        identity = desktop.get("macos_signing_identity")
        if not isinstance(identity, str):
            return None
        return identity.strip() or None
    except Exception as exc:
        print(
            "  (warning: could not load desktop.macos_signing_identity: "
            f"{exc}; falling back to ad-hoc signing)"
        )
        return None


def _codesign_verify(codesign: str, app: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [codesign, "--verify", "--deep", "--strict", str(app)], capture_output=True, **kwargs
    )


def _desktop_macos_has_valid_real_signature(app: Path) -> bool:
    """True when the bundle carries an intact non-ad-hoc (Team ID) signature.

    Makes the relaunch fixup a no-op on properly signed/notarized builds even
    without CSC_LINK / APPLE_SIGNING_IDENTITY in the environment — clobbering a
    Developer ID signature with an ad-hoc one resets TCC grants. A STALE real
    signature fails --verify and returns False so the fixup can repair it.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    try:
        info = subprocess.run(
            [codesign, "-dv", str(app)], check=False, capture_output=True, text=True
        )
        output = f"{info.stdout}\n{info.stderr}"
        if info.returncode != 0 or "TeamIdentifier=" not in output or "TeamIdentifier=not set" in output:
            return False
        return _codesign_verify(codesign, app, check=False).returncode == 0
    except Exception:
        return False


def _desktop_macos_local_codesign(app: Path, *, desktop_dir: Path, identity: str = "-") -> bool:
    """Re-sign a local Desktop build so macOS TCC grants survive rebuilds.

    A plain ``codesign --deep --sign -`` leaves a cdhash-only Designated
    Requirement (changes every rebuild → TCC re-prompts everything) and strips
    electron-builder's entitlements (breaks microphone/JIT under the hardened
    runtime). Instead sign inside-out (standalone Mach-O, nested frameworks/
    helpers, main bundle), preserving the repo's entitlement plists, and pin an
    identifier-based DR when ad-hoc. Raises on signing failure; True after
    strict verification passes.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False

    ent_main = desktop_dir / "electron" / "entitlements.mac.plist"
    ent_inherit = desktop_dir / "electron" / "entitlements.mac.inherit.plist"
    if not (ent_main.exists() and ent_inherit.exists()):
        # Hardened-runtime restrictions apply to ad-hoc signatures too; signing
        # with --options runtime but WITHOUT allow-jit would leave Electron/V8
        # crashing on launch. Bail so the caller falls back to the legacy sign.
        raise FileNotFoundError(f"desktop entitlement plists missing under {desktop_dir / 'electron'}")

    def sign_path(
        path: Path,
        *,
        entitlements: Optional[Path] = None,
        identifier: Optional[str] = None,
        runtime: bool = True,
    ) -> None:
        args = [codesign, "--force", "--sign", identity, "--timestamp=none"]
        if runtime:
            args += ["--options", "runtime"]
        if entitlements is not None and entitlements.exists():
            args += ["--entitlements", str(entitlements)]
        if identifier and identity == "-":
            # Ad-hoc signatures get a cdhash-only DR by default; pin an
            # identifier-based DR so TCC has something stable to persist.
            args += ["--requirements", f'=designated => identifier "{identifier}"']
        args.append(str(path))
        subprocess.run(args, check=True, capture_output=True)

    # 1) Standalone Mach-O files (native modules, dylibs, crashpad handler),
    #    compared relative to the app root — the absolute path always contains
    #    the outer Hermes.app component.
    contents = app / "Contents"
    standalone: list[Path] = []
    for root, _dirs, files in os.walk(contents):
        root_path = Path(root)
        if any(part.endswith(".app") for part in root_path.relative_to(app).parts):
            continue  # nested helper apps are signed as bundles below
        for name in files:
            fp = root_path / name
            if name in {"chrome_crashpad_handler", "spawn-helper"} or fp.suffix in {".node", ".dylib"}:
                standalone.append(fp)
    for fp in sorted(standalone, key=lambda p: len(p.parts), reverse=True):
        sign_path(fp, runtime=False)

    # 2) Nested frameworks and helper apps, deepest first.
    bundles: set[Path] = set()
    frameworks_dir = contents / "Frameworks"
    if frameworks_dir.exists():
        for root, _dirs, _files in os.walk(frameworks_dir):
            p = Path(root)
            if p.suffix in {".framework", ".app"}:
                bundles.add(p)
    for bundle in sorted(bundles, key=lambda p: len(p.parts), reverse=True):
        ent = ent_inherit if bundle.suffix == ".app" and "Helper" in bundle.name else None
        sign_path(bundle, entitlements=ent, identifier=_desktop_macos_bundle_id(bundle))

    # 3) The main bundle, with the app's own entitlements.
    sign_path(app, entitlements=ent_main, identifier=_desktop_macos_bundle_id(app))
    _codesign_verify(codesign, app, check=True)
    return True


def _macos_legacy_adhoc_resign(codesign: str, app: Path) -> bool:
    """Legacy deep ad-hoc re-sign fallback; NEVER deletes the safeStorage keychain item.

    Deleting it would permanently orphan every credential encrypted under it,
    and this path is reached exactly when the entitlement-preserving signer
    failed, so there is no verified successor identity. The keychain prompt
    macOS shows instead is recoverable ("Always Allow"); deletion is not.
    """
    try:
        result = subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(app)], check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign failed (exit {result.returncode}); "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        if _codesign_verify(codesign, app, check=False, text=True).returncode != 0:
            print(
                "  (warning: legacy ad-hoc re-sign did not pass strict verification; "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        print("  → macOS desktop re-signed (legacy ad-hoc); safeStorage keychain item left untouched")
        return True
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")
    return False


def _desktop_macos_relaunchable_fixup(
    desktop_dir: Path,
    *,
    publisher_signing_configured: Optional[bool] = None,
    release_dir: Optional[Path] = None,
) -> bool:
    """Make a locally-built macOS app survive in-place self-update without resetting TCC grants.

    An ad-hoc-signed .app has no stable Designated Requirement, so a rebuilt
    bundle (new cdhash) reports "Hermes is damaged" and loses every TCC grant.
    Clear quarantine xattrs, then re-sign with ``desktop.macos_signing_identity``
    when configured, else identifier-pinned ad-hoc, preserving entitlements. No-op
    when a publisher identity is configured (CSC_LINK / APPLE_SIGNING_IDENTITY —
    callers may pass the decision so a later dotenv load can't reverse it) or the
    bundle already carries an intact Developer ID signature. ``release_dir``
    signs the STAGED bundle before promotion. Falls back to the legacy deep
    ad-hoc sign. Never raises; True when no work was needed or signing verified.
    """
    from hermes_cli.main import _desktop_macos_has_valid_real_signature, _desktop_macos_local_codesign, _desktop_macos_local_signing_identity
    if sys.platform != "darwin":
        return True
    if publisher_signing_configured is None:
        publisher_signing_configured = bool(
            os.environ.get("CSC_LINK") or os.environ.get("APPLE_SIGNING_IDENTITY")
        )
    if publisher_signing_configured:
        return True
    exe = _desktop_packaged_executable_in(release_dir or (desktop_dir / "release"))
    if exe is None:
        return True
    # exe = .../Hermes.app/Contents/MacOS/Hermes  ->  app bundle = .../Hermes.app
    app = exe.parents[2]
    if not str(app).endswith(".app") or not app.is_dir():
        return True
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    if _desktop_macos_has_valid_real_signature(app):
        return True
    subprocess.run(["xattr", "-cr", str(app)], check=False)
    identity = _desktop_macos_local_signing_identity() or "-"
    try:
        if _desktop_macos_local_codesign(app, desktop_dir=desktop_dir, identity=identity):
            label = "keychain identity" if identity != "-" else "stable ad-hoc identity"
            print(f"  → macOS desktop signed with {label}; TCC grants persist across rebuilds")
            return True
    except Exception as exc:
        if identity != "-":
            print(
                f"  (warning: configured macOS signing identity failed: {identity!r}; "
                "falling back to ad-hoc — TCC grants may need to be re-granted)"
            )
        print(f"  (warning: stable macOS signing failed ({exc}); using legacy ad-hoc sign)")
    return _macos_legacy_adhoc_resign(codesign, app)


def _macos_codesigning_identity_valid(security: str, identity: str) -> bool:
    """True when `identity` appears among VALID code-signing identities.

    ``find-identity`` without ``-v`` also lists certs macOS refuses to sign with
    (imported but never trusted for codeSign); only the ``-v`` listing proves
    codesign can use it. Both the idempotency probe and the success
    postcondition for ``--setup-tcc-identity``. Never raises.
    """
    try:
        result = subprocess.run(
            [security, "find-identity", "-v", "-p", "codesigning"], capture_output=True, text=True, check=False,
        )
    except Exception:
        return False

    return f'"{identity}"' in (result.stdout or "")


def _macos_create_signing_identity(
    openssl: str, security: str, codesign: str, keychain: str, identity: str
) -> bool:
    """Create a self-signed code-signing cert (10 years), import it with codesign access, trust it for codeSign."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="hermes-tcc-"))
    try:
        key = tmp_dir / "sign.key"
        crt = tmp_dir / "sign.crt"
        p12 = tmp_dir / "sign.p12"
        subprocess.run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(crt),
                "-days", "3650", "-nodes",
                "-subj", f"/CN={identity}",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,digitalSignature,keyCertSign",
                "-addext", "extendedKeyUsage=codeSigning",
            ],
            capture_output=True, check=True,
        )

        # OpenSSL 3 defaults to AES/SHA-2 PKCS#12 that `security import` rejects
        # with "MAC verification failed". `-legacy` restores the accepted
        # RC2/SHA-1 format but only exists on OpenSSL 3 — so try plain first and
        # fall back to `-legacy` when the IMPORT fails with that signature.
        def _export_p12(extra_args: list) -> None:
            subprocess.run(
                [
                    openssl, "pkcs12", "-export", *extra_args,
                    "-inkey", str(key), "-in", str(crt),
                    "-out", str(p12), "-passout", "pass:hermeslocal",
                ],
                capture_output=True, check=True,
            )

        def _import_p12():
            return subprocess.run(
                [
                    security, "import", str(p12), "-k", keychain,
                    "-P", "hermeslocal",
                    "-T", codesign, "-T", "/usr/bin/codesign_allocate",
                ],
                capture_output=True, text=True, check=False,
            )

        _export_p12([])
        imported = _import_p12()
        if imported.returncode != 0 and "MAC verification failed" in (imported.stderr or ""):
            try:
                _export_p12(["-legacy"])
                imported = _import_p12()
            except subprocess.CalledProcessError:
                pass  # older OpenSSL without -legacy: keep the original failure
        if imported.returncode != 0:
            print(f"  (could not import signing identity into keychain: {imported.stderr.strip()})")
            return False

        # Without explicit trust for the codeSign policy `find-identity -v`
        # reports 0 valid identities. This writes user trust settings, so macOS
        # may prompt for the login password ONCE — the one-time cost this
        # command exists to front-load.
        trusted = subprocess.run(
            [security, "add-trusted-cert", "-r", "trustRoot", "-p", "codeSign", "-k", keychain, str(crt)],
            capture_output=True, text=True, check=False,
        )
        if trusted.returncode != 0:
            print(
                "  (could not trust the certificate for code signing: "
                f"{(trusted.stderr or trusted.stdout).strip()})"
            )
            return False
        print(f"  → created, imported, and trusted self-signed identity: {identity!r}")
        return True
    except Exception as exc:
        print(f"  (certificate creation failed: {exc})")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _desktop_macos_setup_tcc_identity(identity: str = "Hermes Local Signing") -> bool:
    """Create/import a self-signed code-signing cert and configure Hermes to use it.

    One-shot setup for ``hermes desktop --setup-tcc-identity``: creates the cert
    in the login keychain, grants ``codesign`` access, writes
    ``desktop.macos_signing_identity`` to config.yaml, and re-signs the packaged
    app. TCC grants persist against the code-signing identity, not the path; a
    certificate-anchored identity is stable across rebuilds (the yabai/skhd
    mechanism). Idempotent; True on success or already configured. Never raises.
    """
    from hermes_cli.main import PROJECT_ROOT, _desktop_macos_relaunchable_fixup, _desktop_packaged_executable
    if sys.platform != "darwin":
        print("  (--setup-tcc-identity is macOS-only; skipping)")
        return False

    openssl = shutil.which("openssl")
    security = shutil.which("security")
    codesign = shutil.which("codesign")
    if not (openssl and security and codesign):
        print(
            "  (--setup-tcc-identity requires openssl, security, and codesign; "
            f"found openssl={bool(openssl)} security={bool(security)} codesign={bool(codesign)})"
        )
        return False

    keychain = str(Path.home() / "Library" / "Keychains" / "login.keychain-db")
    # Probe with `-v` (valid identities only) so a previously imported-but-
    # untrusted cert is repaired rather than reported as done.
    if _macos_codesigning_identity_valid(security, identity):
        print(f"  → identity {identity!r} already valid in keychain")
    elif not _macos_create_signing_identity(openssl, security, codesign, keychain, identity):
        return False

    # Postcondition gate: name-in-output checks pass for invalid identities;
    # only macOS agreeing the identity is usable counts.
    if not _macos_codesigning_identity_valid(security, identity):
        print(
            f"  (identity {identity!r} was imported but is not a VALID code-signing identity; "
            "run `security find-identity -v -p codesigning` to inspect, and see the manual "
            "Keychain Access steps in the desktop docs)"
        )
        return False

    # config.yaml, not .env — it's not a secret.
    try:
        from hermes_cli.config import set_config_value

        set_config_value("desktop.macos_signing_identity", identity)
        print(f"  → set desktop.macos_signing_identity = {identity!r}")
    except Exception as exc:
        print(f"  (could not write desktop.macos_signing_identity: {exc})")
        return False

    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if _desktop_packaged_executable(desktop_dir) is not None:
        try:
            if _desktop_macos_relaunchable_fixup(desktop_dir):
                print(
                    "  → packaged app re-signed with certificate-anchored identity; "
                    "TCC grants persist across rebuilds"
                )
        except Exception as exc:
            print(f"  (could not re-sign packaged app: {exc})")

    print(
        "\n  Note: macOS will re-prompt for permissions ONE final time (the identity "
        "changed). Grant them and they persist from then on. If a permission gets "
        "stuck, reset it with:  tccutil reset All com.nousresearch.hermes"
    )
    return True


def _force_adhoc_macos_signing(env: dict, *, source_mode: bool) -> bool:
    """Stop electron-builder grabbing a random keychain identity on self-update.

    The self-updater re-signs the .app on the end user's machine; with
    ``CSC_IDENTITY_AUTO_DISCOVERY`` on, electron-builder signs the hardened-
    runtime bundle with whatever personal cert it finds, which stalls the sign
    step or clobbers a real notarized signature. Force ad-hoc for the local
    packaged rebuild instead. No-op for source runs, off-macOS, with a real
    identity configured, or when the caller pinned the flag. Mutates ``env``.
    """
    if sys.platform != "darwin" or source_mode:
        return False
    if env.get("CSC_LINK") or env.get("APPLE_SIGNING_IDENTITY") or "CSC_IDENTITY_AUTO_DISCOVERY" in env:
        return False
    env["CSC_IDENTITY_AUTO_DISCOVERY"] = "false"
    return True


def _desktop_linux_needs_no_sandbox() -> bool:
    """True when Chromium/Electron should bypass the Linux sandbox.

    Ubuntu 23.10+ ``apparmor_restrict_unprivileged_userns`` breaks the userns
    sandbox unless the app ships a root-owned 4755 ``chrome-sandbox``; when we
    can't ``sudo chown/chmod`` it, fall back to ``--no-sandbox`` rather than
    hard-failing. Deliberately NOT True for root: Electron as root without a
    sandbox is a qualitatively riskier path and must stay an explicit choice.
    """
    if os.environ.get("ELECTRON_DISABLE_SANDBOX", 0) == "1":
        return True

    if sys.platform != "linux":
        return False
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return False
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _desktop_linux_userns_sandbox_available() -> bool:
    """True when Chromium's unprivileged user-namespace sandbox works.

    Then Chromium never consults the setuid ``chrome-sandbox`` helper, so
    requiring it root-owned 4755 (and prompting for sudo) is unnecessary. Probe
    the real capability with ``unshare``; fails closed on hosts where user
    namespaces are disabled or AppArmor-restricted.
    """
    if sys.platform != "linux":
        return False
    unshare = shutil.which("unshare")
    if not unshare:
        return False
    try:
        return (
            subprocess.run(
                [unshare, "--user", "--map-root-user", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _sandbox_helper_lstat(packaged_executable: Path) -> tuple[Path, Optional[os.stat_result]]:
    """``(chrome-sandbox path, lstat or None)`` — lstat so a symlink is inspected, not followed."""
    sandbox = packaged_executable.parent / "chrome-sandbox"
    try:
        return sandbox, sandbox.lstat()
    except OSError:
        return sandbox, None


def _sandbox_helper_is_setuid_root(st: os.stat_result) -> bool:
    return st.st_uid == 0 and stat.S_IMODE(st.st_mode) == 0o4755


def _desktop_linux_sandbox_helper_is_regular_file(packaged_executable: Path) -> bool:
    """Return True when ``chrome-sandbox`` exists as a regular file."""
    if sys.platform != "linux":
        return False
    _sandbox, st = _sandbox_helper_lstat(packaged_executable)
    return st is not None and stat.S_ISREG(st.st_mode)


def _desktop_linux_sandbox_fixup(packaged_executable: Path) -> bool:
    """Configure Electron's Linux SUID sandbox helper when required."""
    from hermes_cli.main import _desktop_linux_userns_sandbox_available
    if sys.platform != "linux":
        return True

    sandbox, st = _sandbox_helper_lstat(packaged_executable)
    if not sandbox.exists():
        print(f"✗ Hermes Desktop is missing Electron's Linux sandbox helper: {sandbox}")
        return False
    # Reject symlinks — chown/chmod must not follow an attacker-controlled link.
    if st is None:
        print(f"✗ Cannot stat Electron's Linux sandbox helper: {sandbox}")
        return False
    if not stat.S_ISREG(st.st_mode):
        print(f"✗ Electron's Linux sandbox helper is not a regular file: {sandbox}")
        return False

    if _sandbox_helper_is_setuid_root(st):
        return True

    if _desktop_linux_userns_sandbox_available():
        print("✓ Using Chromium's user-namespace sandbox (setuid helper not needed).")
        return True

    sudo = shutil.which("sudo")
    if not sudo:
        print("✗ Hermes Desktop requires sudo to configure Electron's Linux sandbox helper.")
        return False

    print("→ Configuring Electron Linux sandbox helper (sudo required)...")
    for command in ([sudo, "chown", "root:root", str(sandbox)], [sudo, "chmod", "4755", str(sandbox)]):
        if subprocess.run(command, check=False).returncode != 0:
            print(f"✗ Failed to configure Electron's Linux sandbox helper: {sandbox}")
            return False
    return True


def _desktop_linux_needs_disable_setuid_sandbox(packaged_executable: Path) -> bool:
    """True when Chromium should skip the present-but-non-setuid helper.

    A user-owned ``chrome-sandbox`` still makes Chromium abort with
    ``setuid_sandbox_host`` even when the namespace sandbox works. Call only
    after ``_desktop_linux_sandbox_fixup`` succeeded via the userns path.
    """
    if sys.platform != "linux":
        return False
    _sandbox, st = _sandbox_helper_lstat(packaged_executable)
    return st is not None and stat.S_ISREG(st.st_mode) and not _sandbox_helper_is_setuid_root(st)


_LINUX_PASSWORD_STORES = frozenset({"gnome-libsecret", "kwallet", "kwallet5", "kwallet6", "basic"})


def _detect_linux_password_store() -> str | None:
    """Detect the Chromium password-store backend for the current Linux session.

    safeStorage only reports encryption available when Chromium selects the right
    keychain backend, and its own detection routinely fails under ``hermes
    desktop`` (launcher env doesn't look like a desktop session). Probe order:
    KDE session env, GNOME Keyring control socket, D-Bus ping of
    org.freedesktop.secrets (any Secret Service, e.g. KeePassXC). None when no
    keychain daemon is reachable.
    """
    kde_version = os.environ.get("KDE_SESSION_VERSION", "").strip()
    if kde_version:
        return {"6": "kwallet6", "5": "kwallet5"}.get(kde_version, "kwallet")
    if os.environ.get("KDE_FULL_SESSION"):
        return "kwallet"
    if os.environ.get("GNOME_KEYRING_CONTROL"):
        return "gnome-libsecret"
    try:
        result = subprocess.run(
            [
                "dbus-send", "--session", "--print-reply", "--reply-timeout=2000",
                "--dest=org.freedesktop.secrets",
                "/org/freedesktop/secrets",
                "org.freedesktop.DBus.Peer.Ping",
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "gnome-libsecret"
    except Exception:
        pass
    return None


def _desktop_launch_options() -> tuple[list[str], str, str, str]:
    """Read `desktop.*` launch options from config.yaml.

    Returns ``(electron_flags, disable_gpu, password_store, ozone_hint)``:
    ``disable_gpu`` is "auto"/"1"/"0" (for HERMES_DESKTOP_DISABLE_GPU),
    ``password_store`` is "auto" or a Chromium backend (unknown → "auto"),
    ``ozone_hint`` is "auto"/"x11"/"wayland" (for ELECTRON_OZONE_PLATFORM_HINT).
    Any config error yields ``([], "auto", "auto", "auto")`` so a malformed
    config never blocks the launch.
    """
    flags: list[str] = []
    disable_gpu = password_store = ozone_hint = "auto"
    try:
        from hermes_cli.config import load_config

        desktop_cfg = (load_config() or {}).get("desktop") or {}
    except Exception:
        return flags, disable_gpu, password_store, ozone_hint

    raw_flags = desktop_cfg.get("electron_flags")
    if isinstance(raw_flags, str):
        flags = shlex.split(raw_flags, posix=(os.name != "nt"))
    elif isinstance(raw_flags, (list, tuple)):
        flags = [str(f) for f in raw_flags if str(f).strip()]

    raw_gpu = desktop_cfg.get("disable_gpu", "auto")
    if isinstance(raw_gpu, bool):
        disable_gpu = "1" if raw_gpu else "0"
    elif isinstance(raw_gpu, str):
        low = raw_gpu.strip().lower()
        if low in ("1", "true", "yes", "on"):
            disable_gpu = "1"
        elif low in ("0", "false", "no", "off"):
            disable_gpu = "0"

    raw_store = desktop_cfg.get("password_store", "auto")
    if isinstance(raw_store, str) and raw_store.strip().lower() in _LINUX_PASSWORD_STORES:
        password_store = raw_store.strip().lower()

    raw_ozone = desktop_cfg.get("ozone_platform_hint", "auto")
    if isinstance(raw_ozone, str) and raw_ozone.strip().lower() in ("auto", "x11", "wayland"):
        ozone_hint = raw_ozone.strip().lower()
    return flags, disable_gpu, password_store, ozone_hint


def _register_linux_desktop_entry() -> None:
    """Install the XDG desktop entry for Hermes Desktop (Linux only, best-effort).

    ``Exec`` and ``Icon`` are absolute so the entry works outside a login shell.
    ``hermes uninstall --gui`` removes it.
    """
    from hermes_cli.main import PROJECT_ROOT
    try:
        from hermes_cli.linux_desktop_entry import install_desktop_entry, is_supported

        if not is_supported():
            return
        entry = install_desktop_entry(PROJECT_ROOT)
        if entry:
            print(f"✓ Desktop launcher entry installed: {entry}")
    except Exception as exc:  # never block a launch on launcher plumbing
        print(f"⚠ Could not install the desktop launcher entry: {exc}")


def _install_desktop_workspace_deps(npm: str, env: dict) -> None:
    """npm-install the desktop workspace; exits on a failure that isn't a repairable missing Electron dist."""
    from hermes_cli.main import PROJECT_ROOT, _run_npm_install_deterministic
    from hermes_constants import with_hermes_node_path

    print("→ Installing desktop workspace dependencies...")
    # Managed Node on PATH so npm's child scripts that shell out to bare `node`
    # (e.g. electron-winstaller's select-7z-arch.js) resolve it even when the
    # desktop updater chain lost shell PATH customizations. Wrapping the NixOS
    # env keeps its PYTHON hint while restoring managed Node ahead of PATH.
    nixos_env = with_hermes_node_path(_nixos_build_env())
    install_result = _run_npm_install_deterministic(npm, PROJECT_ROOT, capture_output=False, env=nixos_env)
    if install_result.returncode == 0:
        return
    if not _electron_pkg_staged_missing_dist(PROJECT_ROOT):
        print("✗ Desktop dependency install failed")
        print(f"  Run manually:  cd {PROJECT_ROOT} && npm ci")
        sys.exit(install_result.returncode or 1)
    if _try_redownload_electron_dist(PROJECT_ROOT, env):
        print("  ⚠ Dependency install failed with a missing Electron dist; "
              "repopulated it and continuing.")
    else:
        print("  ⚠ Dependency install failed with a missing Electron dist; "
              "continuing to the build so electron-builder can attempt "
              "the Electron fetch itself.")


def _run_desktop_pack_with_recovery(
    desktop_dir: Path, build_cmd: list[str], npm_build_env: dict, env: dict, staging_dir: Optional[Path]
) -> subprocess.CompletedProcess:
    """Run the desktop build; a packaged build with NO staged exe retries after an Electron re-download, then via mirror.

    Gate on a MISSING packaged executable: that is the signature of the
    corrupt-download class (corrupt cached zip → partial unpack → ENOENT on
    rename). A late failure such as macOS code signing leaves the executable in
    place — redownloading can't repair it, so the retry would only add a slow,
    identical failure.
    """
    from hermes_cli.main import PROJECT_ROOT, _electron_dist_ok, _purge_electron_build_cache, _redownload_electron_dist, _stop_desktop_processes_locking_build

    def _staged_exe() -> Optional[Path]:
        return _desktop_packaged_executable_in(staging_dir) if staging_dir else None

    def _pack(run_env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(build_cmd, cwd=desktop_dir, env=run_env, check=False)

    build_result = _pack(npm_build_env)
    if build_result.returncode != 0 and staging_dir is not None and _staged_exe() is None:
        purged: list[Path] = []
        restored = False
        if not _electron_dist_ok(PROJECT_ROOT):
            purged = _purge_electron_build_cache(desktop_dir, release_dir=staging_dir)
            restored = _redownload_electron_dist(PROJECT_ROOT, env)
        if restored:
            print("  ⚠ Desktop build failed; refreshed the Electron download and retrying once...")
            for p in purged:
                print(f"    - {p}")
            # The purge can't remove a win-unpacked tree whose Hermes.exe is
            # still locked by a running instance; stop it before retry.
            _stop_desktop_processes_locking_build(desktop_dir)
            build_result = _pack(npm_build_env)
    if (
        build_result.returncode != 0
        and staging_dir is not None
        and not env.get("ELECTRON_MIRROR")
        and _staged_exe() is None
    ):
        print("  ⚠ Desktop build still failing; the Electron download from "
              "GitHub looks blocked. Re-downloading via a public mirror "
              "(npmmirror.com)... (set ELECTRON_MIRROR to use another mirror)")
        mirror_env = {**npm_build_env, "ELECTRON_MIRROR": _ELECTRON_FALLBACK_MIRROR}
        if not _electron_dist_ok(PROJECT_ROOT):
            _redownload_electron_dist(PROJECT_ROOT, env, mirror=_ELECTRON_FALLBACK_MIRROR)
        _stop_desktop_processes_locking_build(desktop_dir)
        build_result = _pack(mirror_env)
    return build_result


def _promote_staged_desktop_app(desktop_dir: Path, staging_dir: Path) -> Path:
    """Sign + integrity-gate the STAGED pack, then swap it over the live app. Exits (live app kept) on failure."""
    from hermes_cli.main import _desktop_macos_relaunchable_fixup, _ensure_desktop_exe_launchable, _swap_staged_desktop_app
    staged_executable = _desktop_packaged_executable_in(staging_dir)
    # Locally-built apps are ad-hoc signed; make them relaunchable after an
    # in-place self-update. Signs the STAGED bundle so the live app is never
    # half-signed. No-op on non-macOS and on real-identity builds.
    _desktop_macos_relaunchable_fixup(desktop_dir, release_dir=staging_dir)

    # Windows integrity gate: never declare the rebuild a success on a
    # Hermes.exe Windows cannot load. Verified on the STAGED exe, so a failure
    # simply discards staging and fails loudly for the updater's retry-once.
    verified_executable, rolled_back = _ensure_desktop_exe_launchable(desktop_dir, staged_executable)
    if staged_executable is None or rolled_back or verified_executable is None:
        _discard_desktop_staging(staging_dir)
        if staged_executable is None:
            print(f"✗ Desktop build produced no launchable app in {staging_dir}")
        print(_PREVIOUS_APP_KEPT)
        sys.exit(1)
    packaged_executable = _swap_staged_desktop_app(desktop_dir, staging_dir)
    if packaged_executable is None:
        print(f"✗ Could not install the rebuilt desktop app into {desktop_dir / 'release'}")
        print(_PREVIOUS_APP_KEPT)
        sys.exit(1)
    return packaged_executable


def _build_desktop_app(desktop_dir: Path, *, source_mode: bool, npm: str, env: dict) -> Optional[Path]:
    """npm-install + build the desktop app; stage-and-swap the packaged tree.

    Returns the freshly installed packaged executable (non-source mode) or None
    (source mode builds ``dist/`` in place). Exits on any unrecoverable build
    failure, leaving the previous packaged app untouched.
    """
    from hermes_cli.main import PROJECT_ROOT, _desktop_packaged_executable, _desktop_staging_dir, _stop_desktop_processes_locking_build, _write_desktop_build_stamp

    _install_desktop_workspace_deps(npm, env)

    build_label = "source build" if source_mode else "packaged app"
    print(f"→ Building desktop {build_label}...")
    build_script = "build" if source_mode else "pack"
    if _force_adhoc_macos_signing(env, source_mode=source_mode):
        print("  → No Developer ID configured; ad-hoc signing this local rebuild "
              "(CSC_IDENTITY_AUTO_DISCOVERY=false)")
    npm_build_env = _npm_lifecycle_env(env)
    # Stage-and-swap: electron-builder packs IN PLACE and before-pack.mjs wipes
    # release/<unpacked> first, so a pack that fails afterwards used to leave
    # the user with NO app. Build into a staging dir; the live release/ tree is
    # only replaced — by rename — after the staged result verifies.
    staging_dir: Optional[Path] = None
    build_cmd = [npm, "run", build_script]
    if not source_mode:
        staging_dir = _desktop_staging_dir(desktop_dir)
        build_cmd += ["--", f"-c.directories.output={staging_dir}"]
        # A running desktop instance holds Hermes.exe locked on Windows, so the
        # pack can't replace it ("Access is denied"). Stop it first.
        stopped = _stop_desktop_processes_locking_build(desktop_dir)
        if stopped:
            print(f"  ⚠ Stopped running desktop app to free the build output (pid {', '.join(map(str, stopped))})")

    build_result = _run_desktop_pack_with_recovery(desktop_dir, build_cmd, npm_build_env, env, staging_dir)
    if build_result.returncode != 0:
        print("✗ Desktop GUI build failed")
        if staging_dir is not None:
            _discard_desktop_staging(staging_dir)
            if _desktop_packaged_executable(desktop_dir) is not None:
                print(_PREVIOUS_APP_KEPT)
        print(f"  Run manually:  cd apps/desktop && npm run {build_script}")
        if sys.platform == "win32":
            print("  If this says \"Access is denied\" on Hermes.exe, close any")
            print("  running Hermes desktop window and retry.")
        print("  If the log shows Electron download retries, rebuild via a mirror:")
        print("    ELECTRON_MIRROR=<mirror-base-url> hermes desktop --force-build")
        sys.exit(build_result.returncode or 1)

    packaged_executable = None
    if staging_dir is not None:
        packaged_executable = _promote_staged_desktop_app(desktop_dir, staging_dir)

    # Build succeeded — write the stamp so next run can skip
    _write_desktop_build_stamp(PROJECT_ROOT, source_mode=source_mode)
    return packaged_executable


def _desktop_launch_env(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """Child env for the Electron process plus the config-supplied extra Electron flags.

    Config (``desktop.*``) is bridged to env vars the Electron/Chromium process
    already reads; an explicit env var still wins over config. Linux keychain
    backend for safeStorage: config wins over detection, env wins over both.
    """
    from hermes_cli.main import _desktop_launch_options, _detect_linux_password_store
    from hermes_constants import with_hermes_node_path

    # with_hermes_node_path() copies os.environ when called with no arg.
    env = with_hermes_node_path()
    if getattr(args, "fake_boot", False):
        env["HERMES_DESKTOP_BOOT_FAKE"] = "1"
    if getattr(args, "ignore_existing", False):
        env["HERMES_DESKTOP_IGNORE_EXISTING"] = "1"
    if getattr(args, "hermes_root", None):
        env["HERMES_DESKTOP_HERMES_ROOT"] = str(Path(args.hermes_root).expanduser().resolve())
    if getattr(args, "cwd", None):
        env["HERMES_DESKTOP_CWD"] = str(Path(args.cwd).expanduser().resolve())
    else:
        env["HERMES_DESKTOP_CWD"] = os.getcwd()

    config_electron_flags, config_disable_gpu, config_password_store, config_ozone_hint = (
        _desktop_launch_options()
    )
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu
    if config_ozone_hint != "auto" and "ELECTRON_OZONE_PLATFORM_HINT" not in os.environ:
        env["ELECTRON_OZONE_PLATFORM_HINT"] = config_ozone_hint

    # Without --password-store safeStorage.isEncryptionAvailable() is often
    # false and the desktop app refuses to persist remote gateway tokens.
    if sys.platform == "linux" and "HERMES_DESKTOP_PASSWORD_STORE" not in os.environ:
        password_store = (
            config_password_store if config_password_store != "auto" else _detect_linux_password_store()
        )
        if password_store:
            env["HERMES_DESKTOP_PASSWORD_STORE"] = password_store
    return env, config_electron_flags


def _check_desktop_skip_build(
    desktop_dir: Path, project_root: Path, *, source_mode: bool, packaged_executable: Optional[Path]
) -> None:
    """Validate the pre-built artifact ``--skip-build`` promised; exit with a hint when it's missing."""
    from hermes_cli.main import _desktop_dist_exists
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            print(f"✗ --skip-build --source was passed but no desktop dist found at: {desktop_dir / 'dist'}")
            print("  Pre-build first:  cd apps/desktop && npm run build")
            print("  Or drop --skip-build to install dependencies and build automatically.")
            sys.exit(1)
        if not (_electron_dir(project_root) / "package.json").exists():
            print("✗ --skip-build --source requires existing desktop workspace dependencies.")
            print(f"  Install first:  cd {project_root} && npm ci")
            print("  Or drop --skip-build to install dependencies and build automatically.")
            sys.exit(1)
        print(f"→ Skipping desktop source build (--skip-build --source); using dist at {desktop_dir / 'dist'}")
    elif packaged_executable is None:
        print(f"✗ --skip-build was passed but no packaged desktop app was found at: {desktop_dir / 'release'}")
        print("  Pre-build first:  cd apps/desktop && npm run pack")
        print("  Or drop --skip-build to package automatically.")
        sys.exit(1)
    else:
        print(f"→ Skipping desktop package build (--skip-build); using {packaged_executable}")


def _packaged_desktop_launch_command(packaged_executable: Path) -> list[str]:
    """``[exe, *sandbox flags]`` after the Linux sandbox fixup; exits when the sandbox can't be configured."""
    from hermes_cli.main import _desktop_linux_needs_disable_setuid_sandbox, _desktop_linux_sandbox_fixup
    launch_command = [str(packaged_executable)]
    if not _desktop_linux_sandbox_fixup(packaged_executable):
        if _desktop_linux_needs_no_sandbox() and _desktop_linux_sandbox_helper_is_regular_file(packaged_executable):
            print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces and the Electron sandbox helper could not be configured.")
            launch_command.append("--no-sandbox")
        else:
            sys.exit(1)
    elif _desktop_linux_needs_disable_setuid_sandbox(packaged_executable):
        launch_command.append("--disable-setuid-sandbox")
    return launch_command


def cmd_gui(args: argparse.Namespace):
    """Build and launch the native Electron desktop GUI."""
    from hermes_cli.main import PROJECT_ROOT, _desktop_build_needed, _desktop_dist_exists, _desktop_macos_setup_tcc_identity, _desktop_packaged_executable, _register_linux_desktop_entry, _resolve_node_runtime_npm
    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if not (desktop_dir / "package.json").exists():
        print(f"Desktop GUI source not found at: {desktop_dir}")
        sys.exit(1)

    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

    env, config_electron_flags = _desktop_launch_env(args)

    source_mode = getattr(args, "source", False)
    skip_build = getattr(args, "skip_build", False)
    force_build = getattr(args, "force_build", False)

    # macOS-only one-shot: create a self-signed code-signing identity so TCC
    # grants survive rebuilds, then exit without building/launching.
    if getattr(args, "setup_tcc_identity", False):
        identity = getattr(args, "identity", None) or "Hermes Local Signing"
        sys.exit(0 if _desktop_macos_setup_tcc_identity(identity) else 1)

    packaged_executable = _desktop_packaged_executable(desktop_dir)

    npm = None
    if source_mode or not skip_build:
        npm = _resolve_node_runtime_npm()
        if not npm:
            print("Desktop GUI requires Node.js/npm, but npm was not found on PATH.")
            print("Install Node.js, then run:  hermes gui")
            sys.exit(1)

    if skip_build:
        _check_desktop_skip_build(
            desktop_dir, PROJECT_ROOT, source_mode=source_mode, packaged_executable=packaged_executable
        )
    elif force_build or _desktop_build_needed(desktop_dir, PROJECT_ROOT, source_mode=source_mode):
        # --force-build overrides the content-hash stamp and always rebuilds.
        built = _build_desktop_app(desktop_dir, source_mode=source_mode, npm=npm, env=env)
        if not source_mode:
            packaged_executable = built
    else:
        build_label = "source build" if source_mode else "packaged app"
        print(f"✓ Desktop {build_label} is up to date (content stamp matches)")

    # Best-effort and idempotent; a failure must never stop the app from launching.
    _register_linux_desktop_entry()

    # --build-only: produce the artifact but do NOT launch. The installer's
    # --update flow drives the rebuild headlessly and launches the desktop
    # itself (detached, after the old exe has exited); launching here would
    # block the installer. Verify the artifact exists so a silent "built
    # nothing" can't slip past.
    if getattr(args, "build_only", False):
        if source_mode:
            if not _desktop_dist_exists(desktop_dir):
                print(f"✗ --build-only --source produced no dist at: {desktop_dir / 'dist'}")
                sys.exit(1)
            print(f"✓ Desktop source build ready at {desktop_dir / 'dist'} (not launching; --build-only)")
        elif packaged_executable is None:
            print(f"✗ --build-only produced no launchable app at: {desktop_dir / 'release'}")
            print("  Expected an unpacked Electron app for the current OS.")
            sys.exit(1)
        else:
            print(f"✓ Desktop packaged app ready: {packaged_executable} (not launching; --build-only)")
        return

    if source_mode:
        print("→ Launching Hermes Desktop from source build...")
        launch_command = [npm, "exec", "--", "electron", "."]
    else:
        if packaged_executable is None:
            print(f"✗ Desktop package build completed but no launchable app was found at: {desktop_dir / 'release'}")
            print("  Expected an unpacked Electron app for the current OS.")
            sys.exit(1)
        launch_command = _packaged_desktop_launch_command(packaged_executable)
        launch_command.extend(config_electron_flags)
    if getattr(args, "local", False):
        launch_command.append("--local")
    if not source_mode:
        print(f"→ Launching packaged Hermes Desktop: {' '.join(launch_command)}")
    launch_result = subprocess.run(launch_command, cwd=desktop_dir, env=env, check=False)
    sys.exit(launch_result.returncode)
