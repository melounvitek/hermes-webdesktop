"""Desktop (Electron) app: build/stamp, stage-and-swap pack, exe integrity gate, macOS signing/TCC, Linux sandbox, launch (hermes gui/desktop).

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import logging
import argparse
import hashlib
import json
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
from hermes_cli.main_web_build import _nixos_build_env

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")


def _desktop_dist_exists(desktop_dir: Path) -> bool:
    """Return True when a local desktop renderer build is present."""
    return (desktop_dir / "dist" / "index.html").exists()


def _compute_desktop_content_hash(project_root: Path) -> str:
    """Return a SHA-256 hex digest of all source files that feed the desktop build.

    Covers ``apps/desktop/`` (excluding anything matched by .gitignore)
    plus the root ``package.json`` / ``package-lock.json`` (workspace config
    that determines dependency resolution for the desktop workspace).

    Parses the repo-root ``.gitignore`` via *pathspec* so we automatically
    skip ``node_modules/``, ``dist/``, ``*.pyc``, etc. without maintaining
    a hardcoded skip-list.
    """
    h = hashlib.sha256()

    def _hash_file(path: Path) -> None:
        rel = str(path.relative_to(project_root))
        h.update(rel.encode())
        h.update(b"\0")
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except (OSError, IOError):
            pass
        h.update(b"\0")


    from pathspec import PathSpec

    gitignore = project_root / ".gitignore"
    lines: list[str] = []
    if gitignore.is_file():
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    spec = PathSpec.from_lines("gitignore", lines)

    # Root workspace config
    for name in ("package.json", "package-lock.json"):
        p = project_root / name
        if p.is_file():
            rel = str(p.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(p)

    # Walk apps/desktop/ — prune ignored directories in-place
    desktop_dir = project_root / "apps" / "desktop"
    for dirpath, dirnames, filenames in os.walk(desktop_dir, topdown=True):
        # Prune ignored directories so we never descend into them
        dirnames[:] = [
            d for d in dirnames
            if not spec.match_file(str((Path(dirpath) / d).relative_to(project_root)))
        ]

        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            rel = str(fp.relative_to(project_root))
            if not spec.match_file(rel):
                _hash_file(fp)

    return h.hexdigest()


def _desktop_stamp_path() -> Path:
    """Return the path to the desktop build stamp file under $HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "desktop-build-stamp.json"


def _renderer_bundle_dir(desktop_dir: Path, *, source_mode: bool) -> Optional[Path]:
    """The renderer ``dist`` directory a launch loads, when it is inspectable.

    Source mode builds to ``apps/desktop/dist``. A packaged app ships the same
    bundle twice — inside ``app.asar`` and, because ``asarUnpack`` lists
    ``dist/**``, beside it in ``app.asar.unpacked``. Only the unpacked copy is
    a real directory; that is also the one an interrupted replace tears, so
    checking it catches the failure we care about.
    """
    from hermes_cli.main import _desktop_packaged_executable
    if source_mode:
        return desktop_dir / "dist"

    executable = _desktop_packaged_executable(desktop_dir)
    if executable is None:
        return None

    # macOS: …/Hermes.app/Contents/MacOS/Hermes → …/Contents/Resources
    resources = (
        executable.parent.parent / "Resources"
        if sys.platform == "darwin"
        else executable.parent / "resources"
    )
    return resources / "app.asar.unpacked" / "dist"


# The module files the renderer fetches before any app code runs: Vite emits
# them as `<script type="module" src>` plus `<link rel="modulepreload" href>`.
_HTML_TAG_WITH_URL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["'][^>]*>""", re.IGNORECASE)


_MODULE_TAG = re.compile(r"""\btype=["']module["']|\brel=["']modulepreload["']""", re.IGNORECASE)


def _renderer_bundle_torn(dist_dir: Path) -> bool:
    """True when ``index.html`` names hashed module files that aren't there.

    ``index.html`` and the hashed chunks under ``assets/`` are ONE generation.
    An update that replaces the app while its files are locked (antivirus, a
    still-running instance, an interrupted Windows replace) can leave the two
    behind from different generations. The app then launches and dies on the
    first lazy import with ``Failed to fetch dynamically imported module:
    …/assets/<chunk>-<hash>.js`` — and because the content stamp still matches
    the intact SOURCE tree, ``hermes desktop`` skips the rebuild that would fix
    it, so every relaunch reproduces the crash and reinstalling looks like the
    only way out. Detecting the tear turns it into a normal rebuild.

    Conservative: an unreadable index, or one naming nothing checkable, is NOT
    reported as torn — the missing-bundle guards own those cases.
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
    """Return True when the desktop build output is stale, missing, or torn.

    Compares the current content hash against the saved stamp. Also returns
    True if the expected build artifact doesn't exist (e.g. first run after
    ``hermes update`` that pulled new source but hasn't built yet).
    """
    from hermes_cli.main import _desktop_dist_exists, _desktop_packaged_executable, _desktop_stamp_path
    # If there's no build output at all, we definitely need to build
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            return True
    else:
        if _desktop_packaged_executable(desktop_dir) is None:
            return True

    # A torn renderer bundle is stale no matter what the stamp says: the hash
    # describes the SOURCE tree, which is intact, while the built output is the
    # half-replaced one that crashes on its first lazy import.
    dist_dir = _renderer_bundle_dir(desktop_dir, source_mode=source_mode)
    if dist_dir is not None and _renderer_bundle_torn(dist_dir):
        print(f"  ⚠ A previous update left the desktop bundle incomplete ({dist_dir}); rebuilding it")
        return True

    stamp_file = _desktop_stamp_path()
    if not stamp_file.is_file():
        return True

    try:
        stamp_data = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return True

    # If the mode changed (source vs packaged), force a rebuild
    if stamp_data.get("sourceMode") != source_mode:
        return True

    saved_hash = stamp_data.get("contentHash")
    if not saved_hash:
        return True

    current_hash = _compute_desktop_content_hash(project_root)
    return current_hash != saved_hash


def _write_desktop_build_stamp(project_root: Path, *, source_mode: bool) -> None:
    """Write the desktop build stamp after a successful build."""
    from hermes_cli.main import _desktop_stamp_path
    stamp_file = _desktop_stamp_path()
    try:
        stamp_file.parent.mkdir(parents=True, exist_ok=True)
        content_hash = _compute_desktop_content_hash(project_root)
        from datetime import datetime, timezone
        stamp_data = {
            "contentHash": content_hash,
            "sourceMode": source_mode,
            "builtAt": datetime.now(timezone.utc).isoformat(),
        }
        stamp_file.write_text(json.dumps(stamp_data, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        # Never let stamp-writing block or fail a build
        logger.debug("Failed to write desktop build stamp: %s", exc)


def _desktop_packaged_executable(desktop_dir: Path) -> Optional[Path]:
    """Return the current platform's unpacked Electron app executable."""
    return _desktop_packaged_executable_in(desktop_dir / "release")


def _desktop_packaged_executable_in(release_dir: Path) -> Optional[Path]:
    """Return the unpacked Electron app executable under *release_dir*.

    *release_dir* is electron-builder's ``directories.output`` — the live
    ``apps/desktop/release`` or a stage-and-swap staging dir (#86443).
    """
    from hermes_cli.main import _expected_windows_pe_machines
    if sys.platform == "darwin":
        candidates = list(release_dir.glob("mac*/Hermes.app/Contents/MacOS/Hermes"))
    elif sys.platform == "win32":
        candidates = [
            release_dir / "win-unpacked" / "Hermes.exe",
            release_dir / "win-ia32-unpacked" / "Hermes.exe",
            release_dir / "win-arm64-unpacked" / "Hermes.exe",
        ]
    else:
        candidates = [
            release_dir / "linux-unpacked" / "hermes",
            release_dir / "linux-unpacked" / "Hermes",
            release_dir / "linux-arm64-unpacked" / "hermes",
            release_dir / "linux-arm64-unpacked" / "Hermes",
        ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    if sys.platform == "win32" and len(existing) > 1:
        # Multiple unpacked trees can coexist (e.g. a stale win-arm64-unpacked
        # left behind by a cross-arch experiment next to the real win-unpacked).
        # Picking purely by mtime can then hand a wrong-architecture Hermes.exe
        # to the launcher, which Windows rejects with "This app can't run on
        # your computer" (#69179). Prefer candidates whose PE machine field
        # matches the host; fall back to mtime when none can be parsed.
        expected = _expected_windows_pe_machines()
        matching = [p for p in existing if _pe_machine_or_none(p) in expected]
        if matching:
            existing = matching
    return max(existing, key=lambda p: p.stat().st_mtime)


_DESKTOP_STAGING_PREFIX = ".staging-"


_DESKTOP_PREVIOUS_SUFFIX = ".previous"


def _desktop_staging_dir(desktop_dir: Path) -> Path:
    """Fresh, unique staging output dir: ``apps/desktop/.staging-<pid>-<ts>``.

    A sibling of ``release/`` (same filesystem → the swap is a rename, not a
    copy) but NOT inside it, so nothing globbing ``release/*-unpacked`` or
    ``release/mac*`` can mistake the half-built tree for the live app.
    Leftovers from a killed earlier build are swept first (best-effort).
    """
    for stale in desktop_dir.glob(f"{_DESKTOP_STAGING_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)
    return desktop_dir / f"{_DESKTOP_STAGING_PREFIX}{os.getpid()}-{int(_time_mod.time())}"


def _desktop_unpacked_root(exe: Path, release_dir: Path) -> Path:
    """The directory directly under *release_dir* that holds *exe*
    (``linux-unpacked``, ``win-unpacked``, ``mac-arm64``…) — electron-builder's
    ``appOutDir``, the unit that gets swapped as a whole."""
    unpacked = exe
    while unpacked.parent != release_dir:
        if unpacked.parent == unpacked:
            raise ValueError(f"{exe} is not under {release_dir}")
        unpacked = unpacked.parent
    return unpacked


def _swap_staged_desktop_app(desktop_dir: Path, staging_dir: Path) -> Optional[Path]:
    """Promote a VERIFIED staged pack over the live ``release/`` app.

    ``release/<unpacked>`` → ``release/<unpacked>.previous``,
    ``<staging>/<unpacked>`` → ``release/<unpacked>``, then drop ``.previous``.
    Two renames; the only window with no live app is between them, and a
    failure there rolls ``.previous`` back. Returns the live executable, or
    ``None`` (live app untouched or restored) when the swap could not happen.
    Best-effort cleanup of the staging dir; never raises.
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
        moved_aside = False
        if live_root.exists():
            os.rename(live_root, previous)
            moved_aside = True
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
    _PE_MACHINE_I386: "x86 (32-bit)",
    _PE_MACHINE_AMD64: "x64 (AMD64)",
    _PE_MACHINE_ARM64: "ARM64",
}


_PE_MACHINE_TO_NAME = {
    _PE_MACHINE_ARM64: "ARM64",
    _PE_MACHINE_AMD64: "AMD64",
    _PE_MACHINE_I386: "X86",
}


# MACHINE_ATTRIBUTES bits (processthreadsapi.h). UserEnabled means the host
# can run user-mode code of that machine type — natively or under emulation.
_MACHINE_ATTRIBUTE_USER_ENABLED = 0x00000001


def _windows_native_machine_from_iswow64() -> Optional[str]:
    """Ask IsWow64Process2 for the OS-native machine (None if unavailable/fail).

    ctypes defaults ``GetCurrentProcess``'s restype to ``c_int``, so the
    current-process pseudo-handle ``(HANDLE)-1`` is truncated to
    ``0xFFFFFFFF`` and zero-extended into a 64-bit invalid handle. On Win64
    that makes ``IsWow64Process2`` fail with ``ERROR_INVALID_HANDLE`` (6),
    which is exactly the residual Windows-on-ARM failure after #71218: the
    gate fell through to ``PROCESSOR_ARCHITECTURE=AMD64`` (the emulated
    process arch) and rejected a correctly-built ARM64 ``Hermes.exe``.
    Binding ``restype``/``argtypes`` to ``wintypes.HANDLE`` keeps the full
    ``0xFFFFFFFFFFFFFFFF`` pseudo-handle.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.IsWow64Process2.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    ]
    kernel32.IsWow64Process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not kernel32.IsWow64Process2(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_machine),
        ctypes.byref(native_machine),
    ):
        return None
    return _PE_MACHINE_TO_NAME.get(native_machine.value)


def _windows_user_runnable_pe_machines() -> Optional[set]:
    """PE machines this host can run in user mode, via GetMachineTypeAttributes.

    This asks the question the integrity gate actually cares about — "can this
    Windows host load a PE of machine X?" — instead of inferring it from a
    host-architecture name. It is also the only documented API that reports
    AMD64-on-ARM64 emulation support; ``IsWow64GuestMachineSupported`` only
    answers for 32-bit guests.

    Returns None when the API is unavailable (pre-Windows-11 build 22000) or
    reports nothing runnable, so callers fall back to name-based detection.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetMachineTypeAttributes.argtypes = [
        wintypes.USHORT,
        ctypes.POINTER(ctypes.c_int),
    ]
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
    emulation: the desktop update chain runs an x64 hermes-setup.exe (and thus
    x64 Python) on Windows-on-ARM devices, where ``platform.machine()``
    returns ``AMD64`` even though the OS is ARM64. The #71119 integrity gate
    then rejected the CORRECT ARM64 rebuild as an "architecture mismatch"
    (#69179 follow-up report). Probe order:

    1. ``IsWow64Process2`` with a correctly-typed current-process HANDLE
       (#71218 + HANDLE-truncation fix). This is the only API that tells the
       truth from an x64 process emulated on ARM64.
    2. ``PROCESSOR_ARCHITEW6432`` / ``PROCESSOR_ARCHITECTURE`` — WOW64
       (32-bit) hosts and pre-1511 Windows 10 without the newer API.
    3. ``platform.machine()``.

    Note ``GetNativeSystemInfo`` is deliberately NOT used: Microsoft documents
    that it "also returns emulated processor details when run from an app
    under emulation", so on the very WoA hosts this function exists to serve
    it reports AMD64 — no better than the env-var rung below it.
    """
    if sys.platform == "win32":
        try:
            name = _windows_native_machine_from_iswow64()
        except (OSError, AttributeError, TypeError, ValueError):
            # API missing (pre-1511), DLL load failure in tests, or a
            # mistyped ctypes binding — fall through to the env vars.
            name = None
        if name:
            return name
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get(
            "PROCESSOR_ARCHITECTURE"
        )
        if env_arch:
            return env_arch.upper()
    import platform as _platform

    return (_platform.machine() or "").upper()


def _expected_windows_pe_machines() -> set:
    """PE machine values the current Windows host can natively load.

    Preferred source is ``GetMachineTypeAttributes``, which answers this
    question directly (including AMD64-on-ARM64 emulation) instead of
    inferring it from an architecture name.

    Fallback is name-based: AMD64 hosts run x64 and (via WOW64) x86. ARM64
    hosts run ARM64 and (Windows 11 emulation) x64. 32-bit x86 hosts run only
    x86. Unknown machines return the permissive full set so the integrity gate
    can never brick launch on exotic hosts. Host detection uses the OS-native
    machine (see ``_windows_native_machine``), not the process architecture.
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
    structurally complete PE: missing MZ/PE magic (an HTML error page or JSON
    body saved as .exe), header truncation, or raw section data extending past
    the end of the file (the truncated-download / interrupted-extraction
    shape). Purely a header walk — cheap even on a 200 MB Electron exe.
    """
    import struct

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unreadable: {exc}")
    if file_size < 512:
        raise ValueError(
            f"file is only {file_size} bytes — far too small to be a Windows executable"
        )
    with path.open("rb") as fh:
        head = fh.read(64)
        if len(head) < 64 or head[:2] != b"MZ":
            raise ValueError(
                "missing MZ header — not a Windows executable "
                "(a truncated or non-binary file saved as .exe?)"
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
                f"truncated executable: file is {file_size} bytes but its PE "
                f"sections extend to {max_section_end} bytes"
            )
    return machine


def _pe_machine_or_none(path: Path) -> Optional[int]:
    from hermes_cli.main import _parse_pe_machine
    try:
        return _parse_pe_machine(path)
    except ValueError:
        return None


def _desktop_exe_integrity_error(path: Path) -> Optional[str]:
    """Return a human-readable reason ``path`` cannot run on this Windows host,
    or ``None`` when the exe parses as a complete PE of a loadable architecture.
    """
    from hermes_cli.main import _expected_windows_pe_machines, _parse_pe_machine, _windows_native_machine
    try:
        machine = _parse_pe_machine(path)
    except ValueError as exc:
        return str(exc)
    expected = _expected_windows_pe_machines()
    if machine not in expected:
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

    Returns the restored executable path, or ``None`` when no usable backup
    exists (missing, or its exe fails the same integrity probe). The corrupt
    tree is kept alongside as ``<unpacked-dir>.corrupt`` for diagnostics.
    Best-effort: never raises.
    """
    unpacked = packaged_executable.parent
    backup_dir = _desktop_backup_unpacked_dir(packaged_executable)
    backup_exe = backup_dir / packaged_executable.name
    if not backup_exe.exists():
        return None
    if _desktop_exe_integrity_error(backup_exe) is not None:
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


def _ensure_desktop_exe_launchable(
    desktop_dir: Path, packaged_executable: Optional[Path]
) -> tuple:
    """Windows post-build integrity gate for the self-update rebuild (#69179).

    Returns ``(verified_exe_or_None, rolled_back)``:

    - exe passed the probe → ``(exe, False)``
    - exe corrupt/wrong-arch, previous build restored → ``(old_exe, True)``
    - exe corrupt and nothing restorable → ``(None, False)``

    On any integrity failure the corrupt cached Electron zip is purged and the
    desktop build stamp invalidated, so the updater's retry-once rebuild pulls
    a fresh, SHASUM-verified Electron download instead of re-staging the same
    corrupt bytes. No-op off Windows and when there is no executable to check.
    """
    from hermes_cli.main import _desktop_stamp_path, _purge_electron_build_cache, _rollback_desktop_from_backup
    if packaged_executable is None or sys.platform != "win32":
        return packaged_executable, False

    error = _desktop_exe_integrity_error(packaged_executable)
    if error is None:
        return packaged_executable, False

    print(f"✗ The built Hermes.exe failed its integrity check: {error}")
    print(f"    at: {packaged_executable}")

    # Self-heal setup for the retry: drop the (likely corrupt) cached Electron
    # zip and the content stamp so the next rebuild is a genuine re-download +
    # re-stage rather than a replay of the same broken extraction. Only the
    # exe's OWN output dir is purged (a stage-and-swap staging dir, #86443),
    # never the live release/ tree that still holds the last working app.
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
    """Return the per-user Electron download cache directories for this OS.

    electron-builder's ``app-builder unpack-electron`` extracts the Electron
    distribution from a zip stored in this cache (NOT from node_modules), so a
    corrupt zip here — not a bad workspace install — is what poisons the build.
    Honors the ``electron_config_cache`` / ``ELECTRON_CACHE`` overrides that
    ``@electron/get`` respects, then falls back to the platform defaults.
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

    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        rc = c.expanduser()
        if rc not in seen:
            seen.add(rc)
            out.append(rc)
    return out


def _purge_electron_build_cache(
    desktop_dir: Path, release_dir: Optional[Path] = None
) -> list[Path]:
    """Clear the cached Electron download + half-written unpacked dir so the
    next ``pack`` re-downloads and re-stages from scratch.

    Root cause of the ``ENOENT … rename '…/linux-unpacked/electron' ->
    '…/linux-unpacked/Hermes'`` desktop build failure: a corrupt zip in the
    per-user Electron download cache (a partial download resumed into the same
    file leaves prepended/concatenated junk, or an interrupted write truncates
    it). electron-builder's ``app-builder unpack-electron`` extracts the
    distribution from that cached zip (NOT from node_modules); a bad zip yields
    a partial tree MISSING the 193 MB ``electron`` binary, so the final rename
    dies. Re-running repeats the same broken extraction forever.

    We deliberately do NOT try to detect corruption ourselves. stdlib
    ``zipfile`` silently tolerates the prepended/concatenated junk that is the
    most common corruption here — it reads from the end-of-central-directory
    backward, so ``testzip()`` returns clean on exactly the zips ``unzip -t``
    and ``@electron/get`` reject. Gating the purge on a self-rolled validator
    would therefore skip the real-world case and never self-heal. Instead, on a
    packaged-build failure we unconditionally remove the version's cached zips
    and the stale unpacked dir, then let the caller retry once: ``@electron/get``
    re-downloads with its own SHASUM verification (the real source of truth),
    and ``before-pack.cjs`` re-wipes the unpacked dir. If the failure was
    unrelated, a clean re-download is harmless and the retry fails the same way.

    Best-effort: never raises. Returns the paths removed so the caller can log
    them and decide whether a retry is worthwhile (empty list ⇒ nothing to
    clear, so no point retrying).
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
                # Locked/permission-denied entry is out of our hands; let the
                # build report its own error rather than masking it.
                pass

    # Drop the half-written unpacked dir too: an interrupted prior pack leaves
    # a partial tree that poisons the rename even after the zip is fixed.
    # (before-pack.cjs also handles this, but clearing it here makes the retry
    # robust even if the hook is somehow skipped.) ``release_dir`` lets a
    # stage-and-swap caller point this at its STAGING output so a mid-retry
    # purge never touches the live app under ``release/`` (#86443).
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


# Last-resort Electron mirror after GitHub download fails (#47266). Only used
# when the user hasn't pinned ELECTRON_MIRROR.
_ELECTRON_FALLBACK_MIRROR = "https://npmmirror.com/mirrors/electron/"


def _electron_dir(project_root: Path) -> Path:
    """Return the Electron package directory the desktop workspace installs.

    npm may keep workspace-only dev dependencies under
    ``apps/desktop/node_modules`` instead of hoisting them to the repo root.
    Which layout you get depends on the npm version and what else is installed,
    so a build path that assumes one or the other breaks intermittently across
    machines. ``apps/desktop/package.json`` points electron-builder's
    ``electronDist`` at ``node_modules/electron/dist`` relative to the desktop
    project, so prefer the workspace-local package and fall back to the root
    hoist when that's where npm landed it.
    """
    desktop_local = project_root / "apps" / "desktop" / "node_modules" / "electron"
    if desktop_local.exists():
        return desktop_local
    return project_root / "node_modules" / "electron"


def _electron_dist_binary(project_root: Path) -> Path:
    """Return the path to the Electron main binary inside the installed package.

    electron-builder reads the binary from ``build.electronDist`` since #38673,
    so this is the exact file whose absence makes a pack fail with "The
    specified electronDist does not exist". The basename differs per OS (the
    platform Electron is named for the host the build runs on).
    """
    dist = _electron_dir(project_root) / "dist"
    if sys.platform == "darwin":
        return dist / "Electron.app" / "Contents" / "MacOS" / "Electron"
    if sys.platform == "win32":
        return dist / "electron.exe"
    return dist / "electron"


def _electron_dist_ok(project_root: Path) -> bool:
    """True when ``node_modules/electron/dist`` holds a usable Electron binary.

    A directory that exists but is missing the binary (a partial extraction from
    a corrupt cached zip, or an interrupted postinstall) counts as NOT ok, since
    that is exactly the shape that makes electron-builder throw on the pinned
    electronDist.
    """
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


def _redownload_electron_dist(
    project_root: Path,
    env: dict,
    *,
    mirror: Optional[str] = None,
) -> bool:
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

    dist_dir = electron_dir / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
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
    """Terminate any running desktop app executing from this build's ``release``
    dir so a rebuild can replace its (otherwise locked) executable.

    On Windows a running ``Hermes.exe`` keeps an exclusive lock on
    ``release/win-unpacked/Hermes.exe``. electron-builder's pack then can't
    delete the stale binary and dies with ``remove …\\Hermes.exe: Access is
    denied`` / ``ERR_ELECTRON_BUILDER_CANNOT_EXECUTE`` (before-pack hits the same
    EPERM cleaning the dir). The retry path repeats the failure because the lock
    is still held. POSIX lets you unlink a running binary, so this is a no-op
    off-Windows.

    Scope is deliberately narrow: only processes whose executable lives *inside*
    this desktop's ``release`` tree are stopped — a packaged install elsewhere or
    an unrelated "Hermes" process is never touched. Best-effort: never raises.
    Returns the PIDs we asked to stop.
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
    except Exception:
        return []
    try:
        release_dir = (desktop_dir / "release").resolve()
    except OSError:
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
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or pid == me:
            continue
        try:
            exe_path = Path(exe).resolve()
        except (OSError, ValueError):
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
    """Return the opt-in keychain identity for local macOS desktop signing.

    ``desktop.macos_signing_identity`` in config.yaml names a persistent
    code-signing certificate in the user's login keychain (a self-signed
    "Code Signing" cert made in Keychain Access is enough — no Apple Developer
    account needed). Signing with any identity gives the app a
    certificate-anchored Designated Requirement, which is the strongest way to
    keep macOS TCC grants (Full Disk Access, Accessibility, Automation, Files
    and Folders) stable across local rebuilds. Empty/unset keeps the default
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


def _desktop_macos_has_valid_real_signature(app: Path) -> bool:
    """True when the bundle carries an intact non-ad-hoc (Team ID) signature.

    Used to make the relaunch fixup a no-op on properly signed/notarized
    builds even when CSC_LINK / APPLE_SIGNING_IDENTITY aren't in the
    environment (e.g. a release DMG install being repaired) — clobbering a
    Developer ID signature with an ad-hoc one would reset TCC grants and can
    break the hardened runtime. A *stale* real signature (in-place rebuild
    tampered with the bundle) fails --verify and returns False so the fixup
    can repair it.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    try:
        info = subprocess.run(
            [codesign, "-dv", str(app)], check=False, capture_output=True, text=True
        )
        output = f"{info.stdout}\n{info.stderr}"
        if info.returncode != 0 or "TeamIdentifier=" not in output \
                or "TeamIdentifier=not set" in output:
            return False
        verify = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(app)],
            check=False, capture_output=True,
        )
        return verify.returncode == 0
    except Exception:
        return False


def _desktop_macos_local_codesign(
    app: Path, *, desktop_dir: Path, identity: str = "-"
) -> bool:
    """Re-sign a local Desktop build so macOS TCC grants survive rebuilds.

    A plain ``codesign --deep --sign -`` leaves the bundle with a cdhash-only
    Designated Requirement and strips electron-builder's entitlements. Every
    rebuild changes the cdhash, so TCC (Full Disk Access, Accessibility,
    Automation, Files and Folders: Desktop/Downloads/Documents, microphone)
    treats the rebuilt app as different code and the user must re-grant
    everything — and the lost entitlements break microphone/JIT under the
    hardened runtime.

    Instead, sign inside-out (standalone Mach-O binaries, then nested
    frameworks/helper apps, then the main bundle), preserving the repo's
    entitlement plists, and pin an explicit identifier-based Designated
    Requirement when signing ad-hoc. With a real ``identity`` the certificate
    anchors the DR, so no explicit requirement is needed. Raises on signing
    failure; returns True after strict verification passes.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        return False

    ent_main = desktop_dir / "electron" / "entitlements.mac.plist"
    ent_inherit = desktop_dir / "electron" / "entitlements.mac.inherit.plist"
    if not (ent_main.exists() and ent_inherit.exists()):
        # Hardened-runtime restrictions are enforced even for ad-hoc
        # signatures. Signing with --options runtime but WITHOUT the allow-jit
        # entitlements would leave Electron/V8 crashing on launch — strictly
        # worse than the legacy plain ad-hoc sign. Bail out so the caller
        # falls back to that legacy path instead.
        raise FileNotFoundError(
            f"desktop entitlement plists missing under {desktop_dir / 'electron'}"
        )

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

    # 1) Standalone Mach-O files (native modules, dylibs, crashpad handler).
    #    Compare paths relative to the app root — the absolute path always
    #    contains the outer Hermes.app component, so an absolute-parts check
    #    would skip every file.
    contents = app / "Contents"
    standalone: list[Path] = []
    for root, _dirs, files in os.walk(contents):
        root_path = Path(root)
        rel_parts = root_path.relative_to(app).parts
        if any(part.endswith(".app") for part in rel_parts):
            continue  # nested helper apps are signed as bundles below
        for name in files:
            fp = root_path / name
            if name in {"chrome_crashpad_handler", "spawn-helper"} or fp.suffix in {
                ".node",
                ".dylib",
            }:
                standalone.append(fp)
    for fp in sorted(standalone, key=lambda p: len(p.parts), reverse=True):
        sign_path(fp, runtime=False)

    # 2) Nested frameworks and helper apps, deepest first.
    bundles: list[Path] = []
    frameworks_dir = contents / "Frameworks"
    if frameworks_dir.exists():
        for root, _dirs, _files in os.walk(frameworks_dir):
            p = Path(root)
            if p.suffix in {".framework", ".app"}:
                bundles.append(p)
    for bundle in sorted(set(bundles), key=lambda p: len(p.parts), reverse=True):
        ent = ent_inherit if bundle.suffix == ".app" and "Helper" in bundle.name else None
        sign_path(bundle, entitlements=ent, identifier=_desktop_macos_bundle_id(bundle))

    # 3) The main bundle, with the app's own entitlements.
    sign_path(app, entitlements=ent_main, identifier=_desktop_macos_bundle_id(app))
    subprocess.run(
        [codesign, "--verify", "--deep", "--strict", str(app)],
        check=True, capture_output=True,
    )
    return True


def _desktop_macos_relaunchable_fixup(
    desktop_dir: Path,
    *,
    publisher_signing_configured: Optional[bool] = None,
    release_dir: Optional[Path] = None,
) -> bool:
    """Make a locally-built macOS desktop app survive in-place self-update
    without resetting the user's TCC permission grants.

    An ad-hoc-signed .app has no stable Designated Requirement, so when the
    self-updater rebuilds the bundle in place (new cdhash) Gatekeeper reports
    "Hermes is damaged and can't be opened" — and macOS TCC forgets every
    permission the user granted (Full Disk Access, Desktop/Downloads/Documents,
    Accessibility, Automation, microphone), re-prompting on every launch after
    every update.

    Clear the quarantine xattrs, then re-sign with a stable identity:
    ``desktop.macos_signing_identity`` (a persistent keychain cert — strongest)
    when configured, else ad-hoc with identifier-pinned Designated Requirements,
    preserving the repo's entitlement plists either way. No-op when a real
    publisher identity is configured (CSC_LINK / APPLE_SIGNING_IDENTITY) or the
    bundle already carries an intact Developer ID signature, so a properly
    signed/notarized build is never clobbered. Callers that already made the
    publisher-signing decision may pass it explicitly so a later dotenv load
    can't reverse it. Falls back to the legacy deep ad-hoc sign if the
    entitlement-preserving path fails. Best-effort: never raises. Returns True
    when no work was needed or signing + strict verification succeeded.
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
    # ``release_dir`` (stage-and-swap, #86443): sign the STAGED bundle before
    # it is promoted, so the live app is never touched mid-sign.
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
    try:
        # Legacy ad-hoc fallback: re-sign, but NEVER delete the safeStorage
        # keychain item. Deleting it would permanently orphan every
        # credential encrypted under it (gateway token, native OAuth access/
        # refresh tokens) — and this path is reached exactly when the
        # entitlement-preserving signer failed, so there is no verified
        # successor identity to hand the key to. The keychain prompt macOS
        # shows instead is recoverable ("Always Allow" updates the item's ACL
        # partition list and preserves the key); deletion is not. The real
        # fix (proof-carrying rotation/migration) belongs in Electron, where
        # safeStorage can read the old key. Tracked as follow-up.
        result = subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(app)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign failed (exit {result.returncode}); "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        verify = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(app)],
            check=False, capture_output=True, text=True,
        )
        if verify.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign did not pass strict verification; "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        print("  → macOS desktop re-signed (legacy ad-hoc); safeStorage keychain item left untouched")
        return True
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")
    return False


def _macos_codesigning_identity_valid(security: str, identity: str) -> bool:
    """True when `identity` appears among VALID code-signing identities.

    ``security find-identity -p codesigning`` (without ``-v``) also lists
    certificates macOS will refuse to sign with — e.g. a self-signed cert that
    was imported but never trusted for the codeSign policy. Only the ``-v``
    listing proves codesign can actually use it, so this is both the
    idempotency probe and the success postcondition for
    ``--setup-tcc-identity``. Never raises.
    """
    try:
        result = subprocess.run(
            [security, "find-identity", "-v", "-p", "codesigning"],
            capture_output=True, text=True, check=False,
        )
    except Exception:
        return False

    return f'"{identity}"' in (result.stdout or "")


def _desktop_macos_setup_tcc_identity(identity: str = "Hermes Local Signing") -> bool:
    """Create/import a self-signed code-signing cert and configure Hermes to use it.

    One-shot setup for ``hermes desktop --setup-tcc-identity``. Creates a
    self-signed "Code Signing" certificate in the login keychain (the same
    artifact the docs describe creating manually via Keychain Access), grants
    ``codesign`` access to it, writes ``desktop.macos_signing_identity`` to
    config.yaml, and re-signs the already-packaged app so the next launch uses
    the certificate-anchored identity.

    Why this matters: macOS TCC grants (Full Disk Access, Accessibility,
    Automation, Files and Folders, microphone) persist against the app's
    code-signing identity, not its path. A plain ad-hoc signature gets a
    cdhash-only Designated Requirement, so every rebuild looks like a new app
    and the user must re-grant everything. A certificate-anchored identity is
    stable across rebuilds — the same mechanism yabai/skhd users rely on.

    Idempotent: re-running after an update finds the existing certificate and
    only re-points the config + re-signs. Returns True on success (or when
    already configured), False on failure. Never raises.
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
    # A certificate that merely EXISTS in the keychain is not enough — macOS
    # only treats it as a code-signing identity once it is trusted for the
    # codeSign policy. Probe with `-v` (valid identities only) so a previously
    # imported-but-untrusted cert is repaired rather than reported as done.
    already_imported = _macos_codesigning_identity_valid(security, identity)

    if not already_imported:
        # Create a self-signed code-signing cert (valid 10 years) and import it
        # into the login keychain with codesign access so signing works without
        # an interactive unlock prompt.
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
            # OpenSSL 3 defaults to AES/SHA-2 PKCS#12 encryption that macOS
            # `security import` rejects with "MAC verification failed during
            # PKCS12 import (wrong password?)". The `-legacy` flag restores the
            # RC2/SHA-1 format the importer accepts, but only exists on
            # OpenSSL 3 — so try the plain export first and fall back to
            # `-legacy` when the IMPORT fails with that signature. (Verified
            # E2E on macOS 26.3.1 / OpenSSL 3.6.3 by @ctaylor86 on PR #77189.)
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
                    # Older OpenSSL without -legacy: keep the original failure.
                    pass
            if imported.returncode != 0:
                print(f"  (could not import signing identity into keychain: {imported.stderr.strip()})")
                return False

            # Importing is still not enough: without explicit trust for the
            # codeSign policy, `security find-identity -v -p codesigning`
            # reports 0 valid identities and codesign refuses the cert. Trust
            # the self-signed root for code signing. This writes to the user's
            # trust settings, so macOS may prompt for the login password ONCE
            # here — that is the one-time setup cost this command exists to
            # front-load.
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
        except Exception as exc:
            print(f"  (certificate creation failed: {exc})")
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"  → identity {identity!r} already valid in keychain")

    # Postcondition gate: only report success once macOS actually agrees the
    # identity is usable for code signing. Name-in-output checks pass for
    # invalid identities; this is the check that failed silently before.
    if not _macos_codesigning_identity_valid(security, identity):
        print(
            f"  (identity {identity!r} was imported but is not a VALID code-signing identity; "
            "run `security find-identity -v -p codesigning` to inspect, and see the manual "
            "Keychain Access steps in the desktop docs)"
        )
        return False

    # Point Hermes at the identity (config.yaml, not .env — it's not a secret).
    try:
        from hermes_cli.config import set_config_value

        set_config_value("desktop.macos_signing_identity", identity)
        print(f"  → set desktop.macos_signing_identity = {identity!r}")
    except Exception as exc:
        print(f"  (could not write desktop.macos_signing_identity: {exc})")
        return False

    # Re-sign the packaged app so the current build already uses the identity.
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

    The desktop self-updater rebuilds *and re-signs the .app on the end user's
    machine* (``hermes desktop --build-only`` → electron-builder ``--dir``).
    With ``CSC_IDENTITY_AUTO_DISCOVERY`` on (its default), electron-builder
    signs the ``type=distribution``, hardened-runtime bundle with whatever it
    finds in that user's keychain — typically a personal "Apple Development"
    cert. That stalls/fails the sign step (no Developer ID + no provisioning
    profile) or clobbers your real notarized signature with an unusable one, so
    every post-update launch trips Gatekeeper.

    Force ad-hoc signing for the local packaged rebuild instead: deterministic,
    and exactly what ``_desktop_macos_relaunchable_fixup`` already finishes off.
    No-op for source runs, off-macOS, when a real identity is configured
    (``CSC_LINK`` / ``APPLE_SIGNING_IDENTITY``), or when the caller already
    pinned the flag. Mutates ``env``; returns True when it set the flag.
    """
    if sys.platform != "darwin" or source_mode:
        return False
    if env.get("CSC_LINK") or env.get("APPLE_SIGNING_IDENTITY"):
        return False
    if "CSC_IDENTITY_AUTO_DISCOVERY" in env:
        return False
    env["CSC_IDENTITY_AUTO_DISCOVERY"] = "false"
    return True


def _desktop_linux_needs_no_sandbox() -> bool:
    """Return True when Chromium/Electron should bypass the Linux sandbox.

    Ubuntu 23.10+ can enable AppArmor's
    ``apparmor_restrict_unprivileged_userns`` hardening, which breaks
    Chromium/Electron's user-namespace sandbox for normal users unless the app
    ships a working root-owned 4755 ``chrome-sandbox`` helper. In headless or
    non-interactive CLI contexts we may be unable to ``sudo chown/chmod`` that
    helper, so detect the host restriction and fall back to ``--no-sandbox``
    rather than hard-failing the launcher.

    We intentionally do NOT return True for root users here: running Electron as
    root without a sandbox is a qualitatively riskier path than launching as an
    unprivileged desktop user on an AppArmor-restricted host. The root case
    should remain an explicit user choice.
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
    """Return True when Chromium's unprivileged user-namespace sandbox works.

    When an unprivileged process can create a user namespace, Chromium uses the
    namespace sandbox and never consults the setuid ``chrome-sandbox`` helper,
    so requiring the helper to be root-owned 4755 (and prompting for sudo) is
    unnecessary. Probe the real capability with ``unshare`` instead of reading
    distro-specific sysctls: the probe fails closed on hosts where user
    namespaces are disabled or AppArmor-restricted, which then follow the
    existing setuid-helper path.
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _desktop_linux_sandbox_helper_is_regular_file(packaged_executable: Path) -> bool:
    """Return True when ``chrome-sandbox`` exists as a regular file."""
    if sys.platform != "linux":
        return False
    sandbox = packaged_executable.parent / "chrome-sandbox"
    try:
        sandbox_lstat = sandbox.lstat()
    except OSError:
        return False
    return stat.S_ISREG(sandbox_lstat.st_mode)


def _desktop_linux_sandbox_fixup(packaged_executable: Path) -> bool:
    """Configure Electron's Linux SUID sandbox helper when required."""
    from hermes_cli.main import _desktop_linux_userns_sandbox_available
    if sys.platform != "linux":
        return True

    sandbox = packaged_executable.parent / "chrome-sandbox"
    if not sandbox.exists():
        print(f"✗ Hermes Desktop is missing Electron's Linux sandbox helper: {sandbox}")
        return False

    # Reject symlinks — chown/chmod must not follow an attacker-controlled
    # link to an arbitrary path.  Use lstat() so we inspect the link itself
    # rather than the target, and require a regular file.
    try:
        sandbox_lstat = sandbox.lstat()
    except OSError:
        print(f"✗ Cannot stat Electron's Linux sandbox helper: {sandbox}")
        return False
    if not stat.S_ISREG(sandbox_lstat.st_mode):
        print(f"✗ Electron's Linux sandbox helper is not a regular file: {sandbox}")
        return False

    if sandbox_lstat.st_uid == 0 and stat.S_IMODE(sandbox_lstat.st_mode) == 0o4755:
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
    """Return True when Chromium should skip the present-but-non-setuid helper.

    A user-owned ``chrome-sandbox`` still makes Chromium abort with
    ``setuid_sandbox_host`` even when the namespace sandbox works. Passing
    ``--disable-setuid-sandbox`` keeps the userns sandbox and avoids sudo.
    Call only after ``_desktop_linux_sandbox_fixup`` succeeded without making
    the helper root-owned 4755 (the userns path). Does not re-probe userns.
    """
    if sys.platform != "linux":
        return False
    sandbox = packaged_executable.parent / "chrome-sandbox"
    try:
        sandbox_lstat = sandbox.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(sandbox_lstat.st_mode):
        return False
    if sandbox_lstat.st_uid == 0 and stat.S_IMODE(sandbox_lstat.st_mode) == 0o4755:
        return False
    return True


_LINUX_PASSWORD_STORES = frozenset({"gnome-libsecret", "kwallet", "kwallet5", "kwallet6", "basic"})


def _detect_linux_password_store() -> str | None:
    """Detect the Chromium password-store backend for the current Linux session.

    Electron's safeStorage only reports encryption as available when Chromium
    selects the right keychain backend, and Chromium's own detection routinely
    fails under `hermes desktop` because the launcher environment doesn't look
    like a full desktop session. Probe order: KDE session env vars, GNOME
    Keyring's control socket, then a D-Bus ping of org.freedesktop.secrets
    (covers any Secret Service implementation, e.g. KeePassXC). Returns None
    when no keychain daemon is reachable.
    """
    kde_version = os.environ.get("KDE_SESSION_VERSION", "").strip()
    if kde_version == "6":
        return "kwallet6"
    if kde_version == "5":
        return "kwallet5"
    if kde_version:
        return "kwallet"
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

    Returns ``(electron_flags, disable_gpu, password_store, ozone_hint)`` where
    ``electron_flags`` is a list of extra Electron CLI flags, ``disable_gpu``
    is one of "auto"/"1"/"0" (normalized for the HERMES_DESKTOP_DISABLE_GPU
    env var the Electron app reads), ``password_store`` is "auto" or one
    of the Chromium password-store backends (unknown values normalize to
    "auto"), and ``ozone_hint`` is one of "auto"/"x11"/"wayland" (normalized
    for ``ELECTRON_OZONE_PLATFORM_HINT``). Best-effort: any config error
    yields the safe defaults ``([], "auto", "auto", "auto")`` so a malformed
    config never blocks the launch.
    """
    flags: list[str] = []
    disable_gpu = "auto"
    password_store = "auto"
    ozone_hint = "auto"
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
        else:
            disable_gpu = "auto"

    raw_store = desktop_cfg.get("password_store", "auto")
    if isinstance(raw_store, str):
        low_store = raw_store.strip().lower()
        if low_store in _LINUX_PASSWORD_STORES:
            password_store = low_store

    raw_ozone = desktop_cfg.get("ozone_platform_hint", "auto")
    if isinstance(raw_ozone, str):
        low_ozone = raw_ozone.strip().lower()
        if low_ozone in ("auto", "x11", "wayland"):
            ozone_hint = low_ozone
    return flags, disable_gpu, password_store, ozone_hint


def _register_linux_desktop_entry() -> None:
    """Install the XDG desktop entry for Hermes Desktop (Linux only, best-effort).

    Gives the Electron app a launcher presence: a menu item and an icon.
    ``Exec`` and ``Icon`` are absolute, so the entry works outside a login
    shell. ``hermes uninstall --gui`` removes it.
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


def _build_desktop_app(desktop_dir: Path, *, source_mode: bool, npm: str, env: dict) -> Optional[Path]:
    """npm-install + build the desktop app; stage-and-swap the packaged tree.

    Returns the freshly installed packaged executable (non-source mode) or
    None (source mode builds ``dist/`` in place). Exits the process on any
    unrecoverable build failure, leaving the previous packaged app untouched.
    """
    from hermes_cli.main import PROJECT_ROOT, _desktop_macos_relaunchable_fixup, _desktop_packaged_executable, _desktop_staging_dir, _electron_dist_ok, _ensure_desktop_exe_launchable, _purge_electron_build_cache, _redownload_electron_dist, _run_npm_install_deterministic, _stop_desktop_processes_locking_build, _swap_staged_desktop_app, _write_desktop_build_stamp
    from hermes_constants import with_hermes_node_path

    print("→ Installing desktop workspace dependencies...")
    # Put the Hermes-managed Node on PATH so npm's child scripts (which
    # shell out to bare `node`, e.g. electron-winstaller's
    # select-7z-arch.js) resolve it even when the parent PATH is
    # stripped — the desktop updater chain (Desktop → hermes-setup →
    # hermes update) loses shell PATH customizations. Wrapping the
    # NixOS build env keeps its PYTHON hint while restoring managed Node
    # ahead of a bare PATH (same idiom as the `hermes update` path).
    nixos_env = with_hermes_node_path(_nixos_build_env())
    install_result = _run_npm_install_deterministic(npm, PROJECT_ROOT, capture_output=False, env=nixos_env)
    if install_result.returncode != 0:
        if not _electron_pkg_staged_missing_dist(PROJECT_ROOT):
            print("✗ Desktop dependency install failed")
            print(f"  Run manually:  cd {PROJECT_ROOT} && npm ci")
            sys.exit(install_result.returncode or 1)
        repaired = _try_redownload_electron_dist(PROJECT_ROOT, env)
        if repaired:
            print("  ⚠ Dependency install failed with a missing Electron dist; "
                  "repopulated it and continuing.")
        else:
            print("  ⚠ Dependency install failed with a missing Electron dist; "
                  "continuing to the build so electron-builder can attempt "
                  "the Electron fetch itself.")

    build_label = "source build" if source_mode else "packaged app"
    print(f"→ Building desktop {build_label}...")
    build_script = "build" if source_mode else "pack"
    if _force_adhoc_macos_signing(env, source_mode=source_mode):
        print("  → No Developer ID configured; ad-hoc signing this local rebuild "
              "(CSC_IDENTITY_AUTO_DISCOVERY=false)")
    npm_build_env = _npm_lifecycle_env(env)
    # Stage-and-swap (#86443): electron-builder packs IN PLACE and
    # before-pack.mjs wipes release/<unpacked> first, so a pack that
    # fails afterwards used to leave the user with NO app. Build into
    # a fresh staging output dir instead; the live release/ tree is
    # only replaced — by rename — after the staged result verifies.
    staging_dir: Optional[Path] = None
    build_cmd = [npm, "run", build_script]
    if not source_mode:
        staging_dir = _desktop_staging_dir(desktop_dir)
        build_cmd += ["--", f"-c.directories.output={staging_dir}"]
        # A running desktop instance launched from release/win-unpacked
        # holds Hermes.exe locked on Windows, so the pack can't replace
        # it ("Access is denied" / ERR_ELECTRON_BUILDER_CANNOT_EXECUTE).
        # Stop it first so the rebuild — including the installer's
        # headless --update rebuild — succeeds instead of failing cryptically.
        stopped = _stop_desktop_processes_locking_build(desktop_dir)
        if stopped:
            print(f"  ⚠ Stopped running desktop app to free the build output (pid {', '.join(map(str, stopped))})")

    def _staged_exe() -> Optional[Path]:
        return _desktop_packaged_executable_in(staging_dir) if staging_dir else None

    build_result = subprocess.run(
        build_cmd, cwd=desktop_dir, env=npm_build_env, check=False
    )
    if (
        build_result.returncode != 0
        and not source_mode
        and _staged_exe() is None
    ):
        # Corrupt cached Electron zip → partial unpack → ENOENT on rename.
        # stdlib zipfile won't catch the common concat-junk case, so purge
        # and retry once; @electron/get SHASUM is the real gate.
        #
        # Gate on a MISSING packaged executable: that is the signature of
        # the corrupt-download class this recovery exists for. A late
        # failure such as macOS code signing leaves the executable in
        # place — redownloading Electron can't repair it, so the purge +
        # retry would only add another slow, identical failure (#40187).
        purged: list[Path] = []
        restored = False
        if not _electron_dist_ok(PROJECT_ROOT):
            purged = _purge_electron_build_cache(desktop_dir, release_dir=staging_dir)
            restored = _redownload_electron_dist(PROJECT_ROOT, env)
        if restored:
            print("  ⚠ Desktop build failed; refreshed the Electron download and retrying once...")
            for p in purged:
                print(f"    - {p}")
            # The purge can't remove a win-unpacked tree whose Hermes.exe
            # is still locked by a running instance; stop it before retry.
            _stop_desktop_processes_locking_build(desktop_dir)
            build_result = subprocess.run(
                build_cmd, cwd=desktop_dir, env=npm_build_env, check=False
            )
    if (
        build_result.returncode != 0
        and not source_mode
        and not env.get("ELECTRON_MIRROR")
        and _staged_exe() is None
    ):
        print("  ⚠ Desktop build still failing; the Electron download from "
              "GitHub looks blocked. Re-downloading via a public mirror "
              "(npmmirror.com)... (set ELECTRON_MIRROR to use another mirror)")
        mirror = _ELECTRON_FALLBACK_MIRROR
        mirror_env = dict(npm_build_env)
        mirror_env["ELECTRON_MIRROR"] = mirror
        if not _electron_dist_ok(PROJECT_ROOT):
            _redownload_electron_dist(PROJECT_ROOT, env, mirror=mirror)
        _stop_desktop_processes_locking_build(desktop_dir)
        build_result = subprocess.run(build_cmd, cwd=desktop_dir, env=mirror_env, check=False)
    if build_result.returncode != 0:
        print("✗ Desktop GUI build failed")
        if staging_dir is not None:
            _discard_desktop_staging(staging_dir)
            if _desktop_packaged_executable(desktop_dir) is not None:
                print("  ↩ The previous desktop app was left untouched and still works.")
        print(f"  Run manually:  cd apps/desktop && npm run {build_script}")
        if sys.platform == "win32":
            print("  If this says \"Access is denied\" on Hermes.exe, close any")
            print("  running Hermes desktop window and retry.")
        print("  If the log shows Electron download retries, rebuild via a mirror:")
        print("    ELECTRON_MIRROR=<mirror-base-url> hermes desktop --force-build")
        sys.exit(build_result.returncode or 1)
    if not source_mode:
        assert staging_dir is not None
        staged_executable = _staged_exe()
        # Locally-built apps are ad-hoc signed; make them relaunchable after
        # an in-place self-update (otherwise macOS reports "Hermes is
        # damaged"). No-op on non-macOS and on real-identity builds.
        # Signs the STAGED bundle so the live app is never half-signed.
        _desktop_macos_relaunchable_fixup(desktop_dir, release_dir=staging_dir)

        # Windows integrity gate (#69179): never declare the rebuild a
        # success on a Hermes.exe Windows cannot load (truncated PE from
        # a corrupt cached Electron zip, wrong-arch tree, interrupted
        # rcedit rewrite). Verified on the STAGED exe: a failure here
        # simply discards the staging dir — the live app was never
        # touched — and fails loudly so the updater's retry-once
        # rebuilds from a fresh Electron download.
        verified_executable, rolled_back = _ensure_desktop_exe_launchable(
            desktop_dir, staged_executable
        )
        if staged_executable is None or rolled_back or verified_executable is None:
            _discard_desktop_staging(staging_dir)
            if staged_executable is None:
                print(f"✗ Desktop build produced no launchable app in {staging_dir}")
            print("  ↩ The previous desktop app was left untouched and still works.")
            sys.exit(1)
        # Verified: swap the staged tree over the live one (rename).
        packaged_executable = _swap_staged_desktop_app(desktop_dir, staging_dir)
        if packaged_executable is None:
            print(f"✗ Could not install the rebuilt desktop app into {desktop_dir / 'release'}")
            print("  ↩ The previous desktop app was left untouched and still works.")
            sys.exit(1)

    # Build succeeded — write the stamp so next run can skip
    _write_desktop_build_stamp(PROJECT_ROOT, source_mode=source_mode)
    return packaged_executable if not source_mode else None


def cmd_gui(args: argparse.Namespace):
    """Build and launch the native Electron desktop GUI."""
    from hermes_cli.main import PROJECT_ROOT, _desktop_build_needed, _desktop_dist_exists, _desktop_launch_options, _desktop_linux_needs_disable_setuid_sandbox, _desktop_linux_sandbox_fixup, _desktop_macos_setup_tcc_identity, _desktop_packaged_executable, _detect_linux_password_store, _register_linux_desktop_entry, _resolve_node_runtime_npm
    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if not (desktop_dir / "package.json").exists():
        print(f"Desktop GUI source not found at: {desktop_dir}")
        sys.exit(1)

    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

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

    # Desktop launch options from config.yaml (`desktop.electron_flags`,
    # `desktop.disable_gpu`, `desktop.ozone_platform_hint`). The GPU policy
    # and ozone hint are bridged to env vars the Electron/Chromium process
    # already reads; an explicit env var still wins over config so
    # `HERMES_DESKTOP_DISABLE_GPU=... hermes desktop` and
    # `ELECTRON_OZONE_PLATFORM_HINT=... hermes desktop` keep working.
    config_electron_flags, config_disable_gpu, config_password_store, config_ozone_hint = (
        _desktop_launch_options()
    )
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu
    if config_ozone_hint != "auto" and "ELECTRON_OZONE_PLATFORM_HINT" not in os.environ:
        env["ELECTRON_OZONE_PLATFORM_HINT"] = config_ozone_hint

    # Linux keychain backend for safeStorage (`desktop.password_store`).
    # Chromium needs the --password-store switch to pick the right keychain;
    # without it safeStorage.isEncryptionAvailable() is often false and the
    # desktop app refuses to persist remote gateway tokens. Config wins over
    # detection; an explicit env var wins over both so
    # `HERMES_DESKTOP_PASSWORD_STORE=... hermes desktop` keeps working.
    if sys.platform == "linux" and "HERMES_DESKTOP_PASSWORD_STORE" not in os.environ:
        password_store = (
            config_password_store
            if config_password_store != "auto"
            else _detect_linux_password_store()
        )
        if password_store:
            env["HERMES_DESKTOP_PASSWORD_STORE"] = password_store

    source_mode = getattr(args, "source", False)
    skip_build = getattr(args, "skip_build", False)
    force_build = getattr(args, "force_build", False)

    # macOS-only one-shot: create a self-signed code-signing identity so TCC
    # grants survive rebuilds, then exit without building/launching.
    if getattr(args, "setup_tcc_identity", False):
        identity = getattr(args, "identity", None) or "Hermes Local Signing"
        ok = _desktop_macos_setup_tcc_identity(identity)
        sys.exit(0 if ok else 1)

    packaged_executable = _desktop_packaged_executable(desktop_dir)

    if source_mode or not skip_build:
        npm = _resolve_node_runtime_npm()
        if not npm:
            print("Desktop GUI requires Node.js/npm, but npm was not found on PATH.")
            print("Install Node.js, then run:  hermes gui")
            sys.exit(1)
    else:
        npm = None

    if skip_build:
        if source_mode:
            if not _desktop_dist_exists(desktop_dir):
                print(f"✗ --skip-build --source was passed but no desktop dist found at: {desktop_dir / 'dist'}")
                print("  Pre-build first:  cd apps/desktop && npm run build")
                print("  Or drop --skip-build to install dependencies and build automatically.")
                sys.exit(1)
            if not (_electron_dir(PROJECT_ROOT) / "package.json").exists():
                print("✗ --skip-build --source requires existing desktop workspace dependencies.")
                print(f"  Install first:  cd {PROJECT_ROOT} && npm ci")
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
    else:
        # Check the content-hash stamp before doing any build work.
        # If the source tree hasn't changed since the last successful build,
        # skip the npm install + build entirely (saves a ton of useless work).
        # --force-build overrides the stamp and always rebuilds.
        build_needed = force_build or _desktop_build_needed(
            desktop_dir, PROJECT_ROOT, source_mode=source_mode
        )
        if not build_needed:
            build_label = "source build" if source_mode else "packaged app"
            print(f"✓ Desktop {build_label} is up to date (content stamp matches)")
        else:
            built = _build_desktop_app(desktop_dir, source_mode=source_mode, npm=npm, env=env)
            if not source_mode:
                packaged_executable = built

    # Linux: register the app in the desktop launcher, so Hermes shows up
    # in the application menu with its icon. Best-effort and idempotent.
    # A failure must never stop the app from launching.
    _register_linux_desktop_entry()

    # --build-only: produce the artifact but do NOT launch. The installer's
    # --update flow drives the rebuild headlessly and then launches the desktop
    # itself (detached, after the old exe has exited), so the launch must NOT
    # happen here — it would block the installer and, on Windows, the old exe
    # is still being replaced. Verify the expected artifact exists so a silent
    # "built nothing" can't slip past, then return success.
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
        electron_argv = [npm, "exec", "--", "electron", "."]
        if getattr(args, "local", False):
            electron_argv.append("--local")
        launch_result = subprocess.run(electron_argv, cwd=desktop_dir, env=env, check=False)
        sys.exit(launch_result.returncode)

    if packaged_executable is None:
        print(f"✗ Desktop package build completed but no launchable app was found at: {desktop_dir / 'release'}")
        print("  Expected an unpacked Electron app for the current OS.")
        sys.exit(1)

    launch_command = [str(packaged_executable)]
    if not _desktop_linux_sandbox_fixup(packaged_executable):
        if _desktop_linux_needs_no_sandbox() and _desktop_linux_sandbox_helper_is_regular_file(packaged_executable):
            print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces and the Electron sandbox helper could not be configured.")
            launch_command.append("--no-sandbox")
        else:
            sys.exit(1)
    elif _desktop_linux_needs_disable_setuid_sandbox(packaged_executable):
        launch_command.append("--disable-setuid-sandbox")

    launch_command.extend(config_electron_flags)
    if getattr(args, "local", False):
        launch_command.append("--local")
    print(f"→ Launching packaged Hermes Desktop: {' '.join(launch_command)}")
    launch_result = subprocess.run(launch_command, cwd=desktop_dir, env=env, check=False)
    sys.exit(launch_result.returncode)
