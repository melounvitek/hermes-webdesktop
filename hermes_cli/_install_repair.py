"""Dependency install execution shared between early recovery and full recovery.

- ``hermes_cli._early_recovery.recover_if_needed`` — stdlib-only, runs BEFORE ``hermes_cli.main``'s
third-party imports, so it can complete a pending update while no native extension is mapped yet
(#83569). - ``hermes_cli.main._recover_core_update_marker_locked`` — the historical post-import
recovery path.

This module is deliberately **stdlib-only** so importing it can never fail in the corrupted-venv
state it exists to repair. ``hermes_cli.main`` imports ``managed_uv``, ``hermes_constants``, and
friends only in its late path; the early path must not.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Single source of truth for the recovery-lock lifecycle and uv lookup —
# _early_recovery already owns both, and importing it is free (stdlib-only).
from hermes_cli import _early_recovery as _er


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_termux_env(env: dict | None = None) -> bool:
    """Stdlib Termux probe (hermes_cli.main's version lives behind imports)."""
    env = env if env is not None else os.environ
    try:
        return bool(env.get("TERMUX_VERSION")) or "com.termux" in env.get("PREFIX", "")
    except Exception:
        return False


@contextlib.contextmanager
def _stdout_to_stderr():
    """Route fd 1 (and sys.stdout) to stderr for the duration of an install.

    ``hermes acp`` speaks JSON-RPC on stdout; an inherited-fd install child writing there would
    corrupt the protocol. Mirrors ``main.py::_recover_from_interrupted_install``.
    """
    saved_sys_stdout = sys.stdout
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_fd = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved_sys_stdout
        if saved_fd is not None:
            with contextlib.suppress(OSError):
                os.dup2(saved_fd, 1)
            with contextlib.suppress(OSError):
                os.close(saved_fd)


def _resolve_install_target(root: Path) -> tuple[list[str], dict | None]:
    """(install_cmd_prefix, env) for the project venv — stdlib uv lookup.

    Mirrors ``main.py::_default_venv_install_target`` without ``managed_uv``. ``VIRTUAL_ENV``
    steers ``uv pip`` at the project venv even when invoked from the base interpreter (early
    recovery). Termux strips leaked interpreter-path env vars so uv resolves the venv correctly.
    """
    uv_bin = _er._find_uv_binary()
    if uv_bin:
        from hermes_constants import project_venv_dir

        env = {**os.environ, "VIRTUAL_ENV": str(project_venv_dir(root) or root / "venv")}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _venv_scripts_dir(root: Path) -> Path | None:
    """Project venv Scripts/bin dir, when present. stdlib-only."""
    # hermes_constants is stdlib-only, so the canonical layout helpers are safe
    # to use from this corrupted-venv repair path (#76105: never open-code
    # the Scripts/bin split).
    from hermes_constants import project_venv_dir, venv_bin_dir

    venv_dir = project_venv_dir(root)
    if venv_dir is None:
        return None

    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


#: Launcher command names install.ps1's Set-PathVariable exposes from the
#: managed binary dir (the default Hermes root's ``bin``, next to uv.exe)
#: on the user PATH. Keep in lockstep with the launcher list in
#: scripts/install.ps1.
_WINDOWS_BIN_LAUNCHERS = ("hermes", "hermes-acp")


def _launcher_present(target: Path, name: str) -> bool:
    return (target / f"{name}.exe").exists() or (target / f"{name}.cmd").exists()


def _launchers_missing(target: Path) -> bool:
    return any(not _launcher_present(target, name) for name in _WINDOWS_BIN_LAUNCHERS)


def _default_hermes_root() -> Path | None:
    """Per-machine anchor for the managed-clone gate — the DEFAULT Hermes root, not
    ``get_hermes_home()``: under ``hermes -p <name>`` that returns ``profiles\\<name>``, which
    would fail the gate and silently skip the heal for profile users. ``None`` when unresolvable."""
    from hermes_constants import get_default_hermes_root

    try:
        return Path(get_default_hermes_root())
    except Exception:
        return None


def _venv_is_relocatable(venv_dir: Path) -> bool:
    r"""True when the venv's pyvenv.cfg declares ``relocatable = true``.

    uv writes the flag; ``managed_uv`` builds replacement venvs ``--relocatable``. A relocatable
    venv's console-script trampolines embed a RELATIVE interpreter reference, so a copy placed
    outside ``venv\Scripts`` fails with ``uv trampoline failed to canonicalize script path``;
    non-relocatable venvs embed the absolute path and survive copying. This decides which
    launcher form a PATH dir gets.
    """
    try:
        cfg = (Path(venv_dir) / "pyvenv.cfg").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False
    return any(
        key.strip().lower() == "relocatable" and value.strip().lower() == "true"
        for key, _, value in (line.partition("=") for line in cfg.splitlines())
    )


def _normalize_windows_path(value) -> str:
    """Windows path equality key: backslashes, no trailing separator, lowered.

    Lowercase via ``.lower()`` (what ``ntpath.normcase`` does) rather than ``os.path.normcase`` —
    that is an identity function on POSIX, and this comparison must behave Windows-correct even when
    tests exercise the Windows branch from another host (same rationale as
    ``venv_bin_dir(windows=...)``).
    """
    return str(value).replace("/", "\\").rstrip("\\").lower()


def _windows_user_path_entries() -> list[str]:
    """User PATH entries from the registry — the value install.ps1 writes.

    Falls back to the process PATH when the registry is unreadable. Only called on Windows.
    """
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            raw, _kind = winreg.QueryValueEx(key, "Path")
        value = os.path.expandvars(str(raw))
    except (OSError, ImportError):
        value = os.environ.get("PATH", "")
    return [entry for entry in value.split(";") if entry.strip()]


def ensure_windows_bin_launchers(
    root,
    *,
    windows: bool | None = None,
    user_path_entries: list[str] | None = None,
) -> list[str]:
    r"""Re-stage the Windows ``hermes`` launchers when they vanish.

    On Windows, ``hermes`` resolves through launchers derived from the venv console scripts — never
    ``venv\Scripts`` itself on PATH, which would shadow the user's ``python`` (#83797).

    - canonical managed binary dir: only when *root* is the managed clone (``root.parent ==
    get_default_hermes_root()``), so source checkouts elsewhere never gain launchers; - legacy
    ``<root>\bin``: only when that dir is on the user PATH (registry value, process PATH as
    fallback), i.e.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return []

    root = Path(root)

    home = _default_hermes_root()
    if home is None:
        return []

    targets: list[Path] = []

    # Canonical target — gate on the managed-clone shape. This runs at
    # every hermes_cli.main process start (right after the profile
    # override), so the healthy path must stay at a couple of stat calls.
    if _normalize_windows_path(root.parent) == _normalize_windows_path(home):
        canonical = home / "bin"
        if _launchers_missing(canonical):
            targets.append(canonical)

    # Legacy transition target — the pre-migration in-checkout dir. Only
    # re-staged while the user PATH still points at it (consent), compared
    # as normalized literal strings: the installer wrote the long literal
    # path, and realpath'ing arbitrary PATH entries could hang on dead
    # network shares. An entry stored some other way (8.3 short path,
    # subst drive) misses the re-stage, which fails safe: no-op.
    legacy = root / "bin"
    if _launchers_missing(legacy):
        if user_path_entries is None:
            user_path_entries = _windows_user_path_entries()
        configured = {_normalize_windows_path(entry) for entry in user_path_entries}
        if _normalize_windows_path(legacy) in configured:
            targets.append(legacy)

    if not targets:
        return []

    from hermes_constants import project_venv_dir, venv_bin_dir

    venv_dir = project_venv_dir(root)
    if venv_dir is None:
        return []
    scripts_dir = venv_bin_dir(venv_dir, windows=windows)
    sources = [
        (name, scripts_dir / f"{name}.exe")
        for name in _WINDOWS_BIN_LAUNCHERS
        if (scripts_dir / f"{name}.exe").is_file()
    ]
    if not sources:
        return []
    relocatable = _venv_is_relocatable(venv_dir)

    restored: list[str] = []
    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        for name, source in sources:
            if _launcher_present(target, name):
                continue
            final = target / (f"{name}.cmd" if relocatable else f"{name}.exe")
            staging = target / f"{final.name}.heal.{os.getpid()}"
            try:
                if relocatable:
                    staging.write_text(
                        "@echo off\r\n" f'"{source}" %*\r\n', encoding="ascii"
                    )
                else:
                    shutil.copy2(source, staging)
                os.replace(staging, final)
                restored.append(str(final))
            except OSError:
                with contextlib.suppress(OSError):
                    staging.unlink()
    if restored:
        # Guarded like everything else in this never-raises helper: a
        # closed/broken stderr must not turn a successful heal into a crash.
        with contextlib.suppress(OSError, ValueError):
            print(
                "  ✓ Restored hermes launcher(s): " + ", ".join(restored),
                file=sys.stderr,
            )
    return restored


def _read_user_path_raw() -> tuple[list[str], int]:
    """Raw (unexpanded) user PATH entries + registry value type.

    Raw so a rewrite preserves ``%VARS%`` exactly as the user stored them (same discipline as
    ``hermes_cli.uninstall``). Only called on Windows.
    """
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            raw, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return [], winreg.REG_EXPAND_SZ
    return [entry for entry in str(raw).split(";") if entry], int(kind)


def _write_user_path_raw(entries: list[str], kind: int) -> None:
    """Write the user PATH back, preserving the registry value type."""
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE
    ) as key:
        winreg.SetValueEx(key, "Path", 0, kind, ";".join(entries))


def migrate_windows_bin_path(
    root,
    *,
    windows: bool | None = None,
    read_user_path=None,
    write_user_path=None,
) -> bool:
    """One-time PATH migration to the ``HERMES_HOME\\bin`` launcher layout.

    Runs from the ``hermes update`` tail (and mirrors what install.ps1's
    Set-PathVariable does on fresh installs/repairs, which never reach
    existing installs — updates don't run install.ps1):

    1. stage the launcher copies into the managed binary dir (via
       :func:`ensure_windows_bin_launchers`);
    2. verify both launchers are present there — otherwise STOP, leaving
       the user PATH untouched (never strip a working entry before its
       replacement is proven);
    3. ensure the managed binary dir is on the user PATH (prepend);
    4. strip the legacy entries: ``<root>\\bin`` (in-checkout launcher dir
       the update autostash could sweep) and ``<root>\\venv\\Scripts``
       (shadowed the user's ``python``, #83797).

    The legacy ``<root>\\bin`` FILES are deliberately left in place: editor
    and ACP configs that captured absolute launcher paths keep working
    (the launchers run fine from there — only PATH resolution through a
    dir git could sweep was the bug), and the dir is git-ignored so it
    cannot dirty the tree.

    Registry writes preserve the stored value type and raw ``%VARS%``.
    Never raises; returns True when the canonical layout is in place.

    *read_user_path*/*write_user_path* are injectable for tests.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return False

    root = Path(root)

    from hermes_constants import venv_bin_dir

    home = _default_hermes_root()
    if home is None:
        return False
    if _normalize_windows_path(root.parent) != _normalize_windows_path(home):
        return False  # not the managed clone — nothing to migrate

    ensure_windows_bin_launchers(root, windows=windows, user_path_entries=[])

    home_bin = home / "bin"
    if any(
        not ((home_bin / f"{name}.exe").is_file() or (home_bin / f"{name}.cmd").is_file())
        for name in _WINDOWS_BIN_LAUNCHERS
    ):
        return False  # staging incomplete — leave the PATH alone

    if read_user_path is None:
        read_user_path = _read_user_path_raw
    if write_user_path is None:
        write_user_path = _write_user_path_raw

    try:
        entries, kind = read_user_path()
    except (OSError, ImportError):
        return False

    legacy_keys = {
        _normalize_windows_path(root / "bin"),
        # The pre-#83797 installer put the venv's Scripts dir itself on PATH,
        # always at the literal `venv` layout (never `.venv`) — this strips
        # that stale entry, so it must match what the installer wrote then,
        # not where the venv lives now.
        _normalize_windows_path(venv_bin_dir(root / "venv", windows=True)),
    }
    home_bin_key = _normalize_windows_path(home_bin)

    def _entry_key(entry: str) -> str:
        return _normalize_windows_path(os.path.expandvars(entry))

    kept = [e for e in entries if _entry_key(e) not in legacy_keys]
    have_home_bin = any(_entry_key(e) == home_bin_key for e in kept)
    if not have_home_bin:
        kept = [str(home_bin)] + kept

    if kept != entries:
        try:
            write_user_path(kept, kind)
        except (OSError, ImportError):
            return False
        with contextlib.suppress(OSError, ValueError):
            print(
                f"  ✓ hermes launchers now resolve from {home_bin} "
                "(legacy PATH entries removed)",
                file=sys.stderr,
            )
    return True


def _load_console_script_names(root: Path) -> list[str]:
    """``[project.scripts]`` names from pyproject.toml (tomllib, 3.11+)."""
    project = _er._load_pyproject_project(root)
    try:
        scripts = (project or {}).get("scripts", {}) or {}
        return [str(name) for name in scripts if name]
    except Exception:
        return []


class ShimQuarantineError(RuntimeError):
    """A live shim could not be renamed aside — the venv is contended (#87331).

    Raised BEFORE the install command runs. Callers (early-pass recovery, core-marker recovery)
    catch it like any install failure: the update-incomplete marker survives and a later launch
    retries once the holder exits — the contended venv is never mutated.
    """

    def __init__(self, failed_shims: list[str]):
        self.failed_shims = list(failed_shims)
        super().__init__(
            "could not quarantine live shim(s): " + ", ".join(self.failed_shims)
        )


def _quarantine_running_hermes_exe(
    scripts_dir: Path, *, failed_out: list[str] | None = None
) -> list[tuple[Path, Path]]:
    """Rename live hermes*.exe shims aside so the installer can rewrite them.

    Windows blocks REPLACE on a running .exe but allows RENAME. Best-effort: silently skips anything
    that cannot be renamed. Returns (original, quarantined) pairs. stdlib-only — the console-script
    set comes from pyproject ``[project.scripts]`` (fallback: the well-known trio).
    """
    if not _is_windows():
        return []
    names = set(_load_console_script_names(scripts_dir.parent.parent)) or {
        "hermes",
        "hermes-agent",
        "hermes-acp",
    }
    names.add("hermes-gateway")
    moved: list[tuple[Path, Path]] = []
    for name in sorted(names):
        shim = scripts_dir / f"{name}.exe"
        if not shim.exists():
            continue
        quarantined = shim.with_name(f"{name}.exe.old.{int(time.time() * 1000)}")
        try:
            os.rename(shim, quarantined)
            moved.append((shim, quarantined))
        except OSError:
            if failed_out is not None:
                failed_out.append(shim.name)
    return moved


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Put quarantined shims back when the installer did not replace them.

    Delegates to the shared helper in the stdlib-only ``_early_recovery`` module: one retry ladder
    and one recovery message for every restore site, instead of the near-identical copies that had
    already drifted (#75584).
    """
    _er.restore_quarantined_shims(moved)


def _run_install_cmd(cmd: list[str], *, env: dict | None, root: Path) -> None:
    """Run an install command with quarantine protection for venv shims.

    Fail-closed (#87331): when any live shim cannot be renamed aside, the venv is contended and the
    installer would die partway on the same locks — raise :class:`ShimQuarantineError` WITHOUT
    running it. The caller's marker-keeping failure handling turns that into "retry next launch".

    Raises CalledProcessError on install failure (callers implement the per-extra fallback ladder).
    """
    scripts_dir = _venv_scripts_dir(root) if _is_windows() else None
    failed: list[str] = []
    moved = (
        _quarantine_running_hermes_exe(scripts_dir, failed_out=failed)
        if scripts_dir
        else []
    )
    if failed:
        _restore_quarantined_exes(moved)
        raise ShimQuarantineError(failed)
    try:
        subprocess.run(cmd, cwd=root, check=True, env=env)
    finally:
        # Restore runs on success AND failure: a SUCCESSFUL install can still
        # skip the entry-points step entirely (uv audits an already-satisfied
        # editable install as a no-op and rewrites nothing), which would leave
        # the quarantined shims renamed aside and `hermes` gone from PATH
        # (#75584). _restore_quarantined_exes only renames back when the
        # installer did NOT write a fresh shim, so this is safe in both cases.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)


def _load_installable_optional_extras(root: Path, group: str) -> list[str]:
    """Optional extras referenced by a dependency group (all / termux-all)."""
    project = _er._load_pyproject_project(root)
    if project is None:
        return []
    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []
    referenced: list[str] = []
    for ref in optional_deps.get(group, []):
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)
    return referenced


def run_core_install(root: Path) -> None:
    """Full core ``.[all]`` editable reinstall — the recovery install.

    Equal in behavior to the install half of ``main.py::_recover_core_update_marker_locked``:

    - bootstrap pip via ensurepip (a killed install can leave the venv with no pip module at all) -
    prefer ``uv pip`` with VIRTUAL_ENV pointed at the project venv; fall back to ``python -m pip``
    when no uv binary is available - target ``.[all]`` (or ``.[termux-all]`` on Termux) with the
    per-extra fallback ladder when the combined extras resolve fails - quarantine live
    ``hermes*.exe`` shims on Windows so they can be replaced - route ALL install output to stderr
    (acp/JSON-RPC safety) - Termux strips leaked PYTHONPATH/PYTHONHOME from the uv env
    """
    prefix, env = _resolve_install_target(root)
    group = "termux-all" if _is_termux_env(env) else "all"

    with _stdout_to_stderr():
        _er._run_ensurepip(root)

        try:
            _run_install_cmd(
                prefix + ["install", "-e", f".[{group}]"], env=env, root=root
            )
            return
        except subprocess.CalledProcessError:
            print(
                "  ⚠ Optional extras failed, reinstalling base dependencies "
                "and retrying extras individually..."
            )

        _run_install_cmd(prefix + ["install", "-e", "."], env=env, root=root)

        failed_extras: list[str] = []
        installed_extras: list[str] = []
        for extra in _load_installable_optional_extras(root, group):
            try:
                _run_install_cmd(
                    prefix + ["install", "-e", f".[{extra}]"], env=env, root=root
                )
                installed_extras.append(extra)
            except subprocess.CalledProcessError:
                failed_extras.append(extra)
        if installed_extras:
            print(
                "  ✓ Reinstalled optional extras individually: "
                + ", ".join(installed_extras)
            )
        if failed_extras:
            print(
                "  ⚠ Skipped optional extras that still failed: "
                + ", ".join(failed_extras)
            )


# ---------------------------------------------------------------------------
# Marker metadata (attempt counter for early-pass retry backoff)
# ---------------------------------------------------------------------------


def bump_marker_attempts(marker_path: Path) -> int:
    """Increment an attempts counter stored inside the marker file.

    The marker's existence is the signal; opportunistic JSON body carries the retry count so a
    persistently failing install can back off instead of reinstall-hammering every launch.
    Corrupt/missing bodies restart at 1. Returns the new attempt count. Never raises.
    """
    attempts = _er._read_marker_attempts(marker_path) + 1
    with contextlib.suppress(OSError):
        marker_path.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    return attempts
