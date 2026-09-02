"""Shared constants for Hermes Agent.

Import-safe module with no dependencies — can be imported from anywhere without risk of circular
imports.
"""

import os
import re
import shutil
import stat
import sys
from contextvars import ContextVar, Token
from pathlib import Path


_profile_fallback_warned: bool = False
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar[str | object] = ContextVar("_HERMES_HOME_OVERRIDE", default=_UNSET)

# ── TUI busy-indicator styles ─────────────────────────────────────────
# Single source of truth shared by the CLI /indicator command, the TUI
# gateway config handler, and the /help command registry. Keep in sync
# with ``INDICATOR_STYLES`` / ``DEFAULT_INDICATOR_STYLE`` in
# ``ui-tui/src/app/interfaces.ts`` on the frontend side.
INDICATOR_STYLES: tuple[str, ...] = ("ascii", "emoji", "kaomoji", "unicode")
DEFAULT_INDICATOR_STYLE: str = "kaomoji"


def set_hermes_home_override(path: str | Path | None) -> Token:
    """Set a context-local Hermes home override and return its reset token.

    This is for in-process, per-task scoping. It deliberately does not mutate ``os.environ`` because
    that is shared by every thread in the process.
    """
    value: str | object = _UNSET if path is None else str(path)
    return _HERMES_HOME_OVERRIDE.set(value)


def reset_hermes_home_override(token: Token) -> None:
    """Restore the previous context-local Hermes home override."""
    _HERMES_HOME_OVERRIDE.reset(token)


def get_hermes_home_override() -> str | None:
    """Return the active context-local Hermes home override, if any."""
    override = _HERMES_HOME_OVERRIDE.get()
    return str(override) if override is not _UNSET and override else None


def _get_platform_default_hermes_home() -> Path:
    """Return the platform-native default Hermes home path."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _warn_profile_fallback_once() -> None:
    """Warn once when falling back to the default home while a profile is active.

    Guard: if a non-default profile is sticky-active but ``HERMES_HOME`` is unset, the fallback to
    the default profile is almost certainly wrong.
    """
    global _profile_fallback_warned
    if _profile_fallback_warned:
        return
    try:
        fallback_home = _get_platform_default_hermes_home()
        active_path = fallback_home / "active_profile"
        active = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else ""
    except (UnicodeDecodeError, OSError):
        active = ""
    if active and active != "default":
        _profile_fallback_warned = True
        # Write directly to stderr.  We intentionally do NOT route this
        # through ``logging`` because (a) this function is called at
        # module-import time from 30+ sites, often before logging is
        # configured, and (b) root-logger propagation would double-emit
        # on consoles where a StreamHandler is already attached.
        msg = (
            f"[HERMES_HOME fallback] HERMES_HOME is unset but active "
            f"profile is {active!r}. Falling back to {fallback_home}, which "
            f"is the DEFAULT profile — not {active!r}. Any data this "
            f"process writes will land in the wrong profile. The "
            f"subprocess spawner should pass HERMES_HOME explicitly "
            f"(see issue #18594)."
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: platform-native path).

    Resolution order: context-local override (see :func:`set_hermes_home_override`) →
    ``HERMES_HOME`` env var → the platform-native default. This is the single source of truth — all
    other copies should import this.
    """
    override = get_hermes_home_override()
    if override:
        return Path(override)
    if not os.environ.get("HERMES_HOME", "").strip():
        _warn_profile_fallback_once()
    return get_process_hermes_home()


def hermes_home_key(path: str | Path | None = None) -> str:
    """Return a stable key for a Hermes home/profile directory.

    Runtime registries use this key to isolate plugin-owned entries while keeping built-in
    registrations process-global. ``strict=False`` preserves useful behavior for profiles whose
    directories have not been created yet.
    """
    candidate = Path(path) if path is not None else get_hermes_home()
    resolved = candidate.expanduser().resolve(strict=False)
    return os.path.normcase(str(resolved))


def get_process_hermes_home() -> Path:
    """Return the Hermes home for the running process, ignoring task overrides.

    Unlike :func:`get_hermes_home`, this never follows the context-local override set by
    :func:`set_hermes_home_override`.

    Use this for machine/process-level dashboard-owned assets — theme YAML, dashboard plugin
    manifests — that live under the server's launch home and must stay visible even while a request
    is scoped to another profile (e.g. the embedded ``/chat`` running under ``--open-profile``).
    Shared by :func:`get_hermes_home` so the two never drift.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else _get_platform_default_hermes_home()


# Process-level memo for get_default_hermes_root(). The function resolves
# HERMES_HOME against the native home on every call (~80us of path
# resolution), and it is called at 31+ sites — every _load_global_auth_store()
# (per provider row in the /model picker), kanban, backup, gateway, update.
# Its result depends only on (HERMES_HOME, platform native home), which are
# compared for free on each call, so the memo is freshness-correct even if a
# test or plugin mutates HERMES_HOME mid-process.
_default_hermes_root_memo: "tuple[str, str, Path] | None" = None


def get_default_hermes_root() -> Path:
    """Return the root Hermes directory for profile-level operations.

    In profile mode where ``HERMES_HOME`` is ``<root>/profiles/<name>``, returns ``<root>`` so that
    ``profile list`` can see all profiles. Works both for standard (``~/.hermes/profiles/coder``)
    and Docker (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    global _default_hermes_root_memo
    native_home = _get_platform_default_hermes_home()
    env_home = os.environ.get("HERMES_HOME", "")
    if _default_hermes_root_memo is not None:
        memo_native, memo_env, memo_result = _default_hermes_root_memo
        if memo_native == str(native_home) and memo_env == env_home:
            return memo_result

    result = native_home
    if env_home:
        env_path = Path(env_home)
        try:
            env_path.resolve().relative_to(native_home.resolve())  # under ~/.hermes (normal or profile mode)
        except ValueError:
            # Docker / custom deployment: ``<root>/profiles/<name>`` roots at the grandparent,
            # otherwise HERMES_HOME itself is the root.
            result = env_path.parent.parent if env_path.parent.name == "profiles" else env_path
    _default_hermes_root_memo = (str(native_home), env_home, result)
    return result


# Named-profile deletion must survive stale mkdir from live serve/logging.
# The marker lives beside the profile dir, not inside it, so rmtree cannot
# erase the fact that the profile was deleted.
_DELETED_PROFILES_DIR = ".deleted"

# Files whose presence marks a directory as a real Hermes home. A fresh home
# always gains at least one of these on first use (config save, env backfill,
# session DB), while arbitrary directories that merely contain a ``profiles``
# path segment (e.g. ``/srv/profiles/buildcache``) do not.
_HERMES_HOME_MARKERS = ("config.yaml", ".env", "state.db")


def _is_hermes_profiles_root(profiles_dir: Path) -> bool:
    """Return True when *profiles_dir* is a canonical ``<hermes-home>/profiles``.

    Anchors named-profile recognition so it only fires for directories that provably live under a
    Hermes home: the classic ``~/.hermes`` layout, a root carrying Hermes-home marker files
    (Docker/custom ``HERMES_HOME`` like ``/opt/data``), a ``profiles/.deleted`` tombstone directory
    (only ever created by ``hermes profile delete``), or the process's resolved default Hermes root.
    """
    root = profiles_dir.parent
    if root.name == ".hermes":
        return True
    try:
        if (profiles_dir / _DELETED_PROFILES_DIR).is_dir() or any(
            (root / marker).exists() for marker in _HERMES_HOME_MARKERS
        ):
            return True
    except OSError:
        pass
    try:
        return root.resolve(strict=False) == get_default_hermes_root().resolve(strict=False)
    except OSError:
        return False


def named_profile_home(path: str | Path) -> Path | None:
    """Return ``<root>/profiles/<name>`` when *path* is that home or under it.

    A named profile home is only ``.../profiles/<id>`` where ``<id>`` does not start with ``.`` AND
    the ``profiles`` directory's parent is a real Hermes home (see
    :func:`_is_hermes_profiles_root`). A default Hermes home whose path merely contains a
    ``profiles`` segment (e.g.
    """
    current = Path(path)
    for candidate in (current, *current.parents):
        if (
            candidate.parent.name == "profiles"
            and not candidate.name.startswith(".")
            and _is_hermes_profiles_root(candidate.parent)
        ):
            return candidate
        # Stop at a default Hermes home so a coincidental ``profiles/``
        # ancestor is not treated as a named-profile root.
        if candidate.name == ".hermes":
            return None
    return None


def profile_tombstone_path(profile_home: Path) -> Path:
    return profile_home.parent / _DELETED_PROFILES_DIR / profile_home.name


def named_profile_is_deleted(profile_home: str | Path) -> bool:
    return profile_tombstone_path(Path(profile_home)).exists()


def mark_named_profile_deleted(profile_home: str | Path) -> None:
    marker = profile_tombstone_path(Path(profile_home))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("deleted\n", encoding="utf-8")


def clear_named_profile_deleted(profile_home: str | Path) -> None:
    profile_tombstone_path(Path(profile_home)).unlink(missing_ok=True)


def assert_named_profile_home_live(path: str | Path) -> None:
    """Refuse missing or tombstoned named profile homes."""
    home = named_profile_home(path)
    if home is None:
        return
    if named_profile_is_deleted(home) or not home.exists():
        raise FileNotFoundError(
            f"Named profile home does not exist: {home}. "
            "Create the profile explicitly before using it."
        )


def mkdir_under_hermes_home(path: str | Path) -> Path:
    """Create *path*, but never materialize a deleted/missing named profile."""
    target = Path(path)
    assert_named_profile_home_live(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _packaged_dir(env_var: str, default: Path | None, subdir: str) -> Path:
    """Resolve a package-manager-relocatable directory.

    Resolution order: 1. *env_var* (Nix wrapper / explicit override) 2. caller-supplied ``default``
    (typically the source-checkout path) 3. ``<HERMES_HOME>/<subdir>`` last-resort.
    """
    override = os.getenv(env_var, "").strip()
    if override:
        return Path(override)
    return default if default is not None else get_hermes_home() / subdir


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers."""
    return _packaged_dir("HERMES_OPTIONAL_SKILLS", default, "optional-skills")


def get_optional_mcps_dir(default: Path | None = None) -> Path:
    """Return the optional-mcps directory, honoring package-manager wrappers.

    Mirrors :func:`get_optional_skills_dir`: packaged installs may ship ``optional-mcps`` outside
    the Python package tree and expose it via ``HERMES_OPTIONAL_MCPS``.
    """
    return _packaged_dir("HERMES_OPTIONAL_MCPS", default, "optional-mcps")


def get_bundled_skills_dir(default: Path | None = None) -> Path:
    """Return the bundled skills directory for source and packaged installs.

    Resolution order: 1. ``HERMES_BUNDLED_SKILLS`` env var (Nix wrapper / explicit override) 2.
    Caller-supplied ``default`` (typically the source-checkout path) 3. ``<HERMES_HOME>/skills``
    last-resort
    """
    return _packaged_dir("HERMES_BUNDLED_SKILLS", default, "skills")


def get_hermes_dir(
    new_subpath: str,
    old_name: str,
    *,
    home: Path | None = None,
) -> Path:
    """Resolve a Hermes subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``). ``image_cache``) keep using it
    — no migration required.

    A bare empty ``<old_name>/`` directory does **not** count as "the legacy install is in use" —
    install scaffolds, manual ``mkdir`` work, and cleared-then-abandoned locations all create empty
    stubs that would otherwise silently shadow real data populated at ``<new_subpath>/``.
    """
    home = home or get_hermes_home()
    old_path = home / old_name
    return old_path if _legacy_path_has_content(old_path) else home / new_subpath


def iter_hermes_node_dirs(home: Path | None = None) -> list[Path]:
    """Return Hermes-managed Node.js directories in preferred lookup order.

    Windows installs unpack portable Node into ``%LOCALAPPDATA%\\hermes\\node``; POSIX installs use
    ``$HERMES_HOME/node/bin``. Both shapes are included on every platform so mixed or migrated
    installs still work.
    """
    node_dir = (home or get_hermes_home()) / "node"
    # NOTE: keep this ordering in sync with hermesManagedNodePathEntries() in
    # apps/desktop/electron/backend-env.ts — the Electron main process is Node
    # and cannot import this module, so the platform-ordering rule is mirrored
    # there (once; main.ts imports it rather than keeping its own copy).
    return [node_dir, node_dir / "bin"] if sys.platform == "win32" else [node_dir / "bin", node_dir]


_WINDOWS_NODE_SHIMS = {
    "npm": ["npm.cmd", "npm.exe", "npm"],
    "npx": ["npx.cmd", "npx.exe", "npx"],
    "node": ["node.exe", "node"],
}


def _candidate_node_command_names(command: str) -> list[str]:
    base = Path(command).name
    if sys.platform != "win32" or "." in base:
        return [base]
    # Prefer npm.cmd. PowerShell may block npm.ps1 by execution policy, and
    # CreateProcess cannot launch a bare .ps1 the way it can launch .cmd.
    return _WINDOWS_NODE_SHIMS.get(base.lower(), [f"{base}.cmd", f"{base}.exe", base])


def _iter_managed_node_candidates(names: list[str], home: Path | None = None):
    """Yield existing (and on POSIX, executable) ``<node-dir>/<name>`` files."""
    for directory in iter_hermes_node_dirs(home):
        for name in names:
            candidate = directory / name
            if candidate.is_file() and (
                sys.platform == "win32" or os.access(candidate, os.X_OK)
            ):
                yield candidate


def _first_runnable_managed(names: list[str]) -> tuple[str | None, bool]:
    """Return ``(first runnable candidate, saw a broken one)``."""
    broken = False
    for candidate in _iter_managed_node_candidates(names):
        resolved = str(candidate)
        if node_tool_runnable(resolved):
            return resolved, broken
        broken = True
    return None, broken


def _run_version_probe(argv: list[str], **kwargs):
    """Run ``argv`` (a ``--version`` probe) hidden; ``None`` when it cannot run."""
    import subprocess

    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        return subprocess.run(
            argv,
            capture_output=True,
            timeout=10,
            creationflags=windows_hide_flags(),
            **kwargs,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _version_probe_ok(path: str) -> bool:
    """True when ``<path> --version`` exits 0 under the Hermes-managed Node PATH."""
    result = _run_version_probe([path, "--version"], env=with_hermes_node_path())
    return result is not None and result.returncode == 0


_HERMES_NODE_TARGET_MAJOR = int(os.environ.get("HERMES_NODE_TARGET_MAJOR", "22"))
_managed_node_heal_attempted = False
_NODE_BOOTSTRAP_SCRIPT = Path(__file__).resolve().parent / "scripts" / "lib" / "node-bootstrap.sh"

# Install tree root (this file lives at <install_root>/hermes_constants.py).
# Used by secure_parent_dir() to skip chmod on the install dir — chmodding it
# 0700 breaks hermes-user traversal in Docker (UID 10000). See #25821, #93050.
_INSTALL_ROOT = Path(__file__).resolve().parent


def node_tool_runnable(path: str | None) -> bool:
    """Return True only when *path* is a Node/npm/npx binary that actually runs.

    Probe with ``--version`` (same pattern as :func:`agent_browser_runnable`) so broken managed
    wrappers are detected before use.
    """
    if not path:
        return False
    if sys.platform == "win32":
        if not Path(path).is_file():
            return False
    elif not os.path.exists(path) or not os.access(path, os.X_OK):
        return False
    return _version_probe_ok(path)


def hermes_managed_node_tree_present(home: Path | None = None) -> bool:
    """Return True when any Hermes-managed node/npm/npx shim exists on disk."""
    names = [n for c in ("node", "npm", "npx") for n in _candidate_node_command_names(c)]
    return next(_iter_managed_node_candidates(names, home), None) is not None


def _path_under_any(path: str, roots: list[str]) -> bool:
    """Return True when *path* sits inside one of *roots* (same drive).

    Windows paths are case-insensitive and psutil / env vars can disagree on drive-letter casing,
    so compare through ``normcase`` (a no-op on POSIX). Roots are evaluated individually.
    """
    path_norm = os.path.normcase(os.path.normpath(path))
    for root in roots:
        root_norm = os.path.normcase(os.path.normpath(root))
        try:
            if os.path.commonpath([path_norm, root_norm]) == root_norm:
                return True
        except ValueError:
            # Different drives on Windows — commonpath raises.
            continue
    return False


def managed_node_tree_in_use(home: Path | None = None) -> bool:
    """Return True when any running process executes from the managed Node tree.

    Windows locks executables and loaded scripts against deletion or overwrite while a process runs
    them, so the updater must not rewrite ``%HERMES_HOME%\node`` while the desktop app's Node
    processes hold it — ``PermissionError: [WinError 5]`` on ``npm.cmd`` is the classic symptom
    (#80926).
    """
    if sys.platform != "win32":
        return False
    try:
        import psutil
    except Exception:
        return False
    dirs: list[str] = []
    for directory in iter_hermes_node_dirs(home):
        try:
            dirs.append(str(Path(directory).resolve()))
        except OSError:
            continue
    if not dirs:
        return False
    try:
        procs = psutil.process_iter(["exe", "cmdline"])
    except Exception:
        return False
    for proc in procs:
        try:
            info = proc.info
        except Exception:
            continue
        exe = info.get("exe")
        if exe:
            try:
                exe = str(Path(exe).resolve())
            except (OSError, ValueError):
                exe = str(exe)
        if any(_path_under_any(p, dirs) for p in ([exe] if exe else []) + list(info.get("cmdline") or [])):
            return True
    return False


_managed_node_in_use_notice_printed = False


def _print_managed_node_in_use_notice() -> None:
    """Print the managed-Node deferral notice once per process."""
    global _managed_node_in_use_notice_printed
    if _managed_node_in_use_notice_printed:
        return
    _managed_node_in_use_notice_printed = True
    print(
        "→ Hermes-managed Node.js is in use by a running app; deferring its "
        "upgrade until the app is closed (re-run `hermes update` afterwards).",
        flush=True,
    )


def _heal_managed_node_windows(home: Path | None = None) -> bool | None:
    """Redownload the portable Node zip into ``%HERMES_HOME%\\node`` on Windows.

    Returns ``True`` on success, ``False`` on a genuine failure (offline,
    download error, bad archive), and ``None`` when the tree is in use and the
    heal is deferred — callers must not record the once-per-process attempt
    for ``None`` so a later call can retry once the tree is free.

    The replacement is staging-first: the new tree is fully downloaded and
    extracted to a sibling ``node.new-*`` directory, then the live tree is
    renamed aside (``node.old-*``) and the staged tree renamed into place.
    The live tree is never deleted before its replacement is ready, so an
    interrupted heal cannot gut the running installation. Windows allows
    renaming a tree whose executables are running (images are mapped with
    ``FILE_SHARE_DELETE`` — the same mechanism as the hermes.exe quarantine);
    when the OS refuses the rename, that refusal *is* the in-use signal and
    the heal defers instead of forcing the write and crashing with
    ``PermissionError: [WinError 5]`` on ``npm.cmd`` (#80926).
    """
    import tempfile
    import time
    import urllib.request
    import uuid
    import zipfile

    arch = (os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")).lower()
    node_arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "x86": "x86"}.get(arch)
    if node_arch is None:
        return False

    home = home or get_hermes_home()
    target = home / "node"

    # Cheap pre-check: skip the download and staging work when the tree is
    # already visibly in use.  The rename-based swap below is the
    # authoritative guard — this scan only avoids pointless re-downloads for
    # long-lived processes whose npm resolution retries.
    if managed_node_tree_in_use(home):
        _print_managed_node_in_use_notice()
        return None

    # Best-effort sweep of staging/backup litter from interrupted runs; a
    # locked file simply stays for the next attempt.  Only dirs older than
    # 10 minutes are removed so a concurrent heal's in-flight swap (whose
    # staged/backup dirs are seconds old) is never disturbed.
    cutoff = time.time() - 600
    for stale in (*home.glob("node.old-*"), *home.glob("node.new-*")):
        try:
            if stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            continue

    def _fetch(url: str, timeout: int) -> bytes | None:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except OSError:
            return None

    index_url = f"https://nodejs.org/dist/latest-v{_HERMES_NODE_TARGET_MAJOR}.x/"
    index_bytes = _fetch(index_url, 60)
    if index_bytes is None:
        return False

    match = re.search(
        rf"node-v{_HERMES_NODE_TARGET_MAJOR}\.\d+\.\d+-win-{node_arch}\.zip",
        index_bytes.decode("utf-8", errors="replace"),
    )
    if not match:
        return False

    zip_name = match.group(0)
    zip_bytes = _fetch(f"{index_url}{zip_name}", 300)
    if zip_bytes is None:
        return False

    token = uuid.uuid4().hex[:8]
    staged = home / f"node.new-{token}"
    backup = home / f"node.old-{token}"
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / zip_name
            zip_path.write_bytes(zip_bytes)
            extract_dir = tmp_path / "extract"
            extract_dir.mkdir()
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            extracted = next(extract_dir.glob("node-v*"), None)
            if extracted is None or not extracted.is_dir():
                return False
            # Move the fully-extracted tree to a sibling staging dir so the
            # swap below is a same-volume rename.
            shutil.move(str(extracted), str(staged))
    except OSError:
        return False

    had_live = target.exists()
    if had_live:
        try:
            os.replace(str(target), str(backup))
        except OSError:
            # The OS refuses to move the live tree — a running process holds
            # it.  Defer; the old tree is untouched and the next resolution
            # (e.g. the next update after the app is closed) retries.
            _print_managed_node_in_use_notice()
            shutil.rmtree(staged, ignore_errors=True)
            return None
        # A rename preserves the directory's mtime, so a backup renamed from
        # a long-lived tree would instantly look older than the litter-sweep
        # cutoff to a concurrent heal.  Touch it (best-effort — a failure
        # must not abort the swap, which already succeeded) so the in-flight
        # backup is never swept mid-swap.
        try:
            os.utime(backup, None)
        except OSError:
            pass
    try:
        os.replace(str(staged), str(target))
    except OSError:
        if had_live:
            # Roll the live tree back and report the failure.
            try:
                os.replace(str(backup), str(target))
            except OSError:
                pass
        shutil.rmtree(staged, ignore_errors=True)
        return False
    if had_live:
        # The old tree is no longer canonical; locked files may keep it on
        # disk until the next heal attempt, which is safe.
        shutil.rmtree(backup, ignore_errors=True)
    return node_tool_runnable(str(target / "node.exe"))


def _run_node_bootstrap(func: str, *, timeout: int, **extra_env: str) -> bool:
    """Source ``scripts/lib/node-bootstrap.sh`` and run shell function *func*."""
    if not _NODE_BOOTSTRAP_SCRIPT.is_file():
        return False

    import subprocess

    try:
        result = subprocess.run(
            ["bash", "-c", f'source "{_NODE_BOOTSTRAP_SCRIPT}" && {func}'],
            env={**os.environ, "HERMES_HOME": str(get_hermes_home()), **extra_env},
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def bootstrap_hermes_managed_node() -> str | None:
    """Install a Hermes-managed Node tree and return its npm path.

    Used when the only Node/npm on the machine belongs to the user (system, nvm, brew, Nix) and
    cannot satisfy the repo's ``engines`` requirements — Hermes never modifies a toolchain it does
    not own, so instead it provisions its own tree under ``$HERMES_HOME/node`` (the same tree a
    fresh install creates) and works with that.
    """
    existing = find_hermes_node_executable("npm")
    if existing:
        return existing
    if sys.platform == "win32":
        ok = _heal_managed_node_windows()
    else:
        # POSIX: ``_nb_install_bundled_node`` in node-bootstrap.sh builds the same tree a fresh
        # install creates. HERMES_NODE_SKIP_LINKS=1 keeps node/npm/npx out of ~/.local/bin so the
        # user's own toolchain on PATH is never shadowed.
        ok = _run_node_bootstrap("_nb_install_bundled_node", timeout=600, HERMES_NODE_SKIP_LINKS="1")
    if not ok:
        return None
    return _first_runnable_managed(_candidate_node_command_names("npm"))[0]


def heal_hermes_managed_node() -> bool:
    """Redownload Hermes-managed Node when the tree exists but is broken.

    Runs at most once per process. POSIX shells out to ``heal_managed_node`` in node-bootstrap.sh;
    Windows downloads the portable zip directly. A Windows deferral (tree in use by a running app)
    does NOT record the attempt, so a later call or process can heal once the tree is free.
    """
    global _managed_node_heal_attempted
    if _managed_node_heal_attempted or not hermes_managed_node_tree_present():
        return False
    if sys.platform == "win32":
        result = _heal_managed_node_windows()
        if result is None:
            # In-use deferral: leave the attempt flag clear so a later call
            # in this process can heal after the app releases the tree.
            return False
        _managed_node_heal_attempted = True
        return bool(result)
    _managed_node_heal_attempted = True
    return _run_node_bootstrap("heal_managed_node", timeout=300)


def _managed_node_tree_outdated(home: Path | None = None) -> bool:
    """Return True when the managed tree's node runs but is below the target major.

    An outdated tree heals like a broken one: :func:`find_hermes_node_executable` triggers the
    once-per-process heal, which redownloads the target major, so existing users are upgraded on
    next launch rather than on the next installer run.
    """
    for candidate in _iter_managed_node_candidates(_candidate_node_command_names("node"), home):
        result = _run_version_probe([str(candidate), "--version"])
        if result is None:
            return False  # broken, not outdated — the runnable probe handles it
        try:
            version = result.stdout.decode().strip().lstrip("v")
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            return False
        # A pre-release tree counts as outdated however high its major:
        # nodejs.org publishes a headers tarball only for final releases, so
        # node-gyp cannot build node-pty against one. Without this, an
        # install that adopted such a tree stays broken forever — the heal
        # only fires below the target major, and a pre-release is above it.
        # Mirrors node_satisfies_build() in scripts/install.sh.
        if "-" in version:
            return True
        return major < _HERMES_NODE_TARGET_MAJOR
    return False


def find_hermes_node_executable(command: str) -> str | None:
    """Return a Hermes-managed Node/npm executable path, healing broken trees.

    Outdated trees (major below ``_HERMES_NODE_TARGET_MAJOR``) heal the same way broken ones do.
    When the heal fails (offline, download error) an outdated-but-runnable tree is still returned:
    old Node beats no Node.
    """
    names = _candidate_node_command_names(command)
    resolved, broken_present = _first_runnable_managed(names)
    needs_heal = broken_present or (resolved is not None and _managed_node_tree_outdated())
    if needs_heal and heal_hermes_managed_node():
        healed, _ = _first_runnable_managed(names)
        if healed:
            return healed
    return resolved


def find_node_executable_on_path(command: str) -> str | None:
    """Return a Node/npm executable from PATH with Windows shim ordering.

    ``shutil.which("npm")`` can resolve an extensionless npm shim before the ``.cmd`` shim on
    Windows. Python's CreateProcess cannot execute that shim directly, so prefer the launchable
    variants explicitly for Hermes-owned subprocesses.
    """
    if sys.platform != "win32":
        return shutil.which(command)

    command_str = str(command)
    if any(sep and sep in command_str for sep in (os.sep, os.altsep, "/", "\\")):
        return command_str if Path(command_str).is_file() else None

    for name in _candidate_node_command_names(command_str):
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def find_node_executable(command: str) -> str | None:
    """Resolve a Node.js command, preferring healthy Hermes-managed installs.

    This is for Hermes-owned subprocesses that should not be broken by a bad, missing, or elevation-
    triggering system Node/npm on PATH. When a managed tree exists but cannot be healed, returns
    ``None`` instead of falling back to system npm on PATH.
    """
    managed = find_hermes_node_executable(command)
    if managed:
        return managed
    if hermes_managed_node_tree_present():
        return None
    return find_node_executable_on_path(command)


def with_hermes_node_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return *env* with Hermes-managed Node directories prepended to PATH."""
    merged = dict(os.environ if env is None else env)
    existing = merged.get("PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    managed = [str(path) for path in iter_hermes_node_dirs() if path.is_dir()]
    for entry in reversed(managed):
        if entry not in parts:
            parts.insert(0, entry)
    merged["PATH"] = os.pathsep.join(parts)
    return merged


def agent_browser_runnable(path: str | None) -> bool:
    """Return True only when *path* is an agent-browser CLI that actually runs.

    This validates the candidate by resolving it to a real, executable file and running
    ``--version`` with a short timeout. Returns True only on a clean (exit 0) run, so a dead/wrong-
    arch/hung binary is rejected and the caller can fall through to the next resolution candidate.

    Special cases: * ``None`` / empty → False. * The ``"npx agent-browser"`` fallback form (contains
    a space, not a real file) → True; npx resolves and validates the package at run time, so there
    is nothing to stat here.
    """
    if not path:
        return False
    # The npx fallback is a two-token command string, not a filesystem path.
    if " " in path and path.split()[0].endswith("npx"):
        return True
    # exists() follows symlinks — a dangling link returns False here, so we
    # never even spawn a subprocess for the broken-link case.
    if not os.path.exists(path) or not os.access(path, os.X_OK):
        return False
    return _version_probe_ok(path)


def _legacy_path_has_content(path: Path) -> bool:
    """Return ``True`` iff ``path`` exists and has content worth honouring.

    A populated directory or any non-directory file counts; an empty directory does not, so a
    stale empty stub falls through to the new layout. Any ``OSError`` other than not-found means
    "assume occupied" to avoid orphaning legacy data. Symlinks are resolved first; a dangling
    symlink does NOT count and must not shadow populated new-layout data.
    """
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            st = path.stat()  # judge a symlink on its target; dangling → FileNotFoundError
    except FileNotFoundError:
        return False
    except OSError:
        # PermissionError on a parent, or any other inspection failure:
        # treat as occupied rather than silently orphaning legacy data.
        return True
    if not stat.S_ISDIR(st.st_mode):
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        pass
    return True


def display_hermes_home() -> str:
    """Return a user-friendly display string for the current HERMES_HOME.

    Uses ``~/`` shorthand (``~/.hermes``, ``~/.hermes/profiles/coder``). Use this in user-facing
    messages instead of hardcoding ``~/.hermes``; code needing a real ``Path`` should use
    :func:`get_hermes_home`.
    """
    home = get_hermes_home()
    try:
        # as_posix(): on Windows, str() of a relative Path renders
        # backslashes, producing mixed-separator chimeras like
        # ``~/AppData\Local\hermes/skills/`` once callers append
        # sub-paths. ``~/`` shorthand implies POSIX rendering; keep the
        # whole string consistent (forward slashes work everywhere,
        # including Windows shells and Python APIs).
        return "~/" + home.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(home)


def secure_parent_dir(path: Path) -> None:
    """Chmod ``0o700`` on the parent directory of *path*, but only if safe.

    Refuses to chmod ``/`` or any top-level directory (resolved parent with fewer than 3 parts, i.e.
    ``/`` or any direct child like ``/usr``) to prevent catastrophic host bricking when
    ``HERMES_HOME`` or other path env vars resolve to an unexpected location.
    """
    parent = path.parent.resolve()
    # Refuse root and its direct children (/usr, /home, /var, /tmp, …).
    if parent == Path("/") or len(parent.parts) < 3:
        return
    # Refuse the install tree root. chmodding it 0700 breaks hermes-user
    # traversal in Docker (UID 10000) and any other install where the
    # runtime user doesn't own the install dir. See #25821, #93050.
    if parent == _INSTALL_ROOT or _INSTALL_ROOT in parent.parents:
        # A credential file inside the install tree usually means HERMES_HOME
        # resolved somewhere unexpected — surface it instead of skipping
        # silently, since this same misconfiguration previously caused
        # production lockouts.
        import logging

        logging.getLogger(__name__).warning(
            "Not restricting permissions on %s: it is inside the "
            "hermes-agent install directory (%s). Credential files are "
            "normally stored under the hermes home directory instead.",
            parent,
            _INSTALL_ROOT,
        )
        return
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass


def _norm_home_path(path: str | None) -> str:
    """Return a comparable absolute path string, or ``""`` for empty input."""
    raw = (path or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(raw)))
    except Exception:
        return os.path.normcase(raw)


def _profile_home_path(env: dict[str, str] | None = None) -> str | None:
    """Return ``{HERMES_HOME}/home`` when the profile-home directory exists."""
    hermes_home = get_hermes_home_override() or (env or {}).get("HERMES_HOME") or os.getenv("HERMES_HOME")
    if not hermes_home:
        return None
    profile_home = os.path.join(hermes_home, "home")
    return profile_home if os.path.isdir(profile_home) else None


def _is_profile_home(candidate: str | None, profile_home: str | None) -> bool:
    return bool(candidate and profile_home and _norm_home_path(candidate) == _norm_home_path(profile_home))


def _env_get(env: dict[str, str], key: str, default: str = "") -> str:
    """Stripped *key* from *env*, falling back to the process environment."""
    return str(env.get(key) or os.getenv(key, default)).strip()


def _iter_real_home_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Return likely OS-user home candidates in trust order."""
    env = env or {}
    candidates = [_env_get(env, "HERMES_REAL_HOME"), _env_get(env, "HOME")]
    try:
        import pwd

        candidates.append(pwd.getpwuid(os.getuid()).pw_dir.strip())  # windows-footgun: ok — POSIX-only module inside try/except
    except Exception:
        pass
    candidates.append(_env_get(env, "USERPROFILE"))
    drive, path = _env_get(env, "HOMEDRIVE"), _env_get(env, "HOMEPATH")
    if drive and path:
        candidates.append(f"{drive}{path}" if path.startswith(("\\", "/")) else os.path.join(drive, path))
    expanded = os.path.expanduser("~")
    if expanded != "~":
        candidates.append(expanded)
    return [c for c in candidates if c]


def get_real_home(env: dict[str, str] | None = None) -> str:
    """Return the OS user's real home directory, avoiding Hermes profile HOME.

    ``HERMES_HOME`` scopes Hermes state; ``HOME`` belongs to the OS account and the external CLIs
    that keep credentials under ``~``. If a parent already runs with ``HOME={HERMES_HOME}/home``,
    this repairs back to the account home when possible.
    """
    profile_home = _profile_home_path(env)
    seen: set[str] = set()
    for candidate in _iter_real_home_candidates(env):
        key = _norm_home_path(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if not _is_profile_home(candidate, profile_home):
            return candidate
    return "/tmp"


_HOME_MODE_ALIASES = {
    "isolated": "profile", "profile_home": "profile", "profile-home": "profile",
    "host": "real", "user": "real", "real_home": "real", "real-home": "real",
}


def get_subprocess_home(env: dict[str, str] | None = None) -> str | None:
    """Return a subprocess ``HOME`` override, if one should be applied.

    * ``auto`` (default): host installs keep the real user HOME; containers use
    ``{HERMES_HOME}/home`` for persistent state. If a host parent already has HOME pointed at the
    profile home, repair subprocesses back to real HOME. * ``real``: always prefer the real OS-user
    HOME.
    """
    env = env or {}
    profile_home = _profile_home_path(env)
    mode = _env_get(env, "TERMINAL_HOME_MODE", "auto").lower() or "auto"
    mode = _HOME_MODE_ALIASES.get(mode, mode)

    if mode == "profile":
        return profile_home

    real_home = get_real_home(env)
    current_home = _env_get(env, "HOME")
    repaired = real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None
    if mode == "real":
        return repaired

    if profile_home and is_container():
        return profile_home
    if _is_profile_home(current_home, profile_home):
        return repaired
    return None


def apply_subprocess_home_env(env: dict[str, str]) -> None:
    """Apply Hermes' subprocess HOME contract to *env* in-place."""
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
    home = get_subprocess_home(env)
    if home:
        env["HOME"] = home


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def parse_reasoning_effort(effort) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Returns None for empty/unrecognized input (caller uses the default) and ``{"enabled": False}``
    for "none" and its aliases ("false", "disabled", YAML boolean False): users write
    ``reasoning_effort: false``/``off``/``no`` and that must mean disabled, not "keep thinking".
    Valid levels: none, minimal, low, medium, high, xhigh, max, ultra.
    """
    if effort is False:
        return {"enabled": False}
    if effort is None or effort is True:
        return None
    effort = str(effort).strip().lower()
    if not effort:
        return None
    if effort in {"none", "false", "disabled"}:
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def _canonical_model_variants(model: str) -> list[str]:
    """Generate bounded spelling variants for tolerant override matching.

    Strategy: generate a small set of base forms, then apply version-dot recovery to EACH of them.
    This ensures symmetry: ``claude-opus-4.5``, ``claude-opus-4-5``, and ``claude-opus.4.5`` all
    produce the same variant set.

    Duplicates removed in insertion order (exact always wins).
    """
    # Version-dot regexes — digit-separator-digit interconversion
    _dash_to_dot = lambda s: re.sub(r'(\d)-(\d)', r'\1.\2', s)
    _dot_to_dash = lambda s: re.sub(r'(\d)\.(\d)', r'\1-\2', s)
    seen: set[str] = set()
    variants: list[str] = []

    def _add(*values):
        for v in values:
            if v and v not in seen:
                seen.add(v)
                variants.append(v)

    def _add_with_derivatives(s):
        """Add s plus its dots↔dashes and version-dot derivatives."""
        dashed, dotted = s.replace('.', '-'), s.replace('-', '.')
        _add(s, dashed, dotted, _dash_to_dot(s), _dot_to_dash(s), _dash_to_dot(dashed), _dot_to_dash(dotted))

    _add_with_derivatives(model)
    parts = model.split('/')
    if len(parts) >= 2:  # bare model (strip provider/aggregator prefix)
        _add_with_derivatives(parts[-1])
    if len(parts) >= 3:  # strip aggregator only: "openrouter/anthropic/x" → "anthropic/x"
        _add_with_derivatives('/'.join(parts[1:]))
    known_providers = (
        'anthropic', 'openai', 'google', 'openrouter', 'groq', 'mistral',
        'xai', 'cohere', 'perplexity', 'together', 'fireworks', 'deepseek',
    )
    for v in [v for v in variants if '/' not in v]:
        _add(*(f"{provider}/{v}" for provider in known_providers))
    known_aggregators = ('openrouter', 'opencode', 'fireworks', 'groq', 'together')
    for v in [v for v in variants if v.count('/') == 1]:
        _add(*(f"{agg}/{v}" for agg in known_aggregators))
    return variants


def resolve_per_model_reasoning_effort(model: str, overrides: dict | None) -> dict | None:
    """Lookup a per-model reasoning_effort override with spelling-tolerance.

    Resolution order: 1. Exact match 2. Dots ↔ dashes variants 3. Strip provider prefix (bare model
    name only) 4. Strip aggregator prefix (middle segment only) 5. Prepend known aggregator prefixes
    to bare/single-slash variants

    First non-None parse_reasoning_effort result wins.
    """
    if not overrides or not isinstance(overrides, dict) or not model:
        return None
    for variant in _canonical_model_variants(model):
        if variant in overrides:
            result = parse_reasoning_effort(overrides[variant])
            if result is not None:
                return result
    return None


def resolve_reasoning_config(cfg: dict | None, model: str = "") -> dict | None:
    """Resolve the effective reasoning config for *model* from a config dict.

    Single chokepoint for reasoning-effort resolution, shared by every surface (CLI startup,
    messaging gateway, Desktop/TUI, cron, ``/model`` switch, fallback activation). Priority:
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    agent_cfg = cfg.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}

    if not model:
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            model_cfg = model_cfg.get("default") or model_cfg.get("model") or ""
        model = model_cfg.strip() if isinstance(model_cfg, str) else ""

    overrides = agent_cfg.get("reasoning_overrides") or {}
    per_model = resolve_per_model_reasoning_effort(model, overrides)
    if per_model is not None:
        return per_model

    # Global fallback — keep the raw value; coercing with ``or ""`` turns a
    # YAML boolean False into "", silently re-enabling thinking for users
    # who explicitly disabled it.
    effort = agent_cfg.get("reasoning_effort", "")
    result = parse_reasoning_effort(effort)
    if effort and str(effort).strip() and result is None:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown reasoning_effort '%s', using default (medium)", effort
        )
    return result


def is_termux() -> bool:
    """Return True when running inside a Termux (Android) environment.

    Checks ``TERMUX_VERSION`` (set by Termux) or the Termux-specific ``PREFIX`` path. Import-safe —
    no heavy deps.
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return True when running inside WSL (Windows Subsystem for Linux).

    Checks ``/proc/version`` for the ``microsoft`` marker that both WSL1 and WSL2 inject. Result is
    cached for the process lifetime. Import-safe — no heavy deps.
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


def windows_path_to_wsl(path: str) -> str | None:
    """Convert a Windows drive path (``C:\\...``) to its ``/mnt/<drive>/...`` form."""
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", str(path or "").strip())
    if not match:
        return None
    drive, tail = match.group(1).lower(), match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def wsl_unc_path_to_posix(path: str) -> str | None:
    """Convert a Windows WSL UNC path (``\\\\wsl.localhost\\<distro>\\...`` or the
    legacy ``\\\\wsl$\\...``) to a POSIX path inside the distro."""
    normalized = str(path or "").strip().replace("/", "\\")
    match = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+\\(.*)$", normalized, re.IGNORECASE)
    if not match:
        return None
    tail = match.group(1).replace("\\", "/")
    return f"/{tail}" if tail else "/"


def translate_cwd_for_wsl_backend(cwd: str) -> str:
    """Normalize a cross-boundary cwd when Hermes itself runs inside WSL.

    A Windows-host UI (native picker / drive path / ``\\wsl.localhost\`` UNC) can hand the WSL
    backend a path it can't ``chdir`` into. Map it to the POSIX equivalent so the picker, sidebar,
    and sessions all agree on the workspace. No-op off WSL and for paths that are already POSIX.
    """
    if not is_wsl():
        return cwd
    for translator in (wsl_unc_path_to_posix, windows_path_to_wsl):
        translated = translator(cwd)
        if translated is not None:
            return translated
    return cwd


_container_detected: bool | None = None


def is_container() -> bool:
    """Return True when running inside a container.

    To cover those, also check: * ``KUBERNETES_SERVICE_HOST`` env var — set in every Kubernetes pod.
    * ``kubepods`` / ``containerd`` / ``crio`` markers in ``/proc/1/cgroup``. * the same markers in
    ``/proc/self/mountinfo`` (cgroup-v2 fallback).

    Result is cached for the process lifetime. Import-safe — no heavy deps.
    """
    global _container_detected
    if _container_detected is None:
        _container_detected = _detect_container()
    return _container_detected


def _proc_file_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return False
    return any(marker in content for marker in markers)


def _detect_container() -> bool:
    # Kubernetes always injects KUBERNETES_SERVICE_HOST into pod containers; absent on hosts.
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
        or _proc_file_has_marker("/proc/1/cgroup", ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio"))
    ):
        return True
    # cgroup v2: /proc/1/cgroup is just "0::/" with no marker. The container
    # runtime still shows up in the mount table (overlay rootfs, runtime mount
    # paths), so scan mountinfo as a last resort.
    return _proc_file_has_marker("/proc/self/mountinfo", ("kubepods", "containerd", "crio"))


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under HERMES_HOME."""
    return get_hermes_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under HERMES_HOME."""
    return get_hermes_home() / "skills"


def get_env_path() -> Path:
    """Return the path to the ``.env`` file under HERMES_HOME."""
    return get_hermes_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 connections.

    On servers with broken or unreachable IPv6, Python tries AAAA records first and hangs for the
    full TCP timeout before falling back to IPv4. This affects httpx, requests, urllib, the OpenAI
    SDK — everything that uses ``socket.getaddrinfo``.

    When *force* is True, patches ``getaddrinfo`` so that calls with ``family=AF_UNSPEC`` (the
    default) resolve as ``AF_INET`` instead, skipping IPv6 entirely. If no A record exists, falls
    back to the original unfiltered resolution so pure-IPv6 hosts still work.
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


# ─── Streaming Response Constants ────────────────────────────────────────────

# Response ID for partial stream stubs used during error recovery
PARTIAL_STREAM_STUB_ID = "partial-stream-stub"

FINISH_REASON_LENGTH = "length"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"


# ─── Venv layout ─────────────────────────────────────────────────────────────

def venv_bin_dir(venv_dir, *, windows: bool | None = None) -> Path:
    """Directory holding a venv's executables (``Scripts`` / ``bin``).

    Canonical helper; this was open-coded in many places with three different Windows predicates.
    *windows* lets callers pass their own platform verdict because tests patch predicates such as
    ``hermes_cli.main._is_windows`` to exercise Windows paths on Linux CI; reading ``sys.platform``
    here would drop those paths from coverage. The path is returned unconditionally: callers differ
    on whether a missing venv is an error, so existence checking stays with them.
    """
    if windows is None:
        windows = sys.platform == "win32"
    return Path(venv_dir) / ("Scripts" if windows else "bin")


def project_venv_dir(project_root) -> Path | None:
    """The project's venv directory, ``venv`` or ``.venv``, when one exists.

    ``uv venv`` defaults to ``.venv`` while our installers create ``venv``, so both layouts are in
    the wild. Call sites that only knew about ``venv`` silently no-oped on a ``.venv`` install —
    that is how the Windows shim-lock preflight skipped itself entirely (#79542).
    """
    for name in ("venv", ".venv"):
        candidate = Path(project_root) / name
        if candidate.is_dir():
            return candidate
    return None


def venv_python_path(venv_dir, *, windows: bool | None = None) -> Path:
    """Path to the Python interpreter inside *venv_dir* (may not exist)."""
    windows = sys.platform == "win32" if windows is None else windows
    return venv_bin_dir(venv_dir, windows=windows) / ("python.exe" if windows else "python")


# ─── Partial-update diagnostics ──────────────────────────────────────────────

# Top-level packages/modules that ship as part of Hermes itself. An ImportError
# naming one of these means our own tree is inconsistent; anything else is a
# third-party problem with different remediation. Single source of truth —
# `hermes_cli.update_cmd`'s post-update probe consumes this same set so the
# guard that BLOCKS and the hint that EXPLAINS can never disagree.
FIRST_PARTY_MODULE_ROOTS = frozenset({
    "agent", "acp_adapter", "cli", "cron", "gateway", "model_tools", "plugins",
    "providers", "tools", "toolsets", "run_agent", "tui_gateway", "utils",
})


def is_first_party_module(name: str | None) -> bool:
    """True when *name* is a module that ships with Hermes.

    Matches the first dotted segment against an exact set; a substring or ``startswith`` test would
    also claim third-party ``agents``, ``agentops``, and ``toolsets_x``.
    """
    root = str(name).split(".")[0] if name else ""
    return bool(root) and (root in FIRST_PARTY_MODULE_ROOTS or root.startswith("hermes_"))


def partial_update_hint(exc: BaseException) -> list[str]:
    """Return recovery guidance lines when *exc* looks like a half-updated tree.

    Users hit this as an opaque crash with no indication that the *install*, rather than their
    config, is the problem — and `hermes update` is exactly the command they need but are least
    likely to trust after a failed update. Return the guidance so callers can print it alongside the
    raw error.
    """
    # A missing third-party dependency is a different problem (bad venv, missing
    # extra) with different remediation, so don't claim a partial update.
    if not isinstance(exc, ImportError) or isinstance(exc, ModuleNotFoundError):
        return []
    if not is_first_party_module(getattr(exc, "name", None)):
        return []
    return [
        "",
        "This looks like a partially-updated install: one module was refreshed "
        "and a related one was not.",
        "Re-run the update to bring the whole tree to the same version:",
        "    hermes update",
        "If that also fails, reinstall: https://hermes-agent.nousresearch.com",
    ]


def emit_partial_update_hint(exc: BaseException, *, file=None) -> bool:
    """Print recovery guidance for a half-updated tree."""
    lines = partial_update_hint(exc)
    if not lines:
        return False
    out = sys.stderr if file is None else file
    for line in (f"Error: {exc}", *lines):
        print(line, file=out)
    return True
