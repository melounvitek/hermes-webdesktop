"""Hermes Agent Uninstaller."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.colors import Colors, color

def _logger(mark: str, col: str):
    return lambda msg: print(f"{color(mark, col)} {msg}")


log_info = _logger("→", Colors.CYAN)
log_success = _logger("✓", Colors.GREEN)
log_warn = _logger("⚠", Colors.YELLOW)


def _print_box(middle: str, col: str) -> None:
    """Print the 3-line framed heading used by the uninstall screens."""
    print(color("┌─────────────────────────────────────────────────────────┐", col, Colors.BOLD))
    print(color(middle, col, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", col, Colors.BOLD))


def _prompt(text: str):
    """``input(text).strip().lower()``; None (after printing "Cancelled.") on Ctrl-C/EOF."""
    try:
        return input(text).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return None


def _cancelled() -> None:
    print()
    print("Uninstall cancelled.")


def _confirm_yes(text: str) -> bool:
    """Ask the user to type ``yes``; False (after the cancel line, unless Ctrl-C/EOF) otherwise."""
    confirm = _prompt(f"Type '{color('yes', Colors.YELLOW)}' {text}: ")
    if confirm == "yes":
        return True
    if confirm is not None:
        _cancelled()
    return False


def _remove_each(candidates, remove) -> list:
    """Run ``remove(path)`` for each candidate; collect those it reports removed (truthy).

    Failures are downgraded to the shared ``Could not remove <path>: <err>`` warning.
    """
    removed = []
    for path in candidates:
        try:
            if remove(path):
                removed.append(path)
        except Exception as e:
            log_warn(f"Could not remove {path}: {e}")
    return removed


def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


_SHELL_RC_NAMES = (".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile")


def _strip_hermes_path_lines(content: str) -> str:
    """Drop the ``# Hermes Agent`` marker (+ its PATH line) and any hermes PATH line; squash blank runs."""
    new_lines = []
    skip_next = False
    for line in content.split('\n'):
        if '# Hermes Agent' in line or '# hermes-agent' in line:
            skip_next = True
            continue
        if skip_next and ('hermes' in line.lower() and 'PATH' in line):
            skip_next = False
            continue
        skip_next = False
        if 'hermes' in line.lower() and ('PATH=' in line or 'path=' in line.lower()):
            continue
        new_lines.append(line)
    new_content = '\n'.join(new_lines)
    while '\n\n\n' in new_content:
        new_content = new_content.replace('\n\n\n', '\n\n')
    return new_content


def remove_path_from_shell_configs():
    """Remove Hermes PATH entries from shell configuration files."""
    home = Path.home()
    removed_from = []
    for config_path in (c for c in (home / n for n in _SHELL_RC_NAMES) if c.exists()):
        try:
            content = config_path.read_text(encoding="utf-8")
            new_content = _strip_hermes_path_lines(content)
            if new_content != content:
                from utils import atomic_write_text

                # This is the user's own shell rc, not a Hermes-owned file, and nothing here backs
                # it up. A bare write_text() truncates it before the new content lands, so a crash
                # or SIGINT mid-write leaves an empty/truncated ~/.zshrc -- and the enclosing
                # `except Exception` downgrades that to a warning, so the next login just starts a
                # bare shell. atomic_replace also resolves a symlinked rc file (dotfiles repos keep
                # the symlink). preserve_mode keeps the rc's permission bits (normally 0644) and
                # owner (sudo-run uninstalls) instead of mkstemp's 0600/root.
                atomic_write_text(config_path, new_content, preserve_mode=True)
                removed_from.append(config_path)
        except Exception as e:
            log_warn(f"Could not update {config_path}: {e}")
    return removed_from


def remove_wrapper_script():
    """Remove the hermes wrapper script if it exists."""
    def _unlink_ours(wrapper: Path) -> bool:
        # Check if it's our wrapper (contains hermes_cli reference)
        content = wrapper.read_text(encoding="utf-8")
        if 'hermes_cli' in content or 'hermes-agent' in content:
            wrapper.unlink()
            return True
        return False

    candidates = (
        bin_dir / name
        for bin_dir in (Path.home() / ".local" / "bin", Path("/usr/local/bin"))
        for name in ("hermes", "hermes-acp", "hermes-agent")
    )
    return _remove_each((w for w in candidates if w.exists()), _unlink_ours)


def _node_symlink_candidate_dirs() -> "list[Path]":
    """Directories where the installer may have placed node/npm/npx symlinks."""
    dirs: list[Path] = [Path.home() / ".local" / "bin"]
    if sys.platform == "linux":  # root FHS installs put links in /usr/local/bin
        dirs.append(Path("/usr/local/bin"))
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:  # Termux installs put links in $PREFIX/bin
        dirs.append(Path(prefix) / "bin")
    return dirs


def remove_node_symlinks(hermes_home: Path) -> list:
    """Remove the node/npm/npx symlinks the installer placed on PATH.

    - ``/usr/local/bin/`` on root FHS installs (Linux, uid 0) - ``$PREFIX/bin/`` on Termux -
    ``~/.local/bin/`` otherwise (the common non-root case)

    We check all candidate directories so that uninstall works regardless of how the install was
    done (e.g. a root FHS install that placed links in ``/usr/local/bin``, or an older install that
    used ``~/.local/bin`` before the FHS fix).
    """
    node_dir = (hermes_home / "node").resolve()

    def _unlink_ours(link: Path) -> bool:
        # Only act on symlinks — never delete a real binary the user put here.
        if not link.is_symlink():
            return False
        # Resolve the link target and confirm it points into our node dir. os.readlink + manual
        # join handles broken (dangling) links too; Path.resolve() on a dangling link still
        # returns the target path.
        target = Path(os.readlink(link))
        if not target.is_absolute():
            target = (link.parent / target)
        target = target.resolve()
        if target == node_dir or node_dir in target.parents:
            link.unlink()
            return True
        return False

    candidates = (bin_dir / name for name in ("node", "npm", "npx") for bin_dir in _node_symlink_candidate_dirs())
    return _remove_each(candidates, _unlink_ours)


def uninstall_gateway_service():
    """Stop/uninstall the gateway service on every platform and kill standalone gateways.

    Delegates to the gateway module: systemd user+system units (Linux), launchd (macOS),
    Scheduled Task + Startup folder (Windows), plus standalone ``hermes gateway run`` processes.
    Termux/Android has no systemd, so only the process kill applies there.
    """
    import platform
    stopped_something = False

    # 1. Kill any standalone gateway processes (all platforms, including Termux)
    try:
        from hermes_cli.gateway import kill_gateway_processes, find_gateway_pids
        killed = kill_gateway_processes() if find_gateway_pids() else 0
        if killed:
            log_success(f"Killed {killed} running gateway process(es)")
            stopped_something = True
    except Exception as e:
        log_warn(f"Could not check for gateway processes: {e}")

    # Termux/Android has no systemd and no launchd — nothing left to do.
    if os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in os.getenv("PREFIX", ""):
        return stopped_something

    # 2. Per-platform service removal (systemd / launchd / Scheduled Task).
    step = _GATEWAY_SERVICE_REMOVERS.get(platform.system())
    if step is not None:
        remover, warn_label = step
        try:
            stopped_something = remover() or stopped_something
        except Exception as e:
            log_warn(f"{warn_label}: {e}")

    return stopped_something


def _remove_systemd_gateway() -> bool:
    """Linux: uninstall systemd services (both user and system scopes)."""
    from hermes_cli.gateway import (
        get_systemd_unit_path,
        get_service_name,
        _systemctl_cmd,
    )
    svc_name = get_service_name()
    removed_any = False

    for is_system, scope in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=is_system)
        if not unit_path.exists():
            continue
        try:
            if is_system and os.geteuid() != 0:  # windows-footgun: ok — Linux-only systemd path (dispatched on platform.system())
                log_warn(f"System gateway service exists at {unit_path} but needs sudo to remove")
                continue

            cmd = _systemctl_cmd(is_system)
            for verb in ("stop", "disable"):
                subprocess.run(cmd + [verb, svc_name], capture_output=True, check=False)
            unit_path.unlink()
            subprocess.run(cmd + ["daemon-reload"], capture_output=True, check=False)
            log_success(f"Removed {scope} gateway service ({unit_path})")
            removed_any = True
        except Exception as e:
            log_warn(f"Could not remove {scope} gateway service: {e}")
    return removed_any


def _remove_launchd_gateway() -> bool:
    """macOS: uninstall launchd plist."""
    from hermes_cli.gateway import get_launchd_plist_path
    plist_path = get_launchd_plist_path()
    if not plist_path.exists():
        return False
    subprocess.run(["launchctl", "unload", str(plist_path)],
                   capture_output=True, check=False)
    plist_path.unlink()
    log_success(f"Removed macOS gateway service ({plist_path})")
    return True


def _remove_windows_gateway() -> bool:
    """Windows: uninstall Scheduled Task + Startup-folder entry.

    The gateway_windows module already knows how to locate and remove both code paths (schtasks
    /Delete + .cmd unlink) and how to stop any running detached pythonw gateway process. We call
    into it so the uninstall logic stays in exactly one place.
    """
    from hermes_cli import gateway_windows
    if not any(probe() for probe in (
        gateway_windows.is_installed, gateway_windows.is_task_registered, gateway_windows.is_startup_entry_installed,
    )):
        return False
    try:
        gateway_windows.stop()
    except Exception as e:
        log_warn(f"Could not stop Windows gateway cleanly: {e}")
    try:
        gateway_windows.uninstall()
        log_success("Removed Windows gateway (Scheduled Task + Startup entry)")
        return True
    except Exception as e:
        log_warn(f"Could not fully uninstall Windows gateway: {e}")
        return False


# platform.system() -> (remover, warning label when the remover itself blows up)
_GATEWAY_SERVICE_REMOVERS = {
    "Linux": (_remove_systemd_gateway, "Could not check systemd gateway services"),
    "Darwin": (_remove_launchd_gateway, "Could not remove launchd gateway service"),
    "Windows": (_remove_windows_gateway, "Could not check Windows gateway service"),
}


# ============================================================================
# Windows-specific uninstall helpers
# ============================================================================
#
# The installer (``scripts/install.ps1``) does four Windows-only things that
# ``remove_path_from_shell_configs`` / ``remove_wrapper_script`` don't cover:
#
#   1. Sets User-scope env vars ``HERMES_HOME`` and ``HERMES_GIT_BASH_PATH``
#      via ``[Environment]::SetEnvironmentVariable(..., "User")``.  These
#      don't live in ~/.bashrc — they're in the Windows registry at
#      HKCU\Environment.
#   2. Prepends to User-scope ``PATH`` (same registry location) entries
#      like ``%LOCALAPPDATA%\hermes\git\cmd``, ``%LOCALAPPDATA%\hermes\git\bin``,
#      ``%LOCALAPPDATA%\hermes\git\usr\bin``, ``%LOCALAPPDATA%\hermes\node``.
#      Again not in any rc file — only accessible via the registry or the
#      .NET [Environment] API.
#   3. Downloads PortableGit to ``%LOCALAPPDATA%\hermes\git\`` and Node to
#      ``%LOCALAPPDATA%\hermes\node\`` as user-scoped, isolated copies.
#      These are ~200MB combined and serve no purpose after uninstall.
#   4. On the ``hermes dashboard`` + gateway paths, drops files into
#      ``%LOCALAPPDATA%\hermes\gateway-service\`` and sometimes
#      ``%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`` — the
#      latter is handled by ``gateway_windows.uninstall()`` already.
#
# Running a PowerShell one-liner per operation is overkill and fragile on
# locked-down machines (Constrained Language Mode, restricted ExecutionPolicy).
# Direct registry writes via ``winreg`` work without spawning any subprocess
# and apply immediately for new shells (SendMessage WM_SETTINGCHANGE would
# be nicer but requires ctypes and buys us nothing — the user will log out
# or open a new terminal anyway).


def _hermes_path_markers(hermes_home: Path, *, include_managed_bin: bool = False) -> list[str]:
    """Path-entry substrings that identify Hermes-owned User-PATH entries.

    ``include_managed_bin`` adds the managed binary dir (``<root>\bin``, holding the hermes
    launchers and the managed uv) — only wanted when that dir is about to be deleted (full uninstall
    from the default root), so a keep-data uninstall leaves the still-working managed uv resolvable.
    """
    root = str(hermes_home).rstrip("\\/")
    # Match on prefix so sub-entries (git\cmd, git\bin, git\usr\bin, node, etc.) all get swept.
    subs = ("hermes-agent", "git", "node", "venv") + (("bin",) if include_managed_bin else ())
    return [f"{root}\\{sub}" for sub in subs]


def remove_path_from_windows_registry(hermes_home: Path, *, include_managed_bin: bool = False) -> list[str]:
    """Strip Hermes-owned entries from User-scope PATH in the registry.

    ``include_managed_bin`` adds ``<hermes_home>\bin`` (the managed binary dir holding the hermes
    launchers and the managed uv) to the sweep. Only pass it when that dir is actually being deleted
    — full uninstall from the default root — so a keep-data uninstall leaves the still-working
    managed uv resolvable.
    """
    markers = tuple(m.lower() for m in _hermes_path_markers(hermes_home, include_managed_bin=include_managed_bin))

    def edit(winreg, key, removed):
        try:
            path_value, path_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return
        # Preserve REG_EXPAND_SZ vs REG_SZ so unexpanded %VARS% survive.
        kept: list[str] = []
        for entry in (e for e in path_value.split(";") if e):
            is_ours = entry.rstrip("\\/").lower().startswith(markers)
            (removed if is_ours else kept).append(entry)
        if removed:
            winreg.SetValueEx(key, "Path", 0, path_type, ";".join(kept))

    return _edit_user_environment(edit, warn_label="Could not edit User PATH in registry")


def remove_hermes_env_vars_windows() -> list[str]:
    """Delete HERMES_HOME and HERMES_GIT_BASH_PATH from User-scope env vars."""

    def edit(winreg, key, removed):
        for name in ("HERMES_HOME", "HERMES_GIT_BASH_PATH"):
            try:
                winreg.QueryValueEx(key, name)
            except FileNotFoundError:
                continue
            try:
                winreg.DeleteValue(key, name)
                removed.append(name)
            except OSError as e:
                log_warn(f"Could not delete {name} from User env: {e}")

    return _edit_user_environment(edit, warn_label="Could not open User Environment key")


def _edit_user_environment(edit, *, warn_label: str) -> list[str]:
    """Open HKCU\\Environment read/write and run ``edit(winreg, key, removed)``.

    Returns whatever ``edit`` appended to ``removed`` — even when a later registry call raised, so
    callers report exactly what was touched. ``[]`` off-Windows (no ``winreg``).
    """
    removed: list[str] = []
    try:
        import winreg
    except ImportError:
        return removed  # not on Windows, nothing to do
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as key:
            edit(winreg, key, removed)
    except OSError as e:
        log_warn(f"{warn_label}: {e}")
    return removed


def remove_portable_tooling_windows(hermes_home: Path) -> list[Path]:
    """Delete PortableGit and Node installs the Windows installer created under
    ``%LOCALAPPDATA%\\hermes\\``.  Only called on full uninstall; they're
    isolated from any system Git / Node so they cannot break other tools."""
    def _rmtree(target: Path) -> bool:
        shutil.rmtree(target, ignore_errors=False)
        return True

    targets = (hermes_home / sub for sub in ("git", "node", "gateway-service"))
    return _remove_each((t for t in targets if t.exists()), _rmtree)


def remove_windows_bin_launchers(*, windows: bool | None = None) -> list[Path]:
    """Delete the ``hermes`` launchers install.ps1 staged in the managed ``bin`` dir.

    Every uninstall mode deletes the checkout, so launchers pointing at ``<checkout>/venv``
    would dangle and error, which reads worse than command-not-found; the managed uv is kept for
    keep-data reinstalls. A launcher that is this process's own trampoline is locked against
    deletion but not rename, so it is renamed aside with a non-executable suffix instead.
    """
    if windows is None:
        windows = _is_windows()
    if not windows:
        return []
    try:
        # Lockstep launcher-name list — the same names install.ps1 and the
        # startup heal stage into this dir.
        from hermes_cli._install_repair import _WINDOWS_BIN_LAUNCHERS
        from hermes_constants import get_default_hermes_root

        bin_dir = get_default_hermes_root() / "bin"
    except Exception as e:
        log_warn(f"Could not locate the managed binary dir: {e}")
        return []

    def _unlink_or_rename_aside(launcher: Path) -> bool:
        try:
            launcher.unlink()
        except OSError:
            os.rename(launcher, launcher.with_name(f"{launcher.name}.uninstalled.{os.getpid()}"))
        return True

    candidates = (bin_dir / f"{name}{suffix}" for name in _WINDOWS_BIN_LAUNCHERS for suffix in (".exe", ".cmd"))
    return _remove_each((p for p in candidates if p.exists()), _unlink_or_rename_aside)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_default_hermes_home(hermes_home: Path) -> bool:
    """Return True when ``hermes_home`` points at the default (non-profile) root."""
    try:
        from hermes_constants import get_default_hermes_root
        return hermes_home.resolve() == get_default_hermes_root().resolve()
    except Exception:
        return False


def _discover_named_profiles():
    """Return a list of ``ProfileInfo`` for every non-default profile, or ``[]`` if profile support is
    unavailable or nothing is installed beyond the default root.
    """
    try:
        from hermes_cli.profiles import list_profiles
    except Exception:
        return []
    try:
        return [p for p in list_profiles() if not getattr(p, "is_default", False)]
    except Exception as e:
        log_warn(f"Could not enumerate profiles: {e}")
        return []


def _uninstall_profile(profile) -> None:
    """Fully uninstall a named profile: stop its gateway, remove its alias, wipe its home.

    Shells out to ``hermes -p <name> gateway stop|uninstall`` because service names and unit/
    plist paths derive from the current HERMES_HOME and can't easily be switched in-process.
    """
    name = profile.name
    log_info(f"Uninstalling profile '{name}'...")

    # 1. Stop and remove this profile's gateway service.
    #    Use `python -m hermes_cli.main` so we don't depend on a `hermes`
    #    wrapper that may be half-removed mid-uninstall.
    hermes_invocation = [sys.executable, "-m", "hermes_cli.main", "--profile", name]
    for subcmd in ("stop", "uninstall"):
        try:
            subprocess.run(
                hermes_invocation + ["gateway", subcmd],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_warn(f"  Gateway {subcmd} timed out for '{name}'")
        except Exception as e:
            log_warn(f"  Could not run gateway {subcmd} for '{name}': {e}")

    # 2. Remove the wrapper alias script at ~/.local/bin/<name> (if any).
    alias_path = getattr(profile, "alias_path", None)
    if alias_path and alias_path.exists():
        try:
            alias_path.unlink()
            log_success(f"  Removed alias {alias_path}")
        except Exception as e:
            log_warn(f"  Could not remove alias {alias_path}: {e}")

    # 3. Wipe the profile's HERMES_HOME directory.
    _rmtree_step(profile.path, indent="  ", fully=False)


def run_gui_uninstall(args):
    """GUI-only uninstall: remove the Chat GUI, leave the agent + data intact.

    Mirrors ``hermes uninstall --gui``. Removes the desktop app's built artifacts, the packaged app
    bundle (best-effort), and the Electron userData dir — nothing under ``$HERMES_HOME``
    config/sessions/.env, and never the Python agent or its venv.
    """
    from hermes_cli.gui_uninstall import (
        agent_is_installed,
        gui_install_summary,
        uninstall_gui,
    )

    hermes_home = get_hermes_home()
    summary = gui_install_summary(hermes_home)
    skip_confirm = bool(getattr(args, "yes", False))

    print()
    _print_box("│         ⚕ Hermes Chat GUI Uninstaller                  │", Colors.MAGENTA)
    print()

    if not summary["gui_installed"]:
        print("No Hermes Chat GUI installation was found.")
        print(f"  Checked: {hermes_home}, and the standard app locations for this OS.")
        return

    print(color("This removes the Chat GUI only. The Hermes agent stays installed.", Colors.CYAN))
    print()
    print(color("Will remove:", Colors.YELLOW, Colors.BOLD))
    for p in (*summary["source_built_artifacts"], *summary["packaged_app_paths"]):
        print(f"  • {p}")
    if summary["userdata_exists"]:
        print(f"  • {summary['userdata_dir']}  (desktop app data)")
    print()
    if agent_is_installed(hermes_home):
        print(color("Kept intact:", Colors.GREEN, Colors.BOLD))
        print(f"  • The Hermes agent at {hermes_home / 'hermes-agent'}")
        print(f"  • Your config, sessions, and secrets under {hermes_home}")
        print()

    if not skip_confirm and not _confirm_yes("to remove the Chat GUI"):
        return

    print()
    print(color("Uninstalling Chat GUI...", Colors.CYAN, Colors.BOLD))
    print()
    uninstall_gui(hermes_home)

    print()
    _print_box("│            ✓ Chat GUI Uninstalled!                      │", Colors.GREEN)
    print()
    print("The Hermes agent is still installed. Run 'hermes' to use the CLI,")
    print("or 'hermes uninstall' to remove the agent too.")
    print()


def run_uninstall(args):
    """Run the uninstall process.

    Full uninstall removes code and ``~/.hermes/``; keep-data removes only the code so the
    configs, data and logs survive a reinstall.
    """
    project_root = get_project_root()
    hermes_home = get_hermes_home()

    if bool(getattr(args, "dry_run", False)):
        _print_uninstall_dry_run(
            project_root=project_root,
            hermes_home=hermes_home,
            full_uninstall=bool(getattr(args, "full", False)),
        )
        return

    # Detect named profiles when uninstalling from the default root —
    # offer to clean them up too instead of leaving zombie HERMES_HOMEs
    # and systemd units behind.
    is_default_profile = _is_default_hermes_home(hermes_home)
    named_profiles = _discover_named_profiles() if is_default_profile else []

    # Non-interactive fast path (``--yes``): no prompts. ``--full`` selects a
    # full wipe (code + ~/.hermes data); otherwise keep-data. Named profiles
    # are NOT auto-removed here — that's a destructive, surprising default for
    # an unattended run, so it stays opt-in to the interactive flow. This is
    # the path the desktop app's detached cleanup script uses for its
    # lite/full modes.
    if bool(getattr(args, "yes", False)):
        _perform_uninstall(
            project_root=project_root,
            hermes_home=hermes_home,
            full_uninstall=bool(getattr(args, "full", False)),
            remove_profiles=False,
            named_profiles=named_profiles,
        )
        return

    print()
    _print_box("│            ⚕ Hermes Agent Uninstaller                  │", Colors.MAGENTA)
    print()
    
    # Show what will be affected
    print(color("Current Installation:", Colors.CYAN, Colors.BOLD))
    print(f"  Code:    {project_root}")
    print(f"  Config:  {hermes_home / 'config.yaml'}")
    print(f"  Secrets: {hermes_home / '.env'}")
    print(f"  Data:    {hermes_home / 'cron/'}, {hermes_home / 'sessions/'}, {hermes_home / 'logs/'}")
    print()

    if named_profiles:
        print(color("Other profiles detected:", Colors.CYAN, Colors.BOLD))
        for p in named_profiles:
            print(f"  • {p.name}{' (gateway running)' if getattr(p, 'gateway_running', False) else ''}: {p.path}")
        print()

    # Ask for confirmation
    print(color("Uninstall Options:", Colors.YELLOW, Colors.BOLD))
    print()
    print("  1) " + color("Keep data", Colors.GREEN) + " - Remove code only, keep configs/sessions/logs")
    print("     (Recommended - you can reinstall later with your settings intact)")
    print()
    print("  2) " + color("Full uninstall", Colors.RED) + " - Remove everything including all data")
    print("     (Warning: This deletes all configs, sessions, and logs permanently)")
    print()
    print("  3) " + color("Cancel", Colors.CYAN) + " - Don't uninstall")
    print()
    
    choice = _prompt(color("Select option [1/2/3]: ", Colors.BOLD))
    if choice is None:
        return
    
    if choice in {"3", "c", "cancel", "q", "quit", "n", "no"}:
        _cancelled()
        return
    
    full_uninstall = (choice == "2")

    # When doing a full uninstall from the default profile, also offer to
    # remove any named profiles — stopping their gateway services, unlinking
    # their alias wrappers, and wiping their HERMES_HOME dirs. Otherwise
    # those leave zombie services and data behind.
    remove_profiles = False
    n_profiles = len(named_profiles)
    profile_names = ", ".join(p.name for p in named_profiles)
    if full_uninstall and named_profiles:
        print()
        print(color("Other profiles will NOT be removed by default.", Colors.YELLOW))
        print(f"Found {n_profiles} named profile(s): {profile_names}")
        print()
        resp = _prompt(color(f"Also stop and remove these {n_profiles} profile(s)? [y/N]: ", Colors.BOLD))
        if resp is None:
            return
        remove_profiles = resp in {"y", "yes"}

    # Final confirmation
    print()
    if full_uninstall:
        print(color("⚠️  WARNING: This will permanently delete ALL Hermes data!", Colors.RED, Colors.BOLD))
        print(color("   Including: configs, API keys, sessions, scheduled jobs, logs", Colors.RED))
        if remove_profiles:
            print(color(f"   Plus {n_profiles} profile(s): {profile_names}", Colors.RED))
    else:
        print("This will remove the Hermes code but keep your configuration and data.")
    
    print()
    if not _confirm_yes("to confirm"):
        return

    _perform_uninstall(
        project_root=project_root,
        hermes_home=hermes_home,
        full_uninstall=full_uninstall,
        remove_profiles=remove_profiles,
        named_profiles=named_profiles,
    )


def _print_uninstall_dry_run(*, project_root: Path, hermes_home: Path, full_uninstall: bool) -> None:
    """Print the uninstall plan without stopping services or deleting files."""
    print()
    print(color("Dry run: no files, services, or environment entries will be changed.", Colors.CYAN, Colors.BOLD))
    print()
    print(color("Would inspect/remove:", Colors.YELLOW, Colors.BOLD))
    print("  • Gateway services and standalone gateway processes")
    print("  • Hermes PATH entries from shell configs / Windows User PATH")
    print("  • Hermes wrapper scripts and Hermes-managed node/npm/npx symlinks")
    print("  • Desktop Chat GUI artifacts")
    print(f"  • Code checkout: {project_root}")
    if not full_uninstall:
        print(f"  • Keep Hermes config/data: {hermes_home}")
    else:
        print(f"  • Hermes config/data: {hermes_home}")
        profiles = _discover_named_profiles() if _is_default_hermes_home(hermes_home) else []
        if profiles:
            print("  • Named profiles (interactive uninstall asks before removing):")
            for prof in profiles:
                print(f"    - {prof.name}: {prof.path}")
    print()


def _remove_step(label: str, remove, success_fmt: str, none_msg: str) -> None:
    """Announce ``label``, run ``remove()``, then log one success line per removed item (or ``none_msg``)."""
    log_info(label)
    removed = remove()
    if removed:
        for item in removed:
            log_success(success_fmt.format(item))
    else:
        log_info(none_msg)


def _rmtree_step(path: Path, *, indent: str = "", fully: bool = True) -> None:
    """Best-effort ``rmtree`` with the shared success/warning lines."""
    try:
        if path.exists():
            shutil.rmtree(path)
            log_success(f"{indent}Removed {path}")
    except Exception as e:
        log_warn(f"{indent}Could not {'fully ' if fully else ''}remove {path}: {e}")
        if fully:
            log_info("You may need to manually remove it")


def _perform_uninstall(
    *,
    project_root: Path,
    hermes_home: Path,
    full_uninstall: bool,
    remove_profiles: bool,
    named_profiles: list,
) -> None:
    """Execute the uninstall steps; shared by the interactive and ``--yes`` paths.

    Order: stop gateway -> strip PATH (rc files + Windows registry) -> remove wrapper and node
    symlinks -> remove desktop Chat GUI artifacts -> delete the checkout -> (Windows) remove
    PortableGit/Node -> optionally wipe ``$HERMES_HOME`` and named profiles.
    """
    print()
    print(color("Uninstalling...", Colors.CYAN, Colors.BOLD))
    print()
    
    # 1. Stop and uninstall gateway service + kill standalone processes
    log_info("Checking for running gateway...")
    if not uninstall_gateway_service():
        log_info("No gateway service or processes found")
    
    # 2-3b. PATH entries (POSIX rc files, then the Windows User-scope registry), wrapper script,
    #    Windows launchers, node/npm/npx symlinks. Windows notes: hermes_home is %VAR%-expanded so
    #    marker matching runs against fully resolved paths (install.ps1 writes literal
    #    C:\Users\<u>\AppData\Local\hermes\git\cmd, not %LOCALAPPDATA%); the managed binary dir
    #    (hermes\bin: launchers + managed uv) leaves the PATH only when the full wipe below is
    #    about to delete it, so keep-data mode keeps the still-working uv resolvable; the launchers
    #    themselves always go — both modes delete the checkout, so a surviving launcher would
    #    dangle and error, worse than command-not-found. Symlinks are removed only when they still
    #    point into this Hermes home's node dir (never clobber nvm / user-managed Node).
    windows = _is_windows()
    sweep_managed_bin = windows and full_uninstall and _is_default_hermes_home(hermes_home)
    for on_this_platform, label, remove, success_fmt, none_msg in (
        (True, "Removing PATH entries from shell configs...",
         remove_path_from_shell_configs, "Updated {}", "No PATH entries found to remove in shell rc files"),
        (windows, "Removing PATH entries from Windows User environment...",
         lambda: remove_path_from_windows_registry(
             Path(os.path.expandvars(str(hermes_home))), include_managed_bin=sweep_managed_bin),
         "Removed from User PATH: {}", "No Hermes-owned PATH entries in User environment"),
        (windows, "Removing HERMES_HOME / HERMES_GIT_BASH_PATH User env vars...",
         remove_hermes_env_vars_windows, "Removed User env var: {}", "No Hermes-set User env vars to remove"),
        (True, "Removing hermes command...", remove_wrapper_script, "Removed {}", "No wrapper script found"),
        (windows, "Removing Windows hermes launchers...",
         remove_windows_bin_launchers, "Removed {}", "No Windows hermes launchers found"),
        (True, "Removing Hermes-managed node/npm/npx symlinks...",
         lambda: remove_node_symlinks(hermes_home), "Removed {}", "No Hermes-managed node/npm/npx symlinks found"),
    ):
        if on_this_platform:
            _remove_step(label, remove, success_fmt, none_msg)

    # 3c. Desktop Chat GUI artifacts (built renderer/release, node_modules, packaged app bundle,
    #     Electron userData). Both CLI flows remove the agent code, so the GUI — just another
    #     consumer of the same checkout — goes with it. uninstall_gui() never touches config /
    #     sessions / .env, so it's safe in keep-data mode; the packaged app + Electron userData
    #     live OUTSIDE HERMES_HOME, so the full-uninstall rmtree below would not reach them.
    log_info("Removing desktop Chat GUI artifacts...")
    try:
        from hermes_cli.gui_uninstall import uninstall_gui
        if not uninstall_gui(hermes_home):
            log_info("No desktop GUI artifacts found")
    except Exception as e:
        log_warn(f"Could not remove desktop GUI artifacts: {e}")

    # 4. Remove installation directory (code) — we may be running from inside it.
    log_info("Removing installation directory...")
    _rmtree_step(project_root)

    # 4b. Windows-only installer artifacts that are NOT user data (PortableGit, bundled Node,
    #     gateway-service dir): install tooling under HERMES_HOME, safe to remove in keep-data mode.
    if windows:
        _remove_step(
            "Removing Windows installer artifacts (PortableGit, Node, gateway-service)...",
            lambda: remove_portable_tooling_windows(hermes_home), "Removed {}",
            "No Windows installer artifacts to remove",
        )
    
    # 5. Optionally remove ~/.hermes/ data directory (and named profiles)
    if full_uninstall:
        # 5a. Stop and remove each named profile's gateway service and alias wrapper. The profile
        #     HERMES_HOME dirs live under ``<default>/profiles/<name>/`` and are swept by the rmtree
        #     below, but services + alias scripts live OUTSIDE the default root.
        if remove_profiles:
            for prof in named_profiles:
                _uninstall_profile(prof)

        log_info("Removing configuration and data...")
        _rmtree_step(hermes_home)
    else:
        log_info(f"Keeping configuration and data in {hermes_home}")

    print()
    _print_box("│              ✓ Uninstall Complete!                      │", Colors.GREEN)
    print()

    if not full_uninstall:
        print(color("Your configuration and data have been preserved:", Colors.CYAN))
        print(f"  {hermes_home}/")
        print()
        print("To reinstall later with your existing settings:")
        if windows:
            print(color("  iex (irm https://hermes-agent.nousresearch.com/install.ps1)", Colors.DIM))
        else:
            print(color("  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash", Colors.DIM))
        print()

    if windows:
        print(color("Open a new terminal (PowerShell / Windows Terminal) to pick up", Colors.YELLOW))
        print(color("the updated User PATH and environment variables.", Colors.YELLOW))
    else:
        print(color("Reload your shell to complete the process:", Colors.YELLOW))
        print("  source ~/.bashrc  # or ~/.zshrc")
    print()
    print("Thank you for using Hermes Agent! ⚕")
    print()


class _UninstallArgs:
    """Lightweight args namespace for the module entrypoint below."""

    def __init__(self, *, mode: str):
        self.gui = mode == "gui"
        self.gui_summary = False
        self.full = mode == "full"
        self.yes = True  # the module entrypoint is always non-interactive


def main(argv=None) -> int:
    """Module entrypoint: ``python -m hermes_cli.uninstall --mode <gui|lite|full>``.

    This module imports only stdlib + ``hermes_constants`` + ``hermes_cli.colors`` (and lazily
    ``hermes_cli.gui_uninstall``), so it runs fine under a bare system Python with no site-packages
    from the venv.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hermes_cli.uninstall")
    parser.add_argument(
        "--mode",
        choices=["gui", "lite", "full"],
        required=True,
        help="gui = Chat GUI only; lite = GUI + agent, keep data; full = everything",
    )
    args = _UninstallArgs(mode=parser.parse_args(argv).mode)
    (run_gui_uninstall if args.gui else run_uninstall)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
