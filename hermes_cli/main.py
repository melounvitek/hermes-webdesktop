#!/usr/bin/env python3
"""
Hermes CLI - Main entry point.

Usage:
    hermes                     # Interactive chat (default)
    hermes chat                # Interactive chat
    hermes gateway             # Run gateway in foreground
    hermes gateway start       # Start gateway as service
    hermes gateway stop        # Stop gateway service
    hermes gateway status      # Show gateway status
    hermes gateway install     # Install gateway service
    hermes gateway uninstall   # Uninstall gateway service
    hermes setup               # Interactive setup wizard
    hermes logout              # Clear stored authentication
    hermes status              # Show status of all components
    hermes cron                # Manage cron jobs
    hermes cron list           # List cron jobs
    hermes cron status         # Check if cron scheduler is running
    hermes doctor              # Check configuration and dependencies
    hermes honcho setup                    # Configure Honcho AI memory integration
    hermes honcho status                   # Show Honcho config and connection status
    hermes honcho sessions                 # List directory → session name mappings
    hermes honcho map <name>               # Map current directory to a session name
    hermes honcho peer                     # Show peer names and dialectic settings
    hermes honcho peer --user NAME         # Set user peer name
    hermes honcho peer --ai NAME           # Set AI peer name
    hermes honcho peer --reasoning LEVEL   # Set dialectic reasoning level
    hermes honcho mode                     # Show current memory mode
    hermes honcho mode [hybrid|honcho|local]  # Set memory mode
    hermes honcho tokens                   # Show token budget settings
    hermes honcho tokens --context N       # Set session.context() token cap
    hermes honcho tokens --dialectic N     # Set dialectic result char cap
    hermes honcho identity                 # Show AI peer identity representation
    hermes honcho identity <file>          # Seed AI peer identity from a file (SOUL.md etc.)
    hermes honcho migrate                  # Step-by-step migration guide: OpenClaw native → Hermes + Honcho
    hermes --version           Show version and update status
    hermes update              Update to latest version
    hermes uninstall           Uninstall Hermes Agent
    hermes acp                 Run as an ACP server for editor integration
    hermes sessions browse     Interactive session picker with search

    hermes claw migrate --dry-run  # Preview migration without changes
"""

# IMPORTANT: hermes_bootstrap must be the very first import — it sets up
# UTF-8 stdio on Windows so print()/subprocess children don't hit
# UnicodeEncodeError with non-ASCII characters.  No-op on POSIX.
#
# Guarded against ModuleNotFoundError because ``hermes_bootstrap`` is a
# top-level module registered via pyproject.toml's ``py-modules`` list.
# When the user upgrades code via ``git pull`` (or ``hermes update``
# crashes between ``git reset --hard`` and ``uv pip install -e .``), the
# new code references ``hermes_bootstrap`` but the editable install's
# ``.pth`` file still points at the old set of top-level modules.  Without
# this guard, hermes crashes on import and the user can't run
# ``hermes update`` to recover.  Missing the bootstrap means UTF-8 stdio
# setup is skipped on Windows — degraded, not broken.  POSIX is unaffected.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

# Windows: neutralize CPython's ``platform._syscmd_ver`` before anything else
# imports — it shells out ``cmd /c ver`` (shell=True, no CREATE_NO_WINDOW), so
# any dependency touching ``platform.uname()`` at import time flashes a
# visible console when this process is windowless (pythonw gateway + every
# kanban worker).  No-op on POSIX; never raises.
from hermes_cli._subprocess_compat import suppress_platform_ver_console
from hermes_cli.cli_output import line_input

suppress_platform_ver_console()

import os
import sys

# ── Startup fast-path bootstrap ─────────────────────────────────────────
# Two lines of inline path math so ``python hermes_cli/main.py`` (script
# mode — sys.path[0] is hermes_cli/, not the repo root) can import the
# canonical helpers; everything else lives in hermes_cli._startup_fast.
_bootstrap_root = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)
from hermes_cli import _startup_fast  # noqa: E402

# Early venv self-heal — MUST run before any third-party import below.  When
# a prior ``hermes update`` left a recovery marker and a core package's import
# files were wiped (#57828 — failed lazy backend refresh), the module-level
# ``from hermes_cli.env_loader import ...`` / ``from hermes_cli.config import
# ...`` imports further down would crash before ``main()`` ever reaches
# ``_recover_from_interrupted_install()``.  ``_early_recovery`` is stdlib-only
# (safe to import on a corrupted venv), repairs just enough for this module to
# finish importing, and leaves the marker lifecycle to the full recovery path.
# The module import itself is unguarded on purpose: it lives in this same
# package directory, so if IT can't import, nothing else in hermes_cli can
# either. It is also the canonical home of the probe/repair tables reused by
# the full recovery path below.
from hermes_cli import _early_recovery as _early_recovery_mod

try:
    _early_recovery_mod.recover_if_needed()
except Exception:
    pass


# Startup-liveness watchdog (OOF-298): for gateway runs, arm BEFORE the heavy
# module-level import graph below — an import-time deadlock (native-extension
# init, contended import lock) is exactly the "wedged before the event loop,
# no logs, live PID" class this watchdog exists for. ``hermes_startup_watchdog``
# is stdlib-only, so importing it here cannot itself wedge on application
# code. The match requires the ADJACENT token pair ``gateway run`` (the
# subcommand shape, wherever global flags like ``-p <profile>`` put it) so
# unrelated commands that merely mention both words in different arguments
# never arm a 300s hard-exit timer, while flag-carrying invocations still
# do — under-arming recreates OOF-298. Foreground `hermes gateway run`
# still arms — a pre-loop wedge is just as dead without a supervisor, and
# the stack dump plus exit beats a silent hang; GatewayRunner disarms once
# the event loop is confirmed live.
def _argv_is_gateway_run(argv: list) -> bool:
    return any(
        a == "gateway" and b == "run" for a, b in zip(argv, argv[1:])
    )


if _argv_is_gateway_run(sys.argv[1:]):
    try:
        from hermes_startup_watchdog import arm_startup_watchdog as _arm_sw

        _arm_sw()
        del _arm_sw
    except Exception:
        pass


def _exit_after_oneshot(rc: object) -> None:
    """Exit one-shot mode without letting late native finalizers change rc.

    The SIGABRT this guards against (#30387, #43055) fires in a
    native-extension finalizer during CPython's ``Py_FinalizeEx``, *after*
    the response has printed. Flush streams, shut down file logging, then
    ``os._exit`` past interpreter finalization. The ``atexit`` chain is
    deliberately skipped — several handlers re-enter native code that may
    be the abort source. Stateful cleanup is handled in ``_run_agent`` and
    ``_cleanup_oneshot_runtime``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    if rc is None:
        exit_code = 0
    elif isinstance(rc, int):
        exit_code = rc
    else:
        exit_code = 1
    os._exit(exit_code)


_oneshot_cleanup_done = False


def _cleanup_oneshot_runtime() -> None:
    """Best-effort process-global cleanup before one-shot hard exit.

    ``run_oneshot`` owns the agent-local cleanup (memory provider, agent.close,
    session_db.close — all in ``_run_agent``'s finally block). This mirrors the
    process-global pieces from ``cli.py:_run_cleanup()`` that would otherwise
    be skipped by ``os._exit``.
    """
    global _oneshot_cleanup_done
    if _oneshot_cleanup_done:
        return
    _oneshot_cleanup_done = True
    try:
        from tools.terminal_tool import cleanup_all_environments
        cleanup_all_environments()
    except Exception:
        pass
    try:
        from tools.async_delegation import interrupt_all
        interrupt_all(reason="oneshot shutdown")
    except Exception:
        pass
    try:
        from tools.browser_tool import _emergency_cleanup_all_sessions
        _emergency_cleanup_all_sessions()
    except Exception:
        pass
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except BaseException:
        pass
    try:
        from agent.auxiliary_client import shutdown_cached_clients
        shutdown_cached_clients()
    except Exception:
        pass


def _run_and_exit_oneshot(
    prompt: str,
    *,
    model: object = None,
    provider: object = None,
    toolsets: object = None,
    skills: object = None,
    usage_file: object = None,
) -> None:
    try:
        from hermes_cli.oneshot import run_oneshot

        rc = run_oneshot(
            prompt,
            model=model,
            provider=provider,
            toolsets=toolsets,
            skills=skills,
            usage_file=usage_file,
        )
    except KeyboardInterrupt:
        rc = 130
    except SystemExit as exc:
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
            rc = 1
        else:
            rc = exc.code
    except BaseException:
        # Defense-in-depth. ``run_oneshot`` already converts agent failures
        # into an int return code and only re-raises KeyboardInterrupt /
        # SystemExit (handled above). Anything still escaping here means
        # ``run_oneshot`` itself malfunctioned — surface it on stderr but never
        # fall through to normal interpreter teardown, which is the exact path
        # that aborts with SIGABRT on AL2023 (the bug this routine fixes).
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        rc = 1
    try:
        _cleanup_oneshot_runtime()
    finally:
        # The hard exit is the safety boundary for #43055. Even an interrupt
        # during best-effort cleanup must not fall back into interpreter
        # finalization, where the reported native SIGABRT occurs.
        _exit_after_oneshot(rc)


def _project_root_str_fast() -> str:
    return _startup_fast.project_root_str()


def _ensure_project_root_on_path_fast() -> None:
    _startup_fast.ensure_project_root_on_path()


def _set_process_title() -> None:
    """Set the process title to 'hermes' so tools like 'ps', 'top', and
    'htop' show the app name instead of 'python3.xx'.

    Purely cosmetic — non-fatal on any platform.

    Strategy (try in order):
      1. ``setproctitle`` (opt-in dep — installed via ``hermes tools`` or
         ``pip install setproctitle``, or bundled in a future release).
      2. ctypes ``prctl(PR_SET_NAME)`` (Linux only, 15-char limit).
      3. ctypes ``pthread_setname_np`` (macOS only, kernel thread name —
         changes lldb/top but not ``ps aux``).
      4. No-op on Windows (the .exe name is already ``hermes.exe``).
    """
    # Strategy 1: setproctitle (best — works on macOS, Linux, BSD)
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("hermes")
        return
    except ImportError:
        pass

    # Strategy 2/3: platform-specific ctypes fallback
    import ctypes
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"hermes", 0, 0, 0)  # PR_SET_NAME = 15
        elif system == "Darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            libc.pthread_setname_np(b"hermes")
        # Windows: the .exe name is already ``hermes.exe`` — nothing to do.
    except Exception:
        pass


# Cheap, dependency-free read of `display.interface` from config.yaml for the
# earliest hot-path decisions (mouse-residue suppression, Termux fast launch)
# that run *before* hermes_cli.config is importable. Mirrors the explicit
# precedence used everywhere else: `--cli` always wins, then `--tui`/env, then
# this config value. Cached so the multiple early callers don't re-parse YAML.
_EARLY_INTERFACE_CACHE: "list | None" = None


def _config_default_interface_early() -> str:
    """Return the configured default interface ("cli"/"tui") via a minimal
    YAML read. Best-effort: any error falls back to "cli" (legacy behavior)."""
    global _EARLY_INTERFACE_CACHE
    if _EARLY_INTERFACE_CACHE is not None:
        return _EARLY_INTERFACE_CACHE[0]
    value = "cli"
    try:
        home = os.environ.get("HERMES_HOME")
        if home:
            cfg_path = os.path.join(home, "config.yaml")
        else:
            cfg_path = os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")
        if os.path.exists(cfg_path):
            import yaml as _yaml_iface

            with open(cfg_path, encoding="utf-8") as _f:
                raw = _yaml_iface.load(
                    _f, Loader=getattr(_yaml_iface, "CSafeLoader", None) or _yaml_iface.SafeLoader
                ) or {}
            disp = raw.get("display", {})
            if isinstance(disp, dict):
                iface = disp.get("interface")
                if isinstance(iface, str) and iface.strip().lower() == "tui":
                    value = "tui"
    except Exception:
        value = "cli"  # best-effort — default to classic REPL on any error
    _EARLY_INTERFACE_CACHE = [value]
    return value


def _wants_tui_early(argv: "list[str] | None" = None) -> bool:
    """Earliest TUI decision, usable before argparse/config imports.

    Precedence: explicit ``--cli`` wins (forces classic REPL), then
    explicit ``--tui``/``HERMES_TUI=1``, then a real-TTY gate (a
    non-interactive stdio can't host the Ink UI, so ambient config never
    boots it there), then ``display.interface`` in config.

    The TTY gate is load-bearing for headless spawners — kanban workers,
    cron jobs, pipes run ``hermes … chat -q`` with stdio on a pipe. This
    is the earliest launch decision (it runs before ``cmd_chat`` /
    ``_resolve_use_tui``), so a ``display.interface: tui`` default used to
    boot the TUI here — whose no-TTY bail-out exits 0 without doing the
    task → "protocol violation" on every attempt. An explicit ``--tui``
    still reaches the informative bail-out.
    """
    if argv is None:
        argv = sys.argv[1:]
    if "--cli" in argv:
        return False
    if os.environ.get("HERMES_TUI") == "1" or "--tui" in argv:
        return True
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except Exception:
        return False
    return _config_default_interface_early() == "tui"


# Mouse-tracking residue suppression — runs BEFORE every other import on the
# TUI hot path so the terminal stops emitting SGR/X10 mouse reports while the
# Python launcher is still doing imports (≈100–300ms in cooked + echo mode,
# before the Node TUI takes stdin into raw mode). During that window any
# incoming bytes are echoed straight back to the user's shell scrollback as
# ``^[[<…M`` text. The TUI itself runs `resetTerminalModes()` again in
# `entry.tsx`; this is just the earlier cousin. ``HERMES_TUI_NO_EARLY_DISABLE``
# escapes the behaviour for diagnostics.
def _suppress_mouse_residue_early() -> None:
    if os.environ.get("HERMES_TUI_NO_EARLY_DISABLE") == "1":
        return
    if not _wants_tui_early():
        return
    try:
        # Skip when stdout is redirected (`hermes --tui … >log`, CI capture):
        # the bytes can't reach the terminal anyway and would just pollute
        # the log with raw CSI.
        if not os.isatty(1):
            return
        # Disable every mouse-tracking variant we know about. Idempotent and
        # safe to send even when no tracking is currently asserted.
        os.write(
            1,
            b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
            b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l",
        )
    except OSError:
        pass


_suppress_mouse_residue_early()


def _try_ultrafast_version() -> bool:
    """Handle ``hermes --version`` before config/logging imports."""
    return _startup_fast.try_fast_version()


_ensure_project_root_on_path_fast()

if _try_ultrafast_version():
    raise SystemExit(0)

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


from hermes_cli.subcommands.cron import build_cron_parser
from hermes_cli.subcommands.sync import build_sync_parser
from hermes_cli.subcommands.gateway import build_gateway_parser
from hermes_cli.subcommands.profile import build_profile_parser
from hermes_cli.subcommands.model import build_model_parser
from hermes_cli.subcommands.setup import build_setup_parser

from hermes_cli.subcommands.whatsapp import build_whatsapp_parser, build_whatsapp_cloud_parser
from hermes_cli.subcommands.slack import build_slack_parser
from hermes_cli.subcommands.login import build_login_parser
from hermes_cli.subcommands.logout import build_logout_parser
from hermes_cli.subcommands.auth import build_auth_parser
from hermes_cli.subcommands.status import build_status_parser
from hermes_cli.subcommands.pause import build_pause_parser
from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.subcommands.hooks import build_hooks_parser
from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.subcommands.verify import build_verify_parser
from hermes_cli.subcommands.security import build_security_parser
from hermes_cli.subcommands.approvals import build_approvals_parser
from hermes_cli.subcommands.dump import build_dump_parser
from hermes_cli.subcommands.debug import build_debug_parser
from hermes_cli.subcommands.backup import build_backup_parser
from hermes_cli.subcommands.import_cmd import build_import_cmd_parser
from hermes_cli.subcommands.import_agent import build_import_agent_parser
from hermes_cli.subcommands.config import build_config_parser
from hermes_cli.subcommands.skin import build_skin_parser
from hermes_cli.subcommands.console import build_console_parser
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.subcommands.uninstall import build_uninstall_parser
from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser
from hermes_cli.subcommands.gui import build_gui_parser
from hermes_cli.subcommands.logs import build_logs_parser
from hermes_cli.subcommands.prompt_size import build_prompt_size_parser
from hermes_cli.subcommands.memory import build_memory_parser
from hermes_cli.subcommands.acp import build_acp_parser
from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.subcommands.insights import build_insights_parser
from hermes_cli.subcommands.monitoring import build_monitoring_parser
from hermes_cli.subcommands.skills import build_skills_parser
from hermes_cli.subcommands.pairing import build_pairing_parser
from hermes_cli.subcommands.plugins import build_plugins_parser
from hermes_cli.subcommands.mcp import build_mcp_parser
from hermes_cli.subcommands.claw import build_claw_parser
from hermes_cli.subcommands.moa import build_moa_parser
from hermes_cli.subcommands.fallback import build_fallback_parser
from hermes_cli.subcommands.worktree import build_worktree_parser
from hermes_cli.subcommands.browser import build_browser_parser
from hermes_cli.subcommands.secrets import build_secrets_parser
from hermes_cli.subcommands.egress import build_egress_parser
from hermes_cli.subcommands.migrate import build_migrate_parser
from hermes_cli.subcommands.checkpoints import build_checkpoints_parser
from hermes_cli.subcommands.bundles import build_bundles_parser
from hermes_cli.subcommands.curator import build_curator_parser
from hermes_cli.subcommands.pets import build_pets_parser
from hermes_cli.subcommands.journey import build_journey_parser
from hermes_cli.subcommands.computer_use import build_computer_use_parser
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_cli.subcommands.completion import build_completion_parser


def _require_tty(command_name: str) -> None:
    """Exit with a clear error if stdin is not a terminal.

    Interactive TUI commands (hermes tools, hermes setup, hermes model) use
    curses or input() prompts that spin at 100% CPU when stdin is a pipe.
    This guard prevents accidental non-interactive invocation.
    """
    if not sys.stdin.isatty():
        print(
            f"Error: 'hermes {command_name}' requires an interactive terminal.\n"
            f"It cannot be run through a pipe or non-interactive subprocess.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


# Add project root to path
PROJECT_ROOT = Path(_project_root_str_fast())
_ensure_project_root_on_path_fast()


# ---------------------------------------------------------------------------
# Profile override — MUST happen before any hermes module import.
#
# Many modules cache HERMES_HOME at import time (module-level constants).
# We intercept --profile/-p from sys.argv here and set the env var so that
# every subsequent ``os.getenv("HERMES_HOME", ...)`` resolves correctly.
# The flag is stripped from sys.argv so argparse never sees it.
# Falls back to ~/.hermes/active_profile for sticky default.
# ---------------------------------------------------------------------------
def _apply_profile_override() -> None:
    """Pre-parse --profile/-p and set HERMES_HOME before imports."""
    argv = sys.argv[1:]
    profile_name = None
    consume = 0
    profile_index = None

    def _inside_mcp_add_args(index: int) -> bool:
        """True once argv reaches `hermes mcp add ... --args <command argv>`.

        ``mcp add --args`` is command-argv passthrough. Flags after that point
        belong to the child MCP command (for example Docker MCP Toolkit's
        ``--profile``), not to Hermes' own profile selector.
        """
        try:
            mcp_index = argv.index("mcp", 0, index)
            argv.index("add", mcp_index + 1, index)
        except ValueError:
            return False
        return True

    def _resolve_sudo_user_profile_env(name: str) -> str | None:
        """Resolve `sudo hermes -p <name>` against the invoking user's home.

        `_apply_profile_override()` runs before argparse, so `--run-as-user`
        is not available yet. For sudo invocations, the best available signal
        is SUDO_USER: root is only doing the privileged install/start action,
        while the profile store normally belongs to the user who invoked sudo.
        """
        if name == "default":
            return None
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return None
        sudo_user = os.environ.get("SUDO_USER", "").strip()
        if not sudo_user or sudo_user == "root":
            return None

        try:
            import pwd

            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        except Exception:
            return None

        candidate = home / ".hermes" / "profiles" / name
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            return None
        return None

    # 1. Check for explicit -p / --profile flag. Historically this worked even
    # after the subcommand (`hermes chat -p coder`), so keep scanning broadly.
    # The exception is command-argv passthrough regions such as `mcp add --args`.
    from hermes_cli._parser import top_level_value_flag_sets

    value_flags, optional_value_flags = top_level_value_flag_sets()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if arg == "--args" and _inside_mcp_add_args(i):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            profile_name = argv[i + 1]
            consume = 2
            profile_index = i
            break
        if arg.startswith("--profile="):
            profile_name = arg.split("=", 1)[1]
            consume = 1
            profile_index = i
            break
        if "=" not in arg and arg in value_flags and i + 1 < len(argv):
            i += 2
        elif (
            "=" not in arg
            and arg in optional_value_flags
            and i + 1 < len(argv)
            and not argv[i + 1].startswith("-")
        ):
            i += 2
        else:
            i += 1

    # 1b. Reject values that can't be valid profile names (e.g. pytest's
    # "-p no:xdist" would be misread as profile "no:xdist" otherwise).
    # Mirrors hermes_cli.profiles._PROFILE_ID_RE so we never call
    # resolve_profile_env() with a value it must reject + sys.exit on.
    if profile_name is not None and consume == 2:
        import re as _re

        if not _re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", profile_name):
            profile_name = None
            consume = 0
            profile_index = None

    # 1.5 If HERMES_HOME is already set and no explicit flag was given, trust it
    # only when it already points to a specific profile directory.  The
    # distinguishing heuristic: a profile path has "profiles" as its immediate
    # parent directory name (e.g. ~/.hermes/profiles/coder or
    # /opt/data/profiles/coder).  If HERMES_HOME points to the hermes root
    # instead (e.g. systemd hardcodes HERMES_HOME=/root/.hermes), we must
    # still read active_profile — the user may have switched profiles via
    # `hermes profile use` and the gateway should honour that choice.
    # See issue #22502.
    hermes_home_env = os.environ.get("HERMES_HOME", "")
    if profile_name is None and hermes_home_env:
        if Path(hermes_home_env).parent.name == "profiles":
            return

    # 2. If no flag, check active_profile in the hermes root.
    #
    # EXCEPTION: a supervisor-launched gateway child must NOT follow the
    # sticky active_profile. Each supervised slot has a fixed profile
    # identity: named slots pass ``-p <name>`` explicitly (handled in step 1
    # above) or pin ``HERMES_HOME`` to the profile directory (step 1.5), and
    # a bare invocation means "the root HERMES_HOME profile". If a supervised
    # default-profile child read active_profile here, switching the active
    # profile (e.g. via the dashboard or ``hermes profile use``) would
    # silently redirect the default gateway into that profile — the default
    # gateway then assumes the other profile's identity/credentials (logs
    # under the other profile's tree, connects with its Telegram bot token)
    # and double-polls a token already owned by that profile's own gateway.
    # See issue #74872 and the "Docker & Profiles & Dashboard" report.
    #
    # Supervisor markers honored (see gateway/restart.py
    # ``is_gateway_supervisor_process`` for the sibling detection used by
    # restart routing):
    #   - HERMES_SUPERVISED_CHILD: generalized marker exported by the
    #     generated systemd unit, launchd plist, and Windows Scheduled-Task
    #     launchers (#74872).
    #   - HERMES_S6_SUPERVISED_CHILD: legacy s6 container marker (back-compat;
    #     exported by S6ServiceManager's run-script).
    #   - INVOCATION_ID: set by systemd for service children only (never in
    #     interactive shells) — covers already-installed gateway units that
    #     predate the HERMES_SUPERVISED_CHILD marker. Consulted ONLY for
    #     gateway commands: INVOCATION_ID is inherited by every descendant of
    #     a systemd-launched process (self-hosted CI runners, user services
    #     running unrelated hermes commands), so honoring it globally would
    #     silently disable the sticky active_profile for those.
    #   - HERMES_GATEWAY_EXTERNAL_SUPERVISOR: explicit external-supervisor
    #     opt-in (``hermes gateway run --external-supervisor``).
    #
    # XPC_SERVICE_NAME is deliberately NOT consulted here: interactive macOS
    # terminals set it too, and a false positive would silently break the
    # sticky active_profile for every interactive command.
    def _under_gateway_supervisor() -> bool:
        if os.environ.get("HERMES_SUPERVISED_CHILD"):
            return True
        if os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
            return True
        is_gateway_cmd = next(
            (a for a in argv if not a.startswith("-")), None
        ) == "gateway"
        if is_gateway_cmd and os.environ.get("INVOCATION_ID"):
            return True
        return os.environ.get(
            "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    if profile_name is None and not _under_gateway_supervisor():
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8").strip()
                if name and name != "default":
                    profile_name = name
                    consume = 0  # don't strip anything from argv
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    # 3. If we found a profile, resolve and set HERMES_HOME
    if profile_name is not None:
        try:
            from hermes_cli.profiles import resolve_profile_env

            hermes_home = resolve_profile_env(profile_name)
        except FileNotFoundError as exc:
            hermes_home = _resolve_sudo_user_profile_env(profile_name)
            if not hermes_home:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            # A bug in profiles.py must NEVER prevent hermes from starting
            print(
                f"Warning: profile override failed ({exc}), using default",
                file=sys.stderr,
            )
            return
        os.environ["HERMES_HOME"] = hermes_home
        # Strip the flag from argv so argparse doesn't choke
        if consume > 0 and profile_index is not None:
            start = profile_index + 1  # +1 because argv is sys.argv[1:]
            sys.argv = sys.argv[:start] + sys.argv[start + consume :]


_apply_profile_override()

# Windows launcher self-heal — the ``hermes`` command users run is a COPY of
# the venv console script, staged into the managed binary dir (the default
# Hermes root's ``bin``, next to the managed uv) by install.ps1. That dir
# lives OUTSIDE the git checkout precisely because an earlier layout staged
# the copies at ``<checkout>\bin``, where ``hermes update``'s autostash
# (``git stash push --include-untracked``) swept them off disk; with the
# desktop updater's ``--keep-stash`` nothing restored them and ``hermes``
# stopped resolving in every new terminal (venv\Scripts itself must stay off
# PATH — it shadows the user's ``python``, #83797). Re-staging at process
# start reaches already-broken installs through the one channel that still
# works there: the desktop app spawning its backend via
# ``python -m hermes_cli.main``. Costs a few stat calls when healthy; gates
# fail toward inaction so source checkouts are untouched. Sits AFTER the
# profile override on purpose — no hermes module may be imported before
# profiles resolve. The launcher dir itself is per-machine (the helper
# anchors on the DEFAULT root, not HERMES_HOME), so profile sessions heal
# the same shared dir.
if sys.platform == "win32":
    try:
        from hermes_cli import _install_repair as _install_repair_mod

        _install_repair_mod.ensure_windows_bin_launchers(_bootstrap_root)
    except Exception:
        pass

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.config import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

# Updating dependencies must not import optional secret-manager libraries into
# the updater process before ``uv`` replaces the environment.  On Windows,
# Bitwarden's cryptography import maps ``_rust.pyd`` and the parent updater then
# prevents its own child installer from replacing that file (#73381).  Profile
# flags have already been stripped above, so the first remaining argument is
# the authoritative argparse subcommand.  Dotenv/managed config still loads;
# only external secret fetches are unnecessary for installation maintenance.
load_hermes_dotenv(
    project_env=PROJECT_ROOT / ".env",
    load_external_secrets=sys.argv[1:2] != ["update"],
)

# Bridge security.redact_secrets from config.yaml → HERMES_REDACT_SECRETS env
# var BEFORE hermes_logging imports agent.redact (which snapshots the flag at
# module-import time). Without this, config.yaml's toggle is ignored because
# the setup_logging() call below imports agent.redact, which reads the env var
# exactly once. Env var in .env still wins — this is config.yaml fallback only.
#
# We also read network.force_ipv4 from the same yaml load to avoid two
# separate config.yaml reads (saves ~17ms on every CLI startup — the second
# `load_config()` was doing a full deep-merge for one boolean lookup).
_FORCE_IPV4_EARLY = False
try:
    # Reuse read_raw_config()'s (mtime, size)-keyed cache instead of a bespoke
    # yaml.load — the SAME parse then serves hermes_logging's
    # _read_logging_config and any later raw reads in this process, collapsing
    # 3-4 config.yaml parses per invocation into one.
    from hermes_cli.config import read_raw_config as _read_raw_early

    _cfg_path = get_hermes_home() / "config.yaml"
    if _cfg_path.exists():
        _early_cfg_raw = _read_raw_early() or {}
        # Managed scope: overlay administrator-pinned values so a managed
        # security.redact_secrets / network.force_ipv4 wins here too. This early
        # bridge reads config.yaml directly (before load_config is usable), so
        # without the overlay a managed redact_secrets toggle would be ignored.
        # Fail-open via the shared helper.
        try:
            from hermes_cli import managed_scope
            _early_cfg_raw = managed_scope.apply_managed_overlay(_early_cfg_raw)
        except Exception:
            pass
        if "HERMES_REDACT_SECRETS" not in os.environ:
            _early_sec_cfg = _early_cfg_raw.get("security", {})
            if isinstance(_early_sec_cfg, dict):
                _early_redact = _early_sec_cfg.get("redact_secrets")
                if _early_redact is not None:
                    os.environ["HERMES_REDACT_SECRETS"] = str(_early_redact).lower()
        _early_net_cfg = _early_cfg_raw.get("network", {})
        if isinstance(_early_net_cfg, dict) and _early_net_cfg.get("force_ipv4"):
            _FORCE_IPV4_EARLY = True
        del _early_cfg_raw
    del _cfg_path
except Exception:
    pass  # best-effort — redaction stays at default (enabled) on config errors

# Initialize centralized file logging early — all `hermes` subcommands
# (chat, setup, gateway, config, etc.) write to agent.log + errors.log.
# Dashboard entrypoints bootstrap with GUI mode so gui.log is always present
# during GUI testing, including pre-dispatch startup failures.
try:
    from hermes_logging import setup_logging as _setup_logging

    _setup_logging(
        mode=(
            "gui"
            if next((arg for arg in sys.argv[1:] if not arg.startswith("-")), "")
            in {"dashboard", "serve", "gui", "desktop"}
            else "cli"
        )
    )
except Exception:
    pass  # best-effort — don't crash the CLI if logging setup fails

# Apply IPv4 preference early, before any HTTP clients are created.
# We already determined whether to force IPv4 from the raw yaml read above —
# this just calls the toggle without a redundant load_config() round trip.
if _FORCE_IPV4_EARLY:
    try:
        from hermes_constants import apply_ipv4_preference as _apply_ipv4

        _apply_ipv4(force=True)
    except Exception:
        pass  # best-effort — don't crash if hermes_constants not importable yet

import logging
import threading
import time as _time  # noqa: F401  (tests patch hermes_cli.main._time.sleep)
from datetime import datetime

from hermes_cli import __version__, __release_date__

# Provider model-selection wizard flows extracted to hermes_cli/model_setup_flows.py
# (god-file decomposition Phase 2). Re-imported here so select_provider_and_model and
# existing test monkeypatches (hermes_cli.main._model_flow_*) keep resolving unchanged.
from hermes_cli.model_setup_flows import (
    _prompt_auth_credentials_choice,
    _model_flow_openrouter,
    _model_flow_nous,
    _model_flow_openai_codex,
    _model_flow_xai_oauth,
    _model_flow_qwen_oauth,
    _model_flow_minimax_oauth,
    _model_flow_custom,
    _model_flow_azure_foundry,
    _model_flow_named_custom,
    _model_flow_copilot,
    _model_flow_copilot_acp,
    _model_flow_kimi,
    _model_flow_stepfun,
    _model_flow_bedrock_api_key,
    _model_flow_bedrock,
    _model_flow_vertex,
    _model_flow_api_key_provider,
    _model_flow_anthropic,
    _model_flow_moa,
    _model_flow_ai_gateway,
)
logger = logging.getLogger(__name__)
from hermes_cli.main_dashboard import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    _DASHBOARD_SYSTEMD_UNIT,
    _UpdateOutputStream,
    _dashboard_cmdline_for_pid,
    _dashboard_listening,
    _dashboard_probe_host,
    _extract_scope_from_cgroup,
    _finalize_update_output,
    _find_stale_dashboard_pids,
    _get_pid_cgroup_path,
    _get_systemd_service_for_pid,
    _install_hangup_protection,
    _is_electron_packaged_web_dist,
    _maybe_setup_dashboard_auth_interactively,
    _parse_dashboard_runtime,
    _read_ssh_session_token_file,
    _report_dashboard_status,
    _resolve_dashboard_web_dist,
    _respawn_dashboard_processes,
    _restart_managed_dashboard_service,
    _route_named_profile_dashboard,
    _try_restart_systemd_service,
)
from hermes_cli.main_provider_setup import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    _AUX_TASKS,
    _build_provider_picker_rows,
    _named_custom_provider_map,
    _DEFAULT_QWEN_PORTAL_MODELS,
    _DELEGATION_TASK_DESC,
    _DELEGATION_TASK_KEY,
    _DELEGATION_TASK_NAME,
    _GENERIC_API_KEY_PROVIDERS,
    _all_aux_tasks,
    _auto_provider_name,
    _aux_config_menu,
    _aux_flow_custom_endpoint,
    _aux_flow_provider_model,
    _aux_select_for_task,
    _aux_task_display_name,
    _clear_stale_openai_base_url,
    _custom_provider_api_key_config_value,
    _custom_provider_base_url_config_value,
    _delegation_cfg_as_task,
    _format_aux_current,
    _infer_stepfun_region,
    _is_profile_api_key_provider,
    _prompt_api_key,
    _prompt_custom_api_mode_selection,
    _prompt_provider_choice,
    _prompt_reasoning_effort_selection,
    _remove_custom_provider,
    _reset_aux_to_auto,
    _run_anthropic_oauth_flow,
    _save_aux_choice,
    _save_custom_provider,
    _stepfun_base_url_for_region,
)
from hermes_cli.main_install_repair import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    ShimQuarantineError,
    _LAZY_REFRESH_IMPORT_PROBES,
    _LAZY_REFRESH_REPAIR_PACKAGES,
    _PENDING_RENAME_KEY,
    _PENDING_RENAME_VALUE,
    _QUARANTINE_GRACE_SECONDS,
    _UPDATE_REEXEC_ENV,
    _cleanup_pending_shim_renames,
    _cleanup_quarantined_exes,
    _clear_lazy_refresh_incomplete_marker,
    _clear_marker_file,
    _clear_update_incomplete_marker,
    _default_venv_install_target,
    _detect_broken_lazy_refresh_imports,
    _filter_pending_shim_renames,
    _hermes_exe_shims,
    _insert_python_pin,
    _install_python_dependencies_with_optional_fallback,
    _interpreter_scripts_dir,
    _is_termux_env,
    _is_uv_command,
    _is_windows,
    _is_windows_npm_path,
    _lazy_refresh_marker_path,
    _lazy_refresh_repair_specs,
    _load_console_script_names,
    _load_installable_optional_extras,
    _norm_exe_path,
    _pytest_owns_live_checkout,
    _quarantine_running_hermes_exe,
    _quarantine_stamp_ms,
    _recover_core_update_marker_locked,
    _recover_from_interrupted_install,
    _recover_lazy_refresh_marker_locked,
    _reexec_dependency_sync_off_windows_shim,
    _repair_broken_lazy_refresh_imports,
    _repair_venv_via_import_probes,
    _resolve_install_target_python,
    _resolve_node_runtime_npm,
    _resolve_update_branch,
    _restore_quarantined_exes,
    _run_install_with_heartbeat,
    _run_package_only_install,
    _run_quarantined_install,
    _update_marker_path,
    _venv_scripts_dir,
    _verify_console_scripts_installed,
    _verify_core_dependencies_installed,
    _windows_running_hermes_launcher_locked,
    _windows_shim_in_process_chain,
)
from hermes_cli.main_desktop import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    _DESKTOP_PREVIOUS_SUFFIX,
    _DESKTOP_STAGING_PREFIX,
    _ELECTRON_FALLBACK_MIRROR,
    _HTML_TAG_WITH_URL,
    _LINUX_PASSWORD_STORES,
    _MACHINE_ATTRIBUTE_USER_ENABLED,
    _MODULE_TAG,
    _PE_MACHINE_AMD64,
    _PE_MACHINE_ARM64,
    _PE_MACHINE_I386,
    _PE_MACHINE_NAMES,
    _PE_MACHINE_TO_NAME,
    _build_desktop_app,
    _compute_desktop_content_hash,
    _desktop_backup_unpacked_dir,
    _desktop_build_needed,
    _desktop_dist_exists,
    _desktop_exe_integrity_error,
    _desktop_launch_options,
    _desktop_linux_needs_disable_setuid_sandbox,
    _desktop_linux_needs_no_sandbox,
    _desktop_linux_sandbox_fixup,
    _desktop_linux_sandbox_helper_is_regular_file,
    _desktop_linux_userns_sandbox_available,
    _desktop_macos_bundle_id,
    _desktop_macos_has_valid_real_signature,
    _desktop_macos_local_codesign,
    _desktop_macos_local_signing_identity,
    _desktop_macos_relaunchable_fixup,
    _desktop_macos_setup_tcc_identity,
    _desktop_packaged_executable,
    _desktop_packaged_executable_in,
    _desktop_staging_dir,
    _desktop_stamp_path,
    _desktop_unpacked_root,
    _detect_linux_password_store,
    _discard_desktop_staging,
    _electron_dir,
    _electron_dist_binary,
    _electron_dist_ok,
    _electron_download_cache_dirs,
    _electron_pkg_staged_missing_dist,
    _ensure_desktop_exe_launchable,
    _expected_windows_pe_machines,
    _force_adhoc_macos_signing,
    _macos_codesigning_identity_valid,
    _parse_pe_machine,
    _pe_machine_or_none,
    _purge_electron_build_cache,
    _redownload_electron_dist,
    _register_linux_desktop_entry,
    _renderer_bundle_dir,
    _renderer_bundle_torn,
    _rollback_desktop_from_backup,
    _stop_desktop_processes_locking_build,
    _swap_staged_desktop_app,
    _try_redownload_electron_dist,
    _windows_native_machine,
    _windows_native_machine_from_iswow64,
    _windows_user_runnable_pe_machines,
    _write_desktop_build_stamp,
    cmd_gui,
)
from hermes_cli.main_web_build import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    _BYTECODE_FINGERPRINT_FILE,
    _build_web_ui,
    _compute_web_ui_content_hash,
    _do_build_web_ui,
    _missing_web_build_tool,
    _nixos_build_env,
    _record_bytecode_fingerprint,
    _run_npm_install_deterministic,
    _run_npm_watching_for_engine_failure,
    _run_with_idle_timeout,
    _sweep_stale_bytecode_if_checkout_changed,
    _web_ui_build_needed,
    _web_ui_stamp_path,
    _write_web_ui_build_stamp,
)
from hermes_cli.main_tui_launch import (  # noqa: E402,F401  (re-exported; tests patch hermes_cli.main.<name>)
    _NPM_LOCK_RUNTIME_KEYS,
    _TUI_BUILD_INPUT_DIRS,
    _TUI_BUILD_INPUT_FILES,
    _TUI_BUILD_INPUT_SUFFIXES,
    _apply_tui_python_env,
    _ensure_tui_node,
    _ensure_tui_workspace,
    _find_bundled_tui,
    _iter_tui_build_inputs,
    _launch_tui,
    _make_tui_argv,
    _normalize_tui_toolsets,
    _npm_lifecycle_env,
    _npm_lock_workspace_closure,
    _pin_kanban_board_env,
    _print_tui_exit_summary,
    _read_cgroup_memory_limit,
    _read_tui_active_session_file,
    _resolve_tui_heap_mb,
    _resolve_use_tui,
    _restore_tui_workspace,
    _safe_tui_cwd,
    _sync_bundled_skills_quietly,
    _termux_workspace_install_context,
    _tui_need_npm_install,
    _tui_need_rebuild,
    _tui_selected_workspace_keys,
    _workspace_root,
)


def _is_termux_startup_environment(env: dict[str, str] | None = None) -> bool:
    """Import-safe Termux check for cold-start-sensitive CLI paths."""
    check = env or os.environ
    prefix = str(check.get("PREFIX", ""))
    return bool(
        check.get("TERMUX_VERSION")
        or "com.termux/files/usr" in prefix
        or prefix.startswith("/data/data/com.termux/")
    )


def _read_packed_ref(common_dir: Path, ref: str) -> str | None:
    """Look up a ref in .git/packed-refs without spawning git.

    packed-refs lines look like ``<sha> <ref>`` with optional ``^<sha>``
    peel lines and ``#``-prefixed comments / ``# pack-refs with:`` header.
    """
    try:
        text = (common_dir / "packed-refs").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def _read_git_revision_fingerprint(repo_root: Path) -> str | None:
    """Return a cheap checkout fingerprint without spawning git."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            for line in git_dir.read_text(encoding="utf-8", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "gitdir" and value.strip():
                    git_dir = (repo_root / value.strip()).resolve()
                    break
        # Worktrees point HEAD at a per-worktree gitdir but pack their refs
        # in the main repo's gitdir (referenced via ``commondir``). Resolve
        # that up front so packed-refs lookups hit the right file.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            try:
                rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                if rel:
                    common_dir = (git_dir / rel).resolve()
            except OSError:
                pass
        head_file = git_dir / "HEAD"
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            # Loose refs may live in the worktree gitdir OR the common dir
            # (branches created via `git worktree add` typically live in the
            # common dir's refs/heads/).
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    return f"git:{ref}:{ref_file.read_text(encoding='utf-8', errors='replace').strip()}"
            packed_sha = _read_packed_ref(common_dir, ref)
            if packed_sha:
                return f"git:{ref}:{packed_sha}"
            # Ref name is known but unresolved — still stable across launches,
            # and the version/release fallback in the caller will invalidate
            # after `hermes update`.
            return f"git:{ref}:unresolved"
        return f"git:HEAD:{head}"
    except OSError:
        return None


def _termux_bundled_skills_fingerprint() -> str:
    """Cheap invalidation key for Termux bundled-skill startup sync."""
    git_fp = _read_git_revision_fingerprint(PROJECT_ROOT)
    if git_fp:
        return git_fp
    skills_dir = PROJECT_ROOT / "skills"
    try:
        stat = skills_dir.stat()
        return f"skills:{__version__}:{__release_date__}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"skills:{__version__}:{__release_date__}:missing"


def _termux_bundled_skills_stamp_path() -> Path:
    return get_hermes_home() / "skills" / ".termux_bundled_sync_stamp"


def _termux_bundled_skills_sync_needed() -> bool:
    if not _is_termux_startup_environment():
        return True
    if os.environ.get("HERMES_TERMUX_FORCE_SKILLS_SYNC") == "1":
        return True
    try:
        stamp = _termux_bundled_skills_stamp_path()
        return stamp.read_text(encoding="utf-8").strip() != _termux_bundled_skills_fingerprint()
    except OSError:
        return True


def _mark_termux_bundled_skills_synced() -> None:
    if not _is_termux_startup_environment():
        return
    try:
        stamp = _termux_bundled_skills_stamp_path()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_termux_bundled_skills_fingerprint() + "\n", encoding="utf-8")
    except OSError:
        pass


def _sync_bundled_skills_for_startup() -> bool:
    """Sync bundled skills, but skip unchanged Termux checkouts cheaply.

    Hashing every bundled skill is safe but expensive on older Android
    storage. The git/ref stamp keeps post-update correctness: a changed
    checkout revision forces one real sync, then later starts skip it.
    """
    if _is_termux_startup_environment() and not _termux_bundled_skills_sync_needed():
        return False

    from tools.skills_sync import sync_skills

    sync_skills(quiet=True)
    _mark_termux_bundled_skills_synced()
    return True


def _termux_should_prefetch_update_check() -> bool:
    if not _is_termux_startup_environment():
        return True
    return os.environ.get("HERMES_TERMUX_PREFETCH_UPDATES") == "1"


def _has_any_provider_configured(*, strict_profile_scope: bool = False) -> bool:
    """Check if at least one inference provider is usable.

    ``strict_profile_scope``: the caller has bound a NAMED profile's home and
    secret scope and wants an answer for that profile only — launch-process
    env and host-wide fallbacks (gh auth, Claude Code credentials) must not
    make it appear ready. Unscoped callers keep the legacy behavior.
    """
    from hermes_cli.config import get_env_path, get_hermes_home, load_config
    from hermes_cli.auth import get_auth_status

    # Determine whether Hermes itself has been explicitly configured (model
    # in config that isn't the hardcoded default). Used below to gate external
    # tool credentials (Claude Code, Codex CLI) that shouldn't silently skip
    # the setup wizard on a fresh install.
    from hermes_cli.config import DEFAULT_CONFIG

    _DEFAULT_MODEL = DEFAULT_CONFIG.get("model", "")
    cfg = load_config()
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        _default = model_cfg.get("default")
        if isinstance(_default, dict):
            from hermes_cli.config import split_model_config_default
            _model_name, _ = split_model_config_default(_default)
        else:
            _model_name = (_default or "")
        _model_name = (str(_model_name) if not isinstance(_model_name, str) else _model_name).strip()
    elif isinstance(model_cfg, str):
        _model_name = model_cfg.strip()
    else:
        _model_name = ""
    _has_hermes_config = _model_name and _model_name != _DEFAULT_MODEL

    # Check env vars (may be set by .env or shell).
    # OPENAI_BASE_URL alone counts — local models (vLLM, llama.cpp, etc.)
    # often don't require an API key.
    from hermes_cli.auth import PROVIDER_REGISTRY

    # Collect all provider env vars
    provider_env_vars = {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    }
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if strict_profile_scope:
        from agent.secret_scope import current_secret_scope

        read_provider_env = (current_secret_scope() or {}).get
    else:
        read_provider_env = os.getenv
    if any(read_provider_env(v) for v in provider_env_vars):
        return True

    # Check .env file for keys
    env_file = get_env_path()
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, _, val = line.partition("=")
                val = val.strip().strip("'\"")
                if key.strip() in provider_env_vars and val:
                    return True
        except Exception:
            pass

    # Cheap local checks first: auth.json and config.yaml are on-disk lookups,
    # while the PROVIDER_REGISTRY sweep below spawns subprocesses (gh) and can
    # take 15-20s — long enough that desktop setup.status calls time out.

    # Check for Nous Portal OAuth credentials
    auth_file = get_hermes_home() / "auth.json"
    if auth_file.exists():
        try:
            import json

            auth = json.loads(auth_file.read_text(encoding="utf-8-sig"))
            active = auth.get("active_provider")
            active_config = PROVIDER_REGISTRY.get(str(active or "").strip().lower())
            if active and not (
                strict_profile_scope and active_config and active_config.auth_type == "api_key"
            ):
                status = get_auth_status(active)
                if status.get("logged_in"):
                    return True
        except Exception:
            pass

    # Check config.yaml — if model is a dict with an explicit provider set,
    # the user has gone through setup (fresh installs have model as a plain
    # string).  Also covers custom endpoints that store api_key/base_url in
    # config rather than .env.
    if isinstance(model_cfg, dict):
        cfg_provider = (model_cfg.get("provider") or "").strip()
        cfg_base_url = (model_cfg.get("base_url") or "").strip()
        cfg_api_key = (model_cfg.get("api_key") or "").strip()
        if cfg_provider or cfg_base_url or cfg_api_key:
            return True

    # Check provider-specific auth fallbacks (for example, Copilot via gh auth).
    if not strict_profile_scope:
        try:
            for provider_id, pconfig in PROVIDER_REGISTRY.items():
                if pconfig.auth_type != "api_key":
                    continue
                status = get_auth_status(provider_id)
                if status.get("logged_in"):
                    return True
        except Exception:
            pass

    # Check for Claude Code OAuth credentials (~/.claude/.credentials.json)
    # Only count these if Hermes has been explicitly configured — Claude Code
    # being installed doesn't mean the user wants Hermes to use their tokens.
    if _has_hermes_config and not strict_profile_scope:
        try:
            from agent.anthropic_adapter import (
                read_claude_code_credentials,
                is_claude_code_token_valid,
            )

            creds = read_claude_code_credentials()
            if creds and (
                is_claude_code_token_valid(creds) or creds.get("refreshToken")
            ):
                return True
        except Exception:
            pass

    return False


def _confirm_startup_expensive_model_override(args) -> None:
    """Guard startup -m/--provider overrides before the first API call."""
    explicit_model = (getattr(args, "model", None) or "").strip()
    explicit_provider = (getattr(args, "provider", None) or "").strip()
    if not explicit_model and not explicit_provider:
        return

    try:
        from hermes_cli.config import load_config
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )
    except Exception as exc:
        logger.warning("startup model cost guard unavailable: %s", exc)
        return

    try:
        config = load_config()
    except Exception as exc:
        logger.warning("startup model cost guard could not load config: %s", exc)
        config = {}
    if not isinstance(config, dict):
        config = {}
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    security_cfg = config.get("security") or {}
    if not isinstance(security_cfg, dict):
        security_cfg = {}

    model = explicit_model or (model_cfg.get("default") or "").strip()
    if not model:
        return
    provider = (explicit_provider or model_cfg.get("provider") or "").strip()
    try:
        # Unified registry: cost guard + id-keyed guards (e.g. the
        # data-training-tier warning) all fire at startup too.
        warnings = selection_warnings(
            model,
            provider=provider,
            base_url=(model_cfg.get("base_url") or ""),
            api_key=(model_cfg.get("api_key") or ""),
        )
    except Exception as exc:
        logger.warning("startup model cost guard failed for %s/%s: %s", provider, model, exc)
        return
    if not warnings:
        return

    # Cost and provider-routing confirmation is intentionally independent of
    # --yolo / --accept-hooks: those flags approve local command/tool risk, not
    # paid aggregator spend or a surprising provider route.
    is_interactive = sys.stdin.isatty()
    allow_unattended_data_training = (
        security_cfg.get("allow_data_training_tiers_noninteractive") is True
    )
    if not is_interactive and allow_unattended_data_training:
        acknowledged = [
            warning for warning in warnings if warning.kind == "data_policy"
        ]
        if acknowledged:
            sys.stderr.write(combined_message(acknowledged) + "\n")
            sys.stderr.write(
                "Proceeding in non-interactive mode because "
                "security.allow_data_training_tiers_noninteractive is true.\n"
            )
            warnings = [
                warning for warning in warnings if warning.kind != "data_policy"
            ]
            if not warnings:
                return

    message = combined_message(warnings)
    if not is_interactive:
        sys.stderr.write(message + "\n")
        if any(warning.kind == "data_policy" for warning in warnings):
            sys.stderr.write(
                "To acknowledge data-training tiers for unattended runs, set "
                "security.allow_data_training_tiers_noninteractive to true "
                "in config.yaml.\n"
            )
        sys.stderr.write(
            "Refusing this startup model override in non-interactive mode. "
            "Run interactively and confirm if you intend to use it.\n"
        )
        raise SystemExit(1)

    sys.stderr.write(message + "\n")
    try:
        reply = input("Use this model for this invocation? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = ""
    if reply not in {"y", "yes"}:
        sys.stderr.write("Model override cancelled.\n")
        raise SystemExit(1)


def _resolve_workspace_key() -> Optional[str]:
    """The current workspace identity for cwd-scoped resume.

    Git repo root when CWD is inside a repo (so all sessions across its
    subdirs/worktrees group together), else the CWD itself. Returns None when
    neither can be determined — callers fall back to the global MRU then.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.abspath(result.stdout.strip())
    except Exception:
        pass
    try:
        return os.getcwd()
    except Exception:
        return None


def _resolve_last_session(source: str = "cli") -> Optional[str]:
    """Look up the most recently-used session ID for a source.

    Scoped to the current workspace first (git repo root, else cwd) so
    ``hermes -c`` from repo A continues repo A's last session rather than the
    global MRU. Falls back to the unscoped MRU when no session matches the
    current workspace, preserving the old behaviour for fresh directories.
    """
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        ws_key = _resolve_workspace_key()
        if ws_key:
            sessions = db.search_sessions(source=source, limit=1, workspace_key=ws_key)
            if sessions:
                return sessions[0]["id"]
        # Fallback: global MRU for this source.
        sessions = db.search_sessions(source=source, limit=1)
        return sessions[0]["id"] if sessions else None
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    return None


def _probe_container(cmd: list, backend: str, via_sudo: bool = False):
    """Run a container inspect probe, returning the CompletedProcess.

    Catches TimeoutExpired specifically for a human-readable message;
    all other exceptions propagate naturally.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    except subprocess.TimeoutExpired:
        label = f"sudo {backend}" if via_sudo else backend
        print(
            f"Error: timed out waiting for {label} to respond.\n"
            f"The {backend} daemon may be unresponsive or starting up.",
            file=sys.stderr,
        )
        sys.exit(1)


def _exec_in_container(container_info: dict, cli_args: list):
    """Replace the current process with a command inside the managed container.

    Probes whether sudo is needed (rootful containers), then os.execvp
    into the container. On success the Python process is replaced entirely
    and the container's exit code becomes the process exit code (OS semantics).
    On failure, OSError propagates naturally.

    Args:
        container_info: dict with backend, container_name, exec_user, hermes_bin
        cli_args: the original CLI arguments (everything after 'hermes')
    """

    backend = container_info["backend"]
    container_name = container_info["container_name"]
    exec_user = container_info["exec_user"]
    hermes_bin = container_info["hermes_bin"]

    runtime = shutil.which(backend)
    if not runtime:
        print(
            f"Error: {backend} not found on PATH. Cannot route to container.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Rootful containers (NixOS systemd service) are invisible to unprivileged
    # users — Podman uses per-user namespaces, Docker needs group access.
    # Probe whether the runtime can see the container; if not, try via sudo.
    sudo_path = None
    probe = _probe_container(
        [runtime, "inspect", "--format", "ok", container_name],
        backend,
    )
    if probe.returncode != 0:
        sudo_path = shutil.which("sudo")
        if sudo_path:
            probe2 = _probe_container(
                [sudo_path, "-n", runtime, "inspect", "--format", "ok", container_name],
                backend,
                via_sudo=True,
            )
            if probe2.returncode != 0:
                print(
                    f"Error: container '{container_name}' not found via {backend}.\n"
                    f"\n"
                    f"The container is likely running as root. Your user cannot see it\n"
                    f"because {backend} uses per-user namespaces. Grant passwordless\n"
                    f"sudo for {backend} — the -n (non-interactive) flag is required\n"
                    f"because a password prompt would hang or break piped commands.\n"
                    f"\n"
                    f"On NixOS:\n"
                    f"\n"
                    f"  security.sudo.extraRules = [{{\n"
                    f'    users = [ "{os.getenv("USER", "your-user")}" ];\n'
                    f'    commands = [{{ command = "{runtime}"; options = [ "NOPASSWD" ]; }}];\n'
                    f"  }}];\n"
                    f"\n"
                    f"Or run: sudo hermes {' '.join(cli_args)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"The container may be running under root. Try: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

    is_tty = sys.stdin.isatty()
    tty_flags = ["-it"] if is_tty else ["-i"]

    env_flags = []
    for var in ("TERM", "COLORTERM", "LANG", "LC_ALL"):
        val = os.environ.get(var)
        if val:
            env_flags.extend(["-e", f"{var}={val}"])

    cmd_prefix = [sudo_path, "-n", runtime] if sudo_path else [runtime]
    exec_cmd = (
        cmd_prefix
        + ["exec"]
        + tty_flags
        + ["-u", exec_user]
        + env_flags
        + [container_name, hermes_bin]
        + cli_args
    )

    os.execvp(exec_cmd[0], exec_cmd)


def _resolve_session_by_name_or_id(name_or_id: str) -> Optional[str]:
    """Resolve a session name (title) or ID to a session ID.

    - If it looks like a session ID (contains underscore + hex), try direct lookup first.
    - Otherwise, treat it as a title and use resolve_session_by_title (auto-latest).
    - Falls back to the other method if the first doesn't match.
    - If the resolved session is a compression root, follow the chain forward
      to the latest continuation. Users who remember the old root ID (e.g.
      from an exit summary printed before the bug fix, or from notes) get
      resumed at the live tip instead of a stale parent with no messages.
    """
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()

        # Try as exact session ID first
        session = db.get_session(name_or_id)
        resolved_id: Optional[str] = None
        if session:
            resolved_id = session["id"]
        else:
            # Try as title (with auto-latest for lineage)
            resolved_id = db.resolve_session_by_title(name_or_id)

        if resolved_id:
            # Project forward through compression chain so resumes land on
            # the live tip instead of a dead compressed parent.
            try:
                resolved_id = db.get_compression_tip(resolved_id) or resolved_id
            except Exception:
                pass

        return resolved_id
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    return None


def _create_titled_session(title: str) -> Optional[str]:
    """Create a fresh session with the given title; return its session id.

    Used by ``chat -c <title> --create-if-missing`` (#86794): programmatic
    callers (plugins, scripts) that want "send to this named thread, making
    it if needed" get a deterministic outcome instead of a silent no-op.

    The session id follows the same timestamp+uuid shape the CLI uses for a
    brand-new session; the title is recorded with user provenance so
    auto-titling never overwrites it.
    """
    db = None
    try:
        import uuid as _uuid

        from hermes_state import SessionDB

        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        db = SessionDB()
        db.create_session(new_session_id, source="cli")
        db.set_session_title(new_session_id, title)
        return new_session_id
    except Exception:
        # Programmatic callers (the #86794 use case) rely on --create-if-missing
        # being deterministic; swallow the failure to keep the error path simple,
        # but log the underlying cause so it lands in errors.log and stays
        # debuggable (DB lock, I/O error, import error — all otherwise invisible).
        logger.exception("Failed to create titled session %r", title)
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _resolve_continue_arg(args, *, use_tui: bool) -> None:
    """Resolve ``-c/--continue`` into ``args.resume``.

    Handles both forms:
    - ``-c <name>``: resolve by title/ID. On miss, fail loudly on **stderr**
      (exit 1) so programmatic callers see the error even under quiet mode
      (#86794); with ``--create-if-missing``, create a fresh titled session
      and resume into it instead.
    - bare ``-c``: continue this terminal's breadcrumb session if valid,
      else the most recent session (workspace-scoped MRU, then global
      fallback).
    """
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            # -c "session name" — resolve by title or ID
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            elif getattr(args, "create_if_missing", False):
                # --create-if-missing: no session matches the title — create a
                # new session with that title and proceed. This is the
                # programmatic-caller primitive ("send to this named thread,
                # making it if needed"); without it a background/quiet send to
                # a not-yet-existing named session silently no-ops (#86794).
                new_sid = _create_titled_session(continue_val)
                if new_sid:
                    args.resume = new_sid
                else:
                    print(
                        f"No session found matching '{continue_val}' and "
                        "a new titled session could not be created.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            else:
                print(f"No session found matching '{continue_val}'.", file=sys.stderr)
                print(
                    "Use 'hermes sessions list' to see available sessions, or "
                    "pass --create-if-missing to start a new session with that title.",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # -c with no argument — prefer this terminal's own breadcrumb
            # (written at session start / rotation) so side-by-side terminals
            # each continue their own conversation. Falls back to the
            # most-recent session when there is no valid breadcrumb, or when
            # session.terminal_continue is false in config.yaml.
            if getattr(args, "create_if_missing", False):
                # --create-if-missing only makes sense with a named session;
                # with a bare -c there is nothing to create, so surface the
                # no-op to programmatic callers instead of silently ignoring it.
                print(
                    "--create-if-missing requires a session name: "
                    "`-c <name> --create-if-missing`",
                    file=sys.stderr,
                )
            try:
                from hermes_cli.terminal_breadcrumbs import resolve_breadcrumb_session

                _crumb_id = resolve_breadcrumb_session()
            except Exception:
                _crumb_id = None
            if _crumb_id:
                args.resume = _crumb_id
            else:
                # No valid breadcrumb — continue the most recent session
                source = "tui" if use_tui else "cli"
                last_id = _resolve_last_session(source=source)
                if not last_id and source == "tui":
                    last_id = _resolve_last_session(source="cli")
                if last_id:
                    args.resume = last_id
                else:
                    kind = "TUI" if use_tui else "CLI"
                    print(f"No previous {kind} session found to continue.")
                    sys.exit(1)


def _resolve_chat_session_args(args, use_tui: bool) -> None:
    """Normalize --in / --resume / --continue on ``args`` before agent init.

    Order matters: ``--in DIR`` chdirs first so workspace-scoped "latest"/-c
    lookups key off DIR (and pins the session there, skipping cwd restore);
    then ``--resume latest`` → MRU id, ``--continue`` → ``--resume``,
    ``--resume @claude/@codex`` → imported session id, title → id; finally
    cd back into a resumed session's recorded cwd (best-effort, opt-out via
    --no-restore-cwd, skipped under --worktree).
    """
    # --in DIR: run in DIR. Must happen before any session resolution so the
    # workspace-scoped "latest"/-c lookups key off DIR, and it pins the
    # session there — an explicit --in wins over a resumed session's
    # recorded cwd (so the restore step below is skipped).
    in_dir = getattr(args, "in_dir", None)
    if in_dir:
        # Git Bash / MSYS hands the CLI POSIX-style paths (`--in ~` expands to
        # `/c/Users/x` before Python ever sees it; MSYS2's path conversion is
        # disabled for native executables). Translate the MSYS/Cygwin/WSL
        # drive-root spellings to native Windows form first — no-op elsewhere.
        from tools.environments.local import _msys_to_windows_path

        _target_dir = os.path.abspath(
            os.path.expanduser(_msys_to_windows_path(in_dir))
        )
        if not os.path.isdir(_target_dir):
            print(f"Error: --in directory not found: {in_dir}")
            sys.exit(1)
        try:
            os.chdir(_target_dir)
        except OSError as e:
            print(f"Error: cannot enter --in directory {in_dir}: {e}")
            sys.exit(1)
        args.no_restore_cwd = True

    # --resume latest: keyword for "most recent session" — same resolution
    # as `-c` with no name (workspace-scoped MRU, then global fallback).
    # The keyword wins over a session literally titled "latest"; that
    # session stays reachable via its ID or `-c latest` (title match).
    _resume_raw = getattr(args, "resume", None)
    if isinstance(_resume_raw, str) and _resume_raw.strip().lower() == "latest":
        _source = "tui" if use_tui else "cli"
        _last_id = _resolve_last_session(source=_source)
        if not _last_id and _source == "tui":
            _last_id = _resolve_last_session(source="cli")
        if _last_id:
            args.resume = _last_id
        else:
            kind = "TUI" if use_tui else "CLI"
            print(f"No previous {kind} session found to resume.")
            print("Use 'hermes sessions list' to see available sessions.")
            sys.exit(1)

    # Resolve --continue into --resume with the latest session or by name
    _resolve_continue_arg(args, use_tui=use_tui)

    # --resume @claude / --resume @codex: import a foreign session (Claude
    # Code / Codex CLI) and resume the newly created Hermes session.
    _resume_foreign = getattr(args, "resume", None)
    if isinstance(_resume_foreign, str) and _resume_foreign.strip().lower() in (
        "@claude",
        "@codex",
    ):
        from hermes_cli.foreign_sessions import (
            import_foreign_session,
            pick_foreign_session,
        )

        _foreign_source = _resume_foreign.strip().lower().lstrip("@")
        _picked = pick_foreign_session(_foreign_source)
        if _picked is None:
            sys.exit(1)
        try:
            _imported_id = import_foreign_session(_picked.source, _picked.path)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"✓ Imported as {_imported_id} — resuming it now.")
        print(f"  (later: hermes --resume {_imported_id})")
        args.resume = _imported_id

    # Resolve --resume by title if it's not a direct session ID
    resume_val = getattr(args, "resume", None)
    if resume_val:
        resolved = _resolve_session_by_name_or_id(resume_val)
        if resolved:
            args.resume = resolved
        # If resolution fails, keep the original value — _init_agent will
        # report "Session not found" with the original input

    # Session<->workspace binding: cd back into a resumed session's recorded cwd
    # so it resumes in the repo it belonged to. Opt out with --no-restore-cwd;
    # skipped under --worktree (that path owns its own dir). Best-effort — a
    # missing dir warns and stays put rather than failing the resume.
    if (
        getattr(args, "resume", None)
        and not getattr(args, "no_restore_cwd", False)
        and not getattr(args, "worktree", False)
    ):
        _resume_db = None
        try:
            from hermes_state import SessionDB

            _resume_db = SessionDB()
            _saved_cwd = ((_resume_db.get_session(args.resume) or {}).get("cwd") or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")
        except Exception:
            pass  # never let cwd-restore break a resume
        finally:
            if _resume_db is not None:
                try:
                    _resume_db.close()
                except Exception:
                    pass


def _warn_retired_xai_models() -> None:
    """One-shot xAI retirement warning on stderr; non-blocking, never fails startup."""
    try:
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            RETIREMENT_DATE,
            find_retired_xai_refs,
            format_issue,
        )
        from hermes_cli.config import load_config as _load_config_for_xai_check

        _retired_xai_refs = find_retired_xai_refs(_load_config_for_xai_check())
        if _retired_xai_refs:
            sys.stderr.write(
                f"\033[33m⚠ xAI retires {len(_retired_xai_refs)} model(s) "
                f"in your config on {RETIREMENT_DATE}:\033[0m\n"
            )
            for _ref in _retired_xai_refs:
                sys.stderr.write(f"  \033[33m⚠\033[0m {format_issue(_ref)}\n")
            sys.stderr.write(f"  \033[2mMigration guide: {MIGRATION_GUIDE_URL}\033[0m\n")
            sys.stderr.write("  \033[2mRun 'hermes doctor' for details.\033[0m\n\n")
    except Exception:
        pass


def _start_chat_background_prefetch() -> None:
    """Kick off the update-check/banner prefetch and the bundled-skills sync.

    Update check is opt-in on Termux (it imports rich/prompt_toolkit in the
    foreground and competes for CPU on single-core devices). The skills sync
    is idempotent and hash-gated, so it normally runs in a daemon thread;
    the ONE exception is an unseeded ~/.hermes/skills — there the banner
    prefetch would race the sync and cache an empty skills index, so the
    first run syncs in the foreground and drops the banner's skills cache.
    """
    # Start update check in background (runs while other init happens).
    # On Termux this imports rich/prompt_toolkit in the foreground and then
    # competes for CPU on single-core devices, so keep it opt-in there.
    if _termux_should_prefetch_update_check():
        try:
            from hermes_cli.banner import prefetch_banner_data, prefetch_update_check

            prefetch_update_check()
            # Warm git banner state + skills index off-thread too — their
            # subprocess/file-I/O waits overlap the CPU-bound cli import.
            prefetch_banner_data()
        except Exception:
            pass

    # Sync bundled skills on every CLI launch. Normally runs in a background
    # daemon thread: the sync is idempotent, hash-gated (unchanged skills are
    # skipped), and nothing on the banner path depends on it, yet the scan
    # alone costs ~120-170ms of rglob/hashing on the startup path. Skill
    # loading happens at agent init (first message), by which point the
    # sync has long finished.
    #
    # FIRST RUN is the exception: with an empty ~/.hermes/skills the banner
    # prefetch races the background sync, caches an empty skills index, and
    # the very first launch greets the user with "No skills installed ·
    # 0 skills" while 69 bundled skills land milliseconds later (full-surface
    # CLI QA sweep, Aug 2026). Run the sync in the foreground exactly once —
    # only when the skills dir has no SKILL.md yet — so the first impression
    # matches reality; every later launch keeps the background path.
    def _skills_dir_is_unseeded() -> bool:
        try:
            from hermes_cli.config import get_hermes_home
            skills_dir = Path(get_hermes_home()) / "skills"
            if not skills_dir.is_dir():
                return True
            return next(skills_dir.rglob("SKILL.md"), None) is None
        except Exception:
            return False

    def _skills_sync_bg() -> None:
        try:
            _sync_bundled_skills_for_startup()
        except Exception:
            pass

    if _skills_dir_is_unseeded():
        _skills_sync_bg()
        # The banner prefetch thread (started above) may have scanned the
        # still-empty dir and cached an empty skills index — drop it so the
        # banner recomputes against the freshly seeded tree.
        try:
            import hermes_cli.banner as _banner_mod
            _banner_mod._available_skills_cache = None
        except Exception:
            pass
    else:
        threading.Thread(
            target=_skills_sync_bg, name="bundled-skills-sync", daemon=True
        ).start()


def cmd_chat(args):
    """Run interactive chat CLI."""
    _apply_safe_mode(args)
    _apply_user_config_bypass(args)
    _guard_noninteractive_user_config(args)
    use_tui = _resolve_use_tui(args)

    _resolve_chat_session_args(args, use_tui)

    _warn_retired_xai_models()

    # First-run guard: check if any provider is configured before launching
    if not _has_any_provider_configured():
        print()
        print(
            "It looks like Hermes isn't configured yet -- no API keys or providers found."
        )
        print()
        print("  Run:  hermes setup")
        print()

        from hermes_cli.setup import (
            is_interactive_stdin,
            print_noninteractive_setup_guidance,
        )

        if not is_interactive_stdin():
            print_noninteractive_setup_guidance(
                "No interactive TTY detected for the first-run setup prompt."
            )
            sys.exit(1)

        try:
            reply = input("Run setup now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply in {"", "y", "yes"}:
            cmd_setup(args)
            return
        print()
        print("You can run 'hermes setup' at any time to configure.")
        sys.exit(1)

    _start_chat_background_prefetch()

    # --yolo: bypass all dangerous command approvals.
    # Also set in main() before _prepare_agent_startup() — that is the
    # authoritative site because it runs before tool imports freeze
    # _YOLO_MODE_FROZEN.  This redundant set is a safety net for callers
    # that invoke cmd_chat directly (e.g. subcommand dispatch).
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # --ignore-rules: skip auto-injection of AGENTS.md/SOUL.md/.cursorrules
    # (rules), memory entries, and any preloaded skills coming from user config.
    # Maps to AIAgent(skip_context_files=True, skip_memory=True).
    if getattr(args, "ignore_rules", False):
        os.environ["HERMES_IGNORE_RULES"] = "1"

    # --source: tag session source for filtering (e.g. 'tool' for third-party integrations)
    if getattr(args, "source", None):
        os.environ["HERMES_SESSION_SOURCE"] = args.source

    _pin_kanban_board_env()
    _confirm_startup_expensive_model_override(args)

    if use_tui:
        _launch_tui(
            getattr(args, "resume", None),
            tui_dev=getattr(args, "tui_dev", False),
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            skills=getattr(args, "skills", None),
            verbose=getattr(args, "verbose", None),
            quiet=getattr(args, "quiet", False),
            query=getattr(args, "query", None),
            image=getattr(args, "image", None),
            worktree=getattr(args, "worktree", False),
            checkpoints=getattr(args, "checkpoints", False),
            pass_session_id=getattr(args, "pass_session_id", False),
            max_turns=getattr(args, "max_turns", None),
            accept_hooks=getattr(args, "accept_hooks", False),
        )

    # --query-file: read the single query from a file (or stdin via '-') so
    # callers never have to shell-quote message bodies. This is the transport
    # the Bot Mode DM protocol uses — interpolating arbitrary text into a
    # double-quoted shell argument truncates on quotes and executes $(...)
    # (see tools/bot_mode_probe.py).
    _qfile = getattr(args, "query_file", None)
    if _qfile:
        if args.query:
            # argparse's mutually-exclusive group catches the normal CLI path;
            # this guards programmatic callers that fill the namespace directly.
            print("Error: -q/--query and --query-file are mutually exclusive", file=sys.stderr)
            sys.exit(2)
        try:
            if _qfile == "-":
                args.query = sys.stdin.read()
            else:
                with open(_qfile, "r", encoding="utf-8", errors="replace") as _fh:
                    args.query = _fh.read()
        except OSError as _e:
            print(f"Error: cannot read --query-file {_qfile}: {_e}", file=sys.stderr)
            sys.exit(2)
        if not (args.query or "").strip():
            print(f"Error: --query-file {_qfile} is empty", file=sys.stderr)
            sys.exit(2)

    # Build kwargs from args
    kwargs = {
        "model": args.model,
        "provider": getattr(args, "provider", None),
        "reasoning": getattr(args, "reasoning", None),
        "toolsets": args.toolsets,
        "skills": getattr(args, "skills", None),
        "verbose": getattr(args, "verbose", None),
        "quiet": getattr(args, "quiet", False),
        "query": args.query,
        "oneshot": bool(getattr(args, "oneshot_exit", False)),
        "image": getattr(args, "image", None),
        "resume": getattr(args, "resume", None),
        "worktree": getattr(args, "worktree", False),
        "checkpoints": getattr(args, "checkpoints", False),
        "pass_session_id": getattr(args, "pass_session_id", False),
        "max_turns": getattr(args, "max_turns", None),
        "run_budget": getattr(args, "run_budget", None),
        "ignore_rules": getattr(args, "ignore_rules", False) or getattr(args, "safe_mode", False),
        "ignore_user_config": getattr(args, "ignore_user_config", False) or getattr(args, "safe_mode", False),
        "compact": getattr(args, "compact", False),
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        from cli import main as cli_main

        cli_main(**kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        # Mixed-version installs (new cli.py, older hermes_cli.config) crash
        # here — e.g. missing resolve_turn_limit / split_model_config_default
        # (#96900). The agent-setup mixin prints this hint too late: HermesCLI
        # construction already failed. Fast-chat launch also goes through
        # cmd_chat, so this one catch covers `hermes` / `hermes chat`.
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(e):
            sys.exit(1)
        raise


def cmd_gateway(args):
    """Gateway management commands."""
    _sync_bundled_skills_quietly()

    from hermes_cli.gateway import gateway_command

    gateway_command(args)


def cmd_proxy(args):
    """Local OpenAI-compatible proxy to OAuth providers."""
    # Lazy import — pulls in aiohttp, which is gated behind an extras install
    # for users who don't run the proxy or the messaging gateway.
    from hermes_cli.proxy.cli import cmd_proxy as _cmd_proxy

    rc = _cmd_proxy(args)
    if isinstance(rc, int) and rc != 0:
        raise SystemExit(rc)


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    _require_tty("whatsapp")
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_constants import find_node_executable, with_hermes_node_path

    print()
    print("⚕ WhatsApp Setup")
    print("=" * 50)

    # ── Step 1: Choose mode ──────────────────────────────────────────────
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if not current_mode:
        print()
        print("How will you use WhatsApp with Hermes?")
        print()
        print("  1. Separate bot number (recommended)")
        print("     People message the bot's number directly — cleanest experience.")
        print(
            "     Requires a second phone number with WhatsApp installed on a device."
        )
        print()
        print("  2. Personal number (self-chat)")
        print("     You message yourself to talk to the agent.")
        print("     Quick to set up, but the UX is less intuitive.")
        print()
        try:
            choice = input("  Choose [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return

        if choice == "1":
            save_env_value("WHATSAPP_MODE", "bot")
            wa_mode = "bot"
            print("  ✓ Mode: separate bot number")
            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │  Getting a second number for the bot:           │")
            print("  │                                                 │")
            print("  │  Easiest: Install WhatsApp Business (free app)  │")
            print("  │  on your phone with a second number:            │")
            print("  │    • Dual-SIM: use your 2nd SIM slot            │")
            print("  │    • Google Voice: free US number (voice.google) │")
            print("  │    • Prepaid SIM: $3-10, verify once            │")
            print("  │                                                 │")
            print("  │  WhatsApp Business runs alongside your personal │")
            print("  │  WhatsApp — no second phone needed.             │")
            print("  └─────────────────────────────────────────────────┘")
        else:
            save_env_value("WHATSAPP_MODE", "self-chat")
            wa_mode = "self-chat"
            print("  ✓ Mode: personal number (self-chat)")
    else:
        wa_mode = current_mode
        mode_label = (
            "separate bot number" if wa_mode == "bot" else "personal number (self-chat)"
        )
        print(f"\n✓ Mode: {mode_label}")

    # ── Step 2: Mode is selected, will enable WhatsApp only after pairing ──
    # We intentionally don't write WHATSAPP_ENABLED=true here.  If the user
    # aborts the wizard later (Ctrl+C, failed npm install, missed QR scan),
    # we'd otherwise leave .env claiming WhatsApp is ready when the bridge
    # has no creds.json.  Every subsequent `hermes gateway` then paid a 30s
    # bridge-bootstrap timeout and queued WhatsApp for indefinite retries.
    # Now: aborted setup leaves WHATSAPP_ENABLED unset → gateway skips it.
    # Re-runs that already have WHATSAPP_ENABLED=true (from a prior
    # successful pairing) stay enabled — we just don't write it pre-emptively.
    print()
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true":
        print("✓ WhatsApp is already enabled")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        try:
            response = input("\n  Update allowed users? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            if wa_mode == "bot":
                phone = line_input(
                    "  Phone numbers that can message the bot (comma-separated): "
                ).strip()
            else:
                phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ Updated to: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  Who should be allowed to message the bot?")
            phone = line_input(
                "  Phone numbers (comma-separated, or * for anyone): "
            ).strip()
        else:
            phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Allowed users set: {phone}")
        else:
            print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print(
            "\n→ Installing WhatsApp bridge dependencies (this can take a few minutes)..."
        )
        npm = find_node_executable("npm")
        if not npm:
            print("  ✗ npm not found on PATH — install Node.js first")
            return
        try:
            result = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
                cwd=str(bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=with_hermes_node_path(),
            )
        except KeyboardInterrupt:
            print("\n  ✗ Install cancelled")
            return
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            preview = "\n".join(err.splitlines()[-30:]) if err else "(no output)"
            print("  ✗ npm install failed:")
            print(preview)
            return
        print("  ✓ Dependencies installed")
    else:
        print("✓ Bridge dependencies already installed")

    # ── Step 5: Check for existing session ───────────────────────────────
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if (session_dir / "creds.json").exists():
        print("✓ Existing WhatsApp session found")
        try:
            response = input(
                "\n  Re-pair? This will clear the existing session. [y/N] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
            # Existing pairing — ensure WHATSAPP_ENABLED reflects that.
            # (Older installs may have lost the env var; covers re-runs
            # where the user picked "no, keep my session" but the var
            # was never set or got removed.)
            if (get_env_value("WHATSAPP_ENABLED") or "").lower() != "true":
                save_env_value("WHATSAPP_ENABLED", "true")
            print("\n✓ WhatsApp is configured and paired!")
            print("  Start the gateway with: hermes gateway")
            return

    # ── Step 6: QR code pairing ──────────────────────────────────────────
    print()
    print("─" * 50)
    if wa_mode == "bot":
        print("📱 Open WhatsApp (or WhatsApp Business) on the")
        print("   phone with the BOT's number, then scan:")
    else:
        print("📱 Open WhatsApp on your phone, then scan:")
    print()
    print("   Settings → Linked Devices → Link a Device")
    print("─" * 50)
    print()

    try:
        subprocess.run(
            [
                find_node_executable("node") or "node",
                str(bridge_script),
                "--pair-only",
                "--session",
                str(session_dir),
            ],
            cwd=str(bridge_dir),
            env=with_hermes_node_path(),
        )
    except KeyboardInterrupt:
        pass

    # ── Step 7: Post-pairing ─────────────────────────────────────────────
    print()
    if (session_dir / "creds.json").exists():
        # Only enable WhatsApp now that pairing actually succeeded.  If the
        # user Ctrl+C'd at any earlier step, WHATSAPP_ENABLED stays unset
        # and `hermes gateway` skips it cleanly instead of paying a 30s
        # bridge timeout + queueing the platform for indefinite retries.
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp paired successfully!")
        print()
        if wa_mode == "bot":
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Send a message to the bot's WhatsApp number")
            print("    3. The agent will reply automatically")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
        else:
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Open WhatsApp → Message Yourself")
            print("    3. Type a message — the agent will reply")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
            print("  so you can tell them apart from your own messages.")
        print()
        print("  Or install as a service: hermes gateway install")
    else:
        print("⚠ Pairing may not have completed. Run 'hermes whatsapp' to try again.")


def cmd_whatsapp_cloud(args):
    """Set up WhatsApp Business Cloud API (official Meta integration).

    Walks the user through the Meta-side credentials (Phone Number ID,
    Access Token, App Secret, optional App/WABA IDs) plus webhook
    configuration. Includes field-shape validators that catch the most
    common setup mistakes (e.g. pasting a phone number into the Phone
    Number ID field).

    Distinct from ``hermes whatsapp`` (the Baileys bridge wizard) — the
    two adapters are complementary, not alternatives. See
    ``hermes_cli/setup_whatsapp_cloud.py``.
    """
    _require_tty("whatsapp-cloud")
    from hermes_cli.setup_whatsapp_cloud import run_whatsapp_cloud_setup

    return run_whatsapp_cloud_setup()


def cmd_setup(args):
    """Interactive setup wizard."""
    from hermes_cli.setup import run_setup_wizard

    run_setup_wizard(args)


def cmd_model(args):
    """Select default model — starts with provider selection, then model picker."""
    _require_tty("model")
    if getattr(args, "refresh", False):
        try:
            from hermes_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
            print("  Cleared model picker cache.")
        except Exception:
            pass
    from hermes_cli.setup import run_setup_action_with_navigation

    run_setup_action_with_navigation(
        "Model & Provider",
        lambda: select_provider_and_model(args=args),
        cancelled_message="No change.",
    )


# Provider id -> flow(config, current_model, args). Lambdas resolve the
# _model_flow_* names at call time so test monkeypatches on hermes_cli.main
# keep intercepting. Custom-slug providers (always ``custom:*``),
# remove-custom and the generic API-key set are the fallthrough branches in
# select_provider_and_model.
_PROVIDER_MODEL_FLOWS = {
    "openrouter": lambda c, m, a: _model_flow_openrouter(c, m),
    "moa": lambda c, m, a: _model_flow_moa(c, m),
    "ai-gateway": lambda c, m, a: _model_flow_ai_gateway(c, m),
    "nous": lambda c, m, a: _model_flow_nous(c, m, args=a),
    "openai-codex": lambda c, m, a: _model_flow_openai_codex(c, m),
    "xai-oauth": lambda c, m, a: _model_flow_xai_oauth(c, m, args=a),
    "qwen-oauth": lambda c, m, a: _model_flow_qwen_oauth(c, m),
    "minimax-oauth": lambda c, m, a: _model_flow_minimax_oauth(c, m, args=a),
    "copilot-acp": lambda c, m, a: _model_flow_copilot_acp(c, m),
    "copilot": lambda c, m, a: _model_flow_copilot(c, m),
    "custom": lambda c, m, a: _model_flow_custom(c),
    "anthropic": lambda c, m, a: _model_flow_anthropic(c, m),
    "kimi-coding": lambda c, m, a: _model_flow_kimi(c, m),
    "stepfun": lambda c, m, a: _model_flow_stepfun(c, m),
    "bedrock": lambda c, m, a: _model_flow_bedrock(c, m),
    "vertex": lambda c, m, a: _model_flow_vertex(c, m),
    "azure-foundry": lambda c, m, a: _model_flow_azure_foundry(c, m),
}


def select_provider_and_model(args=None):
    """Core provider selection + model picking logic.

    Shared by ``cmd_model`` (``hermes model``) and the setup wizard
    (``setup_model_provider`` in setup.py).  Handles the full flow:
    provider picker, credential prompting, model selection, and config
    persistence.
    """
    from hermes_cli.auth import (
        resolve_provider,
        AuthError,
        format_auth_error,
    )
    from hermes_cli.config import (
        get_compatible_custom_providers,
        load_config,
        get_env_value,
    )
    from hermes_cli.providers import (
        custom_provider_aliases,
        resolve_provider_full,
    )

    config = load_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        current_model = current_model.get("default", "")
    current_model = current_model or "(not set)"

    # Read effective provider the same way the CLI does at startup:
    # config.yaml model.provider > env var > auto-detect
    config_provider = None
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        config_provider = model_cfg.get("provider")

    effective_provider = (
        config_provider or os.getenv("HERMES_INFERENCE_PROVIDER") or "auto"
    )
    compatible_custom_providers = get_compatible_custom_providers(config)

    def _norm_base_url(url: str) -> str:
        return str(url or "").strip().rstrip("/").lower()

    # Add user-defined custom providers from config.yaml
    _custom_provider_map = _named_custom_provider_map(
        config
    )  # key → {name, base_url, api_key}

    def _canonical_named_custom_key(provider_id: str) -> str:
        requested = str(provider_id or "").strip().lower()
        for key, provider_info in _custom_provider_map.items():
            if requested in custom_provider_aliases(
                provider_info.get("name", ""),
                provider_info.get("provider_key", ""),
            ):
                return key
        return provider_id

    def _active_custom_key_from_base_url() -> str:
        if effective_provider != "custom" or not isinstance(model_cfg, dict):
            return ""
        current_base = _norm_base_url(model_cfg.get("base_url", ""))
        if not current_base:
            return ""
        for key, provider_info in _custom_provider_map.items():
            if _norm_base_url(provider_info.get("base_url", "")) == current_base:
                return key
        return ""

    active = _active_custom_key_from_base_url()
    if active is None:
        active = ""
    if not active and effective_provider != "auto":
        active_def = resolve_provider_full(
            effective_provider,
            config.get("providers"),
            compatible_custom_providers,
        )
        if active_def is not None:
            active = active_def.id
            if active_def.source == "user-config":
                active = _canonical_named_custom_key(active)
        else:
            warning = (
                f"Unknown provider '{effective_provider}'. Check 'hermes model' for "
                "available providers, or run 'hermes doctor' to diagnose config "
                "issues."
            )
            print(f"Warning: {warning} Falling back to auto provider detection.")
    if not active:
        try:
            active = resolve_provider("auto")
        except AuthError as exc:
            if effective_provider == "auto":
                warning = format_auth_error(exc)
                print(f"Warning: {warning} Falling back to auto provider detection.")
            active = None  # no provider yet; default to first in list

    # Detect custom endpoint
    if active == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        active = "custom"

    from hermes_cli.models import _PROVIDER_LABELS

    provider_labels = dict(_PROVIDER_LABELS)  # derive from canonical list
    if active and active in _custom_provider_map:
        active_label = _custom_provider_map[active]["name"]
    else:
        active_label = provider_labels.get(active, active) if active else "none"

    print()
    print(f"  Current model:    {current_model}")
    print(f"  Active provider:  {active_label}")
    print()

    # Step 1: Provider selection (group rows drill into a member sub-picker
    # that resolves back to a concrete slug, so the dispatch below is unchanged).
    ordered, default_idx = _build_provider_picker_rows(
        config, active, provider_labels, _custom_provider_map
    )

    provider_idx = _prompt_provider_choice(
        [label for _, label, _ in ordered],
        default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        print("No change.")
        return

    selected_key = ordered[provider_idx][0]
    selected_members = ordered[provider_idx][2]

    # Group row → drill into a member sub-picker. Default to the active member
    # if the active provider lives in this group. The descriptive text lives on
    # the group row itself, so member rows show only their short label here.
    if selected_members:
        member_default = 0
        if active in selected_members:
            member_default = selected_members.index(active)
        member_labels = [
            provider_labels.get(m, m) for m in selected_members
        ]
        group_label = ordered[provider_idx][1].split(" ▸", 1)[0]
        member_idx = _prompt_provider_choice(
            member_labels,
            default=member_default,
            title=f"Select {group_label} provider:",
        )
        if member_idx is None:
            print("No change.")
            return
        selected_provider = selected_members[member_idx]
    else:
        selected_provider = selected_key

    if selected_provider == "aux-config":
        _aux_config_menu()
        return

    # Step 2: Provider-specific setup + model selection.
    # Flows are looked up by name at call time (globals()) so test
    # monkeypatches on hermes_cli.main._model_flow_* keep intercepting.
    flow = _PROVIDER_MODEL_FLOWS.get(selected_provider)
    if flow is not None:
        flow(config, current_model, args)
    elif (
        selected_provider.startswith("custom:")
        or selected_provider in _custom_provider_map
    ):
        provider_info = _named_custom_provider_map(load_config()).get(selected_provider)
        if provider_info is None:
            print(
                "Warning: the selected saved custom provider is no longer available. "
                "It may have been removed from config.yaml. No change."
            )
            return
        _model_flow_named_custom(config, provider_info)
    elif selected_provider == "remove-custom":
        _remove_custom_provider(config)
    elif (
        selected_provider in _GENERIC_API_KEY_PROVIDERS
        or _is_profile_api_key_provider(selected_provider)
    ):
        _model_flow_api_key_provider(config, selected_provider, current_model)

    # ── Post-switch cleanup: clear stale OPENAI_BASE_URL ──────────────
    # When the user switches to a named provider (anything except "custom"),
    # a leftover OPENAI_BASE_URL in ~/.hermes/.env can poison auxiliary
    # clients that use provider:auto. Clear it proactively.  (#5161)
    if selected_provider not in {
        "custom",
        "cancel",
        "remove-custom",
    } and not selected_provider.startswith("custom:"):
        _clear_stale_openai_base_url()


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary model configuration
#
# Hermes uses lightweight "auxiliary" models for side tasks (vision analysis,
# context compression, web extraction, session search, etc.). Each task has
# its own provider+model pair in config.yaml under `auxiliary.<task>`.
#
# The UI lives behind "Configure auxiliary models..." at the bottom of the
# `hermes model` provider picker. It does NOT re-run credential setup — it
# only routes already-authenticated providers to specific aux tasks. Users
# configure new providers through the normal `hermes model` flow first.
# ─────────────────────────────────────────────────────────────────────────────


# Lazy-export the model catalog at module level. Tests and a handful of
# downstream call sites read `hermes_cli.main._PROVIDER_MODELS` directly,
# so the symbol needs to be reachable as a module attribute. But importing
# the catalog eagerly costs ~55ms on every `hermes` invocation — including
# fast paths like `hermes --version` and slash-command dispatch that never
# touch the catalog. PEP 562 module-level __getattr__ defers the import
# until first attribute access, so the cost is only paid by callers that
# actually look up the catalog. Termux already defers via the same
# mechanism (its model-selection handlers do their own function-local
# imports), so the explicit termux branch from before is no longer needed.
_LAZY_MODEL_EXPORTS = ("_PROVIDER_MODELS",)


# The main.py decomposition moved the sessions/update/dashboard command
# implementations into their own modules, but main.py still re-exports their
# surface so argparse wiring and test monkeypatches on hermes_cli.main.<name>
# keep resolving unchanged. Importing those modules eagerly costs ~50ms on
# every `hermes` invocation, including fast paths like `hermes --version`
# that never run a subcommand. Resolve the re-exports through the module
# __getattr__ below instead, so each module is only imported when one of its
# names is actually touched. Monkeypatching keeps working: patch.object sets
# a real module attribute, which shadows __getattr__.
_LAZY_COMMAND_EXPORTS = {
    "hermes_cli.sessions_cmd": (
        "cmd_sessions",
        "_annotate_session_statuses",
        "_relative_time",
        "_session_browse_picker",
        "_session_status_tag",
        "_size_delta_label",
    ),
    "hermes_cli.dashboard_procs": (
        "_detect_concurrent_hermes_instances",
        "_filter_dashboard_respawn_candidates",
        "_kill_stale_dashboard_processes",
        "_scan_dashboard_processes",
    ),
    "hermes_cli.update_cmd": (
        "_abort_dependency_sync_if_self_locked",
        "_add_upstream_remote",
        "_atomic_replace_dir",
        "_capture_active_lazy_features",
        "_capture_active_tool_dependencies",
        "_capture_head_sha",
        "_classify_concurrent_instance",
        "_clear_fleet_restart_pending_marker",
        "_assess_parked_branch_switch",
        "_branch_head_label",
        "_branch_head_suffix",
        "_cmd_update_check",
        "_cmd_update_impl",
        "_cold_start_windows_gateway_after_update",
        "_count_commits_between",
        "_dependency_sync_would_rewrite",
        "_detect_self_loaded_native_modules",
        "_detect_venv_python_processes",
        "_desktop_owns_gateway_lifecycle",
        "_defer_update_for_self_lock",
        "_discard_lockfile_churn",
        "_discard_stashed_changes",
        "_park_stashed_changes",
        "_ensure_acp_launcher",
        "_ensure_uv_for_termux",
        "_finish_dashboard_update_cleanup",
        "_fleet_probe_expected_runtimes",
        "_fleet_restart_pending_marker_path",
        "_filter_non_gateway_concurrent_instances",
        "_for_each_systemd_gateway_unit",
        "_format_concurrent_instances_message",
        "_format_time_ago",
        "_gateway_service_matches_profile",
        "_gateway_recovery_partition",
        "_handoff_reapable_backend_pids",
        "_ledger_reapable_backend_pids",
        "_purge_stale_hermes_modules",
        "_format_venv_python_holders_message",
        "_gateway_prompt",
        "_get_origin_url",
        "_has_upstream_remote",
        "_install_psutil_android_compat",
        "_invalidate_update_cache",
        "_is_fork",
        "_leftover_pausable_gateway_pids",
        "_ledger_manual_serve_holders",
        "_relaunch_stopped_serves",
        "_serve_relaunch_commands",
        "_log_only_write",
        "_mark_skip_upstream_prompt",
        "_npm_lockfile_changed",
        "_npm_manifests_digest",
        "_ORPHAN_RESCUE_REF_MAX_AGE_DAYS",
        "_ORPHAN_RESCUE_REFS_TO_KEEP",
        "_orphaned_desktop_backend_pids",
        "_pending_fleet_restart_needed",
        "_pause_windows_gateways_for_update",
        "_print_curator_first_run_notice",
        "_prune_orphan_rescue_refs",
        "_print_curator_recent_run_notice",
        "_print_parked_branch_kept_notice",
        "_print_parked_branch_skip_warning",
        "_print_update_completion",
        "_record_npm_lockfile_hash",
        "_refresh_active_lazy_features",
        "_refresh_active_memory_provider_dependencies",
        "_refresh_bootstrap_cache_scripts",
        "_refresh_windows_gateway_launchers",
        "_recover_gateway_restart_after_abort",
        "_reload_updated_runtime_modules",
        "_restart_phase_failure_is_incomplete",
        "_restore_active_tool_dependencies",
        "_restore_stashed_changes",
        "_resume_windows_gateways_after_update",
        "_run_pending_fleet_restart",
        "_run_logged_subprocess",
        "_run_pre_update_backup",
        "_service_unit_supports_graceful_sigusr1_restart",
        "_should_skip_upstream_prompt",
        "_stash_local_changes_if_needed",
        "_stop_process_trees",
        "_surviving_gateway_pids_after_failed_restart",
        "_sync_with_upstream_if_needed",
        "_update_node_dependencies",
        "_update_via_zip",
        "_warn_orphaned_update_autostashes",
        "_upgrade_pip_before_lazy_refresh",
        "_validate_critical_files_syntax",
        "_validate_critical_modules_import",
        "_venv_core_imports_healthy",
        "_venv_launcher_ancestors",
        "_wait_for_windows_update_gateway_exit",
        "_warn_gateway_restart_phase_aborted",
        "_warn_incomplete_gateway_fleet_restart",
        "_warn_pending_fleet_restart_on_startup",
        "_web_build_toolchain_ready",
        "_web_toolchain_roots",
        "_write_fleet_restart_pending_marker",
        "_write_marker_file",
        "_write_update_incomplete_marker",
        "_UPDATE_RUNTIME_RELOAD_MODULES",
        "_UPDATE_CRITICAL_FILES",
        "_UPDATE_CRITICAL_MODULES",
        "_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE",
    ),
}

_LAZY_COMMAND_ATTR_TO_MODULE = {
    attr: module for module, attrs in _LAZY_COMMAND_EXPORTS.items() for attr in attrs
}

# Back-compat alias: some tests and external callers import the old warn-only
# name. The kill behaviour replaced it; resolve to the new name lazily.
_LAZY_COMMAND_ALIASES = {
    "_warn_stale_dashboard_processes": (
        "hermes_cli.dashboard_procs",
        "_kill_stale_dashboard_processes",
    ),
}


def _self():
    """This module, for attribute access at call time.

    Bare-name global lookups inside this module do not go through the PEP 562
    __getattr__ below, so internal callers of the lazily re-exported names use
    _self().<name> instead. That resolves the lazy re-export on first use and
    keeps monkeypatches on hermes_cli.main.<name> working, exactly like a
    globals lookup did. ``sys`` is imported locally because some tests patch
    this module's ``sys`` attribute.
    """
    import sys as _sys

    return _sys.modules[__name__]


def __getattr__(name):
    """Defer the model-catalog and command-module imports until first read."""
    if name in _LAZY_MODEL_EXPORTS:
        from hermes_cli.models import _PROVIDER_MODELS
        # Cache on the module so subsequent accesses skip the import machinery.
        globals()[name] = _PROVIDER_MODELS
        return _PROVIDER_MODELS
    module = _LAZY_COMMAND_ATTR_TO_MODULE.get(name)
    if module is not None:
        import importlib

        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value
    alias = _LAZY_COMMAND_ALIASES.get(name)
    if alias is not None:
        import importlib

        module_name, attr = alias
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def cmd_login(args):
    """Authenticate Hermes CLI with a provider."""
    from hermes_cli.auth import login_command

    login_command(args)


def cmd_logout(args):
    """Clear provider authentication."""
    from hermes_cli.auth import logout_command

    logout_command(args)


def cmd_auth(args):
    """Manage pooled credentials."""
    from hermes_cli.auth_commands import auth_command

    auth_command(args)


def cmd_status(args):
    """Show status of all components."""
    from hermes_cli.status import show_status

    show_status(args)


def cmd_cron(args):
    """Cron job management."""
    from hermes_cli.cron import cron_command

    cron_command(args)


def cmd_sync(args):
    """Skill Sync — personal sync across devices, plus sharing with your org."""
    import json as _json

    sub = getattr(args, "sync_command", None)

    if sub in {None, ""}:
        print(
            "usage: hermes sync "
            "<status|pull|push|now|enable|disable|device|propose>\n"
            "\n"
            "Your skills, across your devices:\n"
            "  status            Show what is synced, and from where\n"
            "  pull              Pull your synced skills\n"
            "  push              Push your opted-in skills\n"
            "  now               Reconcile now: pull then push\n"
            "  enable <skill>    Include a skill in your sync\n"
            "  disable <skill>   Exclude a skill from your sync\n"
            "  device [--name N] Show or set this device's label\n"
            "\n"
            "Shared with your team:\n"
            "  propose <skill>   Share a skill with your organisation",
            file=sys.stderr,
        )
        return 1

    if sub == "device":
        from tools import skills_sync_client as ssc

        name = getattr(args, "device_name", None)
        if name is not None:
            try:
                stored = ssc.set_device_name(name)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"device label set to '{stored}'.")
            print(
                "New commits from this device will use this label; existing "
                "commits keep their previous one.",
                file=sys.stderr,
            )
            return 0
        # No --name: print the current (creating a default on first use).
        print(ssc.stable_device_id())
        return 0

    if sub == "propose":
        from tools import skills_sync_client as ssc

        name = args.name
        try:
            result = ssc.propose_skill(name, message=args.message)
        except ssc.SyncInertError as e:
            print(f"cannot share this skill: {e}", file=sys.stderr)
            return 1
        except ssc.SyncError as e:
            print(f"could not share '{name}': {e}", file=sys.stderr)
            return 1
        if result.get("proposal_pending"):
            print(
                f"Shared '{name}' with your organisation — an admin needs to "
                f"approve it (proposal #{result.get('proposal_id')}). It is "
                f"not live for the team until then."
            )
        else:
            print(f"Added '{name}' to your organisation's shared skills.")
        return 0

    if sub in {"enable", "disable"}:
        from tools.skill_usage import set_sync, is_curation_eligible

        skill = args.skill
        if not is_curation_eligible(skill):
            print(
                f"'{skill}' is not sync-eligible (bundled, hub-installed, "
                f"external, or not found). Only agent-created / user-authored "
                f"skills under ~/.hermes/skills/ can sync.",
                file=sys.stderr,
            )
            return 1
        set_sync(skill, sub == "enable")
        print(f"sync {'enabled' if sub == 'enable' else 'disabled'} for '{skill}'.")
        return 0

    from tools import skills_sync_client as ssc

    if sub == "status":
        status = ssc.sync_status()
        print(_json.dumps(status, indent=2, ensure_ascii=False))
        if status.get("org_available"):
            n = len(status.get("org_skills") or [])
            modified = status.get("org_skills_modified") or []
            print(
                f"\nOrg skills: {n} shared skill(s) from your organisation "
                f"(your role: {status.get('org_role')}). They load alongside "
                f"your own, labeled by origin, and you can edit them.",
                file=sys.stderr,
            )
            if modified:
                print(
                    f"  {len(modified)} with local edits not yet shared: "
                    f"{', '.join(modified)}\n"
                    f"  Share them back with `hermes sync propose <skill>`. "
                    f"Org updates will not overwrite them.",
                    file=sys.stderr,
                )
        elif status.get("logged_in"):
            print(
                "\nOrg skills: not applicable — this account isn't a member "
                "of a shared organisation.",
                file=sys.stderr,
            )
        if not status.get("logged_in"):
            print("\nNot logged into Nous Portal — sync is inert.", file=sys.stderr)
        elif not status.get("nous_admin"):
            print(
                "\nSync is not enabled for your account yet.",
                file=sys.stderr,
            )
        elif not status.get("feature_enabled"):
            print(
                "\nSync feature is off for this instance (set HERMES_SYNC_ENABLED=1 "
                "or config.yaml sync.enabled: true). Sync is inert.",
                file=sys.stderr,
            )
        elif not status.get("base_url"):
            print(
                "\nNo sync base URL configured (config.yaml sync.base_url or "
                "HERMES_SYNC_BASE_URL). Sync is inert.",
                file=sys.stderr,
            )
        return 0

    # pull / push / now — enforce the gate up front with a clear message.
    try:
        identity = ssc.resolve_identity()
    except ssc.SyncInertError as e:
        print(f"sync inert: {e}", file=sys.stderr)
        return 1
    if not identity.get("nous_admin"):
        print(
            "sync unavailable: not enabled for your account yet.",
            file=sys.stderr,
        )
        return 1
    if not ssc.resolve_sync_base_url():
        print(
            "sync inert: no sync base URL configured (config.yaml sync.base_url "
            "or HERMES_SYNC_BASE_URL).",
            file=sys.stderr,
        )
        return 1

    try:
        if sub == "pull":
            result = ssc.pull_skills(identity=identity)
            # Refresh the org mirror too when this account belongs to an
            # organisation (no-op otherwise), so one pull covers both.
            org_result = ssc.maybe_pull_org_skills()
            if org_result:
                n = len(org_result.get("updated") or [])
                print(
                    f"org: refreshed {n} shared skill(s) from your "
                    f"organisation.",
                    file=sys.stderr,
                )
                clashes = org_result.get("conflicted") or []
                if clashes:
                    print(
                        f"org: {len(clashes)} skill(s) have BOTH local edits "
                        f"and org updates, so they were left as-is: "
                        f"{', '.join(clashes)}\n"
                        f"     Your local version is intact. Review it, then "
                        f"either propose it or delete the local copy and pull "
                        f"again to take the org version.",
                        file=sys.stderr,
                    )
        elif sub == "push":
            result = ssc.push_skills(identity=identity, message="hermes sync push")
        elif sub == "now":
            pull_res = ssc.pull_skills(identity=identity)
            push_res = ssc.push_skills(identity=identity, message="hermes sync now")
            result = {"pull": pull_res, "push": push_res}
        else:
            print(f"Unknown sync subcommand: {sub}", file=sys.stderr)
            return 1
    except ssc.SyncError as e:
        print(f"sync failed: {e}", file=sys.stderr)
        return 1

    print(_json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_webhook(args):
    """Webhook subscription management."""
    from hermes_cli.webhook import webhook_command

    webhook_command(args)


def cmd_slack(args):
    """Slack integration helpers.

    Dispatches ``hermes slack <subcommand>``. Currently supports:
      manifest — print or write a Slack app manifest with every gateway
                 command registered as a first-class slash.
    """
    sub = getattr(args, "slack_command", None)
    if sub in {None, ""}:
        # No subcommand — print usage hint.
        print(
            "usage: hermes slack <subcommand>\n"
            "\n"
            "subcommands:\n"
            "  manifest   Generate a Slack app manifest with every gateway\n"
            "             command registered as a native slash\n"
            "\n"
            "Run `hermes slack manifest -h` for details.",
            file=sys.stderr,
        )
        return 1

    if sub == "manifest":
        from hermes_cli.slack_cli import slack_manifest_command

        status = slack_manifest_command(args)
        if status:
            raise SystemExit(status)
        return status

    print(f"Unknown slack subcommand: {sub}", file=sys.stderr)
    return 1


def cmd_kanban(args):
    """Multi-profile collaboration board."""
    from hermes_cli.kanban import kanban_command

    return kanban_command(args)


def cmd_project(args):
    """Manage projects (named, multi-folder workspaces)."""
    from hermes_cli.projects_cmd import projects_command

    return projects_command(args)


def cmd_hooks(args):
    """Shell-hook inspection and management."""
    from hermes_cli.hooks import hooks_command

    hooks_command(args)


def cmd_doctor(args):
    """Check configuration and dependencies."""
    from hermes_cli.doctor import run_doctor

    run_doctor(args)


def cmd_verify(args):
    """Detect a project's run recipe and smoke-test it."""
    from hermes_cli.verify_cmd import run_verify_command

    sys.exit(run_verify_command(args))


def cmd_security(args):
    """Dispatch `hermes security <subcmd>`."""
    sub = getattr(args, "security_command", None)
    if sub in ("audit", None):
        from hermes_cli.security_audit import cmd_security_audit

        # Default subcommand is `audit` when no subcmd is given.
        code = cmd_security_audit(args)
        sys.exit(int(code or 0))
    print(f"unknown security subcommand: {sub}", file=sys.stderr)
    sys.exit(2)


def cmd_approvals(args):
    """Dispatch `hermes approvals <subcmd>`."""
    from hermes_cli.approvals_suggest import approvals_command

    status = approvals_command(args)
    if status:
        sys.exit(status)
    return status


def cmd_dump(args):
    """Dump setup summary for support/debugging."""
    from hermes_cli.dump import run_dump

    run_dump(args)


def cmd_debug(args):
    """Debug tools (share report, etc.)."""
    from hermes_cli.debug import run_debug

    run_debug(args)


def cmd_config(args):
    """Configuration management."""
    from hermes_cli.config import config_command

    try:
        config_command(args)
    except RuntimeError as exc:
        # Safety net for the fail-closed config write guard (unparseable /
        # non-mapping / unreadable config.yaml raises RuntimeError from
        # require_readable_config_before_write). set/unset already surface
        # this per-branch; this covers migrate and future write subcommands
        # so no path ends in a raw traceback.
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_skin(args):
    """Skin management (list / use / set)."""
    from hermes_cli.skin_cmd import skin_command

    skin_command(args)


def cmd_backup(args):
    """Back up Hermes home directory to a zip file."""
    if getattr(args, "quick", False):
        from hermes_cli.backup import run_quick_backup

        run_quick_backup(args)
    else:
        from hermes_cli.backup import run_backup

        run_backup(args)


def cmd_import(args):
    """Restore a Hermes backup from a zip file."""
    from hermes_cli.backup import run_import

    run_import(args)


def _print_version_info(*, check_updates: bool = True) -> None:
    # Single source of truth for version output — shared with the
    # `hermes --version` pre-import fast path (the `version` subcommand
    # was consolidated into `--version`).
    _startup_fast.print_fast_version_info(check_updates=check_updates)


def cmd_version(args):
    """Show version (--version/-V flag)."""
    _print_version_info(check_updates=True)


def cmd_uninstall(args):
    """Uninstall Hermes Agent (or just the Chat GUI with --gui)."""
    # Machine-readable install snapshot for the desktop app's uninstall UI.
    # Must run before any TTY gate — it's called from a non-interactive child.
    if getattr(args, "gui_summary", False):
        from hermes_cli.gui_uninstall import gui_install_summary

        print(json.dumps(gui_install_summary()))
        return

    # GUI-only uninstall. The desktop app shells out to this non-interactively
    # with --yes, so only gate on a TTY when we actually need to prompt.
    if getattr(args, "gui", False):
        if not getattr(args, "yes", False):
            _require_tty("uninstall --gui")
        from hermes_cli.uninstall import run_gui_uninstall

        run_gui_uninstall(args)
        return

    # Full/keep-data uninstall. ``--yes`` runs non-interactively (the desktop
    # app's lite/full modes drive this from a detached cleanup script), so only
    # gate on a TTY when we actually need to prompt for the option + confirm.
    if not getattr(args, "yes", False):
        _require_tty("uninstall")
    from hermes_cli.uninstall import run_uninstall

    run_uninstall(args)


def _clear_bytecode_cache(root: Path) -> int:
    """Remove all __pycache__ directories under *root*.

    Stale .pyc files can cause ImportError after code updates when Python
    loads a cached bytecode file that references names that no longer exist
    (or don't yet exist) in the updated source.  Clearing them forces Python
    to recompile from the .py source on next import.

    Returns the number of directories removed.
    """
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        # Skip venv / node_modules / .git entirely
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"venv", ".venv", "node_modules", ".git", ".worktrees"}
        ]
        if os.path.basename(dirpath) == "__pycache__":
            try:
                shutil.rmtree(dirpath)
                removed += 1
            except OSError:
                pass
            dirnames.clear()  # nothing left to recurse into
    return removed


# Update pipeline lives in hermes_cli/update_cmd.py (main.py decomposition,
# mechanical move). Its names are re-exported lazily through the module-level
# __getattr__ above (see _LAZY_COMMAND_EXPORTS) so argparse wiring and test
# monkeypatches on hermes_cli.main.<name> keep resolving unchanged without
# paying the update_cmd import cost on every CLI invocation.


# ---------------------------------------------------------------------------
# Desktop build stamp — content-hash based skip logic
# ---------------------------------------------------------------------------
# The desktop Electron build is expensive.
# Unlike the web UI (which uses mtime comparison), the desktop uses a
# SHA-256 content hash of the source tree so that:
#   - ``git checkout`` / ``git pull`` that touch mtimes but not content
#     don't trigger a rebuild
#   - ``hermes update`` can unconditionally call ``hermes desktop --build-only``
#     and it will skip if nothing actually changed
#   - ``hermes desktop`` (interactive launch) skips the build when the
#     stamp matches, making repeated launches fast
#
# Stamp file: $HERMES_HOME/desktop-build-stamp.json
# Schema:
#   {
#     "contentHash": "<sha256 hex of source files>",
#     "sourceMode": true | false,
#     "builtAt": "<ISO 8601>"
#   }


# ─── Desktop stage-and-swap pack (#86443) ───────────────────────────────────
#
# electron-builder packs IN PLACE: before-pack.mjs wipes ``release/<platform>-
# unpacked`` (or the mac ``Hermes.app``) and the Electron unpack + asar + rename
# then rebuild it. Any failure after that wipe — corrupt cached zip, blocked
# download, missing dep, disk full — leaves the user with NO app, and
# ``hermes update`` used to report "partially complete" over an empty
# release/. Fix the class, not the predicate: build into a STAGING output
# dir next to release/, verify the staged result, and only then swap it over
# the live tree with renames. On any failure the live app is untouched.


# ─── Desktop exe integrity gate (#69179) ────────────────────────────────────
#
# The desktop self-update chain (Desktop → hermes-setup --update →
# `hermes update` → `hermes desktop --build-only` → relaunch) rebuilds
# Hermes.exe on the end user's machine and used to verify only that the file
# EXISTS before declaring success. A corrupt cached Electron zip whose
# extraction produced a truncated electron.exe, an interrupted rcedit resource
# rewrite, a disk-full pack, or a wrong-arch unpacked tree therefore shipped a
# broken binary that Windows refuses to load ("This app can't run on your
# computer" / 此应用无法在你的电脑上运行). These helpers parse the PE header —
# no signature infrastructure required — so a structurally broken or
# wrong-architecture Hermes.exe is caught BEFORE the updater replaces the
# working app, and the previous build can be restored from the .bak tree that
# apps/desktop/scripts/before-pack.mjs now preserves.


# Dashboard process-hygiene helpers live in hermes_cli/dashboard_procs.py
# (main.py decomposition, mechanical move). Re-exported lazily through the
# module-level __getattr__ above so callers and test monkeypatches on
# hermes_cli.main.<name> keep resolving unchanged.


# Back-compat alias: some tests and any external callers may import the old
# warn-only name.  The new behaviour (kill stale processes) replaces it.
# Resolved lazily via _LAZY_COMMAND_ALIASES near the module __getattr__.


# =========================================================================
# Fork detection and upstream management for `hermes update`
# =========================================================================


def cmd_update(args):
    """Update Hermes Agent to the latest version.

    Thin wrapper around ``_cmd_update_impl``: installs hangup protection,
    runs the update, then restores stdio on the way out (even on
    ``sys.exit`` or unhandled exceptions).
    """
    from hermes_cli.config import (
        is_managed,
        managed_error,
    )

    if is_managed():
        managed_error("update Hermes Agent")
        return

    # --plan is read-only and deployment-kind aware, so it runs BEFORE the
    # docker/nix/apt refusal gates: on an image-managed or package-managed
    # install the plan itself reports "not updatable in place" plus the
    # right mechanism — strictly more useful than the bare refusal text.
    if getattr(args, "plan", False):
        # Read-only plan phase (#91277 Phase 2): inventory every running
        # Hermes runtime across profiles, its supervisor, and its running
        # code version — without mutating anything. Safe on a live fleet.
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            print_update_plan,
        )

        print_update_plan(collect_runtime_inventory())
        return

    # Image-managed / package-managed admission gate (#91277 Phase 3): one
    # shared decision for every mutation surface. Consults the baked image
    # provenance marker first (authoritative, fail-closed on malformed),
    # then the pre-existing docker/nix/apt heuristics. Prints the real
    # update command, records a `refused` receipt so fleet tooling sees the
    # blocked attempt, and exits 2 (refused-by-contract, distinct from
    # exit 1 errors).
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(PROJECT_ROOT)
    if refusal is not None:
        print(refusal.message)
        record_refusal_receipt(refusal)
        sys.exit(2)

    if getattr(args, "check", False):
        # --check honors --branch so the "any new commits?" answer matches
        # what a subsequent `hermes update --branch=<x>` would actually pull.
        branch = _resolve_update_branch(args)
        _self()._cmd_update_check(
            branch=branch,
            branch_explicit=bool(getattr(args, "branch", None)),
        )
        return

    gateway_mode = getattr(args, "gateway", False)

    # Protect against mid-update terminal disconnects (SIGHUP) and tolerate
    # writes to a closed stdout.  No-op in gateway mode.  See
    # _install_hangup_protection for rationale.
    _update_io_state = _install_hangup_protection(gateway_mode=gateway_mode)
    # Cross-process mutual exclusion. The dashboard's Update button spawns
    # this same command detached, and the desktop hands off to the Tauri
    # updater / install-mode bootstrap — all three mutate one checkout. Two of
    # them running together rewrite source under a live interpreter and strand
    # the tree half-updated. Share the marker the Tauri updater and Electron
    # already use rather than inventing a second lock.
    from hermes_cli.update_lock import (
        UPDATE_EXIT_CONCURRENT,
        UpdateLock,
        describe_holder,
    )

    _update_lock = UpdateLock()
    if not _update_lock.acquire():
        print(describe_holder(_update_lock.holder))
        _finalize_update_output(_update_io_state)
        sys.exit(UPDATE_EXIT_CONCURRENT)

    # Exit code for the Windows hand-off child's hard exit (see finally).
    # None = not a SystemExit-shaped outcome; real exceptions keep the
    # normal raise path so their traceback still prints.
    _update_handoff_exit_code: int | None = None
    try:
        _self()._cmd_update_impl(args, gateway_mode=gateway_mode)
    except SystemExit as _update_exit:
        # Receipt boundary (#91283 review): the impl has many early
        # sys.exit paths (concurrent-instance preflight, venv-holder
        # refusal, head-pinned no-op, fetch failure) that never reach an
        # inner finalize. Persist any still-open receipt with the real
        # exit code, then let the exit proceed unchanged. No-op when an
        # inner path already finalized (exactly-once by construction).
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            _code = _update_exit.code if isinstance(_update_exit.code, int) else 1
            finalize_pending_update_receipt(_code, f"sys.exit({_code})")
        except Exception:
            pass
        _update_handoff_exit_code = (
            _update_exit.code if isinstance(_update_exit.code, int) else 0
        )
        raise
    except BaseException as _update_exc:
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            finalize_pending_update_receipt(
                1, f"{type(_update_exc).__name__}: {_update_exc}"
            )
        except Exception:
            pass
        raise
    else:
        try:
            from hermes_cli.update_receipt import finalize_pending_update_receipt

            finalize_pending_update_receipt(0, "completed at command boundary")
        except Exception:
            pass
        _update_handoff_exit_code = 0
    finally:
        _update_lock.release()
        _finalize_update_output(_update_io_state)
        # Windows hand-off child (#93581): the re-exec'd venv child cannot
        # rely on graceful interpreter shutdown — a leftover non-daemon
        # thread from the update tail keeps the console busy long after
        # the receipt is durable (success, exit 0, "completed at command
        # boundary"), freezing the PowerShell window for minutes. By this
        # point every durable step is done (receipt finalized above, lock
        # released, stdio restored), so on the hand-off path only, flush
        # and exit hard instead of waiting for the interpreter to unwind
        # — the same treatment #79040's cron workaround applies. No-op on
        # every non-hand-off invocation: the marker env is set solely by
        # _reexec_dependency_sync_off_windows_shim when it spawns the child.
        if _update_handoff_exit_code is not None and os.environ.get(_UPDATE_REEXEC_ENV) == "1":
            logger.debug(
                "Update hand-off child %s exiting via os._exit(%s)",
                os.getpid(), _update_handoff_exit_code,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(_update_handoff_exit_code)


def _coalesce_session_name_args(argv: list) -> list:
    """Join unquoted multi-word session names after -c/--continue and -r/--resume.

    When a user types ``hermes -c Pokemon Agent Dev`` without quoting the
    session name, argparse sees three separate tokens.  This function merges
    them into a single argument so argparse receives
    ``['-c', 'Pokemon Agent Dev']`` instead.

    Tokens are collected after the flag until we hit another flag (``-*``)
    or a known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat",
        "model",
        "gateway",
        "setup",
        "whatsapp",
        "whatsapp-cloud",
        "login",
        "logout",
        "auth",
        "status",
        "cron",
        "doctor",
        "config",
        "pairing",
        "skills",
        "tools",
        "mcp",
        "sessions",
        "insights",
        "update",
        "uninstall",
        "profile",
        "dashboard",
        "serve",
        "desktop",
        "gui",
        "honcho",
        "claw",
        "plugins",
        "security",
        "acp",
        "webhook",
        "peer",
        "memory",
        "dump",
        "debug",
        "backup",
        "import",
        "completion",
        "logs",
    }
    _SESSION_FLAGS = {"-c", "--continue", "-r", "--resume"}

    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _SESSION_FLAGS:
            result.append(token)
            i += 1
            # Collect subsequent non-flag, non-subcommand tokens as one name
            parts: list = []
            while (
                i < len(argv)
                and not argv[i].startswith("-")
                and argv[i] not in _SUBCOMMANDS
            ):
                parts.append(argv[i])
                i += 1
            if parts:
                result.append(" ".join(parts))
        else:
            result.append(token)
            i += 1
    return result


from hermes_cli.profile_cmd import cmd_profile, _render_distribution_plan  # noqa: E402,F401  (re-export)


def cmd_dashboard(args):
    """Start the web UI server, or (with --stop/--status) manage running ones."""
    _token_file = getattr(args, "ssh_session_token_file", None)
    if _token_file and (
        getattr(args, "status", False) or getattr(args, "stop", False)
    ):
        raise SystemExit("--ssh-session-token-file cannot be used with --status or --stop")

    # --status: report running dashboards and exit, no deps needed.
    if getattr(args, "status", False):
        count = _report_dashboard_status()
        sys.exit(0 if count == 0 else 0)  # status is informational, always 0

    # --stop: kill any running dashboards and exit, no deps needed.
    if getattr(args, "stop", False):
        pids = _find_stale_dashboard_pids()
        if not pids:
            print("No hermes dashboard processes running.")
            sys.exit(0)
        # Reuse the same SIGTERM-grace-SIGKILL path used after `hermes update`.
        _self()._kill_stale_dashboard_processes(reason="requested via --stop")
        # _kill_stale_dashboard_processes prints outcomes itself.  Exit 0 if
        # we killed at least one, 1 if they were all unkillable.
        remaining = _find_stale_dashboard_pids()
        sys.exit(1 if remaining else 0)

    # `serve` is the headless backend: no UI build, no SPA mount, neutral
    # ready sentinel. Resolved once and threaded through the re-exec, the
    # build gate, and start_server.
    _headless_backend = getattr(args, "headless_backend", False)
    # `hermes serve` is headless/non-interactive: fail closed on a corrupt
    # config.yaml instead of silently starting on defaults where provider
    # auto-detection can adopt unnamed .env credentials (issue #81952).
    # Same policy + escape hatch as _guard_noninteractive_user_config.
    if _headless_backend:
        from hermes_cli.config import (
            InvalidUserConfigError,
            require_parseable_user_config,
        )

        try:
            require_parseable_user_config(
                ignore_user_config=bool(getattr(args, "ignore_user_config", False))
            )
        except InvalidUserConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    _ssh_owner_nonce = getattr(args, "ssh_owner_nonce", None)
    if _ssh_owner_nonce and not re.fullmatch(r"[0-9a-f]{16}", _ssh_owner_nonce):
        raise SystemExit("--ssh-owner-nonce must be 16 lowercase hex characters")
    _ssh_session_token = None
    if _token_file and not _headless_backend:
        raise SystemExit("--ssh-session-token-file is only valid with hermes serve")

    # ── Sanitize Desktop-inherited env that hijacks a standalone launch ─
    # Desktop Electron spawns its backend with HERMES_DESKTOP=1 plus
    # HERMES_WEB_DIST=<packaged app.asar[/unpacked]/dist> (and often
    # HERMES_SERVE_HEADLESS=1 on the serve path). A shell that inherits
    # those vars then runs `hermes dashboard` would otherwise:
    #   - serve the desktop renderer → "Desktop IPC bridge is unavailable"
    #     (issue #52945), or
    #   - disable the SPA via inherited HERMES_SERVE_HEADLESS.
    # Only strip Electron-packaged WEB_DIST contamination — caller-managed
    # HERMES_WEB_DIST overrides (dev / custom builds) must still work.
    # The desktop-spawned backend itself (HERMES_DESKTOP=1) keeps its dist.
    # Intentionally headless `serve` re-sets HERMES_SERVE_HEADLESS below.
    if os.environ.get("HERMES_DESKTOP") != "1":
        _inherited_web_dist = os.environ.get("HERMES_WEB_DIST", "")
        if _is_electron_packaged_web_dist(_inherited_web_dist):
            os.environ.pop("HERMES_WEB_DIST", None)
    if not _headless_backend:
        os.environ.pop("HERMES_SERVE_HEADLESS", None)

    _route_named_profile_dashboard(args, _headless_backend, _ssh_owner_nonce, _token_file)

    # Apply the final process/profile policy after dashboard routing, but before
    # importing the web server or opening dashboard state. Applying it before a
    # named-profile re-exec could leak that profile's higher limit into the
    # machine/default dashboard, whose lower policy intentionally cannot undo it.
    # This also covers Desktop SSH's isolated `serve` child, which does not route.
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    if _token_file:
        _ssh_session_token = _read_ssh_session_token_file(_token_file)

    # Attach gui.log early so dashboard startup/build failures are captured in
    # the same logs directory as every other Hermes surface.
    try:
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")
    except Exception:
        pass

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        print("Web UI dependencies not installed (need fastapi + uvicorn).")
        print(
            f"Re-install the package into this interpreter so metadata updates apply:\n"
            f"  cd {PROJECT_ROOT}\n"
            f"  {sys.executable} -m pip install -e .\n"
            "If `pip` is missing in this venv, use:  uv pip install -e ."
        )
        print(f"Import error: {e}")
        sys.exit(1)

    # Seed bundled skills on first dashboard launch so the desktop GUI's
    # skills picker / agent skill discovery sees the bundled library.
    # cmd_chat does this in its own pre-dispatch block; the dashboard
    # backend is the desktop's primary entrypoint and needs the same.
    _sync_bundled_skills_quietly()

    # Bridge terminal.* config into the TERMINAL_* env vars for THIS process,
    # mirroring the CLI (cli.py env_mappings) and gateway (gateway/run.py
    # _terminal_env_map) startup bridges. The dashboard/serve backend runs
    # agents in-process (tui_gateway.ws → server._make_agent) and ticks cron
    # jobs itself when desktop-spawned — without this bridge those consumers
    # saw an unset TERMINAL_ENV and silently ran every command on the host
    # even when config.yaml selects `terminal.backend: docker`
    # (#63141, #54449, #61115, #65696). PTY chat spawns already bridge their
    # child env copy; this covers the in-process consumers.
    try:
        from hermes_cli.config import apply_terminal_config_to_env

        apply_terminal_config_to_env()
    except Exception:
        logger.debug("terminal config → env bridge failed for dashboard/serve",
                     exc_info=True)

    _resolve_dashboard_web_dist(args, _headless_backend)
    # Discover and load plugins so any DashboardAuthProvider plugin
    # (e.g. plugins/dashboard_auth/nous) registers BEFORE start_server's
    # fail-closed gate check runs. The top-level argparse setup skips
    # plugin discovery for built-in subcommands like ``dashboard`` to
    # save ~500ms startup; we have to trigger it explicitly here because
    # the dashboard's server-side runtime depends on plugin-registered
    # providers (image_gen, web, dashboard_auth, …).
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception as exc:
        # Discovery failures must not block dashboard startup outright —
        # log and proceed; the gate's fail-closed branch will surface
        # the missing-provider state if it matters.
        print(f"⚠ Plugin discovery failed: {exc}", file=sys.stderr)

    # Desktop chat uses the dashboard's in-process /api/ws gateway, which builds
    # agents via tui_gateway.server._make_agent.  That path only snapshots the
    # tool registry — it never starts MCP discovery (the stdio TUI does that in
    # tui_gateway/entry.py, which the dashboard process doesn't run).  Without
    # this, a profile's configured MCP servers never connect, so desktop
    # sessions show no MCP tools.  Spawn discovery in the background here so a
    # slow/dead server can't block dashboard startup.
    #
    # Desktop-spawned headless backends start it AFTER the socket binds
    # instead (start_server's ready path): the thread's first act is the
    # ~350ms `mcp` SDK import, which holds the GIL against the main thread's
    # own web_server import and pushes the READY sentinel — and every
    # renderer paint behind it — back by that much. The Desktop can't issue
    # an agent turn until its WebSocket is up anyway, and _make_agent's
    # bounded wait_for_mcp_discovery + the late-binding refresh cover a
    # server that is still connecting when the first turn lands.
    _mcp_discovery_after_bind = _headless_backend and os.environ.get("HERMES_DESKTOP") == "1"
    if not _mcp_discovery_after_bind:
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="dashboard-mcp-discovery",
            )
        except Exception:
            logger.debug(
                "Background MCP tool discovery failed at dashboard startup",
                exc_info=True,
            )

    from hermes_cli.web_server import start_server

    # Interactive auth setup: if this bind will engage the auth gate but no
    # provider is registered yet, offer to configure one here (TTY only)
    # instead of hard-failing inside start_server. Non-interactive callers
    # (Docker/s6, CI, --no-open pipelines) fall through to start_server's
    # fail-closed SystemExit unchanged.
    _maybe_setup_dashboard_auth_interactively(args)

    # The in-browser Chat tab (the embedded TUI over PTY/WebSocket) is always
    # available — the desktop app and the dashboard's own Chat tab both rely on
    # the `/api/ws` + `/api/pty` sockets, so there is no reason to gate them.
    start_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        allow_public=getattr(args, "insecure", False),
        initial_profile=getattr(args, "open_profile", "") or "",
        headless=_headless_backend,
        ssh_session_token=_ssh_session_token,
        ssh_owner_nonce=_ssh_owner_nonce,
        start_mcp_discovery_after_bind=_mcp_discovery_after_bind,
    )


def cmd_dashboard_register(args):
    """Register a self-hosted dashboard OAuth client with Nous Portal."""
    from hermes_cli.dashboard_register import cmd_dashboard_register as _impl

    _impl(args)


def cmd_gateway_enroll(args):
    """Enroll a self-hosted gateway with a relay connector."""
    from hermes_cli.gateway_enroll import cmd_gateway_enroll as _impl

    _impl(args)


def cmd_completion(args, parser=None):
    """Print shell completion script."""
    from hermes_cli.completion import generate_bash, generate_zsh, generate_fish

    shell = getattr(args, "shell", "bash")
    if shell == "zsh":
        print(generate_zsh(parser))
    elif shell == "fish":
        print(generate_fish(parser))
    else:
        print(generate_bash(parser))


def cmd_prompt_size(args):
    """Show a byte/char breakdown of the system prompt + tool schemas."""
    from hermes_cli.prompt_size import cmd_prompt_size as _impl

    _impl(args)


def cmd_logs(args):
    """View and filter Hermes log files."""
    from hermes_cli.logs import tail_log, list_logs

    log_name = getattr(args, "log_name", "agent") or "agent"

    if log_name == "list":
        list_logs()
        return

    tail_log(
        log_name,
        num_lines=getattr(args, "lines", 50),
        follow=getattr(args, "follow", False),
        level=getattr(args, "level", None),
        session=getattr(args, "session", None),
        since=getattr(args, "since", None),
        component=getattr(args, "component", None),
    )


def cmd_console(args):
    """Open the safe Hermes command console."""
    from hermes_cli.console_engine import run_console_repl

    return run_console_repl()


# Top-level subcommands that argparse knows about WITHOUT running plugin
# discovery.  Used to short-circuit eager plugin imports (which can take
# 500ms+ pulling in google.cloud.pubsub_v1, aiohttp, grpc, etc.) when the
# user's invocation clearly doesn't need any plugin-registered subcommand.
#
# Keep this in sync with the ``subparsers.add_parser("NAME", ...)`` calls
# below in ``main()``. Missing an entry here only costs a one-time
# discovery; extra entries here would let a plugin command silently fail
# to parse.
_BUILTIN_SUBCOMMANDS = frozenset(
    {
        "acp", "approvals", "auth", "backup", "bundles", "checkpoints", "claw", "completion",
        "computer-use",
        "config", "console", "cron", "curator", "dashboard", "serve", "debug", "doctor",
        "dump", "egress", "fallback", "gateway", "hooks", "import", "import-agent", "insights",
        "gui", "desktop", "kanban", "login", "logout", "logs", "lsp", "mcp", "memory", "migrate", "moa",
        "journey", "memory-graph", "learning",
        "model", "monitoring", "pairing", "pause", "peer", "pets", "plugins", "portal", "profile",
        "project", "proxy",
        "prompt-size",
        "resume",
        "send", "sessions", "setup",
        "skin", "skills", "slack", "status", "sync", "tools", "uninstall", "update",
        "webhook", "whatsapp", "whatsapp-cloud", "worktree", "chat", "secrets", "security",
        "browser",
        "verify",
        # Help-ish invocations — plugin commands not being listed in
        # top-level --help is an acceptable trade-off for skipping an
        # expensive eager import of every bundled plugin module.
        "help",
    }
)


def _first_positional_argv() -> str | None:
    """Return the first non-flag, non-flag-value token in ``sys.argv[1:]``.

    Used by ``main()`` to decide whether plugin discovery has to run at
    argparse-setup time. Handles common invocations like
    ``hermes -m gpt5 --provider openai chat "msg"`` by skipping the
    values attached to known top-level flags.

    Does NOT fully simulate argparse — unknown ``--foo=bar`` / ``--foo
    bar`` flags degrade gracefully (``bar`` may be wrongly classified as
    a positional, which at worst forces a one-time plugin discovery).
    """
    from hermes_cli._parser import top_level_value_flag_sets

    required_value_flags, optional_value_flags = top_level_value_flag_sets()
    value_flags = required_value_flags | optional_value_flags
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            # Everything after ``--`` is positional.
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if tok.startswith("-"):
            # ``--flag=value`` carries its value inline — single token.
            if "=" in tok:
                i += 1
                continue
            if tok in value_flags and i + 1 < len(argv):
                i += 2
                continue
            i += 1
            continue
        return tok
    return None


def _plugin_cli_discovery_needed() -> bool:
    """True when the CLI might be invoking a plugin-registered subcommand.

    Returning False lets ``main()`` skip plugin discovery entirely during
    argparse setup, saving ~500-650ms per invocation for users whose
    enabled plugins don't contribute any CLI command.
    """
    first = _first_positional_argv()
    if first is None:
        # Bare ``hermes`` or only flags → defaults to ``chat``.
        return False
    if first in _BUILTIN_SUBCOMMANDS:
        return False
    # Unknown token — could be a plugin subcommand, OR a chat prompt
    # starting with a non-flag word. Either way we need discovery: if it
    # IS a plugin command, argparse needs the subparser; if it's a chat
    # prompt, argparse will route it via positional handling and the
    # extra discovery cost is amortized over a full agent run anyway.
    return True


def _resolve_deferred_platform_cli_command(command_name: str | None) -> None:
    """Materialize the deferred platform whose top-level CLI command matches.

    Bundled platform plugins are cheap-registered as *deferred* entries to
    avoid importing every gateway SDK during normal startup. A platform that
    registers a top-level ``hermes <name>`` command (e.g. Photon ->
    ``ctx.register_cli_command(name="photon", ...)``) only runs that side
    effect when its module is imported. On the unknown-top-level-command slow
    path, ``discover_plugins()`` records the deferred loader but does not
    import it, so the CLI registration never happens and ``hermes photon``
    fails with argparse ``invalid choice`` (issue #54678).

    Resolving only the platform whose name matches the first positional token
    keeps normal startup cheap while making the targeted command available.
    """
    if not command_name:
        return
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.get(command_name)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Deferred platform CLI resolution failed for %s: %s",
            command_name,
            exc,
        )


_AGENT_COMMANDS = {None, "chat", "acp", "rl"}
_AGENT_SUBCOMMANDS = {
    "cron": ("cron_command", {"run", "tick"}),
    "gateway": ("gateway_command", {"run"}),
    "mcp": ("mcp_action", {"serve"}),
}


def _is_tui_chat_launch(args) -> bool:
    return bool(getattr(args, "tui", False) or os.environ.get("HERMES_TUI") == "1")


def _command_has_dedicated_mcp_startup(args) -> bool:
    if args.command == "acp":
        return True
    if args.command == "gateway" and getattr(args, "gateway_command", None) == "run":
        return True
    if args.command == "cron" and getattr(args, "cron_command", None) in {"run", "tick"}:
        return True
    return False


def _should_background_mcp_startup(args) -> bool:
    if _is_tui_chat_launch(args):
        return False
    return args.command in {None, "chat", "rl"}


def _prepare_agent_startup(args) -> None:
    """Discover plugins/MCP/hooks for commands that can run an agent turn."""
    # --yolo: chokepoint guarantee that HERMES_YOLO_MODE is set before ANY
    # plugin/tool discovery below imports tools.approval, which freezes
    # _YOLO_MODE_FROZEN at import time (PR #7994 security design).  main()'s
    # dispatch path also sets this earlier, but _prepare_agent_startup() is
    # reachable from other launchers too (e.g. the Termux fast-CLI path),
    # so the guarantee lives here where the import is actually triggered
    # (#60328).
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _apply_safe_mode(args)
    _apply_user_config_bypass(args)
    _guard_noninteractive_user_config(args)

    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    if not (
        args.command in _AGENT_COMMANDS
        or (_sub_attr and getattr(args, _sub_attr, None) in _sub_set)
    ):
        return

    _accept_hooks = bool(getattr(args, "accept_hooks", False))
    if not _is_tui_chat_launch(args):
        # The TUI backend process does its own plugin discovery; the launcher
        # only spawns Node, so discovery here would be thrown-away work.
        try:
            from hermes_cli.plugins import start_background_plugin_discovery

            # Discovery runs in a daemon thread so its ~150ms of manifest
            # scanning + plugin imports overlaps the rest of startup (cli /
            # prompt_toolkit imports, worktree git calls). Correctness is
            # unchanged: every synchronous reader goes through
            # discover_plugins(), which joins this thread first — including
            # the discover_plugins() call model_tools makes at import time,
            # which happens before any tool list is built.
            start_background_plugin_discovery()
        except Exception:
            logger.warning(
                "plugin discovery failed at CLI startup",
                exc_info=True,
            )
    _run_inline_mcp_discovery = True
    if _is_tui_chat_launch(args):
        # The TUI launcher hands off to a dedicated startup path that already
        # backgrounds MCP discovery with a bounded join before the first tool
        # snapshot.
        _run_inline_mcp_discovery = False
    elif _command_has_dedicated_mcp_startup(args):
        # These entrypoints already do their own MCP startup later on the real
        # runtime path (gateway executor, ACP launcher, cron job runner).
        _run_inline_mcp_discovery = False
    elif _should_background_mcp_startup(args):
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="cli-mcp-discovery",
            )
        except Exception:
            logger.debug(
                "Background MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
        _run_inline_mcp_discovery = False
    if _run_inline_mcp_discovery:
        try:
            # MCP tool discovery remains synchronous for entrypoints that do
            # not own a later bounded/executor startup path.
            from tools.mcp_tool import discover_mcp_tools

            discover_mcp_tools()
        except Exception:
            logger.debug(
                "MCP tool discovery failed at CLI startup",
                exc_info=True,
            )
    try:
        from hermes_cli.config import load_config
        from agent.shell_hooks import register_from_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)

        from agent.outbound_webhooks import (
            register_from_config as register_outbound_webhooks,
        )

        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug(
            "shell-hook registration failed at CLI startup",
            exc_info=True,
        )


def _apply_safe_mode(args) -> None:
    if not getattr(args, "safe_mode", False):
        return
    os.environ["HERMES_SAFE_MODE"] = "1"
    os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
    os.environ["HERMES_IGNORE_RULES"] = "1"


def _apply_user_config_bypass(args) -> None:
    """Apply the explicit config bypass before any startup config reads."""
    if getattr(args, "ignore_user_config", False):
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"


def _guard_noninteractive_user_config(args) -> None:
    """Fail closed before a non-interactive invocation initializes providers."""
    if getattr(args, "_noninteractive_config_validated", False):
        return

    is_noninteractive = (
        bool(getattr(args, "oneshot", None))
        or bool(getattr(args, "query", None))
    )
    if not is_noninteractive:
        return

    from hermes_cli.config import (
        InvalidUserConfigError,
        require_parseable_user_config,
    )

    try:
        require_parseable_user_config(
            ignore_user_config=bool(
                getattr(args, "ignore_user_config", False)
                or getattr(args, "safe_mode", False)
            )
        )
    except InvalidUserConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    setattr(args, "_noninteractive_config_validated", True)


def _set_chat_arg_defaults(args) -> None:
    """Fill the chat-parser attrs cmd_chat reads when chat was not parsed."""
    for attr, default in [
        ("query", None),
        ("model", None),
        ("provider", None),
        ("toolsets", None),
        ("verbose", False),
        ("resume", None),
        ("continue_last", None),
        ("worktree", False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)


def _run_oneshot_from_args(args) -> None:
    """Top-level --oneshot / -z: single-shot mode, stdout = final response only.

    Bypasses cli.py entirely; _run_and_exit_oneshot never returns.
    """
    _confirm_startup_expensive_model_override(args)
    _run_and_exit_oneshot(
        args.oneshot,
        model=getattr(args, "model", None),
        provider=getattr(args, "provider", None),
        toolsets=getattr(args, "toolsets", None),
        skills=getattr(args, "skills", None),
        usage_file=getattr(args, "usage_file", None),
    )


def _try_fast_serve_launch() -> bool:
    """Dispatch an unambiguous built-in ``serve`` without the full CLI tree.

    Desktop launches this exact command on every cold start. Building parsers
    for unrelated Hermes commands performs thousands of filesystem-backed
    translation lookups on Windows even though none of those commands are
    usable in this process. Unknown or globally-scoped arguments fall back to
    normal parsing so compatibility and error reporting remain unchanged.
    """
    if os.environ.get("HERMES_DISABLE_FAST_SERVE_LAUNCH") == "1":
        return False

    argv = sys.argv[1:]
    if not argv or argv[0] != "serve" or "-h" in argv or "--help" in argv:
        return False

    # Container routing is top-level policy and must run before host dispatch.
    try:
        from hermes_cli.config import get_container_exec_info

        if get_container_exec_info():
            return False
    except Exception:
        return False

    parser = build_serve_parser(
        cmd_dashboard=cmd_dashboard,
        add_help=False,
        exit_on_error=False,
    )
    try:
        args, unknown = parser.parse_known_args(argv[1:])
    except (argparse.ArgumentError, ValueError):
        return False
    if unknown:
        return False

    cmd_dashboard(args)
    return True


def _try_fast_chat_launch() -> bool:
    """Fast path for unambiguous interactive chat launches (all hosts).

    ``hermes`` / ``hermes -w -s foo --yolo`` / ``hermes chat`` don't need the
    full argparse tree: building all ~40 subcommand parsers costs ~140ms of
    pure-Python argparse setup plus their module imports, none of which the
    chat path uses. Parse the lightweight top-level/chat parser instead and
    dispatch straight to ``cmd_chat``.

    Bails out (returns False) whenever the invocation is not certainly a
    chat launch — a subcommand positional, ``--help``, unknown flags — so
    every other path still goes through the full parser unchanged. Mirrors
    ``_try_termux_fast_cli_launch`` minus the Termux-specific deferred
    startup; kept separate so phone-tuned behavior doesn't leak to desktops.
    """
    if os.environ.get("HERMES_DISABLE_FAST_CHAT_LAUNCH") == "1":
        return False
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    # Container-aware routing must win: when NixOS container mode is
    # active, EVERY invocation is forwarded into the managed container.
    try:
        from hermes_cli.config import get_container_exec_info
        if get_container_exec_info():
            return False
    except Exception:
        return False
    # TUI launches have their own startup path (bounded MCP joins etc.) —
    # keep them on full dispatch outside Termux.
    if _wants_tui_early(argv):
        return False
    if _first_positional_argv() not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    try:
        args, unknown = parser.parse_known_args(_coalesce_session_name_args(argv))
    except SystemExit:
        return False
    if unknown:
        # Flags the light parser doesn't know — could belong to a plugin
        # subcommand or a newer full-parser flag. Fall back to full dispatch.
        return False
    if getattr(args, "version", False):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False

    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _run_oneshot_from_args(args)

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    _set_chat_arg_defaults(args)
    cmd_chat(args)
    return True


def _try_termux_fast_cli_launch() -> bool:
    """Run obvious Termux non-TUI chat/oneshot/version paths on a light parser."""
    if not _is_termux_startup_environment():
        return False
    if os.environ.get("HERMES_TERMUX_DISABLE_FAST_CLI") == "1":
        return False

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    # Let the TUI fast path (or full dispatch) handle anything that resolves to
    # the TUI — explicit --tui/env or display.interface=tui. `--cli` forces this
    # to stay False so the classic fast path still runs.
    if _wants_tui_early(argv):
        return False

    if _startup_fast.is_termux_fast_version_argv(argv):
        _print_version_info(check_updates=True)
        return True

    first = _first_positional_argv()
    has_oneshot = any(
        arg == "-z" or arg == "--oneshot" or arg.startswith("--oneshot=")
        for arg in argv
    )
    if not has_oneshot and first not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    args = parser.parse_args(_coalesce_session_name_args(argv))

    if getattr(args, "version", False):
        _print_version_info(check_updates=True)
        return True

    if getattr(args, "oneshot", None):
        _prepare_agent_startup(args)
        _run_oneshot_from_args(args)

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    if args.command in {None, "chat"}:
        _set_chat_arg_defaults(args)
        interactive_prompt = not getattr(args, "query", None) and not getattr(args, "image", None)
        if interactive_prompt:
            # Bare Termux CLI should reach the prompt first and do agent-only
            # discovery on the first submitted turn instead of before input.
            setattr(args, "compact", True)
            os.environ["HERMES_DEFER_AGENT_STARTUP"] = "1"
            os.environ["HERMES_FAST_STARTUP_BANNER"] = "1"
            if getattr(args, "accept_hooks", False):
                os.environ["HERMES_ACCEPT_HOOKS"] = "1"
        else:
            _prepare_agent_startup(args)
        cmd_chat(args)
        return True

    return False


def _try_termux_fast_tui_launch() -> bool:
    """Launch obvious Termux TUI invocations before building every subparser.

    `hermes --tui` is the hot path on phones. The full parser setup imports
    command modules for model, fallback, migrate, kanban, bundles, plugins,
    etc. even though the TUI immediately execs Node. On Termux only, parse the
    lightweight top-level/chat parser and hand off to ``cmd_chat`` when the
    invocation is unambiguously the built-in TUI/chat path.
    """
    if not _is_termux_startup_environment():
        return False

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return False

    wants_tui = _wants_tui_early(sys.argv[1:])
    if not wants_tui:
        return False

    first = _first_positional_argv()
    if first not in {None, "chat"}:
        return False

    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    args = parser.parse_args(_coalesce_session_name_args(sys.argv[1:]))

    # Preserve top-level behaviours whose semantics are not "launch chat/TUI".
    if getattr(args, "version", False) or getattr(args, "oneshot", None):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False
    if not _resolve_use_tui(args):
        return False

    cmd_chat(args)
    return True


def cmd_memory(args):
    sub = getattr(args, "memory_command", None)
    if sub == "off":
        from hermes_cli.config import load_config, save_config

        config = load_config()
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
    elif sub == "reset":
        from hermes_constants import get_hermes_home, display_hermes_home

        mem_dir = get_hermes_home() / "memories"
        target = getattr(args, "target", "all")
        files_to_reset = []
        if target in {"all", "memory"}:
            files_to_reset.append(("MEMORY.md", "agent notes"))
        if target in {"all", "user"}:
            files_to_reset.append(("USER.md", "user profile"))

        # Check what exists
        existing = [
            (f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()
        ]
        if not existing:
            print(
                f"\n  Nothing to reset — no memory files found in {display_hermes_home()}/memories/\n"
            )
            return

        print("\n  This will permanently erase the following memory files:")
        for f, desc in existing:
            path = mem_dir / f
            size = path.stat().st_size
            print(f"    ◆ {f} ({desc}) — {size:,} bytes")

        if not getattr(args, "yes", False):
            try:
                answer = input("\n  Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                return
            if answer != "yes":
                print("  Cancelled.\n")
                return

        for f, desc in existing:
            (mem_dir / f).unlink()
            print(f"  ✓ Deleted {f} ({desc})")

        print(
            "\n  Memory reset complete. New sessions will start with a blank slate."
        )
        print(f"  Files were in: {display_hermes_home()}/memories/\n")
    else:
        from hermes_cli.memory_setup import memory_command

        memory_command(args)


def cmd_acp(args):
    """Launch Hermes Agent as an ACP server."""
    try:
        from acp_adapter.entry import main as acp_main

        acp_argv = []
        if getattr(args, "acp_version", False):
            acp_argv.append("--version")
        if getattr(args, "check", False):
            acp_argv.append("--check")
        if getattr(args, "setup", False):
            acp_argv.append("--setup")
        if getattr(args, "setup_browser", False):
            acp_argv.append("--setup-browser")
        if getattr(args, "assume_yes", False):
            acp_argv.append("--yes")
        acp_main(acp_argv)
    except ImportError:
        print("ACP dependencies not installed.", file=sys.stderr)
        print("Install them with:  pip install -e '.[acp]'", file=sys.stderr)
        sys.exit(1)


def cmd_tools(args):
    action = getattr(args, "tools_action", None)
    if action in {"list", "disable", "enable"}:
        from hermes_cli.tools_config import tools_disable_enable_command

        tools_disable_enable_command(args)
    elif action == "post-setup":
        from hermes_cli.tools_config import run_post_setup_command

        sys.exit(run_post_setup_command(args))
    else:
        _require_tty("tools")
        from hermes_cli.tools_config import tools_command

        tools_command(args)


def cmd_insights(args):
    db = None
    try:
        from hermes_state import SessionDB
        from agent.insights import InsightsEngine

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=args.days, source=args.source)
        print(engine.format_terminal(report))
    except Exception as e:
        print(f"Error generating insights: {e}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def cmd_monitoring(args):
    """Gateway monitoring status: health & diagnostics export posture."""
    from hermes_cli.config import load_config

    action = getattr(args, "monitoring_action", None) or "status"
    config = load_config()
    mon_raw = config.get("monitoring")
    mon: dict = mon_raw if isinstance(mon_raw, dict) else {}

    if action == "status":
        from agent.monitoring import otlp_exporter

        gh_raw = mon.get("gateway_health_export")
        gh: dict = gh_raw if isinstance(gh_raw, dict) else {}
        export_raw = mon.get("export")
        export_cfg: dict = export_raw if isinstance(export_raw, dict) else {}
        otlp_raw = export_cfg.get("otlp")
        otlp: dict = otlp_raw if isinstance(otlp_raw, dict) else {}

        print("Gateway monitoring")
        print(f"  Health export:  {'enabled' if gh.get('enabled') else 'disabled'} "
              f"(monitoring.gateway_health_export.enabled)")
        if gh.get("enabled"):
            print(f"    Metrics:            {'on' if gh.get('metrics_enabled', True) else 'off'} "
                  f"(interval {gh.get('export_interval_seconds', 60)}s)")
            print(f"    Diagnostic events:  {'on' if gh.get('diagnostic_events_enabled', True) else 'off'}")
            print(f"    Warning/error logs: {'on' if gh.get('warning_error_events_enabled', True) else 'off'} "
                  f"(interval {gh.get('logs_export_interval_seconds', 5)}s)")
            print("    Content safety:     always on "
                  "(rendered messages are never exported; not configurable)")
        endpoint = otlp.get("endpoint") or ""
        if otlp.get("enabled") and endpoint:
            print(f"  OTLP endpoint:  {endpoint}")
        else:
            print("  OTLP endpoint:  not configured (monitoring.export.otlp)")
        print(f"  OTel SDK:       {'installed' if otlp_exporter.is_available() else 'not installed'} "
              f"(optional extra: hermes-agent[otlp])")
        print("\n  Scope: gateway service health + redacted diagnostics only.")
        print("  No prompts, messages, tool args/results, usage analytics, or traces.")
        return

    print(f"Unknown monitoring action: {action}", file=sys.stderr)
    sys.exit(2)


def cmd_skills(args):
    # Route 'config' action to skills_config module
    if getattr(args, "skills_action", None) == "config":
        _require_tty("skills config")
        from hermes_cli.skills_config import skills_command as skills_config_command

        skills_config_command(args)
    elif getattr(args, "skills_action", None) in ("trust", "untrust"):
        _cmd_skills_trust(args)
    else:
        from hermes_cli.skills_hub import skills_command

        skills_command(args)


def _cmd_skills_trust(args):
    """``hermes skills trust [path]`` / ``hermes skills untrust [path]``.

    Manages ``skills.trusted_project_dirs`` in config.yaml. With no path,
    operates on the project root enclosing the current directory (nearest
    ancestor with ``.git``).
    """
    from pathlib import Path

    from agent.skill_utils import (
        PROJECT_SKILLS_SUBDIRS,
        _candidate_project_skills_dirs,
        find_project_root,
        iter_skill_index_files,
    )
    from hermes_cli.config import load_config, save_config

    action = args.skills_action
    raw_path = getattr(args, "path", None)
    if raw_path:
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}")
            return
    else:
        root = find_project_root()
        if root is None:
            print(
                "Not inside a git checkout. Run from a project directory or "
                "pass the project root path explicitly."
            )
            return

    config = load_config()
    skills_cfg = config.setdefault("skills", {})
    trusted = skills_cfg.get("trusted_project_dirs") or []
    if not isinstance(trusted, list):
        trusted = [trusted]
    trusted = [str(t) for t in trusted]
    root_str = str(root)

    if action == "untrust":
        kept = [t for t in trusted if str(Path(t).expanduser().resolve()) != root_str]
        if len(kept) == len(trusted):
            print(f"{root} was not trusted.")
            return
        skills_cfg["trusted_project_dirs"] = kept
        save_config(config)
        print(f"Untrusted: {root}")
        print("Project skills from this repo will no longer load.")
        return

    # trust
    if any(str(Path(t).expanduser().resolve()) == root_str for t in trusted):
        print(f"Already trusted: {root}")
    else:
        trusted.append(root_str)
        skills_cfg["trusted_project_dirs"] = trusted
        save_config(config)
        print(f"Trusted: {root}")

    # Show what this unlocks
    count = 0
    for d in _candidate_project_skills_dirs(root):
        count += sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
    if count:
        print(
            f"{count} project skill(s) will load in sessions started inside "
            "this repo (they take precedence over same-named profile skills)."
        )
    else:
        subdirs = " or ".join(PROJECT_SKILLS_SUBDIRS)
        print(f"No project skills found yet — add them under {subdirs}.")


def cmd_pairing(args):
    from hermes_cli.pairing import pairing_command

    pairing_command(args)


def cmd_plugins(args):
    from hermes_cli.plugins_cmd import plugins_command

    plugins_command(args)


def cmd_mcp(args):
    from hermes_cli.mcp_config import mcp_command

    mcp_command(args)


def cmd_claw(args):
    from hermes_cli.claw import claw_command

    claw_command(args)


def _advertise_agent_env() -> None:
    """Advertise the agent harness to child processes.

    ``AI_AGENT`` is the emerging cross-agent standard (huggingface_hub's agent
    detection reads it; pi and other agents set it — earendil-works/pi#7493)
    so generic tooling can attribute subprocesses to the harness that spawned
    them. The value must be our id in the public agent-harness registry
    (``hermes-agent`` in huggingface.js ``agent-harnesses.ts``): standard-var
    matching is exact, so any other value is counted as "unknown".
    ``HERMES_AGENT`` is the Hermes-specific marker. setdefault: never
    clobber an outer harness (e.g. Hermes running inside another agent's
    terminal).
    """
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")


def _attach_plugin_cli_command(subparsers, cmd_info) -> None:
    """Register one plugin-provided top-level command from its descriptor."""
    plugin_parser = subparsers.add_parser(
        cmd_info["name"],
        help=cmd_info["help"],
        description=cmd_info.get("description", ""),
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    cmd_info["setup_fn"](plugin_parser)
    if cmd_info.get("handler_fn") is not None:
        plugin_parser.set_defaults(func=cmd_info["handler_fn"])


def _register_plugin_cli_commands(subparsers) -> None:
    """Plugin CLI commands — dynamically registered by memory/general plugins.

    Plugins provide a register_cli(subparser) function that builds their own
    argparse tree; no hardcoded plugin commands in main.py. Skipped when the
    invocation already targets a known built-in subcommand (``hermes --help``,
    ``hermes logs``, ...) — eagerly importing every bundled plugin module
    (google.cloud.pubsub_v1, aiohttp, grpc, PIL …) costs 500-650ms.
    """
    if not _plugin_cli_discovery_needed():
        return
    try:
        from plugins.memory import discover_plugin_cli_commands
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        seen_plugin_commands = set()
        for cmd_info in discover_plugin_cli_commands():
            _attach_plugin_cli_command(subparsers, cmd_info)
            seen_plugin_commands.add(cmd_info["name"])

        discover_plugins()
        # A bundled platform whose top-level CLI command is the one being
        # invoked is still only a deferred entry at this point; import it
        # so its register_cli_command side effect runs before we read
        # _cli_commands (issue #54678).
        _resolve_deferred_platform_cli_command(_first_positional_argv())
        for cmd_info in get_plugin_manager()._cli_commands.values():
            if cmd_info["name"] not in seen_plugin_commands:
                _attach_plugin_cli_command(subparsers, cmd_info)
    except Exception as _exc:
        logging.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc)


def _build_cli_parser():
    """Build the full ``hermes`` argparse tree. Returns ``(parser, subparsers)``.

    Registration ORDER is the order subcommands appear in ``hermes --help``;
    keep it stable. Each group lives in ``hermes_cli/subcommands/<group>.py``
    (handlers injected so those modules never import main) or exposes its own
    ``register_cli``/``build_parser`` from a sibling module.
    """
    from hermes_cli._parser import build_top_level_parser

    parser, subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)

    build_model_parser(subparsers, cmd_model=cmd_model)
    build_moa_parser(subparsers)
    build_fallback_parser(subparsers)
    build_worktree_parser(subparsers)
    build_browser_parser(subparsers)
    build_secrets_parser(subparsers)
    # OUTBOUND iron-proxy egress firewall; ``hermes proxy`` (gateway group) is
    # the separate INBOUND OAuth-aggregator reverse proxy.
    build_egress_parser(subparsers)
    build_migrate_parser(subparsers)
    build_gateway_parser(
        subparsers, cmd_gateway=cmd_gateway, cmd_proxy=cmd_proxy, cmd_gateway_enroll=cmd_gateway_enroll
    )

    # LSP is optional infrastructure — never let a registration failure
    # break the CLI overall.
    try:
        from agent.lsp.cli import register_subparser as _lsp_register
        _lsp_register(subparsers)
    except Exception as _lsp_err:  # noqa: BLE001
        logger.debug("LSP CLI registration failed: %s", _lsp_err)

    build_setup_parser(subparsers, cmd_setup=cmd_setup)
    build_whatsapp_parser(subparsers, cmd_whatsapp=cmd_whatsapp)
    build_whatsapp_cloud_parser(subparsers, cmd_whatsapp_cloud=cmd_whatsapp_cloud)
    build_slack_parser(subparsers, cmd_slack=cmd_slack)

    # send command — pipe shell-script output to any configured platform
    from hermes_cli.send_cmd import register_send_subparser
    register_send_subparser(subparsers)

    build_login_parser(subparsers, cmd_login=cmd_login)
    build_logout_parser(subparsers, cmd_logout=cmd_logout)
    build_auth_parser(subparsers, cmd_auth=cmd_auth)
    build_status_parser(subparsers, cmd_status=cmd_status)
    build_pause_parser(subparsers)
    build_cron_parser(subparsers, cmd_cron=cmd_cron)
    build_sync_parser(subparsers, cmd_sync=cmd_sync)
    build_webhook_parser(subparsers, cmd_webhook=cmd_webhook)

    # peer command — bot-to-bot DMs across machines (peer Hermes gateways)
    from hermes_cli.subcommands.peer import build_peer_parser
    build_peer_parser(subparsers)

    # portal command — Nous Portal status + Tool Gateway routing
    from hermes_cli.portal_cli import add_parser as _add_portal_parser
    _add_portal_parser(subparsers)

    # kanban command — multi-profile collaboration board
    from hermes_cli.kanban import build_parser as _build_kanban_parser
    _build_kanban_parser(subparsers).set_defaults(func=cmd_kanban)

    # project command — named, multi-folder workspaces
    from hermes_cli.projects_cmd import build_parser as _build_project_parser
    _build_project_parser(subparsers).set_defaults(func=cmd_project)

    build_hooks_parser(subparsers, cmd_hooks=cmd_hooks)
    build_doctor_parser(subparsers, cmd_doctor=cmd_doctor)
    build_verify_parser(subparsers, cmd_verify=cmd_verify)
    build_security_parser(subparsers, cmd_security=cmd_security)
    build_approvals_parser(subparsers, cmd_approvals=cmd_approvals)
    build_dump_parser(subparsers, cmd_dump=cmd_dump)
    build_debug_parser(subparsers, cmd_debug=cmd_debug)
    build_backup_parser(subparsers, cmd_backup=cmd_backup)
    build_checkpoints_parser(subparsers)
    build_import_cmd_parser(subparsers, cmd_import=cmd_import)
    build_import_agent_parser(subparsers, cmd_import_agent=cmd_import_agent)
    build_config_parser(subparsers, cmd_config=cmd_config)
    build_skin_parser(subparsers, cmd_skin=cmd_skin)
    build_console_parser(subparsers, cmd_console=cmd_console)
    build_pairing_parser(subparsers, cmd_pairing=cmd_pairing)
    build_skills_parser(subparsers, cmd_skills=cmd_skills)
    build_bundles_parser(subparsers)
    build_plugins_parser(subparsers, cmd_plugins=cmd_plugins)

    _register_plugin_cli_commands(subparsers)

    build_curator_parser(subparsers)
    build_pets_parser(subparsers)
    build_journey_parser(subparsers)
    build_memory_parser(subparsers, cmd_memory=cmd_memory)
    build_tools_parser(subparsers, cmd_tools=cmd_tools)
    build_computer_use_parser(subparsers)
    build_mcp_parser(subparsers, cmd_mcp=cmd_mcp)
    # Lazy indirection: sessions_cmd is only imported when the subcommand
    # runs, and monkeypatches on hermes_cli.main.cmd_sessions keep working.
    build_sessions_parser(subparsers, cmd_sessions=lambda a, **kw: _self().cmd_sessions(a, **kw))
    build_insights_parser(subparsers, cmd_insights=cmd_insights)
    build_monitoring_parser(subparsers, cmd_monitoring=cmd_monitoring)
    build_claw_parser(subparsers, cmd_claw=cmd_claw)
    # NOTE: the `hermes version` subcommand was removed — `hermes --version`
    # / `-V` now carries the full output including update status.
    build_update_parser(subparsers, cmd_update=cmd_update)
    build_uninstall_parser(subparsers, cmd_uninstall=cmd_uninstall)
    build_acp_parser(subparsers, cmd_acp=cmd_acp)
    build_profile_parser(subparsers, cmd_profile=cmd_profile)
    build_completion_parser(subparsers, cmd_completion=cmd_completion, parser=parser)
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=cmd_dashboard,
        cmd_dashboard_register=cmd_dashboard_register,
    )
    # desktop (a.k.a. gui): the canonical name is "desktop"; "gui" is kept as
    # a deprecated alias for one release. The Hermes-Setup.exe success screen
    # tells users to run `hermes desktop`, so that name must be the one shown
    # in --help (argparse promotes the primary name; aliases stay hidden).
    build_gui_parser(subparsers, cmd_gui=cmd_gui)
    build_logs_parser(subparsers, cmd_logs=cmd_logs)
    build_prompt_size_parser(subparsers, cmd_prompt_size=cmd_prompt_size)
    return parser, subparsers


def _parse_cli_args(parser, subparsers, argv):
    """Parse ``argv`` with the bpo-9338 subparser-routing workaround.

    argv is first pre-processed so unquoted multi-word session names after
    -c / -r merge into one token (``hermes -c Pokemon Agent Dev`` →
    ``hermes -c 'Pokemon Agent Dev'``).

    On some Python versions (notably <3.11), argparse fails to route
    subcommand tokens when the parent parser has nargs='?' optional arguments
    (--continue): "unrecognized arguments: model" even though 'model' is a
    registered subcommand. Fix: when argv contains a token matching a known
    subcommand, set subparsers.required=True to force deterministic routing.
    If that fails (e.g. 'hermes -c model' where 'model' is consumed as the
    session name for --continue), fall back to the default behaviour.
    """
    import io as _io

    _processed_argv = _coalesce_session_name_args(argv)
    _known_cmds = (
        set(subparsers.choices.keys()) if hasattr(subparsers, "choices") else set()
    )
    _has_cmd_token = any(
        t in _known_cmds for t in _processed_argv if not t.startswith("-")
    )
    if not _has_cmd_token:
        subparsers.required = False
        return parser.parse_args(_processed_argv)

    subparsers.required = True
    _saved_stderr = sys.stderr
    try:
        sys.stderr = _io.StringIO()
        args = parser.parse_args(_processed_argv)
        sys.stderr = _saved_stderr
    except SystemExit as exc:
        sys.stderr = _saved_stderr
        # Help/version flags (exit code 0) already printed output —
        # re-raise immediately to avoid a second parse_args printing
        # the same help text again (#10230).
        if exc.code == 0:
            raise
        # Subcommand name was consumed as a flag value (e.g. -c model).
        # Fall back to optional subparsers so argparse handles it normally.
        subparsers.required = False
        args = parser.parse_args(_processed_argv)
    return args


def _default_to_chat(args) -> None:
    """No subcommand given: run chat (top-level --resume/--continue is a chat shortcut)."""
    if args.resume or args.continue_last:
        args.command = "chat"
    _set_chat_arg_defaults(args)
    cmd_chat(args)


def cmd_import_agent(args):
    from hermes_cli.agent_import import import_agent_command
    import_agent_command(args)


def main():
    """Main entry point for hermes CLI."""
    # Cosmetic: make the process show up as 'hermes' instead of 'python3.11'
    # in ps/top/htop.  Non-fatal — just a nicer UX.
    _set_process_title()

    # Let child processes (and tools like huggingface_hub) detect they run
    # under an AI agent harness.
    _advertise_agent_env()

    # Force UTF-8 stdio on Windows before anything prints.  No-op elsewhere.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    # Sweep stale ``hermes.exe.old.*`` quarantine files left by previous
    # ``hermes update`` runs on Windows. Silent no-op on non-Windows or when
    # there's nothing to clean. See ``_quarantine_running_hermes_exe``.
    try:
        _cleanup_quarantined_exes()
    except Exception:
        pass

    # If the checkout changed since the last launch (hermes update, manual
    # git pull, old-updater update that predates newer clears), sweep stale
    # __pycache__ once so no process — this one's lazy imports included —
    # resolves fresh source against old bytecode. Never raises.
    _sweep_stale_bytecode_if_checkout_changed()

    # Self-heal a venv left half-built by an interrupted ``hermes update``
    # (Ctrl-C, terminal close, WSL OOM mid-install). Skip when the user is
    # *running* update — that flow writes and clears its own marker, and we
    # don't want a recovery install racing the real one. Never raises.
    #
    # The substring match is deliberately loose: argv isn't parsed yet at this
    # point, and the failure modes are asymmetric. Over-matching (e.g.
    # ``hermes skills install update``) merely defers recovery one launch;
    # under-matching (missing ``hermes -p work update``) would race a recovery
    # install against the real one. Loose wins.
    try:
        if "update" not in sys.argv[1:]:
            _recover_from_interrupted_install()
    except Exception:
        pass

    # Cheap hint only (#95294): an interrupted update that pulled code but
    # never restarted the fleet. Do NOT restart here — that is ``hermes
    # update`` catch-up work. Skip when the user is already running update.
    try:
        if "update" not in sys.argv[1:]:
            _warn_pending_fleet_restart_on_startup()
    except Exception:
        pass

    if _try_termux_fast_tui_launch():
        return
    if _try_termux_fast_cli_launch():
        return
    if _try_fast_serve_launch():
        return
    if _try_fast_chat_launch():
        return

    parser, subparsers = _build_cli_parser()

    # ── Container-aware routing ────────────────────────────────────────
    # When NixOS container mode is active, route ALL subcommands into
    # the managed container.  This MUST run before parse_args() so that
    # --help, unrecognised flags, and every subcommand are forwarded
    # transparently instead of being intercepted by argparse on the host.
    from hermes_cli.config import get_container_exec_info

    container_info = get_container_exec_info()
    if container_info:
        _exec_in_container(container_info, sys.argv[1:])
        # Unreachable: os.execvp never returns on success (process is replaced)
        # and raises OSError on failure (which propagates as a traceback).
        sys.exit(1)

    args = _parse_cli_args(parser, subparsers, sys.argv[1:])

    if args.version:
        cmd_version(args)
        return

    # --yolo: set HERMES_YOLO_MODE *before* plugin discovery.  The call to
    # _prepare_agent_startup() below triggers discover_plugins() → tool
    # imports, and tools.approval freezes _YOLO_MODE_FROZEN at module
    # import time (PR #7994, security hardening against prompt-injection).
    # If the env var is set only later (e.g. inside cmd_chat), the frozen
    # value is already False and --yolo silently does nothing.
    if getattr(args, "yolo", False):
        os.environ["HERMES_YOLO_MODE"] = "1"

    # Discover Python plugins and register shell hooks once, before any
    # command that can fire lifecycle hooks.  Both are idempotent; gated
    # so introspection/management commands (hermes hooks list, cron
    # list, gateway status, mcp add, ...) don't pay discovery cost or
    # trigger consent prompts for hooks the user is still inspecting.
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _run_oneshot_from_args(args)

    # No subcommand (optionally with top-level --resume / --continue) → chat.
    if args.command is None:
        _default_to_chat(args)
        return

    # Execute the command.  Propagate the handler's return code as the
    # process exit code so subcommands that signal failure (e.g.
    # ``hermes egress start`` refusing when credential_source=bitwarden
    # is misconfigured) actually exit non-zero.  Handlers that return
    # None are treated as success (exit 0).
    if hasattr(args, "func"):
        rc = args.func(args)
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
