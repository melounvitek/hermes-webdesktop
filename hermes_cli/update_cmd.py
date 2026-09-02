"""Hermes update pipeline — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): ``_cmd_update_impl``, ``_cmd_update_check``
and every module-level helper used only by the update path, plus the update-only
constants they read. Function bodies are lifted verbatim; the only mechanical
change is that references to helpers/constants that STAY in ``hermes_cli.main``
(and to moved-but-test-patched siblings) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` (``PROJECT_ROOT``, ``_is_windows``,
``_run_pre_update_backup``, ...) keep working unchanged. ``main.py`` re-imports
every public-ish name from here (``# noqa: F401``) so the argparse wiring and
the test-patch surface still resolve on ``hermes_cli.main``.

The closures that used to be nested inside ``_cmd_update_impl`` (``_print_items``,
``_wait_for_service_active``, ``_service_restart_sec``, ``_resolve_manage_cmd``,
``_restart_one_systemd_gateway_unit``) now live at module level or inside the
phase helpers ``_cmd_update_impl`` calls in order: ``_pull_updates`` ->
``_sync_python_dependencies_after_pull`` -> ``_run_post_update_maintenance`` ->
``_restart_gateway_fleet_after_update`` -> ``_verify_fleet_after_update``.

Imports are one-way: ``hermes_cli.main`` imports this module, never the reverse
at import time (``_m()`` resolves lazily at call time, when main.py is fully
loaded, so there is no import cycle).
"""

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home
from hermes_constants import get_default_hermes_root, venv_python_path

logger = logging.getLogger(__name__)


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Keeps ``hermes_cli.main.<helper>`` test patches effective in this code
    path and keeps the ``main`` -> ``update_cmd`` import one-way at import time.
    """
    from hermes_cli import main

    return main


def _no_prompt_git_kwargs() -> dict:
    """``subprocess.run`` kwargs for the updater's network git calls.

    GitHub answers anonymous fetches with HTTP 401 during outages (and for
    unreachable repos); git then prompts ``Username for 'https://github.com':``
    on the inherited terminal and the update sits there forever. Disable the
    prompt so the fetch fails fast into ``_classify_fetch_failure``. Only the
    *prompt* is disabled — a configured credential helper / askpass still
    runs, so a private-fork origin keeps authenticating non-interactively.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return {"stdin": subprocess.DEVNULL, "env": env}


_UPDATE_RUNTIME_RELOAD_MODULES = (
    "hermes_constants",
    "tools.environments.local",
    "tools.lazy_deps",
)

#: Package prefixes whose cached modules become stale the moment the checkout
#: changes under this process. Purged (not reloaded) by
#: ``_purge_stale_hermes_modules`` so any LATER import chain resolves against
#: fresh on-disk source only.
_STALE_PURGE_PREFIXES = (
    "hermes_cli",
    "gateway",
    "tools",
    "tui_gateway",
    "agent",
)

#: Modules that must survive the purge: they are (or are referenced by) the
#: code currently EXECUTING the update, so evicting them buys nothing — the
#: running frames keep their module objects alive regardless — and reloading
#: them mid-flight is the one genuinely unsafe move.
_STALE_PURGE_PROTECTED = frozenset(
    {
        "hermes_cli",
        "hermes_cli.main",
        "hermes_cli.update_cmd",
        "hermes_cli.hermes_logging",
    }
)


def _purge_stale_hermes_modules() -> None:
    """Evict every cached Hermes module after the checkout changed in-place.

    ``hermes update`` keeps running in the pre-pull Python process; the
    gateway-restart phase then does function-level imports of NEW source
    inside an OLD ``sys.modules`` world. As soon as new source references a
    symbol added to an already-cached module, the import dies (2026-08-20:
    fresh ``hermes_cli.gateway`` imported ``line_input`` from a stale cached
    ``cli_output`` → restart phase aborted, gateway kept serving old code).

    ``_UPDATE_RUNTIME_RELOAD_MODULES`` fixed this per-symptom; this is the
    class fix: drop EVERY cached module under the Hermes package prefixes so
    later lazy imports rebuild a self-consistent graph from the new checkout.
    Purging only removes the ``sys.modules`` entry — module objects held by
    running frames stay alive and functional. Only genuinely executing modules
    are exempt, because reload-in-place (not purge) is what can pull code out
    from under a running frame.

    Best-effort: never raises.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        purged = []
        for name in list(_m().sys.modules):
            if name in _STALE_PURGE_PROTECTED:
                continue
            if not name.startswith(_STALE_PURGE_PREFIXES):
                continue
            root = name.split(".", 1)[0]
            if root not in _STALE_PURGE_PREFIXES:
                # Prefix-string match caught an unrelated package
                # (e.g. ``gateway_foo``) — leave it alone.
                continue
            if _m().sys.modules.pop(name, None) is not None:
                purged.append(name)
        if purged:
            logger.debug(
                "Purged %d stale Hermes module(s) after checkout update", len(purged)
            )
    except Exception as exc:
        logger.debug("Could not purge stale Hermes modules: %s", exc)


def _reload_updated_runtime_modules() -> None:
    """Reload update-sensitive modules after the checkout changes in-place.

    ``hermes update`` runs in the pre-pull process, so cached modules can
    expose old symbols despite new source on disk. Refresh the small set used
    by lazy-backend refresh before that step imports newly-updated code paths.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        for module_name in _UPDATE_RUNTIME_RELOAD_MODULES:
            module = _m().sys.modules.get(module_name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:
                logger.debug("Could not reload updated module %s: %s", module_name, exc)
    except Exception as exc:
        logger.debug("Could not refresh update runtime modules: %s", exc)


def _reload_config_modules() -> None:
    """Force-reload modules from disk after git pull.

    ``hermes update`` runs in the PRE-pull process, so cached modules hold OLD
    code: ``DEFAULT_CONFIG["_config_version"]`` is stale and
    ``check_config_version()`` reports "up to date" even when the pulled code
    has a newer version with a migration to run. Reloads
    ``config_defaults`` / ``config`` / ``config_migrations`` from disk.

    Also reloads ``_subprocess_compat`` and ``dashboard_procs`` so the later
    dashboard cleanup (``_finish_dashboard_update_cleanup`` →
    ``_scan_dashboard_processes``) sees symbols the update added (e.g.
    ``bounded_probe_run``) instead of dying with ImportError in this process.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli.config_defaults",
        "hermes_cli.config",
        "hermes_cli.config_migrations",
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                logger.debug("Could not reload %s for fresh post-update code: %s", mod_name, exc)


def _run_config_check_fresh() -> tuple:
    """Check config version using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns ``(current_ver, latest_ver)``.
    """
    _reload_config_modules()
    from hermes_cli.config import check_config_version

    return check_config_version()


def _run_migrate_config_fresh(*, interactive: bool = False, quiet: bool = False) -> dict:
    """Run config migration using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns the migration results dict.
    """
    _reload_config_modules()
    from hermes_cli.config import migrate_config

    return migrate_config(interactive=interactive, quiet=quiet)


def _migrate_sibling_profile_configs() -> list[tuple[str, int, int]]:
    """Migrate every SIBLING profile's config.yaml to the current version.

    #91277 Phase 2 (fleet-wide config migration; #20438/#54926/#79048): the
    shared checkout serves every profile, but ``hermes update`` historically
    migrated only the active profile's config — siblings drifted versions
    until their gateway hit a config the new code couldn't read.

    Per profile home (skipping the active one, already migrated by the
    caller): scope config reads/writes via the context-local HERMES_HOME
    override (thread-safe — never ``os.environ``), check the version, and
    run the NON-INTERACTIVE, quiet migration. Prompt-requiring settings are
    left for the profile's own next interactive session, identical to the
    gateway-mode contract for the active profile.

    Returns ``[(profile_name, from_version, to_version), ...]`` for profiles
    actually migrated. Never raises; a failing profile is skipped (its own
    startup migration remains the fallback).
    """
    migrated: list[tuple[str, int, int]] = []
    try:
        from hermes_constants import (
            get_process_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli.profiles import _get_profiles_root, _PROFILE_ID_RE

        active_home = get_process_hermes_home()
        root = _get_profiles_root()
        if not root.is_dir():
            return migrated
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not _PROFILE_ID_RE.match(entry.name):
                continue
            try:
                if entry.resolve() == Path(active_home).resolve():
                    continue
            except OSError:
                continue
            if not (entry / "config.yaml").is_file():
                continue  # profile never configured — nothing to migrate
            token = set_hermes_home_override(entry)
            try:
                current_ver, latest_ver = _run_config_check_fresh()
                if current_ver >= latest_ver:
                    continue
                _run_migrate_config_fresh(interactive=False, quiet=True)
                after_ver, _ = _run_config_check_fresh()
                if after_ver > current_ver:
                    migrated.append((entry.name, current_ver, after_ver))
            except Exception as exc:
                logger.debug(
                    "Config migration for profile %s failed: %s", entry.name, exc
                )
            finally:
                reset_hermes_home_override(token)
    except Exception as exc:
        logger.debug("Sibling profile enumeration failed: %s", exc)
    return migrated

def _check_and_apply_config_migration(
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
) -> None:
    """Check and apply configuration migrations on an update completion path (#91360).

    Must use freshly-reloaded modules (see ``_reload_config_modules``), and
    must run on EVERY completion path — normal post-pull, venv-repair retry,
    and the Node-deps repair on the ``commit_count == 0`` branch — so an
    interrupted update that already pulled new code doesn't strand the user
    on an older config version.
    """
    print()
    print("→ Checking configuration for new options...")

    # Reload config modules BEFORE any config reads so get_missing_*,
    # check_config_version, and migrate_config all use the updated code.
    _reload_config_modules()

    from hermes_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
    )

    # Defensive (#91360): this helper runs on repair/retry completion paths
    # too — a config-check failure must not break an otherwise-successful
    # update. Log, point at the manual command, and return.
    try:
        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = _run_config_check_fresh()
    except Exception as exc:
        logger.debug("Config check during update failed: %s", exc)
        print("  ⚠️  Could not check config version.")
        print("     Run 'hermes config migrate' to check manually.")
        return

    has_new_options = bool(missing_env or missing_config)
    version_bump_only = (
        not has_new_options and current_ver < latest_ver
    )
    needs_migration = has_new_options or current_ver < latest_ver

    if version_bump_only:
        # Only the format version changed (new defaults merge transparently).
        # Prompting "configure new options now?" would look like a no-op on
        # yes (ScottFive / Tt2021) — apply silently and say what happened.
        print()
        print(
            f"  ℹ Updating config format (v{current_ver} → v{latest_ver})…"
        )
        try:
            _mig_results = _run_migrate_config_fresh(
                interactive=False, quiet=True
            )
            print("  ✓ Config format updated (no new settings to configure)")
            # quiet=True also mutes steps that RESET/REMOVE a setting (e.g. the
            # v33→v34 personality reset, #81946). Re-surface them so an
            # unattended update never silently changes config (#86656). Here
            # missing_config is empty, so config_added holds only mutations.
            for _note in _mig_results.get("config_added") or []:
                print(f"  ℹ {_note}")
            for _warn in _mig_results.get("warnings") or []:
                print(f"  ⚠️  {_warn}")
        except Exception as _mig_err:
            print(f"  ⚠️  Config format update failed: {_mig_err}")
            print("     Run 'hermes config migrate' to retry.")
    elif needs_migration:
        print()
        # Show WHAT changed, not just a count, so the user can make an
        # informed yes/no decision (previously the prompt named nothing).
        if missing_env:
            print(
                f"  ⚠️  {len(missing_env)} new required setting(s) need configuration"
            )
            _print_items(missing_env, "New settings", "name")
        if missing_config:
            print(f"  ℹ️  {len(missing_config)} new config option(s) available")
            _print_items(missing_config, "New options", "key")

        print()
        if assume_yes:
            print(
                "  ℹ --yes: auto-applying config migration (skipping API-key prompts)."
            )
            response = "y"
        elif gateway_mode:
            response = (
                _gateway_prompt(
                    "Would you like to configure new options now? [Y/n]", "n"
                )
                .strip()
                .lower()
            )
        elif not (sys.stdin.isatty() and sys.stdout.isatty()):
            print("  ℹ Non-interactive session — applying safe config migrations.")
            response = "auto"
        else:
            try:
                response = (
                    input("Would you like to configure them now? [Y/n]: ")
                    .strip()
                    .lower()
                )
            except EOFError:
                response = "n"
            except UnicodeDecodeError:
                # Non-UTF-8 locales / embedded terminals can make input()
                # raise this; uncaught, it crashes the update at this prompt.
                print(
                    "  ⚠ Could not read input (encoding issue). Skipping. "
                    "Run 'hermes config migrate' manually to configure."
                )
                response = "n"

        if response in {"", "y", "yes", "auto"}:
            print()
            # Gateway mode, --yes and non-interactive contexts can't prompt
            # for API keys; still run the non-interactive pass so new defaults
            # and version bumps land before the restarted gateway validates.
            interactive_migration = not (
                gateway_mode or assume_yes or response == "auto"
            )
            results = _run_migrate_config_fresh(interactive=interactive_migration, quiet=False)

            if results["env_added"] or results["config_added"]:
                print()
                print("✓ Configuration updated!")
            if (gateway_mode or assume_yes or response == "auto") and missing_env:
                print("  ℹ API keys require manual entry: hermes config migrate")
        else:
            print()
            print("Skipped. Run 'hermes config migrate' later to configure.")
    else:
        print("  ✓ Configuration is up to date")

    # Fleet-wide config migration (#91277 Phase 2; #20438/#54926/#79048):
    # the migration above touched only the active profile; siblings drifted
    # (field repro: gateway on new code but config v33 vs v37). Run the same
    # NON-INTERACTIVE migration per sibling home via the context-local
    # HERMES_HOME override (never os.environ — other threads must not see it).
    try:
        _migrated_siblings = _migrate_sibling_profile_configs()
        for _name, _from_ver, _to_ver in _migrated_siblings:
            print(
                f"  ✓ Profile '{_name}': config format updated "
                f"(v{_from_ver} → v{_to_ver})"
            )
    except Exception as exc:
        logger.debug("Sibling config migration failed: %s", exc)

    # Safety net: migrations have left cron/jobs.json valid-but-empty
    # (#34600) and the desktop scheduler has overwritten it with a partial
    # set (#52144). Restore from the pre-update snapshot if jobs went missing.
    try:
        from hermes_cli.backup import restore_cron_jobs_if_emptied

        cron_restore = restore_cron_jobs_if_emptied(pre_update_snapshot_id)
        if cron_restore:
            print()
            print(
                "  ⚠️  cron/jobs.json lost jobs during this update — "
                f"restored {cron_restore['job_count']} job(s) from "
                f"pre-update snapshot {cron_restore['snapshot_id']}."
            )
    except Exception as exc:
        # Never let the cron safety net break an otherwise-good update.
        logger.debug("Cron jobs auto-restore check failed: %s", exc)

    # #64160: Desktop update/repair cycles have rewritten model.provider /
    # model.default and dropped moa: (settings the gateway and cron consume).
    # Restore only those protected keys from the same pre-update snapshot.
    try:
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        cfg_restore = restore_config_model_settings_if_rewritten(
            pre_update_snapshot_id
        )
        if cfg_restore:
            print()
            print(
                "  ⚠️  config.yaml user model settings were rewritten during "
                f"this update — restored {', '.join(cfg_restore['keys'])} "
                f"from pre-update snapshot {cfg_restore['snapshot_id']}."
            )
    except Exception as exc:
        # Never let the config safety net break an otherwise-good update.
        logger.debug("Config model-settings auto-restore check failed: %s", exc)

    # #66140: run the same cron-jobs safety net for every sibling
    # profile against ITS OWN pre-update snapshot (same-generation by
    # construction — both taken by this run).
    try:
        from hermes_cli.backup import restore_cron_jobs_all_profiles

        for _restored in restore_cron_jobs_all_profiles(
            _LAST_SIBLING_SNAPSHOTS
        ):
            print()
            print(
                f"  ⚠️  Profile '{_restored['profile']}': cron/jobs.json "
                f"lost jobs during this update — restored "
                f"{_restored['job_count']} job(s) from pre-update "
                f"snapshot {_restored['snapshot_id']}."
            )
    except Exception as exc:
        logger.debug("Sibling cron auto-restore check failed: %s", exc)

    # #64160: same config model-settings safety net for sibling profiles.
    try:
        from hermes_cli.backup import restore_config_model_settings_all_profiles

        for _cfg_restored in restore_config_model_settings_all_profiles(
            _LAST_SIBLING_SNAPSHOTS
        ):
            print()
            print(
                f"  ⚠️  Profile '{_cfg_restored['profile']}': config.yaml "
                f"user model settings were rewritten during this update — "
                f"restored {', '.join(_cfg_restored['keys'])} from "
                f"pre-update snapshot {_cfg_restored['snapshot_id']}."
            )
    except Exception as exc:
        logger.debug("Sibling config auto-restore check failed: %s", exc)


# Files that must parse right after an update/install (CLI startup imports;
# ``web_server.py`` is the desktop backend a fresh Windows install launches).
# The post-pull syntax guard validates these and auto-rolls-back on failure.
_UPDATE_CRITICAL_FILES = (
    "hermes_cli/main.py",
    "hermes_cli/config.py",
    "hermes_cli/__init__.py",
    "hermes_cli/web_server.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
)

def _record_update_step(step: str, ok: bool, detail: str = "") -> None:
    """Best-effort ``update_receipt.record_step``; the receipt must never break an update."""
    try:
        from hermes_cli.update_receipt import record_step

        record_step(step, ok, detail)
    except Exception:
        pass


def _git_run(git_cmd, args, cwd=None, *, check=False, network=False):
    """Run ``git_cmd + args`` (default cwd: the checkout), capturing utf-8 text.

    ``network=True`` (fetch/pull/push) disables git's terminal prompt so an
    HTTP 401 fails fast instead of hanging on ``Username for ...``.
    """
    return subprocess.run(
        git_cmd + args,
        cwd=_m().PROJECT_ROOT if cwd is None else cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=check,
        **(_no_prompt_git_kwargs() if network else {}),
    )


def _capture_head_sha(git_cmd, cwd) -> str | None:
    """Return the current HEAD SHA, or None if it can't be resolved."""
    try:
        result = _git_run(git_cmd, ["rev-parse", "HEAD"], cwd, check=True)
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None

_ORPHAN_RESCUE_REFS_TO_KEEP = 10
_ORPHAN_RESCUE_REF_MAX_AGE_DAYS = 30

def _prune_orphan_rescue_refs(
    git_cmd,
    cwd,
    branch,
    keep=_ORPHAN_RESCUE_REFS_TO_KEEP,
    max_age_days=_ORPHAN_RESCUE_REF_MAX_AGE_DAYS,
) -> None:
    """Expire old orphan rescue refs so backups stay bounded.

    Each orphan-history divergence (#87694) parks the pre-reset HEAD under
    ``refs/hermes-update-backups/orphan-<branch>-<ts>-<sha>``. A rescue ref
    pins its objects against ``git gc`` — in the incident shape a full
    working-tree snapshot, potentially multi-GB — so a repeatedly corrupted
    install would grow ``.git`` without bound.

    Two limits, both enforced on every orphan incident: keep only the
    ``keep`` most-recent refs, and drop any older than ``max_age_days`` per
    the ``YYYYMMDD-HHMMSS`` stamp in the ref name (unparseable names are left
    alone). Names sort chronologically, so ``for-each-ref`` order is creation
    order. Disk is reclaimed on the next ``git gc``. Best-effort: never
    blocks the update.
    """
    try:
        list_result = _git_run(
            git_cmd,
            ["for-each-ref", "--format=%(refname)", "--sort=refname",
             f"refs/hermes-update-backups/orphan-{branch}-*"],
            cwd,
        )
        if list_result.returncode != 0:
            return
        refs = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
        stale = set(refs[:-keep] if keep > 0 else refs)
        # Age expiry: ref names embed a UTC YYYYMMDD-HHMMSS timestamp right
        # after the branch segment; anything older than max_age_days goes.
        if max_age_days > 0:
            from datetime import timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            prefix = f"refs/hermes-update-backups/orphan-{branch}-"
            for ref in refs:
                stamp = ref[len(prefix):][:15]  # "YYYYMMDD-HHMMSS"
                try:
                    ref_time = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue
                if ref_time < cutoff:
                    stale.add(ref)
        for ref in sorted(stale):
            _git_run(git_cmd, ["update-ref", "-d", ref], cwd)
    except OSError:
        pass

# Files that define the editable install. A pull that touches none of them
# cannot have invalidated it.
_INSTALL_DEFINING_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "uv.lock",
)

def _editable_install_is_current(git_cmd, cwd, pre_pull_sha: str | None) -> bool:
    """True when the pulled commits cannot have invalidated the editable install.

    ``uv pip install -e .`` reinstalls unconditionally and rewrites the
    console-script shims every time. On Windows that rewrite is the only
    reason the running ``hermes.exe`` must be quarantined, and a lost
    quarantine race is the whole ``os error 32`` family — so skip the
    reinstall when it provably cannot change anything.

    Safe because Hermes pins its editable finder to a *static* module list
    (``[tool.setuptools] py-modules`` + ``packages.find.include``): only a
    new top-level module/package can stale it, and that needs a
    ``pyproject.toml`` diff (as do dependencies and ``[project.scripts]``).
    New submodules under an already-mapped package need no reinstall.

    Fails closed: an unresolvable pre-pull SHA (shallow checkout, ZIP swap)
    or a failed ``git diff`` returns False and the install runs as before.
    """
    if not pre_pull_sha:
        return False
    try:
        result = subprocess.run(
            git_cmd
            + ["diff", "--name-only", f"{pre_pull_sha}..HEAD", "--"]
            + list(_INSTALL_DEFINING_FILES),
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return not result.stdout.strip()

def _validate_python_files_syntax(
    root, relpaths
) -> tuple[bool, str | None, str | None]:
    """Compile *relpaths* under *root* without writing bytecode into the tree."""
    import py_compile
    import tempfile

    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in relpaths:
            path = root / relpath
            if not path.exists():
                continue
            cfile = Path(tmpdir) / (str(relpath).replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not read: {exc}"
    return True, None, None


def _validate_critical_files_syntax(root) -> tuple[bool, str | None, str | None]:
    """Compile each file in ``_UPDATE_CRITICAL_FILES`` to catch SyntaxErrors.

    These are imported on every ``hermes`` startup; a syntax error (orphan
    conflict markers, etc.) means the CLI can't bootstrap, so we validate
    after ``git pull`` and auto-roll-back instead of leaving a bricked install.

    The ``.pyc`` goes to a temp dir, not the tree's ``__pycache__/``: avoids
    racing concurrent test workers and leaving a stale pyc behind when the
    next interpreter run uses a different Python. Only the compile-or-not
    signal matters.

    Returns ``(ok, failing_path, error_message)``.
    """
    return _validate_python_files_syntax(root, _UPDATE_CRITICAL_FILES)


# Modules imported on every agent startup. Unlike _UPDATE_CRITICAL_FILES (which
# is only parsed), these are actually *imported* so that cross-module breakage
# is caught — a file can be syntactically perfect and still fail to import
# because a name it pulls from a sibling module no longer exists.
_UPDATE_CRITICAL_MODULES = (
    "hermes_cli.main",
    "run_agent",
    "model_tools",
    "toolsets",
)


def _critical_module_import_failures(
    root, *, report_runtime_errors: bool = False
) -> dict[str, tuple[str, str]]:
    """Import each module in ``_UPDATE_CRITICAL_MODULES`` in a subprocess.

    ``_validate_critical_files_syntax`` only *parses*, so a partially-updated
    tree (new ``agent/``, old ``tools/``) parses fine yet dies at startup
    with ``ImportError: cannot import name ...``. That skew is reachable on
    the Windows ZIP-update path, whose copy loop replaces top-level entries
    one at a time in ``os.listdir`` order.

    Runs in a subprocess (~0.4s) so the half-updated tree's import-time side
    effects don't pollute the updater's ``sys.modules``. Uses the project
    venv's interpreter when present (like ``_venv_core_imports_healthy``):
    ``hermes update`` may be driven by a different Python than the install's.

    Returns every failing module in probe order. Generic import-time
    exceptions are tolerated by default (they can depend on local config);
    ``report_runtime_errors=True`` exposes them so a caller can compare two
    states of the same checkout without one failure masking another.
    """
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS

    import secrets

    marker = f"__HERMES_IMPORT_HEALTH_{secrets.token_hex(16)}__"
    probe = (
        "import importlib, json, sys\n"
        "failures = []\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as exc:\n"
        # A missing *third-party* module means dependencies aren't installed
        # yet, not a skewed checkout. Only our own packages count as breakage.
        # The root set is injected from hermes_constants so this can't drift
        # from the hint the user is shown (they disagreed once already).
        "        missing = (getattr(exc, 'name', '') or '').split('.')[0]\n"
        "        if missing in %r or missing.startswith('hermes_') or %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except ImportError as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except Exception as exc:\n"
        "        if %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except BaseException as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "sys.stdout.write('\\n%s' + json.dumps(failures))\n"
        % (
            _UPDATE_CRITICAL_MODULES,
            tuple(sorted(FIRST_PARTY_MODULE_ROOTS)),
            report_runtime_errors,
            report_runtime_errors,
            marker,
        )
    )
    try:
        interpreter = sys.executable
        try:
            venv_python = venv_python_path(
                Path(root) / "venv", windows=_m()._is_windows()
            )
            if venv_python.exists():
                interpreter = str(venv_python)
        except Exception:
            pass  # fall back to the running interpreter
        result = subprocess.run(
            [interpreter, "-c", probe],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "critical-module probe": (
                "TimeoutExpired",
                "timed out before reporting import health",
            )
        }
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return {}
    output = result.stdout or ""
    if marker not in output:
        return {
            "critical-module probe": (
                "ProbeTerminated",
                "terminated before reporting import health "
                f"(exit code {result.returncode})",
            )
        }
    try:
        import json

        failures = json.loads(output.rsplit(marker, 1)[1])
        if not isinstance(failures, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
            for item in failures
        ):
            raise ValueError("invalid import-health payload")
        return {
            str(module): (str(kind), str(detail))
            for module, kind, detail in failures
        }
    except (TypeError, ValueError):
        return {
            "critical-module probe": (
                "MalformedPayload",
                "reported malformed import health data",
            )
        }


def _validate_critical_modules_import(
    root, *, report_runtime_errors: bool = False
) -> tuple[bool, str | None, str | None]:
    """Return the first critical-module import failure, if any."""
    failures = _critical_module_import_failures(
        root, report_runtime_errors=report_runtime_errors
    )
    if failures:
        module = next(iter(failures))
        return False, module, failures[module][1]
    return True, None, None

def _gateway_prompt(prompt_text: str, default: str = "", timeout: float = 300.0) -> str:
    """File-based IPC prompt for gateway mode.

    Writes a prompt marker file for the gateway to forward to the user, then
    polls for a response file; falls back to *default* on timeout. Lets
    ``hermes update --gateway`` forward prompts (stash restore, config
    migration) to the messenger instead of silently skipping them.
    """
    import json as _json
    import uuid as _uuid
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    prompt_path = home / ".update_prompt.json"
    response_path = home / ".update_response"

    response_path.unlink(missing_ok=True)

    payload = {
        "prompt": prompt_text,
        "default": default,
        "id": str(_uuid.uuid4()),
    }
    tmp = prompt_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload), encoding="utf-8")
    tmp.replace(prompt_path)

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if response_path.exists():
            try:
                answer = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                return answer if answer else default
            except (OSError, ValueError):
                pass
        _time.sleep(0.5)

    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    print(f"  (no response after {int(timeout)}s, using default: {default!r})")
    return default

def _npm_bin_exists(bin_dir: Path, name: str) -> bool:
    """True when an npm bin shim for *name* exists (POSIX or Windows)."""
    return any(
        (bin_dir / candidate).exists()
        for candidate in (name, f"{name}.cmd", f"{name}.ps1", f"{name}.exe")
    )

def _web_build_toolchain_ready(*roots: Path) -> bool:
    """True when ``tsc`` and ``vite`` shims are reachable from any of *roots*.

    Callers must pass every root the build would search; checking only one
    reports a healthy tree as broken.
    """
    bin_dirs = [
        bin_dir
        for bin_dir in (root / "node_modules" / ".bin" for root in roots)
        if bin_dir.is_dir()
    ]
    return bool(bin_dirs) and all(
        any(_npm_bin_exists(bin_dir, tool) for bin_dir in bin_dirs)
        for tool in ("tsc", "vite")
    )

def _web_toolchain_roots(web_dir: Path) -> tuple[Path, ...]:
    """Roots whose ``node_modules/.bin`` can satisfy the web build.

    ``npm run build`` prepends ``node_modules/.bin`` for the package and each
    of its ancestors, so shims hoisted to the workspace root and shims nested
    under a package that owns its lockfile (#42973) are equally valid.
    """
    return (web_dir, web_dir.parent)

def _print_curator_first_run_notice() -> None:
    """Print a short heads-up about the skill curator after `hermes update`.

    Only fires when the curator is enabled AND has no recorded run yet, which
    is exactly the window where the gateway ticker used to fire Curator
    against a fresh skill library immediately after an update. We defer the
    first real pass by one ``interval_hours``; this notice tells the user how
    to preview or disable before then. Silent on steady state.
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        if not curator.is_enabled():
            return
        state = curator.load_state()
    except Exception:
        return
    if state.get("last_run_at"):
        # Curator has run before (real or already seeded) — no notice needed.
        return
    try:
        hours = curator.get_interval_hours()
    except Exception:
        hours = 24 * 7
    days = max(1, hours // 24)
    print()
    print("ℹ Skill curator")
    print(
        f"  Background skill maintenance is enabled. First pass is deferred "
        f"~{days}d after installation; only agent-created skills are in "
        f"scope and nothing is ever auto-deleted (archive is recoverable)."
    )
    print("  Preview now:  hermes curator run --dry-run")
    print("  Pause it:     hermes curator pause")
    print(
        "  Docs:         https://hermes-agent.nousresearch.com/docs/user-guide/features/curator"
    )

def _print_fts_optimize_available_notice() -> None:
    """Advertise the opt-in v23 search-index optimization after `hermes update`.

    Only fires when the current profile's state.db is still on the legacy
    (pre-v23) inline FTS layout. Leads with the reclaimable-space figure and
    points at the exact command. Honors ``sessions.fts_optimize_notice``:
    ``advise`` (default) prints an advisory notice, ``require`` prints a
    firmer required-upgrade notice, ``off`` suppresses it. Silent for
    fresh/already-optimized installs.
    """
    mode = "advise"
    try:
        from hermes_cli.config import load_config

        mode = str(
            ((load_config() or {}).get("sessions") or {}).get(
                "fts_optimize_notice", "advise"
            )
        ).strip().lower()
    except Exception:
        mode = "advise"
    if mode == "off":
        return

    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
    except Exception:
        return
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return
    try:
        size_gb = db_path.stat().st_size / (1024 ** 3)
    except OSError:
        return
    # Skip the notice for trivially small DBs — the win isn't worth the nag.
    if size_gb < 0.5:
        return
    db = None
    interrupted = False
    try:
        db = SessionDB(db_path=db_path, read_only=True)
        # read_only opens skip schema init, so probe the layout directly.
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        # An interrupted `optimize-storage` run: the table is already the
        # v23 shape, but backfill markers / demoted trash tables remain.
        # Offer the command again — re-running resumes and finishes it.
        interrupted = bool(
            db._conn.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', 'fts_cjk_stale') LIMIT 1"
            ).fetchone()
        )
    except Exception:
        return
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    sql = (row[0] if row else "") or ""
    if not sql or ("tool_name" in sql and not interrupted):
        # v23 layout already present (fresh/optimized) — nothing to offer.
        return

    if interrupted:
        print()
        print("◆ Session database optimization incomplete")
        print(
            "  A previous `hermes sessions optimize-storage` run was "
            "interrupted. Search still works; re-run the command to resume "
            "and finish reclaiming disk:"
        )
        print("    hermes sessions optimize-storage")
        return

    # Concrete size framing — lead with the savings the user cares about.
    est_reclaim = size_gb * 0.6
    print()
    if mode == "require":
        print("◆ Session database upgrade required")
        print(
            f"  Your search index uses the OLD storage layout and should be "
            f"upgraded. The new layout typically frees ~60% of state.db "
            f"(≈{est_reclaim:.1f} GB of your current {size_gb:.1f} GB) and is "
            f"required for continued optimal operation."
        )
    else:
        print("◆ Reclaim ~60% of your session database disk")
        print(
            f"  Your search index uses the old storage layout. Upgrading it "
            f"typically frees ~60% of state.db — about {est_reclaim:.1f} GB "
            f"of your current {size_gb:.1f} GB."
        )
    print("  Run when convenient:  hermes sessions optimize-storage")
    print(
        "  It runs in the foreground with a progress bar, is safe to "
        "interrupt/re-run, and never changes your conversations."
    )

def _print_curator_recent_run_notice() -> None:
    """Print the most recent curator run summary, exactly once.

    The curator runs in the background, so users only notice consolidations
    by stumbling into a rename; ``hermes update`` is a high-attention surface
    to show the rename map. Show-once: stamps ``last_run_summary_shown_at``
    after printing. Silent when the curator never ran, the summary was already
    shown, or it has no rename info (no archives).
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        state = curator.load_state()
    except Exception:
        return

    last_run_at = state.get("last_run_at")
    if not last_run_at:
        return  # no curator run yet — first-run notice handles this case

    if state.get("last_run_summary_shown_at") == last_run_at:
        return  # already shown for this run

    summary = state.get("last_run_summary") or ""
    if not summary:
        return

    # Only a multi-line summary (rename map appended) is worth showing; a
    # bare "auto: no changes; llm: no change" isn't.
    if "\n" not in summary:
        # Still stamp it shown so we don't reconsider it on every update.
        try:
            state["last_run_summary_shown_at"] = last_run_at
            curator.save_state(state)
        except Exception:
            pass
        return

    when = _format_time_ago(last_run_at)
    print()
    print(f"ℹ Skill curator — last run {when}")
    for line in summary.splitlines():
        print(f"  {line}")
    print(
        "  (This message shows once per curator run. "
        "View anytime: hermes curator status)"
    )

    # Stamp shown so we don't repeat on the next update.
    try:
        state["last_run_summary_shown_at"] = last_run_at
        curator.save_state(state)
    except Exception:
        pass

def _format_time_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as `Xh ago` / `Xd ago` / `Xm ago`. Best effort."""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "recently"

def _reload_process_scan_modules() -> None:
    """Force-reload the process-scan modules from disk after an update.

    ``_finish_dashboard_update_cleanup`` runs in the PRE-update process, but
    ``_scan_dashboard_processes`` lazily imports from ``_subprocess_compat``;
    a symbol the update added (``bounded_probe_run``, #87134) is missing from
    the cached OLD module and the cleanup crashes with ImportError after the
    code update already succeeded. Reload dependency-first so
    ``dashboard_procs`` binds against the fresh ``_subprocess_compat``.

    Called from the cleanup entry point (not only ``_reload_config_modules``)
    so EVERY caller — git path, Windows ZIP fallback, future ones — is covered.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                # warning, not debug: a failed reload here surfaces seconds
                # later as an ImportError in the same process — leave a trail.
                logger.warning(
                    "Could not reload %s for post-update cleanup: %s",
                    mod_name,
                    exc,
                )


def _finish_dashboard_update_cleanup(
    node_failures: list[str], already_restarted_units: "set[str] | None" = None
) -> None:
    """Refresh managed dashboards or stop stale manual ones after an update.

    *already_restarted_units* forwards the systemd unit names (no
    ``.service`` suffix) that the fleet-restart loop already restarted
    directly, so a Serve-only install's freshly restarted process isn't
    found and restarted a second time here (review on #83595).
    """
    if node_failures:
        print()
        print("  ℹ Leaving running dashboard process(es) untouched because the")
        print("    Node.js dependency refresh did not complete.")
        return

    # The scan path lazy-imports symbols from _subprocess_compat; make sure
    # both modules reflect the freshly-updated source before touching them.
    _reload_process_scan_modules()

    stop_result = _m()._kill_stale_dashboard_processes(
        restart_managed=True, already_restarted_units=already_restarted_units
    )
    if not stop_result.get("unrecovered"):
        return

    print()
    print(
        "⚠ A web dashboard/serve process was stopped during update and could "
        "not be auto-restarted."
    )
    print("  Re-launch it when you want the web UI back:")
    print("    hermes dashboard --port <port>")

def _atomic_replace_dir(src: str, dst: str) -> None:
    """Replace directory *dst* with *src* without leaving *dst* half-deleted.

    Naive ``rmtree(dst); copytree(src, dst)`` has a destructive window: a
    copy that fails partway (common on the Windows ZIP path, which only runs
    because file I/O is already flaky) leaves the old tree gone and nothing
    in its place (#49145: ``ui-tui/`` vanished and broke the TUI).

    Now a thin alias over the two-phase helpers below (#76104); retained as
    part of the ``hermes_cli.main`` re-export surface and the #49145 guard.
    """
    _commit_staged_replacements([(_stage_replacement(src, dst), dst)])


def _stage_replacement(src: str, dst: str) -> str:
    """Copy *src* to a sibling staging path for *dst*; return the staging path.

    Phase 1 of the two-phase replace. Handles both directories and plain
    files. Touches nothing live, so a failure here leaves the whole install
    untouched.
    """
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # A previous run may have died between "move dst aside" and "move staging
    # in", leaving the backup as the ONLY copy. Restore it BEFORE clearing
    # leftovers: deleting it and then failing to stage (disk exhaustion is
    # likely here) would leave a hole with nothing to roll back to.
    if not os.path.exists(dst) and os.path.exists(backup):
        os.rename(backup, dst)
    for leftover in (staging, backup):
        if os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
        elif os.path.exists(leftover):
            os.remove(leftover)
    if os.path.isdir(src):
        shutil.copytree(src, staging)
    else:
        shutil.copy2(src, staging)
    return staging


def _discard_staged(staged) -> None:
    """Remove staging paths for entries that were never committed.

    Otherwise a phase-1 failure (typically disk exhaustion) orphans one
    staging copy per processed entry — up to a full second tree — and the
    advised "re-run `hermes update`" retry fails harder with less free space.
    """
    for staging, _dst in staged:
        try:
            if os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
            elif os.path.exists(staging):
                os.remove(staging)
        except OSError as exc:  # best-effort cleanup, never fatal
            logger.warning("could not remove staging path %s: %s", staging, exc)


def _commit_staged_replacements(staged) -> None:
    """Phase 2: swap every staged entry into place, rolling back all on failure.

    ``_atomic_replace_dir`` made each *individual* swap safe, but the ZIP
    update loops over ~90 top-level entries and nothing made the loop atomic
    *as a whole*: a partway failure left a mixed-version tree — every file
    valid, the combination unbootable (#76104; also #76091, #63717).

    Covers plain files too: the repo root holds 20 first-party modules, so a
    files-only failure reproduces the same bug class. Every swap is an
    ``os.rename`` onto a just-moved-aside path — atomic on POSIX and NTFS —
    so a file swap can't leave a half-written module the way ``copy2`` onto
    a live path can.

    Stage-all-then-swap-all shrinks the failure window from "a full tree
    copy" to "N renames" and makes it recoverable: a failed swap restores
    every entry already swapped, so the tree lands wholly new or wholly old.
    """
    swapped: list[tuple[str, str]] = []  # (dst, backup) in swap order; "" = absent
    try:
        for staging, dst in staged:
            backup = f"{dst}.hermes-update-old"
            if os.path.exists(dst):
                os.rename(dst, backup)
                swapped.append((dst, backup))
            else:
                swapped.append((dst, ""))
            os.rename(staging, dst)
    except OSError:
        # Undo every swap already made so the install stays self-consistent.
        for dst, backup in reversed(swapped):
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                elif os.path.exists(dst):
                    os.remove(dst)
                if backup and os.path.exists(backup):
                    os.rename(backup, dst)
            except OSError as exc:
                # Keep restoring the rest — a silent failure here is the one
                # thing that turns a recoverable rollback into a mixed tree,
                # so say so rather than swallowing it.
                logger.warning("rollback failed for %s: %s", dst, exc)
        raise
    # All swaps succeeded — drop the backups (best-effort, never fatal).
    for _dst, backup in swapped:
        if backup and os.path.isdir(backup):
            shutil.rmtree(backup, ignore_errors=True)
        elif backup and os.path.exists(backup):
            try:
                os.remove(backup)
            except OSError:
                pass


def _branch_head_label(git_cmd=None, cwd=None) -> str | None:
    """``"<branch> @ <short-sha>"`` for the checkout, or None when unknown.

    Appended to update summary lines so branch drift is visible (2026-08-17
    incident: a checkout parked on a stale feature branch got "✓ Update
    complete!" with nothing saying WHERE it sat). Never raises.
    """
    try:
        cmd = list(git_cmd) if git_cmd else ["git"]
        root = cwd if cwd is not None else _m().PROJECT_ROOT
        branch = subprocess.run(
            cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        sha = subprocess.run(
            cmd + ["rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        branch_name = branch.stdout.strip()
        sha_text = sha.stdout.strip()
        if branch.returncode != 0 or sha.returncode != 0 or not sha_text:
            return None
        if not branch_name:
            return None
        label = "detached" if branch_name == "HEAD" else branch_name
        return f"{label} @ {sha_text}"
    except Exception:
        return None


def _branch_head_suffix(git_cmd=None, cwd=None) -> str:
    """`` [<branch> @ <sha>]`` suffix for summary lines ("" when unknown)."""
    label = _branch_head_label(git_cmd, cwd)
    return f" [{label}]" if label else ""


def _assess_parked_branch_switch(
    git_cmd: list[str], cwd: Path, current_branch: str, target_branch: str
) -> tuple[bool, str]:
    """Decide whether it is safe to auto-switch a parked feature branch back
    to the update target.

    Live incident (2026-08-17): the checkout sat on a stale feature branch;
    ``hermes update`` autostashed, ran post-update steps and printed
    "✓ Code updated!" while the running code stayed days behind main.

    - (True, "") — tree + index clean AND every parked commit is already in
      ``origin/<target_branch>`` (``git cherry`` reports no ``+`` lines).
    - (True, "unmerged:<count>") — tree clean but the branch has commits not
      in the target. Switching is safe (``git checkout`` never discards
      committed work) but the caller must print a LOUD notice naming the
      branch and count. Non-interactive callers (desktop button, gateway
      /update, cron) rely on this: they can't resolve a skip, so a clean
      checkout must always reach the target.
    - (False, <reason>) — dirty tree, git errors, or the
      ``updates.auto_switch_parked_branch: false`` opt-out; caller must NOT
      touch the branch. A dirty tree is the genuinely unsafe case: uncommitted
      work riding an autostash across branches is how the incident started.

    Block reasons: "disabled", "dirty", "unverifiable".
    """
    try:
        from hermes_cli.config import load_config

        _update_cfg = (load_config() or {}).get("updates", {})
        if isinstance(_update_cfg, dict) and not bool(
            _update_cfg.get("auto_switch_parked_branch", True)
        ):
            return False, "disabled"
    except Exception as exc:
        # A config read failure must not disable the guard's safety checks —
        # fall through to them with the default (auto-switch allowed).
        logger.debug("Could not read updates.auto_switch_parked_branch: %s", exc)

    status = _git_run(git_cmd, ["status", "--porcelain"], cwd)
    if status.returncode != 0:
        return False, "unverifiable"
    if status.stdout.strip():
        return False, "dirty"

    cherry = _git_run(git_cmd, ["cherry", f"origin/{target_branch}"], cwd)
    if cherry.returncode != 0:
        return False, "unverifiable"
    unmerged = [
        line for line in cherry.stdout.splitlines() if line.startswith("+")
    ]
    if unmerged:
        # Clean tree: switching is safe (checkout keeps the commits on the
        # branch). The reason string tells the caller to print the loud
        # "branch kept with N unmerged commit(s)" notice.
        return True, f"unmerged:{len(unmerged)}"
    return True, ""


def _print_parked_branch_skip_warning(
    git_cmd: list[str],
    cwd: Path,
    current_branch: str,
    target_branch: str,
    reason: str,
) -> None:
    """LOUD block explaining why the code update was skipped on a parked
    branch, with the behind-count and the exact commands to resolve."""
    behind = None
    try:
        behind_result = _git_run(git_cmd, ["rev-list", f"HEAD..origin/{target_branch}", "--count"], cwd)
        if behind_result.returncode == 0 and behind_result.stdout.strip():
            behind = int(behind_result.stdout.strip())
    except Exception:
        behind = None

    if reason == "dirty":
        why = "the working tree has uncommitted changes"
    elif reason == "disabled":
        why = "updates.auto_switch_parked_branch is set to false in config.yaml"
    else:
        why = (
            f"the branch state could not be verified against "
            f"origin/{target_branch}"
        )

    bar = "=" * 68
    print()
    print(bar)
    print(f"⚠ CODE UPDATE SKIPPED — checkout is parked on '{current_branch}'")
    print(f"  Not auto-switching to {target_branch}: {why}.")
    if behind is not None and behind > 0:
        print(
            f"  This checkout is {behind} commit(s) BEHIND "
            f"origin/{target_branch} — the code you are running is stale."
        )
    print()
    print("  To resolve, inspect the branch and switch back yourself:")
    print(f"    git -C {cwd} status")
    print(f"    git -C {cwd} checkout {target_branch} && hermes update")
    print(
        "  (commit or stash your work on the branch first if you want to "
        "keep it)"
    )
    print(bar)


def _print_parked_branch_kept_notice(
    current_branch: str, target_branch: str, unmerged_count: str
) -> None:
    """LOUD notice printed when a clean parked branch with unmerged commits
    is auto-switched back to the update target.

    Non-interactive callers can't resolve a skip, so a clean checkout always
    proceeds — but the unmerged work must be impossible to miss. The commits
    stay on the branch (``git checkout`` never discards committed work).
    """
    bar = "=" * 68
    print()
    print(bar)
    print(
        f"⚠ Checkout was parked on '{current_branch}' with "
        f"{unmerged_count} commit(s) not merged into origin/{target_branch}."
    )
    print(
        f"  Switching to {target_branch} so the update can proceed — your "
        f"commit(s) are safe on '{current_branch}'."
    )
    print()
    print("  To pick the work back up later:")
    print(f"    git checkout {current_branch}")
    print(bar)


def _print_update_completion(message: str) -> None:
    """Print an update outcome plus, when the dashboard launched this run with
    an action id, a terminal receipt line the Desktop can match after the
    dashboard restarts (#47359 / #58764). The outcome line carries the
    branch + HEAD short-sha so branch drift is visible (2026-08-17 incident)."""
    print(f"{message}{_branch_head_suffix()}")
    action_id = os.environ.get("HERMES_ACTION_ID", "")
    if len(action_id) == 32 and all(char in "0123456789abcdef" for char in action_id):
        print(f"=== hermes-update completed {action_id} ===")


def _called_process_error_cmd_parts(exc: subprocess.CalledProcessError) -> list[str]:
    """Normalize ``CalledProcessError.cmd`` into argv-style tokens."""
    cmd = exc.cmd
    if cmd is None:
        return []
    if isinstance(cmd, (str, bytes)):
        text = cmd.decode("utf-8", "replace") if isinstance(cmd, bytes) else cmd
        try:
            return shlex.split(text, posix=os.name != "nt")
        except ValueError:
            return text.split()
    return [str(part) for part in cmd]


def _called_process_error_is_git(exc: subprocess.CalledProcessError) -> bool:
    """True when the failed subprocess was git itself."""
    parts = _called_process_error_cmd_parts(exc)
    if not parts:
        return False
    # Windows argv may use backslashes; basename() on POSIX would otherwise
    # keep the whole path. Normalize separators before taking the name.
    name = os.path.basename(parts[0].replace("\\", "/")).lower()
    return name in {"git", "git.exe"}


def _called_process_error_is_python_dep_install(
    exc: subprocess.CalledProcessError,
) -> bool:
    """True when the failed subprocess was a uv/pip (or ensurepip) install."""
    parts = [part.lower() for part in _called_process_error_cmd_parts(exc)]
    if not parts:
        return False
    exe = os.path.basename(parts[0].replace("\\", "/"))
    if "ensurepip" in parts:
        return True
    if "install" in parts and (
        "pip" in parts or exe in {"pip", "pip.exe", "pip3", "pip3.exe", "uv", "uv.exe"}
    ):
        return True
    return False


def _format_update_failure_stage(exc: subprocess.CalledProcessError) -> str:
    """Name the update stage that actually failed.

    The git pull and the Python-dependency install share one ``try`` in
    ``_cmd_update_impl``. Calling every ``CalledProcessError`` a git failure
    (the historical Windows message) sent users hunting in the wrong place
    and, worse, keyed the ZIP overlay on exception *type* rather than on git
    actually having failed (#87304, #85840).
    """
    if _called_process_error_is_python_dep_install(exc):
        return "Python dependency install failed"
    if _called_process_error_is_git(exc):
        return "Git update failed"
    return "Update step failed"


def _shim_quarantine_error_type() -> "type[BaseException]":
    """The strict-quarantine refusal type, resolved lazily through ``_m()``.

    Falls back to a never-raised private type when main.py lacks it (torn
    mid-update tree), so the ``except`` clause stays valid.
    """
    cls = getattr(_m(), "ShimQuarantineError", None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls

    class _Never(Exception):
        pass

    return _Never


def _refuse_update_for_contended_shims(exc: BaseException) -> None:
    """Refuse the dependency sync when live shims could not be quarantined.

    #87331 fail-closed half: a shim rename that failed every retry proves a
    process holds the venv without FILE_SHARE_DELETE — running the installer
    anyway is exactly how the venv ends up stranded between versions. The
    code swap (when one happened) is already committed; only the dependency
    install is deferred, via the update-incomplete marker, to the next fresh
    launch after the holder exits. Exits 2 (refused) so the command-boundary
    receipt net records it as a refusal, not a failure.
    """
    print("✗ Cannot continue the update: live Hermes launcher(s) could not be")
    print("  moved aside:")
    for name in getattr(exc, "failed_shims", []) or ["hermes.exe"]:
        print(f"    {name}")
    print("  Another process is holding this install's venv — typically Hermes")
    print("  Desktop, a gateway, or another hermes REPL — and mutating the venv")
    print("  now would strand it half-updated.")
    print("  The dependency install has been deferred: close the process(es)")
    print("  above, then run any `hermes` command to finish it automatically.")
    # Idempotent: the git path already dropped the marker before the sync;
    # this covers the ZIP/repair paths so the deferral is never silent.
    _write_update_incomplete_marker()
    sys.exit(2)


def _should_zip_fallback_on_update_error(exc: BaseException) -> bool:
    """ZIP fallback is for Windows git file-I/O breakage, not later stages.

    A dependency-install failure (locked ``hermes.exe`` / ``uv pip install``
    exit 2) is not a git failure. The pull has already succeeded by then, so
    re-downloading the source ZIP cannot fix the install and would replace
    every top-level entry except ``venv`` / ``node_modules`` / ``.git`` /
    ``.env`` — permanently deleting uncommitted edits and untracked files.
    """
    return (
        isinstance(exc, subprocess.CalledProcessError)
        and _m()._is_windows()
        and _called_process_error_is_git(exc)
    )


def _print_called_process_error_tail(
    exc: subprocess.CalledProcessError, *, limit: int = 12
) -> None:
    """Print a captured stderr/stdout tail when the failing call recorded one."""
    blob = exc.stderr or exc.stdout or ""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    lines = [line for line in str(blob).splitlines() if line.strip()]
    if not lines:
        return
    print("  Last output:")
    for line in lines[-limit:]:
        print(f"    {line}")


def _zip_overlay_block_reason(
    root: Path, *, ignore_staging_artifacts: bool = False
) -> Optional[str]:
    """Why overlaying a ZIP onto ``root`` would destroy work, or None if safe.

    The ZIP path swaps every top-level entry (minus a tiny preserve set) and
    deletes the backups, so uncommitted edits and untracked files are gone.
    Fails closed when git status cannot run (#87304).

    ``ignore_staging_artifacts`` is for the pre-swap re-check: phase 1 leaves
    ``*.hermes-update-staging`` siblings that git reports as untracked; they
    are our own artifacts, and without the filter the re-check always refuses.
    """
    if not (root / ".git").exists():
        return None
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
    result = subprocess.run(
        # -uall: a user-level ``status.showUntrackedFiles = no`` must not
        # blind this guard. --ignored=matching: gitignored files are still
        # USER DATA the overlay would delete (#87392); ``matching`` reports an
        # ignored dir as one ``dir/`` line (cheaper, same verdict below).
        # NOTE: ``--ignored=all`` is NOT a valid git mode — exits 128 and
        # would fail-close every ZIP update.
        git_cmd + ["status", "--porcelain", "--untracked-files=all", "--ignored=matching"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f" ({detail[0]})" if detail else ""
        return f"could not check the working tree{suffix}"
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    # --ignored=all reports the ZIP path's own preserved entries (venv,
    # node_modules are gitignored on every normal install). The swap never
    # touches those top-level entries, so they must not turn into a false
    # dirty-tree refusal. Everything else — including ignored files — blocks.
    lines = [line for line in lines if not _is_zip_preserved_entry_status_line(line)]
    if ignore_staging_artifacts:
        lines = [
            line for line in lines if not _is_zip_staging_artifact_status_line(line)
        ]
    if lines:
        return "the working tree has uncommitted changes or untracked files"
    return None


_ZIP_STAGING_ARTIFACT_SUFFIXES = (".hermes-update-staging", ".hermes-update-old")
# Single source of truth for the top-level entries the ZIP swap preserves —
# consumed by both the dirty-tree filter below and _update_via_zip's swap loop.
_ZIP_PRESERVED_TOP_LEVEL = {"venv", "node_modules", ".git", ".env"}


def _is_zip_preserved_entry_status_line(line: str) -> bool:
    """True when every path on a porcelain status line sits under a top-level
    entry the ZIP swap preserves.

    The ``" -> "`` split applies ONLY to rename/copy codes (R/C): porcelain
    v1 doesn't quote plain filenames with spaces, so an ignored file named
    ``venv -> node_modules`` on a ``!!``/``??`` line is ONE path — splitting
    would fail-open into the destructive swap. Requiring EVERY path preserved
    keeps renames out of a preserved dir (``R venv/x -> src/x``) blocking.
    """
    status, payload = (line[:2], line[3:]) if len(line) >= 3 else ("", line)
    is_rename = any(code in "RC" for code in status)
    paths = payload.split(" -> ") if is_rename else [payload]
    for path in paths:
        top_level = (
            path.strip().strip('"').replace("\\", "/").rstrip("/").split("/", 1)[0]
        )
        if top_level not in _ZIP_PRESERVED_TOP_LEVEL:
            return False
    return True


def _is_zip_staging_artifact_status_line(line: str) -> bool:
    """True when a porcelain status line is our own two-phase-swap artifact."""
    payload = line[3:] if len(line) >= 3 else line
    top_level = (
        payload.strip().strip('"').replace("\\", "/").rstrip("/").split("/", 1)[0]
    )
    return top_level.endswith(_ZIP_STAGING_ARTIFACT_SUFFIXES)


def _abort_zip_update_if_dirty_tree() -> None:
    """Refuse to overlay a ZIP onto a dirty git checkout (#87304)."""
    reason = _zip_overlay_block_reason(_m().PROJECT_ROOT)
    if reason is None:
        return
    print(f"✗ ZIP fallback refused: {reason}.")
    print(
        "  Overlaying the ZIP would overwrite uncommitted edits and permanently "
        "delete untracked files."
    )
    print("  Stash or commit your changes, then rerun `hermes update`.")
    print("  To inspect: git status --porcelain")
    _m().sys.exit(1)


def _read_project_version() -> str | None:
    """Read the ``version`` field from the checkout's pyproject.toml.

    On-disk file, not importlib.metadata: after a pull the installed
    metadata still describes the OLD version. Returns None on any failure —
    version reporting is cosmetic and must never break an update.
    """
    try:
        import tomllib

        with open(_m().PROJECT_ROOT / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            version = tomllib.load(fh).get("project", {}).get("version")
        return str(version) if version else None
    except Exception:
        return None


def _update_complete_message(pre_version: str | None) -> str:
    """Completion line with the version transition when it is known.

    Ported from PrimeIntellect-ai/prime-agent#630: show ``v0.19.4 → v0.20.0``
    after a self-update. Plain message when either side is unknown or the
    version did not change (several commits within one release).
    """
    post_version = _read_project_version()
    if pre_version and post_version and pre_version != post_version:
        return f"✓ Update complete! (v{pre_version} → v{post_version})"
    if post_version:
        return f"✓ Update complete! (v{post_version})"
    return "✓ Update complete!"


def _post_update_sqlite_runtime_status():
    """Return whether the interpreter used after update has safe SQLite."""
    from hermes_constants import project_venv_dir
    from hermes_cli.sqlite_runtime import probe_sqlite_runtime

    venv_dir = project_venv_dir(_m().PROJECT_ROOT)
    python = (
        venv_python_path(venv_dir, windows=_m()._is_windows())
        if venv_dir is not None
        else Path(sys.executable)
    )
    info = probe_sqlite_runtime(python)
    return info is not None and not info.wal_reset_vulnerable, info


def _print_verified_update_completion(message: str) -> bool:
    """Print a success completion only after probing the next Hermes runtime."""
    if not message.startswith("✓"):
        _print_update_completion(message)
        return False
    sqlite_runtime_ok, sqlite_info = _post_update_sqlite_runtime_status()
    if sqlite_info is None:
        # Grace path: an unprobeable interpreter (no venv in a dev checkout,
        # probe subprocess unavailable) must not fail an otherwise-successful
        # update — only a POSITIVE vulnerable probe withholds success
        # (same contract as _venv_core_imports_healthy's unknown states).
        logger.debug("Post-update SQLite runtime probe unavailable; not blocking")
        _print_update_completion(message)
        return True
    if sqlite_runtime_ok:
        _print_update_completion(message)
        return True
    print()
    detail = (
        f"SQLite {sqlite_info.sqlite_version_string} still has the "
        "WAL-reset corruption bug"
    )
    print(f"⚠ Update partially complete — {detail}.")
    print(
        "  Rebuild the Hermes venv with a uv-managed Python, restart Hermes, "
        "then verify with `hermes doctor`."
    )
    return False


def _clear_stale_sqlite_sidecars(db_path: Path) -> None:
    """Delete the WAL / shared-memory / rollback-journal files next to *db_path*.

    Call immediately before overwriting a database with a snapshot image.
    Quick snapshots come from ``sqlite3.backup()`` (``backup._safe_copy_db``),
    so the image is checkpointed and owns no WAL — which is why
    ``backup._EXCLUDED_SUFFIXES`` ships no sidecars. Copying the image
    replaces only the main file, so a ``-wal``/``-shm`` left by the *old*
    database (crashed writer, undrained second process) is replayed over the
    fresh image on next open: it passes ``PRAGMA integrity_check`` while
    serving the old contents, and the first checkpoint makes that permanent.

    Safe here because the sidecars belong to a database the caller has already
    declared corrupt and is about to discard.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _print_update_summary(
    *,
    node_failures: list,
    desktop_build_ok: bool,
    pre_update_version: str | None,
) -> bool:
    """Final update banner. A failed Desktop rebuild is non-fatal for the
    Python side, but must not print ``✓ Update complete!`` (#88251)."""
    sqlite_runtime_ok, sqlite_info = _post_update_sqlite_runtime_status()
    if sqlite_info is None:
        # Grace path: an unprobeable interpreter must not fail the update —
        # only a POSITIVE vulnerable probe demotes success to partial.
        sqlite_runtime_ok = True
    print()
    if node_failures or not desktop_build_ok or not sqlite_runtime_ok:
        parts = []
        if node_failures:
            parts.append(
                f"Node.js dependencies for {', '.join(node_failures)} did not refresh"
            )
        if not desktop_build_ok:
            parts.append(
                "the desktop app was not rebuilt and is still on the previous build"
            )
        if not sqlite_runtime_ok and sqlite_info is not None:
            parts.append(
                f"SQLite {sqlite_info.sqlite_version_string} still has the "
                "WAL-reset corruption bug"
            )
        print("⚠ Update partially complete — " + "; ".join(parts) + ".")
        if node_failures:
            print("  Code and Python deps are updated, but the dashboard/TUI may")
            print("  be in a mixed state until the Node deps are rebuilt.")
        if not desktop_build_ok:
            print("  Run `hermes desktop` to retry the desktop rebuild.")
        if not sqlite_runtime_ok:
            print(
                "  The Python runtime remediation did not complete. Run `hermes "
                "update` again; if SQLite is unchanged, rebuild the Hermes venv "
                "with a uv-managed Python, restart Hermes, then verify with "
                "`hermes doctor`."
            )
    else:
        _print_update_completion(_update_complete_message(pre_update_version))
    return desktop_build_ok and sqlite_runtime_ok


def _write_gateway_update_exit_code(ok: bool) -> None:
    path = get_hermes_home() / ".update_exit_code"
    try:
        path.write_text("0" if ok else "1", encoding="utf-8")
    except OSError:
        pass


def _restore_state_db_from_snapshot(state_path: Path, snap_state: Path) -> bool:
    """Replace *state_path* with the snapshot image at *snap_state*.

    Shared by the ZIP and git-pull auto-restore paths. Stale sidecars are
    cleared before the copy so the corrupt database's WAL replay cannot
    silently overwrite the restored image (:func:`_clear_stale_sqlite_sidecars`).

    Refuses (``False``) while another process — or a live connection in THIS
    process — holds the database or its sidecars: copying over a live
    writer's inode desyncs its page cache/WAL index from the file bytes and
    its next checkpoint clobbers pages (#90950 page-1 clobber). ``None``
    (scan unavailable) proceeds: gateways are already drained, and refusing
    on "unknown" would disable auto-restore on every non-Linux host.

    Returns ``True`` when the restored file passes an integrity check. Raises
    ``OSError`` if the copy itself fails (callers already report it).
    """
    from hermes_cli.backup import _foreign_db_holder_pids, verify_sqlite_integrity
    from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access

    holders = _foreign_db_holder_pids(state_path)
    if holders:
        print(
            f"  ✗ Auto-restore refused: process(es) {holders} still hold "
            "state.db or its WAL open. Stop them (hermes gateway stop), "
            "then restore manually with /snapshot restore."
        )
        return False
    # The foreign-pid scan excludes THIS process, but an in-process SessionDB
    # handle is just as live: unlinking -wal/-shm and copy2-ing under it
    # leaves this process checkpointing through deleted-inode sidecars (the
    # #90950 split brain, reproduced live via `/proc/self/fd`).
    # ``offline_file_access`` fails CLOSED on any tracked connection and holds
    # the lifecycle lock across clear + copy so none can appear mid-swap.
    try:
        with offline_file_access(state_path, what="restore a snapshot over"):
            _clear_stale_sqlite_sidecars(state_path)
            shutil.copy2(snap_state, state_path)
    except LiveConnectionError as exc:
        print(
            f"  ✗ Auto-restore refused: {exc} Close the in-process database "
            "handles (or restart Hermes) and retry."
        )
        return False
    restored = verify_sqlite_integrity(
        state_path, check_header=True, run_pragma=True
    )
    return bool(restored.get("valid"))


def _verify_and_restore_one_state_db(home: Path, *, label: str) -> None:
    """Post-update integrity check + auto-restore for ONE home's state.db.

    Shared by the root-DB and sibling-profile guards (ZIP update path and
    git-pull path both route here). A corrupt live DB is restored from the
    most recent valid snapshot under that home's own state-snapshots dir.
    Never raises: a guard that crashes the update tail would be worse than
    the corruption it detects.
    """
    try:
        from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity

        state_path = home / "state.db"
        if not state_path.exists():
            return
        ok = verify_sqlite_integrity(state_path, check_header=True, run_pragma=True)
        if ok.get("valid"):
            logger.debug(
                "Post-update state.db integrity OK (%s): %s",
                label,
                ok.get("message"),
            )
            return
        print()
        print(
            f"⚠ state.db is corrupted after update ({label}): "
            + ok.get("message", "unknown error")
        )
        snap_root = _quick_snapshot_root(home)
        if not snap_root.exists():
            print("  ⚠ No pre-update snapshot for this home")
            return
        for snap_dir in sorted(
            (d for d in snap_root.iterdir() if d.is_dir()), reverse=True
        ):
            snap_state = snap_dir / "state.db"
            if not snap_state.exists():
                continue
            snap_ok = verify_sqlite_integrity(
                snap_state, check_header=True, run_pragma=True
            )
            if not snap_ok.get("valid"):
                continue
            try:
                if _restore_state_db_from_snapshot(state_path, snap_state):
                    print(
                        f"  ✓ Auto-restored from snapshot {snap_dir.name} ({label})"
                    )
                else:
                    print(
                        "  ✗ Auto-restore FAILED — restored copy also failed "
                        "integrity"
                    )
            except OSError as exc:
                print(f"  ✗ Auto-restore file copy failed: {exc}")
            return
        print("  ⚠ No valid pre-update snapshot found for this home")
    except Exception as exc:
        logger.debug(
            "Post-update state.db guard (%s) failed: %s", label, exc
        )


def _verify_and_restore_state_dbs_post_update() -> None:
    """Post-update integrity guard for the ROOT state.db AND every sibling
    profile's state.db (#97994).

    The pre-update snapshot already covers siblings (#66140), but the guard
    only verified the root DB — a corrupted profile DB was never detected or
    restored, its sessions silently gone while the root passed.
    """
    home = get_hermes_home()
    _verify_and_restore_one_state_db(home, label="default home")
    try:
        from hermes_cli.backup import _sibling_profile_homes

        for name, profile_home in _sibling_profile_homes(home):
            _verify_and_restore_one_state_db(profile_home, label=f"profile {name}")
    except Exception as exc:
        logger.debug("Sibling-profile state.db guard sweep failed: %s", exc)


def _ensure_venv_pip(pip_cmd: list, python_exe: str) -> None:
    """Bootstrap pip back into the venv via ensurepip when ``pip --version`` fails
    (some environments lose it); call before the editable install."""
    try:
        subprocess.run(
            pip_cmd + ["--version"],
            cwd=_m().PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [python_exe, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=_m().PROJECT_ROOT,
            check=True,
        )


def _print_bundled_skills_sync_report() -> None:
    """Run ``sync_skills`` (copies new, updates changed, respects user deletions) and print its summary."""
    from tools.skills_sync import sync_skills

    result = sync_skills(quiet=True)
    if result["copied"]:
        print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
    if result.get("updated"):
        print(
            f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
        )
    if result.get("user_modified"):
        print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
        print(
            "    → see them: hermes skills list-modified  "
            "(diff/reset to resume updates)"
        )
    if result.get("cleaned"):
        print(f"  − {len(result['cleaned'])} removed from manifest")
    if result.get("relocated"):
        print(
            f"  → {len(result['relocated'])} moved to new upstream paths: "
            f"{', '.join(result['relocated'])}"
        )
    if not result["copied"] and not result.get("updated"):
        print("  ✓ Skills are up to date")


def _update_via_zip(args, *, had_desktop_app_before_update: bool = False) -> bool:
    """Update Hermes Agent by downloading a ZIP archive.

    Used on Windows when git file I/O is broken (antivirus, NTFS filter
    drivers causing 'Invalid argument' errors on file creation).

    Returns ``False`` when a Desktop rebuild ran and failed; ``True`` otherwise.
    """
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    import tempfile
    import zipfile
    from urllib.request import urlretrieve

    # Snapshot the pre-update version before files are replaced so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()

    # The static GitHub archive is fine for "main" but would silently ignore
    # --branch — the exact silent-divergence bug --branch was added to
    # prevent. Refuse rather than lie.
    branch = _m()._resolve_update_branch(args)
    if branch != "main":
        print(
            f"✗ --branch={branch} is not supported on the Windows ZIP-fallback "
            "update path."
        )
        print(
            "  This path runs when git file I/O is broken on the system. "
            "Either resolve the git-side breakage (typically an antivirus "
            "or NTFS filter holding files open) and rerun `hermes update "
            f"--branch {branch}`, or update against main with `hermes update`."
        )
        _m().sys.exit(1)
    _abort_zip_update_if_dirty_tree()
    zip_url = (
        f"https://github.com/NousResearch/hermes-agent/archive/refs/heads/{branch}.zip"
    )

    print("→ Downloading latest version...")
    tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
    try:
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)

        print("→ Extracting...")
        import stat as _stat
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Reject zip-slip (path traversal) AND symlink members: a
            # hermes-agent source ZIP never legitimately contains symlinks,
            # and a compromised mirror could use them to plant files anywhere.
            tmp_dir_real = os.path.realpath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
                if (
                    not member_path.startswith(tmp_dir_real + os.sep)
                    and member_path != tmp_dir_real
                ):
                    raise ValueError(
                        f"Zip-slip detected: {member.filename} escapes extraction directory"
                    )
                # Unix mode lives in the upper 16 bits of external_attr;
                # mask to the file-type bits.
                mode = (member.external_attr >> 16) & 0o170000
                if _stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP contains unsupported symlink member: {member.filename}"
                    )
            zf.extractall(tmp_dir)

        # GitHub ZIPs extract to hermes-agent-<branch>/
        extracted = os.path.join(tmp_dir, f"hermes-agent-{branch}")
        if not os.path.isdir(extracted):
            for d in os.listdir(tmp_dir):
                candidate = os.path.join(tmp_dir, d)
                if os.path.isdir(candidate) and d != "__MACOSX":
                    extracted = candidate
                    break

        preserve = _ZIP_PRESERVED_TOP_LEVEL
        entries = [i for i in os.listdir(extracted) if i not in preserve]

        # Two-phase replace (#76104): phase 1 stages every entry (dirs AND
        # top-level files — the repo root holds 20 first-party modules) beside
        # its target; phase 2 swaps all in with same-filesystem renames and
        # rolls back on any failure. One-at-a-time replacement left `agent/`
        # new and `tools/` stale on interruption: all files valid, tree
        # unbootable. Staging costs one extra tree copy — check space up front.
        need = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for entry in entries
            for dirpath, _dirs, files in os.walk(os.path.join(extracted, entry))
            for f in files
        ) + sum(
            os.path.getsize(os.path.join(extracted, e))
            for e in entries
            if os.path.isfile(os.path.join(extracted, e))
        )
        # Swaps are renames, so only the staging copy is new: require it plus
        # 20% headroom, not 2x — which would block updates on exactly the
        # space-constrained machines most likely to hit this path.
        required = int(need * 1.2)
        free = shutil.disk_usage(str(_m().PROJECT_ROOT)).free
        if free < required:
            raise RuntimeError(
                f"not enough free disk space to stage the update safely "
                f"(need ~{required // (1024 * 1024)} MB, have "
                f"{free // (1024 * 1024)} MB)"
            )

        staged: list[tuple[str, str]] = []
        try:
            for item in entries:
                src = os.path.join(extracted, item)
                dst = os.path.join(str(_m().PROJECT_ROOT), item)
                staged.append((_stage_replacement(src, dst), dst))
                # #70337/#87331: the source ZIP lacks apps/desktop/release/
                # (the BUILT desktop app); swapping `apps` without it deletes
                # the build and breaks the shortcut. Graft the live release
                # dir into the staged copy BEFORE the swap.
                if item == "apps":
                    live_release = os.path.join(dst, "desktop", "release")
                    staged_release = os.path.join(
                        staged[-1][0], "desktop", "release"
                    )
                    if os.path.isdir(live_release) and not os.path.exists(
                        staged_release
                    ):
                        os.makedirs(os.path.dirname(staged_release), exist_ok=True)
                        shutil.copytree(live_release, staged_release)
        except Exception:
            # Nothing is live yet; drop the partial staging copies so a retry
            # starts from the same free space this attempt did.
            _discard_staged(staged)
            raise

        try:
            # Re-check right before the swap (#87304 TOCTOU): download +
            # extract + staging can take minutes, and work created meanwhile
            # would be destroyed. Our own staging siblings are filtered out.
            recheck_reason = _zip_overlay_block_reason(
                _m().PROJECT_ROOT, ignore_staging_artifacts=True
            )
            if recheck_reason is not None:
                _discard_staged(staged)
                print(f"✗ ZIP fallback aborted before the swap: {recheck_reason}.")
                print(
                    "  Files appeared in the checkout while the update was "
                    "downloading; committing the swap would delete them."
                )
                print("  Stash or commit your changes, then rerun `hermes update`.")
                _m().sys.exit(1)
            _commit_staged_replacements(staged)
        except Exception:
            # Rollback restored the swapped entries, but staging copies for
            # the rest (possibly most of a tree) remain. Drop them, or the
            # retry's up-front free-space check (which runs BEFORE per-entry
            # leftover cleanup) fails on our litter. Safe post-rollback:
            # _discard_staged skips paths that no longer exist.
            _discard_staged(staged)
            raise
        update_count = len(staged)

        print(f"✓ Updated {update_count} items from ZIP")

    except Exception as e:
        print(f"✗ ZIP update failed: {e}")
        # The two-phase replace either commits every entry or rolls them all
        # back, so a failure here does not leave a mixed-version tree — don't
        # scare the user toward a reinstall they don't need.
        print("  Your existing install was left in place.")
        print(
            "  Re-run `hermes update` to retry; if the agent won't start, "
            "reinstall from https://hermes-agent.nousresearch.com"
        )
        _m().sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _sweep_bytecode_after_update(branch)

    # Reinstall Python deps: prefer .[all]; if one extra breaks, keep base
    # deps and retry the remaining extras individually so working
    # capabilities aren't silently stripped. Self-lock deferral (#86735): the
    # code swap is committed; defer only the dependency sync when this
    # process holds a native extension the sync must rewrite.
    _m()._abort_dependency_sync_if_self_locked()
    print("→ Updating Python dependencies...")

    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [_m().sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    if uv_bin:
        # Same third-party UV-env isolation as the main update path (#83914):
        # a user-level UV_PYTHON_INSTALL_DIR / UV_PYTHON from unrelated
        # software must not steer which interpreter uv resolves here.
        from hermes_cli.managed_uv import managed_python_env

        uv_env = managed_python_env()
        uv_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
        if _m()._is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
        try:
            _m()._install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
        except _shim_quarantine_error_type() as _sqe:
            # #87331: this runs inside the ZIP-fallback error handler, so the
            # boundary except clause in cmd_update cannot catch it — refuse
            # here with the same defer-via-marker contract.
            _refuse_update_for_contended_shims(_sqe)
    else:
        # sys.executable -m pip avoids PEP 668 'externally-managed-environment' errors.
        _ensure_venv_pip(pip_cmd, _m().sys.executable)
        _m()._install_python_dependencies_with_optional_fallback(pip_cmd)

    install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
    install_env = uv_env if uv_bin else None
    _m()._restore_active_tool_dependencies(
        active_tool_dependencies,
        install_prefix,
        env=install_env,
    )

    # ZIP path parity: heal the active memory provider's bridge packages
    # after the dependency reinstall, same as the git-pull path (#53272,
    # #70636).
    _m()._refresh_active_memory_provider_dependencies()

    # Verify the tree actually imports (catches the parse-OK-but-skewed tree
    # an interrupted copy leaves). Placed *after* the dependency reinstall so
    # a genuinely-new third-party requirement isn't misreported as a partial
    # copy. No SHA to roll back to here — surface a concrete recovery step
    # instead of reporting success over a bricked install.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print("✗ Update left the install in an unimportable state:")
        print(f"  {failing_module}: {import_error}")
        print()
        print("  This usually means the copy was interrupted partway through.")
        print("  Re-run `hermes update` to complete it.")
        _m().sys.exit(1)

    node_failures = _update_node_dependencies()
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    desktop_build_ok = _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    )

    try:
        print("→ Syncing bundled skills...")
        _print_bundled_skills_sync_report()
    except Exception:
        pass

    # Seed the model-catalog disk cache from the freshly-unpacked checkout
    # (same rationale as the git-pull path in _cmd_update_impl). Non-fatal.
    try:
        from hermes_cli.model_catalog import seed_cache_from_checkout

        if seed_cache_from_checkout(_m().PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")
    except Exception as e:
        logger.debug("Model catalog seed during zip update failed: %s", e)

    # Post-update state.db integrity guard (#68474, #97994): root home AND
    # every sibling profile, each auto-restored from its own snapshot.
    try:
        _verify_and_restore_state_dbs_post_update()
    except Exception as exc:
        logger.debug(
            "Post-update state.db integrity check (zip path) failed: %s", exc
        )

    update_complete = _print_update_summary(
        node_failures=node_failures,
        desktop_build_ok=desktop_build_ok,
        pre_update_version=pre_update_version,
    )
    try:
        _print_curator_first_run_notice()
    except Exception as e:
        logger.debug("Curator first-run notice failed: %s", e)
    try:
        _print_curator_recent_run_notice()
    except Exception as e:
        logger.debug("Curator recent-run notice failed: %s", e)
    # Don't stop a working dashboard when the Node refresh failed — see the
    # git-update path for rationale (#30271).
    _finish_dashboard_update_cleanup(node_failures)
    try:
        from hermes_cli.update_receipt import finalize_update_receipt

        finalize_update_receipt(
            "success" if update_complete and not node_failures else "partial"
        )
    except Exception as _receipt_exc:
        logger.debug("Update receipt finalize (zip path) failed: %s", _receipt_exc)
    return update_complete

def _stash_local_changes_if_needed(git_cmd: list[str], cwd: Path) -> Optional[str]:
    status = _git_run(git_cmd, ["status", "--porcelain"], cwd, check=True)
    if not status.stdout.strip():
        return None

    # If the index has unmerged entries (e.g. from an interrupted merge/rebase),
    # git stash will fail with "needs merge / could not write index".  Clear the
    # conflict state with `git reset` so the stash can proceed.  Working-tree
    # changes are preserved; only the index conflict markers are dropped.
    unmerged = _git_run(git_cmd, ["ls-files", "--unmerged"], cwd)
    if unmerged.stdout.strip():
        print("→ Clearing unmerged index entries from a previous conflict...")
        subprocess.run(git_cmd + ["reset"], cwd=cwd, capture_output=True)

    from datetime import datetime, timezone

    stash_name = datetime.now(timezone.utc).strftime(
        f"{_AUTOSTASH_NAME_PREFIX}%Y%m%d-%H%M%S"
    )
    print("→ Local changes detected — stashing before update...")
    prev_stash = _git_run(git_cmd, ["rev-parse", "--verify", "refs/stash"], cwd).stdout.strip()
    push = _git_run(git_cmd, ["stash", "push", "--include-untracked", "-m", stash_name], cwd)
    if push.stdout.strip():
        print(push.stdout.strip())
    stash_probe = _git_run(git_cmd, ["rev-parse", "--verify", "refs/stash"], cwd)
    stash_ref = stash_probe.stdout.strip()
    stash_created = (
        stash_probe.returncode == 0 and bool(stash_ref) and stash_ref != prev_stash
    )

    if push.returncode != 0:
        if stash_created:
            # stash push exits non-zero when it saved everything but couldn't
            # delete some swept untracked files (e.g. a root-owned dir:
            # "failed to remove ...: Permission denied"). The entry is
            # complete, so not a failure — leave the files and continue.
            if push.stderr.strip():
                print(push.stderr.strip())
            print(
                "  ⚠ Some untracked files could not be removed from the "
                "working tree (permission denied)."
            )
            print(
                "    They were still saved to the stash and were left in "
                "place — the update will continue."
            )
            # A partially-failed stash push also aborts its working-tree
            # cleanup for TRACKED modifications — they are saved in the stash
            # but still dirty the tree, which would break the checkout/pull
            # that follows. Safe to reset: everything is in the stash entry.
            subprocess.run(
                git_cmd + ["reset", "--hard", "HEAD"],
                cwd=cwd,
                capture_output=True,
            )
        else:
            # No stash entry was created: the changes were NOT saved.  This
            # is a real failure — bail out before the update touches HEAD.
            print("✗ Could not stash local changes — update aborted.")
            if push.stderr.strip():
                print(f"  {push.stderr.strip().splitlines()[0]}")
            print(
                "  Commit, stash, or clean up your local changes manually, "
                "then re-run `hermes update`."
            )
            raise subprocess.CalledProcessError(
                push.returncode, push.args, output=push.stdout, stderr=push.stderr
            )

    return stash_ref

def _resolve_stash_selector(
    git_cmd: list[str], cwd: Path, stash_ref: str
) -> Optional[str]:
    stash_list = _git_run(git_cmd, ["stash", "list", "--format=%gd %H"], cwd, check=True)
    for line in stash_list.stdout.splitlines():
        selector, _, commit = line.partition(" ")
        if commit.strip() == stash_ref:
            return selector.strip()
    return None

#: Producer/consumer contract for update autostash names: the stash subject is
#: this prefix + a UTC YYYYMMDD-HHMMSS stamp (see _stash_local_changes_if_needed
#: and _warn_orphaned_update_autostashes).
_AUTOSTASH_NAME_PREFIX = "hermes-update-autostash-"

#: Age past which a leftover ``hermes-update-autostash-*`` entry is called out
#: at update time. Entries younger than this are normal (a parked stash from
#: the desktop updater's --keep-stash run minutes ago); older ones are almost
#: always forgotten (#63717 problem 6: an orphan persisted 9+ days unnoticed).
_AUTOSTASH_WARN_AGE_DAYS = 7


def _warn_orphaned_update_autostashes(git_cmd: list[str], cwd: Path) -> int:
    """Surface leftover update autostashes older than the warn threshold.

    Autostashes legitimately outlive a run (``--keep-stash`` parks them; a
    failed restore preserves them), but nothing re-surfaces them — they sit
    invisibly for weeks (#63717 problem 6). Prints a notice with recovery/
    cleanup guidance. Deliberately NOT a GC: a stash entry can be the only
    copy of the user's uncommitted work, so Hermes never drops one.

    Best-effort — any git failure returns 0. Returns the stale-entry count.
    """
    from datetime import timedelta, timezone

    try:
        stash_list = _git_run(git_cmd, ["stash", "list", "--format=%gd %s"], cwd)
        if stash_list.returncode != 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=_AUTOSTASH_WARN_AGE_DAYS
        )
        marker = _AUTOSTASH_NAME_PREFIX
        stale: list[tuple[str, str]] = []
        for line in stash_list.stdout.splitlines():
            selector, _, subject = line.strip().partition(" ")
            pos = subject.find(marker)
            if pos < 0:
                continue
            stamp = subject[pos + len(marker):][:15]  # "YYYYMMDD-HHMMSS"
            try:
                stash_time = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                # Unparseable name — age unknown; leave it alone rather than
                # guess (same posture as _prune_orphan_rescue_refs).
                continue
            if stash_time < cutoff:
                stale.append((selector, stamp))
        if not stale:
            return 0
        print()
        print(
            f"⚠ {len(stale)} leftover update autostash entr"
            f"{'y is' if len(stale) == 1 else 'ies are'} more than "
            f"{_AUTOSTASH_WARN_AGE_DAYS} days old:"
        )
        for selector, stamp in stale:
            print(f"    {selector}  ({_AUTOSTASH_NAME_PREFIX}{stamp})")
        print("  These hold local changes stashed by earlier updates and never")
        print("  restored. Review with: git stash show -p <entry>")
        print("  Restore with: git stash apply <entry>   Discard with: git stash drop <entry>")
        return len(stale)
    except Exception as exc:
        logger.debug("Autostash age check failed: %s", exc)
        return 0


def _print_stash_cleanup_guidance(
    stash_ref: str, stash_selector: Optional[str] = None
) -> None:
    print(
        "  Check `git status` first so you don't accidentally reapply the same change twice."
    )
    print("  Find the saved entry with: git stash list --format='%gd %H %s'")
    if stash_selector:
        print(f"  Remove it with: git stash drop {stash_selector}")
    else:
        print(
            f"  Look for commit {stash_ref}, then drop its selector with: git stash drop stash@{{N}}"
        )

def _stash_apply_failed_only_on_existing_untracked(stderr: str) -> bool:
    """True when a ``git stash apply`` failure is ONLY about untracked files
    that already exist in the working tree.

    This is the tail end of the permission-denied autostash class: ``git stash
    push --include-untracked`` swept undeletable files (e.g. a root-owned
    ``packaging/`` directory) into the stash but could not remove them from
    disk.  On restore, git applies all tracked changes, then refuses to
    overwrite those still-present files (``already exists, no checkout`` /
    ``could not restore untracked files from stash``) and exits non-zero even
    though nothing was lost.  Any other error line (e.g. ``would be
    overwritten by merge`` / ``Aborting``) means the tracked apply itself
    failed and this returns False.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return False
    saw_untracked_error = False
    for ln in lines:
        if "already exists, no checkout" in ln:
            saw_untracked_error = True
        elif "could not restore untracked files from stash" in ln:
            saw_untracked_error = True
        elif ln.startswith(("warning:", "hint:")):
            continue
        else:
            return False
    return saw_untracked_error

def _park_stashed_changes(stash_ref: str) -> None:
    """Leave a pre-update autostash parked instead of re-applying it.

    Used by ``hermes update --keep-stash`` (the desktop updater's mode): the
    stash made the update possible on a dirty tree, but local source edits
    must never be silently re-applied onto the updated code. Nothing is
    lost — the entry stays in ``git stash`` with printed recovery guidance.
    """
    print()
    print("ℹ️  Local changes were stashed before updating and were NOT re-applied (--keep-stash).")
    print(f"  Stash ref: {stash_ref}")
    print(f"  Restore manually with: git stash apply {stash_ref}")


def _git_untracked_paths(git_cmd: list[str], cwd: Path) -> set[str] | None:
    """Return untracked paths, or ``None`` when Git cannot enumerate them."""
    try:
        result = subprocess.run(
            git_cmd + ["ls-files", "--others", "--exclude-standard", "-z"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is None or result.returncode != 0:
        print(
            "  ⚠ Could not enumerate untracked files while validating the "
            "restored stash."
        )
        return None
    return {path for path in result.stdout.split("\0") if path}


def _restored_python_paths(
    git_cmd: list[str], cwd: Path
) -> tuple[str, ...] | None:
    """Return restored ``.py`` paths changed from ``HEAD``.

    This deliberately validates Python source only; non-Python entry scripts
    remain outside the executable import-health check.
    """
    try:
        changed = subprocess.run(
            git_cmd + ["diff", "--name-only", "-z", "HEAD", "--", "*.py"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        )
    except (OSError, subprocess.SubprocessError):
        changed = None
    if changed is None or changed.returncode != 0:
        print("  ⚠ Could not enumerate tracked Python files restored from the stash.")
        return None
    paths = set(changed.stdout.split("\0"))
    untracked = _git_untracked_paths(git_cmd, cwd)
    if untracked is None:
        return None
    paths.update(path for path in untracked if path.endswith(".py"))
    paths.discard("")
    return tuple(sorted(paths))


def _reject_unsafe_stash_restore(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
    preexisting_untracked: set[str],
    failing_target: str,
    detail: str | None,
) -> None:
    """Restore the clean updated tree, preserve the stash, and abort the update."""
    print()
    print("✗ Restored local changes made the Hermes agent unexecutable.")
    print(f"  Health check failed: {failing_target}")
    if detail:
        for line in str(detail).splitlines()[:6]:
            print(f"    {line}")

    current_untracked = _git_untracked_paths(git_cmd, cwd)
    restored_untracked = (
        current_untracked - preexisting_untracked
        if current_untracked is not None
        else set()
    )
    try:
        reset = subprocess.run(
            git_cmd + ["reset", "--hard", "HEAD"], cwd=cwd, capture_output=True
        )
    except (OSError, subprocess.SubprocessError):
        reset = None

    clean = None
    if restored_untracked:
        try:
            clean = subprocess.run(
                git_cmd + ["clean", "-fd", "--", *sorted(restored_untracked)],
                cwd=cwd,
                capture_output=True,
            )
        except (OSError, subprocess.SubprocessError):
            clean = None
    cleanup_ok = (
        current_untracked is not None
        and reset is not None
        and reset.returncode == 0
        and (not restored_untracked or (clean is not None and clean.returncode == 0))
    )
    if cleanup_ok:
        try:
            verify = subprocess.run(
                git_cmd + ["diff", "--quiet", "HEAD", "--"],
                cwd=cwd,
                capture_output=True,
            )
            cleanup_ok = verify.returncode == 0
        except (OSError, subprocess.SubprocessError):
            cleanup_ok = False

    if cleanup_ok:
        print("  The clean updated tree has been restored; the gateway was not restarted.")
    else:
        print("  ⚠ The clean updated tree could not be fully restored automatically.")
        print("    Inspect `git status` and run `git reset --hard HEAD` before retrying.")
    print("  Platform connectivity alone does not mean the agent can execute turns.")
    print(f"  Your local changes remain preserved in stash: {stash_ref}")
    print(f"  Inspect them with: git stash show --stat {stash_ref}")
    print(f"  Restore manually after fixing them: git stash apply {stash_ref}")
    raise SystemExit(1)


def _restore_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
    prompt_user: bool = False,
    input_fn=None,
) -> bool:
    if prompt_user:
        remote_prompt = input_fn is not None
        prompt_suffix = "[y/N]" if remote_prompt else "[Y/n]"
        print()
        print("⚠ Local changes were stashed before updating.")
        print(
            "  Restoring them may reapply local customizations onto the updated codebase."
        )
        print("  Review the result afterward if Hermes behaves unexpectedly.")
        print(f"Restore local changes now? {prompt_suffix}")
        if input_fn is not None:
            response = input_fn(f"Restore local changes now? {prompt_suffix}", "n")
        else:
            try:
                response = input().strip().lower()
            except (EOFError, UnicodeDecodeError):
                # A closed stdin or terminal-encoding error must not crash the
                # update mid-restore; fall through to the skip-restore path.
                response = "n"
        accepted = response in {"y", "yes"} or (not remote_prompt and response == "")
        if not accepted:
            print("Skipped restoring local changes.")
            print("Your changes are still preserved in git stash.")
            print(f"Restore manually with: git stash apply {stash_ref}")
            return False

    preexisting_untracked = _git_untracked_paths(git_cmd, cwd)
    if preexisting_untracked is None:
        print("  The stash was not restored because its cleanup baseline is unknown.")
        print(f"  Restore manually with: git stash apply {stash_ref}")
        return False
    clean_import_failures = _critical_module_import_failures(
        cwd, report_runtime_errors=True
    )
    print("→ Restoring local changes...")
    restore = _git_run(git_cmd, ["stash", "apply", stash_ref], cwd)

    # Check for unmerged (conflicted) files — can happen even when returncode is 0
    unmerged = _git_run(git_cmd, ["diff", "--name-only", "--diff-filter=U"], cwd)
    has_conflicts = bool(unmerged.stdout.strip())

    if restore.returncode != 0 and not has_conflicts and (
        _stash_apply_failed_only_on_existing_untracked(restore.stderr)
    ):
        # Tracked changes applied cleanly; the only "failure" is untracked files
        # git couldn't delete at stash time and now refuses to overwrite. Their
        # content is untouched — treat as restored.
        print(
            "  ⚠ Some stashed untracked files already exist in the working "
            "tree and were kept as-is."
        )
    elif restore.returncode != 0 or has_conflicts:
        print("✗ Update pulled new code, but restoring local changes hit conflicts.")
        if restore.stdout.strip():
            print(restore.stdout.strip())
        if restore.stderr.strip():
            print(restore.stderr.strip())

        conflicted_files = unmerged.stdout.strip()
        if conflicted_files:
            print("\nConflicted files:")
            for f in conflicted_files.splitlines():
                print(f"  • {f}")

        print("\nYour stashed changes are preserved — nothing is lost.")
        print(f"  Stash ref: {stash_ref}")

        # Always reset: conflict markers in source make hermes unrunnable
        # (SyntaxError on import). The user's changes remain in the stash.
        subprocess.run(
            git_cmd + ["reset", "--hard", "HEAD"],
            cwd=cwd,
            capture_output=True,
        )
        print("Working tree reset to clean state.")
        print(f"Restore your changes later with: git stash apply {stash_ref}")
        # Don't exit: the code update succeeded; let cmd_update continue with
        # pip install, skill sync, and gateway restart.
        return False

    restored_python = _restored_python_paths(git_cmd, cwd)
    if restored_python is None:
        _reject_unsafe_stash_restore(
            git_cmd,
            cwd,
            stash_ref,
            preexisting_untracked,
            "restored Python source discovery",
            "could not determine which restored Python files require validation",
        )
    syntax_ok, failing_path, syntax_error = _validate_python_files_syntax(
        cwd, restored_python
    )
    if not syntax_ok:
        _reject_unsafe_stash_restore(
            git_cmd,
            cwd,
            stash_ref,
            preexisting_untracked,
            failing_path or "restored Python source",
            syntax_error,
        )

    restored_import_failures = _critical_module_import_failures(
        cwd, report_runtime_errors=True
    )
    changed_import_failure = next(
        (
            (module, error)
            for module, error in restored_import_failures.items()
            if clean_import_failures.get(module) != error
        ),
        None,
    )
    if changed_import_failure is not None:
        failing_module, import_error = changed_import_failure
        _reject_unsafe_stash_restore(
            git_cmd,
            cwd,
            stash_ref,
            preexisting_untracked,
            f"agent import {failing_module or 'unknown'}",
            import_error[1],
        )

    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop."
        )
        print(
            "  The stash was left in place. You can remove it manually after checking the result."
        )
        _print_stash_cleanup_guidance(stash_ref)
    else:
        drop = _git_run(git_cmd, ["stash", "drop", stash_selector], cwd)
        if drop.returncode != 0:
            print(
                "⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry."
            )
            if drop.stdout.strip():
                print(drop.stdout.strip())
            if drop.stderr.strip():
                print(drop.stderr.strip())
            print(
                "  The stash was left in place. You can remove it manually after checking the result."
            )
            _print_stash_cleanup_guidance(stash_ref, stash_selector)

    print("⚠ Local changes were restored on top of the updated codebase.")
    print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
    return True

def _discard_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
) -> bool:
    """Drop a pre-update stash without applying it.

    Only for NON-interactive updates with
    ``updates.non_interactive_local_changes: discard``. Unlike ``git reset
    --hard`` + ``git clean -fd``, this touches only what was stashed — ignored
    paths (node_modules, venv, build outputs) are never affected.

    Returns True if dropped, False on git failure (stash left in place).
    """
    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Configured to discard local changes on non-interactive update, "
            "but Hermes couldn't find the stash entry to drop."
        )
        _print_stash_cleanup_guidance(stash_ref)
        return False

    drop = _git_run(git_cmd, ["stash", "drop", stash_selector], cwd)
    if drop.returncode != 0:
        print(
            "⚠ Configured to discard local changes, but Hermes couldn't drop "
            "the saved stash entry."
        )
        if drop.stderr.strip():
            print(f"  {drop.stderr.strip().splitlines()[0]}")
        _print_stash_cleanup_guidance(stash_ref, stash_selector)
        return False

    print("→ Discarded local source changes (updates.non_interactive_local_changes=discard).")
    return True

OFFICIAL_REPO_URLS = {
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
}

OFFICIAL_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"

SKIP_UPSTREAM_PROMPT_FILE = ".skip_upstream_prompt"

def _get_origin_url(git_cmd: list[str], cwd: Path) -> Optional[str]:
    """Get the URL of the origin remote, or None if not set."""
    try:
        result = _git_run(git_cmd, ["remote", "get-url", "origin"], cwd)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False
    # Normalize URL for comparison (strip trailing .git if present)
    normalized = origin_url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    for official in OFFICIAL_REPO_URLS:
        official_normalized = official.rstrip("/")
        if official_normalized.endswith(".git"):
            official_normalized = official_normalized[:-4]
        if normalized == official_normalized:
            return False
    return True

def _has_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Check if an 'upstream' remote already exists."""
    try:
        result = _git_run(git_cmd, ["remote", "get-url", "upstream"], cwd)
        return result.returncode == 0
    except Exception:
        return False

def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    try:
        result = _git_run(git_cmd, ["remote", "add", "upstream", OFFICIAL_REPO_URL], cwd)
        return result.returncode == 0
    except Exception:
        return False

def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    try:
        result = _git_run(git_cmd, ["rev-list", "--count", f"{base}..{head}"], cwd)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1

def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()

def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    try:
        from hermes_constants import get_hermes_home

        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()
    except Exception:
        pass

def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Attempt to push updated main to origin (sync fork).

    Returns True if push succeeded, False otherwise.
    """
    try:
        result = _git_run(git_cmd, ["push", "origin", "main", "--force-with-lease"], cwd, network=True)
        return result.returncode == 0
    except Exception:
        return False

def _sync_with_upstream_if_needed(
    git_cmd: list[str],
    cwd: Path,
    *,
    assume_yes: bool = False,
    input_fn=None,
) -> bool:
    """Check if fork is behind upstream and fast-forward if safe.

    Offers to add the ``upstream`` remote, compares origin/main with
    upstream/main, pulls when strictly behind, then tries to push origin.

    Returns True only when origin/main was actually verified against
    upstream/main; False when the check never happened (prompt declined,
    remote add/fetch/compare failed) so the caller never reports "up to date"
    on an origin-only comparison (#97052).
    """
    has_upstream = _has_upstream_remote(git_cmd, cwd)

    if not has_upstream:
        if _should_skip_upstream_prompt():
            return False

        print()
        print("ℹ Your fork is not tracking the official Hermes repository.")
        print("  This means you may miss updates from NousResearch/hermes-agent.")
        print()

        if assume_yes or (
            input_fn is None and not (sys.stdin.isatty() and sys.stdout.isatty())
        ):
            # --yes means "don't block", not "mutate my git remotes". Skip
            # without persisting the decline so interactive runs still get asked.
            print("  Skipping upstream setup (non-interactive run).")
            print(
                "  Add it later with: git remote add upstream https://github.com/NousResearch/hermes-agent.git"
            )
            return False

        if input_fn is not None:
            response = (
                input_fn("Add official repo as 'upstream' remote? [y/N]", "n")
                .strip()
                .lower()
            )
        else:
            try:
                response = (
                    input("Add official repo as 'upstream' remote? [Y/n]: ")
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt, UnicodeDecodeError):
                print()
                response = "n"

        if response in {"", "y", "yes"}:
            print("→ Adding upstream remote...")
            if _add_upstream_remote(git_cmd, cwd):
                print(
                    "  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git"
                )
                has_upstream = True
            else:
                print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
                return False
        else:
            print(
                "  Skipped. Run 'git remote add upstream https://github.com/NousResearch/hermes-agent.git' to add later."
            )
            _mark_skip_upstream_prompt()
            return False

    # Fetch only upstream/main: a bare fetch drags in thousands of
    # auto-generated branches.
    print()
    print("→ Fetching upstream...")
    try:
        subprocess.run(
            git_cmd + ["fetch", "upstream", "main", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=True,
            **_no_prompt_git_kwargs(),
        )
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return False

    # Compare origin/main with upstream/main
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(
        git_cmd, cwd, "origin/main", "upstream/main"
    )

    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return False

    # If origin/main has commits not on upstream, don't trample
    if origin_ahead > 0:
        print()
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        print("  If you want to merge upstream changes, run:")
        print("    git pull upstream main")
        return True

    if upstream_ahead == 0:
        print("  ✓ Fork is up to date with upstream")
        return True

    # origin/main is strictly behind upstream/main (can fast-forward)
    print()
    print(f"→ Fork is {upstream_ahead} commit(s) behind upstream")
    print("→ Pulling from upstream...")

    try:
        subprocess.run(
            git_cmd + ["pull", "--ff-only", "upstream", "main"],
            cwd=cwd,
            check=True,
            **_no_prompt_git_kwargs(),
        )
    except subprocess.CalledProcessError:
        print(
            "  ✗ Failed to pull from upstream. You may need to resolve conflicts manually."
        )
        return False

    print("  ✓ Updated from upstream")

    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print(
            "  ℹ Got updates from upstream but couldn't push to fork (no write access?)"
        )
        print("    Your local repo is updated, but your fork on GitHub may be behind.")
    return True

def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles.

    The git repo is shared, so one profile's update makes every profile
    current; a per-profile cache would show a stale "commits behind" banner.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)
    from hermes_constants import get_default_hermes_root

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass

def _write_marker_file(path: Path, *, label: str) -> None:
    """Drop an update-recovery breadcrumb. Never raises."""
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping %s marker under pytest (live checkout)", label)
        return
    try:
        path.write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write %s marker: %s", label, exc)

def _write_update_incomplete_marker() -> None:
    """Drop the interrupted core-install breadcrumb. Never raises."""
    _write_marker_file(_m()._update_marker_path(), label="update-incomplete")

def _write_lazy_refresh_incomplete_marker() -> None:
    """Drop the interrupted lazy-refresh breadcrumb. Never raises."""
    _write_marker_file(_m()._lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


# Lives under HERMES_HOME (not next to the venv). Unlike the venv-repair
# markers, this records the fleet-restart obligation after a pull advanced
# HEAD (#95294); cleared only when the restart completes or nothing was running.
_FLEET_RESTART_PENDING_NAME = "fleet_restart_pending"


def _fleet_restart_pending_marker_path() -> Path:
    """HERMES_HOME breadcrumb for a pull that has not yet restarted the fleet."""
    return get_hermes_home() / _FLEET_RESTART_PENDING_NAME


def _write_fleet_restart_pending_marker(*, expected_sha: str = "") -> None:
    """Drop the pull→restart obligation breadcrumb. Never raises."""
    path = _fleet_restart_pending_marker_path()
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping fleet-restart-pending marker under pytest (live checkout)")
        return
    try:
        lines = [f"started={_time.time()}", f"pid={os.getpid()}"]
        if expected_sha:
            lines.append(f"expected_sha={expected_sha}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write fleet-restart-pending marker: %s", exc)


def _clear_fleet_restart_pending_marker() -> None:
    """Remove the pull→restart obligation breadcrumb. Never raises."""
    _m()._clear_marker_file(
        _fleet_restart_pending_marker_path(), label="fleet-restart-pending"
    )


def _current_checkout_sha() -> str | None:
    """Current on-disk checkout HEAD, or None if it cannot be resolved."""
    try:
        from hermes_cli.build_info import get_code_identity

        sha = (get_code_identity(refresh=True) or {}).get("sha")
        return str(sha) if sha else None
    except Exception:
        return _capture_head_sha(["git"], _m().PROJECT_ROOT)


def _receipt_looks_unfinished(receipt: dict) -> bool:
    """True when *receipt* is from an update that did not finish cleanly."""
    if receipt.get("stop_reason"):
        return True
    exit_code = receipt.get("exit_code")
    if exit_code not in (0, None):
        return True
    outcome = receipt.get("outcome")
    if outcome in ("failed", "partial", "running"):
        return True
    gateway_restart = receipt.get("gateway_restart")
    if isinstance(gateway_restart, dict) and gateway_restart.get("incomplete"):
        return True
    return False


def _receipt_reports_stale_runtime(expected_sha: str | None = None) -> bool:
    """True when ``update_receipts/latest.json`` records a runtime SHA skew.

    Prefer the post-restart ``fleet`` matrix. ``plan.runtimes[].code_sha`` is
    captured *before* the pull, so a finished update's plan always shows stale
    SHAs and must not retrigger a restart; consult it only for an unfinished
    receipt (#95294).
    """
    try:
        from hermes_cli.update_receipt import read_latest_receipt

        receipt = read_latest_receipt()
    except Exception:
        receipt = None
    if not isinstance(receipt, dict):
        return False
    if not expected_sha:
        expected_sha = _current_checkout_sha()
    if not expected_sha:
        return False

    def _sha_mismatch(code_sha) -> bool:
        return bool(code_sha) and str(code_sha) != str(expected_sha)

    fleet = receipt.get("fleet")
    if isinstance(fleet, list) and fleet:
        for entry in fleet:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "stale":
                return True
            if _sha_mismatch(entry.get("code_sha")):
                return True
        return False

    if not _receipt_looks_unfinished(receipt):
        return False
    plan = receipt.get("plan")
    if not isinstance(plan, dict):
        return False
    for runtime in plan.get("runtimes") or []:
        if isinstance(runtime, dict) and _sha_mismatch(runtime.get("code_sha")):
            return True
    return False


def _pending_fleet_restart_needed() -> bool:
    """True when a prior pull still owes the fleet a restart (#95294)."""
    try:
        if _fleet_restart_pending_marker_path().is_file():
            return True
    except OSError:
        pass
    return _receipt_reports_stale_runtime()


def _warn_pending_fleet_restart(*, startup: bool = False) -> None:
    """Print the specific interrupted-update fleet-restart warning."""
    stream = sys.stderr if startup else sys.stdout
    print(
        "⚠ A previous `hermes update` pulled new code but did not "
        "restart running gateways.",
        file=stream,
    )
    print(
        "  Gateways may still be serving pre-update modules (mixed sys.modules).",
        file=stream,
    )
    if startup:
        print(
            "  Run `hermes update` or `hermes gateway restart`.",
            file=stream,
        )


def _warn_pending_fleet_restart_on_startup() -> None:
    """Cheap CLI-startup hint. Never restarts; never raises."""
    try:
        if not _pending_fleet_restart_needed():
            return
        _warn_pending_fleet_restart(startup=True)
    except Exception:
        pass


def _restart_systemd_gateway_units_best_effort(failed: list) -> None:
    """Best-effort ``systemctl restart`` of every hermes-gateway/serve unit."""
    for scope, scope_cmd in (
        ("user", ["systemctl", "--user"]),
        ("system", ["systemctl"]),
    ):
        try:
            result = _systemctl(
                scope_cmd + ["list-units", "hermes-gateway*", "hermes-serve*",
                             "--plain", "--no-legend", "--no-pager"],
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue

        def process_unit(svc_name: str, _scope=scope, _cmd=scope_cmd) -> None:
            restart_cmd = list(_cmd) + ["--no-ask-password", "restart", svc_name]
            if (
                _scope == "system"
                and hasattr(os, "geteuid")
                and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
            ):
                restart_cmd = ["sudo", "-n"] + restart_cmd
            _systemctl(restart_cmd, timeout=30)

        def on_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
            failed.append(svc_name)

        _for_each_systemd_gateway_unit(
            result.stdout,
            process_unit=process_unit,
            on_unit_timeout=on_timeout,
        )


def _run_pending_fleet_restart() -> bool:
    """Catch-up restart for gateways left on pre-update code (#95294).

    Returns True when restart completed or no services were running.
    Returns False if restart was incomplete. Never raises.
    """
    print("→ Restarting gateways left on pre-update code...")
    try:
        _m()._purge_stale_hermes_modules()
    except Exception:
        pass
    try:
        from hermes_cli.gateway import (
            find_gateway_pids,
            is_macos,
            is_windows,
            kill_gateway_processes,
            supports_systemd_services,
            _wait_for_gateway_exit,
        )
    except Exception as exc:
        _warn_gateway_restart_phase_aborted(exc, None)
        return False

    try:
        pids = list(find_gateway_pids(all_profiles=True))
    except Exception as exc:
        logger.debug("Pending fleet restart: gateway probe failed: %s", exc)
        pids = None

    if pids == []:
        print("  ✓ No running gateways — nothing to restart.")
        return True

    failed: list = []
    try:
        if supports_systemd_services():
            _restart_systemd_gateway_units_best_effort(failed)
        if is_macos():
            restarted: list = []
            try:
                _restart_macos_launchd_gateways(restarted, failed, 45.0)
            except Exception as exc:
                logger.debug("Pending fleet restart: launchd failed: %s", exc)
                failed.append("launchd")
        if is_windows():
            try:
                from hermes_cli import gateway_windows

                if gateway_windows.is_installed():
                    gateway_windows.restart()
            except Exception as exc:
                logger.debug("Pending fleet restart: Windows failed: %s", exc)
                failed.append("windows-gateway")
        leftover: list = []
        try:
            leftover = list(find_gateway_pids(all_profiles=True))
        except Exception:
            leftover = list(pids or [])
        if leftover:
            try:
                kill_gateway_processes(all_profiles=True)
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
            except Exception as exc:
                logger.debug("Pending fleet restart: PID stop failed: %s", exc)
        if failed:
            _warn_incomplete_gateway_fleet_restart(failed)
            return False
        print("  ✓ Pending fleet restart completed.")
        return True
    except Exception as exc:
        surviving = None
        try:
            surviving = list(find_gateway_pids(all_profiles=True))
        except Exception:
            surviving = pids
        _warn_gateway_restart_phase_aborted(exc, surviving)
        return False


def _apply_pending_fleet_restart_catchup() -> None:
    """On an already-up-to-date ``hermes update``, finish a skipped restart.

    No-op when nothing is pending. Exits 1 when the catch-up restart is
    incomplete so automation does not treat the fleet as healthy.
    """
    if not _pending_fleet_restart_needed():
        return
    print()
    _warn_pending_fleet_restart()
    print("→ Running the pending fleet restart...")
    if _run_pending_fleet_restart():
        _clear_fleet_restart_pending_marker()
        return
    print("  ⚠ Fleet restart incomplete. Recover with: hermes gateway restart")
    sys.exit(1)


def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)


def _classify_concurrent_instance(pid: int) -> str:
    """Return ``"gateway"`` when ``pid``'s command line is a gateway runtime.

    Delegates to ``_is_pausable_gateway`` — the same canonical ``gateway run``
    matcher used by the Desktop preflight exemption and the venv-holder guard
    — so a PID classified ``"gateway"`` here is exactly the set the downstream
    pause/kill+restart machinery will stop. That symmetry lets the pre-update
    concurrent gate skip the abort for gateway-only matches instead of making
    the user kill a gateway that is about to be paused anyway.

    Returns ``"non-gateway"`` when the cmdline doesn't match and ``"unknown"``
    when psutil can't read it; the gate treats ``"unknown"`` as non-gateway
    (better to block than proceed against an unidentified process).
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return "unknown"

    try:
        proc = psutil.Process(int(pid))
        cmdline_list = proc.cmdline()
    except Exception:
        return "unknown"

    from hermes_cli._scan_venv_blockers import _is_pausable_gateway  # noqa: PLC0415

    cmdline = " ".join(cmdline_list or [])
    if _is_pausable_gateway(cmdline):
        return "gateway"
    return "non-gateway"


def _filter_non_gateway_concurrent_instances(
    matches: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return only the concurrent-instance matches that are NOT the gateway.

    If every concurrent instance is a gateway, the pause machinery and the
    post-update kill+restart handle it and the update proceeds. Anything else
    (TUI shell, Desktop backend child, another ``hermes`` REPL) has no pause
    machinery downstream, so the gate still aborts.
    """
    non_gateway: list[tuple[int, str]] = []
    for pid, name in matches:
        if _classify_concurrent_instance(pid) != "gateway":
            non_gateway.append((pid, name))
    return non_gateway

def _upgrade_pip_before_lazy_refresh(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Upgrade pip before lazy-backend refreshes.

    Older pip (e.g. 24.0 on Python 3.11) can fail setuptools-backed source
    builds during lazy installs and leave a partially-written venv (#57828).
    Never raises.
    """
    try:
        _m()._run_package_only_install(
            install_cmd_prefix + ["install", "--upgrade", "pip"],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.debug("pip upgrade before lazy refresh failed: %s", exc)


def _capture_active_lazy_features() -> list[str]:
    """Snapshot active lazy backends before a managed runtime is replaced."""
    try:
        from tools import lazy_deps

        return lazy_deps.active_features()
    except Exception as exc:
        logger.debug("Could not snapshot active lazy features: %s", exc)
        return []


def _capture_active_tool_dependencies() -> list[str]:
    """Snapshot Python dependencies installed explicitly through ``hermes tools``."""
    try:
        from hermes_cli import tools_config

        return tools_config.active_restorable_python_tool_dependencies()
    except Exception as exc:
        logger.debug("Could not snapshot active Hermes Tools dependencies: %s", exc)
        return []


def _restore_active_tool_dependencies(
    dependencies: list[str],
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Restore allowlisted ``hermes tools`` dependencies into a rebuilt venv.

    The dependency names came from a pre-rebuild import probe and are resolved
    through a static package allowlist. Never raises: a failed optional tool
    must not block the core update, but the user must be told what stayed
    unavailable.
    """
    if not dependencies:
        return

    try:
        from hermes_cli import tools_config
    except Exception as exc:
        logger.debug("Hermes Tools dependency restore skipped (import failed): %s", exc)
        return

    target_python = _m()._resolve_install_target_python(install_cmd_prefix, env)
    missing: list[tuple[str, tuple[str, ...]]] = []
    for name in dependencies:
        spec = tools_config.restorable_python_tool_dependency(name)
        if spec is None:
            continue
        module_name, install_args = spec
        if target_python is not None:
            try:
                probe = subprocess.run(
                    [
                        str(target_python),
                        "-c",
                        "import importlib.util,sys; "
                        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                        module_name,
                    ],
                    capture_output=True,
                    env=env,
                    check=False,
                )
                if probe.returncode == 0:
                    continue
            except (subprocess.SubprocessError, OSError):
                # An indeterminate probe is safer to repair than to treat as
                # proof that a pre-rebuild dependency survived.
                pass
        missing.append((name, install_args))

    if not missing:
        return

    print()
    print(f"→ Restoring {len(missing)} Hermes Tools dependency set(s)...")
    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, install_args in missing:
        try:
            _m()._run_package_only_install(
                install_cmd_prefix + ["install", *install_args, "--quiet"],
                env=env,
            )
            restored.append(name)
        except Exception as exc:
            # Best-effort optional tooling: surface failures without aborting
            # the core update.
            failed.append((name, str(exc)))

    if restored:
        print(f"  ✓ {len(restored)} restored: {', '.join(restored)}")
    for name, reason in failed:
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {name} failed to restore: {reason}")


def _refresh_active_lazy_features(
    install_cmd_prefix: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    features: list[str] | None = None,
) -> bool:
    """Refresh lazy-installed backends after a code update.

    ``uv pip install -e .[all]`` never touches ``tools/lazy_deps.py`` backends,
    so a bumped :data:`LAZY_DEPS` pin (CVE, transitive fix) would otherwise
    leave already-activated backends stale forever. Reinstalls only the
    features the user previously activated; cold backends stay untouched.

    Returns True when the venv is safe to use (refresh succeeded, nothing
    active, or post-failure import repair succeeded); False when a failed
    lazy install left broken core imports that repair could not fix (#57828).

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return True

    if features is None:
        try:
            active = lazy_deps.active_features()
        except Exception as exc:
            logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
            return True
    else:
        active = features

    if not active:
        return True

    print()
    print(f"→ Refreshing {len(active)} active lazy backend(s)...")

    unexpected_failure = False
    try:
        if features is None:
            results = lazy_deps.refresh_active_features(prompt=False)
        else:
            results = lazy_deps.restore_features(active)
    except Exception as exc:
        # refresh_active_features is documented as never-raise, but defend
        # the update flow against future regressions.
        print(f"  ⚠ Lazy refresh failed unexpectedly: {exc}")
        results = {}
        unexpected_failure = True

    refreshed = [f for f, s in results.items() if s in {"refreshed", "restored"}]
    current = [f for f, s in results.items() if s == "current"]
    failed = [(f, s) for f, s in results.items() if s.startswith("failed:")]
    skipped = [(f, s) for f, s in results.items() if s.startswith("skipped:")]

    if refreshed:
        print(f"  ↑ {len(refreshed)} refreshed: {', '.join(refreshed)}")
    if current:
        print(f"  ✓ {len(current)} already current")
    if skipped:
        # Most common reason: security.allow_lazy_installs=false. Show one
        # line so the user knows why; not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")

    if not failed and not unexpected_failure:
        return True

    for feature, status in failed:
        reason = status.split(": ", 1)[-1]
        # Clip noisy pip stderr to keep update output legible.
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {feature} failed to refresh: {reason}")

    if install_cmd_prefix is None:
        print("  ⚠ Lazy refresh failed; rerun `hermes update` once resolved.")
        return False

    # Immediate import-based recovery — metadata-only verifiers miss the case
    # where DISTRIBUTION-INFO remains but import files were wiped (#57828).
    # Unavailable probes are indeterminate, not healthy — keep the lazy marker.
    status = _m()._repair_venv_via_import_probes(install_cmd_prefix, env=env)
    if status == "repaired":
        print(
            "  Lazy backend(s) keep their previous version until refresh succeeds."
        )
        return True
    if status == "healthy":
        print(
            "  Lazy backend(s) keep their previous version; probed packages look intact."
        )
        print("  Rerun `hermes update` once the upstream issue is resolved.")
        return True
    if status == "indeterminate":
        print(
            "  ⚠ Leaving `.lazy-refresh-incomplete` until import probes can confirm health."
        )
    return False

def _refresh_active_memory_provider_dependencies() -> None:
    """Refresh pip dependencies for the configured external memory provider.

    Provider bridge packages are declared in each provider's ``plugin.yaml``
    (plus mode extras like Hindsight's ``hindsight-all``), not in Hermes'
    extras or ``LAZY_DEPS``, so the core reinstall can strip or downgrade
    them (#53272, #70636). Re-run the ACTIVE provider's install after the
    core install and lazy refresh so its writes to shared packages land last.

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (config load failed): %s", exc)
        return

    provider = ""
    if isinstance(cfg, dict):
        memory_cfg = cfg.get("memory")
        if isinstance(memory_cfg, dict):
            if memory_cfg.get("enabled") is False:
                return
            provider = str(memory_cfg.get("provider") or "").strip()

    # "default" / empty is the built-in file-backed store — no pip deps.
    if not provider or provider in {"default", "builtin", "none"}:
        return

    try:
        from hermes_cli.memory_setup import _install_dependencies
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (import failed): %s", exc)
        return

    print()
    print(f"→ Refreshing active memory provider dependencies ({provider})...")

    try:
        _install_dependencies(provider, force=True)
    except Exception as exc:
        print(f"  ⚠ {provider} dependencies failed to refresh: {exc}")

def _is_android_python() -> bool:
    return _m().sys.platform == "android"

def _install_psutil_android_compat(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup gates Linux sources behind ``sys.platform.startswith('linux')``;
    Termux reports ``'android'``, so setup aborts although the Linux source path
    compiles fine. Only the extracted build tree for this attempt is patched.

    Stopgap: remove (together with the standalone installer's use of the same
    helper) once https://github.com/giampaolo/psutil/pull/2762 ships.
    """
    import tempfile
    import urllib.request
    from hermes_cli.psutil_android import PSUTIL_URL, prepare_patched_psutil_sdist

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        src_root = prepare_patched_psutil_sdist(archive, tmp_path)

        _m()._run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)],
            env=env,
        )

def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux for faster update installs.

    The official uv installer may not work on Termux (glibc vs bionic). Prefer
    a uv already on PATH (``pkg install uv``); otherwise fall back to a
    wheel-only ``pip install uv`` so the Rust crate is never source-built.
    """
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _m()._is_termux_env():
        return None
    # A Termux-packaged uv lands on PATH but not in the managed bin dir, so
    # resolve_uv() misses it. Use it before pip, which has no Android wheel and
    # would otherwise build uv from source on a low-memory device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    try:
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=_m().PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
    except Exception:
        pass
    return resolve_uv() or shutil.which("uv")

def _npm_manifest_paths() -> tuple[Path, ...]:
    """Manifests whose changes must defeat the update-skip.

    The lockfile alone is not a sufficient key: a dev can edit a package.json
    (root or workspace) without running npm, and `hermes update` is exactly
    the step expected to sync node_modules (`npm install` fallback in
    _run_npm_install_deterministic).

    Workspaces come from the root package.json's `workspaces` globs so a new
    workspace can never escape the key. Every workspace manifest counts —
    desktop included, though the install names only ui-tui and web — because
    the single lockfile spans the whole workspace graph. Falls back to root
    manifests only if package.json is unreadable (never skips more than main
    would have installed).
    """
    root_pkg = _m().PROJECT_ROOT / "package.json"
    paths = [_m().PROJECT_ROOT / "package-lock.json", root_pkg]
    try:
        workspaces = json.loads(root_pkg.read_text(encoding="utf-8")).get(
            "workspaces", []
        )
        if isinstance(workspaces, dict):  # legacy {"packages": [...]} form
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            for match in sorted(_m().PROJECT_ROOT.glob(str(pattern))):
                manifest = match / "package.json"
                if manifest.is_file():
                    paths.append(manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return tuple(paths)

def _npm_manifests_digest() -> str | None:
    """Combined sha256 over the lockfile + all workspace package.json files.

    Returns None when the lockfile is missing (never skip then).
    """
    if not (_m().PROJECT_ROOT / "package-lock.json").exists():
        return None
    h = hashlib.sha256()
    for p in _npm_manifest_paths():
        h.update(str(p.relative_to(_m().PROJECT_ROOT)).encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()

def _npm_lockfile_changed(hermes_root: Path) -> bool:
    current = _npm_manifests_digest()
    if current is None:
        return True
    # Also check that node_modules exists; a matching hash with missing
    # node_modules means the cache was recorded by another checkout.
    if not (_m().PROJECT_ROOT / "node_modules").is_dir():
        return True
    # A matching hash must NOT skip the reinstall when the web build toolchain
    # never landed, or every later update rebuilds against a half-installed tree.
    web_dir = _m().PROJECT_ROOT / "web"
    if (web_dir / "package.json").is_file() and not _web_build_toolchain_ready(
        *_web_toolchain_roots(web_dir)
    ):
        return True
    try:
        # Key the cache by PROJECT_ROOT so parallel worktrees don't collide.
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        if not cache_file.exists():
            return True
        return cache_file.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True

def _record_npm_lockfile_hash(hermes_root: Path) -> None:
    digest = _npm_manifests_digest()
    if digest is None:
        return
    try:
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        cache_file.write_text(digest, encoding="utf-8")
    except OSError:
        logger.debug("Could not write npm lockfile hash cache")

def _repair_node_deps_on_current_checkout(
    print_completion,
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    completion_message: str = "✓ Already up to date!",
    had_desktop_app_before_update: bool = False,
) -> bool:
    """Repair Node deps on the ``commit_count == 0`` path (#77211).

    A current checkout does not imply healthy Node deps: a failed npm install
    (EBADENGINE, network timeout, interrupt) says "re-run hermes update", but
    the early return never reached the Node refresh. ``_update_node_dependencies``
    self-gates on the lockfile hash, recorded only after a SUCCESSFUL install
    (and re-tripped when node_modules or the web toolchain is missing), so this
    is a cheap no-op on healthy installs and a real repair after a failed one.
    """
    node_failures = _update_node_dependencies()
    if node_failures:
        print(f"  ⚠ Node.js refresh failed for: {', '.join(node_failures)}")
        print("    Fix npm and re-run `hermes update`.")
        print_completion(
            "⚠ Checkout is current, but Node.js dependencies could not be repaired."
        )
        return False
    # Pair the refresh with the web build like every other
    # _update_node_dependencies call site; it staleness-checks internally,
    # so this is a no-op when nothing changed.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    _check_and_apply_config_migration(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
    )
    # A current checkout can still owe a Desktop rebuild (#97343) — e.g. the
    # Windows hand-off child never reaches the commits-pulled rebuild — leaving
    # a stale app behind a successful-looking update. Self-gates on the build stamp.
    if not _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    ):
        # _rebuild_desktop_after_update already printed the retry hint; withhold
        # success rather than claiming the update finished (#88251).
        print_completion(
            "⚠ Update partially complete — the desktop app was not rebuilt "
            "and is still on the previous build."
        )
        return False
    return bool(print_completion(completion_message))


def _update_node_dependencies() -> list[str]:
    """Refresh Node deps for the ui-tui and web workspaces.

    Returns the list of labels whose npm install failed (empty on success),
    so the caller can treat a Node refresh failure as a partial update rather
    than silently reporting ``Update complete!`` (#30271).
    """
    if not (_m().PROJECT_ROOT / "package.json").exists():
        return []

    npm = _m()._resolve_node_runtime_npm()
    if not npm:
        # If the only npm reachable inside this WSL shell is the Windows one,
        # flag it loudly: silently skipping leaves ui-tui deps stale while the
        # rest of the update proceeds, and running it would corrupt the tree.
        from hermes_constants import is_wsl

        path_npm = shutil.which("npm")
        if is_wsl() and path_npm and _m()._is_windows_npm_path(path_npm):
            print("→ Updating Node.js dependencies...")
            print("  ⚠ Skipped: only a Windows npm is reachable from this WSL shell.")
            print("    Install Node.js inside the WSL distro (nvm, or your distro's")
            print("    package manager), then re-run `hermes update`.")
            failed = []
            if any(
                (_m().PROJECT_ROOT / workspace / "package.json").exists()
                for workspace in ("ui-tui", "web")
            ):
                failed.append("ui-tui, web workspaces")
            return failed
        return []

    from hermes_constants import get_default_hermes_root

    # node_modules is shared by every profile on this checkout, so keep one
    # per-checkout cache under the shared root instead of one per profile.
    shared_hermes_root = get_default_hermes_root()

    # Best-effort npx cache warm for agent-browser (#43564), before the
    # lockfile-unchanged early return (the common case). Can block ~11s on a
    # cold cache — print first so it doesn't look like a hang.
    print("→ Warming npx cache for agent-browser...")
    try:
        from tools.browser_tool import warm_agent_browser_npx_cache
        warm_agent_browser_npx_cache()
    except Exception:
        pass

    if not _m()._npm_lockfile_changed(shared_hermes_root):
        logger.info("npm lockfile unchanged, skipping npm install")
        return []

    # Root package.json has no dependencies of its own (#43564: agent-browser
    # resolves via `npx` at runtime, @streamdown/math moved to apps/desktop),
    # so a workspace-scoped install prunes nothing root-only. apps/desktop is
    # deliberately never named: its Electron devDependency has a ~200MB
    # postinstall download, so desktop deps install on demand
    # (see _desktop_build_needed).
    print("→ Updating Node.js dependencies...")

    def _partial_update_failure(*labels: str) -> list[str]:
        print()
        print("  ⚠ Node.js dependency refresh did not complete cleanly; the")
        print("    installation may be in a mixed state (updated code, stale Node")
        print("    deps). Fix npm and re-run `hermes update`.")
        return list(labels)

    install_args = [
        "--no-fund", "--no-audit", "--prefer-offline", "--progress=false",
        "--workspace", "ui-tui", "--workspace", "web",
        # Root's own devDependencies (the shared ESLint flat config every
        # workspace imports) would otherwise be pruned by this scoped install
        # and have nowhere else to live. apps/desktop is still excluded since
        # it is never named above.
        "--include-workspace-root",
    ]

    from hermes_constants import with_hermes_node_path

    nixos_env = with_hermes_node_path(_m()._nixos_build_env())

    # capture_output=False is deliberate (#18840): optional postinstall scripts
    # print download progress, and capturing it makes a long download look
    # hung. The npm-deprecation noise comes from the desktop build (captured
    # to update.log), not this step.
    result = _m()._run_npm_install_deterministic(
        npm,
        _m().PROJECT_ROOT,
        extra_args=tuple(install_args),
        capture_output=False,
        env=nixos_env,
    )
    if result.returncode == 0:
        _record_npm_lockfile_hash(shared_hermes_root)
        print("  ✓ ui-tui, web workspaces installed (desktop skipped)")
        failures: list[str] = []
    else:
        print("  ⚠ npm install failed")
        stderr = (result.stderr or "").strip() if result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        failures = _partial_update_failure("ui-tui, web workspaces")

    return failures

def _log_only_write(text: str) -> None:
    """Write ``text`` to ``~/.hermes/logs/update.log`` only, never the terminal.

    During ``hermes update`` ``sys.stdout`` is an ``_UpdateOutputStream``
    mirroring to terminal and log; this reaches past it to the log handle so
    loud, low-signal subprocess output (npm, Electron/vite, cua-driver "Next
    steps") stays debuggable without flooding the terminal.
    """
    if not text:
        return
    stream = _m().sys.stdout
    log_file = getattr(stream, "_log", None)
    if log_file is None:
        return
    try:
        log_file.write(text if text.endswith("\n") else text + "\n")
        log_file.flush()
    except Exception:
        pass

def _run_logged_subprocess(cmd, *, cwd=None, env=None):
    """Run ``cmd`` capturing combined output into update.log (not the terminal).

    Returns the ``CompletedProcess`` (with ``stdout`` populated) so the caller
    can decide whether to surface the captured output on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _log_only_write(result.stdout or "")
    return result

def _classify_fetch_failure(stderr: str) -> str:
    """Map git-fetch stderr to a one-line, user-facing diagnosis.

    Order matters: curl reports HTTP failures as ``unable to access '<url>':
    The requested URL returned error: 429``, so the rate-limit/outage checks
    must run BEFORE the generic "unable to access" network check. The caller
    always prints the first raw stderr line too — this adds guidance, it
    never replaces the wire error.
    """

    def _has_http_code(*codes: str) -> bool:
        return any(
            f"HTTP {code}" in stderr or f"returned error: {code}" in stderr
            for code in codes
        )

    if _has_http_code("429") or "rate limit" in stderr.lower():
        return (
            "✗ GitHub is rate limiting requests or having an outage (HTTP 429)"
            " — try again in 5 minutes."
        )
    if _has_http_code("500", "502", "503", "504"):
        return (
            "✗ GitHub appears to be having an outage — try again in a few"
            " minutes (https://www.githubstatus.com)."
        )
    if "Could not resolve host" in stderr or "unable to access" in stderr:
        return "✗ Network error — cannot reach the remote repository."
    if "could not read Username" in stderr or "terminal prompts disabled" in stderr:
        # Anonymous fetch of a public repo got HTTP 401. GitHub does this
        # during outages (and for renamed/private repos) — it is not a
        # credentials problem on the user's side.
        return (
            "✗ GitHub rejected the anonymous fetch (asked for a login) — this"
            " usually means a GitHub outage; try again in a few minutes"
            " (https://www.githubstatus.com). If it persists, check"
            " `git remote -v` points at a public repo."
        )
    if "Authentication failed" in stderr:
        return "✗ Authentication failed — check your git credentials or SSH key."
    return "✗ Failed to fetch updates from origin."


def _print_fetch_failure(stderr: str) -> None:
    """Print the classified diagnosis plus the first raw stderr line."""
    stderr = (stderr or "").strip()
    print(_classify_fetch_failure(stderr))
    if stderr:
        print(f"  {stderr.splitlines()[0]}")


def _cmd_update_check(branch: str = "main", *, branch_explicit: bool = False):
    """Implement ``hermes update --check``: fetch and report without installing.

    ``branch`` selects which branch the check compares against. Default is
    "main"; callers can pass another branch to ask "are there new commits
    on origin/<branch>?" without performing the update.

    ``branch_explicit`` is True iff the caller passed --branch on the CLI.
    Installs that can't honor non-default branches (e.g. Docker) surface a
    one-line notice instead of silently dropping the flag.
    """
    # Shared admission gate (#91277 Phase 3): same marker-first decision as
    # the apply path, so --check can never report git state for an install
    # whose real update mechanism is an image pull.
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(_m().PROJECT_ROOT)
    if refusal is not None:
        print(refusal.message)
        record_refusal_receipt(refusal)
        sys.exit(2)

    git_dir = _m().PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("✗ Not a git repository — cannot check for updates.")
        sys.exit(1)

    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # An interrupted fetch can leave .git/shallow.lock (or another lock) behind,
    # making every later fetch fail with "File exists". Self-heal before fetching.
    from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

    cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
    for lock_path in cleared:
        print(f"  (removed stale git lock: {lock_path})")
    # Aborted fetches on flaky lines also strand tmp_pack_* debris in
    # .git/objects/pack — unchecked it reached 6 GB and corrupted the pack
    # dir outright (#93732). Same age+process safety contract as the locks.
    swept = clear_stale_tmp_packs(_m().PROJECT_ROOT)
    if swept:
        print(f"  (removed {len(swept)} aborted-fetch pack temp file(s))")

    # Fetch only <branch>: a bare `git fetch <remote>` pulls thousands of
    # auto-generated branches. Prefer upstream as canonical, but only for main
    # (a fork's non-default branch has no upstream counterpart). Installer
    # checkouts are shallow (`--depth 1`); a plain fetch would unshallow them
    # and rev-list would report a huge bogus "behind" count, so fetch with
    # --depth 1 and report presence-only.
    is_shallow = (
        _git_run(git_cmd, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
        == "true"
    )
    depth_args = ["--depth", "1"] if is_shallow else []

    if branch == "main":
        # Probe locally (~6 ms) for an 'upstream' remote before spending a
        # network fetch (~0.3-1 s) that non-fork installs would always fail.
        has_upstream_remote = (
            _git_run(git_cmd, ["remote", "get-url", "upstream"]).returncode
            == 0
        )
        fetch_result = None
        if has_upstream_remote:
            print("→ Fetching from upstream...")
            fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["upstream", branch], network=True)
        if fetch_result is not None and fetch_result.returncode == 0:
            compare_branch = f"upstream/{branch}"
        else:
            # No upstream remote, or the upstream fetch failed — use origin.
            print("→ Fetching from origin...")
            fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["origin", branch], network=True)
            compare_branch = f"origin/{branch}"
    else:
        # Non-default branch: compare against origin/<branch> directly.
        print("→ Fetching from origin...")
        fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["origin", branch], network=True)
        compare_branch = f"origin/{branch}"

    if fetch_result.returncode != 0:
        _print_fetch_failure(fetch_result.stderr)
        sys.exit(1)

    # Verify the compare ref exists first: rev-list on a bogus ref exits 128
    # and (with check=True) would surface a Python traceback.
    verify_result = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", compare_branch])
    if verify_result.returncode != 0:
        print(f"✗ Branch '{branch}' not found on {compare_branch.split('/', 1)[0]}.")
        sys.exit(1)

    if is_shallow:
        # No history across the shallow boundary: compare tip SHAs (like the
        # banner's _check_via_local_git), then recover the exact count via the
        # GitHub compare API, whose graph is complete.
        head_sha = _git_run(git_cmd, ["rev-parse", "HEAD"]).stdout.strip()
        target_sha = _git_run(git_cmd, ["rev-parse", compare_branch]).stdout.strip()
        if head_sha and target_sha and head_sha == target_sha:
            print("✓ Already up to date.")
        else:
            from hermes_cli.banner import _github_compare_behind
            from hermes_cli.config import recommended_update_command

            counted = _github_compare_behind(head_sha, target_sha)
            if counted == 0:
                # Local commits on top of the remote tip — not behind.
                print("✓ Already up to date.")
                return
            if counted is not None:
                commits_word = "commit" if counted == 1 else "commits"
                print(f"⚕ Update available: {counted} {commits_word} behind {compare_branch}.")
            else:
                print(f"⚕ Update available (behind {compare_branch}).")
            print(f"  Run '{recommended_update_command()}' to install.")
        return

    rev_result = _git_run(git_cmd, ["rev-list", f"HEAD..{compare_branch}", "--count"], check=True)
    behind = int(rev_result.stdout.strip())

    if behind == 0:
        print("✓ Already up to date.")
    else:
        commits_word = "commit" if behind == 1 else "commits"
        print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
        from hermes_cli.config import recommended_update_command

        print(f"  Run '{recommended_update_command()}' to install.")

def _ensure_fhs_path_guard() -> None:
    """Ensure /usr/local/bin is on PATH for RHEL-family root non-login shells.

    Mirrors the post-symlink probe in ``scripts/install.sh`` so existing FHS
    root installs on RHEL/CentOS/Rocky/Alma 8+ get repaired on ``hermes
    update``. In non-login interactive shells there (su, sudo -s, tmux panes)
    neither /etc/bashrc nor /root/.bash_profile adds /usr/local/bin, so
    ``hermes`` prints ``command not found`` despite the symlink.

    Silent no-op on non-Linux, non-root, non-FHS installs, and wherever
    ``bash -i -c 'command -v hermes'`` already resolves. Idempotent.
    """
    if _m().sys.platform != "linux":
        return
    try:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux FHS helper, guarded by sys.platform == "linux" above + AttributeError catch
            return
    except AttributeError:
        return
    # Only act when this is actually an FHS-layout install (command link at
    # /usr/local/bin/hermes, code at /usr/local/lib/hermes-agent).
    fhs_link = Path("/usr/local/bin/hermes")
    if not fhs_link.is_symlink() and not fhs_link.exists():
        return

    # Probe a fresh non-login interactive bash the way the user will use it.
    # ``bash -i -c`` sources ~/.bashrc but NOT ~/.bash_profile or /etc/profile,
    # which is the exact scenario where RHEL root loses /usr/local/bin.
    home = os.environ.get("HOME") or "/root"
    try:
        probe = subprocess.run(
            [
                "env",
                "-i",
                f"HOME={home}",
                f"TERM={os.environ.get('TERM', 'dumb')}",
                "bash",
                "-i",
                "-c",
                "command -v hermes",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no bash or probe hung — don't block update on this
    if probe.returncode == 0:
        return  # already on PATH, nothing to do

    path_line = 'export PATH="/usr/local/bin:$PATH"'
    path_comment = (
        "# Hermes Agent — ensure /usr/local/bin is on PATH " "(RHEL non-login shells)"
    )
    wrote_any = False
    for candidate in (".bashrc", ".bash_profile"):
        cfg = Path(home) / candidate
        if not cfg.is_file():
            continue
        try:
            existing = cfg.read_text(errors="replace", encoding="utf-8")
        except OSError:
            continue
        # Idempotency: skip if any uncommented PATH= line already references
        # /usr/local/bin.  Mirrors the grep pattern used by install.sh.
        already_guarded = any(
            "/usr/local/bin" in line
            and "PATH" in line
            and not line.lstrip().startswith("#")
            for line in existing.splitlines()
        )
        if already_guarded:
            continue
        try:
            with cfg.open("a", encoding="utf-8") as f:
                f.write("\n" + path_comment + "\n" + path_line + "\n")
        except OSError as e:
            print(f"  ⚠ Could not update {cfg}: {e}")
            continue
        print(f"  ✓ Added /usr/local/bin to PATH in {cfg}")
        wrote_any = True
    if wrote_any:
        print("    (reload your shell or run 'source ~/.bashrc' to pick it up)")

def _ensure_acp_launcher() -> None:
    r"""Self-heal: install a ``hermes-acp`` launcher next to the ``hermes`` one.

    Mirrors the launcher block in ``scripts/install.sh``. ACP hosts (Zed,
    JetBrains, Buzz Desktop) resolve ``hermes-acp`` on the login-shell PATH,
    but the console script lives inside the venv, so they report Hermes as
    not installed. The shim just delegates to the sibling ``hermes`` launcher
    with the ``acp`` subcommand, which is correct for every install layout.

    No-op on Windows: install.ps1 stages launchers into ``$HermesHome\bin``
    and puts THAT on PATH — never ``venv\Scripts``, which would shadow the
    user's ``python`` (#83797); ``ensure_windows_bin_launchers`` re-stages
    them. Also no-op where ``hermes-acp`` already exists next to ``hermes``.
    Unwritable dirs (``/usr/local/bin`` as non-root) are skipped. Idempotent.
    """
    if _m().sys.platform == "win32":
        # Windows launcher staging/repair lives in _install_repair
        # (ensure_windows_bin_launchers at process start,
        # migrate_windows_bin_path in this command's tail) — not here.
        return
    for bin_dir in (Path.home() / ".local" / "bin", Path("/usr/local/bin")):
        hermes_cmd = bin_dir / "hermes"
        acp_cmd = bin_dir / "hermes-acp"
        try:
            if not (hermes_cmd.is_file() or hermes_cmd.is_symlink()):
                continue
            # Already present (console script, earlier shim, or symlink).
            # is_symlink() catches broken symlinks exists() misses; never
            # follow-and-overwrite (#21454).
            if acp_cmd.exists() or acp_cmd.is_symlink():
                continue
            shim = (
                "#!/usr/bin/env bash\n"
                "# Hermes Agent — ACP launcher (written by `hermes update`).\n"
                "# ACP hosts (Zed, JetBrains, Buzz) resolve the agent by this\n"
                "# command name on the login-shell PATH.\n"
                f'exec "{hermes_cmd}" acp "$@"\n'
            )
            acp_cmd.write_text(shim, encoding="utf-8")
            acp_cmd.chmod(acp_cmd.stat().st_mode | 0o755)
        except OSError:
            continue
        print(f"  ✓ Installed hermes-acp launcher → {acp_cmd}")

_PRE_UPDATE_SNAPSHOT_KEEP = 1
# {profile: snapshot_id} from this run's pre-update backup, consumed by the
# post-update per-profile cron-jobs safety net (#66140). Module-level because
# snapshot and restore run far apart in _cmd_update_impl.
_LAST_SIBLING_SNAPSHOTS: dict = {}

# Per-file cap for the quick snapshot; larger files are skipped with a warning.
# The snapshot protects small, hard-to-regenerate state (pairing JSONs, cron,
# config, auth) — not a multi-GB state.db (a 24 GB one cost ~60s and 24 GB/update).
_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30  # 1 GiB

def _resolve_pre_update_backup_mode(args) -> str:
    """Resolve the pre-update backup mode: ``"off"``, ``"quick"``, or ``"full"``.

    CLI flags win over config; ``--no-backup`` beats ``--backup``. Config
    accepts the mode strings plus legacy booleans: ``true`` → ``full``,
    ``false`` → ``off`` (an explicit opt-out also disables the quick
    snapshot). Missing key defaults to ``quick``.
    """
    if getattr(args, "no_backup", False):
        return "off"
    if getattr(args, "backup", False):
        return "full"

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not load config for pre-update backup: %s", exc
        )
        cfg = {}

    updates_cfg = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    raw = updates_cfg.get("pre_update_backup", "quick")

    if raw is True:
        return "full"
    if raw is False:
        return "off"
    mode = str(raw).strip().lower()
    if mode in ("off", "false", "none", "disabled"):
        return "off"
    if mode in ("full", "zip", "true"):
        return "full"
    if mode == "quick":
        return "quick"
    logging.getLogger(__name__).warning(
        "Unknown updates.pre_update_backup value %r — using 'quick'", raw
    )
    return "quick"

def _run_pre_update_backup(args) -> Optional[str]:
    """Run the pre-update safety backup and return the quick-snapshot id.

    Gated on ``updates.pre_update_backup``:

    - ``off``   — nothing runs; explicit opt-out is honored fully.
    - ``quick`` (default) — snapshot of critical small files
      (``_QUICK_STATE_FILES``) under ``state-snapshots/``; files over 1 GiB
      are skipped so a bloated state.db can never stall the update (#15733,
      #34600).
    - ``full``  — quick snapshot PLUS a zip of HERMES_HOME under ``backups/``
      (restorable via ``hermes import``; exists because of the #48200 wipe).

    ``--backup`` forces ``full``; ``--no-backup`` forces ``off``. Never raises.
    Returns the quick-snapshot id (used by the post-update cron-jobs restore),
    or ``None`` when mode is ``off`` or the snapshot failed.
    """
    mode = _resolve_pre_update_backup_mode(args)

    if mode == "off":
        if getattr(args, "no_backup", False):
            print("◆ Pre-update backup: skipped (--no-backup)")
            print()
        # Config-level off is silent — the user opted out; don't spam them
        # on every update.
        return None

    snapshot_id = None
    try:
        from hermes_cli.backup import (
            _quick_snapshot_root,
            create_quick_snapshot,
            verify_sqlite_integrity,
        )

        # NOTE: this function later does `from hermes_constants import
        # get_hermes_home`, which makes the name function-local — the
        # module-level import is shadowed and unbound here. Alias explicitly.
        from hermes_cli.config import get_hermes_home as _get_home

        snapshot_id = create_quick_snapshot(
            label="pre-update",
            keep=_PRE_UPDATE_SNAPSHOT_KEEP,
            max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        )

        # Verify the live state.db is still intact after the snapshot: a
        # concurrent process (antivirus, force-killed gateway, Windows filter
        # driver) can corrupt it at any point, and a silent zeroing would
        # otherwise proceed to exit 0 — the #68474 symptom.
        if snapshot_id:
            _src_path = _get_home() / "state.db"
            if _src_path.exists():
                _integrity = verify_sqlite_integrity(
                    _src_path,
                    check_header=True,
                    run_pragma=True,
                    max_bytes=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
                )
                if not _integrity.get("valid"):
                    _msg = _integrity.get("message", "unknown error")
                    print(
                        f"  ⚠ state.db integrity check FAILED after snapshot: {_msg}"
                    )
                    # Check if the snapshot itself is valid.
                    _snap_root = _quick_snapshot_root(_get_home())
                    _snap_state = _snap_root / snapshot_id / "state.db"
                    if _snap_state.exists():
                        _snap_ok = verify_sqlite_integrity(
                            _snap_state, check_header=True, run_pragma=True
                        )
                        if _snap_ok.get("valid"):
                            print(
                                "  ✓ Snapshot copy is valid — continuing update."
                            )
                            print(
                                "    If state.db is lost after update it will be auto-restored."
                            )
                        else:
                            print(
                                "  ✗ Snapshot copy ALSO failed integrity — "
                                "the source was already corrupted before the backup."
                            )
                    else:
                        print(
                            "  ⚠ Snapshot does not contain state.db (was skipped or too large)."
                        )
                    print()
        if snapshot_id:
            print(f"◆ Pre-update snapshot: {snapshot_id}")

        # #66140: the code swap + fleet restart touch EVERY profile, so
        # every profile gets the same snapshot (same set, same 1GiB cap,
        # keep=1) under its own state-snapshots/. Best-effort per profile.
        try:
            from hermes_cli.backup import create_pre_update_snapshots_all_profiles

            _sibling_snaps = create_pre_update_snapshots_all_profiles(
                keep=_PRE_UPDATE_SNAPSHOT_KEEP,
                max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
            )
            if _sibling_snaps:
                print(
                    f"◆ Sibling profile snapshot(s): "
                    + ", ".join(sorted(_sibling_snaps))
                )
                _record_update_step(
                    "sibling_profile_snapshots",
                    True,
                    ", ".join(
                        f"{k}={v}" for k, v in sorted(_sibling_snaps.items())
                    ),
                )
                global _LAST_SIBLING_SNAPSHOTS
                _LAST_SIBLING_SNAPSHOTS = _sibling_snaps
        except Exception as _sib_exc:
            logging.getLogger(__name__).debug(
                "Sibling profile snapshots failed: %s", _sib_exc
            )
    except Exception as exc:
        # Never let a snapshot failure block an update.
        logging.getLogger(__name__).debug("Pre-update snapshot failed: %s", exc)

    if mode != "full":
        if snapshot_id:
            print()
        return snapshot_id

    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(
            f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update."
        )
        print()
        return snapshot_id

    try:
        from hermes_cli.config import load_config

        _keep = (load_config() or {}).get("updates", {}).get("backup_keep", 5)
    except Exception:
        _keep = 5

    print("◆ Creating pre-update backup...")
    t0 = _time.monotonic()
    try:
        out_path = create_pre_update_backup(keep=int(_keep))
    except Exception as exc:  # defensive — helper already swallows, but just in case
        print(f"  ⚠ Backup failed: {exc}")
        print("  Continuing with update.")
        print()
        return snapshot_id

    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return snapshot_id

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    # Human-readable size
    from hermes_cli.sizefmt import format_bytes

    size_str = format_bytes(size_bytes)

    # Render path using display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        try:
            display_path = f"{display_hermes_home()}/{out_path.relative_to(home)}"
        except ValueError:
            display_path = str(out_path)
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({size_str}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  set updates.pre_update_backup: quick (or off) in config.yaml")
    print()
    return snapshot_id

def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False

def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors

def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs inside the venv interpreter (NOT this process — ``hermes update`` may
    run under a different Python). Catches a half-updated venv: checkout
    current but a dependency sync failed or was killed partway (e.g. Windows
    access-denied on a loaded .pyd). Without it, a current checkout prints
    "Already up to date!" and never re-syncs, so the install stays broken.

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv interpreter. Normal for a dev checkout (report healthy to
        # avoid forced reinstalls), but on a MANAGED install (bootstrap stamp
        # or `.update-incomplete` present) the venv IS the install — its
        # absence means a repair was interrupted after the old venv was moved
        # aside, and "Already up to date!" would be a lie.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete",
            _m()._update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=60,
            cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""

def _self_and_non_gateway_ancestor_pids(psutil) -> set[int]:
    """PIDs a venv-holder scan must never nominate: this process and its ancestry.

    #87594: do NOT blanket-exclude ancestors. When ``/update`` runs from a
    messaging platform the updater is a CHILD of the gateway; hiding it means
    the pause machinery never sees the one process it exists to stop and the
    update dead-ends on ``venv-blocked``. Keep a GATEWAY ancestor visible (the
    pause path stops it gracefully; a detached child survives on Windows) and
    exclude every other ancestor — an updater must never nominate its own
    interactive ancestry as a blocker.
    """
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    skip: set[int] = {os.getpid()}
    try:
        for anc in psutil.Process().parents():
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass
    return skip


def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest Windows lock-holder class:
    the Desktop backend (``python.exe -m hermes_cli.main serve``) and anything
    running off ``venv\\Scripts\\python(w).exe``. They keep native ``.pyd``
    files mapped, so a mid-update dependency sync dies with access-denied and
    strands the venv half-updated.

    Killing them is pointless (the Desktop app respawns its backend), so the
    caller should refuse and ask the user to close the app. Returns
    ``(pid, name, cmdline)`` tuples; empty off-Windows / without psutil / no
    matches. This process and its ancestors are excluded. Never raises.
    """
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(_m().PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(_m().PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip |= _self_and_non_gateway_ancestor_pids(psutil)

    matches: list[tuple[int, str, str]] = []
    try:
        # On Windows cmdline/cwd are expensive per-process queries; with 500+
        # processes prefetching them can exceed the Desktop preflight watchdog.
        # Collect cheap identity fields first, fetch cmdline/cwd lazily for
        # plausible Python/uv/Hermes candidates.
        proc_iter = psutil.process_iter(["pid", "exe", "name"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        name = str(info.get("name") or Path(exe).name)
        name_low = name.lower()

        if not is_holder and not (
            name_low.startswith(("python", "pypy"))
            or name_low in {"uv.exe", "uvx.exe", "hermes.exe"}
        ):
            continue

        try:
            cmdline_raw = " ".join(proc.cmdline() or [])
        except Exception:
            cmdline_raw = ""
        cmdline_low = cmdline_raw.lower()
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv yet still holds its .pyd files. Match on the cmdline
        # instead: this venv's path, or `-m hermes_cli.main` tied to this
        # install (root in the cmdline or as cwd).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            try:
                cwd_low = str(proc.cwd() or "").lower().rstrip(os.sep) + os.sep
            except Exception:
                cwd_low = os.sep
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        # Return the FULL cmdline: callers parse it (the Desktop preflight's
        # pausable-gateway exemption looks for `gateway run`). Truncating here
        # once cut long interpreter paths before the argv, so autostarted
        # gateways were misreported as blockers. Truncate only at display time.
        matches.append((int(pid), str(name), cmdline_raw))
    return matches

# Native-extension modules that pin files inside the venv once imported. If
# the updater itself has one loaded, Windows blocks REPLACE on the mapped
# ``.pyd``/``.dll`` and the sync dies with ``os error 5`` between uninstall
# and reinstall, stranding the venv half-updated (#83569). ``cryptography``
# is the canonical case; PyYAML's ``_yaml`` is loaded by every CLI process.
# Kept as defence-in-depth against future eager imports, but the guard must
# be HONEST (#86735/#86780/#86781: a preflight firing on every run, before
# the fetch, re-bricked the flow it protected). Two honesty gates:
#
# 1. Fire only when the sync would actually REWRITE the loaded distribution
#    (``_dependency_sync_would_rewrite``); a satisfied pin means uv/pip
#    never touch the mapped ``.pyd``.
# 2. Run AFTER the code swap, right before the venv rewrite — so gate 1
#    compares against the NEW pyproject and a deferral leaves the user on
#    new code with only the dependency install pending for the next launch's
#    marker recovery.
#
# Keys are ``sys.modules`` prefixes; values are ``(display name, PyPI dist)``.
_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml"),
}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether ``uv pip install -e .[all]`` would replace *dist_name*'s files.

    Compares the installed version against every applicable requirement in
    the on-disk ``pyproject.toml`` (base deps plus all extras). ``False`` —
    every pin satisfied, a mapped extension is NOT at risk; ``True`` — some
    pin unsatisfied or dist missing; ``None`` — undeterminable.

    Never raises. Callers treat ``None`` as fail-OPEN (no deferral): PyYAML
    is loaded by every process, so deferring on uncertainty would recreate
    the #86735 always-firing loop.
    """
    try:
        from importlib import metadata as _ilmd

        installed = _ilmd.version(dist_name)
    except Exception:
        return True  # not installed → the sync will definitely install it
    try:
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        pyproject = _m().PROJECT_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        req_strings: list[str] = list(project.get("dependencies") or [])
        for extra_reqs in (project.get("optional-dependencies") or {}).values():
            req_strings.extend(extra_reqs or [])

        target = canonicalize_name(dist_name)
        installed_v = Version(installed)
        saw_pin = False
        for req_str in req_strings:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            if canonicalize_name(req.name) != target:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            saw_pin = True
            if installed_v not in req.specifier:
                return True
        if saw_pin:
            return False
        # Not pinned anywhere in pyproject: the resolver may still move it
        # as a transitive — we cannot cheaply predict that, so stay honest
        # about the uncertainty.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Native venv extensions loaded into THIS process that the sync would rewrite.

    Returns display names (empty off Windows — POSIX lets a running process
    keep using an unlinked inode, so self-locking is a Windows-only hazard).
    A loaded module whose installed version already satisfies the on-disk
    pyproject pins is NOT reported: the dependency sync will not touch its
    files, so there is no swap at risk (#86735 — the always-firing variant
    of this preflight bricked every Windows update).  Never raises.
    """
    if not _m()._is_windows():
        return []
    found = []
    for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items():
        if prefix not in sys.modules:
            continue
        # Defer ONLY on a CONFIRMED pending rewrite; "unknown" must fail OPEN,
        # since PyYAML is loaded in every CLI process and treating unknown as
        # at-risk recreated the always-firing loop (#86735). A missed deferral
        # only yields the pre-existing mid-sync os error 5, which marker
        # recovery already handles — far less harmful than an update that
        # can never run.
        if _m()._dependency_sync_would_rewrite(dist) is not True:
            continue
        found.append(display)
    return sorted(set(found))


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer the venv rewrite when THIS process holds something it must replace.

    Runs after the code swap, right before the venv rewrite, so a deferral
    leaves the user on NEW code with only the dependency install pending.
    No-op when nothing at-risk is held. Two hazards with different recoveries:

    - A mapped native extension (``.pyd``): exit 2 and let the next launch's
      marker recovery finish the install before importing anything heavy.
    - The ``hermes.exe`` shim we were launched from (#88838, #89599): every
      future launch is also the shim, so the marker would defer forever.
      Hand the install to a child under the venv interpreter and exit.
    """
    locked = _m()._detect_self_loaded_native_modules()
    if locked:
        _m()._defer_update_for_self_lock(locked)
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(2)

    if _m()._reexec_dependency_sync_off_windows_shim():
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(0)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the dependency sync when the updater holds a lock.

    The install cannot win this race from inside the locked process — even
    killing threads would not unmap the image — so defer it: drop the
    update-incomplete marker (next launch's fresh process completes the
    install before importing anything heavy), explain, and exit 2 like the
    other preflight refusals.
    """
    print("✗ This updater process has already loaded native venv modules that")
    print("  the dependency sync must replace:")
    for name in loaded:
        print(f"    {name}")
    print()
    print("  On Windows a mapped extension cannot be replaced by the process")
    print("  holding it. The code update has been applied; only the dependency")
    print("  sync has been deferred: the next `hermes` launch will complete it")
    print("  in a fresh process before anything imports these modules.")
    _m()._write_update_incomplete_marker()


_HOLDER_VALUE_FLAGS_FALLBACK = frozenset(
    {
        "--profile", "-p", "--config",
        "--model", "-m", "--provider", "--reasoning",
        "--toolsets", "-t", "--skills", "-s",
        "--continue", "-c", "--resume", "-r",
        "--oneshot", "-z", "--in", "--usage-file",
    }
)
_holder_value_flags_cache: frozenset | None = None


def _holder_value_flags() -> frozenset:
    """Top-level CLI flags that consume a value — derived from the REAL parser.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0) so
    the holder classifier can't drift from argparse (#91869: a handwritten
    subset misparsed ``--reasoning high serve`` as subcommand ``high``). The
    pre-argparse profile selectors (``--profile``/``-p``, ``--config``) are
    added explicitly since they're stripped before argparse sees argv. Falls
    back to a static snapshot when the parser can't be imported (the updater
    must classify holders even on a broken tree). Cached per process.
    """
    global _holder_value_flags_cache
    if _holder_value_flags_cache is not None:
        return _holder_value_flags_cache
    flags: set[str] = {"--profile", "-p", "--config"}
    try:
        from hermes_cli._parser import build_top_level_parser

        parser = build_top_level_parser()[0]
        for action in parser._actions:
            if action.option_strings and action.nargs != 0:
                flags.update(action.option_strings)
        _holder_value_flags_cache = frozenset(flags)
    except Exception:
        _holder_value_flags_cache = _HOLDER_VALUE_FLAGS_FALLBACK
    return _holder_value_flags_cache


def _hermes_holder_subcommand(cmdline: str) -> str | None:
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None.

    Token-based, never substring (#90778: ``kanban --preserve-cache`` contains
    \"serve\" and got labeled as the Desktop backend). Finds the
    ``hermes_cli.main`` / ``hermes(.exe)`` entry token, then returns the first
    following token that isn't a flag or a flag's value (profile selectors
    skipped). None when undeterminable — callers must NOT guess a label.
    """
    try:
        import shlex

        tokens = shlex.split(cmdline, posix=False)
    except Exception:
        tokens = cmdline.split()

    entry_idx: int | None = None
    for i, token in enumerate(tokens):
        low = token.lower().strip('"')
        if low.endswith("hermes_cli.main") and i > 0 and tokens[i - 1] == "-m":
            entry_idx = i
            break
        base = low.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if base in ("hermes", "hermes.exe"):
            entry_idx = i
            break
    if entry_idx is None:
        return None

    value_flags = _holder_value_flags()
    i = entry_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags or token.split("=", 1)[0] in value_flags:
            # --flag value consumes two tokens; --flag=value consumes one.
            i += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token.lower()
    return None


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them.

    Holder labels come from the parsed SUBCOMMAND, never substring matching
    (#90778): a standalone ``hermes dashboard`` must not be labeled as the
    Desktop backend (advice to close an app that isn't running), and flags
    like ``--preserve-cache`` must not match \"serve\". Unknown argv gets no
    hint rather than a wrong one.
    """
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    hint_by_subcommand = {
        "serve": "  ← Hermes backend (if the Desktop app is open, close it)",
        "dashboard": "  ← hermes dashboard (stop it: hermes dashboard stop, or close that terminal)",
        "gateway": "  ← gateway",
    }
    for pid, name, cmdline in matches[:6]:
        sub = _hermes_holder_subcommand(cmdline)
        hint = hint_by_subcommand.get(sub or "", "")
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)

def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Return venv-interpreter ancestors of *pids* that hold the install open.

    On Windows a gateway started through the venv shim is a two-process chain:
    ``venv\\Scripts\\python.exe`` (the launcher, which keeps venv ``.pyd``
    files mapped) spawns the real interpreter from uv's managed CPython. The
    PID file is written by the *child*, so ``find_gateway_pids()`` / the pause
    set only see the uv-side worker, while ``_detect_venv_python_processes()``
    (venv path prefix) sees the *launcher*. The sets are disjoint, so a paused
    gateway still tripped the venv-holder guard and aborted the update.

    Walk one hop up from each mapped gateway PID and keep only ancestors under
    the project venv; unrelated ancestors (the Scheduled Task's ``cmd.exe``,
    an operator's shell) are ignored to bound the blast radius. Never raises.
    """
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep

    skip = _self_and_non_gateway_ancestor_pids(psutil)

    found: list[int] = []
    for pid in pids:
        try:
            parent = psutil.Process(int(pid)).parent()
        except Exception:
            continue
        if parent is None:
            continue
        ppid = int(parent.pid)
        if ppid in skip or ppid in found or ppid in set(pids):
            continue
        try:
            exe = (parent.exe() or "").lower()
        except Exception:
            continue
        if exe.startswith(venv_prefix):
            found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining venv holder is a pausable gateway.

    ``_pause_windows_gateways_for_update()`` stops the gateways its discovery
    finds, but the venv-holder guard sees the process table as it is *now*: a
    gateway respawned by its supervisor inside the pause→guard window, or one
    started through an unmapped spawn path, still holds venv ``.pyd`` files and
    would dead-end the update on exactly the process the pause exists to stop.

    Holders are classified with the same matcher the Desktop preflight uses
    (``_is_pausable_gateway``) so exemption and tolerance cannot drift apart.
    The scan keeps only a 120-char cmdline prefix, so live argv is re-read via
    psutil when possible, falling back to the prefix.

    Returns ``None`` when any holder is not a pausable gateway (operator REPL,
    stray script, Desktop backend) — nothing downstream can pause it, so the
    guard must keep refusing.
    """
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            try:
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
            except Exception:
                pass
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _refuse_gateway_ancestor_tree_kill(
    pids: list[int], *, gateway_mode: bool
) -> bool:
    """Refuse a plain Windows update that would kill its own process tree.

    A chat agent can run plain ``hermes update`` via its terminal tool, making
    the updater a child of the gateway; leftover-holder recovery uses
    ``taskkill /T /F``, so force-stopping that gateway kills the updater before
    it mutates the checkout (#98814). ``/update`` (``--gateway`` hand-off) is
    exempt: it detaches the updater with file-based progress/result delivery.
    Otherwise refuse only when a nominated gateway is positively an ancestor of
    this process; if ancestry cannot be established, keep existing recovery.
    """
    if gateway_mode or not pids:
        return False

    try:
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        ancestors = [
            int(pid)
            for pid in pids
            if _is_pid_ancestor_of_current_process(int(pid))
        ]
    except Exception as exc:
        logger.debug("Could not inspect gateway ancestry before tree-kill: %s", exc)
        return False

    if not ancestors:
        return False

    rendered = ", ".join(str(pid) for pid in ancestors)
    print(
        "✗ Refusing to stop the gateway process tree because this updater "
        f"is running inside it (gateway PID(s): {rendered})."
    )
    print(
        "  On Windows, taskkill /T would terminate the updater before the "
        "update can run."
    )
    print("  From a chat platform, use `/update` instead.")
    print("  Otherwise, run `hermes update` from a separate terminal.")
    return True


def _ledger_manual_serve_holders(
    matches: list[tuple[int, str, str]],
) -> list[dict]:
    """Ledger entries for venv holders that are MANUAL serve/dashboard backends.

    Positive identity only (#63206): the process self-registered in the spawn
    ledger with purpose serve/dashboard, its (pid, create_time) still matches
    a live process, and its recorded spawner is NOT alive (a Desktop-owned
    backend keeps its live Electron spawner and must keep the refusal — the
    app would respawn what we kill; a PowerShell-launched serve has no live
    Hermes spawner). Returns the full ledger entries so the relauncher can
    rebuild the launch command from structured host/port/profile instead of
    parsing argv.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead
    except Exception:
        return []
    holder_pids = {int(pid) for pid, _name, _cmd in matches}
    out: list[dict] = []
    for entry in ledger_entries():
        if entry.get("purpose") not in ("serve", "dashboard"):
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or pid not in holder_pids:
            continue
        if spawner_is_dead(entry) is False:
            continue  # live Desktop supervisor owns it — keep refusing
        out.append(entry)
    return out


def _serve_relaunch_commands(entries: list[dict]) -> list[list[str]]:
    """Rebuild launch commands for stopped serves from structured identity.

    Uses the ledger's host/port/profile fields — never argv parsing (a
    joined argv string cannot round-trip Windows paths with spaces). Entries
    without a recorded port are skipped; the caller prints the manual hint
    for those.
    """
    commands: list[list[str]] = []
    hermes = None
    try:
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            for name in ("hermes.exe", "hermes"):
                candidate = scripts_dir / name
                if candidate.is_file():
                    hermes = str(candidate)
                    break
    except Exception:
        hermes = None
    if hermes is None:
        hermes = "hermes"
    for entry in entries:
        port = entry.get("port")
        if not isinstance(port, int) or port <= 0:
            continue
        cmd = [hermes]
        profile = str(entry.get("profile") or "")
        if profile and profile != "default":
            cmd += ["--profile", profile]
        cmd.append(str(entry.get("purpose")))
        host = str(entry.get("host") or "")
        if host:
            cmd += ["--host", host]
        cmd += ["--port", str(port)]
        commands.append(cmd)
    return commands


def _relaunch_stopped_serves(token: dict) -> None:
    """Idempotent atexit relaunch of manual serves stopped by the venv guard.

    Mirrors the gateway resume token contract: `pending` flips False on the
    first invocation so the explicit call and the atexit registration cannot
    double-spawn (#63206).
    """
    if not token.get("pending"):
        return
    token["pending"] = False
    entries = token.get("entries") or []
    if not entries:
        return
    commands = _serve_relaunch_commands(entries)
    skipped = len(entries) - len(commands)
    failed: list = []
    if commands:
        print("  ⟲ Relaunching stopped serve/dashboard backend(s)")
        failed = _m()._respawn_dashboard_processes(commands)
    if skipped or failed:
        print(
            "  ⚠ Some stopped backends could not be relaunched automatically; "
            "restart them manually (hermes serve --host <ip> --port <port>)."
        )
    _record_update_step(
        "serve_relaunch",
        not failed and not skipped,
        f"relaunched={len(commands) - len(failed)} failed={len(failed)} skipped={skipped}",
    )


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[tuple[int, int]] | None:
    """PIDs from *matches* when every remaining holder is an ORPHANED backend.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by
    design: while the Desktop is open, killing it is futile (the app respawns
    it within seconds). But in the GUI-updater hand-off the Desktop has
    *already exited* — by contract it tree-kills its backends before spawning
    hermes-setup, and the update-in-progress marker parks any relaunched
    Desktop (#50238). A ``serve`` backend still holding the venv then is a
    straggler whose supervisor is gone (SIGTERM raced its spawn, or a crashed
    window); refusing on it dead-ends the update with "Hermes is still
    running" while the user sees zero open windows.

    A holder qualifies only when BOTH hold:

    - its cmdline is a Hermes backend (``hermes_cli.main`` + ``serve`` /
      ``dashboard``), and
    - its supervising parent is demonstrably gone: the parent PID no longer
      exists, or was reused (parent created *after* the child).

    Tree-aware: the scanner may also return an orphan's managed-runtime child
    (the ``.hermes-runtime`` interpreter), which has a live parent and is not
    a ``serve`` cmdline. Holders inside an accepted orphan root's tree are
    folded into that root; only roots are returned (``taskkill /T`` reaps
    descendants).

    Any other live-parent backend, non-backend holder outside an orphan tree,
    or unprovable case disqualifies the whole set → ``None`` (keep refusing).
    Also ``None`` when psutil is unavailable. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[tuple[int, int]] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Holder exited between scan and classification — nothing to
            # reap, nothing blocking. Skip it.
            continue
        except Exception:
            pass
        low = argv.lower()
        if not _is_backend(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            # Fingerprint from the SAME psutil handle, quantized to centiseconds
            # like gateway.status.get_process_start_time on Windows, so it
            # round-trips through pid_is_hermes at kill time (/proc/<pid>/stat
            # would read the HOST table in different units).
            process_start_time = int(round(proc.create_time() * 100))
        except psutil.NoSuchProcess:
            # The candidate itself exited during classification; there is
            # nothing left to reap and no identity to pass to taskkill.
            continue
        except Exception:
            return None

        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a
                # recycled PID, not the real (dead) supervisor.
                if parent.create_time() <= proc.create_time():
                    # Live parent — not a root, but possibly an orphan root's
                    # descendant (the venv python.exe trampoline re-execs the
                    # uv interpreter with the SAME argv). Defer to pass 2.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append((int(pid), process_start_time))

    # Pass 2: every non-backend holder must be a descendant of an accepted
    # orphan root — then it dies with the root's tree reap. Anything else
    # (operator REPL, stray script) keeps the refusal.
    root_set = {pid for pid, _start_time in roots}
    for pid, _low in remaining:
        if not root_set:
            return None
        try:
            ancestors = {int(a.pid) for a in psutil.Process(pid).parents()}
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
        if not (ancestors & root_set):
            return None
    return roots


def _ledger_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int]:
    """PIDs positively identified by the spawn ledger as orphaned backends.

    The strongest rung: look each venv holder up in the machine spawn ledger
    (``hermes_cli.process_identity``) instead of inferring lineage from PPIDs
    or cmdline shape. A holder qualifies when ALL of:

    - its ``(pid, create_time)`` matches a live ledger entry (PID reuse
      cannot forge this pair);
    - the entry's purpose is a reapable backend kind (serve/dashboard/
      gateway — never interactive processes);
    - the entry's recorded SPAWNER is provably dead (``spawner_is_dead``).

    Safe in ANY update context — the process itself declared its supervisor
    and that supervisor is gone. Holders not in the ledger fall through to
    later rungs and never disqualify identified ones. Never raises.
    """
    try:
        from hermes_cli.process_identity import (
            REAPABLE_PURPOSES,
            ledger_entries,
            spawner_is_dead,
        )

        entries = ledger_entries()
    except Exception:
        return []
    by_pid = {e.get("pid"): e for e in entries if isinstance(e.get("pid"), int)}
    roots: list[int] = []
    for pid, _name, _cmdline in matches:
        entry = by_pid.get(int(pid))
        if not entry:
            continue
        if entry.get("purpose") not in REAPABLE_PURPOSES:
            continue
        if spawner_is_dead(entry) is True:
            roots.append(int(pid))
    return roots


def _handoff_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs of Hermes ``serve``/``dashboard`` backends safe to reap during a
    GUI-updater hand-off, INCLUDING ones with a still-live parent.

    Complements ``_orphaned_desktop_backend_pids``, which returns ``None``
    (keep refusing) the moment ANY holder has a live parent. That produced a
    field incident: a Windows Desktop hand-off (``update --yes --gateway
    --force``) left a swarm of per-profile ``serve`` backends holding
    ``cryptography\\_rust.pyd``, several with a lingering parent (tearing-down
    Electron, or the launcher→worker chain mid-exit), so the orphan check
    disqualified the WHOLE set and the update hung for 12 minutes.

    The hand-off is the safe signal: with the update-incomplete marker claimed
    AND a ``--gateway`` run AND no live Desktop shim (``hermes.exe``), nothing
    legitimate supervises or respawns a ``serve`` backend from this venv (the
    Desktop tree-kills its backends and parks relaunch behind the marker,
    #50238). Any ``serve`` backend still holding the venv is a leak, live
    parent or not, and tree-reaping it is correct rather than a race.

    Guards: only Hermes backends (``hermes_cli.main`` + ``serve``/``dashboard``)
    qualify — a non-backend holder disqualifies the whole set → ``None``; the
    CALLER must have confirmed the hand-off gate above (outside it the stricter
    orphan-only path stands); psutil unavailable → ``None``. Returns backend
    root PIDs to tree-reap, or ``None`` to leave the decision to the caller's
    other rungs. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    roots: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Exited between scan and classification — nothing to reap.
            continue
        except Exception:
            pass
        if not _is_backend(argv.lower()):
            # A non-backend holder during a hand-off is unexpected; refuse the
            # whole set rather than reap something we cannot justify.
            return None
        roots.append(int(pid))

    return roots or None


def _stop_process_trees(
    pids: list[int] | list[tuple[int, int]],
) -> None:
    """Force-stop each PID with its full child tree (Windows).

    ``taskkill /T /F`` mirrors the Desktop's ``forceKillProcessTree`` and
    install.ps1's venv sweep: stopping only the parent can leave a managed
    ``.hermes-runtime`` interpreter child alive and holding the install open
    (#70026). Best effort; never raises.
    """
    from gateway.status import get_process_start_time
    from hermes_cli._subprocess_compat import pid_is_hermes, windows_hide_flags

    for entry in pids:
        if isinstance(entry, tuple):
            pid, expected_start_time = entry
        else:
            pid = int(entry)
            expected_start_time = get_process_start_time(pid)
        try:
            if expected_start_time is None:
                logger.debug(
                    "Skipping taskkill of PID %s: process identity unavailable",
                    pid,
                )
                continue
            if not pid_is_hermes(
                pid,
                expected_start_time=expected_start_time,
            ):
                logger.debug(
                    "Skipping taskkill of non-Hermes or changed PID %s",
                    pid,
                )
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _looks_like_desktop_control_plane(cmdline: str) -> bool:
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv.

    That is the Desktop control plane, not the messaging gateway (serve and
    dashboard host no platform adapters, #92091); do not feed this into
    ``looks_like_gateway_command_line``. Token-based via the parser-derived
    subcommand classifier — never substring (#90778/#91869: ``kanban
    --preserve-cache`` contains "serve", ``-m dashboard chat`` contains
    " dashboard"). An undeterminable subcommand is NOT a control plane.
    """
    if "hermes_cli.main" not in (cmdline or "").lower():
        return False
    return _hermes_holder_subcommand(cmdline) in ("serve", "dashboard")


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane.

    The updater must not steal gateway start in that case: Desktop owns
    start/stop via ``/api/gateway/*``. This is *not* proof messaging is
    already served — a live serve process is the control plane, and the
    gateway is a detached sibling (#76129 / #92091).

    Prefer the spawn ledger (owned identity). Fall back to the install-scoped
    venv-holder scan already used by the lock guard; an orphaned control-plane
    process (supervisor gone) does not count.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            if spawner_is_dead(entry) is False:
                return True
    except Exception as exc:
        logger.debug("Desktop-lifecycle ledger probe failed: %s", exc)

    try:
        import psutil
    except Exception:
        psutil = None

    try:
        holders = _m()._detect_venv_python_processes()
    except Exception as exc:
        logger.debug("Desktop-lifecycle holder scan failed: %s", exc)
        return False

    for pid, _name, cmdline in holders:
        if not _looks_like_desktop_control_plane(cmdline):
            continue
        if psutil is None:
            # Cannot prove orphanhood; a live this-install control plane is
            # enough to refuse stealing gateway start.
            return True
        try:
            proc = psutil.Process(int(pid))
            parent = proc.parent()
            if parent is None or not parent.is_running():
                continue
            if parent.create_time() > proc.create_time():
                continue
            return True
        except Exception:
            continue
    return False


def _stop_windows_gateway_service(
    name: str,
    *,
    expected_processes: tuple[tuple[int, float], ...] = (),
    expected_service_identity: tuple[int, float] | None = None,
    expected_gateway_identity: tuple[int, float] | None = None,
    timeout: float = 30.0,
) -> None:
    """Stop one verified Windows service and wait until SCM reports it down."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    if expected_service_identity is not None:
        try:
            current_status = str(service.status())
            current_service_pid = int(service.pid() or 0)
        except Exception as exc:
            raise RuntimeError(
                f"Windows service {name} SCM identity is unavailable before stop"
            ) from exc
        if current_status != "running":
            raise RuntimeError(
                f"Windows service {name} is not stably running before stop: {current_status}"
            )
        if current_service_pid != int(expected_service_identity[0]):
            raise RuntimeError(
                f"Windows service {name} SCM process identity changed before stop"
            )
    for label, identity in (
        ("service", expected_service_identity),
        ("gateway", expected_gateway_identity),
    ):
        if identity is None:
            continue
        pid, create_time = identity
        try:
            current = float(psutil.Process(int(pid)).create_time())
        except Exception as exc:
            raise RuntimeError(
                f"Windows {label} process identity is unavailable before stop"
            ) from exc
        if abs(current - float(create_time)) > 0.001:
            raise RuntimeError(
                f"Windows {label} process identity changed before stop"
            )
    if expected_service_identity is not None and expected_gateway_identity is not None:
        service_pid = int(expected_service_identity[0])
        gateway_pid = int(expected_gateway_identity[0])
        try:
            ancestor_pids = {
                int(parent.pid) for parent in psutil.Process(gateway_pid).parents()
            }
        except Exception as exc:
            raise RuntimeError(
                "Windows gateway ancestry is unavailable before service stop"
            ) from exc
        if service_pid not in ancestor_pids:
            raise RuntimeError(
                f"Windows gateway is no longer owned by service {name}"
            )
    result = subprocess.run(
        ["sc.exe", "stop", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "stopped":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe stop failed with {result.returncode}")

    def _original_process_is_alive(pid: int, create_time: float) -> bool:
        try:
            current = float(psutil.Process(pid).create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # A vanished process is clear.
            return False
        except Exception:
            # AccessDenied or any unknown probe failure stays fail-closed
            # because the venv may still be locked.
            return True
        return abs(current - create_time) <= 0.001

    alive = [
        pid
        for pid, create_time in expected_processes
        if _original_process_is_alive(pid, create_time)
    ]
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        service_stopped = service.status() == "stopped"
        alive = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if service_stopped and not alive:
            return
        _time.sleep(0.2)
    if service.status() == "stopped":
        # Return only if the original processes are gone too; a lingering
        # matching-identity process means venv mutation is unsafe — fail closed.
        alive_after_stop = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if alive_after_stop:
            raise RuntimeError(
                f"Windows service {name} stopped but its process tree is still alive: "
                f"{alive_after_stop}"
            )
        return
    # Timeout with the original descendants still alive — fail closed; venv mutation is unsafe.
    raise RuntimeError(
        f"Windows service {name} did not stop within {timeout:.0f}s; venv mutation unsafe."
    )


def _start_windows_gateway_service(name: str, *, timeout: float = 30.0) -> None:
    """Start one previously paused Windows service and verify it is running."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    result = subprocess.run(
        ["sc.exe", "start", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "running":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe start failed with {result.returncode}")
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if service.status() == "running":
            return
        _time.sleep(0.2)
    raise RuntimeError(f"Windows service {name} did not start within {timeout:.0f}s")


def _restore_windows_gateway_service(name: str, *, timeout: float = 60.0) -> None:
    """Restore a service after an uncertain stop, including STOP_PENDING."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        status = service.status()
        if status == "running":
            return
        if status == "stopped":
            _start_windows_gateway_service(name)
            return
        _time.sleep(0.2)
    raise RuntimeError(
        f"Windows service {name} did not reach a restorable state within {timeout:.0f}s"
    )


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
    """
    if not _m()._is_windows():
        return None

    try:
        from gateway.status import get_process_start_time, terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
            find_windows_gateway_services,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare Windows gateway pause for update: {exc}"
        ) from exc

    try:
        profile_process_list = find_profile_gateway_processes(strict=True)
        profile_processes = {proc.pid: proc for proc in profile_process_list}
    except Exception as exc:
        raise RuntimeError(
            f"Could not map Windows gateway PIDs to profiles: {exc}"
        ) from exc

    try:
        service_gateways = find_windows_gateway_services(
            profile_processes=profile_process_list
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not determine Windows gateway service ownership: {exc}"
        ) from exc

    service_gateway_pids = {int(service.gateway_pid) for service in service_gateways}
    try:
        running_pids = list(
            dict.fromkeys(
                [
                    *find_gateway_pids(all_profiles=True),
                    *sorted(profile_processes),
                    *sorted(service_gateway_pids),
                ]
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover Windows gateway PIDs before update: {exc}"
        ) from exc
    if not running_pids:
        # No gateway is running, but an installed autostart entry (Scheduled
        # Task / Startup-folder login item) is an explicit "I want a gateway"
        # signal. A gateway that died between updates (e.g. its spawning
        # terminal closed) would otherwise stay down until next login, since
        # the resume path only relaunches gateways that were running. Cold-start
        # one after the update; gateway-less users get nothing forced on them.
        #
        # Exception: Desktop owns this install's gateway lifecycle (live
        # supervised serve/dashboard); a vestigial autostart entry is not the
        # owner, and spawning ``gateway run`` beside Desktop races ports/state
        # (#76129). The skip is ownership, not liveness (#92091).
        try:
            if _desktop_owns_gateway_lifecycle():
                logger.debug(
                    "Skipping Windows gateway cold-start plan: "
                    "Desktop owns gateway lifecycle"
                )
                return None
        except Exception as exc:
            logger.debug(
                "Could not check Desktop gateway-lifecycle ownership before update: %s",
                exc,
            )
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
        return None

    profiles: dict[str, int] = {}
    mapped_pids = []
    socket_acks: list[dict] = []
    for pid in running_pids:
        if pid in service_gateway_pids:
            continue
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))
        # Socket-first pause (#92091 step 2): ask the gateway to drain and exit
        # itself. A positive ACK means it runs its own graceful restart path
        # (same drain as SIGUSR1/service restarts) and releases venv handles on
        # exit. No answer (older gateway, no socket) → the marker poll /
        # force-kill ladder below behaves exactly as before.
        try:
            from gateway.control_socket import pause_gateway_for_update

            ack = pause_gateway_for_update(Path(proc.path))
            if ack and (ack.get("pausing") or ack.get("already_stopping")):
                socket_acks.append(ack)
        except Exception as exc:
            logger.debug(
                "Socket pause unavailable for gateway %s: %s", pid, exc
            )

    # Resolve each mapped worker's venv-side launcher BEFORE draining: a
    # gracefully drained worker is gone when the wait returns, and a dead
    # pid's parent cannot be recovered (psutil raises NoSuchProcess). The
    # snapshot is stopped after the drain alongside the survivors.
    #
    # The drain targets the PID-file writer (uv-side worker); its parent is
    # usually the venv ``python.exe`` launcher, which keeps venv ``.pyd``
    # files mapped and is what ``_detect_venv_python_processes()`` reports.
    # Left alive, it trips the venv-holder guard though the gateway is stopped.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    if socket_acks:
        # A socket-paused gateway drains its ACTIVE TURN before exiting; honor
        # the budget it declared (plus teardown grace) so a mid-turn gateway
        # isn't force-killed by a too-short wait.
        try:
            declared = max(
                float(a.get("drain_timeout") or 0.0) for a in socket_acks
            )
            drain_timeout = max(drain_timeout, declared + 10.0)
        except Exception:
            pass
        print(
            f"  → {len(socket_acks)} gateway(s) ACKed socket pause; "
            f"waiting up to {int(drain_timeout)}s for graceful exit"
        )
    survivors = _m()._wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [
        pid
        for pid in running_pids
        if pid not in profile_processes and pid not in service_gateway_pids
    ]

    # Snapshot each unmapped gateway's argv *before* force-killing it so
    # ``_resume_windows_gateways_after_update`` can replay it. Unmapped = no
    # profile→PID-file mapping (e.g. a Scheduled Task running ``pythonw.exe -m
    # hermes_cli.main gateway run``); without this they were never restarted (#50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Stop drain survivors, unmapped gateways, and the pre-drain launcher
    # snapshot. ``terminate_pid(force=True)`` is a tree kill; a launcher that
    # already exited with its worker raises ProcessLookupError and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        try:
            pid_int = int(pid)
            terminate_pid(
                pid_int,
                force=True,
                expected_start_time=get_process_start_time(pid_int),
            )
            force_killed.append(pid_int)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    token = {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

    # Stop SCM-supervised gateways only after every fallible step for ordinary
    # gateways is done; from here on any error restores attempted services and
    # already-paused ordinary gateways before aborting.
    paused_services = []
    current_service_name = None
    try:
        for service in service_gateways:
            current_service_name = str(service.name)
            _stop_windows_gateway_service(
                current_service_name,
                expected_processes=tuple(
                    getattr(service, "descendant_identities", ())
                ),
                expected_service_identity=(
                    int(service.service_pid),
                    float(service.service_create_time),
                ),
                expected_gateway_identity=(
                    int(service.gateway_pid),
                    float(service.gateway_create_time),
                ),
            )
            paused_services.append(current_service_name)
            current_service_name = None
        if paused_services:
            token["services"] = paused_services
            token["expected_services"] = list(paused_services)
            token["restarted_services"] = []
            token["service_profiles"] = {
                str(service.name): str(service.profile)
                for service in service_gateways
                if str(service.name) in paused_services
            }
            print(
                "  ✓ Paused Windows gateway service(s): "
                + ", ".join(paused_services)
            )
        return token
    except Exception as exc:
        restore_names = []
        if current_service_name:
            restore_names.append(current_service_name)
        restore_names.extend(reversed(paused_services))
        rollback_failures = []
        for service_name in dict.fromkeys(restore_names):
            try:
                _restore_windows_gateway_service(service_name)
            except Exception as restore_exc:
                rollback_failures.append(f"{service_name}: {restore_exc}")
        if profiles or unmapped:
            try:
                _resume_windows_gateways_after_update(token)
            except Exception as restore_exc:
                rollback_failures.append(f"ordinary gateways: {restore_exc}")
        failed_service = current_service_name or "unknown"
        detail = f"Could not stop Windows gateway service {failed_service}: {exc}"
        if rollback_failures:
            detail += "; rollback failures: " + "; ".join(rollback_failures)
        raise RuntimeError(detail) from exc


def _cold_start_windows_gateway_after_update() -> bool:
    """Start a fresh detached gateway after update when one is installed but down.

    Called from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running at update start,
    but an autostart entry is installed. Unlike the relaunch paths (watch an
    old PID, respawn on exit) this is a direct spawn via the same
    hidden-console + breakaway path as ``hermes gateway start``
    (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't duplicate.

    A successful ``Popen`` only proves the process was created, not that it
    survived (a job object denying breakaway kills it before it logs, #84185),
    so the success line is gated on the same post-spawn liveness poll every
    other ``_spawn_detached`` caller uses (``_report_gateway_start``).
    """
    if not _m()._is_windows():
        return True
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway cold-start helpers: {exc}"
        ) from exc

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return True
    except Exception as exc:
        raise RuntimeError(
            f"Could not re-check gateway liveness before cold-start: {exc}"
        ) from exc

    try:
        if _desktop_owns_gateway_lifecycle():
            logger.debug(
                "Skipping Windows gateway cold-start: Desktop owns gateway lifecycle"
            )
            return True
    except Exception as exc:
        raise RuntimeError(
            "Could not re-check Desktop gateway-lifecycle ownership before cold-start: "
            f"{exc}"
        ) from exc

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        raise RuntimeError(f"Could not cold-start Windows gateway after update: {exc}") from exc

    if not pid:
        raise RuntimeError("Windows gateway cold-start did not return a process ID")
    ready_pids = gateway_windows._wait_for_gateway_ready()
    if not ready_pids:
        raise RuntimeError(
            f"Windows gateway cold-start PID {pid} did not become ready"
        )
    print()
    print(
        "✓ Gateway started via cold-start after update "
        f"(PID: {', '.join(map(str, ready_pids))})"
    )
    # Persist the PIDs this ✓ vouched for so a death AFTER the updater exits
    # (parent Job Object teardown, #91675) is reported by the next CLI
    # invocation instead of staying silent. Best-effort.
    try:
        gateway_windows._write_start_attestation(
            ready_pids, "cold-start after update"
        )
    except Exception:
        pass
    return True


def _systemctl(cmd: list, *, timeout: float):
    """Run a systemctl (or sudo systemctl) invocation, capturing utf-8 text with a timeout."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def _systemctl_reset_and_restart(manage_cmd: list, svc_name: str):
    """``reset-failed`` then ``restart`` a unit. Always clear failed state first: if
    systemd's own auto-restart attempts already parked the unit in a failed state,
    a plain ``restart`` can wedge against the RestartSec backoff and leave it dead."""
    _systemctl(manage_cmd + ["reset-failed", svc_name], timeout=10)
    return _systemctl(manage_cmd + ["restart", svc_name], timeout=15)


def _for_each_systemd_gateway_unit(
    list_units_stdout: str,
    *,
    process_unit,
    on_unit_timeout,
) -> None:
    """Process each ``hermes-gateway*.service``/``hermes-serve*.service`` unit
    from ``systemctl list-units``.

    ``subprocess.TimeoutExpired`` raised by ``process_unit`` is isolated to
    that unit via ``on_unit_timeout`` so one wedged systemctl call cannot
    abort the rest of the fleet (#68523).
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service"):
            continue
        # list-units is already pattern-filtered, but keep the name gate so a
        # stray line cannot enter the restart path. Require the exact base unit
        # or hyphenated profile family: ``startswith("hermes-serve")`` would
        # also accept the unrelated ``hermes-server.service`` (#83595).
        if not (
            unit == "hermes-gateway.service"
            or unit.startswith("hermes-gateway-")
            or unit == "hermes-serve.service"
            or unit.startswith("hermes-serve-")
        ):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)

def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` units run ``gateway/run.py`` (the SIGUSR1
    handler). ``hermes-serve*`` units (#83438) don't: SIGUSR1 would just
    terminate them and burn the full drain budget, so they go straight to the
    blunt ``systemctl restart`` path.

    Same strict exact/hyphenated shape as the unit-name gate in
    ``_for_each_systemd_gateway_unit``, so a near-prefix unit like
    ``hermes-gatewayd`` can't be sent a SIGUSR1 it doesn't handle.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    from hermes_cli.gateway import is_macos

    if not failed_units:
        return
    # Preserve discovery order while de-duplicating.
    seen = set()
    ordered = []
    for name in failed_units:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    if is_macos():
        # A launchd label lands here when launchd was not supervising a live
        # process after the restart (#88848) — very likely deregistered, which
        # `launchctl kickstart` cannot revive.
        print("  Listed services may be deregistered from launchd, or still")
        print("  running pre-update code (mixed sys.modules). Recover with:")
        print("    hermes gateway status")
        print("    launchctl list | grep <label>")
        print("    launchctl bootstrap gui/$(id -u) "
              "~/Library/LaunchAgents/<label>.plist")
        return
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    if any(not name.startswith("ai.hermes.") for name in ordered):
        print("    systemctl --user restart <unit>   # user-scope")
        print("    sudo systemctl restart <unit>     # system-scope")
    if any(name.startswith("ai.hermes.") for name in ordered):
        print("    launchctl kickstart -k gui/$UID/<label>   # macOS (or user/$UID)")


def _restart_launchd_gateway_after_update(
    *, supervision_verify: bool = True
) -> tuple[list, list]:
    """Restart the invoking profile's launchd gateway after an update.

    No ``launchctl list``-based classification (#74973): a *booted-out* job
    (plist present, definition deregistered — crashed helper, manual bootout,
    failed prior update) fails that check, and ``launchctl list`` is
    session-scoped and can exit non-zero while the job is alive, so gating on
    it silently skipped the restart while still printing "Update complete!".
    When the plist exists, ``launchd_restart()`` always runs — it drains a
    live PID, kickstarts with ``-k``, and owns the bootout/bootstrap/kickstart
    ladder for the unloaded state. Every failure path is loud and names the
    manual recovery command.

    Returns ``(restarted_labels, failed_labels)``. With ``supervision_verify``
    (the update path), success additionally requires launchd reporting a fresh
    supervised PID (#88848 — "the call returned" is not "supervised").
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        wait_for_launchd_gateway_supervision,
    )

    current_label = get_launchd_label()
    try:
        if not get_launchd_plist_path().exists():
            return [], []  # not a launchd install — nothing to do or warn
        try:
            launchd_restart()
        except subprocess.CalledProcessError as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            print(
                f"  ⚠ Gateway restart failed: {stderr}\n"
                "    The gateway may be DOWN on pre-update code. "
                "Recover manually: hermes gateway restart"
            )
            return [], [current_label]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # A plist exists, so a gateway is SUPPOSED to be supervised; a broken/
        # wedged launchctl is not proof nothing needs restarting (#74973's
        # second silent variant). Count it and tell the operator.
        print(
            "  ⚠ Could not restart the gateway "
            f"({e.__class__.__name__}: {e}).\n"
            "    Recover manually: hermes gateway restart"
        )
        return [], [current_label]

    if not supervision_verify:
        return [current_label], []

    # launchd_restart() returning only means "restart REQUESTED" — both the
    # self-restart branch and a plist reload are asynchronous. A helper that
    # dies before its first bootstrap (#88848), or a bootstrap that exits 0
    # without registering (seen on macOS 26.6.1), would otherwise reach "Update
    # complete!" unsupervised. Verified domain-agnostically: a domain locate
    # fails on macOS-26 hosts whose per-user domains reject service management.
    if wait_for_launchd_gateway_supervision(label=current_label):
        return [current_label], []
    print(
        f"  ✗ {current_label} restarted but launchd is not supervising it.\n"
        "    Check logs, then: hermes gateway restart"
    )
    return [], [current_label]


def _restart_macos_launchd_gateways(
    restarted_services: list,
    failed_or_stale_units: list,
    drain_budget: float,
) -> None:
    """Restart every launchd-managed gateway after an update (macOS).

    The git pull is shared across profiles, so every ``ai.hermes.gateway*``
    LaunchAgent must reload it; restarting only the invoking profile leaves
    siblings on pre-update ``sys.modules`` (#41403). Parity with systemd.

    The invoking profile keeps ``launchd_restart()`` (self-restart request →
    drain → kickstart). Siblings get the same drain-first sequence with their
    launchd domain resolved per label (``gui/<uid>`` vs ``user/<uid>``) so
    none is kickstarted in the wrong domain. ``subprocess.TimeoutExpired`` is
    isolated per label so one wedged launchctl call cannot strand the fleet
    on old code (#68523).
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        launchd_gateway_labels_for_install,
        _graceful_restart_via_sigusr1,
        _launchd_kickstart,
        _locate_launchd_gateway_service,
        _wait_for_launchd_service_pid,
    )

    # --- Current profile: unchanged single-service path ---------------------
    _restarted, _failed = _restart_launchd_gateway_after_update(
        supervision_verify=True
    )
    restarted_services.extend(_restarted)
    failed_or_stale_units.extend(_failed)
    current_label = get_launchd_label()

    # --- Sibling profiles ---------------------------------------------------
    for label in launchd_gateway_labels_for_install():
        if label == current_label:
            continue
        try:
            # Locate = liveness + domain in one probe; the kickstart and
            # fresh-PID checks below reuse that domain so a sibling is never
            # probed in one gui/user domain and restarted in another.
            domain, old_pid = _locate_launchd_gateway_service(label)
            if domain is None:
                # Installed but not bootstrapped (stopped/uninstalled
                # mid-way) — nothing is running old code here.
                continue
            graceful_ok = False
            if old_pid is not None and old_pid > 0:
                print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
                graceful_ok = _graceful_restart_via_sigusr1(
                    old_pid, drain_timeout=drain_budget
                )
            if graceful_ok and _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=10.0, domain=domain
            ):
                # Unconditional KeepAlive already respawned it on the new
                # code — a hard kickstart now would kill the fresh process.
                restarted_services.append(label)
                continue
            try:
                _launchd_kickstart(label, domain)
            except subprocess.CalledProcessError as e:
                stderr = (getattr(e, "stderr", "") or "").strip()
                failed_or_stale_units.append(label)
                print(
                    f"  ⚠ Failed to restart {label}: {stderr}\n"
                    f"    Recover manually: launchctl kickstart -k {domain}/{label}"
                )
                continue
            if _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=15.0, domain=domain
            ):
                restarted_services.append(label)
            else:
                failed_or_stale_units.append(label)
                print(
                    f"  ✗ {label} failed to come back after restart.\n"
                    f"    Check logs, then: launchctl kickstart -k {domain}/{label}"
                )
        except subprocess.TimeoutExpired:
            failed_or_stale_units.append(label)
            print(
                f"  ⚠ launchctl timed out restarting {label}; "
                "continuing with remaining gateways"
            )


def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    ``None`` when undeterminable — notably when ``hermes_cli.gateway`` no
    longer imports, one of the ways the restart phase aborts (the checkout was
    replaced under a process holding old modules). The caller treats ``None``
    and a non-empty list as "assume stale"; only a positive empty result proves
    nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids

        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None


_FRESH_RESTART_SUPERVISORS = frozenset({"systemd", "launchd", "service", "s6"})


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label to a profile.

    Profile names must not be matched as substrings: ``foo`` must not claim
    that ``hermes-gateway-foobar.service`` was already restarted.  These are
    the service/label shapes produced by the existing systemd, launchd, and
    s6 lifecycle implementations.
    """
    name = str(service).removesuffix(".service")
    if profile == "default":
        return name in {
            "hermes-gateway",
            "ai.hermes.gateway",
            "gateway",
            "gateway-default",
        }
    return name in {
        f"hermes-gateway-{profile}",
        f"ai.hermes.gateway-{profile}",
        f"gateway-{profile}",
    }


# Abort recovery lives in its own bounded module (review on #96235). Re-exported
# here because `hermes_cli.main` and the update flow below address these names
# through `update_cmd`.
from hermes_cli.update_abort_recovery import (  # noqa: E402
    _abort_recovery_is_complete,
    _qualified_serve_skips,
    _recover_gateway_restart_after_abort,
    _serve_unit_recovery_available,
    _surviving_pre_update_serve_runtimes,
    _warn_stale_serve_runtimes,
)


def _gateway_recovery_partition(
    plan, *, skip_profiles: set[str] | None = None
) -> tuple[dict[str, str], list[dict]]:
    """Partition pre-update runtimes into fresh-restart candidates and skips.

    Uses only the inventory captured before the checkout changed: re-importing
    ``hermes_cli.gateway`` in the failing interpreter is exactly what can raise
    the original ``ImportError``.

    Returns ``(candidates, skipped)``: ``candidates`` maps profile → supervisor
    for supervised gateway runtimes the fresh process may restart; ``skipped``
    lists every other inventoried runtime, each with an explicit reason, so
    nothing from the spawn ledger vanishes silently (manual gateways have no
    relaunch authority; serve/dashboard runtimes have no per-profile command).

    A ``skipped`` serve/dashboard entry does NOT mean unrecoverable: the fresh
    child runs a separate ``hermes-serve*`` systemd pass enumerating units from
    systemd, because the ledger collector cannot classify a systemd-launched
    ``hermes serve`` (no spawner ⇒ ``manual-serve``). Leftovers are caught by
    :func:`_surviving_pre_update_serve_runtimes` (#92145).
    """
    skip_profiles = skip_profiles or set()
    candidates: dict[str, str] = {}
    skipped: list[dict] = []
    try:
        for runtime in getattr(plan, "runtimes", ()) or ():
            kind = getattr(runtime, "kind", None)
            profile = getattr(runtime, "profile", None)
            supervisor = getattr(runtime, "supervisor", None)
            if not isinstance(profile, str) or not profile:
                continue
            if kind == "gateway":
                if profile in skip_profiles:
                    continue
                if supervisor in _FRESH_RESTART_SUPERVISORS:
                    candidates.setdefault(profile, str(supervisor))
                else:
                    skipped.append(
                        {
                            "profile": profile,
                            "kind": "gateway",
                            "supervisor": str(supervisor),
                            "reason": (
                                "manual gateway has no supervisor relaunch"
                                " authority; left running for explicit operator"
                                " restart"
                            ),
                        }
                    )
            elif kind in ("serve", "dashboard"):
                if supervisor == "desktop":
                    reason = (
                        "desktop app owns and respawns this serve backend;"
                        " the recovery pass must not restart it out from under"
                        " its supervisor"
                    )
                else:
                    # NOT a claim that no supervisor exists: a systemd-launched
                    # `hermes serve` sets neither HERMES_SPAWN nor
                    # HERMES_PARENT_PID, so it reads as "manual-serve".
                    # Unit-backed serves are recovered by the fresh child's
                    # systemd pass (enumerated from systemd); survivors are
                    # reported by _surviving_pre_update_serve_runtimes (#92145).
                    reason = (
                        "no per-profile relaunch command reaches a serve/"
                        "dashboard runtime; recovered by the fresh systemd"
                        " unit pass when it owns a hermes-serve* unit, else"
                        " left running for explicit operator restart"
                    )
                skipped.append(
                    {
                        "profile": profile,
                        "kind": str(kind),
                        "supervisor": str(supervisor),
                        "reason": reason,
                    }
                )
    except Exception as exc:
        logger.debug("Could not prepare fresh gateway restart profiles: %s", exc)
    return candidates, skipped


def _gateway_restart_recovery_profiles(
    plan, *, skip_profiles: set[str] | None = None
) -> list[str]:
    """Supervised gateway profiles a fresh process may restart (kept: re-exported by ``hermes_cli.main``)."""
    candidates, _ = _gateway_recovery_partition(plan, skip_profiles=skip_profiles)
    return sorted(candidates)


def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    #78574: the phase was wrapped in a blanket ``except Exception`` logged at
    debug level, so an early failure (e.g. importing ``hermes_cli.gateway``
    from the fresh checkout) erased every drain/restart line; the update
    printed "Update complete!" and exited 0 while the gateway kept serving
    pre-update modules and died on the next turn with an ImportError.
    """
    print()
    print(f"⚠ Update incomplete — gateway auto-restart failed: {exc}")
    if pids:
        listed = ", ".join(str(pid) for pid in pids)
        print(f"  Gateway process(es) still running pre-update code: {listed}")
    else:
        print("  Any gateway still running is serving pre-update code")
        print("  (mixed sys.modules) against the updated checkout.")
    print("  Restart it manually, then verify:")
    print("    hermes gateway restart")
    print("    hermes gateway status")

def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` +
    ``gateway.vbs``) are written once at install time, so installs predating
    the hidden-console rework (aa2ae36c3f) kept launching via ``pythonw.exe``:
    conhost flashes (#54220/#56747) and, since #70344, a startup death with
    ``RuntimeError: sys.stderr is None`` (#71671).

    The task's /TR points at a stable script path, so rewriting in place
    retargets it without schtasks (no UAC). ``_write_task_script`` is
    idempotent — a no-op for modern installs. Best-effort: never fails the update.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")
    except Exception as exc:
        logger.debug("Could not refresh Windows gateway launchers after update: %s", exc)

def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Sync the installer's bootstrap-cache scripts from the fresh checkout.

    The Desktop GUI updater (``hermes-setup.exe``) runs
    ``$HERMES_HOME/bootstrap-cache/install-<ref>.ps1`` (or ``.sh``) for its
    repair/bootstrap stages. Installers built before the #67193 cache-refresh
    fix NEVER re-download a cached branch-ref script, so a stale
    ``install-main.ps1`` runs months-old code forever (the 2026-08-09
    incident: a cached script lacked the #81327 process-tree sweep and died on
    ``Access denied``). The binary has no self-update path.

    Overwriting the cached script for *branch* with the freshly pulled
    ``scripts/install.ps1`` / ``scripts/install.sh`` on every update turns
    that unconditional reuse into a feature. Post-#67193 installers
    re-download anyway, so for them this is a harmless pre-seed.

    Scope guards, mirroring ``install_script.rs``:

    - Only the cache key for the update-target *branch* is rewritten
      (``sanitize_ref``: non ``[A-Za-z0-9._-]`` chars become ``_``, so
      ``bb/gui`` → ``install-bb_gui.ps1``); sibling refs cache other
      branches' scripts and must not be clobbered.
    - Commit-SHA pins are immutable and never touched. ``is_valid_commit()``
      accepts **7–40** hex chars, so abbreviated pins count too; the sanitized
      *branch* must also not itself look like a pin (defense in depth).

    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format
    (#67193). Best-effort: a failed refresh must never fail the update.
    """
    try:
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): 7-40 hex chars is an
        # immutable commit pin — abbreviated SHAs included. Never rewrite.
        if _re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            if not src.is_file():
                continue
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                # Match the installer's cache format: PowerShell needs the
                # UTF-8 BOM or localized/em-dash text mis-decodes (#67193).
                data = b"\xef\xbb\xbf" + data
            if cached.read_bytes() == data:
                continue  # already current
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print(
                "  ✓ Refreshed installer bootstrap-cache script(s): "
                + ", ".join(sorted(refreshed))
            )
    except Exception as exc:
        logger.debug("Could not refresh bootstrap-cache scripts after update: %s", exc)

def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    if not token or not token.get("resume_needed"):
        return
    if not _m()._is_windows():
        token["resume_needed"] = False
        return

    # Regenerate the persisted launcher scripts before respawning anything,
    # so a legacy pythonw-era Scheduled Task / Startup entry comes back on
    # current hidden-console design at the next login too.
    _m()._refresh_windows_gateway_launchers()

    services = list(token.get("services") or [])
    token.setdefault("expected_services", list(services))
    verified_restarts = list(token.get("restarted_services") or [])
    restarted_services = []
    failed_services = []
    for service_name in services:
        try:
            _start_windows_gateway_service(str(service_name))
            restarted_services.append(str(service_name))
            if str(service_name) not in verified_restarts:
                verified_restarts.append(str(service_name))
        except Exception as exc:
            logger.warning(
                "Could not restart Windows gateway service %s after update: %s",
                service_name,
                exc,
            )
            print(f"  ⚠ Could not restart Windows gateway service: {service_name}")
            failed_services.append(str(service_name))

    if failed_services:
        token["services"] = failed_services
        token["restarted_services"] = verified_restarts
        raise RuntimeError(
            "Could not restart Windows gateway service(s): "
            + ", ".join(failed_services)
        )
    token["services"] = []
    token["restarted_services"] = verified_restarts
    if restarted_services:
        print()
        print(
            "  ✓ Restarted Windows gateway service(s): "
            + ", ".join(restarted_services)
        )

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            if not _m()._cold_start_windows_gateway_after_update():
                raise RuntimeError("Windows gateway cold-start was not verified")
            token["cold_start_if_installed"] = False
        token["resume_needed"] = False
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway restart helper: {exc}"
        ) from exc

    relaunched = []
    failed_profiles = {}
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
            else:
                failed_profiles[str(profile)] = int(old_pid)
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )
            failed_profiles[str(profile)] = int(old_pid)

    # Surface the outcome on the token (#91277 plan-vs-execution
    # reconciliation): the git-update fleet reconciliation cross-checks every
    # planned runtime against relaunched_profiles etc., which this Windows
    # pause/resume never fed, so a correctly relaunched gateway was reported
    # "unaccounted" (loud warning + exit 1). The caller merges this into the
    # shared list. A profile whose relaunch failed is deliberately left off so
    # it still surfaces as unaccounted (Windows has no watcher to recover it).
    token["relaunched_profiles"] = relaunched

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
    unmapped_relaunched = 0
    failed_unmapped = []
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            failed_unmapped.append(entry)
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
            else:
                failed_unmapped.append(entry)
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )
            failed_unmapped.append(entry)

    token["profiles"] = failed_profiles
    token["unmapped"] = failed_unmapped
    if failed_profiles or failed_unmapped:
        raise RuntimeError("Could not restart every paused Windows gateway")

    # A truthy launch result only proves the detached watcher was created, not
    # that the respawned gateway survived: a parent Job Object denying
    # CREATE_BREAKAWAY_FROM_JOB kills it on updater teardown before it logs
    # anything, yet "✓ Restarting" was printed (#48820). Verify a stable
    # gateway exists with the same provisional-hit + confirmation-window poll
    # every other spawn path uses (#91675); all_profiles=True because the
    # resume covers the fleet.
    if relaunched or unmapped_relaunched:
        try:
            from hermes_cli import gateway_windows
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Windows gateway liveness helpers: {exc}"
            ) from exc
        ready_pids = gateway_windows._wait_for_gateway_ready(
            timeout_s=30.0, all_profiles=True
        )
        if not ready_pids:
            token["profiles"] = dict(profiles)
            token["unmapped"] = list(unmapped)
            print()
            print(
                "  ⚠ Windows gateway restart could not be verified — no stable "
                "gateway process appeared after relaunch."
            )
            print(
                "    (The respawned gateway may have been killed by a parent "
                "Job Object during updater teardown, #48820.)"
            )
            print("    Recover with: hermes gateway restart")
            raise RuntimeError(
                "Windows gateway relaunch after update was not verified alive"
            )
        # Persist the PIDs this ✓ vouches for so a death AFTER the updater
        # exits (parent Job Object teardown, #91675) is reported by the next
        # CLI invocation instead of staying silent. Best-effort.
        try:
            gateway_windows._write_start_attestation(
                ready_pids, "post-update relaunch"
            )
        except Exception:
            pass

    token["resume_needed"] = False

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )

def _git_is_trampoline(git_cmd: list) -> bool:
    """Whether *git_cmd* resolves to a Git-for-Windows trampoline launcher.

    Git for Windows ships ~46KB shims (``bin\\git.exe``, ``cmd\\git.exe``) that
    re-exec ``mingw64\\libexec\\git-core\\git.exe``. When the shim cannot find
    git-core, every git call dies with the launcher's guard message — a broken
    PATH entry, not a network/filesystem problem (#87876). Never raises;
    unknown states report False so a probe failure can't block an update.
    """
    try:
        result = subprocess.run(
            git_cmd + ["--version"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=15,
        )
    except Exception:
        return False
    output = ((result.stdout or "") + (result.stderr or "")).lower()
    return "fork bomb" in output


def _portable_git_candidates() -> list:
    """PortableGit candidate paths: shared root first, then profile home.

    The Hermes-managed PortableGit tree lives under the SHARED root
    (``<root>/git/...``), not the profile-scoped HERMES_HOME
    (``<root>/profiles/<name>``), so a profile-scoped ``hermes update`` must
    look there (monerostar review, #87876). The profile-home candidate is
    kept as a fallback for custom layouts that place it there.
    """
    candidates = []
    try:
        for root in (get_default_hermes_root(), Path(get_hermes_home())):
            candidates.append(
                root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
            )
    except Exception:
        pass
    return candidates


def _locate_real_git() -> Optional[Path]:
    """Find a real Git-for-Windows binary that is not a broken trampoline.

    The ~46KB ``bin\\git.exe`` / ``cmd\\git.exe`` shims fail to re-exec
    git-core while ``mingw64\\libexec\\git-core\\git.exe`` (≈4.4MB) works
    directly (#87876). Check standard Git for Windows locations plus the
    Hermes-managed PortableGit; accept the first candidate that runs without
    the trampoline guard. None when nothing suits — callers keep the broken
    command and let the fetch-failure ZIP fallback handle it.
    """
    candidates = [
        Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe"),
        Path(r"C:\Program Files (x86)\Git\mingw64\libexec\git-core\git.exe"),
    ] + _portable_git_candidates()
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=15,
            )
        except Exception:
            continue
        output = ((result.stdout or "") + (result.stderr or "")).lower()
        if "fork bomb" in output:
            continue
        return candidate
    return None


def _ensure_non_trampoline_git(git_cmd: list) -> list:
    """Swap a broken Git-for-Windows trampoline for a real git binary.

    Runs right after the git command is built. If ``git`` is a broken
    trampoline, rebuild the command around the real binary so fetch/pull/
    checkout keep working instead of degrading to the ZIP fallback; if none is
    found, leave the command untouched (the fetch-failure handler falls back to
    ZIP on Windows). No-op off Windows and when git is healthy.
    """
    if sys.platform != "win32":
        return git_cmd
    if not _git_is_trampoline(git_cmd):
        return git_cmd
    real_git = _locate_real_git()
    if real_git is None:
        print(
            "⚠ Detected a broken git trampoline and could not locate a real "
            "git binary — the update will fall back to the ZIP path."
        )
        return git_cmd
    print(
        f"⚠ Detected a broken git trampoline; switching to real git at "
        f"{real_git}"
    )
    return [str(real_git)] + list(git_cmd[1:])


def _discard_lockfile_churn(git_cmd, repo_root):
    """Restore tracked ``package-lock.json`` files that npm dirtied locally.

    npm rewrites lockfiles non-deterministically at install/build time. On a
    managed install those diffs are never intentional, so we discard them so
    ``hermes update`` sees a clean tree instead of autostashing every run.
    Best-effort; only ever touches files named ``package-lock.json``.
    """
    try:
        diff = _git_run(git_cmd, ["diff", "--name-only"], repo_root)
        if diff.returncode != 0:
            return
        dirty_package_dirs = {
            Path(line.strip()).parent
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package.json")
        }
        dirty = [
            line.strip()
            for line in diff.stdout.splitlines()
            if line.strip().endswith("package-lock.json")
            and Path(line.strip()).parent not in dirty_package_dirs
        ]
        if not dirty:
            return
        _git_run(git_cmd, ["checkout", "--", *dirty], repo_root)
        print(f"→ Discarded npm lockfile churn ({len(dirty)} file(s))")
    except Exception:
        # Never let lockfile cleanup block an update.
        pass

def _normalize_managed_eol(git_cmd, repo_root):
    """Take a managed checkout off ``core.autocrlf=true`` without leaving it dirty.

    Git for Windows ships ``core.autocrlf=true`` system-wide, which turns this
    repo's LF files CRLF in the working tree and breaks ``git checkout`` on
    update ("Your local changes would be overwritten"); ``install.ps1`` pins
    ``core.autocrlf=false`` on the managed clone (#67730). Older checkouts never
    got the pin and the bootstrap installer reuses its build-pinned
    ``install.ps1`` forever, so ``hermes update`` is the only path that can fix them.

    The pin and the cleanup are one operation: under ``autocrlf=true`` a CRLF
    tree reads clean, so pinning alone would expose every text file as
    modified and hand the update a whole-tree autostash. The pin is written
    only after the tree is verified clean under it; a checkout we cannot fully
    normalize is left as it was. Best-effort: never blocks an update.
    """
    # -c, not config: evaluate the tree as it WOULD look pinned, without
    # persisting anything we might not be able to follow through on.
    probe = git_cmd + ["-c", "core.autocrlf=false"]

    def _dirty(*extra):
        out = subprocess.run(
            probe + ["diff", "-z", "--name-only", *extra],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return None
        return {p for p in out.stdout.split("\0") if p}

    def _real_dirty():
        # Files with a *content* change once CRLF differences are ignored.
        # ``diff --name-only --ignore-cr-at-eol`` still LISTS CR-only files
        # (names come from blob/stat differences before the CR filter), so use
        # ``--numstat``, which honors the filter: a CR-only file produces no
        # record. Parse the paths out of numstat.
        out = subprocess.run(
            probe + ["-c", "core.quotepath=false",
                     "diff", "--numstat", "--ignore-cr-at-eol"],
            cwd=repo_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return None
        paths = set()
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            # Format: "<added>\t<deleted>\t<path>". Rename detection is off in
            # plain diff, so there is exactly one path field per record.
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[2]:
                paths.add(parts[2])
        return paths

    def _eol_only():
        all_dirty, real_dirty = _dirty(), _real_dirty()
        if all_dirty is None or real_dirty is None:
            return None
        return all_dirty - real_dirty

    try:
        effective = _git_run(git_cmd, ["config", "--get", "core.autocrlf"], repo_root)
        # Only "true" rewrites LF to CRLF on checkout. Unset, false, and input
        # all leave the working tree alone, so there is nothing to repair.
        if effective.stdout.strip().lower() != "true":
            return

        eol_only = _eol_only()
        if eol_only is None:
            return
        if eol_only:
            # Pathspec over stdin, not argv: a fully renormalized checkout is
            # thousands of paths, well past the Windows command-line limit.
            subprocess.run(
                probe
                + ["checkout", "--pathspec-from-file=-", "--pathspec-file-nul", "--"],
                cwd=repo_root,
                input="\0".join(sorted(eol_only)),
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False,
            )
            if _eol_only():
                # Still dirty — persisting the pin here would only surface churn
                # we failed to clear. Leave the checkout as we found it.
                return
            print(f"→ Normalized line-ending churn ({len(eol_only)} file(s))")

        subprocess.run(
            git_cmd + ["config", "core.autocrlf", "false"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except Exception:
        # Never let line-ending cleanup block an update.
        pass


def _desktop_app_present(desktop_dir: Path) -> bool:
    """Return whether a packaged or source Desktop build exists."""
    return (
        _m()._desktop_packaged_executable(desktop_dir) is not None
        or _m()._desktop_dist_exists(desktop_dir)
    )


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool
) -> bool:
    """Rebuild an installed Desktop app when its source or artifact changed.

    Returns ``False`` only when a rebuild was attempted and failed, so the
    caller can withhold ``✓ Update complete!`` and (in gateway mode) write
    a failing ``.update_exit_code`` (#88251). Every other outcome — nothing
    to rebuild, up to date, build succeeded, Desktop never installed —
    returns ``True``.
    """
    # The release tree is ignored by git and can disappear during an update.
    # Its pre-update presence is enough to restore it; do not make people who
    # have never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app
    ):
        return True

    print("→ Checking if desktop app needs rebuilding...")
    # Consult the content-hash stamp IN-PROCESS first: the spawned
    # `hermes desktop --build-only` re-imports the whole CLI stack (~1-3 s)
    # just to reach the same _m()._desktop_build_needed check. The update path
    # never passes --source, so mirror source_mode=False. Any pre-check error
    # falls through to the subprocess.
    skip_desktop_build = False
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False
        )
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return True

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the (very loud) Electron/vite build output into update.log. On
    # a nonzero exit, retry once (covers a still-settling rebuild window), then
    # surface the captured tail so the failure is debuggable.
    #
    # Put the Hermes-managed Node on PATH: inside the desktop updater chain
    # (Desktop → hermes-setup → hermes update) shell PATH customizations are
    # lost, so a bare-PATH child fails with `node: not found` before cmd_gui
    # can self-heal.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
    )
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
        )
    if build_result.returncode != 0:
        print("  ⚠ Desktop build failed (run `hermes desktop` to retry)")
        tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
        if tail:
            print(tail)
        from hermes_constants import display_hermes_home as _dhh

        print(f"  Full build log: {_dhh()}/logs/update.log")
        return False
    print("  ✓ Desktop app up to date")
    return True


def _path_uid(path) -> Optional[int]:
    """Owner uid of ``path`` via ``os.stat`` — ``None`` when unreadable.

    Separate seam so tests can simulate root-owned files without chown
    (which needs root). Never raises.
    """
    try:
        return os.stat(path, follow_symlinks=False).st_uid
    except OSError:
        return None


def _venv_foreign_owned_paths(venv_root, limit: int = 5) -> list:
    """Bounded scan for venv entries not owned by the current user (#83529).

    A venv ever touched by ``sudo pip`` / ``sudo hermes`` contains root-owned
    files (classically ``*.dist-info/INSTALLER``); a later normal ``hermes
    update`` then dies mid-mutation inside ``uv pip install -e .`` with
    ``venv/bin/hermes`` already deleted — the CLI is bricked. Same philosophy
    as the contended-venv gate (#87331): never mutate a venv we cannot safely mutate.

    Deliberately BOUNDED (no full recursion): the venv root, direct entries of
    ``venv/bin``, top-level entries of the first ``lib/python*/site-packages``,
    and direct children of each ``*.dist-info`` there; ~2000 stat calls max,
    at most ``limit`` paths returned. POSIX-only: ``[]`` on Windows and as
    root. Swallows every per-entry ``OSError`` and returns ``[]`` on any
    structural surprise — must NEVER raise or add noticeable latency.

    Returns ``(path_str, uid)`` tuples, at most ``limit`` long.
    """
    try:
        if not hasattr(os, "geteuid"):
            return []  # windows-footgun: ok — POSIX ownership concept only
        euid = os.geteuid()  # windows-footgun: ok — guarded by hasattr above
        if euid == 0:
            return []  # root can rewrite anything; nothing to refuse

        venv_root = Path(venv_root)
        budget = 2000  # max stat() calls — hard bound on preflight cost
        foreign: list = []

        def _check(p) -> bool:
            """stat one path; True while scan should continue."""
            nonlocal budget
            if budget <= 0 or len(foreign) >= limit:
                return False
            budget -= 1
            uid = _path_uid(p)
            if uid is not None and uid != euid:
                foreign.append((str(p), uid))
            return budget > 0 and len(foreign) < limit

        def _scan_dir(d, recurse_dist_info: bool = False) -> None:
            try:
                entries = list(os.scandir(d))
            except OSError:
                return
            for entry in entries:
                if not _check(entry.path):
                    return
                if recurse_dist_info and entry.name.endswith(".dist-info"):
                    try:
                        children = list(os.scandir(entry.path))
                    except OSError:
                        continue
                    for child in children:
                        if not _check(child.path):
                            return

        if not _check(venv_root):
            return foreign[:limit]
        _scan_dir(venv_root / "bin")

        # First lib/python*/site-packages (POSIX venv layout).
        site_packages = next(
            iter(sorted(venv_root.glob("lib/python*/site-packages"))), None
        )
        if site_packages is not None:
            _scan_dir(site_packages, recurse_dist_info=True)

        return foreign[:limit]
    except Exception:
        # Preflight is advisory: any structural surprise means "no verdict",
        # never a crashed or blocked update.
        return []


def _refuse_update_if_venv_foreign_owned(project_root) -> None:
    """Refuse-before-mutate ownership gate for the dependency install (#83529).

    Runs after the code pull and immediately before the first venv mutation:
    foreign-owned venv files would make ``uv pip install -e .`` die
    mid-mutation and brick the install, so refuse up front with the exact
    recovery command while the venv is intact. No subprocess calls here —
    update tests mock ``subprocess.run`` with sequenced side effects.
    """
    foreign = _venv_foreign_owned_paths(Path(project_root) / "venv")
    if not foreign:
        return
    print("\n✗ Update stopped: this install's venv contains files owned by another user.")
    print("  Updating now would fail midway (Permission denied) and leave Hermes broken.")
    print("  This usually happens after running hermes or pip with sudo. Offending paths:")
    for p, uid in foreign:
        print(f"    - {p} (owner uid {uid})")
    print("\n  Fix ownership, then re-run the update:")
    print(f"    sudo chown -R $(id -un): {project_root}")
    print("    hermes update")
    print("\n  Nothing in the venv was modified.")
    sys.exit(1)


def _drain_or_signal_gateway_for_update(
    pid: int,
    drain_budget: float,
    label: str,
) -> bool:
    """Decide how ``hermes update`` hands a running gateway over to new code.

    Three-way triage shared by the systemd and bare-process restart paths:

    1. **Gateway is an ancestor of this process** — deadlock break (#100179).
       When ``hermes update`` runs INSIDE the gateway's process tree (the
       hermes-auto-update cron job), waiting for the gateway is circular:
       gateway waits on in-flight work units (#77184) → cron session waits on
       ``hermes update`` → ``hermes update`` waits on the gateway. The
       wedged-loop probe can't break it (the cron session posts activity every
       ~180s, so it is never marked wedged) and the gateway burns the full
       1800s force-drain cap. Fire-and-forget instead: signal the restart and
       return; the gateway's own restart completes once THIS process exits.
    2. **Event loop provably wedged** (#81642) — SIGUSR1 can never drain it;
       bounded escalation (SIGTERM grace → SIGKILL) instead.
    3. **Live, out-of-tree gateway** — normal graceful SIGUSR1 drain, waiting
       up to ``drain_budget`` (including the #86684 cron floor).

    Returns True when the gateway was signalled/stopped successfully.
    """
    from hermes_cli.gateway import (
        GATEWAY_LOOP_WEDGED,
        _escalate_wedged_gateway,
        _graceful_restart_via_sigusr1,
        _is_pid_ancestor_of_current_process,
        _request_gateway_self_restart,
        probe_gateway_loop_liveness,
    )

    if _is_pid_ancestor_of_current_process(pid):
        print(
            f"  → {label}: update is running inside this gateway's "
            "process tree — signalling restart and letting the gateway "
            "drain itself (avoids the cron-update deadlock, #100179)"
        )
        return _request_gateway_self_restart(pid)
    if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        print(
            f"  ⚠ {label}: gateway event loop is unresponsive — "
            "skipping drain, forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        return True
    print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
    return _graceful_restart_via_sigusr1(pid, drain_timeout=drain_budget)


def _resolve_manage_cmd(cache: dict, scope_: str, scope_cmd_: list, svc_name_: str):
    """Resolve the command prefix for manage-units operations.

    Read-only systemctl calls work unprivileged, but manage-units verbs
    (``reset-failed``, ``start``, ``restart``) on a *system* service trigger a
    polkit auth prompt for non-root users. That prompt runs inside our captured
    10-15s subprocess — it flashes and dies before the user can answer, and the
    TimeoutExpired used to be swallowed silently.

    Strategy: root → plain systemctl. Otherwise try ``sudo -n`` — a blanket
    probe, then a targeted ``systemctl reset-failed`` probe so a least-privilege
    sudoers entry scoped to ``systemctl ... hermes-gateway*`` also qualifies
    (``reset-failed`` is an idempotent no-op we run before every privileged
    restart anyway). If neither works return None: the caller must SKIP the
    restart (without draining the gateway first!) and print manual steps.
    ``--no-ask-password`` guarantees polkit can never hang this path.
    """
    if scope_ in cache:
        return cache[scope_]
    cmd = scope_cmd_ + ["--no-ask-password"]
    if (
        scope_ == "system"
        and hasattr(os, "geteuid")
        and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
    ):
        sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
        sudo_ok = False
        try:
            _probe = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True,
                timeout=5,
            )
            sudo_ok = _probe.returncode == 0
            if not sudo_ok:
                # Blanket sudo refused — a targeted sudoers entry
                # (NOPASSWD for systemctl ... hermes-gateway*)
                # may still allow the exact commands we need.
                _probe = subprocess.run(
                    sudo_cmd + ["reset-failed", svc_name_],
                    capture_output=True,
                    timeout=5,
                )
                sudo_ok = _probe.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            sudo_ok = False
        cmd = sudo_cmd if sudo_ok else None
    cache[scope_] = cmd
    return cmd


def _restart_systemd_gateway_units(
    restarted_services, failed_or_stale_units, restarted_scoped_units, drain_budget
):
    """Restart every active hermes-gateway*/hermes-serve* systemd unit (user + system scope).

    Appends settled units to ``restarted_services`` (bare names) and
    ``restarted_scoped_units`` (``scope/name``), failures to ``failed_or_stale_units``.
    Per-unit timeouts are isolated so one wedged unit never aborts the fleet.
    """
    from hermes_cli.gateway import supports_systemd_services, _ensure_user_systemd_env

    _manage_cmd_cache: dict = {}

    # --- Systemd services (Linux) ---
    # Discover all hermes-gateway* units (default + profiles) plus
    # hermes-serve* units (the Desktop app's backend, #83438).
    if supports_systemd_services():
        try:
            _ensure_user_systemd_env()
        except Exception:
            pass

        for scope, scope_cmd in [
            ("user", ["systemctl", "--user"]),
            ("system", ["systemctl"]),
        ]:
            try:
                result = _systemctl(
                    scope_cmd + ["list-units", "hermes-gateway*", "hermes-serve*",
                                 "--plain", "--no-legend", "--no-pager"],
                    timeout=10,
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired as exc:
                # Discovery timeout — skip this scope, keep the other.
                print(
                    f"  ⚠ systemctl timed out listing {scope}-scope "
                    f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
                    f"Check the gateway with: hermes gateway status"
                )
                continue

            def _restart_one_systemd_gateway_unit(svc_name: str) -> None:
                # Check if active
                check = _systemctl(scope_cmd + ["is-active", svc_name], timeout=5)
                if check.stdout.strip() != "active":
                    return

                # Resolve how we may run manage-units verbs for this scope.
                # None ⇒ no non-interactive privilege path; avoid those verbs
                # entirely or polkit throws an auth prompt inside our captured
                # 10-15s subprocess (it flashes and "exits directly").
                _manage_cmd = _resolve_manage_cmd(_manage_cmd_cache, 
                    scope, scope_cmd, svc_name
                )

                # Prefer a graceful SIGUSR1 restart so in-flight agent runs
                # drain instead of being SIGKILLed: the handler calls
                # request_restart(via_service=True) → drain → exit, and
                # Restart=always respawns the unit. hermes-serve has no such
                # handler, so it skips straight to the blunt restart below.
                _main_pid = 0
                if _service_unit_supports_graceful_sigusr1_restart(svc_name):
                    try:
                        _show = _systemctl(scope_cmd + ["show", svc_name, "--property=MainPID", "--value"], timeout=5)
                        _main_pid = int((_show.stdout or "").strip() or 0)
                    except (
                        ValueError,
                        subprocess.TimeoutExpired,
                        FileNotFoundError,
                    ):
                        _main_pid = 0

                _graceful_ok = False
                if _main_pid > 0:
                    # Three-way triage (#100179 ancestor / #81642 wedged /
                    # graceful drain), shared with the bare-process path.
                    _graceful_ok = _drain_or_signal_gateway_for_update(
                        _main_pid, drain_budget, svc_name
                    )

                if _graceful_ok:
                    # Gateway exited after a planned restart. ``Restart=always``
                    # respawns the unit only after ``RestartSec`` (60s on our
                    # unit file) — a crash-loop guard that is dead time for a
                    # voluntary update restart. ``reset-failed`` + ``start``
                    # skips RestartSec (we initiate the unit manually), taking
                    # ~1-3s on a warm box; if RestartSec already elapsed while
                    # draining, ``start`` is a no-op and we fall through to the
                    # poll below. Either way the 60s+ delay collapses to ~5s.
                    #
                    # The shortcut needs manage-units privileges; without them
                    # skip it — systemd's auto-restart still relaunches the
                    # unit after RestartSec.
                    if _manage_cmd is not None:
                        _systemctl(_manage_cmd + ["reset-failed", svc_name], timeout=10)
                        _systemctl(_manage_cmd + ["start", svc_name], timeout=15)
                        # Short poll: RestartSec was bypassed, so it should be up in seconds.
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=10.0,
                        ):
                            restarted_services.append(svc_name)
                            return
                    # Passive poll: systemd's auto-restart fires after
                    # RestartSec regardless of privileges — the primary path
                    # when _manage_cmd is None, the fallback otherwise.
                    _restart_sec = _service_restart_sec(
                        scope_cmd,
                        svc_name,
                        default=0.0,
                    )
                    _post_drain_timeout = max(
                        10.0,
                        _restart_sec + 10.0,
                    )
                    if _manage_cmd is None and _restart_sec > 5.0:
                        print(
                            f"  → {svc_name}: waiting for systemd "
                            f"auto-restart (~{int(_restart_sec)}s; "
                            "no root for an immediate restart)..."
                        )
                    if _wait_for_service_active(
                        scope_cmd,
                        svc_name,
                        timeout=_post_drain_timeout,
                    ):
                        restarted_services.append(svc_name)
                        return
                    # Exited but not respawned (older unit without
                    # Restart=on-failure / RestartForceExitStatus=75); fall
                    # through to systemctl start/restart.
                    print(
                        f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                    )

                # Forcing a restart needs manage-units privileges. Without a
                # non-interactive path, systemctl would spawn a polkit prompt
                # inside a captured 10-15s subprocess (flashes and dies before
                # the user can answer) — skip with clear instructions.
                if _manage_cmd is None:
                    failed_or_stale_units.append(svc_name)
                    print(
                        f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                        f"    Restart it manually to load the new version:\n"
                        f"      sudo systemctl restart {svc_name}\n"
                        f"    To let `hermes update` restart it automatically, allow\n"
                        f"    passwordless sudo for systemctl, or run updates with sudo."
                    )
                    return

                # Fallback: blunt systemctl restart — only reached when the graceful
                # path failed (unit missing SIGUSR1 wiring, drain exceeded the budget,
                # restart-policy mismatch). Mirrors `hermes gateway restart`
                # (`systemd_restart()`, PR #20949).
                restart = _systemctl_reset_and_restart(_manage_cmd, svc_name)
                if restart.returncode == 0:
                    # systemctl restart returns 0 even if the new process
                    # crashes immediately — verify it survived.
                    if _wait_for_service_active(
                        scope_cmd,
                        svc_name,
                        timeout=10.0,
                    ):
                        restarted_services.append(svc_name)
                    else:
                        # Retry once — transient startup failures (stale module
                        # cache, import race) often resolve on the second try.
                        # Clear failed state first so the retry isn't blocked.
                        print(
                            f"  ⚠ {svc_name} died after restart, retrying..."
                        )
                        _systemctl_reset_and_restart(_manage_cmd, svc_name)
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=10.0,
                        ):
                            restarted_services.append(svc_name)
                            print(f"  ✓ {svc_name} recovered on retry")
                        else:
                            failed_or_stale_units.append(svc_name)
                            _scope_flag = "--user " if scope == "user" else ""
                            _sudo_hint = "sudo " if scope == "system" else ""
                            print(
                                f"  ✗ {svc_name} failed to stay running after restart.\n"
                                f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                f"    Recover manually:\n"
                                f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                            )
                else:
                    failed_or_stale_units.append(svc_name)
                    print(
                        f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                    )

            def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
                # Isolate the timeout to this unit and keep going
                # (#68523). A scope-wide handler used to abort every
                # later gateway and leave the fleet on mixed code.
                failed_or_stale_units.append(svc_name)
                print(
                    f"  ⚠ systemctl timed out restarting {svc_name} "
                    f"({exc.cmd if exc.cmd else 'unknown command'}); "
                    f"continuing with remaining gateways"
                )

            # Qualify everything this scope appended to ``restarted_services``
            # before the next scope can add a same-named unit; ``finally`` so a
            # mid-scope abort still carries the units it settled.
            _scope_mark = len(restarted_services)
            try:
                _for_each_systemd_gateway_unit(
                    result.stdout,
                    process_unit=_restart_one_systemd_gateway_unit,
                    on_unit_timeout=_on_unit_timeout,
                )
            finally:
                restarted_scoped_units.update(
                    f"{scope}/{name}"
                    for name in restarted_services[_scope_mark:]
                )


@dataclass
class _GatewayRestartOutcome:
    """Bookkeeping the post-update gateway restart phase hands back to the update flow.

    ``restarted_services`` keeps bare unit names (the fleet probe, receipt and
    operator summary all read it); ``incomplete`` means at least one gateway
    may still be serving pre-update code.
    """

    incomplete: bool
    phase_errors: list
    pre_restart_gateway_pids: "list | None"
    restarted_services: list
    failed_or_stale_units: list
    relaunched_profiles: list
    externally_supervised_profiles: list
    killed_pids: set


def _restart_gateway_fleet_after_update(_pre_update_plan, gateway_mode: bool):
    """Restart every running gateway (systemd, launchd, manual) so it picks up the pulled code.

    Never raises: a phase abort runs the fresh-child recovery and fails closed
    (``incomplete=True``) unless every planned gateway is verifiably covered.
    """
    gateway_fleet_restart_incomplete = False
    gateway_restart_phase_errors: list[str] = []
    # Gateways running before we touch anything. Stays empty until the probe
    # is imported and we are about to stop/drain, so an early exception has
    # nothing to fail closed on, while a failure after stopping a discovered
    # gateway fails closed on an empty survivor probe (#78574).
    _pre_restart_gateway_pids: list | None = []
    # Declared outside the try/except (never reset to None) so it is safe to
    # read even if the block raises early — already-restarted units are
    # forwarded to ``_finish_dashboard_update_cleanup`` (#83595).
    restarted_services: list = []
    # Scope-qualified twin of ``restarted_services`` (``user/hermes-serve``
    # vs ``system/hermes-serve`` are different processes; abort recovery
    # needs to know WHICH settled). ``restarted_services`` keeps bare names
    # for the fleet probe, receipt and summary (#92145).
    restarted_scoped_units: set = set()
    # Defined up front so abort recovery and fleet reconciliation can read
    # them even when the phase raises before its imports initialize them.
    failed_or_stale_units: list = []
    relaunched_profiles: list = []
    externally_supervised_profiles: list = []
    # Same treatment: the fleet version check uses killed_pids to decide
    # whether to wait for settle, and the except path forwards it to the receipt.
    killed_pids: set = set()

    # The pulled code is shared across profiles, so EVERY running gateway
    # restarts. Purge stale cached Hermes modules FIRST: the import below
    # loads new gateway source into this pre-update interpreter, and a
    # cached sibling (cli_output, status, ...) missing a symbol the new
    # source expects would ImportError and abort the whole phase.
    _m()._purge_stale_hermes_modules()
    try:
        from hermes_cli.gateway import (
            is_macos,
            find_gateway_pids,
            find_profile_gateway_processes,
            _prepare_profile_gateway_update_restart,
            _get_service_pids,
            _wait_for_gateway_exit,
        )
        import signal as _signal

        # Wait budget for graceful SIGUSR1 restarts: covers both the
        # ``restart_after_turn_timeout`` deferral (#77184) and the
        # ``restart_drain_timeout`` inside stop(), so we don't hard-kill a
        # gateway still waiting on a turn. Units without SIGUSR1 wiring
        # just time out and fall back to ``systemctl restart``.
        try:
            from hermes_cli.gateway import _get_restart_exit_wait_budget

            _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
        except Exception:
            _drain_budget = 45.0

        failed_or_stale_units = []
        killed_pids = set()
        relaunched_profiles = []
        externally_supervised_profiles = []

        # Snapshot running gateways before any stop/drain so an empty
        # survivor probe later reads as "stopped and never came back", not
        # "nothing was running" (#78574). If the probe raises, None fails closed.
        try:
            _pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            _pre_restart_gateway_pids = None

        _restart_systemd_gateway_units(
            restarted_services, failed_or_stale_units, restarted_scoped_units, _drain_budget
        )

        # --- Launchd services (macOS): EVERY ai.hermes.gateway* LaunchAgent,
        # parity with systemd (#41403). Per-label TimeoutExpired isolation inside.
        if is_macos():
            try:
                _restart_macos_launchd_gateways(
                    restarted_services,
                    failed_or_stale_units,
                    _drain_budget,
                )
            except (FileNotFoundError, ImportError):
                pass

        # --- Manual (non-service) gateways --- excluding PIDs of
        # just-restarted services so we don't kill what systemd/launchd spawned.
        service_pids = _get_service_pids(all_profiles=True)
        manual_pids = find_gateway_pids(
            exclude_pids=service_pids, all_profiles=True
        )
        profile_processes = {
            proc.pid: proc
            for proc in find_profile_gateway_processes(exclude_pids=service_pids)
            if proc.pid in manual_pids
        }
        # Profile gateways we could not arm a relaunch for must NOT keep
        # running on pre-update modules (#88654): hand them to the unmapped
        # sweep below, which stops them and lists them under "Restart manually".
        unrestartable_pids = set()
        for pid, proc in profile_processes.items():
            restart_mode = _prepare_profile_gateway_update_restart(
                proc.profile, pid
            )
            if restart_mode is None:
                # Previously a bare ``continue``: the gateway was neither
                # relaunched nor stopped nor mentioned, so it kept serving
                # from stale modules with no operator signal at all.
                print(
                    f"  ⚠ {proc.profile}: could not arm an automatic "
                    f"gateway restart for PID {pid} — stopping it instead "
                    "so it cannot keep running pre-update code"
                )
                unrestartable_pids.add(pid)
                continue
            # Graceful SIGUSR1 drain first (in-flight runs finish), SIGTERM
            # fallback if unsupported or over budget — the watcher relaunches
            # either way. Three-way triage (ancestor fire-and-forget #100179 /
            # wedged escalation #81642 / normal drain) shared with the systemd
            # path; the helper announces its choice first because a silent
            # full-budget wait reads as a hung update (#44515).
            drained = _drain_or_signal_gateway_for_update(
                pid, _drain_budget, proc.profile
            )
            if not drained:
                try:
                    os.kill(pid, _signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            # Wait up to 5s for the old process to exit before the watcher
            # respawns. Telegram keeps the old getUpdates session alive ~30s;
            # a new gateway connecting inside that window gets a 409 that
            # _handle_polling_conflict() retries through, but a brief wait
            # avoids that path on fast machines (watcher restarts in <1s).
            _wait_for_gateway_exit(timeout=5.0, force_after=None)
            killed_pids.add(pid)
            if restart_mode == "external-supervisor":
                externally_supervised_profiles.append(proc.profile)
            else:
                relaunched_profiles.append(proc.profile)

        for pid in manual_pids:
            if pid in profile_processes and pid not in unrestartable_pids:
                continue
            try:
                os.kill(pid, _signal.SIGTERM)
                killed_pids.add(pid)
            except (ProcessLookupError, PermissionError):
                pass

        if restarted_services or killed_pids:
            print()
            for svc in restarted_services:
                print(f"  ✓ Restarted {svc}")
            if relaunched_profiles:
                names = ", ".join(relaunched_profiles)
                print(f"  ✓ Restarting manual gateway profile(s): {names}")
            if externally_supervised_profiles:
                names = ", ".join(externally_supervised_profiles)
                print(
                    "  ✓ Handed gateway profile(s) back to their external "
                    f"supervisor: {names}"
                )
            unmapped_count = (
                len(killed_pids)
                - len(relaunched_profiles)
                - len(externally_supervised_profiles)
            )
            if unmapped_count:
                print(f"  → Stopped {unmapped_count} manual gateway process(es)")
                print("    Restart manually: hermes gateway run")
                if unmapped_count > 1:
                    print(
                        "    (or: hermes -p <profile> gateway run  for each profile)"
                    )

        if failed_or_stale_units:
            gateway_fleet_restart_incomplete = True
            if gateway_mode:
                _write_gateway_update_exit_code(False)
        _warn_incomplete_gateway_fleet_restart(failed_or_stale_units)

        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=bool(failed_or_stale_units),
            )
        except Exception:
            pass


        # --- Post-restart survivor sweep (#17648) ---------------------
        # Gateways that ignore SIGTERM (stuck drain, blocked I/O, zombie)
        # never exit, so the 120s profile watcher never respawns and the
        # user keeps hitting ImportError on stale sys.modules. Give graceful
        # paths a moment, then SIGKILL remaining pre-update PIDs.
        try:
            _time.sleep(3.0)
            _service_pids_after = _get_service_pids(all_profiles=True)
            _surviving = find_gateway_pids(
                exclude_pids=_service_pids_after,
                all_profiles=True,
            )
            # Only PIDs we already tried to kill; anything newer started
            # AFTER our restart attempt and is left alone.
            _stuck = [pid for pid in _surviving if pid in killed_pids]
            if _stuck:
                print()
                print(
                    f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
                )
                from gateway.status import (
                    get_process_start_time as _get_process_start_time,
                    terminate_pid as _terminate_pid,
                )
                for pid in _stuck:
                    try:
                        # taskkill /T /F on Windows, SIGKILL on POSIX —
                        # _signal.SIGKILL doesn't exist on Windows.
                        _terminate_pid(
                            pid,
                            force=True,
                            expected_start_time=_get_process_start_time(pid),
                        )
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                # Give the OS a beat to reap the processes so the
                # watchers see them exit and respawn.
                _time.sleep(1.5)
        except Exception as _sweep_exc:
            logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)

    except Exception as e:
        logger.debug("Gateway restart during update failed: %s", e)
        gateway_restart_phase_errors.append(str(e))
        # An escaped exception means the restart output never printed. Treat
        # the fleet as stale unless we can positively prove no gateway runs
        # (#78574). An empty ``_surviving`` proves safety only if nothing was
        # running beforehand; a pre-restart gateway that is gone now was
        # stopped without a verified replacement, so ``[]`` still fails closed.
        _surviving = _surviving_gateway_pids_after_failed_restart()
        _already_restarted_profiles = set(relaunched_profiles)
        _already_restarted_profiles.update(externally_supervised_profiles)
        for runtime in getattr(_pre_update_plan, "runtimes", ()) or ():
            if getattr(runtime, "kind", None) != "gateway":
                continue
            profile = getattr(runtime, "profile", None)
            if not isinstance(profile, str):
                continue
            if any(
                _gateway_service_matches_profile(profile, service)
                for service in restarted_services
            ):
                _already_restarted_profiles.add(profile)
        _recovery_result = _recover_gateway_restart_after_abort(
            _pre_update_plan,
            gateway_mode=gateway_mode,
            skip_profiles=_already_restarted_profiles,
            skip_units=set(restarted_scoped_units),
        )
        _recovery_serve_units = _recovery_result.get("serve_units") or {}
        _serve_units_failed = list(_recovery_serve_units.get("failed") or [])
        # Deliberately NOT merged into ``restarted_services`` (gateway-phase
        # vocabulary feeding the fleet-probe expectation); serve coverage
        # lives in the recovery result and receipt. A serve/dashboard runtime
        # that is still the SAME pre-update process is live on old code
        # (#92145, e.g. tui_gateway under `hermes serve`, unreachable by any
        # `gateway restart`). Recovery may not claim success while one
        # remains, and must never kill one: manual/Desktop-owned serves have
        # no relaunch authority.
        _stale_runtime_rows = _surviving_pre_update_serve_runtimes(
            _pre_update_plan
        )
        _recovery_result["stale_runtimes"] = _stale_runtime_rows
        # Only systemd-VERIFIED outcomes may claim supervisor coverage.
        # A relaunch that merely exited 0 ("relaunch_attempted") was never
        # observed by the code and must not clear the incomplete flag.
        _recovery_verified = set(_recovery_result.get("verified") or [])
        if _recovery_verified:
            relaunched_profiles.extend(
                profile
                for profile in sorted(_recovery_verified)
                if profile not in relaunched_profiles
            )
        _planned_gateway_runtimes = [
            runtime
            for runtime in getattr(_pre_update_plan, "runtimes", ()) or ()
            if getattr(runtime, "kind", None) == "gateway"
            and isinstance(getattr(runtime, "profile", None), str)
        ]
        _planned_gateway_profiles = {
            runtime.profile for runtime in _planned_gateway_runtimes
        }
        _covered_gateway_profiles = (
            _already_restarted_profiles | _recovery_verified
        )
        _recovery_complete = _abort_recovery_is_complete(
            planned_gateway_profiles=_planned_gateway_profiles,
            covered_gateway_profiles=_covered_gateway_profiles,
            recovery_result=_recovery_result,
            stale_runtime_rows=_stale_runtime_rows,
        )
        if _recovery_complete:
            # The fresh child is the recovery terminal result. Leave the
            # final fleet-version matrix below as the authoritative
            # read-back before the update is declared successful.
            gateway_fleet_restart_incomplete = False
        elif (
            _restart_phase_failure_is_incomplete(
                _surviving, _pre_restart_gateway_pids
            )
            or _stale_runtime_rows
            or _serve_units_failed
        ):
            gateway_fleet_restart_incomplete = True
            _warn_gateway_restart_phase_aborted(e, _surviving)
            _warn_stale_serve_runtimes(_stale_runtime_rows)
            if gateway_mode:
                _write_gateway_update_exit_code(False)
        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=gateway_fleet_restart_incomplete,
                phase_error=str(e),
                fresh_recovery=_recovery_result,
            )
        except Exception:
            pass

    return _GatewayRestartOutcome(
        incomplete=gateway_fleet_restart_incomplete,
        phase_errors=gateway_restart_phase_errors,
        pre_restart_gateway_pids=_pre_restart_gateway_pids,
        restarted_services=restarted_services,
        failed_or_stale_units=failed_or_stale_units,
        relaunched_profiles=relaunched_profiles,
        externally_supervised_profiles=externally_supervised_profiles,
        killed_pids=killed_pids,
    )


def _repair_current_checkout(
    *,
    assume_yes,
    gateway_mode,
    pre_update_snapshot_id,
    desktop_dir,
    had_desktop_app_before_update,
    active_lazy_features,
    active_tool_dependencies,
    upstream_checked,
    _windows_gateway_resume,
) -> bool:
    """Already-up-to-date path: keep the managed runtime current and repair a broken venv.

    A current checkout does not imply a healthy install (a prior dependency
    sync may have died partway), and the Windows shim hand-off child lands
    here BY DESIGN to run the sync its parent could not. Returns whether the
    checkout can be reported as complete.
    """
    # "No new commits" does not mean the managed interpreter is safe.
    # uv can retain the same CPython patch while python-build-standalone
    # refreshes the embedded SQLite underneath it. Keep the existing
    # update-boundary hook active on this retry path too.
    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    runtime_repairs = []
    update_managed_uv(repair_observer=runtime_repairs.append)
    ensure_uv(repair_observer=runtime_repairs.append)
    runtime_repaired = next(
        (result for result in runtime_repairs if result.repaired),
        None,
    )

    # A current checkout does NOT imply a healthy install: a prior sync may
    # have died partway (Windows: locked .pyd → uv/pip access-denied, venv
    # stranded between versions). Probe core imports and repair, or
    # "Already up to date!" hides a bricked install.
    healthy, detail = _venv_core_imports_healthy()
    # The Windows shim hand-off child exists to run the sync its parent
    # could not; the checkout is current BY DESIGN, so the pending sync —
    # not venv health — is the question. Without this it would print
    # "Already up to date!" and skip its one job.
    handed_off_sync = os.environ.get(_m()._UPDATE_REEXEC_ENV) == "1"
    current_checkout_complete = True
    if handed_off_sync:
        print("→ Finishing the dependency install handed off by hermes.exe...")
    elif not healthy:
        print("⚠ Checkout is current, but the venv is unhealthy:")
        print(f"  {detail}")
        print("→ Repairing Python dependencies...")
    if handed_off_sync or not healthy:
        # Self-lock deferral (#86735): the repair rewrites the venv
        # too — same mapped-extension hazard as the update sync.
        _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
        _write_update_incomplete_marker()
        from hermes_cli.managed_uv import ensure_uv

        repair_uv = ensure_uv()
        # A managed install whose venv is gone entirely (interrupted
        # repair after the old venv was moved aside) needs the venv
        # recreated before dependencies can be installed into it.
        venv_python_missing = not (
            venv_python_path(
                _m().PROJECT_ROOT / "venv", windows=_m()._is_windows()
            )
        ).exists()
        if venv_python_missing and repair_uv:
            print("→ Recreating virtual environment...")
            subprocess.run(
                [repair_uv, "venv", "venv"],
                cwd=_m().PROJECT_ROOT,
                check=False,
            )
        if repair_uv:
            # Isolated from third-party UV env vars (#83914), same as
            # the main-path and git-path dependency syncs.
            from hermes_cli.managed_uv import managed_python_env

            repair_env = managed_python_env()
            repair_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
            _m()._install_python_dependencies_with_optional_fallback(
                [repair_uv, "pip"], env=repair_env, group="all"
            )
            _m()._refresh_active_lazy_features(
                [repair_uv, "pip"],
                env=repair_env,
                features=active_lazy_features,
            )
            _m()._restore_active_tool_dependencies(
                active_tool_dependencies,
                [repair_uv, "pip"],
                env=repair_env,
            )
        else:
            _m()._install_python_dependencies_with_optional_fallback(
                [sys.executable, "-m", "pip"], group="all"
            )
            _m()._refresh_active_lazy_features(
                [sys.executable, "-m", "pip"],
                features=active_lazy_features,
            )
            _m()._restore_active_tool_dependencies(
                active_tool_dependencies,
                [sys.executable, "-m", "pip"],
            )
        _m()._clear_update_incomplete_marker()
        healthy_after, detail_after = _venv_core_imports_healthy()
        if healthy_after:
            print("✓ Dependencies repaired!")
            _check_and_apply_config_migration(
                assume_yes=assume_yes,
                gateway_mode=gateway_mode,
                pre_update_snapshot_id=pre_update_snapshot_id,
            )
            # The hand-off child never reaches the commits-pulled rebuild,
            # so rebuild the Desktop app here or it stays on the old build (#97343).
            if _rebuild_desktop_after_update(
                desktop_dir,
                had_desktop_app_before_update=had_desktop_app_before_update,
            ):
                current_checkout_complete = _print_verified_update_completion(
                    "✓ Update complete!"
                )
            else:
                current_checkout_complete = False
                _print_update_completion(
                    "⚠ Update partially complete — the desktop app was "
                    "not rebuilt and is still on the previous build."
                )
        else:
            current_checkout_complete = False
            print(f"⚠ Venv still unhealthy after repair: {detail_after}")
            print("  Close all Hermes windows/gateways and re-run: hermes update")
    else:
        current_checkout_complete = _repair_node_deps_on_current_checkout(
            _print_verified_update_completion,
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            completion_message=(
                "✓ Already up to date!"
                if upstream_checked
                else "✓ Up to date with your fork (official repo not checked)."
            ),
            had_desktop_app_before_update=had_desktop_app_before_update,
        )
    if runtime_repaired is not None and not _m()._is_windows():
        print()
        print(
            "⚠ Restart required to finish the managed Python runtime repair."
        )
        print(
            "  Any running Hermes gateways, Desktop backends, or other "
            "long-lived processes still use the previous runtime."
        )
        print("  Restart each of them to pick up the repaired runtime.")
    return current_checkout_complete


def _pull_updates(
    git_cmd,
    branch,
    auto_stash_ref,
    *,
    prompt_for_restore,
    gw_input_fn,
    discard_local_changes,
    keep_stash,
):
    """Fast-forward the checkout onto ``origin/<branch>`` and settle the autostash.

    Divergence is handled by shape (custom branch -> merge, same branch ->
    reset, orphan history -> rescue ref first); a post-pull syntax error in a
    critical file rolls back to the pre-pull SHA. Exits the process on
    failure. Returns the pre-pull HEAD SHA (or None).
    """
    update_succeeded = False
    # Pre-pull SHA for auto-rollback when pulled code has a syntax error in
    # a critical file (PR #28452: stray conflict markers in config.py
    # bricked every updater for 7 minutes).
    pre_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
    try:
        # Merge the ref we already fetched above (→ Fetching updates...)
        # instead of `git pull`, which performs a SECOND network fetch of
        # the same branch (~0.5-1.5 s of redundant round-trip per update).
        # `merge --ff-only origin/<branch>` is byte-identical in effect to
        # `pull --ff-only origin <branch>` given the fresh tracking ref;
        # the divergence fallback below is unchanged.
        pull_result = _git_run(git_cmd, ["merge", "--ff-only", f"origin/{branch}"])
        if pull_result.returncode != 0:
            # ff-only failed — local and remote have diverged. Before
            # assuming an upstream force-push, check WHY: a checkout on a
            # custom branch (local commits on top of origin/<branch>) also
            # cannot fast-forward, and `reset --hard` here would silently
            # discard that work. Merge instead and stop cleanly on
            # conflict — an update must never destroy local commits.
            _cur_branch = (
                _git_run(git_cmd, ["branch", "--show-current"]).stdout
                or ""
            ).strip()
            if _cur_branch and _cur_branch != branch:
                print(
                    f"  ⚠ Checkout is on custom branch '{_cur_branch}' — "
                    f"merging origin/{branch} instead of resetting so local commits survive..."
                )
                # Best-effort safety tag; recovery anchor if anything goes wrong.
                subprocess.run(
                    git_cmd
                    + ["tag", f"pre-update-{_time.strftime('%Y%m%d-%H%M%S')}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    check=False,
                )
                merge_result = _git_run(git_cmd, ["merge", "--no-edit", f"origin/{branch}"])
                if merge_result.returncode != 0:
                    subprocess.run(
                        git_cmd + ["merge", "--abort"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        check=False,
                    )
                    print(
                        "✗ Merge conflict between local commits and upstream — "
                        "update stopped, nothing was changed."
                    )
                    print(
                        f"  Resolve manually: cd {_m().PROJECT_ROOT} && "
                        f"git merge origin/{branch}"
                    )
                    print(
                        "  Then re-run the update. Local work is untouched."
                    )
                    sys.exit(1)
            else:
                # Same branch as the target — a true upstream force-push/
                # rebase; local changes are stashed, so reset to the remote.
                # Orphan divergence (no common ancestor: corrupted HEAD, repo
                # re-init — #87694) would lose the whole local commit graph,
                # so park pre_pull_sha behind a rescue ref first.
                merge_base_result = _git_run(git_cmd, ["merge-base", "HEAD", f"origin/{branch}"])
                has_common_ancestor = bool(
                    merge_base_result.returncode == 0
                    and merge_base_result.stdout.strip()
                )
                if not has_common_ancestor and pre_pull_sha:
                    from datetime import datetime as _dt, timezone

                    # SHA suffix (not just a 1s timestamp) so two updates in
                    # the same second get distinct refs instead of overwriting.
                    rescue_ref = (
                        f"refs/hermes-update-backups/orphan-{branch}-"
                        f"{_dt.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                        f"-{pre_pull_sha[:12]}"
                    )
                    update_ref_result = _git_run(git_cmd, ["update-ref", rescue_ref, pre_pull_sha])
                    if update_ref_result.returncode == 0:
                        print(
                            "  ⚠ Local history shares no common ancestor with "
                            f"origin/{branch} (orphan divergence) — backed up "
                            f"current HEAD to {rescue_ref} before resetting. "
                            f"This backup expires after "
                            f"{_ORPHAN_RESCUE_REF_MAX_AGE_DAYS} days."
                        )
                    else:
                        # update-ref's return code is intentionally not
                        # fatal (disk full, permissions) — but don't tell
                        # the user a backup exists when the write failed.
                        print(
                            "  ⚠ Local history shares no common ancestor with "
                            f"origin/{branch} (orphan divergence) — attempted "
                            f"to back up current HEAD to {rescue_ref} before "
                            "resetting, but the backup write failed "
                            f"(pre-reset SHA was {pre_pull_sha})."
                        )
                    _prune_orphan_rescue_refs(git_cmd, _m().PROJECT_ROOT, branch)
                print(
                    "  ⚠ Fast-forward not possible (history diverged), resetting to match remote..."
                )
                reset_result = _git_run(git_cmd, ["reset", "--hard", f"origin/{branch}"])
                if reset_result.returncode != 0:
                    print(f"✗ Failed to reset to origin/{branch}.")
                    if reset_result.stderr.strip():
                        print(f"  {reset_result.stderr.strip()}")
                    print(
                        f"  Try manually: git fetch origin && git reset --hard origin/{branch}"
                    )
                    sys.exit(1)

        # Post-pull syntax guard: a bad commit that slipped past CI
        # (admin-merge bypass) is caught here and rolled back so the CLI
        # stays bootable until a fix lands.
        syntax_ok, failing_path, syntax_error = _validate_critical_files_syntax(
            _m().PROJECT_ROOT
        )
        if not syntax_ok:
            print()
            print("✗ Pulled code has a syntax error in a critical file:")
            print(f"  {failing_path}")
            if syntax_error:
                # py_compile errors can be multi-line; show the first
                # ~6 lines so the user sees the actual SyntaxError text.
                for line in str(syntax_error).splitlines()[:6]:
                    print(f"    {line}")
            if pre_pull_sha:
                print()
                print(f"→ Rolling back to {pre_pull_sha[:10]}...")
                rollback_result = _git_run(git_cmd, ["reset", "--hard", pre_pull_sha])
                if rollback_result.returncode == 0:
                    print("  ✓ Rollback complete — your install is unchanged.")
                    print("  Try ``hermes update`` again later once a fix lands.")
                else:
                    print("  ✗ Rollback failed. Recover manually with:")
                    print(f"    cd {_m().PROJECT_ROOT} && git reset --hard {pre_pull_sha}")
                    if rollback_result.stderr.strip():
                        print(f"    ({rollback_result.stderr.strip().splitlines()[0]})")
            else:
                print()
                print("  Could not capture pre-pull SHA — recover manually with:")
                print(f"    cd {_m().PROJECT_ROOT} && git reflog && git reset --hard <prev-sha>")
            sys.exit(1)

        update_succeeded = True
    finally:
        if auto_stash_ref is not None:
            # Don't attempt stash restore if the code update itself failed —
            # working tree is in an unknown state.
            if not update_succeeded:
                print(
                    f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})"
                )
                print("  Restore manually with: git stash apply")
            elif discard_local_changes:
                # Non-interactive update + user opted into discarding local
                # source edits (updates.non_interactive_local_changes:
                # discard). Throw the stash away instead of re-applying it.
                _m()._discard_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                )
            elif keep_stash:
                # --keep-stash (desktop updater): the update landed; leave
                # local edits parked in the stash instead of silently
                # re-applying them onto the updated code.
                _m()._park_stashed_changes(auto_stash_ref)
            else:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
    return pre_pull_sha


def _sweep_bytecode_after_update(branch: str) -> None:
    """Clear stale ``__pycache__`` (prevents ImportError on gateway restart when new
    source references names absent from old bytecode), re-stamp the fingerprint
    and refresh the bootstrap cache scripts."""
    removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
    if removed:
        print(
            f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
        )
    _m()._record_bytecode_fingerprint()
    _m()._refresh_bootstrap_cache_scripts(branch)


def _sync_python_dependencies_after_pull(
    git_cmd,
    branch,
    pre_pull_sha,
    *,
    active_lazy_features,
    active_tool_dependencies,
    _windows_gateway_resume,
):
    """Reinstall Python dependencies for the freshly pulled checkout.

    Order matters: ownership preflight -> self-lock deferral -> core-install
    marker -> ``.[all]`` (uv or pip) -> bytecode sweep -> lazy-feature and
    tool-dependency refresh (own marker) -> memory-provider bridge deps ->
    critical-import probe (warn only; stale-bytecode self-heals next launch).
    """
    _refuse_update_if_venv_foreign_owned(_m().PROJECT_ROOT)
    #
    # Self-lock deferral (relocated preflight — #86735): if THIS process
    # holds a native extension the sync must rewrite, defer NOW — after
    # the code swap, so only the dependency install is pending and the
    # next fresh launch completes it via the marker.
    _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
    #
    # Drop the core-install breadcrumb BEFORE touching the venv. If the
    # install is killed mid-flight (Ctrl-C, terminal close, WSL OOM), the
    # marker survives and the next ``hermes`` launch finishes the install
    # via ``_recover_from_interrupted_install``. Cleared after the core
    # ``.[all]`` install completes — lazy refresh uses a separate marker.
    _write_update_incomplete_marker()
    deps_current = _editable_install_is_current(
        git_cmd, _m().PROJECT_ROOT, pre_pull_sha
    )
    if deps_current:
        print("→ Python dependencies unchanged — skipping reinstall")
    else:
        print("→ Updating Python dependencies...")
    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    install_group = "all"

    if uv_bin:
        # Use official managed_python_env() isolation so third-party
        # UV_PYTHON_INSTALL_DIR (e.g. WorkBuddy) cannot hijack uv; then
        # point VIRTUAL_ENV at this install's venv.
        from hermes_cli.managed_uv import managed_python_env

        uv_env = managed_python_env()
        uv_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
        if _m()._is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
            install_group = "termux-all"
            print("  → Termux detected: using uv + curated termux-all optional profile...")
        if not deps_current:
            if _m()._is_termux_env(uv_env) and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat([uv_bin, "pip"], env=uv_env)
            _m()._install_python_dependencies_with_optional_fallback(
                [uv_bin, "pip"], env=uv_env, group=install_group
            )
    else:
        # sys.executable -m pip avoids PEP 668 'externally-managed-environment' errors.
        pip_cmd = [sys.executable, "-m", "pip"]
        _ensure_venv_pip(pip_cmd, sys.executable)
        if _m()._is_termux_env():
            install_group = "termux-all"
            print("  → Termux detected: using curated termux-all optional profile...")
        if not deps_current:
            if _m()._is_termux_env() and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat(pip_cmd)
            _m()._install_python_dependencies_with_optional_fallback(pip_cmd, group=install_group)

    install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
    lazy_env = uv_env if uv_bin else None

    if deps_current:
        # The verification normally runs inside the install we just
        # skipped. Run it here so a wrong skip self-heals into a real
        # install (both verifiers reinstall what they find missing)
        # instead of leaving a venv nobody checked.
        _m()._verify_core_dependencies_installed(
            install_prefix, env=lazy_env, group=install_group
        )
        _m()._verify_console_scripts_installed(install_prefix, env=lazy_env)

    # Core ``.[all]`` install finished. Clear the generic core breadcrumb
    # before the lazy-refresh phase — that phase uses its own marker so a
    # later lazy failure cannot be "healed" by clearing the core marker
    # based on a narrow 7-package import probe (#58004 review).
    _m()._clear_update_incomplete_marker()

    # The update process is still the old Python interpreter process. Run
    # one final cache/module refresh immediately before lazy backend
    # refresh, which imports newly-pulled modules that may depend on fresh
    # symbols in hermes_constants or lazy_deps. The dependency install
    # above may also have regenerated bytecode from build-cache copies —
    # this second sweep catches those stragglers (#60242, #65240).
    _sweep_bytecode_after_update(branch)
    _m()._reload_updated_runtime_modules()

    # Upgrade pip before lazy refreshes — stale pip can fail source builds
    # and leave partially-written packages (#57828).
    _write_lazy_refresh_incomplete_marker()
    _m()._upgrade_pip_before_lazy_refresh(install_prefix, env=lazy_env)

    # Lazy refresh can corrupt the venv when a backend install fails.
    # Clear the lazy marker only when refresh/repair is confirmed healthy.
    lazy_ok = _m()._refresh_active_lazy_features(
        install_prefix,
        env=lazy_env,
        features=active_lazy_features,
    )
    if lazy_ok:
        _m()._clear_lazy_refresh_incomplete_marker()
    else:
        print(
            "  ⚠ Lazy-refresh recovery incomplete — run `hermes` again "
            "to finish import-based venv repair."
        )

    _m()._restore_active_tool_dependencies(
        active_tool_dependencies,
        install_prefix,
        env=lazy_env,
    )

    # Heal the active memory provider's bridge packages last — the core
    # reinstall + lazy refresh above may have stripped or downgraded
    # plugin.yaml-declared deps that aren't in extras (#53272, #70636).
    _m()._refresh_active_memory_provider_dependencies()

    # All transient-ImportError sources have run, so a module that still
    # won't import is real breakage. Warn only — never roll back: `cannot
    # import name X` is also the stale-bytecode signature (#6207, #60242),
    # which _sweep_stale_bytecode_if_checkout_changed() self-heals next launch.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print(f"  ⚠ {failing_module} still fails to import after updating:")
        print(f"      {import_error}")
        print("    Run `hermes update` again — if it persists, reinstall:")
        print("    https://hermes-agent.nousresearch.com")


def _run_post_update_maintenance(
    *,
    assume_yes,
    gateway_mode,
    pre_update_snapshot_id,
    had_desktop_app_before_update,
    node_failures,
    desktop_build_ok,
    pre_update_version,
) -> bool:
    """Post-pull housekeeping that runs once the code + deps are in place.

    state.db integrity restore, catalog/skills/profile syncs, config
    migration, the update summary (whose verdict is returned), and the
    best-effort notices/self-heals (FTS, curator, PATH, launchers, cua-driver).
    Every step is isolated so none can fail the update.
    """
    # ── macOS TCC stale-grant notice (#86385) ──────────────────────
    # Desktop bundles are re-signed each update; grants made to a pre-#73681
    # binary stay stale (toggle ON, yet macOS re-prompts with no Allow
    # button). One line tells affected users how to re-grant once.
    if sys.platform == "darwin" and had_desktop_app_before_update:
        print()
        print(
            "  ℹ macOS: if Hermes re-prompts for permissions you already "
            "granted (toggle shows ON), the stored grant is stale — run "
            "`tccutil reset ScreenCapture com.nousresearch.hermes` (repeat "
            "per affected service), toggle it ON in System Settings, then "
            "fully quit & relaunch once."
        )

    # macOS TCC interpreter anchor (#95596): dylib-complete re-land.
    # Boot-gated — a failed probe leaves the venv untouched.
    try:
        from hermes_cli.macos_tcc_anchor import ensure_tcc_anchor

        ensure_tcc_anchor()
    except Exception:
        logger.debug("macOS TCC anchor refresh skipped", exc_info=True)

    # ── Post-update state.db integrity guard (#68474, #97994) ─────────
    # Check state.db in the root home AND every profile; restore a corrupted
    # one from its own pre-update snapshot instead of silently losing sessions.
    try:
        _verify_and_restore_state_dbs_post_update()
    except Exception as exc:
        logger.debug("Post-update state.db integrity check failed: %s", exc)

    # Seed ~/.hermes/cache/model_catalog.json from the just-pulled
    # website/static/api/model-catalog.json instead of a (bot-gated,
    # flaky) network fetch. Non-fatal: the picker refreshes on next open.
    try:
        from hermes_cli.model_catalog import seed_cache_from_checkout

        if seed_cache_from_checkout(_m().PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")
    except Exception as e:
        logger.debug("Model catalog seed during update failed: %s", e)

    # Sync bundled skills (copies new, updates changed, respects user deletions)
    try:
        print()
        print("→ Syncing bundled skills...")
        _print_bundled_skills_sync_report()
    except Exception as e:
        logger.debug("Skills sync during update failed: %s", e)

    # Sync bundled skills to all profiles (including the active one).
    # seed_profile_skills() uses subprocess with an explicit HERMES_HOME so
    # it is not affected by sync_skills()'s module-level HERMES_HOME cache,
    # which means the active profile is reliably synced regardless of whether
    # the caller's HERMES_HOME env var points at the default or a named profile.
    try:
        from hermes_cli.profiles import (
            list_profiles,
            seed_profile_skills,
        )

        all_profiles = list_profiles()
        if all_profiles:
            print()
            print("→ Syncing bundled skills to all profiles...")
            for p in all_profiles:
                try:
                    r = seed_profile_skills(p.path, quiet=True)
                    if r and r.get("skipped_opt_out"):
                        status = "opted out (--no-skills)"
                    elif r:
                        copied = len(r.get("copied", []))
                        updated = len(r.get("updated", []))
                        modified = len(r.get("user_modified", []))
                        parts = []
                        if copied:
                            parts.append(f"+{copied} new")
                        if updated:
                            parts.append(f"↑{updated} updated")
                        if modified:
                            parts.append(f"~{modified} user-modified")
                        status = ", ".join(parts) if parts else "up to date"
                    else:
                        status = "sync failed"
                    print(f"  {p.name}: {status}")
                except Exception as pe:
                    print(f"  {p.name}: error ({pe})")
    except Exception:
        pass  # profiles module not available or no profiles

    # Backfill per-profile .env files for profiles created before the
    # .env-seeding fix (#44792). Copies the default install's .env so
    # those profiles keep the credentials they were effectively using.
    try:
        from hermes_cli.profiles import backfill_profile_envs

        backfilled = backfill_profile_envs(quiet=True)
        if backfilled:
            print()
            print(
                f"→ Seeded .env for {len(backfilled)} profile(s) "
                f"(copied from default): {', '.join(backfilled)}"
            )
    except Exception:
        pass  # profiles module not available or no profiles

    # Sync Honcho host blocks to all profiles
    try:
        from plugins.memory.honcho.cli import sync_honcho_profiles_quiet

        synced = sync_honcho_profiles_quiet()
        if synced:
            print(f"\n-> Honcho: synced {synced} profile(s)")
    except Exception:
        pass  # honcho plugin not installed or not configured

    # Check for config migrations (#91360).
    _check_and_apply_config_migration(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
    )

    update_complete = _print_update_summary(
        node_failures=node_failures,
        desktop_build_ok=desktop_build_ok,
        pre_update_version=pre_update_version,
    )

    # v23 search-index notice: the compact layout is opt-in (existing
    # indexes are untouched), so surface the command and size win here,
    # only when a legacy index is present.
    try:
        _print_fts_optimize_available_notice()
    except Exception as e:
        logger.debug("FTS optimize notice failed: %s", e)

    # Curator first-run heads-up. Only prints when curator is enabled AND
    # has never run — i.e. the window where the ticker would otherwise
    # have fired against a fresh skill library. Kept silent on steady
    # state so we don't nag.
    try:
        _print_curator_first_run_notice()
    except Exception as e:
        logger.debug("Curator first-run notice failed: %s", e)

    # Latest curator run notice (rename map `old-name → umbrella`),
    # self-stamped so it shows once per run.
    try:
        _print_curator_recent_run_notice()
    except Exception as e:
        logger.debug("Curator recent-run notice failed: %s", e)

    # Repair RHEL-family root installs where /usr/local/bin isn't on PATH
    # for non-login interactive shells.  No-op on every other platform.
    try:
        _ensure_fhs_path_guard()
    except Exception as e:
        logger.debug("FHS PATH guard check failed: %s", e)

    # Self-heal the hermes-acp launcher for installs that predate it, so
    # ACP hosts (Zed, JetBrains, Buzz) can resolve Hermes on PATH without
    # a reinstall.  No-op on Windows (the launcher migration below owns
    # that) and when already present.
    try:
        _ensure_acp_launcher()
    except Exception as e:
        logger.debug("hermes-acp launcher self-heal failed: %s", e)

    # Migrate/repair Windows launchers into the managed bin dir. In-checkout
    # launchers (hermes-agent\bin) were swept by the pre-update autostash
    # (--include-untracked) and with --keep-stash never restored, so
    # `hermes` stopped resolving. Updates never run install.ps1, so this is
    # how existing installs reach the new layout. No-op on POSIX/source checkouts.
    try:
        from hermes_cli._install_repair import migrate_windows_bin_path

        migrate_windows_bin_path(_m().PROJECT_ROOT)
    except Exception as e:
        logger.debug("Windows bin launcher migration failed: %s", e)

    # Refresh cua-driver (Computer Use) — no-op unless already on PATH.
    # Tied to ``hermes update`` for a predictable cadence without a
    # per-launch GitHub API call.
    try:
        refresh_cua_driver = True
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                refresh_cua_driver = bool(
                    _update_cfg.get("refresh_cua_driver", True)
                )
        except Exception as cfg_exc:
            logger.debug("Could not read updates.refresh_cua_driver: %s", cfg_exc)

        if (
            refresh_cua_driver
            and sys.platform in ("darwin", "win32", "linux")
            and shutil.which("cua-driver")
        ):
            from hermes_cli.tools_config import install_cua_driver

            print()
            print("→ Refreshing cua-driver (Computer Use)...")
            # require_confirmed_update: run the slow silent installer only
            # when check-update positively reports a newer release; an
            # indeterminate check keeps the current version (`hermes update`
            # must stay fast; `computer-use install --upgrade` is the force
            # path). Windows defers even confirmed updates there because the
            # installer may need console/UAC consent.
            install_cua_driver(
                upgrade=True,
                require_confirmed_update=True,
                show_installer_progress=False,
            )
    except Exception as e:
        logger.debug("cua-driver refresh failed: %s", e)
    return update_complete


def _resume_windows_gateways_and_merge_outcome(outcome, _windows_gateway_resume, gateway_mode: bool):
    """Resume the gateways paused for a Windows update and fold that outcome into ``outcome``.

    Feeds the pause/resume token's relaunched/restarted/pending lists into
    the same bookkeeping the systemd/launchd phase populates so the
    plan-vs-execution reconciliation never reports a healthy Windows gateway
    as unaccounted. Best-effort: must never itself abort the update.
    """
    try:
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
    except Exception as _windows_resume_exc:
        outcome.incomplete = True
        outcome.phase_errors.append(str(_windows_resume_exc))
        print(
            "  ⚠ Windows gateway service restart incomplete: "
            f"{_windows_resume_exc}"
        )
        if gateway_mode:
            _write_gateway_update_exit_code(False)

    if isinstance(_windows_gateway_resume, dict):
        # Fold the Windows pause/resume outcome into the same bookkeeping
        # the systemd/launchd phase fills, so the #91277 reconciliation
        # doesn't flag a healthy Windows gateway as "unaccounted". Genuinely
        # failed relaunches are left out of the token so they still surface.
        # Best-effort: must never abort the update.
        try:
            for _win_profile in _windows_gateway_resume.get("relaunched_profiles") or []:
                if _win_profile not in outcome.relaunched_profiles:
                    outcome.relaunched_profiles.append(_win_profile)
        except Exception as _win_reconcile_exc:
            logger.debug(
                "Could not merge Windows relaunch outcome into fleet "
                "reconciliation bookkeeping: %s",
                _win_reconcile_exc,
            )
        windows_restarted = list(
            _windows_gateway_resume.get("restarted_services") or []
        )
        for service_name in windows_restarted:
            if service_name not in outcome.restarted_services:
                outcome.restarted_services.append(service_name)
        service_profiles = _windows_gateway_resume.get("service_profiles") or {}
        for service_name in windows_restarted:
            profile_name = service_profiles.get(service_name)
            if profile_name and profile_name not in outcome.relaunched_profiles:
                outcome.relaunched_profiles.append(profile_name)
        pending_services = list(_windows_gateway_resume.get("services") or [])
        for service_name in pending_services:
            label = str(service_profiles.get(service_name) or service_name)
            if label not in outcome.failed_or_stale_units:
                outcome.failed_or_stale_units.append(label)

        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=outcome.restarted_services,
                relaunched_profiles=outcome.relaunched_profiles,
                externally_supervised_profiles=outcome.externally_supervised_profiles,
                killed_pids=sorted(outcome.killed_pids),
                failed_units=outcome.failed_or_stale_units,
                incomplete=(
                    outcome.incomplete
                    or bool(outcome.failed_or_stale_units)
                ),
                phase_error="; ".join(outcome.phase_errors) or None,
            )
        except Exception:
            pass


def _verify_fleet_after_update(
    restart,
    *,
    _pre_update_plan,
    _windows_gateway_resume,
    node_failures,
    update_complete,
):
    """Post-restart verification: legacy-unit warning, dashboard cleanup, stale serve
    probe, fleet version matrix, plan-vs-execution reconciliation, receipt finalize.

    Exits 1 (leaving ``fleet_restart_pending`` in place for the next catch-up)
    when any gateway may still be serving pre-update code; otherwise clears
    the pending marker.
    """
    # Legacy hermes.service + hermes-gateway.service SIGTERM-fight over the
    # same bot token (PR #11909); warn on every update until migrated.
    try:
        from hermes_cli.gateway import (
            has_legacy_hermes_units,
            _find_legacy_hermes_units,
            supports_systemd_services,
        )

        if supports_systemd_services() and has_legacy_hermes_units():
            print()
            print("⚠ Legacy Hermes gateway unit(s) detected:")
            for name, path, is_sys in _find_legacy_hermes_units():
                scope = "system" if is_sys else "user"
                print(f"    {path}  ({scope} scope)")
            print()
            print("  These pre-rename units (hermes.service) fight the current")
            print("  hermes-gateway.service for the bot token and cause SIGTERM")
            print("  flap loops. Remove them with:")
            print()
            print("    hermes gateway migrate-legacy")
            print()
            print("  (add `sudo` if any are in system scope)")
    except Exception as e:
        logger.debug("Legacy unit check during update failed: %s", e)

    # Restart a managed dashboard via systemd or stop stale manual ones
    # (raw-killing a systemd-owned PID reads as a clean stop and leaves the
    # Cloudflare origin dead). A failed Node refresh leaves the running
    # dashboard untouched. Already-restarted units (incl. hermes-serve*,
    # #83438) are forwarded so they aren't restarted twice (#83595).
    _finish_dashboard_update_cleanup(
        node_failures, already_restarted_units=set(restart.restarted_services)
    )

    # Success-path twin of the abort-recovery probe (#100479): the restart
    # phase only touches units, so a unit-less `hermes serve` keeps its
    # pre-update sys.modules and its cron ticker ImportErrors. Runs AFTER
    # dashboard cleanup so a respawned manual dashboard isn't a survivor.
    # Rows feed the reconciliation below (survivor → exit 1); ``None``
    # means the probe failed and reconciliation stays fail-closed.
    _stale_serve_rows: "list | None" = None
    try:
        _stale_serve_rows = _surviving_pre_update_serve_runtimes(_pre_update_plan)
        if _stale_serve_rows:
            _warn_stale_serve_runtimes(_stale_serve_rows)
    except Exception as _serve_warn_exc:
        logger.debug("Failed to check for surviving serve runtimes: %s", _serve_warn_exc)

    print()
    print("Tip: You can now select a provider and model:")
    print("  hermes model              # Select provider and model")

    # Phase 1 (#91277): post-update fleet version verification. Compare
    # every live gateway's stamped code_sha against the freshly-updated
    # checkout and surface any gateway still serving pre-update code —
    # instead of assuming the restart phase worked (#88654, #69754).
    _fleet_snapshot: list = []
    try:
        from hermes_cli.update_receipt import (
            collect_fleet_versions,
            print_fleet_version_matrix,
        )

        # Cross-platform "we expected fleet rows" signal (#93406). The
        # old (restart.restarted_services or restart.killed_pids) condition never fires
        # on Windows: the pause/resume phase populates neither list, so
        # a healthy resumed gateway yielded zero rows and exit 0.
        _fleet_rows_expected = _m()._fleet_probe_expected_runtimes(
            _pre_update_plan,
            restart.pre_restart_gateway_pids,
            _windows_gateway_resume,
            restart.restarted_services,
            restart.killed_pids,
        )
        # Settle window (skipped when nothing was running): restarted
        # gateways need time to rewrite gateway_state.json. Windows resumes
        # DETACHED and may take ~10s to boot, so a single 2s sleep reported
        # "no rows" (exit 1) on healthy resumes and the retry re-killed the
        # new gateway. Poll a bounded window instead.
        _fleet_snapshot = []
        if _fleet_rows_expected:
            _fleet_deadline = _time.monotonic() + 30.0
            while True:
                _time.sleep(2.0)
                # Pass the pre-restart PID snapshot so a gateway the
                # restart phase stopped WITHOUT a verified replacement
                # shows as a DOWN row (exit 1) instead of silently
                # producing no row at all.
                _fleet_snapshot = collect_fleet_versions(
                    pre_restart_pids=restart.pre_restart_gateway_pids
                )
                # A "down" row may just be a detached replacement still
                # booting; keep polling until no "down" rows remain or the
                # deadline passes, so a slow gateway isn't misread.
                if _fleet_snapshot and not any(
                    row.get("state") == "down" for row in _fleet_snapshot
                ):
                    break
                if _time.monotonic() >= _fleet_deadline:
                    break
        else:
            _fleet_snapshot = collect_fleet_versions(
                pre_restart_pids=restart.pre_restart_gateway_pids
            )
        if print_fleet_version_matrix(_fleet_snapshot):
            restart.incomplete = True
        elif not _fleet_snapshot and _fleet_rows_expected:
            # Zero rows although a gateway was (or may have been) live
            # pre-update. collect_fleet_versions() swallows every failure,
            # so an empty list is indistinguishable from a healthy fleet —
            # treat it as verification failure (receipt "partial", exit 1) (#93406).
            print(
                "\n⚠ Fleet version check returned no rows even though"
                " gateway runtimes were expected — verification incomplete."
            )
            restart.incomplete = True
    except Exception as _fleet_exc:
        logger.debug("Fleet version verification failed: %s", _fleet_exc)

    # Plan-vs-execution reconciliation (#91277): every runtime the PLAN saw
    # must appear in the restart bookkeeping; an unaccounted one is a
    # silent miss and escalates like a STALE/DOWN row.
    _runtime_outcomes: list = []
    try:
        if _pre_update_plan is not None and _pre_update_plan.runtimes:
            from hermes_cli.update_inventory import (
                match_runtime_outcomes,
                report_unaccounted_runtimes,
            )

            _runtime_outcomes = match_runtime_outcomes(
                _pre_update_plan,
                restarted_services=restart.restarted_services,
                relaunched_profiles=restart.relaunched_profiles,
                externally_supervised_profiles=restart.externally_supervised_profiles,
                killed_pids=restart.killed_pids,
                failed_units=restart.failed_or_stale_units,
                # Serve/dashboard runtimes reconcile by incarnation
                # liveness, not by the gateway's unit names (#100479).
                stale_serve_pids=(
                    {row.get("pid") for row in _stale_serve_rows}
                    if _stale_serve_rows is not None
                    else None
                ),
            )
            if report_unaccounted_runtimes(_runtime_outcomes):
                restart.incomplete = True
            try:
                import hermes_cli.update_receipt as _ur

                if _ur._current is not None:
                    _ur._current.data["runtime_outcomes"] = _runtime_outcomes
            except Exception:
                pass
    except Exception as _outcome_exc:
        logger.debug("Runtime-outcome reconciliation failed: %s", _outcome_exc)

    try:
        from hermes_cli.update_receipt import finalize_update_receipt

        _receipt_path = finalize_update_receipt(
            (
                "partial"
                if restart.incomplete or not update_complete
                else "success"
            ),
            fleet=_fleet_snapshot,
        )
        if _receipt_path is not None:
            logger.info("Update receipt written: %s", _receipt_path)
    except Exception as _receipt_exc:
        logger.debug("Update receipt finalize failed: %s", _receipt_exc)

    if restart.incomplete:
        # Code update itself succeeded, but at least one gateway still
        # runs pre-update modules — surface that as a failed update so
        # automation / operators do not treat the fleet as healthy.
        # Leave ``fleet_restart_pending`` in place so the next
        # ``hermes update`` still runs the catch-up restart.
        sys.exit(1)
    _clear_fleet_restart_pending_marker()


def _clear_windows_venv_holders_or_exit(args, gateway_mode: bool, _windows_gateway_resume):
    """Windows: stop every venv-python holder we can positively identify, else exit 2.

    Rungs, in order: leftover pausable gateways -> ledger-identified orphaned
    backends -> orphaned Desktop backends -> ledger-identified manual serve
    (relaunched at exit on the same bind) -> GUI-updater hand-off leaks.
    Anything still holding the venv afterwards is refused (the sync would
    corrupt against a locked .pyd) and the paused gateways are resumed.
    """
    _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        _gateway_holders = _m()._leftover_pausable_gateway_pids(_venv_holders)
        if _gateway_holders is not None:
            if _refuse_gateway_ancestor_tree_kill(
                _gateway_holders, gateway_mode=gateway_mode
            ):
                _m()._resume_windows_gateways_after_update(
                    _windows_gateway_resume
                )
                sys.exit(2)
            # Remaining holders are gateways the pause machinery owns
            # (supervisor respawn in the pause→guard window, or an unmapped
            # spawn path). Stop and re-check; the post-update resume brings them back.
            from gateway.status import get_process_start_time, terminate_pid

            print(
                f"  ⚠ {len(_gateway_holders)} gateway process(es) still "
                "hold the venv after the pause; stopping them"
            )
            for _pid in _gateway_holders:
                try:
                    pid_int = int(_pid)
                    terminate_pid(
                        pid_int,
                        force=True,
                        expected_start_time=get_process_start_time(pid_int),
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not stop leftover gateway %s: %s", _pid, exc
                    )
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Positive-identity rung (runs FIRST, any update context): holders
        # the spawn ledger proves are orphaned Hermes backends — the
        # process self-registered (pid, create_time, purpose, spawner) at
        # startup and its recorded spawner is provably dead. No PPID
        # archaeology, no hand-off contract required.
        _ledger_backends = _m()._ledger_reapable_backend_pids(_venv_holders)
        if _ledger_backends:
            print(
                f"  ⚠ {len(_ledger_backends)} ledger-identified orphaned "
                "Hermes backend process(es) hold the venv; stopping their trees"
            )
            _m()._stop_process_trees(_ledger_backends)
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        _orphan_backends = _m()._orphaned_desktop_backend_pids(_venv_holders)
        if _orphan_backends:
            # Remaining holders are Desktop `serve` backends whose app is
            # GONE (Electron lost the SIGTERM race on teardown). Nothing
            # respawns an orphan, so reap the tree and re-check. Backends
            # with a live Desktop never reach here (returns None) — the app
            # would just respawn what we kill, so that path keeps refusing.
            print(
                f"  ⚠ {len(_orphan_backends)} orphaned Desktop backend "
                "process(es) still hold the venv; stopping their trees"
            )
            _m()._stop_process_trees(_orphan_backends)
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Manual serve/dashboard rung (#63206): a `hermes serve --host <ip>`
        # powering a REMOTE Desktop used to dead-end the update with exit 2.
        # Ledger identity only (spawner not alive; Desktop-owned backends
        # keep the refusal). Stop them and register an idempotent atexit
        # relaunch on the SAME host/port/profile — success or failure.
        _serve_entries = _m()._ledger_manual_serve_holders(_venv_holders)
        if _serve_entries:
            print(
                f"  ⚠ {len(_serve_entries)} manual serve/dashboard "
                "backend(s) hold the venv; stopping them for the update "
                "(they will be relaunched on their recorded endpoints)"
            )
            _m()._stop_process_trees(
                [int(e["pid"]) for e in _serve_entries]
            )
            _serve_resume_token = {
                "pending": True,
                "entries": _serve_entries,
            }
            _record_update_step(
                "serve_pause",
                True,
                f"stopped={len(_serve_entries)}",
            )
            import atexit as _serve_atexit

            _serve_atexit.register(
                _m()._relaunch_stopped_serves, _serve_resume_token
            )
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Final rung: a GUI-updater hand-off (`update --gateway --force` with
        # the update-incomplete marker) means the Desktop is contractually
        # gone and nothing legitimate respawns a `serve` backend. The
        # orphan-only reap bails on ANY live parent (mid-teardown Electron,
        # launcher→worker chain), which hung updates; in hand-off context
        # surviving backends are leaks regardless — reap by cmdline.
        _handoff = False
        try:
            _handoff = bool(getattr(args, "gateway", False)) and _m()._update_marker_path().exists()
        except Exception:
            _handoff = False
        # Fail closed: if we cannot positively verify the shim state
        # (scripts dir unresolvable, detection raised), assume a live
        # shim exists and keep refusing rather than reap.
        _no_live_shim = False
        try:
            _scripts_dir = _m()._venv_scripts_dir()
            if _scripts_dir is not None:
                _no_live_shim = not _m()._detect_concurrent_hermes_instances(_scripts_dir)
        except Exception:
            _no_live_shim = False
        if _handoff and _no_live_shim:
            _handoff_backends = _m()._handoff_reapable_backend_pids(_venv_holders)
            if _handoff_backends:
                print(
                    f"  ⚠ {len(_handoff_backends)} Hermes backend process(es) "
                    "still hold the venv after the Desktop hand-off; "
                    "stopping their trees"
                )
                _m()._stop_process_trees(_handoff_backends)
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        print(_format_venv_python_holders_message(_venv_holders))
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        sys.exit(2)


@dataclass
class _CheckoutPlan:
    """What the pre-pull checkout phase decided (see ``_prepare_checkout_for_update``)."""

    auto_stash_ref: "str | None"
    commit_count: int
    in_place_update: bool
    parked_branch_switched: bool
    prompt_for_restore: bool
    switch_block_reason: "str | None"
    upstream_checked: bool



def _prepare_checkout_for_update(
    git_cmd,
    branch,
    current_branch,
    *,
    is_fork,
    assume_yes,
    gateway_mode,
    gw_input_fn,
    switch_branch,
    _windows_gateway_resume,
):
    """Apply the parked-branch guard, land on the update target, stash, and count new commits.

    Exits the process when the checkout is unsafe to move or the target branch
    does not exist. ``commit_count`` is 0 when up to date, -1 when tips differ
    but the shallow count is unrecoverable.
    """
    switch_block_reason = None  # only meaningful when parked_branch_switched
    # Parked-branch guard: a checkout parked on a stale feature branch
    # used to stash-switch-pull-switch-back, "updating" main while the
    # running code stayed behind. Routing by branch contents +
    # updates.parked_branch_strategy:
    #   fully merged  -> switch back to the target.
    #   unmerged: N   -> "switch" (default): switch anyway (commits are
    #                    safe on the branch) with a loud "kept" notice;
    #                    deterministic for non-interactive callers.
    #                    "update_in_place": merge origin/<target> INTO
    #                    the branch — checkout never moves, local commits
    #                    survive. --switch-branch overrides for one run.
    #   anything else -> dirty/unverifiable/opted out: touch nothing,
    #                    warn, mark the code update SKIPPED, stop.
    parked_branch_switched = False
    in_place_update = False
    if current_branch != branch and current_branch != "HEAD":
        switch_safe, switch_block_reason = _m()._assess_parked_branch_switch(
            git_cmd, _m().PROJECT_ROOT, current_branch, branch
        )
        if not switch_safe:
            _m()._print_parked_branch_skip_warning(
                git_cmd,
                _m().PROJECT_ROOT,
                current_branch,
                branch,
                switch_block_reason,
            )
            print()
            print(
                "⚠ Update finished — code update SKIPPED"
                f"{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}"
            )
            _m()._resume_windows_gateways_after_update(
                _windows_gateway_resume
            )
            sys.exit(1)
        if switch_block_reason.startswith("unmerged:"):
            _in_place_configured = False
            try:
                from hermes_cli.config import load_config as _load_cfg

                _upd_cfg = (_load_cfg() or {}).get("updates", {})
                _in_place_configured = (
                    isinstance(_upd_cfg, dict)
                    and _upd_cfg.get("parked_branch_strategy", "switch")
                    == "update_in_place"
                )
            except Exception as exc:
                logger.debug(
                    "Could not read updates.parked_branch_strategy: %s", exc
                )
            if _in_place_configured and not switch_branch:
                # The merge source must exist upstream; --branch typos
                # previously surfaced through the checkout failing, which
                # does not run on this path.
                verify_ref = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", f"origin/{branch}"])
                if verify_ref.returncode != 0:
                    print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                    sys.exit(1)
                in_place_update = True
                print(
                    f"  ℹ On branch '{current_branch}' — updating it in place from "
                    f"origin/{branch} (no branch switch; local commits preserved)."
                )
            else:
                parked_branch_switched = True
                _m()._print_parked_branch_kept_notice(
                    current_branch,
                    branch,
                    switch_block_reason.split(":", 1)[1],
                )
        else:
            parked_branch_switched = True
            print(
                f"  ⚠ Checkout was parked on '{current_branch}' "
                f"(fully merged) — switching back to {branch}..."
            )

    if not in_place_update and current_branch != branch:
        if current_branch == "HEAD":
            print(
                f"  ⚠ Currently on detached HEAD — switching to {branch} "
                "for update..."
            )
        # Stash before checkout so uncommitted work isn't lost
        auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)
        checkout_result = _git_run(git_cmd, ["checkout", branch])
        if checkout_result.returncode != 0:
            # Branch not local yet — set it up tracking origin/<branch>.
            track_result = _git_run(git_cmd, ["checkout", "-B", branch, f"origin/{branch}"])
            if track_result.returncode != 0:
                # Restore the user's prior stash before bailing
                # so we don't leave them stranded in a weird state.
                if auto_stash_ref is not None:
                    _m()._restore_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                        prompt_user=False,
                        input_fn=gw_input_fn,
                    )
                print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                if track_result.stderr.strip():
                    print(f"  {track_result.stderr.strip().splitlines()[0]}")
                sys.exit(1)
    else:
        auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)

    prompt_for_restore = (
        auto_stash_ref is not None
        and not assume_yes
        and (gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty()))
    )

    # Check if there are updates. On shallow checkouts `rev-list --count`
    # walks the truncated graph and can report the entire remote ancestry
    # (e.g. "Found 9980 new commit(s)" on a depth-1 install — #53479).
    # The zero/nonzero gate is still sound (HEAD == origin/<branch> counts
    # 0), so keep it, but treat the shallow NUMBER as unknown and recover
    # the real one via the GitHub compare API when possible.
    result = _git_run(git_cmd, ["rev-list", f"HEAD..origin/{branch}", "--count"], check=True)
    commit_count = int(result.stdout.strip())

    apply_is_shallow = (
        _git_run(git_cmd, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
        == "true"
    )
    if commit_count > 0 and apply_is_shallow:
        from hermes_cli.banner import _github_compare_behind

        head_sha = _git_run(git_cmd, ["rev-parse", "HEAD"]).stdout.strip()
        target_sha = _git_run(git_cmd, ["rev-parse", f"origin/{branch}"]).stdout.strip()
        counted = _github_compare_behind(head_sha, target_sha)
        # counted == 0 means local-ahead (remote tip reachable from HEAD):
        # not behind, fall through to the up-to-date path.
        commit_count = counted if counted is not None else -1

    # A fork can match origin yet trail upstream, so the upstream sync can
    # move HEAD with commit_count == 0. Detect that BEFORE the no-update
    # return so deps, restarts AND the fleet matrix still run (#73108 —
    # the sync used to live inside the early-return branch and verified
    # nothing). Non-forks have no upstream question.
    upstream_checked = True
    if commit_count == 0 and is_fork and branch == "main":
        pre_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        upstream_checked = _m()._sync_with_upstream_if_needed(
            git_cmd,
            _m().PROJECT_ROOT,
            assume_yes=assume_yes,
            input_fn=gw_input_fn,
        )
        post_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if pre_sync_sha and post_sync_sha and pre_sync_sha != post_sync_sha:
            synced_count = _count_commits_between(
                git_cmd,
                _m().PROJECT_ROOT,
                pre_sync_sha,
                post_sync_sha,
            )
            # HEAD moving is itself proof of an update. Keep the update
            # path active even if the informational count cannot be read.
            commit_count = max(1, synced_count)

    return _CheckoutPlan(
        auto_stash_ref=auto_stash_ref,
        commit_count=commit_count,
        in_place_update=in_place_update,
        parked_branch_switched=parked_branch_switched,
        prompt_for_restore=prompt_for_restore,
        switch_block_reason=switch_block_reason,
        upstream_checked=upstream_checked,
    )


def _cmd_update_impl(args, gateway_mode: bool):
    """Body of ``cmd_update`` — kept separate so the wrapper can always
    restore stdio even on ``sys.exit``."""
    # A managed-runtime refresh can replace site-packages before the normal
    # ``.[all]`` install runs. Snapshot while the old environment can still
    # prove which optional backends the user had activated.
    active_lazy_features = _m()._capture_active_lazy_features()
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    # Snapshot the pre-update version before any code is pulled so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (
        (lambda prompt, default="": _gateway_prompt(prompt, default))
        if gateway_mode
        else None
    )
    assume_yes = bool(getattr(args, "yes", False))
    # --keep-stash (desktop updater): stash local changes so the update can
    # proceed, but never re-apply them afterward — they stay parked in git
    # stash. Only applies when an update actually landed; abort/no-op paths
    # still restore, since the tree they restore onto is unchanged.
    keep_stash = bool(getattr(args, "keep_stash", False))
    # --switch-branch: on a branch carrying unmerged commits, prefer switching
    # to the update target over an in-place merge, so the branch's history is
    # never written to by an update (#89507 review feedback). Only meaningful
    # when updates.parked_branch_strategy is "update_in_place".
    switch_branch = bool(getattr(args, "switch_branch", False))

    # Whether this update is running without a human at the keyboard.
    # Interactive terminal updates always stash-and-ask (unchanged behavior);
    # only non-interactive updates (desktop/chat app, gateway, `--yes`) consult
    # the `updates.non_interactive_local_changes` config setting to decide
    # whether to auto-restore stashed local source changes or throw them away.
    _non_interactive_update = (
        gateway_mode
        or assume_yes
        or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    discard_local_changes = False
    if _non_interactive_update:
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                _mode = str(_update_cfg.get("non_interactive_local_changes", "stash")).lower()
                discard_local_changes = _mode == "discard"
        except Exception as exc:
            # Never let a config read failure change the safe default.
            logger.debug("Could not read updates.non_interactive_local_changes: %s", exc)
            discard_local_changes = False

    print("⚕ Updating Hermes Agent...")
    print()

    # Phase 1 (#91277): structured update receipt — record what this run
    # discovers, does, and skips, so silent-failure classes (#88848,
    # #74973, #85753, #81193) become diagnosable from disk.
    try:
        from hermes_cli.update_receipt import begin_update_receipt

        begin_update_receipt()
    except Exception as _receipt_exc:
        logger.debug("Update receipt unavailable: %s", _receipt_exc)

    # Plan phase (#91277): snapshot every running runtime, supervisor and
    # code version into the receipt (read-only; probe failure records
    # nothing). Re-read AFTER the restart phase to reconcile planned
    # runtimes against bookkeeping — the plan is the worklist.
    _pre_update_plan = None
    try:
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            record_plan_in_receipt,
        )

        _pre_update_plan = collect_runtime_inventory()
        record_plan_in_receipt(_pre_update_plan)
        if _pre_update_plan.runtimes:
            _n = len(_pre_update_plan.runtimes)
            _profiles = ", ".join(
                sorted({r.profile for r in _pre_update_plan.runtimes})
            )
            print(f"→ Fleet: {_n} running service(s) across profiles: {_profiles}")
    except Exception as _plan_exc:
        logger.debug("Update plan phase failed: %s", _plan_exc)

    # Windows: abort if another hermes.exe holds the venv shim — continuing
    # yields WinError 32 spam and a deferred-rename leftover or silent ZIP
    # fallback (#26670). Exception (#37039): instances positively identified
    # as gateways are paused by ``_pause_windows_gateways_for_update`` below
    # and restarted afterwards; anything else (TUI, Desktop backend,
    # unreadable cmdline) still aborts.
    if _m()._is_windows() and not getattr(args, "force", False):
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            concurrent = _m()._detect_concurrent_hermes_instances(scripts_dir)
            if concurrent:
                non_gateway = _m()._filter_non_gateway_concurrent_instances(
                    concurrent
                )
                if non_gateway:
                    print(
                        _format_concurrent_instances_message(
                            non_gateway, scripts_dir
                        )
                    )
                    sys.exit(2)

    # Pre-update backup — runs before any git/file mutation so users can
    # always roll back to the exact state they had before this update.
    # Returns the quick-snapshot id (or None when disabled/failed); the
    # post-update cron-jobs safety net uses it to detect job loss.
    pre_update_snapshot_id = _m()._run_pre_update_backup(args)
    _record_update_step(
        "pre_update_backup",
        pre_update_snapshot_id is not None,
        f"snapshot={pre_update_snapshot_id}" if pre_update_snapshot_id else "disabled or failed",
    )

    _windows_gateway_resume = _m()._pause_windows_gateways_for_update()
    if _windows_gateway_resume:
        import atexit as _atexit

        _atexit.register(
            _m()._resume_windows_gateways_after_update,
            _windows_gateway_resume,
        )

    # With gateways paused, any venv python still running (typically the
    # Desktop `hermes serve` backend) keeps .pyd files locked and would
    # corrupt the sync; refuse rather than race (the app respawns a killed
    # backend). NOT bypassed by --force: the desktop updater passes it to
    # skip the shim guard but only probes the shim and app.asar.
    # --force-venv is the explicit escape hatch.
    if _m()._is_windows() and not getattr(args, "force_venv", False):
        _clear_windows_venv_holders_or_exit(args, gateway_mode, _windows_gateway_resume)

    # Self-lock deferral moved: the venv-holder sweep above excludes this
    # process by design (a CLI `hermes update` IS the venv python), and an
    # updater that has imported a native venv extension cannot rewrite its
    # own mapped .pyd (#83569). That check used to run HERE — before the
    # fetch — but firing pre-fetch meant a deferral stranded the user on the
    # OLD checkout, and any startup path that eagerly loaded cryptography
    # turned every Windows update into an exit-2 loop (#86735/#86780/#86781).
    # It now runs via _abort_dependency_sync_if_self_locked() after the code
    # swap, immediately before the dependency sync — the only phase the lock
    # can actually break — and only when the sync would truly rewrite the
    # loaded distribution.

    # Capture this after every fail-closed venv guard, but before either
    # update path can remove the ignored release tree.
    desktop_dir = _m().PROJECT_ROOT / "apps" / "desktop"
    had_desktop_app_before_update = _desktop_app_present(desktop_dir)

    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = _m().PROJECT_ROOT / ".git"

    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            print("✗ Not a git repository. Please reinstall:")
            print(
                "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
            )
            sys.exit(1)

    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists():
        subprocess.run(
            [
                "git",
                "-c",
                "windows.appendAtomically=false",
                "config",
                "windows.appendAtomically",
                "false",
            ],
            cwd=_m().PROJECT_ROOT,
            check=False,
            capture_output=True,
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
    # A broken Git-for-Windows trampoline refuses every git call with a
    # "BUG (fork bomb)" guard instead of running; swap in a real binary up
    # front so the normal git path survives instead of degrading to ZIP
    # (#87876).
    git_cmd = _ensure_non_trampoline_git(git_cmd)

    # Discard npm lockfile churn before stash/branch logic: npm rewrites
    # package-lock.json non-deterministically, which is never an intentional
    # edit on a managed install but forces an autostash every update.
    _discard_lockfile_churn(git_cmd, _m().PROJECT_ROOT)
    # Same rationale, different generator: line-ending churn is machine-made
    # dirt on a managed checkout, so clear it (and stop generating it) before
    # the stash/branch logic rather than autostashing the entire tree.
    _normalize_managed_eol(git_cmd, _m().PROJECT_ROOT)

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _m()._get_origin_url(git_cmd, _m().PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        try:
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
        finally:
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        if gateway_mode:
            _write_gateway_update_exit_code(desktop_build_ok)
        return

    # Fetch and pull
    try:

        # Resolve the branch first so the fetch is scoped: a bare `git fetch
        # origin` pulls thousands of auto-generated branches and can stall for minutes.
        branch = _m()._resolve_update_branch(args)

        # Self-heal abandoned git lock files (e.g. .git/shallow.lock left by a
        # crashed fetch) before the fetch — otherwise the update fails with
        # "Unable to create .../shallow.lock: File exists" and never reaches
        # the network.
        from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

        cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
        if cleared:
            print("  (removed stale git lock(s): %s)" % ", ".join(cleared))
        swept = clear_stale_tmp_packs(_m().PROJECT_ROOT)
        if swept:
            print("  (removed %d aborted-fetch pack temp file(s))" % len(swept))

        # Surface autostash entries left behind by earlier updates (#63717
        # problem 6) — parked --keep-stash runs and failed restores preserve
        # the stash but nothing ever mentioned it again.
        _m()._warn_orphaned_update_autostashes(git_cmd, _m().PROJECT_ROOT)

        print("→ Fetching updates...")
        fetch_result = _git_run(git_cmd, ["fetch", "origin", branch], network=True)
        if fetch_result.returncode != 0:
            _print_fetch_failure(fetch_result.stderr)
            sys.exit(1)

        # Get current branch (returns literal "HEAD" when detached)
        result = _git_run(git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"], check=True)
        current_branch = result.stdout.strip()

        _plan = _prepare_checkout_for_update(
            git_cmd,
            branch,
            current_branch,
            is_fork=is_fork,
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            gw_input_fn=gw_input_fn,
            switch_branch=switch_branch,
            _windows_gateway_resume=_windows_gateway_resume,
        )
        auto_stash_ref = _plan.auto_stash_ref
        commit_count = _plan.commit_count
        in_place_update = _plan.in_place_update
        parked_branch_switched = _plan.parked_branch_switched
        prompt_for_restore = _plan.prompt_for_restore
        switch_block_reason = _plan.switch_block_reason
        upstream_checked = _plan.upstream_checked

        if commit_count == 0:
            _invalidate_update_cache()

            # Restore stash and switch back to original branch if we moved.
            # EXCEPTION: a parked feature branch we verified clean + fully
            # merged stays on the target — re-parking the checkout on the
            # stale branch is the 2026-08-17 incident all over again.
            if auto_stash_ref is not None:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if parked_branch_switched:
                if switch_block_reason.startswith("unmerged:"):
                    _count = switch_block_reason.split(":", 1)[1]
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' — "
                        f"switched back to {branch}; {_count} unmerged "
                        f"commit(s) kept on '{current_branch}'."
                    )
                else:
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' (fully "
                        f"merged) — switched back to {branch}."
                    )
            elif current_branch not in {branch, "HEAD"}:
                _git_run(git_cmd, ["checkout", current_branch])

            current_checkout_complete = _repair_current_checkout(
                assume_yes=assume_yes,
                gateway_mode=gateway_mode,
                pre_update_snapshot_id=pre_update_snapshot_id,
                desktop_dir=desktop_dir,
                had_desktop_app_before_update=had_desktop_app_before_update,
                active_lazy_features=active_lazy_features,
                active_tool_dependencies=active_tool_dependencies,
                upstream_checked=upstream_checked,
                _windows_gateway_resume=_windows_gateway_resume,
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            # A prior pull may still owe the fleet a restart (#95294); catch
            # up even on the "Already up to date" path, and BEFORE the exit
            # gate below so a partial outcome can't strand the fleet on stale
            # code (#91277 fleet contract).
            _apply_pending_fleet_restart_catchup()
            if not current_checkout_complete:
                if gateway_mode:
                    _write_gateway_update_exit_code(False)
                try:
                    from hermes_cli.update_receipt import finalize_update_receipt

                    finalize_update_receipt("partial")
                except Exception as _receipt_exc:
                    logger.debug(
                        "Update receipt finalize (current checkout) failed: %s",
                        _receipt_exc,
                    )
                sys.exit(1)
            return

        if commit_count > 0:
            print(f"→ Found {commit_count} new commit(s)")
        else:
            # Shallow checkout, exact count unrecoverable (offline/rate-limited
            # compare API) — the tips differ, so there IS an update.
            print("→ Updates available (commit count unknown on this shallow checkout)")

        print("→ Pulling updates...")
        pre_pull_sha = _pull_updates(
            git_cmd,
            branch,
            auto_stash_ref,
            prompt_for_restore=prompt_for_restore,
            gw_input_fn=gw_input_fn,
            discard_local_changes=discard_local_changes,
            keep_stash=keep_stash,
        )

        _invalidate_update_cache()

        # Verify HEAD moved (#79678): a detached checkout pinned to a SHA can
        # report "N new commit(s)" and a successful ``merge --ff-only`` yet
        # stay on the old commit, so the old code reinstalled deps and
        # claimed "✓ Code updated!". Surface the no-op instead.
        post_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if pre_pull_sha and post_pull_sha == pre_pull_sha:
            print()
            print("✗ Code did not move — update was a no-op.")
            print(
                f"  HEAD is pinned to {pre_pull_sha[:10]} (detached checkout); "
                f"origin/{branch} advanced but the working tree stayed put."
            )
            print(
                "  Reattach to the branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # Verify HEAD is on the target branch; otherwise "✓ Code updated!"
        # would be a lie. An IN-PLACE update is the one legitimate way to end
        # elsewhere: origin/<target> was merged INTO the checked-out branch,
        # so the running code *is* current.
        post_pull_branch = _git_run(git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        if (
            not in_place_update
            and post_pull_branch
            and post_pull_branch not in {branch, "HEAD"}
        ):
            print()
            print(
                f"✗ Update pulled origin/{branch}, but the checkout is on "
                f"'{post_pull_branch}' — not claiming success."
            )
            print(
                "  Switch to the target branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # #95294: HEAD advanced; running gateways still serve pre-pull
        # modules until the restart phase below. Any interrupt between here
        # and a completed (or no-op) restart leaves this marker so the next
        # ``hermes update`` can catch up even when git is already up to date.
        # Distinct from ``.update-incomplete`` (venv/install repair).
        _write_fleet_restart_pending_marker(expected_sha=post_pull_sha or "")

        # Clear stale .pyc bytecode cache — prevents ImportError on gateway
        # restart when updated source references names that didn't exist in
        # the old bytecode (e.g. get_hermes_home added to hermes_constants).
        _sweep_bytecode_after_update(branch)

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _m()._sync_with_upstream_if_needed(
                git_cmd,
                _m().PROJECT_ROOT,
                assume_yes=assume_yes,
                input_fn=gw_input_fn,
            )

        # Reinstall deps: .[all], falling back to base + remaining extras
        # individually so a broken extra doesn't strip working capabilities.
        # Ownership preflight (#83529) refuses first if the venv has
        # foreign-owned (sudo-pip) files that would brick the install mid-mutation.
        _sync_python_dependencies_after_pull(
            git_cmd,
            branch,
            pre_pull_sha,
            active_lazy_features=active_lazy_features,
            active_tool_dependencies=active_tool_dependencies,
            _windows_gateway_resume=_windows_gateway_resume,
        )

        node_failures = _update_node_dependencies()
        _m()._build_web_ui(_m().PROJECT_ROOT / "web")

        desktop_build_ok = _rebuild_desktop_after_update(
            desktop_dir,
            had_desktop_app_before_update=had_desktop_app_before_update,
        )

        print()
        print(f"✓ Code updated!{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}")

        update_complete = _run_post_update_maintenance(
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            had_desktop_app_before_update=had_desktop_app_before_update,
            node_failures=node_failures,
            desktop_build_ok=desktop_build_ok,
            pre_update_version=pre_update_version,
        )

        # Write the exit code *before* the restart attempt: under ``update
        # --gateway`` this process lives in the gateway's systemd cgroup, and
        # the ``systemctl restart`` fallback SIGKILLs the cgroup (KillMode=
        # mixed) — us and the wrapping shell included — so the marker would
        # never be written and the new gateway's watcher would poll 30 min
        # and report a spurious timeout. The verified summary already folds
        # in Desktop and SQLite-runtime health (gateway/run.py).
        if gateway_mode:
            _write_gateway_update_exit_code(update_complete)

        _restart = _restart_gateway_fleet_after_update(_pre_update_plan, gateway_mode)
        _resume_windows_gateways_and_merge_outcome(_restart, _windows_gateway_resume, gateway_mode)
        _verify_fleet_after_update(
            _restart,
            _pre_update_plan=_pre_update_plan,
            _windows_gateway_resume=_windows_gateway_resume,
            node_failures=node_failures,
            update_complete=update_complete,
        )

    except _shim_quarantine_error_type() as e:
        # Fail-closed shim contention (#87331): strict quarantine refused
        # BEFORE any installer ran — defer via marker, exit 2, no ZIP.
        _refuse_update_for_contended_shims(e)
    except subprocess.CalledProcessError as e:
        stage = _format_update_failure_stage(e)
        if _should_zip_fallback_on_update_error(e):
            print(f"⚠ {stage}: {e}")
            print("→ Falling back to ZIP download...")
            print()
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
            if gateway_mode:
                _write_gateway_update_exit_code(desktop_build_ok)
        else:
            print(f"✗ {stage}: {e}")
            _print_called_process_error_tail(e)
            if _called_process_error_is_python_dep_install(e):
                print(
                    "  The git update already finished. Re-downloading the source "
                    "ZIP cannot fix a dependency install error and would overwrite "
                    "local files."
                )
                if _m()._is_windows():
                    print("  Retry through the venv interpreter:")
                    print(
                        '    venv\\Scripts\\python.exe -c '
                        '"from hermes_cli.main import main; main()" update --yes'
                    )
            try:
                from hermes_cli.update_receipt import finalize_update_receipt

                finalize_update_receipt("failed")
            except Exception:
                pass
            sys.exit(1)

# --- Hoisted from the body of _cmd_update_impl (self-contained, no closure state) ---

def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped gateway-restart-phase exception must fail the update.

    Fail closed unless the fleet is provably safe: ``surviving is None``
    (probe couldn't determine state, e.g. new ``hermes_cli.gateway`` no
    longer imports) or non-empty -> stale. ``surviving == []`` is proof of
    safety ONLY if nothing ran beforehand; a pre-restart gateway
    (``pre_restart_pids`` non-empty, or ``None`` = unreadable) that is gone
    now was stopped without a verified replacement (#78574).
    """
    if surviving is None or surviving:
        return True
    # surviving == []: safe only if we know nothing was running beforehand.
    return pre_restart_pids is None or bool(pre_restart_pids)


def _fleet_probe_expected_runtimes(
    pre_update_plan,
    pre_restart_pids,
    windows_resume_token,
    restarted_services,
    killed_pids,
) -> bool:
    """Whether the post-update fleet probe should have produced rows.

    The zero-rows fail-open (#93406): ``collect_fleet_versions()`` swallows
    every probe failure via ``logger.debug()`` and ``print_fleet_version_matrix([])``
    early-returns ``False``, so an empty snapshot reads as \"healthy fleet\" and
    the update exits 0.  An empty snapshot is only proof-of-safety when NOTHING
    says a gateway existed before the update.  Any of these signals means at
    least one runtime was (or may have been) live pre-update, so zero rows is
    verification failure, not health:

    * ``restarted_services`` / ``killed_pids`` — the POSIX restart phase
      touched live gateways.
    * ``pre_restart_pids`` non-empty, or ``None`` (pre-state unreadable —
      cannot prove nothing was running; same contract as
      ``_restart_phase_failure_is_incomplete``, #78574).
    * the pre-update plan inventoried ≥1 runtime.

    ``windows_resume_token`` is deliberately EXCLUDED (#93406 residual). The
    pause/resume token is bookkeeping for ``_pause_windows_gateways_for_update``
    / ``_resume_windows_gateways_after_update`` — it is not a runtime
    inventory, and its entries do not correspond to rows
    ``collect_fleet_versions()`` is capable of returning:

    * ``unmapped`` entries (Scheduled-Task gateways) never publish
      ``gateway_state.json`` rows at all, and
    * a paused profile gateway is resumed as a DETACHED relaunch that may not
      republish its identity within the probe window.

    Counting the token therefore made ``_fleet_rows_expected`` True on every
    Windows update that had paused a gateway, the probe's polling window ran
    out with zero rows on a perfectly healthy update, and verification
    reported "no rows … verification incomplete" and exited 1 after a long
    silent wait. Expected-runtimes must key only on signals that map to rows
    the probe can actually see; a genuinely live pre-update Windows gateway
    is already covered by ``pre_restart_pids`` and the plan inventory. The
    parameter stays in the signature so the call site keeps passing the token
    (cheap, explicit, and the docstring is where the exclusion is explained).

    The same condition gates the 2.0s settle sleep: a freshly restarted
    gateway needs the settle window to rewrite ``gateway_state.json``.

    Note this keys ONLY on zero-rows-despite-expected-runtimes.  A non-empty
    snapshot — including rows in ``unknown`` state — is still judged solely by
    ``print_fleet_version_matrix``.
    """
    del windows_resume_token  # excluded on purpose — see docstring (#93406)
    if restarted_services or killed_pids:
        return True
    if pre_restart_pids is None or pre_restart_pids:
        return True
    try:
        if pre_update_plan is not None and pre_update_plan.runtimes:
            return True
    except Exception:
        pass
    return False


def _print_items(items, label, key, fallback_key=None):
    if not items:
        return
    print(f"  {label}:")
    shown = items[:8]
    for it in shown:
        if isinstance(it, dict):
            name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
            desc = (it.get("description") or "").strip()
        else:
            # Defensive: some callers/mocks pass bare name strings.
            name = str(it)
            desc = ""
        if desc:
            print(f"      • {name} — {desc}")
        else:
            print(f"      • {name}")
    extra = len(items) - len(shown)
    if extra > 0:
        print(f"      … and {extra} more")

def _wait_for_service_active(
    scope_cmd_: list,
    svc_name_: str,
    timeout: float = 10.0,
) -> bool:
    """Poll ``systemctl is-active`` until the unit reports active.

    systemd's Stopped -> Started transition after a graceful exit
    (or a hard restart) is not instantaneous; a one-shot check
    races that window and falsely reports the unit as down.
    Poll every 0.5s up to ``timeout`` seconds before giving up.
    """
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        try:
            _verify = _systemctl(scope_cmd_ + ["is-active", svc_name_], timeout=5)
            if _verify.stdout.strip() == "active":
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)

def _service_restart_sec(
    scope_cmd_: list,
    svc_name_: str,
    default: float = 0.0,
) -> float:
    """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

    After a graceful exit-75, systemd waits ``RestartSec`` before
    respawning the unit.  Callers that poll for ``is-active``
    must use a timeout >= ``RestartSec`` + transition slack, or
    they'll give up *during* the cooldown window and wrongly
    conclude the unit didn't relaunch.
    """
    try:
        _show = _systemctl(scope_cmd_ + ["show", svc_name_, "--property=RestartUSec", "--value"], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # systemd emits values like "30s", "100ms", "1min 30s", or
    # "infinity".  Parse conservatively; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in (
            ("ms", 0.001),
            ("us", 0.000001),
            ("min", 60.0),
            ("s", 1.0),
        ):
            if part.endswith(_suf):
                try:
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                except ValueError:
                    pass
                break
    return total if matched else default
