"""Config-schema migration after ``hermes update``: run the fresh-process config check / migrate for the active profile and every sibling profile.

Split out of ``hermes_cli/update_cmd.py``; every moved name is re-imported there, so
``hermes_cli.update_cmd.<name>`` keeps resolving (and monkeypatching) as before.
Origin-internal helpers are imported lazily inside each function (no import cycle;
test patches on ``hermes_cli.update_cmd.<name>`` stay effective).
"""

import logging
import sys
from pathlib import Path

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


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
    from hermes_cli.update_cmd import _reload_config_modules
    _reload_config_modules()
    from hermes_cli.config import check_config_version

    return check_config_version()


def _run_migrate_config_fresh(*, interactive: bool = False, quiet: bool = False) -> dict:
    """Run config migration using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns the migration results dict.
    """
    from hermes_cli.update_cmd import _reload_config_modules
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
    from hermes_cli.update_cmd import _run_config_check_fresh, _run_migrate_config_fresh
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
    from hermes_cli.update_cmd import (
        _gateway_prompt,
        _migrate_sibling_profile_configs,
        _reload_config_modules,
        _run_config_check_fresh,
        _run_migrate_config_fresh,
    )
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


# {profile: snapshot_id} from this run's pre-update backup, consumed by the
# post-update per-profile cron-jobs safety net (#66140). Module-level because
# snapshot and restore run far apart in _cmd_update_impl.
_LAST_SIBLING_SNAPSHOTS: dict = {}


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
