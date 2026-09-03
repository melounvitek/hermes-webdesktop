"""Post-``hermes update`` config-schema migration for the active profile and every sibling.

Split out of ``update_cmd.py``; names are re-imported there so ``hermes_cli.update_cmd.<name>``
still resolves/monkeypatches. Origin helpers are imported lazily per function (no cycle; patches hold).
"""

import logging
import sys
from pathlib import Path

from hermes_cli.update_cmd_common import _best_effort

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


def _reload_config_modules() -> None:
    """Force-reload modules from disk after git pull.

    ``hermes update`` runs in the PRE-pull process, so cached ``config_defaults`` /
    ``config`` / ``config_migrations`` hold OLD code and ``check_config_version()``
    reports "up to date" despite a newer pulled version with a migration to run.
    Also reloads ``_subprocess_compat`` / ``dashboard_procs`` so the later dashboard
    cleanup sees symbols the update added instead of dying with ImportError.
    """
    import importlib
    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli.config_defaults",
        "hermes_cli.config",
        "hermes_cli.config_migrations",
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                logger.debug("Could not reload %s for fresh post-update code: %s", mod_name, exc)


def _run_config_check_fresh() -> tuple:
    """Return ``(current_ver, latest_ver)`` using freshly-reloaded modules (see ``_reload_config_modules``)."""
    from hermes_cli.update_cmd import _reload_config_modules
    _reload_config_modules()
    from hermes_cli.config import check_config_version
    return check_config_version()


def _run_migrate_config_fresh(*, interactive: bool = False, quiet: bool = False) -> dict:
    """Run config migration using freshly-reloaded modules (see ``_reload_config_modules``); returns results dict."""
    from hermes_cli.update_cmd import _reload_config_modules
    _reload_config_modules()
    from hermes_cli.config import migrate_config
    return migrate_config(interactive=interactive, quiet=quiet)


def _migrate_sibling_profile_configs() -> list[tuple[str, int, int]]:
    """Migrate every SIBLING profile's config.yaml (the shared checkout serves all profiles;
    migrating only the active one left siblings on configs the new code couldn't read).

    Per sibling home (active one skipped — caller already did it): scope reads/writes via the
    context-local HERMES_HOME override (thread-safe — never ``os.environ``) and run the
    NON-INTERACTIVE quiet migration; prompt-requiring settings wait for that profile's own
    interactive session. Returns ``[(profile_name, from_version, to_version), ...]`` for
    profiles actually migrated. Never raises; a failing profile is skipped (its startup
    migration remains the fallback).
    """
    from hermes_cli.update_cmd import _run_config_check_fresh, _run_migrate_config_fresh
    migrated: list[tuple[str, int, int]] = []
    with _best_effort('Sibling profile enumeration failed: %s'):
        from hermes_constants import (
            get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override)
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
                logger.debug("Config migration for profile %s failed: %s", entry.name, exc)
            finally:
                reset_hermes_home_override(token)
    return migrated


def _restore_snapshot_safety_nets(pre_update_snapshot_id) -> None:
    """Post-migration safety nets: restore cron jobs / protected model settings lost during the update,
    for the active profile (from *pre_update_snapshot_id*) and every sibling profile (own snapshot)."""
    # Safety net: migrations/desktop scheduler have emptied or truncated cron/jobs.json;
    # restore from the pre-update snapshot if jobs went missing.
    try:
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        cron_restore = restore_cron_jobs_if_emptied(pre_update_snapshot_id)
        if cron_restore:
            print()
            print(
                "  ⚠️  cron/jobs.json lost jobs during this update — "
                f"restored {cron_restore['job_count']} job(s) from "
                f"pre-update snapshot {cron_restore['snapshot_id']}.")
    except Exception as exc:
        # Never let the cron safety net break an otherwise-good update.
        logger.debug("Cron jobs auto-restore check failed: %s", exc)

    # Desktop update/repair cycles have rewritten model.provider/model.default and dropped
    # moa:; restore only those protected keys from the same pre-update snapshot.
    try:
        from hermes_cli.backup import restore_config_model_settings_if_rewritten
        cfg_restore = restore_config_model_settings_if_rewritten(pre_update_snapshot_id)
        if cfg_restore:
            print()
            print(
                "  ⚠️  config.yaml user model settings were rewritten during "
                f"this update — restored {', '.join(cfg_restore['keys'])} "
                f"from pre-update snapshot {cfg_restore['snapshot_id']}.")
    except Exception as exc:
        # Never let the config safety net break an otherwise-good update.
        logger.debug("Config model-settings auto-restore check failed: %s", exc)

    # Same cron-jobs safety net per sibling profile against ITS OWN pre-update snapshot.
    with _best_effort('Sibling cron auto-restore check failed: %s'):
        from hermes_cli.backup import restore_cron_jobs_all_profiles
        for _restored in restore_cron_jobs_all_profiles(
            _LAST_SIBLING_SNAPSHOTS):
            print()
            print(
                f"  ⚠️  Profile '{_restored['profile']}': cron/jobs.json "
                f"lost jobs during this update — restored "
                f"{_restored['job_count']} job(s) from pre-update "
                f"snapshot {_restored['snapshot_id']}.")

    # Same config model-settings safety net for sibling profiles.
    with _best_effort('Sibling config auto-restore check failed: %s'):
        from hermes_cli.backup import restore_config_model_settings_all_profiles
        for _cfg_restored in restore_config_model_settings_all_profiles(
            _LAST_SIBLING_SNAPSHOTS):
            print()
            print(
                f"  ⚠️  Profile '{_cfg_restored['profile']}': config.yaml "
                f"user model settings were rewritten during this update — "
                f"restored {', '.join(_cfg_restored['keys'])} from "
                f"pre-update snapshot {_cfg_restored['snapshot_id']}.")


def _check_and_apply_config_migration(
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None) -> None:
    """Check/apply config migrations on an update completion path.

    Must use freshly-reloaded modules (see ``_reload_config_modules``) and run on EVERY
    completion path (post-pull, venv-repair retry, Node-deps repair on ``commit_count == 0``)
    so an interrupted update that already pulled code doesn't strand an old config version.
    """
    from hermes_cli.update_cmd import (
        _gateway_prompt,
        _migrate_sibling_profile_configs,
        _reload_config_modules,
        _run_config_check_fresh,
        _run_migrate_config_fresh)
    print()
    print("→ Checking configuration for new options...")

    # Reload BEFORE any config reads so all checks use the updated code.
    _reload_config_modules()

    from hermes_cli.config import (
        get_missing_env_vars, get_missing_config_fields)

    # A config-check failure must not break an otherwise-successful update.
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
    version_bump_only = not has_new_options and current_ver < latest_ver
    needs_migration = has_new_options or current_ver < latest_ver

    if version_bump_only:
        # Only the format version changed (defaults merge transparently); prompting
        # would look like a no-op on yes — apply silently and say what happened.
        print()
        print(f"  ℹ Updating config format (v{current_ver} → v{latest_ver})…")
        try:
            _mig_results = _run_migrate_config_fresh(interactive=False, quiet=True)
            print("  ✓ Config format updated (no new settings to configure)")
            # quiet=True also mutes steps that RESET/REMOVE a setting; re-surface them so an
            # unattended update never silently changes config (config_added holds only mutations here).
            for _note in _mig_results.get("config_added") or []:
                print(f"  ℹ {_note}")
            for _warn in _mig_results.get("warnings") or []:
                print(f"  ⚠️  {_warn}")
        except Exception as _mig_err:
            print(f"  ⚠️  Config format update failed: {_mig_err}")
            print("     Run 'hermes config migrate' to retry.")
    elif needs_migration:
        print()
        # Show WHAT changed, not just a count, for an informed yes/no.
        if missing_env:
            print(f"  ⚠️  {len(missing_env)} new required setting(s) need configuration")
            _print_items(missing_env, "New settings", "name")
        if missing_config:
            print(f"  ℹ️  {len(missing_config)} new config option(s) available")
            _print_items(missing_config, "New options", "key")

        print()
        if assume_yes:
            print("  ℹ --yes: auto-applying config migration (skipping API-key prompts).")
            response = "y"
        elif gateway_mode:
            response = (
                _gateway_prompt(
                    "Would you like to configure new options now? [Y/n]", "n")
                .strip()
                .lower())
        elif not (sys.stdin.isatty() and sys.stdout.isatty()):
            print("  ℹ Non-interactive session — applying safe config migrations.")
            response = "auto"
        else:
            try:
                response = (input("Would you like to configure them now? [Y/n]: ").strip().lower())
            except EOFError:
                response = "n"
            except UnicodeDecodeError:
                # Non-UTF-8 locales / embedded terminals can make input() raise this.
                print(
                    "  ⚠ Could not read input (encoding issue). Skipping. "
                    "Run 'hermes config migrate' manually to configure.")
                response = "n"

        if response in {"", "y", "yes", "auto"}:
            print()
            # Gateway/--yes/non-interactive can't prompt for API keys; still run the
            # non-interactive pass so defaults and version bumps land before the gateway restarts.
            interactive_migration = not (gateway_mode or assume_yes or response == "auto")
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

    # The migration above touched only the active profile; run the same NON-INTERACTIVE
    # migration per sibling home via the context-local HERMES_HOME override (never os.environ).
    with _best_effort('Sibling config migration failed: %s'):
        _migrated_siblings = _migrate_sibling_profile_configs()
        for _name, _from_ver, _to_ver in _migrated_siblings:
            print(f"  ✓ Profile '{_name}': config format updated " f"(v{_from_ver} → v{_to_ver})")

    _restore_snapshot_safety_nets(pre_update_snapshot_id)


# {profile: snapshot_id} from this run's pre-update backup, consumed by the per-profile
# safety nets. Module-level because snapshot and restore run far apart in _cmd_update_impl.
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
