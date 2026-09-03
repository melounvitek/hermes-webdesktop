"""hermes claw — OpenClaw migration commands."""

import importlib.util
import itertools
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from hermes_cli.config import get_hermes_home, get_config_path, load_config, save_config
from hermes_constants import get_optional_skills_dir
from hermes_cli.setup import (
    Colors, color, print_header, print_info, print_success, print_error, prompt_yes_no,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_SCRIPT_REL = Path("migration", "openclaw-migration", "scripts", "openclaw_to_hermes.py")
_OPENCLAW_SCRIPT = get_optional_skills_dir(PROJECT_ROOT / "optional-skills") / _SCRIPT_REL
# Fallback: user may have installed the skill from the Hub
_OPENCLAW_SCRIPT_INSTALLED = get_hermes_home() / "skills" / _SCRIPT_REL

# Known OpenClaw directory names (current + legacy)
_OPENCLAW_DIR_NAMES = (".openclaw", ".clawdbot", ".moltbot")

# `hermes claw migrate` flags and their defaults. Secrets are never included implicitly — they
# must be requested via --migrate-secrets even under --preset full (mirrors OpenClaw's two-phase
# migrate-hermes posture), so a full run cannot silently import API keys.
_MIGRATE_ARG_DEFAULTS = (
    ("source", None), ("dry_run", False), ("preset", "full"), ("overwrite", False),
    ("migrate_secrets", False), ("workspace_target", None), ("skill_conflict", "skip"),
    ("no_backup", False), ("yes", False),
)

# (status, heading, color, default reason) — printed in this order after migrated items.
_REPORT_REASON_GROUPS = (
    ("conflict", "  ⚠ Conflicts (skipped — use --overwrite to force):", Colors.YELLOW,
     "already exists"),
    ("skipped", "  ─ Skipped:", Colors.DIM, ""),
    ("error", "  ✗ Errors:", Colors.RED, "unknown error"),
)

# Workspace marker files listed by `hermes claw cleanup` (name, display label, presence check).
_WORKSPACE_ITEM_LABELS = (
    ("todo.json", "todo.json", Path.exists), ("sessions", "sessions/", Path.is_dir),
    ("SOUL.md", "SOUL.md", Path.exists), ("MEMORY.md", "MEMORY.md", Path.exists),
)


def _print_banner(title: str) -> None:
    """Print the magenta boxed banner shared by the claw subcommands."""
    print()
    rule = "─" * 57
    for line in (f"┌{rule}┐", f"│          ⚕ Hermes — {title:<35s}│", f"└{rule}┘"):
        print(color(line, Colors.MAGENTA))


def _error_block(headline: str, *lines: str) -> None:
    """Print a blank line, an error headline, then each line as info."""
    print()
    print_error(headline)
    for line in lines:
        print_info(line)


def _confirm(auto_yes: bool, question: str, *, default: bool, declined: str,
             non_tty: Optional[tuple[str, ...]] = None) -> Optional[bool]:
    """Ask to proceed unless --yes; print ``declined`` and return False when the user says no.

    With ``non_tty`` set, a non-interactive stdin prints those lines and returns None instead of
    prompting (callers treat None as "don't apply" but never as an explicit refusal).
    """
    if auto_yes:
        return True
    if non_tty is not None and not sys.stdin.isatty():
        for line in non_tty:
            print_info(line)
        return None
    if prompt_yes_no(question, default=default):
        return True
    print_info(declined)
    return False


def _warn_token_conflict(auto_yes: bool, headline: str, lines: list[str], declined: str,
                         non_tty: Optional[tuple[str, ...]] = None) -> None:
    """Print a bot-token conflict warning and exit 0 if the user explicitly declines to continue."""
    _error_block(headline, *lines)
    print()
    if _confirm(auto_yes, "Continue anyway?", default=False, declined=declined,
                non_tty=non_tty) is False:
        sys.exit(0)


def _detect_openclaw_processes() -> list[str]:
    """Detect running OpenClaw processes and services."""
    found: list[str] = []
    if sys.platform == "win32":
        # bounded_probe_run: a plain subprocess.run(timeout=...) can hang forever on Windows in
        # post-timeout cleanup when a conhost.exe descendant holds duplicated pipe handles — and
        # a hang is not an exception, so the try/except here can't save the caller.
        from hermes_cli._subprocess_compat import bounded_probe_run

        try:
            for exe in ("openclaw.exe", "clawd.exe"):
                result = bounded_probe_run(["tasklist", "/FI", f"IMAGENAME eq {exe}"], timeout=5)
                if result is not None and exe in (result.stdout or "").lower():
                    found.append(f"process: {exe}")
            # Node.js-hosted OpenClaw — tasklist doesn't show command lines, so use PowerShell.
            ps_cmd = (
                'Get-CimInstance Win32_Process -Filter "Name = \'node.exe\'" | '
                'Where-Object { $_.CommandLine -match "openclaw|clawd" } | '
                'Select-Object -First 1 ProcessId'
            )
            result = bounded_probe_run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=5)
            if result is not None and (result.stdout or "").strip():
                pid = result.stdout.strip()
                found.append(f"node.exe process with openclaw in command line (PID {pid})")
        except Exception:
            pass
        return found

    result = _posix_probe(["systemctl", "--user", "is-active", "openclaw-gateway.service"], 5)
    if result is not None and result.stdout.strip() == "active":
        found.append("systemd service: openclaw-gateway.service")
    result = _posix_probe(["pgrep", "-f", "openclaw"], 3)
    if result is not None and result.returncode == 0:
        found.append(f"openclaw process(es) (PIDs: {', '.join(result.stdout.strip().split())})")
    return found


def _posix_probe(cmd: list[str], timeout: int):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _warn_if_openclaw_running(auto_yes: bool) -> None:
    """Warn if OpenClaw is still running: Telegram/Discord/Slack allow one session per bot token."""
    running = _detect_openclaw_processes()
    if running:
        _warn_token_conflict(
            auto_yes, "OpenClaw appears to be running:",
            [f"  * {detail}" for detail in running] + [
                "Messaging platforms (Telegram, Discord, Slack) only allow one "
                "active session per bot token. If you continue, both OpenClaw and "
                "Hermes may try to use the same token, causing disconnects.",
                "Recommendation: stop OpenClaw before migrating."],
            declined="Migration cancelled. Stop OpenClaw and try again.",
            non_tty=("Non-interactive session — continuing to preview only.",),
        )


def _warn_if_gateway_running(auto_yes: bool) -> None:
    """Warn if a Hermes gateway has connected platforms (token conflicts, e.g. Telegram 409)."""
    from gateway.status import get_running_pid, read_runtime_status

    if not get_running_pid():
        return
    platforms = (read_runtime_status() or {}).get("platforms") or {}
    connected = [name for name, info in platforms.items()
                 if isinstance(info, dict) and info.get("state") == "connected"]
    if connected:
        _warn_token_conflict(
            auto_yes, "Hermes gateway is running with active connections: " + ", ".join(connected),
            ["Migrating bot tokens while the gateway is active will cause "
             "conflicts (Telegram, Discord, and Slack only allow one active "
             "session per token).",
             "Recommendation: stop the gateway first with 'hermes gateway stop'."],
            declined="Migration cancelled. Stop the gateway and try again.",
        )


def _find_migration_script() -> Path | None:
    """Find the openclaw_to_hermes.py script in known locations."""
    return next((c for c in (_OPENCLAW_SCRIPT, _OPENCLAW_SCRIPT_INSTALLED) if c.exists()), None)


def _load_migration_module(script_path: Path):
    """Dynamically load the migration script as a module."""
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes", script_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module (Python 3.11+ requires this).
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def _find_openclaw_dirs() -> list[Path]:
    """Find all OpenClaw directories on disk."""
    return [d for d in (Path.home() / name for name in _OPENCLAW_DIR_NAMES) if d.is_dir()]


def _scan_workspace_state(source_dir: Path) -> list[tuple[Path, str]]:
    """List state files in an OpenClaw directory root, then in its workspace-like subdirs."""
    candidates = [(source_dir / name, "Root") for name in ("todo.json", "sessions", "logs")]
    try:
        children = sorted(source_dir.iterdir())
    except OSError:
        children = []
    candidates += [
        (child / name, "Workspace")
        for child in children if child.is_dir() and not child.name.startswith(".")
        for name in ("todo.json", "sessions", "logs", "memory")
    ]
    return [
        (p, f"{scope} {'directory' if p.is_dir() else 'file'}: "
            f"{p.relative_to(source_dir).as_posix()}")
        for p, scope in candidates if p.exists()
    ]


def _archive_directory(source_dir: Path, dry_run: bool = False) -> Path:
    """Rename an OpenClaw directory to .pre-migration (date-stamped, then numbered, if taken)."""
    base = f"{source_dir.name}.pre-migration"
    stamped = f"{base}-{datetime.now().strftime('%Y%m%d')}"
    names = itertools.chain((base, stamped), (f"{stamped}-{n}" for n in itertools.count(2)))
    archive_path = next(p for p in (source_dir.parent / n for n in names) if not p.exists())
    if not dry_run:
        source_dir.rename(archive_path)
    return archive_path


def claw_command(args):
    """Route hermes claw subcommands."""
    action = getattr(args, "claw_action", None)
    if action == "migrate":
        _cmd_migrate(args)
    elif action in {"cleanup", "clean"}:
        _cmd_cleanup(args)
    else:
        print("Usage: hermes claw <command> [options]\n\nCommands:\n"
              "  migrate          Migrate settings from OpenClaw to Hermes\n"
              "  cleanup          Archive leftover OpenClaw directories after migration\n\n"
              "Run 'hermes claw <command> --help' for options.")


def _cmd_migrate(args):
    """Run the OpenClaw → Hermes migration: preflight, preview, confirm, back up, apply."""
    opts = SimpleNamespace(**{k: getattr(args, k, d) for k, d in _MIGRATE_ARG_DEFAULTS})
    # Explicit --source, else first existing of current + legacy names; default to ~/.openclaw.
    opts.source_dir = (Path(opts.source) if opts.source
                       else next(iter(_find_openclaw_dirs()), Path.home() / ".openclaw"))
    _print_banner("OpenClaw Migration")

    if not opts.source_dir.is_dir():
        _error_block(
            f"OpenClaw directory not found: {opts.source_dir}",
            "Make sure your OpenClaw installation is at the expected path.",
            "You can specify a custom path: hermes claw migrate --source /path/to/.openclaw",
        )
        return
    script_path = _find_migration_script()
    if not script_path:
        _error_block("Migration script not found.", "Expected at one of:", f"  {_OPENCLAW_SCRIPT}",
                     f"  {_OPENCLAW_SCRIPT_INSTALLED}",
                     "Make sure the openclaw-migration skill is installed.")
        return

    opts.hermes_home = get_hermes_home()
    print()
    print_header("Migration Settings")
    print_info(f"Source:      {opts.source_dir}")
    print_info(f"Target:      {opts.hermes_home}")
    print_info(f"Preset:      {opts.preset}")
    print_info(f"Overwrite:   {'yes' if opts.overwrite else 'no (skip conflicts)'}")
    print_info(f"Secrets:     {'yes (allowlisted only)' if opts.migrate_secrets else 'no'}")
    if opts.skill_conflict != "skip":
        print_info(f"Skill conflicts: {opts.skill_conflict}")
    if opts.workspace_target:
        print_info(f"Workspace:   {opts.workspace_target}")
    print()
    # Migrating tokens while OpenClaw or the gateway is active causes conflicts (e.g. Telegram 409).
    _warn_if_openclaw_running(opts.yes)
    _warn_if_gateway_running(opts.yes)
    # Ensure config.yaml exists before migration tries to read it
    if not get_config_path().exists():
        save_config(load_config())

    run_migrator = _load_migrator(script_path, opts)
    if run_migrator is None or not _preview_migration(run_migrator, opts):
        return
    print()
    if _confirm(opts.yes, "Proceed with migration?", default=True, declined="Migration cancelled.",
                non_tty=("Non-interactive session — preview only.",
                         "To execute, re-run with: hermes claw migrate --yes")):
        _apply_migration(run_migrator, opts)
    # Source directory is left untouched — archiving is `hermes claw cleanup`'s job.


def _load_migrator(script_path: Path, opts: SimpleNamespace) -> Optional[Callable[[bool], dict]]:
    """Load the migration script; return ``run(execute) -> report`` or None (error printed)."""
    try:
        mod = _load_migration_module(script_path)
    except Exception as e:
        _error_block(f"Could not load migration script: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return None
    if mod is None:
        print_error("Could not load migration script.")
        return None
    selected = mod.resolve_selected_options(None, None, preset=opts.preset)
    ws_target = Path(opts.workspace_target).resolve() if opts.workspace_target else None

    def _run_migrator(execute: bool) -> dict:
        return mod.Migrator(
            source_root=opts.source_dir.resolve(), target_root=opts.hermes_home.resolve(),
            execute=execute, workspace_target=ws_target, overwrite=opts.overwrite,
            migrate_secrets=opts.migrate_secrets, output_dir=None, selected_options=selected,
            preset_name=opts.preset, skill_conflict_mode=opts.skill_conflict,
        ).migrate()

    return _run_migrator


def _preview_migration(run_migrator: Callable[[bool], dict], opts: SimpleNamespace) -> bool:
    """Always preview first; return True only if the run should proceed to apply."""
    try:
        preview_report = run_migrator(False)
    except Exception as e:
        _error_block(f"Migration preview failed: {e}")
        logger.debug("OpenClaw migration preview error", exc_info=True)
        return False
    preview_summary = preview_report.get("summary", {})
    preview_count = preview_summary.get("migrated", 0)
    preview_conflicts = preview_summary.get("conflict", 0)

    # "Nothing to migrate" means nothing migrated AND nothing blocked by conflicts. With
    # conflicts, still show the plan and surface the --overwrite guidance instead of bailing.
    if preview_count == 0 and preview_conflicts == 0:
        print()
        print_info("Nothing to migrate from OpenClaw.")
        _print_migration_report(preview_report, dry_run=True)
        return False
    print()
    print_header(
        f"Migration Preview — {preview_count} item(s) would be imported"
        if preview_count > 0
        else f"Migration Preview — {preview_conflicts} conflict(s), nothing would be imported"
    )
    print_info("No changes have been made yet. Review the list below:")
    _print_migration_report(preview_report, dry_run=True)
    if opts.dry_run:
        return False

    # Modelled on OpenClaw's assertConflictFreePlan(): apply is a safe no-op on conflicts unless
    # the user opts in to overwriting — otherwise "yes, proceed" would silently skip every
    # conflicting item.
    if preview_conflicts > 0 and not opts.overwrite:
        _error_block(
            f"Plan has {preview_conflicts} conflict(s). Refusing to apply.",
            "Each conflict is an item whose target already exists in ~/.hermes/. "
            "Re-run with --overwrite to replace conflicting targets (item-level "
            "backups are written to the migration report directory).",
            "Or re-run with --dry-run to review the full plan.",
        )
        return False
    return True


def _apply_migration(run_migrator: Callable[[bool], dict], opts: SimpleNamespace) -> None:
    """Take a pre-migration backup (unless --no-backup), execute, and print the report.

    The backup shares implementation with the pre-update backup (same exclusions, SQLite
    safe-copy, zip format) so it is restorable with `hermes import` — one atomic restore point
    before any mutation, auto-pruned to the last 5 pre-migration zips.
    """
    backup_archive: Optional[Path] = None
    if not opts.no_backup:
        try:
            from hermes_cli.backup import create_pre_migration_backup, _format_size
            backup_archive = create_pre_migration_backup(hermes_home=opts.hermes_home)
            if backup_archive:
                size_str = _format_size(backup_archive.stat().st_size)
                print()
                print_success(f"Pre-migration backup: {backup_archive} ({size_str})")
                print_info(f"Restore with: hermes import {backup_archive.name}")
        except Exception as e:
            _error_block(
                f"Could not create pre-migration backup: {e}",
                "Re-run with --no-backup to skip, or free up disk space under the Hermes home.",
            )
            logger.debug("Pre-migration backup error", exc_info=True)
            return
    try:
        report = run_migrator(True)
    except Exception as e:
        _error_block(f"Migration failed: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        if backup_archive:
            print_info(f"A pre-migration backup is available at: {backup_archive}")
            print_info(f"Restore with: hermes import {backup_archive.name}")
        return
    _print_migration_report(report, dry_run=False)


def _cmd_cleanup(args):
    """Offer to rename leftover OpenClaw directories to .pre-migration to free disk space."""
    dry_run = getattr(args, "dry_run", False)
    auto_yes = getattr(args, "yes", False)
    explicit_source = getattr(args, "source", None)

    _print_banner("OpenClaw Cleanup")
    dirs_to_check = [Path(explicit_source)] if explicit_source else _find_openclaw_dirs()
    if not dirs_to_check:
        print()
        print_success("No OpenClaw directories found. Nothing to clean up.")
        return

    # Archiving while the service is active makes it recreate an empty skeleton directory.
    running = _detect_openclaw_processes()
    if running:
        _error_block(
            "OpenClaw appears to be still running:",
            *(f"  * {detail}" for detail in running),
            "Archiving .openclaw/ while the service is active may cause it to "
            "immediately recreate an empty skeleton directory, destroying your config.",
            "Stop OpenClaw first: systemctl --user stop openclaw-gateway.service",
        )
        print()
        if not _confirm(
            auto_yes, "Proceed anyway?", default=False,
            non_tty=("Non-interactive session — aborting. Stop OpenClaw and re-run.",),
            declined="Aborted. Stop OpenClaw first, then re-run: hermes claw cleanup",
        ):
            return

    total_archived = 0
    for source_dir in dirs_to_check:
        _describe_openclaw_dir(source_dir)
        if dry_run:
            archive_path = _archive_directory(source_dir, dry_run=True)
            print_info(f"Would archive: {source_dir} → {archive_path}")
        elif _confirm(auto_yes, f"Archive {source_dir}?", default=True, declined="Skipped.",
                      non_tty=(f"Non-interactive session — would archive: {source_dir}",
                               "To execute, re-run with: hermes claw cleanup --yes")):
            try:
                archive_path = _archive_directory(source_dir)
                print_success(f"Archived: {source_dir} → {archive_path}")
                total_archived += 1
            except OSError as e:
                print_error(f"Could not archive: {e}")
                print_info(f"Try manually: mv {source_dir} {source_dir}.pre-migration")

    print()
    n = len(dirs_to_check) if dry_run else total_archived
    word = "directory" if n == 1 else "directories"
    if dry_run:
        print_info(f"Dry run complete. {n} {word} would be archived.")
        print_info("Run without --dry-run to archive them.")
    elif total_archived:
        print_success(f"Cleaned up {total_archived} OpenClaw {word}.")
        print_info("Directories were renamed, not deleted. You can undo by renaming them back.")
    else:
        print_info("No directories were archived.")


def _describe_openclaw_dir(source_dir: Path) -> None:
    """Print the workspace directories (first 5) and state files (first 8) of one OpenClaw dir."""
    print()
    print_header(f"Found: {source_dir}")
    state_files = _scan_workspace_state(source_dir)
    try:
        workspace_dirs = [
            d for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and any((d / n).exists() for n in ("todo.json", "SOUL.md", "MEMORY.md", "USER.md"))
        ]
    except OSError:
        workspace_dirs = []

    if workspace_dirs:
        print_info(f"Workspace directories: {len(workspace_dirs)}")
        _print_rows([_workspace_row(ws) for ws in workspace_dirs], 5)
    if state_files:
        print()
        print(color(f"  {len(state_files)} state file(s) found:", Colors.YELLOW))
        _print_rows([desc for _path, desc in state_files], 8)
    print()


def _workspace_row(ws: Path) -> str:
    items = [label for name, label, check in _WORKSPACE_ITEM_LABELS if check(ws / name)]
    return f"{ws.name}/  ({', '.join(items) or 'empty'})"


def _print_rows(rows: list[str], shown: int) -> None:
    """Print the first ``shown`` rows indented, then a '... and N more' trailer if truncated."""
    for row in rows[:shown]:
        print(f"      {row}")
    if len(rows) > shown:
        print(f"      ... and {len(rows) - shown} more")


def _print_migration_report(report: dict, dry_run: bool):
    """Print a formatted migration report."""
    summary = report.get("summary", {})
    migrated = summary.get("migrated", 0)
    print()
    print_header("Dry Run Results" if dry_run else "Migration Results")
    if dry_run:
        print_info("No files were modified. This is a preview of what would happen.")
    print()

    items = report.get("items", [])
    migrated_items = [i for i in items if i.get("status") == "migrated"]
    if migrated_items:
        print(color(f"  ✓ {'Would migrate' if dry_run else 'Migrated'}:", Colors.GREEN))
        for item in migrated_items:
            kind, dest = item.get("kind", "unknown"), item.get("destination", "")
            print(f"      {kind:<22s} → {str(dest).replace(str(Path.home()), '~')}" if dest
                  else f"      {kind}")
        print()
    for status, heading, heading_color, default_reason in _REPORT_REASON_GROUPS:
        group = [i for i in items if i.get("status") == status]
        if not group:
            continue
        print(color(heading, heading_color))
        for item in group:
            print(f"      {item.get('kind', 'unknown'):<22s}  {item.get('reason', default_reason)}")
        print()

    counts = ((migrated, "would migrate" if dry_run else "migrated"),
              (summary.get("conflict", 0), "conflict(s)"), (summary.get("skipped", 0), "skipped"),
              (summary.get("error", 0), "error(s)"))
    parts = [f"{count} {label}" for count, label in counts if count]
    print_info(f"Summary: {', '.join(parts)}" if parts else "Nothing to migrate.")
    if report.get("output_dir"):
        print_info(f"Full report saved to: {report['output_dir']}")

    if dry_run:
        print()
        print_info("To execute the migration, run without --dry-run:")
        print_info(f"  hermes claw migrate --preset {report.get('preset', 'full')}")
    elif migrated:
        print()
        print_success("Migration complete!")
        # Warn if API keys were skipped (migrate_secrets not enabled)
        if any(i.get("kind") == "provider-keys" and i.get("status") == "skipped" for i in items):
            print()
            for line in (
                "  ⚠ API keys were NOT migrated (secrets migration is disabled by default).",
                "  Your OPENROUTER_API_KEY and other provider keys must be added manually.",
            ):
                print(color(line, Colors.YELLOW))
            print()
            print_info("To migrate API keys, re-run with:")
            print_info("  hermes claw migrate --migrate-secrets")
            print()
            print_info("Or add your key manually:")
            print_info("  hermes config set OPENROUTER_API_KEY sk-or-v1-...")
