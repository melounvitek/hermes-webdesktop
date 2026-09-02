"""Autostash handling for ``hermes update``: stash local changes before the pull, restore/park/discard them afterwards, warn about orphaned autostashes.

Split out of ``hermes_cli/update_cmd.py``; every moved name is re-imported there, so
``hermes_cli.update_cmd.<name>`` keeps resolving (and monkeypatching) as before.
Origin-internal helpers are imported lazily inside each function (no import cycle;
test patches on ``hermes_cli.update_cmd.<name>`` stay effective).
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


def _stash_local_changes_if_needed(git_cmd: list[str], cwd: Path) -> Optional[str]:
    from hermes_cli.update_cmd import _git_run
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
    from hermes_cli.update_cmd import _git_run
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
    from hermes_cli.update_cmd import _git_run
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
    from hermes_cli.update_cmd import _git_untracked_paths
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
    from hermes_cli.update_cmd import _git_untracked_paths
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
    from hermes_cli.update_cmd import (
        _critical_module_import_failures,
        _git_run,
        _git_untracked_paths,
        _restored_python_paths,
        _validate_python_files_syntax,
    )
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
    from hermes_cli.update_cmd import _git_run
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
