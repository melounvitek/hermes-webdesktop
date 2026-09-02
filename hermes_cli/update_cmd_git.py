"""Git plumbing for ``hermes update``: fork/upstream sync, trampoline-git detection, lockfile/EOL churn cleanup, orphan rescue refs, parked-branch assessment, fetch-failure classification.

Split out of ``update_cmd.py``, which re-imports every name so ``hermes_cli.update_cmd.<name>``
still resolves/monkeypatches. Origin helpers are imported lazily per function (no cycle;
test patches on ``update_cmd`` stay effective).
"""

import logging
from contextlib import suppress
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


_ORPHAN_RESCUE_REFS_TO_KEEP = 10


_ORPHAN_RESCUE_REF_MAX_AGE_DAYS = 30


def _prune_orphan_rescue_refs(
    git_cmd,
    cwd,
    branch,
    keep=_ORPHAN_RESCUE_REFS_TO_KEEP,
    max_age_days=_ORPHAN_RESCUE_REF_MAX_AGE_DAYS,
) -> None:
    """Expire old orphan rescue refs (``refs/hermes-update-backups/orphan-<branch>-<ts>-<sha>``).

    Each ref pins a possibly multi-GB snapshot against ``git gc``, so a repeatedly corrupted
    install would grow ``.git`` unbounded. Keep the ``keep`` newest AND drop any older than
    ``max_age_days`` by the ``YYYYMMDD-HHMMSS`` stamp (unparseable names left alone); names
    sort chronologically so ``for-each-ref`` order is creation order. Best-effort, never blocks.
    """
    from hermes_cli.update_cmd import _git_run
    with suppress(OSError):
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


def _branch_head_label(git_cmd=None, cwd=None) -> str | None:
    """``"<branch> @ <short-sha>"`` for the checkout, or None when unknown. Never raises.

    Appended to summary lines so a checkout parked on a stale branch is visible.
    """
    from hermes_cli.update_cmd import _m
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
    """Decide whether a parked feature branch may be auto-switched back to the update target.

    - (True, "") — tree clean and every parked commit is in ``origin/<target>`` (no ``git cherry +``).
    - (True, "unmerged:<n>") — tree clean but commits not in target; switching is safe (checkout
      keeps committed work) but caller must print a LOUD notice. Non-interactive callers
      (desktop, gateway /update, cron) can't resolve a skip, so a clean checkout must reach target.
    - (False, "disabled"|"dirty"|"unverifiable") — caller must NOT touch the branch. Dirty is
      the genuinely unsafe case: uncommitted work riding an autostash across branches.
    """
    from hermes_cli.update_cmd import _git_run
    try:
        from hermes_cli.config import load_config

        _update_cfg = (load_config() or {}).get("updates", {})
        if isinstance(_update_cfg, dict) and not bool(
            _update_cfg.get("auto_switch_parked_branch", True)
        ):
            return False, "disabled"
    except Exception as exc:
        # Config read failure must not disable the safety checks; fall through with default.
        logger.debug("Could not read updates.auto_switch_parked_branch: %s", exc)

    status = _git_run(git_cmd, ["status", "--porcelain"], cwd)
    if status.returncode != 0:
        return False, "unverifiable"
    if status.stdout.strip():
        return False, "dirty"

    cherry = _git_run(git_cmd, ["cherry", f"origin/{target_branch}"], cwd)
    if cherry.returncode != 0:
        return False, "unverifiable"
    unmerged = [line for line in cherry.stdout.splitlines() if line.startswith("+")]
    if unmerged:
        # Safe (checkout keeps commits); reason tells caller to print the loud notice.
        return True, f"unmerged:{len(unmerged)}"
    return True, ""


def _print_parked_branch_skip_warning(
    git_cmd: list[str],
    cwd: Path,
    current_branch: str,
    target_branch: str,
    reason: str,
) -> None:
    """LOUD block: why the update was skipped on a parked branch, behind-count, fix commands."""
    from hermes_cli.update_cmd import _git_run
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
        why = f"the branch state could not be verified against origin/{target_branch}"

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
    print("  (commit or stash your work on the branch first if you want to keep it)")
    print(bar)


def _print_parked_branch_kept_notice(
    current_branch: str, target_branch: str, unmerged_count: str
) -> None:
    """LOUD notice when a clean parked branch with unmerged commits is auto-switched.

    Non-interactive callers can't resolve a skip, so we proceed — but the unmerged work
    (still safe on its branch) must be impossible to miss.
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
    from hermes_cli.update_cmd import _git_run
    with suppress(Exception):
        result = _git_run(git_cmd, ["remote", "get-url", "origin"], cwd)
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False
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
    from hermes_cli.update_cmd import _git_run
    try:
        result = _git_run(git_cmd, ["remote", "get-url", "upstream"], cwd)
        return result.returncode == 0
    except Exception:
        return False


def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    from hermes_cli.update_cmd import _git_run
    try:
        result = _git_run(git_cmd, ["remote", "add", "upstream", OFFICIAL_REPO_URL], cwd)
        return result.returncode == 0
    except Exception:
        return False


def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    from hermes_cli.update_cmd import _git_run
    with suppress(Exception):
        result = _git_run(git_cmd, ["rev-list", "--count", f"{base}..{head}"], cwd)
        if result.returncode == 0:
            return int(result.stdout.strip())
    return -1


def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()


def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    with suppress(Exception):
        from hermes_constants import get_hermes_home

        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()


def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Push updated main to origin (sync fork); True on success."""
    from hermes_cli.update_cmd import _git_run
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
    """Offer to add ``upstream``, compare origin/main vs upstream/main, ff-pull when strictly
    behind, then push origin.

    Returns True only when origin/main was actually verified against upstream/main; False when
    the check never happened, so the caller never reports "up to date" on an origin-only compare.
    """
    from hermes_cli.update_cmd import (
        _add_upstream_remote,
        _count_commits_between,
        _has_upstream_remote,
        _mark_skip_upstream_prompt,
        _no_prompt_git_kwargs,
        _should_skip_upstream_prompt,
    )
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
            # --yes means "don't block", not "mutate my remotes"; don't persist the decline.
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
                print("  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git")
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

    # Only upstream/main: a bare fetch drags in thousands of auto-generated branches.
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

    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(git_cmd, cwd, "origin/main", "upstream/main")

    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return False

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
        print("  ✗ Failed to pull from upstream. You may need to resolve conflicts manually.")
        return False

    print("  ✓ Updated from upstream")

    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print("  ℹ Got updates from upstream but couldn't push to fork (no write access?)")
        print("    Your local repo is updated, but your fork on GitHub may be behind.")
    return True


def _classify_fetch_failure(stderr: str) -> str:
    """Map git-fetch stderr to a one-line diagnosis (caller also prints the raw first line).

    Order matters: curl reports HTTP errors as ``unable to access '<url>': ... error: 429``,
    so rate-limit/outage checks must run BEFORE the generic "unable to access" check.
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
        # Anonymous fetch got HTTP 401: GitHub does this during outages (and for
        # renamed/private repos) — not a user credentials problem.
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


def _git_is_trampoline(git_cmd: list) -> bool:
    """Whether *git_cmd* is a broken Git-for-Windows trampoline shim.

    The ~46KB ``bin\\git.exe``/``cmd\\git.exe`` shims re-exec git-core; when they can't find
    it every call dies with the launcher's guard message (a PATH problem, not network).
    Never raises; unknown states report False so a probe failure can't block an update.
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
    """PortableGit candidates: shared root first (where the managed tree actually lives,
    not the profile-scoped HERMES_HOME), then profile home as a fallback for custom layouts."""
    from hermes_cli.update_cmd import get_default_hermes_root, get_hermes_home
    candidates = []
    with suppress(Exception):
        for root in (get_default_hermes_root(), Path(get_hermes_home())):
            candidates.append(root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe")
    return candidates


def _locate_real_git() -> Optional[Path]:
    """Find a real Git-for-Windows ``git-core/git.exe`` (standard locations + managed PortableGit)
    that runs without the trampoline guard. None when nothing suits — callers keep the broken
    command and let the fetch-failure ZIP fallback handle it."""
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
    """Swap a broken Git-for-Windows trampoline for a real git binary so fetch/pull/checkout
    keep working; if none is found leave the command untouched (fetch-failure handler falls
    back to ZIP). No-op off Windows and when git is healthy."""
    from hermes_cli.update_cmd import _locate_real_git
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
    print(f"⚠ Detected a broken git trampoline; switching to real git at {real_git}")
    return [str(real_git)] + list(git_cmd[1:])


def _discard_lockfile_churn(git_cmd, repo_root):
    """Restore ``package-lock.json`` files npm rewrote non-deterministically, so the update
    sees a clean tree instead of autostashing every run. Only touches lockfiles whose
    package.json is NOT also dirty. Best-effort."""
    from hermes_cli.update_cmd import _git_run
    with suppress(Exception):
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


def _normalize_managed_eol(git_cmd, repo_root):
    """Take a managed checkout off ``core.autocrlf=true`` without leaving it dirty.

    Git for Windows sets ``autocrlf=true`` system-wide, turning LF files CRLF and breaking
    ``git checkout`` on update; install.ps1 pins ``false`` but older checkouts never got it
    and only ``hermes update`` can fix them. Pin and cleanup are one operation: under
    ``autocrlf=true`` a CRLF tree reads clean, so pinning alone would expose every file as
    modified (whole-tree autostash). Pin only after the tree verifies clean under it; a
    checkout we can't fully normalize is left as-is. Best-effort.
    """
    from hermes_cli.update_cmd import _git_run
    # -c, not config: evaluate the tree as it WOULD look pinned, persisting nothing.
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
        # Files with a *content* change ignoring CRLF. ``--name-only --ignore-cr-at-eol``
        # still LISTS CR-only files; ``--numstat`` honors the filter (no record for them).
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
            # "<added>\t<deleted>\t<path>"; rename detection off, so exactly one path.
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[2]:
                paths.add(parts[2])
        return paths

    def _eol_only():
        all_dirty, real_dirty = _dirty(), _real_dirty()
        if all_dirty is None or real_dirty is None:
            return None
        return all_dirty - real_dirty

    with suppress(Exception):
        effective = _git_run(git_cmd, ["config", "--get", "core.autocrlf"], repo_root)
        # Only "true" rewrites LF->CRLF; unset/false/input leave the tree alone.
        if effective.stdout.strip().lower() != "true":
            return

        eol_only = _eol_only()
        if eol_only is None:
            return
        if eol_only:
            # Pathspec via stdin: thousands of paths exceed the Windows argv limit.
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
                # Still dirty: pinning would only surface churn we failed to clear.
                return
            print(f"→ Normalized line-ending churn ({len(eol_only)} file(s))")

        subprocess.run(
            git_cmd + ["config", "core.autocrlf", "false"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
