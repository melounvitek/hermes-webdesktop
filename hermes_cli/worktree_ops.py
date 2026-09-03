"""Git worktree isolation for ``hermes -w`` sessions: create, classify, prune.

Every git call goes through ``_git``/``_git_quiet`` (UTF-8 text, captured, bounded
timeout). Classification helpers fail SAFE toward "preserve". ``cli`` re-exports
these names; ``_cprint`` is imported lazily from ``cli`` to avoid a cycle.
"""
import concurrent.futures
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("cli")


def _cprint(text: str) -> None:
    from cli import _cprint as _impl

    _impl(text)


def _git(args, cwd, timeout: float = 10, **kwargs):
    """Run ``git *args`` in *cwd* capturing UTF-8 text; raises like ``subprocess.run``."""
    import subprocess

    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=cwd, **kwargs,
    )


def _git_quiet(args, cwd, timeout: float = 10, log: str | None = None, **kwargs) -> None:
    """Fail-soft ``_git``: swallow every error, optionally logging it at DEBUG with *log* as prefix."""
    try:
        _git(args, cwd, timeout=timeout, **kwargs)
    except Exception as e:
        if log:
            logger.debug("%s: %s", log, e)


def _normalize_git_bash_path(p: Optional[str]) -> Optional[str]:
    """Translate a Git Bash path (``/c/Users/...``, ``/cygdrive/c/...``, ``/mnt/c/...``)
    to native Windows form (``C:\\Users\\...``). No-op on non-Windows and native paths."""
    if not p or sys.platform != "win32":
        return p
    m = re.match(r"^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    return p


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD (Git-Bash-normalized), or None if not in a repo."""
    try:
        result = _git(["rev-parse", "--show-toplevel"], None, timeout=5)
        if result.returncode == 0:
            return _normalize_git_bash_path(result.stdout.strip())
    except Exception:
        pass
    return None


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Return True when a resolved path stays within the expected root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _cleanup_failed_worktree_add(repo_root: str, wt_path: Path, branch_name: str) -> None:
    """Sweep the leftovers of a failed/timed-out ``git worktree add``.

    ``worktree add`` is not transactional: killed mid-checkout it leaves the partial
    directory, a LOCKED admin entry under ``.git/worktrees/<name>`` naming the *live*
    pid (so the startup pruner's dead-pid unlock never touches it), and sometimes the
    branch. Any retry of the same name then fails. Every step is fail-soft.
    """
    try:
        # Unlock first: `worktree remove --force` refuses a locked tree.
        _git_quiet(["worktree", "unlock", str(wt_path)], repo_root, timeout=15, check=False)
        _git_quiet(["worktree", "remove", "--force", str(wt_path)], repo_root, timeout=15, check=False)
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # `remove` needs the dir; `prune` drops the admin entry when it is already gone.
        _git_quiet(["worktree", "prune"], repo_root, timeout=15, check=False)
        _git_quiet(["branch", "-D", branch_name], repo_root, timeout=15, check=False)
    except Exception as e:
        logger.debug("cleanup after failed worktree add: %s", e)


_PACK_SPRAWL_THRESHOLD = 15


def _maintain_pack_health(repo_root: str) -> None:
    """Repack the object store when pack files sprawl (background thread, fail-soft).

    ``gc --auto`` only triggers at 50 non-kept packs; past a few dozen packs every
    object lookup scans every pack index and worktree creation can blow its timeout
    under concurrent load. ``nice`` keeps the repack off the startup path.
    """
    import subprocess

    try:
        pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
        if not pack_dir.is_dir():
            return
        packs = len(list(pack_dir.glob("*.pack")))
        if packs < _PACK_SPRAWL_THRESHOLD:
            return
        logger.info("git pack sprawl (%d packs) — repacking in background", packs)
        cmd = ["git", "repack", "-a", "-d", "--quiet"]
        if os.name == "posix":
            cmd = ["nice", "-n", "19", *cmd]
        subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=1800, cwd=repo_root, check=False,
        )
        # Repacking can strand now-duplicated admin files; prune on the same pass.
        _git(["worktree", "prune"], repo_root, timeout=60, check=False)
    except Exception as e:
        logger.debug("pack maintenance skipped: %s", e)


def _resolve_worktree_base(
    repo_root: str,
    fetch_timeout: float = 5,
    freshness_window: float = 300,
) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from.

    The standalone clone's ``HEAD`` can lag the remote by hundreds of commits, so
    branching from it roots every new branch on a stale base. Strategy, each step
    falling back to the next: (1) the current branch's upstream, refreshed;
    (2) the remote default branch (``origin/HEAD``), refreshed; (3) local ``HEAD``.

    Refresh is deliberately cheap on the startup path: the fetch is skipped when
    ``FETCH_HEAD`` is younger than *freshness_window* seconds and capped at
    *fetch_timeout*; on failure the cached remote-tracking ref is used (never a
    second fetch). Genuine staleness is backstopped by the pre-push stale-base gate.

    Returns ``(base_ref, label)`` — *label* is the human-readable banner description.
    """
    import subprocess

    from hermes_cli._subprocess_compat import noninteractive_git_env

    def _run(args, timeout: float = 20):
        return _git(args, repo_root, timeout=timeout, stdin=subprocess.DEVNULL, env=noninteractive_git_env())

    def _ref_exists(ref: str) -> bool:
        try:
            return _run(["rev-parse", "--verify", "--quiet", ref + "^{commit}"]).returncode == 0
        except Exception:
            return False

    def _fetch_head_age() -> Optional[float]:
        try:
            gd = _run(["rev-parse", "--git-dir"])
            if gd.returncode != 0:
                return None
            git_dir = Path(gd.stdout.strip())
            if not git_dir.is_absolute():
                git_dir = Path(repo_root) / git_dir
            fetch_head = git_dir / "FETCH_HEAD"
            if not fetch_head.exists():
                return None
            return max(0.0, time.time() - fetch_head.stat().st_mtime)
        except Exception:
            return None

    def _refresh(remote: str, branch: str, ref: str) -> tuple:
        """(ref, label) after a best-effort refresh; never raises, never fetches twice."""
        age = _fetch_head_age()
        if age is not None and age < freshness_window and _ref_exists(ref):
            return ref, f"{ref} (fetched {int(age)}s ago)"
        try:
            fetched = _run(["fetch", remote, branch], timeout=fetch_timeout)
            if fetched.returncode == 0:
                return ref, f"{ref} (fetched)"
            reason = "fetch failed"
        except subprocess.TimeoutExpired:
            reason = f"fetch timed out after {fetch_timeout:g}s"
        except Exception as e:
            reason = f"fetch error: {e}"
        if _ref_exists(ref):
            logger.debug("worktree base: %s — using cached %s", reason, ref)
            return ref, f"{ref} (cached — {reason})"
        return "HEAD", f"HEAD (local — {reason}, no cached {ref})"

    # 1. Current branch's upstream, if it tracks one.
    try:
        up = _run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()  # e.g. "origin/main"
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return _refresh(remote, branch, upstream)
    except Exception as e:
        logger.debug("worktree base: upstream resolution failed: %s", e)

    # 2. Remote default branch (origin/HEAD).
    try:
        head_ref = _run(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        default_ref = ""
        if head_ref.returncode == 0:
            default_ref = head_ref.stdout.strip().replace("refs/remotes/", "", 1)
        if not default_ref:
            # origin/HEAD not set locally; ask the remote (network — capped like the fetch).
            show = _run(["remote", "show", "origin"], timeout=max(fetch_timeout, 5))
            for line in show.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    _branch = line.split(":", 1)[1].strip()
                    # A remote with no default branch reports "(unknown)".
                    if _branch and _branch != "(unknown)":
                        default_ref = "origin/" + _branch
                    break
        if default_ref and "/" in default_ref:
            remote, branch = default_ref.split("/", 1)
            return _refresh(remote, branch, default_ref)
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Fall back to local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"


def _ensure_worktrees_gitignored(repo_root: str) -> None:
    """Append ``.worktrees/`` to the repo's .gitignore when missing (fail-soft)."""
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        # utf-8-sig: a Notepad BOM would glue to the first line and defeat the
        # membership check (duplicating the entry); the append writes UTF-8.
        existing = (
            gitignore.read_text(encoding="utf-8-sig", errors="replace")
            if gitignore.exists()
            else ""
        )
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)


def _copy_worktree_includes(repo_root: str, wt_path: Path) -> None:
    """Copy/symlink the entries listed in ``.worktreeinclude`` (gitignored files the agent needs)."""
    include_file = Path(repo_root) / ".worktreeinclude"
    if not include_file.exists():
        return
    try:
        repo_root_resolved = Path(repo_root).resolve()
        wt_path_resolved = wt_path.resolve()
        # utf-8-sig, not the locale default: on a cp1251/GBK Windows machine a UTF-8
        # include list would decode to mojibake paths or raise (swallowed below),
        # copying nothing; a Notepad BOM would glue to the first entry.
        for line in include_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = Path(repo_root) / entry
            dst = wt_path / entry
            # Path-traversal / symlink-escape guard: both resolved endpoints must stay
            # inside their roots before any file or symlink operation happens.
            try:
                src_resolved = src.resolve(strict=False)
                dst_resolved = dst.resolve(strict=False)
            except (OSError, ValueError):
                logger.debug("Skipping invalid .worktreeinclude entry: %s", entry)
                continue
            if not _path_is_within_root(src_resolved, repo_root_resolved):
                logger.warning("Skipping .worktreeinclude entry outside repo root: %s", entry)
                continue
            if not _path_is_within_root(dst_resolved, wt_path_resolved):
                logger.warning("Skipping .worktreeinclude entry that escapes worktree: %s", entry)
                continue
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            elif src.is_dir() and not dst.exists():
                # Symlink directories (fast, no disk). Windows needs Developer Mode or
                # elevation for symlinks — fall back to a recursive copy there.
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(src_resolved), str(dst))
                except (OSError, NotImplementedError) as _sym_err:
                    if sys.platform != "win32":
                        raise
                    logger.info(
                        ".worktreeinclude: symlink failed (%s) — "
                        "falling back to copytree on Windows.",
                        _sym_err,
                    )
                    try:
                        shutil.copytree(
                            str(src_resolved),
                            str(dst),
                            symlinks=True,
                            dirs_exist_ok=False,
                        )
                    except Exception as _copy_err:
                        logger.warning(
                            ".worktreeinclude: copy fallback "
                            "also failed for %s -> %s: %s",
                            src, dst, _copy_err,
                        )
    except Exception as e:
        logger.debug("Error copying .worktreeinclude entries: %s", e)


def _worktree_add(repo_root: str, wt_path: Path, branch_name: str, base_ref: str, base_label: str):
    """Run ``git worktree add`` with a local-HEAD retry; returns ``(base_ref, base_label)`` or None on failure.

    Any failed/timed-out attempt is swept with ``_cleanup_failed_worktree_add``
    (git leaves a partial dir plus a lock naming THIS live pid, poisoning retries).
    """
    import subprocess

    from hermes_cli._subprocess_compat import noninteractive_git_env

    def _add(cfg):
        # 120s, not 30: on a multi-agent box the ~10k-file materialization contends
        # with sibling checkouts for the disk (measured 113s wall under load vs 1.2s idle).
        return _git(
            [*cfg, "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
            repo_root, timeout=120, stdin=subprocess.DEVNULL, env=noninteractive_git_env(),
        )

    # checkout.workers parallelizes file materialization (0.6s serial -> ~0.2s);
    # unknown -c keys are ignored by older git, and the retry drops them anyway.
    try:
        result = _add(["-c", "checkout.workers=8", "-c", "checkout.thresholdForParallelism=100"])
        if result.returncode != 0:
            if base_ref != "HEAD":
                # A partial fetch can leave the remote ref unusable — retry from local
                # HEAD so creation never hard-fails on a sync hiccup.
                logger.warning(
                    "worktree add from %s failed (%s); retrying from local HEAD",
                    base_ref, result.stderr.strip(),
                )
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                base_ref, base_label = "HEAD", "HEAD (fallback — remote base failed)"
                result = _add([])
            if result.returncode != 0:
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                _cprint(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
                return None
    except Exception as e:
        _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
        _cprint(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None
    return base_ref, base_label


def _setup_worktree(repo_root: str = None, sync_base: bool = True,
                    name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns ``{path, branch, repo_root, base}`` on success, None on failure.
    *sync_base* branches from the freshly-fetched remote tip (see
    ``_resolve_worktree_base``); ``worktree_sync: false`` branches from local HEAD.
    *name* (``/worktree new <name>``) replaces the random ``hermes-<id>``; named
    trees skip the ``hermes-`` prefix so the pruner ages them on its slower schedule.
    """
    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        _cprint("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    wt_name = f"hermes-{uuid.uuid4().hex[:8]}"
    if name:
        wt_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")[:40] or wt_name
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name
    if name and wt_path.exists():
        _cprint(f"\033[31m✗ Worktree already exists: {wt_path}\033[0m")
        print("  Pick a different name, or remove it with: "
              f"git worktree remove {wt_path}")
        return None

    _ensure_worktrees_gitignored(repo_root)

    if sync_base:
        base_ref, base_label = _resolve_worktree_base(repo_root)
    else:
        base_ref, base_label = "HEAD", "HEAD (local — worktree_sync disabled)"

    added = _worktree_add(repo_root, wt_path, branch_name, base_ref, base_label)
    if added is None:
        return None
    base_ref, base_label = added

    _copy_worktree_includes(repo_root, wt_path)

    # Lock so other processes (and `git worktree remove`) see it is in use. Fail-soft.
    try:
        _git(["worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(wt_path)], repo_root)
        logger.debug("Worktree locked: %s (pid=%s)", wt_path, os.getpid())
    except Exception as e:
        logger.debug("git worktree lock failed (non-fatal): %s", e)

    _cprint(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")
    print(f"  Base:   {base_label}")

    return {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
        "base": base_ref,
    }


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has commits not reachable from any remote branch.

    No remote-tracking refs at all means no usable baseline -> False. Fails SAFE
    toward True. SHALLOW-CLONE CAVEAT: the shallow boundary can disconnect an older
    HEAD from origin/*, making public commits look unpushed; callers that can afford
    it should ``_deepen_shallow_repo`` first (the startup pruner does).
    """
    try:
        remote_refs = _git(["for-each-ref", "--format=%(refname)", "refs/remotes"], worktree_path, timeout=timeout)
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            return False

        result = _git(["log", "--oneline", "HEAD", "--not", "--remotes"], worktree_path, timeout=timeout)
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _worktree_is_dirty(worktree_path: str, timeout: int = 10) -> bool:
    """Whether a worktree has staged/unstaged/untracked changes. Fails SAFE toward True."""
    try:
        result = _git(["status", "--porcelain"], worktree_path, timeout=timeout)
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _repo_is_shallow(repo_path: str, timeout: int = 5) -> bool:
    """Whether *repo_path* is a shallow clone (the installer default, ``--depth 1``).

    Shallowness poisons every history-connectivity verdict: an older worktree HEAD is
    disconnected from ``origin/main`` so its commits misreport as unpushed forever.
    Fails toward False so callers don't take shallow-specific branches on unknown state.
    """
    try:
        result = _git(["rev-parse", "--is-shallow-repository"], repo_path, timeout=timeout)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def _deepen_shallow_repo(repo_root: str, timeout: int = 600) -> bool:
    """One-time blobless unshallow (``--unshallow --filter=blob:none``) so history verdicts are correct.

    Fetches the commit/tree graph without historical blobs; falls back to a plain
    ``--unshallow`` if the server rejects filters. Background paths only. Returns
    whether the repo is non-shallow afterwards; on failure callers keep preserving.
    """
    import subprocess

    if not _repo_is_shallow(repo_root):
        return True

    try:
        remotes = _git(["remote"], repo_root)
        names = [r.strip() for r in remotes.stdout.splitlines() if r.strip()]
        if remotes.returncode != 0 or not names:
            return False
        remote = "origin" if "origin" in names else names[0]

        for extra in (["--filter=blob:none"], []):
            try:
                result = _git(["fetch", remote, "--unshallow", *extra], repo_root, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            if result.returncode == 0:
                break
            logger.debug(
                "git fetch --unshallow%s failed: %s",
                " " + " ".join(extra) if extra else "",
                result.stderr.strip()[-500:],
            )
    except Exception as e:
        logger.debug("Deepening shallow repo failed (non-fatal): %s", e)
        return False

    deepened = not _repo_is_shallow(repo_root)
    if deepened:
        logger.info(
            "Deepened shallow clone at %s so worktree cleanup can verify "
            "push state", repo_root,
        )
    return deepened


# Upper bound on retained `git cherry` verdict entries (~90 bytes each -> ~90 KB cap).
_WORKTREE_MERGE_CACHE_MAX = 1000


def _worktree_merge_cache_path() -> Path:
    """Path of the patch-equivalence verdict cache (profile-aware)."""
    return get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def _load_worktree_merge_cache() -> Dict[str, bool]:
    """Load the ``git cherry`` verdict cache. Missing/corrupt cache = empty."""
    try:
        raw = json.loads(
            _worktree_merge_cache_path().read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("verdicts")
    if not isinstance(entries, dict):
        return {}
    # A hand-edited or partially written cache must never inject a non-bool verdict.
    return {k: v for k, v in entries.items() if isinstance(v, bool)}


def _save_worktree_merge_cache(verdicts: Dict[str, bool]) -> None:
    """Persist the verdict cache atomically, bounded to the newest ``_WORKTREE_MERGE_CACHE_MAX``. Never raises."""
    path = _worktree_merge_cache_path()
    tmp = None
    try:
        items = list(verdicts.items())
        if len(items) > _WORKTREE_MERGE_CACHE_MAX:
            items = items[-_WORKTREE_MERGE_CACHE_MAX:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "verdicts": dict(items)}),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as e:
        logger.debug("Could not persist worktree merge cache: %s", e)
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _worktree_commits_all_merged_upstream(
    worktree_path: str,
    timeout: int = 30,
    max_ahead: int = 20,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Whether every local-only commit is patch-equivalent (``git cherry``) to an upstream commit.

    Catches the dominant ``.worktrees/`` leak: a squash-merged/cherry-picked PR whose
    remote branch was deleted leaves local commits unreachable from ``refs/remotes/*``
    forever. Returns False (preserve) when more than *max_ahead* commits ahead — a
    stale-base tree, too expensive to diff-hash. Fails SAFE toward False.

    *cache* memoizes the verdict on ``(base_sha, head_sha, max_ahead)`` — the exact
    inputs ``git cherry`` consumes, so a hit is identical to recomputation.
    """
    base = None
    for candidate in ("origin/HEAD", "origin/main", "origin/master"):
        try:
            probe = _git(["rev-parse", "--verify", "--quiet", candidate], worktree_path, timeout=timeout)
            if probe.returncode == 0 and probe.stdout.strip():
                base = candidate
                break
        except Exception:
            return False
    if base is None:
        return False

    try:
        cache_key = None
        if cache is not None:
            revs = _git(["rev-parse", f"{base}^{{commit}}", "HEAD^{commit}"], worktree_path, timeout=timeout)
            if revs.returncode == 0:
                shas = revs.stdout.split()
                if len(shas) == 2:
                    cache_key = f"{shas[0]}..{shas[1]}:{max_ahead}"
                    if cache_key in cache:
                        return cache[cache_key]

        def _memo(verdict: bool) -> bool:
            if cache is not None and cache_key is not None:
                cache[cache_key] = verdict
            return verdict

        ahead = _git(["rev-list", "--count", f"{base}..HEAD"], worktree_path, timeout=timeout)
        if ahead.returncode != 0:
            return False
        count = int(ahead.stdout.strip() or "0")
        if count == 0:
            return _memo(True)
        if count > max_ahead:
            return _memo(False)

        cherry = _git(["cherry", base, "HEAD"], worktree_path, timeout=timeout)
        if cherry.returncode != 0:
            return False
        lines = [ln for ln in cherry.stdout.splitlines() if ln.strip()]
        # "-" = patch-equivalent commit exists upstream; "+" = unique local work
        return _memo(bool(lines) and all(ln.startswith("-") for ln in lines))
    except Exception:
        return False


def _worktree_current_branch(worktree_path: str, timeout: int) -> Optional[str]:
    """Checked-out branch name, or None when detached or git fails. May raise on subprocess errors."""
    head = _git(["rev-parse", "--abbrev-ref", "HEAD"], worktree_path, timeout=timeout)
    if head.returncode != 0:
        return None
    branch = head.stdout.strip()
    if not branch or branch == "HEAD":  # detached
        return None
    return branch


def _worktree_branch_pr_merged(
    worktree_path: str,
    timeout: int = 15,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Whether the worktree branch's PR is MERGED on GitHub (``gh pr list``).

    Escape hatch for what ``git cherry`` cannot catch: a rebase-merge that altered the
    diff changes the patch-id, so merged trees survive the cherry check forever.
    Verdicts are memoized on ``(branch, head_sha)``; MERGED is monotonic so only True
    is cached (the PR may merge later without new local commits).
    Fails SAFE toward False: no gh, offline, rate-limited, detached, parse failure.
    """
    try:
        branch = _worktree_current_branch(worktree_path, timeout)
        if branch is None:
            return False

        cache_key = None
        if cache is not None:
            sha = _git(["rev-parse", "HEAD"], worktree_path, timeout=timeout)
            if sha.returncode == 0 and sha.stdout.strip():
                cache_key = f"pr-merged:{branch}:{sha.stdout.strip()}"
                if cache.get(cache_key) is True:
                    return True

        import subprocess

        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "merged",
             "--json", "number", "--limit", "1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return False
        prs = json.loads(result.stdout or "[]")
        merged = isinstance(prs, list) and len(prs) > 0
        if merged and cache is not None and cache_key is not None:
            cache[cache_key] = True
        return merged
    except Exception:
        return False


def _fetch_remote_branch_heads(repo_root: str, timeout: int = 20) -> Optional[Dict[str, str]]:
    """Return ``{branch_name: sha}`` for every branch on origin (one ``ls-remote``), or None.

    Managed installs fetch a single-branch refspec, so ``refs/remotes/origin/<branch>``
    never exists for pushed PR branches and they read as unpushed forever; one bounded
    network round-trip answers "is this branch pushed?" for the whole sweep.
    Fails SAFE toward None — callers must treat None as "cannot verify — preserve".
    """
    try:
        result = _git(["ls-remote", "--heads", "origin"], repo_root, timeout=timeout)
        if result.returncode != 0:
            return None
        heads: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                heads[parts[1][len("refs/heads/"):].strip()] = parts[0].strip()
        return heads
    except Exception:
        return None


def _worktree_branch_pushed_exact(
    worktree_path: str,
    remote_heads: Optional[Dict[str, str]],
    timeout: int = 10,
) -> bool:
    """Whether the worktree's branch head is EXACTLY what origin holds.

    True means the checkout is redundant — the work lives on the remote (typically an
    open PR) and in the local branch ref, so reaping the TREE while keeping the BRANCH
    loses nothing. Exact match is deliberately the only True case: a head ahead of or
    diverged from origin has commits origin lacks, and without remote-tracking refs
    ancestry can't be proven cheaply — anything but equality fails SAFE toward preserve.
    """
    if not remote_heads:
        return False
    try:
        branch = _worktree_current_branch(worktree_path, timeout)
        if branch is None:
            return False
        remote_sha = remote_heads.get(branch)
        if not remote_sha:
            return False
        local = _git(["rev-parse", "HEAD"], worktree_path, timeout=timeout)
        if local.returncode != 0:
            return False
        return local.stdout.strip() == remote_sha
    except Exception:
        return False


def _worktree_lock_is_live(repo_root: str, worktree_path: str, timeout: int = 10):
    """Classify a worktree's git lock: ``"live"`` (owning pid running — skip), ``"dead"``
    (pid gone or a non-hermes lock reason — safe to unlock + reap), or None (unlocked).

    ``hermes -w`` locks with reason ``hermes pid=<pid>``; a crashed session leaves the
    lock forever and ``worktree remove --force`` refuses locked trees, so dead-locked
    worktrees would accumulate indefinitely. Fails SAFE toward ``"live"``.
    """
    try:
        result = _git(["worktree", "list", "--porcelain"], repo_root, timeout=timeout)
        if result.returncode != 0:
            return "live"
    except Exception:
        return "live"

    target = Path(worktree_path).resolve()
    current: Optional[Path] = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                current = Path(line[len("worktree "):].strip()).resolve()
            except Exception:
                current = None
        elif line == "locked" or line.startswith("locked "):
            if current != target:
                continue
            reason = line[len("locked"):].strip()
            m = re.search(r"hermes pid=(\d+)", reason)
            if not m:
                # A foreign lock on a hermes -w worktree is almost certainly a leftover;
                # the age/dirty/unpushed gates already ran before we got here.
                return "dead"
            pid = int(m.group(1))
            if pid == os.getpid():
                return "live"
            try:
                from gateway.status import _pid_exists
                return "live" if _pid_exists(pid) else "dead"
            except Exception:
                return "live"
    return None


def _prune_candidates(worktrees_dir: Path, max_age_hours: int, now: float) -> list:
    """Phase 1 — stat-only age filter: ``[(entry, mtime, force)]`` for trees past their soft cutoff.

    Kanban task trees (``t_<hex>``) are owned by the kanban dispatcher's gc and skipped.
    Scratch trees (``hermes-*``) age on *max_age_hours*; named trees (salvage/review
    lanes created deliberately) get 3x. *force* marks the hard (3x) tier.
    """
    kanban_re = re.compile(r"^t_[0-9a-f]+$")
    candidates: list = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or kanban_re.match(entry.name):
            continue
        tier_hours = max_age_hours if entry.name.startswith("hermes-") else max_age_hours * 3
        soft_cutoff = now - (tier_hours * 3600)
        hard_cutoff = now - (tier_hours * 3 * 3600)
        try:
            mtime = entry.stat().st_mtime
            if mtime > soft_cutoff:
                continue  # Too recent — skip
        except Exception:
            continue
        candidates.append((entry, mtime, mtime <= hard_cutoff))
    return candidates


def _classify_prune_candidates(repo_root: str, candidates: list) -> list:
    """Phase 2 — read-only git classification of every candidate, in parallel.

    Returns ``[(entry, mtime, force, verdict, lock_state)]`` with verdict in
    ``dirty`` / ``unpushed`` / ``locked-live`` / ``reap`` / ``reap-keep-branch``.
    Each check is a read-only query against a distinct worktree (no repo-wide lock),
    so a bounded thread pool is safe; the mutating phase stays serial. ``git cherry``
    verdicts are memoized on disk (see ``_worktree_commits_all_merged_upstream``).
    """
    merge_cache = _load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    cache_lock = threading.Lock()

    # Lazy, once-per-sweep ls-remote — only paid when some tree reaches the pushed-tier
    # check (the TUI runs this pruner synchronously; offline costs one bounded timeout).
    _remote_heads_memo: dict = {}
    _remote_heads_lock = threading.Lock()

    def _get_remote_heads():
        with _remote_heads_lock:
            if "heads" not in _remote_heads_memo:
                _remote_heads_memo["heads"] = _fetch_remote_branch_heads(
                    repo_root, timeout=10
                )
            return _remote_heads_memo["heads"]

    def _classify(item):
        entry, mtime, force = item
        # Never delete real work regardless of age: only clean, fully merged/pushed trees are reaped.
        if _worktree_is_dirty(str(entry), timeout=5):
            return (entry, mtime, force, "dirty", None)
        keep_branch = False
        if _worktree_has_unpushed_commits(str(entry), timeout=5):
            # Squash-merge escape hatch: patch-equivalent commits are merged, not unpushed.
            with cache_lock:
                snapshot = dict(merge_cache)
            merged = _worktree_commits_all_merged_upstream(
                str(entry), timeout=30, cache=snapshot
            )
            if not merged:
                # Rebase-merge escape hatch: cherry misses changed patch-ids, GitHub knows.
                merged = _worktree_branch_pr_merged(
                    str(entry), timeout=15, cache=snapshot
                )
            with cache_lock:
                merge_cache.update(snapshot)
            if not merged:
                # Pushed-branch tier: head EXACTLY matches origin -> the checkout is
                # redundant; reap the tree, keep the branch ref (open-PR lane anchor).
                if _worktree_branch_pushed_exact(str(entry), _get_remote_heads(), timeout=10):
                    keep_branch = True
                else:
                    return (entry, mtime, force, "unpushed", None)

        # A live-locked tree is in use by a running hermes; a dead lock is unlocked in phase 3.
        lock_state = _worktree_lock_is_live(repo_root, str(entry), timeout=5)
        if lock_state == "live":
            return (entry, mtime, force, "locked-live", None)
        return (entry, mtime, force,
                "reap-keep-branch" if keep_branch else "reap", lock_state)

    # Enough workers to hide git's per-process startup latency without dozens of gits.
    workers = max(1, min(8, (os.cpu_count() or 4), len(candidates)))
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="hermes-wt-prune"
            ) as pool:
                verdicts = list(pool.map(_classify, candidates))
        else:
            verdicts = [_classify(c) for c in candidates]
    except Exception as e:
        logger.debug("Parallel worktree classification failed (%s); serial", e)
        verdicts = [_classify(c) for c in candidates]

    if len(merge_cache) != cache_size_before:
        _save_worktree_merge_cache(merge_cache)
    return verdicts


def _reap_prune_verdicts(repo_root: str, verdicts: list, stale_work_cutoff: float) -> tuple[list, set]:
    """Phase 3 — serial unlock / remove / branch -D.

    Returns ``(preserved_stale, kept_branches)``: trees preserved for dirty/unpushed work
    older than the cutoff (reported once), and branch refs the pushed tier kept on
    purpose (must survive the orphaned-branch pass). Branch deletion is gated on
    ``worktree remove`` succeeding so a failed removal never orphans reachable commits.
    """
    preserved_stale: list = []
    kept_branches: set = set()
    for entry, mtime, force, verdict, lock_state in verdicts:
        if verdict == "dirty":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (uncommitted changes)")
            continue
        if verdict == "unpushed":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (unpushed commits)")
            continue
        if verdict == "locked-live":
            logger.debug("Skipping live-locked worktree: %s", entry.name)
            continue

        if lock_state == "dead":
            _git_quiet(["worktree", "unlock", str(entry)], repo_root,
                       log=f"Failed to unlock dead worktree {entry.name}")

        try:
            branch = _git(["branch", "--show-current"], str(entry), timeout=5).stdout.strip()
            remove_result = _git(["worktree", "remove", str(entry), "--force"], repo_root, timeout=15)
            if remove_result.returncode != 0:
                logger.debug(
                    "Failed to remove worktree %s: %s",
                    entry.name, remove_result.stderr.strip(),
                )
                continue
            if branch and verdict == "reap-keep-branch":
                kept_branches.add(branch)
            elif branch:
                _git(["branch", "-D", branch], repo_root)
            logger.debug("Pruned stale worktree: %s (force=%s)", entry.name, force)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)
    return preserved_stale, kept_branches


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove stale worktrees and orphaned branches on startup.

    Covers every directory under ``.worktrees/`` except kanban task trees. Tiers:
    ``hermes-*`` skip under 24h, reap 24h+ when clean and merged/pushed, 72h+ is the
    aggressive tier (still never deletes real work); named trees run at 3x.

    Work-preservation guards (all tiers, any age): dirty trees are never removed;
    unpushed commits are never removed UNLESS patch-equivalent to upstream (squash-merge
    case), the PR is MERGED on GitHub, or the branch head EXACTLY matches origin (pushed
    tier — tree reaped, branch ref kept and shielded from the orphaned-branch pass).
    Live-locked trees are skipped at any age; dead-locked ones are unlocked first.
    Trees preserved for >7 days are listed in one WARNING so in-flight work can't rot
    silently. Phases: ``_prune_candidates`` -> ``_classify_prune_candidates`` (parallel,
    read-only) -> ``_reap_prune_verdicts`` (serial) -> ``_prune_orphaned_branches``.
    """
    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        _prune_orphaned_branches(repo_root)
        return

    # A shallow clone disconnects old worktree HEADs from origin/main, so every aged
    # tree would read as unpushed forever. Deepen once (blobless, fail-soft).
    if _repo_is_shallow(repo_root):
        _deepen_shallow_repo(repo_root)

    now = time.time()
    candidates = _prune_candidates(worktrees_dir, max_age_hours, now)
    if not candidates:
        _prune_orphaned_branches(repo_root)
        return

    verdicts = _classify_prune_candidates(repo_root, candidates)
    preserved_stale, kept_branches = _reap_prune_verdicts(repo_root, verdicts, now - (7 * 24 * 3600))

    if preserved_stale:
        logger.warning(
            "Preserving %d worktree(s) older than 7 days with unmerged work "
            "(run `hermes worktree prune` to review and reclaim): %s",
            len(preserved_stale), ", ".join(sorted(preserved_stale)),
        )

    _prune_orphaned_branches(repo_root, protect=kept_branches)

    # Escalation notice: the startup pass is conservative, so installs accumulate
    # preserved trees it can never reclaim — say so once per launch past a threshold.
    try:
        from hermes_cli.worktree_gc import worktrees_summary

        count, size_mb = worktrees_summary(repo_root)
        if count >= 10 or (size_mb or 0) >= 5120:
            size_txt = f"{size_mb / 1024:.1f}GB" if size_mb else "unknown size"
            logger.warning(
                ".worktrees/ holds %d tree(s) (%s) — run `hermes worktree list` "
                "to audit and `hermes worktree prune` to reclaim safely.",
                count, size_txt,
            )
    except Exception:
        pass


def _prune_orphaned_branches(repo_root: str, protect: Optional[set] = None) -> None:
    """Delete local ``hermes/hermes-*`` and ``pr-*`` branches with no worktree.

    *protect*: branch names never deleted this pass — the pushed-tier reap removes a
    tree while deliberately keeping its branch (an open PR's local anchor).
    """
    try:
        result = _git(["branch", "--format=%(refname:short)"], repo_root)
        if result.returncode != 0:
            return
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    except Exception:
        return

    # Branches actively checked out in a worktree
    active_branches: set = set()
    try:
        wt_result = _git(["worktree", "list", "--porcelain"], repo_root)
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())
    except Exception:
        return  # Can't determine active branches — bail

    # Also protect the currently checked-out branch and main
    try:
        current = _git(["branch", "--show-current"], repo_root, timeout=5).stdout.strip()
        if current:
            active_branches.add(current)
    except Exception:
        pass
    active_branches.add("main")

    orphaned = [
        b for b in all_branches
        if b not in active_branches
        and b not in (protect or ())
        and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
    ]

    if not orphaned:
        return

    for i in range(0, len(orphaned), 50):
        _git_quiet(["branch", "-D"] + orphaned[i:i + 50], repo_root, timeout=30,
                   log="Failed to prune orphaned branches")

    logger.debug("Pruned %d orphaned branches", len(orphaned))
