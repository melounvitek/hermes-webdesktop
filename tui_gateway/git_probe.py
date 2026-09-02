"""Git working-tree probing for the gateway: run git, resolve repo roots, fold
linked worktrees under their common root.

Probing runs where the gateway runs, so it covers local and remote backends
(the desktop's electron probe only sees the local fs). Roots go through a
thread-safe single-flight cache: gateway handlers run on worker threads, so
concurrent identical probes share one ``git`` spawn instead of racing a dict.

Positive results are cached for the process lifetime; negatives (not a repo, or
a deleted dir) only for ``_NEG_TTL``. Caching negatives matters: ``build_tree``
resolves a cwd once *per session*, so hundreds of sessions in non-git/deleted
dirs would otherwise re-spawn ``git`` on every sidebar open (multi-second
"Projects" load). The TTL keeps a not-yet-repo cwd re-probable — we ``git init``
a new project's folder on its first worktree, and a frozen "" would mislabel its
main lane by dir basename. ``invalidate()`` drops everything after a known
mutation.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from hermes_cli._subprocess_compat import bounded_git_probe

_GIT_TIMEOUT = 1.5
_WARM_WORKERS = 8

# "Not a git repo" cache TTL: short enough that a freshly `git init`-ed folder
# shows correctly within seconds, long enough to collapse a tree build's
# hundreds of redundant probes.
_NEG_TTL = 30.0


def run_git(cwd: str, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure.

    Uses :func:`bounded_git_probe` so post-kill cleanup is bounded on Windows —
    a plain ``subprocess.run(timeout=...)`` deadlocked Desktop session readiness
    when a killed git left a suspended descendant holding the pipe handles.
    """
    if not cwd or not os.path.isdir(cwd):
        # `git -C` on a missing dir can only fail, at the price of a fork; deleted
        # worktrees dominate a long session history's cwds, so the stat pays off.
        return ""
    return bounded_git_probe(["git", "-C", cwd, *args], timeout=_GIT_TIMEOUT)


def branch(cwd: str) -> str:
    return run_git(cwd, "branch", "--show-current") or run_git(cwd, "rev-parse", "--short", "HEAD")


class _RootCache:
    """Thread-safe, single-flight cache of git-root probes: positives live for
    the process, negatives for ``_NEG_TTL``; followers wait on the leader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roots: dict[str, str] = {}
        self._neg: dict[str, float] = {}  # key -> monotonic expiry
        self._inflight: dict[str, threading.Event] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._roots.clear()
            self._neg.clear()
            self._inflight.clear()

    def resolve(self, key: str, probe) -> str:
        while True:
            with self._lock:
                hit = self._roots.get(key)
                if hit:
                    return hit
                expiry = self._neg.get(key)
                if expiry is not None:
                    if expiry > time.monotonic():
                        return ""  # recent "not a repo": trust it briefly
                    del self._neg[key]  # TTL elapsed: re-probe (may be a repo now)
                gate = self._inflight.get(key)
                leader = gate is None
                if leader:
                    gate = self._inflight[key] = threading.Event()

            if not leader:
                # Another thread is probing this key — wait, then re-read.
                gate.wait(timeout=_GIT_TIMEOUT + 0.5)
                continue

            value = ""
            try:
                value = probe()
            finally:
                with self._lock:
                    if value:
                        self._roots[key] = value
                    else:
                        self._neg[key] = time.monotonic() + _NEG_TTL
                    self._inflight.pop(key, None)
                gate.set()
            return value


_cache = _RootCache()


def invalidate() -> None:
    """Drop cached roots after a known mutation (e.g. a worktree was added)."""
    _cache.invalidate()


def repo_root(cwd: str) -> str:
    """Top-level git repo root for ``cwd`` (``""`` when not a repo)."""
    if not cwd:
        return ""
    return _cache.resolve(cwd, lambda: run_git(cwd, "rev-parse", "--show-toplevel"))


def common_repo_root(cwd: str) -> str:
    """The MAIN (common) repo root for ``cwd``, folding linked worktrees.

    ``--show-toplevel`` returns a linked worktree's OWN root, splitting every
    worktree into its own "repo"; the parent of the shared ``--git-common-dir``
    is the one true root (fallback: the toplevel root).

    The result is normalized to git's forward-slash spelling so it compares
    equal to :func:`repo_root` (raw ``--show-toplevel``). ``os.path.realpath``
    uses native ``\\`` on Windows, so without this the main checkout compared
    unequal to its own common root, was misread as a linked worktree, and the
    desktop sidebar rendered it twice.
    """
    # Not a repo: nothing to fold. Checking the (warmed, negative-cached)
    # toplevel first spares every non-repo cwd a second `git` spawn that the
    # parallel warm can't absorb (`resolve()` only reaches here for repos).
    if not cwd or not repo_root(cwd):
        return ""

    def _probe() -> str:
        gitdir = run_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if gitdir:
            gitdir = os.path.realpath(gitdir)
            if os.path.basename(gitdir) == ".git":
                return os.path.dirname(gitdir).replace(os.sep, "/")
        return repo_root(cwd)

    return _cache.resolve(f"common:{cwd}", _probe)


def resolve(cwd: str) -> dict | None:
    """Inject-able resolver for ``project_tree.build_tree``.

    Returns ``{"repo_root": <common root>, "worktree_root": <this checkout>}``
    or ``None`` when ``cwd`` is not in a git repo. ``build_tree`` treats
    ``worktree_root == repo_root`` as the main checkout.
    """
    worktree_root = repo_root(cwd)
    if not worktree_root:
        return None
    return {"repo_root": common_repo_root(cwd) or worktree_root, "worktree_root": worktree_root}


def warm_roots(cwds: Iterable[str], max_workers: int = _WARM_WORKERS) -> None:
    """Pre-resolve many cwds' roots in parallel (bounded) so a cold first paint
    doesn't serialize one git spawn per session cwd; results land in the cache."""
    pending = sorted({(cwd or "").strip() for cwd in cwds} - {""})
    if not pending:
        return
    if len(pending) == 1:
        resolve(pending[0])
        return
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
        list(pool.map(resolve, pending))
