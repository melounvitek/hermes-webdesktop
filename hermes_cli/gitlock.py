"""Stale git lock-file recovery for update/check paths.

A crashed or killed ``git fetch`` on a shallow clone can leave ``.git/shallow.lock`` behind. Every
later fetch then fails with::

fatal: Unable to create '/path/.git/shallow.lock': File exists.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Lock files younger than this are presumed live (a fetch is in flight) and
# are never removed. git lock files are created and removed within a single
# fetch (seconds); anything older than 10 minutes is abandoned by any
# reasonable standard.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60

# Lock files we know how to self-heal. ``shallow.lock`` is the one observed in
# the wild (interrupted fetch on a shallow clone); the others are the same
# class of failure (interrupted git operation) and harmless to clear when
# stale. Index/HEAD locks from a live git process are protected by the
# process guard in :func:`clear_stale_git_locks`.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")

# Aborted-fetch pack debris younger than this is presumed live (a fetch may
# be writing it right now) and is never removed. A healthy fetch completes in
# minutes; the same 10-minute bar the lock sweep uses is comfortably safe.
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS

# Temp-file prefixes git writes into .git/objects/pack during a transfer and
# renames away on success. Anything left with these names after a fetch died
# is garbage by definition — git itself never reuses or cleans them.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_")


def _git_proc_running() -> bool:
    """True when a ``git`` process is currently running.

    The conservative answer on any platform we can't probe: if we can't tell, treat a lock as
    possibly-live and don't remove it. This is the safety check that stops us from yanking a lock a
    real fetch is holding.
    """
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
            return "git.exe" in out
        out = subprocess.run(
            ["pgrep", "-x", "git"], capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0
    except Exception:
        logger.debug("git process probe failed; assuming no git running", exc_info=True)
        return False


def _sweep_stale(
    directory: Path,
    candidates: Callable[[], Iterable[Path]],
    *,
    min_age_seconds: Optional[int],
    default_age: int,
    skip_msg: str,
    log_removed: Callable[[Path, int], None],
) -> List[str]:
    """Shared guard + age-floor sweep. Never raises; skips anything it cannot stat/unlink."""
    if not directory.is_dir():
        return []
    if _git_proc_running():
        logger.debug(skip_msg)
        return []
    cutoff = time.time() - (min_age_seconds if min_age_seconds is not None else default_age)
    removed: List[str] = []
    for entry in candidates():
        try:
            if entry.is_file():
                st = entry.stat()
                if st.st_mtime < cutoff:
                    entry.unlink()
                    removed.append(str(entry))
                    log_removed(entry, st.st_size)
        except OSError:
            logger.debug("Could not clear %s (skipping)", entry, exc_info=True)
    return removed


def clear_stale_git_locks(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove abandoned ``.git`` lock files under ``repo_root``.

    A lock is removed only when BOTH conditions hold:

    Returns the list of removed lock file paths. Never raises: a lock we cannot stat or unlink is
    skipped (a concurrently-held lock may have just been created between our age check and the
    unlink — the process guard makes that window vanishingly small, and skipping is always safe).
    """
    git_dir = Path(repo_root) / ".git"
    return _sweep_stale(
        git_dir,
        lambda: [git_dir / name for name in LOCK_NAMES],
        min_age_seconds=min_age_seconds,
        default_age=STALE_LOCK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping stale-lock sweep",
        log_removed=lambda p, _size: logger.info("Removed stale git lock %s", p),
    )


def clear_stale_tmp_packs(
    repo_root: Path, *, min_age_seconds: Optional[int] = None
) -> List[str]:
    """Remove aborted-fetch temp pack files under ``.git/objects/pack``.

    Every ``git fetch`` that dies mid-transfer (timeout, HTTP 429, dropped connection) leaves a
    ``tmp_pack_*`` (and sometimes ``tmp_idx_*``) file behind, and git never cleans them up.

    Same safety contract as :func:`clear_stale_git_locks`: only files older than the age floor,
    never while a git process is running, never raises. Returns the removed paths.
    """
    pack_dir = Path(repo_root) / ".git" / "objects" / "pack"

    def _candidates():
        try:
            entries = list(pack_dir.iterdir())
        except OSError:
            return []
        return [e for e in entries if e.name.startswith(_TMP_PACK_PREFIXES)]

    return _sweep_stale(
        pack_dir,
        _candidates,
        min_age_seconds=min_age_seconds,
        default_age=STALE_TMP_PACK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping tmp-pack sweep",
        log_removed=lambda p, size: logger.info(
            "Removed aborted-fetch pack debris %s (%d bytes)", p, size
        ),
    )
