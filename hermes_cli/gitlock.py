"""Stale git lock-file and aborted-fetch pack-debris recovery for update/check paths.

A crashed or killed ``git fetch`` can leave ``.git/shallow.lock`` behind (every later fetch then
fails with "Unable to create '.../shallow.lock': File exists") and ``tmp_pack_*`` files under
``.git/objects/pack`` that git itself never cleans up.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Files younger than this are presumed live (a fetch may be in flight) and are never removed. git
# lock files live for seconds and a healthy fetch completes in minutes; 10 minutes is abandoned by
# any reasonable standard.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS

# ``shallow.lock`` is the one observed in the wild; the others are the same class of failure
# (interrupted git operation). Locks held by a live git process are protected by the process guard.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")

# Temp-file prefixes git writes into .git/objects/pack during a transfer and renames away on
# success. Anything left with these names after a fetch died is garbage by definition.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_")


def _git_proc_running() -> bool:
    """True when a ``git`` process is currently running.

    This is the safety check that stops us from yanking a lock a real fetch is holding. A failed
    probe logs and returns False; the age floor in the sweep still applies.
    """
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
            return "git.exe" in out
        return subprocess.run(["pgrep", "-x", "git"], capture_output=True, text=True, timeout=10).returncode == 0
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
    """Remove abandoned ``.git`` lock files under ``repo_root``; returns the removed paths.

    A lock is removed only when it is older than the age floor AND no git process is running. Never
    raises: a lock we cannot stat or unlink is skipped (a concurrently-held lock may have been
    created between the age check and the unlink; skipping is always safe).
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


def clear_stale_tmp_packs(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove aborted-fetch temp pack files under ``.git/objects/pack``.

    Same safety contract as :func:`clear_stale_git_locks`. Returns the removed paths.
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
        log_removed=lambda p, size: logger.info("Removed aborted-fetch pack debris %s (%d bytes)", p, size),
    )
