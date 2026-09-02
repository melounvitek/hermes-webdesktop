"""Symlink-safe creation helpers for spill/cache files.

Spill files (terminal output, hook context, subagent summaries, web_extract
text) live in predictable directories under ``~/.hermes``; a plain
``open(path, "w")`` there would follow a pre-planted symlink and let a local
process redirect the write onto ``~/.bashrc``, ``authorized_keys``, etc.

Every helper refuses symlinks by construction: new files use
``O_CREAT | O_EXCL`` (fails on ANY existing path, including a dangling link);
overwrites ``lstat`` + ``unlink`` the existing path first (removes the link,
never its target) and then create exclusively, so the pair can't be raced.

Privacy tiers: ``private=True`` (default) forces ``0o700`` dirs / ``0o600``
files for spills that may hold pre-redaction secrets; ``private=False`` keeps
umask-default perms for cache dirs bind-mounted into remote terminal backends
(``credential_files._CACHE_DIRS``), where a non-root container UID must read them.

Disk failures are the caller's concern: helpers raise ``OSError``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO

__all__ = [
    "ensure_spill_dir",
    "open_exclusive",
    "write_text_exclusive",
]

# O_NOFOLLOW is POSIX-only; on Windows O_EXCL alone already refuses every
# pre-existing path.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def ensure_spill_dir(path: Path, *, private: bool = True) -> Path:
    """Create ``path`` (and parents) as a directory, refusing symlinks.

    ``private=True`` creates the leaf ``0o700`` and tightens an existing leaf to
    ``0o700``. Raises ``OSError`` if the leaf is not a real directory.
    """
    path = Path(path)
    path.mkdir(mode=0o700 if private else 0o777, parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"spill dir is not a directory (symlink?): {path}")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def open_exclusive(
    path: Path,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> IO[str]:
    """Open ``path`` for writing via exclusive create; never follows a link.

    ``overwrite=True`` first unlinks an existing path (``lstat``-checked, so only
    the link itself is removed and directories are refused), then creates
    exclusively — the overwrite path cannot be redirected through a symlink either.
    """
    path = Path(path)
    if overwrite:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(st.st_mode):
                raise OSError(f"refusing to overwrite a directory: {path}")
            os.unlink(path)
    mode = 0o600 if private else 0o666  # non-private honors umask
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, mode)
    try:
        return os.fdopen(fd, "w", encoding=encoding, errors=errors)
    except Exception:
        os.close(fd)
        raise


def write_text_exclusive(
    path: Path,
    text: str,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> None:
    """``Path.write_text`` equivalent that refuses to follow symlinks."""
    with open_exclusive(
        path, private=private, overwrite=overwrite, encoding=encoding, errors=errors
    ) as fh:
        fh.write(text)
