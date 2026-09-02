"""
Resolve where the Photon sidecar runs from and where its Node deps live.

The sidecar source ships inside the installed plugin tree. Hosted/managed
images keep that tree read-only (``/opt/hermes``, EROFS on every install /
self-heal path), so resolution mirrors ``resolve_whatsapp_bridge_dir``:

1. ``PHOTON_SIDECAR_DIR`` env override — used as-is.
2. Source dir writable → run in place (dev installs).
3. Source dir read-only but ``node_modules`` is baked and current → run in
   place (the Dockerfile bakes deps with ``npm ci`` at build time).
4. Source dir read-only and deps missing/stale → mirror the sidecar source
   files to ``$HERMES_HOME/photon/sidecar`` (durable, writable) and return it.

The mirror is refreshed on every resolve by content compare (not mtime);
``node_modules`` is left alone so the adapter's lockfile-vs-install-marker
staleness check triggers the ``npm ci`` self-heal inside the mirror.

Import-light on purpose: both ``adapter.py`` and ``cli.py`` use it, and
resolution never happens at import time (it probes/copies on disk).
"""

from __future__ import annotations

import filecmp
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SOURCE_SIDECAR_DIR = Path(__file__).parent / "sidecar"

# Files that define the sidecar; node_modules is deliberately absent (baked
# on managed images or installed by npm in the mirror).
_MIRROR_FILES = ("index.mjs", "package.json", "package-lock.json", "patch-spectrum-mixed-attachments.mjs")

# Tests monkeypatch these module globals directly; the accessors honor a
# non-None value and only resolve/derive when unset.
_SIDECAR_DIR: Optional[Path] = None
# Written by `hermes photon install-sidecar` on npm failure so
# check_requirements() can surface the root cause later; cleared on success.
_NPM_ERROR_LOG: Optional[Path] = None
_NPM_ERROR_LOG_MAX_CHARS = 300


def dir_writable(path: Path) -> bool:
    """True when we can create files in ``path`` (probe, not stat — stat lies on
    root-squash / read-only bind mounts)."""
    probe = path / ".hermes-write-probe"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


_dir_writable = dir_writable


def _lock_newer_than_install(sidecar_dir: Path) -> bool:
    """True when the committed lockfile postdates npm's install marker
    (``node_modules/.package-lock.json``) — the same signal ``npm ci`` uses.
    False on any stat failure so an odd filesystem never blocks start."""
    lockfile = sidecar_dir / "package-lock.json"
    marker = sidecar_dir / "node_modules" / ".package-lock.json"
    try:
        return lockfile.stat().st_mtime > marker.stat().st_mtime
    except OSError:
        return False


def resolve_sidecar_dir(source_dir: Optional[Path] = None) -> Path:
    """Return the directory the sidecar should run from (see module doc)."""
    source = Path(source_dir) if source_dir is not None else SOURCE_SIDECAR_DIR
    override = os.getenv("PHOTON_SIDECAR_DIR")
    if override:
        return Path(override)
    if _dir_writable(source):
        return source
    # Read-only tree with baked, current deps: run in place (the sidecar never
    # writes inside its own directory).
    if (source / "node_modules").exists() and not _lock_newer_than_install(source):
        return source
    from hermes_constants import get_hermes_home
    mirror = get_hermes_home() / "photon" / "sidecar"
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in _MIRROR_FILES:
            src = source / name
            if not src.exists():
                continue
            dst = mirror / name
            if not dst.exists() or not filecmp.cmp(str(src), str(dst), shallow=False):
                shutil.copy2(str(src), str(dst))
        return mirror
    except OSError as exc:
        logger.warning(
            "[photon] install tree is read-only and mirroring the sidecar "
            "to %s failed (%s) — falling back to the read-only source dir; "
            "dependency installs will not be possible",
            mirror,
            exc,
        )
        return source


def _sidecar_dir() -> Path:
    """Sidecar runtime dir, resolved once on first use (never at import)."""
    global _SIDECAR_DIR
    if _SIDECAR_DIR is None:
        _SIDECAR_DIR = resolve_sidecar_dir()
    return _SIDECAR_DIR


def _npm_error_log() -> Path:
    """Path of the persisted npm-failure log (derived from the sidecar dir)."""
    if _NPM_ERROR_LOG is not None:
        return _NPM_ERROR_LOG
    return _sidecar_dir() / ".photon-npm-error.log"
