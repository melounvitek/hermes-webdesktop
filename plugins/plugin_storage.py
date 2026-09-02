"""Per-plugin persistent storage convention.

Plugins must NOT park state inside ``<hermes home>/plugins/<name>/`` — that is the
install dir, which ``hermes plugins remove`` deletes and ``update`` git-pulls into. The
sanctioned root is ``<hermes home>/plugin-data/<name>/``: user-owned, untouched by
install/update/remove, inspectable in one place. Secrets are deliberately NOT part of
this convention — credential reads go through ``agent.secret_scope`` / ``.env``.

Usage::

    from plugins.plugin_storage import plugin_data_dir, plugin_db

    state_file = plugin_data_dir("my-plugin") / "state.json"
    conn = plugin_db("my-plugin")             # <data dir>/data.db
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

__all__ = ["plugin_data_dir", "plugin_db"]

# Mirrors the plugin-name shape `hermes plugins install` accepts; anything else could
# escape the data root via separators or traversal.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"invalid plugin name for storage: {name!r}")
    return name


def plugin_data_dir(name: str) -> Path:
    """Return (and create) ``<hermes home>/plugin-data/<name>/``.

    Resolves ``get_hermes_home()`` on every call so it follows the active profile —
    don't cache the result across profile switches.
    """
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "plugin-data" / _validate_name(name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def plugin_db(name: str, filename: str = "data.db") -> sqlite3.Connection:
    """Open ``<data dir>/<filename>`` (created on first use).

    WAL so a dashboard reader and an agent-tool writer coexist; ``check_same_thread=False``
    matches the multi-threaded FastAPI/tool environment — the caller owns transaction discipline.
    """
    if Path(filename).name != filename or not filename:
        raise ValueError(f"invalid plugin db filename: {filename!r}")

    conn = sqlite3.connect(plugin_data_dir(name) / filename, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
