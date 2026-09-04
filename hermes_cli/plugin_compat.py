"""Once-per-process notice for external plugins still importing from pre-decomposition module paths.

The Sep 2026 decomposition (PR #102117) moved most of Hermes's internals into ``<stem>_<topic>`` sibling
modules. Old paths keep resolving through ``PLUGIN-COMPAT`` blocks so plugins have time to update; each
resolution through such a block calls :func:`warn_once` so the plugin author sees, exactly once per process
per name, where the code went and when the old path disappears. Internal Hermes code never reaches this
(``scripts/check_compat_pointers.py`` fails CI if it does).

This module is part of the compat layer and is removed with it.
"""
from __future__ import annotations

import os
import warnings

# Set by the maintainers when the removal is scheduled; surfaces in every warning and in COMPAT_MANIFEST.md.
COMPAT_REMOVAL = os.environ.get("HERMES_PLUGIN_COMPAT_REMOVAL", "the next minor release after the announced date (see COMPAT_MANIFEST.md)")


class HermesPluginCompatWarning(FutureWarning):
    """A plugin imported a name from its pre-decomposition module path."""


_seen: set[tuple[str, str]] = set()


def warn_once(facade: str, name: str, target_module: str, target_name: str) -> None:
    key = (facade, name)
    if key in _seen:
        return
    _seen.add(key)
    new = f"{target_module}.{target_name}" if target_name != name else f"{target_module}.{name}"
    warnings.warn(
        f"hermes plugin compat: `{facade}.{name}` moved to `{new}`. The old path is kept only for external "
        f"plugins and is removed in {COMPAT_REMOVAL}; update your import.",
        HermesPluginCompatWarning,
        stacklevel=3,
    )
