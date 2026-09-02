#!/usr/bin/env python3
"""Close the Hermes desktop GUI's preview pane, or one of its tabs.

Registration moved into `desktop_preview`; kept for its ``preview.close`` action. The
renderer drops the matching tab — or the whole pane when no url is given — for the
window that asked, never a background session's view.
"""

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target


def close_preview_tool(url: str = "") -> str:
    """Ask the desktop GUI to close the preview pane, or the tab for ``url``."""
    target = _normalize_target(url or "")
    return desktop_ui.emit_or_error(
        "preview.close",
        {"url": target},
        "Failed to close the preview pane: ",
        "The preview pane is only available in the Hermes desktop app.",
        {"success": True, "url": target},
    )
