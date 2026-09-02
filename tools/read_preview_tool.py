#!/usr/bin/env python3
"""Read the in-app browser / preview pane in the Hermes desktop GUI.

The preview's content lives in the renderer (a sandboxed ``<webview>``), so this
round-trips through the gateway's blocking-prompt bridge like ``read_terminal``
(``preview.read.request`` -> ``preview.read.respond``). Registration moved into
`desktop_preview`; the agent dispatches action=read here with the injected callback.
"""

from typing import Callable, Optional

from tools.desktop_ui import passthrough_json
from tools.registry import tool_error


def read_preview_tool(
    start: Optional[int] = None,
    count: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Return the active preview tab's contents (+ metadata) as a JSON string."""
    if callback is None:
        return tool_error("read_preview is only available in the Hermes desktop app.")

    try:
        window = {
            key: max(floor, int(val))
            for key, val, floor in (("start", start, 0), ("count", count, 1))
            if val is not None
        }
    except (TypeError, ValueError):
        return tool_error("start and count must be integers.")

    try:
        raw = callback(**window)
    except Exception as exc:
        return tool_error(f"Failed to read the preview pane: {exc}")

    if not raw:
        return tool_error("No preview tab is open, or the read timed out.")
    return passthrough_json(raw)
