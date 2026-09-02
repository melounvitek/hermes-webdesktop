#!/usr/bin/env python3
"""Persistent element annotations in the Hermes desktop GUI's in-app browser.

``drive_preview`` draws transient marks (one per action, self-retiring). An
annotation outlines an element — or, with ``hold``, the whole visible field —
and stays until the agent removes it. Annotations bind to elements, not
coordinates: they ride scrolls/reflows and vanish with their element, so a
navigation clears them. Rides the same ``preview.act`` bridge as
``drive_preview`` (the renderer resolves ``@e`` refs and owns the overlay).
Lives in the ``desktop_ui`` toolset, enabled only for desktop-sourced sessions.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

ACTIONS = ("add", "hold", "remove", "clear")

# Verbs the renderer knows, keyed by ours. `clear` is `unpin` with nothing to
# aim at, which the overlay reads as "all of them".
WIRE = {"add": "pin", "hold": "hold", "remove": "unpin", "clear": "unpin"}


def annotate_preview_tool(
    action: str = "add",
    ref: Optional[str] = None,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Put one annotation up, take one down, or clear them all."""
    if callback is None:
        return tool_error("annotate_preview is only available in the Hermes desktop app.")

    verb = (action or "add").strip().lower()
    if verb not in ACTIONS:
        return tool_error(f"action must be one of: {', '.join(ACTIONS)}.")

    if verb in ("add", "remove") and not (ref or selector):
        return tool_error(
            f"{verb} needs a ref from drive_preview action='elements' "
            "(e.g. 'btn-sign-in') or a CSS selector."
        )

    payload = {
        name: val
        for name, val in (
            ("action", WIRE[verb]),
            ("ref", None if verb in ("clear", "hold") else ref),
            ("selector", None if verb in ("clear", "hold") else selector),
            ("text", label),
        )
        if val is not None
    }

    try:
        raw = callback(payload)
    except Exception as exc:
        return tool_error(f"Failed to annotate the in-app browser: {exc}")

    if not raw:
        return tool_error(
            "The annotation timed out, or no GUI window answered. "
            "Open a page with open_preview first."
        )

    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)


ANNOTATE_PREVIEW_SCHEMA = {
    "name": "annotate_preview",
    "description": (
        "Highlight elements on the preview-pane page, lastingly (drive_preview's own "
        "marks fade; annotations stay until removed) — point at findings, "
        "flag what you're about to change, keep your place. Use the refs "
        "from drive_preview action='elements'. add: outline one element "
        "(optional short label — a word or two, drawn on the page). hold: "
        "freeze the whole visible field, every element outlined and named. "
        "remove/clear: take one/all down. Marks follow their element on "
        "scroll; navigation clears them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "description": "Defaults to 'add'.",
            },
            "ref": {
                "type": "string",
                "description": "Ref from drive_preview elements.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector fallback. Prefer ref.",
            },
            "label": {
                "type": "string",
                "description": "Optional caption, e.g. 'cheapest'.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="annotate_preview",
    toolset="desktop_ui",
    schema=ANNOTATE_PREVIEW_SCHEMA,
    handler=lambda args, **kw: annotate_preview_tool(
        action=args.get("action", "add"),
        ref=args.get("ref"),
        selector=args.get("selector"),
        label=args.get("label"),
        callback=kw.get("callback"),
    ),
    emoji="🔖",
)
