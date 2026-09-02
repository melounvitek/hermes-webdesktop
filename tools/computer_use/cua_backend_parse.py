"""Pure parsing helpers for the cua-driver backend: MCP result flattening,
``list_windows`` / ``get_window_state`` payload normalisation, key combos.

No I/O, no module state — everything here is a function of its inputs, which
is what makes it safe to share between the MCP and CLI transports.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, UIElement

_MISSING = object()

# Linux/X11 can surface GNOME Shell / desktop backdrop windows before real app
# windows with no useful z-order. They are targetable X11 windows but capture
# as empty through get_window_state, so default app capture must skip them.
_NON_APP_WINDOW_TITLE_PREFIXES = (
    "@!",          # GNOME Shell background/monitor helper windows
    "Desktop",
    "gnome-shell",
    "GNOME Shell",
)

_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)'
    r'(?:'
      r'\s*=\s*"([^"]*)"'              # = "value"
      r'|\s+"([^"]*)"'                 # "value"
      r'|\s+\((?!\d+\))([^)]*)\)'      # (value) but not a pure-digit (order) number
    r')?'
    r'(?:\s+(?:\(\d+\)\s+)?id=([^\s\[\]]+))?',  # optional id=value (after an optional (order))
    re.MULTILINE,
)
"""Element line of the get_window_state AX-tree markdown.

cua-driver renders each actionable node as ``[N] AXRole`` followed by a label
in one of four forms — ``= "value"``, ``"quoted"``, ``(paren)``, ``id=Label``
(optionally after an ``(order)`` number). A parenthesised pure-digit group is
an ORDER index, not a label, and is excluded so the id= label wins. Group 1 is
the index, group 2 the role, groups 3-6 the label in whichever form matched.
"""


def _mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x rename.

    mcp 2.0 exposes snake_case attributes and keeps camelCase only as a
    serialization alias, so ``getattr(result, "isError", False)`` reads False
    for every result on 2.x and a denied call would look like a success.
    Deliberately duplicated from ``tools.mcp_tool.mcp_field`` so computer_use
    never loads the much larger config-driven MCP client module.
    """
    value = getattr(obj, snake, _MISSING)
    if value is not _MISSING:
        return value
    value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value


def _action_result_from(
    name: str,
    ok: bool,
    message: str,
    meta: Dict[str, Any],
    structured: Dict[str, Any],
    *,
    requested_delivery: Optional[str] = None,
) -> ActionResult:
    """Build an ActionResult, lifting cua-driver's structured verdict.

    structuredContent is canonical, the flattened ``meta`` copy the fallback.
    Every structured field is additive: a driver that omits one leaves the
    attribute ``None`` so old drivers see unchanged behavior.
    """
    sc = structured if isinstance(structured, dict) else {}

    def _pick(key: str) -> Any:
        return sc.get(key) if key in sc else meta.get(key)

    def _typed(value: Any, typ) -> Any:
        return value if isinstance(value, typ) else None

    return ActionResult(
        ok=ok,
        action=name,
        message=message,
        meta=meta,
        verified=_typed(_pick("verified"), bool),
        effect=_typed(_pick("effect"), str),
        escalation=_typed(_pick("escalation"), dict),
        path=_typed(_pick("path"), str),
        degraded=_typed(_pick("degraded"), bool),
        # What we asked for; the driver's `path` records the rung that ran.
        delivery_mode=_typed(requested_delivery, str),
        # Refusal/limitation code — drivers spell it "code" or "reason_code".
        code=_typed(_pick("code") or _pick("reason_code"), str),
    )


def _z_index_uninformative(windows: List[Dict[str, Any]]) -> bool:
    """True when every window shares the same z_index (common on Linux/X11)."""
    if not windows:
        return True
    return len({w.get("z_index", 0) for w in windows}) <= 1


def _parse_xprop_net_active_window(stdout: str) -> Optional[int]:
    """Parse ``xprop -root _NET_ACTIVE_WINDOW`` stdout into a window id.

    Accepts the ``window id # 0x...`` form, falling back to the first hex token.
    """
    text = stdout or ""
    match = re.search(r"window id # (0x[0-9a-fA-F]+)", text) or re.search(r"(0x[0-9a-fA-F]+)", text)
    if not match:
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


def _is_real_app_window(w: Dict[str, Any]) -> bool:
    """Return False for desktop/shell helper windows that capture as empty."""
    title = w.get("title", "")
    return not any(
        title.startswith(p) or title.lower().startswith(p.lower())
        for p in _NON_APP_WINDOW_TITLE_PREFIXES
    )


def _parse_elements_from_tree(markdown: str) -> List[UIElement]:
    """Parse UIElements from get_window_state AX-tree markdown.

    Last-resort fallback for drivers without ``structuredContent.elements``.
    Bounds always come back ``(0, 0, 0, 0)`` — the markdown carries none —
    which is fine for element-index clicks (the driver resolves the frame).
    """
    return [
        UIElement(
            index=int(m.group(1)),
            role=m.group(2),
            # groups 3-6: value / quoted / paren / id= label (first non-None wins)
            label=m.group(3) or m.group(4) or m.group(5) or m.group(6) or "",
            bounds=(0, 0, 0, 0),
        )
        for m in _ELEMENT_LINE_RE.finditer(markdown)
    ]


def _parse_elements_from_structured(raw_elements: List[Dict[str, Any]]) -> List[UIElement]:
    """Read the canonical ``structuredContent.elements`` array.

    Each entry has ``element_index``, ``role``, ``label`` and, when the AT-SPI /
    AXFrame call returned usable bounds, ``frame`` ``{x, y, w, h}`` — so real
    pixel bounds survive (the markdown path loses them). Malformed entries are
    skipped rather than failing the whole walk.
    """
    elements: List[UIElement] = []
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("element_index")
        if not isinstance(idx, int):
            continue
        role = raw.get("role") if isinstance(raw.get("role"), str) else ""
        label = raw.get("label") if isinstance(raw.get("label"), str) else ""
        frame = raw.get("frame") if isinstance(raw.get("frame"), dict) else None
        bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        if frame:
            try:
                bounds = tuple(int(frame.get(k, 0)) for k in ("x", "y", "w", "h"))  # type: ignore[assignment]
            except (TypeError, ValueError):
                bounds = (0, 0, 0, 0)
        # Opaque element_token (`s{snapshot_hex}:{index}`) — the driver owns
        # the parse + LRU semantics; we treat it as a black-box string.
        raw_token = raw.get("element_token")
        elements.append(UIElement(
            index=idx,
            role=role,
            label=label,
            bounds=bounds,
            element_token=raw_token if isinstance(raw_token, str) and raw_token else None,
        ))
    return elements


def _image_dimensions_from_bytes(raw: bytes) -> Tuple[int, int]:
    """Best-effort PNG/JPEG dimension sniffing without extra dependencies."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        if width > 0 and height > 0:
            return width, height

    if raw.startswith(b"\xff\xd8"):
        i = 2
        n = len(raw)
        while i + 9 < n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > n:
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > n:
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                if segment_len >= 7:
                    height = int.from_bytes(raw[i + 3:i + 5], "big")
                    width = int.from_bytes(raw[i + 5:i + 7], "big")
                    if width > 0 and height > 0:
                        return width, height
                break
            i += segment_len

    return 0, 0


def _split_tree_text(full_text: str) -> Tuple[str, str]:
    """Split get_window_state text into (summary_line, tree_markdown)."""
    summary, _, tree = full_text.partition("\n")
    return summary, tree


_MODIFIER_NAMES = frozenset({"cmd", "command", "shift", "option", "alt", "ctrl", "control", "fn"})
_KEY_ALIASES = {"command": "cmd", "alt": "option", "control": "ctrl"}


def _parse_key_combo(keys: str) -> Tuple[Optional[str], List[str]]:
    """Parse 'cmd+s' / 'ctrl-alt-t' into (key, modifiers); last non-modifier wins."""
    modifiers: List[str] = []
    key = None
    for part in (p.strip().lower() for p in re.split(r'[+\-]', keys) if p.strip()):
        normalized = _KEY_ALIASES.get(part, part)
        if normalized in _MODIFIER_NAMES:
            modifiers.append(normalized)
        else:
            key = part
    return key, modifiers


def _extract_tool_result(mcp_result: Any) -> Dict[str, Any]:
    """Flatten an mcp CallToolResult into
    ``{data, images, image_mime_types, structuredContent, isError}``.

    ``data`` is the joined text parts (parsed as JSON when it looks like JSON);
    ``image_mime_types`` is parallel to ``images`` with ``""`` where the part
    carried no mimeType (older drivers — callers then sniff the base64 prefix).
    """
    data: Any = None
    images: List[str] = []
    image_mime_types: List[str] = []
    # Identity, not truthiness: mocks/proxies synthesize truthy attributes.
    is_error = _mcp_field(mcp_result, "is_error", "isError", False) is True
    structured: Optional[Dict] = (
        _mcp_field(mcp_result, "structured_content", "structuredContent") or None
    )
    text_chunks: List[str] = []
    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_chunks.append(getattr(part, "text", "") or "")
        elif ptype == "image":
            b64 = getattr(part, "data", None)
            if b64:
                images.append(b64)
                image_mime_types.append(_mcp_field(part, "mime_type", "mimeType") or "")
    if text_chunks:
        joined = "\n".join(t for t in text_chunks if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined
    return {
        "data": data,
        "images": images,
        "image_mime_types": image_mime_types,
        "structuredContent": structured,
        "isError": is_error,
    }


def _image_from_tool_result(out: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Pull ``(b64, mime_type)`` out of a flattened tool result.

    cua-driver delivers screenshots either as an MCP ``image`` part
    (``out["images"]``) or as ``screenshot_png_b64`` inside structuredContent
    (newer builds, and the CLI transport); checking both keeps capture()
    robust when the driver moves the image between the two.
    """
    images = out.get("images") or []
    if images and images[0]:
        mimes = out.get("image_mime_types") or []
        return images[0], (mimes[0] if mimes and mimes[0] else None)
    structured = out.get("structuredContent") or {}
    b64 = structured.get("screenshot_png_b64") or structured.get("png_b64")
    if b64:
        return b64, (structured.get("screenshot_mime_type") or structured.get("mime_type") or None)
    return None, None


def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer, rejecting booleans and malformed values."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _is_placeholder_id(value: Any) -> bool:
    """True when *value* is a schema-filler id (``0`` / negative) rather than a target.

    Some providers emit every optional integer zero-filled; treating that as a
    targeting request would drop the caller's ``app=``. Non-numeric values are
    NOT placeholders — they still reach the validation error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return False
    try:
        return int(value) <= 0
    except ValueError:
        return False


def _ingest_windows(raw_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise cua-driver ``list_windows`` entries, dropping unusable ones.

    Every downstream call needs an integer ``pid`` and ``window_id``. On X11
    the PID comes from the optional ``_NET_WM_PID`` property, so root/panel/
    popup windows report ``pid: null`` — skip those instead of aborting the
    whole enumeration. ``z_index``: higher = closer to front; Wayland's null
    (undefined stacking) sorts lowest so real windows stay above the desktop.
    """
    windows: List[Dict[str, Any]] = []
    for w in raw_windows:
        if not isinstance(w, dict):  # untrusted compatibility envelopes
            continue
        pid_int = _positive_int(w.get("pid"))
        window_id_int = _positive_int(w.get("window_id"))
        if pid_int is None or window_id_int is None:
            continue
        z_raw = w.get("z_index")
        app_name = w.get("app_name", "")
        title = w.get("title", "")
        windows.append({
            "app_name": app_name if isinstance(app_name, str) else "",
            "pid": pid_int,
            "window_id": window_id_int,
            # Only explicit False means off-screen; null (Linux 0.6.x) means unknown.
            "off_screen": w.get("is_on_screen") is False,
            "title": title if isinstance(title, str) else "",
            "z_index": z_raw if isinstance(z_raw, (int, float)) and not isinstance(z_raw, bool) else 0,
        })
    return windows


def _first_nonempty_list(*containers: Any, keys: Tuple[str, ...]) -> List[Any]:
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, list) and value:
                return value
    return []


def _windows_from_tool_result(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list_windows payloads across cua-driver result shapes."""
    structured = out.get("structuredContent")
    if isinstance(structured, dict):
        windows = structured.get("windows")
        if isinstance(windows, list) and windows:
            return windows
    return _first_nonempty_list(out.get("data"), out, keys=("windows", "_legacy_windows"))


def _apps_from_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apps: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for summary in _ingest_windows(windows):
        name = summary["app_name"]
        key = (name, summary["pid"])
        if not name or key in seen:
            continue
        seen.add(key)
        apps.append({"name": name, "pid": summary["pid"]})
    return apps
