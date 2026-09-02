"""Abstract backend interface for computer use.

Any implementation (cua-driver over MCP, pyautogui, noop, future Linux/Windows)
returns the shapes below. All methods are synchronous; async is handled inside
the backend implementation if needed.
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_JPEG_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def image_dimensions_from_bytes(raw: bytes) -> Optional[Tuple[int, int]]:
    """Return (width, height) for PNG / JPEG bytes, or None when unreadable.

    PNG: IHDR. JPEG: walk segments (skipping 0xFF fill bytes) to the first SOF
    marker; stop at SOS. Used by the tool layer's provider min-size guard.
    """
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        try:
            width, height = struct.unpack(">II", raw[16:24])
            return int(width), int(height)
        except Exception:
            return None
    if raw.startswith(b"\xff\xd8") and len(raw) > 4:
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            while marker == 0xFF and i < len(raw):
                marker = raw[i]
                i += 1
            if marker in {0xD8, 0xD9}:
                continue
            if marker == 0xDA or i + 2 > len(raw):
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > len(raw):
                break
            if marker in _JPEG_SOF_MARKERS and segment_len >= 7:
                height = int.from_bytes(raw[i + 3:i + 5], "big")
                width = int.from_bytes(raw[i + 5:i + 7], "big")
                return int(width), int(height)
            i += segment_len
    return None


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int                       # 1-based SOM index
    role: str                        # AX role (AXButton, AXTextField, ...)
    label: str = ""                  # AXTitle / AXDescription / AXValue snippet
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""                    # owning bundle ID or app name
    pid: int = 0                     # owning process PID
    window_id: int = 0               # SkyLight / CG window ID
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Opaque per-snapshot handle from cua-driver. Passed alongside `index` for explicit
    # stale-detection: a stale token errors instead of silently re-resolving to a
    # different element. None for older drivers that lack the field.
    element_token: Optional[str] = None

    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture call.

    At least one of png_b64 / elements is populated depending on capture mode:
    mode="vision" → png_b64 only; mode="ax" → elements only; mode="som" (default)
    → both: the PNG already carries numbered overlays drawn by the backend and
    `elements` holds the matching index → element mapping.
    """

    mode: str
    width: int                      # screenshot width (logical px, pre-Anthropic-scale)
    height: int
    png_b64: Optional[str] = None
    elements: List[UIElement] = field(default_factory=list)
    app: str = ""                   # target app/window the elements were captured for
    window_title: str = ""
    png_bytes_len: int = 0          # raw bytes sent to Anthropic, for token estimation
    # MIME type of `png_b64` when the backend supplied it (cua-driver-rs emits `mimeType`
    # on every image part). None → consumers fall back to base64-prefix sniffing (older drivers).
    image_mime_type: Optional[str] = None
    # Guidance appended to the summary by capture lanes that intentionally return no
    # elements (e.g. full-screen composited grabs) to point the model at an interactive lane.
    note: str = ""


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait).

    ``ok`` is tool/transport success only — NOT the semantic verdict. Read
    ``effect`` / ``escalation`` (cua-driver's structured verdict) to decide the
    next rung of the verify → escalate ladder. All structured fields are optional
    and additive: an older driver that omits ``structuredContent`` leaves them
    ``None`` and behavior is unchanged.
    """

    ok: bool
    action: str
    message: str = ""                # human-readable summary
    capture: Optional[CaptureResult] = None  # trailing screenshot, when requested / always-on
    meta: Dict[str, Any] = field(default_factory=dict)  # debugging / telemetry extras
    # ── cua-driver structured verdict (additive; None on old drivers) ──
    verified: Optional[bool] = None  # AX read-back: True confirmed, False unconfirmed, None n/a
    effect: Optional[str] = None     # "confirmed" | "unverifiable" | "suspected_noop"
    # {"recommended": "px"|"foreground"|"page", "reason": str} — only when driver recommends climbing
    escalation: Optional[Dict[str, Any]] = None
    path: Optional[str] = None       # delivery rung that ran (e.g. "ax", "x11_pixel", "cgevent_fg")
    degraded: Optional[bool] = None  # AX walk found no actionable elements (act by px instead)
    delivery_mode: Optional[str] = None  # the delivery_mode the caller requested, echoed back
    code: Optional[str] = None       # refusal code, e.g. "background_unavailable", "desktop_scope_disabled"


class ComputerUseBackend(ABC):
    """Lifecycle: `start()` before first use, `stop()` at shutdown.

    Pointer/keyboard actions take ``delivery_mode`` (background (default) | foreground)
    and ``bring_to_front``; ``button`` is left | right | middle; ``modifiers`` a list of
    key names. ``element`` args are 1-based SOM indices from a prior capture.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend can be used on this host right now (check_fn gating, setup wizard)."""

    # ── Capture ─────────────────────────────────────────────────────
    @abstractmethod
    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult: ...

    # ── Pointer actions ─────────────────────────────────────────────
    @abstractmethod
    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[List[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[Tuple[int, int]] = None, to_xy: Optional[Tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[List[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[List[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        """`direction` is up | down | left | right; `amount` is wheel ticks."""

    # ── Keyboard ────────────────────────────────────────────────────
    @abstractmethod
    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult: ...

    @abstractmethod
    def key(self, keys: str, *, delivery_mode: Optional[str] = None,
            bring_to_front: bool = False) -> ActionResult:
        """Send a key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return'."""

    # ── Introspection ───────────────────────────────────────────────
    @abstractmethod
    def list_apps(self) -> List[Dict[str, Any]]:
        """Return running apps with bundle IDs, PIDs, window counts."""

    def list_windows(self) -> List[Dict[str, Any]]:
        """Visible native windows with PID and window identifiers. Optional compatibility
        hook: backends that predate window discovery stay instantiable and report none."""
        return []

    @abstractmethod
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Route input to `app` (by name or bundle ID). Default: focus without raise."""

    @abstractmethod
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a native value on an element (e.g. AXPopUpButton selection)."""

    def wait(self, seconds: float) -> ActionResult:
        """Default implementation: time.sleep."""
        import time
        time.sleep(max(0.0, min(seconds, 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")
