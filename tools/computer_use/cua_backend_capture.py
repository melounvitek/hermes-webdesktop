"""Capture side of the cua-driver backend: window discovery, capture-target
selection and the capture()/list_windows()/list_apps()/focus_app() methods
(mixed into ``CuaDriverBackend``).

Logger name is kept as ``tools.computer_use.cua_backend`` so log-based tests
and operators see one backend logger.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.computer_use.cua_backend_input import _BTF_UNSUPPORTED_MSG
from tools.computer_use.cua_backend_parse import (
    _apps_from_windows,
    _image_dimensions_from_bytes,
    _image_from_tool_result,
    _ingest_windows,
    _is_placeholder_id,
    _is_real_app_window,
    _parse_elements_from_structured,
    _parse_elements_from_tree,
    _parse_xprop_net_active_window,
    _positive_int,
    _split_tree_text,
    _windows_from_tool_result,
    _z_index_uninformative,
)

logger = logging.getLogger("tools.computer_use.cua_backend")

# Whole-screen intents: app="screen"/... -> composited `get_desktop_state`
# (pixels only); app="desktop" -> the OS shell window via list_windows, WITH
# interactable elements (desktop icons, taskbar).
_FULL_SCREEN_SENTINELS = {"screen", "fullscreen", "full screen", "all"}
_DESKTOP_SHELL_SENTINELS = {"desktop"}
# Shell window identifiers (substring of app_name + title, case-insensitive).
# Windows: Progman/WorkerW = desktop, Shell_TrayWnd = taskbar; macOS: Finder/Dock.
_DESKTOP_WINDOW_NAMES = (
    "progman", "workerw", "program manager", "shell_traywnd", "taskbar",
    "finder", "desktop", "dock",
)
# Backdrop subset preferred over the taskbar when both are present.
_DESKTOP_BACKDROP_NAMES = ("progman", "workerw", "program manager", "finder", "desktop")

_WINDOW_TITLE_RE = re.compile(r'AXWindow\s+"([^"]+)"')
_LEGACY_APP_LINE_RE = re.compile(r'(.+?)\s+\(pid\s+(\d+)\)')

_NO_DESKTOP_WINDOW_MSG = (
    "<no desktop/shell window found for app={app!r}; cua-driver captures one "
    "window at a time and exposes no whole-virtual-desktop or per-monitor "
    "capture. Call list_apps / capture(app='<AppName>') to target a specific "
    "window instead. On Windows the taskbar is 'Shell_TrayWnd' and the desktop "
    "is 'Progman'.>"
)
_NO_APP_MATCH_MSG = (
    "<no on-screen window matched app={app!r}; call list_apps to see available "
    "app names or bundle IDs (macOS reports localized names, e.g. '計算機' "
    "instead of 'Calculator'; some Linux/Qt apps only resolve via list_apps "
    "metadata)>"
)
_NO_DESKTOP_IMAGE_MSG = (
    "<get_desktop_state returned no image; the driver may predate the desktop "
    "capture lane — try capture(app='<AppName>') for a specific window>"
)
_FULL_SCREEN_NOTE = (
    "full-screen capture has no interactable elements; to act on what you see, "
    "call capture(app='<AppName>') for that app's clickable element list, or "
    "capture(app='desktop') for the desktop shell (wallpaper icons / taskbar) "
    "with elements"
)


def _linux_x11_active_window_id() -> Optional[int]:
    """Best-effort read of ``_NET_ACTIVE_WINDOW`` via xprop. Never raises."""
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=2, check=False, stdin=subprocess.DEVNULL)
    except Exception:
        return None
    return _parse_xprop_net_active_window(proc.stdout or "") if proc.returncode == 0 else None


def _select_capture_target(
    windows: List[Dict[str, Any]],
    *,
    app_requested: bool,
    exact_target: bool = False,
) -> Dict[str, Any]:
    """Best window from z-sorted (frontmost-first) list_windows output.

    Unqualified default captures on Linux (no app filter, no exact target) skip
    desktop/shell helper windows first — targetable but capture as empty — and
    when every remaining candidate shares one ``z_index`` (the common X11 case)
    ``_NET_ACTIVE_WINDOW`` beats list order. Exact-target captures never pay
    for the ``xprop`` probe.
    """
    pool = [w for w in windows if not w["off_screen"]]
    if not exact_target and not app_requested and sys.platform == "linux":
        pool = [w for w in pool if _is_real_app_window(w)] or pool
        if pool and _z_index_uninformative(pool):
            active_id = _linux_x11_active_window_id()
            if active_id is not None:
                for w in pool:
                    if w.get("window_id") == active_id:
                        return w
    return pool[0] if pool else windows[0]


def _sorted_windows(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalised windows from a list_windows result, ``z_index`` DESCENDING
    (frontmost at index 0 — the default target for capture()/focus_app())."""
    windows = _ingest_windows(_windows_from_tool_result(out))
    windows.sort(key=lambda w: w["z_index"], reverse=True)
    return windows


def _tree_and_title(out: Dict[str, Any]) -> Tuple[str, str]:
    """``(tree_markdown, window_title)`` from a get_window_state result."""
    data = out.get("data")
    _, tree = _split_tree_text(data if isinstance(data, str) else "")
    match = _WINDOW_TITLE_RE.search(tree)
    return tree, (match.group(1) if match else "")


def _gws_is_empty(out: Dict[str, Any]) -> bool:
    """True when a get_window_state result carries neither a screenshot nor a
    parseable tree. Modern drivers put the payload in structuredContent with
    no markdown tree — that is NOT empty."""
    if out.get("images"):
        return False
    sc_ = out.get("structuredContent") or {}
    if sc_.get("elements") or sc_.get("screenshot_png_b64"):
        return False
    tree, _ = _tree_and_title(out)
    return not tree.strip()


def _png_metrics(png_b64: str, width: int, height: int) -> Tuple[int, int, int]:
    """Return ``(png_bytes_len, width, height)``, replacing the given size with
    the sniffed one when the bytes decode to a readable PNG/JPEG header."""
    try:
        raw = base64.b64decode(png_b64, validate=False)
        png_bytes_len = len(raw)
        detected_width, detected_height = _image_dimensions_from_bytes(raw)
        if detected_width and detected_height:
            width, height = detected_width, detected_height
    except Exception:
        png_bytes_len = len(png_b64) * 3 // 4
    return png_bytes_len, width, height


def _is_desktop_window(w: Dict[str, Any], names: Tuple[str, ...] = _DESKTOP_WINDOW_NAMES) -> bool:
    haystack = f"{w.get('app_name', '')} {w.get('title', '')}".lower()
    return any(name in haystack for name in names)


def _app_aliases(raw_app: Dict[str, Any]) -> set:
    return {
        value.strip().lower()
        for key in ("bundle_id", "bundleId", "name", "app_name", "display_name")
        if isinstance((value := raw_app.get(key)), str) and value.strip()
    }


class _CaptureMixin:
    """capture()/list_windows()/list_apps()/focus_app() and their window-discovery helpers."""

    # ── Failure plumbing ───────────────────────────────────────────
    def _failed_capture(self, mode: str, message: str = "") -> CaptureResult:
        """Return an empty capture after disarming any prior target context."""
        self._clear_active_target()
        return CaptureResult(mode=mode, width=0, height=0, png_b64=None, elements=[],
                             app="", window_title=message, png_bytes_len=0)

    def _call_capture_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a capture-stage tool and disarm state on transport or logical failure."""
        try:
            out = self._session.call_tool(name, args)
        except Exception:
            self._clear_active_target()
            raise
        if out.get("isError") is True:
            message = out.get("data")
            self._clear_active_target()
            raise RuntimeError(
                f"cua-driver {name} failed"
                + (f": {message}" if isinstance(message, str) and message else "")
            )
        return out

    def _cli_refetch(self, name: str, args: Dict[str, Any], timeout: float,
                     what: str) -> Optional[Dict[str, Any]]:
        """One-shot call over the CLI transport (different daemon socket) after
        MCP came back empty/imageless without raising. None on failure."""
        try:
            cli_out = self._session._call_tool_via_cli(name, args, timeout)
        except Exception as cli_exc:
            logger.error("cua-driver CLI re-fetch for %s failed: %s", what, cli_exc)
            return None
        if cli_out.get("isError") is True:
            if name == "list_windows":
                logger.error("cua-driver CLI re-fetch for list_windows returned an error")
            self._clear_active_target()
            return None
        return cli_out

    # ── Window discovery ───────────────────────────────────────────
    def _list_windows_args(self) -> Dict[str, Any]:
        return {"on_screen_only": True, "session": self._session_id}

    def _load_windows(self) -> List[Dict[str, Any]]:
        """Visible windows frontmost-first, re-fetching over the CLI transport
        when MCP returns nothing."""
        windows = _sorted_windows(self._call_capture_tool("list_windows", self._list_windows_args()))
        if windows:
            return windows
        logger.warning(
            "cua-driver list_windows returned no windows over MCP; "
            "re-fetching via CLI transport",
        )
        cli_out = self._cli_refetch("list_windows", self._list_windows_args(), 20.0, "list_windows")
        return _sorted_windows(cli_out) if cli_out is not None else []

    def _load_windows_or_disarm(self) -> List[Dict[str, Any]]:
        """``_load_windows`` that forgets the sticky target when discovery raises."""
        try:
            return self._load_windows()
        except Exception:
            self._clear_active_target()
            raise

    def _match_windows_for_app(
        self, windows: List[Dict[str, Any]], app: str
    ) -> List[Dict[str, Any]]:
        """Resolve ``app=``: exact window names, then exact list_apps aliases
        (Linux ``list_windows`` can omit the app name that ``list_apps`` keeps),
        then substrings — querying ``Code`` must not silently select
        ``Visual Studio Code`` because it is frontmost."""
        app_lower = app.strip().lower()
        if not app_lower:
            return []

        def _by_name(exact: bool) -> List[Dict[str, Any]]:
            if exact:
                return [w for w in windows if app_lower == str(w.get("app_name", "")).strip().lower()]
            return [w for w in windows if app_lower in str(w.get("app_name", "")).lower()]

        direct_exact = _by_name(exact=True)
        if direct_exact:
            return direct_exact

        try:
            running_apps = self.list_apps()
        except Exception as exc:
            # A title can still be the only usable identity on X11 when app
            # enumeration is unavailable, so keep the title fallback below.
            logger.debug("computer_use list_apps fallback failed for %r: %s", app, exc)
            running_apps = []

        exact_pids: set[int] = set()
        partial_pids: set[int] = set()
        for raw_app in running_apps:
            if not isinstance(raw_app, dict) or raw_app.get("running") is False:
                continue
            pid = _positive_int(raw_app.get("pid"))
            if pid is None:
                continue
            aliases = _app_aliases(raw_app)
            if app_lower in aliases:
                exact_pids.add(pid)
            elif any(app_lower in alias for alias in aliases):
                partial_pids.add(pid)

        for matched in (
            [w for w in windows if w.get("pid") in exact_pids],
            _by_name(exact=False),
            [w for w in windows if w.get("pid") in partial_pids],
        ):
            if matched:
                return matched

        # Some X11 backends expose a title but no app name. Restrict this final
        # fallback to nameless rows so a localized app name is not overridden
        # merely because its title happens to be in the caller's language.
        return [
            w for w in windows
            if not str(w.get("app_name", "")).strip()
            and app_lower in str(w.get("title", "")).lower()
        ]

    def _resolve_capture_windows(
        self,
        mode: str,
        app: Optional[str],
        pid: Optional[int],
        window_id: Optional[int],
    ) -> "List[Dict[str, Any]] | CaptureResult":
        """Candidate windows for capture(), or a failed CaptureResult."""
        if pid is not None or window_id is not None:
            # An exact pid/window pair is both the stable capture_after target
            # and the escape hatch when discovery is unavailable on X11.
            if pid is None or window_id is None:
                return self._failed_capture(mode, "<capture targeting requires both pid and window_id>")
            target_pid, target_window_id = _positive_int(pid), _positive_int(window_id)
            if target_pid is None or target_window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires positive integer pid and window_id>",
                )
            return [{"app_name": app or "", "pid": target_pid, "window_id": target_window_id,
                     "off_screen": False, "title": "", "z_index": 0}]

        windows = self._load_windows_or_disarm()
        if not windows:
            # Diagnose instead of a bare 0x0: the dominant real-world cause on
            # Linux is a locked desktop session.
            from tools.computer_use import cua_backend as _cb

            return self._failed_capture(mode, _cb._empty_discovery_reason())
        if not app:
            return windows

        if app.strip().lower() in _DESKTOP_SHELL_SENTINELS:
            # Desktop-shell request: the OS shell window WITH its interactable
            # elements (desktop icons), so "click the taskbar" works. Prefer the
            # backdrop (Progman/WorkerW/Finder) over the taskbar so the capture
            # shows the full desktop rather than the task strip.
            desktop = [w for w in windows if _is_desktop_window(w)]
            if not desktop:
                return self._failed_capture(mode, _NO_DESKTOP_WINDOW_MSG.format(app=app))
            return sorted(desktop, key=lambda w: 0 if _is_desktop_window(w, _DESKTOP_BACKDROP_NAMES) else 1)

        # When the filter matches nothing, say so instead of silently capturing
        # the frontmost window — on macOS list_windows returns the localized
        # app name (e.g. "計算機"), so `app="Calculator"` legitimately misses.
        return (self._match_windows_for_app(windows, app)
                or self._failed_capture(mode, _NO_APP_MATCH_MSG.format(app=app)))

    # ── Capture ────────────────────────────────────────────────────
    def _gws_args(self) -> Dict[str, Any]:
        return {"pid": self._active_pid, "window_id": self._active_window_id, "session": self._session_id}

    def _capture_vision(self) -> Tuple[Optional[str], Optional[str], str]:
        """Pixels only, no elements: ``(png_b64, mime, window_title)``.

        Drivers advertising the cheaper standalone ``screenshot`` tool use it;
        current drivers folded PNG capture into ``get_window_state`` (tree
        DISCARDED here). Before discovery ran we still try ``screenshot`` first
        and fall back, so the path self-heals on any driver version.
        """
        png_b64: Optional[str] = None
        image_mime_type: Optional[str] = None
        window_title = ""
        if self._session._has_tool("screenshot") or not self._session.capabilities_discovered:
            sc_out = self._call_capture_tool("screenshot", {
                "window_id": self._active_window_id, "format": "jpeg", "quality": 85,
                "session": self._session_id,
            })
            png_b64, image_mime_type = _image_from_tool_result(sc_out)
        if not png_b64:
            # "Unknown tool: screenshot" or an empty image part -> get_window_state.
            gws_out = self._call_capture_tool("get_window_state", self._gws_args())
            png_b64, image_mime_type = _image_from_tool_result(gws_out)
            # The title is cheap and useful; `elements` stays empty by contract.
            _, window_title = _tree_and_title(gws_out)
        if not png_b64:
            logger.warning(
                "cua-driver vision capture returned no image over MCP "
                "(window_id=%s); re-fetching via CLI transport",
                self._active_window_id,
            )
            cli_out = self._cli_refetch("get_window_state", self._gws_args(), 30.0, "vision screenshot")
            if cli_out is not None and cli_out.get("images"):
                png_b64, image_mime_type = cli_out["images"][0], "image/png"
        return png_b64, image_mime_type, window_title

    def _capture_window_state(self) -> Tuple[Optional[str], Optional[str], List[UIElement], str]:
        """AX tree + screenshot. Returns ``(png_b64, mime, elements, window_title)``."""
        gws_out = self._call_capture_tool("get_window_state", self._gws_args())
        # A flaky bridge can return a degenerate result (no screenshot AND no
        # parseable tree) WITHOUT raising — a silent 0x0 to the model. Distinct
        # from the EAGAIN path handled in call_tool: here MCP "succeeded".
        if _gws_is_empty(gws_out):
            logger.warning(
                "cua-driver get_window_state returned an empty result over MCP "
                "(pid=%s window_id=%s); re-fetching via CLI transport",
                self._active_pid, self._active_window_id,
            )
            cli_out = self._cli_refetch("get_window_state", self._gws_args(), 30.0, "get_window_state")
            if cli_out is not None and not _gws_is_empty(cli_out):
                gws_out = cli_out

        tree, window_title = _tree_and_title(gws_out)
        # Prefer the canonical structuredContent.elements (real frames); the
        # markdown regex fallback yields (0,0,0,0) bounds.
        sc_elements = (gws_out.get("structuredContent") or {}).get("elements")
        if isinstance(sc_elements, list) and sc_elements:
            elements = _parse_elements_from_structured(sc_elements)
        else:
            elements = _parse_elements_from_tree(tree) if tree else []
        # Tokens are tied to this snapshot: overwrite the whole map (and clear
        # it when the new capture carries none).
        self._snapshot_tokens = {e.index: e.element_token for e in elements if e.element_token}
        png_b64, image_mime_type = _image_from_tool_result(gws_out)
        return png_b64, image_mime_type, elements, window_title

    def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> CaptureResult:
        """Capture the frontmost on-screen window or an exact known target:
        `list_windows` + `get_window_state` (ax/som) or `screenshot` (vision).
        Only the structured ``structuredContent.windows`` shape is supported."""
        # Schema-filler ids (models zero-fill optional properties) must not read
        # as a targeting request.
        pid = None if _is_placeholder_id(pid) else pid
        window_id = None if _is_placeholder_id(window_id) else window_id
        exact_target = pid is not None or window_id is not None
        # Full-screen lane bypasses enumeration entirely (also keeps
        # screenshots working when Windows UIA enumeration hangs).
        # app='desktop' deliberately does NOT take it: desktop icons stay clickable.
        if not exact_target and app and app.strip().lower() in _FULL_SCREEN_SENTINELS:
            return self._capture_full_screen(mode)

        windows = self._resolve_capture_windows(mode, app, pid, window_id)
        if isinstance(windows, CaptureResult):
            return windows

        target = _select_capture_target(windows, app_requested=bool(app), exact_target=exact_target)
        self._set_active_target(target)
        app_name = target["app_name"]
        # Record the resolved app so capture_after= follow-ups re-target the
        # same app rather than falling back to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name or app or ""

        elements: List[UIElement] = []
        if mode == "vision":
            png_b64, image_mime_type, window_title = self._capture_vision()
        else:
            png_b64, image_mime_type, elements, window_title = self._capture_window_state()

        png_bytes_len, width, height = _png_metrics(png_b64, 0, 0) if png_b64 else (0, 0, 0)
        return CaptureResult(mode=mode, width=width, height=height, png_b64=png_b64,
                             elements=elements, app=app_name, window_title=window_title,
                             png_bytes_len=png_bytes_len, image_mime_type=image_mime_type)

    def _capture_full_screen(self, mode: str) -> CaptureResult:
        """Composited PrtScn-style grab via `get_desktop_state` (the shell window
        would only show wallpaper + icons). Never enumerates, so it also works
        when Windows UIA hangs. Pixels only — `elements` is empty and `note`
        points the model at the interactive lanes. ``capture_scope`` is switched
        to desktop for the call and restored afterwards."""
        self._clear_active_target()
        previous_scope: Optional[str] = None
        try:
            cfg = self._session.call_tool("get_config", {"session": self._session_id}, timeout=10.0)
            sc = cfg.get("structuredContent") or {}
            if isinstance(sc, dict) and isinstance(sc.get("capture_scope"), str):
                previous_scope = sc["capture_scope"]
        except Exception as e:
            logger.debug("cua-driver get_config before full-screen capture failed: %s", e)

        def _set_scope(value: str) -> None:
            self._session.call_tool(
                "set_config",
                {"key": "capture_scope", "value": value, "session": self._session_id},
                timeout=10.0,
            )

        try:
            if previous_scope != "desktop":
                _set_scope("desktop")
            out = self._call_capture_tool("get_desktop_state", {"session": self._session_id})
        finally:
            if previous_scope and previous_scope != "desktop":
                try:
                    _set_scope(previous_scope)
                except Exception as e:
                    logger.debug("cua-driver restore capture_scope failed: %s", e)

        png_b64, image_mime_type = _image_from_tool_result(out)
        if not png_b64:
            return self._failed_capture(mode, _NO_DESKTOP_IMAGE_MSG)
        structured = out.get("structuredContent") or {}
        png_bytes_len, width, height = _png_metrics(
            png_b64,
            int(structured.get("screenshot_width") or structured.get("screen_width") or 0),
            int(structured.get("screenshot_height") or structured.get("screen_height") or 0),
        )
        return CaptureResult(
            mode="vision", width=width, height=height, png_b64=png_b64, elements=[],
            app="screen", window_title="Full screen (composited)",
            png_bytes_len=png_bytes_len, image_mime_type=image_mime_type, note=_FULL_SCREEN_NOTE,
        )

    # ── Introspection ──────────────────────────────────────────────
    def list_windows(self) -> List[Dict[str, Any]]:
        return self._load_windows()

    def list_apps(self) -> List[Dict[str, Any]]:
        out = self._session.call_tool("list_apps", {"session": self._session_id})
        structured = out.get("structuredContent")
        data = out.get("data")
        # structuredContent is canonical; empty lists fall through so a
        # populated compatibility envelope (older drivers, CLI fallback) can
        # still recover.
        def _apps_in(container: Any) -> List[Any]:
            apps = container.get("apps") if isinstance(container, dict) else None
            return apps if isinstance(apps, list) else []

        if _apps_in(structured):
            return _apps_in(structured)
        if isinstance(data, list) and data:
            return data
        for container in (data, out):
            if _apps_in(container):
                return _apps_in(container)
        derived = _apps_from_windows(_windows_from_tool_result(out))
        if derived:
            return derived
        # Old text-only drivers retain a small, name/PID-only fallback.
        if isinstance(data, str):
            return [
                {"name": m.group(1).strip(), "pid": int(m.group(2))}
                for m in map(_LEGACY_APP_LINE_RE.search, data.splitlines()) if m
            ]
        return []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Pure window-selector (store pid/window_id so later input hits the
        right process) — background automation never needs to raise a window.
        ``raise_window=True`` is explicit, separately approved, and uses the
        standalone ``bring_to_front`` tool."""
        matched = self._match_windows_for_app(self._load_windows_or_disarm(), app)
        # No silent fallback to the frontmost window: that hides the real
        # failure (often a localized macOS app-name mismatch).
        if not matched:
            self._clear_active_target()
            return ActionResult(ok=False, action="focus_app",
                                message=f"No on-screen window found for app '{app}'.")
        target = matched[0]
        self._set_active_target(target)
        self._last_app = target["app_name"] or app  # retained for back-compat diagnostics
        if raise_window:
            if not self._session._has_tool("bring_to_front"):
                return ActionResult(ok=False, action="focus_app", code="bring_to_front_unsupported",
                                    message=_BTF_UNSUPPORTED_MSG)
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
            focused.action = "focus_app"
            focused.meta["target_selected"] = True
            return focused
        return ActionResult(ok=True, action="focus_app",
                            message=f"Targeted {target['app_name']} (pid {self._active_pid}, "
                                    f"window {self._active_window_id}) without raising window.")
