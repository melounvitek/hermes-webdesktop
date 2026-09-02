"""Input side of the cua-driver backend: delivery-mode handling and the
pointer / keyboard / value-setter methods (mixed into ``CuaDriverBackend``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_parse import _parse_key_combo


_NO_TARGET_MSG = "No active window — call capture() first."

_BTF_UNSUPPORTED_MSG = "The connected cua-driver does not advertise the standalone bring_to_front tool."


class _InputMixin:
    """Pointer / keyboard / value-setter actions against the sticky target."""

    def _no_target(self, action: str, *, need_window: bool = False) -> Optional[ActionResult]:
        if self._active_pid is None or (need_window and self._active_window_id is None):
            return ActionResult(ok=False, action=action, message=_NO_TARGET_MSG)
        return None

    # ── Input delivery ─────────────────────────────────────────────
    def _apply_delivery(
        self,
        action: str,
        args: Dict[str, Any],
        delivery_mode: Optional[str],
    ) -> Optional[ActionResult]:
        """Attach delivery_mode to an input-action args dict.

        Background is the default and needs no flag. Foreground is only sent
        when the live action schema accepts it; on an older driver we refuse
        with ``foreground_unsupported`` instead of silently downgrading to
        background (which would land input where the model didn't expect).
        Returns an ActionResult to short-circuit on refusal, or None to proceed.
        """
        if not delivery_mode or delivery_mode == "background":
            return None
        if delivery_mode != "foreground":
            return ActionResult(ok=False, action=action, code="bad_delivery_mode",
                                message=f"unknown delivery_mode {delivery_mode!r} — use background|foreground.")
        if not self._session.supports_input_property(action, "delivery_mode"):
            return ActionResult(
                ok=False, action=action, code="foreground_unsupported", delivery_mode="foreground",
                message=("The connected cua-driver action schema does not accept "
                         "delivery_mode, so foreground delivery is unavailable. "
                         "Use another verified rung without assuming the reported "
                         "package version describes the live schema."),
            )
        args["delivery_mode"] = "foreground"
        return None

    def _run_input_action(
        self,
        action: str,
        args: Dict[str, Any],
        delivery_mode: Optional[str],
        bring_to_front: bool,
    ) -> ActionResult:
        """Apply one delivery rung, optionally focusing via its own tool.

        ``bring_to_front`` is never an input-action property: when requested,
        the separately approved standalone focus action runs first, then the
        original foreground input runs unchanged.
        """
        refusal = self._apply_delivery(action, args, delivery_mode)
        if refusal is not None:
            return refusal
        if bring_to_front:
            if delivery_mode != "foreground":
                return ActionResult(ok=False, action=action, code="bring_to_front_requires_foreground",
                                    message="bring_to_front requires delivery_mode='foreground'.")
            if not self._session._has_tool("bring_to_front"):
                return ActionResult(ok=False, action=action, code="bring_to_front_unsupported",
                                    delivery_mode="foreground", message=_BTF_UNSUPPORTED_MSG)
            if self._active_pid is None or self._active_window_id is None:
                return ActionResult(
                    ok=False, action=action, code="bring_to_front_target_required",
                    delivery_mode="foreground",
                    message="Capture an exact target before requesting persistent foreground focus.",
                )
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
        result = self._action(action, args)
        if bring_to_front:
            result.meta["foreground_focus"] = {"invoked": True, "tool": "bring_to_front"}
        return result

    # ── Pointer ────────────────────────────────────────────────────
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("click")
        if missing is not None:
            return missing
        # Tool is chosen by click_count only; `button` goes through click's
        # enum (the driver rejects unknown buttons). `right_click` /
        # `middle_click` MCP tools are deprecated aliases and never invoked here.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return ActionResult(ok=False, action="click",
                                message=f"unknown button {button!r} — expected left, right, middle.")
        tool = "double_click" if click_count == 2 else "click"

        args: Dict[str, Any] = {"pid": self._active_pid, "button": button_norm}
        if element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for element_index click.")
            args["element_index"] = element
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for coordinate click.")
            args["x"] = x
            args["y"] = y
        else:
            return ActionResult(ok=False, action=tool, message="click requires element= or x/y.")
        args["window_id"] = self._active_window_id
        if modifiers:
            args["modifier"] = modifiers
        return self._run_input_action(tool, args, delivery_mode, bring_to_front)

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("drag")
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid}
        if from_element is not None and to_element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for element-based drag.")
            args["from_element"] = from_element
            args["to_element"] = to_element
        elif from_xy is not None and to_xy is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for coordinate drag.")
            args["from_x"], args["from_y"] = int(from_xy[0]), int(from_xy[1])
            args["to_x"], args["to_y"] = int(to_xy[0]), int(to_xy[1])
        else:
            return ActionResult(ok=False, action="drag",
                                message="drag requires from_element/to_element or from_coordinate/to_coordinate.")
        args["window_id"] = self._active_window_id
        return self._run_input_action("drag", args, delivery_mode, bring_to_front)

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
        delivery_mode: Optional[str] = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        missing = self._no_target("scroll")
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid, "direction": direction,
                                "amount": max(1, min(50, amount))}
        if element is not None and self._active_window_id is not None:
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="scroll",
                                    message="No active window_id for coordinate scroll.")
            # Some driver schemas reject x/y on scroll: only send coordinates
            # when the driver advertises support; otherwise it scrolls the
            # targeted window (window_id is still sent for routing).
            if self._session.supports_capability("input.scroll.coordinates", tool="scroll"):
                args["x"] = x
                args["y"] = y
            args["window_id"] = self._active_window_id
        return self._run_input_action("scroll", args, delivery_mode, bring_to_front)

    # ── Keyboard ───────────────────────────────────────────────────
    def type_text(self, text: str, *, delivery_mode: Optional[str] = None,
                  bring_to_front: bool = False) -> ActionResult:
        missing = self._no_target("type_text", need_window=True)
        if missing is not None:
            return missing
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id, "text": text}
        return self._run_input_action("type_text", args, delivery_mode, bring_to_front)

    def key(self, keys: str, *, delivery_mode: Optional[str] = None,
            bring_to_front: bool = False) -> ActionResult:
        missing = self._no_target("key", need_window=True)
        if missing is not None:
            return missing
        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return ActionResult(ok=False, action="key",
                                message=f"Could not parse key from '{keys}'.")
        args: Dict[str, Any] = {"pid": self._active_pid, "window_id": self._active_window_id}
        if modifiers:  # hotkey requires at least one modifier + one key
            args["keys"] = modifiers + [key_name]
            return self._run_input_action("hotkey", args, delivery_mode, bring_to_front)
        args["key"] = key_name
        return self._run_input_action("press_key", args, delivery_mode, bring_to_front)

    # ── Value setter ────────────────────────────────────────────────
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        missing = self._no_target("set_value", need_window=True)
        if missing is not None:
            return missing
        if element is None:
            return ActionResult(ok=False, action="set_value",
                                message="set_value requires element= (element index).")
        return self._action("set_value", {"pid": self._active_pid, "window_id": self._active_window_id,
                                          "element_index": element, "value": value})
