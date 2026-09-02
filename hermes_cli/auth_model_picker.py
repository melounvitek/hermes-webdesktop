"""Interactive model picker used after OAuth login.

Split out of ``hermes_cli/auth.py``; every moved name is re-imported there, so
``hermes_cli.auth.<name>`` keeps resolving (and monkeypatching) as before. Origin-internal
helpers are imported lazily inside each function (no import cycle; patches on
``hermes_cli.auth.<helper>`` still intercept).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Dict, List, Optional
from hermes_cli.auth_constants import DEFAULT_NOUS_PORTAL_URL

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")


def _confirm_selection_guards(
    model_id: str,
    *,
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
    include_kinds: Optional[List[str]] = None,
) -> bool:
    """Prompt before saving a model that trips any selection guard.

    Runs the unified guard registry (cost, data-policy, future guards) and shows one [y/N] confirm
    listing every warning that fired. Returns True to proceed, False to cancel.
    """
    try:
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )

        warnings = selection_warnings(
            model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            include_kinds=include_kinds,
        )
    except Exception:
        warnings = []
    if not warnings:
        return True

    print()
    print("=" * 72)
    print(combined_message(warnings))
    print("=" * 72)
    try:
        response = input("Switch anyway? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return response in {"y", "yes"}


class _ModelPickerRows:
    """Column-aligned model rows (name + $/Mtok prices + Nous sale chrome) for the model picker.

    Sale chrome (★ / -N% / was) is drawn as curses/ANSI segments (yellow % / dim "was"), not baked
    into one plain string — curses addnstr would otherwise render escape bytes literally.
    """

    def __init__(
        self,
        all_models: List[str],
        pricing: Optional[Dict[str, Dict[str, str]]],
        *,
        current_model: str,
        sale_chrome: bool,
    ) -> None:
        from hermes_cli.models import _format_price_per_mtok, compute_sale_discount

        self.current_model = current_model
        self.has_pricing = bool(pricing and any(pricing.get(m) for m in all_models))
        # Leave room for a leading "★ " on sale rows (Nous only).
        name_pad = 3 if sale_chrome else 2
        self.name_col = (
            max((len(m) for m in all_models), default=0) + name_pad
            if self.has_pricing
            else 0
        )
        # (inp, out, cache, pct|None, was_inp, was_out)
        self._price_cache: dict[str, tuple[str, str, str, int | None, str, str]] = {}
        self.price_col = 3  # minimum width
        self.cache_col = 0  # only set if any model has cache pricing
        self.has_cache = False
        self.any_on_sale = False
        if not self.has_pricing:
            return
        for mid in all_models:
            p = pricing.get(mid)  # type: ignore[union-attr]
            pct: int | None = None
            was_inp = was_out = ""
            if p:
                inp = _format_price_per_mtok(p.get("prompt", ""))
                out = _format_price_per_mtok(p.get("completion", ""))
                cache_read = p.get("input_cache_read", "")
                cache = _format_price_per_mtok(cache_read) if cache_read else ""
                if cache:
                    self.has_cache = True
                if sale_chrome:
                    sale = compute_sale_discount(
                        p.get("prompt", ""),
                        p.get("completion", ""),
                        p.get("original"),
                    )
                    if sale is not None:
                        self.any_on_sale = True
                        pct, was_prompt_raw, was_out_raw = sale
                        # Natively-free models (no gateway original) carry
                        # empty was_* raws — leave them empty so the row
                        # shows bare "-100%" with no "was ?/?" suffix.
                        if was_prompt_raw == "" and was_out_raw == "":
                            was_inp = was_out = ""
                        else:
                            was_inp = (
                                _format_price_per_mtok(was_prompt_raw)
                                if was_prompt_raw != ""
                                else "?"
                            )
                            was_out = (
                                _format_price_per_mtok(was_out_raw)
                                if was_out_raw != ""
                                else "?"
                            )
            else:
                inp, out, cache = "", "", ""
            self._price_cache[mid] = (inp, out, cache, pct, was_inp, was_out)
            self.price_col = max(self.price_col, len(inp), len(out))
            self.cache_col = max(self.cache_col, len(cache))
        if self.has_cache:
            self.cache_col = max(self.cache_col, 5)  # minimum: "Cache" header

    def segments(self, mid: str) -> list[tuple[str, str | None]]:
        """Build a rich radiolist row: yellow ★/% , dim was, plain prices."""
        if not self.has_pricing:
            segs: list[tuple[str, str | None]] = [(mid, None)]
            if mid == self.current_model:
                segs.append(("  ← currently in use", None))
            return segs

        inp, out, cache, pct, was_inp, was_out = self._price_cache.get(
            mid, ("", "", "", None, "", "")
        )
        on_sale = pct is not None
        # Reserve 2 columns for "★ " so sale and non-sale names share alignment.
        star_w = 2
        if on_sale:
            name_segs: list[tuple[str, str | None]] = [
                ("★ ", "yellow"),
                (f"{mid:<{self.name_col - star_w}}", None),
            ]
        else:
            name_segs = [(f"{mid:<{self.name_col}}", None)]

        price_part = f" {inp:>{self.price_col}}  {out:>{self.price_col}}"
        if self.has_cache:
            price_part += f"  {cache:>{self.cache_col}}"
        segs = [*name_segs, (price_part, None)]
        if on_sale:
            segs.append((f"  -{pct}%", "yellow"))
            if was_inp or was_out:
                segs.append((f"  was {was_inp}/{was_out}", "dim"))
        if mid == self.current_model:
            segs.append(("  ← currently in use", None))
        return segs

    def label(self, mid: str) -> str:
        return "".join(text for text, _style in self.segments(mid))

    def menu_title(self) -> str:
        """``Select default model:`` plus an aligned pricing header hint when priced."""
        title = "Select default model:"
        if self.has_pricing:
            # Align the header with the model column.
            # Each choice is "  {label}" (2 spaces) and we prepend
            # a 3-char cursor region ("-> " or "   "), so content starts at col 5.
            pad = " " * 5
            header = f"\n{pad}{'':>{self.name_col}} {'In':>{self.price_col}}  {'Out':>{self.price_col}}"
            if self.has_cache:
                header += f"  {'Cache':>{self.cache_col}}"
            # Legend lives on the column-header line so it reads as a key
            # (★ = on sale), not a fake menu row.
            title += header + "  $/Mtok"
            if self.any_on_sale:
                title += "  ★ = on sale"
        return title


def _prompt_model_selection(
    model_ids: List[str],
    current_model: str = "",
    pricing: Optional[Dict[str, Dict[str, str]]] = None,
    unavailable_models: Optional[List[str]] = None,
    portal_url: str = "",
    unavailable_message: str = "",
    confirm_provider: str = "",
    confirm_base_url: str = "",
    confirm_api_key: str = "",
) -> Optional[str]:
    """Interactive model picker; current_model listed first. Returns the chosen model ID or None.

    With *pricing* (``{model_id: {prompt, completion}}``) a compact price column is shown; models in
    *unavailable_models* render grayed out and unselectable with an upgrade link to *portal_url*.
    """
    from hermes_cli.cli_output import line_input

    _unavailable = unavailable_models or []
    # Sale chrome (★ / -N% / was) is Nous Portal-only — never for OpenRouter
    # or other providers even if pricing.original is somehow present.
    sale_chrome = (confirm_provider or "").strip().lower() == "nous"

    def _confirmed_selection(mid: str) -> Optional[str]:
        if not mid:
            return None
        # Unified guard registry (hermes_cli.model_selection_guards): the cost
        # guard only runs when a provider is known (pricing lookups need one);
        # id-keyed guards like the data-policy guard always run — they must
        # fire even via a custom endpoint or gateway.
        _kinds = None if confirm_provider else ["data_policy"]
        if not _confirm_selection_guards(
            mid,
            provider=confirm_provider,
            base_url=confirm_base_url,
            api_key=confirm_api_key,
            include_kinds=_kinds,
        ):
            return None
        return mid

    # Reorder: current model first, then the rest (deduplicated)
    ordered = []
    if current_model and current_model in model_ids:
        ordered.append(current_model)
    for mid in model_ids:
        if mid not in ordered:
            ordered.append(mid)

    # All models for column-width computation (selectable + unavailable)
    rows = _ModelPickerRows(
        list(ordered) + list(_unavailable), pricing,
        current_model=current_model, sale_chrome=sale_chrome,
    )
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    # Default cursor on the current model (index 0 if it was reordered to top)
    default_idx = 0
    menu_title = rows.menu_title()
    _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")

    # Try arrow-key menu first, fall back to number input.
    try:
        from hermes_cli.curses_ui import curses_radiolist

        choices = [rows.segments(mid) for mid in ordered]
        choices.append("Enter custom model name")
        choices.append("Skip (keep current)")

        unavailable_footer = unavailable_message.strip()
        if not unavailable_footer and _unavailable:
            unavailable_footer = f"Upgrade at {_upgrade_url} for paid models"

        # The pricing column header (and any unavailable-models block) is shown
        # as a multi-line description above the list so it survives the curses
        # screen clear. menu_title already embeds the aligned price header.
        desc_lines: list[str] = []
        if rows.has_pricing:
            # menu_title is "Select default model:\n<pad><header>  $/Mtok\n…"
            # Keep only the header/legend portion for the description.
            header_part = menu_title.split("\n", 1)
            if len(header_part) > 1:
                desc_lines.extend(header_part[1].splitlines())
        if _unavailable:
            for mid in _unavailable:
                desc_lines.append(f"   {rows.label(mid)}")
            desc_lines.append(f"  ── {unavailable_footer} ──")
        description = "\n".join(desc_lines) if desc_lines else None

        # Search haystacks keep pricing labels visible while adding aliases
        # for brand-less wire ids (e.g. Kimi Coding `k3` ↔ query "kimi").
        from hermes_cli.model_search import model_search_text

        model_search_labels = []
        for mid in ordered:
            label = rows.label(mid)
            haystack = model_search_text(mid)
            # model_search_text always starts with the wire id; only append when
            # aliases add tokens beyond the bare id already in the label.
            model_search_labels.append(
                label if haystack == mid else f"{label} {haystack}"
            )
        model_search_labels.append("Enter custom model name")
        model_search_labels.append("Skip (keep current)")

        idx = curses_radiolist(
            "Select default model:",
            choices,
            selected=default_idx,
            cancel_returns=-1,
            description=description,
            searchable=True,
            search_labels=model_search_labels,
        )
        if idx < 0:
            return None
        print()
        if idx < len(ordered):
            return _confirmed_selection(ordered[idx])
        elif idx == len(ordered):
            try:
                custom = line_input("Enter model name: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            return _confirmed_selection(custom) if custom else None
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    # Fallback: numbered list (ANSI colors for sale chrome)
    from hermes_cli.curses_ui import format_radio_item_ansi
    from hermes_cli.colors import Colors, color

    for line in menu_title.splitlines():
        if "★" in line:
            print(line.replace("★", color("★", Colors.YELLOW), 1))
        else:
            print(line)
    num_width = len(str(len(ordered) + 2))
    for i, mid in enumerate(ordered, 1):
        print(f"  {i:>{num_width}}. {format_radio_item_ansi(rows.segments(mid))}")
    n = len(ordered)
    print(f"  {n + 1:>{num_width}}. Enter custom model name")
    print(f"  {n + 2:>{num_width}}. Skip (keep current)")

    if _unavailable:
        unavailable_footer = unavailable_message.strip() or (
            f"Unavailable models (requires paid tier — upgrade at {_upgrade_url})"
        )
        print()
        print(f"  {_DIM}── {unavailable_footer} ──{_RESET}")
        for mid in _unavailable:
            print(f"  {'':>{num_width}}  {_DIM}{rows.label(mid)}{_RESET}")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: skip): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return _confirmed_selection(ordered[idx - 1])
            elif idx == n + 1:
                custom = line_input("Enter model name: ").strip()
                return _confirmed_selection(custom) if custom else None
            elif idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None


def _save_model_choice(model_id: str) -> None:
    """Save the selected model to config.yaml (single source of truth).

    The model is stored in config.yaml only — NOT in .env. This avoids conflicts in multi-agent
    setups where env vars would stomp each other.
    """
    from hermes_cli.config import save_config, load_config

    config = load_config()
    # Always use dict format so provider/base_url can be stored alongside
    if isinstance(config.get("model"), dict):
        config["model"]["default"] = model_id
    else:
        config["model"] = {"default": model_id}
    save_config(config)
