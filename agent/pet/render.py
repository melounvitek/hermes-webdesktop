"""Decode a pet spritesheet and encode frames for a terminal.

Shared by the base CLI (writes escape bytes to stdout) and the TUI (ships the
encoded bytes to Ink) so decode + capability detection + protocol encoding
exist once. Output modes, in fidelity order: ``kitty`` (kitty, Ghostty,
WezTerm), ``iterm`` (iTerm2, WezTerm), ``sixel`` (xterm -ti vt340, foot,
mlterm, …), ``unicode`` (24-bit half-blocks; any truecolor terminal). Missing
Pillow or spritesheet degrades to an empty string rather than raising.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from agent.pet.constants import (
    DEFAULT_SCALE,
    FRAME_H,
    FRAME_W,
    FRAMES_PER_STATE,
    PetState,
    state_row_index,
)

logger = logging.getLogger(__name__)

# Public render-mode names accepted by ``display.pet.render_mode``.
RENDER_MODES = ("auto", "kitty", "iterm", "sixel", "unicode", "off")


# ───────────────────────── terminal capability detection ─────────────────────────


def _is_wezterm() -> bool:
    return os.environ.get("TERM_PROGRAM", "").lower() == "wezterm" or bool(os.environ.get("WEZTERM_PANE"))


def detect_terminal_graphics() -> str:
    """Best-effort richest protocol from env vars only (never a DA1 query that could hang a pipe).

    Returns ``kitty`` / ``iterm`` / ``sixel`` / ``unicode``; unknown terminals get
    ``unicode``, which works anywhere with truecolor.
    """
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()

    # VS Code/Cursor set TERM_PROGRAM=vscode but don't scrub inherited
    # ITERM_SESSION_ID/KITTY_WINDOW_ID; trusting those emits a protocol xterm.js
    # can't show (blank frame). Inline images there are opt-in, so default to
    # half-blocks; users who enabled them can pin display.pet.render_mode.
    if term_program == "vscode":
        return "unicode"
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term or term_program == "ghostty":
        return "kitty"
    if _is_wezterm():  # speaks kitty and iterm; kitty has richer placement
        return "kitty"
    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return "iterm"
    if term_program == "mintty" or "foot" in term or "mlterm" in term or "sixel" in term:
        return "sixel"
    return "unicode"


def supports_kitty_placeholders() -> bool:
    """True when the terminal paints kitty Unicode placeholders (U+10EEEE).

    Narrower than ``detect_terminal_graphics() == "kitty"``: WezTerm accepts
    kitty APC transmits but lacks the placeholder grid (cells render as tofu).
    """
    return detect_terminal_graphics() == "kitty" and not _is_wezterm()


def resolve_mode(configured: str | None, *, stream=None) -> str:
    """Effective render mode from ``display.pet.render_mode`` + env; ``off`` when not a TTY."""
    mode = (configured or "auto").strip().lower()
    if mode not in RENDER_MODES:
        mode = "auto"
    if mode == "off":
        return "off"

    stream = stream or sys.stdout
    try:
        if not (hasattr(stream, "isatty") and stream.isatty()):
            return "off"
    except (ValueError, OSError):
        return "off"

    return detect_terminal_graphics() if mode == "auto" else mode


# ───────────────────────── frame decoding ─────────────────────────

# Max alpha at/below which a frame is blank padding. petdex sheets are
# left-packed, so a state with fewer real frames than FRAMES_PER_STATE has
# fully transparent trailing cells; animating into one flashes the pet blank.
_BLANK_ALPHA = 8


def _frame_is_blank(frame) -> bool:
    return frame.getchannel("A").getextrema()[1] <= _BLANK_ALPHA


@lru_cache(maxsize=16)
def _raw_frames(sheet_path: str, state_value: str, frame_w: int, frame_h: int, frames_per_state: int) -> tuple:
    """Cropped RGBA frames for one state row, stopping at the first blank column.

    Cached; returns ``()`` on any decode failure.
    """
    try:
        from PIL import Image

        sheet = Image.open(Path(sheet_path)).convert("RGBA")
        cols = max(1, sheet.width // frame_w)
        rows = max(1, sheet.height // frame_h)
        top = state_row_index(state_value, rows) * frame_h
        # Clamp to the sheet: some pets ship fewer rows than the taxonomy reserves.
        if top + frame_h > sheet.height:
            top = max(0, sheet.height - frame_h)

        frames = []
        for i in range(min(frames_per_state, cols)):
            left = i * frame_w
            frame = sheet.crop((left, top, left + frame_w, top + frame_h))
            if _frame_is_blank(frame):
                break
            frames.append(frame)
        return tuple(frames)
    except Exception as exc:  # noqa: BLE001 - cosmetic feature, never fatal
        logger.debug("pet frame decode failed (%s, %s): %s", sheet_path, state_value, exc)
        return ()


@lru_cache(maxsize=8)
def _frames_for(
    sheet_path: str,
    state_value: str,
    frame_w: int,
    frame_h: int,
    frames_per_state: int,
    scale_w: int,
    scale_h: int,
):
    """Scaled :func:`_raw_frames` (both cached, so animation-time requests are free)."""
    raw = _raw_frames(sheet_path, state_value, frame_w, frame_h, frames_per_state)
    if not raw or (scale_w, scale_h) == (frame_w, frame_h):
        return list(raw)
    from PIL import Image

    return [f.resize((scale_w, scale_h), Image.LANCZOS) for f in raw]


def state_frame_counts(
    sheet_path: str | Path,
    *,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    frames_per_state: int = FRAMES_PER_STATE,
) -> dict[str, int]:
    """Each driven :class:`PetState` → its real (padding-trimmed) frame count.

    The gateway ships this map to the desktop canvas, which steps its own loop.
    """
    return {
        state.value: len(_raw_frames(str(sheet_path), state.value, frame_w, frame_h, frames_per_state))
        for state in PetState
    }


# ───────────────────────── encoders ─────────────────────────


def _png_b64(frame) -> str:
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _crop_frames_to_alpha_union(frames):
    """Crop every frame to the union opaque bbox (a stable trim across the animation).

    kitty paints the whole transmitted rectangle, transparent margins included,
    so an untrimmed pet looks small and adrift inside its cell box.
    """
    boxes = []
    for frame in frames:
        try:
            bbox = frame.getchannel("A").getbbox()
        except Exception:  # noqa: BLE001 - cosmetic; fail open
            bbox = None
        if bbox:
            boxes.append(bbox)
    if not boxes:
        return frames
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    return [f.crop(union) for f in frames]


# Nominal terminal cell size in pixels. kitty fits an image to its cell
# rectangle preserving aspect, so a frame that isn't a whole cell multiple
# rounds up, clipping the bottom row ("clipped feet") and letterboxing a blank
# row. Snapping to an exact multiple avoids that (cf. ratatui-image #57).
_CELL_W = 8
_CELL_H = 16


def _snap_frames_to_cell_grid(frames):
    """Resize frames so width/height are exact multiples of the cell box (all frames share the union-cropped size)."""
    if not frames:
        return frames
    from PIL import Image

    w, h = frames[0].size
    target = (max(1, round(w / _CELL_W)) * _CELL_W, max(1, round(h / _CELL_H)) * _CELL_H)
    if (w, h) == target:
        return frames
    return [f.resize(target, Image.LANCZOS) for f in frames]


def _kitty_apc(ctrl: str, data: str) -> str:
    """kitty APC escape for *data*, chunked into ≤4096-byte ``m`` pieces."""
    pieces = [data[i : i + 4096] for i in range(0, len(data), 4096)] or [""]
    last = len(pieces) - 1
    return "".join(
        f"\x1b_G{ctrl + ',' if i == 0 else ''}m={0 if i == last else 1};{piece}\x1b\\" for i, piece in enumerate(pieces)
    )


def _encode_kitty(frame, *, cell_cols: int | None = None, cell_rows: int | None = None) -> str:
    """kitty transmit+display at the cursor; ``c``/``r`` pin the cell box so frames overwrite each other."""
    ctrl = "f=100,a=T,q=2"
    if cell_cols:
        ctrl += f",c={cell_cols}"
    if cell_rows:
        ctrl += f",r={cell_rows}"
    return _kitty_apc(ctrl, _png_b64(frame))


# ───────────────────────── kitty Unicode placeholders ─────────────────────────
# Ink owns the screen and measures every cell's width, so it can't host raw
# kitty image escapes. The placeholder protocol is the grid-safe path: transmit
# once as a virtual placement (U=1), then print ordinary-width placeholder cells
# (U+10EEEE + diacritics) whose foreground color encodes the image id; Ink counts
# them as width-1 text and the terminal paints the image underneath.
#   https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders

_KITTY_PLACEHOLDER = "\U0010eeee"

# Row/column diacritics, index → diacritic, verbatim from kitty's
# gen/rowcolumn-diacritics.txt. We only ever need the row index.
_ROWCOL_DIACRITICS: tuple[int, ...] = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A,
    0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365,
    0x0366, 0x0367, 0x0368, 0x0369, 0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F,
    0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592, 0x0593, 0x0594, 0x0595, 0x0597,
    0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1, 0x05A8, 0x05A9,
    0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6,
    0x06D7, 0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2,
    0x06E4, 0x06E7, 0x06E8, 0x06EB, 0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736,
    0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743, 0x0745, 0x0747, 0x0749, 0x074A,
    0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3, 0x0816, 0x0817,
    0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951,
    0x0953, 0x0954, 0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD,
    0x193A, 0x1A17, 0x1A75, 0x1A76, 0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C,
    0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73, 0x1CD0, 0x1CD1,
    0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4, 0x1DC5, 0x1DC6,
    0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF,
    0x1DE0, 0x1DE1, 0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1,
    0x20D4, 0x20D5, 0x20D6, 0x20D7, 0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0,
    0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2, 0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6,
    0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE, 0x2DEF, 0x2DF0,
    0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1,
    0xA8E0, 0xA8E1, 0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9,
    0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED, 0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2,
    0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1, 0xFE20, 0xFE21, 0xFE22, 0xFE23,
    0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186, 0x1D187, 0x1D188,
    0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244,
)


def kitty_image_id(slug: str) -> int:
    """Deterministic per-slug image id in ``[1, 0x7FFF]`` (non-zero; encoded in the placeholder fg color) so re-renders reuse the terminal-side image."""
    import zlib

    return (zlib.crc32(slug.encode("utf-8")) % 0x7FFE) + 1


def kitty_color_hex(image_id: int) -> str:
    """Hex foreground color (``#rrggbb``) that encodes *image_id* for kitty."""
    return "#%06x" % (image_id & 0xFFFFFF)


def kitty_placeholder_rows(cols: int, rows: int) -> list[str]:
    """Placeholder text grid: first cell carries the row diacritic, the rest auto-increment the column.

    The foreground color (image id) is applied by the caller / Ink, not here.
    """
    cols = max(1, cols)
    out: list[str] = []
    for r in range(max(1, rows)):
        idx = min(r, len(_ROWCOL_DIACRITICS) - 1)
        out.append(_KITTY_PLACEHOLDER + chr(_ROWCOL_DIACRITICS[idx]) + _KITTY_PLACEHOLDER * (cols - 1))
    return out


def _encode_kitty_virtual(frame, *, image_id: int, cols: int, rows: int) -> str:
    """Transmit a frame as a kitty virtual placement (``U=1``; ``q=2`` mutes replies that would corrupt Ink's output).

    Re-sending with the same ``i`` replaces the image, so static placeholder cells animate underneath.
    """
    return _kitty_apc(f"a=T,U=1,i={image_id},c={cols},r={rows},f=100,q=2", _png_b64(frame))


def _encode_iterm(frame, *, cell_cols: int | None = None, cell_rows: int | None = None) -> str:
    """iTerm2 inline image (OSC 1337 File)."""
    payload = _png_b64(frame)
    args = ["inline=1", f"size={len(payload)}", "preserveAspectRatio=1"]
    if cell_cols:
        args.append(f"width={cell_cols}")
    if cell_rows:
        args.append(f"height={cell_rows}")
    return f"\x1b]1337;File={';'.join(args)}:{payload}\x07"


def _encode_sixel(frame) -> str:
    """DEC sixel via a compact hand-rolled encoder (Pillow has no sixel writer).

    Quantizes to ≤255 adaptive colors; transparent pixels are skipped (render as background).
    """
    from PIL import Image

    pal = frame.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    palette = pal.getpalette() or []
    px = pal.load()
    alpha = frame.getchannel("A").load()
    w, h = pal.size

    out = ["\x1bP0;1;0q", '"1;1;%d;%d' % (w, h)]
    used = sorted({px[x, y] for y in range(h) for x in range(w)})
    for idx in used:  # color registers on a 0..100 scale
        r, g, b = (palette[idx * 3 + c] if idx * 3 + c < len(palette) else 0 for c in range(3))
        out.append("#%d;2;%d;%d;%d" % (idx, r * 100 // 255, g * 100 // 255, b * 100 // 255))

    for band in range(0, h, 6):
        for color_idx in used:
            line = ["#%d" % color_idx]
            run_char = None
            run_len = 0

            def flush():
                nonlocal run_char, run_len
                if run_char is None:
                    return
                line.append("!%d%s" % (run_len, run_char) if run_len > 3 else run_char * run_len)
                run_char, run_len = None, 0

            for x in range(w):
                bits = 0
                for bit in range(6):
                    y = band + bit
                    if y < h and alpha[x, y] > 32 and px[x, y] == color_idx:
                        bits |= 1 << bit
                ch = chr(63 + bits)
                if ch == run_char:
                    run_len += 1
                else:
                    flush()
                    run_char, run_len = ch, 1
            flush()
            out.append("".join(line) + "$")  # carriage return within band
        out.append("-")  # next band
    out.append("\x1b\\")
    return "".join(out)


_HALF_BLOCK = "▀"

# A single half-block cell: top pixel + bottom pixel as (r, g, b, a) tuples.
Cell = tuple[tuple[int, int, int, int], tuple[int, int, int, int]]


def _downscale_cells(frame, *, target_cols: int) -> list[list[Cell]]:
    """Downscale a frame to rows of half-block cells (one terminal row = two pixel rows).

    Framework-neutral representation shared by the ANSI encoder (CLI) and the
    structured ``cells`` API (Ink).
    """
    from PIL import Image

    target_cols = max(4, target_cols)
    aspect = frame.height / max(1, frame.width)
    target_rows = max(2, int(round(target_cols * aspect * 0.5)) * 2)
    px = frame.resize((target_cols, target_rows), Image.LANCZOS).convert("RGBA").load()
    return [
        [(px[x, y], px[x, y + 1] if y + 1 < target_rows else (0, 0, 0, 0)) for x in range(target_cols)]
        for y in range(0, target_rows, 2)
    ]


def _encode_unicode(frame, *, target_cols: int) -> str:
    """Truecolor ANSI half-blocks (one char = 2 vertical pixels)."""
    lines: list[str] = []
    for row in _downscale_cells(frame, target_cols=target_cols):
        cells = [
            "\x1b[0m " if ta < 32 and ba < 32 else f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m{_HALF_BLOCK}"
            for (tr, tg, tb, ta), (br, bg, bb, ba) in row
        ]
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)


# ───────────────────────── public renderer ─────────────────────────


class PetRenderer:
    """Holds a pet's spritesheet and yields encoded frames per (state, index).

    Construct once per pet, then call :meth:`frame` on an animation timer;
    decoded frames are cached so repeated calls are cheap.
    """

    def __init__(
        self,
        spritesheet: str | Path,
        *,
        mode: str = "unicode",
        scale: float = DEFAULT_SCALE,
        unicode_cols: int = 20,
        frame_w: int = FRAME_W,
        frame_h: int = FRAME_H,
        frames_per_state: int = FRAMES_PER_STATE,
    ) -> None:
        self.spritesheet = str(spritesheet)
        self.mode = mode if mode in RENDER_MODES else "unicode"
        self.scale = scale
        self.unicode_cols = unicode_cols
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frames_per_state = frames_per_state

    @property
    def available(self) -> bool:
        return self.mode != "off" and Path(self.spritesheet).is_file()

    def frame_count(self, state: PetState | str) -> int:
        return len(self._frames(state))

    def _frames(self, state: PetState | str):
        return _frames_for(
            self.spritesheet,
            state.value if isinstance(state, PetState) else str(state),
            self.frame_w,
            self.frame_h,
            self.frames_per_state,
            max(1, int(self.frame_w * self.scale)),
            max(1, int(self.frame_h * self.scale)),
        )

    def cells(self, state: PetState | str, index: int, *, cols: int | None = None) -> list[list[Cell]]:
        """One frame as a half-block cell grid for Ink's native color props; ``[]`` when unavailable."""
        frames = self._frames(state)
        if not frames:
            return []
        return _downscale_cells(frames[index % len(frames)], target_cols=cols or self.unicode_cols)

    def _cell_box(self, frame) -> tuple[int, int]:
        """Terminal cell box (~8×16 px per cell) for a scaled frame.

        kitty stretches the image to fill ``c``×``r`` cells, so this must track
        the scaled pixel size, not a native-aspect column count (that upscales small pets).
        """
        return max(1, frame.width // 8), max(1, frame.height // 16)

    def kitty_payload(self, state: PetState | str, *, image_id: int) -> dict | None:
        """kitty Unicode-placeholder payload ``{cols, rows, placeholder, frames}`` for one state.

        ``frames`` are transmit escapes (all reusing ``image_id``); ``placeholder``
        is the static text grid Ink paints. ``None`` when no frame is available.
        """
        frames = self._frames(state)
        if not frames:
            return None
        frames = _snap_frames_to_cell_grid(_crop_frames_to_alpha_union(frames))
        cols, rows = self._cell_box(frames[0])
        return {
            "cols": cols,
            "rows": rows,
            "placeholder": kitty_placeholder_rows(cols, rows),
            "frames": [_encode_kitty_virtual(f, image_id=image_id, cols=cols, rows=rows) for f in frames],
        }

    def frame(self, state: PetState | str, index: int) -> str:
        """Encoded escape string for one frame (``index`` taken modulo the frame count), or ``""``."""
        if self.mode == "off":
            return ""
        frames = self._frames(state)
        if not frames:
            return ""
        frame = frames[index % len(frames)]
        cell_cols, cell_rows = self._cell_box(frame)

        try:
            if self.mode == "kitty":
                return _encode_kitty(frame, cell_cols=cell_cols, cell_rows=cell_rows)
            if self.mode == "iterm":
                return _encode_iterm(frame, cell_cols=cell_cols, cell_rows=cell_rows)
            if self.mode == "sixel":
                return _encode_sixel(frame)
            return _encode_unicode(frame, target_cols=self.unicode_cols)
        except Exception as exc:  # noqa: BLE001 - degrade silently
            logger.debug("pet frame encode failed (mode=%s): %s", self.mode, exc)
            return ""


def build_renderer(
    spritesheet: str | Path,
    *,
    configured_mode: str | None = None,
    scale: float = DEFAULT_SCALE,
    unicode_cols: int = 20,
    stream=None,
) -> PetRenderer:
    """Resolve the mode from config+env, then construct a :class:`PetRenderer`."""
    mode = resolve_mode(configured_mode, stream=stream)
    return PetRenderer(spritesheet, mode=mode, scale=scale, unicode_cols=unicode_cols)
