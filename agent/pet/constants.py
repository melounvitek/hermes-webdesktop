"""Pet sprite geometry + animation-state taxonomy.

Common petdex/Codex pet geometry. ``pet.json`` usually only carries
``id``/``displayName``/``description``/``spritesheetPath``; row taxonomy is
inferred from the atlas shape so Hermes renders both legacy 8-row sheets and
current 9-row Codex sheets.
"""

from __future__ import annotations

from enum import Enum

# Frame geometry (pixels). Current Codex/petdex sheets are 8 columns x 9 rows
# (1536x1872); older Hermes/petdex sheets were 9 columns x 8 rows (1728x1664).
# Renderers derive row taxonomy and real column count from the concrete sheet.
FRAME_W = 192
FRAME_H = 208

# Frames consumed per animation state (the petdex web app uses CSS ``steps(6)``).
# A sheet may physically contain more columns; we only step through the first N.
FRAMES_PER_STATE = 6

# Full-loop duration for one state, milliseconds (petdex default).
LOOP_MS = 1100

# Default on-screen scale relative to native frame size. ``display.pet.scale`` is
# the single master scalar: the desktop canvas multiplies native pixels by it and
# every terminal surface derives its half-block/kitty column width from it
# (:func:`cols_for_scale`), so one number shrinks all three interfaces together.
# petdex's own clients render at 0.7; we default smaller so the mascot stays a
# glanceable corner sprite. The half-block fallback can't shrink as far and
# clamps to ``UNICODE_MIN_COLS`` instead.
DEFAULT_SCALE = 0.33

# User-settable scale bounds (``/pet scale``, desktop slider). Floor keeps the
# pet clickable/visible; ceiling stops a fat-fingered value from filling the screen.
MIN_SCALE = 0.1
MAX_SCALE = 3.0


def clamp_scale(scale: float) -> float:
    """Clamp *scale* to ``[MIN_SCALE, MAX_SCALE]`` (the single validation point)."""
    return max(MIN_SCALE, min(MAX_SCALE, scale))


# Terminal cells one native frame spans at ``scale == 1.0``: a cell is ~8px wide,
# a frame 192px → 24 cells. Mirrors the kitty placement (``scaled_px // 8``) so at
# full scale every renderer agrees.
BASE_UNICODE_COLS = FRAME_W // 8

# Legibility floor for the half-block fallback. A half-block cell samples the
# sprite at only 1 horizontal + 2 vertical taps, so below this width a 192×208
# pet collapses into an unreadable blob regardless of scale (kitty/GUI draw true
# pixels and have no such floor). ``scale`` shrinks the unicode pet down TO this
# floor, not past it into noise.
UNICODE_MIN_COLS = 16


def cols_for_scale(scale: float) -> int:
    """Half-block width implied by *scale*, clamped to the legibility floor.

    Above the floor it tracks the kitty cell box (``scaled_px // 8``) so the two
    renderers converge at larger sizes.
    """
    return max(UNICODE_MIN_COLS, round(BASE_UNICODE_COLS * (scale or DEFAULT_SCALE)))


def resolve_cols(scale: float, unicode_cols: int = 0) -> int:
    """Resolve terminal width: explicit *unicode_cols* override, else from *scale*."""
    return int(unicode_cols) if unicode_cols and int(unicode_cols) > 0 else cols_for_scale(scale)


class PetState(str, Enum):
    """Animation state a pet can be shown in.

    Hermes' activity state names; not always identical to the source atlas row
    names (Codex pets use ``jumping``/``running`` rows while the UI keeps the
    shorter ``jump``/``run``).
    """

    IDLE = "idle"
    WAVE = "wave"
    RUN = "run"
    FAILED = "failed"
    REVIEW = "review"
    JUMP = "jump"
    WAITING = "waiting"


# Legacy Hermes/petdex row order (top -> bottom) for the older 8-row, 9-column atlas.
LEGACY_STATE_ROWS: list[str] = ["idle", "wave", "run", "failed", "review", "jump", "extra1", "extra2"]

# Current Petdex row order (top -> bottom) for 1536x1872 atlases (8 cols x 9 rows).
CODEX_STATE_ROWS: list[str] = ["idle", "running-right", "running-left", "waving", "jumping", "failed", "waiting", "running", "review"]

# Default/fallback for callers without a sheet: generated pets and the public
# Codex pet contract use the 9-row format.
STATE_ROWS: list[str] = CODEX_STATE_ROWS

# Canonical Hermes activity names -> accepted row-name aliases in descending
# preference, so internal names stay stable while matching Petdex's taxonomy.
_CODEX_NAMES = {"wave": "waving", "jump": "jumping", "run": "running"}
STATE_ALIASES: dict[str, tuple[str, ...]] = {
    s.value: (s.value, _CODEX_NAMES[s.value]) if s.value in _CODEX_NAMES else (s.value,) for s in PetState
}


def state_aliases_for(state: "PetState | str") -> tuple[str, ...]:
    """Return accepted row-name aliases for *state* (always non-empty)."""
    value = state.value if isinstance(state, PetState) else str(state)
    return STATE_ALIASES.get(value) or (value,)


def state_rows_for_grid(row_count: int | None) -> list[str]:
    """Return the row taxonomy for a spritesheet with *row_count* rows."""
    try:
        rows = int(row_count or 0)
    except (TypeError, ValueError):
        rows = 0
    return CODEX_STATE_ROWS if rows >= len(CODEX_STATE_ROWS) else LEGACY_STATE_ROWS


def state_row_index(state: "PetState | str", row_count: int | None = None) -> int:
    """Return the spritesheet row index for *state* (clamped, never raises)."""
    rows = state_rows_for_grid(row_count)
    return next((rows.index(name) for name in state_aliases_for(state) if name in rows), 0)  # 0 = idle row
