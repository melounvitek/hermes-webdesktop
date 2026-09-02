"""Deterministic spritesheet assembly — generated row strips → Hermes atlas.

Image models draw a row of poses well but can't do exact grid geometry, so the
model never owns the layout: it emits one loose horizontal strip per state and
these ops slice it into centered transparent ``192x208`` cells packed into the
petdex/Codex atlas (8 columns x 9 rows, ``1536x1872``) that
:mod:`agent.pet.render` reads via :data:`agent.pet.constants.CODEX_STATE_ROWS`.
``running`` is the in-place *working* state; ``running-right``/``-left`` are the
directional walk cycles. Segmentation/fit/residue logic is adapted from
OpenAI's ``hatch-pet`` skill (openai/skills, Apache-2.0).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

from agent.pet.constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)

CELL_WIDTH = FRAME_W
CELL_HEIGHT = FRAME_H

# (state, row index, frame count). Order/row indices MUST match
# ``constants.CODEX_STATE_ROWS``; frame counts mirror the petdex ``hatch-pet``
# spec. Rows shorter than 8 leave their tail transparent (renderer trims it).
ROW_SPECS: list[tuple[str, int, int]] = [
    ("idle", 0, 6),
    ("running-right", 1, 8),
    ("running-left", 2, 8),
    ("waving", 3, 4),
    ("jumping", 4, 5),
    ("failed", 5, 8),
    ("waiting", 6, 6),
    ("running", 7, 6),
    ("review", 8, 6),
]

ROWS = len(ROW_SPECS)
COLUMNS = max(count for _, _, count in ROW_SPECS)
ATLAS_WIDTH = COLUMNS * CELL_WIDTH
ATLAS_HEIGHT = ROWS * CELL_HEIGHT

_ALPHA_FLOOR = 16  # alpha at/below which a pixel is "background"
_CELL_PAD = 10  # padding kept around a fitted sprite
# Small margin for the normalized pass so cells fill like real petdex pets
# (~5px from the edges); the width clamp, not the pad, prevents clipping.
_NORMALIZE_PAD = 14
# Side-lobe cutoff: adjacent-pose bleed shows as a small separated lobe; keep
# sizeable lobes so a legitimate wide pose isn't punished.
_SIDE_LOBE_RATIO = 0.18
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _median(values) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _blank(size=(CELL_WIDTH, CELL_HEIGHT)):
    from PIL import Image

    return Image.new("RGBA", size, (0, 0, 0, 0))


def _place(sprite, size: tuple[int, int], offset: tuple[int, int] = (0, 0)):
    """*sprite* alpha-composited at *offset* onto a new transparent canvas of *size*."""
    canvas = _blank(size)
    canvas.alpha_composite(sprite, offset)
    return canvas


def _clear_region(image, box: tuple[int, int, int, int]) -> None:
    """Make the ``(left, top, right, bottom)`` region of *image* fully transparent, in place."""
    image.paste(_blank((box[2] - box[0], box[3] - box[1])), (box[0], box[1]))


def _load_rgba(image):
    """Open a path (or take an image) as RGBA."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGBA")
    return image.convert("RGBA")


def _border_coords(w: int, h: int):
    """Every border pixel: top/bottom rows column-wise, then left/right columns."""
    for x in range(w):
        for y in (0, h - 1):
            yield x, y
    for y in range(h):
        for x in (0, w - 1):
            yield x, y


def _flood(w: int, h: int, visited: bytearray, seeds, accept) -> list[tuple[int, int]]:
    """4-connected BFS from *seeds* over pixels passing *accept*; returns the visited pixels."""
    queue = deque(seeds)
    pixels: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        pixels.append((x, y))
        for dx, dy in _NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                idx = ny * w + nx
                if not visited[idx]:
                    visited[idx] = 1
                    if accept(nx, ny):
                        queue.append((nx, ny))
    return pixels


def _border_flood(w: int, h: int, visited: bytearray, accept) -> list[tuple[int, int]]:
    """Flood from every border pixel passing *accept* (edge-connected region only)."""
    seeds: list[tuple[int, int]] = []
    for x, y in _border_coords(w, h):
        if not visited[y * w + x] and accept(x, y):
            visited[y * w + x] = 1
            seeds.append((x, y))
    return _flood(w, h, visited, seeds, accept)


def _unvisited_components(w: int, h: int, visited: bytearray, accept):
    """Yield each 4-connected component of not-yet-visited pixels passing *accept*, in scan order."""
    for start in range(w * h):
        if visited[start]:
            continue
        visited[start] = 1
        x, y = start % w, start // w
        if accept(x, y):
            yield _flood(w, h, visited, [(x, y)], accept)


# ───────────────────────── background removal ─────────────────────────


def _has_transparency(image) -> bool:
    """True if the strip already carries a real alpha background."""
    if image.getchannel("A").getextrema()[0] > _ALPHA_FLOOR:
        return False
    transparent = sum(image.getchannel("A").histogram()[: _ALPHA_FLOOR + 1])
    return transparent > image.width * image.height * 0.05


def _dominant_corner_color(image) -> tuple[int, int, int]:
    """Most common opaque color among the four corners."""
    from collections import Counter

    w, h = image.width, image.height
    px = image.load()
    counter: Counter = Counter()
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        r, g, b, a = px[x, y]
        if a > _ALPHA_FLOOR:
            counter[(r, g, b)] += 1
    return counter.most_common(1)[0][0] if counter else (0, 255, 0)


def _near_key_mask(image, key: tuple[int, int, int], tol: int = 48):
    """``L`` mask, 255 where a pixel is within *tol* per-channel of *key*.

    Tight on purpose: marks only near-pure backdrop so trapped chroma pockets
    seed the flood while chroma-tinted character pixels stay outside it.
    """
    from PIL import ImageChops

    r, g, b, _a = image.split()
    kr, kg, kb = key
    return ImageChops.darker(
        ImageChops.darker(
            r.point(lambda v: 255 if abs(v - kr) <= tol else 0),
            g.point(lambda v: 255 if abs(v - kg) <= tol else 0),
        ),
        b.point(lambda v: 255 if abs(v - kb) <= tol else 0),
    )


def _remove_masked(rgba, mask):
    """Clear the pixels *mask* (``L``, 255 = remove) selects, then erode alpha by 1px (3x3 min).

    The erosion drops the antialiased key/sprite blend ring (too far from the key to
    match); the sprite's own thick outline keeps the silhouette intact.
    """
    from PIL import Image, ImageFilter

    out = Image.composite(_blank(rgba.size), rgba, mask)
    out.putalpha(out.getchannel("A").filter(ImageFilter.MinFilter(3)))
    return out


def remove_background(image, *, chroma_key: tuple[int, int, int] | None = None, threshold: float = 90.0):
    """Return *image* (RGBA) with its flat background keyed out to transparent.

    Already-transparent strips are left alone (holes repaired). Otherwise key out
    *chroma_key* (or the dominant corner color) via a border flood-fill: a global
    color match punched holes wherever an interior highlight matched the backdrop.
    """
    from PIL import Image, ImageChops

    rgba = image.convert("RGBA")
    if _has_transparency(rgba):
        return _repair_internal_alpha_holes(rgba)

    key = chroma_key or _dominant_corner_color(rgba)
    w, h = rgba.width, rgba.height
    px = rgba.load()

    def _is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a > _ALPHA_FLOOR and math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2) <= threshold

    # Fast path for saturated chroma keys (our prompts use hot magenta): C-level
    # channel ops clear border backdrop and enclosed pockets alike, no Python flood.
    if max(key) - min(key) >= 120:
        opaque = rgba.getchannel("A").point(lambda a: 255 if a > _ALPHA_FLOOR else 0)
        return _remove_masked(rgba, ImageChops.darker(_near_key_mask(rgba, key), opaque))

    # Border-only flood on purpose: a desaturated near-white/gray key must never
    # seed from the character's interior (that is the hole-punching case).
    # Mark removals in a flat mask and composite once in C — per-pixel writes
    # were ~3M PixelAccess calls and stalled the gateway.
    remove = bytearray(w * h)
    for x, y in _border_flood(w, h, bytearray(w * h), _is_bg):
        remove[y * w + x] = 1
    return _remove_masked(rgba, Image.frombytes("L", (w, h), bytes(remove)).point(lambda v: 255 if v else 0))


def _repair_internal_alpha_holes(image):
    """Fill transparent islands fully enclosed by opaque sprite pixels.

    Some providers return "transparent" PNGs with swiss-cheese alpha inside the
    character. Edge-connected transparent components stay background; enclosed
    ones are filled with the average color of their opaque neighbours.
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    visited = bytearray(w * h)

    def _is_transparent(x: int, y: int) -> bool:
        return px[x, y][3] <= _ALPHA_FLOOR

    _border_flood(w, h, visited, _is_transparent)  # edge-connected transparency = background

    def _fill_color(hole: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        samples: list[tuple[int, int, int]] = []
        seen = set(hole)
        for x, y in hole:
            for dx, dy in _NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                    r, g, b, a = px[nx, ny]
                    if a > _ALPHA_FLOOR:
                        samples.append((r, g, b))
        if not samples:
            return (0, 0, 0, 255)
        r, g, b = (round(sum(c[i] for c in samples) / len(samples)) for i in range(3))
        return (r, g, b, 255)

    for hole in _unvisited_components(w, h, visited, _is_transparent):
        color = _fill_color(hole)
        for hx, hy in hole:
            px[hx, hy] = color
    return rgba


# ───────────────────────── frame extraction ─────────────────────────


def _fit_to_cell(image):
    """Crop to content, scale to fit a padded cell, and center on transparent."""
    from PIL import Image

    image = _drop_side_bleed(image)
    bbox = image.getbbox()
    if bbox is None:
        return _blank()

    sprite = image.crop(bbox)
    scale = min((CELL_WIDTH - _CELL_PAD) / sprite.width, (CELL_HEIGHT - _CELL_PAD) / sprite.height, 1.0)
    if scale != 1.0:
        # NEAREST: interpolating resamples blur the hard pixel-art edges.
        sprite = sprite.resize(
            (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
            Image.Resampling.NEAREST,
        )
    return _place(sprite, (CELL_WIDTH, CELL_HEIGHT), ((CELL_WIDTH - sprite.width) // 2, (CELL_HEIGHT - sprite.height) // 2))


def _drop_side_bleed(image):
    """Remove tiny separated left/right lobes (neighbour-pose slivers) before fitting.

    Component extraction may already have grouped a near sliver with the subject;
    a horizontal alpha projection still reveals it as a low-mass side lobe. Only
    those are dropped so wide poses and real limbs survive.
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    runs = _content_runs(_column_profile(rgba))
    if len(runs) < 2:
        return rgba
    keep_mass = max(m for _run, m in runs) * _SIDE_LOBE_RATIO
    keep = [run for run, m in runs if m >= keep_mass]
    if len(keep) == len(runs):
        return rgba

    rgba = rgba.copy()
    prev = 0
    for left, right in keep:
        if left > prev:
            _clear_region(rgba, (prev, 0, left, h))
        prev = right
    if prev < w:
        _clear_region(rgba, (prev, 0, w, h))
    return rgba


def _erase_long_axis_lines(image):
    """Remove thin slot-spanning guide/floor/divider lines.

    Models sometimes draw literal floors or panel dividers; they survive keying
    and connect otherwise clean poses. Only *thin* near-full-span rows/columns go.
    """
    rgba = image.convert("RGBA").copy()
    w, h = rgba.size
    alpha = rgba.getchannel("A")

    def _thin_groups(indices: list[int]) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        start = prev = -2
        for idx in [*indices, -2]:  # -2 sentinel flushes the last run
            if start >= 0 and idx == prev + 1:
                prev = idx
                continue
            if start >= 0 and prev - start + 1 <= 4:
                groups.append((start, prev + 1))
            start = prev = idx
        return groups

    wide_rows = [y for y in range(h) if sum(1 for x in range(w) if alpha.getpixel((x, y)) > _ALPHA_FLOOR) >= w * 0.85]
    tall_cols = [x for x in range(w) if sum(1 for y in range(h) if alpha.getpixel((x, y)) > _ALPHA_FLOOR) >= h * 0.85]

    for top, bottom in _thin_groups(wide_rows):
        _clear_region(rgba, (0, top, w, bottom))
    for left, right in _thin_groups(tall_cols):
        _clear_region(rgba, (left, 0, right, h))
    return rgba


def _component_boxes(image) -> list[tuple[tuple[int, int, int, int], int]]:
    """Connected opaque components as ``[(bbox, mass)]``."""
    rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        return []
    l0, t0, r0, b0 = bbox
    w, h = r0 - l0, b0 - t0
    alpha = rgba.getchannel("A").load()
    out: list[tuple[tuple[int, int, int, int], int]] = []

    def _opaque(x: int, y: int) -> bool:
        return alpha[l0 + x, t0 + y] > _ALPHA_FLOOR

    for pixels in _unvisited_components(w, h, bytearray(w * h), _opaque):
        xs = [x for x, _ in pixels]
        ys = [y for _, y in pixels]
        out.append(((l0 + min(xs), t0 + min(ys), l0 + max(xs) + 1, t0 + max(ys) + 1), len(pixels)))
    return out


def _isolate_slot_subject(image):
    """Keep the slot's real subject; drop detached effects/noise."""
    rgba = _erase_long_axis_lines(image)
    comps = _component_boxes(rgba)
    if not comps:
        return rgba

    main_box, main_mass = max(comps, key=lambda item: item[1])
    ml, mt, mr, mb = main_box
    mw = max(1, mr - ml)
    keep: list[tuple[int, int, int, int]] = []
    for box, mass in comps:
        if box == main_box:
            keep.append(box)
            continue
        left, _top, right, _bottom = box
        overlap = max(0, min(right, mr) - max(left, ml))
        center_x = (left + right) / 2
        near_main = (ml - mw * 0.25) <= center_x <= (mr + mw * 0.25)
        # Keep attached-looking accessories (halos); drop sparkles/tears/noise.
        if mass >= max(24, main_mass * 0.035) and (overlap >= mw * 0.3 or near_main):
            keep.append(box)

    out = _blank(rgba.size)
    for box in keep:
        out.alpha_composite(rgba.crop(box), (box[0], box[1]))
    return out


def _has_margin(size: tuple[int, int], box: tuple[int, int, int, int], fx: float, fy: float) -> bool:
    """True when *box* leaves empty room on all four edges of an image of *size* (≥4px, ≤12/16px)."""
    w, h = size
    left, top, right, bottom = box
    min_x = max(4, min(12, round(w * fx)))
    min_y = max(4, min(16, round(h * fy)))
    return left >= min_x and top >= min_y and w - right >= min_x and h - bottom >= min_y


def _group_component_rows(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group component boxes into visual rows, then sort left→right."""
    if not boxes:
        return []
    row_tol = max(12, _median(max(1, b[3] - b[1]) for b in boxes) * 0.55)
    rows: list[list[tuple[int, int, int, int]]] = []
    centers: list[float] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (box[1] + box[3]) / 2
        for i, center in enumerate(centers):
            if abs(cy - center) <= row_tol:
                rows[i].append(box)
                centers[i] = sum((b[1] + b[3]) / 2 for b in rows[i]) / len(rows[i])
                break
        else:
            rows.append([box])
            centers.append(cy)
    ordered = [row for _center, row in sorted(zip(centers, rows, strict=False), key=lambda item: item[0])]
    for row in ordered:
        row.sort(key=lambda b: (b[0] + b[2]) / 2)
    return ordered


def _merge_related_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge disconnected parts of one subject (capes, tails, props) on the same row.

    Merges when vertical spans overlap and the horizontal gap is tiny relative to
    the component size; never bridges the larger gaps between separate poses.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            al, at, ar, ab = a
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bl, bt, br, bb = boxes[j]
                v_overlap = max(0, min(ab, bb) - max(at, bt))
                min_h = max(1, min(ab - at, bb - bt))
                gap = max(0, max(al, bl) - min(ar, br))
                min_w = max(1, min(ar - al, br - bl))
                if v_overlap >= min_h * 0.45 and gap <= max(14, min_w * 0.22):
                    al, at, ar, ab = min(al, bl), min(at, bt), max(ar, br), max(ab, bb)
                    used[j] = True
                    changed = True
            merged.append((al, at, ar, ab))
        boxes = merged
    return boxes


def _component_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Extract frame subjects as connected non-background objects.

    Robust path for models that emit a 2D grid instead of one row: count real
    subject components, discard tiny effects, sort in reading order, return
    exactly *frame_count* frames (or ``None`` when the contract can't be met).
    """

    def attempt(source) -> list | None:
        subjects = _significant_subject_boxes(source, min_mass=64)
        if len(subjects) < frame_count:
            return None

        rows = _group_component_rows(subjects)
        ordered = [box for row in rows for box in row][:frame_count]
        if len(ordered) < frame_count:
            return None

        if require_padding and not all(_has_margin(source.size, box, 0.01, 0.015) for box in ordered):
            return None

        multirow = len(rows) > 1
        frames = []
        for left, top, right, bottom in ordered:
            pad_x = max(8, round((right - left) * 0.08))
            pad_y = max(8, round((bottom - top) * 0.08))
            if multirow:
                crop_box = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    min(source.width, right + pad_x),
                    min(source.height, bottom + pad_y),
                )
            elif frame_count == 1:
                crop_box = (0, 0, source.width, source.height)
            else:
                # Keep full height for true one-row strips so vertical motion survives.
                crop_box = (max(0, left - pad_x), 0, min(source.width, right + pad_x), source.height)
            # No second component filter here: capes/tails can be legitimate
            # disconnected lobes inside the chosen subject box.
            frames.append(
                _place(
                    source.crop((left, top, right, bottom)),
                    (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                    (left - crop_box[0], top - crop_box[1]),
                )
            )
        return frames

    return attempt(strip) or attempt(_erase_long_axis_lines(strip))


def _sever_expected_gutters(strip, frame_count: int):
    """Cut narrow transparent bands at expected frame boundaries before labeling.

    Shared shadows/smears/1px bridges connect neighbouring poses into one blob;
    losing a few boundary pixels beats exporting merged frames.
    """
    if frame_count <= 1:
        return strip

    out = strip.copy()
    alpha = out.getchannel("A")  # zero alpha only; RGB is left untouched
    slot = out.width / frame_count
    half = max(3, min(18, round(slot * 0.06)))
    for i in range(1, frame_count):
        x = round(i * slot)
        alpha.paste(0, (max(0, x - half), 0, min(out.width, x + half + 1), out.height))
    out.putalpha(alpha)
    return out


def _clean_slot(image):
    return _drop_side_bleed(_isolate_slot_subject(image))


def _slot_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Slice *strip* into *frame_count* uniform, independently cleaned columns.

    Equal-width columns keep every frame in one shared coordinate frame so
    :func:`normalize_cells` preserves the row's real motion without sliding.
    """
    w, h = strip.size
    frames = []
    for i in range(frame_count):
        slot = _clean_slot(strip.crop((round(i * w / frame_count), 0, round((i + 1) * w / frame_count), h)))
        bbox = slot.getbbox()
        if require_padding and (bbox is None or not _has_margin(slot.size, bbox, 0.025, 0.02)):
            return None
        frames.append(slot)
    return frames


def _content_runs(profile: list[int], *, threshold: int = 2) -> list[tuple[tuple[int, int], int]]:
    """``[((left, right), mass)]`` column spans whose alpha exceeds *threshold* (candidate frames)."""
    runs: list[tuple[tuple[int, int], int]] = []
    start: int | None = None
    for x, v in enumerate(list(profile) + [0]):
        if v > threshold:
            if start is None:
                start = x
        elif start is not None:
            runs.append(((start, x), sum(profile[start:x])))
            start = None
    return runs


def _frame_x_ranges(strip, frame_count: int) -> list[tuple[int, int]] | None:
    """Per-frame ``(left, right)`` column ranges from the row's empty gutters.

    spans == frames → one per frame; spans > frames → merge across the smallest
    gaps (a detached halo sits a tiny gap from its body, the inter-pose gutter is
    the big gap that survives); spans < frames → ``None`` (poses touching).
    Ranges span X only; the caller crops full height so tall ears/halos survive.
    """
    runs = _content_runs(_column_profile(strip))
    if not runs:
        return None

    floor = max(m for _run, m in runs) * 0.02
    groups = [[l, r] for (l, r), m in runs if m >= floor]
    if len(groups) < frame_count:
        return None

    while len(groups) > frame_count:
        gi = min(range(len(groups) - 1), key=lambda i: groups[i + 1][0] - groups[i][1])
        groups[gi][1] = groups[gi + 1][1]
        del groups[gi + 1]
    return [tuple(g) for g in groups]


def _significant_subject_boxes(image, *, min_mass: int = 32) -> list[tuple[int, int, int, int]]:
    """Merged boxes of components carrying meaningful mass (≥12% of the largest)."""
    comps = _component_boxes(image)
    if not comps:
        return []
    max_mass = max(mass for _box, mass in comps)
    return _merge_related_boxes([box for box, mass in comps if mass >= max(min_mass, max_mass * 0.12)])


def _is_multi_pose_outlier(width: int, height: int, med_w: int, med_h: int) -> bool:
    """A frame several times wider than the median but not proportionally taller."""
    return width > max(med_w * 3.0, med_w + 96) and height <= med_h * 1.6


def _validate_extracted_frames(frames: list, frame_count: int) -> None:
    """Reject rows where one "frame" is really multiple poses.

    A collapsed strip of tiny repeated poses would make normalization shrink the
    whole pet to postage-stamp size; catching it here lets hatch regenerate.
    """
    if len(frames) != frame_count:
        raise ValueError(f"expected {frame_count} frames, got {len(frames)}")

    boxes = []
    for i, frame in enumerate(frames):
        bbox = frame.getbbox()
        if bbox is None:
            raise ValueError(f"frame {i} is empty")
        if len(_significant_subject_boxes(frame)) >= 3:
            raise ValueError(f"frame {i} contains multiple separated subjects")
        boxes.append(bbox)

    if frame_count <= 1:
        return

    med_w = max(1, _median(b[2] - b[0] for b in boxes))
    med_h = max(1, _median(b[3] - b[1] for b in boxes))
    for i, (left, top, right, bottom) in enumerate(boxes):
        if _is_multi_pose_outlier(right - left, bottom - top, med_w, med_h):
            raise ValueError(f"frame {i} is a multi-pose width outlier")


def extract_strip_frames(
    strip,
    frame_count: int,
    *,
    chroma_key: tuple[int, int, int] | None = None,
    method: str = "auto",
    fit: bool = True,
) -> list:
    """Turn one generated row strip into *frame_count* frames.

    Keys out the background, then treats the frame count as source of truth:
    isolate padded subjects (component pass, then equal slots). When that fails,
    ``components`` raises and ``auto`` falls back to lenient salvage (gutters →
    severed gutters → raw slots). *fit* centers each frame into a 192x208 cell;
    hatching passes ``fit=False`` so :func:`normalize_cells` can register the
    whole pet with one shared scale + baseline.
    """
    strip = remove_background(_load_rgba(strip), chroma_key=chroma_key)

    frames = _component_crops(strip, frame_count, require_padding=True)
    if frames is None:
        frames = _slot_crops(strip, frame_count, require_padding=True)
    if frames is None:
        if method == "components":
            raise ValueError(f"could not segment {frame_count} padded sprites from strip")
        frames = _component_crops(strip, frame_count, require_padding=False)
    if frames is None:
        source = strip
        ranges = _frame_x_ranges(source, frame_count)
        if ranges is None:
            source = _sever_expected_gutters(strip, frame_count)
            ranges = _frame_x_ranges(source, frame_count)

        if ranges is None:
            frames = _slot_crops(source, frame_count, require_padding=False) or []
        else:
            h = source.height
            pad = max(2, min(16, round((source.width / max(1, frame_count)) * 0.04)))
            frames = [
                _clean_slot(source.crop((max(0, left - pad), 0, min(source.width, right + pad), h)))
                for left, right in ranges
            ]
    _validate_extracted_frames(frames, frame_count)
    return [_fit_to_cell(f) for f in frames] if fit else frames


def _column_profile(image) -> list[int]:
    """Per-column alpha mass — collapse to a 1px-tall strip (fast in C)."""
    from PIL import Image

    return list(image.getchannel("A").resize((image.width, 1), Image.BILINEAR).getdata())


def _best_shift(ref: list[int], prof: list[int], window: int) -> int:
    """Integer dx that best aligns *prof* onto *ref* (1-D cross-correlation).

    The body dominates the column profile, so the peak locks onto the body and a
    flipping arm/cape barely moves the match (~9px drift → ~1px).
    """
    n = len(ref)

    def score(d: int) -> int:
        return sum(ref[x] * prof[x - d] for x in range(max(0, d), min(n, n + d)))

    return max(range(-window, window + 1), key=score)  # ties → smallest dx, as before


def normalize_cells(frames_by_state: dict[str, list], *, pad: int = _NORMALIZE_PAD) -> dict[str, list]:
    """Register every frame into a 192x208 cell — the deterministic anti-jitter math.

    Per-frame crop→scale→center jitters (bbox shifts with a limb, per-frame scale
    pulses). Instead: cross-correlate each frame's column profile against the
    state's median profile to lock the body, union-crop through one shared state
    window, then scale every state by a single global factor keyed to its median
    pose height so the character is the same size in every row.
    """
    from PIL import Image

    out: dict[str, list] = {}
    prepared: dict[str, tuple[list, tuple[int, int, int, int], tuple[int, int]]] = {}
    target_w = CELL_WIDTH - pad
    target_h = CELL_HEIGHT - pad

    for state, frames in frames_by_state.items():
        rgba = [f.convert("RGBA") for f in frames]
        if not any(f.getbbox() for f in rgba):
            out[state] = [_blank() for _ in frames]
            continue

        # Pad every frame to a common canvas so column profiles are comparable.
        w0 = max(f.width for f in rgba)
        h0 = max(f.height for f in rgba)
        canvas = [f if f.size == (w0, h0) else _place(f, (w0, h0)) for f in rgba]

        profiles = [_column_profile(f) for f in canvas]
        ref = [_median(p[x] for p in profiles) for x in range(w0)]
        window = max(8, w0 // 5)
        margin = window
        aligned = [
            _place(f, (w0 + 2 * margin, h0), (margin + _best_shift(ref, prof, window), 0))
            for f, prof in zip(canvas, profiles)
        ]

        boxes = [b for b in (a.getbbox() for a in aligned) if b]
        prepared[state] = (
            aligned,
            (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)),
            (_median(b[2] - b[0] for b in boxes), _median(b[3] - b[1] for b in boxes)),
        )

    if not prepared:
        return out

    # K is the one global cap keeping the tallest/widest motion envelope (a
    # jump's lift) inside the cell; a still row's union ≈ pose so it fills fully.
    K = target_h
    for _aligned, (left, top, right, bottom), (_pose_w, pose_h) in prepared.values():
        uw, uh = right - left, bottom - top
        K = min(K, target_h * pose_h / max(1, uh), target_w * pose_h / max(1, uw))

    for state, (aligned, (left, top, right, bottom), (_pose_w, pose_h)) in prepared.items():
        uw, uh = right - left, bottom - top
        scale = K / max(1, pose_h)
        sw, sh = max(1, round(uw * scale)), max(1, round(uh * scale))
        px, py = round((CELL_WIDTH - sw) / 2), round((CELL_HEIGHT - pad // 2) - sh)

        cells = []
        for a in aligned:
            crop = a.crop((left, top, right, bottom))
            if crop.size != (sw, sh):
                crop = crop.resize((sw, sh), Image.Resampling.NEAREST)  # keep pixel edges crisp
            cells.append(_place(crop, (CELL_WIDTH, CELL_HEIGHT), (px, py)))
        out[state] = cells
    return out


# ───────────────────────── atlas composition ─────────────────────────


def single_frame(image, *, fit: bool = True):
    """One frame from a standalone image (idle fallback so a pet always renders).

    *fit* yields a finished cell; ``fit=False`` the raw keyed sprite for
    :func:`normalize_cells`.
    """
    keyed = remove_background(_load_rgba(image))
    return _fit_to_cell(keyed) if fit else _drop_side_bleed(keyed)


def _clear_transparent_rgb(image):
    """Zero the RGB of fully-transparent pixels (no colored-halo residue)."""
    from PIL import Image

    rgba = image.convert("RGBA")
    data = bytearray(rgba.tobytes())
    for i in range(0, len(data), 4):
        if data[i + 3] == 0:
            data[i] = data[i + 1] = data[i + 2] = 0
    return Image.frombytes("RGBA", rgba.size, bytes(data))


def mirror_frames(frames: list) -> list:
    """Flip each frame horizontally (per-frame, so order/timing is preserved).

    Derives ``running-left`` from ``running-right``; NOT a strip reverse.
    """
    from PIL import Image

    flip = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT
    return [frame.convert("RGBA").transpose(flip) for frame in frames]


def compose_atlas(frames_by_state: dict[str, list]):
    """Pack per-state frame lists into the atlas; short states leave trailing cells transparent."""
    atlas = _blank((ATLAS_WIDTH, ATLAS_HEIGHT))
    for state, row, count in ROW_SPECS:
        frames = frames_by_state.get(state) or []
        for col, frame in enumerate(frames[:count]):
            cell = frame.convert("RGBA")
            if cell.size != (CELL_WIDTH, CELL_HEIGHT):
                cell = _fit_to_cell(cell)
            atlas.alpha_composite(cell, (col * CELL_WIDTH, row * CELL_HEIGHT))
    return _clear_transparent_rgb(atlas)


def validate_atlas(atlas) -> dict:
    """Check geometry, per-cell occupancy, and transparency invariants.

    Returns ``{ok, width, height, errors, warnings, filled_states}``; errors are
    blockers, warnings soft (a whole state row blank).
    """
    atlas = _load_rgba(atlas)
    if atlas.size != (ATLAS_WIDTH, ATLAS_HEIGHT):
        errors: list[str] = [f"expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}, got {atlas.width}x{atlas.height}"]
        warnings: list[str] = []
        filled_states: list[str] = []
    else:
        errors, warnings, filled_states = _check_atlas_cells(atlas)
    return {
        "ok": not errors,
        "width": atlas.width,
        "height": atlas.height,
        "errors": errors,
        "warnings": warnings,
        "filled_states": filled_states,
    }


def _check_atlas_cells(atlas) -> tuple[list[str], list[str], list[str]]:
    """Occupancy/collapse/residue checks for a correctly-sized atlas → ``(errors, warnings, filled_states)``."""
    errors: list[str] = []
    warnings: list[str] = []
    filled_states: list[str] = []
    cell_boxes_by_state: dict[str, list[tuple[int, int, int, int]]] = {}
    for state, row, count in ROW_SPECS:
        row_pixels = 0
        boxes: list[tuple[int, int, int, int]] = []
        for col in range(count):
            left, top = col * CELL_WIDTH, row * CELL_HEIGHT
            cell = atlas.crop((left, top, left + CELL_WIDTH, top + CELL_HEIGHT))
            row_pixels += sum(cell.getchannel("A").histogram()[1:])
            bbox = cell.getbbox()
            if bbox is not None:
                boxes.append(bbox)
        if row_pixels > 0:
            filled_states.append(state)
            cell_boxes_by_state[state] = boxes
        else:
            warnings.append(f"state '{state}' has no frames")

    if not filled_states:
        errors.append("atlas is empty — no state produced any frames")

    # A valid pet must occupy the cell: one bad row can poison global
    # normalization and shrink every state while still passing "non-empty".
    all_boxes = [b for boxes in cell_boxes_by_state.values() for b in boxes]
    global_med_w = global_med_h = 0
    if all_boxes:
        global_med_w = _median(r - l for l, _t, r, _b in all_boxes)
        global_med_h = _median(b - t for _l, t, _r, b in all_boxes)
        if global_med_h < max(56, round(CELL_HEIGHT * 0.28)):
            errors.append(f"atlas sprites are too small after normalization (median frame height {global_med_h}px)")

    for state, boxes in cell_boxes_by_state.items():
        if len(boxes) <= 1:
            continue
        widths = [right - left for left, _top, right, _bottom in boxes]
        heights = [bottom - top for _left, top, _right, bottom in boxes]
        med_w, med_h = max(1, _median(widths)), max(1, _median(heights))
        if _is_multi_pose_outlier(max(widths), max(heights), med_w, med_h):
            errors.append(f"state '{state}' contains a multi-pose frame outlier")
        # Per-state collapse guard: one malformed row must not pass on the
        # strength of the healthy ones.
        if (global_med_w and global_med_h) and (
            med_w < max(32, round(global_med_w * 0.42)) or med_h < max(40, round(global_med_h * 0.50))
        ):
            errors.append(
                f"state '{state}' appears collapsed (median {med_w}x{med_h}px, global median {global_med_w}x{global_med_h}px)"
            )

    data = atlas.tobytes()
    residue = sum(1 for i in range(0, len(data), 4) if data[i + 3] == 0 and (data[i] or data[i + 1] or data[i + 2]))
    if residue:
        errors.append(f"{residue} transparent pixels retain RGB residue")
    return errors, warnings, filled_states
