"""Terminal renderer for the learning timeline (learned skills + memories).

The desktop starmap (``apps/desktop/src/app/starmap``) is a GPU constellation;
here the same data becomes a timeline bar chart (date rows, skill/memory bars
colored by dominant category, cumulative trajectory sparkline) plus per-slice
bucket metadata the TUI walks as a tree. Age gradient and memory ink are ported
from the desktop source. Grids are style runs ``[text, style, alpha, hex?]``:
consumers map style + brightness onto their palette; hex overrides the base
color (category heatmap). Pure, stdlib-only.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

# time-axis.ts LEAD_IN: the oldest node sits just off recency 0.
LEAD_IN = 0.06

# constants.ts AGE_GRADIENT — old quiet, recent bright.
AGE_OLD_INK = 0.42
AGE_MID_INK = 0.74
AGE_NEW_INK = 0.95
AGE_MID = 0.52

# Style keys consumers map to base colors (brightness = the run alpha).
STYLE_BG = "bg"
STYLE_SKILL = "skill"
STYLE_MEMORY = "memory"
STYLE_LABEL = "label"
STYLE_DIM = "dim"

# Legend glyphs mirror NODE_SHAPE (skill = circle, memory = diamond).
SKILL_GLYPH = "●"
MEMORY_GLYPH = "◆"
_LABEL_KEYS = tuple("123456789abc")

Run = list  # [text, style, alpha, hex?]
Row = list  # list[Run]
Grid = list  # list[Row]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _smoothstep(p: float) -> float:
    p = _clamp(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def _is_memory(node: dict[str, Any]) -> bool:
    return node.get("kind") == "memory"


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id", ""))


def _node_ts(node: dict[str, Any]) -> Optional[float]:
    try:
        return None if node.get("timestamp") is None else float(node["timestamp"])
    except (TypeError, ValueError):
        return None


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def recency_ink(rec: float) -> float:
    """Port of geometry.ts ``recencyInk`` — smoothstep age → ink alpha."""
    t = _clamp(rec, 0.0, 1.0)
    if t <= AGE_MID:
        return AGE_OLD_INK + (AGE_MID_INK - AGE_OLD_INK) * _smoothstep(t / AGE_MID)
    return AGE_MID_INK + (AGE_NEW_INK - AGE_MID_INK) * _smoothstep((t - AGE_MID) / (1 - AGE_MID))


def format_date(ts: Optional[float]) -> str:
    if not ts:
        return "unknown"
    try:
        dt = _utc(float(ts))
        return f"{dt.day} {dt.strftime('%b %Y')}"
    except (ValueError, OSError, OverflowError):
        return "unknown"


def compute_recency(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Port of time-axis.ts ``computeRecency`` (id → recency ratio, timed flag).

    Untimed graphs (no spread of timestamps) fall back to ordinal position so
    every node still gets a distinct recency.
    """
    known = [t for t in (_node_ts(n) for n in nodes) if t is not None]
    min_ts = min(known) if known else None
    max_ts = max(known) if known else None
    timed = min_ts is not None and max_ts is not None and max_ts > min_ts

    ordered = sorted(nodes, key=lambda n: (_node_ts(n) if _node_ts(n) is not None else math.inf, _node_id(n)))
    last = max(len(ordered) - 1, 1)
    ord_ratio = {_node_id(n): (i / last if len(ordered) > 1 else 0.0) for i, n in enumerate(ordered)}

    rec: dict[str, float] = {}
    for n in nodes:
        nid, ts = _node_id(n), _node_ts(n)
        ratio = (ts - min_ts) / (max_ts - min_ts) if timed and ts is not None else ord_ratio.get(nid, 0.0)
        rec[nid] = LEAD_IN + (1 - LEAD_IN) * _clamp(ratio, 0.0, 1.0)
    return {"rec": rec, "timed": timed, "minTs": min_ts, "maxTs": max_ts}


def _date_at(rec: dict[str, Any], reveal: float) -> Optional[float]:
    lo, hi = rec.get("minTs"), rec.get("maxTs")
    if not rec.get("timed") or lo is None or hi is None:
        return None
    return round(lo + _clamp(reveal, 0, 1) * (hi - lo))


# ── Color: ported from color.ts so memory ink + age fade match the desktop ──


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 255, 215, 0


def rgb_to_hex(c: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(_clamp(v, 0, 255)) for v in c))


def mix_rgb(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    p = _clamp(t, 0.0, 1.0)
    return tuple(round(a[i] + (b[i] - a[i]) * p) for i in range(3))  # type: ignore[return-value]


def _rgb_to_hsl(c: tuple) -> tuple[float, float, float]:
    r, g, b = (x / 255 for x in c)
    mx, mn = max(r, g, b), min(r, g, b)
    light = (mx + mn) / 2
    d = mx - mn
    if not d:
        return 0.0, 0.0, light
    s = d / (2 - mx - mn) if light > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60, s, light


# Hue sextant → (r, g, b) as a permutation of (c, x, 0).
_HUE_SEXTANTS = (
    lambda c, x: (c, x, 0.0),
    lambda c, x: (x, c, 0.0),
    lambda c, x: (0.0, c, x),
    lambda c, x: (0.0, x, c),
    lambda c, x: (x, 0.0, c),
    lambda c, x: (c, 0.0, x),
)


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    hue = ((h % 360) + 360) % 360
    c = (1 - abs(2 * light - 1)) * s
    x = c * (1 - abs(((hue / 60) % 2) - 1))
    m = light - c / 2
    return tuple(round((v + m) * 255) for v in _HUE_SEXTANTS[min(int(hue // 60), 5)](c, x))  # type: ignore[return-value]


def _complementary_ink(c: tuple) -> tuple[int, int, int]:
    h, s, light = _rgb_to_hsl(c)
    return _hsl_to_rgb(h + 165, max(s, 0.5), _clamp(light, 0.5, 0.7))


def derive_palette(primary_hex: str, *, dark: bool = True) -> dict[str, str]:
    """Port of color.ts ``computePalette`` (the bits a terminal needs)."""
    primary = hex_to_rgb(primary_hex)
    base = (255, 255, 255) if dark else (0, 0, 0)
    bg = (8, 8, 12) if dark else (250, 250, 250)
    return {
        "primary": primary_hex,
        # Memories are drillable → primary "clickable" ink; skills are dead-ends → muted complement.
        "memory": rgb_to_hex(mix_rgb(primary, base, 0.12 if dark else 0.18)),
        "skill": rgb_to_hex(mix_rgb(_complementary_ink(primary), bg, 0.45)),
        "label": rgb_to_hex(mix_rgb(base, bg, 0.35)),
        "dim": rgb_to_hex(mix_rgb(base, bg, 0.7)),
        "bg": rgb_to_hex(bg),
    }


def _node_score(node: dict[str, Any], rec: float) -> float:
    """Pick which visible objects deserve map markers + label rows."""
    if _is_memory(node):
        return 3.5 + rec
    use = float(node.get("useCount", 0) or 0)
    return rec * 2 + math.sqrt(max(0.0, use)) + (2.0 if node.get("pinned") else 0.0)


def _node_raw_label(node: dict[str, Any]) -> str:
    return str(node.get("label") or node.get("id") or "unknown").strip()


def _node_card(node: dict[str, Any]) -> dict[str, Any]:
    """Shared glyph/label/meta/style fields for label rows and bucket trees."""
    mem = _is_memory(node)
    text = _node_raw_label(node)
    date = format_date(_node_ts(node))
    if mem:
        meta = f"{'profile memory' if node.get('memorySource') == 'profile' else 'memory'} · {date}"
    else:
        count = int(node.get("useCount", 0) or 0)
        bits = [str(node.get("category") or "skill"), date] + ([f"x{count}"] if count else []) + (["pinned"] if node.get("pinned") else [])
        meta = " · ".join(bits)
    return {
        "glyph": MEMORY_GLYPH if mem else SKILL_GLYPH,
        "label": text if len(text) <= 26 else text[:23].rstrip() + "…",
        "meta": meta,
        "style": STYLE_MEMORY if mem else STYLE_SKILL,
    }


def _skill_category_counts(nodes: Iterable[dict[str, Any]]) -> Counter:
    return Counter(str(node.get("category") or "skill") for node in nodes if not _is_memory(node))


# ── Timeline chart frame ─────────────────────────────────────────────────────


class _ChartBucket:
    __slots__ = ("label", "ts", "nodes", "rec")

    def __init__(self, label: str, ts: float):
        self.label, self.ts, self.rec = label, ts, 1.0
        self.nodes: list[dict[str, Any]] = []

    @property
    def memories(self) -> int:
        return sum(1 for n in self.nodes if _is_memory(n))

    @property
    def skills(self) -> int:
        return len(self.nodes) - self.memories

    @property
    def total(self) -> int:
        return len(self.nodes)

    def add(self, node: dict[str, Any]) -> None:
        self.nodes.append(node)

    def category(self) -> Optional[str]:
        counts = _skill_category_counts(self.nodes)
        return max(counts, key=lambda k: counts[k]) if counts else None


# granularity → (period key, row label) from a UTC datetime.
_PERIODS: dict[str, tuple] = {
    "day": (lambda dt: (dt.year, dt.month, dt.day), lambda dt: f"{dt.day} {dt.strftime('%b')}"),
    "month": (lambda dt: (dt.year, dt.month), lambda dt: dt.strftime("%b %Y")),
    "year": (lambda dt: (dt.year,), lambda dt: dt.strftime("%Y")),
}


def _period(ts: float, granularity: str) -> tuple[tuple[int, ...], str]:
    key_fn, label_fn = _PERIODS.get(granularity, _PERIODS["year"])
    dt = _utc(ts)
    return key_fn(dt), label_fn(dt)


def _fill_even_bins(buckets: list[_ChartBucket], nodes: Iterable[dict[str, Any]], rec: dict[str, Any]) -> None:
    """Drop each node into the bin its recency ratio maps to (order preserved)."""
    n_bins = len(buckets)
    for node in nodes:
        r = rec["rec"].get(_node_id(node), 0.0)
        buckets[int(_clamp(math.floor(r * n_bins), 0, n_bins - 1))].add(node)


def _build_chart_buckets(nodes: list[dict[str, Any]], rec: dict[str, Any], max_rows: int) -> list[_ChartBucket]:
    """Timeline rows: finest date granularity that fits, oldest → newest."""
    if not nodes:
        return []
    if not rec["timed"]:
        ordered = sorted(nodes, key=lambda n: rec["rec"].get(_node_id(n), 0.0))
        buckets = [_ChartBucket(f"#{i + 1}", float(i)) for i in range(min(max_rows, len(ordered)))]
        _fill_even_bins(buckets, ordered, rec)
        return buckets

    chosen: Optional[list[_ChartBucket]] = None
    for granularity in ("day", "month", "year"):
        groups: dict[tuple[int, ...], _ChartBucket] = {}
        for node in nodes:
            ts = _node_ts(node)
            if ts is not None:
                key, label = _period(ts, granularity)
                groups.setdefault(key, _ChartBucket(label, ts)).add(node)
        # For short spans, keep the useful day-by-day graph even when the caller
        # asked for fewer rows; scrollback beats collapsing a month into one bar.
        if len(groups) <= max_rows or (granularity == "day" and len(groups) <= 32):
            chosen = [groups[key] for key in sorted(groups)]
            break

    min_ts, max_ts = rec.get("minTs"), rec.get("maxTs")
    if chosen is None:
        # Even yearly buckets overflow → fall back to even time bins.
        n_bins = max(1, max_rows)
        chosen = [
            _ChartBucket(format_date(ts), ts)
            for ts in (min_ts + (i / max(1, n_bins - 1)) * (max_ts - min_ts) if min_ts and max_ts else float(i) for i in range(n_bins))
        ]
        _fill_even_bins(chosen, nodes, rec)

    span = (max_ts - min_ts) if min_ts is not None and max_ts is not None and max_ts > min_ts else 0
    for bucket in chosen:
        bucket.rec = LEAD_IN + (1 - LEAD_IN) * ((bucket.ts - min_ts) / span) if span else 1.0
    return chosen


def _bucket_rows(buckets: list[_ChartBucket], payload: dict[str, Any]) -> list[dict[str, Any]]:
    cmap = category_color_map(payload)
    memory_lookup = {
        f"memory:{card.get('source')}:{idx}": card
        for idx, card in enumerate(payload.get("memory", []) or [])
        if isinstance(card, dict)
    }
    rows: list[dict[str, Any]] = []
    for idx, bucket in enumerate(buckets):
        cat = bucket.category()
        nodes = []
        # Chronological within the slice so the TUI tree reads oldest → newest.
        for node in sorted(bucket.nodes, key=lambda n: _node_ts(n) or bucket.ts):
            card = _node_card(node)
            memory = memory_lookup.get(_node_id(node))
            nodes.append({
                "id": _node_id(node), "glyph": card["glyph"], "label": card["label"],
                "fullLabel": _node_raw_label(node), "meta": card["meta"],
                "body": str(memory.get("body", "")) if memory else "", "style": card["style"],
            })
        rows.append({
            "index": idx, "label": bucket.label, "date": format_date(bucket.ts),
            "skills": bucket.skills, "memories": bucket.memories, "total": bucket.total,
            "category": cat, "color": cmap.get(cat) if cat else None,
            "nodes": nodes,
        })
    return rows


def _category_counts(payload: dict[str, Any]) -> list[tuple[str, int]]:
    clusters = [
        (str(c.get("category")), int(c.get("count", 0)))
        for c in payload.get("clusters", []) or []
        if c.get("category") and c.get("category") != "memory"
    ]
    if clusters:
        return clusters
    counts = _skill_category_counts(payload.get("nodes", []))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def category_color_map(payload: dict[str, Any]) -> dict[str, str]:
    """Deterministic, evenly-spread hue per skill category (theme-independent).
    Golden-angle spacing so adjacent categories never collide in color."""
    return {cat: rgb_to_hex(_hsl_to_rgb((i * 137.508) % 360, 0.55, 0.62)) for i, (cat, _c) in enumerate(_category_counts(payload))}


def category_legend(payload: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    cmap = category_color_map(payload)
    cats = _category_counts(payload)
    out = [{"glyph": "●", "color": cmap.get(cat, ""), "label": f"{cat} ({count})"} for cat, count in cats[:limit]]
    if len(cats) > limit:
        out.append({"glyph": "·", "color": "", "label": f"+{len(cats) - limit}"})
    return out


def _trajectory_row(buckets: list[_ChartBucket], width: int, reveal: float) -> Row:
    """Cumulative learning curve as a compact star-path sparkline."""
    if not buckets:
        return []
    total = sum(b.total for b in buckets) or 1
    visible = int(_clamp(math.ceil(reveal * len(buckets)), 0, len(buckets)))
    cells = [" "] * width
    acc = last = 0
    for b in buckets[:visible]:
        acc += b.total
        p = round((acc / total) * (width - 1))
        for x in range(min(last, p), max(last, p) + 1):
            if 0 <= x < width and cells[x] == " ":
                cells[x] = "·"
        if 0 <= p < width:
            cells[p] = "✦"
        last = p
    return [["trajectory ", STYLE_LABEL, 0.55], ["".join(cells), STYLE_SKILL, 0.48]]


def _bar_lengths(bucket: _ChartBucket, max_total: int, bar_w: int) -> tuple[int, int, int]:
    """(bar, skill, memory) cell counts; a present kind never rounds to zero."""
    bar_len = max(1, round((bucket.total / max_total) * bar_w)) if bucket.total else 0
    skill_len = round((bucket.skills / bucket.total) * bar_len) if bucket.total else 0
    if bucket.skills and skill_len == 0:
        skill_len = 1
    memory_len = bar_len - skill_len
    if bucket.memories and memory_len == 0 and bar_len > 1:
        memory_len = 1
        skill_len = bar_len - 1
    return bar_len, skill_len, memory_len


def render_graph(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, reveal: float = 1.0) -> dict[str, Any]:
    """Render one timeline frame at ``reveal`` (0→1): date rows with proportional
    skill/memory bars colored by dominant category, numbered markers tied to
    label rows, and a cumulative trajectory sparkline underneath."""
    reveal, cols, rows = _clamp(reveal, 0.0, 1.0), max(44, cols), max(14, rows)
    nodes = list(payload.get("nodes", []))
    if not nodes:
        placeholder = [["no learning yet — keep using Hermes and it maps out here", STYLE_DIM, 0.7]]
        return {"grid": [placeholder], "date": "", "reveal": reveal, "visible": 0}

    rec = compute_recency(nodes)
    cmap = category_color_map(payload)
    buckets = _build_chart_buckets(nodes, rec, max_rows=max(4, rows - 3))
    n_buckets = len(buckets)
    visible_bucket_count = int(_clamp(math.ceil(reveal * n_buckets), 0, n_buckets))
    max_total = max((b.total for b in buckets), default=1) or 1
    label_w = min(9, max(len(b.label) for b in buckets))
    bar_w = max(14, cols - label_w - 16)

    grid: Grid = []
    labels: list[dict[str, Any]] = []
    visible = 0
    for i, bucket in enumerate(buckets):
        if i >= visible_bucket_count:
            grid.append([])
            continue
        visible += bucket.total
        ink = recency_ink(bucket.rec)
        bar_len, skill_len, memory_len = _bar_lengths(bucket, max_total, bar_w)

        marker = ""
        if bucket.nodes and len(labels) < 6:
            node = max(bucket.nodes, key=lambda n: _node_score(n, _node_ts(n) or bucket.ts))
            marker = _LABEL_KEYS[len(labels)]
            labels.append({"key": marker, **_node_card(node), "alpha": round(ink, 3)})

        cat = bucket.category()
        cat_hex = cmap.get(cat) if cat else None

        row: Row = [[f"{bucket.label:>{label_w}} ", STYLE_LABEL, ink], ["│ ", STYLE_DIM, 0.55]]
        if marker:
            row.append([marker, STYLE_LABEL, 0.95])
        elif bucket.total:
            head_hex = cat_hex if bucket.skills else None
            row.append(["✦" if bucket.skills else "◆", STYLE_SKILL if bucket.skills else STYLE_MEMORY, ink, head_hex])
        if skill_len:
            # Bar colored by the day's dominant category — a learning heatmap.
            row.append(["━" * skill_len, STYLE_SKILL, ink, cat_hex])
        if memory_len:
            mem_trail = "◆" if memory_len == 1 else "◆" + ("━" * (memory_len - 2)) + "◆"
            row.append([mem_trail, STYLE_MEMORY, max(0.65, ink)])
        if bar_len < bar_w:
            # Empty space keeps counts aligned; starmap texture lives in the trajectory row.
            row.append([" " * (bar_w - bar_len), STYLE_BG, 1.0])
        row.append(["  ", STYLE_BG, 1.0])
        row.append([str(bucket.skills), STYLE_SKILL, max(0.72, ink)])
        if bucket.memories:
            row.append(["+", STYLE_DIM, 0.6])
            row.append([str(bucket.memories), STYLE_MEMORY, max(0.72, ink)])
        if i == visible_bucket_count - 1:
            row.append(["  ◀ now", STYLE_LABEL, 0.9])
        elif bucket.total == max_total and max_total > 1:
            row.append(["  ☄ peak", STYLE_LABEL, 0.75])
        grid.append(row)

    grid.append([[(" " * (label_w + 2)), STYLE_BG, 1.0], *_trajectory_row(buckets, max(12, cols - label_w - 13), reveal)])
    return {"grid": grid, "date": format_date(_date_at(rec, reveal)), "reveal": reveal, "visible": visible, "labels": labels}


# ── Trimmings ──────────────────────────────────────────────────────────────


def build_legend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes", [])
    memories = sum(1 for n in nodes if _is_memory(n))
    return [
        {"glyph": SKILL_GLYPH, "style": STYLE_SKILL, "label": f"skills ({len(nodes) - memories})"},
        {"glyph": MEMORY_GLYPH, "style": STYLE_MEMORY, "label": f"memories ({memories})"},
    ]


def axis_labels(payload: dict[str, Any]) -> dict[str, str]:
    rec = compute_recency(list(payload.get("nodes", [])))
    if not rec["timed"]:
        return {"start": "oldest", "end": "now"}
    return {"start": format_date(rec["minTs"]), "end": format_date(rec["maxTs"])}


def _peak_day(payload: dict[str, Any]) -> Optional[str]:
    counts: Counter = Counter()
    labels: dict[tuple[int, ...], str] = {}
    for node in payload.get("nodes", []):
        ts = _node_ts(node)
        if ts is not None:
            key, labels[key] = _period(ts, "day")
            counts[key] += 1
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    return f"busiest day {labels[best]} · {counts[best]} learned"


def build_summary(payload: dict[str, Any]) -> list[str]:
    stats = payload.get("stats", {}) or {}
    learned = stats.get("learned_skills", stats.get("nodes", 0))
    lines = [f"{learned} learned skills · {stats.get('memory_nodes', 0)} memories · {stats.get('related_edges', 0)} skill links"]
    extra = [f"{stats['memory_skill_edges']} memory↔skill links"] if stats.get("memory_skill_edges") else []
    extra += filter(None, [_peak_day(payload)])
    if extra:
        lines.append(" · ".join(extra))
    return lines


def render_frames(payload: dict[str, Any], *, cols: int = 80, rows: int = 16, frames: int = 48) -> dict[str, Any]:
    """Pre-render a full play-through (reveal 0→1) plus static legend/summary."""
    frames = max(2, min(frames, 240))
    nodes = list(payload.get("nodes", []))
    # Mirror render_graph's bucketing so the interactive row list lines up with what the user sees.
    buckets = _build_chart_buckets(nodes, compute_recency(nodes), max_rows=max(4, rows - 3)) if nodes else []
    out_frames = []
    for i in range(frames):
        frame = render_graph(payload, cols=cols, rows=rows, reveal=i / (frames - 1))
        out_frames.append({k: frame[k] for k in ("reveal", "date", "visible", "grid")} | {"labels": frame.get("labels", [])})
    return {
        "frames": out_frames,
        "legend": build_legend(payload),
        "categories": category_legend(payload),
        "buckets": _bucket_rows(buckets, payload),
        "summary": build_summary(payload),
        "axis": axis_labels(payload),
        "count": len(payload.get("nodes", [])),
        "cols": cols,
        "rows": rows,
    }
