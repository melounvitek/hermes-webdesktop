"""Curated starter catalog for the managed local runtime.

Small and honest: every entry carries the estimator inputs (measured on real GGUFs) so the picker
can price a model BEFORE the user downloads gigabytes. Once a file is on disk, profile_from_gguf()
is the authority and the catalog numbers are only used for the download decision.

Validation lifecycle: builds proven end-to-end on real hardware are marked validated. Day-0 entries
ship before that proof (they simply lack the validated flag) — ensure_model_ready's touch generation
still gates every first load at runtime.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from hermes_cli.local_runtime.context_policy import (
    FLOOR,
    RUNTIME_OVERHEAD_BYTES,
    TARGET_WINDOW,
    ub_logits_bytes,
)
from hermes_cli.local_runtime.estimator import (
    HardwareBudget,
    LayerKind,
    ModelProfile,
    ctx_bytes,
)

logger = logging.getLogger(__name__)

_PART_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$")


@dataclass(frozen=True)
class AssetFile:
    """One downloadable file: repo-relative path and exact bytes (the size feeds the estimator and the
    download progress bar; there is no download-time integrity check by design — a corrupt file
    surfaces as a llama.cpp load error). ``local`` overrides the on-disk name (repos reuse generic
    names like mmproj-BF16.gguf across models). Non-model extras live under the models dir's assets/
    subdirectory so the router never lists them.
    """

    path: str                   # repo-relative (may include a subdir)
    size_bytes: int
    local: str | None = None

    @property
    def local_name(self) -> str:
        return self.local or PurePosixPath(self.path).name


@dataclass(frozen=True)
class QuantVariant:
    """One downloadable build of a model. Split GGUFs list every part in
    files; the model loads from the first part."""

    quant: str                  # e.g. "UD-Q4_K_M"
    files: tuple                # AssetFile, first = the load target
    validated: bool = False     # proven end-to-end on real hardware

    @property
    def model_id(self) -> str:
        stem = PurePosixPath(self.files[0].path).name.removesuffix(".gguf")
        return _PART_SUFFIX.sub("", stem)

    @property
    def size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def weights_bytes(self) -> int:
        """Pre-download weights estimate: GGUF bytes ≈ tensor bytes + a small header (<2%) — a safe,
        slightly conservative stand-in until profile_from_gguf reads the real table.
        """
        return self.size_bytes


@dataclass(frozen=True)
class CatalogEntry:
    id: str                     # stable family id (variant-independent)
    display_name: str
    description: str            # one line, plain language
    repo: str                   # HF repo
    variants: tuple             # QuantVariant (exactly one, Q4-class)
    # Estimator inputs (measured or config-derived; quant changes weights,
    # never KV). Entries with gated upstream configs carry a conservative
    # same-family prior — the GGUF header is the authority after download.
    n_ctx_train: int
    full_layers: int
    recurrent_layers: int
    per_layer_f16: int          # KV bytes/token per full-attention layer
    swa_layers: int = 0
    swa_window: int = 0
    moe: bool = False
    mtp: bool = False           # ships MTP heads (spec decode when loaded)
    # Speculative draft depth for MTP models. Per-model and measured:
    # deeper drafting pays only while draft acceptance holds, and the
    # break-even depth differs by model.
    mtp_draft_depth: int = 3
    # Vocab size prices the GPU logits buffers (ubatch x vocab x fp32,
    # doubled under MTP backend sampling) — a multi-GiB term at large
    # vocab sizes that a weights-only fit would miss.
    n_vocab: int = 0
    mmproj: "AssetFile | None" = None    # vision projector, downloads with model
    draft: "AssetFile | None" = None     # spec-decode draft model (e.g. DSpark)
    sampling: dict = field(default_factory=dict)  # INI long-form launch defaults
    # Oldest llama.cpp release tag that can load this model (day-0
    # architectures need the release where their support landed). Empty
    # means any installed engine. The pane gates download/activate on it.
    min_engine: str = ""
    # Editorial quality ordering (higher = smarter), authored once,
    # globally, at catalog-authoring time — Artificial Analysis-informed
    # where they cover the model (scripts/aa_quality_sync.py proposes,
    # the commit decides), editorial elsewhere. Ranks entries for the
    # per-machine recommendation; never displayed as a score (it grades
    # the full-precision model, not our Q4 build).
    quality: int = 0
    # Fraction of the build's bytes read per decoded token: 1.0 for dense
    # models (every weight streams every token), the active slice for MoE
    # (attention + shared + routed experts over total). With memory
    # bandwidth this predicts decode speed — the physics half of the
    # recommendation.
    decode_fraction: float = 1.0

    def profile(self, variant: QuantVariant) -> ModelProfile:
        layers = ([(LayerKind.FULL, self.per_layer_f16)] * self.full_layers
                  + [(LayerKind.SWA, self.per_layer_f16)] * self.swa_layers
                  + [(LayerKind.RECURRENT, 0)] * self.recurrent_layers)
        return ModelProfile(
            name=variant.model_id, weights_bytes=variant.weights_bytes,
            embd_table_bytes=0, n_ctx_train=self.n_ctx_train,
            layers=layers, swa_window=self.swa_window, moe=self.moe,
            n_vocab=self.n_vocab,
            kv_scale=1.2 if self.mtp else 1.0)

    def download_files(self, variant: QuantVariant) -> tuple:
        """Everything a download job fetches for this variant, in order."""
        extras = tuple(a for a in (self.mmproj, self.draft) if a is not None)
        return tuple(variant.files) + extras

    def download_bytes(self, variant: QuantVariant) -> int:
        return sum(f.size_bytes for f in self.download_files(variant))


@dataclass(frozen=True)
class VariantChoice:
    """Selection result: which build this machine should download and why.
    reason_key is a UI-copy discriminator, not display text."""

    variant: QuantVariant
    zero_spill: bool
    reason_key: str  # "best-large-window" | "best-fits" | "smallest-fits-spilled"


def select_variant(entry: CatalogEntry, budget: HardwareBudget) -> VariantChoice | None:
    """Fit the entry's one build (Q4-class) to this machine.

    Every entry ships exactly one variant (see the module docstring for why there is no quant
    ladder); headroom buys a bigger window, never a bigger quant. The fit shapes:

    - "best-large-window": zero-spills at TARGET_WINDOW - "best-fits": zero-spills at the 64K floor
    - "smallest-fits-spilled": weights spill to host RAM, priced honestly - None: even spilled,
    physics refuses (the machine can't run it)
    """
    overhead = (RUNTIME_OVERHEAD_BYTES
                + (entry.mmproj.size_bytes if entry.mmproj else 0)
                + ub_logits_bytes(entry.n_vocab, mtp_capable=entry.mtp))
    native = entry.n_ctx_train or FLOOR
    variant = entry.variants[-1]
    profile = entry.profile(variant)
    need = variant.weights_bytes + overhead
    vram = budget.usable_vram_bytes
    if need + ctx_bytes(profile, min(TARGET_WINDOW, native)) <= vram:
        return VariantChoice(variant, zero_spill=True, reason_key="best-large-window")
    floor_kv = ctx_bytes(profile, min(FLOOR, native))
    if need + floor_kv <= vram:
        return VariantChoice(variant, zero_spill=True, reason_key="best-fits")
    if need + floor_kv <= vram + budget.ram_available_bytes:
        return VariantChoice(variant, zero_spill=False, reason_key="smallest-fits-spilled")
    return None


# ── recommendation: best quality that fits and isn't miserably slow ──
#
# Two axes, each living where it belongs. QUALITY is a judgment made once,
# globally, at authoring time (entry.quality — AA-informed, editorially
# owned). SPEED is physics computed per machine: decode is memory-bound,
# so predicted tok/s ≈ bandwidth / bytes-read-per-token, and the bytes per
# token are the build's size scaled by its decode fraction (dense reads
# everything; MoE reads the active slice). The pick: highest quality among
# entries that run resident and clear a pleasant speed floor; else the
# fastest resident entry; else the least-painful spilled one.
#
# The bandwidth axis is the `uma` flag for now: every discrete card that
# matters is 900+ GB/s GDDR while the unified-memory class measures ~1/5th
# of that, so the flag IS the high/low split. A measured per-machine
# bandwidth (one cached memcpy probe) can replace these class constants
# without touching the rule; predictions order candidates and gate the
# floor — they are not display values.

_DISCRETE_BANDWIDTH_GB_S = 1000.0   # representative GDDR6X/GDDR7 class
_UMA_BANDWIDTH_GB_S = 210.0         # measured on unified-memory NVIDIA
_HOST_BANDWIDTH_GB_S = 80.0         # spilled weights stream over host DRAM

# The one editorial constant in the tree: below this predicted decode
# speed a model stops feeling pleasant for agentic use (roughly reading
# speed with headroom for tool-call bursts). Distinct from the growth
# policy's 6 tok/s compress floor, which marks unusable, not unpleasant.
PLEASANT_FLOOR_TOK_S = 20.0


def predicted_decode_tok_s(entry: CatalogEntry, variant: QuantVariant,
                           budget: HardwareBudget, *,
                           spilled: bool = False) -> float:
    """Memory-bound decode prediction for ordering and floor-gating."""
    bandwidth = (_HOST_BANDWIDTH_GB_S if spilled
                 else _UMA_BANDWIDTH_GB_S if budget.uma
                 else _DISCRETE_BANDWIDTH_GB_S)
    bytes_per_token = max(1.0, variant.size_bytes * entry.decode_fraction)
    return bandwidth * 1e9 / bytes_per_token


def recommended_entry(budget: HardwareBudget,
                      entries: "tuple[CatalogEntry, ...] | None" = None
                      ) -> "tuple[CatalogEntry, str] | None":
    """The catalog's default pick for THIS machine, with its reason.

    Callers pass pre-filtered entries when some are ineligible for reasons the catalog can't know
    (engine too old); default is the full catalog.

    best-quality-resident quality won among resident entries that clear the pleasant floor speed-
    gated-quality same, but the floor eliminated a HIGHER quality candidate — the exact 'why not the
    big model?' a unified-memory owner asks fastest-resident nothing resident clears the floor; the
    quickest resident entry wins least-painful-spilled nothing runs resident; fastest from host
    memory (MoE by construction)
    """
    pool = CATALOG if entries is None else entries
    fitting = [(e, c) for e in pool if (c := select_variant(e, budget)) is not None]
    if not fitting:
        return None

    def speed(t, spilled=False):
        return predicted_decode_tok_s(t[0], t[1].variant, budget, spilled=spilled)

    resident = [(e, c) for e, c in fitting if c.zero_spill]
    pleasant = [t for t in resident if speed(t) >= PLEASANT_FLOOR_TOK_S]
    if pleasant:
        pick = max(pleasant, key=lambda t: (t[0].quality, -t[1].variant.size_bytes))[0]
        floor_gated = any(e.quality > pick.quality for e, _ in resident)
        return (pick, "speed-gated-quality" if floor_gated else "best-quality-resident")
    if resident:
        return (max(resident, key=speed)[0], "fastest-resident")
    # Everything spills: take the least painful — fastest predicted decode
    # from host memory (MoE wins here by construction; a dense spill
    # streams every weight over the host bus).
    return (max(fitting, key=lambda t: speed(t, spilled=True))[0], "least-painful-spilled")


# ── catalog data: packaged JSON, refreshed from GitHub in memory ─
#
# The catalog DATA lives in catalog.json (checked in beside this module
# and shipped as package data); this module keeps all policy. At import
# we load the packaged copy — no network on the import path. A TTL-gated
# background refresh fetches the same file from the repo's main branch
# and swaps it in memory only: nothing on disk changes, so a git
# checkout never sees a dirty tracked file and the packaged copy remains
# the offline truth. A reverted commit on main heals every install on
# its next fetch, and day-0 entries reach users without an app release.

_CATALOG_URL = ("https://raw.githubusercontent.com/NousResearch/hermes-agent"
                "/main/hermes_cli/local_runtime/catalog.json")
_SCHEMA_VERSION = 1
_REFRESH_TTL_S = 6 * 3600
_refresh_lock = threading.Lock()
_last_refresh_attempt = 0.0


def _asset_from(d: "dict | None") -> "AssetFile | None":
    if not d:
        return None
    return AssetFile(path=d["path"], size_bytes=int(d["size_bytes"]),
                     local=d.get("local"))


# Scalar CatalogEntry fields parsed from JSON: key -> (coerce, default);
# a None default means the key is required.
_SCALAR_FIELDS = {
    "n_ctx_train": (int, None), "full_layers": (int, None),
    "recurrent_layers": (int, None), "per_layer_f16": (int, None),
    "swa_layers": (int, 0), "swa_window": (int, 0),
    "moe": (bool, False), "mtp": (bool, False), "mtp_draft_depth": (int, 3),
    "n_vocab": (int, 0), "sampling": (dict, {}), "min_engine": (str, ""),
    "quality": (int, 0), "decode_fraction": (float, 1.0),
}


def _load_catalog(doc: dict) -> "tuple[CatalogEntry, ...]":
    """Parse a catalog document into entries. Unknown fields are ignored
    (newer catalogs stay readable by older apps); a major schema bump is
    the signal that they wouldn't be, and the caller skips the document."""
    if int(doc.get("schema_version", 0)) != _SCHEMA_VERSION:
        raise ValueError(f"catalog schema {doc.get('schema_version')!r} "
                         f"(this build reads {_SCHEMA_VERSION})")
    entries = []
    for m in doc["models"]:
        variants = tuple(
            QuantVariant(quant=v["quant"],
                         files=tuple(_asset_from(f) for f in v["files"]),
                         validated=bool(v.get("validated")))
            for v in m["variants"])
        scalars = {k: coerce(m[k] if default is None else m.get(k, default))
                   for k, (coerce, default) in _SCALAR_FIELDS.items()}
        entries.append(CatalogEntry(
            id=m["id"], display_name=m["display_name"],
            description=m["description"], repo=m["repo"], variants=variants,
            mmproj=_asset_from(m.get("mmproj")), draft=_asset_from(m.get("draft")),
            **scalars))
    return tuple(entries)


def _packaged_catalog() -> "tuple[CatalogEntry, ...]":
    from importlib.resources import files

    raw = files("hermes_cli.local_runtime").joinpath("catalog.json").read_text(
        encoding="utf-8")
    return _load_catalog(json.loads(raw))


CATALOG: "tuple[CatalogEntry, ...]" = _packaged_catalog()


def refresh_catalog(force: bool = False) -> bool:
    """Fetch the current catalog from the repo and swap it in memory.

    Best-effort by design: any failure (offline, GitHub down, unreadable schema) leaves the running
    catalog untouched and retries after the TTL. Returns True when a fetched document replaced the
    catalog.
    """
    global CATALOG, _last_refresh_attempt

    now = time.monotonic()
    with _refresh_lock:
        if not force and now - _last_refresh_attempt < _REFRESH_TTL_S:
            return False
        _last_refresh_attempt = now
    try:
        req = urllib.request.Request(
            _CATALOG_URL, headers={"User-Agent": "hermes-local-runtime"})
        with urllib.request.urlopen(req, timeout=10) as r:
            fetched = _load_catalog(json.load(r))
    except Exception as exc:  # noqa: BLE001
        logger.debug("catalog refresh skipped: %s", exc)
        return False
    if fetched != CATALOG:
        logger.info("catalog refreshed from repo (%d models)", len(fetched))
    CATALOG = fetched
    return True


def refresh_catalog_soon() -> None:
    """TTL-gated background refresh; returns immediately. The caller's current request serves the
    catalog it already has — the refresh lands for the next one.
    """
    if time.monotonic() - _last_refresh_attempt < _REFRESH_TTL_S:
        return
    threading.Thread(target=refresh_catalog, daemon=True,
                     name="catalog-refresh").start()


def catalog_by_id() -> dict[str, CatalogEntry]:
    return {entry.id: entry for entry in CATALOG}


def find_entry_for_model(model_id: str) -> "tuple[CatalogEntry, QuantVariant] | None":
    """Locate the entry + variant that owns a staged model id."""
    for entry in CATALOG:
        for variant in entry.variants:
            if variant.model_id == model_id:
                return entry, variant
    return None
