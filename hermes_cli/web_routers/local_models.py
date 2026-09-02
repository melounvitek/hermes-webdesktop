"""Local-models dashboard routes — the desktop's window into the managed
llama.cpp runtime.

Every payload carries plain-language, pre-formatted facts the UI can show
verbatim (what will this model do ON THIS MACHINE, how big is the download,
what is the runtime doing right now), never raw internals.

Long jobs (runtime install, model download) follow the repo's job pattern:
start-POST -> {job_id} -> GET poll with byte progress. Downloads are
byte-size checked against what the server declared (no hash verification by
design); a short download deletes the file and reports it plainly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from starlette.concurrency import run_in_threadpool

from hermes_cli import config as config_mod, web_deps
from hermes_cli.local_runtime import (
    binaries,
    bootstrap,
    catalog,
    context_policy,
    estimator,
    growth,
    hardware,
    hf_browse,
    load_progress,
    presets,
    supervisor,
)
from hermes_cli.local_runtime.endpoint import _state_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()

_GIB = 1 << 30
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_LLAMACPP_PROVIDERS = ("llamacpp", "llama.cpp", "llama-cpp")


def _human_gb(n: int | float) -> str:
    return f"{n / _GIB:.1f} GB"


def _job(kind: str, target: str, model_id: str | None = None) -> Dict[str, Any]:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "kind": kind,               # "runtime-install" | "model-download" | ...
        "target": target,
        "model_id": model_id,       # catalog id for downloads; None otherwise
        "status": "running",        # running | done | error
        "phase": "starting",        # human-readable step name
        "detail": "",
        "total_bytes": None,
        "done_bytes": 0,
        "started_at": time.time(),
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["job_id"]] = job
    return job


def _job_view(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    if out["total_bytes"]:
        out["percent"] = min(100, round(out["done_bytes"] / out["total_bytes"] * 100))
    return out


def _finish(job: Dict[str, Any], detail: str) -> None:
    job["phase"] = "done"
    job["status"] = "done"
    job["detail"] = detail


def _spawn_job(job: Dict[str, Any], name: str, body: Callable[[], None], *,
               fail_msg: str | None = None,
               on_exit: Callable[[], None] | None = None) -> None:
    """Run ``body`` on a daemon thread; any exception marks the job errored
    (optionally warning ``fail_msg`` with the exception). ``on_exit`` always
    runs last (lock release)."""
    def _run():
        try:
            body()
        except Exception as exc:  # noqa: BLE001
            if fail_msg:
                logger.warning(fail_msg, exc)
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            if on_exit is not None:
                on_exit()

    threading.Thread(target=_run, daemon=True, name=name).start()


def _refresh_runtime(skip_msg: str) -> None:
    """Bounce a running router so it rescans the models dir; the router only
    scans at spawn, so a new/deleted file is invisible until then. Never
    raises — the file operation already succeeded."""
    try:

        bootstrap.refresh_local_runtime()
    except Exception:  # noqa: BLE001
        logger.debug(skip_msg, exc_info=True)


def _router_request(endpoint: Dict[str, Any], path: str, *, timeout: float,
                    payload: dict | None = None) -> Any:
    """Call the local router (base_url minus ``/v1``) with its bearer key.
    GET (no payload) returns the parsed JSON body; POST returns None."""
    headers = {"Authorization": f"Bearer {endpoint.get('api_key', '')}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint["base_url"].rsplit("/v1", 1)[0] + path, data=data, headers=headers,
        method="POST" if payload is not None else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return None if payload is not None else json.loads(r.read())


# ── fast download: ranged parallel streams ───────────────────

# One TCP stream to a CDN rarely fills a fast line; 8 ranged connections
# writing into a preallocated file saturate consumer gigabit.
_DOWNLOAD_CONNECTIONS = 8
_CHUNK = 4 << 20


def _probe_range_support(url: str) -> int:
    """Total size when the server honors Range requests, else 0.

    A 401/403 from the CDN means the repo is gated or the catalog names a
    wrong repo — raise with a plain-language message, not a bare status."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status == 206:
                content_range = r.headers.get("Content-Range", "")
                if "/" in content_range:
                    return int(content_range.rsplit("/", 1)[1])
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "The model host refused the download (gated or moved). "
                "This is a catalog problem, not yours — please report it.") from exc
        raise
    except Exception:  # noqa: BLE001
        pass
    return 0


def _model_id_for(gguf: Path) -> str:
    """Variant model id for a staged file (strips split-part suffixes)."""
    return re.sub(r"-\d{5}-of-\d{5}$", "", gguf.stem)


def _variant_files_on_disk(model_id: str) -> "list[Path]":
    """Every local file belonging to a staged model: all split parts plus
    its catalog-declared assets (mmproj/draft) when present."""

    files = [p for p in _models_dir().glob("*.gguf") if _model_id_for(p) == model_id]
    hit = catalog.find_entry_for_model(model_id)
    if hit is not None:
        for asset in (hit[0].mmproj, hit[0].draft):
            if asset is not None and (bootstrap.assets_dir() / asset.local_name).exists():
                files.append(bootstrap.assets_dir() / asset.local_name)
    return files


def download_file(url: str, dest: Path, job: Dict[str, Any],
                  *,
                  base_done: int = 0, keep_totals: bool = False) -> None:
    """Download url -> dest with byte progress on ``job``.

    Ranged-parallel when the server supports it, single-stream fallback
    otherwise. No integrity check against the CATALOG by design: catalog
    sizes may lag an upstream re-upload, and a newer file must download
    fine. Completeness is checked only against what the SERVER declared for
    this transfer (range-probe total / Content-Length), so a dropped
    connection still errors instead of staging a truncated file. Never
    leaves a .part behind.

    Multi-file variants: ``base_done`` offsets the progress so this file's
    bytes accumulate onto the files before it, and ``keep_totals=True``
    stops the per-file size from overwriting the variant's total.
    """
    tmp = dest.with_suffix(".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_done = [0]
    progress_lock = threading.Lock()

    def bump(n: int) -> None:
        with progress_lock:
            file_done[0] += n
            job["done_bytes"] = base_done + file_done[0]

    def pump(r, f) -> None:
        while True:
            chunk = r.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            bump(len(chunk))

    try:
        # The probe and the preallocation both take real seconds on a 20+ GB
        # file — narrate them, or the pane shows a dead '— of X GB'.
        job["detail"] = "Connecting"
        total = _probe_range_support(url)
        if total:
            if not keep_totals:
                job["total_bytes"] = total
            # Preallocate so each worker writes at its own offset.
            job["detail"] = f"Reserving {_human_gb(total)} of disk space"
            with open(tmp, "wb") as f:
                f.truncate(total)
            job["detail"] = ""
            errors: list[Exception] = []
            bounds = [(i * total // _DOWNLOAD_CONNECTIONS,
                       (i + 1) * total // _DOWNLOAD_CONNECTIONS - 1)
                      for i in range(_DOWNLOAD_CONNECTIONS)]

            def fetch_range(start: int, end: int) -> None:
                try:
                    req = urllib.request.Request(
                        url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r, \
                            open(tmp, "r+b") as f:
                        f.seek(start)
                        pump(r, f)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=fetch_range, args=b, daemon=True,
                                        name=f"lm-dl-{i}")
                       for i, b in enumerate(bounds)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            if file_done[0] != total:
                raise RuntimeError(
                    f"download incomplete ({file_done[0]} of {total} bytes)")
        else:
            # No range support: single stream. Completeness is judged by the
            # server's own Content-Length when it sent one — never the catalog.
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                length = int(r.headers.get("Content-Length") or 0)
                if length and not keep_totals:
                    job["total_bytes"] = length
                pump(r, f)
            if length and file_done[0] != length:
                raise RuntimeError(
                    f"Download ended at {file_done[0]:,} bytes but the server "
                    f"said {length:,} — connection dropped? Removed; try again")

        shutil.move(str(tmp), str(dest))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _models_dir() -> Path:

    return bootstrap.models_dir()


def _hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _download_plan(entry, variant) -> list:
    """Everything a variant needs: split parts + mmproj/draft assets, as
    (url, dest, bytes) tuples."""

    plan = [(_hf_url(entry.repo, a.path), _models_dir() / a.local_name, a.size_bytes)
            for a in variant.files]
    plan += [(_hf_url(entry.repo, a.path), bootstrap.assets_dir() / a.local_name, a.size_bytes)
             for a in (entry.mmproj, entry.draft) if a is not None]
    return plan


def _run_download_plan(job: Dict[str, Any], plan: list, label: str) -> None:
    """Download every missing file in ``plan``; already-present files count
    toward progress without a transfer."""
    total = sum(p[2] for p in plan)
    job["phase"] = "downloading"
    job["detail"] = f"{label} — {_human_gb(total)}"
    done_before = 0
    for url, dest, size in plan:
        if not dest.exists():
            download_file(url, dest, job, base_done=done_before, keep_totals=True)
            job["phase"] = "downloading"
        done_before += size
        job["done_bytes"] = done_before


def _engine_too_old(min_engine: str) -> bool:
    """True when the installed llama.cpp predates a model's requirement.
    Tags are release numbers (b10362); no engine installed compares as
    too old only when the model states a requirement."""
    if not min_engine:
        return False
    try:

        tags = binaries.installed_tags() or [binaries.default_tag()]
        newest = max(int(t.lstrip("b")) for t in tags if t.lstrip("b").isdigit())
        return newest < int(min_engine.lstrip("b"))
    except Exception:  # noqa: BLE001
        return False


def _load_config() -> dict:

    try:
        return config_mod.load_config()
    except Exception:  # noqa: BLE001
        return {}


def _runtime_section() -> dict:
    return (_load_config() or {}).get("local_runtime") or {}


def _set_runtime_enabled(enabled: bool) -> dict:
    """Persist ``local_runtime.enabled`` and return the config written."""

    config = config_mod.load_config()
    config.setdefault("local_runtime", {})["enabled"] = enabled
    config_mod.save_config(config)
    return config


def _resolve_backend(section: dict, requested: str | None = None) -> str:

    backend = requested or section.get("backend", "auto")
    return binaries.select_backend(bootstrap._detect_gpu_vendor()) if backend == "auto" else backend


def _eligible_entries():
    """Catalog entries this engine can activate today (engine-gated ones
    can't be the recommendation either)."""

    return tuple(e for e in catalog.CATALOG if not _engine_too_old(e.min_engine))


def _ensure_server(job: Dict[str, Any], config: dict, model_id: str, *,
                   fail_detail: str, skip_msg: str) -> None:
    """Start the local server if needed and self-heal a stale router: the
    model list is spawn-only, so a server started before ``model_id``
    finished downloading can't serve it — bounce it when it doesn't know
    the model."""

    job["phase"] = "starting-server"
    job["detail"] = "Starting the local server"
    sup = bootstrap.ensure_local_runtime(config, force=True)
    if sup is None and _state_endpoint() is None:
        raise RuntimeError(fail_detail)
    if sup is not None:
        try:
            if model_id not in sup.models():
                job["detail"] = "Refreshing the local server"
                bootstrap.refresh_local_runtime()
        except Exception:  # noqa: BLE001
            logger.debug(skip_msg, exc_info=True)


def _assign_default(job: Dict[str, Any], model_id: str) -> None:
    """Make ``model_id`` the main model through the same machinery as
    /api/model/set (late-bound so tests can stub web_deps.late)."""

    job["phase"] = "setting-default"
    job["detail"] = "Making it your default"
    web_deps.late("_apply_model_assignment_sync")("main", "llamacpp", model_id, "", "", "")


# ── status: the one call the pane opens with ─────────────────


def _loaded_models(running: Dict[str, Any]) -> "tuple[Dict[str, str], Dict[str, Any]]":
    """Which staged models are resident right now, plus how each is placed.

    Placement is the granted window from the child itself and the plan's
    spill facts from the preset decision — the difference between 'fast'
    and 'why is my CPU busy', so it must be inspectable, not inferred."""

    data = _router_request(running, "/models", timeout=3)
    # Everything resident or becoming resident: 'loading' renders as its own
    # state in the pane (a 20-GB load in flight is the most important thing
    # the pane can show).
    loaded = {
        m["id"]: m.get("status", {}).get("value", "unknown")
        for m in data.get("data", [])
        if m.get("status", {}).get("value") in ("loaded", "ready", "loading")
    }
    placement: Dict[str, Any] = {}
    decisions = presets.read_preset_decisions()
    for model_id, state in loaded.items():
        facts: Dict[str, Any] = {}
        plan = decisions.get(model_id)
        if plan is not None:
            facts["window"] = plan.window
            facts["window_label"] = f"{plan.window // 1024}K"
            facts["spilled"] = plan.spilled
        if state in ("loaded", "ready"):
            try:
                props = _router_request(running, f"/props?model={model_id}", timeout=3)
                n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
                if n_ctx:
                    facts["granted_window"] = int(n_ctx)
                    facts["granted_window_label"] = f"{int(n_ctx) // 1024}K"
            except Exception:  # noqa: BLE001
                pass
        if facts:
            placement[model_id] = facts
    return loaded, placement


@router.get("/api/local-models/status")
def local_models_status():
    """Cheap, immediate, never blocks on probes: config state + installed
    runtime + staged models + supervisor state. GPU facts come from
    /api/local-models/hardware (slower, polled).

    Sync def on purpose: blocking urlopen/scans run in FastAPI's threadpool
    instead of stalling the event loop."""

    section = _runtime_section()
    configured_tag = section.get("tag") or binaries.default_tag()
    have = binaries.installed_tags()

    # The tag actually serving (boot ladder: configured if installed, else
    # newest installed).
    tag = configured_tag if configured_tag in have else (have[0] if have else configured_tag)

    # A pending engine update exists when the user runs the local engine
    # (enabled + something installed) and the configured tag — pinned or the
    # Hermes-release default — is newer than anything on disk. The download
    # is a button click, never automatic.
    update_available = bool(
        section.get("enabled") and have and configured_tag not in have)

    runtime_installed = False
    runtime_backend = None
    root = binaries.runtimes_root() / tag
    if root.exists():
        for backend_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                binaries.server_binary(backend_dir)
                runtime_installed = True
                runtime_backend = backend_dir.name
                break
            except Exception:  # noqa: BLE001
                continue

    staged = []
    mdir = _models_dir()
    if mdir.exists():

        for gguf in bootstrap.staged_models():
            model_id = _model_id_for(gguf)
            # Split models: report the whole variant's bytes, not one part's.
            hit = catalog.find_entry_for_model(model_id)
            size = hit[1].size_bytes if hit is not None else gguf.stat().st_size
            staged.append({"id": model_id, "size_bytes": size, "size_label": _human_gb(size)})

    running = _state_endpoint()

    # Resident models from the live router; {} when down. Feeds the pane's
    # Loaded pills and eject buttons.
    loaded: Dict[str, str] = {}
    placement: Dict[str, Any] = {}
    if running is not None:
        try:
            loaded, placement = _loaded_models(running)
        except Exception as exc:  # noqa: BLE001
            # Never silent: an empty dict here renders as 'Not in memory'
            # on a machine whose VRAM is visibly full.
            logger.warning("loaded-models read failed: %r", exc)
            loaded = {}

    # The active main model, when it is one of ours (config authority: the
    # same model.provider + model.default that /api/model/set writes).
    active_model_id = None
    try:
        model_section = (_load_config() or {}).get("model") or {}
        if str(model_section.get("provider", "")).strip().lower() in _LLAMACPP_PROVIDERS:
            active_model_id = str(
                model_section.get("default") or model_section.get("name") or ""
            ).strip() or None
    except Exception:  # noqa: BLE001
        pass

    return {
        "enabled": bool(section.get("enabled")),
        "tag": tag,
        "configured_tag": configured_tag,
        "update_available": update_available,
        "runtime_installed": runtime_installed,
        "runtime_backend": runtime_backend,
        "server_running": running is not None,
        "server_base_url": (running or {}).get("base_url"),
        "active_model_id": active_model_id,
        "loaded_models": loaded,
        # Live load progress per model (SSE-fed): {model_id: {stage, value,
        # percent}}. The chat's loading bar and the picker rows poll this.
        "loading": _loading_progress(),
        "placement": placement,
        "models": staged,
        "models_dir": str(mdir),
    }


def _loading_progress() -> Dict[str, Any]:
    try:

        return load_progress.get_loading_progress()
    except Exception:  # noqa: BLE001 — progress is garnish, never a 500
        return {}


# ── hardware: what this machine can do ───────────────────────


@router.get("/api/local-models/hardware")
def local_models_hardware():
    """The budget as plain facts. Polled by the pane and the statusbar
    resource item (throttled client-side). Sync def on purpose: shells out
    to nvidia-smi and probes budgets — threadpool, not loop."""

    budget = hardware.probe_budget()
    ram_total, ram_avail = hardware._ram_bytes()
    out = {
        "uma": budget.uma,
        "vram_total_bytes": budget.total_device_bytes,
        "vram_usable_bytes": budget.usable_vram_bytes,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_avail,
        "vram_label": _human_gb(budget.total_device_bytes),
        "gpu_name": None,
        "gpu_util_percent": None,
        "vram_used_bytes": None,
    }
    # GPU identity + live utilization (NVIDIA; other vendors degrade to None
    # and the UI hides those readouts).
    try:

        smi_exe = hardware._nvidia_smi_path()
        smi = subprocess.run(
            [smi_exe, "--query-gpu=name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5) if smi_exe else None
        if smi and smi.returncode == 0 and smi.stdout.strip():
            name, util, used_mib = (x.strip() for x in smi.stdout.strip().splitlines()[0].split(","))
            out["gpu_name"] = name
            out["gpu_util_percent"] = int(util)
            out["vram_used_bytes"] = int(used_mib) << 20
    except Exception:  # noqa: BLE001
        pass
    return out


# ── catalog: priced for THIS machine before download ─────────

_QUANT_REASONS = {
    "best-large-window": ("Recommended build ({quant}) — the quant class this "
                          "engine is optimized for; runs fully on your GPU with a "
                          "large context window"),
    "best-fits": ("Recommended build ({quant}) — the quant class this "
                  "engine is optimized for; runs fully on your GPU"),
}
_QUANT_REASON_COMPACT = ("Compact build sized for this machine ({quant}) — "
                         "larger than GPU memory, runs slower")


@router.get("/api/local-models/catalog")
def local_models_catalog():
    """Every entry answers the user's three questions up front: how big is
    the download, will it fit, and what context/speed shape will I get —
    from the catalog's measured numbers + this machine's budget. The row
    advertises the BEST build for this machine (highest quality that runs
    fully on the GPU at the 64K floor; else the smallest that works, spilled
    and priced). No entry is hidden; unaffordable models show WHY. Sync def
    on purpose: probe_budget + catalog I/O block — threadpool, not loop."""

    # Serve the catalog already in memory; a TTL-gated background fetch lands
    # new entries for the next call (day-0 models without an app release).
    catalog.refresh_catalog_soon()

    # Planning budget: price against machine capacity, not live-free VRAM —
    # a loaded model must not make every row unaffordable.
    budget = hardware.probe_budget(planning=True)
    # The default pick for THIS machine: quality-ranked, fit- and speed-gated.
    # The reason key ships with the row so the Recommended badge's tooltip is
    # the branch that actually fired, not a re-derivation that can drift.
    picked = catalog.recommended_entry(budget, _eligible_entries())
    recommended = picked[0].id if picked is not None else None
    recommended_reason = picked[1] if picked is not None else None
    # Completeness-checked staging (split parts all present) — same answer the
    # picker and router see, so a mid-download model never reads as downloaded.
    staged_ids = set(bootstrap.staged_model_ids())
    entries = []
    for entry in catalog.CATALOG:
        choice = catalog.select_variant(entry, budget)
        # Any variant of this family on disk counts as downloaded.
        downloaded_variant = next(
            (v for v in entry.variants if v.model_id in staged_ids), None)
        row: Dict[str, Any] = {
            "id": entry.id,
            "display_name": entry.display_name,
            "description": entry.description,
            "native_context": entry.n_ctx_train,
            "native_context_label": f"{entry.n_ctx_train // 1024}K",
            "recommended": entry.id == recommended,
            "recommended_reason": recommended_reason if entry.id == recommended else None,
            "downloaded": downloaded_variant is not None,
            "downloaded_model_id": downloaded_variant.model_id if downloaded_variant else None,
            "downloaded_quant": downloaded_variant.quant if downloaded_variant else None,
            "mtp": entry.mtp,
            "vision": entry.mmproj is not None,
            # Day-0 architectures need the llama.cpp release where their support
            # landed. True gates download/activate in the pane until the engine
            # updates; the row still renders (visible + explained beats hidden).
            "needs_engine": _engine_too_old(entry.min_engine),
            "min_engine": entry.min_engine or None,
        }
        if choice is None:
            smallest = min(entry.variants, key=lambda v: v.size_bytes)
            smallest_total = entry.download_bytes(smallest)
            row.update({
                "fits": False,
                "size_bytes": smallest_total,
                "size_label": _human_gb(smallest_total),
                "fit_summary": "Needs more memory than this machine has",
                "fit_detail": (f"even the most compact build ({smallest.quant}, "
                               f"{_human_gb(smallest_total)}) exceeds GPU + system memory"),
            })
            entries.append(row)
            continue

        variant = choice.variant
        # Same overhead the launch decision prices (runtime buffers + vision
        # projector + microbatch/MTP logits): the row must advertise the window
        # the model will actually get, not a paper number.
        overhead = (context_policy.RUNTIME_OVERHEAD_BYTES
                    + (entry.mmproj.size_bytes if entry.mmproj else 0)
                    + context_policy.ub_logits_bytes(entry.n_vocab, mtp_capable=entry.mtp))
        decision = context_policy.initial_window(entry.profile(variant), budget, overhead_bytes=overhead)
        download_total = entry.download_bytes(variant)
        row.update({
            "fits": True,
            "model_id": variant.model_id,
            "quant": variant.quant,
            "quant_validated": variant.validated,
            "size_bytes": download_total,
            "size_label": _human_gb(download_total),
            "variant_count": len(entry.variants),
            "quant_reason": _QUANT_REASONS.get(
                choice.reason_key, _QUANT_REASON_COMPACT).format(quant=variant.quant),
        })
        if not isinstance(decision, estimator.PhysicsRefusal):
            row["start_window"] = decision.window
            row["start_window_label"] = f"{decision.window // 1024}K"
            row["spilled"] = decision.spilled
            if decision.window >= entry.n_ctx_train:
                shape = f"runs at its full {row['native_context_label']} context"
            else:
                shape = (f"starts at {row['start_window_label']} and grows toward "
                         f"{row['native_context_label']} as you use it")
            if decision.spilled:
                shape += " (larger than your GPU memory — runs slower)"
            row["fit_summary"] = shape
        else:
            row["fit_summary"] = row["quant_reason"]
        entries.append(row)
    return {"models": entries}


# ── runtime install (job) ────────────────────────────────────


class RuntimeInstallBody(BaseModel):
    backend: Optional[str] = None   # None/auto -> detect


def _runtime_progress_hook(job: Dict[str, Any]):
    """Adapter: ensure_runtime_installed's progress stream -> job fields.

    Throttled to ~4 updates/s. Byte counters are CUMULATIVE across the plan:
    a multi-asset engine (CUDA zip + cudart zip) reads as one growing
    download, not a bar that restarts per asset; the total grows as each
    asset's size becomes known. Unpack/verify keep the download's counters —
    a bar bouncing back to zero after the bytes finished reads as failure."""
    state = {"last": 0.0, "banked": 0, "asset": None, "asset_total": 0}

    def hook(stage: str, done: int, total: int, label: str) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.25 and done < total:
            return
        state["last"] = now
        suffix = f" ({label})" if label else ""
        if stage == "download":
            if label != state["asset"]:
                # Previous asset finished: bank its bytes so the counters keep
                # climbing instead of restarting for the next asset.
                state["banked"] += state["asset_total"]
                state["asset"] = label
            state["asset_total"] = total or done
            plan_done = state["banked"] + done
            plan_total = state["banked"] + (total or 0)
            job["phase"] = "downloading-runtime"
            job["detail"] = f"Downloading the local engine{suffix} — {_human_gb(plan_done)}"
            if total:
                job["detail"] += f" of {_human_gb(plan_total)}"
            job["done_bytes"] = plan_done
            job["total_bytes"] = plan_total or None
        elif stage == "extract":
            job["phase"] = "unpacking-runtime"
            pct = f" — {min(100, round(done / total * 100))}%" if total else ""
            job["detail"] = f"Unpacking the engine{suffix}{pct}"
        else:  # verify
            job["phase"] = "verifying-runtime"
            job["detail"] = f"Verifying the engine{suffix}"

    return hook


@router.post("/api/local-models/runtime/install")
async def local_models_runtime_install(body: RuntimeInstallBody):

    section = _runtime_section()
    tag = section.get("tag") or binaries.default_tag()
    backend = _resolve_backend(section, body.backend)
    # Resolve first so an impossible combination fails the POST, not the job.
    try:
        plan = binaries.resolve_assets(tag, backend)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    job = _job("runtime-install", f"llama.cpp {tag} ({backend})")

    def _run():

        previous = binaries.installed_tags()
        job["phase"] = "downloading"
        job["detail"] = f"Fetching {len(plan.assets)} package(s) for {backend}"
        binaries.ensure_runtime_installed(tag, backend, progress=_runtime_progress_hook(job))

        # Engine update path: a server already running on an older tag moves
        # to the new one now — the click was the consent. Fresh installs (no
        # server) skip this; Use/boot handles their start.
        restarted = False
        try:

            if bootstrap.get_supervisor() is not None and previous and tag not in previous:
                job["phase"] = "restarting"
                job["detail"] = "Switching the running server to the new build"
                bootstrap.shutdown_local_runtime()
                bootstrap.ensure_local_runtime(_load_config(), force=True)
                restarted = True
        except Exception as exc:  # noqa: BLE001
            # The new build is installed either way; the next boot serves it.
            logger.warning("post-update restart skipped: %s", exc)

        # N-1 retention, only after the new tag verified: keep it and the
        # newest previous build as the rollback pin target.
        try:
            binaries.prune_old_tags([tag] + [t for t in previous if t != tag][:1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime prune skipped: %s", exc)

        _finish(job, f"llama.cpp {tag} ready ({backend})"
                + (" — server restarted on the new build" if restarted else ""))

    _spawn_job(job, "lr-runtime-install", _run, fail_msg="runtime install failed: %s")
    return {"job_id": job["job_id"], "backend": backend, "tag": tag}


# ── model download (job with byte progress) ──────────────────


class ModelDownloadBody(BaseModel):
    model_id: str


@router.post("/api/local-models/download")
async def local_models_download(body: ModelDownloadBody):
    """Accepts either a family id (downloads this machine's selected
    variant) or an exact variant model_id."""

    entry = catalog.catalog_by_id().get(body.model_id)
    variant = None
    if entry is not None:
        if _engine_too_old(entry.min_engine):
            raise HTTPException(
                status_code=409,
                detail=(f"{entry.display_name} needs llama.cpp {entry.min_engine} "
                        f"or newer — update the engine first"))
        # Same planning budget as the catalog — the user downloads exactly
        # the build the row advertised.
        choice = catalog.select_variant(entry, hardware.probe_budget(planning=True))
        if choice is None:
            raise HTTPException(status_code=409,
                                detail=f"no variant of {entry.id} fits this machine")
        variant = choice.variant
    else:
        hit = catalog.find_entry_for_model(body.model_id)
        if hit is not None:
            entry, variant = hit
    if entry is None or variant is None:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model_id}")

    if variant.model_id in bootstrap.staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": variant.model_id}

    plan = _download_plan(entry, variant)
    job = _job("model-download", f"{entry.display_name} ({variant.quant})",
               model_id=entry.id)
    job["total_bytes"] = sum(p[2] for p in plan)

    def _run():
        _run_download_plan(job, plan, entry.display_name)
        _finish(job, f"{entry.display_name} ready")
        _refresh_runtime("post-download runtime refresh skipped")

    _spawn_job(job, "lr-model-download", _run, fail_msg="model download failed: %s")
    return {"job_id": job["job_id"], "model_id": variant.model_id}


@router.delete("/api/local-models/models/{model_id}")
async def local_models_delete(model_id: str):
    """Remove a staged model: every split part plus its private assets, then
    bounce the router off the request thread (deleting the active file
    mid-serve is exactly the stale state the refresh exists for)."""
    files = _variant_files_on_disk(model_id)
    if not files:
        raise HTTPException(status_code=404, detail="model not found")
    for path in files:
        path.unlink(missing_ok=True)
    # Growth state dies with the model: a re-download starts back at its
    # zero-spill window instead of inheriting a stale grown one.
    try:

        growth.clear_window_override(model_id)
    except Exception:  # noqa: BLE001
        logger.debug("window-override clear skipped", exc_info=True)

    threading.Thread(target=_refresh_runtime, args=("post-delete runtime refresh skipped",),
                     daemon=True, name="lr-post-delete").start()
    return {"ok": True}


# ── server lifecycle: turn the engine on/off ─────────────────


class ServerActionBody(BaseModel):
    action: str                 # "stop" | "start"


# ── quickstart: one click from nothing to a working default ──


class QuickstartBody(BaseModel):
    model_id: str | None = None   # default: the catalog's recommended entry


# One quickstart at a time: the job sequences installs, downloads, a
# server bounce, and a config write — two racing runs would interleave
# all four. Held for the job's lifetime, released in the worker.
_QUICKSTART_LOCK = threading.Lock()


@router.post("/api/local-models/quickstart")
async def local_models_quickstart(body: QuickstartBody):
    """The dummy-proof path: one job that installs the runtime (if missing),
    downloads this machine's build of the recommended model (if missing),
    and makes it the default for new chats. Each leg is the same code the
    individual routes run — this route only sequences them, so 'Configure'
    and quickstart can never disagree about what gets installed.

    Preflight rejects (no servable entry, engine too old) fail the POST
    synchronously so the button can explain itself; everything slow runs in
    the job with the usual phase/byte progress.
    """

    # Resolve the target entry: explicit id, else this machine's
    # recommendation, else the first catalog entry this machine can serve.
    budget = hardware.probe_budget(planning=True)
    if body.model_id:
        entry = catalog.catalog_by_id().get(body.model_id)
        if entry is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown model {body.model_id}")
        candidates = [entry]
    else:
        picked = catalog.recommended_entry(budget, _eligible_entries())
        best = picked[0] if picked is not None else None
        candidates = ([best] if best is not None else []) + [
            e for e in catalog.CATALOG if best is None or e.id != best.id]
    chosen = None
    for candidate in candidates:
        choice = catalog.select_variant(candidate, budget)
        if choice is not None and not _engine_too_old(candidate.min_engine):
            chosen = (candidate, choice.variant)
            break
    if chosen is None:
        raise HTTPException(
            status_code=409,
            detail="no catalog model fits this machine — open Local Models "
                   "to browse for a smaller build")
    entry, variant = chosen

    section = _runtime_section()
    tag = section.get("tag") or binaries.default_tag()
    backend = _resolve_backend(section)
    need_runtime = not binaries.installed_tags()
    if need_runtime:
        # Same preflight as /runtime/install: impossible combos fail the POST.
        try:
            binaries.resolve_assets(tag, backend)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))

    need_download = variant.model_id not in bootstrap.staged_model_ids()
    download_plan = _download_plan(entry, variant) if need_download else []
    download_bytes = sum(p[2] for p in download_plan)

    if not _QUICKSTART_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409,
                            detail="Setup is already running")

    job = _job("quickstart", entry.display_name, model_id=entry.id)
    job["total_bytes"] = download_bytes or None

    def _run():
        if need_runtime:

            job["phase"] = "installing-runtime"
            job["detail"] = "Installing the local engine"
            binaries.ensure_runtime_installed(tag, backend, progress=_runtime_progress_hook(job))

        if need_download:
            # The runtime leg repurposed the byte counters for its own
            # stages — reset them to the model plan before download.
            job["done_bytes"] = 0
            job["total_bytes"] = download_bytes
            _run_download_plan(job, download_plan, entry.display_name)

        # Activate: same sequence as /activate's job body.
        _ensure_server(
            job, _set_runtime_enabled(True), variant.model_id,
            fail_detail="The local server could not start — open Local Models for details",
            skip_msg="quickstart rescan check skipped")
        _assign_default(job, variant.model_id)
        _finish(job, f"{entry.display_name} is ready — new chats use it")

    _spawn_job(job, "lr-quickstart", _run, fail_msg="quickstart failed: %s",
               on_exit=_QUICKSTART_LOCK.release)
    return {
        "job_id": job["job_id"],
        "model_id": entry.id,
        "display_name": entry.display_name,
        "needs_runtime": need_runtime,
        "needs_download": need_download,
        "download_bytes": download_bytes,
    }


def _stop_server() -> None:

    if bootstrap.get_supervisor() is not None:
        bootstrap.shutdown_local_runtime()
    elif _state_endpoint() is not None:
        # Server owned by another process (or an orphan): best-effort
        # terminate via the state file's pid, then clear the state.
        try:
            import psutil  # type: ignore


            state = json.loads(supervisor.state_path().read_text(encoding="utf-8"))
            pid = int(state.get("pid") or 0)
            if pid > 0 and psutil.pid_exists(pid):
                psutil.Process(pid).terminate()
            supervisor.state_path().unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    _set_runtime_enabled(False)


def _start_server() -> None:

    sup = bootstrap.ensure_local_runtime(_set_runtime_enabled(True), force=True)
    if sup is None and _state_endpoint() is None:
        raise RuntimeError("The local server could not start — check the "
                           "runtime is installed")


_SERVER_ACTIONS = {"stop": _stop_server, "start": _start_server}


@router.post("/api/local-models/server")
async def local_models_server(body: ServerActionBody):
    """Turn the local engine off (stop the server, free ALL GPU memory, and
    disable auto-start) or back on. The off switch is the whole-engine
    counterpart of per-model eject — and unlike eject it IS durable: the
    user said off, so boots stay off until they say on."""
    action = (body.action or "").strip().lower()
    if action not in _SERVER_ACTIONS:
        raise HTTPException(status_code=400, detail="action must be 'stop' or 'start'")
    try:
        await asyncio.to_thread(_SERVER_ACTIONS[action])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "action": action}


# ── activate: make a downloaded model THE model ──────────────


class ModelEjectBody(BaseModel):
    model_id: str


@router.post("/api/local-models/eject")
def local_models_eject(body: ModelEjectBody):
    """Free a loaded model's GPU memory now. Nothing reloads it except
    demand — the next message to it (residency v2: no automatic loading
    exists anywhere). Sync def on purpose: the fallback path blocks on a
    urlopen with a 120s timeout — threadpool, never the event loop."""

    sup = bootstrap.get_supervisor()
    if sup is not None:
        try:
            sup.unload_model(body.model_id)
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Server owned by another process (or state-file only): drive the
    # router directly with the persisted endpoint.
    endpoint = _state_endpoint()
    if endpoint is None:
        raise HTTPException(status_code=409, detail="local server is not running")
    try:
        _router_request(endpoint, "/models/unload", timeout=120,
                        payload={"model": body.model_id})
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class ModelActivateBody(BaseModel):
    model_id: str               # exact variant id (a staged .gguf stem)


@router.post("/api/local-models/activate")
async def local_models_activate(body: ModelActivateBody):
    """Make a downloaded model the default for new chats. Pure selection
    (residency v2): a config write through the same machinery as
    /api/model/set, plus making sure the server is up. NO model loading —
    models load on first inference, always; an empty router costs nothing.
    Kept as a job for UI continuity."""
    # Split variants stage under their first part — resolve like the rest
    # of the routes instead of assuming a single flat file.

    if body.model_id not in bootstrap.staged_model_ids():
        raise HTTPException(status_code=404, detail=f"{body.model_id} is not downloaded")

    job = _job("model-activate", body.model_id, model_id=body.model_id)

    def _run():

        _ensure_server(
            job, config_mod.load_config(), body.model_id,
            fail_detail="The local server could not start — check the runtime is installed",
            skip_msg="activate rescan check skipped")
        job["phase"] = "setting-default"
        job["detail"] = "Making it your default"
        _set_runtime_enabled(True)
        _assign_default(job, body.model_id)
        _finish(job, f"{body.model_id} is the default for new chats")

    _spawn_job(job, "lr-model-activate", _run, fail_msg="model activate failed: %s")
    return {"job_id": job["job_id"]}


# ── job polling ──────────────────────────────────────────────


@router.get("/api/local-models/jobs")
async def local_models_jobs():
    """All recent jobs, running first — the pane and the app-level poller
    rediscover in-flight work here after a remount or app restart."""
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(),
                      key=lambda j: (j["status"] != "running", -j["started_at"]))
    return {"jobs": [_job_view(job) for job in jobs[:20]]}


@router.get("/api/local-models/jobs/{job_id}")
async def local_models_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_view(job)


# ── Hugging Face browser: search, repo files, arbitrary download ─


@router.get("/api/local-models/search")
async def local_models_search(q: str, limit: int = 20):
    """Full-text HF search over GGUF models — the firehose behind the
    curated catalog. Per-quant fit pills come from the repo-files call once
    the user opens a hit."""


    if not q.strip():
        return {"hits": []}
    try:
        hits = await run_in_threadpool(hf_browse.search_models, q, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"Hugging Face search unavailable: {exc}") from exc
    return {"hits": [h.__dict__ for h in hits]}


@router.get("/api/local-models/search/files")
async def local_models_search_files(repo: str):
    """The servable GGUFs in one HF repo with a rough pre-download fit
    verdict per quant (file size + conservative fill-ins — the GGUF header
    refines it after download)."""


    try:
        groups = await run_in_threadpool(
            hf_browse.priced_repo_files, repo, hardware.probe_budget(planning=True))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"Could not list {repo}: {exc}") from exc
    return {"files": [dict(g.__dict__, paths=list(g.paths)) for g in groups]}


class BrowsedDownloadBody(BaseModel):
    repo: str
    paths: list[str]            # one GGUF, or every part of a split, in order


@router.post("/api/local-models/download-browsed")
async def local_models_download_browsed(body: BrowsedDownloadBody):
    """Download an arbitrary HF GGUF (browsed or pasted) into the managed
    models dir. From the moment it lands it is a normal staged model: the
    post-download bounce regenerates presets from its real header and the
    fit policy owns its launch. No catalog entry — it serves 'unverified',
    capabilities answered from the live server only."""

    paths = [p for p in (body.paths or []) if p.lower().endswith(".gguf")]
    if not paths:
        raise HTTPException(status_code=422, detail="no .gguf files given")
    first = paths[0].rsplit("/", 1)[-1]
    model_id = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first, flags=re.IGNORECASE)
    model_id = model_id[:-5] if model_id.lower().endswith(".gguf") else model_id
    if model_id in bootstrap.staged_model_ids():
        return {"job_id": None, "already_downloaded": True, "model_id": model_id}

    job = _job("model-download", f"{model_id} (from {body.repo})",
               model_id=model_id)

    def _run():
        job["phase"] = "downloading"
        for p in paths:
            dest = _models_dir() / p.rsplit("/", 1)[-1]
            if dest.exists():
                continue
            download_file(_hf_url(body.repo, urllib.parse.quote(p)), dest, job,
                          base_done=int(job.get("done_bytes") or 0),
                          keep_totals=bool(job.get("total_bytes")))
            job["phase"] = "downloading"
        _finish(job, f"{model_id} ready")
        _refresh_runtime("post-download runtime refresh skipped")

    _spawn_job(job, "lm-download-browsed", _run)
    return {"job_id": job["job_id"], "model_id": model_id}


class SideloadBody(BaseModel):
    path: str                   # absolute path to a .gguf on this machine


@router.post("/api/local-models/sideload")
async def local_models_sideload(body: SideloadBody):
    """Register a GGUF that already exists on this machine: link it into
    the managed models dir (copy only when linking is impossible) and
    bounce the router so it serves immediately. The original stays where
    it is; delete-from-Hermes removes only our link."""

    src = Path(body.path)
    if not src.is_file() or src.suffix.lower() != ".gguf":
        raise HTTPException(status_code=422, detail="Pick a .gguf model file")
    dest = _models_dir() / src.name
    if dest.exists():
        return {"ok": True, "model_id": dest.stem, "already_present": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)          # hardlink: instant, no extra disk
    except OSError:
        try:
            os.symlink(src, dest)   # cross-volume fallback
        except OSError:
            await run_in_threadpool(shutil.copyfile, src, dest)
    _refresh_runtime("post-sideload runtime refresh skipped")
    return {"ok": True, "model_id": dest.stem}
