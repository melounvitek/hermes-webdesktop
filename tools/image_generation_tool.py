#!/usr/bin/env python3
"""
Image Generation Tools Module

Provides image generation via FAL.ai. Multiple FAL models are supported and
selectable via ``hermes tools`` → Image Generation; the active model is
persisted to ``image_gen.model`` in ``config.yaml``.

Architecture:
- ``FAL_MODELS`` (``tools.image_generation_catalog``) holds per-model metadata
  (size-style family, defaults, ``supports`` whitelist, upscaler flag).
- ``_build_fal_payload()`` / ``_build_fal_edit_payload()`` translate the
  unified inputs into the model-specific payload, filtered to the whitelist so
  models never receive rejected keys.
- Upscaling (Clarity Upscaler) is strictly per-call opt-in: chained by default
  it degraded text rendering, CJK and faces, so ``upscale`` is False everywhere.
"""

import json
import logging
import os
import datetime
import threading
import uuid
from typing import Any, Dict, Optional

# fal_client is imported lazily (see _load_fal_client): an eager import cost
# ~64 ms on every CLI cold start because discover_builtin_tools() imports this
# module unconditionally. Tests that monkeypatch this attribute keep working:
# _load_fal_client() short-circuits when it is already truthy.
fal_client: Any = None


def _load_fal_client() -> Any:
    """Lazily import fal_client into the module global (idempotent; keeps a test-installed mock)."""
    global fal_client
    if fal_client is not None:
        return fal_client
    from tools.fal_common import import_fal_client
    fal_client = import_fal_client()
    return fal_client


from tools.debug_helpers import DebugSession
from tools.fal_common import (
    _ManagedFalSyncClient,
    _extract_http_status,
    _normalize_fal_queue_url_format,  # noqa: F401 — re-exported for tests
)
from tools.image_generation_catalog import (  # noqa: F401 — re-exported (plugins/tests/tools_config)
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MODEL,
    FAL_MODELS,
    UPSCALER_CREATIVITY,
    UPSCALER_DEFAULT_PROMPT,
    UPSCALER_FACTOR,
    UPSCALER_GUIDANCE_SCALE,
    UPSCALER_MODEL,
    UPSCALER_NEGATIVE_PROMPT,
    UPSCALER_NUM_INFERENCE_STEPS,
    UPSCALER_RESEMBLANCE,
    UPSCALER_SAFETY_CHECKER,
    VALID_ASPECT_RATIOS,
)
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    NOUS_MANAGED_PROVIDER,
    fal_key_is_configured,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    read_selection,
    selection_error,
)

logger = logging.getLogger(__name__)

_debug = DebugSession("image_tools", env_var="IMAGE_TOOLS_DEBUG")
_managed_fal_client = None
_managed_fal_client_config = None
_managed_fal_client_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Managed FAL gateway (Nous Subscription)
# ---------------------------------------------------------------------------
def _resolve_managed_fal_gateway():
    """Resolve the FAL route from the stored `hermes tools` selection.

    - ``"nous"`` (or legacy ``use_gateway: true``) → managed gateway ONLY; not
      entitled/unreachable is a selection-naming error, never a silent FAL_KEY fallback.
    - any other stored provider → direct FAL ONLY; missing FAL_KEY is an error
      naming FAL_KEY and the selection, never a silent managed reroute.
    - never configured → legacy autodetect: direct when FAL_KEY is set, else
      the managed gateway when resolvable, else None.

    Returns the managed gateway config, or ``None`` for the direct route.
    """
    selected = read_selection("image_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        gateway = resolve_managed_tool_gateway("fal-queue")
        if gateway is None:
            raise ValueError(selection_error(
                "image_gen",
                NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or "
                "unreachable)",
            ))
        return gateway
    if selected is not None:
        if not fal_key_is_configured():
            raise ValueError(selection_error(
                "image_gen",
                selected,
                "FAL_KEY is not set",
            ))
        return None
    # Never-configured category: legacy credential autodetect (do NOT persist).
    if fal_key_is_configured():
        return None
    return resolve_managed_tool_gateway("fal-queue")


def _get_managed_fal_client(managed_gateway):
    """Reuse the managed FAL client so its internal httpx.Client is not leaked per call."""
    global _managed_fal_client, _managed_fal_client_config

    client_config = (
        managed_gateway.gateway_origin.rstrip("/"),
        managed_gateway.nous_user_token,
    )
    with _managed_fal_client_lock:
        if _managed_fal_client is not None and _managed_fal_client_config == client_config:
            return _managed_fal_client

        # Resolve fal_client on this module so monkeypatching
        # ``image_generation_tool.fal_client`` still takes effect.
        _load_fal_client()
        _managed_fal_client = _ManagedFalSyncClient(
            fal_client,
            key=managed_gateway.nous_user_token,
            queue_run_origin=managed_gateway.gateway_origin,
        )
        _managed_fal_client_config = client_config
        return _managed_fal_client


class ImageGenerationInterrupted(Exception):
    """Raised when the user interrupts while a FAL job is in flight."""


def _wait_fal_result(handler, *, poll_seconds: float = 0.5):
    """Interrupt-aware replacement for a blind ``handler.get()``.

    ``handler.get()`` blocks inside the FAL SDK for 30-60s, during which a user
    interrupt was invisible. Run it on a daemon worker and poll the per-thread
    interrupt bit between join slices; on interrupt, abandon the worker (the
    remote job keeps running) and raise ``ImageGenerationInterrupted``.
    """
    from tools.interrupt import is_interrupted

    result_box: list = []
    error_box: list = []

    def _get():
        try:
            result_box.append(handler.get())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            error_box.append(exc)

    worker = threading.Thread(target=_get, daemon=True, name="fal-result-wait")
    worker.start()
    while worker.is_alive():
        if is_interrupted():
            raise ImageGenerationInterrupted(
                "Image generation interrupted by user — abandoned the "
                "in-flight FAL job."
            )
        worker.join(timeout=poll_seconds)
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def _submit_fal_request(model: str, arguments: Dict[str, Any]):
    """Submit a FAL request using direct credentials or the managed queue gateway."""
    _load_fal_client()
    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = _resolve_managed_fal_gateway()
    if managed_gateway is None:
        return fal_client.submit(model, arguments=arguments, headers=request_headers)

    managed_client = _get_managed_fal_client(managed_gateway)
    try:
        return managed_client.submit(
            model,
            arguments=arguments,
            headers=request_headers,
        )
    except Exception as exc:
        # A 4xx from the managed gateway usually means the portal doesn't proxy
        # this model (allowlist miss, billing gate) — give actionable remediation
        # instead of a raw httpx error.
        status = _extract_http_status(exc)
        if status is not None and 400 <= status < 500:
            gateway_message = ""
            if status in {401, 402, 403}:
                gateway_message = (
                    "\n\n"
                    + nous_tool_gateway_unavailable_message(
                        "managed FAL image generation",
                        force_fresh=True,
                    )
                )
            raise ValueError(
                f"Nous Subscription gateway rejected model '{model}' "
                f"(HTTP {status}). This model may not yet be enabled on "
                f"the Nous Portal's FAL proxy. Either:\n"
                f"  • Set FAL_KEY in your environment to use FAL.ai directly, or\n"
                f"  • Pick a different model via `hermes tools` → Image Generation."
                f"{gateway_message}"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Config readers, model resolution + payload construction
# ---------------------------------------------------------------------------
def _read_image_gen_key(key: str) -> Optional[str]:
    """Return the stripped ``image_gen.<key>`` string from config.yaml, or None."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.%s: %s", key, exc)
    return None


def _read_configured_image_model():
    """Return the value of ``image_gen.model`` from config.yaml, or None."""
    return _read_image_gen_key("model")


def _read_configured_image_provider():
    """Return ``image_gen.provider`` from config.yaml, or None.

    The plugin registry is consulted only when this is explicitly set — an
    unset value keeps users on the in-tree FAL fallback even when other
    providers happen to be registered (e.g. OPENAI_API_KEY present for other
    features). ``"fal"`` explicitly routes through ``plugins/image_gen/fal/``,
    which delegates back into this module via call-time indirection.
    """
    return _read_image_gen_key("provider")


def _resolve_fal_model() -> tuple:
    """Resolve the active FAL model from config.yaml (primary) or default.

    Returns (model_id, metadata_dict). Falls back to DEFAULT_MODEL if the
    configured model is unknown (logged as a warning).
    """
    # FAL_IMAGE_MODEL is an undocumented escape hatch (backward-compat for tests/scripts).
    model_id = _read_image_gen_key("model") or os.getenv("FAL_IMAGE_MODEL", "").strip()

    if not model_id:
        return DEFAULT_MODEL, FAL_MODELS[DEFAULT_MODEL]

    if model_id not in FAL_MODELS:
        logger.warning(
            "Unknown FAL model '%s' in config; falling back to %s",
            model_id, DEFAULT_MODEL,
        )
        return DEFAULT_MODEL, FAL_MODELS[DEFAULT_MODEL]

    return model_id, FAL_MODELS[model_id]


def _build_payload(
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    seed: Optional[int],
    overrides: Optional[Dict[str, Any]],
    image_urls: Optional[list] = None,
) -> Dict[str, Any]:
    """Shared text-to-image / edit payload builder (``image_urls`` selects edit mode).

    Translates aspect_ratio into the model's native size spec, merges model
    defaults, applies caller overrides, then filters to the model's whitelist.
    Edit endpoints mostly auto-infer output size from the input image, so the
    size key is only sent when ``edit_supports`` advertises it. ``prompt`` (and
    ``image_urls`` on edits) are required by every FAL endpoint and are kept
    even if a whitelist omits them, so a catalog gap can't send a broken request.
    """
    meta = FAL_MODELS[model_id]
    edit = image_urls is not None
    supports = (meta.get("edit_supports") or set()) if edit else meta["supports"]
    size_style = meta["size_style"]
    sizes = meta["sizes"]

    aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
    if aspect not in sizes:
        aspect = DEFAULT_ASPECT_RATIO

    payload: Dict[str, Any] = dict(meta.get("defaults", {}))
    payload["prompt"] = (prompt or "").strip()
    required = {"prompt"}
    if edit:
        payload["image_urls"] = list(image_urls)
        required.add("image_urls")

    if size_style in {"image_size_preset", "gpt_literal"}:
        size_key = "image_size"
    elif size_style == "aspect_ratio":
        size_key = "aspect_ratio"
    elif edit:
        size_key = None
    else:
        raise ValueError(f"Unknown size_style: {size_style!r}")
    if size_key is not None and (not edit or size_key in supports):
        payload[size_key] = sizes[aspect]

    if seed is not None and isinstance(seed, int):
        payload["seed"] = seed

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                payload[k] = v

    return {k: v for k, v in payload.items() if k in supports or k in required}


def _build_fal_payload(
    model_id: str,
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a FAL text-to-image payload for `model_id` from unified inputs."""
    return _build_payload(model_id, prompt, aspect_ratio, seed, overrides)


def _build_fal_edit_payload(
    model_id: str,
    prompt: str,
    image_urls: list,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a FAL *edit* (image-to-image) payload: ``image_urls`` + prompt, filtered to ``edit_supports``."""
    return _build_payload(model_id, prompt, aspect_ratio, seed, overrides, image_urls=image_urls)


# ---------------------------------------------------------------------------
# Upscaler
# ---------------------------------------------------------------------------
def _upscale_image(image_url: str, original_prompt: str) -> Optional[Dict[str, Any]]:
    """Upscale via FAL's Clarity Upscaler; None on failure (caller keeps the original)."""
    try:
        logger.info("Upscaling image with Clarity Upscaler...")

        upscaler_arguments = {
            "image_url": image_url,
            "prompt": f"{UPSCALER_DEFAULT_PROMPT}, {original_prompt}",
            "upscale_factor": UPSCALER_FACTOR,
            "negative_prompt": UPSCALER_NEGATIVE_PROMPT,
            "creativity": UPSCALER_CREATIVITY,
            "resemblance": UPSCALER_RESEMBLANCE,
            "guidance_scale": UPSCALER_GUIDANCE_SCALE,
            "num_inference_steps": UPSCALER_NUM_INFERENCE_STEPS,
            "enable_safety_checker": UPSCALER_SAFETY_CHECKER,
        }

        handler = _submit_fal_request(UPSCALER_MODEL, arguments=upscaler_arguments)
        result = _wait_fal_result(handler)

        if result and "image" in result:
            upscaled_image = result["image"]
            logger.info(
                "Image upscaled successfully to %sx%s",
                upscaled_image.get("width", "unknown"),
                upscaled_image.get("height", "unknown"),
            )
            return {
                "url": upscaled_image["url"],
                "width": upscaled_image.get("width", 0),
                "height": upscaled_image.get("height", 0),
                "upscaled": True,
                "upscale_factor": UPSCALER_FACTOR,
            }
        logger.error("Upscaler returned invalid response")
        return None

    except ImageGenerationInterrupted:
        # A user interrupt must not degrade into a silent "use original" fallback.
        raise
    except Exception as e:
        logger.error("Error upscaling image: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Artifact path hinting for non-local terminal backends
# ---------------------------------------------------------------------------
def _looks_like_absolute_file_path(value: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    lower = value.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return False
    if os.path.isabs(value):
        return True
    return len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}


def _active_terminal_env(task_id: str | None):
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id or "default")
    except Exception as exc:  # noqa: BLE001 - artifact hinting must not break generation
        logger.debug("Could not inspect active terminal environment: %s", exc)
        return None


def _agent_cache_base_for_env(env: Any) -> str | None:
    if env is not None:
        # Optional extension hook: an environment may expose its own agent-visible
        # cache root. No backend defines it yet; the guards make it a safe no-op.
        explicit = getattr(env, "agent_visible_cache_base", None)
        if callable(explicit):
            try:
                value = explicit()
                if value:
                    return str(value).rstrip("/")
            except Exception as exc:  # noqa: BLE001
                logger.debug("active env agent_visible_cache_base failed: %s", exc)

        remote_home = getattr(env, "_remote_home", None)
        if remote_home:
            return f"{str(remote_home).rstrip('/')}/.hermes"

        env_name = env.__class__.__name__
        if env_name in {"DockerEnvironment", "SingularityEnvironment", "ModalEnvironment"}:
            return "/root/.hermes"

    # No environment yet: only backends with deterministic cache roots can be
    # translated without side effects. SSH can use a shell-visible tilde path;
    # its first environment sync uploads the cache file before the first command.
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if backend in {"docker", "singularity", "modal"}:
        return "/root/.hermes"
    if backend == "ssh":
        return "~/.hermes"
    return None


def _agent_visible_cache_path(host_path: str, env: Any) -> str | None:
    if not _looks_like_absolute_file_path(host_path):
        return None

    cache_base = _agent_cache_base_for_env(env)
    if not cache_base:
        return None

    try:
        from tools.credential_files import map_cache_path_to_container

        return map_cache_path_to_container(host_path, container_base=cache_base)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not translate image cache path for backend: %s", exc)
    return None


def _force_artifact_sync(env: Any) -> None:
    sync_manager = getattr(env, "_sync_manager", None)
    if sync_manager is None:
        return
    try:
        sync_manager.sync(force=True)
    except Exception as exc:  # noqa: BLE001 - keep generation success; log for operators
        logger.warning("Could not force-sync generated image artifact: %s", exc)


def _postprocess_image_generate_result(raw: str, task_id: str | None = None) -> str:
    """Annotate successful local image results with backend-visible paths.

    ``image`` stays the host/gateway-deliverable path; when the active terminal
    backend has a different filesystem, ``agent_visible_image`` is the path the
    agent can use with terminal/file tools.
    """
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return raw

    if not isinstance(payload, dict) or not payload.get("success"):
        return raw

    image = payload.get("image")
    if not isinstance(image, str) or not _looks_like_absolute_file_path(image):
        return raw

    env = _active_terminal_env(task_id)
    agent_path = _agent_visible_cache_path(image, env)
    if not agent_path or agent_path == image:
        return raw

    if env is not None:
        _force_artifact_sync(env)

    payload.setdefault("host_image", image)
    payload.setdefault("agent_visible_image", agent_path)
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------
def image_generate_tool(
    prompt: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_images: Optional[int] = None,
    output_format: Optional[str] = None,
    seed: Optional[int] = None,
    image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None,
    upscale: Optional[bool] = None,
) -> str:
    """Generate an image from a text prompt, or edit a source image, via FAL.

    Routing: ``image_url`` / ``reference_image_urls`` plus a model with an
    ``edit_endpoint`` → image-to-image; otherwise text-to-image. The extra
    kwargs are overrides for direct Python callers, filtered per-model via the
    ``supports`` / ``edit_supports`` whitelist (unsupported ones are dropped
    silently so legacy callers survive model switches).

    Returns a JSON string with ``{"success": bool, "image": url | None,
    "modality": "text" | "image", "error": str, "error_type": str}``.
    """
    model_id, meta = _resolve_fal_model()

    # Collect any source images (primary + references) into one ordered list.
    source_images: list = []
    if isinstance(image_url, str) and image_url.strip():
        source_images.append(image_url.strip())
    if isinstance(reference_image_urls, (list, tuple)):
        for ref in reference_image_urls:
            if isinstance(ref, str) and ref.strip():
                source_images.append(ref.strip())

    edit_endpoint = meta.get("edit_endpoint")
    use_edit = bool(source_images) and bool(edit_endpoint)
    modality = "image" if use_edit else "text"

    debug_call_data = {
        "model": model_id,
        "parameters": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "seed": seed,
            "modality": modality,
            "source_images": len(source_images),
        },
        "error": None,
        "success": False,
        "images_generated": 0,
        "generation_time": 0,
    }

    start_time = datetime.datetime.now()

    try:
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt is required and must be a non-empty string")

        # A stored-but-broken selection raises the selection-naming error from
        # _resolve_managed_fal_gateway(); only the never-configured path can
        # report "no backend at all".
        if not (fal_key_is_configured() or _resolve_managed_fal_gateway()):
            raise ValueError(_build_no_backend_setup_message())

        # Source images on a model without an edit endpoint: fail clearly rather
        # than silently dropping them and producing an unrelated picture.
        if source_images and not edit_endpoint:
            raise ValueError(
                f"Model '{meta.get('display', model_id)}' ({model_id}) is not "
                f"capable of image-to-image / editing. Provide a text-only "
                f"prompt (omit image_url), or switch to an edit-capable model "
                f"via `hermes tools` → Image Generation."
            )

        aspect_lc = (aspect_ratio or DEFAULT_ASPECT_RATIO).lower().strip()
        if aspect_lc not in VALID_ASPECT_RATIOS:
            logger.warning(
                "Invalid aspect_ratio '%s', defaulting to '%s'",
                aspect_ratio, DEFAULT_ASPECT_RATIO,
            )
            aspect_lc = DEFAULT_ASPECT_RATIO

        overrides: Dict[str, Any] = {
            k: v for k, v in (
                ("num_inference_steps", num_inference_steps),
                ("guidance_scale", guidance_scale),
                ("num_images", num_images),
                ("output_format", output_format),
            ) if v is not None
        }

        if use_edit:
            # Clamp reference count to the model's declared cap.
            max_refs = int(meta.get("max_reference_images") or 1)
            clamped_sources = source_images[:max_refs] if max_refs > 0 else source_images
            arguments = _build_fal_edit_payload(
                model_id, prompt, clamped_sources, aspect_lc,
                seed=seed, overrides=overrides,
            )
            endpoint = edit_endpoint
            logger.info(
                "Editing image with %s (%s) — %d source image(s), prompt: %s",
                meta.get("display", model_id), endpoint, len(clamped_sources),
                prompt[:80],
            )
        else:
            arguments = _build_fal_payload(
                model_id, prompt, aspect_lc, seed=seed, overrides=overrides,
            )
            endpoint = model_id
            logger.info(
                "Generating image with %s (%s) — prompt: %s",
                meta.get("display", model_id), model_id, prompt[:80],
            )

        handler = _submit_fal_request(endpoint, arguments=arguments)
        result = _wait_fal_result(handler)

        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        if not result or "images" not in result:
            raise ValueError("Invalid response from FAL.ai API — no images returned")

        images = result.get("images", [])
        if not images:
            raise ValueError("No images were generated")

        # An explicit ``upscale`` wins over the catalog default, including for
        # edits (an explicit request is intentional). The catalog default never
        # upscales edits: Clarity is a text-to-image quality pass and must not
        # silently alter edit compositions.
        if upscale is not None:
            should_upscale = bool(upscale)
        else:
            should_upscale = bool(meta.get("upscale", False)) and not use_edit

        formatted_images = []
        for img in images:
            if not (isinstance(img, dict) and "url" in img):
                continue
            original_image = {
                "url": img["url"],
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }

            if should_upscale:
                upscaled_image = _upscale_image(img["url"], prompt.strip())
                if upscaled_image:
                    formatted_images.append(upscaled_image)
                    continue
                logger.warning("Using original image as fallback (upscale failed)")

            original_image["upscaled"] = False
            formatted_images.append(original_image)

        if not formatted_images:
            raise ValueError("No valid image URLs returned from API")

        upscaled_count = sum(1 for img in formatted_images if img.get("upscaled"))
        logger.info(
            "Generated %s image(s) in %.1fs (%s upscaled) via %s [%s]",
            len(formatted_images), generation_time, upscaled_count, endpoint,
            modality,
        )

        response_data = {
            "success": True,
            "image": formatted_images[0]["url"] if formatted_images else None,
            "modality": modality,
            "upscaled": bool(formatted_images and formatted_images[0].get("upscaled")),
        }

        debug_call_data["success"] = True
        debug_call_data["images_generated"] = len(formatted_images)
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)

    except Exception as e:
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error generating image: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        response_data = {
            "success": False,
            "image": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }

        debug_call_data["error"] = error_msg
        debug_call_data["generation_time"] = generation_time
        _debug.log_call("image_generate_tool", debug_call_data)
        _debug.save()

        return json.dumps(response_data, indent=2, ensure_ascii=False)


def check_fal_api_key() -> bool:
    """True if the FAL backend selected via `hermes tools` (or, never configured, any FAL backend) is available.

    A stored-but-broken selection reports False here (registry gating); the
    selection-naming error surfaces at call time from ``_resolve_managed_fal_gateway``.
    """
    selected = read_selection("image_gen")
    if selected == NOUS_MANAGED_PROVIDER:
        return bool(resolve_managed_tool_gateway("fal-queue"))
    if selected is not None:
        return fal_key_is_configured()
    return bool(fal_key_is_configured() or resolve_managed_tool_gateway("fal-queue"))


def _build_no_backend_setup_message() -> str:
    """Actionable error when no FAL backend is reachable: FAL_KEY signup,
    managed-gateway status (if Nous tools enabled), and the plugin alternative."""
    lines = ["Image generation is unavailable in this environment.", ""]
    lines.append("Missing requirements:")
    if managed_nous_tools_enabled():
        lines.append(
            "  - FAL_KEY is not set and the managed FAL gateway is unreachable"
        )
    else:
        lines.append("  - FAL_KEY environment variable is not set")
        gateway_message = nous_tool_gateway_unavailable_message(
            "managed FAL image generation",
        )
        if gateway_message:
            lines.append(f"  - {gateway_message}")
    lines.append("")
    lines.append("To enable image generation, do one of:")
    lines.append(
        "  1. Get a free API key at https://fal.ai and set "
        "FAL_KEY=<your-key> (then restart the session)"
    )
    if managed_nous_tools_enabled():
        lines.append(
            "  2. Sign in to a Nous account that has the managed FAL "
            "gateway enabled (`hermes setup`)"
        )
    lines.append(
        "  3. Configure a different image_gen provider via `hermes tools` "
        "→ Image Generation (run `hermes plugins list` to see installed "
        "backends)"
    )
    return "\n".join(lines)


def _get_plugin_provider(name: str):
    """Discover plugins (import is local so importing this module never triggers discovery) and return the named provider."""
    from agent.image_gen_registry import get_provider
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()
    return get_provider(name)


def check_image_generation_requirements() -> bool:
    """True if FAL or the explicitly configured image backend is available."""
    try:
        if check_fal_api_key():
            # The lazy import doubles as the SDK presence check: ImportError
            # when ``fal-client`` isn't installed falls through to plugin probing.
            _load_fal_client()
            return True
    except ImportError:
        pass

    configured = _read_configured_image_provider()
    if not configured or configured in ("fal", NOUS_MANAGED_PROVIDER):
        return False

    # Probe only the explicitly selected plugin. Merely possessing a cloud
    # provider key must not opt a user into a paid image-generation backend.
    try:
        provider = _get_plugin_provider(configured)
        return bool(provider and provider.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    # Placeholder — description AND params are rebuilt dynamically at
    # get_tool_definitions() time from the active backend's declared
    # capabilities (FAL catalog metadata, or plugin provider.capabilities()).
    # Edit-only args (image_url, reference_image_urls) and upscale are
    # advertised ONLY when the active model actually supports them; the
    # handler accepts them regardless (replay compat + teaching errors).
    # See _build_dynamic_image_schema().
    "description": (
        "Generate images from text prompts. The active model's edit/reference "
        "capabilities are rendered at serving time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The text prompt describing the desired image (text-to-"
                    "image) or the edit to apply (image-to-image). Be detailed "
                    "and descriptive."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            # image_url / reference_image_urls / upscale are added per-capability
            # by _build_dynamic_image_schema. Do not re-add them statically.
        },
        "required": ["prompt"],
    },
}


# ---------------------------------------------------------------------------
# Plugin provider dispatch + managed-mode Krea routing
# ---------------------------------------------------------------------------
def _provider_error(error: str, error_type: str) -> str:
    """JSON error envelope shared by every provider-dispatch failure path."""
    return json.dumps({
        "success": False,
        "image": None,
        "error": error,
        "error_type": error_type,
    })


def _add_provider_kwargs(
    kwargs: Dict[str, Any],
    image_url: Optional[str],
    reference_image_urls: Optional[list],
    upscale: Optional[bool],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Add the optional ``provider.generate(**kwargs)`` args in place (edit args only when supplied)."""
    if model:
        kwargs["model"] = model
    if isinstance(image_url, str) and image_url.strip():
        kwargs["image_url"] = image_url.strip()
    norm_refs = None
    if reference_image_urls is not None:
        from agent.image_gen_provider import normalize_reference_images

        norm_refs = normalize_reference_images(reference_image_urls)
    if norm_refs:
        kwargs["reference_image_urls"] = norm_refs
    if upscale is not None:
        kwargs["upscale"] = bool(upscale)
    return kwargs


def _dispatch_to_plugin_provider(
    prompt: str,
    aspect_ratio: str,
    image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None,
    upscale: Optional[bool] = None,
):
    """Route the call to a plugin-registered provider when one is selected.

    Returns a JSON string on dispatch, or ``None`` to fall through to the
    in-tree FAL pipeline. Fires when ``image_gen.provider`` is set to anything
    other than unset / ``"fal"`` / ``"nous"`` — those run the legacy pipeline
    (``"nous"`` routes it through the managed fal-queue gateway).

    ``image_url`` / ``reference_image_urls`` are forwarded so the backend can
    route to its edit endpoint; ``upscale`` requests a post-generation
    high-res pass (providers without it ignore it via ``**kwargs``).
    """
    configured = _read_configured_image_provider()
    if not configured or configured in ("fal", NOUS_MANAGED_PROVIDER):
        return None

    configured_model = _read_configured_image_model()

    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        provider = _get_plugin_provider(configured)
    except Exception as exc:
        logger.debug("image_gen plugin dispatch skipped: %s", exc)
        return None

    if provider is None:
        try:
            # Long-lived sessions may have discovered plugins before a bundled
            # backend was patched in or config changed: retry once with a forced
            # refresh before surfacing a missing-provider error.
            from agent.image_gen_registry import get_provider

            _ensure_plugins_discovered(force=True)
            provider = get_provider(configured)
        except Exception as exc:
            logger.debug("image_gen plugin force-refresh skipped: %s", exc)

    if provider is None:
        return _provider_error(
            f"image_gen.provider='{configured}' is set but no plugin "
            f"registered that name. Run `hermes plugins list` to see "
            f"available image gen backends.",
            "provider_not_registered",
        )

    pname = getattr(provider, "name", "?")
    kwargs: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    try:
        _add_provider_kwargs(
            kwargs, image_url, reference_image_urls, upscale, model=configured_model,
        )
        result = provider.generate(**kwargs)
    except TypeError as exc:
        # A provider whose generate() predates image_url support (third-party
        # plugin not yet updated): text-to-image keeps working, but surface a
        # clear note when the user actually asked for an edit.
        if "image_url" in kwargs or "reference_image_urls" in kwargs:
            logger.warning(
                "image_gen provider '%s' rejected image-to-image kwargs "
                "(signature too narrow): %s",
                pname, exc,
            )
            return _provider_error(
                f"Provider '{pname}' does not "
                f"support image-to-image / editing (its generate() "
                f"signature is out of date with the image_generate schema). "
                f"Omit image_url for text-to-image, or pick a backend that "
                f"supports editing via `hermes tools` → Image Generation.",
                "modality_unsupported",
            )
        logger.warning("Image gen provider '%s' raised TypeError: %s", pname, exc)
        return _provider_error(f"Provider '{pname}' error: {exc}", "provider_exception")
    except Exception as exc:
        logger.warning("Image gen provider '%s' raised: %s", pname, exc)
        return _provider_error(f"Provider '{pname}' error: {exc}", "provider_exception")
    if not isinstance(result, dict):
        return _provider_error("Provider returned a non-dict result", "provider_contract")
    return json.dumps(result)


# Native ``krea-2-*`` plugin model ids are served by the dedicated Krea managed
# gateway; ``fal-ai/krea/v2/*`` catalog ids stay on the FAL path. Routing only
# fires in managed mode — direct/BYO users keep their unchanged pipeline.
_KREA_NATIVE_MODELS = {"krea-2-medium", "krea-2-large", "krea-2-medium-turbo"}


def _normalize_krea_model(model_id: Optional[str]) -> Optional[str]:
    """Return the native Krea plugin model id when ``model_id`` is ``krea-2-*``."""
    if not isinstance(model_id, str):
        return None
    candidate = model_id.strip()
    if candidate in _KREA_NATIVE_MODELS:
        return candidate
    return None


def _maybe_route_managed_krea(
    prompt: str,
    aspect_ratio: str,
    image_url: Optional[str] = None,
    reference_image_urls: Optional[list] = None,
    upscale: Optional[bool] = None,
) -> Optional[str]:
    """Route a native ``krea-2-*`` model to the managed Krea gateway, in managed mode.

    Returns a JSON result string when handled, or ``None`` to fall through to
    the normal plugin/FAL pipeline. Fires only when the configured model is a
    native ``krea-2-*`` id AND no explicit ``image_gen.provider`` other than the
    managed ``"nous"`` selection is stored (a picker choice dispatches normally)
    AND the managed Krea gateway is resolvable.
    """
    configured_provider = _read_configured_image_provider()
    if configured_provider is not None and configured_provider != NOUS_MANAGED_PROVIDER:
        return None

    normalized = _normalize_krea_model(_read_configured_image_model())
    if normalized is None:
        return None

    try:
        from plugins.image_gen.krea import _resolve_managed_krea_gateway

        if _resolve_managed_krea_gateway() is None:
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea routing probe failed: %s", exc)
        return None

    try:
        provider = _get_plugin_provider("krea")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Managed Krea routing: provider unavailable: %s", exc)
        return None
    if provider is None:
        return None

    kwargs: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio, "model": normalized}
    try:
        _add_provider_kwargs(kwargs, image_url, reference_image_urls, upscale)
        result = provider.generate(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Managed Krea routing failed: %s", exc)
        return _provider_error(f"Managed Krea generation error: {exc}", "provider_exception")
    if not isinstance(result, dict):
        return _provider_error("Krea provider returned a non-dict result", "provider_contract")
    return json.dumps(result)


def _confine_source_images(
    image_url, reference_image_urls, task_id, *, permitted: tuple = ("image",)
):
    """Route path-like source images through the sandbox-aware resolver.

    Under a non-local terminal backend (ssh/docker/…), model-supplied local
    paths resolve via ``tools.image_source`` (in-sandbox exec-read, media-cache
    host reads, credential guard) into ``data:`` URLs before any provider sees
    them, so generation obeys the same confinement boundary as vision/video
    analysis and sandbox-only files work as edit sources. URLs and data: URLs
    pass through; the local backend is a no-op (providers keep host reads).

    Returns ``(image_url, reference_image_urls, error_json_or_None)``.
    """
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if backend in ("", "local"):
        return image_url, reference_image_urls, None

    from model_tools import _run_async
    from tools.image_source import ImageResolutionError, resolve_local_source_to_data_url

    try:
        if isinstance(image_url, str) and image_url.strip():
            image_url = _run_async(resolve_local_source_to_data_url(
                image_url, task_id, permitted=permitted))
        if isinstance(reference_image_urls, (list, tuple)):
            reference_image_urls = [
                _run_async(resolve_local_source_to_data_url(ref, task_id, permitted=permitted))
                if isinstance(ref, str) else ref
                for ref in list(reference_image_urls)
            ]
    except ImageResolutionError as exc:
        return image_url, reference_image_urls, _provider_error(
            f"Could not read source image: {exc}", type(exc).__name__,
        )
    return image_url, reference_image_urls, None


def _handle_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    aspect_ratio = args.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    image_url = args.get("image_url")
    reference_image_urls = args.get("reference_image_urls")
    upscale = args.get("upscale")
    if not isinstance(upscale, bool):
        upscale = None
    task_id = kw.get("task_id")

    # Confinement chokepoint: path-like sources become data: URLs BEFORE any
    # dispatch, so plugin, managed Krea and in-tree FAL all get sandbox-confined bytes.
    image_url, reference_image_urls, confine_error = _confine_source_images(
        image_url, reference_image_urls, task_id)
    if confine_error is not None:
        return confine_error

    # Order matters: explicit plugin provider (incl. provider == "krea"), then
    # model-driven managed Krea interception (only when no provider is set, so
    # the BYO/direct FAL path stays untouched), then the in-tree FAL pipeline.
    raw = _dispatch_to_plugin_provider(
        prompt, aspect_ratio,
        image_url=image_url,
        reference_image_urls=reference_image_urls,
        upscale=upscale,
    )
    if raw is None:
        raw = _maybe_route_managed_krea(
            prompt, aspect_ratio,
            image_url=image_url,
            reference_image_urls=reference_image_urls,
            upscale=upscale,
        )
    if raw is None:
        raw = image_generate_tool(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_url=image_url,
            reference_image_urls=reference_image_urls,
            upscale=upscale,
        )
    return _postprocess_image_generate_result(raw, task_id=task_id)


# ---------------------------------------------------------------------------
# Dynamic schema — reflect the active backend's image-to-image capability
# ---------------------------------------------------------------------------
# Whether the active model can edit depends on the configured backend + model;
# telling the model up front saves a wasted turn. Memoized by config.yaml mtime
# in model_tools.get_tool_definitions(), so it rebuilds on provider/model switch.


def _active_image_capabilities() -> Dict[str, Any]:
    """Best-effort capabilities of the active backend/model; never raises.

    Resolution mirrors runtime dispatch: a set ``image_gen.provider`` asks that
    plugin, otherwise the in-tree FAL catalog. Fail-closed on every axis: an
    undeclared capability is advertised as absent (an under-declaring provider
    is that provider's bug, not a safety problem).
    """
    info: Dict[str, Any] = {
        "modalities": ["text"],
        "max_reference_images": 0,
        "supports_upscale": False,
    }

    configured_provider = _read_configured_image_provider()
    if configured_provider and configured_provider != "fal":
        try:
            provider = _get_plugin_provider(configured_provider)
            if provider is not None:
                caps = {}
                try:
                    caps = provider.capabilities() or {}
                except Exception:  # noqa: BLE001
                    caps = {}
                info["provider"] = provider.display_name
                info["model"] = _read_configured_image_model() or (provider.default_model() or "")
                if caps.get("modalities"):
                    info["modalities"] = list(caps["modalities"])
                if caps.get("max_reference_images"):
                    info["max_reference_images"] = int(caps["max_reference_images"])
                # Plugins opt in explicitly; absent = no upscale param.
                info["supports_upscale"] = bool(caps.get("supports_upscale"))
                return info
        except Exception:  # noqa: BLE001
            pass

    # In-tree FAL path (provider unset or == "fal").
    try:
        model_id, meta = _resolve_fal_model()
        info["provider"] = "FAL.ai"
        info["model"] = meta.get("display", model_id)
        if meta.get("edit_endpoint"):
            info["modalities"] = ["text", "image"]
            info["max_reference_images"] = int(meta.get("max_reference_images") or 1)
        else:
            info["modalities"] = ["text"]
            info["max_reference_images"] = 0
        # FAL: Clarity is a separate endpoint chained on explicit request for ANY
        # catalog model (the per-model ``upscale`` key is only the default flag).
        info["supports_upscale"] = True
    except Exception:  # noqa: BLE001
        pass

    return info


# Param snippets assembled per-capability by _build_dynamic_image_schema.
_IMAGE_URL_PARAM = {
    "type": "string",
    "description": (
        "Source image to edit/transform (image-to-image). A public URL or "
        "an absolute local file path from the conversation. Omit for "
        "text-to-image."
    ),
}

_UPSCALE_PARAM = {
    "type": "boolean",
    "description": (
        "Post-generation high-resolution pass (~2x, extra cost/latency), "
        "off by default. A creative enhancer that can alter fine detail "
        "(rendered text, faces) — use only when resolution matters more "
        "than fidelity."
    ),
}


def _build_dynamic_image_schema() -> Dict[str, Any]:
    """Render description AND params from the active model's capabilities.

    Args a model cannot honor are NOT advertised — the handler still accepts
    them (replay compat) and answers with a capability error.
    """
    base_desc = (
        "Generate high-quality images from text prompts{edit_clause}. "
        "Returns the result in the `image` field — a URL or an absolute "
        "file path; reference it in your response using the current "
        "platform's file-delivery convention."
    )

    try:
        info = _active_image_capabilities()
    except Exception:  # noqa: BLE001
        info = {"modalities": ["text"], "max_reference_images": 0,
                "supports_upscale": False}

    modalities = set(info.get("modalities") or ["text"])
    max_refs = int(info.get("max_reference_images") or 0)
    can_edit = "image" in modalities

    properties: Dict[str, Any] = {
        "prompt": IMAGE_GENERATE_SCHEMA["parameters"]["properties"]["prompt"],
        "aspect_ratio": IMAGE_GENERATE_SCHEMA["parameters"]["properties"]["aspect_ratio"],
    }

    if can_edit:
        edit_clause = (
            ", or edit / transform an existing image by passing image_url"
        )
        properties["image_url"] = _IMAGE_URL_PARAM
        if max_refs > 1:
            properties["reference_image_urls"] = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_refs,
                "description": (
                    f"Up to {max_refs} additional reference images (style, "
                    "character, or composition) guiding an edit. URLs or "
                    "absolute local paths."
                ),
            }
    else:
        edit_clause = (
            " (text-to-image only — the active model cannot edit existing "
            "images)"
        )

    if info.get("supports_upscale"):
        properties["upscale"] = _UPSCALE_PARAM

    description = base_desc.format(edit_clause=edit_clause)

    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["prompt"],
        },
    }


registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_generation_requirements,
    requires_env=[],
    is_async=False,   # sync fal_client API to avoid "Event loop is closed" in gateway
    emoji="🎨",
    dynamic_schema_overrides=_build_dynamic_image_schema,
)
