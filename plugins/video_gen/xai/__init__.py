"""xAI Grok-Imagine video generation backend.

Surface: text-, image- and reference-to-video through the unified video provider; xAI
edit/extend are exposed by ``tools.xai_video_tools`` via ``run_xai_video_edit`` / ``run_xai_video_extend``.

Authentication: xAI Grok OAuth tokens (preferred — billed to the user's SuperGrok / X Premium+
subscription) or ``XAI_API_KEY``, both via ``tools.xai_http.resolve_xai_http_credentials`` so one
login covers chat + TTS + image gen + video gen + transcription. When xAI storage is enabled, the
primary ``video`` / ``public_url`` fields are the stored files-cdn HTTPS link; pass that public MP4
URL as ``video_url`` for edit/extend (sent to xAI as ``video.url``).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx

from agent.video_gen_provider import VideoGenProvider, error_response, success_response

logger = logging.getLogger(__name__)


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TEXT_TO_VIDEO_MODEL = "grok-imagine-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "grok-imagine-video-1.5"
DEFAULT_MODEL = DEFAULT_TEXT_TO_VIDEO_MODEL
DEFAULT_DURATION = 8
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "720p"
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_EXTEND_DURATION = 6

VALID_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"}
VALID_RESOLUTIONS = {"480p", "720p"}
MAX_REFERENCE_IMAGES = 7

_REMOTE_PREFIXES = ("http://", "https://")
_TERMINAL_POLL_STATUSES = {"done", "failed", "error", "expired", "cancelled"}

_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-imagine-video": {
        "display": "Grok Imagine Video", "speed": "~60-240s", "strengths": "Text-to-video; legacy image-to-video fallback.",
        "price": "see https://docs.x.ai/developers/models/grok-imagine-video", "modalities": ["text", "image"],
    },
    "grok-imagine-video-1.5": {
        "display": "Grok Imagine Video 1.5", "speed": "~60-240s", "strengths": "Latest xAI image-to-video model.",
        "price": "see https://docs.x.ai/developers/pricing", "modalities": ["image"],
    },
}

_IMAGE_TO_VIDEO_COMPAT_MODEL_IDS = {"grok-imagine-video-1.5-preview", "grok-imagine-video-1.5-2026-05-30"}

_AUTH_REQUIRED_MSG = (
    "No xAI credentials found. Sign in via `hermes auth add xai-oauth` "
    "(SuperGrok / Premium+) or set XAI_API_KEY from https://console.x.ai/."
)
_PUBLIC_URL_HINT = "(e.g. the `image`/`public_url` from a prior Imagine result)"


# ---- Credentials / HTTP helpers -------------------------------------------


def _resolve_xai_credentials() -> Tuple[str, str]:
    """Return ``(api_key, base_url)`` from the shared xAI credential resolver.

    Order: runtime provider (xai-oauth pool entry) → singleton ``auth.json`` OAuth tokens →
    ``XAI_API_KEY`` env var. ``api_key`` is empty when no source is available; callers must check.
    """
    try:
        from tools.xai_http import resolve_xai_http_credentials

        creds = resolve_xai_http_credentials() or {}
    except Exception as exc:
        logger.debug("xAI credential resolver failed: %s", exc)
        creds = {}
    base_url = str(creds.get("base_url") or os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL)
    return str(creds.get("api_key") or os.getenv("XAI_API_KEY", "")).strip(), base_url.strip().rstrip("/")


def _xai_headers(api_key: str) -> Dict[str, str]:
    try:
        from tools.xai_http import hermes_xai_user_agent

        user_agent = hermes_xai_user_agent()
    except Exception:
        user_agent = "hermes-agent/video_gen"
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": user_agent}


def _xai_error(error: str, error_type: str, prompt: str, model: str = "", aspect_ratio: str = "") -> Dict[str, Any]:
    return error_response(error=error, error_type=error_type, provider="xai", model=model, prompt=prompt, aspect_ratio=aspect_ratio)


# ---- Input normalization --------------------------------------------------


def _media_ref_to_xai_url(value: str, *, kind: str, fallback_mime: str) -> str:
    """Return a URL/data URI accepted by xAI for ``kind`` (``image``/``video``) inputs.

    Remote URLs and matching data URIs pass through; a readable local file of the right MIME
    class is inlined as base64; anything else is returned as-is so the caller's URL check rejects
    it clearly. Local reads go through Hermes' read deny-list (same credential-store guard as the
    image providers), which fails open if its machinery is unavailable.
    """
    ref = (value or "").strip()
    if not ref or ref.lower().startswith(_REMOTE_PREFIXES + (f"data:{kind}/",)):
        return ref
    path = Path(ref).expanduser()
    if not path.is_file():
        return ref
    try:
        from agent.file_safety import raise_if_read_blocked
    except Exception as exc:  # noqa: BLE001 - guard must never break loading
        logger.debug("xAI media input read guard unavailable: %s", exc)
    else:
        raise_if_read_blocked(ref)
    mime = mimetypes.guess_type(path.name)[0] or fallback_mime
    if not mime.startswith(f"{kind}/"):
        return ref
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _image_ref_to_xai_input(value: str) -> Optional[Dict[str, str]]:
    ref = _media_ref_to_xai_url(value, kind="image", fallback_mime="application/octet-stream")
    return {"url": ref} if ref and ref.lower().startswith(_REMOTE_PREFIXES + ("data:image/",)) else None


async def _video_input_from_public_url(value: str, *, api_key: str, base_url: str) -> Optional[Dict[str, str]]:
    """Build xAI ``video`` input using a public HTTPS URL (``url`` field only)."""
    ref = (value or "").strip()
    if ref and Path(ref).expanduser().is_file():
        ref = _media_ref_to_xai_url(ref, kind="video", fallback_mime="video/mp4")
        return {"url": ref} if ref else None
    return {"url": ref} if ref.lower().startswith(_REMOTE_PREFIXES) else None


def _clamp_duration(duration: Optional[int], *, has_reference_images: bool = False, max_seconds: int = 15,
                    default: int = DEFAULT_DURATION) -> int:
    """Clamp to ``[1, max_seconds]``; reference-to-video additionally caps at 10s."""
    value = max(1, min(max_seconds, duration if duration is not None else default))
    return min(value, 10) if has_reference_images else value


def _resolve_model_for_modality(model: Optional[str], *, modality: str, explicit_model: bool) -> str:
    """Select xAI's text/video model without treating config as a prompt override.

    ``grok-imagine-video-1.5`` rejects text-only generation but is the desired image-to-video
    backend. Explicit tool ``model=`` still wins for users who intentionally request another model.
    """
    requested = (model or "").strip()
    if explicit_model and requested:
        return requested
    if modality == "image":
        return DEFAULT_IMAGE_TO_VIDEO_MODEL
    if requested == DEFAULT_IMAGE_TO_VIDEO_MODEL or requested in _IMAGE_TO_VIDEO_COMPAT_MODEL_IDS:
        return DEFAULT_TEXT_TO_VIDEO_MODEL
    return requested or DEFAULT_TEXT_TO_VIDEO_MODEL


# ---- Provider ---------------------------------------------------------------


class XAIVideoGenProvider(VideoGenProvider):
    """xAI Grok Imagine video backend."""

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI"

    def is_available(self) -> bool:
        return has_xai_video_credentials()

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution lives in the shared ``xai_grok`` post_setup hook (hermes_cli/tools_config.py) so the
        # picker doesn't prompt for an API key when already signed in via xAI Grok OAuth; the hook offers an
        # OAuth-vs-API-key choice when neither is configured.
        try:
            from tools.xai_http import xai_storage_notice_text

            storage_notice = xai_storage_notice_text("video_gen")
        except Exception:
            storage_notice = ""
        tag = (
            "grok-imagine-video for text/reference; grok-imagine-video-1.5 for image-to-video; edit/extend: pass "
            "the stored public HTTPS MP4 (`video` / `public_url` from a prior Imagine result); uses xAI Grok OAuth "
            "or XAI_API_KEY"
        )
        if storage_notice:
            tag += f". {storage_notice}"
        return {"name": "xAI Grok Imagine", "badge": "paid", "tag": tag, "env_vars": [], "post_setup": "xai_grok"}

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"], "aspect_ratios": sorted(VALID_ASPECT_RATIOS),
            "resolutions": sorted(VALID_RESOLUTIONS), "max_duration": 15, "min_duration": 1,
            "supports_audio": False, "supports_negative_prompt": False, "supports_seed": True,
            "supports_upscale": False, "max_reference_images": MAX_REFERENCE_IMAGES,
        }

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None, duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO, resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None, audio: Optional[bool] = None, seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _run_xai_video_coroutine(
            lambda api_key, base_url: _generate_xai_video_async(
                api_key=api_key, base_url=base_url, prompt=prompt, model=model,
                explicit_model=bool(kwargs.get("_model_override_explicit")), image_url=image_url,
                reference_image_urls=reference_image_urls, duration=duration, aspect_ratio=aspect_ratio,
                resolution=resolution,
            ),
            operation_label="generation", model=model, prompt=prompt, aspect_ratio=aspect_ratio,
        )


# ---- Sync entry points (provider + tools.xai_video_tools) -------------------


def has_xai_video_credentials() -> bool:
    return bool(_resolve_xai_credentials()[0])


def run_xai_video_edit(*, prompt: str, video_url: str, model: Optional[str] = None) -> Dict[str, Any]:
    return _run_xai_video_mutation(prompt, video_url, model, endpoint="edits", operation="edit", duration=DEFAULT_DURATION)


def run_xai_video_extend(*, prompt: str, video_url: str, duration: Optional[int] = None,
                         model: Optional[str] = None) -> Dict[str, Any]:
    return _run_xai_video_mutation(
        prompt, video_url, model, endpoint="extensions", operation="extend",
        duration=_clamp_duration(duration, max_seconds=10, default=DEFAULT_EXTEND_DURATION),
    )


def _run_xai_video_mutation(prompt: str, video_url: str, model: Optional[str], *, endpoint: str, operation: str,
                            duration: int) -> Dict[str, Any]:
    return _run_xai_video_coroutine(
        lambda api_key, base_url: _mutate_xai_video_async(
            api_key=api_key, base_url=base_url, prompt=prompt, video_url=video_url, model=model,
            endpoint=endpoint, operation=operation, duration=duration,
        ),
        operation_label=operation, model=model, prompt=prompt, aspect_ratio=DEFAULT_ASPECT_RATIO,
    )


def _run_xai_video_coroutine(
    start: Callable[[str, str], Coroutine[Any, Any, Dict[str, Any]]], *, operation_label: str,
    model: Optional[str], prompt: str, aspect_ratio: str,
) -> Dict[str, Any]:
    """Resolve credentials, then drive ``start(api_key, base_url)`` on a fresh event loop;
    any escaped exception → api_error response."""
    api_key, base_url = _resolve_xai_credentials()
    if not api_key:
        return _xai_error(_AUTH_REQUIRED_MSG, "auth_required", prompt)
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(start(api_key, base_url))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("xAI video %s unexpected failure: %s", operation_label, exc, exc_info=True)
        return _xai_error(
            f"xAI video {operation_label} failed: {exc}", "api_error", prompt,
            model=model or DEFAULT_MODEL, aspect_ratio=aspect_ratio,
        )


# ---- Async flows ------------------------------------------------------------


async def _generate_xai_video_async(
    *, api_key: str, base_url: str, prompt: str, model: Optional[str], explicit_model: bool, image_url: Optional[str],
    reference_image_urls: Optional[List[str]], duration: Optional[int], aspect_ratio: str, resolution: str,
) -> Dict[str, Any]:
    prompt = (prompt or "").strip()
    image_input = _image_ref_to_xai_input(image_url) if (image_url or "").strip() else None
    if (image_url or "").strip() and not image_input:
        return _xai_error(f"image_url must be a public HTTPS URL or data URI {_PUBLIC_URL_HINT}", "invalid_image_url", prompt)
    aspect_ratio = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip()
    resolution = (resolution or DEFAULT_RESOLUTION).strip().lower()
    refs = [_image_ref_to_xai_input(url.strip()) for url in reference_image_urls or [] if (url or "").strip()]
    if not all(refs):
        return _xai_error(
            f"reference_image_urls must be public HTTPS URLs or data URIs {_PUBLIC_URL_HINT}", "invalid_reference_image_urls", prompt,
        )
    if not prompt:
        return _xai_error("prompt is required for xAI video generation", "missing_prompt", prompt)
    if len(refs) > MAX_REFERENCE_IMAGES:
        return _xai_error(
            f"reference_image_urls supports at most {MAX_REFERENCE_IMAGES} images on xAI", "too_many_references", prompt,
        )
    if image_input and refs:
        return _xai_error("image_url and reference_image_urls cannot be combined on xAI", "conflicting_inputs", prompt)

    # Unsupported values silently fall back to defaults rather than erroring.
    aspect_ratio = aspect_ratio if aspect_ratio in VALID_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO
    resolution = resolution if resolution in VALID_RESOLUTIONS else DEFAULT_RESOLUTION

    modality_used = "reference" if refs else ("image" if image_input else "text")
    resolved_model = _resolve_model_for_modality(model, modality=modality_used, explicit_model=explicit_model)
    # Reference-to-video only exists on the text model: explicit other model = error, implicit (config) = corrected.
    if refs and resolved_model != DEFAULT_TEXT_TO_VIDEO_MODEL:
        if explicit_model:
            return _xai_error(
                f"xAI reference-to-video requires {DEFAULT_TEXT_TO_VIDEO_MODEL}; got {resolved_model}",
                "unsupported_model", prompt, model=resolved_model,
            )
        resolved_model = DEFAULT_TEXT_TO_VIDEO_MODEL

    clamped_duration = _clamp_duration(duration, has_reference_images=bool(refs))
    payload = {"model": resolved_model, "prompt": prompt, "duration": clamped_duration, "aspect_ratio": aspect_ratio,
               "resolution": resolution}
    payload.update({k: v for k, v in (("image", image_input), ("reference_images", refs)) if v})
    return await _submit_xai_video_payload(
        api_key=api_key, base_url=base_url, endpoint="generations", payload=payload,
        prompt=prompt, resolved_model=resolved_model, modality=modality_used,
        aspect_ratio=aspect_ratio, duration=clamped_duration, operation="generate", resolution=resolution,
    )


async def _mutate_xai_video_async(
    *, api_key: str, base_url: str, prompt: str, video_url: str, model: Optional[str], endpoint: str, operation: str,
    duration: int,
) -> Dict[str, Any]:
    """Edit or extend using a public HTTPS ``video_url`` input (``url`` on the wire)."""
    prompt = (prompt or "").strip()
    video_input = await _video_input_from_public_url(video_url or "", api_key=api_key, base_url=base_url)
    if not prompt:
        return _xai_error("prompt is required for xAI video edit/extend", "missing_prompt", prompt)
    if not video_input:
        msg = "video_url must be a public HTTPS MP4 URL (the `video`/`public_url` from a prior Imagine result)"
        return _xai_error(msg, "missing_video", prompt)
    resolved_model = _resolve_model_for_modality(model, modality="text", explicit_model=bool(model))
    payload: Dict[str, Any] = {"model": resolved_model, "prompt": prompt, "video": video_input}
    if endpoint == "extensions":
        payload["duration"] = duration
    return await _submit_xai_video_payload(
        api_key=api_key, base_url=base_url, endpoint=endpoint, payload=payload,
        prompt=prompt, resolved_model=resolved_model, modality=operation,
        aspect_ratio=DEFAULT_ASPECT_RATIO, duration=duration, operation=operation,
    )


async def _submit_xai_video_payload(
    *, api_key: str, base_url: str, endpoint: str, payload: Dict[str, Any], prompt: str, resolved_model: str,
    modality: str, aspect_ratio: str, duration: int, operation: str, resolution: Optional[str] = None,
) -> Dict[str, Any]:
    """POST ``payload`` to ``/videos/{endpoint}``, poll ``/videos/{request_id}`` to a terminal status, shape the response."""
    try:
        from tools.xai_http import build_xai_storage_options, maybe_mark_xai_storage_notice_seen, read_xai_imagine_storage_config

        storage_options = build_xai_storage_options("video_gen", filename_prefix="hermes-xai-video", extension="mp4")
        storage_notice = maybe_mark_xai_storage_notice_seen("video_gen")
        storage_cfg = read_xai_imagine_storage_config("video_gen")
    except Exception:
        storage_options, storage_notice, storage_cfg = None, None, {"enabled": False}
    if storage_options is not None:
        payload["storage_options"] = storage_options

    headers = _xai_headers(api_key)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/videos/{endpoint}", headers={**headers, "x-idempotency-key": str(uuid.uuid4())},
                json=payload, timeout=60,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:
                pass
            return _xai_error(
                f"xAI submit failed ({exc.response.status_code}): {detail or exc}", "api_error", prompt, model=resolved_model,
            )
        request_id = response.json().get("request_id")
        if not request_id:
            raise RuntimeError("xAI video response did not include request_id")

        elapsed = 0.0
        status, body = "queued", {}
        while elapsed < DEFAULT_TIMEOUT_SECONDS:
            response = await client.get(f"{base_url}/videos/{request_id}", headers=headers, timeout=30)
            response.raise_for_status()
            body = response.json()
            status = (body.get("status") or "").lower()
            if status in _TERMINAL_POLL_STATUSES:
                break
            await asyncio.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
            elapsed += DEFAULT_POLL_INTERVAL_SECONDS
        else:
            return _xai_error(
                f"Timed out waiting for xAI video request after {DEFAULT_TIMEOUT_SECONDS}s", "timeout", prompt,
                model=resolved_model,
            )

    if status != "done":
        message = (body.get("error", {}) or {}).get("message") or body.get("message")
        return _xai_error(message or f"xAI video request ended with status '{status}'", f"xai_{status}", prompt, model=resolved_model)

    video = body.get("video") if isinstance(body.get("video"), dict) else {}
    # Primary URL is the stored files-cdn HTTPS MP4 (``public_url``) when storage is enabled, else xAI's
    # temporary ``video.url``; pass it as ``video_url`` for edit/extend chaining. The temporary URL is only
    # reported when it differs from the stored one.
    file_output = video.get("file_output")
    file_output = file_output if isinstance(file_output, dict) else {}
    stored_public = file_output.get("public_url")
    stored_public = stored_public.strip() if isinstance(stored_public, str) else None
    temporary = video.get("url")
    temporary = temporary.strip() if isinstance(temporary, str) else None
    public_video_url = stored_public or temporary or ""
    if not public_video_url:
        return _xai_error(
            "xAI video request completed without a video URL", "empty_response", prompt,
            model=body.get("model") or resolved_model,
        )
    extra: Dict[str, Any] = {"request_id": request_id, "operation": operation, "storage_enabled": bool(storage_cfg.get("enabled"))}
    if resolution:
        extra["resolution"] = resolution
    if storage_notice:
        extra["storage_notice"] = storage_notice
    if stored_public:
        extra["public_url"] = stored_public
        if temporary and temporary != stored_public:
            extra["temporary_url"] = temporary
    extra.update({k: file_output[k] for k in ("filename", "expires_at", "public_url_expires_at", "public_url_error", "storage_error")
                  if k in file_output})
    if body.get("usage"):
        extra["usage"] = body["usage"]
    return success_response(
        video=public_video_url, model=body.get("model") or resolved_model, prompt=prompt, modality=modality,
        aspect_ratio=aspect_ratio, duration=video.get("duration") or duration, provider="xai", extra=extra,
    )


def register(ctx) -> None:
    """Plugin entry point — wire ``XAIVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(XAIVideoGenProvider())
