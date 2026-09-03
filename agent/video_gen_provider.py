"""
Video Generation Provider ABC
=============================

Pluggable-backend interface for video generation. Providers register via
``PluginContext.register_video_gen_provider()``; the one selected by
``video_gen.provider`` services every ``video_generate`` call. Providers live in
``<repo>/plugins/video_gen/<name>/`` (built-in) or
``~/.hermes/plugins/video_gen/<name>/`` (user, opt-in). Mirrors
``agent/image_gen_provider.py`` so the two surfaces stay learnable together.

One tool covers text-to-video and image-to-video: ``image_url`` present routes
to the provider's image-to-video endpoint, absent routes to text-to-video. Users
pick one model family; the provider picks the FAL/xAI endpoint. Video edit and
extend are deliberately NOT exposed — backends are too inconsistent for one
unified tool.

Response shape (built by :func:`success_response` / :func:`error_response`)::

    success         bool
    video           str | None      URL or absolute file path
    model           str             provider-specific model identifier
    prompt          str             echoed prompt
    modality        str             "text" | "image" (which mode was used)
    aspect_ratio    str             provider-native (e.g. "16:9") or ""
    duration        int             seconds (0 if not applicable)
    provider        str             provider name (for diagnostics)
    error           str             only when success=False
    error_type      str             only when success=False
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent import provider_media
from agent.provider_base import CatalogProviderBase

logger = logging.getLogger(__name__)


# Advertised as an enum hint in the tool schema; providers may accept a narrower
# or wider set and are responsible for clamping.
COMMON_ASPECT_RATIOS: Tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3")
DEFAULT_ASPECT_RATIO = "16:9"

COMMON_RESOLUTIONS: Tuple[str, ...] = ("480p", "540p", "720p", "1080p")
DEFAULT_RESOLUTION = "720p"


class VideoGenProvider(CatalogProviderBase):
    """Abstract base class for a video generation backend.

    Subclasses must implement :attr:`name` and :meth:`generate`; everything else
    has defaults. ``list_models`` entries are **model families** and may add
    ``speed`` / ``strengths`` / ``price`` / advisory ``modalities``.
    """

    def capabilities(self) -> Dict[str, Any]:
        """What this provider supports (all keys optional): ``modalities``,
        ``aspect_ratios``, ``resolutions``, ``max_duration`` / ``min_duration``,
        ``supports_audio`` / ``supports_negative_prompt`` / ``supports_seed`` /
        ``supports_upscale``, ``max_reference_images``.

        Used for soft validation, capability-gated params in the dynamic
        ``video_generate`` schema (args a backend can't honor aren't advertised),
        and the picker. Default fails closed: text-only, no optional features.
        """
        return {
            "modalities": ["text"],
            "aspect_ratios": list(COMMON_ASPECT_RATIOS),
            "resolutions": list(COMMON_RESOLUTIONS),
            "max_duration": 10,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "supports_seed": False,
            "supports_upscale": False,
            "max_reference_images": 0,
        }

    @abc.abstractmethod
    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None, duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO, resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None, audio: Optional[bool] = None,
        seed: Optional[int] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a video from a prompt, or animate ``image_url`` when given.

        Return :func:`success_response` / :func:`error_response`. Unknown
        ``kwargs`` MUST be ignored (forward compat). Known optional kwarg:
        ``upscale`` (bool) — a post-generation high-res pass; providers that
        honor it report ``upscaled: True`` in ``extra``.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_b64_video(b64_data: str,*, prefix: str="video", extension: str="mp4") -> Path:
    """Decode base64 video data into ``$HERMES_HOME/cache/videos/``; return the path."""
    return provider_media.save_b64("videos", b64_data, prefix=prefix, extension=extension)


def save_bytes_video(raw: bytes,*, prefix: str="video", extension: str="mp4") -> Path:
    """Write raw video bytes (e.g. an HTTP download body) to the cache."""
    return provider_media.save_bytes("videos", raw, prefix=prefix, extension=extension)


_URL_VIDEO_CONTENT_TYPES = {
    "video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov", "video/x-matroska": "mkv"
}


def save_url_video(
    url: str, *, prefix: str = "video", timeout: float = 180.0, max_bytes: int = 200 * 1024 * 1024
) -> Path:
    """Download an (often ephemeral) video URL into ``$HERMES_HOME/cache/videos/``.

    Raises on network / HTTP / oversize / empty errors so callers can fall back
    to returning the bare URL. See :mod:`agent.provider_media`.
    """
    return provider_media.save_url(
        "videos", url, prefix=prefix, timeout=timeout, max_bytes=max_bytes,
        chunk_size=256 * 1024, content_types=_URL_VIDEO_CONTENT_TYPES,
        url_extensions=("mp4", "webm", "mov", "mkv"), default_extension="mp4",
        label="Video", empty_error="Video at {url} was empty (0 bytes).",
    )


def success_response(
    *, video: str, model: str, prompt: str, modality: str = "text", aspect_ratio: str = "",
    duration: int = 0, provider: str, extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Uniform success dict; ``extra`` keys are added without overriding standard ones."""
    payload: Dict[str, Any] = {
        "success": True,
        "video": video,
        "model": model,
        "prompt": prompt,
        "modality": modality,
        "aspect_ratio": aspect_ratio,
        "duration": int(duration) if duration else 0,
        "provider": provider,
    }
    if extra:
        for k, v in extra.items():
            payload.setdefault(k, v)
    return payload


def error_response(
    *, error: str, error_type: str = "provider_error", provider: str = "", model: str = "",
    prompt: str = "", aspect_ratio: str = "",
) -> Dict[str, Any]:
    """Build a uniform error response dict."""
    return {
        "success": False,
        "video": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
    }


# ---------------------------------------------------------------------------
# Reusable OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAICompatibleVideoGenProvider(VideoGenProvider):
    """Generic text/image-to-video over the OpenAI ``client.videos`` API.

    DeepInfra, OpenAI/Sora, and OpenRouter share the ``POST /videos`` async-job
    shape (``create`` → poll → ``download_content``), so the SDK call lives here
    once; a concrete backend declares identity and credentials::

        class FooVideoGenProvider(OpenAICompatibleVideoGenProvider):
            name = "foo"
            _env_key = "FOO_API_KEY"
            _default_base_url = "https://api.foo.com/v1/openai"
            def list_models(self):
                return [...]   # entries with an "id" key; default_model() uses [0]

    ``image_url`` routes to image-to-video; its absence routes to text-to-video.
    Provider-specific fields (``image_url``/``negative_prompt``/``seed``) ride
    in ``extra_body`` so they pass through the SDK unchanged.
    """

    _env_key: str = "OPENAI_API_KEY"
    _default_base_url: str = "https://api.openai.com/v1"

    # The SDK's ``create_and_poll`` polls ~1/s forever on a non-terminal status,
    # pinning the tool-executor thread on a stuck job; we poll coarsely with a
    # hard wall-clock deadline instead.
    _poll_interval_s: float = 5.0
    _poll_deadline_s: float = 900.0

    def _api_key(self) -> str:
        import os

        return os.environ.get(self._env_key, "").strip()

    def is_available(self) -> bool:
        return bool(self._api_key())

    def _create_and_poll(self, client: Any, call_kwargs: Dict[str, Any]) -> Any:
        """Create the job and poll to a terminal status (any); raise
        :class:`TimeoutError` when ``_poll_deadline_s`` passes first."""
        import time

        video = client.videos.create(**call_kwargs)
        terminal = {"completed", "succeeded", "failed", "error", "cancelled", "canceled"}
        deadline = time.monotonic() + self._poll_deadline_s
        while getattr(video, "status", None) not in terminal:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"video job {getattr(video, 'id', '?')} did not reach a terminal "
                    f"status within {int(self._poll_deadline_s)}s "
                    f"(last status={getattr(video, 'status', None)!r})"
                )
            time.sleep(self._poll_interval_s)
            video = client.videos.retrieve(video.id)
        return video

    def _base_url(self) -> str:
        import os

        override = os.environ.get(f"{self.name.upper()}_BASE_URL", "").strip()
        return override or self._default_base_url

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None, duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO, resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None, audio: Optional[bool] = None,
        seed: Optional[int] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            return error_response(
                error="prompt is required", error_type="invalid_request", provider=self.name
            )
        if not self._api_key():
            return error_response(
                error=f"{self._env_key} is not set", error_type="missing_credentials",
                provider=self.name,
            )
        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency", provider=self.name,
            )

        model_id = model or self.default_model()
        if not model_id:
            return error_response(
                error=f"no {self.name} video model available (live catalog empty?)",
                error_type="no_model", provider=self.name,
            )

        # Fields ``videos.create`` doesn't name natively ride in ``extra_body``.
        extra_body = {
            k: v
            for k, v in {
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "image_url": image_url,  # presence ⇒ image-to-video
                "seed": seed,
            }.items()
            if v is not None
        }
        call_kwargs: Dict[str, Any] = {"model": model_id, "prompt": prompt}
        if duration:
            call_kwargs["seconds"] = str(duration)
        if resolution:
            call_kwargs["size"] = resolution
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        client = openai.OpenAI(api_key=self._api_key(), base_url=self._base_url())
        try:
            try:
                video = self._create_and_poll(client, call_kwargs)
            except Exception as exc:  # noqa: BLE001 - surface any SDK/API/timeout failure uniformly
                logger.debug("%s video generation failed", self.name, exc_info=True)
                return error_response(
                    error=f"{self.name} video generation failed: {exc}", error_type="api_error",
                    provider=self.name, model=model_id, prompt=prompt, aspect_ratio=aspect_ratio,
                )

            # DeepInfra reports "succeeded", OpenAI/Sora "completed" — accept both.
            status = getattr(video, "status", None)
            if status not in ("completed", "succeeded"):
                # ``video.error`` is a pydantic object — str() keeps the dict JSON-serializable.
                job_error = getattr(video, "error", None)
                return error_response(
                    error=str(job_error) if job_error else f"video job ended with status={status!r}",
                    error_type="job_failed", provider=self.name, model=model_id, prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

            # Output is a delivery URL in ``data`` (DeepInfra/FAL) or only reachable
            # via the SDK download endpoint (OpenAI/Sora). Save locally either way —
            # DeepInfra's delivery URLs are short-lived.
            url = None
            for item in getattr(video, "data", None) or []:
                candidate = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
                if candidate:
                    url = candidate
                    break

            try:
                if url:
                    video_ref = str(save_url_video(url, prefix=self.name))
                else:
                    raw = client.videos.download_content(video.id).read()
                    video_ref = str(save_bytes_video(raw, prefix=self.name))
            except Exception as exc:  # noqa: BLE001
                if url:
                    logger.debug("%s: saving video locally failed (%s); returning URL", self.name, exc)
                    video_ref = url
                else:
                    return error_response(
                        error=f"{self.name} video job succeeded but no output could be retrieved: {exc}",
                        error_type="empty_response", provider=self.name, model=model_id,
                        prompt=prompt, aspect_ratio=aspect_ratio,
                    )

            return success_response(
                video=video_ref, model=model_id, prompt=prompt,
                modality="image" if image_url else "text", aspect_ratio=aspect_ratio,
                duration=duration or 0, provider=self.name,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
