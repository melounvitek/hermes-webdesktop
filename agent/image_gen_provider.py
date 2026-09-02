"""
Image Generation Provider ABC
=============================

Pluggable-backend interface for image generation. Providers register via
``PluginContext.register_image_gen_provider()``; the one selected by
``image_gen.provider`` services every ``image_generate`` call. Providers live in
``<repo>/plugins/image_gen/<name>/`` (built-in) or
``~/.hermes/plugins/image_gen/<name>/`` (user, opt-in).

One tool covers text-to-image and image-to-image/editing: the presence of
``image_url`` (and/or ``reference_image_urls``) routes to the provider's edit
endpoint, otherwise text-to-image. Users pick one model; the provider picks the
endpoint. Mirrors ``agent/video_gen_provider.py`` so the two stay learnable.

Response shape (built by :func:`success_response` / :func:`error_response`)::

    success        bool
    image          str | None       URL or absolute file path
    model          str              provider-specific model identifier
    prompt         str              echoed prompt
    aspect_ratio   str              "landscape" | "square" | "portrait"
    modality       str              "text" | "image" (which mode was used)
    provider       str              provider name (for diagnostics)
    error          str              only when success=False
    error_type     str              only when success=False
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent import provider_media
from agent.provider_base import CatalogProviderBase

logger = logging.getLogger(__name__)


VALID_ASPECT_RATIOS: Tuple[str, ...] = ("landscape", "square", "portrait")
DEFAULT_ASPECT_RATIO = "landscape"


class ImageGenProvider(CatalogProviderBase):
    """Abstract base class for an image generation backend.

    Subclasses must implement :attr:`name` and :meth:`generate`; everything else
    has defaults. ``list_models`` entries may add ``speed`` / ``strengths`` /
    ``price`` for the picker.
    """

    def capabilities(self) -> Dict[str, Any]:
        """What this provider supports: ``modalities`` (``"text"`` and/or
        ``"image"``) and ``max_reference_images``.

        The tool layer surfaces this in the dynamic schema so the model knows
        when ``image_url`` is honored. Default is text-only so a provider that
        doesn't override advertises only text-to-image (backward compatible).
        """
        return {
            "modalities": ["text"],
            "max_reference_images": 0,
        }

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image, or edit/transform a source image.

        ``image_url`` is the primary source to edit; ``reference_image_urls``
        are extra style/composition references (clamp to ``max_reference_images``).
        Any source image routes to the edit endpoint, otherwise text-to-image.
        Return :func:`success_response` / :func:`error_response`. Unknown
        ``kwargs`` MUST be ignored (forward compat). Known optional kwarg:
        ``upscale`` (bool) — a post-generation high-res pass; providers that
        honor it report ``upscaled: True`` in ``extra``.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_aspect_ratio(value: Optional[str]) -> str:
    """Clamp to :data:`VALID_ASPECT_RATIOS`; invalid values coerce to landscape so
    the tool surface forgives agent mistakes instead of rejecting them."""
    if not isinstance(value, str):
        return DEFAULT_ASPECT_RATIO
    v = value.strip().lower()
    if v in VALID_ASPECT_RATIOS:
        return v
    return DEFAULT_ASPECT_RATIO


def normalize_reference_images(value: Any) -> Optional[List[str]]:
    """Coerce a str or list into a clean list of non-blank strings; ``None`` when
    nothing usable remains so providers treat "no refs" as one sentinel."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out or None


def _images_cache_dir() -> Path:
    """Return ``$HERMES_HOME/cache/images/``, creating parents as needed."""
    return provider_media.cache_dir("images")


def save_b64_image(
    b64_data: str,
    *,
    prefix: str = "image",
    extension: str = "png",
) -> Path:
    """Decode base64 image data into ``$HERMES_HOME/cache/images/``; return the path."""
    return provider_media.save_b64("images", b64_data, prefix=prefix, extension=extension)


_URL_IMAGE_CONTENT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def save_url_image(
    url: str,
    *,
    prefix: str = "image",
    timeout: float = 60.0,
    max_bytes: int = 25 * 1024 * 1024,
) -> Path:
    """Download an (often ephemeral) image URL into ``$HERMES_HOME/cache/images/``.

    Raises on network / HTTP / oversize / empty errors so callers can fall back
    to returning the bare URL with a clear message. See :mod:`agent.provider_media`.
    """
    return provider_media.save_url(
        "images", url, prefix=prefix, timeout=timeout, max_bytes=max_bytes,
        chunk_size=64 * 1024, content_types=_URL_IMAGE_CONTENT_TYPES,
        url_extensions=("png", "jpg", "jpeg", "webp", "gif"), default_extension="png",
        label="Image", empty_error="Image at {url} returned 0 bytes; refusing to cache.",
    )


def success_response(
    *,
    image: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    provider: str,
    modality: str = "text",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Uniform success dict; ``extra`` keys are added without overriding standard ones."""
    payload: Dict[str, Any] = {
        "success": True,
        "image": image,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "modality": modality,
        "provider": provider,
    }
    if extra:
        for k, v in extra.items():
            payload.setdefault(k, v)
    return payload


def error_response(
    *,
    error: str,
    error_type: str = "provider_error",
    provider: str = "",
    model: str = "",
    prompt: str = "",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> Dict[str, Any]:
    """Build a uniform error response dict."""
    return {
        "success": False,
        "image": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
    }
