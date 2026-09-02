"""Meta Model API image generation backend.

Exposes Meta's ``muse-image`` model(s) as an :class:`ImageGenProvider`.
The Meta Model API (https://api.meta.ai/v1) is OpenAI-compatible, so we reuse
the OpenAI Python SDK pointed at Meta's base URL and authenticate with
``META_MODEL_API_KEY``. Output is base64 JSON (WebP) → ``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins): ``model`` kwarg forwarded by the
dispatcher (the ``hermes tools`` pick) → ``META_IMAGE_MODEL`` env →
``image_gen.meta-ai.model`` → ``image_gen.model`` (when it's one of our IDs)
→ :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)
from plugins.image_gen._common import (
    api_key_setup_schema,
    catalog_rows,
    error_factory,
    import_openai,
    openai_importable,
    prompt_required_error,
    resolve_static_model,
    size_for,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
# Auth env vars, in priority order. Mirrors the bundled ``meta-ai`` chat
# provider (plugins/model-providers/meta-ai): MODEL_API_KEY is Meta's
# documented var; the rest are accepted aliases.
API_KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY")
# Primary key shown in setup prompts / error messages.
API_KEY_ENV = "META_MODEL_API_KEY"
# Optional base-url override (same var the chat provider honors).
BASE_URL_ENV = "META_BASE_URL"


def _resolve_api_key() -> Optional[str]:
    """First non-empty auth env var, checked in priority order."""
    for env in API_KEY_ENVS:
        val = get_secret(env)
        if val:
            return val
    return None


def _resolve_base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or "").strip() or DEFAULT_BASE_URL


# Catalog shown in `hermes tools` and matched against `image_gen.model`.
# The model id is sent verbatim to the Meta Model API (`/v1/images/generations`).
_MODELS: Dict[str, Dict[str, Any]] = {
    "muse-image-1.0": {
        "display": "Muse Image 1.0",
        "speed": "~10s",
        "strengths": "Meta Model API image generation",
        "price": "$0.01/image",
    },
}
DEFAULT_MODEL = "muse-image-1.0"


def _resolve_model(caller_model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Return (model_id, metadata); ``caller_model`` is the dispatcher's ``model`` kwarg."""
    return resolve_static_model(
        _MODELS, DEFAULT_MODEL, env_var="META_IMAGE_MODEL", config_key="meta-ai",
        explicit=caller_model,
    )


class MetaImageGenProvider(ImageGenProvider):
    """Meta Model API ``images.generate`` backend (muse-image)."""

    @property
    def name(self) -> str:
        return "meta-ai"

    @property
    def display_name(self) -> str:
        return "Meta Model API"

    def is_available(self) -> bool:
        return bool(_resolve_api_key()) and openai_importable()

    def list_models(self) -> List[Dict[str, Any]]:
        return catalog_rows(_MODELS)

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return api_key_setup_schema(
            "Meta Model API", "paid", "Muse Image via Meta Model API (api.meta.ai)",
            key=API_KEY_ENV, prompt="Meta Model API key (LLM|... token)", url="https://api.meta.ai",
        )

    def capabilities(self) -> Dict[str, Any]:
        # Text-to-image only until image-to-image is verified against the Meta endpoint.
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return prompt_required_error("meta-ai", aspect)

        api_key = _resolve_api_key()
        if not api_key:
            return error_factory("meta-ai", aspect)(
                f"{API_KEY_ENV} not set. Run `hermes tools` -> Image "
                "Generation -> Meta Model API to configure.",
                "auth_required",
            )

        openai, err = import_openai("meta-ai", aspect)
        if err:
            return err

        model_id, _meta = _resolve_model(kwargs.get("model"))
        size = size_for(aspect)
        fail = error_factory("meta-ai", aspect, model=model_id, prompt=prompt)
        client = openai.OpenAI(api_key=api_key, base_url=_resolve_base_url())

        try:
            response = client.images.generate(model=model_id, prompt=prompt, size=size, n=1)
        except Exception as exc:
            logger.debug("Meta image generation failed", exc_info=True)
            return fail(f"Meta image generation failed: {exc}", "api_error")

        try:
            first = response.data[0]
        except (AttributeError, IndexError, TypeError):
            return fail("Meta response contained no image data", "empty_response")

        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        try:
            if b64:
                image_ref = str(save_b64_image(b64, prefix="meta", extension="webp"))
            elif url:
                image_ref = str(save_url_image(url, prefix="meta"))
            else:
                return fail("Meta response contained neither b64_json nor URL", "empty_response")
        except Exception as exc:
            return fail(f"Failed to save Meta image: {exc}", "io_error")

        extra: Dict[str, Any] = {"size": size}
        revised_prompt = getattr(first, "revised_prompt", None)
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="meta-ai",
            modality="text",
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point -- wire ``MetaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MetaImageGenProvider())
