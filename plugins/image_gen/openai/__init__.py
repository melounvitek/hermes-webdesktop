"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers
(``gpt-image-2-low`` ~15s, ``-medium`` ~40s default, ``-high`` ~2min) as
virtual model ids so the picker and ``image_gen.model`` behave like any other
multi-model backend. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence: ``OPENAI_IMAGE_MODEL`` env → ``image_gen.openai.model``
→ ``image_gen.model`` (when it is one of our tier ids) → :data:`DEFAULT_MODEL`.
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
    success_response,
)
from plugins.image_gen._common import (
    GPT_IMAGE_2_API_MODEL as API_MODEL,
    GPT_IMAGE_2_DEFAULT as DEFAULT_MODEL,
    GPT_IMAGE_2_TIERS,
    api_key_setup_schema,
    catalog_rows,
    collect_source_images,
    error_factory,
    import_openai,
    materialize_image,
    openai_importable,
    prompt_required_error,
    resolve_static_model,
    size_for,
)

logger = logging.getLogger(__name__)

_MODELS: Dict[str, Dict[str, Any]] = dict(GPT_IMAGE_2_TIERS)


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    return resolve_static_model(
        _MODELS, DEFAULT_MODEL, env_var="OPENAI_IMAGE_MODEL", config_key="openai"
    )


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load ``(data, filename)`` from a URL, data URI or local path; raises on IO/network error."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        resp = requests.get(ref, timeout=60)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(b64), f"image.{ext}"
    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    return data, os.path.basename(ref) or "image.png"


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` / ``images.edit`` backend — gpt-image-2."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        return bool(get_secret("OPENAI_API_KEY")) and openai_importable()

    def list_models(self) -> List[Dict[str, Any]]:
        return catalog_rows(_MODELS, price="varies")

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return api_key_setup_schema(
            "OpenAI", "paid",
            "gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing",
            key="OPENAI_API_KEY", prompt="OpenAI API key",
            url="https://platform.openai.com/api-keys",
        )

    def capabilities(self) -> Dict[str, Any]:
        # images.edit() accepts up to 16 source images.
        return {"modalities": ["text", "image"], "max_reference_images": 16}

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
            return prompt_required_error("openai", aspect)

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            return error_factory("openai", aspect)(
                "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                "Generation → OpenAI to configure, or `hermes setup` "
                "to add the key.",
                "auth_required",
            )

        openai, err = import_openai("openai", aspect)
        if err:
            return err

        tier_id, meta = _resolve_model()
        size = size_for(aspect)
        sources = collect_source_images(image_url, reference_image_urls, limit=16)
        is_edit = bool(sources)
        fail = error_factory("openai", aspect, model=tier_id, prompt=prompt)
        client = openai.OpenAI(api_key=api_key)

        if is_edit:
            # images.edit() expects named file-like objects for correct multipart.
            import io

            try:
                files = []
                for ref in sources:
                    data, fname = _load_image_bytes(ref)
                    bio = io.BytesIO(data)
                    bio.name = fname
                    files.append(bio)
            except Exception as exc:
                return fail(f"Could not load source image for editing: {exc}", "io_error")

            try:
                response = client.images.edit(
                    model=API_MODEL,
                    image=files if len(files) > 1 else files[0],
                    prompt=prompt,
                    size=size,  # type: ignore[arg-type]  # OPENAI_SIZES values are valid gpt-image sizes
                    quality=meta["quality"],
                    n=1,
                )
            except Exception as exc:
                logger.debug("OpenAI image edit failed", exc_info=True)
                return fail(f"OpenAI image editing failed: {exc}", "api_error")
        else:
            # gpt-image-2 returns b64_json unconditionally and REJECTS
            # ``response_format`` as an unknown parameter. Don't send it.
            try:
                response = client.images.generate(
                    model=API_MODEL, prompt=prompt, size=size, n=1, quality=meta["quality"],
                )
            except Exception as exc:
                logger.debug("OpenAI image generation failed", exc_info=True)
                return fail(f"OpenAI image generation failed: {exc}", "api_error")

        data = getattr(response, "data", None) or []
        if not data:
            return fail("OpenAI returned no image data", "empty_response")

        first = data[0]
        image_ref, err = materialize_image(
            getattr(first, "b64_json", None), getattr(first, "url", None),
            prefix=f"openai_{tier_id}", label="OpenAI", provider="openai",
            model=tier_id, prompt=prompt, aspect=aspect, log=logger,
        )
        if err:
            return err

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
        revised_prompt = getattr(first, "revised_prompt", None)
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            modality="image" if is_edit else "text",
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
