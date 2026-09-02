"""DeepInfra image generation backend.

Exposes DeepInfra's image-gen catalog (FLUX, Qwen-Image-Edit, …) through the
OpenAI-compatible ``/v1/openai/images/generations`` endpoint.

**Fully dynamic model discovery.** DeepInfra publishes one tagged catalog at
``https://api.deepinfra.com/v1/openai/models?filter=true&sort_by=hermes``
where each entry's ``metadata.tags`` declares its surface (``image-gen``
here). ``list_models()`` filters it via
:func:`hermes_cli.models._fetch_deepinfra_models_by_tag`, so no model ids are
hardcoded here: new models appear in ``hermes tools`` automatically and
retired ones disappear on the next fetch.

Model selection (first hit wins): ``DEEPINFRA_IMAGE_MODEL`` env →
``image_gen.deepinfra.model`` → first model from the live catalog. When all
three are absent ``generate()`` returns an error rather than guessing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

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
    error_factory,
    import_openai,
    load_image_gen_config,
    prompt_required_error,
    size_for,
)

logger = logging.getLogger(__name__)


def _live_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch ``image-gen``-tagged models from the DeepInfra catalog."""
    try:
        from hermes_cli.models import _fetch_deepinfra_models_by_tag
    except Exception as exc:
        logger.debug("Cannot import _fetch_deepinfra_models_by_tag: %s", exc)
        return None
    return _fetch_deepinfra_models_by_tag("image-gen")


def _format_catalog_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format a catalog item into the picker row shape."""
    mid = item.get("id", "")
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    row: Dict[str, Any] = {
        "id": mid,
        "display": mid.split("/", 1)[-1] if "/" in mid else mid,
        "strengths": metadata.get("description", ""),
    }
    pricing = metadata.get("pricing")
    if isinstance(pricing, dict) and pricing.get("per_image_unit") is not None:
        try:
            row["price"] = f"${float(pricing['per_image_unit']):.4f}/image"
        except (TypeError, ValueError):
            pass
    for key in ("default_width", "default_height", "default_iterations"):
        if metadata.get(key) is not None:
            row[key] = metadata[key]
    return row


def _resolve_model(catalog: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[str]:
    """Pick the model id (env > config > first live result, else None).

    Takes the already-loaded ``image_gen.deepinfra`` config so ``generate()``
    reads config once instead of via a second ``load_config`` deepcopy.
    """
    env_override = os.environ.get("DEEPINFRA_IMAGE_MODEL", "").strip()
    if env_override:
        return env_override
    cfg_model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()
    if catalog:
        first = catalog[0].get("id")
        if isinstance(first, str) and first:
            return first
    return None


class DeepInfraImageGenProvider(ImageGenProvider):
    """DeepInfra ``images.generations`` backend; catalog discovered live by the ``image-gen`` tag."""

    @property
    def name(self) -> str:
        return "deepinfra"

    @property
    def display_name(self) -> str:
        return "DeepInfra"

    def is_available(self) -> bool:
        return bool((get_secret("DEEPINFRA_API_KEY", "") or "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [_format_catalog_row(item) for item in _live_models() or []]

    def default_model(self) -> Optional[str]:
        rows = self.list_models()
        return rows[0].get("id") if rows else None

    def capabilities(self) -> Dict[str, Any]:
        """DeepInfra's OpenAI-compatible generation surface is text-only."""
        return {"modalities": ["text"], "max_reference_images": 0}

    def get_setup_schema(self) -> Dict[str, Any]:
        return api_key_setup_schema(
            "DeepInfra", "paid", "FLUX, Qwen-Image, … — live catalog from api.deepinfra.com",
            key="DEEPINFRA_API_KEY", prompt="DeepInfra API key",
            url="https://deepinfra.com/dash/api_keys",
        )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        fail = error_factory("deepinfra", aspect)

        if kwargs.get("image_url") or kwargs.get("reference_image_urls"):
            return fail(
                "DeepInfra image generation is text-to-image only in this "
                "backend; image_url and reference_image_urls are unsupported.",
                "modality_unsupported", prompt=prompt,
            )

        if not prompt:
            return prompt_required_error("deepinfra", aspect)

        api_key = (get_secret("DEEPINFRA_API_KEY", "") or "").strip()
        if not api_key:
            return fail(
                "DEEPINFRA_API_KEY not set. Run `hermes tools` → Image "
                "Generation → DeepInfra to configure, or `hermes setup` "
                "to add the key.",
                "auth_required",
            )

        di_cfg = load_image_gen_config("deepinfra")
        model_id = _resolve_model(_live_models() or [], di_cfg)
        if not model_id:
            return fail(
                "No DeepInfra image-gen model available. Pin one in "
                "config.yaml under image_gen.deepinfra.model, set "
                "DEEPINFRA_IMAGE_MODEL, or check connectivity to "
                "api.deepinfra.com so the live catalog can be fetched.",
                "no_model_available", prompt=prompt,
            )
        size = size_for(aspect)
        from hermes_cli.models import deepinfra_base_url

        # OpenAI-compatible endpoint — the openai SDK supplies retry, timeout
        # and error mapping (mirrors the OpenAI image-gen plugin).
        openai, err = import_openai("deepinfra", aspect)
        if err:
            return err

        fail = error_factory("deepinfra", aspect, model=model_id, prompt=prompt)
        client = openai.OpenAI(api_key=api_key, base_url=deepinfra_base_url(di_cfg))
        try:
            response = client.images.generate(model=model_id, prompt=prompt, size=size, n=1)
        except Exception as exc:
            logger.debug("DeepInfra image generation failed", exc_info=True)
            return fail(f"DeepInfra image generation failed: {exc}", "api_error")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        data = getattr(response, "data", None) or []
        if not data:
            return fail("DeepInfra returned no image data", "empty_response")

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        # Drop the ``vendor/`` prefix and any colons so the saved filename
        # stays a single path component on every OS.
        prefix = f"deepinfra_{model_id.split('/', 1)[-1].replace(':', '_')}"

        if b64:
            try:
                image_ref = str(save_b64_image(b64, prefix=prefix))
            except Exception as exc:
                return fail(f"Could not save image to cache: {exc}", "io_error")
        elif url:
            # Delivery URLs are often short-lived; materialise locally so downstream
            # consumers (Telegram send_photo, browser fetch) don't get a dead link.
            # Best-effort: fall back to the bare URL if the download fails.
            try:
                image_ref = str(save_url_image(url, prefix=prefix))
            except Exception as exc:
                logger.debug("DeepInfra: caching delivery URL failed (%s); returning URL", exc)
                image_ref = url
        else:
            return fail("DeepInfra response contained neither b64_json nor URL", "empty_response")

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="deepinfra",
            extra={"size": size},
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``DeepInfraImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(DeepInfraImageGenProvider())
