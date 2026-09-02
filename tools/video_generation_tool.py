#!/usr/bin/env python3
"""
Video Generation Tool
=====================

Single ``video_generate`` tool that dispatches to a plugin-registered
video generation provider. Mirrors the ``image_generate`` design:

- ``agent/video_gen_provider.py`` defines the :class:`VideoGenProvider` ABC.
- ``agent/video_gen_registry.py`` holds the active providers (populated by
  plugins at import time).
- Each provider lives under ``plugins/video_gen/<name>/``.

The tool is backend-agnostic and ships **no in-tree provider** — enable a
plugin (``hermes plugins enable video_gen/<name>``) and select it in
``hermes tools`` → Video Generation.

One tool covers text-to-video, image-to-video and reference-to-video with a
compact schema (prompt, image_url, reference_image_urls, duration,
aspect_ratio, resolution, negative_prompt, audio, seed, model). Providers
ignore parameters they do not support: the tool layer does only lightweight
validation (type/required-prompt) and each provider clamps inside
:meth:`VideoGenProvider.generate`, so the surface stays stable as providers
with different capabilities ship. Video edit/extend are intentionally not
exposed here; providers with those workflows expose separate tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    COMMON_ASPECT_RATIOS,
    COMMON_RESOLUTIONS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    error_response,
)
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


VIDEO_GENERATE_SCHEMA: Dict[str, Any] = {
    "name": "video_generate",
    # Placeholder — description AND params are rebuilt dynamically at
    # get_tool_definitions() time from the active provider's declared
    # capabilities() and the active model's catalog entry. Optional args
    # (image_url, reference_image_urls, negative_prompt, audio, seed,
    # upscale) are advertised ONLY when the active backend/model honors
    # them; the handler accepts them regardless (replay compat — providers
    # clamp/ignore). See _build_dynamic_video_schema().
    "description": "(rebuilt at get_definitions() time — see _build_dynamic_video_schema)",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text instruction describing the desired video, motion, "
                    "subject, style, camera movement, etc."
                ),
            },
            "duration": {
                "type": "integer",
                "description": (
                    "Desired video duration in seconds. Providers clamp to "
                    "their supported range. Omit for the provider default."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(COMMON_ASPECT_RATIOS),
                "description": "Output aspect ratio.",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "resolution": {
                "type": "string",
                "enum": list(COMMON_RESOLUTIONS),
                "description": "Output resolution.",
                "default": DEFAULT_RESOLUTION,
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override; defaults to the configured "
                    "``video_gen.model``. Unknown models are rejected."
                ),
            },
            # image_url / reference_image_urls / negative_prompt / audio / seed /
            # upscale are added per-capability by _build_dynamic_video_schema.
            # Do not re-add them statically.
        },
        "required": ["prompt"],
    },
}


# ---------------------------------------------------------------------------
# Config readers (mirror image_generation_tool.py)
# ---------------------------------------------------------------------------


def _read_video_gen_key(key: str) -> Optional[str]:
    """Return the stripped ``video_gen.<key>`` string from config.yaml, or None."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        value = section.get(key) if isinstance(section, dict) else None
    except Exception as exc:
        logger.debug("Could not read video_gen config: %s", exc)
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_configured_video_provider() -> Optional[str]:
    return _read_video_gen_key("provider")


def _read_configured_video_model() -> Optional[str]:
    return _read_video_gen_key("model")


# ---------------------------------------------------------------------------
# Availability check + provider resolution
# ---------------------------------------------------------------------------


def check_video_generation_requirements() -> bool:
    """True when at least one registered provider reports available.

    Triggers plugin discovery (idempotent) so user-installed plugins are
    visible to the toolset gate.
    """
    try:
        from agent.video_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        for provider in list_providers():
            try:
                if provider.is_available():
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _resolve_active_provider():
    """Return the active provider object or None.

    Forces a discovery refresh on a miss — handles long-lived sessions that
    started before a plugin was installed.
    """
    try:
        from agent.video_gen_registry import get_active_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_active_provider()
        if provider is None:
            _ensure_plugins_discovered(force=True)
            provider = get_active_provider()
        return provider
    except Exception as exc:
        logger.debug("video_gen provider resolution failed: %s", exc)
        return None


def _missing_provider_error(configured: Optional[str]) -> str:
    if configured:
        msg = (
            f"video_gen.provider='{configured}' is set but no plugin "
            f"registered that name. Run `hermes plugins list` to see "
            f"installed video gen backends, or `hermes tools` → Video "
            f"Generation to pick one."
        )
        return json.dumps(error_response(
            error=msg, error_type="provider_not_registered",
            provider=configured,
        ))
    msg = (
        "No video generation backend is configured. Run `hermes tools` → "
        "Video Generation to enable one (xAI, FAL, or Google Veo)."
    )
    return json.dumps(error_response(
        error=msg, error_type="no_provider_configured",
    ))


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    return None


def _normalize_reference_images(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    out = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return out or None


def _handle_video_generate(args: Dict[str, Any], **_kw: Any) -> str:
    prompt = (args.get("prompt") or "").strip()
    image_url = (args.get("image_url") or "").strip() or None
    reference_image_urls = _normalize_reference_images(args.get("reference_image_urls"))
    task_id = _kw.get("task_id")

    # Confinement chokepoint (mirrors image_generate): under a non-local
    # backend, path-like source images reach providers as data: URLs.
    from tools.image_generation_tool import _confine_source_images

    image_url, reference_image_urls, confine_error = _confine_source_images(
        image_url, reference_image_urls, task_id)
    if confine_error is not None:
        return confine_error
    duration = _coerce_int(args.get("duration"))
    aspect_ratio = (args.get("aspect_ratio") or DEFAULT_ASPECT_RATIO).strip() or DEFAULT_ASPECT_RATIO
    resolution = (args.get("resolution") or DEFAULT_RESOLUTION).strip() or DEFAULT_RESOLUTION
    negative_prompt = (args.get("negative_prompt") or "").strip() or None
    audio = _coerce_bool(args.get("audio"))
    seed = _coerce_int(args.get("seed"))
    upscale = _coerce_bool(args.get("upscale"))
    model_override = (args.get("model") or "").strip() or None

    # Soft validation — providers do their own. The backend may accept
    # image-only on its image-to-video endpoint, but our surface always needs a prompt.
    if not prompt:
        return tool_error("prompt is required for video generation")
    if "operation" in args or "video_url" in args:
        return tool_error(
            "video_generate only supports text-to-video, image-to-video, and "
            "reference-to-video; use a provider-specific tool for video edit/extend"
        )

    configured = _read_configured_video_provider()
    provider = _resolve_active_provider()
    if provider is None:
        return _missing_provider_error(configured)

    # Explicit arg wins, then config, then provider default.
    model = model_override or _read_configured_video_model() or provider.default_model()

    kwargs: Dict[str, Any] = {
        "model": model,
        "_model_override_explicit": bool(model_override),
        "image_url": image_url,
        "reference_image_urls": reference_image_urls,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "negative_prompt": negative_prompt,
        "audio": audio,
        "seed": seed,
        "upscale": upscale,
    }
    # Drop None entries so providers see clean defaults.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pname = getattr(provider, "name", "?")

    def _err(error: str, error_type: str) -> str:
        return json.dumps(error_response(
            error=error, error_type=error_type,
            provider=getattr(provider, "name", ""), model=model or "", prompt=prompt,
        ))

    try:
        result = provider.generate(prompt=prompt, **kwargs)
    except TypeError as exc:
        # A provider that hasn't widened its signature is a plugin bug, not a
        # caller error — surface a clear contract message.
        logger.warning(
            "video_gen provider '%s' rejected kwargs (signature too narrow): %s",
            pname, exc,
        )
        return _err(
            f"Provider '{pname}' signature is "
            f"out of date with the video_generate schema. Report this "
            f"to the plugin author.",
            "provider_contract",
        )
    except Exception as exc:
        logger.warning("video_gen provider '%s' raised: %s", pname, exc)
        return _err(f"Provider '{pname}' error: {exc}", "provider_exception")

    if not isinstance(result, dict):
        return _err("Provider returned a non-dict result", "provider_contract")

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Dynamic schema — reflect the active backend's actual capabilities
# ---------------------------------------------------------------------------
# The configured backend determines which modalities, aspect ratios,
# resolutions, durations and audio/negative-prompt flags are real; surfacing
# the per-model surface in the description means the model usually gets the
# call right first try. model_tools.get_tool_definitions() keys its cache on
# config.yaml mtime, so the schema rebuilds when provider/model changes.


_GENERIC_DESCRIPTION = (
    "Generate a video from a text prompt (text-to-video), animate a "
    "still image (image-to-video), or guide generation with reference images. "
    "Pass `image_url` to animate an image or `reference_image_urls` for "
    "reference-to-video. Video edit/extend workflows are not part of this "
    "unified surface; use a dedicated provider-specific tool when one is "
    "available. The backend and model family are user-configured via "
    "`hermes tools` → Video Generation; the agent does not pick them. "
    "Long-running generations may take 30 seconds to several minutes — "
    "the call blocks until the video is ready. Returns the result in the "
    "`video` field — either an HTTP URL or an absolute file path. To show "
    "it to the user, reference that path/URL in your response using the "
    "file-delivery convention for the current platform (your platform "
    "guidance describes how files are delivered here)."
)


def _build_dynamic_video_schema() -> Dict[str, Any]:
    """Render description AND params from the active backend's declared surface.

    Optional args are advertised only when the resolved provider/model honors
    them (capabilities() + the model's catalog entry); enums and duration
    bounds tighten to the active model's sets. The handler still accepts
    unadvertised args (replay compat): providers clamp or ignore.
    """
    static_props = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
    parts: List[str] = [_GENERIC_DESCRIPTION]

    configured_model = _read_configured_video_model()
    provider = _resolve_active_provider()

    if provider is None:
        parts.append(
            "\nNo video backend is available. Calls will return an error "
            "until the user picks one via `hermes tools` → Video Generation."
        )
        return {
            "description": "\n".join(parts),
            "parameters": {
                "type": "object",
                "properties": {"prompt": static_props["prompt"]},
                "required": ["prompt"],
            },
        }

    try:
        caps = provider.capabilities() or {}
    except Exception:
        caps = {}
    try:
        models = provider.list_models() or []
    except Exception:
        models = []

    active_model = configured_model or provider.default_model()
    model_meta = next(
        (m for m in models if isinstance(m, dict) and m.get("id") == active_model),
        {},
    )

    # ---- description -------------------------------------------------
    # Model caveats surface only what differs from the backend's overall
    # capabilities. FAL's plugin uses the singular ``modality`` key for
    # single-modality entries.
    model_modalities = set(model_meta.get("modalities") or [])
    modality = model_meta.get("modality")
    if modality:
        model_modalities.add(modality)
    if "image" in model_modalities and "text" not in model_modalities:
        parts.append(
            "- this model is image-to-video only — image_url is REQUIRED; "
            "text-only calls will be rejected"
        )
    elif "text" in model_modalities and "image" not in model_modalities:
        parts.append("- this model is text-to-video only — image_url is not supported")

    effective_modalities = model_modalities or set(caps.get("modalities") or [])
    can_i2v = "image" in effective_modalities
    t2v = "text" in effective_modalities
    if can_i2v and not t2v:
        parts.append("- image-to-video only: image_url is REQUIRED")
    elif not can_i2v:
        parts.append("- text-to-video only (no image input)")

    if provider.name == "xai":
        parts.append(
            "- chaining: for edit/extend pass the public HTTPS MP4 in `video` "
            "or `public_url` from the prior Imagine result (files-cdn). For "
            "image-to-video / reference-to-video pass public image URLs the "
            "same way"
        )
        try:
            from tools.xai_http import xai_storage_notice_text

            notice = xai_storage_notice_text("video_gen")
        except Exception:
            notice = ""
        if notice:
            parts.append(f"- storage: {notice}")

    # ---- params ------------------------------------------------------
    properties: Dict[str, Any] = {"prompt": static_props["prompt"]}

    if can_i2v:
        properties["image_url"] = {
            "type": "string",
            "description": (
                "Public HTTPS URL of a still image to animate "
                "(image-to-video). Omit for text-to-video."
            ),
        }
        max_refs = int(caps.get("max_reference_images") or 0)
        if max_refs > 0:
            properties["reference_image_urls"] = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_refs,
                "description": (
                    f"Up to {max_refs} public HTTPS reference image URLs "
                    "(style or character refs)."
                ),
            }

    min_duration = model_meta.get("min_duration", caps.get("min_duration"))
    max_duration = model_meta.get("max_duration", caps.get("max_duration"))
    duration_param = dict(static_props["duration"])
    if min_duration and max_duration:
        duration_param["minimum"] = int(min_duration)
        duration_param["maximum"] = int(max_duration)
        duration_param["description"] = (
            f"Video duration in seconds ({min_duration}-{max_duration}). "
            "Omit for the provider default."
        )
    properties["duration"] = duration_param

    # Tighten enums to the active backend's actual sets when declared.
    for key, caps_key in (("aspect_ratio", "aspect_ratios"), ("resolution", "resolutions")):
        param = dict(static_props[key])
        if caps.get(caps_key):
            param["enum"] = list(caps[caps_key])
        properties[key] = param

    if caps.get("supports_negative_prompt"):
        properties["negative_prompt"] = {
            "type": "string",
            "description": "Content to avoid in the output.",
        }
    if caps.get("supports_audio"):
        properties["audio"] = {
            "type": "boolean",
            "description": (
                "Enable native audio generation (affects pricing tier)."
            ),
        }
    elif caps.get("audio_always_on"):
        parts.append(
            "- audio: native stereo audio is generated with every video "
            "(always on; no toggle) — describe the desired sound in the "
            "prompt"
        )
    if caps.get("supports_seed"):
        properties["seed"] = {
            "type": "integer",
            "description": "Seed for reproducible outputs.",
        }
    if caps.get("supports_upscale"):
        properties["upscale"] = {
            "type": "boolean",
            "description": (
                "High-resolution pass via the backend's video upscaler "
                "(~2x, extra cost/latency). Omit for native resolution."
            ),
        }

    properties["model"] = static_props["model"]

    return {
        "description": "\n".join(parts),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["prompt"],
        },
    }


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=[],
    is_async=False,
    emoji="🎬",
    dynamic_schema_overrides=_build_dynamic_video_schema,
)
