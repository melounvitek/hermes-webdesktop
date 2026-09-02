"""FAL image model catalog + upscaler constants for ``tools.image_generation_tool``.

Each entry declares how to translate the unified inputs (prompt + aspect_ratio)
into the model's native payload. ``size_style`` picks the family:
``"image_size_preset"`` (FAL preset enum), ``"aspect_ratio"`` (ratio enum),
``"gpt_literal"`` (literal "WxH" strings). ``supports`` / ``edit_supports`` are
whitelists — keys outside them are stripped so models never receive rejected
parameters. ``upscale`` (Clarity Upscaler chained after generation) is False
everywhere: Clarity redraws content (creativity 0.35) and degraded text/CJK/
faces when default-on, so upscaling is strictly per-call opt-in.
Pricing strings are as-of-commit and allowed to drift.
"""

from typing import Any, Dict

_PRESET_SIZES = {
    "landscape": "landscape_16_9",
    "square": "square_hd",
    "portrait": "portrait_16_9",
}
_ASPECT_SIZES = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}

FAL_MODELS: Dict[str, Dict[str, Any]] = {
    "fal-ai/flux-2/klein/9b": {
        "display": "FLUX 2 Klein 9B",
        "speed": "<1s",
        "strengths": "Fast, crisp text",
        "price": "$0.006/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_inference_steps": 4,
            "output_format": "png",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "seed",
            "output_format", "enable_safety_checker",
        },
        "upscale": False,
        # Image-to-image / editing: FLUX.2 [klein] 9B edit endpoint takes
        # `image_urls` (list). Natural-language edits, multi-ref.
        "edit_endpoint": "fal-ai/flux-2/klein/9b/edit",
        "edit_supports": {
            "prompt", "image_urls", "num_inference_steps", "seed",
            "output_format", "enable_safety_checker",
        },
        "max_reference_images": 9,
    },
    "fal-ai/flux-2-pro": {
        "display": "FLUX 2 Pro",
        "speed": "~6s",
        "strengths": "Studio photorealism",
        "price": "$0.03/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_inference_steps": 50,
            "guidance_scale": 4.5,
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
            "safety_tolerance": "5",
            "sync_mode": True,
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "enable_safety_checker",
            "safety_tolerance", "sync_mode", "seed",
        },
        "upscale": False,
        # Edit endpoint accepts up to 9 reference images.
        "edit_endpoint": "fal-ai/flux-2-pro/edit",
        "edit_supports": {
            "prompt", "image_urls", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "enable_safety_checker",
            "safety_tolerance", "sync_mode", "seed",
        },
        "max_reference_images": 9,
    },
    "fal-ai/z-image/turbo": {
        "display": "Z-Image Turbo",
        "speed": "~2s",
        "strengths": "Bilingual EN/CN, 6B",
        "price": "$0.005/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_inference_steps": 8,
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
            "enable_prompt_expansion": False,  # avoid the extra per-request charge
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "num_images",
            "seed", "output_format", "enable_safety_checker",
            "enable_prompt_expansion",
        },
        "upscale": False,
    },
    "fal-ai/nano-banana-pro": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "~8s",
        "strengths": "Gemini 3 Pro, reasoning depth, text rendering",
        "price": "$0.15/image (1K)",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "5",
            # "1K" is the cheapest tier; 4K doubles the per-image cost.
            # Users on Nous Subscription should stay at 1K for predictable billing.
            "resolution": "1K",
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "output_format",
            "safety_tolerance", "seed", "sync_mode", "resolution",
            "enable_web_search", "limit_generations",
        },
        "upscale": False,
        # Nano Banana Pro edit (Gemini 3 Pro Image): natural-language edits
        # with up to 2 reference images via `image_urls`.
        "edit_endpoint": "fal-ai/nano-banana-pro/edit",
        "edit_supports": {
            "prompt", "image_urls", "aspect_ratio", "num_images",
            "output_format", "safety_tolerance", "seed", "sync_mode",
            "resolution", "enable_web_search", "limit_generations",
        },
        "max_reference_images": 2,
    },
    "fal-ai/nano-banana-2": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "speed": "~3s",
        "strengths": "Fast reasoning, multilingual text, infographics",
        "price": "Lower-cost Flash tier",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "4",
            "resolution": "1K",
            "limit_generations": True,
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "output_format",
            "safety_tolerance", "seed", "sync_mode", "system_prompt",
            "resolution", "enable_web_search", "limit_generations",
            "thinking_level",
        },
        "upscale": False,
        "edit_endpoint": "fal-ai/nano-banana-2/edit",
        "edit_supports": {
            "prompt", "image_urls", "aspect_ratio", "num_images",
            "output_format", "safety_tolerance", "seed", "sync_mode",
            "system_prompt", "resolution", "enable_web_search",
            "limit_generations", "thinking_level",
        },
        "max_reference_images": 14,
    },
    "fal-ai/gpt-image-1.5": {
        "display": "GPT Image 1.5",
        "speed": "~15s",
        "strengths": "Prompt adherence",
        "price": "$0.034/image",
        "size_style": "gpt_literal",
        "sizes": {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        },
        "defaults": {
            # Quality is pinned to medium to keep portal billing predictable
            # across all users (low is too rough, high is 4-6x more expensive).
            "quality": "medium",
            "num_images": 1,
            "output_format": "png",
        },
        "supports": {
            "prompt", "image_size", "quality", "num_images", "output_format",
            "background", "sync_mode",
        },
        "upscale": False,
        # Edit endpoint: high-fidelity edits preserving composition/lighting.
        "edit_endpoint": "fal-ai/gpt-image-1.5/edit",
        "edit_supports": {
            "prompt", "image_urls", "image_size", "quality", "num_images",
            "output_format", "sync_mode",
        },
        "max_reference_images": 16,
    },
    "fal-ai/gpt-image-2": {
        "display": "GPT Image 2",
        "speed": "~20s",
        "strengths": "SOTA text rendering + CJK, world-aware photorealism",
        "price": "$0.04–0.06/image",
        # GPT Image 2 uses FAL's standard preset enum (unlike 1.5's literal
        # dimensions). We map to the 4:3 variants — the 16:9 presets
        # (1024x576) fall below GPT-Image-2's 655,360 min-pixel requirement
        # and would be rejected. 4:3 keeps us above the minimum on all
        # three aspect ratios.
        "size_style": "image_size_preset",
        "sizes": {
            "landscape": "landscape_4_3",   # 1024x768
            "square": "square_hd",            # 1024x1024
            "portrait": "portrait_4_3",       # 768x1024
        },
        "defaults": {
            # Same quality pinning as gpt-image-1.5: medium keeps Nous
            # Portal billing predictable. "high" is 3-4x the per-image
            # cost at the same size; "low" is too rough for production use.
            "quality": "medium",
            "num_images": 1,
            "output_format": "png",
        },
        "supports": {
            "prompt", "image_size", "quality", "num_images", "output_format",
            "sync_mode",
            # openai_api_key (BYOK) intentionally omitted — all users go
            # through the shared FAL billing path.
        },
        "upscale": False,
        # GPT Image 2 edit endpoint lives under the OpenAI namespace on FAL
        # (NOT fal-ai/). Takes `image_urls` (list) + optional mask. We don't
        # send `image_size` on edit so the model auto-infers from input.
        "edit_endpoint": "openai/gpt-image-2/edit",
        "edit_supports": {
            "prompt", "image_urls", "quality", "num_images", "output_format",
            "sync_mode", "mask_image_url",
        },
        "max_reference_images": 16,
    },
    "fal-ai/ideogram/v3": {
        "display": "Ideogram V3",
        "speed": "~5s",
        "strengths": "Best typography",
        "price": "$0.03-0.09/image",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "rendering_speed": "BALANCED",
            "expand_prompt": True,
            "style": "AUTO",
        },
        "supports": {
            "prompt", "image_size", "rendering_speed", "expand_prompt",
            "style", "seed",
        },
        "upscale": False,
        # Ideogram V3 edit endpoint takes `image_urls` (list).
        "edit_endpoint": "fal-ai/ideogram/v3/edit",
        "edit_supports": {
            "prompt", "image_urls", "rendering_speed", "expand_prompt",
            "style", "seed",
        },
        "max_reference_images": 1,
    },
    "fal-ai/recraft/v4/pro/text-to-image": {
        "display": "Recraft V4 Pro",
        "speed": "~8s",
        "strengths": "Design, brand systems, production-ready",
        "price": "$0.25/image",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            # V4 Pro dropped V3's required `style` enum — defaults handle taste now.
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "enable_safety_checker",
            "colors", "background_color",
        },
        "upscale": False,
    },
    "fal-ai/qwen-image": {
        "display": "Qwen Image",
        "speed": "~12s",
        "strengths": "LLM-based, complex text",
        "price": "$0.02/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_inference_steps": 30,
            "guidance_scale": 2.5,
            "num_images": 1,
            "output_format": "png",
            "acceleration": "regular",
        },
        "supports": {
            "prompt", "image_size", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "acceleration", "seed", "sync_mode",
        },
        "upscale": False,
        # Qwen edit uses the Qwen Image 2.0 Pro editing endpoint, which takes
        # `image_urls` (list) + natural-language edit instructions.
        "edit_endpoint": "fal-ai/qwen-image-2/pro/edit",
        "edit_supports": {
            "prompt", "image_urls", "num_inference_steps", "guidance_scale",
            "num_images", "output_format", "acceleration", "seed", "sync_mode",
        },
        "max_reference_images": 3,
    },
    # Krea 2 on FAL — same model family as ``plugins/image_gen/krea``, but billed
    # through FAL / the FAL managed gateway. Native ``krea-2-*`` ids route to the
    # dedicated Krea plugin instead.
    "fal-ai/krea/v2/medium/text-to-image": {
        "display": "Krea 2 Medium",
        "speed": "~15-25s",
        "strengths": "Illustration, anime, painting, expressive/artistic styles",
        "price": "$0.030 (text) / $0.035 (style refs)",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "creativity": "medium",
        },
        "supports": {
            "prompt", "aspect_ratio", "creativity", "seed",
            "image_style_references",
        },
        "upscale": False,
    },
    "fal-ai/krea/v2/large/text-to-image": {
        "display": "Krea 2 Large",
        "speed": "~25-60s",
        "strengths": "Photorealism, raw textured looks (motion blur, grain, film)",
        "price": "$0.060 (text) / $0.065 (style refs)",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "creativity": "medium",
        },
        "supports": {
            "prompt", "aspect_ratio", "creativity", "seed",
            "image_style_references",
        },
        "upscale": False,
    },
    # ─── Aug 2026 catalog expansion ────────────────────────────────────────
    # Endpoint ids, `supports` whitelists and enum defaults below are taken
    # from each model's FAL OpenAPI schema, so a key we send is a key the
    # vendor declares. Paired `/edit` apps hang off their text-to-image entry
    # rather than appearing as separate picker rows.
    "bytedance/seedream/v5/pro/text-to-image": {
        "display": "Seedream 5.0 Pro",
        "speed": "~10s",
        "strengths": "ByteDance flagship, dense layouts, native text in 14 languages",
        "price": "$0.0675/image (≤1536²)",
        "size_style": "image_size_preset",
        # Pro requires total pixels between 1024x1024 and 2048x2048 —
        # explicit ImageSize dicts keep every aspect inside that window.
        "sizes": {
            "landscape": {"width": 2048, "height": 1152},
            "square": {"width": 1536, "height": 1536},
            "portrait": {"width": 1152, "height": 2048},
        },
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "num_images", "output_format",
            "sync_mode", "enable_safety_checker",
        },
        "upscale": False,
        # Region-precise editing with up to 10 reference images.
        "edit_endpoint": "bytedance/seedream/v5/pro/edit",
        "edit_supports": {
            "prompt", "image_urls", "image_size", "num_images",
            "output_format", "sync_mode", "enable_safety_checker",
        },
        "max_reference_images": 10,
    },
    "bytedance/seedream/v5/lite/text-to-image": {
        "display": "Seedream 5.0 Lite",
        "speed": "~5s",
        "strengths": "Fast/cheap Seedream tier, high-res output",
        "price": "$0.035/image",
        "size_style": "image_size_preset",
        # Lite wants total pixels between 2560x1440 and 4096x4096. Use the
        # documented presets (FAL auto-scales if a preset is under the floor)
        # instead of hand-rolled ImageSize dicts that drift from the schema.
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_images": 1,
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "num_images", "max_images",
            "sync_mode", "enable_safety_checker",
        },
        "upscale": False,
    },
    "ideogram/v4/instant": {
        "display": "Ideogram V4 (Instant)",
        "speed": "<1s",
        "strengths": "Latest Ideogram typography, posters/logos, instant",
        "price": "$0.0075/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "expansion_model": "Medium",
            "output_format": "png",
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "expansion_model", "num_images",
            "seed", "sync_mode", "enable_safety_checker", "output_format",
        },
        "upscale": False,
    },
    "ideogram/v4/fast": {
        "display": "Ideogram V4 (Fast)",
        "speed": "~1s",
        "strengths": "Ideogram V4 quality tiers via rendering_speed",
        "price": "$0.005-0.018/MP",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "expansion_model": "Medium",
            "rendering_speed": "BALANCED",
        },
        "supports": {
            "prompt", "image_size", "expansion_model", "rendering_speed",
            "num_images", "seed", "sync_mode",
        },
        "upscale": False,
    },
    "alibaba/qwen-image-3/text-to-image": {
        "display": "Qwen Image 3",
        "speed": "~8s",
        "strengths": "Complex CN/EN text rendering, prompt-guided resolution",
        "price": "$0.04 (1K) / $0.075 (2K) per image",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "enable_prompt_expansion": False,  # avoid the LLM rewrite surprise
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "negative_prompt", "image_size", "num_images",
            "seed", "sync_mode", "output_format",
            "enable_prompt_expansion", "enable_safety_checker",
        },
        "upscale": False,
        # Qwen Image 3 edit: 1-3 reference images, identity-preserving edits.
        "edit_endpoint": "alibaba/qwen-image-3/edit",
        "edit_supports": {
            "prompt", "image_urls", "negative_prompt", "num_images",
            "seed", "sync_mode", "output_format",
            "enable_prompt_expansion", "enable_safety_checker",
        },
        "max_reference_images": 3,
    },
    "microsoft/mai-image-2.5-pro": {
        "display": "MAI Image 2.5 Pro",
        "speed": "~10s",
        "strengths": "Microsoft flagship, hero imagery, precise typography",
        "price": "~$0.17/image",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "output_format",
            "sync_mode",
        },
        "upscale": False,
    },
    "google/nano-banana-2-lite": {
        "display": "Nano Banana 2 Lite",
        "speed": "<2s",
        "strengths": "Gemini image family, sub-2s, 14 aspect ratios incl. extreme",
        "price": "~$0.04/image (1K fixed)",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "5",
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "seed",
            "output_format", "safety_tolerance", "sync_mode",
            "system_prompt", "limit_generations", "thinking_level",
        },
        "upscale": False,
        # Fast multi-turn local edits with reference images via `image_urls`.
        "edit_endpoint": "google/nano-banana-2-lite/edit",
        "edit_supports": {
            "prompt", "image_urls", "aspect_ratio", "num_images",
            "seed", "output_format", "safety_tolerance", "sync_mode",
            "system_prompt",
        },
        "max_reference_images": 4,
    },
    "fal-ai/recraft/v4.1/text-to-image": {
        "display": "Recraft V4.1",
        "speed": "~8s",
        "strengths": "Design-first raster, brand systems, editorial",
        "price": "$0.035/image",
        "size_style": "image_size_preset",
        "sizes": _PRESET_SIZES,
        "defaults": {
            "enable_safety_checker": False,
        },
        "supports": {
            "prompt", "image_size", "enable_safety_checker",
            "colors", "background_color",
        },
        "upscale": False,
    },
    "xai/grok-imagine-image/v2.0/text-to-image": {
        "display": "Grok Imagine Image 2.0",
        "speed": "~5s",
        "strengths": "xAI. Design-grade typography/layout, instruction following",
        "price": "$0.06/image (1K medium)",
        "size_style": "aspect_ratio",
        "sizes": _ASPECT_SIZES,
        "defaults": {
            "num_images": 1,
            "output_format": "png",
            # 1k + medium is the cheapest sensible tier ($0.06/image);
            # 2k roughly +33% per image.
            "resolution": "1k",
            "quality": "medium",
        },
        "supports": {
            "prompt", "aspect_ratio", "num_images", "output_format",
            "resolution", "quality", "sync_mode",
        },
        "upscale": False,
        # Edit endpoint takes `image_urls` (max 3) + the same knobs;
        # aspect_ratio defaults to "auto" (follows the first input image),
        # so we don't send it on edits.
        "edit_endpoint": "xai/grok-imagine-image/v2.0/edit",
        "edit_supports": {
            "prompt", "image_urls", "num_images", "output_format",
            "resolution", "quality", "sync_mode",
        },
        "max_reference_images": 3,
    },
}


# Fastest reasonable option; cheap and sub-1s.
DEFAULT_MODEL = "fal-ai/flux-2/klein/9b"

DEFAULT_ASPECT_RATIO = "landscape"
VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")

# Clarity Upscaler settings.
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18
