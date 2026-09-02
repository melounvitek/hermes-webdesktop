"""browser_vision helpers: Lightpanda pre-route, native provider vision, auxiliary-LLM screenshot analysis.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools.browser_tool_origin import origin_module as _origin


def _vision_mode_label() -> str:
    _bt = _origin()
    _cp = _bt._get_cloud_provider()
    return "local" if _cp is None else f"cloud ({_cp.provider_name()})"


def _lightpanda_vision_preroute(
    effective_task_id: str, annotate: bool, screenshot_path: Path,
) -> Tuple[bool, Optional[str], Path]:
    """Capture the vision screenshot through the Chrome fallback when Lightpanda is the engine.

    Lightpanda has no graphical renderer, so the normal path would fail with a
    CDP error or return a placeholder PNG. Returns ``(prerouted, fallback_warning,
    screenshot_path)``; on fallback failure ``prerouted`` is False and the caller
    takes the normal screenshot path (forcing Chrome) so ``_run_browser_command``
    still produces the standard fallback metadata/error.
    """
    _bt = _origin()
    engine = _bt._get_browser_engine()
    if engine != "lightpanda" or not _bt._should_inject_engine(engine):
        return False, None, screenshot_path
    _bt.logger.debug("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)")
    screenshot_args = ["--annotate"] if annotate else []
    fb_result = _bt._chrome_fallback_screenshot(effective_task_id, screenshot_args, _bt._get_command_timeout())
    fb_result = _bt._annotate_lightpanda_fallback(fb_result, _bt._LP_VISION_FALLBACK_REASON)
    if not fb_result.get("success"):
        _bt.logger.warning("Lightpanda Chrome fallback vision screenshot failed: %s", fb_result.get("error"))
        return False, None, screenshot_path
    fb_path = fb_result.get("data", {}).get("path", "")
    if fb_path and os.path.exists(fb_path):
        import uuid as uuid_mod
        from hermes_constants import get_hermes_dir

        screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        persistent_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
        shutil.copy2(fb_path, persistent_path)
        screenshot_path = persistent_path
    return True, fb_result.get("fallback_warning"), screenshot_path


def _native_vision_result(
    screenshot_path: Path, question: str, annotate: bool,
    result: Dict[str, Any], lp_fallback_warning: Optional[str],
) -> Dict[str, Any]:
    """Multimodal tool-result envelope: the main model inspects the pixels itself.

    History-reuse cap: this embed is baked into the tool result and re-sent on
    every later turn, exactly like vision_analyze's native path — apply the same
    proactive resize so full-res screenshots can't enter immutable history
    uncapped. The helper's stat/dimension quick-estimate skips the resize when
    already under both caps; without Pillow it fails open to the raw bytes.
    """
    from tools.vision_tools import (
        _EMBED_MAX_DIMENSION,
        _EMBED_TARGET_BYTES,
        _build_native_vision_tool_result,
        _resize_image_for_vision,
    )

    data_url = _resize_image_for_vision(
        screenshot_path,
        mime_type="image/png",
        max_base64_bytes=_EMBED_TARGET_BYTES,
        max_dimension=_EMBED_MAX_DIMENSION,
        force_jpeg=True,
    )
    native_result = _build_native_vision_tool_result(
        image_url=str(screenshot_path),
        question=question,
        image_data_url=data_url,
        image_size_bytes=screenshot_path.stat().st_size,
    )
    meta = native_result.setdefault("meta", {})
    meta["screenshot_path"] = str(screenshot_path)
    if lp_fallback_warning:
        meta["fallback_warning"] = lp_fallback_warning
    if annotate and result.get("data", {}).get("annotations"):
        meta["annotations"] = result["data"]["annotations"]
    native_result["text_summary"] = (
        f"{native_result.get('text_summary', '')} " f"Screenshot path: {screenshot_path}"
    ).strip()
    return native_result


def _analyze_screenshot_with_aux_llm(screenshot_path: Path, question: str) -> str:
    """One-shot aux vision-LLM analysis (not baked into history), secret-redacted.

    Encodes at full resolution; on a size-related provider rejection the image
    is downscaled once and retried. Timeout/temperature come from
    ``auxiliary.vision.*`` — local vision models (llama.cpp, ollama) can take
    well over 30s, so the default timeout is generous.
    """
    _bt = _origin()
    import base64

    vision_prompt = (
        f"You are analyzing a screenshot of a web browser.\n\n"
        f"User's question: {question}\n\n"
        f"Provide a detailed and helpful answer based on what you see in the screenshot. "
        f"If there are interactive elements, describe them. If there are verification challenges "
        f"or CAPTCHAs, describe what type they are and what action might be needed. "
        f"Focus on answering the user's specific question."
    )
    _screenshot_bytes = screenshot_path.read_bytes()
    _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{_screenshot_b64}"
    vision_model = _bt._get_vision_model()
    _bt.logger.debug("browser_vision: analysing screenshot (%d bytes)",
                 len(_screenshot_bytes))

    vision_timeout = 120.0
    vision_temperature = 0.1
    try:
        from hermes_cli.config import load_config
        _vision_cfg = _bt.cfg_get(load_config(), "auxiliary", "vision", default={})
        _vt = _vision_cfg.get("timeout")
        if _vt is not None:
            vision_timeout = float(_vt)
        _vtemp = _vision_cfg.get("temperature")
        if _vtemp is not None:
            vision_temperature = float(_vtemp)
    except Exception:
        pass

    call_kwargs = {
        "task": "vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": vision_temperature,
        "timeout": vision_timeout,
    }
    if vision_model:
        call_kwargs["model"] = vision_model
    try:
        response = _bt._lazy_call_llm(**call_kwargs)
    except Exception as _api_err:
        from tools.vision_tools import (
            _is_image_size_error, _resize_image_for_vision, _RESIZE_TARGET_BYTES,
        )
        if not (_is_image_size_error(_api_err) and len(data_url) > _RESIZE_TARGET_BYTES):
            raise
        _bt.logger.info(
            "Vision API rejected screenshot (%.1f MB); "
            "auto-resizing to ~%.0f MB and retrying...",
            len(data_url) / (1024 * 1024),
            _RESIZE_TARGET_BYTES / (1024 * 1024),
        )
        data_url = _resize_image_for_vision(screenshot_path, mime_type="image/png")
        call_kwargs["messages"][0]["content"][1]["image_url"]["url"] = data_url
        response = _bt._lazy_call_llm(**call_kwargs)

    analysis = (response.choices[0].message.content or "").strip()
    # Redact secrets the vision LLM may have read from the screenshot.
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(analysis)
