"""Routing helpers for inbound user-attached images.

Two modes:

  native  — attach images as OpenAI-style ``image_url`` content parts on the
            user turn; provider adapters translate these to vendor formats.
  text    — run ``vision_analyze`` on each image up-front and prepend the
            (lossy) description to the user's text. Still the right choice
            for non-vision models.

:func:`decide_image_input_mode` decides once per message turn from
``agent.image_input_mode`` (``auto`` | ``native`` | ``text``, default ``auto``)
plus the active model's capability metadata. In ``auto`` mode:

  - An explicitly configured ``auxiliary.vision`` backend is the DE-FACTO route
    (``text``): a user who named a dedicated vision model wants it used even
    when the main model has native vision. ``image_input_mode: native`` is the
    absolute override.
  - Otherwise, ``supports_vision=True`` (config override or catalog) → native.
  - Otherwise text via the default vision_analyze flow.

``vision_analyze`` stays surfaced as a tool in every session so skills that
chain it keep working; routing only affects how *user-attached images on the
current turn* are presented to the main model.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "native", "text"})


# Extensions extract_image_refs() auto-attaches. Kept tight: documents/archives
# are excluded because the gateway routes them via send_document, and we never
# want a PDF attached as a vision part.
_IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic",
)
_IMAGE_EXT_PATTERN = "|".join(e.lstrip(".") for e in _IMAGE_EXTS)

# Absolute / home-relative local image path — same shape as gateway's
# extract_local_files(): anchored to ``~/`` or ``/``, the lookbehind skips
# matches inside URLs, case-insensitive extension.
_LOCAL_IMAGE_PATH_RE = re.compile(
    r"(?<![/:\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:" + _IMAGE_EXT_PATTERN + r")\b",
    re.IGNORECASE,
)

# http(s) URL ending in an image extension, optional query string. Strict
# ``http(s)://`` so ``file://`` and other schemes are not grabbed.
_IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>\"']+?\.(?:" + _IMAGE_EXT_PATTERN + r")(?:\?[^\s<>\"']*)?",
    re.IGNORECASE,
)


def extract_image_refs(text: str) -> Tuple[List[str], List[str]]:
    """Scan free-form text for image references the model should see.

    Returns ``(local_paths, urls)``, each order-preserving and deduplicated.
    Local paths (``/`` or ``~/``) must exist on disk as files; URLs are not
    validated (the provider fetches them). Matches inside fenced code blocks
    and inline backticks are skipped so pasted example snippets aren't treated
    as live attachments (mirrors ``BaseAdapter.extract_local_files``).
    """
    if not isinstance(text, str) or not text:
        return [], []

    code_spans: list[tuple[int, int]] = [
        (m.start(), m.end())
        for pattern, flags in ((r"```[^\n]*\n.*?```", re.DOTALL), (r"`[^`\n]+`", 0))
        for m in re.finditer(pattern, text, flags)
    ]

    def _in_code(pos: int) -> bool:
        return any(s <= pos < e for s, e in code_spans)

    local_paths: list[str] = []
    for match in _LOCAL_IMAGE_PATH_RE.finditer(text):
        if _in_code(match.start()):
            continue
        expanded = os.path.expanduser(match.group(0))
        try:
            if not os.path.isfile(expanded):
                continue
        except OSError:
            # ENAMETOOLONG / EINVAL on pathological inputs — skip rather than crash.
            continue
        if expanded not in local_paths:
            local_paths.append(expanded)

    urls: list[str] = []
    for match in _IMAGE_URL_RE.finditer(text):
        if _in_code(match.start()):
            continue
        # Trailing punctuation is almost certainly prose ("see https://x/a.png.").
        url = match.group(0).rstrip(".,;:!?)]>")
        if url not in urls:
            urls.append(url)

    return local_paths, urls


# Strict YAML/JSON boolean coercion for capability overrides. ``bool("false")``
# is True, so a quoted ``supports_vision: "false"`` would silently enable native
# routing on a model that can't handle it. Accept only YAML boolean tokens,
# real bools and 0/1; anything else is None so the caller falls through to
# models.dev rather than honouring garbage.
_TRUE_TOKENS = frozenset({"true", "yes", "on", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "off", "0"})


def _coerce_capability_bool(raw: Any) -> Optional[bool]:
    """Return True/False for recognised boolean values, None otherwise."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw) if raw in (0, 1) else None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
    return None


def _dict_or_empty(raw: Any) -> Dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _clean_str(raw: Any) -> str:
    return str(raw or "").strip()


def _runtime_main(key: str) -> str:
    """Stripped context-local main-runtime value, or "" when unavailable."""
    try:
        from agent.auxiliary_client import _runtime_main_value

        return _clean_str(_runtime_main_value(key))
    except Exception:
        return ""


def _model_supports_vision_override(models_cfg: Any, model: str) -> Optional[bool]:
    """Per-model ``supports_vision`` (or ``vision`` alias) from a ``models`` mapping."""
    per_model = _dict_or_empty(_dict_or_empty(models_cfg).get(model))
    return _coerce_capability_bool(per_model.get("supports_vision", per_model.get("vision")))


def _custom_provider_entries(cfg: Dict[str, Any], names: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """Yield legacy ``custom_providers`` list entries whose ``name`` matches ``names``.

    Iterates ``names`` in the given priority order (outer loop) so list order
    cannot let a persisted default shadow the live route.
    """
    custom_providers = cfg.get("custom_providers")
    if not isinstance(custom_providers, list):
        return
    entries = [e for e in custom_providers if isinstance(e, dict)]
    for name in names:
        wanted = name.strip().lower()
        for entry in entries:
            if _clean_str(entry.get("name")).lower() == wanted:
                yield entry


def _supports_vision_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Resolve user-declared vision capability from config.yaml; None when unset.

    First hit wins: ``model.supports_vision`` → ``providers.<p>.models.<model>``
    → legacy ``custom_providers[].models.<model>``. Named custom providers are
    rewritten to ``provider="custom"`` at runtime while config keeps the user's
    name under ``model.provider``, so the requested, runtime and config
    identities are all tried, plus the bare ``<name>`` of any ``custom:<name>``.
    """
    if not isinstance(cfg, dict):
        return None

    model_cfg = _dict_or_empty(cfg.get("model"))
    top = _coerce_capability_bool(model_cfg.get("supports_vision"))
    if top is not None:
        return top

    provider_candidates: List[str] = []
    for candidate in (requested_provider, provider, _clean_str(model_cfg.get("provider"))):
        if candidate:
            provider_candidates.append(candidate)
            if candidate.startswith("custom:") and candidate[len("custom:"):]:
                provider_candidates.append(candidate[len("custom:"):])
    provider_candidates = list(dict.fromkeys(provider_candidates))

    providers_cfg = _dict_or_empty(cfg.get("providers"))
    for p in provider_candidates:
        coerced = _model_supports_vision_override(_dict_or_empty(providers_cfg.get(p)).get("models"), model)
        if coerced is not None:
            return coerced

    for entry in _custom_provider_entries(cfg, provider_candidates):
        coerced = _model_supports_vision_override(entry.get("models"), model)
        if coerced is not None:
            return coerced

    return None


def _resolve_inference_value(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    key: str,
    *,
    runtime_ok: Callable[[str], bool],
) -> str:
    """Shared resolution for ``base_url`` / ``api_key`` of the active inference provider.

    Order: context-local runtime value (when ``runtime_ok`` accepts it) →
    ``model.<key>`` → ``providers.<name>.<key>`` → ``custom_providers[].<key>``,
    where ``<name>`` covers the provider and ``model.provider`` in both bare
    and ``custom:``-prefixed forms.
    """
    runtime = _runtime_main(key)
    if runtime and runtime_ok(runtime):
        return runtime

    if not isinstance(cfg, dict):
        return ""

    model_cfg = _dict_or_empty(cfg.get("model"))
    value = _clean_str(model_cfg.get(key))
    if value:
        return value

    candidate_names: set[str] = set()
    for p in filter(None, (provider, _clean_str(model_cfg.get("provider")))):
        candidate_names.add(p)
        if p.lower().startswith("custom:"):
            candidate_names.add(p.split(":", 1)[1])
        else:
            candidate_names.add(f"custom:{p}")

    providers_cfg = cfg.get("providers")
    if isinstance(providers_cfg, dict):
        for name in candidate_names:
            entry = providers_cfg.get(name)
            if isinstance(entry, dict):
                value = _clean_str(entry.get(key))
                if value:
                    return value

    custom_providers = cfg.get("custom_providers")
    if isinstance(custom_providers, list):
        lowered = {n.lower() for n in candidate_names}
        for entry_raw in custom_providers:
            if not isinstance(entry_raw, dict):
                continue
            entry_name = _clean_str(entry_raw.get("name"))
            if entry_name not in candidate_names and entry_name.lower() not in lowered:
                continue
            value = _clean_str(entry_raw.get(key))
            if value:
                return value

    return ""


def _resolve_inference_base_url(
    cfg: Optional[Dict[str, Any]],
    provider: str,
) -> str:
    """Best-effort base URL for the active inference provider.

    The runtime base_url is only trusted when it belongs to the requested
    provider (or no provider was requested).
    """
    requested_provider = _clean_str(provider).lower()

    def _runtime_ok(_: str) -> bool:
        return not requested_provider or requested_provider == _runtime_main("provider").lower()

    return _resolve_inference_value(cfg, provider, "base_url", runtime_ok=_runtime_ok)


def _resolve_inference_api_key(
    cfg: Optional[Dict[str, Any]],
    provider: str,
) -> str:
    """Best-effort API key for the active inference provider.

    Mirrors :func:`_resolve_inference_base_url` so the key matches the base URL
    actually probed; otherwise the local server-type probe hits a keyed remote
    endpoint without Authorization and sprays 401s on every image turn.
    """
    return _resolve_inference_value(cfg, provider, "api_key", runtime_ok=lambda _: True)


def _should_probe_ollama_vision(
    provider: str, base_url: str, api_key: str = ""
) -> bool:
    """True when the active provider likely fronts a local Ollama server.

    Server-fingerprint probing is only valid for LOCAL endpoints: remote
    OpenAI-compatible APIs (sglang, vLLM) expose Ollama-compat routes that can
    misidentify, and probing them without an api_key returns 401 on every leg.
    """
    if (provider or "").strip().lower() == "ollama":
        return True
    if not base_url:
        return False
    try:
        from agent.model_metadata import detect_local_server_type, is_local_endpoint

        if not is_local_endpoint(base_url):
            return False
        # Forward the key: an unauthorized probe can never produce a positive verdict.
        return detect_local_server_type(base_url, api_key=api_key) == "ollama"
    except Exception:
        return False


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes (default ``auto``)."""
    if isinstance(raw, str) and raw.strip().lower() in _VALID_MODES:
        return raw.strip().lower()
    return "auto"


def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when the user configured a specific ``auxiliary.vision`` backend.

    An explicit backend is the DE-FACTO image route in ``auto`` mode even when
    the main model could take images natively. ``auto``/empty provider with no
    model and no base_url is not explicit.
    """
    if not isinstance(cfg, dict):
        return False
    aux = cfg.get("auxiliary") or {}
    if not isinstance(aux, dict):
        return False
    vision = aux.get("vision") or {}
    if not isinstance(vision, dict):
        return False

    provider = _clean_str(vision.get("provider")).lower()
    return not (
        provider in {"", "auto"}
        and not _clean_str(vision.get("model"))
        and not _clean_str(vision.get("base_url"))
    )


def _lookup_supports_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Return True/False if vision capability can be resolved, None if unknown.

    Order: config ``supports_vision`` override → managed local runtime →
    models.dev catalog → Ollama probe for local endpoints.
    """
    # Named custom providers are canonicalized to ``provider="custom"``; the
    # original name lives in the context-local main runtime. Borrow it only on an
    # exact provider+model match so background/auxiliary lookups never take
    # another turn's identity.
    if not requested_provider:
        if (
            _runtime_main("provider").lower() == _clean_str(provider).lower()
            and _runtime_main("model") == _clean_str(model)
        ):
            requested_provider = _runtime_main("requested_provider")

    override = _supports_vision_override(
        cfg,
        provider,
        model,
        requested_provider=requested_provider,
    )
    if override is not None:
        return override
    if not provider or not model:
        return None

    # Managed local runtime: the server receiving the image is the authority on
    # whether it can see (its /props reports modalities). Cloud catalogs have
    # never heard of a local GGUF, so without this every local model reads as
    # text-only and screenshots detour to a cloud auxiliary.
    try:
        from hermes_cli.local_runtime.capabilities import (
            is_managed_provider,
            managed_model_supports_vision,
        )

        if is_managed_provider(provider, _resolve_inference_base_url(cfg, provider) or ""):
            managed = managed_model_supports_vision(model)
            if managed is not None:
                return managed
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("image_routing: managed-runtime caps lookup failed for %s:%s — %s",
                     provider, model, exc)

    caps = None
    try:
        from agent.models_dev import get_model_capabilities
        # allow_network=True on purpose: this runs only when an image needs
        # routing, and the text-only-main guard depends on catalog data — a cold
        # cache returning "unknown" would reintroduce attempting the call. The
        # fetch is cached (4h TTL) and backoff-limited.
        caps = get_model_capabilities(provider, model, allow_network=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("image_routing: caps lookup failed for %s:%s — %s", provider, model, exc)
    if caps is not None:
        return bool(caps.supports_vision)

    base_url = _resolve_inference_base_url(cfg, provider)
    if not base_url and (provider or "").strip().lower() == "ollama":
        base_url = "http://localhost:11434/v1"

    resolved_api_key = _resolve_inference_api_key(cfg, provider)

    if _should_probe_ollama_vision(provider, base_url, api_key=resolved_api_key):
        try:
            from agent.model_metadata import query_ollama_supports_vision

            ollama_vision = query_ollama_supports_vision(
                model, base_url, api_key=resolved_api_key
            )
            if ollama_vision is not None:
                return ollama_vision
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "image_routing: ollama vision probe failed for %s:%s — %s",
                provider,
                model,
                exc,
            )
    return None


def decide_image_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
    *,
    requested_provider: str = "",
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn.

    Args:
      provider: active inference provider ID (e.g. ``"anthropic"``).
      model:    active model slug as sent to the provider.
      cfg:      loaded config.yaml dict, or None (behaves as auto).
      requested_provider: provider identity before runtime canonicalization.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("image_input_mode"))

    if mode_cfg != "auto":
        return mode_cfg

    # auto: an explicit auxiliary.vision backend wins (see module docstring);
    # native remains the default for unconfigured installs.
    if _explicit_aux_vision_override(cfg):
        return "text"
    if requested_provider:
        supports = _lookup_supports_vision(
            provider,
            model,
            cfg,
            requested_provider=requested_provider,
        )
    else:
        # Keep the three-argument call contract for callers/tests that replace
        # the capability lookup hook.
        supports = _lookup_supports_vision(provider, model, cfg)
    return "native" if supports is True else "text"


# Image size handling is REACTIVE: attach at full size regardless of provider
# and let ``run_agent._try_shrink_image_parts_in_messages`` shrink + retry on
# rejection (e.g. Anthropic's 5 MB ceiling as HTTP 400). Provider ceilings are
# partial and evolving (OpenAI 49 MB+, Anthropic 5 MB, Gemini 100 MB); a
# proactive table would go stale and silently degrade quality for providers
# that would have accepted the full image — worse than one extra API call.


# Magic-byte signatures, checked in order. Filename-based detection is
# unreliable when platforms lie about content-type (Discord serves PNG as
# ``image/webp`` for proxied stickers); Anthropic rejects a media_type that
# does not match the bytes with HTTP 400, so we sniff.
_HEIC_BRANDS = frozenset({
    b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis",
})
_MAGIC_PREFIXES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_mime_from_bytes(raw: bytes) -> Optional[str]:
    """Detect image MIME from magic bytes; None if unrecognised."""
    if not raw:
        return None
    for prefix, mime in _MAGIC_PREFIXES:
        if raw.startswith(prefix):
            return mime
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    # ISO-BMFF family (HEIC/HEIF/AVIF): 'ftyp' at 4..8, major brand at 8..12.
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12]
        if brand in {b"avif", b"avis"}:
            return "image/avif"
        if brand in _HEIC_BRANDS:
            return "image/heic"
    if raw[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if raw[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"
    # SVG is text: look for an <svg tag near the start (skip BOM/whitespace).
    head = raw[:512].lstrip().lower()
    if (head.startswith(b"<?xml") or head.startswith(b"<svg")) and b"<svg" in head:
        return "image/svg+xml"
    return None


# Formats every major vision provider accepts natively. Anything else must be
# transcoded to PNG before declaring media_type or the provider returns HTTP
# 400 and the whole turn fails. Chat platforms freely accept AVIF (Chromium
# screenshots), HEIC (iPhone), TIFF, BMP and ICO, so users do hit this. SVG is
# vector — Pillow cannot rasterize it — and is skipped (logged) instead.
_UNIVERSALLY_SUPPORTED_MIMES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
})


def _transcode_to_png(raw: bytes) -> Optional[bytes]:
    """Decode image bytes with Pillow and re-encode as PNG; None when impossible.

    HEIC/HEIF and AVIF need optional Pillow plugins, registered on demand; a
    missing plugin just looks like "Pillow can't decode this" so the caller
    skips the image and the rest of the turn proceeds.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.info(
            "image_routing: Pillow not installed; cannot transcode "
            "non-standard image format to PNG. Install with `pip install Pillow` "
            "(and `pillow-heif` / `pillow-avif-plugin` for those formats)."
        )
        return None
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        import pillow_avif  # type: ignore  # noqa: F401  -- registers AVIF on import
    except Exception:
        pass
    try:
        from io import BytesIO

        with Image.open(BytesIO(raw)) as im:
            # Normalise exotic modes to RGBA so PNG can serialise and any
            # source transparency survives.
            if im.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
                im = im.convert("RGBA")
            buf = BytesIO()
            im.save(buf, format="PNG", optimize=False)
            return buf.getvalue()
    except Exception as exc:
        logger.info(
            "image_routing: Pillow could not transcode image to PNG -- %s", exc
        )
        return None


_SUFFIX_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _guess_mime(path: Path, raw: Optional[bytes] = None) -> str:
    """Image MIME for *path*: magic bytes (authoritative) → ``mimetypes`` → suffix → jpeg."""
    if raw is not None:
        sniffed = _sniff_mime_from_bytes(raw)
        if sniffed:
            return sniffed
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    # mimetypes on some Linux distros mis-maps .jpg; default to jpeg.
    return _SUFFIX_MIMES.get(path.suffix.lower(), "image/jpeg")


def _file_to_data_url(path: Path) -> Optional[str]:
    """Encode a local image as a base64 data URL at its native size.

    Size is NOT limited here (the agent retry loop shrinks on the provider's
    first rejection, so lenient providers pay no silent quality tax). Format
    compatibility IS handled: MIMEs outside the accepted set are transcoded to
    PNG. Returns None when the file can't be read, is blocked by the read
    guard, or can't be transcoded — the caller reports it in ``skipped``.
    """
    try:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(str(path))
    except ValueError as exc:
        logger.warning("image_routing: blocked local image attachment %s -- %s", path, exc)
        return None
    except Exception:
        # Keep attachment routing best-effort if the guard itself is unavailable.
        pass

    try:
        raw = path.read_bytes()
    except Exception as exc:
        logger.warning("image_routing: failed to read %s — %s", path, exc)
        return None
    mime = _guess_mime(path, raw=raw)
    accepted = _UNIVERSALLY_SUPPORTED_MIMES
    # The managed local server decodes fewer formats (no WebP — and a WebP part
    # fails SILENTLY: the model confabulates a description). Narrow the accepted
    # set so those formats transcode here instead of vanishing server-side.
    try:
        from agent.auxiliary_client import _runtime_main_value
        from hermes_cli.local_runtime.capabilities import (
            ACCEPTED_IMAGE_MIMES,
            is_managed_provider,
        )

        if is_managed_provider(
                str(_runtime_main_value("provider") or ""),
                str(_runtime_main_value("base_url") or "")):
            accepted = ACCEPTED_IMAGE_MIMES
    except Exception:  # noqa: BLE001 — best-effort narrowing only
        pass
    if mime not in accepted:
        transcoded = _transcode_to_png(raw)
        if transcoded is None:
            logger.warning(
                "image_routing: %s is %s which is not accepted by the "
                "active provider and could not be transcoded to PNG; "
                "skipping this attachment.",
                path, mime,
            )
            return None
        logger.info(
            "image_routing: transcoded %s (%s) -> image/png for provider compatibility",
            path.name, mime,
        )
        raw = transcoded
        mime = "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
    image_urls: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Local paths are embedded as base64 ``data:`` URLs; remote URLs pass through
    verbatim. When at least one image attaches, a single text part combines the
    caption (or a neutral default) with one hint per image —
    ``[Image attached at: <path>]`` / ``[Image attached: <url>]`` — giving the
    model a string handle for tools that take an image path/URL, mirroring the
    text-mode hint from ``Runner._enrich_message_with_vision``.

    Returns ``(content_parts, skipped)``; ``skipped`` holds local paths that
    could not be read. URLs are never skipped (not validated here).
    """
    skipped: List[str] = []
    image_parts: List[Dict[str, Any]] = []
    hint_lines: List[str] = []

    for raw_path in image_paths:
        p = Path(raw_path)
        data_url = _file_to_data_url(p) if p.exists() and p.is_file() else None
        if not data_url:
            skipped.append(str(raw_path))
            continue
        image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        hint_lines.append(f"[Image attached at: {raw_path}]")

    for url in image_urls or []:
        url = (url or "").strip()
        if not url:
            continue
        image_parts.append({"type": "image_url", "image_url": {"url": url}})
        hint_lines.append(f"[Image attached: {url}]")

    text = (user_text or "").strip()

    if image_parts:
        base_text = text or "What do you see in this image?"
        combined_text = f"{base_text}\n\n" + "\n".join(hint_lines)
        return [{"type": "text", "text": combined_text}, *image_parts], skipped

    # No images attached — plain text-only behaviour.
    return ([{"type": "text", "text": text}] if text else []), skipped


__all__ = [
    "decide_image_input_mode",
    "build_native_content_parts",
    "extract_image_refs",
]
