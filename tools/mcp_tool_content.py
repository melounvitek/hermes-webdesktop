"""Rendering of MCP tool-result content blocks into model-facing text: size
capping, _meta filtering, image/audio caching to MEDIA tags, resource links and
embedded resources."""

import logging
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import mcp_field, _core
from tools.mcp_tool_schema import mcp_prefixed_tool_name

logger = logging.getLogger("tools.mcp_tool")


# Hard allocation ceiling for one MCP text payload (chars): the first line of
# defense against a multi-megabyte flood being JSON-encoded and handed
# downstream. Deliberately far ABOVE the budget layer's 50K spillover threshold
# so ordinary large results reach spillover intact; only pathological floods
# are lossy-truncated here.
_MCP_HARD_RESULT_CAP_CHARS = 2_000_000


def _truncate_mcp_text_result(text: str, max_chars: int = _MCP_HARD_RESULT_CAP_CHARS) -> str:
    """Pass text at or under ``max_chars`` unchanged; otherwise keep a 40% head /
    60% tail split with an omission notice between."""
    if len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.4)
    tail_chars = max_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n\n... [MCP RESULT TRUNCATED - {omitted:,} chars omitted "
          f"out of {len(text):,} total] ...\n\n"
        + text[-tail_chars:]
    )


def _is_reserved_mcp_meta_key(key: str) -> bool:
    """True if an MCP ``_meta`` key uses a protocol-reserved prefix: a
    ``modelcontextprotocol`` or ``mcp`` label followed by at least one more
    label. A trailing one (``com.example.mcp/...``) is a vendor namespace."""
    slash = key.find("/")
    if slash <= 0:
        return False
    labels = key[:slash].split(".")
    return any(
        label in ("modelcontextprotocol", "mcp") and i < len(labels) - 1
        for i, label in enumerate(labels)
    )


def _strip_reserved_meta_keys(meta) -> "Optional[Dict[str, Any]]":
    """Drop protocol-reserved keys from ``_meta``; None if nothing model-facing
    remains or the input wasn't a mapping."""
    if not isinstance(meta, dict):
        return None
    out = {k: v for k, v in meta.items()
           if isinstance(k, str) and not _is_reserved_mcp_meta_key(k)}
    return out or None


def _mcp_image_extension_for_mime_type(mime_type: str) -> str:
    """File extension for an MCP image MIME type (``.png`` fallback)."""
    import mimetypes
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    if normalized in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return mimetypes.guess_extension(normalized) or ".png"


def _cache_mcp_image_block(block) -> str:
    """Cache an ``ImageContent`` block and return a ``MEDIA:<path>`` tag.

    Returns "" (logging, not raising) when the block isn't an image, the base64
    is malformed, or the cache rejects the bytes: one bad block must not kill
    the tool result, and the caller falls through to any text blocks.
    """
    import base64

    data = getattr(block, "data", None)
    mime_type = mcp_field(block, "mime_type", "mimeType")
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if data is None or not normalized_mime.startswith("image/"):
        return ""

    try:
        raw_bytes = base64.b64decode(data)
    except (TypeError, ValueError) as exc:
        logger.warning("MCP image block decode failed (%s): %s", normalized_mime, exc)
        return ""

    try:
        from gateway.platforms.base import cache_image_from_bytes

        image_path = cache_image_from_bytes(
            raw_bytes,
            ext=_mcp_image_extension_for_mime_type(normalized_mime),
        )
    except ImportError:
        # gateway.platforms.base unavailable (e.g. cron without gateway deps):
        # drop silently, callers get any text blocks that parsed.
        logger.debug("MCP image caching skipped — gateway.platforms.base unavailable")
        return ""
    except Exception as exc:
        logger.warning("MCP image block cache failed: %s", exc)
        return ""

    return f"MEDIA:{image_path}"


# Hard cap on decoded resource bytes from one block, so a misbehaving server
# can't fill the cache disk.
_MCP_RESOURCE_MAX_BYTES = 50 * 1024 * 1024


# Base64 expands ~4/3; reject oversized payloads BEFORE decoding so a multi-GB
# blob string is never transiently doubled in memory.
_MCP_RESOURCE_MAX_B64_CHARS = _MCP_RESOURCE_MAX_BYTES * 4 // 3 + 4


def _mcp_resource_filename(uri: str, mime_type: str) -> str:
    """Safe display filename from the URI's last path segment, used only as a
    name hint: ``cache_document_from_bytes`` re-sanitizes and prefixes it, so
    remote path components can't steer the cache location."""
    import mimetypes
    import re as _re
    from pathlib import Path
    from urllib.parse import urlparse, unquote

    name = ""
    if uri:
        try:
            name = Path(unquote(urlparse(str(uri)).path or "")).name
        except (ValueError, TypeError):
            name = ""
    # Strip control chars (hostile URIs could inject newlines/ANSI into the
    # filename and transcript marker) and cap length, preserving the extension.
    name = _re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if len(name) > 150:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 12:
            name = stem[: 150 - len(ext) - 1] + "." + ext
        else:
            name = name[:150]
    if not name or name in {".", ".."}:
        normalized = (mime_type or "").split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(normalized) or ".bin"
        name = f"resource{ext}"
    return name


def _cache_mcp_audio_block(block) -> str:
    """Cache an ``AudioContent`` block and return a ``MEDIA:`` tag; "" when not
    audio or on any failure (same fail-open contract as the image path)."""
    import base64

    data = getattr(block, "data", None)
    mime_type = str(mcp_field(block, "mime_type", "mimeType") or "").split(";", 1)[0].strip().lower()
    if data is None or not mime_type.startswith("audio/"):
        return ""
    if len(data) > _core._MCP_RESOURCE_MAX_B64_CHARS:
        return f"[MCP audio resource too large to cache: ~{len(data) * 3 // 4} bytes]"
    try:
        raw_bytes = base64.b64decode(data)
    except (TypeError, ValueError) as exc:
        logger.warning("MCP audio block decode failed (%s): %s", mime_type, exc)
        return ""
    if len(raw_bytes) > _core._MCP_RESOURCE_MAX_BYTES:
        return f"[MCP audio resource too large to cache: {len(raw_bytes)} bytes]"
    try:
        from gateway.platforms.base import cache_audio_from_bytes
        import mimetypes

        ext = (
            {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav"}.get(mime_type)
            or mimetypes.guess_extension(mime_type)
            or ".ogg"
        )
        audio_path = cache_audio_from_bytes(raw_bytes, ext=ext)
    except ImportError:
        logger.debug("MCP audio caching skipped — gateway.platforms.base unavailable")
        return ""
    except Exception as exc:
        logger.warning("MCP audio block cache failed: %s", exc)
        return ""
    return f"MEDIA:{audio_path}"


def _render_mcp_resource_block(block, server_name: str = "") -> str:
    """Render a ``ResourceLink`` or ``EmbeddedResource`` block as text.

    Embedded text → the text; embedded blob → decoded (size-capped) into the
    document cache with a path marker; link → the URI plus a pointer at the
    server's read_resource tool (no fetch here — links are only readable via
    the originating session). "" for non-resource blocks; failures are
    reported inline rather than silently dropped.
    """
    block_type = getattr(block, "type", "")

    if block_type == "resource_link" or (
        hasattr(block, "uri") and not hasattr(block, "resource") and block_type != "text"
    ):
        uri = getattr(block, "uri", None)
        if not uri:
            return ""
        name = getattr(block, "name", "") or ""
        mime = mcp_field(block, "mime_type", "mimeType", "") or ""
        details = f"uri={uri}"
        if name:
            details += f", name={name}"
        if mime:
            details += f", mimeType={mime}"
        reader = (
            mcp_prefixed_tool_name(server_name, "read_resource")
            if server_name
            else "the MCP server's read_resource tool"
        )
        return f"[MCP resource link: {details} — fetch it with {reader}]"

    resource = getattr(block, "resource", None)
    if resource is None:
        return ""

    text = getattr(resource, "text", None)
    if text is not None:
        return strip_unicode_tags(str(text))

    blob = getattr(resource, "blob", None)
    if blob is None:
        return ""

    import base64

    uri = str(getattr(resource, "uri", "") or "")
    mime = str(mcp_field(resource, "mime_type", "mimeType", "") or "")
    if len(blob) > _core._MCP_RESOURCE_MAX_B64_CHARS:
        return f"[MCP embedded resource too large to cache: ~{len(blob) * 3 // 4} bytes, uri={uri}]"
    try:
        raw_bytes = base64.b64decode(blob)
    except (TypeError, ValueError) as exc:
        logger.warning("MCP embedded resource decode failed (%s): %s", mime or uri, exc)
        return f"[MCP embedded resource could not be decoded: {mime or uri}]"
    if len(raw_bytes) > _core._MCP_RESOURCE_MAX_BYTES:
        return f"[MCP embedded resource too large to cache: {len(raw_bytes)} bytes, uri={uri}]"
    try:
        from gateway.platforms.base import cache_document_from_bytes

        path = cache_document_from_bytes(raw_bytes, _mcp_resource_filename(uri, mime))
    except ImportError:
        logger.debug("MCP resource caching skipped — gateway.platforms.base unavailable")
        return f"[MCP embedded resource received ({len(raw_bytes)} bytes, {mime or 'unknown type'}) but document cache unavailable in this process]"
    except Exception as exc:
        logger.warning("MCP embedded resource cache failed: %s", exc)
        return f"[MCP embedded resource could not be cached: {mime or uri}]"
    return f"[MCP resource saved to {path} ({mime or 'unknown type'}, {len(raw_bytes)} bytes) — read it with read_file or terminal tools]"
