"""Attachment staging for prompt.submit / prompt.attach: image sniffing, size caps, per-session attachment dirs, gateway attachment path resolution.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import re as _re

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes → file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)

# Context-ref values containing any of these must be quoted (desktop formatRefValue parity).
_ATTACHMENT_REF_NEEDS_QUOTING_RE = _re.compile(r"""[\s()\[\]{}<>"'`]""")
del _re  # bodies are rebound onto server globals: import inside functions only


def _decode_attach_base64(raw: str, *, mime_prefix: str) -> bytes | None:
    """Decode a base64 payload, optionally ``data:<mime_prefix>...;base64,``-wrapped,
    tolerating embedded whitespace. ``None`` when not valid base64."""
    import base64 as _base64
    import re as _re

    cleaned = raw.strip()
    m = _re.match(
        rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$", cleaned, _re.DOTALL,
    )
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _decode_attach_payload(rid, raw_b64: str, *, mime_prefix: str, max_bytes: int,
                           label: str, empty_msg: str):
    """``(bytes, None)`` or ``(None, error)`` for an upload: 4017 on bad/empty
    base64, 4018 over *max_bytes*."""
    data = _decode_attach_base64(raw_b64, mime_prefix=mime_prefix)
    if data is None:
        return None, _err(rid, 4017, "data is not valid base64")
    if not data:
        return None, _err(rid, 4017, empty_msg)
    if len(data) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        return None, _err(rid, 4018, f"{label} too large ({len(data)} bytes; cap is {mb} MB)")
    return data, None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Extension from the filename hint, else magic bytes (WebP needs the RIFF/WEBP
    container check), else ``.png``."""
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return ".png"


def _allowed_image_extensions() -> frozenset[str]:
    try:
        from cli import _IMAGE_EXTENSIONS

        return frozenset(_IMAGE_EXTENSIONS)
    except Exception:
        return frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def _session_home_dir(session: dict, name: str) -> Path:
    """``<session home>/<name>``, anchored on the session's stored ``profile_home``.

    Attach RPCs run BEFORE ``prompt.submit`` installs the profile HERMES_HOME
    override, so ``get_hermes_home()`` would return the gateway's launch home —
    while the sandbox mounts and the vision host-read allowlist resolve the
    *session profile's* dirs at run time. Writing anywhere else means the agent
    can never see the file.
    """
    profile_home = session.get("profile_home")
    base = Path(profile_home) if profile_home else _hermes_home
    return base / name


def _session_images_dir(session: dict) -> Path:
    """Uploads ``images/`` dir for the session (see ``_session_home_dir``)."""
    return _session_home_dir(session, "images")


def _queue_attached_image(session: dict, img_bytes: bytes, ext: str, *, prefix: str) -> Path:
    """Write image bytes into the session images dir and append to
    ``session["attached_images"]`` so the next ``prompt.submit`` picks them up."""
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _session_images_dir(session)
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
    try:
        img_path.write_bytes(img_bytes)
    except Exception:
        session["image_counter"] = max(0, session["image_counter"] - 1)
        raise
    session.setdefault("attached_images", []).append(str(img_path))
    return img_path


def _format_ref_value(value: str) -> str:
    """Quote a context-ref value containing whitespace/brackets/quotes so the staged
    ``@file:`` ref round-trips through ``agent.context_references``."""
    if not value or not _ATTACHMENT_REF_NEEDS_QUOTING_RE.search(value):
        return value
    if "`" not in value:
        return f"`{value}`"
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return value


def _attachment_ref_path(session: dict, target: Path) -> str:
    """Workspace-relative path for an attachment, or the absolute path if outside."""
    workspace = Path(_session_cwd(session)).resolve()
    try:
        rel = target.resolve().relative_to(workspace)
        return str(rel).replace(os.sep, "/")
    except ValueError:
        return str(target.resolve())


def _desktop_attachment_dir(session: dict) -> Path:
    """File-attachment staging dir (``attachments/``, see ``_session_home_dir``);
    registered in ``tools.credential_files._CACHE_DIRS`` and auto-mounted into
    containers, so a staged file lands where the bind mount points."""
    root = _session_home_dir(session, "attachments")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_attachment_name(name: str) -> str:
    import re as _re

    candidate = Path(str(name or "").strip()).name
    candidate = _re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "attachment"


def _unique_attachment_path(root: Path, filename: str) -> Path:
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem or "attachment"
    suffix = Path(filename).suffix
    counter = 2
    while True:
        next_candidate = root / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def _resolve_gateway_attachment_path(raw: str) -> Path | None:
    """Resolve a raw path token to a gateway-visible file, or None."""
    if not raw:
        return None
    try:
        from cli import _detect_file_drop, _resolve_attachment_path, _split_path_input
    except Exception:
        return None

    dropped = _detect_file_drop(raw)
    if dropped:
        return Path(dropped["path"]).resolve()
    path_token, _remainder = _split_path_input(raw)
    resolved = _resolve_attachment_path(path_token)
    return Path(resolved).resolve() if resolved is not None else None


def _decode_attachment_data_url(data_url: str) -> bytes:
    """Decode a ``data:<any-mime>;base64,<b64>`` payload (any media type, unlike the
    image-specific ``_decode_attach_base64``); bare base64 also accepted."""
    import base64 as _base64
    import binascii as _binascii
    import re as _re

    cleaned = (data_url or "").strip()
    m = _re.match(r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", cleaned, _re.DOTALL | _re.I)
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except (ValueError, _binascii.Error) as exc:
        raise ValueError("invalid data_url payload") from exc


def _stage_session_file_attachment(
    session: dict,
    *,
    raw_path: str,
    data_url: str,
    name: str,
) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the gateway agent.

    1. Path resolves INSIDE the session workspace → use as-is (``uploaded=False``).
    2. Gateway-visible file OUTSIDE the workspace → copy into ``attachments/``
       (bind-mounted into container backends) so ``@file:`` resolves in the sandbox.
    3. Not on the gateway (remote client disk) → decode ``data_url`` bytes into
       ``attachments/``.
    Returns ``(stored_path, uploaded)``.
    """
    workspace = Path(_session_cwd(session)).resolve()
    resolved = _resolve_gateway_attachment_path(raw_path)
    if resolved is not None:
        try:
            resolved.relative_to(workspace)
            return resolved, False
        except ValueError:
            payload = resolved.read_bytes()
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        payload = _decode_attachment_data_url(data_url)
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)

    upload_dir = _desktop_attachment_dir(session)
    target = _unique_attachment_path(upload_dir, _sanitize_attachment_name(filename))
    target.write_bytes(payload)
    return target.resolve(), True


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
