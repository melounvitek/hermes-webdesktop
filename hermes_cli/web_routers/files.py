"""Managed-files, chat image upload, /api/media and /api/fs dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import base64
import binascii
import mimetypes
import os
import re
import stat
import tempfile
import asyncio
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from fastapi import APIRouter
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.web_models import ManagedFileUpload, ChatImageUpload, ManagedDirectoryCreate, ManagedFileDelete, FsWriteText
from pathlib import Path
from typing import Any, Dict, Optional

router = APIRouter()


# Directory names whose entire subtree is credential material. Both canonical
# guards deny these as directory trees, not basenames:
#   * gateway.platforms.base._ROOT_CREDENTIAL_DIRS = {"pairing", "mcp-tokens"}
#   * agent.file_safety.get_read_block_error (mcp-tokens/ prefix match)
# The managed-files API lets the browser descend into subdirs, so a
# basename-only guard would still expose e.g. ``mcp-tokens/<server>.json``
# (live MCP OAuth tokens) and ``pairing/<x>``. We match on ANY path component
# so these trees are blocked wherever they appear under the browsable root,
# without needing to resolve them relative to HERMES_HOME.
_SENSITIVE_MANAGED_DIR_NAMES = frozenset({
    "mcp-tokens",
    "pairing",
})


def _is_sensitive_filename(name: str) -> bool:
    """Return True for a basename the managed-files API must never expose.

    Covers ``.env`` / ``.env.<suffix>`` / ``.envrc`` variants plus the
    canonical Hermes credential-store basenames (see
    ``_SENSITIVE_MANAGED_FILE_BASENAMES`` above).

    Case-insensitive so ``.ENV`` / ``.Env.local`` / ``Auth.JSON`` on
    case-insensitive filesystems (macOS/Windows mounts) can't slip past
    the guard.

    Basename-only: for the directory-tree credential stores
    (``mcp-tokens/``, ``pairing/``) that the canonical guards also deny,
    use :func:`_is_sensitive_path`, which the API call sites route through.
    """
    from hermes_cli.web_server import _SENSITIVE_MANAGED_FILE_BASENAMES
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_MANAGED_FILE_BASENAMES


def _is_sensitive_path(path: Path) -> bool:
    """Return True for any path the managed-files API must never expose.

    Combines the basename denylist (:func:`_is_sensitive_filename`) with a
    credential-directory-tree check: a path is sensitive if its own basename
    is sensitive OR any of its path components is a credential directory
    (``mcp-tokens`` / ``pairing``). The component match is case-insensitive
    and needs no HERMES_HOME resolution, so it blocks these trees wherever
    they sit under the operator-configured managed root — closing the gap
    the canonical guards cover as directory trees but a basename-only check
    would miss.

    Read-side only: this guards list/read/download (the #57505 exfil surface).
    The write endpoints (upload/mkdir/delete) are a separate threat class
    handled by the write-path checks; extending this guard to them is out of
    scope for this fix.
    """
    if _is_sensitive_filename(path.name):
        return True
    return any(part.lower() in _SENSITIVE_MANAGED_DIR_NAMES for part in path.parts)


_FS_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024


_FS_TEXT_PREVIEW_MAX_BYTES = 512 * 1024


# Upper bound for the in-app spot editor's save. The editor only opens
# non-truncated text (<= the preview cap), so this is a safety ceiling against
# a pasted-in megablob, not the expected payload size.
_FS_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024


_FS_PREVIEW_LANGUAGE_BY_EXT = {
    ".c": "c",
    ".conf": "ini",
    ".cpp": "cpp",
    ".css": "css",
    ".csv": "csv",
    ".go": "go",
    ".graphql": "graphql",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".svg": "xml",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "shell",
}


_FS_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


def _fs_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FS_MIME_TYPES:
        return _FS_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _fs_looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\0" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in {9, 10, 13})
    return suspicious / len(data) > 0.12


def _fs_regular_file(path: Path) -> tuple[Path, os.stat_result]:
    from hermes_cli.web_server import _fs_path
    target = _fs_path(str(path))
    try:
        st = target.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except NotADirectoryError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    return target, st


def _fs_find_git_root(start: Path) -> str | None:
    directory = start
    for _ in range(50):
        try:
            if (directory / ".git").exists():
                return str(directory)
        except OSError:
            return None
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent
    return None


def _fs_default_cwd() -> str:
    from hermes_cli.web_server import load_config
    cfg_terminal = load_config().get("terminal") or {}
    raw = str(cfg_terminal.get("cwd") or os.environ.get("TERMINAL_CWD") or "").strip()
    if raw and raw not in {".", "auto", "cwd"}:
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.is_dir():
                return str(candidate)
        except (OSError, RuntimeError):
            pass
    return str(Path.cwd())


def _fs_git_branch(cwd: str) -> str:
    try:
        run_kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 2,
            "check": False,
        }
        if sys.platform == "win32":
            run_kwargs["creationflags"] = windows_hide_flags()
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            **run_kwargs,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _media_serve_roots() -> list[Path]:
    """Directories ``GET /api/media`` is allowed to read from.

    Confined to where the agent and attach pipeline actually write media on the
    gateway host — its images dir and cache subtree. This stops an authenticated
    client from reading image-extension files anywhere on disk (e.g. a renamed
    key or a screenshot outside the cache) merely because the suffix passes the
    allowlist.
    """
    from hermes_cli.web_server import get_hermes_home
    home = get_hermes_home()
    roots = [home / "images", home / "screenshots", home / "cache"]
    out: list[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except (OSError, RuntimeError):
            continue
    return out


@router.get("/api/media")
async def get_media(path: str):
    """Return a gateway-local image file as a base64 data URL.

    Lets remote clients (the desktop app over the network, or the web dashboard
    in a browser) display images the agent wrote to *this* machine's filesystem
    — they can't read the gateway's local disk directly.

    Auth-gated by the session token like every other /api route. Restricted to
    an image-extension allowlist, a size cap, AND the gateway's own media roots
    (resolved, symlink-safe) so it can't be used to read arbitrary files.
    """
    from hermes_cli.web_server import _MEDIA_CONTENT_TYPES, _MEDIA_MAX_BYTES
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if target.suffix.lower() not in _MEDIA_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    roots = _media_serve_roots()
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail="Path outside media roots")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {"data_url": f"data:{_MEDIA_CONTENT_TYPES[target.suffix.lower()]};base64,{encoded}"}


def _local_dashboard_request(request: Request) -> bool:
    if getattr(request.app.state, "auth_required", False):
        return False
    host = (request.url.hostname or "").lower()
    client_host = (request.client.host if request.client else "").lower()
    local_hosts = {"", "localhost", "127.0.0.1", "::1", "testserver", "testclient"}
    return host in local_hosts or client_host in local_hosts


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    from hermes_cli.web_server import _MANAGED_FILE_MAX_BYTES
    text = (data_url or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise HTTPException(status_code=400, detail="Upload payload must be a data URL")
    header, encoded = text.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Upload payload must be base64 encoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Upload payload is not valid base64")
    if len(data) > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    return data, mime_type


_CHAT_IMAGE_UPLOAD_MAX_BYTES = 25 * 1024 * 1024


_CHAT_IMAGE_ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


_CHAT_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def _sanitize_chat_image_filename(filename: str | None) -> str:
    candidate = Path(str(filename or "").strip()).name
    candidate = re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "pasted-image"


def _chat_image_extension(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _CHAT_IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return None


def _decode_chat_image_upload(payload: ChatImageUpload) -> tuple[bytes, str, str]:
    data, mime_type = _decode_data_url(payload.data_url)
    if not mime_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload payload must be an image")
    if len(data) > _CHAT_IMAGE_UPLOAD_MAX_BYTES:
        mb = _CHAT_IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image is too large; cap is {mb} MB")

    ext = _chat_image_extension(data)
    if ext not in _CHAT_IMAGE_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return data, mime_type, ext


@router.post("/api/chat/image-upload")
async def upload_chat_image(payload: ChatImageUpload, profile: Optional[str] = None):
    """Persist a browser-provided chat image where the embedded TUI can read it.

    The dashboard /chat page runs Hermes inside an xterm.js PTY. Browser
    clipboard image bytes are not visible to the server-side clipboard, so the
    page uploads them here, then drives the TUI's ``/image <path>`` command
    with the returned gateway-visible path. Files land under
    ``HERMES_HOME/images/`` — the same directory ``clipboard.paste`` /
    ``image.attach`` already use.
    """
    from hermes_cli.web_server import _profile_scope, get_hermes_home
    def _run():
        data, mime_type, ext = _decode_chat_image_upload(payload)
        with _profile_scope(profile) as scoped_home:
            home = scoped_home or get_hermes_home()
            img_dir = Path(home) / "images"
            try:
                img_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise HTTPException(status_code=403, detail="Image directory is not writable")
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not create image directory: {exc}")

            stem = Path(_sanitize_chat_image_filename(payload.filename)).stem or "pasted-image"
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "pasted-image"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = img_dir / f"dashboard_{ts}_{secrets.token_hex(4)}_{stem}{ext}"

            try:
                target.write_bytes(data)
            except PermissionError:
                raise HTTPException(status_code=403, detail="Image directory is not writable")
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not write image: {exc}")

        return {
            "ok": True,
            "path": str(target),
            "name": target.name,
            "bytes": len(data),
            "mime_type": mime_type,
        }

    # _profile_scope acquires _SKILLS_PROFILE_LOCK and the body does file I/O —
    # keep both off the event loop (asyncio.to_thread copies the contextvar
    # context, so the profile override stays scoped to the worker thread).
    return await asyncio.to_thread(_run)


@router.get("/api/files")
async def list_managed_files(request: Request, path: Optional[str] = None):
    from hermes_cli.web_server import (
        _managed_file_entry,
        _managed_response_meta,
        _resolve_managed_path,
    )
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        with os.scandir(target) as scan:
            entries = [
                _managed_file_entry(policy, Path(entry.path))
                for entry in scan
                if not _is_sensitive_path(Path(entry.path))
            ]
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read directory: {exc}")

    entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
    locked_root = policy.locked_root
    parent = None
    if target.parent != target and (locked_root is None or target != locked_root):
        parent = str(target.parent)
    return {
        "path": display_path,
        "parent": parent,
        "entries": entries,
        **_managed_response_meta(policy),
    }


@router.get("/api/files/read")
async def read_managed_file(request: Request, path: str):
    from hermes_cli.web_server import (
        _MANAGED_FILE_MAX_BYTES,
        _managed_response_meta,
        _resolve_managed_path,
    )
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}")

    return {
        "name": target.name,
        "path": display_path,
        "size": size,
        "mime_type": mime_type,
        "data_url": f"data:{mime_type};base64,{encoded}",
        **_managed_response_meta(policy),
    }


def _managed_file_response(
    request: Request,
    path: str,
    *,
    content_disposition_type: str,
    media_only: bool = False,
) -> FileResponse:
    """Build a range-aware response after applying managed-file policy."""
    from hermes_cli.web_server import (
        _MANAGED_FILE_MAX_BYTES,
        _STREAMABLE_MEDIA_EXTENSIONS,
        _resolve_managed_path,
    )
    policy, target, _display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    if media_only and target.suffix.lower() not in _STREAMABLE_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"

    return FileResponse(
        path=str(target),
        media_type=mime_type,
        filename=target.name,
        content_disposition_type=content_disposition_type,
        headers={"X-Content-Type-Options": "nosniff"} if media_only else None,
    )


@router.get("/api/files/download")
async def download_managed_file(request: Request, path: str):
    """Stream a managed file as an attachment download.

    Remote clients (desktop app, browser dashboard) open agent-written files
    that live on *this* gateway's disk, not theirs. Auth-gated like every other
    managed-files route — ``auth_middleware`` additionally accepts the session
    token as a ``?token=`` query param here so a shell/browser-opened download
    (which can't set the session header) still authenticates. See ``/api/pty``
    for the same query-token precedent. Chromium identifies ``<audio>`` and
    ``<video>`` subresource requests through ``Sec-Fetch-Dest``; serve those
    inline for compatibility with Desktop builds that still use this route as
    their player source, while preserving attachment semantics for ordinary
    link/document requests.
    """
    fetch_destination = request.headers.get("sec-fetch-dest", "").lower()
    is_media_subresource = fetch_destination in {"audio", "video"}
    return _managed_file_response(
        request,
        path,
        content_disposition_type="inline" if is_media_subresource else "attachment",
        media_only=is_media_subresource,
    )


@router.get("/api/files/stream")
@router.head("/api/files/stream")
async def stream_managed_file(request: Request, path: str):
    """Stream managed audio/video inline with HTTP Range support.

    Electron's Chromium media pipeline may reject an attachment response used
    as an ``<audio>`` or ``<video>`` source. This route shares the download
    endpoint's authentication, size cap, sensitive-file guard, MIME detection,
    and Starlette ``FileResponse`` range handling, but explicitly marks the
    response inline so metadata loading, playback, and seeking work remotely.
    """
    return _managed_file_response(
        request,
        path,
        content_disposition_type="inline",
        media_only=True,
    )


@router.post("/api/files/upload")
async def upload_managed_file(payload: ManagedFileUpload, request: Request):
    from hermes_cli.web_server import (
        _managed_file_entry,
        _managed_response_meta,
        _resolve_managed_path,
    )
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not payload.overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    data, _mime_type = _decode_data_url(payload.data_url)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@router.post("/api/files/upload-stream")
async def upload_managed_file_stream(
    request: Request,
    file: UploadFile = File(...),
    path: str = Form(...),
    overwrite: bool = Form(True),
):
    from hermes_cli.web_server import (
        _MANAGED_FILE_MAX_BYTES,
        _UPLOAD_CHUNK_BYTES,
        _managed_file_entry,
        _managed_response_meta,
        _resolve_managed_path,
    )
    policy, target, display_path = _resolve_managed_path(path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create parent directory: {exc}")

    # Write to a sibling temp file first so a partial/aborted upload never
    # clobbers an existing file, then atomically rename into place.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".upload", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="File is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")
    finally:
        # Clean up the temp file on every non-success exit, including
        # BaseException paths the `except` clauses above don't catch — most
        # importantly asyncio.CancelledError when a browser aborts a large
        # upload mid-stream (the exact NS-501 scenario). os.replace clears
        # tmp_path on success, so only unlink when the rename didn't happen.
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@router.post("/api/files/mkdir")
async def create_managed_directory(payload: ManagedDirectoryCreate, request: Request):
    from hermes_cli.web_server import (
        _managed_file_entry,
        _managed_response_meta,
        _resolve_managed_path,
    )
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=409, detail="A file already exists at that path")

    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create directory: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@router.delete("/api/files")
async def delete_managed_file(payload: ManagedFileDelete, request: Request):
    from hermes_cli.web_server import _managed_response_meta, _resolve_managed_path
    policy, target, display_path = _resolve_managed_path(payload.path, request)
    if policy.locked_root is not None and target == policy.locked_root:
        raise HTTPException(status_code=400, detail="Cannot delete the managed files root")
    if target.parent == target:
        raise HTTPException(status_code=400, detail="Cannot delete the filesystem root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if target.is_dir():
            if payload.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
    except OSError as exc:
        status_code = 409 if target.is_dir() and not payload.recursive else 500
        raise HTTPException(status_code=status_code, detail=f"Could not delete path: {exc}")

    return {"ok": True, "path": display_path, **_managed_response_meta(policy)}


@router.get("/api/fs/list")
async def fs_list(path: str):
    from hermes_cli.web_server import _FS_READDIR_HIDDEN, _fs_path
    target = _fs_path(path)
    try:
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name in _FS_READDIR_HIDDEN:
                    continue
                entries.append({
                    "name": entry.name,
                    "path": str(target / entry.name),
                    "isDirectory": entry.is_dir(follow_symlinks=False),
                })
        entries.sort(key=lambda item: (not item["isDirectory"], item["name"].lower(), item["name"]))
        return {"entries": entries}
    except FileNotFoundError:
        return {"entries": [], "error": "ENOENT"}
    except NotADirectoryError:
        return {"entries": [], "error": "ENOTDIR"}
    except PermissionError:
        return {"entries": [], "error": "EACCES"}
    except OSError as exc:
        return {"entries": [], "error": getattr(exc, "strerror", None) or "read-error"}


@router.get("/api/fs/read-text")
async def fs_read_text(path: str):
    from hermes_cli.web_server import _fs_path
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_TEXT_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    bytes_to_read = min(st.st_size, _FS_TEXT_PREVIEW_MAX_BYTES)
    try:
        with target.open("rb") as handle:
            data = handle.read(bytes_to_read)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {
        "binary": _fs_looks_binary(data[:4096]),
        "byteSize": st.st_size,
        "language": _FS_PREVIEW_LANGUAGE_BY_EXT.get(target.suffix.lower(), "text"),
        "mimeType": _fs_mime_type(target),
        "path": str(target),
        "text": data.decode("utf-8", errors="replace"),
        "truncated": st.st_size > _FS_TEXT_PREVIEW_MAX_BYTES,
    }


@router.post("/api/fs/write-text")
async def fs_write_text(payload: FsWriteText):
    """Overwrite (or create) a UTF-8 text file for the in-app spot editor.

    Mirrors the local Electron ``hermes:fs:writeText`` hardening: the path is
    resolved + validated by ``_fs_path``, the parent directory must already
    exist (we never build directory trees), only regular files may be replaced,
    and the payload is size-capped. The write is staged to a sibling temp file
    and ``os.replace``-d into place so a crash mid-write can't truncate the
    original. Stale-on-disk detection is the client's job (re-read before save),
    so both transports behave identically.
    """
    from hermes_cli.web_server import _fs_path
    target = _fs_path(payload.path)
    text = payload.content or ""
    if len(text.encode("utf-8")) > _FS_TEXT_WRITE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content too large")

    try:
        st: Optional[os.stat_result] = target.stat()
    except FileNotFoundError:
        st = None
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")

    if st is not None and stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if st is not None and not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be written")
    if not target.parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    tmp = target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {"ok": True, "path": str(target), "byteSize": len(text.encode("utf-8"))}


@router.get("/api/fs/read-data-url")
async def fs_read_data_url(path: str):
    from hermes_cli.web_server import _FS_DATA_URL_MAX_BYTES, _fs_path
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_DATA_URL_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {"dataUrl": f"data:{_fs_mime_type(target)};base64,{encoded}"}


@router.get("/api/fs/download")
async def fs_download(path: str):
    from hermes_cli.web_server import _fs_path
    target, _st = _fs_regular_file(_fs_path(path))
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return FileResponse(
        path=str(target),
        media_type=_fs_mime_type(target),
        filename=target.name,
        content_disposition_type="attachment",
    )


@router.get("/api/fs/git-root")
async def fs_git_root(path: str):
    from hermes_cli.web_server import _fs_path
    target = _fs_path(path)
    try:
        st = target.stat()
        start = target if stat.S_ISDIR(st.st_mode) else target.parent
    except OSError:
        start = target
    return {"root": _fs_find_git_root(start)}


@router.get("/api/fs/default-cwd")
async def fs_default_cwd():
    cwd = _fs_default_cwd()
    return {"cwd": cwd, "branch": _fs_git_branch(cwd)}
