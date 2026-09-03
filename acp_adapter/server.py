"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import base64
import contextlib
import contextvars
import json
import logging
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Optional
from urllib.parse import unquote, urlparse

import acp
from acp.schema import (
    AgentCapabilities, AgentMessageChunk, AudioContentBlock, AuthenticateResponse, AvailableCommand,
    AvailableCommandsUpdate, BlobResourceContents, ClientCapabilities, EmbeddedResourceContentBlock,
    ForkSessionResponse, ImageContentBlock, Implementation, InitializeResponse, ListSessionsResponse,
    LoadSessionResponse, McpServerHttp, McpServerSse, McpServerStdio, ModelInfo, NewSessionResponse,
    PromptCapabilities, PromptResponse, ResourceContentBlock, ResumeSessionResponse, SessionCapabilities,
    SessionForkCapabilities, SessionInfo, SessionInfoUpdate, SessionListCapabilities, SessionMode,
    SessionModeState, SessionModelState, SessionResumeCapabilities, SetSessionConfigOptionResponse,
    SetSessionModeResponse, SetSessionModelResponse, TextContentBlock, TextResourceContents,
    UnstructuredCommandInput, Usage, UsageUpdate, UserMessageChunk,
)

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider
from acp_adapter.events import (
    _build_plan_update_from_todo_result, make_message_cb, make_step_cb, make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.provenance import session_provenance_meta
from acp_adapter.session import SessionManager, SessionState, _expand_acp_enabled_toolsets
from acp_adapter.tools import build_tool_complete, build_tool_start
from agent.context_compressor import (COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor)
from agent.interrupt_compat import request_hard_interrupt
from tools.approval import (reset_hermes_interactive_context, set_hermes_interactive_context)

logger = logging.getLogger(__name__)

PromptBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


def _named_custom_provider_catalogs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(slug, label, [(model_id, description), ...])`` for named endpoints (v12 ``providers:``
    and legacy ``custom_providers:``), which canonical provider enumeration never lists.

    Models = the entry's declared models, refreshed from the live ``/models`` listing when a
    credential exists and ``discover_models`` isn't disabled; declared models survive a failed
    discovery (some endpoints have no ``/models`` route). Slugs use the ``custom:<name>`` shape
    ``parse_model_input``/``resolve_runtime_provider`` resolve, so choice ids round-trip.
    """
    try:
        from hermes_cli.config import (get_compatible_custom_providers, is_provider_enabled, load_config)
        from hermes_cli.model_switch import (
            _NativePickerModelList, _declared_model_ids, _entry_models_discovered, _fetch_picker_live_models,
            _models_config_is_allowlist,
        )
        from hermes_cli.model_switch_providers import _discover_flag
        from hermes_cli.models import should_use_ollama_native_catalog
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return []

    try:
        cfg = load_config()
        entries = get_compatible_custom_providers(cfg)
    except Exception:
        logger.debug("Could not load named custom providers", exc_info=True)
        return []

    # ``get_compatible_custom_providers`` drops ``enabled``; read disabled keys from raw config.
    raw_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    disabled_keys = {
        str(key).strip().lower()
        for key, raw in (raw_providers.items() if isinstance(raw_providers, dict) else ())
        if isinstance(raw, dict) and not is_provider_enabled(raw)
    }

    def _entry_catalog(entry: dict) -> tuple[str, str, list[tuple[str, str]]] | None:
        provider_key = str(entry.get("provider_key", "") or "").strip()
        name = str(entry.get("name", "") or "").strip()
        base_url = str(entry.get("base_url", "") or "").strip()
        if provider_key.lower() in disabled_keys or not name or not base_url:
            return None
        slug = custom_provider_slug(name, provider_key)

        api_key = str(entry.get("api_key", "") or "").strip()
        if not api_key:
            key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
            api_key = os.environ.get(key_env, "").strip() if key_env else ""

        declared: list[str] = []
        models_cfg = entry.get("models")
        for mid in [str(entry.get("model", "") or "").strip(), *_declared_model_ids(models_cfg)]:
            if mid and mid not in declared:
                declared.append(mid)

        native_headers = entry.get("extra_headers") or None
        is_ollama_key = provider_key.lower() in {"ollama", "custom:ollama"}
        is_native_ollama = should_use_ollama_native_catalog(
            provider_key if is_ollama_key else "custom", base_url, headers=native_headers
        )
        if not api_key and not declared and not is_native_ollama:
            return None  # nothing to discover with and nothing declared: not addressable

        model_ids = list(declared)
        live = None
        if _discover_flag(entry) and (api_key or is_native_ollama):
            try:
                live = _fetch_picker_live_models(
                    api_key, base_url, provider_key if is_native_ollama and is_ollama_key else "custom",
                    _models_config_is_allowlist(models_cfg, _entry_models_discovered(entry)),
                    headers=native_headers, timeout=1.5, api_mode=entry.get("api_mode"),
                )
            except Exception:
                live = None
            if isinstance(live, _NativePickerModelList):
                model_ids = list(live)
            elif live is not None:
                model_ids = declared + [m for m in live if m not in declared]

        if not model_ids and not isinstance(live, _NativePickerModelList):
            return None
        return slug, name, [(mid, "") for mid in model_ids]

    catalogs = [_entry_catalog(entry) for entry in entries if isinstance(entry, dict)]
    return [c for c in catalogs if c is not None]

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Runs the synchronous AIAgent off the event loop.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")

# ListSessionsRequest has no client-side limit; clients paginate via `cursor`/`next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 50
# Per-provider row cap (clients render all `availableModels` in one dropdown; mirrors the
# MoA picker cap). Not a total cap; the current model is always kept via the fallback insert.
ACP_MAX_MODELS_PER_PROVIDER = 200
_MAX_ACP_RESOURCE_BYTES = 512 * 1024
_TEXT_RESOURCE_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/typescript",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
}


def _resource_display_name(uri: str, name: str | None = None, title: str | None = None) -> str:
    """Human-readable attachment name for prompt context."""
    raw_name = (name or "").strip()
    raw_title = (title or "").strip()
    if raw_title and raw_name and raw_title != raw_name:
        return f"{raw_title} ({raw_name})"
    if raw_title or raw_name:
        return raw_title or raw_name
    parsed = urlparse(uri)
    candidate = parsed.path if parsed.scheme else uri
    return Path(unquote(candidate)).name or uri or "resource"


def _mime_main(mime_type: str | None) -> str:
    return (mime_type or "").split(";", 1)[0].strip().lower()


def _is_text_resource(mime_type: str | None) -> bool:
    mime = _mime_main(mime_type)
    return mime.startswith("text/") or mime in _TEXT_RESOURCE_MIME_TYPES


def _is_image_resource(mime_type: str | None) -> bool:
    return _mime_main(mime_type).startswith("image/")


_IMAGE_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _path_from_file_uri(uri: str) -> Path | None:
    """Local file URI/path from an ACP client -> readable Path (None for non-file URIs).
    Windows drive forms (Zed via wsl.exe) become ``/mnt/<drive>/...``."""
    raw = (uri or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        return None

    if parsed.scheme == "file" and parsed.netloc and parsed.netloc not in {"", "localhost"}:
        return None
    path_text = unquote(parsed.path or "") if parsed.scheme == "file" else unquote(raw)

    # file:///C:/Users/... or C:\Users\...
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":" and path_text[1].isalpha():
        drive, rest = path_text[1], path_text[3:]
    elif len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha():
        drive, rest = path_text[0], path_text[2:]
    else:
        return Path(path_text)
    return Path("/mnt") / drive.lower() / rest.lstrip("/\\").replace("\\", "/")


def _decode_text_bytes(data: bytes, mime_type: str | None) -> str | None:
    """Decode resource bytes if they are probably text; return None for binary."""
    if b"\x00" in data and not _is_text_resource(mime_type):
        return None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _format_resource_text(
    *, uri: str, body: str, name: str | None = None, title: str | None = None, note: str | None = None
) -> str:
    display = _resource_display_name(uri, name=name, title=title)
    header = f"[Attached file: {display}]"
    if note:
        header += f" ({note})"
    return f"{header}\nURI: {uri}\n\n{body}"


def _text_parts(**kwargs: Any) -> list[dict[str, Any]]:
    """Single OpenAI text part wrapping ``_format_resource_text(**kwargs)``."""
    return [{"type": "text", "text": _format_resource_text(**kwargs)}]


def _image_parts(uri: str, display: str, data: bytes, mime: str) -> list[dict[str, Any]]:
    """Text header + image_url data URL so vision models can see the attachment."""
    return [
        {"type": "text", "text": f"[Attached image: {display}]" + (f"\nURI: {uri}" if uri else "")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}},
    ]


def _resource_link_to_parts(block: ResourceContentBlock) -> list[dict[str, Any]]:
    """ACP resource_link -> OpenAI content parts: images become a text header + image_url,
    everything else a single text part with the inlined body (or a binary-omit note)."""
    uri = str(getattr(block, "uri", "") or "").strip()
    if not uri:
        return []

    name = str(getattr(block, "name", "") or "").strip() or None
    title = str(getattr(block, "title", "") or "").strip() or None
    mime_type = str(getattr(block, "mime_type", "") or "").strip() or None
    path = _path_from_file_uri(uri)
    ident = dict(uri=uri, name=name, title=title)

    if path is None:
        return _text_parts(
            **ident, body="[Resource link only; Hermes cannot read non-file ACP resource URIs directly.]"
        )

    image_mime = mime_type if _is_image_resource(mime_type) else _IMAGE_SUFFIX_MIME.get(path.suffix.lower())
    if image_mime and _is_image_resource(image_mime):
        try:
            size = path.stat().st_size
            if size > _MAX_ACP_RESOURCE_BYTES:
                return _text_parts(
                    **ident, body=f"[Image too large to inline: {size} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]"
                )
            with path.open("rb") as fh:
                data = fh.read()
        except OSError as exc:
            logger.warning("ACP image resource read failed: %s", uri, exc_info=True)
            return _text_parts(**ident, body=f"[Could not read attached image: {exc}]")
        return _image_parts(uri, _resource_display_name(uri, name=name, title=title), data, image_mime)

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            data = fh.read(min(size, _MAX_ACP_RESOURCE_BYTES))
        text = _decode_text_bytes(data, mime_type)
        if text is None:
            return _text_parts(**ident, body=f"[Binary file omitted: {size} bytes, mime={mime_type or 'unknown'}]")
        note = f"truncated to {_MAX_ACP_RESOURCE_BYTES} of {size} bytes" if size > _MAX_ACP_RESOURCE_BYTES else None
        return _text_parts(**ident, body=text, note=note)
    except OSError as exc:
        logger.warning("ACP resource read failed: %s", uri, exc_info=True)
        return _text_parts(**ident, body=f"[Could not read attached file: {exc}]")


def _embedded_resource_to_parts(block: EmbeddedResourceContentBlock) -> list[dict[str, Any]]:
    resource = getattr(block, "resource", None)
    if resource is None:
        return []

    uri = str(getattr(resource, "uri", "") or "").strip()
    mime_type = str(getattr(resource, "mime_type", "") or "").strip() or None

    if isinstance(resource, TextResourceContents):
        return _text_parts(uri=uri, body=resource.text)

    if isinstance(resource, BlobResourceContents):
        blob = resource.blob or ""
        try:
            data = base64.b64decode(blob, validate=True)
        except Exception:
            data = blob.encode("utf-8", errors="replace")

        if _is_image_resource(mime_type):
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                return _text_parts(
                    uri=uri,
                    body=f"[Embedded image too large to inline: {len(data)} bytes, cap={_MAX_ACP_RESOURCE_BYTES}]",
                )
            return _image_parts(uri, _resource_display_name(uri), data, mime_type or "image/png")

        text = _decode_text_bytes(data[:_MAX_ACP_RESOURCE_BYTES], mime_type)
        if text is None:
            body = f"[Binary embedded file omitted: {len(data)} bytes, mime={mime_type or 'unknown'}]"
        else:
            body = text
            if len(data) > _MAX_ACP_RESOURCE_BYTES:
                body += f"\n\n[Truncated to {_MAX_ACP_RESOURCE_BYTES} of {len(data)} bytes]"
        return _text_parts(uri=uri, body=body)

    text = getattr(resource, "text", None)
    if text:
        return _text_parts(uri=uri, body=str(text))
    return []


def _extract_text(prompt: list[PromptBlock]) -> str:
    """Extract plain text from ACP content blocks for display/commands."""
    return "\n".join(str(block.text) for block in prompt if hasattr(block, "text"))


def _image_block_to_openai_part(block: ImageContentBlock) -> dict[str, Any] | None:
    """Convert an ACP image content block to OpenAI-style multimodal content."""
    data = str(getattr(block, "data", "") or "").strip()
    uri = str(getattr(block, "uri", "") or "").strip()
    mime_type = str(getattr(block, "mime_type", "") or "image/png").strip() or "image/png"

    if data:
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
    elif uri:
        url = uri
    else:
        return None

    return {"type": "image_url", "image_url": {"url": url}}


def _append_parts(parts: list, text_parts: list[str], new_parts: list[dict[str, Any]]) -> None:
    for part in new_parts:
        parts.append(part)
        if part.get("type") == "text":
            text_parts.append(part["text"])


def _content_blocks_to_openai_user_content(prompt: list[PromptBlock]) -> str | list[dict[str, Any]]:
    """Convert ACP prompt blocks into a Hermes/OpenAI-compatible user content payload."""
    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
        elif isinstance(block, ImageContentBlock):
            image_part = _image_block_to_openai_part(block)
            if image_part is not None:
                parts.append(image_part)
        elif isinstance(block, ResourceContentBlock):
            _append_parts(parts, text_parts, _resource_link_to_parts(block))
        elif isinstance(block, EmbeddedResourceContentBlock):
            _append_parts(parts, text_parts, _embedded_resource_to_parts(block))

    if not parts:
        return _extract_text(prompt)

    # Pure text stays a string (slash commands, text-only providers); structured only for media.
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(text_parts)

    return parts


def _semantic_provider(provider_id: str, normalize_provider: Callable[[str], str]) -> str:
    raw = str(provider_id or "").strip().lower()
    if raw in {"ollama", "custom:ollama"}:
        return "ollama"
    if raw.startswith("custom:"):
        return raw
    return normalize_provider(raw)


def _empty_catalog_applies(
    provider_id: str, empty_authoritative: set[str], normalize_provider: Callable[[str], str]
) -> bool:
    """True when a named endpoint with an authoritative-empty catalog owns ``provider_id``."""
    raw = str(provider_id or "").strip().lower()
    normalized = normalize_provider(raw)
    if normalized == "custom":
        return any(
            candidate == raw
            or f"custom:{candidate}" == raw
            or (raw == "custom" and candidate == "custom")
            for candidate in empty_authoritative
        )
    return any(
        candidate == raw
        or candidate == f"custom:{normalized}"
        or candidate == f"custom:{raw}"
        or normalize_provider(candidate) == normalized
        for candidate in empty_authoritative
    )


def _choice_provider(model_id: str) -> str:
    """Provider prefix of an encoded choice id; longest configured ``custom:`` slug wins."""
    parts = model_id.split(":")
    if parts[:1] == ["custom"] and len(parts) > 1:
        from hermes_cli.models import _configured_custom_provider_ids

        lowered = model_id.lower()
        for candidate in sorted(
            (p for p in _configured_custom_provider_ids() if p.startswith("custom:")), key=len, reverse=True,
        ):
            if lowered.startswith(candidate + ":"):
                return candidate
        return "custom"
    return parts[0]


def _estimate_tokens(history: list, agent: Any, system_prompt: str | None = None, tools: Any = None) -> int:
    """Rough request-token estimate over history + system prompt + tool schemas."""
    from agent.model_metadata import estimate_request_tokens_rough

    if system_prompt is None:
        system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    if tools is None:
        tools = getattr(agent, "tools", None) or None
    return estimate_request_tokens_rough(history, system_prompt=system_prompt, tools=tools)


def _flatten_history_text(value: Any) -> str:
    """Persisted content/reasoning (str, or list of ``{"text"}`` / ``{"type": "text", "content"}``
    parts) -> one stripped string; whitespace-only collapses to ``""`` ("nothing to emit")."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def _history_reasoning_text(message: dict[str, Any]) -> str:
    """First non-empty of ``reasoning_content`` and ``reasoning`` — both live keys, for
    different transports (not old-vs-new)."""
    for key in ("reasoning_content", "reasoning"):
        text = _flatten_history_text(message.get(key))
        if text:
            return text
    return ""


def _history_summary_meta(message: dict[str, Any], text: str) -> dict[str, Any] | None:
    """``_meta`` for a replayed compaction summary, else None.

    Summaries persist as ordinary messages, standalone (either role) or merged into the first
    preserved tail message. Two keys so clients can't hide real content: ``compactionSummary``
    (whole chunk; safe to collapse) vs ``containsCompactionSummary`` (real content + summary).
    Uses the in-process flag, falling back to content classification for DB-reloaded sessions.
    """
    kind = ContextCompressor.classify_summary_content(text)
    if kind is None and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
        # Flagged but unclassified (prefix drift): the flag only marks summaries -> standalone.
        kind = "standalone"
    if kind == "standalone":
        return {"hermes": {"compactionSummary": True}}
    if kind == "merged":
        return {"hermes": {"containsCompactionSummary": True}}
    return None


_HISTORY_CHUNK_TYPES = {
    "user": (UserMessageChunk, "user_message_chunk"),
    "assistant": (AgentMessageChunk, "agent_message_chunk"),
}


def _history_message_update(
    *, role: str, text: str, field_meta: dict[str, Any] | None = None
) -> UserMessageChunk | AgentMessageChunk | None:
    """ACP history replay update for a user/assistant message."""
    spec = _HISTORY_CHUNK_TYPES.get(role)
    if spec is None:
        return None
    cls, session_update = spec
    return cls(session_update=session_update, content=TextContentBlock(type="text", text=text), field_meta=field_meta)


def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract function name/arguments from an OpenAI-style tool_call."""
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
    raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            raw_args = {"raw": raw_args}
    if not isinstance(raw_args, dict):
        raw_args = {}
    return name, raw_args


def _mcp_server_config(server: McpServerStdio | McpServerHttp | McpServerSse) -> dict:
    if isinstance(server, McpServerStdio):
        return {"command": server.command, "args": list(server.args), "env": {i.name: i.value for i in server.env}}
    return {"url": server.url, "headers": {i.name: i.value for i in server.headers}}


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _bind_guarded(stack: contextlib.ExitStack, label: str, setup: Callable[[], Callable[[], None]]) -> None:
    """Run ``setup`` (returns its teardown) and register the teardown; failures in either half only
    log — the turn must still run without the binding."""
    try:
        teardown = setup()
    except Exception:
        logger.debug("Could not set ACP %s", label, exc_info=True)
        return

    def _teardown() -> None:
        try:
            teardown()
        except Exception:
            logger.debug("Could not restore ACP %s", label, exc_info=True)

    stack.callback(_teardown)


def _attach_interrupted_prompt(interrupted_prompt: str, guidance: str) -> str:
    return f"{interrupted_prompt}\n\nUser correction/guidance after interrupt: {guidance}"


def _take_interrupted_prompt(state: SessionState) -> tuple[bool, str]:
    """``(idle, interrupted_prompt)``; consumes the cancelled prompt only when the session is idle."""
    with state.runtime_lock:
        if state.is_running:
            return False, ""
        text, state.interrupted_prompt_text = state.interrupted_prompt_text, ""
        return True, text


@dataclass
class _ModelCatalog:
    """Deduplicated ACP model rows from the inventory + named endpoints.

    Dedupes on the encoded choice id AND a semantic ``provider:model`` id (``ollama`` ==
    ``custom:ollama``). A bare/``custom`` current provider whose base_url matches an ollama
    inventory row is resolved to ``custom:ollama``.
    """

    normalize_provider: Callable[[str], str]
    current_model: str
    current_choice_provider: str
    current_base_url: str
    models: list[ModelInfo] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    seen_semantic_ids: set[str] = field(default_factory=set)
    empty_authoritative: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.current_choice_provider == "ollama":
            self.current_choice_provider = "custom:ollama"
        self._identity_resolved = self.current_choice_provider not in {"", "custom"}

    def semantic(self, provider_id: str) -> str:
        return _semantic_provider(provider_id, self.normalize_provider)

    def add(self, provider_id: str, model_id: str, name: str, description: str) -> None:
        choice_id = HermesACPAgent._encode_model_choice(provider_id, model_id)
        semantic_id = f"{self.semantic(provider_id)}:{model_id}"
        if not choice_id or choice_id in self.seen_ids or semantic_id in self.seen_semantic_ids:
            return
        self.models.append(ModelInfo(model_id=choice_id, name=name, description=description))
        self.seen_ids.add(choice_id)
        self.seen_semantic_ids.add(semantic_id)

    def add_inventory_rows(self, rows: list, provider_label: Callable[[str], str]) -> None:
        for row in rows:
            raw_row_provider = str(row.get("slug") or "").strip().lower()
            row_provider = self.normalize_provider(raw_row_provider)
            row_base_url = str(row.get("api_url") or "").strip().rstrip("/").lower()
            if row.get("native_catalog_empty"):
                self.empty_authoritative.add(raw_row_provider)
            if (
                not self._identity_resolved
                and raw_row_provider in {"ollama", "custom:ollama"}
                and self.current_base_url
                and row_base_url == self.current_base_url
            ):
                self.current_choice_provider = "custom:ollama"
                self._identity_resolved = True
            if not row_provider:
                continue
            provider_name = str(row.get("name") or "").strip() or provider_label(row_provider)
            row_models = row.get("models")
            if not isinstance(row_models, (list, tuple)):
                continue
            encoded_provider = (
                "custom:ollama" if raw_row_provider == "ollama"
                else raw_row_provider if raw_row_provider.startswith("custom:")
                else row_provider
            )
            for model_entry in row_models:
                if isinstance(model_entry, dict):
                    model_entry = model_entry.get("id") or model_entry.get("model") or model_entry.get("name")
                rendered_model = str(model_entry or "").strip()
                if not rendered_model:
                    continue
                is_current = (
                    self.semantic(encoded_provider) == self.semantic(self.current_choice_provider)
                    and rendered_model == self.current_model
                )
                self.add(
                    encoded_provider, rendered_model, f"{provider_name} · {rendered_model}",
                    f"Provider: {provider_name}" + (" • current" if is_current else ""),
                )

    def add_named_catalogs(self, catalogs: list, normalized_provider: str) -> None:
        """Named user-defined endpoints (providers: / custom_providers:) are invisible
        to canonical enumeration — append them like the TUI /model picker. An empty
        catalog marks that slug authoritative-empty."""
        for named_slug, named_label, named_catalog in catalogs:
            if not named_catalog:
                self.empty_authoritative.add(str(named_slug).strip().lower())
                continue
            for named_model, named_desc in named_catalog:
                named_parts = [f"Provider: {named_label}"]
                if named_desc:
                    named_parts.append(str(named_desc).strip())
                if named_slug == normalized_provider and named_model == self.current_model:
                    named_parts.append("current")
                self.add(named_slug, named_model, named_model, " • ".join(part for part in named_parts if part))


@dataclass
class _TurnCallbacks:
    """Per-turn ACP streaming callbacks; all None when no client is connected."""

    tool_progress_cb: Any = None
    reasoning_cb: Any = None
    step_cb: Any = None
    stream_delta_cb: Any = None
    approval_cb: Any = None
    edit_approval_requester: Any = None
    streamed: bool = False


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    # name -> (help text, advertised description, input hint)
    _COMMANDS: dict[str, tuple[str, str, str | None]] = {
        "help": ("Show available commands", "List available commands", None),
        "model": (
            "Show or change current model",
            "Show current model and provider, or switch models",
            "model name to switch to",
        ),
        "tools": ("List available tools", "List available tools with descriptions", None),
        "context": ("Show conversation context info", "Show conversation message counts by role", None),
        "reset": ("Clear conversation history", "Clear conversation history", None),
        "compress": ("Compress conversation context", "Compress conversation context", None),
        "steer": (
            "Inject guidance into the currently running agent turn",
            "Inject guidance into the currently running agent turn",
            "guidance for the active turn",
        ),
        "queue": (
            "Queue a prompt to run after the current turn finishes",
            "Queue a prompt to run after the current turn finishes",
            "prompt to run next",
        ),
        "version": ("Show Hermes version", "Show Hermes version", None),
    }

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"
    # mode id -> (edit approval policy, display name, description)
    _MODES: dict[str, tuple[str, str, str]] = {
        "default": ("ask", "Default", "Ask before edits."),
        "accept_edits": (
            "workspace_session",
            "Accept Edits",
            "Auto-allow workspace and /tmp edits; still asks for sensitive paths.",
        ),
        "dont_ask": (
            "session", "Don't Ask", "Auto-allow file edits for this session except sensitive paths."
        ),
    }
    _MODE_TO_EDIT_APPROVAL_POLICY = {mode: spec[0] for mode, spec in _MODES.items()}
    _EDIT_APPROVAL_POLICY_TO_MODE = {spec[0]: mode for mode, spec in _MODES.items()}

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    async def _send(self, session_id: str, update: Any, *, fail_msg: str, level: int = logging.WARNING) -> bool:
        """``session_update`` that logs instead of raising; False on failure."""
        try:
            await self._conn.session_update(session_id=session_id, update=update)
            return True
        except Exception:
            logger.log(level, fail_msg, session_id, exc_info=True)
            return False

    def _schedule_soon(self, make_coro: Callable[[], Any]) -> None:
        """Run a notification coroutine right after the current response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, make_coro())

    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Edit-approval policy as ACP modes. Zed renders ``config_options`` in the model
        picker's slot; modes (as Claude/Codex use) coexist with the picker."""
        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODES:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[SessionMode(id=m, name=n, description=d) for m, (_p, n, d) in self._MODES.items()],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

    @staticmethod
    def _encode_model_choice(provider: str | None, model: str | None) -> str:
        """``provider:model`` so ACP clients keep provider context."""
        raw_model = str(model or "").strip()
        if not raw_model:
            return ""
        raw_provider = str(provider or "").strip().lower()
        return f"{raw_provider}:{raw_model}" if raw_provider else raw_model

    def _build_model_state(self, state: SessionState) -> SessionModelState | None:
        """Authenticated providers + models, from the shared Hermes inventory (same substrate
        as ``hermes model``/TUI/dashboard) so the selector isn't just the current curated list."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = getattr(state.agent, "provider", None) or detect_provider() or "openrouter"

        try:
            from hermes_cli.inventory import build_models_payload, load_picker_context
            from hermes_cli.models import normalize_provider, provider_label

            normalized_provider = normalize_provider(provider)
            context = load_picker_context().with_overrides(
                current_provider=normalized_provider, current_model=model,
                current_base_url=str(getattr(state.agent, "base_url", "") or ""),
            )
            payload = build_models_payload(
                context, explicit_only=True, include_unconfigured=False, picker_hints=False,
                canonical_order=True, pricing=False, capabilities=False, refresh=False,
                probe_custom_providers=False, probe_current_custom_provider=False,
                max_models=ACP_MAX_MODELS_PER_PROVIDER,
            )

            cat = _ModelCatalog(
                normalize_provider=normalize_provider, current_model=model,
                current_choice_provider=str(provider or "").strip().lower(),
                current_base_url=str(getattr(state.agent, "base_url", "") or "").strip().rstrip("/").lower(),
            )
            cat.add_inventory_rows(payload.get("providers") or [], provider_label)
            cat.add_named_catalogs(_named_custom_provider_catalogs(), normalized_provider)
            available_models = cat.models

            def empty_applies(provider_id: str) -> bool:
                return _empty_catalog_applies(provider_id, cat.empty_authoritative, normalize_provider)

            if cat.empty_authoritative:
                available_models = [m for m in available_models if not empty_applies(_choice_provider(m.model_id))]

            current_is_empty = empty_applies(cat.current_choice_provider)
            if current_is_empty:
                available_models = [m for m in available_models if " • current" not in str(m.description or "")]
            current_model_id = "" if current_is_empty else self._encode_model_choice(cat.current_choice_provider, model)
            if current_model_id and current_model_id not in {item.model_id for item in available_models}:
                provider_name = provider_label(normalized_provider)
                available_models.insert(0, ModelInfo(
                    model_id=current_model_id, name=f"{provider_name} · {model}",
                    description=f"Provider: {provider_name} • current",
                ))

            if not available_models and current_is_empty:
                return SessionModelState(available_models=[], current_model_id="")
            if available_models:
                return SessionModelState(
                    available_models=available_models,
                    current_model_id=(
                        current_model_id if current_model_id or current_is_empty else available_models[0].model_id
                    ),
                )
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None

        fallback_choice = self._encode_model_choice(provider, model)
        return SessionModelState(
            available_models=[ModelInfo(model_id=fallback_choice, name=model)], current_model_id=fallback_choice
        )

    @staticmethod
    def _resolve_model_selection(raw_model: str, current_provider: str) -> tuple[str, str]:
        """Resolve ``provider:model`` input into the provider and normalized model id."""
        target_provider = current_provider
        new_model = raw_model.strip()

        try:
            from hermes_cli.models import detect_provider_for_model, parse_model_input

            target_provider, new_model = parse_model_input(new_model, current_provider)
            if target_provider == current_provider:
                detected = detect_provider_for_model(new_model, current_provider)
                if detected:
                    target_provider, new_model = detected
        except Exception:
            logger.debug("Provider detection failed, using model as-is", exc_info=True)

        return target_provider, new_model

    def _switch_model(
        self, state: SessionState, raw_model: str, *, keep_endpoint: bool = False
    ) -> tuple[str | None, str, str]:
        """Rebuild the session agent on a new model -> (old provider, new provider, model).
        ``keep_endpoint`` carries base_url/api_mode over when the provider is unchanged."""
        current_provider = getattr(state.agent, "provider", None)
        target_provider, new_model = self._resolve_model_selection(raw_model, current_provider or "openrouter")
        state.model = new_model
        endpoint: dict[str, Any] = {}
        if keep_endpoint and not (current_provider and target_provider != current_provider):
            endpoint = {
                "base_url": getattr(state.agent, "base_url", None),
                "api_mode": getattr(state.agent, "api_mode", None),
            }
        state.agent = self.session_manager._make_agent(
            session_id=state.session_id, cwd=state.cwd, model=new_model,
            requested_provider=target_provider, **endpoint,
        )
        self.session_manager.save_session(state.session_id)
        return current_provider, target_provider, new_model

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """``usage_update`` for Zed's context indicator: ``size`` = context window, ``used`` =
        estimated request pressure (system prompt + history + tool schemas)."""
        agent = state.agent
        compressor = getattr(agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None

        try:
            used = _estimate_tokens(state.history, agent)
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)

        return UsageUpdate(session_update="usage_update", size=max(size, 0), used=max(used, 0))

    async def _send_usage_update(self, state: SessionState) -> None:
        if not self._conn:
            return
        update = self._build_usage_update(state)
        if update is None:
            return
        await self._send(state.session_id, update, fail_msg="Failed to send ACP usage update for session %s")

    def _provenance_meta(
        self, acp_session_id: str, current_hermes_session_id: str, previous_hermes_session_id: Optional[str] = None
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(), acp_session_id, current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug("Could not build ACP session provenance for %s", acp_session_id, exc_info=True)
            return None

    async def _send_session_info_update(
        self, session_id: str, *,
        current_hermes_session_id: Optional[str] = None, previous_hermes_session_id: Optional[str] = None,
    ) -> None:
        """Session metadata update; pass ``previous_hermes_session_id`` when the internal head
        rotated (compression split) so provenance flags the reason."""
        if not self._conn:
            return
        try:
            row = self.session_manager._get_db().get_session(session_id)
        except Exception:
            logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
            return
        if not row:
            return

        title = row.get("title")
        # `sessions` has no `updated_at`; "now" is right since this fires when the title changed.
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title if isinstance(title, str) and title.strip() else None,
            updated_at=datetime.now(timezone.utc).isoformat(),
            field_meta=self._provenance_meta(
                session_id, current_hermes_session_id or session_id, previous_hermes_session_id
            ),
        )
        await self._send(
            session_id, update, fail_msg="Could not send ACP session info update for %s", level=logging.DEBUG
        )

    async def _register_session_mcp_servers(
        self, state: SessionState, mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return

        try:
            from tools.mcp_tool import register_mcp_servers

            config_map = {server.name: _mcp_server_config(server) for server in mcp_servers}
            await asyncio.to_thread(register_mcp_servers, config_map)
        except Exception:
            logger.warning("Session %s: failed to register ACP MCP servers", state.session_id, exc_info=True)
            return

        try:
            from model_tools import get_tool_definitions
            from agent.memory_manager import inject_memory_provider_tools

            enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"],
                mcp_server_names=[server.name for server in mcp_servers],
            )
            state.agent.enabled_toolsets = enabled_toolsets
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=getattr(state.agent, "disabled_toolsets", None), quiet_mode=True,
            )
            state.agent.valid_tool_names = {tool["function"]["name"] for tool in state.agent.tools or []}
            inject_memory_provider_tools(state.agent)
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id, len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id, exc_info=True,
            )

    def _schedule_mcp_late_refresh(self, state: SessionState) -> None:
        """Refresh the tool snapshot when background MCP discovery lands after agent build
        (``_make_agent`` only joins ~1.5s). Waits up to 30s off the critical path, then rebuilds
        via ``refresh_agent_mcp_tools`` (same as ``/reload-mcp``).

        Cache safety: only pre-first-turn (nothing cached yet); afterwards the snapshot stays
        frozen and late servers land via the between-turns prologue refresh
        (``agent/turn_context.py``). No-op if discovery finished, join timed out, registry
        unchanged, or session closed.
        """
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight
        except Exception:
            return
        if not mcp_discovery_in_flight():
            return

        import threading

        agent = state.agent
        session_id = state.session_id

        def _wait_then_refresh() -> None:
            try:
                from hermes_cli.mcp_startup import join_mcp_discovery

                if not join_mcp_discovery(timeout=30.0):
                    return

                # In-memory only: ``get_session()`` would restore from DB and build a new AIAgent.
                with self.session_manager._lock:
                    current = self.session_manager._sessions.get(session_id)
                if current is None or current.agent is not agent:
                    return

                # ``prompt()`` flips ``is_running`` under ``runtime_lock`` before dispatching, so
                # holding it here closes the window where a refresh would swap ``tools=`` mid-turn.
                with current.runtime_lock:
                    if current.is_running:
                        return
                    if (
                        int(getattr(agent, "_user_turn_count", 0) or 0) > 0
                        or int(getattr(agent, "_api_call_count", 0) or 0) > 0
                    ):
                        return

                    from tools.mcp_tool import refresh_agent_mcp_tools

                    added = refresh_agent_mcp_tools(agent, quiet_mode=True)
                if added:
                    logger.info(
                        "Session %s: late MCP refresh added %d tools: %s",
                        session_id, len(added), ", ".join(sorted(added)),
                    )
            except Exception:
                logger.debug("Session %s: late MCP refresh failed", session_id, exc_info=True)

        threading.Thread(
            target=_wait_then_refresh, name=f"acp-mcp-late-refresh-{session_id}", daemon=True
        ).start()

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self, protocol_version: int | None = None, client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None, **kwargs: Any,
    ) -> InitializeResponse:
        auth_methods = build_auth_methods()
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_info.name if client_info else "unknown",
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(), list=SessionListCapabilities(), resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only acknowledge the method_id advertised in initialize().
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth runs setup out-of-band; succeed only once credentials exist.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay history as user/assistant/thought chunks plus reconstructed tool-call
        start/complete events so the editor shows the transcript, not a clean thread."""
        if not self._conn or not state.history:
            return

        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        async def send(update: Any) -> bool:
            return await self._send(
                state.session_id, update, fail_msg="Failed to replay ACP history for session %s"
            )

        async def send_message(role: str, message: dict[str, Any]) -> bool:
            text = _flatten_history_text(message.get("content"))
            if not text:
                return True
            update = _history_message_update(
                role=role, text=text, field_meta=_history_summary_meta(message, text)
            )
            return update is None or await send(update)

        for message in state.history:
            role = str(message.get("role") or "")

            if role == "user":
                if not await send_message(role, message):
                    return

            elif role == "assistant":
                thought = _history_reasoning_text(message)
                if thought and not await send(acp.update_agent_thought_text(thought)):
                    return
                if not await send_message(role, message):
                    return
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = str(
                            tool_call.get("id") or tool_call.get("call_id") or tool_call.get("tool_call_id") or ""
                        ).strip()
                        if not tool_call_id:
                            continue
                        tool_name, args = _history_tool_call_name_args(tool_call)
                        active_tool_calls[tool_call_id] = (tool_name, args)
                        if not await send(build_tool_start(tool_call_id, tool_name, args)):
                            return

            elif role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                tool_name = str(message.get("tool_name") or "").strip()
                function_args: dict[str, Any] | None = None
                if tool_call_id in active_tool_calls:
                    tool_name, function_args = active_tool_calls.pop(tool_call_id)
                if not tool_call_id or not tool_name:
                    continue
                result = message.get("content")
                result_text = result if isinstance(result, str) else None
                update = build_tool_complete(tool_call_id, tool_name, result=result_text, function_args=function_args)
                if not await send(update):
                    return
                if tool_name == "todo":
                    plan_update = _build_plan_update_from_todo_result(result_text)
                    if plan_update is not None and not await send(plan_update):
                        return

    async def _replay_history_guarded(self, state: SessionState, verb: str) -> None:
        """Per ACP spec, load/resume must stream history via ``session/update`` BEFORE
        responding (Codex/Claude Code/OpenCode/Zed rely on this; deferring via ``call_soon``
        broke them). Best-effort: a corrupt message must not turn the load into an error."""
        try:
            await self._replay_session_history(state)
        except Exception:
            logger.warning(
                f"ACP history replay raised during session/{verb} for %s — "
                f"{verb} will still succeed, partial transcript may be missing",
                state.session_id,
                exc_info=True,
            )

    def _session_response_fields(self, state: SessionState) -> dict[str, Any]:
        """``models``/``modes``/``field_meta`` for session responses; schedules command
        advertisement + usage refresh."""
        self._schedule_available_commands_update(state.session_id)
        self._schedule_soon(lambda: self._send_usage_update(state))
        return {
            "models": self._build_model_state(state),
            "modes": self._session_modes(state),
            "field_meta": self._provenance_meta(
                state.session_id, getattr(state.agent, "session_id", state.session_id)
            ),
        }

    async def new_session(self, cwd: str, mcp_servers: list | None = None, **kwargs: Any) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        return NewSessionResponse(session_id=state.session_id, **self._session_response_fields(state))

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("Loaded session %s", session_id)
        await self._replay_history_guarded(state, "load")
        return LoadSessionResponse(**self._session_response_fields(state))

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info("Resumed session %s", state.session_id)
        await self._replay_history_guarded(state, "resume")
        return ResumeSessionResponse(**self._session_response_fields(state))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            with state.runtime_lock:
                if state.is_running and state.current_prompt_text:
                    state.interrupted_prompt_text = state.current_prompt_text
                # Cancel + hard-stop under the lock so no other prompt mistakes this turn for
                # redirectable work.
                state.cancel_event.set()
                try:
                    if state.agent:
                        request_hard_interrupt(state.agent)
                except Exception:
                    logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        if state is None:
            logger.info("Forked session %s -> %s", session_id, "")
            return ForkSessionResponse(session_id="")
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, state.session_id)
        self._schedule_available_commands_update(state.session_id)
        return ForkSessionResponse(
            session_id=state.session_id, models=self._build_model_state(state), modes=self._session_modes(state)
        )

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """``cursor`` is a ``session_id`` returned as ``next_cursor``; results resume after it
        (unknown cursor -> empty page, never the full list). Pages cap at the fixed size."""
        infos = self.session_manager.list_sessions(cwd=cwd)

        if cursor:
            for idx, s in enumerate(infos):
                if s["session_id"] == cursor:
                    infos = infos[idx + 1:]
                    break
            else:
                infos = []

        has_more = len(infos) > _LIST_SESSIONS_PAGE_SIZE
        infos = infos[:_LIST_SESSIONS_PAGE_SIZE]

        sessions = [
            SessionInfo(
                session_id=s["session_id"], cwd=s["cwd"], title=s.get("title"),
                updated_at=None if s.get("updated_at") is None else str(s["updated_at"]),
            )
            for s in infos
        ]

        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    def _rewrite_prompt_for_interrupt(
        self, state: SessionState, user_text: str, user_content: Any, text_only: bool
    ) -> tuple[str, Any]:
        """Idle ``/steer`` has nothing to inject into (gateway parity): if a prompt was just
        cancelled, replay it with the steer text as explicit correction; otherwise run the steer
        payload as a plain prompt rather than silently queueing it as if ``/queue`` was typed.
        Plain text after a cancel likewise keeps the cancelled request attached ("stop and
        send" clients) so deictic follow-ups have a target."""
        if not (text_only and isinstance(user_content, str)):
            return user_text, user_content

        if user_text.startswith("/steer"):
            split = user_text.split(maxsplit=1)
            steer_text = split[1].strip() if len(split) > 1 else ""
            if not steer_text:
                return user_text, user_content
            idle, interrupted_prompt = _take_interrupted_prompt(state)
            if interrupted_prompt:
                return (_attach_interrupted_prompt(interrupted_prompt, steer_text),) * 2
            return (steer_text, steer_text) if idle else (user_text, user_content)

        if not user_text.startswith("/"):
            _idle, interrupted_prompt = _take_interrupted_prompt(state)
            if interrupted_prompt:
                return (_attach_interrupted_prompt(interrupted_prompt, user_text),) * 2

        return user_text, user_content

    def _claim_turn_or_queue(
        self, state: SessionState, session_id: str, user_text: str, user_content: Any, text_only: bool
    ) -> str | None:
        """Mark the session running; if a turn is active, redirect it (text-only, supported
        runtime) or queue it. Returns the client message when absorbed, else None."""
        redirected = False
        queued_depth: int | None = None
        with state.runtime_lock:
            if state.is_running:
                if (
                    text_only
                    and isinstance(user_content, str)
                    and getattr(state.agent, "_supports_active_turn_redirect", False) is True
                    and hasattr(state.agent, "redirect")
                ):
                    try:
                        redirected = bool(state.agent.redirect(user_content))
                    except Exception:
                        logger.debug("ACP active-turn redirect failed for %s", session_id, exc_info=True)
                if not redirected:
                    state.queued_prompts.append(user_text or "[Image attachment]")
                    queued_depth = len(state.queued_prompts)
            else:
                state.is_running = True
                state.current_prompt_text = user_text or "[Image attachment]"

        if redirected:
            return "Redirected the active turn with your correction."
        if queued_depth is not None:
            return f"Queued for the next turn. ({queued_depth} queued)"
        return None

    def _run_agent_turn(
        self, *, state: SessionState, session_id: str, user_text: str, user_content: Any, conn: Any,
        loop: asyncio.AbstractEventLoop, approval_cb: Any, edit_approval_requester: Any,
    ) -> dict:
        """Executor-thread body of one turn, run inside ``contextvars.copy_context()`` so
        ContextVar writes are isolated from concurrent sessions.

        Approval routing is thread-local, so it MUST be bound here, not on the loop thread.
        Interactive routing is a ``tools.approval`` contextvar, not ``HERMES_INTERACTIVE`` in
        os.environ, so concurrent workers can't race a global flag onto the non-interactive
        auto-approve path (GHSA-96vc-wcxf-jjff).
        """
        agent = state.agent
        with contextlib.ExitStack() as stack:
            # HERMES_SESSION_KEY scopes per-session caches (interactive sudo password) to this
            # session, not the reused thread. ``cwd`` pins what the system prompt reports as the
            # working directory — otherwise it advertises the Hermes workspace while tools are
            # rooted at the client's project and edits land outside it. ``cron_session=""`` masks
            # any leaked process-global HERMES_CRON_SESSION.
            def _session_context() -> Callable[[], None]:
                from gateway.session_context import clear_session_vars, set_session_vars

                tokens = set_session_vars(
                    session_key=session_id, session_id=session_id, cwd=state.cwd, cron_session="",
                )
                return lambda: clear_session_vars(tokens)

            def _approval() -> Callable[[], None]:
                from tools import terminal_tool

                previous = terminal_tool._get_approval_callback()
                terminal_tool.set_approval_callback(approval_cb)
                return lambda: terminal_tool.set_approval_callback(previous)

            def _edit_approval() -> Callable[[], None]:
                from acp_adapter.edit_approval import reset_edit_approval_requester, set_edit_approval_requester

                token = set_edit_approval_requester(edit_approval_requester)
                return lambda: reset_edit_approval_requester(token)

            _bind_guarded(stack, "session context", _session_context)
            if approval_cb:
                _bind_guarded(stack, "approval callback", _approval)
            if edit_approval_requester:
                _bind_guarded(stack, "edit approval requester", _edit_approval)
            stack.callback(reset_hermes_interactive_context, set_hermes_interactive_context(True))
            # Tools tag side-effects with the ACP session (``kanban_create``); save/restore it.
            stack.callback(_restore_env, "HERMES_SESSION_ID", os.environ.get("HERMES_SESSION_ID"))
            os.environ["HERMES_SESSION_ID"] = session_id

            # Auto-titling fires in the turn prologue; push the title now as a session-info update.
            def _notify_title_update(_title: str, _source: str) -> None:
                if conn:
                    loop.call_soon_threadsafe(asyncio.create_task, self._send_session_info_update(session_id))

            agent._on_session_title = _notify_title_update
            try:
                return agent.run_conversation(
                    user_message=user_content, conversation_history=state.history, task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}

    async def prompt(self, prompt: list[PromptBlock], session_id: str, **kwargs: Any) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)
        text_only_prompt = all(isinstance(block, TextContentBlock) for block in prompt)
        has_content = bool(user_text) or (isinstance(user_content, list) and bool(user_content))
        if not has_content:
            return PromptResponse(stop_reason="end_turn")

        user_text, user_content = self._rewrite_prompt_for_interrupt(
            state, user_text, user_content, text_only_prompt
        )

        # Slash commands are text-only; a prompt with media goes to the agent even if it starts with "/".
        if text_only_prompt and isinstance(user_content, str) and user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if response_text is not None:
                if self._conn:
                    await self._conn.session_update(session_id, acp.update_agent_message_text(response_text))
                    await self._send_usage_update(state)
                return PromptResponse(stop_reason="end_turn")

        absorbed = self._claim_turn_or_queue(state, session_id, user_text, user_content, text_only_prompt)
        if absorbed is not None:
            if self._conn:
                await self._conn.session_update(session_id, acp.update_agent_message_text(absorbed))
            return PromptResponse(stop_reason="end_turn")

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        cbs = self._wire_turn_callbacks(state, session_id, conn, loop)

        def _run_agent() -> dict:
            return self._run_agent_turn(
                state=state, session_id=session_id, user_text=user_text, user_content=user_content, conn=conn,
                loop=loop, approval_cb=cbs.approval_cb, edit_approval_requester=cbs.edit_approval_requester,
            )

        try:
            # ACP `session_id` is the stable handle; agent.session_id is the internal head that
            # compression may rotate — snapshot it to detect rotation after the turn.
            pre_turn_hermes_id = getattr(state.agent, "session_id", None)
            # Fresh context copy: concurrent sessions on the shared executor must not share ContextVars.
            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
            return PromptResponse(stop_reason="end_turn")

        return await self._finish_turn(state, session_id, conn, result, pre_turn_hermes_id, cbs.streamed)

    def _wire_turn_callbacks(
        self, state: SessionState, session_id: str, conn: Any, loop: asyncio.AbstractEventLoop
    ) -> _TurnCallbacks:
        """Install the ACP streaming callbacks on the session agent for one turn."""
        cbs = _TurnCallbacks()
        if conn:
            tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
            tool_call_meta: dict[str, dict[str, Any]] = {}
            policy_getter = lambda: self._edit_approval_policy_for_state(state)  # noqa: E731
            cbs.tool_progress_cb = make_tool_progress_cb(
                conn, session_id, loop, tool_call_ids, tool_call_meta, edit_approval_policy_getter=policy_getter
            )
            cbs.reasoning_cb = make_thinking_cb(conn, session_id, loop)
            cbs.step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta)
            message_cb = make_message_cb(conn, session_id, loop)

            def stream_delta_cb(text: str) -> None:
                if text:
                    cbs.streamed = True
                message_cb(text)

            cbs.stream_delta_cb = stream_delta_cb
            cbs.approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
            try:
                from acp_adapter.edit_approval import make_acp_edit_approval_requester

                cbs.edit_approval_requester = make_acp_edit_approval_requester(
                    conn.request_permission, loop, session_id, auto_approve_getter=policy_getter
                )
            except Exception:
                logger.debug("Could not create ACP edit approval requester", exc_info=True)

        agent = state.agent
        agent.tool_progress_callback = cbs.tool_progress_cb
        # Thought panes get provider reasoning only — no local status updates, no fake accordion.
        agent.thinking_callback = None
        agent.reasoning_callback = cbs.reasoning_cb
        agent.step_callback = cbs.step_cb
        agent.stream_delta_callback = cbs.stream_delta_cb
        return cbs

    async def _finish_turn(
        self, state: SessionState, session_id: str, conn: Any, result: dict, pre_turn_hermes_id: Any,
        streamed_message: bool,
    ) -> PromptResponse:
        """Persist, emit provenance/final text, drain queued prompts, report usage."""
        if result.get("messages"):
            state.history = result["messages"]
            self.session_manager.save_session(session_id)

        # Head rotated (compression split): emit provenance so clients can render the boundary.
        post_turn_hermes_id = getattr(state.agent, "session_id", None)
        if (
            conn
            and post_turn_hermes_id
            and pre_turn_hermes_id
            and post_turn_hermes_id != pre_turn_hermes_id
        ):
            try:
                await self._send_session_info_update(
                    session_id, current_hermes_session_id=post_turn_hermes_id,
                    previous_hermes_session_id=pre_turn_hermes_id,
                )
            except Exception:
                logger.debug("Could not emit ACP provenance update after rotation for %s", session_id, exc_info=True)

        final_response = result.get("final_response", "")
        cancelled = bool(state.cancel_event and state.cancel_event.is_set())
        interrupted = bool(result.get("interrupted")) or cancelled
        # The local "waiting for model" interrupt status is metadata, not prose; stop_reason carries it.
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        suppress_interrupt_response = interrupted and final_response.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)
        # Send the final text unless already streamed — or if a plugin hook transformed it after.
        if (
            final_response
            and conn
            and not suppress_interrupt_response
            and (not streamed_message or result.get("response_transformed"))
        ):
            await conn.session_update(session_id, acp.update_agent_message_text(final_response))

        # Go idle before draining so recursive prompt() calls can acquire the session.
        with state.runtime_lock:
            state.is_running = False
            state.current_prompt_text = ""

        while True:
            with state.runtime_lock:
                if not state.queued_prompts:
                    break
                next_prompt = state.queued_prompts.pop(0)
            if conn:
                await conn.session_update(session_id, acp.update_user_message_text(next_prompt))
            await self.prompt(prompt=[TextContentBlock(type="text", text=next_prompt)], session_id=session_id)

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0), output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0), thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        await self._send_usage_update(state)

        return PromptResponse(stop_reason="cancelled" if cancelled else "end_turn", usage=usage)

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        return [
            AvailableCommand(name=name, description=desc, input=UnstructuredCommandInput(hint=hint) if hint else None)
            for name, (_help, desc, hint) in cls._COMMANDS.items()
        ]

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return
        update = AvailableCommandsUpdate(
            session_update="available_commands_update", available_commands=self._available_commands()
        )
        await self._send(session_id, update, fail_msg="Failed to advertise ACP slash commands for session %s")

    def _schedule_available_commands_update(self, session_id: str) -> None:
        self._schedule_soon(lambda: self._send_available_commands_update(session_id))

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command; ``None`` for unknown ones so they fall through to the LLM."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd not in self._COMMANDS:
            return None
        handler = getattr(self, f"_cmd_{cmd}")

        # Handlers run on the loop thread, outside the per-turn cwd-pinning context. ``/compress``
        # and ``/model`` REBUILD the system prompt, so unpinned they'd bake the Hermes install tree
        # into the persisted cached prompt. Pin inside a fresh context: no leak, no teardown.
        def _dispatch() -> str | None:
            try:
                from agent.runtime_cwd import set_session_cwd

                set_session_cwd(state.cwd)
            except Exception:
                logger.debug("Could not pin ACP session cwd for slash command", exc_info=True)
            return handler(args, state)

        try:
            return contextvars.copy_context().run(_dispatch)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        lines.extend(f"  /{cmd:10s}  {desc}" for cmd, (desc, _adv, _hint) in self._COMMANDS.items())
        lines.extend(["", "Unrecognized /commands are sent to the model as normal messages."])
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider, target_provider, new_model = self._switch_model(state, args)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider or "openrouter"
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            from types import SimpleNamespace
            from agent.memory_manager import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"])
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={t.get("function", {}).get("name") for t in tools or [] if isinstance(t, dict)},
                enabled_toolsets=toolsets, _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = (t.get("function") or {}).get("name", "?")
                desc = (t.get("function") or {}).get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)
        roles = Counter(msg.get("role", "unknown") for msg in state.history)

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            approx_tokens = _estimate_tokens(state.history, agent)
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages" if n_messages else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0:
            if context_length > 0:
                usage_pct = (approx_tokens / context_length) * 100
                lines.append(f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)")
            else:
                lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0:
            if approx_tokens > 0:
                threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
                pct_note = f", {threshold_pct:.0f}%" if threshold_pct else ""
                if approx_tokens >= threshold_tokens:
                    lines.append(f"Compression: due now (threshold ~{threshold_tokens:,}{pct_note}). Run /compress.")
                else:
                    remaining = max(threshold_tokens - approx_tokens, 0)
                    lines.append(
                        f"Compression: ~{remaining:,} tokens until threshold "
                        f"(~{threshold_tokens:,}{pct_note})."
                    )
            else:
                lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        if getattr(agent, "compression_enabled", True) is False:
            lines.append(
                "Auto-compaction is disabled (compression.enabled: false); "
                "/compress still compresses manually."
            )
        else:
            lines.append("Tip: run /compress to compress manually before the threshold.")

        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        reset_failed = False
        try:
            reset_session_state = getattr(state.agent, "reset_session_state", None)
            if callable(reset_session_state):
                reset_session_state()
        except Exception:
            reset_failed = True
            logger.warning("ACP session state reset failed for %s", state.session_id, exc_info=True)
        finally:
            self.session_manager.save_session(state.session_id)
        if reset_failed:
            return "Conversation history cleared. Agent session state reset failed; see logs."
        return "Conversation history cleared."

    def _cmd_compress(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            # No compression_enabled gate: it only disables *automatic* compaction (CLI/gateway parity).
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            original_count = len(state.history)
            # Include system prompt + tool schemas so the figure reflects real request pressure.
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            approx_tokens = _estimate_tokens(state.history, agent, _sys_prompt, _tools)
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # Stable ACP session id: suppress _compress_context's SQLite session split.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history, _sys_prompt, approx_tokens=approx_tokens, task_id=state.session_id, force=True,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_tokens = _estimate_tokens(
                state.history, agent, getattr(agent, "_cached_system_prompt", "") or _sys_prompt,
                getattr(agent, "tools", None) or _tools,
            )
            return (
                f"Context compressed: {original_count} -> {len(state.history)} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _queue_prompt(self, state: SessionState, text: str) -> int:
        with state.runtime_lock:
            state.queued_prompts.append(text)
            return len(state.queued_prompts)

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        depth = self._queue_prompt(state, steer_text)
        return f"No active turn — queued for the next turn. ({depth} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        depth = self._queue_prompt(state, queued_text)
        return f"Queued for the next turn. ({depth} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- Session settings (ACP protocol methods) -----------------------------

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state:
            _old, requested_provider, resolved_model = self._switch_model(state, model_id, keep_endpoint=True)
            logger.info(
                "Session %s: model switched to %s via provider %s", session_id, resolved_model, requested_provider
            )
            return SetSessionModelResponse()
        logger.warning("Session %s: model switch requested for missing session", session_id)
        return None

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        normalized_mode = str(mode_id or "").strip()
        if normalized_mode not in self._MODES:
            normalized_mode = self._MODE_DEFAULT
        state.mode = normalized_mode
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, normalized_mode)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        if str(config_id) == self._EDIT_APPROVAL_POLICY_CONFIG_ID:
            state.mode = self._EDIT_APPROVAL_POLICY_TO_MODE.get(str(value), self._MODE_DEFAULT)
        else:
            options = getattr(state, "config_options", None)
            if not isinstance(options, dict):
                options = {}
            options[str(config_id)] = value
            state.config_options = options
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])
