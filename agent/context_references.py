"""@-reference expansion (``@file:``, ``@folder:``, ``@diff``, ``@git:``, ``@url:`` + plugin prefixes)."""

from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.model_metadata import estimate_tokens_rough
from hermes_cli._subprocess_compat import (
    IS_WINDOWS,
    harden_git_argv,
    noninteractive_git_env,
    windows_hide_flags,
)
from hermes_cli.sizefmt import format_bytes

# ---------------------------------------------------------------------------
# Plugin context-reference provider API
# ---------------------------------------------------------------------------

BUILTIN_PREFIXES = frozenset({"diff", "staged", "file", "folder", "git", "url"})

_context_reference_providers: dict[str, "ContextReferenceProvider"] = {}


class ContextCompletionItem:
    """A single autocomplete result from a context reference provider."""

    __slots__ = ("text", "display", "meta")

    def __init__(self, text: str, display: str = "", meta: str = "") -> None:
        self.text = text
        self.display = display or text
        self.meta = meta


class ContextReferenceProvider(ABC):
    """Base class for plugin @-prefix providers, registered via ``PluginContext.register_context_reference()``."""

    prefix: str = ""  # e.g. "issue", "channel", "doc"
    description: str = ""  # shown in autocomplete meta column

    @abstractmethod
    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        """Return autocomplete items for the given query string."""
        ...

    @abstractmethod
    async def expand(self, target: str) -> str | None:
        """Expand *target* to prompt content.  Return ``None`` to skip."""
        ...


def register_context_reference_provider(provider: ContextReferenceProvider) -> None:
    """Register a plugin context reference provider."""
    if not isinstance(provider, ContextReferenceProvider):
        raise TypeError("provider must be a ContextReferenceProvider instance")
    prefix = provider.prefix.lower().strip()
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    if prefix in BUILTIN_PREFIXES:
        raise ValueError(f"prefix '{prefix}' is reserved for built-in references")
    if prefix in _context_reference_providers:
        raise ValueError(f"prefix '{prefix}' is already registered")
    _context_reference_providers[prefix] = provider


def get_context_reference_providers() -> dict[str, ContextReferenceProvider]:
    """Return a snapshot of all registered plugin providers."""
    return dict(_context_reference_providers)


_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
# Plugin fallback: any @<word>:<value> the built-in regex did not claim.
_PLUGIN_REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?P<kind>[a-zA-Z][a-zA-Z0-9_-]*):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+)"
)

TRAILING_PUNCTUATION = ",.;!?"
_NEEDS_QUOTING = re.compile(r"""[\s()\[\]{}<>"'`]""")
_SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh")
_SENSITIVE_HERMES_DIRS = (Path("skills") / ".hub",)
_SENSITIVE_HOME_FILES = (
    Path(".ssh") / "authorized_keys",
    Path(".ssh") / "id_rsa",
    Path(".ssh") / "id_ed25519",
    Path(".ssh") / "config",
    Path(".bashrc"),
    Path(".zshrc"),
    Path(".profile"),
    Path(".bash_profile"),
    Path(".zprofile"),
    Path(".netrc"),
    Path(".pgpass"),
    Path(".npmrc"),
    Path(".pypirc"),
)


@dataclass(frozen=True)
class ContextReference:
    raw: str
    kind: str
    target: str
    start: int
    end: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False


def format_reference_value(value: str) -> str:
    """Quote a reference value so ``REFERENCE_PATTERN`` reads it back whole.

    The unquoted alternative is ``\\S+``, so a path with a space would parse as a
    truncated ref. Mirrors ``formatRefValue`` in the desktop's directive-text.tsx.
    """
    if not _NEEDS_QUOTING.search(value):
        return value
    for quote in ("`", '"', "'"):
        if quote not in value:
            return f"{quote}{value}{quote}"
    return value


def parse_context_references(message: str) -> list[ContextReference]:
    refs: list[ContextReference] = []
    if not message:
        return refs

    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        if simple:
            refs.append(ContextReference(raw=match.group(0), kind=simple, target="", start=match.start(), end=match.end()))
            continue
        kind = match.group("kind")
        value = _strip_trailing_punctuation(match.group("value") or "")
        if kind == "file":
            target, line_start, line_end = _parse_file_reference_value(value)
        else:
            target, line_start, line_end = _strip_reference_wrappers(value), None, None
        refs.append(
            ContextReference(
                raw=match.group(0),
                kind=kind,
                target=target,
                start=match.start(),
                end=match.end(),
                line_start=line_start,
                line_end=line_end,
            )
        )

    # Second pass: plugin-registered prefixes the built-in pattern missed.
    if _context_reference_providers:
        for match in _PLUGIN_REFERENCE_PATTERN.finditer(message):
            kind = match.group("kind")
            if kind in BUILTIN_PREFIXES or kind not in _context_reference_providers:
                continue
            if any(r.kind == kind and r.start == match.start() for r in refs):
                continue
            value = _strip_trailing_punctuation(match.group("value") or "")
            refs.append(
                ContextReference(
                    raw=match.group(0),
                    kind=kind,
                    target=_strip_reference_wrappers(value),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return refs


def preprocess_context_references(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    """Sync wrapper; safe both without a loop (CLI) and inside a running loop (gateway)."""
    coro = preprocess_context_references_async(
        message,
        cwd=cwd,
        context_length=context_length,
        url_fetcher=url_fetcher,
        allowed_root=allowed_root,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def preprocess_context_references_async(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd_path = Path(cwd).expanduser().resolve()
    # Default root = cwd so @ references cannot escape the workspace unless a caller widens it.
    allowed_root_path = Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    warnings: list[str] = []
    blocks: list[str] = []
    injected_tokens = 0

    # Expand concurrently (each ref is independent; several @url: refs would otherwise
    # serialize web_extract round-trips). gather preserves order, so warnings/blocks
    # are assembled in ref order; the token-budget check runs once afterwards.
    expanded = await asyncio.gather(
        *(_expand_reference(ref, cwd_path, url_fetcher=url_fetcher, allowed_root=allowed_root_path) for ref in refs)
    )
    for warning, block in expanded:
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            injected_tokens += estimate_tokens_rough(block)

    hard_limit = max(1, int(context_length * 0.50))
    soft_limit = max(1, int(context_length * 0.25))
    if injected_tokens > hard_limit:
        warnings.append(
            f"@ context injection refused: {injected_tokens} tokens exceeds the 50% hard limit ({hard_limit})."
        )
        return ContextReferenceResult(
            message=message,
            original_message=message,
            references=refs,
            warnings=warnings,
            injected_tokens=injected_tokens,
            expanded=False,
            blocked=True,
        )
    if injected_tokens > soft_limit:
        warnings.append(
            f"@ context injection warning: {injected_tokens} tokens exceeds the 25% soft limit ({soft_limit})."
        )

    # The `@file:`/`@folder:` tokens stay where the user typed them: the token IS the
    # reference (clients render it as an inline chip); stripping it left a hole in the
    # sentence and forced the desktop to re-derive refs from the attached block.
    final = message
    if warnings:
        final = f"{final}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)

    return ContextReferenceResult(
        message=final.strip(),
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(blocks or warnings),
        blocked=False,
    )


def _git_log_args(ref: ContextReference) -> list[str]:
    count = max(1, min(int(ref.target or "1"), 10))
    return ["log", f"-{count}", "-p"]


# Git-backed reference kinds -> f(ref) -> git argv (the label is "git " + argv).
_GIT_REFERENCE_ARGS: dict[str, Callable[[ContextReference], list[str]]] = {
    "diff": lambda ref: ["diff"],
    "staged": lambda ref: ["diff", "--staged"],
    "git": _git_log_args,
}


async def _expand_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(warning, block)`` for one reference; exactly one side is set."""
    try:
        if ref.kind == "file":
            return _expand_file_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "folder":
            return _expand_folder_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind in _GIT_REFERENCE_ARGS:
            git_args = _GIT_REFERENCE_ARGS[ref.kind](ref)
            return _expand_git_reference(ref, cwd, git_args, "git " + " ".join(git_args))
        if ref.kind == "url":
            content = await _fetch_url_content(ref.target, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no content extracted", None
            return None, f"🌐 {ref.raw} ({estimate_tokens_rough(content)} tokens)\n{content}"
    except Exception as exc:
        return f"{ref.raw}: {exc}", None

    provider = _context_reference_providers.get(ref.kind)
    if provider is not None:
        try:
            plugin_content = await provider.expand(ref.target)
            if plugin_content is not None:
                return None, f"📌 {ref.raw} ({estimate_tokens_rough(plugin_content)} tokens)\n{plugin_content}"
        except Exception as exc:
            return f"{ref.raw}: plugin expansion error: {exc}", None

    return f"{ref.raw}: unsupported reference type", None


def _expand_file_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: file not found", None
    if not path.is_file():
        return f"{ref.raw}: path is not a file", None
    if _is_binary_file(path):
        # A bare "not supported" warning was a dead end (the model gave up); the file IS
        # on disk where the agent's tools run, so hand it an actionable block instead.
        return None, _binary_reference_block(ref, path)

    text = path.read_text(encoding="utf-8")
    if ref.line_start is not None:
        lines = text.splitlines()
        start_idx = max(ref.line_start - 1, 0)
        end_idx = min(ref.line_end or ref.line_start, len(lines))
        text = "\n".join(lines[start_idx:end_idx])

    lang = _code_fence_language(path)
    return None, f"📄 {ref.raw} ({estimate_tokens_rough(text)} tokens)\n```{lang}\n{text}\n```"


def _expand_folder_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: folder not found", None
    if not path.is_dir():
        return f"{ref.raw}: path is not a folder", None
    listing = _build_folder_listing(path, cwd)
    return None, f"📁 {ref.raw} ({estimate_tokens_rough(listing)} tokens)\n{listing}"


def _run_quiet(
    cmd: list[str], cwd: Path, timeout: int, env: dict | None = None
) -> subprocess.CompletedProcess:
    """subprocess.run with captured text output, no stdin, and no console flash on Windows."""
    popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    if env is not None:
        popen_kwargs["env"] = env
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        **popen_kwargs,
    )


def _expand_git_reference(
    ref: ContextReference,
    cwd: Path,
    args: list[str],
    label: str,
) -> tuple[str | None, str | None]:
    try:
        # Repo-supplied config/attributes must never execute code (GHSA-7x36-8jrh-v4pw).
        result = _run_quiet(
            ["git", *harden_git_argv(args)], cwd, 30, env=noninteractive_git_env()
        )
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: git command timed out (30s)", None
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "git command failed"
        return f"{ref.raw}: {stderr}", None
    content = result.stdout.strip() or "(no output)"
    return None, f"🧾 {label} ({estimate_tokens_rough(content)} tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(
    url: str,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> str:
    fetcher = url_fetcher or _default_url_fetcher
    content = fetcher(url)
    if inspect.isawaitable(content):
        content = await content
    return str(content or "").strip()


async def _default_url_fetcher(url: str) -> str:
    from tools.web_tools import web_extract_tool

    raw = await web_extract_tool([url], format="markdown")
    docs = json.loads(raw).get("results", [])
    if not docs:
        return ""
    doc = docs[0]
    return str(doc.get("content") or doc.get("raw_content") or "").strip()


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path | None = None) -> Path:
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    if allowed_root is not None:
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("path is outside the allowed workspace") from exc
    return resolved


def _ensure_reference_path_allowed(path: Path) -> None:
    """Refuse credential/internal paths. Fails CLOSED: the gateway feeds untrusted remote text here."""
    from hermes_constants import get_hermes_home
    home = Path(os.path.expanduser("~")).resolve()
    hermes_home = get_hermes_home().resolve()

    blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES}
    blocked_exact.add(hermes_home / ".env")
    blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS]
    blocked_dirs.extend(hermes_home / rel for rel in _SENSITIVE_HERMES_DIRS)

    if path in blocked_exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")
    for blocked_dir in blocked_dirs:
        try:
            path.relative_to(blocked_dir)
        except ValueError:
            continue
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")

    # Anchor to the canonical read deny-list (agent/file_safety.get_read_block_error): the
    # narrow list above never caught auth.json, .anthropic_oauth.json, mcp-tokens/, webhook
    # secrets or project .env files, and it grows automatically with that deny-list.
    try:
        from agent.file_safety import get_read_block_error

        if get_read_block_error(str(path)) is not None:
            raise ValueError(
                "path is a sensitive credential or internal Hermes path and cannot be attached"
            )
    except ValueError:
        raise
    except Exception:
        # If the canonical lookup fails, falling through would re-open the exact hole this
        # guard closes; a spurious block is recoverable, a leaked credential is not.
        raise ValueError(
            "path could not be verified against the credential deny-list and cannot be attached"
        )


def _strip_trailing_punctuation(value: str) -> str:
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    while stripped.endswith((")", "]", "}")):
        closer = stripped[-1]
        opener = {")": "(", "]": "[", "}": "{"}[closer]
        if stripped.count(closer) > stripped.count(opener):
            stripped = stripped[:-1]
            continue
        break
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
        return value[1:-1]
    return value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    quoted_match = re.match(
        r'^(?P<quote>`|"|\')(?P<path>.+?)(?P=quote)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$',
        value,
    )
    if quoted_match:
        line_start = quoted_match.group("start")
        line_end = quoted_match.group("end")
        return (
            quoted_match.group("path"),
            int(line_start) if line_start is not None else None,
            int(line_end or line_start) if line_start is not None else None,
        )

    range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if range_match:
        line_start = int(range_match.group("start"))
        return (
            range_match.group("path"),
            line_start,
            int(range_match.group("end") or range_match.group("start")),
        )

    return _strip_reference_wrappers(value), None, None


_TEXT_EXTENSIONS = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")


def _is_binary_file(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not mime.startswith("text/") and not path.name.endswith(_TEXT_EXTENSIONS):
        return True
    return b"\x00" in path.read_bytes()[:4096]


def _build_folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    lines = [f"{path.relative_to(cwd)}/"]
    entries = _iter_visible_entries(path, cwd, limit=limit)
    base_depth = len(path.relative_to(cwd).parts)
    for entry in entries:
        indent = "  " * max(len(entry.relative_to(cwd).parts) - base_depth - 1, 0)
        if entry.is_dir():
            lines.append(f"{indent}- {entry.name}/")
        else:
            lines.append(f"{indent}- {entry.name} ({_file_metadata(entry)})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


def _iter_visible_entries(path: Path, cwd: Path, limit: int) -> list[Path]:
    rg_entries = _rg_files(path, cwd, limit=limit)
    if rg_entries is not None:
        output: list[Path] = []
        seen_dirs: set[Path] = set()
        for rel in rg_entries:
            full = cwd / rel
            for parent in full.parents:
                if parent == cwd or parent in seen_dirs or path not in {parent, *parent.parents}:
                    continue
                seen_dirs.add(parent)
                output.append(parent)
            output.append(full)
        return sorted({p for p in output if p.exists()}, key=lambda p: (not p.is_dir(), str(p)))

    output = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        files = sorted(f for f in files if not f.startswith("."))
        root_path = Path(root)
        for name in dirs + files:
            output.append(root_path / name)
            if len(output) >= limit:
                return output
    return output


def _rg_files(path: Path, cwd: Path, limit: int) -> list[Path] | None:
    try:
        result = _run_quiet(["rg", "--files", str(path.relative_to(cwd))], cwd, 10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return files[:limit]


def _agent_visible_path(path: Path) -> str:
    """Map a host path to what the agent's tools can read in the active backend.

    Under a container backend the host path dangles inside the sandbox; files staged
    into an auto-mounted cache dir are translated via ``tools.credential_files``.
    Falls back to the host path when the backend is local or translation fails.
    """
    try:
        # In-process gateways may not have bridged terminal.* config into TERMINAL_ENV
        # yet; run the idempotent bridge so the translation gate sees the active backend.
        from tools.terminal_tool import _ensure_terminal_env_bridged

        _ensure_terminal_env_bridged()
        from tools.credential_files import to_agent_visible_cache_path

        return to_agent_visible_cache_path(str(path))
    except Exception:
        return str(path)


def _binary_reference_block(ref: ContextReference, path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    try:
        size = format_bytes(path.stat().st_size)
    except OSError:
        size = "unknown size"
    return (
        f"📎 {ref.raw} ({mime}, {size}) — binary file, not inlined as text. "
        f"It is available on disk at `{_agent_visible_path(path)}`. Use your tools to work with it "
        f"(read or convert it, extract its text, or view/render it as needed); "
        f"do not tell the user the file type is unsupported."
    )


def _file_metadata(path: Path) -> str:
    if _is_binary_file(path):
        return f"{path.stat().st_size} bytes"
    try:
        line_count = path.read_text(encoding="utf-8").count("\n") + 1
    except Exception:
        return f"{path.stat().st_size} bytes"
    return f"{line_count} lines"


_FENCE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
}


def _code_fence_language(path: Path) -> str:
    return _FENCE_LANGUAGES.get(path.suffix.lower(), "")
