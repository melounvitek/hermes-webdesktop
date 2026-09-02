"""Completion helpers (@-mention / path fuzzy ranking, repo file listing) for the complete.* RPCs.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# ── Methods: complete ─────────────────────────────────────────────────

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset(
    {".git", ".hg", ".svn", ".next", ".cache", ".venv", "venv", "node_modules", "__pycache__",
     "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _list_repo_files(root: str) -> list[str]:
    """File paths relative to ``root`` (tracked + untracked via ``git ls-files`` from the
    repo top; files outside ``root`` excluded so the picker stays Cmd-P scoped). Falls
    back to a bounded ``os.walk(root)`` outside a git repo. Cached per-root for
    ``_FUZZY_CACHE_TTL_S`` so rapid keystrokes don't respawn git."""
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]

    files: list[str] = []
    from hermes_cli._subprocess_compat import windows_hide_flags

    run_kw = dict(capture_output=True, timeout=2.0, check=False, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
    try:
        top_result = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], **run_kw)
        if top_result.returncode == 0:
            top = top_result.stdout.decode("utf-8", "replace").strip()
            list_result = subprocess.run(
                ["git", "-C", top, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], **run_kw
            )
            if list_result.returncode == 0:
                for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
                    if not p:
                        continue
                    rel = os.path.relpath(os.path.join(top, p), root).replace(os.sep, "/")
                    if rel.startswith("../"):  # parents/siblings of cwd: keep Cmd-P workspace scope
                        continue
                    files.append(rel)
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not files:
        # Fallback walk skips vendor/build dirs + dot-dirs; dotfiles survive (the ranker
        # decides based on whether the query starts with `.`).
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")]
                rel_dir = os.path.relpath(dirpath, root)
                for f in filenames:
                    rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
                if len(files) >= _FUZZY_CACHE_MAX_FILES:
                    break
        except OSError:
            pass

    with _fuzzy_cache_lock:
        _fuzzy_cache[root] = (now, files)

    return files


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    """Rank ``name`` against ``query`` as (tier, len(name)); lower wins, None rejects.
    Tiers: 0 exact · 1 prefix · 2 word-boundary/camelCase hit (`chrome` → `appChrome.tsx`)
    · 3 substring · 4 subsequence (query chars appear in order)."""
    if not query:
        return (3, len(name))

    nl = name.lower()
    ql = query.lower()

    if nl == ql:
        return (0, len(name))

    if nl.startswith(ql):
        return (1, len(name))

    # Split on -_. and camelCase (`appChrome` → ["app","Chrome"]); cheap approximation,
    # falls through to substring/subsequence if it misses.
    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            if buf:
                parts.append(buf)
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for p in parts:
        if p.lower().startswith(ql):
            return (2, len(name))

    if ql in nl:
        return (3, len(name))

    i = 0
    for ch in nl:
        if ch == ql[i]:
            i += 1
            if i == len(ql):
                return (4, len(name))

    return None


def _abs_completion_prefix_exists(path_part: str) -> bool:
    """True when ``path_part`` reads sensibly as an absolute path: the parent dir exists
    and a partially-typed final segment matches at least one entry. Decides whether
    `@/foo` is the absolute `/foo` or shorthand for `foo` under the cwd."""
    expanded = _normalize_completion_path(path_part)
    parent = os.path.dirname(expanded.rstrip("/")) or "/"
    tail = os.path.basename(expanded.rstrip("/"))

    if not os.path.isdir(parent):
        return False

    if not tail or expanded.endswith("/"):
        return os.path.isdir(expanded) or expanded == "/"

    try:
        tail_lower = tail.lower()
        return any(e.lower().startswith(tail_lower) for e in os.listdir(parent))
    except OSError:
        return False


def _details_completion_item(value: str, meta: str = "") -> dict:
    return {"text": value, "display": value, "meta": meta}


def _details_root_completion_item(value: str, meta: str, needs_leading_space: bool) -> dict:
    return _details_completion_item(f" {value}" if needs_leading_space else value, meta)


_DETAILS_SECTIONS = ("thinking", "tools", "subagents", "activity")
_DETAILS_MODES = ("hidden", "collapsed", "expanded")


def _details_root_meta(candidate: str) -> str:
    if candidate in _DETAILS_SECTIONS:
        return "section override"
    return "cycle global mode" if candidate == "cycle" else "global mode"


def _details_completions(text: str) -> list[dict] | None:
    if not text.lower().startswith("/details"):
        return None

    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None

    body = text[len("/details") :]
    if body.startswith(" "):
        body = body[1:]
    parts = body.split()
    has_trailing_space = text.endswith(" ")
    sections, modes = _DETAILS_SECTIONS, _DETAILS_MODES
    root_candidates = (*modes, "cycle", *sections)

    if not body or (len(parts) == 0 and has_trailing_space):
        return [_details_root_completion_item(c, _details_root_meta(c), not has_trailing_space) for c in root_candidates]

    if len(parts) == 1 and not has_trailing_space:
        prefix = parts[0].lower()
        return [
            _details_completion_item(c, _details_root_meta(c))
            for c in root_candidates
            if c.startswith(prefix) and c != prefix
        ]

    section = parts[0].lower() if parts else ""
    if section not in sections:
        return []

    def section_meta(candidate: str) -> str:
        return f"clear {section} override" if candidate == "reset" else f"set {section}"

    if len(parts) == 1 and has_trailing_space:
        return [_details_completion_item(c, section_meta(c)) for c in (*modes, "reset")]

    if len(parts) == 2 and not has_trailing_space:
        prefix = parts[1].lower()
        return [
            _details_completion_item(c, section_meta(c)) for c in (*modes, "reset") if c.startswith(prefix) and c != prefix
        ]

    return []


def _model_picker_context(agent):
    """Layer live session state onto config without losing custom identity."""
    from hermes_cli.inventory import load_picker_context

    ctx = load_picker_context()
    provider = getattr(agent, "provider", "") if agent else ""
    base_url = getattr(agent, "base_url", "") if agent else ""
    model = getattr(agent, "model", "") if agent else ""
    if str(provider or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            provider = (
                canonical_custom_identity(
                    base_url=base_url or None, config_provider=ctx.current_provider, model=model or None
                )
                or provider
            )
        except Exception:
            logger.debug("custom provider identity recovery failed (model picker)", exc_info=True)

    return ctx.with_overrides(
        current_provider=provider, current_model=model or _resolve_model(), current_base_url=base_url
    )


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
