"""Progressive subdirectory hint discovery.

As the agent navigates into subdirectories via tool calls, this module loads
project context files (AGENTS.md, CLAUDE.md, .cursorrules) from those
directories and appends them to the tool result — context arrives without
touching the system prompt (preserving prompt caching). Complements the
startup CWD-only loading in ``prompt_builder.py``. Inspired by goose's
SubdirectoryHintTracker.
"""

import hashlib
import logging
import os
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, Set

from agent.prompt_builder import _scan_context_content

logger = logging.getLogger(__name__)

# Same filenames as prompt_builder.py, in priority order (first match wins per dir).
_HINT_FILENAMES = [
    "AGENTS.override.md",
    "AGENTS.md", "agents.md",
    "CLAUDE.md", "claude.md",
    ".cursorrules",
]
_MAX_HINT_CHARS = 8_000
_PATH_ARG_KEYS = {"path", "file_path", "workdir"}
_COMMAND_TOOLS = {"terminal"}
# Ancestor levels walked per path — bounds the scan for deeply nested paths.
_MAX_ANCESTOR_WALK = 5

# Directories that hold *copies* of context files (backups, vendored deps,
# VCS internals, caches), never authoritative project context.
_EXCLUDED_DIR_NAMES = frozenset({
    "node_modules", "venv", ".venv", "__pycache__",
    ".git", ".hg", ".svn",
    ".Trash", ".cache", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", "dist-packages",
    "backups", "backup", ".backups",
    "vendor", "third_party",
})


def _is_ancestor_or_same(a: Path, b: Path) -> bool:
    """True if *a* is *b* or one of its ancestors."""
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


class SubdirectoryHintTracker:
    """Track which directories the agent visits and load hints on first access.

    Usage: after each tool call, ``hints = tracker.check_tool_call(name, args)``
    and append the returned text to the tool result.
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        # The working dir is pre-marked loaded (startup context handles it).
        self._loaded_dirs: Set[Path] = {self.working_dir}
        # Content digests already injected: the same file reached through
        # symlinks/hardlinks/copies is never re-sent.
        self._loaded_digests: Set[str] = set()
        self._seed_working_dir_digest()

    def _seed_working_dir_digest(self) -> None:
        """Record the CWD context file's digest (prompt_builder already loaded it)."""
        for filename in _HINT_FILENAMES:
            candidate = self.working_dir / filename
            try:
                if not candidate.is_file():
                    continue
                content = candidate.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if content:
                self._loaded_digests.add(
                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                )
            break  # first match wins, mirroring startup loading

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Return formatted hint text for newly visited directories, or None."""
        all_hints = []
        for d in self._extract_directories(tool_name, tool_args):
            hints = self._load_hints_for_directory(d)
            if hints:
                all_hints.append(hints)
        if not all_hints:
            return None
        return "\n\n" + "\n\n".join(all_hints)

    def _extract_directories(
        self, tool_name: str, args: Dict[str, Any]
    ) -> list:
        """Extract directory paths from tool call arguments."""
        candidates: Set[Path] = set()
        for key in _PATH_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self._add_path_candidate(val, candidates)
        if tool_name in _COMMAND_TOOLS:
            cmd = args.get("command", "")
            if isinstance(cmd, str):
                self._extract_paths_from_command(cmd, candidates)
        return list(candidates)

    def _add_path_candidate(self, raw_path: str, candidates: Set[Path]):
        """Add a raw path's directory and its ancestors to candidates.

        Walks up toward the root, stopping at the first already-loaded
        directory or after ``_MAX_ANCESTOR_WALK`` levels, so reading
        ``project/src/main.py`` still discovers ``project/AGENTS.md``.
        """
        try:
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = self.working_dir / p
            p = p.resolve()
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            for _ in range(_MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    break
                if self._is_valid_subdir(p):
                    candidates.add(p)
                parent = p.parent
                if parent == p:
                    break  # filesystem root
                p = parent
        except (OSError, ValueError, RuntimeError):
            pass

    def _extract_paths_from_command(self, cmd: str, candidates: Set[Path]):
        """Extract path-like tokens (contain / or .; not flags or URLs) from a shell command."""
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        for token in tokens:
            if token.startswith("-"):
                continue
            if "/" not in token and "." not in token:
                continue
            if token.startswith(("http://", "https://", "git@")):
                continue
            self._add_path_candidate(token, candidates)

    def _within_working_dir(self, path: Path) -> bool:
        """Reject paths outside the working-dir tree.

        Loading ~/.codex/AGENTS.md or ~/.claude/CLAUDE.md would mix another
        agent's instructions into this session. ``is_relative_to`` handles
        symlinked paths; the ancestor check is a best-effort fallback.
        """
        try:
            return path.is_relative_to(self.working_dir)
        except (OSError, ValueError):
            return _is_ancestor_or_same(self.working_dir, path)

    def _is_valid_subdir(self, path: Path) -> bool:
        """Directory inside the working-dir tree, not yet loaded, not an excluded copy dir."""
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        if path in self._loaded_dirs:
            return False
        if not self._within_working_dir(path):
            return False
        return not self._is_excluded(path)

    def _is_excluded(self, path: Path) -> bool:
        """True when a segment *below* the working dir is an excluded copy dir.

        Only segments under ``working_dir`` are screened: a user deliberately
        working inside ``vendor/`` keeps that segment legitimate.
        """
        try:
            rel_parts = path.relative_to(self.working_dir).parts
        except ValueError:
            return True  # outside the tree — already rejected upstream
        return any(part in _EXCLUDED_DIR_NAMES for part in rel_parts)

    def _load_hints_for_directory(self, directory: Path) -> Optional[str]:
        """Load the first hint file in *directory*; formatted text or None."""
        self._loaded_dirs.add(directory)
        if not self._within_working_dir(directory):
            logger.debug(
                "Skipping hint files in %s — outside working_dir %s",
                directory, self.working_dir,
            )
            return None

        found_hints = []
        for filename in _HINT_FILENAMES:
            hint_path = directory / filename
            try:
                if not hint_path.is_file():
                    continue
            except OSError:
                continue
            try:
                content = hint_path.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in self._loaded_digests:
                    logger.debug(
                        "Skipping duplicate hint content at %s (digest %s)",
                        hint_path,
                        digest[:12],
                    )
                    break
                self._loaded_digests.add(digest)
                # Same security scan as startup context loading.
                content = _scan_context_content(content, filename)
                if len(content) > _MAX_HINT_CHARS:
                    content = (
                        content[:_MAX_HINT_CHARS]
                        + f"\n\n[...truncated {filename}: {len(content):,} chars total]"
                    )
                rel_path = str(hint_path)
                try:
                    rel_path = str(hint_path.relative_to(self.working_dir))
                except (ValueError, RuntimeError):
                    try:
                        # as_posix: "~/" shorthand implies POSIX rendering
                        # (avoids ~/AppData\Local\... chimeras on Windows).
                        rel_path = "~/" + hint_path.relative_to(Path.home()).as_posix()
                    except (ValueError, RuntimeError):
                        pass  # keep absolute
                found_hints.append((rel_path, content))
                break  # first match wins per directory (like startup loading)
            except Exception as exc:
                logger.debug("Could not read %s: %s", hint_path, exc)

        if not found_hints:
            return None

        sections = [
            f"[Subdirectory context discovered: {rel_path}]\n{content}"
            for rel_path, content in found_hints
        ]
        logger.debug(
            "Loaded subdirectory hints from %s: %s",
            directory,
            [h[0] for h in found_hints],
        )
        return "\n\n".join(sections)
