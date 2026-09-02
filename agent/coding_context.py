"""Coding-context awareness — base Hermes, every interactive surface.

When Hermes runs inside a code workspace (CLI, TUI, desktop, ACP editor) it
shifts into a **coding posture**. This module is the single place that decides
whether we're in that posture and what it implies, so nothing else re-derives
"are we coding?". The posture is a frozen :class:`RuntimeMode` selected from a
small :class:`ContextProfile` registry (``coding`` / ``general``); a profile is
*data* (toolset, operating brief, skill-index hints) that every domain reads:

  * System prompt — ``RuntimeMode.system_prompt_parts()`` → operating brief +
    live git/workspace snapshot (``agent/system_prompt.py``).
  * Toolset — ``RuntimeMode.toolset_selection()`` → ``coding`` toolset + enabled
    MCP servers, ONLY under the opt-in ``focus`` mode. The default posture is
    prompt-only and never strips a toolset the user explicitly enabled.
  * Delegation — subagents inherit the toolset and prompt builder, so the
    posture propagates for free.

Cache safety: the mode is resolved once and immutable; the workspace snapshot
is built once at prompt-build time and never re-probed per turn (the brief
tells the model to re-check with ``git``). A ``/coding`` flip takes effect next
session.

Activation (config ``agent.coding_context``): ``auto`` (default) — posture on an
interactive surface in a code workspace, prompt-only; ``focus`` — also collapse
the toolset and demote non-coding skill categories to names-only (never
hidden); ``on`` — force the posture anywhere; ``off`` — disable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli._subprocess_compat import bounded_git_probe

logger = logging.getLogger("hermes.coding_context")

CODING_TOOLSET = "coding"

# Surfaces where a coding posture makes sense under ``auto``. Messaging
# platforms are intentionally absent — a chat bot in a group is not pairing.
INTERACTIVE_CODING_PLATFORMS = {"cli", "tui", "acp", "desktop", ""}

# Project-root signals that mark a directory as a code workspace even when it
# isn't (yet) a git repo. Cheap filename checks — no parsing.
_PROJECT_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "tsconfig.json", "deno.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "mix.exs", "pubspec.yaml",
    "CMakeLists.txt", "Makefile", "Dockerfile",
    "AGENTS.md", "CLAUDE.md", ".cursorrules",
)

# Agent-instruction files surfaced separately from manifests in the snapshot.
_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", ".cursorrules")

# Source extensions that make a manifest-less git repo a *code* workspace, so
# `git init` on a notes/writing folder does not flip the session into coding.
_CODE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".rb", ".php", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".mm", ".dart", ".ex", ".exs",
    ".lua", ".sh", ".bash", ".zsh", ".sql", ".vue", ".svelte", ".r", ".jl",
    ".hs", ".clj", ".erl", ".pl",
})

_CODE_SCAN_SKIP_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "target", ".next", ".turbo", "vendor",
})
# Bounded sweep: a code workspace reveals itself in the first handful of entries.
_CODE_SCAN_MAX_ENTRIES = 500


def _has_code_files(root: Path) -> bool:
    """Bounded check for source files in the root and its immediate subdirs."""
    seen = 0
    stack = [(root, True)]
    while stack:
        directory, is_root = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > _CODE_SCAN_MAX_ENTRIES:
                        return False
                    name = entry.name
                    try:
                        if entry.is_file():
                            if os.path.splitext(name)[1].lower() in _CODE_EXTENSIONS:
                                return True
                        elif is_root and entry.is_dir() and name not in _CODE_SCAN_SKIP_DIRS and not name.startswith("."):
                            stack.append((Path(entry.path), False))
                    except OSError:
                        continue
        except OSError:
            continue
    return False


# Lockfile → package manager, checked in priority order.
_PY_LOCKFILES = (("uv.lock", "uv"), ("poetry.lock", "poetry"), ("Pipfile.lock", "pipenv"))
_JS_LOCKFILES = (
    ("pnpm-lock.yaml", "pnpm"), ("bun.lockb", "bun"), ("bun.lock", "bun"),
    ("yarn.lock", "yarn"), ("package-lock.json", "npm"),
)

# package.json scripts / Makefile targets worth surfacing as verify commands.
_VERIFY_TARGETS = ("test", "tests", "lint", "typecheck", "check", "build", "fmt", "format")
_MAX_VERIFY_COMMANDS = 8
_MAX_FACT_FILE_BYTES = 256 * 1024

_GIT_TIMEOUT = 2.5


# Per-model edit-format steering: nudge each family toward the `patch` mode it
# was trained on (unknown families get nothing). GPT/Codex get V4A for ALL
# edits incl. single-file — codex-rs ships apply_patch as its ONLY editor and
# its prompts say to use it even for single files, so a replace-mode nudge
# would steer them toward a format their first-party harness never taught.
# Anthropic and most open-weight coding models were RL'd on str_replace-style
# editors. Substrings match the model id; aligned with TOOL_USE_ENFORCEMENT_MODELS.
_EDIT_FORMAT_GUIDANCE: dict[str, tuple[tuple[str, ...], str]] = {
    "patch": (
        ("gpt", "codex"),
        "- Edit format: author new files with `write_file`; for edits to "
        "existing code use `patch` with `mode='patch'` (V4A diff) — including "
        "single-file edits. It's the edit format you handle most reliably.",
    ),
    "replace": (
        ("claude", "sonnet", "opus", "haiku",
         "gemini", "gemma", "deepseek", "qwen", "kimi", "glm", "grok",
         "hermes", "llama", "mistral", "devstral", "minimax"),
        "- Edit format: author new files with `write_file`; for edits to "
        "existing code prefer `patch` in `mode='replace'` — match a unique "
        "snippet and swap it. Reach for `mode='patch'` (V4A) only when an edit "
        "genuinely spans several files at once.",
    ),
}


def _model_family(model: Optional[str]) -> Optional[str]:
    """Edit-format family key for a model id, or ``None`` (neutral wording applies)."""
    if not model:
        return None
    lowered = model.lower()
    for family, (needles, _line) in _EDIT_FORMAT_GUIDANCE.items():
        if any(n in lowered for n in needles):
            return family
    return None


def _edit_format_line(model: Optional[str]) -> str:
    """The edit-format guidance line for this model's family (``""`` if none)."""
    family = _model_family(model)
    return "" if family is None else _EDIT_FORMAT_GUIDANCE[family][1]


# Operating brief for the coding posture. Tool names referenced here are in the
# coding toolset and in _HERMES_CORE_TOOLS, so they exist on every surface this fires on.
CODING_AGENT_GUIDANCE = (
    "You are a coding agent pairing with the user inside their codebase. "
    "Operate like a careful senior engineer.\n"
    "\n"
    "Gather context first:\n"
    "- Read the relevant files with `read_file` and locate code with "
    "`search_files` before changing anything. Trace a symbol to its definition "
    "and usages rather than guessing its shape.\n"
    "- Batch independent lookups: when several reads/searches don't depend on "
    "each other, issue them together in one turn instead of one at a time.\n"
    "- Never invent files, symbols, APIs, or imports. If you haven't seen it in "
    "the repo, go look. Don't assume a library is available — check the project "
    "manifest (pyproject.toml / package.json / Cargo.toml / go.mod) and how "
    "neighbouring files import it.\n"
    "\n"
    "Make changes through the tools, not the chat:\n"
    "- Edit with `patch`/`write_file`. Do NOT print code blocks to the user as "
    "a substitute for editing — apply the change, then summarise it. Only show "
    "code when the user explicitly asks to see it.\n"
    "- Match the project's existing style and conventions; AGENTS.md / "
    "CLAUDE.md / .cursorrules already in context win over your defaults. Touch "
    "only what the task needs — no drive-by refactors, renames, or reformatting "
    "— and add any imports/dependencies your code requires.\n"
    "- If an edit fails to apply, re-read the file to get the current exact "
    "contents before retrying — don't repeat a stale patch. If the same region "
    "fails twice, rewrite the enclosing function or file with `write_file` "
    "instead of attempting a third patch.\n"
    "\n"
    "Verify, and know when to stop:\n"
    "- Use `terminal` for git, builds, tests, and inspection. Run the relevant "
    "tests/linter/build and confirm they pass before claiming the work is done.\n"
    "- Terminal state persists across calls: current directory and exported "
    "environment variables carry forward. Activate a virtualenv or export setup "
    "vars once, then reuse that state instead of re-sourcing it before every "
    "test command.\n"
    "- Fix root causes, not symptoms: when you find a bug, check sibling call "
    "paths for the same flaw and fix the class, not just the reported site.\n"
    "- When fixing linter/type errors on a file, stop after about three "
    "attempts on the same file and ask the user rather than looping.\n"
    "- Track multi-step work with `todo_list`. Reference code as `path:line` instead "
    "of pasting whole files.\n"
    "\n"
    "Respect the user's repo: don't commit, push, or rewrite history unless "
    "asked, and never read, print, or commit secrets — leave `.env` and "
    "credential files alone unless the user explicitly asks. The Workspace "
    "block below is a snapshot from session start — re-run `git status`/"
    "`git branch` before relying on it. Be concise: lead with the change or "
    "answer, not a preamble."
)

_TODO_SENTENCE = (
    "- Track multi-step work with `todo_list`. Reference code as "
    "`path:line` instead of pasting whole files."
)
_NO_TODO_SENTENCE = "- Reference code as `path:line` instead of pasting whole files."


# ── Context profiles (declarative posture definitions) ──────────────────────


@dataclass(frozen=True)
class ContextProfile:
    """A named operating posture. Pure data — consumers read these fields.

    ``toolset``: collapse to this toolset (+ enabled MCP) under ``focus``;
    ``None`` keeps the platform default. ``guidance``: operating brief for the
    stable system prompt. ``model_hint``: routing preference (extension seam).
    ``compact_skill_categories``: categories DEMOTED to names-only in the skill
    index under ``focus`` — deny-list, never hidden, so recall keeps working.
    """

    name: str
    toolset: Optional[str] = None
    guidance: str = ""
    model_hint: Optional[str] = None
    compact_skill_categories: tuple[str, ...] = ()


# Clearly non-coding skill categories (deny-list: custom categories keep full
# entries). Coding-adjacent ones (devops, github, mcp, research, …) are absent.
_NON_CODING_SKILL_CATEGORIES = (
    "apple", "communication", "cooking", "creative", "email", "finance",
    "gaming", "gifs", "health", "media", "music", "note-taking",
    "productivity", "shopping", "smart-home", "social-media", "travel",
    "yuanbao",
)


GENERAL_PROFILE = ContextProfile(name="general")
CODING_PROFILE = ContextProfile(
    name="coding",
    toolset=CODING_TOOLSET,
    guidance=CODING_AGENT_GUIDANCE,
    model_hint="coding",
    compact_skill_categories=_NON_CODING_SKILL_CATEGORIES,
)

_PROFILES: dict[str, ContextProfile] = {
    GENERAL_PROFILE.name: GENERAL_PROFILE,
    CODING_PROFILE.name: CODING_PROFILE,
}


def get_profile(name: str) -> ContextProfile:
    """Return a registered profile, falling back to ``general``."""
    return _PROFILES.get(name, GENERAL_PROFILE)


# ── Helpers ─────────────────────────────────────────────────────────────────

_MODE_ALIASES = {
    **dict.fromkeys(("focus", "strict", "lean"), "focus"),
    **dict.fromkeys(("on", "true", "yes", "1", "always"), "on"),
    **dict.fromkeys(("off", "false", "no", "0", "never"), "off"),
}


def _agent_config_value(config: Optional[dict[str, Any]], key: str, default: Any, *, readonly: bool) -> Any:
    """``config["agent"][key]``, loading config when none was passed."""
    if config is None:
        try:
            from hermes_cli.config import load_config, load_config_readonly

            config = load_config_readonly() if readonly else load_config()
        except Exception:
            config = {}
    return ((config or {}).get("agent", {}) or {}).get(key, default)


def _coding_mode(config: Optional[dict[str, Any]]) -> str:
    """Normalized ``agent.coding_context`` mode (auto/focus/on/off)."""
    raw = _agent_config_value(config, "coding_context", "auto", readonly=True)
    return _MODE_ALIASES.get(str(raw).strip().lower(), "auto")


def _coding_instructions(config: Optional[dict[str, Any]]) -> str:
    """Standing operator instructions (``agent.coding_instructions``: str or list).

    Appended to the brief as an extra stable block so a user can pin
    project-wide workflow rules without editing the shipped brief.
    """
    raw = _agent_config_value(config, "coding_instructions", "", readonly=False)
    if isinstance(raw, (list, tuple)):
        return "\n".join(str(item).strip() for item in raw if str(item).strip())
    return str(raw or "").strip()


def _resolve_cwd(cwd: Optional[str | Path]) -> Path:
    if cwd:
        return Path(cwd).expanduser()
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return resolve_agent_cwd()
    except Exception:
        return Path(os.getcwd())


def _git_root(cwd: Path) -> Optional[Path]:
    current = cwd.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _home() -> Optional[Path]:
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def _marker_root(cwd: Path) -> Optional[Path]:
    """Nearest ancestor (≤6 levels) that looks like a project root, or ``None``.

    ``$HOME`` and the shared temp root are skipped: a Makefile/AGENTS.md in the
    home dir is global user config, and a stray manifest in /tmp must not flip
    every session whose cwd lives under it into the coding posture.
    """
    current = cwd.resolve()
    home = _home()
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
    except Exception:
        temp_root = None
    for depth, parent in enumerate([current, *current.parents]):
        if depth > 6:
            break
        if parent == home or (temp_root is not None and parent == temp_root):
            continue
        for marker in _PROJECT_MARKERS:
            if (parent / marker).exists():
                return parent
    return None


def _detect_profile_name(mode: str, platform: str, cwd_str: str) -> str:
    """Resolve which profile applies.

    ``auto``/``focus``: coding when the surface is interactive AND the cwd is a
    code workspace (project root, or a git repo that actually holds code).
    ``on``: always coding. ``off``: always general. A git repo rooted at
    ``$HOME`` (dotfiles) is NOT a workspace signal. Deliberately not memoized:
    a long-lived gateway/TUI process serves sessions from different cwds.
    """
    if mode == "off":
        return GENERAL_PROFILE.name
    if mode == "on":
        return CODING_PROFILE.name
    if platform and platform.strip().lower() not in INTERACTIVE_CODING_PLATFORMS:
        return GENERAL_PROFILE.name
    cwd = Path(cwd_str)
    if _marker_root(cwd) is not None:
        return CODING_PROFILE.name
    git_root = _git_root(cwd)
    if git_root is not None and git_root != _home() and _has_code_files(git_root):
        return CODING_PROFILE.name
    return GENERAL_PROFILE.name


# ── RuntimeMode (the seam) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeMode:
    """The resolved operating posture for a session. Immutable by construction.

    Built once via :func:`resolve_runtime_mode`; never re-resolved mid-session
    (that would break the prompt cache).
    """

    profile: ContextProfile
    surface: str
    cwd: Path
    # Normalized ``agent.coding_context`` mode; toolset collapse is gated on ``focus``.
    config_mode: str = "auto"
    # Model id, used only to steer edit-format guidance (fixed per session).
    model: Optional[str] = None
    # ``agent.coding_instructions``, appended as an extra stable block.
    instructions: str = ""

    @property
    def kind(self) -> str:
        return self.profile.name

    @property
    def is_coding(self) -> bool:
        return self.profile.name == CODING_PROFILE.name

    def toolset_selection(self, config: Optional[dict[str, Any]] = None) -> Optional[list[str]]:
        """Toolset list for this posture, or ``None`` to keep the platform default.

        Non-``None`` only under ``focus``. Callers apply it only when the user
        hasn't pinned an explicit selection (``--toolsets``, ``HERMES_TUI_TOOLSETS``).
        """
        if self.config_mode != "focus" or self.profile.toolset is None:
            return None
        return [self.profile.toolset, *_enabled_mcp_servers(config)]

    def system_prompt_parts(
        self, valid_tool_names=None
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (prefix, workspace, trailing) posture blocks.

        The brief carries the model-family edit-format nudge appended to it
        (one cached string). ``valid_tool_names`` drops the ``todo_list``
        sentence when that tool isn't loaded (e.g. Blank Slate). The three
        lists preserve the historical flat order — brief, workspace snapshot,
        operator instructions — so prompt assembly can put a cache boundary
        before the snapshot without changing the persisted bytes.
        """
        if not self.is_coding:
            return [], [], []
        prefix: list[str] = []
        if self.profile.guidance:
            brief = self.profile.guidance
            if valid_tool_names is not None and "todo_list" not in valid_tool_names:
                brief = brief.replace(_TODO_SENTENCE, _NO_TODO_SENTENCE)
            edit_line = _edit_format_line(self.model)
            if edit_line:
                brief = f"{brief}\n{edit_line}"
            prefix.append(brief)
        workspace = build_coding_workspace_block(self.cwd)
        workspace_parts = [workspace] if workspace else []
        # Operator instructions ride their own block so the brief stays
        # byte-stable independently of user config.
        trailing = (
            [f"Operator instructions (from config):\n{self.instructions}"]
            if self.instructions else []
        )
        return prefix, workspace_parts, trailing

    def system_blocks(self) -> list[str]:
        """Posture blocks as one flat list in historical order (compat helper)."""
        prefix, workspace, trailing = self.system_prompt_parts()
        return [*prefix, *workspace, *trailing]

    def compact_skill_categories(self) -> frozenset[str]:
        """Skill categories to demote to names-only in the skill index.

        Gated on ``focus`` like the toolset collapse — index changes under
        ``auto`` proved too surprising. Demoted, never hidden: fully pruning
        them caused silent capability loss (agent-created skills are the
        model's project memory and models don't reliably re-run ``skills_list``).
        """
        if not self.is_coding or self.config_mode != "focus":
            return frozenset()
        return frozenset(self.profile.compact_skill_categories)


def resolve_runtime_mode(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
) -> RuntimeMode:
    """Resolve the operating posture once (a handful of ``stat`` calls).

    The single entry point every domain should call; the result is immutable
    and safe to hold for the session. ``model`` only steers edit-format guidance.
    """
    resolved_cwd = _resolve_cwd(cwd)
    mode = _coding_mode(config)
    name = _detect_profile_name(
        mode, (platform or "").strip().lower(), str(resolved_cwd)
    )
    return RuntimeMode(
        profile=get_profile(name),
        surface=platform or "",
        cwd=resolved_cwd,
        config_mode=mode,
        model=model,
        instructions=_coding_instructions(config),
    )


# ── Back-compat surface (thin wrappers over RuntimeMode) ────────────────────


def is_coding_context(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether Hermes should operate in its coding posture right now."""
    return resolve_runtime_mode(platform=platform, cwd=cwd, config=config).is_coding


def coding_selection(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
) -> Optional[list[str]]:
    """Toolset selection for the coding posture (``None`` unless ``focus`` and active)."""
    return resolve_runtime_mode(
        platform=platform, cwd=cwd, config=config
    ).toolset_selection(config)


def coding_system_prompt_parts(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
    valid_tool_names=None,
) -> tuple[list[str], list[str], list[str]]:
    """Return coding prefix, workspace snapshot, and trailing guidance."""
    return resolve_runtime_mode(
        platform=platform, cwd=cwd, config=config, model=model
    ).system_prompt_parts(valid_tool_names=valid_tool_names)


def coding_compact_skill_categories(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
) -> frozenset[str]:
    """Skill categories the active posture demotes to names-only (empty outside ``focus``)."""
    return resolve_runtime_mode(
        platform=platform, cwd=cwd, config=config
    ).compact_skill_categories()


def _enabled_mcp_servers(config: Optional[dict[str, Any]]) -> list[str]:
    """Names of MCP servers the user has enabled — kept in the coding posture."""
    try:
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag

        servers = read_raw_config().get("mcp_servers") or {}
        return [
            str(name)
            for name, cfg in servers.items()
            if isinstance(cfg, dict)
            and _parse_enabled_flag(cfg.get("enabled", True), default=True)
        ]
    except Exception:
        return []


# ── git/workspace probe ─────────────────────────────────────────────────────


def _git(cwd: Path, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure.

    :func:`bounded_git_probe` bounds the post-kill cleanup on Windows — a plain
    ``subprocess.run(timeout=...)`` deadlocked when a killed git left a
    suspended descendant holding the pipe handles.
    """
    return bounded_git_probe(["git", "-C", str(cwd), *args], timeout=_GIT_TIMEOUT)


def _parse_status(porcelain: str) -> tuple[dict[str, str], dict[str, int]]:
    """Parse ``git status --porcelain=2 --branch`` into branch + counts."""
    branch: dict[str, str] = {}
    counts = {"staged": 0, "modified": 0, "untracked": 0, "conflicts": 0}
    for line in porcelain.splitlines():
        if line.startswith("# branch.head"):
            branch["head"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.upstream"):
            branch["upstream"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.ab"):
            parts = line.split()
            branch["ahead"], branch["behind"] = parts[2].lstrip("+"), parts[3].lstrip("-")
        elif line.startswith(("1 ", "2 ")):
            xy = line.split(maxsplit=2)[1]
            if xy[0] != ".":
                counts["staged"] += 1
            if xy[1] != ".":
                counts["modified"] += 1
        elif line.startswith("u "):
            counts["conflicts"] += 1
        elif line.startswith("? "):
            counts["untracked"] += 1
    return branch, counts


def _read_small(path: Path) -> str:
    """Read a small text file, or ``""`` — never raises, never reads huge files."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FACT_FILE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@dataclass(frozen=True)
class ProjectFacts:
    """Structured project facts — exposed so non-prompt consumers (desktop verify UI) don't re-detect."""

    manifests: list[str]
    package_managers: list[str]
    verify_commands: list[str]
    context_files: list[str]


def detect_project_facts(root: Path) -> ProjectFacts:
    """Detect manifests, package manager(s), verify commands, and context files.

    Single source of truth for the prompt snapshot and the gateway's
    ``project.facts``. Cheap: stat calls plus a couple of small file reads.
    """
    manifests = [m for m in _PROJECT_MARKERS if m not in _CONTEXT_FILES and (root / m).is_file()]
    package_managers = list(
        dict.fromkeys(pm for lock, pm in (*_PY_LOCKFILES, *_JS_LOCKFILES) if (root / lock).is_file())
    )

    verify: list[str] = []
    if (root / "scripts" / "run_tests.sh").is_file():
        verify.append("scripts/run_tests.sh")
    if (root / "package.json").is_file():
        try:
            scripts = json.loads(_read_small(root / "package.json") or "{}").get("scripts") or {}
        except (json.JSONDecodeError, AttributeError):
            scripts = {}
        js_pm = next((pm for lock, pm in _JS_LOCKFILES if (root / lock).is_file()), "npm")
        verify.extend(f"{js_pm} run {name}" for name in _VERIFY_TARGETS if name in scripts)
    if (root / "pytest.ini").is_file() or "[tool.pytest" in _read_small(root / "pyproject.toml"):
        verify.append("pytest")
    makefile = _read_small(root / "Makefile")
    if makefile:
        verify.extend(
            f"make {name}" for name in _VERIFY_TARGETS
            if re.search(rf"^{re.escape(name)}\s*:", makefile, re.MULTILINE)
        )

    return ProjectFacts(
        manifests=manifests,
        package_managers=package_managers,
        verify_commands=list(dict.fromkeys(verify))[:_MAX_VERIFY_COMMANDS],
        context_files=[c for c in _CONTEXT_FILES if (root / c).is_file()],
    )


def _project_facts(root: Path) -> list[str]:
    """Render :func:`detect_project_facts` as workspace-snapshot lines (byte-stable)."""
    f = detect_project_facts(root)
    facts: list[str] = []
    if f.manifests:
        line = f"- Project: {', '.join(f.manifests[:6])}"
        if f.package_managers:
            line += f" ({'/'.join(f.package_managers)})"
        facts.append(line)
    if f.verify_commands:
        facts.append(f"- Verify: {'; '.join(f.verify_commands)}")
    if f.context_files:
        facts.append(f"- Context files: {', '.join(f.context_files)}")
    return facts


def _workspace_roots(cwd: Optional[str | Path]) -> tuple[Optional[Path], Optional[Path]]:
    """(git_root, workspace_root) for *cwd*; workspace root is git root else marker root."""
    resolved = _resolve_cwd(cwd)
    git_root = _git_root(resolved)
    return git_root, git_root or _marker_root(resolved)


def project_facts_for(cwd: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    """Structured project facts for ``cwd`` — ``None`` outside a workspace.

    Same detection the system-prompt snapshot uses, exposed for non-prompt
    consumers (the desktop verify UI).
    """
    _, root = _workspace_roots(cwd)
    if root is None:
        return None
    f = detect_project_facts(root)
    return {
        "root": str(root),
        "manifests": f.manifests,
        "packageManagers": f.package_managers,
        "verifyCommands": f.verify_commands,
        "contextFiles": f.context_files,
    }


def build_coding_workspace_block(cwd: Optional[str | Path] = None) -> str:
    """Workspace snapshot for the system prompt (empty outside a workspace).

    Git state when the cwd is in a repo, plus detected project facts — so
    marker-only (non-git) projects still get a snapshot.
    """
    git_root, root = _workspace_roots(cwd)
    if root is None:
        return ""

    lines = ["Workspace (snapshot at session start — re-check with `git` before acting on it):"]
    lines.append(f"- Root: {root}")

    if git_root is not None:
        branch, counts = _parse_status(_git(root, "status", "--porcelain=2", "--branch"))
        head = branch.get("head", "")
        if head and head != "(detached)":
            line = f"- Branch: {head}"
            if branch.get("upstream"):
                line += f" \u2192 {branch['upstream']}"
                ahead, behind = branch.get("ahead", "0"), branch.get("behind", "0")
                if ahead != "0" or behind != "0":
                    line += f" (ahead {ahead}, behind {behind})"
            lines.append(line)
        elif head == "(detached)":
            lines.append("- Branch: (detached HEAD)")

        # Linked worktree: say so (branches/stashes are shared state) but do
        # NOT expose the primary tree path — a second absolute path makes the
        # model run commands in the wrong directory.
        git_dir, common_dir = _git(root, "rev-parse", "--git-dir"), _git(root, "rev-parse", "--git-common-dir")
        if git_dir and common_dir and Path(git_dir).resolve() != Path(common_dir).resolve():
            lines.append("- Worktree: linked (git state shared with primary tree)")

        dirty = [f"{n} {label}" for label, n in (
            ("staged", counts["staged"]), ("modified", counts["modified"]),
            ("untracked", counts["untracked"]), ("conflicts", counts["conflicts"]),
        ) if n]
        lines.append(f"- Status: {', '.join(dirty) if dirty else 'clean'}")

        recent = _git(root, "log", "-3", "--pretty=%h %s")
        if recent:
            lines.append("- Recent commits:")
            lines.extend(f"    {c}" for c in recent.splitlines())

    lines.extend(_project_facts(root))
    return "\n".join(lines)
