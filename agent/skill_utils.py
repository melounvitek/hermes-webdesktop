"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.

Import-light by design: no tool registry, CLI config, or provider resolution.
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_constants import get_config_path, get_skills_dir, is_termux

logger = logging.getLogger(__name__)

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

EXCLUDED_SKILL_DIRS = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".curator_backups",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )
)

# Progressive-disclosure support dirs inside a skill package: loaded explicitly
# via skill_view(skill, file_path=...), never scanned as standalone skills.
SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))

# ── Org-shared skills (sync contract) ───────────────────────────
# Org mirrors live under ~/.hermes/skills/_org/<org_id>/. Resolution is
# TOKEN-GATED via a marker the sync client writes after verifying the token:
# only the marked org's mirror is scanned; no marker ⇒ no org skills load.
# The marker persists offline so already-pulled org skills keep working.
ORG_MIRROR_DIR_NAME = "_org"
ORG_ACTIVE_MARKER = ".active_org"
ORG_PROVENANCE_FILE = ".org-provenance.json"
# Fingerprint of each skill as upstream sent it, so a local edit is detectable.
ORG_BASELINE_FILE = ".org-baseline.json"


def read_active_org_id(skills_dir: Path) -> Optional[str]:
    """The org id whose mirror may resolve, or None (no org skills load)."""
    try:
        marker = skills_dir / ORG_MIRROR_DIR_NAME / ORG_ACTIVE_MARKER
        if not marker.exists():
            return None
        return marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _org_rel_parts(path, skills_dir: Path) -> Optional[Tuple[str, ...]]:
    try:
        return Path(path).resolve().relative_to(Path(skills_dir).resolve()).parts
    except (OSError, ValueError):
        return None


def is_org_mirror_path(path, skills_dir: Path) -> bool:
    """True when *path* is inside the org mirror (``_org/``)."""
    parts = _org_rel_parts(path, skills_dir)
    return bool(parts) and parts[0] == ORG_MIRROR_DIR_NAME


def org_id_of_path(path, skills_dir: Path) -> Optional[str]:
    """The ``<org_id>`` segment for a path under ``_org/<org_id>/...``."""
    parts = _org_rel_parts(path, skills_dir)
    if parts and len(parts) >= 2 and parts[0] == ORG_MIRROR_DIR_NAME:
        return parts[1]
    return None


def is_excluded_skill_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* (Path or str) should be skipped by active skill scanners.

    Apply to every SKILL.md from a direct ``rglob`` scan so all scanning sites
    share one exclusion set (dependency/VCS/cache dirs + support packages).
    """
    parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts) or is_skill_support_path(
        path, root=root
    )


def is_skill_support_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* is under a support dir sitting directly inside a skill root.

    ``skills/scripts/foo`` stays discoverable: its ``scripts`` component is not
    directly under a directory containing ``SKILL.md``.
    """
    path_obj = path if isinstance(path, Path) else Path(str(path))
    parts = path_obj.parts
    # Only components before the leaf can be containing support directories.
    for idx, part in enumerate(parts[:-1]):
        if part not in SKILL_SUPPORT_DIRS or idx == 0:
            continue
        skill_root = Path(*parts[:idx])
        if root is not None and not path_obj.is_absolute():
            skill_root = root / skill_root
        if (skill_root / "SKILL.md").exists():
            return True
    return False


# ── Lazy YAML loader ─────────────────────────────────────────────────────

_yaml_load_fn = None


def yaml_load(content: str):
    """Parse YAML with lazy import and CSafeLoader preference."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


# ── Frontmatter parsing ──────────────────────────────────────────────────


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown; returns (frontmatter_dict, body).

    Falls back to simple key:value splitting for malformed YAML. A single
    leading UTF-8 BOM is stripped first: Windows editors prepend one and it
    would otherwise defeat the ``---`` fence check and silently drop the
    frontmatter (see CONTRIBUTING.md "File encoding").
    """
    frontmatter: Dict[str, Any] = {}
    if content.startswith("\ufeff"):
        content = content[1:]
    body = content

    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


# ── Platform matching ─────────────────────────────────────────────────────


def skill_matches_platform_list(platforms: Any) -> bool:
    """Return True when *platforms* is compatible with the current OS."""
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    running_in_termux = is_termux()
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
        # Termux is a Linux userland on Android: accept linux-tagged skills
        # whether sys.platform is "linux" (pre-3.13) or "android" (3.13+),
        # plus explicit termux/android tags.
        if running_in_termux and mapped in ("linux", "termux", "android"):
            return True
    return False


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """True when the skill's ``platforms:`` list (absent = all) matches this OS."""
    return skill_matches_platform_list(frontmatter.get("platforms"))


# ── Environment matching ──────────────────────────────────────────────────
# An ``environments:`` tag is a *relevance* gate for offer surfaces (index,
# autocomplete, slash commands), not a compatibility gate: an explicit load
# (skill_view, --skills) always succeeds. Detection is cached per process.

_KNOWN_ENVIRONMENTS = frozenset({"kanban", "docker", "s6"})

_ENV_DETECT_CACHE: Dict[str, bool] = {}


def _detect_kanban() -> bool:
    # Mirror the signals tools/kanban_tools.py gates on: a dispatcher-spawned
    # worker (HERMES_KANBAN_TASK/BOARD in env — but only when this execution
    # owns the task; a delegate_task child or in-process cron job sees the
    # worker's vars without being that worker) or a profile opted into the
    # kanban toolset.
    if os.getenv("HERMES_KANBAN_TASK") or os.getenv("HERMES_KANBAN_BOARD"):
        try:
            from agent.delegation_context import is_dispatcher_owned_worker_context

            if is_dispatcher_owned_worker_context():
                return True
        except Exception:
            return True
    try:
        from tools.kanban_tools import _profile_has_kanban_toolset

        return bool(_profile_has_kanban_toolset())
    except Exception:
        return False


def _detect_docker() -> bool:
    try:
        from hermes_constants import is_container

        return is_container()
    except Exception:
        return False


def _detect_s6() -> bool:
    # The Hermes Docker image runs s6-overlay as PID 1; either marker means
    # we're inside an s6-supervised container.
    return os.path.isdir("/run/s6") or os.path.isdir("/package/admin/s6-overlay")


_ENV_DETECTORS: Dict[str, Callable[[], bool]] = {
    "kanban": _detect_kanban,
    "docker": _detect_docker,
    "s6": _detect_s6,
}


def _detect_environment(env: str) -> bool:
    """True when the named runtime environment is active.

    Cached per process EXCEPT ``kanban``: that verdict is context-dependent
    (delegate children / in-process cron see the worker's vars), so a
    process-wide cache would leak the first asker's answer to the others.
    """
    if env != "kanban" and env in _ENV_DETECT_CACHE:
        return _ENV_DETECT_CACHE[env]
    detector = _ENV_DETECTORS.get(env)
    result = detector() if detector else True
    _ENV_DETECT_CACHE[env] = result
    return result


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """True when ANY declared ``environments:`` tag is active (absent = all).

    Offer-time filter only; unknown tags fail open.
    """
    environments = frontmatter.get("environments")
    if not environments:
        return True
    if not isinstance(environments, list):
        environments = [environments]
    for env in environments:
        normalized = str(env).lower().strip()
        if not normalized:
            continue
        if normalized not in _KNOWN_ENVIRONMENTS or _detect_environment(normalized):
            return True
    return False


# ── Disabled skills ───────────────────────────────────────────────────────


_RAW_CONFIG_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def _raw_config_cache_clear() -> None:
    """Test hook — drop the shared raw config cache."""
    _RAW_CONFIG_CACHE.clear()


def _load_raw_config() -> Dict[str, Any]:
    """Read config.yaml with an mtime+size keyed cache (no hermes_cli.config import)."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    try:
        stat = config_path.stat()
        cache_key = (str(config_path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _RAW_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return {}
    if not isinstance(parsed, dict):
        return {}

    if cache_key is not None:
        _RAW_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE[cache_key] = parsed
    return parsed


def _skills_cfg() -> Optional[Dict[str, Any]]:
    """The ``skills:`` mapping from config.yaml, or None when absent/malformed."""
    parsed = _load_raw_config()
    skills_cfg = parsed.get("skills") if parsed else None
    return skills_cfg if isinstance(skills_cfg, dict) else None


def _expand_path(entry: str) -> Path:
    """Expand ``~`` and ``${VAR}`` in a config path entry."""
    return Path(os.path.expanduser(os.path.expandvars(entry)))


# Always available regardless of config: `hermes-agent` is the agent's own
# operating manual and the system prompt points at it unconditionally.
ESSENTIAL_SKILLS: frozenset = frozenset({"hermes-agent"})


def get_disabled_skill_names(platform: str | None = None) -> Set[str]:
    """Disabled skill names from config.yaml: global list ∪ platform list.

    *platform* defaults to ``HERMES_PLATFORM`` / ``HERMES_SESSION_PLATFORM``.
    """
    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return set()

    from gateway.session_context import get_session_env
    resolved_platform = (
        platform
        or os.getenv("HERMES_PLATFORM")
        or get_session_env("HERMES_SESSION_PLATFORM")
    )
    global_disabled = _normalize_string_set(skills_cfg.get("disabled"))
    if resolved_platform:
        platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(
            resolved_platform
        )
        if platform_disabled is not None:
            return (
                global_disabled | _normalize_string_set(platform_disabled)
            ) - ESSENTIAL_SKILLS
    return global_disabled - ESSENTIAL_SKILLS


def parse_config_string_list(value) -> List[str]:
    """Normalize a config value that may hold a JSON-array string into a list.

    ``hermes config set`` stores lists as quoted JSON/Python-literal strings;
    treating one as a single name would silently filter nothing (#86661). A
    scalar string still means one name (#13026).
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return []


def _normalize_string_set(values) -> Set[str]:
    return {name.strip() for name in parse_config_string_list(values) if name.strip()}


# ── External skills directories ──────────────────────────────────────────

# (config_path_str, mtime_ns) -> resolved external dirs. Called once per skill
# during banner / tool-registry scans; re-parsing config each time dominated
# cold-start.
_EXTERNAL_DIRS_CACHE: Dict[Tuple[str, int], List[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()
    _raw_config_cache_clear()


def get_external_skills_dirs() -> List[Path]:
    """Validated, deduplicated ``skills.external_dirs`` (existing dirs only).

    Entries are expanded (``~``, ``${VAR}``); relative paths resolve against
    HERMES_HOME; the local ``~/.hermes/skills/`` is skipped.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return []

    try:
        stat = config_path.stat()
        cache_key: Tuple[str, int] = (str(config_path), stat.st_mtime_ns)
    except OSError:
        cache_key = None  # type: ignore[assignment]

    if cache_key is not None:
        cached = _EXTERNAL_DIRS_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)  # copy so callers can't mutate the cache

    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return []

    raw_dirs = skills_cfg.get("external_dirs")
    if not raw_dirs:
        result: List[Path] = []
        if cache_key is not None:
            _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
        return result
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    local_skills = get_skills_dir().resolve()
    seen: Set[Path] = set()
    result = []

    for entry in raw_dirs:
        entry = str(entry).strip()
        if not entry:
            continue
        p = _expand_path(entry)
        p = (hermes_home / p).resolve() if not p.is_absolute() else p.resolve()
        if p == local_skills or p in seen:
            continue
        if p.is_dir():
            seen.add(p)
            result.append(p)
        else:
            logger.debug("External skills dir does not exist, skipping: %s", p)

    if cache_key is not None:
        _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
    return result


def get_skill_create_dir() -> Optional[Path]:
    """Configured ``skills.create_dir`` (need not exist yet), or None when unset.

    Relative paths resolve against HERMES_HOME; a value equal to the local
    skills dir counts as unset.
    """
    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return None
    raw = skills_cfg.get("create_dir")
    if not raw or not isinstance(raw, (str, os.PathLike)):
        return None
    entry = str(raw).strip()
    if not entry:
        return None

    from hermes_constants import get_hermes_home

    p = _expand_path(entry)
    if not p.is_absolute():
        p = get_hermes_home() / p
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    try:
        if resolved == get_skills_dir().resolve():
            return None
    except OSError:
        pass
    return resolved


def display_skill_create_dir() -> str:
    """User-facing path where new skills are created (``~/`` shorthand when possible).

    Used by tool schema descriptions and prompts so a configured
    ``skills.create_dir`` changes every instruction that names the path.
    """
    from hermes_constants import display_hermes_home

    create_dir = get_skill_create_dir()
    if create_dir is None:
        return f"{display_hermes_home()}/skills/"
    try:
        return "~/" + create_dir.relative_to(Path.home()).as_posix() + "/"
    except ValueError:
        return create_dir.as_posix() + "/"


def get_all_skills_dirs() -> List[Path]:
    """Skill dirs: local ``~/.hermes/skills/`` first, then create_dir, then external.

    Trusted project-local dirs are NOT included — they have *higher* precedence
    than the local dir; see :func:`get_project_skills_dirs`.
    """
    dirs = [get_skills_dir()]
    create_dir = get_skill_create_dir()
    if create_dir is not None and create_dir.is_dir():
        dirs.append(create_dir)
    for d in get_external_skills_dirs():
        if d not in dirs:
            dirs.append(d)
    return dirs


# ── Project-local skills directories ──────────────────────────────────────
# A checkout can carry skills at <root>/.hermes/skills/ or <root>/.agents/skills/
# (root = nearest ancestor with .git). TRUST GATE: skills are procedure docs an
# agent will follow, so auto-sourcing them from any cloned repo is a prompt-
# injection vector — they load only when the root is in
# ``skills.trusted_project_dirs``. Trusted project skills override same-named
# profile/bundled skills. cwd and the trust list are fixed for a session, so
# the skills index (and system prompt) stays byte-stable.

PROJECT_SKILLS_SUBDIRS = (
    os.path.join(".hermes", "skills"),
    os.path.join(".agents", "skills"),
)

# Walk-up bound: don't scan the whole filesystem on pathological cwds.
_PROJECT_ROOT_MAX_DEPTH = 64


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Nearest ancestor containing ``.git`` (dir or worktree file), or None.

    Without *start*, the surface's ``TERMINAL_CWD`` wins over process cwd so
    cron/API surfaces inherit an interactive trust decision by project identity;
    a surface with no workdir resolves no project (#48975).
    """
    try:
        if start is None:
            from agent.runtime_cwd import scope_terminal_cwd

            env_cwd = scope_terminal_cwd()
            start = Path(env_cwd) if env_cwd else Path.cwd()
        cur = Path(start).resolve()
    except OSError:
        return None
    home = Path.home().resolve()
    for _ in range(_PROJECT_ROOT_MAX_DEPTH):
        try:
            if (cur / ".git").exists():
                # A dotfiles checkout AT home would make every session
                # project-scoped; treat home itself as non-project.
                return None if cur == home else cur
        except OSError:
            return None
        if cur.parent == cur:
            return None
        cur = cur.parent
    return None


def _project_trusted_dirs_from_config() -> Set[Path]:
    """Resolved set of trusted project roots from ``skills.trusted_project_dirs``."""
    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return set()
    raw = skills_cfg.get("trusted_project_dirs")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return set()
    result: Set[Path] = set()
    for entry in raw:
        entry = str(entry).strip()
        if not entry:
            continue
        try:
            result.add(_expand_path(entry).resolve())
        except OSError:
            continue
    return result


def is_project_root_trusted(root: Path) -> bool:
    """True when *root* is listed in ``skills.trusted_project_dirs``."""
    try:
        return Path(root).resolve() in _project_trusted_dirs_from_config()
    except OSError:
        return False


def _candidate_project_skills_dirs(root: Path) -> List[Path]:
    """Existing skill dirs under *root*, excluding the profile's own skills dir.

    Matters when HERMES_HOME itself lives inside a git checkout.
    """
    local_skills = get_skills_dir().resolve()
    dirs: List[Path] = []
    for sub in PROJECT_SKILLS_SUBDIRS:
        cand = root / sub
        try:
            if cand.is_dir() and cand.resolve() != local_skills:
                dirs.append(cand.resolve())
        except OSError:
            continue
    return dirs


def _project_discovery_disabled() -> bool:
    skills_cfg = _skills_cfg()
    return skills_cfg is not None and skills_cfg.get("project_discovery") is False


def get_project_skills_dirs() -> List[Path]:
    """Trusted project-local skill dirs for the current cwd (may be empty)."""
    if _project_discovery_disabled():
        return []
    root = find_project_root()
    if root is None or not is_project_root_trusted(root):
        return []
    return _candidate_project_skills_dirs(root)


def get_untrusted_project_skills_root() -> Optional[Tuple[Path, int]]:
    """(root, skill_count) when cwd's project has skills but is NOT trusted, else None."""
    if _project_discovery_disabled():
        return None
    root = find_project_root()
    if root is None or is_project_root_trusted(root):
        return None
    count = 0
    for d in _candidate_project_skills_dirs(root):
        try:
            count += sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
        except OSError:
            continue
    if count == 0:
        return None
    return root, count


# ── Project skill quarantine (scan-time injection defense) ────────────────
# Trust is a repo-level decision made once, but repo content changes with every
# pull — without this gate a `git pull` could inject a malicious skill into an
# already-trusted repo with no scan anywhere (#48974). Every project SKILL.md
# is scanned with the hub's skills_guard scanner (content-hash cached); a
# "dangerous" verdict excludes the skill from index, list, view, and slash
# commands ("caution" loads, as on the hub). Scan cache lives under HERMES_HOME,
# never inside the repo.

_PROJECT_SCAN_SOURCE = "project-local"
# skill_dir -> quarantined; avoids re-reading the attestation JSON per call.
_PROJECT_QUARANTINE_CACHE: Dict[str, bool] = {}


def _project_scan_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "project_skill_scans"


def is_quarantined_project_skill(skill_md) -> bool:
    """True when a project skill's scan verdict is ``dangerous``.

    Fail-closed: a scanner crash or missing scanner quarantines the skill.
    Scans unconditionally — non-project callers should not call this.
    """
    skill_dir = Path(skill_md).parent
    try:
        key = str(skill_dir.resolve())
    except OSError:
        key = str(skill_dir)
    cached = _PROJECT_QUARANTINE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from tools.skills_guard import scan_skill_cached

        result, _prov = scan_skill_cached(
            skill_dir,
            source=_PROJECT_SCAN_SOURCE,
            cache_dir=_project_scan_cache_dir(),
        )
        quarantined = result.verdict == "dangerous"
        if quarantined:
            logger.warning(
                "Project skill quarantined (verdict=dangerous): %s — %s",
                skill_dir,
                result.summary,
            )
    except Exception:
        logger.warning(
            "Project skill scan failed — quarantining (fail closed): %s",
            skill_dir,
            exc_info=True,
        )
        quarantined = True
    _PROJECT_QUARANTINE_CACHE[key] = quarantined
    return quarantined


def iter_project_skill_files(project_dir: Path):
    """Yield non-quarantined SKILL.md files under a trusted project dir.

    The single iteration chokepoint for the project tier, so the quarantine
    cannot be bypassed by a new call site forgetting the check.
    """
    for skill_md in iter_skill_index_files(project_dir, "SKILL.md"):
        if not is_quarantined_project_skill(skill_md):
            yield skill_md


def normalize_skill_lookup_name(identifier: str) -> str:
    """Translate a trusted absolute skill path to the relative form ``skill_view()`` accepts.

    Slash commands and cron jobs may store absolute paths under the skills
    root or ``skills.external_dirs``; skill_view() rejects absolute names.
    """
    raw_identifier = (identifier or "").strip()
    if not raw_identifier:
        return raw_identifier

    identifier_path = Path(raw_identifier).expanduser()
    if not identifier_path.is_absolute():
        return raw_identifier.lstrip("/")

    # Resolve the primary root via tools.skills_tool at CALL time: tests patch
    # ``tools.skills_tool.SKILLS_DIR`` and skill_view() enforces ``_skills_dir()``
    # (which also follows the live profile-scoped HERMES_HOME, #67277), so
    # normalization must agree with that exact root. Import deferred (cycle).
    try:
        from tools import skills_tool as _skills_tool
        primary_root = _skills_tool._skills_dir()
    except Exception:
        primary_root = get_skills_dir()

    trusted_roots = [primary_root]
    for getter in (get_project_skills_dirs, get_external_skills_dirs):
        try:
            trusted_roots.extend(getter())
        except Exception:
            pass

    # Prefer the lexical path under a trusted root before resolving symlinks:
    # ~/.hermes/skills/<name> may be a symlink to a checkout elsewhere, and
    # resolving first would turn that trusted path into one skill_view rejects.
    for root in trusted_roots:
        try:
            return str(identifier_path.relative_to(root))
        except ValueError:
            continue

    try:
        return str(identifier_path.resolve().relative_to(primary_root.resolve()))
    except Exception:
        logger.debug(
            "Skill identifier %r is an absolute path outside trusted skills "
            "roots — passing through unchanged (skill_view will reject it)",
            raw_identifier,
        )
        return raw_identifier


def _resolve_for_skill_ownership(path) -> Path:
    path_obj = path if isinstance(path, Path) else Path(str(path))
    try:
        return path_obj.expanduser().resolve()
    except (OSError, RuntimeError):
        return path_obj.expanduser().absolute()


def is_external_skill_path(path) -> bool:
    """True when ``path`` lives under an external or trusted project skills dir.

    Those dirs are externally owned: autonomous lifecycle maintenance must
    treat them as read-only (user-directed tool calls may still edit them).
    """
    candidate = _resolve_for_skill_ownership(path)
    roots: List[Path] = list(get_external_skills_dirs())
    try:
        roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    for root in roots:
        try:
            candidate.relative_to(_resolve_for_skill_ownership(root))
            return True
        except ValueError:
            continue
    return False


# ── Frontmatter metadata extraction ───────────────────────────────────────


def _hermes_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """``metadata.hermes`` mapping from frontmatter, or ``{}`` when malformed."""
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    hermes = metadata.get("hermes") or {}
    return hermes if isinstance(hermes, dict) else {}


def extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List]:
    """Extract conditional activation fields from parsed frontmatter."""
    hermes = _hermes_metadata(frontmatter)
    return {
        "fallback_for_toolsets": hermes.get("fallback_for_toolsets", []),
        "requires_toolsets": hermes.get("requires_toolsets", []),
        "fallback_for_tools": hermes.get("fallback_for_tools", []),
        "requires_tools": hermes.get("requires_tools", []),
        # Gateway-channel gate: session platforms the skill is FOR (hidden from
        # the index elsewhere). Unlike ``platforms:`` (host OS). Empty = everywhere.
        "session_platforms": hermes.get("session_platforms", []),
    }


def extract_skill_config_vars(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ``metadata.hermes.config`` declarations (key/description/default/prompt).

    Entries missing ``key`` or ``description`` are skipped; ``prompt`` defaults
    to the description.
    """
    raw = _hermes_metadata(frontmatter).get("config")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if not key or key in seen:
            continue
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        entry: Dict[str, Any] = {"key": key, "description": desc}
        default = item.get("default")
        if default is not None:
            entry["default"] = default
        prompt_text = item.get("prompt")
        entry["prompt"] = (
            prompt_text.strip()
            if isinstance(prompt_text, str) and prompt_text.strip()
            else desc
        )
        seen.add(key)
        result.append(entry)
    return result


def discover_all_skill_config_vars() -> List[Dict[str, Any]]:
    """Config var declarations across all enabled, platform-compatible skills.

    Deduplicated by key; each dict carries a ``skill`` attribution key.
    """
    all_vars: List[Dict[str, Any]] = []
    seen_keys: set = set()

    disabled = get_disabled_skill_names()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.is_dir():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            skill_name = frontmatter.get("name") or skill_file.parent.name
            if str(skill_name) in disabled or not skill_matches_platform(frontmatter):
                continue

            for var in extract_skill_config_vars(frontmatter):
                if var["key"] not in seen_keys:
                    var["skill"] = str(skill_name)
                    all_vars.append(var)
                    seen_keys.add(var["key"])

    return all_vars


# Skill config vars are stored under skills.config.<logical key> in config.yaml.
SKILL_CONFIG_PREFIX = "skills.config"


def _resolve_dotpath(config: Dict[str, Any], dotted_key: str):
    """Walk a nested dict following a dotted key; None if any part is missing."""
    current = config
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def resolve_skill_config_values(
    config_vars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Map logical skill config keys to current values (or declared defaults).

    Path-like string values are ``~``/``${VAR}`` expanded.
    """
    config = _load_raw_config()

    resolved: Dict[str, Any] = {}
    for var in config_vars:
        logical_key = var["key"]
        value = _resolve_dotpath(config, f"{SKILL_CONFIG_PREFIX}.{logical_key}")

        if value is None or (isinstance(value, str) and not value.strip()):
            value = var.get("default", "")

        if isinstance(value, str) and ("~" in value or "${" in value):
            value = os.path.expanduser(os.path.expandvars(value))

        resolved[logical_key] = value

    return resolved


# ── Description extraction ────────────────────────────────────────────────

SKILL_PROMPT_DESC_LIMIT = 60


def _normalize_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Normalize a skill's description field for comparison/truncation."""
    raw_desc = frontmatter.get("description", "")
    return str(raw_desc).strip().strip("'\"") if raw_desc else ""


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract a system-prompt-length description from parsed frontmatter."""
    desc = _normalize_skill_description(frontmatter)
    if len(desc) > SKILL_PROMPT_DESC_LIMIT:
        return desc[:SKILL_PROMPT_DESC_LIMIT - 3] + "..."
    return desc


def is_skill_description_truncated_for_prompt(frontmatter: Dict[str, Any]) -> bool:
    """True when the description will be truncated in the system prompt skill index."""
    return len(_normalize_skill_description(frontmatter)) > SKILL_PROMPT_DESC_LIMIT


# ── File iteration ────────────────────────────────────────────────────────


def iter_skill_index_files(skills_dir: Path, filename: str):
    """Walk skills_dir yielding sorted paths matching *filename*.

    Prunes EXCLUDED_SKILL_DIRS and support dirs of skill roots. Org mirrors
    (``_org/``) are TOKEN-GATED: only the active org's subdir is walked, so
    leaving an org stops its skills resolving without manual cleanup.
    """
    skills_dir_str = str(skills_dir)
    active_org = read_active_org_id(skills_dir)
    org_root = os.path.join(skills_dir_str, ORG_MIRROR_DIR_NAME)
    matches: list[str] = []
    for root, dirs, files in os.walk(skills_dir_str, followlinks=True):
        has_skill_md = "SKILL.md" in files
        if root == skills_dir_str and ORG_MIRROR_DIR_NAME in dirs and active_org is None:
            dirs.remove(ORG_MIRROR_DIR_NAME)
        elif root == org_root:
            dirs[:] = [d for d in dirs if d == active_org]
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDED_SKILL_DIRS
            and not (has_skill_md and d in SKILL_SUPPORT_DIRS)
        ]
        if filename in files:
            matches.append(os.path.join(root, filename))
    for path in sorted(matches):
        yield Path(path)


# ── Namespace helpers for plugin-provided skills ───────────────────────────

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def parse_qualified_name(name: str) -> Tuple[Optional[str], str]:
    """Split ``'namespace:skill-name'`` into ``(namespace, bare_name)``; ``(None, name)`` without ``':'``."""
    if ":" not in name:
        return None, name
    return tuple(name.split(":", 1))  # type: ignore[return-value]


def is_valid_namespace(candidate: Optional[str]) -> bool:
    """Check whether *candidate* is a valid namespace (``[a-zA-Z0-9_-]+``)."""
    return bool(candidate) and bool(_NAMESPACE_RE.match(candidate))
