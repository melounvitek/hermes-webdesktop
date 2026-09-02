"""Authoritative project -> repo -> lane -> session tree builder.

Single source of truth for how the desktop sidebar groups sessions. Pure (git
resolution is injected via ``resolve``) so it is unit-testable and shared by the
``projects.tree`` / ``projects.project_sessions`` RPCs.

Emitted ids and lane keys must stay byte-compatible with the renderer's persisted
state (pins, manual ordering, dismissal), which keys off these exact strings:

  - explicit project id .......... ``p_<hex>`` (from projects.db)
  - auto/discovered project id ... the repo root path
  - home (no-project) bucket ..... ``__no_project__``
  - repo node id ................. the repo root path
  - main branch lane id .......... ``<repoRoot>::branch::<branch>`` (or ``::branch::``)
  - kanban bucket lane id ........ ``<repoRoot>::kanban``
  - linked worktree lane id ...... the worktree path

Linked worktrees are folded under their MAIN repo via a git common-dir probe
(``git rev-parse --show-toplevel`` returns the worktree's own root, which is why
the old client-side grouping double-counted them).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

# cwd -> ``{"repo_root", "worktree_root"}``: ``repo_root`` is the COMMON (main)
# repo root shared across worktrees, ``worktree_root`` this cwd's own checkout
# root. ``None`` when not in a git repo or unprobeable (remote backend).
Resolve = Callable[[str], Optional[dict]]

# "does this directory still exist?" predicate. Defaults to True-for-everything
# so callers that can't stat (remote backends) don't wrongly hide a project that
# lives on the other host.
Exists = Callable[[str], bool]

# Only KANBAN-TASK worktrees (`<repo>/.worktrees/t_<hex>`, the id kanban_db mints)
# collapse into one lane; user-named dirs under `.worktrees/` stay their own lanes.
_KANBAN_DIR_RE = re.compile(r"^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$")
_TRAILING_SEP_RE = re.compile(r"[/\\]+$")
_TRUNK_BRANCHES = {"main", "master", "trunk", "develop"}
DEFAULT_BRANCH_LABEL = "main"

# Synthetic bucket for every session no project claimed (no cwd, bare home dir,
# HERMES state, deleted workspace). The desktop labels it "Home"; the id/flag
# name what the bucket MEANS since membership keys off them.
NO_PROJECT_ID = "__no_project__"
NO_PROJECT_LABEL = "Home"

# Sibling candidates tried when recovering a deleted worktree's parent repo
# (``_probe_sibling_worktree``). Each miss costs a git probe; real suffixes are
# one or two segments.
_MAX_SIBLING_PROBES = 4


def stamp_profile(projects: list[dict], profile: str) -> None:
    """Stamp every session row with the request-scope profile (authoritative even
    for legacy rows whose ``profile_name`` is NULL) for cross-profile routing."""
    for project in projects:
        for session in project.get("previewSessions") or []:
            session["profile"] = profile
        for repo in project.get("repos") or []:
            for group in repo.get("groups") or []:
                for session in group.get("sessions") or []:
                    session["profile"] = profile


def _branch_lane_id(repo_root: str, branch: str = "") -> str:
    """The one definition of a main-checkout lane id (must match the desktop)."""
    return f"{repo_root}::branch::{(branch or '').strip()}"


def _kanban_lane_id(repo_root: str) -> str:
    return f"{repo_root}::kanban"


# ---------------------------------------------------------------------------
# Path helpers (match the TS segment logic so labels/ids line up)
# ---------------------------------------------------------------------------


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[/\\]", (path or "").rstrip("/\\")) if s]


def _is_windows_path(path: str) -> bool:
    # Drive-letter (`C:\…`), UNC (`\\srv`, `//srv`), or any backslash-rooted path
    # (`\wsl.localhost\…`, `\Users\…`). A single leading `/` stays POSIX.
    value = (path or "").strip()
    return bool(re.match(r"^[A-Za-z]:[/\\]", value)) or value.startswith(("\\", "//"))


def _comparison_segments(path: str) -> list[str]:
    """Segments for identity comparison: Windows paths casefold (even when running
    on POSIX); display paths and emitted IDs keep their spelling."""
    segs = _segments(path)
    return [s.casefold() for s in segs] if _is_windows_path(path) else segs


def _path_key(path: str) -> str:
    """Canonical comparison key (separator/trailing-slash agnostic)."""
    return "/".join(_comparison_segments(path))


def _lane_key(path_or_lane: str) -> str:
    """Canonicalize only the path portion of a lane id; branch labels stay
    byte-preserved so equivalent Windows spellings don't fork lanes."""
    for marker in ("::branch::", "::kanban"):
        if marker in path_or_lane:
            root, suffix = path_or_lane.split(marker, 1)
            return f"{_path_key(root)}{marker}{suffix}"
    return _path_key(path_or_lane)


def base_name(path: str) -> str:
    segs = _segments(path)
    return segs[-1] if segs else ""


def kanban_worktree_dir(path: str) -> Optional[str]:
    """The ``<repo>/.worktrees`` dir for a ``.../.worktrees/<task>`` path, else None."""
    m = _KANBAN_DIR_RE.match(path or "")
    return m.group(1) if m else None


def _strip_trailing_sep(path: str) -> str:
    return _TRAILING_SEP_RE.sub("", path or "")


def _with_base_name(path: str, name: str) -> str:
    return re.sub(r"[^/\\]+$", name, _strip_trailing_sep(path))


def _parent_dir(path: str) -> str:
    """The containing directory of ``path`` (``""`` once the root is passed)."""
    return _strip_trailing_sep(_with_base_name(path, ""))


def _branch_label(branch: str) -> str:
    # An unrecorded branch folds into the one trunk lane so a repo never shows two
    # "main" lanes (recorded "main" + the empty-branch bucket).
    return (branch or "").strip() or DEFAULT_BRANCH_LABEL


def _session_time(session: dict) -> float:
    return float(session.get("last_active") or session.get("started_at") or 0)


def _last_active(sessions: list[dict]) -> float:
    return max((_session_time(s) for s in sessions), default=0.0)


# ---------------------------------------------------------------------------
# Lane placement
# ---------------------------------------------------------------------------


def _placement(
    repo_root: str, lane_key: str, lane_label: str, lane_path: str, is_main: bool, is_kanban: bool
) -> dict:
    return {
        "repo_key": repo_root,
        "repo_label": base_name(repo_root) or repo_root,
        "repo_path": repo_root,
        "lane_key": lane_key,
        "lane_label": lane_label,
        "lane_path": lane_path,
        "is_main": is_main,
        "is_kanban": is_kanban,
    }


def _trunk_placement(repo_root: str, branch: str) -> dict:
    b = _branch_label(branch)
    return _placement(repo_root, _branch_lane_id(repo_root, b), b, repo_root, True, False)


def _kanban_placement(repo_root: str, kanban_dir: str) -> dict:
    return _placement(repo_root, _kanban_lane_id(repo_root), "kanban", kanban_dir, False, True)


def _probe_sibling_worktree(cwd: str, resolve: Resolve) -> str:
    """The parent repo root of a deleted ``<repo>-<suffix>`` worktree, else ``""``.

    A deleted dir can't be probed, so trim one ``-<segment>`` at a time off its
    name and return the first sibling that resolves. The cwd is frequently a
    SUBDIR of the deleted worktree (``<repo>-<suffix>/apps/desktop``) whose
    basename shares nothing with the repo, so the trim is applied to each
    ANCESTOR, deepest first — otherwise the dead path gets minted as its own
    project. Probes are bounded in total (each is a git invocation).
    """
    probes = 0
    path = _strip_trailing_sep(cwd)
    while path and probes < _MAX_SIBLING_PROBES:
        parts = base_name(path).split("-")
        for i in range(len(parts) - 1, 0, -1):
            if probes >= _MAX_SIBLING_PROBES:
                break
            probes += 1
            info = resolve(_with_base_name(path, "-".join(parts[:i])))
            if info and info.get("repo_root"):
                return (info["repo_root"] or "").strip()
        path = _parent_dir(path)
    return ""


def _place_by_heuristic(path: str) -> Optional[dict]:
    """Path-only fallback when there is no git probe and no persisted root."""
    base = base_name(path)
    if not base:
        return None
    kanban_dir = kanban_worktree_dir(path)
    if kanban_dir:
        return _kanban_placement(_strip_trailing_sep(_with_base_name(kanban_dir, "")), kanban_dir)
    m = re.match(r"^(.+)-wt-(.+)$", base)
    if m:
        return _placement(_with_base_name(path, m.group(1)), path, m.group(2), path, False, False)
    return _placement(path, _branch_lane_id(path, DEFAULT_BRANCH_LABEL), base, path, True, False)


def _place(cwd: str, branch: str, resolve: Optional[Resolve], persisted_root: str) -> Optional[dict]:
    info = resolve(cwd) if resolve else None
    if info and info.get("repo_root") and info.get("worktree_root"):
        repo_root, worktree_root = info["repo_root"], info["worktree_root"]
        if _path_key(worktree_root) == _path_key(repo_root) or info.get("is_main"):
            return _trunk_placement(repo_root, branch)
        kanban_dir = kanban_worktree_dir(worktree_root)
        if kanban_dir:
            return _kanban_placement(repo_root, kanban_dir)
        label = base_name(worktree_root) or worktree_root
        return _placement(repo_root, worktree_root, label, worktree_root, False, False)

    # No live probe: trust the backend-persisted root (split main by the recorded
    # branch). Kanban tasks still collapse by path shape.
    if persisted_root:
        kanban_dir = kanban_worktree_dir(cwd)
        if kanban_dir:
            return _kanban_placement(persisted_root, kanban_dir)
        return _trunk_placement(persisted_root, branch)

    # Unresolvable cwd: a deleted ``<repo>-<suffix>`` worktree still belongs to its
    # parent; absorb it into the trunk lane rather than stranding a dead-path lane.
    sibling_root = _probe_sibling_worktree(cwd, resolve) if resolve else ""
    if sibling_root:
        return _trunk_placement(sibling_root, branch)
    return _place_by_heuristic(cwd)


def _place_session(session: dict, resolve: Optional[Resolve]) -> Optional[dict]:
    """``_place`` for a session row; ``None`` when it has no cwd."""
    cwd = (session.get("cwd") or "").strip()
    if not cwd:
        return None
    return _place(
        cwd,
        (session.get("git_branch") or "").strip(),
        resolve,
        (session.get("git_repo_root") or "").strip(),
    )


def _session_repo_root(session: dict, resolve: Optional[Resolve]) -> str:
    """The COMMON repo root a session belongs to (folds linked worktrees)."""
    cwd = (session.get("cwd") or "").strip()
    if cwd and resolve:
        info = resolve(cwd)
        if info and info.get("repo_root"):
            return info["repo_root"]
    return (session.get("git_repo_root") or "").strip()


# ---------------------------------------------------------------------------
# Ordering + label disambiguation (parity with the old client tree)
# ---------------------------------------------------------------------------


def _lane_sort_key(group: dict) -> tuple:
    # Trunk pins to the top; the kanban aggregate sinks to the bottom; the rest
    # (branches + linked worktrees) sort by most-recent activity, then label.
    is_trunk = bool(group.get("isMain")) and group["label"].lower() in _TRUNK_BRANCHES
    return (
        0 if is_trunk else 1,
        1 if group.get("isKanban") else 0,
        -_last_active(group.get("sessions") or []),
        group["label"].lower(),
    )


def _disambiguate_labels(items: list[dict]) -> None:
    """Grow colliding basenames into path-prefixed labels (in place)."""
    by_label: dict[str, list[dict]] = {}
    for item in items:
        by_label.setdefault(item["label"], []).append(item)

    for bucket in by_label.values():
        pathed = [g for g in bucket if g.get("path")]
        if len(pathed) < 2:
            continue
        parents = {id(g): _segments(g["path"])[:-1] for g in pathed}
        max_depth = max(len(p) for p in parents.values())
        for depth in range(1, max_depth + 1):
            counts: dict[str, int] = {}
            for g in pathed:
                prefix = "/".join(parents[id(g)][-depth:])
                base = base_name(g["path"]) or g["path"]
                g["label"] = f"{prefix}/{base}" if prefix else base
                counts[g["label"]] = counts.get(g["label"], 0) + 1
            if all(c == 1 for c in counts.values()):
                break


# ---------------------------------------------------------------------------
# Repo subtree assembly
# ---------------------------------------------------------------------------


def _repo_node(root: str, label: str) -> dict:
    return {"id": root, "label": label, "path": root, "groups": [], "sessionCount": 0}


def _build_repos(sessions: list[dict], resolve: Optional[Resolve], hydrate: bool) -> list[dict]:
    """Build the ``repo -> lane -> sessions`` subtree for a set of sessions."""
    lanes: dict[str, tuple[dict, dict]] = {}  # lane identity -> (group, placement)
    for session in sessions:
        placement = _place_session(session, resolve)
        if not placement:
            continue
        lane_identity = _lane_key(placement["lane_key"])
        if lane_identity not in lanes:
            lanes[lane_identity] = (
                {
                    "id": placement["lane_key"],
                    "label": placement["lane_label"],
                    "path": placement["lane_path"],
                    "isMain": placement["is_main"],
                    "isKanban": placement["is_kanban"],
                    "sessions": [],
                },
                placement,
            )
        lanes[lane_identity][0]["sessions"].append(session)

    repos: dict[str, dict] = {}
    for group, placement in lanes.values():
        group["sessions"].sort(key=_session_time, reverse=True)
        repo = repos.setdefault(
            _path_key(placement["repo_key"]),
            _repo_node(placement["repo_key"], placement["repo_label"]),
        )
        repo["groups"].append(group)
        repo["sessionCount"] += len(group["sessions"])

    repo_list = list(repos.values())
    for repo in repo_list:
        repo["groups"] = sorted(repo["groups"], key=_lane_sort_key)
        _disambiguate_labels(repo["groups"])
        # Drop per-lane rows only AFTER sorting: _lane_sort_key derives recency
        # from them. Counts were captured above, so the overview payload stays slim.
        if not hydrate:
            for group in repo["groups"]:
                group["sessions"] = []
    _disambiguate_labels(repo_list)
    return repo_list


def _seed_folder_repos(repos: list[dict], folders: list[dict], resolve: Optional[Resolve]) -> list[dict]:
    """Ensure every declared project folder shows as a repo, even with 0 sessions.

    Without it the desktop's entered-project view renders blank (early-returns on
    no repos) and the optimistic live-session overlay has no lane to drop a
    fresh session into until a full tree refresh. Folders already covered by a
    session-derived repo (same git root) are left untouched.
    """
    seen = {_path_key(v) for repo in repos for v in (repo.get("id"), repo.get("path")) if v}
    seeded = list(repos)
    for folder in folders or []:
        raw = (folder.get("path") or "").strip()
        if not raw:
            continue
        info = resolve(raw) if resolve else None
        root = (info or {}).get("repo_root") or _strip_trailing_sep(raw)
        root_key = _path_key(root)
        if not root_key or root_key in seen:
            continue
        seeded.append(_repo_node(root, base_name(root) or root))
        seen.add(root_key)
    if len(seeded) != len(repos):
        _disambiguate_labels(seeded)
    return seeded


# ---------------------------------------------------------------------------
# Explicit-project ownership
# ---------------------------------------------------------------------------


class _FolderIndex:
    """Normalized folder path -> (owning project, depth), so a session is matched
    by walking its cwd's ancestors (O(path depth) lookups) instead of scanning
    every project x folder per session."""

    def __init__(self, projects: list[dict]) -> None:
        self._by_path: dict[str, tuple[dict, int]] = {}
        for project in projects:
            for folder in project.get("folders") or []:
                segs = _comparison_segments(folder.get("path") or "")
                if not segs:
                    continue
                key = "/".join(segs)
                # Deepest folder wins; ties keep the first project (scan order).
                existing = self._by_path.get(key)
                if existing is None or len(segs) > existing[1]:
                    self._by_path[key] = (project, len(segs))

    def match(self, target: str) -> tuple[Optional[dict], int]:
        """Owning project for ``target`` by longest ancestor folder, + its depth."""
        segs = _comparison_segments(target or "")
        for end in range(len(segs), 0, -1):
            hit = self._by_path.get("/".join(segs[:end]))
            if hit:
                return hit
        return None, -1


def _project_for_session(session: dict, index: _FolderIndex, resolve: Optional[Resolve]) -> Optional[dict]:
    cwd = (session.get("cwd") or "").strip()
    if not cwd:
        return None
    repo_root = _session_repo_root(session, resolve)
    candidates = [cwd, repo_root] if repo_root and repo_root != cwd else [cwd]
    best, best_len = None, -1
    for target in candidates:
        match, length = index.match(target)
        if match and length > best_len:
            best, best_len = match, length
    return best


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def _session_cost(session: dict) -> float:
    """A session's spend, billed if the provider reported it, else estimated."""
    for key in ("actual_cost_usd", "estimated_cost_usd"):
        if session.get(key):
            return float(session[key])
    return 0.0


def _project_node(
    *,
    pid: str,
    label: str,
    path: Optional[str],
    repos: list[dict],
    session_count: int,
    last_active: float,
    preview_sessions: list[dict],
    sessions: Optional[list[dict]] = None,
    color: Any = None,
    icon: Any = None,
    is_auto: bool = False,
    is_no_project: bool = False,
) -> dict:
    return {
        "id": pid,
        "label": label,
        "path": path,
        "color": color,
        "icon": icon,
        "isAuto": is_auto,
        "isNoProject": is_no_project,
        "sessionCount": session_count,
        "lastActive": last_active,
        # Totals over the same sessions `sessionCount` counts, so a project header
        # adds up to what its rows show.
        "totalTokens": sum((s.get("input_tokens") or 0) + (s.get("output_tokens") or 0) for s in sessions or []),
        "totalCostUsd": sum(_session_cost(s) for s in sessions or []),
        "repos": repos,
        "previewSessions": preview_sessions,
    }


def build_tree(
    projects: list[dict],
    sessions: list[dict],
    discovered_repos: list[dict],
    resolve: Optional[Resolve] = None,
    *,
    preview_limit: int = 3,
    hydrate: bool = False,
    is_junk_root: Optional[Callable[[str], bool]] = None,
    is_junk_cwd: Optional[Callable[[str], bool]] = None,
    exists: Optional[Exists] = None,
) -> dict:
    """Build the authoritative project tree.

    ``projects`` are ``projects_db.Project.to_dict()`` shapes (non-archived).
    ``sessions`` are projected session-row dicts (``id``, ``cwd``, ``git_branch``,
    ``git_repo_root``, ``started_at``, ``last_active``). ``discovered_repos`` are
    ``{"root", "label", "sessions", "last_active"}``. ``is_junk_root`` flags git
    roots that must never become an AUTO project (bare home dir, HERMES_HOME);
    ``is_junk_cwd`` is the narrower policy for non-git session folders (selected
    descendants may be intentional workspaces). User-created projects are honored
    regardless. ``exists`` keeps a DELETED workspace (removed worktree, /tmp
    scratch) from being promoted to a phantom AUTO project; omit it (remote
    backends) to keep every candidate.

    Returns ``{"projects": [...], "scoped_session_ids": [...]}``. With
    ``hydrate`` False (overview) lane ``sessions`` are emptied but counts are
    preserved and each project carries up to ``preview_limit`` ``previewSessions``;
    True (drill-in) keeps full session rows.
    """
    active_projects = [p for p in projects if not p.get("archived")]
    _junk = is_junk_root or (lambda _root: False)
    _junk_cwd = is_junk_cwd or (lambda _cwd: False)
    _exists = exists or (lambda _path: True)
    folder_index = _FolderIndex(active_projects)

    by_project: dict[str, list[dict]] = {}
    unowned: list[dict] = []
    for session in sessions:
        owner = _project_for_session(session, folder_index, resolve)
        if owner:
            by_project.setdefault(owner["id"], []).append(session)
        else:
            unowned.append(session)

    scoped_ids: list[str] = []
    result: list[dict] = []

    def _previews(project_sessions: list[dict]) -> list[dict]:
        if preview_limit <= 0:
            return []
        return sorted(project_sessions, key=_session_time, reverse=True)[:preview_limit]

    def _scope(project_sessions: list[dict]) -> None:
        scoped_ids.extend(s["id"] for s in project_sessions if s.get("id"))

    # Tier 1: explicit, user-created projects (always shown, even with 0 sessions).
    for project in active_projects:
        psessions = by_project.get(project["id"], [])
        _scope(psessions)
        result.append(
            _project_node(
                pid=project["id"],
                label=project.get("name") or project["id"],
                path=project.get("primary_path"),
                color=project.get("color"),
                icon=project.get("icon"),
                repos=_seed_folder_repos(_build_repos(psessions, resolve, hydrate), project.get("folders") or [], resolve),
                session_count=len(psessions),
                last_active=_last_active(psessions),
                preview_sessions=_previews(psessions),
                sessions=psessions,
            )
        )

    # Tier 2: auto projects from leftover sessions. Prefer the common git repo
    # root, then fall back to the session cwd for historical/non-git workspaces
    # (the pre-Projects desktop grouped every non-empty cwd; dropping that would
    # flatten those sessions into Recents on upgrade).
    by_auto_root: dict[str, dict] = {}
    homeless: list[dict] = []  # every session no tier could place -> Home bucket

    def _add_auto(root: str, session: dict) -> None:
        key = _path_key(root)
        if not key:
            homeless.append(session)
            return
        by_auto_root.setdefault(key, {"root": root, "sessions": []})["sessions"].append(session)

    for session in unowned:
        root = _session_repo_root(session, resolve)
        if root:
            # A real git root uses the stricter repo policy; never reinterpret a
            # filtered internal repo as a cwd-only project. A root no longer on
            # disk is a stale persisted value and must not resurrect as a project.
            if not _junk(root) and _exists(root):
                _add_auto(root, session)
            else:
                homeless.append(session)
            continue
        cwd = (session.get("cwd") or "").strip()
        if not cwd or _junk_cwd(cwd):
            homeless.append(session)
            continue
        placement = _place_session(session, resolve)
        # A placement that only echoes back an unresolvable cwd is the path-only
        # heuristic guessing. If that dir is also gone from disk, promoting it
        # mints a phantom project that can only be dismissed by hand -> Home.
        if placement and _exists(placement["repo_key"]):
            _add_auto(placement["repo_key"], session)
        else:
            homeless.append(session)

    seen: set[str] = set()
    for bucket in by_auto_root.values():
        auto_root, auto_sessions = bucket["root"], bucket["sessions"]
        auto_key = _path_key(auto_root)
        repos = _build_repos(auto_sessions, resolve, hydrate)
        repo_node = next(
            (r for r in repos if _path_key(r.get("id") or r.get("path") or "") == auto_key), None
        )
        if repo_node is None:
            homeless.extend(auto_sessions)
            continue
        seen.add(auto_key)
        _scope(auto_sessions)
        result.append(
            _project_node(
                pid=auto_root,
                label=base_name(auto_root) or auto_root,
                path=auto_root,
                repos=repos,
                session_count=repo_node["sessionCount"],
                last_active=_last_active(auto_sessions),
                preview_sessions=_previews(auto_sessions),
                sessions=auto_sessions,
                is_auto=True,
            )
        )

    # Tier 3: repos discovered from full history / disk scan with no loaded
    # sessions, folded to their common root and not owned by an explicit project.
    for repo in discovered_repos or []:
        raw_root = (repo.get("root") or "").strip()
        if not raw_root:
            continue
        info = resolve(raw_root) if resolve else None
        root = (info or {}).get("repo_root") or raw_root
        root_key = _path_key(root)
        if root_key in seen or _junk(root) or folder_index.match(root)[0]:
            continue
        seen.add(root_key)
        label = repo.get("label") or base_name(root) or root
        result.append(
            _project_node(
                pid=root,
                label=label,
                path=root,
                repos=[_repo_node(root, label)],
                session_count=int(repo.get("sessions") or 0),
                last_active=float(repo.get("last_active") or 0),
                preview_sessions=[],
                is_auto=True,
            )
        )

    # Auto projects are labelled by repo basename, which can collide; grow path
    # prefixes so each is distinct. Explicit projects keep their user-chosen names.
    _disambiguate_labels([p for p in result if p.get("isAuto")])

    # Tier 0: everything above could not place, so the grouped view loses no
    # session. No folder => no repo/lane structure; the one synthetic lane just
    # carries the rows. Leads the list; omitted entirely when empty.
    if homeless:
        homeless.sort(key=_session_time, reverse=True)
        _scope(homeless)
        lane = {
            "id": NO_PROJECT_ID,
            "label": NO_PROJECT_LABEL,
            "path": None,
            "isMain": False,
            "isKanban": False,
            "sessions": homeless if hydrate else [],
        }
        home_repo = {
            "id": NO_PROJECT_ID,
            "label": NO_PROJECT_LABEL,
            "path": None,
            "groups": [lane],
            "sessionCount": len(homeless),
        }
        result.insert(
            0,
            _project_node(
                pid=NO_PROJECT_ID,
                label=NO_PROJECT_LABEL,
                path=None,
                repos=[home_repo],
                session_count=len(homeless),
                last_active=_last_active(homeless),
                preview_sessions=_previews(homeless),
                sessions=homeless,
                is_no_project=True,
            ),
        )

    return {"projects": result, "scoped_session_ids": scoped_ids}
