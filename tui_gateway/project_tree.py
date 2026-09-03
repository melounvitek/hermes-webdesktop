"""Authoritative project -> repo -> lane -> session tree builder (pure; git via ``resolve``).

Emitted ids/lane keys must stay byte-compatible with the renderer's persisted state
(pins, ordering, dismissal), which keys off: explicit project id ``p_<hex>``; auto
project id / repo node id = repo root path; home bucket ``__no_project__``; main lane
``<repoRoot>::branch::<branch>``; kanban lane ``<repoRoot>::kanban``; linked worktree
lane = the worktree path. Linked worktrees fold under their MAIN repo (common-dir probe).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

# cwd -> ``{"repo_root", "worktree_root"}`` (COMMON main root shared across worktrees /
# this cwd's own checkout root); ``None`` when not in git or unprobeable (remote backend).
Resolve = Callable[[str], Optional[dict]]
# "does this directory still exist?"; defaults to always-True so callers that can't
# stat (remote backends) don't hide a project living on the other host.
Exists = Callable[[str], bool]

# Only KANBAN-TASK worktrees (`<repo>/.worktrees/t_<hex>`, the id kanban_db mints)
# collapse into one lane; user-named dirs under `.worktrees/` stay their own lanes.
_KANBAN_DIR_RE = re.compile(r"^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$")
_TRUNK_BRANCHES = {"main", "master", "trunk", "develop"}
DEFAULT_BRANCH_LABEL = "main"

# Synthetic bucket for every session no project claimed (no cwd, bare home, HERMES
# state, deleted workspace); the id/flag name what it MEANS since membership keys off them.
NO_PROJECT_ID = "__no_project__"
NO_PROJECT_LABEL = "Home"

# Sibling probes when recovering a deleted worktree's parent repo; each miss is a git
# invocation and real suffixes are one or two segments.
_MAX_SIBLING_PROBES = 4


def stamp_profile(projects: list[dict], profile: str) -> None:
    """Stamp every session row with the request-scope profile (authoritative even
    for legacy rows whose ``profile_name`` is NULL) for cross-profile routing."""
    for project in projects:
        lanes = [g for repo in project.get("repos") or [] for g in repo.get("groups") or []]
        lane_rows = [s for g in lanes for s in g.get("sessions") or []]
        for session in (project.get("previewSessions") or []) + lane_rows:
            session["profile"] = profile


def _branch_lane_id(repo_root: str, branch: str = "") -> str:
    """The one definition of a main-checkout lane id (must match the desktop)."""
    return f"{repo_root}::branch::{(branch or '').strip()}"


def _kanban_lane_id(repo_root: str) -> str:
    return f"{repo_root}::kanban"


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
    marker = next((m for m in ("::branch::", "::kanban") if m in path_or_lane), None)
    if marker is None:
        return _path_key(path_or_lane)
    root, suffix = path_or_lane.split(marker, 1)
    return f"{_path_key(root)}{marker}{suffix}"


def base_name(path: str) -> str:
    segs = _segments(path)
    return segs[-1] if segs else ""


def kanban_worktree_dir(path: str) -> Optional[str]:
    """The ``<repo>/.worktrees`` dir for a ``.../.worktrees/<task>`` path, else None."""
    m = _KANBAN_DIR_RE.match(path or "")
    return m.group(1) if m else None


def _with_base_name(path: str, name: str) -> str:
    return re.sub(r"[^/\\]+$", name, (path or "").rstrip("/\\"))


def _parent_dir(path: str) -> str:
    """The containing directory of ``path`` (``""`` once the root is passed)."""
    return _with_base_name(path, "").rstrip("/\\")


def _field(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def _branch_label(branch: str) -> str:
    # An unrecorded branch folds into the one trunk lane so a repo never shows two
    # "main" lanes (recorded "main" + the empty-branch bucket).
    return (branch or "").strip() or DEFAULT_BRANCH_LABEL


def _session_time(session: dict) -> float:
    return float(session.get("last_active") or session.get("started_at") or 0)


def _last_active(sessions: list[dict]) -> float:
    return max((_session_time(s) for s in sessions), default=0.0)


def _placement(
    repo_root: str, lane_key: str, lane_label: str, lane_path: str, is_main: bool, is_kanban: bool
) -> dict:
    return {
        "repo_key": repo_root, "repo_label": base_name(repo_root) or repo_root,
        "lane_key": lane_key, "lane_label": lane_label, "lane_path": lane_path,
        "is_main": is_main, "is_kanban": is_kanban,
    }


def _trunk_placement(repo_root: str, branch: str) -> dict:
    b = _branch_label(branch)
    return _placement(repo_root, _branch_lane_id(repo_root, b), b, repo_root, True, False)


def _kanban_placement(repo_root: str, kanban_dir: str) -> dict:
    return _placement(repo_root, _kanban_lane_id(repo_root), "kanban", kanban_dir, False, True)


def _probe_sibling_worktree(cwd: str, resolve: Resolve) -> str:
    """The parent repo root of a deleted ``<repo>-<suffix>`` worktree, else ``""``.

    A deleted dir can't be probed, so trim one ``-<segment>`` at a time off its name
    and return the first sibling that resolves. The cwd is often a SUBDIR of the dead
    worktree (``<repo>-<suffix>/apps/desktop``), so the trim runs on each ANCESTOR,
    deepest first. Probes are bounded in total (each is a git invocation).
    """
    probes = 0
    path = (cwd or "").rstrip("/\\")
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
        return _kanban_placement(_parent_dir(kanban_dir), kanban_dir)
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
    cwd = _field(session, "cwd")
    if not cwd:
        return None
    return _place(cwd, _field(session, "git_branch"), resolve, _field(session, "git_repo_root"))


def _session_repo_root(session: dict, resolve: Optional[Resolve]) -> str:
    """The COMMON repo root a session belongs to (folds linked worktrees)."""
    cwd = _field(session, "cwd")
    if cwd and resolve:
        info = resolve(cwd)
        if info and info.get("repo_root"):
            return info["repo_root"]
    return _field(session, "git_repo_root")


def _lane_sort_key(group: dict) -> tuple:
    # Trunk pins to the top; the kanban aggregate sinks to the bottom; the rest
    # (branches + linked worktrees) sort by most-recent activity, then label.
    is_trunk = bool(group.get("isMain")) and group["label"].lower() in _TRUNK_BRANCHES
    return (0 if is_trunk else 1, 1 if group.get("isKanban") else 0,
            -_last_active(group.get("sessions") or []), group["label"].lower())


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
        for depth in range(1, max(len(p) for p in parents.values()) + 1):
            for g in pathed:
                prefix = "/".join(parents[id(g)][-depth:])
                base = base_name(g["path"]) or g["path"]
                g["label"] = f"{prefix}/{base}" if prefix else base
            if len({g["label"] for g in pathed}) == len(pathed):
                break


# Lane group wire fields <- placement keys (same order).
_LANE_FIELDS = ("id", "label", "path", "isMain", "isKanban")
_PLACEMENT_LANE_KEYS = ("lane_key", "lane_label", "lane_path", "is_main", "is_kanban")


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
            group = dict(zip(_LANE_FIELDS, (placement[k] for k in _PLACEMENT_LANE_KEYS)))
            group["sessions"] = []
            lanes[lane_identity] = (group, placement)
        lanes[lane_identity][0]["sessions"].append(session)

    repos: dict[str, dict] = {}
    for group, placement in lanes.values():
        group["sessions"].sort(key=_session_time, reverse=True)
        repo_key = placement["repo_key"]
        repo = repos.setdefault(_path_key(repo_key), _repo_node(repo_key, placement["repo_label"]))
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
    """Ensure every declared project folder shows as a repo, even with 0 sessions:
    otherwise the desktop's entered-project view renders blank and the optimistic
    live-session overlay has no lane for a fresh session until a full refresh.
    Folders already covered by a session-derived repo (same git root) are untouched."""
    seen = {_path_key(v) for repo in repos for v in (repo.get("id"), repo.get("path")) if v}
    seeded = list(repos)
    for folder in folders or []:
        raw = _field(folder, "path")
        if not raw:
            continue
        info = resolve(raw) if resolve else None
        root = (info or {}).get("repo_root") or raw.rstrip("/\\")
        root_key = _path_key(root)
        if not root_key or root_key in seen:
            continue
        seeded.append(_repo_node(root, base_name(root) or root))
        seen.add(root_key)
    if len(seeded) != len(repos):
        _disambiguate_labels(seeded)
    return seeded


class _FolderIndex:
    """Normalized folder path -> (owning project, depth): a session is matched by
    walking its cwd's ancestors instead of scanning every project x folder."""

    def __init__(self, projects: list[dict]) -> None:
        self._by_path: dict[str, tuple[dict, int]] = {}
        for project in projects:
            for folder in project.get("folders") or []:
                segs = _comparison_segments(folder.get("path") or "")
                # Deepest folder wins; ties keep the first project (scan order).
                if segs and len(segs) > self._by_path.get("/".join(segs), (None, -1))[1]:
                    self._by_path["/".join(segs)] = (project, len(segs))

    def match(self, target: str) -> tuple[Optional[dict], int]:
        """Owning project for ``target`` by longest ancestor folder, + its depth."""
        segs = _comparison_segments(target or "")
        for end in range(len(segs), 0, -1):
            hit = self._by_path.get("/".join(segs[:end]))
            if hit:
                return hit
        return None, -1


def _project_for_session(session: dict, index: _FolderIndex, resolve: Optional[Resolve]) -> Optional[dict]:
    cwd = _field(session, "cwd")
    if not cwd:
        return None
    repo_root = _session_repo_root(session, resolve)
    candidates = [cwd, repo_root] if repo_root and repo_root != cwd else [cwd]
    # Longest folder match wins; ties keep the cwd match (max() keeps the first maximum).
    return max((index.match(t) for t in candidates), key=lambda hit: hit[1])[0]


def _session_cost(session: dict) -> float:
    """A session's spend, billed if the provider reported it, else estimated."""
    for key in ("actual_cost_usd", "estimated_cost_usd"):
        if session.get(key):
            return float(session[key])
    return 0.0


def _project_node(
    pid: str, label: str, path: Optional[str], repos: list[dict], session_count: int,
    last_active: float, preview_sessions: list[dict], sessions: Optional[list[dict]] = None,
    **flags: Any,
) -> dict:
    """``flags`` overrides ``color`` / ``icon`` / ``isAuto`` / ``isNoProject`` (key order is
    fixed by the defaults below — the renderer's wire shape)."""
    node = {
        "id": pid, "label": label, "path": path, "color": None, "icon": None,
        "isAuto": False, "isNoProject": False,
        "sessionCount": session_count, "lastActive": last_active,
        # Totals over the same sessions `sessionCount` counts, so a project header
        # adds up to what its rows show.
        "totalTokens": sum((s.get("input_tokens") or 0) + (s.get("output_tokens") or 0) for s in sessions or []),
        "totalCostUsd": sum(_session_cost(s) for s in sessions or []),
        "repos": repos, "previewSessions": preview_sessions,
    }
    node.update(flags)
    return node


def _auto_buckets(
    unowned: list[dict], resolve: Optional[Resolve], junk: Callable, junk_cwd: Callable,
    exists: Callable,
) -> tuple[dict[str, dict], list[dict]]:
    """Group leftover sessions by auto-project root; the rest go to the Home bucket.
    Prefer the common git root, then the session cwd for non-git workspaces (the
    pre-Projects desktop grouped every cwd; dropping that flattens them into Recents)."""
    by_auto_root: dict[str, dict] = {}
    homeless: list[dict] = []

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
            if not junk(root) and exists(root):
                _add_auto(root, session)
            else:
                homeless.append(session)
            continue
        cwd = _field(session, "cwd")
        if not cwd or junk_cwd(cwd):
            homeless.append(session)
            continue
        placement = _place_session(session, resolve)
        # A placement that only echoes back an unresolvable cwd is the path-only
        # heuristic guessing. If that dir is also gone from disk, promoting it
        # mints a phantom project that can only be dismissed by hand -> Home.
        if placement and exists(placement["repo_key"]):
            _add_auto(placement["repo_key"], session)
        else:
            homeless.append(session)
    return by_auto_root, homeless


def _home_project(homeless: list[dict], hydrate: bool, previews: list[dict]) -> dict:
    """The synthetic Home bucket: no folder => no repo/lane structure, one lane carries the rows."""
    lane = {
        "id": NO_PROJECT_ID, "label": NO_PROJECT_LABEL, "path": None, "isMain": False,
        "isKanban": False, "sessions": homeless if hydrate else [],
    }
    home_repo = {
        "id": NO_PROJECT_ID, "label": NO_PROJECT_LABEL, "path": None, "groups": [lane],
        "sessionCount": len(homeless),
    }
    return _project_node(
        NO_PROJECT_ID, NO_PROJECT_LABEL, None, [home_repo], len(homeless), _last_active(homeless),
        previews, homeless, isNoProject=True)


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
    exists: Optional[Exists] = None) -> dict:
    """Build the authoritative project tree -> ``{"projects", "scoped_session_ids"}``.

    ``is_junk_root`` flags git roots that must never become an AUTO project (bare home,
    HERMES_HOME); ``is_junk_cwd`` is the narrower policy for non-git folders; explicit
    projects are honored regardless. ``exists`` keeps a DELETED workspace from becoming
    a phantom AUTO project (omit on remote backends). ``hydrate`` False (overview) empties
    lane ``sessions`` but keeps counts + ``preview_limit`` ``previewSessions``.
    """
    active_projects = [p for p in projects if not p.get("archived")]
    _junk = is_junk_root or (lambda _root: False)
    _junk_cwd = is_junk_cwd or (lambda _cwd: False)
    _exists = exists or (lambda _path: True)
    folder_index = _FolderIndex(active_projects)
    by_project: dict[str, list[dict]] = {}  # explicit project id -> owned rows
    unowned: list[dict] = []
    for session in sessions:
        owner = _project_for_session(session, folder_index, resolve)
        (by_project.setdefault(owner["id"], []) if owner else unowned).append(session)

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
        repos = _build_repos(psessions, resolve, hydrate)
        repos = _seed_folder_repos(repos, project.get("folders") or [], resolve)
        result.append(_project_node(
            project["id"], project.get("name") or project["id"], project.get("primary_path"), repos,
            len(psessions), _last_active(psessions), _previews(psessions), psessions,
            color=project.get("color"), icon=project.get("icon")))

    # Tier 2: auto projects from leftover sessions.
    by_auto_root, homeless = _auto_buckets(unowned, resolve, _junk, _junk_cwd, _exists)
    seen: set[str] = set()
    for bucket in by_auto_root.values():
        auto_root, auto_sessions = bucket["root"], bucket["sessions"]
        auto_key = _path_key(auto_root)
        repos = _build_repos(auto_sessions, resolve, hydrate)
        repo_node = next(
            (r for r in repos if _path_key(r.get("id") or r.get("path") or "") == auto_key), None)
        if repo_node is None:
            homeless.extend(auto_sessions)
            continue
        seen.add(auto_key)
        _scope(auto_sessions)
        result.append(_project_node(
            auto_root, base_name(auto_root) or auto_root, auto_root, repos,
            repo_node["sessionCount"], _last_active(auto_sessions), _previews(auto_sessions),
            auto_sessions, isAuto=True))

    # Tier 3: repos discovered from full history / disk scan with no loaded
    # sessions, folded to their common root and not owned by an explicit project.
    for repo in discovered_repos or []:
        raw_root = _field(repo, "root")
        if not raw_root:
            continue
        info = resolve(raw_root) if resolve else None
        root = (info or {}).get("repo_root") or raw_root
        root_key = _path_key(root)
        if root_key in seen or _junk(root) or folder_index.match(root)[0]:
            continue
        seen.add(root_key)
        label = repo.get("label") or base_name(root) or root
        result.append(_project_node(
            root, label, root, [_repo_node(root, label)], int(repo.get("sessions") or 0),
            float(repo.get("last_active") or 0), [], isAuto=True))

    # Auto projects are labelled by repo basename, which can collide; grow path
    # prefixes so each is distinct. Explicit projects keep their user-chosen names.
    _disambiguate_labels([p for p in result if p.get("isAuto")])

    # Tier 0: everything above could not place, so the grouped view loses no
    # session. Leads the list; omitted entirely when empty.
    if homeless:
        homeless.sort(key=_session_time, reverse=True)
        _scope(homeless)
        result.insert(0, _home_project(homeless, hydrate, _previews(homeless)))

    return {"projects": result, "scoped_session_ids": scoped_ids}
