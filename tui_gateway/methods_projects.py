"""Projects RPC surface: first-class, per-profile, multi-folder workspaces, plus repo
discovery and the sidebar project tree.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# JSON-RPC error codes for the projects surface.
_E_PROJECTS = 5061  # generic failure
_E_NO_PROJECT = 5062  # id resolved to nothing
_E_PROJECT_ARG = 5063  # invalid argument (e.g. bad name/slug)


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb
    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping.

    Binds ``params['profile']`` (via ``@_profile_scoped``) so app-global remote
    mode reads that profile's ``projects.db``. Missing id maps to 5062, bad args
    to 5063, everything else to 5061.
    """

    def decorator(fn):
        @method(name)
        @_registry.profile_scoped
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb
                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))
        return handler
    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


@_projects_method("projects.list")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.get")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_projects_method("projects.create")
def _(rid, params, pdb, conn) -> dict:
    pid = pdb.create_project(
        conn, name=str(params.get("name") or ""), slug=params.get("slug"),
        folders=params.get("folders") or [], primary_path=params.get("primary_path"),
        description=params.get("description"), icon=params.get("icon"), color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(rid, {"project": proj.to_dict() if proj else None})


@_projects_method("projects.update")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.update_project(
        conn, proj.id, name=params.get("name"), description=params.get("description"),
        icon=params.get("icon"), color=params.get("color"), board_slug=params.get("board_slug"),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.add_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.add_folder(
        conn, proj.id, str(params.get("path") or ""), label=params.get("label"),
        is_primary=bool(params.get("is_primary")),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.remove_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.set_primary")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.set_primary(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.archive")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.delete")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.delete_project(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.set_active")
def _(rid, params, pdb, conn) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(rid, {"active_id": pdb.get_active_id(conn)})


@_projects_method("projects.for_cwd")
def _(rid, params, pdb, conn) -> dict:
    cwd = _completion_cwd({"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(rid, {"project": proj.to_dict() if proj else None, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)})


def _non_workspace_dirs() -> set[str]:
    """Directories that are never a workspace, whichever tier proposes them.

    The filesystem root, the user's home, and the directory homes live in —
    ``/home`` on Linux, ``/Users`` on macOS, ``C:\\Users`` on Windows. Both
    POSIX spellings are excluded on every host because both are reachable as a
    cwd anywhere: macOS ships an empty ``/home`` autofs stub, and a container or
    remote shell hands back Linux paths. Promoting one of these mints a
    catch-all project that swallows unplaced sessions, and ``/home`` in
    particular renders as a second row reading "home" next to the Home bucket.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    candidates = (os.sep, home, os.path.dirname(home), "/home", "/Users")
    return {os.path.normcase(os.path.realpath(path)) for path in candidates if path}


def _is_repo_junk(root: str) -> bool:
    """A git root we never auto-surface as a project: a non-workspace dir (see
    :func:`_non_workspace_dirs`) or anything under HERMES_HOME (~/.hermes by
    default) — config/sessions/skills, not a workspace. User-created projects
    pointing there are still honored."""
    if not root:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.realpath(root)
    hermes_home = os.path.realpath(str(get_hermes_home()))
    return (
        os.path.normcase(real) in _non_workspace_dirs()
        or real == hermes_home
        or real.startswith(hermes_home + os.sep)
    )


def _is_session_cwd_junk(cwd: str) -> bool:
    """A non-git cwd that should stay in flat Recents rather than auto-group.

    Unlike discovered git roots, an explicitly selected descendant of
    HERMES_HOME may be an intentional prose/data workspace. The pre-Projects
    desktop surfaced every such cwd, so exclude only the broad defaults that
    would create catch-all projects: HERMES_HOME itself and the dirs in
    :func:`_non_workspace_dirs`.
    """
    if not cwd:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.normcase(os.path.realpath(cwd))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real in _non_workspace_dirs() or real == hermes_home


def _repo_discovery_policy(raw: dict | None = None) -> dict:
    """Return the effective, profile-local Desktop repository scan policy."""
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (_load_cfg().get("desktop") or {})
    if not isinstance(source, dict):
        source = {}
    enabled = source.get("enabled", source.get("repo_scan_enabled", defaults["repo_scan_enabled"]))
    roots = source.get("roots", source.get("repo_scan_roots", defaults["repo_scan_roots"]))
    excludes = source.get(
        "exclude_paths", source.get("repo_scan_exclude_paths", defaults["repo_scan_exclude_paths"]),
    )
    return {
        "enabled": enabled if isinstance(enabled, bool) else defaults["repo_scan_enabled"],
        "roots": [value.strip() for value in roots if isinstance(value, str) and value.strip()]
        if isinstance(roots, list)
        else list(defaults["repo_scan_roots"]),
        "exclude_paths": [
            value.strip()
            for value in excludes
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(excludes, list)
        else list(defaults["repo_scan_exclude_paths"]),
    }


def _repo_discovery_policy_key(policy: dict) -> str:
    def _paths(values: list[str]) -> list[str]:
        normalized = set()
        home = os.path.expanduser("~")
        for value in values:
            expanded = os.path.expanduser(value)
            if not os.path.isabs(expanded):
                expanded = os.path.join(home, expanded)
            normalized.add(os.path.normcase(os.path.abspath(expanded)))
        return sorted(normalized)
    canonical = {
        "enabled": bool(policy["enabled"]), "roots": _paths(policy["roots"]),
        "exclude_paths": _paths(policy["exclude_paths"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _repo_discovery_policy_is_default(policy: dict) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG
    return _repo_discovery_policy_key(policy) == _repo_discovery_policy_key(
        _repo_discovery_policy(DEFAULT_CONFIG["desktop"])
    )


def _scan_discovered_repos_remote(conn, policy: dict) -> bool:
    """Backend-side disk scan of the discovery policy roots.

    The desktop's native repo scan only runs on the local filesystem. On a
    remote gateway connection the host must scan its own disk so repos with
    zero Hermes sessions still appear in the sidebar (#81723). Mirrors the
    desktop's behavior: walk each root (bounded depth), find `.git`
    directories, record (root, label) pairs into the discovery cache.

    Best-effort: any failure logs and leaves the cache untouched — the
    session-derived repos from `_discover_repos_payload` still surface.

    Returns True when the scan is authoritative (every root was walked to
    completion without error and the per-scan cap was not hit). Only then may
    the caller treat the result as a full replacement and pass ``replace=True``
    to the cache write — a partial or errored scan must merge, never wipe, so
    a failed remote refresh can't blank the previously cached repos into the
    silent, unpopulated sidebar of #81723.
    """
    from hermes_cli import projects_db as pdb
    roots = policy.get("roots") or []
    excludes = policy.get("exclude_paths") or []
    pairs: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    authoritative = True

    def _is_excluded(path: str) -> bool:
        return any(path == ex or path.startswith(ex.rstrip("/\\") + os.sep) for ex in excludes if ex)
    for root in roots:
        if not os.path.isdir(root):
            # `os.walk` on a missing root yields nothing instead of raising; an
            # unmounted volume would look like an empty scan and let the
            # authoritative replace wipe every cached repo under it.
            authoritative = False
            logger.debug("discover_repos scan root missing, skipping: %s", root)
            continue
        try:
            for dirpath, dirnames, _filenames in os.walk(root):
                if _is_excluded(dirpath):
                    dirnames[:] = []
                    continue
                # A `.git` directory marks this directory as a repo root. Check
                # BEFORE pruning hidden dirs — `.git` is itself hidden, so a
                # prune-first order would drop it and never detect any repo.
                if ".git" in dirnames:
                    repo_root = dirpath
                    if repo_root not in seen:
                        seen.add(repo_root)
                        pairs.append((repo_root, os.path.basename(repo_root)))
                    # Don't descend into the repo's own .git to hunt nested repos.
                    dirnames[:] = []
                else:
                    # Not a repo: skip hidden dirs (e.g. .hermes) and node_modules.
                    dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules",)]
                if len(pairs) >= 500:
                    break
        except Exception:
            # A root that can't be walked yields no authoritative set — fall back
            # to merging, never replacing, so the prior cache survives.
            authoritative = False
            logger.debug("discover_repos scan failed for root %s", root, exc_info=True)
        if len(pairs) >= 500:
            # Cap hit means the walk didn't cover the full roots; the collected
            # set must not be treated as the complete authoritative universe.
            authoritative = False
            break
    if pairs:
        try:
            pdb.record_discovered_repos(
                conn, pairs, replace=authoritative, policy_key=_repo_discovery_policy_key(policy)
            )
        except Exception:
            logger.debug("discover_repos cache write failed", exc_info=True)
            authoritative = False
    return authoritative


def _discover_repos_payload(
    db, *, conn=None, backfill: bool = True, include_cached: bool = True
) -> list[dict]:
    """Merge filesystem-scanned repos (cached) with session-derived repo roots.

    Repo-first: the disk scan (persisted by `projects.record_repos`) surfaces
    repos even with zero hermes sessions. Session-derived roots cover repos
    outside the scan roots. Both are junk-filtered (hermes home subtree + bare
    home) and carry their session totals for the overview.

    ``conn`` reuses an already-open projects.db connection (the tree path holds
    one); ``backfill`` persists resolved roots back onto session rows — kept off
    the per-turn tree path (grouping uses the live git resolver regardless) and
    done only on the explicit discover/record refresh.
    """
    _is_junk = _is_repo_junk
    repos: dict[str, dict] = {}

    def _agg(root: str) -> dict:
        return repos.setdefault(root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})

    # Session-derived roots (common repo root, folding worktrees; cached) +
    # backfill the column so persisted git_repo_root matches the tree grouping.
    cwd_rows = list(db.distinct_session_cwds())
    # Warm the per-cwd git probes in parallel so a cold first paint doesn't
    # serialize one subprocess per distinct cwd before this loop reads the cache.
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = _git_common_repo_root_for_cwd(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))
    if backfill:
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            logger.debug("failed to backfill repo roots", exc_info=True)
    if include_cached:
        # Filesystem-scanned roots from the cache (may have zero sessions). Reuse
        # the caller's projects.db connection when given, else a short-lived one.
        try:
            from hermes_cli import projects_db as pdb

            def _read(c) -> None:
                for entry in pdb.list_discovered_repos(c):
                    root = str(entry.get("root") or "")
                    if not root or _is_junk(root):
                        continue
                    agg = _agg(root)
                    if entry.get("label"):
                        agg["label"] = entry["label"]
                    # `last_seen` is scan time, not user activity; folding it
                    # into `last_active` made every scanned repo "just now".
            if conn is not None:
                _read(conn)
            else:
                with pdb.connect_closing() as own:
                    _read(own)
        except Exception:
            logger.debug("failed to read discovered repo cache", exc_info=True)
    out = sorted(repos.values(), key=lambda r: r["last_active"], reverse=True)
    for r in out:
        r["label"] = r["label"] or os.path.basename(r["root"].rstrip("/\\")) or r["root"]
    return out


# Not user conversations (cron has its own section; kanban runs are read on
# the board). Subagent/compression children are dropped by include_children=False.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban"]


def _project_tree_row(r: dict) -> dict:
    """Project a SessionDB row to the minimal shape the sidebar renders.

    Keeps the fields the grouping needs (cwd / git_branch / git_repo_root) plus
    everything ``SidebarSessionRow`` reads, and drops the heavy columns
    (system_prompt, model_config, ...) so the tree payload stays lean.
    """
    return {
        "id": r.get("id"),
        "_lineage_root_id": r.get("_lineage_root_id"),
        "_lineage_ids": r.get("_lineage_ids"),
        # The sidebar nests branch/fork sessions under their parent
        # (flattenSessionsWithBranches keys on this); without it, lane rows can't
        # draw the └─ connector the flat Recents list shows.
        "parent_session_id": r.get("parent_session_id"),
        "title": r.get("title"),
        "preview": r.get("preview"),
        "started_at": r.get("started_at") or 0,
        "ended_at": r.get("ended_at"),
        "last_active": r.get("last_active") or r.get("started_at") or 0,
        "source": r.get("source"),
        "archived": bool(r.get("archived")),
        "message_count": r.get("message_count") or 0,
        "tool_call_count": r.get("tool_call_count") or 0,
        "input_tokens": r.get("input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        # Cost is one of the fields SidebarSessionRow renders, so a lane row has
        # to carry it too — without it, switching Show → cost filled in every
        # figure in Recents and left the same sessions blank under a project.
        "actual_cost_usd": r.get("actual_cost_usd"),
        "estimated_cost_usd": r.get("estimated_cost_usd"),
        "model": r.get("model"),
        "is_active": False,
        "cwd": r.get("cwd"),
        "git_branch": r.get("git_branch"),
        "git_repo_root": r.get("git_repo_root"),
    }


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.

    ``include_discovered`` is the zero-session-repo overview tier; the entered
    view (drill-in) skips it entirely — it only needs the project it's showing,
    which already has sessions — avoiding the distinct-cwd scan + git probes on
    that per-turn path. One projects.db connection serves both reads.
    """
    rows = db.list_sessions_rich(
        limit=session_limit,
        offset=0,
        order_by_last_active=True,
        min_message_count=1,
        include_children=False,
        exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False,
        # `_project_tree_row` keeps ~18 fields and drops the rest, so selecting
        # the system-prompt blob only to discard it costs tens of MB of B-tree
        # reads per build on a long-lived database.
        compact_rows=True,
    )
    sessions = [_project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver reads it instead of
    # cold-probing each cwd in sequence (matters on the drill-in path, which
    # skips the discovery warm-up below).
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))
    from hermes_cli import projects_db as pdb
    policy = _repo_discovery_policy()
    policy_key = _repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(
                conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(policy),
            )
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = (
            _discover_repos_payload(db, conn=conn, backfill=False, include_cached=policy["enabled"])
            if include_discovered
            else []
        )
    return sessions, projects, discovered, active_id


# Per-build memo for `_dir_exists_cached`. Cleared at the top of every
# `_build_project_tree`, so a dir created or deleted between sidebar refreshes
# is seen on the next one.
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _dir_exists_cached(path: str) -> bool:
    """``os.path.isdir`` for the project tree, memoized per build.

    ``build_tree`` asks per SESSION, not per distinct path, so a power user with
    hundreds of sessions across a handful of dirs would otherwise fire hundreds
    of redundant stats on every sidebar open. The memo is per build, so a dir
    created or deleted between refreshes is picked up on the next one.
    """
    hit = _DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = os.path.isdir(path)
        _DIR_EXISTS_CACHE[path] = hit
    return hit


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree
    _DIR_EXISTS_CACHE.clear()
    sessions, projects, discovered, active_id = _project_tree_inputs(
        db, session_limit, include_discovered=include_discovered
    )
    # build_tree resolves every declared project folder and every discovered
    # repo root too, and those paths are not session cwds — without this they
    # are the one part of the build still probing git one directory at a time.
    git_probe.warm_roots(
        [str(f.get("path") or "") for p in projects for f in (p.get("folders") or [])]
        + [str(r.get("root") or "") for r in discovered]
    )
    tree = project_tree.build_tree(
        projects, sessions, discovered, _resolve_cwd_git, preview_limit=preview_limit,
        hydrate=hydrate, is_junk_root=_is_repo_junk, is_junk_cwd=_is_session_cwd_junk,
        exists=_dir_exists_cached,
    )
    return tree, active_id


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
