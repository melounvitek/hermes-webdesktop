"""Config / projects / setup JSON-RPC handlers.

Handlers and module-level helpers are rebound onto server.py's globals at
install time (see method_ctx.bind_module), so bodies reference server.py
globals bare (``_ok``, ``_err``, ``_load_cfg``, ``_sessions``, ...).
``config.set`` still lives in server.py.
"""


from .method_ctx import HandlerRegistry, bind_module

from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _reconcile_repo_discovery(pdb, conn, policy, policy_key):
    pdb.reconcile_discovered_repos_policy(
        conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(policy)
    )


@method("projects.discover_repos")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    try:
        with _profile_db(params) as db:
            if db is None:
                return _ok(rid, {"repos": []})
            from hermes_cli import projects_db as pdb

            policy = _repo_discovery_policy()
            with pdb.connect_closing() as conn:
                _reconcile_repo_discovery(pdb, conn, policy, _repo_discovery_policy_key(policy))
                # `scan=true` (desktop in remote-gateway mode): the desktop's
                # native scan only sees its local filesystem, so ask the host
                # to scan the policy roots itself so zero-session repos surface.
                if params.get("scan") and policy["enabled"]:
                    _scan_discovered_repos_remote(conn, policy)
                repos = _discover_repos_payload(db, conn=conn, include_cached=policy["enabled"])
            return _ok(rid, {"repos": repos, "discovery_policy": policy})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.record_repos")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Persist git repo roots found by the client's filesystem scan (the native
    crawl runs on the desktop), then return the merged repo list."""
    try:
        from hermes_cli import projects_db as pdb

        policy = _repo_discovery_policy()
        policy_key = _repo_discovery_policy_key(policy)
        incoming_raw = params.get("discovery_policy")
        incoming_policy = (
            _repo_discovery_policy(incoming_raw) if isinstance(incoming_raw, dict) else None
        )
        incoming_matches = (
            incoming_policy is not None
            and _repo_discovery_policy_key(incoming_policy) == policy_key
        )
        accept_legacy_default = (
            incoming_policy is None and _repo_discovery_policy_is_default(policy)
        )

        pairs: list[tuple[str, str | None]] = []
        for item in params.get("repos") or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get("root"):
                pairs.append((str(item["root"]), item.get("label")))

        with pdb.connect_closing() as conn:
            _reconcile_repo_discovery(pdb, conn, policy, policy_key)
            accepted = bool(policy["enabled"] and (incoming_matches or accept_legacy_default))
            if accepted:
                pdb.record_discovered_repos(conn, pairs, replace=True, policy_key=policy_key)
            elif not policy["enabled"]:
                pdb.clear_discovered_repos(conn, policy_key=policy_key)

        with _profile_db(params) as db:
            return _ok(
                rid,
                {
                    "repos": _discover_repos_payload(db, include_cached=policy["enabled"])
                    if db is not None
                    else [],
                    "accepted": accepted,
                    "discovery_policy": policy,
                },
            )
    except Exception as e:
        return _err(rid, 5061, str(e))


def _stamped_project_tree(db, params, **kwargs):
    """``_build_project_tree`` + profile stamping shared by the two tree RPCs."""
    from tui_gateway.project_tree import stamp_profile

    tree, active_id = _build_project_tree(db, **kwargs)
    stamp_profile(tree["projects"], _response_profile_name(params.get("profile")))
    return tree, active_id


@method("projects.tree")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Authoritative project overview: project -> repo -> lane structure with
    counts + a few preview sessions per project, plus the flat set of session
    ids claimed by any project (so the desktop excludes them from flat Recents).
    Lanes carry no session rows here; drill-in uses ``projects.project_sessions``.
    """
    try:
        with _profile_db(params) as db:
            if db is None:
                return _ok(rid, {"projects": [], "active_id": None, "scoped_session_ids": []})
            tree, active_id = _stamped_project_tree(
                db,
                params,
                preview_limit=int(params.get("preview_limit") or 3),
                hydrate=False,
                session_limit=int(params.get("session_limit") or 2000),
                include_discovered=True,
            )
            return _ok(
                rid,
                {
                    "projects": tree["projects"],
                    "active_id": active_id,
                    "scoped_session_ids": tree["scoped_session_ids"],
                },
            )
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.project_sessions")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Fully hydrated lanes (repo -> lane -> session rows) for one project,
    built from the same authoritative grouping as ``projects.tree`` so ids and
    membership match exactly."""
    try:
        project_id = str(params.get("project_id") or "")
        if not project_id:
            return _err(rid, 5063, "project_id required")

        with _profile_db(params) as db:
            if db is None:
                return _ok(rid, {"project": None})
            # Drill-in only needs the entered project (which has sessions):
            # skip the zero-session discovery tier.
            tree, _active = _stamped_project_tree(
                db,
                params,
                preview_limit=0,
                hydrate=True,
                session_limit=int(params.get("session_limit") or 5000),
                include_discovered=False,
            )
            proj = next((p for p in tree["projects"] if p["id"] == project_id), None)
            return _ok(rid, {"project": proj})
    except Exception as e:
        return _err(rid, 5061, str(e))


# ---------------------------------------------------------------------------
# config.get — one getter per key. Each returns the result payload (a dict) or
# a full ``_err`` response (dicts containing "error" pass through untouched).
# ---------------------------------------------------------------------------


def _display_cfg() -> dict:
    display = _load_cfg().get("display")
    return display if isinstance(display, dict) else {}


def _display_mode(cfg: dict, key: str, allowed: frozenset, default: str) -> str:
    raw = str((cfg.get("display") or {}).get(key, default) or default).strip().lower()
    return raw if raw in allowed else default


_DETAILS_MODES = frozenset({"hidden", "collapsed", "expanded"})
_THINKING_MODES = frozenset({"collapsed", "truncated", "full"})


def _cfg_get_provider(rid, params):
    try:
        from hermes_cli.models import list_available_providers, normalize_provider

        model = _resolve_model()
        parts = model.split("/", 1)
        return {
            "model": model,
            "provider": normalize_provider(parts[0]) if len(parts) > 1 else "unknown",
            "providers": list_available_providers(),
        }
    except Exception as e:
        return _err(rid, 5013, str(e))


def _cfg_get_profile(rid, params):
    from hermes_constants import display_hermes_home

    return {"home": str(_hermes_home), "display": display_hermes_home()}


def _cfg_get_project(rid, params):
    cfg_terminal = _load_cfg().get("terminal") or {}
    raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
    cwd = _completion_cwd({"cwd": raw} if raw else {})
    return {"cwd": cwd, "branch": _git_branch_for_cwd(cwd)}


def _cfg_get_indicator(rid, params):
    # Normalize so a hand-edited config.yaml (stray casing / unknown value)
    # reads back the SAME value the TUI rendered (frontend falls back to
    # DEFAULT_INDICATOR_STYLE for the same inputs).
    norm = str((_load_cfg().get("display") or {}).get("tui_status_indicator", "")).strip().lower()
    return {"value": norm if norm in INDICATOR_STYLES else DEFAULT_INDICATOR_STYLE}


def _cfg_get_personality(rid, params):
    # EFFECTIVE personality via the single owner — a stale/unknown name in
    # config must not display as active.
    from hermes_cli.personality import active_personality_name

    return {"value": active_personality_name(_load_cfg()) or "none"}


def _cfg_get_reasoning(rid, params):
    cfg = _load_cfg()
    session = _sessions.get(params.get("session_id", ""))
    reasoning_config = None
    if session is not None:
        if isinstance(session.get("create_reasoning_override"), dict):
            reasoning_config = session.get("create_reasoning_override")
        else:
            agent_reasoning = getattr(session.get("agent"), "reasoning_config", None)
            if isinstance(agent_reasoning, dict):
                reasoning_config = agent_reasoning

    if isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            effort = "none"
        else:
            effort = str(reasoning_config.get("effort") or "medium")
    else:
        raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
        # YAML `reasoning_effort: false` means thinking disabled, not "unset".
        effort = "none" if raw_effort is False else str(raw_effort or "medium")
    display = "show" if bool((cfg.get("display") or {}).get("show_reasoning", True)) else "hide"
    return {"value": effort, "display": display}


def _cfg_get_fast(rid, params):
    # `config.set fast` is session-scoped, so prefer the session's live/pinned
    # value over the global key; a pre-build session keeps its pin in
    # create_service_tier_override.
    session = _sessions.get(params.get("session_id", ""))
    tier = None
    if session is not None:
        agent = session.get("agent")
        if agent is not None:
            tier = getattr(agent, "service_tier", None)
        elif session.get("create_service_tier_override") is not None:
            tier = session["create_service_tier_override"]
    if tier is None:
        tier = _load_service_tier()
    return {"value": "fast" if tier == "priority" else "normal"}


def _cfg_get_approval_mode(rid, params):
    try:
        return {"value": _load_approval_mode()}
    except Exception as e:
        return _err(rid, 5001, str(e))


def _cfg_get_thinking_mode(rid, params):
    cfg = _load_cfg()
    raw = str((cfg.get("display") or {}).get("thinking_mode", "") or "").strip().lower()
    if raw in _THINKING_MODES:
        return {"value": raw}
    dm = _display_mode(cfg, "details_mode", _DETAILS_MODES, "collapsed")
    return {"value": "full" if dm == "expanded" else "collapsed"}


def _cfg_get_theme(rid, params):
    raw = str(_display_cfg().get("tui_theme", "auto")).strip().lower()
    return {"value": raw if raw in {"auto", "light", "dark"} else "auto"}


def _cfg_get_focus(rid, params):
    on = bool(_display_cfg().get("focus_view", False))
    return {"value": "on" if on else "off", "tool_progress": _load_tool_progress_mode()}


def _cfg_get_mtime(rid, params):
    cfg_path = _hermes_home / "config.yaml"
    try:
        mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
    except Exception:
        return {"mtime": 0}
    # mcp_rev: hash of the MCP-relevant config sections so the TUI's poller
    # reloads MCP servers only when their config changed — a /skin write bumps
    # mtime but must not cost a multi-second MCP reconnect.
    return {"mtime": mtime, "mcp_rev": _compute_mcp_rev()}


def _config_getters() -> dict:
    """key -> getter(rid, params). Built inside a function (not a module-level
    dict) so, once rebound onto server.py, every entry resolves to the rebound
    helper copies rather than this module's un-rebound originals."""
    return {
        "provider": _cfg_get_provider,
        "profile": _cfg_get_profile,
        "project": _cfg_get_project,
        "full": lambda rid, params: {"config": _load_cfg()},
        "prompt": lambda rid, params: {"prompt": _load_cfg().get("custom_prompt", "")},
        "skin": lambda rid, params: {"value": (_load_cfg().get("display") or {}).get("skin", "default")},
        "indicator": _cfg_get_indicator,
        "personality": _cfg_get_personality,
        "reasoning": _cfg_get_reasoning,
        "fast": _cfg_get_fast,
        "busy": lambda rid, params: {"value": _load_busy_input_mode()},
        "approval_mode": _cfg_get_approval_mode,
        "approvals.mode": _cfg_get_approval_mode,
        "details_mode": lambda rid, params: {
            "value": _display_mode(_load_cfg(), "details_mode", _DETAILS_MODES, "collapsed")
        },
        "thinking_mode": _cfg_get_thinking_mode,
        "density": lambda rid, params: {
            "value": "on" if bool((_load_cfg().get("display") or {}).get("tui_compact", False)) else "off"
        },
        "theme": _cfg_get_theme,
        "statusbar": lambda rid, params: {
            "value": _coerce_statusbar(_display_cfg().get("tui_statusbar", "top"))
        },
        "focus": _cfg_get_focus,
        "mouse": lambda rid, params: {"value": _display_mouse_tracking(_load_cfg().get("display"))},
        "mtime": _cfg_get_mtime,
    }


@method("config.get")
@_profile_scoped
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    getter = _config_getters().get(key)
    if getter is None:
        return _err(rid, 4002, f"unknown config key: {key}")
    payload = getter(rid, params)
    if "error" in payload:
        return payload
    return _ok(rid, payload)


# ---------------------------------------------------------------------------
# setup readiness
# ---------------------------------------------------------------------------


def _readiness_profile_scope(params: dict):
    """Resolve the optional ``profile`` param of the setup readiness RPCs.

    Returns ``(profile, scope)``: ``scope`` binds that profile's HERMES_HOME and
    ``.env`` secret scope (ContextVars, so concurrent checks stay isolated); the
    launch profile / no param yields ``("", nullcontext())``. A profile unknown
    to this host raises ``FileNotFoundError`` — a readiness check must never
    quietly answer for the launch profile instead.
    """
    import contextlib

    profile = str(params.get("profile") or "").strip() if isinstance(params, dict) else ""
    if not profile:
        return "", contextlib.nullcontext()
    from hermes_cli import profiles as profiles_mod

    if not profiles_mod.profile_exists(profile):
        raise FileNotFoundError(f"Profile '{profile}' does not exist on this backend.")
    home = _profile_home(profile)
    if home is None:
        return profile, contextlib.nullcontext()
    return profile, _session_profile_runtime_scope({"profile_home": str(home)})


def _readiness_check(rid, params, probe):
    """Shared shell of setup.status / setup.runtime_check.

    ``probe(profile)`` runs inside the profile scope and returns the payload;
    an unknown profile answers ``ok=False`` (never a JSON-RPC error).
    """
    try:
        profile, scope = _readiness_profile_scope(params)
    except FileNotFoundError as e:
        return _ok(rid, {"ok": False, "profile": params.get("profile"), "error": str(e)})
    with scope:
        payload = probe(profile)
    return _ok(rid, payload)


@method("setup.status")
def _(rid, params: dict) -> dict:
    """Loose provider check; ``profile`` (optional) scopes it to that profile's home."""
    try:
        from hermes_cli.main import _has_any_provider_configured

        def probe(profile):
            configured = bool(_has_any_provider_configured(strict_profile_scope=bool(profile)))
            payload = {"provider_configured": configured}
            if profile:
                payload["profile"] = profile
            return payload

        return _readiness_check(rid, params, probe)
    except Exception as e:
        return _err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: dict) -> dict:
    """Strict provider check: does the configured/default model resolve to a usable runtime?

    Unlike setup.status (True if ANY provider auth state is discoverable, incl.
    indirect fallbacks like ``gh auth token``), this runs the same
    resolve_runtime_provider() the agent uses on session creation and returns
    ok=False with the auth error when the model cannot actually be served, so
    UIs can surface onboarding before a doomed prompt. ``profile`` (optional)
    answers for THAT profile's config.yaml pin and ``.env``; an unknown profile
    answers ``ok=False`` rather than the launch profile's readiness.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured

        requested = str(params.get("provider") or "").strip() or None

        def probe(profile):
            runtime = resolve_runtime_provider(requested=requested)
            provider_configured = bool(
                _has_any_provider_configured(strict_profile_scope=bool(profile))
            )
            scoped = {"profile": profile} if profile else {}
            provider = runtime.get("provider") or "provider"
            source = str(runtime.get("source") or "")
            if (
                not provider_configured
                and provider == "bedrock"
                and source in {"iam-role", "aws-sdk-default-chain"}
            ):
                return {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": source,
                    "error": "No Hermes provider is configured.",
                    **scoped,
                }

            api_key = runtime.get("api_key")
            api_key_text = "" if callable(api_key) else str(api_key or "").strip()
            credential_ok = (
                callable(api_key)
                or api_key_text in {"aws-sdk", "no-key-required"}
                or has_usable_secret(api_key_text)
                or bool(runtime.get("command"))
            )
            if not credential_ok:
                return {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": runtime.get("source"),
                    "error": f"No usable credentials found for {provider}.",
                    **scoped,
                }
            return {
                "ok": True,
                "provider": runtime.get("provider"),
                "model": runtime.get("model"),
                "source": runtime.get("source"),
                **scoped,
            }

        return _readiness_check(rid, params, probe)
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


@method("diagnostics.share_nous")
def _(rid, params: dict) -> dict:
    """Upload a redacted debug bundle to Nous-internal diagnostics storage.

    Same collection + force-redaction pipeline as ``hermes debug share --nous``;
    redaction is NOT client-controllable. Consent lives with the CALLER (the
    desktop shows the privacy notice + Upload button first). Structured
    ``ok``/``error`` envelope rather than JSON-RPC errors so the client can
    render upload failures inline.

    Params (optional): ``error_context`` (client text about the failure,
    redacted, attached as ``error-context.txt``), ``extra_files`` ({label →
    text} client-side artifacts such as a remote desktop.log; force-redacted,
    labels sanitized and size-capped), ``log_lines`` (default 200).
    """
    try:
        from hermes_cli.debug import _redact_log_text, build_nous_bundle, collect_share_bundle
        from hermes_cli.diagnostics_upload import share_to_nous

        log_lines = params.get("log_lines")
        if not isinstance(log_lines, int) or not (10 <= log_lines <= 2000):
            log_lines = 200

        bundle = collect_share_bundle(log_lines=log_lines, redact=True)

        # Client text goes through the SAME upload-safe redactor as backend
        # logs (force secret redaction + email masking), never the weaker bare
        # secret pass.
        error_context = params.get("error_context")
        if isinstance(error_context, str) and error_context.strip():
            bundle["error-context.txt"] = _redact_log_text(error_context.strip()[:8_000])

        # Bounded: at most 4 files, 512KB each, sanitized labels — a
        # diagnostics channel, not an arbitrary upload surface.
        extra_files = params.get("extra_files")
        if isinstance(extra_files, dict):
            for label, text in list(extra_files.items())[:4]:
                if not isinstance(label, str) or not isinstance(text, str):
                    continue
                safe_label = "".join(ch for ch in label if ch.isalnum() or ch in "._- ()").strip()[:64]
                # Collapse dot-runs / leading dots so traversal-shaped labels
                # can't survive even cosmetically.
                while ".." in safe_label:
                    safe_label = safe_label.replace("..", ".")
                safe_label = safe_label.lstrip(".").strip()
                if not safe_label or not text.strip():
                    continue
                bundle[f"client/{safe_label}"] = _redact_log_text(text[:524_288])

        res = share_to_nous(build_nous_bundle(bundle, redact=True))
        view_url = res.get("viewUrl") or res.get("view_url")
        upload_id = res.get("id")
        if not view_url and not upload_id:
            # An upload the user can't reference is useless to support.
            return _ok(
                rid, {"ok": False, "error": "upload succeeded but returned no view URL or id"}
            )
        return _ok(
            rid,
            {
                "ok": True,
                "view_url": view_url,
                "upload_id": upload_id,
                "expires_at": res.get("expiresAt") or res.get("expires_at"),
            },
        )
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


def register(server) -> None:
    """Publish helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
