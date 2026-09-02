"""Profile JSON-RPC handlers — the ws twin of the dashboard's /api/profiles.

Desktop plugins reach the backend only through the ws JSON-RPC door, so bot rosters
and profile pickers need profile enumeration/creation/editing here, delegating to
the same `hermes_cli.profiles` primitives the REST endpoints use.

Every function here is rebound onto server.py's globals at install time
(method_ctx.bind_module): bodies use server.py globals bare (`_ok`, `_err`, `os`,
`json`, `Path`, `is_truthy_value`, `get_hermes_home`, `*_hermes_home_override`,
`_profile_ui_meta_lock`), and module-level names are published onto server.py, so
they must not collide with its own globals.
"""

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# ext -> mime; iteration order is the on-disk lookup order for assets.
_ASSET_EXTS = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
# session.list's deny-list: sub-agent and kanban dispatcher workers.
_WORKER_SOURCES = frozenset({"kanban", "tool"})


def _lazy(module, name):
    """Late-bound attribute lookup (the wrapped modules are heavy / cyclic at import time).

    Uses the ``__import__`` builtin: rebound bodies only see server.py globals, so this
    module's own imports (e.g. ``importlib``) are NOT available here.
    """
    return getattr(__import__(module, fromlist=[name]), name)


def _pin_profile_model(profile_dir, provider, model) -> None:
    _lazy("hermes_cli.web_routers.profiles", "_write_profile_model")(profile_dir, provider, model)


def _launch_mcp_catalog() -> dict:
    mcp = (_lazy("hermes_cli.config", "load_config_readonly")() or {}).get("mcp_servers")
    return mcp if isinstance(mcp, dict) else {}


def _try(fn, default):
    """``fn()`` or ``default`` on any exception — best-effort sections must never fail each other."""
    try:
        return fn()
    except Exception:
        return default


def _best_effort(fn) -> bool:
    """Run ``fn``; True on success, False on any exception."""
    return _try(lambda: (fn(), True)[1], False)


def _read_text_if_file(path) -> str:
    return _try(lambda: path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "", "")


@contextlib.contextmanager
def _hermes_home_scope(path):
    """Scope config/auth resolution to ``path`` for the block."""
    token = set_hermes_home_override(str(path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _profile_dir_or_err(rid, name):
    """``(profile_dir, None)`` for an existing profile, else ``(None, 4064 error)``."""
    from hermes_cli.profiles import get_profile_dir
    profile_dir = Path(get_profile_dir(name))
    if not profile_dir.is_dir():
        return None, _err(rid, 4064, f"profile '{name}' not found")
    return profile_dir, None


def _resolve_profile(rid, params):
    """``(name, profile_dir, err)`` — err is the 4063 (name required) / 4064 (not found) response."""
    name = str(params.get("name") or "").strip()
    if not name:
        return name, None, _err(rid, 4063, "name required")
    profile_dir, err = _profile_dir_or_err(rid, name)
    return name, profile_dir, err


def _read_profile_yaml(profile_dir) -> dict:
    """profile.yaml as a mapping; ``{}`` when missing, unparseable, or not a mapping."""
    import yaml

    meta_path = profile_dir / "profile.yaml"
    loaded = (yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}) if meta_path.is_file() else {}
    return loaded if isinstance(loaded, dict) else {}


def _clean_revisions(raw: dict) -> dict:
    """Normalise a ``_ui_meta_revisions`` map: str keys, non-bool ints clamped at 0."""
    return {str(k): max(0, int(v)) for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)}


def _latest_message_preview(db, session_id):
    """Excerpt (≤80 chars) of the NEWEST active user/assistant message, or "".

    Messaging-app semantics for rosters (latest exchange), unlike the first-message
    preview session lists use for recognition.  Agent-delivery prefixes are kept.
    Same query shape as ``SessionDB.latest_message_row_id`` — keep them in step.
    """
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT content FROM messages"
                " WHERE session_id = ? AND role IN ('user', 'assistant')"
                " AND active = 1 AND content IS NOT NULL AND TRIM(content) != ''"
                " ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
    except Exception:
        return ""
    if not row:
        return ""
    text = " ".join(str(row[0] or "").split()).strip()
    return text[:80] + "..." if len(text) > 80 else text


def _open_profile_session_db_readonly(profile_path):
    """Read-only attach for roster previews, or None.

    A writable ``SessionDB()`` waits up to 20s for the write lock and runs schema init;
    the roster polls every 5s while the live backend holds the writer, which stalled the
    RPC past the desktop timeout.  ``read_only=True`` takes neither the lock nor the DDL.
    """
    db_path = Path(profile_path) / "state.db"
    if not _try(db_path.exists, False):
        return None
    return _try(lambda: _lazy("hermes_state", "SessionDB")(db_path=db_path, read_only=True), None)


def _resurrect_recoverable_canonical(db, profile_path, session_id):
    """Un-archive an accidentally archived canonical row, or False.

    Recoverability is judged on the read-only handle first; the write uses a
    short-lived writable handle so the 20s lock patience is never paid on the poll.
    """
    try:
        row = db.get_session(session_id)
        if not row or not row.get("archived"):
            return False
        tip_id = _try(lambda: db.get_compression_tip(session_id), None) or session_id
        tip = (_try(lambda: db.get_session(tip_id), None) or row) if tip_id != session_id else row
        from hermes_state import SessionDB, get_shared_session_db
        if (tip.get("end_reason") or "") not in SessionDB.RECOVERABLE_END_REASONS:
            return False
        wdb = get_shared_session_db(Path(profile_path) / "state.db")
        try:
            return bool(wdb.unarchive_recoverable_session(session_id))
        finally:
            _best_effort(lambda: _lazy("hermes_state", "release_or_close")(wdb))
    except Exception:
        return False


def _canonical_session_row(db, profile_path):
    """Summary of the profile's canonical "Bot Chat" registry row, or None.

    Identity is the NAME (UNIQUE(title) ⇒ ≤1 row), so preview and click target agree
    without a client pointer.  Exact lookup: hidden rows resolve (canonical chats are
    always hidden); lineages resolve via ``get_compression_tip``, NOT the resume walker
    whose unmarked-child fallback can pick an ordinary child.  Worker sources count as
    absent.  ``id`` is the durable registry row, ``resolved_id`` the live tip.
    """
    if db is None:
        return None
    try:
        row = db.get_session_by_title("Bot Chat")
        if not row:
            return None
        session_id = str(row.get("id") or "").strip()
        if not session_id or (row.get("source") or "").strip().lower() in _WORKER_SOURCES:
            return None
        # Archived usually means the user retired it — report absent — but the
        # ws-orphan reaper / older cleanup can archive by accident: resurrect those.
        if row.get("archived") and not _resurrect_recoverable_canonical(db, profile_path, session_id):
            return None
        tip = _try(lambda: db.get_compression_tip(session_id), None) or session_id
        tip_row = db.get_session(tip) or row
        started = row.get("started_at") or 0
        return {
            "id": session_id,
            "resolved_id": tip,
            "root_title": row.get("title") or "",
            "title": tip_row.get("title") or "",
            "preview": _latest_message_preview(db, tip),
            "started_at": tip_row.get("started_at") or started,
            "last_active": tip_row.get("last_activity_at") or tip_row.get("started_at") or started,
            "message_count": tip_row.get("message_count") or 0,
        }
    except Exception:
        return None


def _latest_profile_session_rows(db):
    """(newest human-facing session, newest worker session) for a profile.

    The second is the newest DENIED row so rosters can show a profile as working
    even though worker sessions never surface in conversation lists (workers
    heartbeat ``last_activity_at`` every ≤60s; the client picks a liveness window).
    """
    if db is None:
        return None, None
    try:
        human = worker = None
        for s in db.list_sessions_rich(source=None, limit=20, order_by_last_active=True, compact_rows=True):
            src = (s.get("source") or "").strip().lower()
            title = s.get("title") or ""
            last_active = s.get("last_active") or s.get("started_at") or 0
            if src in _WORKER_SOURCES:
                if worker is None:
                    worker = {"id": s["id"], "source": src, "title": title, "last_active": last_active}
                continue
            if human is not None:
                continue
            human = {
                "id": s["id"],
                "title": title,
                "preview": s.get("preview") or "",
                "started_at": s.get("started_at") or 0,
                "last_active": last_active,
                "message_count": s.get("message_count") or 0,
            }
            # Rosters want "where the conversation IS": prefer the newest text.
            human["preview"] = _latest_message_preview(db, s["id"]) or human["preview"]
            if worker is not None:
                break
        return human, worker
    except Exception:
        return None, None


def _profile_session_fields(row, profile_path):
    """Attach last_session / worker_session / canonical_session to a roster row."""
    db = _open_profile_session_db_readonly(profile_path)
    try:
        row["last_session"], row["worker_session"] = _latest_profile_session_rows(db)
        # Resolved server-side on every listing so no client carries a session pointer.
        row["canonical_session"] = _canonical_session_row(db, profile_path)
    finally:
        if db is not None:
            _best_effort(db.close)


@method("profiles.list")
def _(rid, params: dict) -> dict:
    """List Hermes profiles (name, path, model, description, skill count).

    ``include_sessions`` (default true) adds ``last_session`` / ``worker_session``
    / ``canonical_session`` so a roster paints per-agent previews without N calls.
    """
    try:
        from hermes_cli.profiles import list_profiles
        include_sessions = is_truthy_value(params.get("include_sessions", True))
        out = []
        for p in list_profiles():
            row = {
                "name": p.name,
                "path": str(p.path),
                "is_default": bool(p.is_default),
                "model": p.model,
                "provider": p.provider,
                "description": getattr(p, "description", "") or "",
                "display_name": getattr(p, "display_name", "") or "",
                "skill_count": getattr(p, "skill_count", 0) or 0,
            }
            if include_sessions:
                _profile_session_fields(row, p.path)
            # Client-agnostic UI metadata lives in profile.yaml so every client paints
            # the same roster.  ``ui_meta_revisions`` is always present: it
            # feature-detects gateway-owned CAS even for a brand-new profile.
            profile_dir = Path(str(p.path))
            row["ui_meta_revisions"] = {}
            raw_meta = _try(lambda: _read_profile_yaml(profile_dir), {})
            ui_meta = raw_meta.get("ui_meta")
            if isinstance(ui_meta, dict) and ui_meta:
                row["ui_meta"] = ui_meta
            revisions = raw_meta.get("_ui_meta_revisions")
            if isinstance(revisions, dict) and revisions:
                row["ui_meta_revisions"] = _try(lambda: _clean_revisions(revisions), {})
            # Cheap existence flag so rosters skip a get_asset probe per paint.
            row["has_avatar"] = _try(lambda: any((profile_dir / "assets" / f"avatar.{e}").is_file() for e in _ASSET_EXTS), False)
            out.append(row)
        # Capability flag: this backend injects the Bot Mode teammate-messaging
        # protocol into every session, so clients must not append it to SOUL.md.
        return _ok(rid, {"profiles": out, "bot_mode_protocol": True})
    except Exception as e:
        return _err(rid, 5061, str(e))


def _has_real_env_content(env_path) -> bool:
    """True when .env has any non-comment, non-blank line."""
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return any(s and not s.startswith("#") for s in (line.strip() for line in lines))


def _copy_secret_file(src, dst) -> None:
    import shutil
    shutil.copy2(src, dst)
    with contextlib.suppress(OSError):
        os.chmod(str(dst), 0o600)


def _mirror_env(path, launch_home) -> bool:
    """Copy the launch .env only over the seeded comment-only stub (never a clone's secrets)."""
    src, dst = launch_home / ".env", path / ".env"
    if not (src.is_file() and _has_real_env_content(src) and not _try(lambda: _has_real_env_content(dst), False)):
        return False
    _copy_secret_file(src, dst)
    return True


def _mirror_auth(path, launch_home) -> bool:
    """Copy the launch auth.json when absent, dropping single-use OAuth grants.

    Skipped under ``share_auth`` so the profile reads token state via the global-root
    fallback (refreshes write through): a copy forks token state and the first refresh
    in either store strands the other.  Static .env keys have no refresh semantics.
    """
    src, dst = launch_home / "auth.json", path / "auth.json"
    if not (src.is_file() and not dst.exists()):
        return False
    _copy_secret_file(src, dst)
    # Never fork single-use OAuth grants (Anthropic / Codex / xAI): the first profile
    # to refresh strands every sibling.  API keys stay; OAuth rows are dropped and
    # read from the root grant via the pool fallback.
    _best_effort(lambda: _lazy("hermes_cli.auth", "strip_cloned_single_use_oauth_grants")(path))
    return True


def _mirror_voice_sections(path) -> bool:
    """Copy voice config (stt/tts/voice) from the launch profile; True if written.

    Dictation/TTS resolve ``stt`` inside the TARGET profile's home, and a fresh
    profile has only a ``model`` section, so voice fell back to defaults.  Goes
    through the canonical loaders under the home override (config-read-guard).
    """
    try:
        from hermes_cli.config import load_config_readonly, read_user_config_raw, save_config
        src_cfg = load_config_readonly() or {}
        sections = {k: src_cfg[k] for k in ("stt", "tts", "voice") if src_cfg.get(k)}
        if not sections:
            return False
        with _hermes_home_scope(path):
            # Round-trip the RAW file: load_config() merges DEFAULT_CONFIG, making every
            # section look present (no-op mirror) and save_config would then persist
            # the whole default tree into the fresh profile.
            dst_cfg = read_user_config_raw() or {}
            missing = {k: v for k, v in sections.items() if k not in dst_cfg}
            if missing:
                dst_cfg.update(missing)
                save_config(dst_cfg)
        return bool(missing)
    except Exception:
        return False


def _inherit_launch_model(path) -> bool:
    """Inherit the launch profile's model.provider/default when the new profile has none.

    Gate on the MODEL SECTION being absent, not on config.yaml existing: voice
    mirroring legitimately creates the file first, and a file-existence gate
    silently skipped inheritance for every non-clone bot.  Clones keep theirs.
    """
    from hermes_cli.config import load_config_readonly, read_user_config_raw

    with _hermes_home_scope(path):
        dst_model = (read_user_config_raw() or {}).get("model") or {}
    if dst_model.get("provider") and dst_model.get("default"):
        return False
    model_cfg = (load_config_readonly() or {}).get("model") or {}
    provider, model = str(model_cfg.get("provider") or ""), str(model_cfg.get("default") or "")
    if not (provider and model):
        return False
    _pin_profile_model(path, provider, model)
    return True


@method("profiles.create")
def _(rid, params: dict) -> dict:
    """Create a profile — the ws twin of POST /api/profiles.

    Params: ``name`` (lowercase slug), ``description``, ``clone_from`` (omitted =
    fresh profile with bundled skills), ``clone_all``, ``no_skills``, ``soul``,
    ``model`` + ``provider`` (optional pin), ``share_auth``, ``mirror_credentials``
    (default true: copy the launch .env, auth.json and voice sections; inherit its
    model when unpinned).  Mirroring exists because ``create_profile()`` seeds a
    comment-only .env and never copies auth.json, so a headlessly created profile had
    NO inference provider and no interactive ``hermes setup`` to recover.
    """
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4061, "name required")
    try:
        from hermes_cli import profiles as profiles_mod
        clone_from = str(params.get("clone_from") or "").strip() or None
        clone_all = is_truthy_value(params.get("clone_all", False))
        path = profiles_mod.create_profile(
            name=name,
            clone_from=clone_from,
            clone_all=clone_all,
            clone_config=bool(clone_from) and not clone_all,
            no_skills=is_truthy_value(params.get("no_skills", False)),
            description=str(params.get("description") or "").strip() or None,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        return _err(rid, 4062, str(e))
    except Exception as e:
        return _err(rid, 5062, str(e))

    # Mirror the CLI/REST create flow: bundled skills for fresh profiles, then the
    # alias wrapper.  Both best-effort.
    if not clone_from:
        _best_effort(lambda: profiles_mod.seed_profile_skills(path, quiet=True))
    _best_effort(lambda: profiles_mod.check_alias_collision(name) or profiles_mod.create_wrapper_script(name))

    soul = params.get("soul")
    soul_written = False
    if isinstance(soul, str) and soul.strip():
        soul_written = _best_effort(lambda: (path / "SOUL.md").write_text(soul, encoding="utf-8"))

    mirrored = {"env": False, "auth": False, "model_inherited": False, "voice": False}
    share_auth = is_truthy_value(params.get("share_auth", False))
    if share_auth:
        mirrored["auth"] = "shared"
    mirror = is_truthy_value(params.get("mirror_credentials", True))
    if mirror:
        launch_home = get_hermes_home()
        mirrored["env"] = _try(lambda: _mirror_env(path, launch_home), False)
        if not share_auth:
            mirrored["auth"] = _try(lambda: _mirror_auth(path, launch_home), False)
        mirrored["voice"] = _mirror_voice_sections(path)

    model = str(params.get("model") or "").strip()
    provider = str(params.get("provider") or "").strip()
    model_set = False
    if model and provider:
        model_set = _best_effort(lambda: _pin_profile_model(path, provider, model))
    elif mirror:
        mirrored["model_inherited"] = _try(lambda: _inherit_launch_model(path), False)

    return _ok(
        rid,
        {"ok": True, "name": name, "path": str(path), "soul_written": soul_written, "model_set": model_set, "mirrored": mirrored},
    )


def _describe_toolsets(cfg):
    """``(toolsets, pinned_set)`` as the `hermes tools` checklist presents them.

    Configurable universe minus platform-restricted, enablement resolved as the runtime
    does.  The raw registry leaks internal platform composites and reports everything
    "enabled" when the profile has no pin.
    """
    from hermes_cli.tools_config import _get_effective_configurable_toolsets, _get_platform_tools, _toolset_allowed_for_platform
    from toolsets import resolve_toolset
    pinned = (cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}).get("enabled_toolsets")
    pinned_set = _clean_names(pinned) if isinstance(pinned, list) else None
    platform_enabled = _try(lambda: set(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False)), set())
    default_off = _try(lambda: _lazy("hermes_cli.tools_config", "_DEFAULT_OFF_TOOLSETS"), set())
    toolsets_out = []
    for ts_name, ts_label, ts_desc in _get_effective_configurable_toolsets():
        if not _toolset_allowed_for_platform(ts_name, "cli"):
            continue
        enabled = ts_name in pinned_set if pinned_set is not None else ts_name in platform_enabled
        # Default-off integrations (a2a, spotify, ...) and the equally opt-in yuanbao
        # are noise in a per-profile editor unless already enabled.
        if (ts_name in default_off or ts_name == "yuanbao") and not enabled:
            continue
        tool_count = _try(lambda: len(set(resolve_toolset(ts_name))), 0)
        toolsets_out.append(
            {"name": ts_name, "label": ts_label, "description": ts_desc or "", "tool_count": tool_count, "enabled": enabled}
        )
    return toolsets_out, pinned_set


def _describe_mcp_servers(cfg):
    """``[{name, enabled, transport}]`` for the profile's ``mcp_servers`` (best-effort)."""
    mcp_cfg = cfg.get("mcp_servers")
    if not isinstance(mcp_cfg, dict):
        return []
    return _try(
        lambda: [
            {
                "name": str(srv_name),
                "enabled": not is_truthy_value(entry.get("disabled", False)),
                "transport": str(entry.get("transport") or "http") if entry.get("url") else "stdio",
            }
            for srv_name in sorted(mcp_cfg.keys())
            for entry in (mcp_cfg[srv_name],)
            if isinstance(entry, dict)
        ],
        [],
    )


@method("profiles.describe")
def _(rid, params: dict) -> dict:
    """Full configuration snapshot of one profile, for an editor UI.

    Result: ``{name, description, soul, model: {provider, default}, skills:
    [{name, enabled}], toolsets: [...], toolsets_pinned, mcp_servers}``.  Skill
    enablement mirrors the disabled-list model (installed = enabled unless in
    ``skills.disabled``).  All reads are scoped to the profile via the home override.
    """
    try:
        name, profile_dir, err = _resolve_profile(rid, params)
        if err is not None:
            return err
        with _hermes_home_scope(profile_dir):
            from hermes_cli.config import load_config
            from hermes_cli.skills_config import get_disabled_skills
            cfg = load_config() or {}
            disabled = {s.lower() for s in get_disabled_skills(cfg)}
            skills_root = profile_dir / "skills"
            installed = [
                {"name": md.parent.name, "enabled": md.parent.name.lower() not in disabled}
                for md in (sorted(skills_root.rglob("SKILL.md")) if skills_root.is_dir() else ())
            ]
            toolsets_out, pinned_set = _describe_toolsets(cfg)
            soul = _read_text_if_file(profile_dir / "SOUL.md")
            mcp_out = _describe_mcp_servers(cfg)
            model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
            meta = _try(lambda: _lazy("hermes_cli.profiles", "read_profile_meta")(profile_dir), {})
            result = {
                "name": name,
                "description": str(meta.get("description") or ""),
                "soul": soul,
                "model": {"provider": str(model_cfg.get("provider") or ""), "default": str(model_cfg.get("default") or "")},
                "skills": installed,
                "toolsets": toolsets_out,
                "toolsets_pinned": pinned_set is not None,
                "mcp_servers": mcp_out,
            }
            return _ok(rid, result)
    except Exception as e:
        return _err(rid, 5063, str(e))


def _configure_ui_meta(profile_dir, params, applied) -> None:
    """Merge ``params["ui_meta"]`` key-wise into profile.yaml (None deletes a key).

    Size-capped (64KB) because it rides profiles.list on every roster paint.
    ``ui_meta_expected_revisions`` are per-key CAS preconditions; any mismatch rejects
    the whole write.  Revisions survive deletion so a stale client cannot recreate a
    removed key by presenting the initial revision.
    """
    try:
        incoming = params["ui_meta"]
        if len(json.dumps(incoming)) > 65536:
            applied["ui_meta"] = False
            return
        expected = params.get("ui_meta_expected_revisions")
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("ui_meta_expected_revisions must be an object")
        with _profile_ui_meta_lock:
            existing = _try(lambda: _read_profile_yaml(profile_dir), {})
            raw_revisions = existing.get("_ui_meta_revisions")
            revisions = _clean_revisions(raw_revisions if isinstance(raw_revisions, dict) else {})
            conflicts = {}
            for key in incoming if isinstance(expected, dict) else ():
                wanted, actual = expected.get(key), revisions.get(key, 0)
                if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 0 or wanted != actual:
                    conflicts[key] = {"expected": wanted, "actual": actual}
            if conflicts:
                applied["ui_meta"] = False
                applied["ui_meta_conflicts"] = conflicts
                applied["ui_meta_revisions"] = {key: revisions.get(key, 0) for key in incoming}
                return
            current = existing.get("ui_meta")
            current = current if isinstance(current, dict) else {}
            for key, value in incoming.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
                revisions[key] = revisions.get(key, 0) + 1
            if current:
                existing["ui_meta"] = current
            else:
                existing.pop("ui_meta", None)
            existing["_ui_meta_revisions"] = revisions
            from utils import atomic_yaml_write
            atomic_yaml_write(profile_dir / "profile.yaml", existing, sort_keys=False)
            applied["ui_meta"] = True
            applied["ui_meta_revisions"] = {key: revisions[key] for key in incoming}
    except Exception:
        applied["ui_meta"] = False


def _configure_model(profile_dir, params, applied):
    """Apply a ``model`` + ``provider`` pin; returns a confirm message instead of writing.

    Same handshake as ``config.set model``: without ``confirm_expensive_model`` a
    guarded (data-policy / expensive) pick answers ``confirm_required`` and writes
    NOTHING; the client resends with the flag once confirmed.  A misbehaving guard
    must never break the save (treated as "no warning"), matching ``_apply_model_switch``.
    """
    model = str(params.get("model") or "").strip()
    provider = str(params.get("provider") or "").strip()
    confirm_message = None
    if not (model and provider):
        return None
    if not is_truthy_value(params.get("confirm_expensive_model", False)):
        confirm_message = _try(
            lambda: getattr(
                _lazy("hermes_cli.model_selection_guards", "combined_selection_warning")(model, provider=provider or None),
                "message",
                None,
            ),
            None,
        )
    if confirm_message is None:
        applied["model"] = _best_effort(lambda: _pin_profile_model(profile_dir, provider, model))
    return confirm_message


def _configure_cfg_sections(profile_dir, params, applied) -> None:
    """Apply ``disabled_skills`` / ``enabled_toolsets`` / ``enabled_mcp_servers`` (replace semantics).

    An empty ``enabled_toolsets`` clears the pin.  ``enabled_mcp_servers`` toggles the
    ``disabled`` flag; enabling a server the profile doesn't define copies its
    definition from the LAUNCH profile's catalog — unknown names are skipped, never
    invented.  Server defs are config, not secrets; credentials stay in .env/auth.
    """
    want_mcp = isinstance(params.get("enabled_mcp_servers"), list)
    # Launch profile's MCP catalog, read BEFORE the home override flips config
    # resolution to the target profile.
    launch_mcp = _try(_launch_mcp_catalog, {}) if want_mcp else {}

    with _hermes_home_scope(profile_dir):
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
        if isinstance(params.get("disabled_skills"), list):
            try:
                from hermes_cli.skills_config import save_disabled_skills
                save_disabled_skills(cfg, _clean_names(params["disabled_skills"]))
                applied["skills"] = True
                cfg = load_config() or {}
            except Exception:
                applied["skills"] = False
        if isinstance(params.get("enabled_toolsets"), list):
            applied["toolsets"] = _best_effort(lambda: _save_toolset_pin(cfg, params["enabled_toolsets"], save_config))
        if want_mcp:
            applied["mcp_servers"] = _best_effort(
                lambda: _save_mcp_toggles(load_config() or {}, params["enabled_mcp_servers"], launch_mcp, save_config)
            )


def _clean_names(values) -> set:
    return {str(v).strip() for v in values if str(v).strip()}


def _save_toolset_pin(cfg, enabled, save_config) -> None:
    wanted = sorted(_clean_names(enabled))
    tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
    if wanted:
        tools_cfg["enabled_toolsets"] = wanted
    else:
        tools_cfg.pop("enabled_toolsets", None)
    cfg["tools"] = tools_cfg
    save_config(cfg)


def _save_mcp_toggles(cfg, enabled, launch_mcp, save_config) -> None:
    wanted = _clean_names(enabled)
    mcp_cfg = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
    for srv in wanted:
        if srv in mcp_cfg and isinstance(mcp_cfg[srv], dict):
            mcp_cfg[srv].pop("disabled", None)
        elif srv in launch_mcp and isinstance(launch_mcp[srv], dict):
            mcp_cfg[srv] = dict(launch_mcp[srv])
            mcp_cfg[srv].pop("disabled", None)
    for srv, entry in mcp_cfg.items():
        if srv not in wanted and isinstance(entry, dict):
            entry["disabled"] = True
    if mcp_cfg:
        cfg["mcp_servers"] = mcp_cfg
    save_config(cfg)


@method("profiles.configure")
def _(rid, params: dict) -> dict:
    """Apply configuration changes to a profile (editor Save).

    Params: ``name`` plus any of ``ui_meta`` (+ ``ui_meta_expected_revisions``), ``soul``,
    ``description``, ``model`` + ``provider`` (+ ``confirm_expensive_model``),
    ``disabled_skills``, ``enabled_toolsets``, ``enabled_mcp_servers``.  Sections are
    independent and best-effort; ``applied`` reports per-section success.
    """
    try:
        _name, profile_dir, err = _resolve_profile(rid, params)
        if err is not None:
            return err
        applied = {}
        if isinstance(params.get("ui_meta"), dict):
            _configure_ui_meta(profile_dir, params, applied)
        if isinstance(params.get("soul"), str):
            applied["soul"] = _best_effort(lambda: (profile_dir / "SOUL.md").write_text(params["soul"], encoding="utf-8"))
        if isinstance(params.get("description"), str):
            applied["description"] = _best_effort(
                lambda: _lazy("hermes_cli.profiles", "write_profile_meta")(
                    profile_dir, description=params["description"].strip(), description_auto=False
                )
            )
        confirm_message = _configure_model(profile_dir, params, applied)
        if any(isinstance(params.get(k), list) for k in ("disabled_skills", "enabled_toolsets", "enabled_mcp_servers")):
            _configure_cfg_sections(profile_dir, params, applied)

        result = {"ok": all(applied.values()) if applied else True, "applied": applied}
        if confirm_message is not None:
            # Same shape config.set returns, so clients reuse one confirm handler.
            result["confirm_required"] = True
            result["confirm_message"] = confirm_message
        return _ok(rid, result)
    except Exception as e:
        return _err(rid, 5064, str(e))


def _sniff_asset_ext(blob):
    """Extension for a PNG/JPEG/WebP blob by magic bytes (never trust declared mime), or None."""
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if blob[:3] == b"\xff\xd8\xff":
        return "jpg"
    return "webp" if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP" else None


def _unlink_asset_files(assets_dir, asset) -> int:
    """Delete every ``<asset>.<ext>`` in ``assets_dir``; returns how many existed."""
    present = [t for t in (assets_dir / f"{asset}.{ext}" for ext in _ASSET_EXTS) if t.is_file()]
    for target in present:
        target.unlink()
    return len(present)


@method("profiles.set_asset")
def _(rid, params: dict) -> dict:
    """Store a small binary asset (avatar image) as ``assets/<asset>.<ext>``, atomically.

    Params: ``name``, ``asset`` (only ``"avatar"``), ``data`` (data URL or raw base64;
    PNG/JPEG/WebP; decoded ≤2MB), or ``clear: true``.  Result: ``{ok, asset, size}``.
    """
    asset = str(params.get("asset") or "avatar").strip().lower()
    if not str(params.get("name") or "").strip():
        return _err(rid, 4063, "name required")
    if asset != "avatar":
        return _err(rid, 4066, f"unknown asset '{asset}' (supported: avatar)")
    try:
        import base64
        import re
        _name, profile_dir, err = _resolve_profile(rid, params)
        if err is not None:
            return err
        assets_dir = profile_dir / "assets"
        if is_truthy_value(params.get("clear", False)):
            removed = _unlink_asset_files(assets_dir, asset)
            return _ok(rid, {"ok": True, "asset": asset, "size": 0, "removed": removed})
        data = str(params.get("data") or "")
        if not data:
            return _err(rid, 4067, "data required (data URL or base64)")
        match = re.match(r"^data:(image/(?:png|jpeg|webp));base64,(.*)$", data, re.DOTALL)
        try:
            blob = base64.b64decode(match.group(2) if match else data, validate=True)
        except Exception:
            return _err(rid, 4068, "data is not valid base64")
        if len(blob) > 2_000_000:
            return _err(rid, 4069, f"asset too large ({len(blob)} bytes; max 2MB)")
        ext = _sniff_asset_ext(blob)
        if ext is None:
            return _err(rid, 4070, "unsupported image format (PNG/JPEG/WebP only)")
        assets_dir.mkdir(parents=True, exist_ok=True)
        _unlink_asset_files(assets_dir, asset)  # one canonical file per asset
        target = assets_dir / f"{asset}.{ext}"
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(target)
        return _ok(rid, {"ok": True, "asset": asset, "size": len(blob)})
    except Exception as e:
        return _err(rid, 5065, str(e))


@method("profiles.get_asset")
def _(rid, params: dict) -> dict:
    """Fetch a profile asset as a data URL: ``{found, data?, mime?, size?}``.

    ``found: false`` (not an error) when absent, so rosters can probe cheaply.
    """
    asset = str(params.get("asset") or "avatar").strip().lower()
    try:
        import base64
        _name, profile_dir, err = _resolve_profile(rid, params)
        if err is not None:
            return err
        for ext, mime in _ASSET_EXTS.items():
            target = profile_dir / "assets" / f"{asset}.{ext}"
            if target.is_file():
                blob = target.read_bytes()
                data = f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"
                return _ok(rid, {"found": True, "mime": mime, "size": len(blob), "data": data})
        return _ok(rid, {"found": False})
    except Exception as e:
        return _err(rid, 5066, str(e))


def register(server) -> None:
    bind_module(globals(), server)
