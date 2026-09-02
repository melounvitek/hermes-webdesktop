"""Session / delegation / spawn-tree / billing / pet JSON-RPC handlers.

Handler bodies are rebound onto server.py's globals at install time (see
method_ctx.py), so they reference server helpers (``_sessions``, ``_ok``,
``_err``, ...) bare. Module-level helpers defined here are published onto
server.py by :func:`register` the same way, so handlers and helpers share one
namespace (and tests that monkeypatch ``server.X`` still intercept).
"""

import contextlib
from dataclasses import dataclass

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── shared handler plumbing ──────────────────────────────────────────


def _with_session(fn):
    """Resolve ``params.session_id`` via ``_sess_nowait`` and pass the record as a 3rd arg."""

    def handler(rid, params: dict) -> dict:
        session, err = _sess_nowait(params, rid)
        if err:
            return err
        return fn(rid, params, session)

    return handler


def _with_live_session(fn):
    """Like :func:`_with_session` but via ``_sess`` (waits for the agent build)."""

    def handler(rid, params: dict) -> dict:
        session, err = _sess(params, rid)
        if err:
            return err
        return fn(rid, params, session)

    return handler


def _new_runtime_ids(params: dict) -> tuple[str, str]:
    """Fresh runtime sid + resolved DB ``source`` for a session minted from ``params``."""
    return (
        uuid.uuid4().hex[:8],
        _resolve_session_source(str(params.get("source") or "").strip() or None),
    )


@contextlib.contextmanager
def _profile_build_scope(profile_home):
    """Bind HERMES_HOME + the profile's secret scope while building/initializing an agent.

    The home override alone only moves config/skills/memory; credentials resolve
    through get_secret(), which without a scope falls through to the LAUNCH
    profile's .env — so both are installed together. No-op for the launch profile.
    """
    if not profile_home:
        yield
        return
    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
    try:
        yield
    finally:
        reset_hermes_home_override(home_token)
        reset_secret_scope(secret_token)


def _branch_title(db, parent_key: str) -> str:
    """Next title in the parent's lineage (mirrors the TUI /branch naming)."""
    current = db.get_session_title(parent_key) or "branch"
    if hasattr(db, "get_next_title_in_lineage"):
        return db.get_next_title_in_lineage(current)
    return f"{current} (branch)"


def _cwd_info(session: dict, cwd: str, branch=None) -> dict:
    """session.info after a cwd change: the full agent view, or the lazy shape."""
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent, session)
    return {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd) if branch is None else branch,
        "project": _project_info_for_cwd(cwd),
        "lazy": True,
    }


def _session_row_summary(row: dict, *, tip_row: dict | None = None, resolved_id=None) -> dict:
    """Compact session.list row; ``tip_row``/``resolved_id`` come from the compression tip."""
    tip_row = tip_row or row
    out = {"id": row["id"]}
    if resolved_id is not None:
        out["resolved_id"] = resolved_id
    out.update(
        {
            "title": row.get("title") or "",
            "preview": tip_row.get("preview") or "",
            "started_at": row.get("started_at") or 0,
            "message_count": tip_row.get("message_count") or 0,
            "source": row.get("source") or "",
        }
    )
    return out


# Sources hidden from human-facing listings: ``tool`` sub-agent runs and
# ``kanban`` dispatcher workers. A deny-list (not an allow-list) so new
# platforms / custom HERMES_SESSION_SOURCE values surface automatically.
_LISTING_DENY_SOURCES = frozenset({"kanban", "tool"})


def _denied_source(row: dict) -> bool:
    return (row.get("source") or "").strip().lower() in _LISTING_DENY_SOURCES


def _pet_display_cfg() -> dict:
    """``display.pet`` config block, ``{}`` when config is unreadable."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        return display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        return {}


def _pet_guard(name: str, *, fail_open=None):
    """Wrap a pet handler so any exception is logged at debug and never breaks the surface.

    ``fail_open`` is the result payload to return (``pet.info`` style); without it
    the caller gets ``_err(5031, "<name> failed: ...")``.
    """

    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                return fn(rid, params)
            except Exception as exc:  # noqa: BLE001 - cosmetic surface
                logger.debug("%s failed: %s", name, exc)
                if fail_open is not None:
                    return _ok(rid, fail_open(params) if callable(fail_open) else dict(fail_open))
                return _err(rid, 5031, f"{name} failed: {exc}")

        return handler

    return deco


def _billing_call(rid, fn, extra: dict | None = None) -> dict:
    """Run a portal call; typed BillingError → serialized envelope, anything else → generic.

    ``extra`` is appended to both ERROR envelopes (e.g. the idempotency key the
    TUI reuses on retry); the success payload is whatever ``fn`` returns.
    """
    from hermes_cli.nous_billing import BillingError

    try:
        return _ok(rid, fn())
    except BillingError as exc:
        return _ok(rid, {**_serialize_billing_error(exc), **(extra or {})})
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), **(extra or {})})


def _billing_invalid(rid, message: str, error: str = "invalid_request") -> dict:
    return _ok(rid, {"ok": False, "error": error, "message": message})


# ── session.create / list / most_recent / facts ──────────────────────


def _create_branch_rows(
    db, new_key: str, parent_key: str, title: str, history: list, *, source, cwd, profile_name, copy_fields=()
) -> None:
    """Create a branch child row + copy the parent transcript in bounded-chunk transactions.

    ``_branched_from`` is the stable marker that keeps the branch visible in
    list_sessions_rich(): the TUI branch leaves the parent live (no
    end_reason='branched'), so the legacy end_reason heuristic never matches it.
    ``profile_name`` is stamped explicitly (not just parent-backfill) — NULL rows
    drop out of profile-keyed sidebar matching and deep-link resolution.
    """
    db.create_session(
        new_key,
        source=source,
        model=_resolve_model(),
        model_config={"_branched_from": parent_key},
        parent_session_id=parent_key,
        cwd=cwd,
        profile_name=profile_name,
    )
    db.append_messages_batch(
        new_key,
        [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content"),
                **{field: msg.get(field) for field in copy_fields},
            }
            for msg in history
        ],
        chunk_rows=500,
    )
    db.set_session_title(new_key, title)


def _seed_branch_row(sid: str, key: str, parent_session_id: str, history: list, source: str, profile_home) -> None:
    """Persist a seeded desktop branch child up front (the one session.create exception to lazy rows).

    A branch carries parent_session_id AND a seeded transcript — explicit intent, not
    an abandoned draft. The renderer's post-create resume re-fetches the child via REST
    + defer_history hydration, both of which read the DB, so an unpersisted child 404s
    and the fail-latch spins forever. Best-effort: on failure the lazy first-prompt
    path stays as the fallback, exactly as for plain drafts.
    """
    try:
        with _session_db(_sessions[sid]) as db:
            if db is None:
                return
            branch_title = _branch_title(db, parent_session_id)
            try:
                _create_branch_rows(
                    db,
                    key,
                    parent_session_id,
                    branch_title,
                    history,
                    source=source,
                    cwd=_sessions[sid]["cwd"],
                    profile_name=(Path(profile_home).name if profile_home else None),
                )
            except Exception as exc:
                # Compensation: if the transcript copy / title write failed AFTER the
                # row committed, a durable-but-empty row would defeat the INSERT OR
                # IGNORE first-prompt seed. Roll back just this child so it can retry.
                from hermes_state import is_disk_full_error

                if is_disk_full_error(exc):
                    raise
                try:
                    db.delete_session(key)
                except Exception:
                    logger.debug("branch seed compensation delete failed for %s", key, exc_info=True)
                raise
            _sessions[sid]["pending_title"] = None
    except Exception:
        logger.warning(
            "seeded-branch persistence failed for %s; falling back to lazy row creation",
            key,
            exc_info=True,
        )


@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    history = _coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    # A branch: copies an existing conversation and links back so list_sessions_rich
    # keeps it visible and the sidebar nests it (mirrors the TUI /branch marker).
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    # Only an explicitly chosen (existing) workspace is persisted as the session's
    # cwd (_ensure_session_db_row); the gateway launch dir fallback lands in "No workspace".
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    except Exception:
        explicit_cwd = False
    resolved_cwd = _completion_cwd(params)
    source = _resolve_session_source(str(params.get("source") or "").strip() or None)
    _enable_gateway_prompts()

    # ``profile`` (app-global remote mode): build + persist against THAT profile's
    # home/state.db. Stored on the session so the build and every turn re-bind HERMES_HOME.
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # Composer model/effort/fast are PER-SESSION overrides, never a global config
    # write. provider is optional (resolved at build).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )
    create_reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        try:
            from hermes_constants import parse_reasoning_effort

            create_reasoning_override = parse_reasoning_effort(effort)
        except Exception:
            create_reasoning_override = None
    # ``fast`` presence is the contract: omitted inherits the profile, true pins
    # priority, false pins normal ("" — _make_agent uses None for inheritance).
    create_service_tier_override = None
    if "fast" in params:
        create_service_tier_override = "priority" if is_truthy_value(params.get("fast")) else ""

    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": threading.Event(),
            "attached_images": [],
            "close_on_disconnect": is_truthy_value(params.get("close_on_disconnect", False)),
            "active_session_lease": None,  # claimed lazily on the first turn (_ensure_active_session_slot)
            "cols": cols,
            "created_at": now,
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "image_counter": 0,
            "cwd": resolved_cwd,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "pending_hidden": is_truthy_value(params.get("hidden", False)),
            "room_plumbing": is_truthy_value(params.get("room_plumbing", False)),
            "follow_profile_config": is_truthy_value(params.get("follow_profile_config", False)),
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": _load_show_reasoning(),
            "source": source,
            "slash_worker": None,
            "tool_progress_mode": _load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
        }
        _register_session_cwd(_sessions[sid])

    # No DB row here: every launch/draft opens a session just to paint the
    # composer, and eager rows left "Untitled" litter. The row is created lazily
    # on the first prompt (_ensure_session_db_row + prompt.submit) — except for
    # seeded branch children, which must exist immediately.
    if parent_session_id and history:
        _seed_branch_row(sid, key, parent_session_id, history, source, profile_home)

    # Return immediately so Ink can paint; the real AIAgent builds right after
    # this response is flushed (no first prompt needed to hydrate tools/skills).
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

    return _ok(
        rid,
        {
            "session_id": sid,
            "stored_session_id": key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": {
                # Reflect the per-session model override immediately so the client
                # doesn't briefly clobber its sticky pick with the global default.
                "model": (
                    session_model_override.get("model") if session_model_override else _resolve_model()
                ),
                **(
                    {"provider": session_model_override["provider"]}
                    if session_model_override and session_model_override.get("provider")
                    else {}
                ),
                "tools": {},
                "skills": {},
                "cwd": _sessions[sid]["cwd"],
                "branch": _git_branch_for_cwd(_sessions[sid]["cwd"]),
                "project": _project_info_for_cwd(_sessions[sid]["cwd"]),
                "lazy": True,
                "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                "profile_name": _response_profile_name(profile),
            },
        },
    )


def _session_list_by_title(rid, db, title_lookup: str) -> dict:
    """EXACT-title registry lookup (not a listing) for callers that treat a title as identity.

    Hidden rows resolve (canonical chats are born hidden); archived rows and
    deny-listed sources do not; compression lineages resolve to the live tip
    (``resolved_id``), mirroring profiles.list's canonical_session resolver.
    """
    row = db.get_session_by_title(title_lookup)
    if row and row.get("archived"):
        from tools.bot_mode_probe import BOT_CHAT_TITLE

        # The canonical Bot Chat is identity-scoped: an archive stamped by the
        # ws-orphan reaper / agent_close is an accident, and hiding it makes the
        # desktop mint transient replacements forever. Resurrect recoverable
        # reasons only; deliberate archives still hide. Re-fetch by ID — title
        # has no DB-level UNIQUE, so a title re-query could grab a duplicate.
        if title_lookup == BOT_CHAT_TITLE and db.unarchive_recoverable_session(row["id"]):
            row = db.get_session(row["id"])
    if not row or row.get("archived") or _denied_source(row):
        return _ok(rid, {"sessions": []})
    try:
        # Only a real compression continuation: the generic resume resolver's
        # legacy unmarked-child fallback could redirect the canonical Bot Chat
        # to an unrelated normal child.
        tip = db.get_compression_tip(row["id"]) or row["id"]
    except Exception:
        tip = row["id"]
    tip_row = (db.get_session(tip) or row) if tip != row["id"] else row
    return _ok(rid, {"sessions": [_session_row_summary(row, tip_row=tip_row, resolved_id=tip)]})


@method("session.list")
def _(rid, params: dict) -> dict:
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5006)
        try:
            # Older clients never send ``title``; newer clients on older gateways
            # just get the windowed listing back and scan it.
            title_lookup = str(params.get("title") or "").strip()
            if title_lookup:
                return _session_list_by_title(rid, db, title_lookup)

            limit = int(params.get("limit", 200) or 200)
            # ``include_hidden``: only for surfaces that OWN hidden sessions (Bots
            # pane, plugin pickers); off for the resume picker and every global caller.
            include_hidden = is_truthy_value(params.get("include_hidden", False))
            # Over-fetch so per-source filtering (and tip projection merging in
            # list_sessions_rich) doesn't leave us short.
            fetch_limit = max(limit * 2, 200)
            rows = [
                s
                for s in db.list_sessions_rich(
                    source=None,
                    limit=fetch_limit,
                    order_by_last_active=True,
                    compact_rows=True,
                    include_hidden=include_hidden,
                )
                if not _denied_source(s)
            ][:limit]
            return _ok(rid, {"sessions": [_session_row_summary(s) for s in rows]})
        except Exception as e:
            return _err(rid, 5006, str(e))


@method("session.most_recent")
def _(rid, params: dict) -> dict:
    """Most recent human-facing session id, or ``None`` (same deny-list as session.list).

    ``{"session_id": null}`` means "no eligible session right now"; errors fold
    into that shape (and log) so callers never special-case error envelopes.
    Honors ``params.profile`` (mirrors ``session.resume``).
    """
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"session_id": None})
        try:
            # Generous over-fetch so heavy sub-agent users (many ``tool`` rows)
            # don't get a false "no eligible session".
            rows = db.list_sessions_rich(
                source=None, limit=200, order_by_last_active=True, compact_rows=True
            )
            for row in rows:
                if _denied_source(row):
                    continue
                return _ok(
                    rid,
                    {
                        "session_id": row.get("id"),
                        "title": row.get("title") or "",
                        "started_at": row.get("started_at") or 0,
                        "source": row.get("source") or "",
                    },
                )
            return _ok(rid, {"session_id": None})
        except Exception:
            logger.exception("session.most_recent failed")
            return _ok(rid, {"session_id": None})


@method("project.facts")
def _(rid, params: dict) -> dict:
    """Project facts for a cwd (manifests, package manager, verify commands, context files).

    Same detection the coding-context posture bakes into the system prompt,
    exposed so UIs consume it instead of re-sniffing. ``{"facts": null}`` = not a code workspace.
    """
    try:
        from agent.coding_context import project_facts_for

        return _ok(rid, {"facts": project_facts_for(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return _ok(rid, {"facts": None})


@method("verification.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Best known coding verification evidence for a cwd/session.

    Read-only consumer of the core ledger: never runs checks, never upgrades
    targeted evidence into a repository-wide guarantee.
    """
    try:
        from agent.verification_evidence import verification_status

        return _ok(
            rid,
            {
                "verification": verification_status(
                    session_id=params.get("session_id") or params.get("session_key"),
                    cwd=params.get("cwd"),
                )
            },
        )
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


# ── session.resume ───────────────────────────────────────────────────


@dataclass
class _Resume:
    """Per-call state for ``session.resume`` shared by the path helpers below.

    ``owns_db`` tracks the DEDICATED profile-scoped handle: it is ours to close
    (the handler's ``finally``) until a path hands it to the hydration worker or
    the agent (``_init_session``), which flips it False.
    """

    rid: object
    params: dict
    target: str
    cols: int
    profile: str | None
    profile_home: object
    lazy: bool
    defer_history: bool
    omit_messages: bool
    eager_build: bool
    db: object = None
    owns_db: bool = False
    found: dict | None = None
    profile_resume_cwd: str = ""

    def record(self, source: str, history: list, **extra) -> dict:
        """``_deferred_session_record`` with this resume's common fields; the active-session
        lease is always claimed lazily on the first turn (_ensure_active_session_slot)."""
        return _deferred_session_record(
            self.target,
            cols=self.cols,
            cwd=self.profile_resume_cwd or _default_session_cwd(),
            history=history,
            lease=None,
            source=source,
            close_on_disconnect=is_truthy_value(self.params.get("close_on_disconnect", False)),
            profile_home=self.profile_home,
            explicit_cwd=bool(self.profile_resume_cwd),
            **extra,
        )

    def resume_failed(self, exc) -> dict:
        return _err(self.rid, 5000, f"resume failed: {exc}")


def _find_live_unpersisted(needle: str, home) -> str:
    """Runtime sid of a live, not-yet-persisted session matched by stored key or pending title."""
    want_home = str(home) if home is not None else None
    for live_sid, record in list(_sessions.items()):
        if not isinstance(record, dict):
            continue
        if (record.get("profile_home") or None) != want_home:
            continue
        if str(record.get("session_key") or "") == needle or (record.get("pending_title") or "") == needle:
            return live_sid
    return ""


def _resume_live_unpersisted(ctx: _Resume, live_sid: str, live: dict) -> dict:
    """Reattach a LIVE lazy session (no state.db row yet — every fresh Bot Chat).

    session.create persists no row until the first prompt, so a resume by stored
    key / pending title for a never-messaged session lands here; a hard 404 killed
    messaging for exactly the bots that had never spoken. A WS drop may have
    sentinel-parked the record, so rebind the transport and cancel the armed
    orphan-reap Timer or it fires against a client that is attached right now.
    """
    if ctx.owns_db:
        with contextlib.suppress(Exception):
            from hermes_state import release_or_close

            release_or_close(ctx.db)
    live["last_active"] = time.time()
    transport = current_transport()
    if transport is not None:
        with live.setdefault("history_lock", threading.Lock()):
            live["transport"] = transport
            live.setdefault("viewers", {})[transport] = time.time()
    _cancel_ws_orphan_reap(live_sid)
    history = live.get("history") or []
    return _ok(
        ctx.rid,
        _attach_todo_state(
            {
                "session_id": live_sid,
                "stored_session_id": str(live.get("session_key") or ""),
                "message_count": len(history),
                "messages": [] if ctx.omit_messages else _history_to_messages(history),
                "info": {"model": _resolve_model(), "lazy": True, "profile_name": ctx.profile or ""},
            },
            live,
        ),
    )


def _resume_adopt_stranded(ctx: _Resume) -> None:
    """Adopt a lineage stranded in the DEFAULT store into this profile's db (profile-scoped only).

    Before session RPCs routed by their TARGET session, a profile bot's turns ran
    on the focused tile's backend, so its canonical session accumulated in the
    default profile's state.db; without adoption that chat 4001s forever.
    Exact-id match ONLY: title lookup has no archived filter and bot titles
    collide by design, so a title-matched donor could retire an UNRELATED
    conversation. Never re-adopt an already-retired donor (two "canonical" clones).
    """
    try:
        default_db = _get_db()
        donor_row = default_db.get_session(ctx.target) if default_db is not None else None
        if donor_row and donor_row.get("archived"):
            donor_row = None
        if donor_row:
            adoption = ctx.db.adopt_session_lineage_from(default_db, donor_row["id"])
            if adoption.get("adopted"):
                logger.info(
                    "adopted stranded session %s (lineage of %s segment(s)) from default store into profile %s",
                    donor_row["id"],
                    len(adoption.get("imported_ids") or []) + len(adoption.get("skipped_ids") or []),
                    ctx.profile or "?",
                )
                ctx.found = ctx.db.get_session(donor_row["id"])
                if ctx.found:
                    ctx.target = ctx.found["id"]
    except Exception:
        logger.exception("stranded-session adoption failed for %s", ctx.target)


def _resume_locate(ctx: _Resume) -> dict | None:
    """Resolve ``ctx.target`` to a stored row (``ctx.found``); a dict is an early response."""
    db = ctx.db
    ctx.found = db.get_session(ctx.target)
    if ctx.found:
        return None
    ctx.found = db.get_session_by_title(ctx.target)
    if ctx.found:
        ctx.target = ctx.found["id"]
        return None
    if ctx.lazy and _child_run_active(ctx.target):
        # Race: a watch window opened on a freshly-spawned subagent. The child
        # relays `subagent.start` BEFORE its first run_conversation() flushes the
        # DB row, so the row is momentarily missing (reliably on WSL2). The child
        # is provably live, so proceed lazily with empty history — the live
        # mirror streams the turn and the row exists by upgrade time.
        ctx.found = {}
        return None
    live_sid = _find_live_unpersisted(ctx.target, ctx.profile_home)
    live = _sessions.get(live_sid) if live_sid else None
    if live is not None:
        return _resume_live_unpersisted(ctx, live_sid, live)
    if ctx.owns_db:
        _resume_adopt_stranded(ctx)
    if not ctx.found:
        return _err(ctx.rid, 4007, "session not found")
    return None


def _resume_follow_tip(ctx: _Resume) -> None:
    """Rebind a rotated-out parent id to its compression-continuation tip.

    Auto-compression ends the session and forks a child; resuming the original
    id would reload the parent transcript and lose the post-compression reply.
    Resolving here also re-anchors the live fast path so a rotated live session
    is reused (by its new key) instead of rebuilding a duplicate on the stale
    parent. Skipped for lazy watch windows (they attach to the exact child).
    Bot Chat stays on a proven compression edge so an unmarked side chat cannot
    steal the open; other sessions keep the legacy unmarked-child walker.
    """
    if not ctx.found or ctx.lazy:
        return
    try:
        from tools.bot_mode_probe import BOT_CHAT_TITLE

        if (ctx.found.get("title") or "").strip() == BOT_CHAT_TITLE:
            tip = ctx.db.get_compression_tip(ctx.target) or ctx.target
        else:
            tip = ctx.db.resolve_resume_session_id(ctx.target)
    except Exception:
        tip = ctx.target
    if tip and tip != ctx.target:
        ctx.target = tip
        ctx.found = ctx.db.get_session(tip) or ctx.found


def _resume_guard(ctx: _Resume) -> dict | None:
    """Refuse a runaway transcript before any history read (sessions.max_resume_messages).

    Only the non-deferred, non-omitted resume reads the whole lineage; the
    deferred Desktop resume, omit_messages resume and lazy watch load the TIP
    segment only, so they are guarded tip-only (a full-lineage count rejected
    exactly the well-compressed conversations compaction produces). Metadata
    fallback keeps lightweight adaptor DBs compatible. Fails OPEN on guard errors
    — only a genuine over-limit blocks.
    """
    from hermes_state import SessionResumeTooLargeError, resolved_max_resume_messages

    guard_tip_only = ctx.lazy or ctx.omit_messages or (ctx.defer_history and not ctx.eager_build)
    safety_check = getattr(ctx.db, "assert_resume_safe", None)
    try:
        if callable(safety_check):
            if guard_tip_only:
                safety_check(ctx.target, tip_only=True)
            else:
                safety_check(ctx.target)
        else:
            resume_limit = resolved_max_resume_messages()
            stored_message_count = int(ctx.found.get("message_count") or 0)
            if resume_limit and stored_message_count > resume_limit:
                raise SessionResumeTooLargeError(stored_message_count, resume_limit)
    except SessionResumeTooLargeError as exc:
        return _err(ctx.rid, 4130, str(exc))
    except Exception as exc:
        logger.warning(
            "resume safety check failed for %s (proceeding without guard): %s", ctx.target, exc
        )
    return None


def _resume_reuse_live(ctx: _Resume, sid: str, session: dict) -> dict:
    """Reattach an already-live session under the resume lock.

    Holding the lock across the client-gone check, transport rebind and reap
    cancel makes grace expiry atomic across every reuse path (slow-path claim
    races discover a winner after releasing their own lock).
    """
    with _session_resume_lock:
        if _sessions.get(sid) is not session:
            return _err(ctx.rid, 4007, "session no longer live; retry resume")
        if session.get("_client_gone_interrupt_requested"):
            return _err(ctx.rid, 4009, "session disconnect interrupt settling")
        # Cancel unconditionally (the payload's rebind only cancels when a
        # transport is passed) so the fast path can never race the reap Timer.
        _cancel_ws_orphan_reap(sid)
        payload = _live_session_payload(
            sid,
            session,
            cols=ctx.cols,
            touch=True,
            transport=current_transport() or _stdio_transport,
            omit_messages=ctx.omit_messages,
        )
        payload["resumed"] = ctx.target
        if ctx.defer_history:
            payload["messages"] = []
            payload["message_count"] = int(session.get("resume_message_count") or payload["message_count"])
            payload["hydrating"] = bool(session.get("resume_hydrating"))
        # A lazy watch session never owns a run loop (running always False) —
        # overlay the child-run registry so a reconnecting window stays busy.
        if session.get("agent") is None and _child_run_active(ctx.target):
            payload["running"] = True
            payload["status"] = "streaming"
        return _ok(ctx.rid, payload)


def _resume_info(ctx: _Resume, cwd: str, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    model_override = overrides.get("model_override") or {}
    return _lazy_resume_info(
        cwd,
        model=model_override.get("model") or "",
        provider=overrides.get("provider_override") or "",
        profile=ctx.profile,
    )


def _resume_response(
    ctx: _Resume,
    sid: str,
    record: dict,
    *,
    messages: list,
    message_count: int,
    info: dict,
    running: bool,
    status: str,
    hydrating: bool | None = None,
    started_at=None,
    auto_continue=None,
) -> dict:
    payload = {
        "session_id": sid,
        "resumed": ctx.target,
        "message_count": message_count,
        "messages": messages,
    }
    if hydrating is None:
        payload["messages_omitted"] = ctx.omit_messages
    else:
        payload["hydrating"] = hydrating
    payload.update(
        {
            "info": info,
            "inflight": None,
            "running": running,
            "session_key": ctx.target,
            "started_at": record["created_at"] if started_at is None else started_at,
            "status": status,
        }
    )
    if auto_continue is not None:
        payload["auto_continue"] = auto_continue
    return _ok(ctx.rid, _attach_todo_state(payload, record))


def _resume_read_history(ctx: _Resume):
    """One lineage SELECT feeds both projections: model-fed copy alternation-repaired
    for live replay, display copy verbatim (inspection/export shows what is stored)."""
    ctx.db.reopen_session(ctx.target)
    if ctx.omit_messages:
        raw = ctx.db.get_messages_as_conversation(ctx.target, repair_alternation=True, include_row_ids=True)
        return raw, []
    return ctx.db.get_resume_conversations(ctx.target)


def _resume_lazy(ctx: _Resume) -> dict:
    """Lazy/watch resume: register the live session WITHOUT building an agent.

    Used by the desktop's subagent windows — the child runs inside the parent's
    turn, so the window only needs stored history plus a transport for the
    child-mirror's live events. A later prompt.submit upgrades it via
    _start_agent_build (resume_session_id keeps it on the stored conversation).
    """
    sid, source = _new_runtime_ids(ctx.params)
    try:
        ctx.db.reopen_session(ctx.target)
        # The child's OWN conversation only (include_ancestors would prepend the
        # parent's transcript). repair_alternation heals a durable ``user;user``
        # once here instead of re-firing the pre-request repair every turn.
        history = ctx.db.get_messages_as_conversation(
            ctx.target, repair_alternation=True, include_row_ids=True
        )
    except Exception as e:
        return ctx.resume_failed(e)
    record = ctx.record(source, history, lazy=True, todo_state=_todo_state_from_history(history))
    if (live := _claim_or_reuse_live(sid, ctx.target, record, None)) is not None:
        return _resume_reuse_live(ctx, *live)
    # A delegated child mid-run emits no session events of its own — report
    # liveness from the relay registry so the window shows a busy turn.
    child_running = _child_run_active(ctx.target)
    # Display uses the VERBATIM projection (child-only, matching the repaired
    # read) so model-invisible rows survive in the watch window as on the eager
    # + REST paths; the repaired ``history`` still feeds live replay.
    try:
        display_history = ctx.db.get_messages_as_conversation(
            ctx.target, repair_alternation=False, include_row_ids=True
        )
    except Exception:
        logger.debug("child-watch display projection read failed", exc_info=True)
        display_history = history
    messages = [] if ctx.omit_messages else _history_to_messages(display_history)
    return _resume_response(
        ctx,
        sid,
        record,
        messages=messages,
        message_count=len(display_history) if ctx.omit_messages else len(messages),
        info=_lazy_resume_info(record["cwd"], profile=ctx.profile),
        running=child_running,
        status="streaming" if child_running else "idle",
    )


def _resume_deferred(ctx: _Resume) -> dict:
    """Bounded acknowledgement; the transcript hydrates in the background and the
    display copy pages over REST. defer_history SUPERSEDES omit_messages: the
    response never carries a transcript and the ONE history read happens in the
    hydration worker, so it is never loaded twice for one resume."""
    sid, source = _new_runtime_ids(ctx.params)
    _enable_gateway_prompts()
    overrides = _stored_session_runtime_overrides(ctx.found) or {}
    record = ctx.record(
        source,
        [],
        model_override=overrides.get("model_override"),
        resume_runtime_overrides=overrides or None,
    )
    record["resume_history_ready"] = threading.Event()
    record["resume_hydrating"] = True
    record["resume_message_count"] = int(ctx.found.get("message_count") or 0)
    if (live := _claim_or_reuse_live(sid, ctx.target, record, None)) is not None:
        return _resume_reuse_live(ctx, *live)

    _schedule_resume_hydration(sid, ctx.target, ctx.db, close_db=ctx.owns_db)
    # The hydration worker now owns a profile-scoped handle and closes it after
    # the read. The shared launch DB is process-owned.
    ctx.owns_db = False
    _schedule_session_cap_enforcement()
    return _resume_response(
        ctx,
        sid,
        record,
        messages=[],
        message_count=record["resume_message_count"],
        info=_resume_info(ctx, record["cwd"], overrides),
        running=False,
        status="resuming",
        hydrating=True,
    )


def _resume_cold(ctx: _Resume) -> dict:
    """Cold resume default: register the session and read its transcript, but build
    the agent OFF the response path — _make_agent can block for seconds and every
    resume caller awaits this RPC before it paints. Pre-warms on a short timer
    (session.create's deferred-build contract); _sess() builds on demand if the
    first prompt beats it. Unlike the lazy branch this restores the full ancestor
    history and persisted runtime identity, and is a real (upgradable) session."""
    sid, source = _new_runtime_ids(ctx.params)
    _enable_gateway_prompts()
    try:
        raw_history, display_history = _resume_read_history(ctx)
    except Exception as e:
        return ctx.resume_failed(e)
    # Display keeps the full transcript; the model-fed history drops a dangling
    # tool-call tail so a session killed mid-loop does not replay it forever.
    prefix = [] if ctx.omit_messages else ctx.db.get_ancestor_display_prefix(ctx.target)
    history = sanitize_replay_history(raw_history)
    # Restore model/provider/reasoning/tier so the deferred build matches the
    # eager path — without them the build drops the provider.
    overrides = _stored_session_runtime_overrides(ctx.found) or {}
    record = ctx.record(
        source,
        history,
        display_history_prefix=prefix,
        model_override=overrides.get("model_override"),
        resume_runtime_overrides=overrides or None,
        todo_state=_todo_state_from_history(history),
    )
    if (live := _claim_or_reuse_live(sid, ctx.target, record, None)) is not None:
        return _resume_reuse_live(ctx, *live)

    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap
    auto_continue = _maybe_schedule_auto_continue(sid, record, ctx.target)

    messages = [] if ctx.omit_messages else _history_to_messages(display_history)
    return _resume_response(
        ctx,
        sid,
        record,
        messages=messages,
        message_count=len(raw_history) if ctx.omit_messages else len(messages),
        info=_resume_info(ctx, record["cwd"], overrides),
        running=False,
        status="idle",
        auto_continue=auto_continue,
    )


def _resume_eager(ctx: _Resume) -> dict:
    """Synchronous build (``eager_build: true``, e.g. build-race tests).

    The agent is built OUTSIDE _session_resume_lock (it can block for seconds and
    would stall session.close on the dispatch thread), then double-checked: if a
    concurrent resume won meanwhile, discard our agent and reuse theirs.
    """
    sid, source = _new_runtime_ids(ctx.params)
    _enable_gateway_prompts()
    with _profile_build_scope(ctx.profile_home):
        try:
            raw_history, display_history = _resume_read_history(ctx)
            display_history_prefix = [] if ctx.omit_messages else ctx.db.get_ancestor_display_prefix(ctx.target)
            history = sanitize_replay_history(raw_history)
            messages = [] if ctx.omit_messages else _history_to_messages(display_history)
            tokens = _set_session_context(ctx.target)
            try:
                # The profile's db so turns persist to the right state.db; runtime
                # identity from the stored row so switching chats does not inherit
                # whatever global model another chat last selected.
                stored_runtime_overrides = _stored_session_runtime_overrides(ctx.found)
                agent = _make_agent(
                    sid,
                    ctx.target,
                    session_id=ctx.target,
                    session_db=ctx.db,
                    platform_override=source,
                    context_cwd_is_launch_artifact=(
                        source in _LAUNCH_CWD_NOT_A_WORKSPACE and not ctx.profile_resume_cwd
                    ),
                    **stored_runtime_overrides,
                )
            finally:
                _clear_session_context(tokens)
        except Exception as e:
            return ctx.resume_failed(e)

    with _session_resume_lock:
        live = _find_live_session_by_key(ctx.target, ctx.profile_home)
        if live is not None:
            try:
                if hasattr(agent, "close"):
                    agent.close()
            except Exception:
                pass
            return _resume_reuse_live(ctx, *live)
        try:
            with _profile_build_scope(ctx.profile_home):
                _init_session(
                    sid,
                    ctx.target,
                    agent,
                    history,
                    cols=ctx.cols,
                    cwd=ctx.profile_resume_cwd,
                    session_db=ctx.db,
                    source=source,
                    explicit_cwd=bool(ctx.profile_resume_cwd),
                )
                # Ownership TRANSFER: the registered agent holds this handle for
                # its life and AIAgent.close() releases it at teardown
                # (_init_session never closes a caller-supplied db). The drop is
                # UNCONDITIONAL — past this line the session is registered against
                # the handle, so the finally must not close it even if the transfer
                # was refused (a leak is survivable; "Cannot operate on a closed
                # database" on every later turn is not). The transfer is gated on
                # owns_db: the SHARED launch handle must never move onto one
                # session, or session.close tears down the process-wide database.
                if ctx.owns_db:
                    _transfer_db_to_agent(agent, ctx.db)
                ctx.owns_db = False
            if sid in _sessions:
                if stored_runtime_overrides.get("model_override") is not None:
                    _sessions[sid]["model_override"] = stored_runtime_overrides["model_override"]
                _sessions[sid]["display_history_prefix"] = display_history_prefix
                # Each turn re-binds HERMES_HOME (mid-turn home reads — memory,
                # skills — must resolve to the resumed profile too).
                if ctx.profile_home is not None:
                    _sessions[sid]["profile_home"] = str(ctx.profile_home)
                _sessions[sid]["active_session_lease"] = None  # claimed lazily on the first turn
        except Exception as e:
            # _init_session registers _sessions[sid] BEFORE its first read through
            # this handle ("database is locked" is the realistic trigger). Left in
            # place, the live fast path would serve that dead session forever;
            # owns_db still True means the registration is ours to undo.
            if ctx.owns_db:
                with _sessions_lock:
                    _sessions.pop(sid, None)
            return ctx.resume_failed(e)
        session = _sessions.get(sid) or {}
    auto_continue = _maybe_schedule_auto_continue(sid, session, ctx.target) if session else None
    return _resume_response(
        ctx,
        sid,
        session,
        messages=messages,
        message_count=len(raw_history) if ctx.omit_messages else len(messages),
        info=_session_info(agent, session),
        running=False,
        status="idle",
        started_at=float(session.get("created_at") or time.time()),
        auto_continue=auto_continue,
    )


@method("session.resume")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80
    # ``profile`` (app-global remote mode): resume from another local profile's state.db.
    profile = (params.get("profile") or "").strip() or None
    ctx = _Resume(
        rid=rid,
        params=params,
        target=target,
        cols=cols,
        profile=profile,
        profile_home=_profile_home(profile),
        lazy=is_truthy_value(params.get("lazy", False)),
        defer_history=is_truthy_value(params.get("defer_history", False)),
        # Desktop hydrates transcripts over REST in parallel; suppress the
        # duplicate WebSocket copy only when explicitly asked.
        omit_messages=is_truthy_value(params.get("omit_messages", False)),
        eager_build=is_truthy_value(params.get("eager_build", False)),
    )
    # Profile scope opens a DEDICATED handle we own until the agent takes it;
    # otherwise the shared launch db, which outlives the RPC and is never closed here.
    if ctx.profile_home is not None:
        from hermes_state import get_shared_session_db

        ctx.db = get_shared_session_db(ctx.profile_home / "state.db")
        ctx.owns_db = True
    else:
        ctx.db = _get_db()
    try:
        if ctx.db is None:
            return _db_unavailable_error(rid, code=5000)
        if (resp := _resume_locate(ctx)) is not None:
            return resp
        _resume_follow_tip(ctx)
        if (resp := _resume_guard(ctx)) is not None:
            return resp
        ctx.profile_resume_cwd = str(ctx.found.get("cwd") or "").strip() or _profile_configured_cwd(
            ctx.profile_home
        )
        # Fast path: reuse a session already live IN THIS PROFILE — never another
        # profile's runtime of the same stored id.
        with _session_resume_lock:
            live = _find_live_session_by_key(ctx.target, ctx.profile_home)
        if live is not None:
            return _resume_reuse_live(ctx, *live)
        if ctx.lazy:
            return _resume_lazy(ctx)
        if ctx.defer_history and not ctx.eager_build:
            return _resume_deferred(ctx)
        if not ctx.eager_build:
            return _resume_cold(ctx)
        return _resume_eager(ctx)
    finally:
        # Every return that does not transfer the handle abandons it. Refcounting
        # alone does not release the sqlite fds: SessionDB pins ITSELF once its
        # background token writer starts (atexit.register in hermes_state), which
        # only close() unregisters.
        if ctx.owns_db and ctx.db is not None:
            with contextlib.suppress(Exception):
                ctx.db.close()


# ── cwd / workspace / live-session bookkeeping ───────────────────────


@method("session.cwd.set")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    info = _cwd_info(session, cwd)
    _emit("session.info", params.get("session_id", ""), info)
    return _ok(rid, info)


@method("session.workspace.move")
def _(rid, params: dict) -> dict:
    """Re-home a STORED session's workspace (by ``session_key``) into another folder/project.

    Unlike ``session.cwd.set`` no live agent is required. The git branch/root
    columns are REPLACED, not enriched — a stale ``git_repo_root`` would keep the
    session grouped under the project it left. A live agent bound to the row
    follows too, even mid-turn (refusing made the UI claim success while state.db
    kept the old cwd); in-flight tool calls keep their cwd, the NEXT one moves.
    """
    target = str(params.get("session_key") or "").strip()
    if not target:
        return _err(rid, 4007, "session_key required")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    from hermes_constants import translate_cwd_for_wsl_backend

    resolved = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(raw)))
    if not os.path.isdir(resolved):
        return _err(rid, 4017, f"working directory does not exist: {raw}")

    # Snapshot under the lock — concurrent RPCs mutate _sessions.
    live = None
    live_sid = ""
    with _sessions_lock:
        for sid, sess in list(_sessions.items()):
            if sess.get("session_key") == target:
                live, live_sid = sess, sid
                break

    branch = _git_branch_for_cwd(resolved)
    root = _git_common_repo_root_for_cwd(resolved)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        # A brand-new draft has no row yet; the live re-home still applies and
        # the row inherits the cwd when first written.
        row_exists = bool(db.get_session(target))
        if not row_exists and live is None:
            return _err(rid, 4007, "session not found")
        if row_exists:
            try:
                db.update_session_cwd(target, resolved, branch, root, replace_git_meta=True)
            except Exception as e:
                return _err(rid, 5007, f"move failed: {e}")

    if live is not None:
        try:
            _set_session_cwd(live, resolved)
        except ValueError as e:
            return _err(rid, 4017, str(e))
        _emit("session.info", live_sid, _cwd_info(live, resolved, branch=branch))

    return _ok(rid, {"cwd": resolved, "branch": branch, "git_repo_root": root})


@method("session.active_list")
def _(rid, params: dict) -> dict:
    """Live TUI sessions in this process (not a DB browser): only sessions with
    in-memory agents/workers the current TUI can switch to without closing siblings."""
    current = str(params.get("current_session_id") or "")
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")

    # ``_finalized`` sessions are dead (teardown begun) but may linger in
    # ``_sessions`` until the reaper pops them; counting them inflated the footer
    # forever. Do NOT filter on the WS-detached sentinel: a detached session is
    # still attachable via reconnect until grace-reap finalizes it, and a
    # standalone ``hermes --tui`` rides the real stdio transport. Keep insertion
    # order — the focused session must not jump to the top.
    rows = [
        _session_live_item(sid, session, current)
        for sid, session in snapshot
        if not session.get("_finalized")
    ]
    return _ok(rid, {"sessions": rows})


@method("session.activate")
def _(rid, params: dict) -> dict:
    """Attach the frontend to an already-live TUI session (does not close the
    previously focused one — just enough state for Ink to redraw)."""
    sid = str(params.get("session_id") or "")
    session, err = _sess_nowait({"session_id": sid}, rid)
    if err:
        return err
    return _ok(
        rid,
        _live_session_payload(
            sid,
            session,
            touch=True,
            transport=current_transport() or _stdio_transport,
            omit_messages=is_truthy_value(params.get("omit_messages", False)),
        ),
    )


@method("session.delete")
def _(rid, params: dict) -> dict:
    """Delete a stored session and its transcript files (TUI resume picker ``d``).

    Refuses sessions live in this process — removing rows under a live agent
    corrupts message ordering and trips FK constraints on the next flush.
    Honors ``params.profile`` (mirrors ``session.resume``).
    """
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    # Snapshot via list(): _sessions is mutated by concurrent RPCs. If even the
    # snapshot raises, fail CLOSED (refuse the delete).
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")
    active = {s.get("session_key") for s in snapshot if s.get("session_key")}
    if target in active:
        return _err(rid, 4023, "cannot delete an active session")
    profile_home = _profile_home((params.get("profile") or "").strip() or None)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5036)
        sessions_dir = (Path(profile_home) if profile_home is not None else get_hermes_home()) / "sessions"
        try:
            deleted = db.delete_session(target, sessions_dir=sessions_dir)
        except Exception as e:
            return _err(rid, 5036, f"delete failed: {e}")
        if not deleted:
            return _err(rid, 4007, "session not found")
        return _ok(rid, {"deleted": target})


def _title_read(rid, params: dict, session: dict, db) -> dict:
    """``session.title`` without ``title``: read it, applying a queued pending_title if possible."""
    key = session["session_key"]
    fallback = session.get("pending_title") or ""
    try:
        resolved_title = db.get_session_title(key) or ""
        if fallback:
            if db.set_session_title(key, fallback):
                session["pending_title"] = None
                resolved_title = fallback
            else:
                existing_row = db.get_session(key)
                existing_title = ((existing_row or {}).get("title") or "").strip()
                if existing_title == fallback:
                    session["pending_title"] = None
                    resolved_title = fallback
                elif not resolved_title:
                    resolved_title = fallback
        elif resolved_title:
            session["pending_title"] = None
    except Exception:
        resolved_title = fallback
    _emit_session_info_for_session(params.get("session_id", ""), session)
    return _ok(rid, {"title": resolved_title, "session_key": key})


@method("session.title")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        if "title" not in params:
            return _title_read(rid, params, session, db)
        key = session["session_key"]
        title = (params.get("title", "") or "").strip()
        if not title:
            return _err(rid, 4021, "title required")
        sid = params.get("session_id", "")

        def _done(pending: bool, value: str) -> dict:
            session["pending_title"] = value if pending else None
            _emit_session_info_for_session(sid, session)
            return _ok(rid, {"pending": pending, "title": value})

        try:
            if db.set_session_title(key, title):
                return _done(False, title)
            # rowcount == 0 can mean "same value" as well as "missing row".
            existing_row = db.get_session(key)
            if existing_row:
                return _done(False, existing_row.get("title") or title)
            # No row yet (deferred to the first prompt). An explicit /title is
            # clear intent, so persist the row NOW (mirrors the messaging
            # gateway's _handle_title_command) instead of queuing pending_title
            # and hoping the post-turn apply block lands under this key. The
            # min-messages sidebar filter keeps a titled 0-message row hidden.
            _ensure_session_db_row(session)
            with _session_db(session) as scoped_db:
                if scoped_db is not None and scoped_db.set_session_title(key, title):
                    return _done(False, title)
            # Row creation didn't take (DB unavailable / concurrent writer) —
            # queue so the post-turn apply block can still recover.
            return _done(True, title)
        except ValueError as e:
            return _err(rid, 4022, str(e))
        except Exception as e:
            return _err(rid, 5007, str(e))


@method("session.set_hidden")
def _(rid, params: dict) -> dict:
    """Set/clear the ``hidden`` flag on a session (and its compression lineage).

    A hidden session is dropped from the default Sessions list but stays fully
    resumable by the surface that owns it. Two-tier resolution: a LIVE runtime
    id first (covers the not-yet-persisted draft via ``pending_hidden``), then a
    durable stored id/key in the target profile's state.db — plugins reconciling
    sessions they own hold stored ids for chats that aren't live right now.
    """
    hidden = is_truthy_value(params.get("hidden", True))
    session, err = _sess_nowait(params, rid)
    if session is not None:
        with _session_db(session) as db:
            if db is None:
                return _db_unavailable_error(rid, code=5007)
            key = session["session_key"]
            try:
                if not db.set_session_hidden(key, hidden):
                    # No row yet: remember the intent so _ensure_session_db_row
                    # is born hidden (mirrors the pending_title deferral).
                    session["pending_hidden"] = hidden
                return _ok(rid, {"hidden": hidden, "session_key": key})
            except Exception as e:
                return _err(rid, 5007, str(e))
    # ``resolve_session_id`` follows key/title aliases like the REST pin/archive path.
    target = str(params.get("session_id") or "").strip()
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            resolved = db.resolve_session_id(target) if hasattr(db, "resolve_session_id") else target
            if not resolved:
                return err
            db.set_session_hidden(resolved, hidden)
            return _ok(rid, {"hidden": hidden, "session_key": resolved})
        except Exception as e:
            return _err(rid, 5007, str(e))


@method("message.react")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    """Set or clear one author's emoji reaction on a persisted message.

    iOS Tapback semantics enforced in the DB layer: one reaction per author per
    message, re-sending the same emoji retracts it, ``emoji: null`` clears.
    ``row_id`` is the durable ``messages.id``; a live message that hasn't
    round-tripped through a resume can instead name ``newest_role``.
    """
    newest_role = str(params.get("newest_role") or "").strip()
    row_id = params.get("row_id")
    if row_id is None and newest_role not in {"user", "assistant"}:
        return _err(rid, 4023, "row_id or newest_role required")

    emoji = params.get("emoji")
    if emoji is not None:
        emoji = str(emoji).strip()
        if not emoji:
            return _err(rid, 4024, "emoji must be a non-empty string or null")

    author = str(params.get("author") or "user").strip()
    if author not in {"user", "agent"}:
        return _err(rid, 4025, "author must be 'user' or 'agent'")

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if row_id is None:
                row_id = db.latest_message_row_id(session["session_key"], role=newest_role)
                if row_id is None:
                    return _err(rid, 4040, "no message to react to yet")
            reactions = db.set_message_reaction(session["session_key"], int(row_id), emoji, author=author)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if reactions is None:
        return _err(rid, 4040, "message not found in this session")
    return _ok(rid, {"row_id": int(row_id), "reactions": reactions})


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Single stateless LLM request outside any conversation (e.g. a commit message).

    Accepts a named ``template`` + ``variables`` or ``instructions``/``input``.
    A live ``session_id`` lends its agent's model; otherwise the auxiliary
    ``task`` backend. Never mutates session history (prompt cache untouched).
    """
    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    task = (params.get("task") or "title_generation").strip() or "title_generation"

    try:
        max_tokens = int(params.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = None

    if not template and not str(instructions).strip() and not str(user_input).strip():
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    session = _sessions.get(params.get("session_id") or "")
    main_runtime = _main_runtime_from_agent(session.get("agent")) if session else None

    try:
        from agent.oneshot import run_oneshot

        text = run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as e:
        return _err(rid, 4031, str(e))
    except ValueError as e:
        return _err(rid, 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")

    return _ok(rid, {"text": text})


# ── handoff ──────────────────────────────────────────────────────────


@method("handoff.request")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    """Queue a handoff of this session to a messaging platform (desktop parity with /handoff).

    Only writes ``handoff_state='pending'`` on the persisted row; the separate
    ``hermes gateway`` process's ``_handoff_watcher`` claims it, re-binds the
    session to the platform's home channel and forges a synthetic turn. The
    desktop then polls ``handoff.state``.
    """
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — wait for the current turn to finish, then retry the handoff"
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate against the live gateway config up front: an unconfigured platform
    # or missing home channel would leave the handoff pending forever.
    try:
        from gateway.config import Platform, load_gateway_config
    except Exception as e:  # pragma: no cover — gateway pkg always ships
        return _err(rid, 5021, f"could not load gateway config: {e}")
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        with _session_profile_runtime_scope(session):
            gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return _err(rid, 4025, f"platform '{platform_name}' is not configured/enabled in the gateway")
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return _err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    # The watcher transfers a persisted row, so make sure one exists even for a
    # brand-new empty chat (mirrors the CLI's set_session_title stub).
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(
            rid, 4027, "session is already in flight for handoff — wait for it to settle, then retry"
        )
    return _ok(
        rid, {"queued": True, "session_key": key, "platform": platform_name, "home_name": home.name}
    )


@method("handoff.state")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    """Poll ``{state, platform, error}``; ``state`` is pending|running|completed|failed or empty."""
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return _ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Mark a not-yet-claimed handoff failed so the user can retry (desktop poll timeout).

    Only PENDING rows change (compare-and-swap in ``fail_handoff``): once the
    watcher has claimed the row (``running``) it owns the terminal state — failing
    it from the waiter races the dispatch and later flips failed→completed after
    the user was told it failed. A ``running`` row yields
    ``{"failed": False, "state": "running"}`` ("still transferring").
    """
    # Undecorated on purpose: tests rebind this handler's __code__ directly.
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            failed = db.fail_handoff(key, reason, only_states=("pending",))
        except TypeError:
            # Older SessionDB without only_states: fail only when still pending.
            record = db.get_handoff_state(key) or {}
            failed = (record.get("state") or "") == "pending"
            if failed:
                db.fail_handoff(key, reason)
        if failed:
            return _ok(rid, {"failed": True, "state": "failed"})
        record = db.get_handoff_state(key) or {}

    return _ok(rid, {"failed": False, "state": record.get("state") or ""})


# ── usage ────────────────────────────────────────────────────────────


@method("session.usage")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    agent = session.get("agent")
    usage: dict = _session_usage_snapshot(session)
    if agent is None and not usage:
        usage = {"calls": 0, "input": 0, "output": 0, "total": 0}
    # Nous credits are agent-independent (portal fetch) so they show even with
    # zero API calls. Fail-open: absent when not logged in / portal hiccup.
    try:
        from agent.account_usage import nous_credits_lines

        credits = nous_credits_lines()
        if credits:
            usage["credits_lines"] = credits
    except Exception:
        pass
    return _ok(rid, usage)


@method("session.context_breakdown")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    agent = session.get("agent")
    if agent is None:
        usage = _session_usage_snapshot(session) or _get_usage(None)
        return _ok(
            rid,
            {
                "categories": [],
                "context_max": usage.get("context_max", 0) or 0,
                "context_percent": usage.get("context_percent", 0) or 0,
                "context_used": usage.get("context_used", 0) or 0,
                "estimated_total": usage.get("context_used", 0) or usage.get("total", 0) or 0,
                "model": _metadata_mirror(session).get("model", ""),
            },
        )
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        from agent.context_breakdown import compute_session_context_breakdown

        payload = compute_session_context_breakdown(agent, history)
    except Exception as exc:
        return _err(rid, 5000, f"Could not compute context breakdown: {exc}")
    return _ok(rid, payload)


# ── pet ──────────────────────────────────────────────────────────────

_PET_OFF = {"enabled": False}


@method("pet.info")
@_profile_scoped
@_pet_guard("pet.info", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Active petdex pet for sprite-rendering surfaces (desktop canvas + TUI half-block).

    Carries the spritesheet (base64) plus frame geometry + state-row taxonomy so
    the renderer is a thin consumer. Agent-independent; fail-open ``enabled=False``.
    """
    enabled, pet, scale = _pet_active_selection()
    if not enabled or pet is None or not pet.exists:
        return _ok(rid, {"enabled": False})
    payload = {"enabled": True, **_pet_sprite_payload(pet, scale=scale)}
    # Send-once for the multi-MB sheet: a caller holding revision R gets
    # metadata only (spritesheetUnchanged) when the sheet hasn't changed.
    known_revision = str(params.get("knownRevision", "") or "")
    if known_revision and known_revision == payload.get("spritesheetRevision"):
        payload.pop("spritesheetBase64", None)
        payload["spritesheetUnchanged"] = True
    return _ok(rid, payload)


@method("pet.info.meta")
@_profile_scoped
@_pet_guard("pet.info.meta", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    enabled, pet, scale = _pet_active_selection()
    if not enabled or pet is None or not pet.exists:
        return _ok(rid, {"enabled": False})
    return _ok(
        rid,
        {
            "enabled": True,
            "slug": pet.slug,
            "displayName": pet.display_name,
            "scale": scale,
            "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        },
    )


@method("pet.cells")
@_profile_scoped
@_pet_guard("pet.cells", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Half-block cell frames for one pet state (TUI renderer).

    Each cell is ``[tr,tg,tb,ta, br,bg,bb,ba]`` (top + bottom pixel).
    Params: ``state`` (idle/run/review/failed/wave/jump), ``cols``, ``graphics``.
    """
    from agent.pet import constants, render, store
    from agent.pet.render import PetRenderer

    pet_cfg = _pet_display_cfg()
    if not is_truthy_value(pet_cfg.get("enabled"), default=False):
        return _ok(rid, {"enabled": False})
    pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
    if pet is None or not pet.exists:
        return _ok(rid, {"enabled": False})

    state = str(params.get("state") or constants.PetState.IDLE.value)
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))
    base = {"enabled": True, "slug": pet.slug, "displayName": pet.display_name, "state": state}

    # Graphics path: a real TTY speaking kitty gets a Unicode-placeholder image
    # instead of half-blocks. Env detection is shared with the Ink process (it
    # spawns us); the dashboard PTY has no such env and falls through. Only
    # kitty is grid-safe in Ink — iTerm/sixel stay on the fallback.
    if params.get("graphics"):
        configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
        gmode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
        if gmode == "kitty":
            image_id = render.kitty_image_id(pet.slug)
            # kitty sizes from scaled pixels, so unicode_cols is moot here.
            payload = PetRenderer(str(pet.spritesheet), mode="kitty", scale=scale).kitty_payload(
                state, image_id=image_id
            )
            if payload:
                return _ok(
                    rid,
                    {
                        **base,
                        "graphics": "kitty",
                        "imageId": image_id,
                        "color": render.kitty_color_hex(image_id),
                        "cols": payload["cols"],
                        "rows": payload["rows"],
                        "placeholder": payload["placeholder"],
                        "frames": payload["frames"],
                        "frameMs": constants.LOOP_MS / max(1, len(payload["frames"]) or 1),
                        "scale": scale,
                    },
                )

    renderer = PetRenderer(str(pet.spritesheet), mode="unicode", scale=scale, unicode_cols=cols)
    count = renderer.frame_count(state) or 1
    frames = [
        [[[*top, *bottom] for (top, bottom) in row] for row in renderer.cells(state, i, cols=cols)]
        for i in range(count)
    ]
    return _ok(
        rid,
        {**base, "cols": cols, "frameMs": constants.LOOP_MS / max(1, count), "frames": frames, "scale": scale},
    )


@method("pet.gallery")
@_profile_scoped
@_pet_guard("pet.gallery", fail_open={"enabled": False, "active": "", "pets": []})
def _(rid, params: dict) -> dict:
    """Adoptable pets for the desktop picker: petdex gallery merged with local install state.

    Fail-open to whatever is installed locally when the gallery is unreachable.
    ``localOnly`` skips the remote manifest so the user's own pets render instantly.
    """
    local_only = bool(params.get("localOnly"))
    from agent.pet import store

    pet_cfg = _pet_display_cfg()
    installed = {p.slug: p for p in store.installed_pets()}

    gallery: list[dict] = []
    seen: set[str] = set()
    try:
        from agent.pet.manifest import fetch_manifest, prefetch

        # Local-only still warms the manifest cache in the background.
        if local_only:
            prefetch()

        for entry in [] if local_only else fetch_manifest():
            seen.add(entry.slug)
            gallery.append(
                {
                    "slug": entry.slug,
                    "displayName": entry.display_name,
                    "installed": entry.slug in installed,
                    "spritesheetUrl": entry.spritesheet_url,
                    # petdex has no popularity metric; "curated" (its hand-picked
                    # set, identified by asset path) is the closest signal.
                    "curated": "/curated/" in entry.spritesheet_url,
                    "generated": entry.slug in installed and installed[entry.slug].generated,
                }
            )
    except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
        logger.debug("pet.gallery manifest fetch failed: %s", exc)

    for slug, pet in installed.items():
        if slug not in seen:
            gallery.append(
                {
                    "slug": slug,
                    "displayName": pet.display_name,
                    "installed": True,
                    "spritesheetUrl": "",
                    "generated": pet.generated,
                }
            )

    return _ok(
        rid,
        {
            "enabled": is_truthy_value(pet_cfg.get("enabled"), default=False),
            "active": str(pet_cfg.get("slug", "") or ""),
            "pets": gallery,
        },
    )


def _with_slug(fn):
    """Require ``params.slug`` (4004 "missing slug") and pass it as a 3rd arg."""

    def handler(rid, params: dict) -> dict:
        slug = str(params.get("slug") or "").strip()
        if not slug:
            return _err(rid, 4004, "missing slug")
        return fn(rid, params, slug)

    return handler


@method("pet.select")
@_profile_scoped
@_pet_guard("pet.select")
@_with_slug
def _(rid, params: dict, slug: str) -> dict:
    """Adopt a pet: install (if needed) + activate; writes ``display.pet.*`` to config."""
    from agent.pet import store
    from agent.pet.manifest import ManifestError
    from hermes_cli.pets import _set_active

    try:
        pet = store.install_pet(slug)
    except (store.PetStoreError, ManifestError) as exc:
        return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
    _set_active(slug)
    return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})


@method("pet.remove")
@_profile_scoped
@_pet_guard("pet.remove")
@_with_slug
def _(rid, params: dict, slug: str) -> dict:
    """Uninstall a pet (delete its directory); if it was active, turn the display off."""
    from agent.pet import store
    from hermes_cli.pets import _clear_active_if

    removed = store.remove_pet(slug)
    try:
        _clear_active_if(slug)
    except Exception as exc:  # noqa: BLE001 - removal already succeeded
        logger.debug("pet.remove config update failed: %s", exc)
    return _ok(rid, {"ok": removed, "slug": slug})


@method("pet.export")
@_profile_scoped
@_pet_guard("pet.export")
@_with_slug
def _(rid, params: dict, slug: str) -> dict:
    """Export an installed pet as a re-importable ``.zip`` → ``{ok, filename, zipBase64}``."""
    import base64

    from agent.pet import store

    filename, data = store.export_pet(slug)
    return _ok(
        rid,
        {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
    )


@method("pet.rename")
@_profile_scoped
@_pet_guard("pet.rename")
@_with_slug
def _(rid, params: dict, slug: str) -> dict:
    """Rename a pet's display name + realign its slug/dir; follows the active slug in config."""
    name = str(params.get("name") or "").strip()
    if not name:
        return _err(rid, 4004, "missing name")
    from agent.pet import store

    new_slug = store.rename_pet(slug, name)
    if not new_slug:
        return _err(rid, 5031, "pet.rename failed")
    if new_slug != slug:
        try:
            from hermes_cli.pets import _rename_active_if

            _rename_active_if(slug, new_slug)
        except Exception as exc:  # noqa: BLE001 - rename already succeeded
            logger.debug("pet.rename config update failed: %s", exc)
    return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})


@method("pet.thumb")
@_profile_scoped
@_pet_guard("pet.thumb", fail_open=lambda params: {"ok": False, "slug": str(params.get("slug") or "").strip()})
@_with_slug
def _(rid, params: dict, slug: str) -> dict:
    """Small idle-frame PNG data URI for the picker preview (same-origin; the desktop
    CSP / R2 hotlink rules break a CDN ``<img>``). ``url`` serves not-yet-installed pets."""
    import base64

    from agent.pet import store

    data = store.thumbnail_png(slug, source_url=str(params.get("url") or ""))
    if not data:
        return _ok(rid, {"ok": False, "slug": slug})
    return _ok(
        rid,
        {
            "ok": True,
            "slug": slug,
            "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
        },
    )


@method("pet.disable")
@_profile_scoped
@_pet_guard("pet.disable")
def _(rid, params: dict) -> dict:
    """``display.pet.enabled=false`` from the desktop picker."""
    from hermes_cli.pets import _set_enabled

    _set_enabled(False)
    return _ok(rid, {"ok": True})


@method("pet.scale")
@_profile_scoped
@_pet_guard("pet.scale")
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` (clamped to engine bounds) from the desktop slider."""
    from hermes_cli.pets import set_pet_scale

    scale, err = set_pet_scale(params.get("scale"))
    if err:
        return _err(rid, 4004, err)
    return _ok(rid, {"ok": True, "scale": scale})


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Signal an in-flight ``pet.generate``/``pet.hatch`` (by token) to stop.

    Idempotent; stays off the worker pool so it lands while a generation occupies it.
    """
    token = str(params.get("token") or "").strip()
    if token:
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@method("pet.generate.status")
@_pet_guard("pet.generate.status", fail_open={"available": False, "providers": []})
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible: a reference-capable image backend is configured."""
    from agent.pet.generate.imagegen import GenerationError, list_sprite_providers, resolve_provider

    try:
        resolve_provider(require_references=True)
        available = True
    except GenerationError:
        available = False
    try:
        providers = list_sprite_providers()
    except Exception as exc:  # noqa: BLE001 - picker is best-effort
        logger.debug("pet provider list failed: %s", exc)
        providers = []
    return _ok(rid, {"available": available, "providers": providers})


@method("pet.generate")
@_pet_guard("pet.generate")
def _(rid, params: dict) -> dict:
    """Generate candidate base looks for a new pet (the draft step). Heavy: worker pool.

    Params: ``prompt`` (required unless ``referenceImage`` — a data URL every draft
    is grounded on), ``count`` (≤4), ``style``, ``provider``. Returns
    ``{ok, token, drafts:[{index, dataUri}]}``; the token keys a later ``pet.hatch``.
    """
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    import shutil

    from agent.pet.generate import generate_base_drafts
    from agent.pet.generate.imagegen import GenerationError, resolve_provider

    root = _pet_gen_root()
    _pet_gen_sweep(root)

    # Token up front so each draft is staged + streamed the moment it lands.
    token = uuid.uuid4().hex[:12]
    _pet_cancel_arm(token)
    stage = root / token
    stage.mkdir(parents=True, exist_ok=True)

    reference_images = None
    if ref_raw:
        try:
            reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
        except ValueError as exc:
            _pet_cancel_release(token)
            return _err(rid, 4004, str(exc))

    # Resolve a picker-chosen provider up front so a bad pick fails fast, not mid-fan-out.
    provider_name = str(params.get("provider") or "").strip()
    sprite = None
    if provider_name:
        try:
            sprite = resolve_provider(require_references=bool(reference_images), prefer=provider_name)
        except GenerationError as exc:
            _pet_cancel_release(token)
            return _err(rid, 5031, str(exc))

    concept = prompt or "a pet based on the reference image"
    out: list[dict] = []

    # Token-only init event so a Stop fired before the first draft can target this run.
    try:
        _emit("pet.generate.progress", "", {"token": token, "count": count})
    except Exception as exc:  # noqa: BLE001 - streaming is best-effort
        logger.debug("pet.generate init emit failed: %s", exc)

    def _on_draft(index: int, src) -> None:
        dest = stage / f"draft-{index}.png"
        try:
            shutil.copyfile(src, dest)
            data_uri = _pet_png_data_uri(dest)
        except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
            logger.debug("pet.generate draft %d failed: %s", index, exc)
            return
        out.append({"index": index, "dataUri": data_uri})
        # Stream the draft so the grid fills live; a transport hiccup must not abort generation.
        try:
            _emit(
                "pet.generate.progress",
                "",
                {"token": token, "index": index, "dataUri": data_uri, "count": count},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pet.generate progress emit failed: %s", exc)

    try:
        generate_base_drafts(
            concept,
            n=count,
            style=style,
            reference_images=reference_images,
            provider=sprite,
            on_draft=_on_draft,
            is_cancelled=lambda: _pet_is_cancelled(token),
        )
    except GenerationError as exc:
        _pet_cancel_release(token)
        return _err(rid, 5031, str(exc))

    cancelled = _pet_is_cancelled(token)
    _pet_cancel_release(token)
    if cancelled:
        return _err(rid, 5031, "generation cancelled")
    if not out:
        return _err(rid, 5031, "generation produced no usable drafts")
    out.sort(key=lambda d: d["index"])
    return _ok(rid, {"ok": True, "token": token, "drafts": out})


@method("pet.hatch")
@_pet_guard("pet.hatch")
def _(rid, params: dict) -> dict:
    """Turn a chosen base draft into a full pet — installed but NOT yet active. Heavy: worker pool.

    The result is a preview the surface plays before the user commits (``pet.select``
    adopts, ``pet.remove`` discards). Params: ``token`` + ``index`` (from
    ``pet.generate``), ``name`` (required), ``description``, ``prompt``, ``style``,
    ``cancelToken``. Returns ``{ok, slug, displayName, warnings, pet}``.
    """
    token = str(params.get("token") or "").strip()
    # Hatch cancellation rides its own key: pet.generate may still be releasing
    # `token`, which would wipe the arm set here. Falls back for old clients.
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(params.get("index", 0))
    except (TypeError, ValueError):
        index = 0

    from agent.pet import store
    from agent.pet.generate import hatch_pet
    from agent.pet.generate.imagegen import GenerationError, resolve_provider

    base = _pet_gen_root() / token / f"draft-{index}.png"
    if not base.is_file():
        return _err(rid, 4004, "draft expired — generate again")

    # Picker override (rows always need reference grounding).
    provider_name = str(params.get("provider") or "").strip()
    sprite = None
    if provider_name:
        try:
            sprite = resolve_provider(require_references=True, prefer=provider_name)
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))

    _pet_cancel_arm(cancel_token)
    slug = store.unique_slug(name)

    def _on_progress(event: str, detail: str) -> None:
        # Row progress is "<state>:<done>:<total>" so the egg screen can show
        # "Drawing <state>… (n/total)"; other phases pass through as-is.
        payload: dict = {"event": event, "detail": detail}
        if event == "row" and detail.count(":") == 2:
            state, done, total = detail.split(":")
            payload = {"event": "row", "state": state, "done": done, "total": total}
        try:
            _emit("pet.hatch.progress", "", payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pet.hatch progress emit failed: %s", exc)

    try:
        result = hatch_pet(
            base_image=base,
            slug=slug,
            display_name=name,
            description=str(params.get("description") or ""),
            concept=str(params.get("prompt") or name),
            style=str(params.get("style") or "auto").strip() or "auto",
            provider=sprite,
            on_progress=_on_progress,
            is_cancelled=lambda: _pet_is_cancelled(cancel_token),
        )
    except GenerationError as exc:
        return _err(rid, 5031, str(exc))
    finally:
        _pet_cancel_release(cancel_token)

    pet = store.load_pet(result.slug)
    payload = _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}
    return _ok(
        rid,
        {
            "ok": True,
            "slug": result.slug,
            "displayName": result.display_name,
            "warnings": result.validation.get("warnings", []),
            "pet": payload,
        },
    )


# ── billing / subscription ───────────────────────────────────────────
# All fail-open: a logged-out / unreachable portal yields an ``ok`` envelope
# with a typed ``error`` (via _serialize_billing_error) rather than a JSON-RPC
# error, so the TUI maps it to the right copy. ``billing:manage`` routes return
# error=insufficient_scope on 403, which drives the ``billing.step_up`` device flow.


@method("billing.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/state → serialized BillingState. No scope required."""
    try:
        from agent.billing_view import build_billing_state

        return _ok(rid, _serialize_billing_state(build_billing_state()))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load billing state"})


@method("usage.bars")
def _(rid, params: dict) -> dict:
    """Shared dollar usage model (two-bar view) for /usage + /subscription."""
    try:
        from agent.billing_usage import build_usage_model

        return _ok(rid, _serialize_usage_model(build_usage_model()))
    except Exception:
        return _ok(rid, {"ok": True, "available": False})


@method("subscription.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/subscription → serialized SubscriptionState (read-only)."""
    try:
        from agent.subscription_view import build_subscription_state

        return _ok(rid, _serialize_subscription_state(build_subscription_state()))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load subscription state"})


@method("subscription.preview")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/preview → chargeless effect quote. billing:manage."""
    from agent.subscription_view import subscription_change_preview_from_payload
    from hermes_cli.nous_billing import post_subscription_preview

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _billing_invalid(rid, "subscription_type_id is required")
    return _billing_call(
        rid,
        lambda: _serialize_subscription_preview(
            subscription_change_preview_from_payload(post_subscription_preview(subscription_type_id=tier_id))
        ),
    )


@method("subscription.change")
def _(rid, params: dict) -> dict:
    """PUT /api/billing/subscription/pending-change: schedule a downgrade / same-price
    change OR a period-end cancellation (chargeless). billing:manage."""
    from hermes_cli.nous_billing import put_subscription_pending_change

    cancel = bool(params.get("cancel"))
    tier_id = params.get("subscription_type_id")
    if not cancel and not tier_id:
        return _billing_invalid(rid, "subscription_type_id or cancel is required")

    def call():
        result = put_subscription_pending_change(subscription_type_id=tier_id, cancel=cancel)
        return {"ok": True, "message": result.get("message"), "payload": result}

    return _billing_call(rid, call)


@method("subscription.resume")
def _(rid, params: dict) -> dict:
    """DELETE /api/billing/subscription/pending-change: clear a scheduled downgrade /
    cancellation. Re-enables recurring spend → billing:manage + kill-switch."""
    from hermes_cli.nous_billing import delete_subscription_pending_change

    def call():
        result = delete_subscription_pending_change()
        return {"ok": True, "message": result.get("message"), "payload": result}

    return _billing_call(rid, call)


@method("subscription.upgrade")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/upgrade — the single money route: prorate + charge + flip plan.

    SCA / decline come back as status requires_action / payment_failed with a
    recovery_url. The idempotency key is minted if absent and echoed (also on
    error) so the TUI reuses it on retry of the SAME upgrade. billing:manage.
    """
    from agent.billing_view import new_idempotency_key
    from hermes_cli.nous_billing import post_subscription_upgrade

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _billing_invalid(rid, "subscription_type_id is required")
    key = params.get("idempotency_key") or new_idempotency_key()

    def call():
        result = post_subscription_upgrade(subscription_type_id=tier_id, idempotency_key=key)
        return {
            "ok": True,
            "status": result.get("status"),
            "target_tier_name": result.get("targetTierName"),
            "recovery_url": result.get("recoveryUrl"),
            "reason": result.get("reason"),
            "idempotency_key": key,
        }

    return _billing_call(rid, call, extra={"idempotency_key": key})


@method("billing.charge")
def _(rid, params: dict) -> dict:
    """POST /api/billing/charge → {ok, charge_id, idempotency_key}; key minted if absent
    and echoed (also on error) so the TUI reuses it on retry of the SAME purchase."""
    from hermes_cli.nous_billing import post_charge
    from agent.billing_view import new_idempotency_key

    amount = params.get("amount_usd")
    if amount is None:
        return _billing_invalid(rid, "amount_usd is required")
    key = params.get("idempotency_key") or new_idempotency_key()

    def call():
        result = post_charge(amount_usd=amount, idempotency_key=key)
        return {"ok": True, "charge_id": result.get("chargeId"), "idempotency_key": key}

    return _billing_call(rid, call, extra={"idempotency_key": key})


@method("billing.charge_status")
def _(rid, params: dict) -> dict:
    """GET /api/billing/charge/{id} — a single status read; the caller drives the poll cadence."""
    from hermes_cli.nous_billing import get_charge_status

    charge_id = params.get("charge_id")
    if not charge_id:
        return _billing_invalid(rid, "charge_id is required", error="invalid_charge_id")

    def call():
        result = get_charge_status(charge_id)
        return {
            "ok": True,
            "status": result.get("status"),
            "amount_usd": result.get("amountUsd"),
            "settled_at": result.get("settledAt"),
            "reason": result.get("reason"),
        }

    return _billing_call(rid, call)


@method("billing.auto_reload")
def _(rid, params: dict) -> dict:
    """PATCH /api/billing/auto-top-up. params: {enabled, threshold, top_up_amount}."""
    from hermes_cli.nous_billing import patch_auto_top_up

    enabled = bool(params.get("enabled"))
    threshold = params.get("threshold")
    top_up_amount = params.get("top_up_amount")
    if threshold is None or top_up_amount is None:
        return _billing_invalid(rid, "threshold and top_up_amount are required")

    def call():
        patch_auto_top_up(enabled=enabled, threshold=threshold, top_up_amount=top_up_amount)
        return {"ok": True}

    return _billing_call(rid, call)


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """Lazy billing:manage step-up device flow → {ok, granted}; granted:false when the
    server silently downscopes.

    Runs on the thread pool (_LONG_HANDLERS): the device flow blocks for minutes.
    The verification URL/code reach the TUI via the out-of-band
    ``billing.step_up.verification`` event (a print would be lost in the JSON-RPC
    stdout pipe) and the browser is opened TUI-side — never via the gateway's
    headless webbrowser.open (open_browser=False).
    """
    sid = params.get("session_id") or ""

    def call():
        from hermes_cli.auth import step_up_nous_billing_scope

        def _on_verification(url: str, code: str) -> None:
            _emit("billing.step_up.verification", sid, {"verification_url": url, "user_code": code})

        granted = step_up_nous_billing_scope(open_browser=False, on_verification=_on_verification)
        return {"ok": True, "granted": bool(granted)}

    return _billing_call(rid, call, extra={"granted": False})


# ── session status / history / undo / compress / save / close ────────


@method("session.status")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")

    def _row(db) -> dict:
        try:
            return db.get_session(key) or {}
        except Exception:
            return {}

    meta = {}
    # Prefer the live session's bound profile db, else params.profile / launch.
    with _session_db(session) as db:
        if db is not None:
            if key:
                meta = _row(db)
        else:
            with _profile_db(params) as db2:
                if db2 and key:
                    meta = _row(db2)

    def _dt(value, fallback: datetime | None = None) -> datetime:
        if value:
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                pass
        return fallback or datetime.now()

    created = _dt(meta.get("started_at"))
    updated = created
    for field in ("updated_at", "last_updated_at", "last_activity_at"):
        if meta.get(field):
            updated = _dt(meta.get(field), created)
            break

    mirror = _metadata_mirror(session)
    usage = _session_usage_snapshot(session)
    provider = getattr(agent, "provider", None) or mirror.get("provider") or "unknown"
    model = getattr(agent, "model", None) or mirror.get("model") or "(unknown)"
    project = _project_info_for_cwd(_display_session_cwd(session))
    lines = ["Hermes TUI Status", "", f"Session ID: {key}", f"Path: {display_hermes_home()}"]
    if project:
        lines.append(f"Project: {project['name']}")
    title = (meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")
    lines.extend(
        [
            f"Model: {model} ({provider})",
            f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {int(usage.get('total') or 0):,}",
            f"Agent Running: {'Yes' if session.get('running') else 'No'}",
        ]
    )
    return _ok(rid, {"output": "\n".join(lines)})


@method("session.history")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    history = list(session.get("history", []))
    if session.get("session_key"):
        with _session_db(session) as db:
            if db is not None:
                try:
                    # include_row_ids: the durable row id is how clients address a
                    # persisted turn (reactions, content-based truncation targets);
                    # _history_to_messages only forwards row_id when stamped.
                    history = db.get_messages_as_conversation(
                        session["session_key"], include_ancestors=True, include_row_ids=True
                    )
                except Exception:
                    pass
    return _ok(rid, {"count": len(history), "messages": _history_to_messages(history)})


@method("session.undo")
@_with_live_session
def _(rid, params: dict, session: dict) -> dict:
    # Mutating history under a running turn would make prompt.submit's post-run
    # write either clobber the undo or drop the agent's output — /interrupt first.
    busy = _err(rid, 4009, "session busy — /interrupt the current turn before /undo")
    if session.get("running"):
        return busy
    removed = 0
    with session["history_lock"]:
        if session.get("running"):
            return busy
        history = _history_without_ephemeral_scaffolding(session.get("history", []))
        # Truncate from the last *real* user turn: popping trailing assistant/tool
        # then one user left timeline markers / compaction handoffs as the target.
        from agent.context_compressor import user_originated_turn_view

        user_indices = [
            index for index, message in enumerate(history) if user_originated_turn_view(message) is not None
        ]
        if user_indices:
            try:
                _installed, _live_view, removed = _rewind_active_session_history(
                    session, len(user_indices) - 1
                )
            except Exception as exc:
                return _err(rid, 5008, f"undo: {exc}")
    return _ok(rid, {"removed": removed})


def _compress_via_compute_host(rid, params: dict, session: dict) -> dict:
    """``session.compress`` for a turn-isolated session: forward ``/compress`` to the host."""
    sid = str(params.get("session_id") or "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
    command = "/compress" + (f" {focus_topic}" if focus_topic else "")

    def _on_late_ack(late: dict, _sid=sid) -> None:
        _adopt_late_compute_host_compress_ack(_sid, session, late, route_name="session.compress")

    try:
        ack = _send_compute_host_control(
            sid,
            route_name="session.compress",
            command=command,
            wait=True,
            # Follows compression.context_total_ceiling_seconds: the host legitimately runs that long.
            timeout=_compute_host_compress_wait_seconds(),
            on_late_ack=_on_late_ack,
        )
    except queue.Empty:
        # The waiter gave up but the host is still compressing; the late-ack
        # handler adopts the rotated session and pushes session.info when it
        # lands. Not an error (a 5019 here made clients report a timeout while
        # compression later succeeded silently).
        return _ok(
            rid,
            {
                "status": "pending",
                "turn_isolation": True,
                "message": (
                    "compression still running in the background; "
                    "the transcript will refresh when it finishes"
                ),
            },
        )
    except Exception as exc:
        return _err(rid, 5019, f"compute-host compress failed: {exc}")
    if ack.get("type") in {"control.error", "error"}:
        return _err(rid, 4009, str(ack.get("message") or "compute-host compress failed"))
    _apply_compute_host_metadata_mirror(session, ack)
    host_result = ack.get("result")
    if isinstance(host_result, dict):
        # The host owns the isolated session; preserve its structured result
        # verbatim (it carries `status: aborted` / `summary.aborted`).
        return _ok(rid, {**host_result, "turn_isolation": True})
    host_info = ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
    host_messages = _history_to_messages(ack.get("messages")) if isinstance(ack.get("messages"), list) else []
    # `messages` goes at top level for the transcript replacement; don't send the
    # same (large) transcript a second time inside the ack.
    host_ack = {key: value for key, value in ack.items() if key != "messages"}
    return _ok(
        rid,
        {
            "status": "compressed",
            "turn_isolation": True,
            "host_ack": host_ack,
            "info": host_info,
            "messages": host_messages,
            "usage": host_info.get("usage") if isinstance(host_info.get("usage"), dict) else {},
        },
    )


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if _session_uses_compute_host(session):
        return _compress_via_compute_host(rid, params, session)
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before /compress")
    from agent.conversation_compression import finalize_context_engine_compression_notification

    sid = params.get("session_id", "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
    try:
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough

        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
        before_count = len(before_messages)
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None

        def _tokens(msgs, sys_prompt, tools) -> int:
            return estimate_request_tokens_rough(msgs, system_prompt=sys_prompt, tools=tools) if msgs else 0

        before_tokens = _tokens(before_messages, _sys_prompt, _tools)

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            _status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages (~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = _compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            # Re-read prompt + tools: _compress_context may have rebuilt the system prompt.
            after_tokens = _tokens(
                messages,
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt,
                getattr(_agent, "tools", None) or _tools,
            )
            agent = session["agent"]
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
            finalize_context_engine_compression_notification(agent, committed=True)
            return _ok(
                rid,
                {
                    "status": "aborted" if summary["aborted"] else "compressed",
                    "removed": removed,
                    "before_messages": before_count,
                    "after_messages": after_count,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "summary": summary,
                    "usage": usage,
                    "info": info,
                    # Same projection as session.resume / session.history: raw tool
                    # results belong in persisted history, not the transcript response.
                    "messages": _history_to_messages(messages),
                },
            )
        finally:
            # Always clear the pinned compressing status (success, no-op, or raise).
            _status_update(sid, "ready")
    except CompressionLockHeld as e:
        _status_update(sid, "ready")
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return _ok(rid, {"compressed": False, "lock_held": True, "message": describe_compression_lock_skip(e.holder)})
    except Exception as e:
        finalize_context_engine_compression_notification(session["agent"], committed=False)
        return _err(rid, 5005, str(e))


@method("session.save")
@_with_live_session
def _(rid, params: dict, session: dict) -> dict:
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        try:
            ack = _send_compute_host_control(sid, route_name="session.save", wait=True)
        except Exception as exc:
            return _err(rid, 5011, f"compute-host session save failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 5011, str(ack.get("message") or "compute-host session save failed"))
        result = ack.get("result")
        if not isinstance(result, dict):
            return _err(rid, 5011, "compute-host session save returned an invalid response")
        return _ok(rid, result)

    agent = session["agent"]
    # Mirror the classic CLI /save: snapshot under the profile home (not the
    # workspace cwd) and include the system prompt so it matches the dashboard save.
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")

    path = saved_dir / f"hermes_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    # Prefer the agent's session_start (classic CLI export); else the gateway created_at.
    agent_start = getattr(agent, "session_start", None)
    if isinstance(agent_start, datetime):
        session_start = agent_start.isoformat()
    else:
        created_at = session.get("created_at")
        session_start = datetime.fromtimestamp(created_at).isoformat() if isinstance(created_at, (int, float)) else ""

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": getattr(agent, "model", ""),
                    "session_id": session_id,
                    "session_start": session_start,
                    "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                    "messages": messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return _ok(rid, {"file": str(path)})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    # Serialize only the ownership claim against session.resume / the reaper;
    # finalization may run arbitrary plugin cleanup and must not block other resumes.
    with _session_resume_lock:
        session = _pop_session_by_id(sid)
    closed = _teardown_popped_session(session, end_reason="tui_close")
    return _ok(rid, {"closed": closed})


# ── session.branch ───────────────────────────────────────────────────


def _visible_branch_history(messages) -> list:
    """user/assistant rows with visible text, as FULL row copies (reasoning fields and
    timeline-marker tags — display_kind/display_metadata — must survive the branch)."""
    visible = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        if not _coerce_message_text(message.get("content")).strip():
            continue
        visible.append(dict(message))
    return visible


_BRANCH_COPY_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    # Timeline markers ride as role=user; dropping the tag re-plants them as bare
    # user turns after a restart, corrupting the truncate ordinal address space.
    "display_kind",
    "display_metadata",
    # Branch copies are history, not new activity: keep the parent's timestamps.
    "timestamp",
)


@method("session.branch")
@_with_live_session
def _(rid, params: dict, session: dict) -> dict:
    # Branch writes into the parent's profile-scoped state.db (app-global remote
    # mode); the launch handle would orphan branch rows + history.
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        with session["history_lock"]:
            in_memory_history = [
                dict(msg)
                for msg in list(session.get("display_history_prefix") or []) + list(session.get("history", []))
                if isinstance(msg, dict)
            ]

        # The live history is the MODEL projection — after compaction only a
        # summary + protected tail. Snapshot the persisted display projection
        # instead, or the child permanently loses every turn archived before the fork.
        history = None
        get_resume_conversations = getattr(db, "get_resume_conversations", None)
        if callable(get_resume_conversations):
            try:
                _, display_history = get_resume_conversations(old_key)
                display_history = _reconcile_display_with_live(display_history, in_memory_history)
                history = _visible_branch_history(display_history)
            except Exception:
                logger.debug("branch display projection read failed", exc_info=True)
        if not history:
            history = _visible_branch_history(in_memory_history)
        if not history:
            return _err(rid, 4008, "nothing to branch — send a message first")
        count = params.get("count")
        if isinstance(count, int) and count > 0:
            history = history[:count]
        new_key = _new_session_key()
        new_sid = uuid.uuid4().hex[:8]
        source = _session_source(session)
        try:
            title = params.get("name", "") or _branch_title(db, old_key)
            _create_branch_rows(
                db,
                new_key,
                old_key,
                title,
                history,
                source=source,
                cwd=_session_cwd(session),
                profile_name=(
                    Path(session["profile_home"]).name if session.get("profile_home") else _current_profile_name()
                ),
                copy_fields=_BRANCH_COPY_FIELDS,
            )
        except Exception as e:
            return _err(rid, 5008, f"branch failed: {e}")
    # Bound before the try so the ownership finally can never see them unbound.
    branch_db = None
    branch_owns_db = False
    try:
        # Bind the branched AGENT to the parent's profile like session.create/
        # resume: home + secret scope for the build, and the profile's own state.db
        # handle so message flushes and later compression rotation persist there.
        parent_home = session.get("profile_home")
        if parent_home:
            # DEDICATED handle, same ownership rule as session.resume: ours until
            # the branched agent takes it below.
            from hermes_state import get_shared_session_db

            branch_db = get_shared_session_db(Path(parent_home) / "state.db")
            branch_owns_db = True
        with _profile_build_scope(parent_home):
            tokens = _set_session_context(new_key)
            try:
                agent = _make_agent(
                    new_sid,
                    new_key,
                    session_id=new_key,
                    session_db=branch_db,
                    platform_override=source,
                    context_cwd_is_launch_artifact=_context_cwd_is_launch_artifact(session),
                )
            finally:
                _clear_session_context(tokens)
            _init_session(
                new_sid,
                new_key,
                agent,
                list(history),
                cols=session.get("cols", 80),
                cwd=_session_cwd(session),
                session_db=branch_db,
                source=source,
                profile_home=parent_home,
                explicit_cwd=bool(session.get("explicit_cwd")),
            )
            # Ownership TRANSFER (unconditional drop, as in session.resume): past
            # _init_session the branched session is registered against this handle.
            _transfer_db_to_agent(agent, branch_db)
            branch_owns_db = False
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = None  # claimed lazily on the first turn
    except Exception as e:
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    finally:
        if branch_owns_db and branch_db is not None:
            with contextlib.suppress(Exception):
                from hermes_state import release_or_close

                release_or_close(branch_db)
    branched_session = _sessions.get(new_sid)
    return _ok(
        rid,
        {
            "session_id": new_sid,
            "stored_session_id": new_key,
            "title": title,
            "parent": old_key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": _session_info(agent, branched_session),
        },
    )


# ── interrupt / steer / redirect ─────────────────────────────────────


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    # Keypress barge-in also silences streaming TTS (voice is process-global).
    _tts_stream_stop()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    expected_hosted_task_id = str(params.get("expected_hosted_task_id") or "").strip()
    if expected_hosted_task_id:
        with session["history_lock"]:
            active_task = session.get("_hosted_room_task")
            if (
                not session.get("running")
                or not isinstance(active_task, dict)
                or active_task.get("task_id") != expected_hosted_task_id
            ):
                return _ok(rid, {"status": "not_interrupted", "interrupted": False})
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        try:
            _interrupt_session_turn(sid, session, request_id=f"interrupt-{rid}")
        except Exception as exc:
            return _err(rid, 5019, f"compute-host interrupt failed: {exc}")
        return _ok(rid, {"status": "interrupted", "turn_isolation": True})
    session, err = _sess(params, rid)
    if err:
        return err
    _interrupt_session_turn(str(params.get("session_id") or ""), session)
    # Retire the crash-recovery marker on a confirmed local Stop now: waiting for
    # the run thread's finally leaves a window where a backend exit looks like a
    # crash and session.resume auto-continues the turn the user just stopped.
    # Extra key covers compression rotating session_key mid-turn.
    with session["history_lock"]:
        active_marker_key = str(session.pop("_active_turn_marker_key", "") or "")
    _retire_turn_marker(session, active_marker_key)
    return _ok(rid, {"status": "interrupted"})


def _record_accepted_correction(session: dict, text: str) -> None:
    """Record a steer/redirect on the live turn so a mid-turn resume rebuilds the user
    bubble, and purge server-queue self-copies of the live original so post-turn
    drain cannot re-fire the pre-correction prompt."""
    with session["history_lock"]:
        _record_inflight_correction(session, text)
        _drop_queued_duplicates_of_inflight_user(session)
        session["last_active"] = time.time()


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(): the text lands on the last tool result of the next
    tool batch. No interrupt, no new user turn, no role alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    if accepted:
        _record_accepted_correction(session, text)
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("session.redirect")
def _(rid, params: dict) -> dict:
    """Redirect the active model turn while preserving valid work/context."""
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    # Turn-build window: a fresh turn flips running=True with agent still None.
    # Queue the correction server-side for the next turn instead of a misleading
    # 4010 the client swallows into a lost follow-up.
    if agent is None and session.get("running"):
        _enqueue_prompt(session, text, current_transport() or _stdio_transport)
        session["last_active"] = time.time()
        return _ok(rid, {"status": "queued", "text": text})
    if (
        agent is None
        or getattr(agent, "_supports_active_turn_redirect", False) is not True
        or not hasattr(agent, "redirect")
    ):
        return _err(rid, 4010, "agent does not support active-turn redirect")
    try:
        accepted = agent.redirect(text)
    except Exception as exc:
        return _err(rid, 5000, f"redirect failed: {exc}")
    if accepted:
        _record_accepted_correction(session, text)
    return _ok(rid, {"status": "redirected" if accepted else "rejected", "text": text})


# ── delegation / spawn trees ─────────────────────────────────────────


@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import (
        is_spawn_paused,
        list_active_subagents,
        _get_max_concurrent_children,
        _get_max_spawn_depth,
    )

    return _ok(
        rid,
        {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": _get_max_spawn_depth(),
            "max_concurrent_children": _get_max_concurrent_children(),
        },
    )


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused

    return _ok(rid, {"paused": set_spawn_paused(bool(params.get("paused", True)))})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    return _ok(rid, {"found": interrupt_subagent(subagent_id), "subagent_id": subagent_id})


@method("subagent.steer")
def _(rid, params: dict) -> dict:
    """Queue steering text into a live delegated child without stopping it.

    Resolves the child in the delegation registry and calls AIAgent.steer(); the
    in-flight tool call is never cut. "queued" is not "delivered": a child past
    its final tool batch has no boundary left, and that race surfaces as
    ``missed_steer`` on the parent's completion entry.
    """
    from tools.delegate_tool import steer_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    _invoking_session, err = _sess_nowait(params, rid)
    if err:
        return err
    invoking_session_id = str(params.get("session_id") or "").strip()
    invoking_transport, invoking_session = _current_session_steer_authority(invoking_session_id)
    queued = False
    if invoking_transport is not None and invoking_session is not None:
        queued = steer_subagent(
            subagent_id,
            text,
            owner_session_id=invoking_session_id,
            owner_transport=invoking_transport,
            owner_session_record=invoking_session,
        )
    return _ok(rid, {"status": "queued" if queued else "rejected", "subagent_id": subagent_id, "text": text})


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or time.time()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / f"{ts}.json"
    try:
        payload = {
            "session_id": session_id,
            "started_at": float(started_at) if started_at else None,
            "finished_at": float(finished_at),
            "label": label,
            "subagents": subagents,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")

    _append_spawn_tree_index(
        d,
        {
            "path": str(path),
            "session_id": session_id,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "label": label,
            "count": len(subagents),
        },
    )
    return _ok(rid, {"path": str(path), "session_id": session_id})


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    if bool(params.get("cross_session")):
        roots = [p for p in _spawn_trees_root().iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for d in roots:
        indexed = _read_spawn_tree_index(d)
        if indexed:
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(e for e in indexed if (p := e.get("path")) and Path(p).exists())
            continue
        # Legacy (pre-index) sessions: full scan, once per session until the next save.
        for p in d.glob("*.json"):
            if p.name == _SPAWN_TREE_INDEX:
                continue
            try:
                stat = p.stat()
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append(
                    {
                        "path": str(p),
                        "session_id": raw.get("session_id") or d.name,
                        "finished_at": raw.get("finished_at") or stat.st_mtime,
                        "started_at": raw.get("started_at"),
                        "label": raw.get("label") or "",
                        "count": len(subagents) if isinstance(subagents, list) else 0,
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")
    # Reject paths escaping the spawn-trees root.
    root = _spawn_trees_root().resolve()
    try:
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")
    return _ok(rid, payload)


# ── terminal / event replay ──────────────────────────────────────────


@method("terminal.resize")
@_with_session
def _(rid, params: dict, session: dict) -> dict:
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


@method("session.events.since")
def _(rid, params: dict) -> dict:
    """Replay recorded events newer than the client's last-seen seq (WS reconnect contract).

    Frames older than the ring window report ``truncated`` so the client refetches
    history instead of silently accepting a gap.
    """
    sid = str(params.get("session_id") or "")
    try:
        last_seen = int(params.get("last_seen", 0))
    except (TypeError, ValueError):
        return _err(rid, -32602, "invalid params: last_seen must be an integer")
    from tui_gateway import event_replay

    frames = event_replay.events_since(sid, last_seen)
    return _ok(
        rid,
        {
            "events": frames,
            "latest_seq": event_replay.latest_seq(sid),
            "truncated": event_replay.is_truncated(sid, last_seen),
            "count": len(frames),
            # seq counters are in-process: clients compare this against the epoch
            # from gateway.ready and reset watermarks on mismatch (restart detection).
            "epoch": event_replay.replay_epoch(),
        },
    )


@method("session.events.stats")
def _(rid, params: dict) -> dict:
    """Replay-buffer telemetry (ops/debug)."""
    from tui_gateway import event_replay

    return _ok(rid, event_replay.replay_stats())


def register(server) -> None:
    """Publish this module's helpers onto ``server`` and install its handlers.

    Helpers are module-level functions/classes, so install() alone would leave them
    bound to THIS module's (empty) globals; ``bind_module`` rebinds them onto
    server.py's namespace so they resolve the same free names as the handlers.
    """
    bind_module(globals(), server, skip=("_",))
