"""Session working-directory + durable session row: cwd resolution/healing, session.db row ensure, branch seed, history rewind, git meta persistence.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib
from tui_gateway import git_probe

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if (
            len(normalized) >= 3
            and normalized[1] == ":"
            and normalized[2] == "/"
            and normalized[0].isalpha()
        ):
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


def _completion_cwd(params: dict | None = None) -> str:
    params = params or {}
    raw = (
        params.get("cwd")
        or _sessions.get(params.get("session_id") or "", {}).get("cwd")
        # A session bound to another profile resolves its workspace from THAT
        # profile's config before falling back to the launch profile's env var.
        or _profile_configured_cwd(_profile_home(params.get("profile")))
        # The launch profile's dashboard /chat attaches to the dashboard's
        # in-memory gateway, which does NOT inherit the PTY child's bridged
        # TERMINAL_CWD. Read the launch profile's config.yaml directly so a
        # configured terminal.cwd wins over a stale process env / launch dir.
        or _launch_configured_cwd()
        or os.environ.get("TERMINAL_CWD")
        or os.getcwd()
    )
    try:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return os.getcwd()


def _terminal_task_cwd(session: dict | None) -> str:
    """Return the cwd that terminal_tool should use for this TUI session.

    ``_completion_cwd`` validates paths on the host so file completion does not
    point at nonsense.  Non-local terminal backends are different: their cwd is
    inside the target environment, so an SSH path like /home/user/workspace may
    not exist on the local macOS host but is still the correct execution cwd.

    When ``TERMINAL_ENV`` is unset (dashboard/TUI process) the config's
    ``terminal.backend`` is consulted as a fallback so the non-local cwd
    resolution path is taken even when the dashboard entrypoint did not call
    ``apply_terminal_config_to_env`` on its own ``os.environ``.
    """
    return _terminal_task_cwd_with_source(session)[0]


def _terminal_task_cwd_with_source(session: dict | None) -> tuple[str, str]:
    """Like :func:`_terminal_task_cwd` but also names the value's ORIGIN.

    Returns ``(cwd, source)`` where source is:

    * ``"session"`` — the workspace the user attached to THIS session
      (``explicit_cwd``), or this session's own tracked directory.
    * ``"process"`` — the process-global ``TERMINAL_CWD`` env var / config
      ``terminal.cwd`` fallback.  On a shared-container backend this is the
      normal seed; under per-session docker isolation it is a launch
      artifact from a PREVIOUS session (the workspace picker persists it
      process-wide) and must never become a fresh session's bind mount —
      terminal_tool refuses ``cwd_source: "process"`` as a mount source.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if not backend or backend == "local":
        # Fall back to config when TERMINAL_ENV is unset (dashboard/TUI process
        # never calls apply_terminal_config_to_env on os.environ).
        try:
            terminal_cfg = _load_cfg().get("terminal", {})
            if isinstance(terminal_cfg, dict):
                cfg_backend = str(terminal_cfg.get("backend") or "").strip().lower()
                if cfg_backend and cfg_backend != "local":
                    backend = cfg_backend
        except Exception:
            pass

    if backend and backend != "local":
        # A workspace the user explicitly attached to THIS session wins over
        # the process-global env var — the env var is whatever the LAST
        # session's picker wrote, not this session's choice.
        if session and session.get("explicit_cwd") and session.get("cwd"):
            return str(session["cwd"]), "session"
        raw = os.environ.get("TERMINAL_CWD", "").strip()
        if not raw:
            try:
                terminal_cfg = _load_cfg().get("terminal", {})
                if isinstance(terminal_cfg, dict):
                    raw = str(terminal_cfg.get("cwd") or "").strip()
            except Exception:
                raw = ""
        if raw and raw not in {".", "auto", "cwd"}:
            return raw, "process"
        if backend == "ssh":
            return "~", "process"

    if session and session.get("cwd"):
        return str(session["cwd"]), "session"
    return _completion_cwd(), "process"


# Git working-tree probing (run git, resolve roots, fold worktrees) lives in a
# focused, single-flight-cached module; these stay as the in-server names every
# call site already uses.
_git = git_probe.run_git
_git_branch_for_cwd = git_probe.branch
_git_repo_root_for_cwd = git_probe.repo_root
_git_common_repo_root_for_cwd = git_probe.common_repo_root
_resolve_cwd_git = git_probe.resolve


def _session_cwd(session: dict | None) -> str:
    if session and session.get("cwd"):
        return str(session["cwd"])
    return _completion_cwd()


# Sources whose launch directory is an artifact of how the app was started, not
# a workspace the user picked. Everything else is terminal-started: the process
# runs in a directory the user deliberately cd'd into.
_LAUNCH_CWD_NOT_A_WORKSPACE = {"desktop"}


def _context_cwd_is_launch_artifact(session: dict | None) -> bool:
    """Whether the session cwd came from app launch rather than user intent."""
    return bool(
        session
        and not session.get("explicit_cwd")
        and _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE
    )


def _persisted_session_cwd(session: dict) -> str | None:
    """The cwd to stamp on the session's DB row, or None to leave it unset.

    See :func:`_ensure_session_db_row` for why the launch directory counts as a
    workspace for terminal sessions but not for the desktop.
    """
    if session.get("explicit_cwd"):
        return _session_cwd(session)
    if _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE:
        return None
    # Only the session's OWN directory. `_session_cwd` falls back to the
    # gateway-wide completion cwd, which belongs to no session in particular —
    # stamping that would invent a workspace for a session that never had one.
    return str(session.get("cwd") or "") or None


def _heal_dead_cwd(cwd: str) -> str:
    """Resolve a session cwd that points at a now-deleted directory.

    A session anchored to a linked worktree (``<repo>/.worktrees/<name>``) keeps
    that path after the worktree is removed (branch merged, `git worktree
    remove`, etc). The literal dir is gone, so a probe of it returns nothing and
    the composer shows no branch — while the sidebar still folds the path up to
    the repo's main lane. Heal the mismatch: walk up to the first existing
    ancestor, then resolve its common git root, so a dead-worktree cwd collapses
    to the live repo root (and its real current branch).

    Only meaningful for local backends; a remote/SSH cwd may legitimately not
    exist on the host, so callers must skip healing there.
    """
    raw = (cwd or "").strip()
    if not raw or os.path.isdir(raw):
        return raw

    probe = raw
    # Climb to the first ancestor that still exists on disk.
    for _ in range(64):
        parent = os.path.dirname(probe)
        if not parent or parent == probe:
            break
        probe = parent
        if os.path.isdir(probe):
            break

    if not os.path.isdir(probe):
        return raw

    try:
        root = _git_common_repo_root_for_cwd(probe) or _git_repo_root_for_cwd(probe)
    except Exception:
        root = ""

    return root or probe


def _is_local_terminal_backend() -> bool:
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    return not backend or backend == "local"


def _effective_terminal_backend() -> str:
    """Active terminal backend name (``local``, ``docker``, ``ssh``, ...).

    ``TERMINAL_ENV`` is authoritative when set (launchers bridge
    ``terminal.backend`` into env at startup). Desktop/TUI in-process gateways
    skip that bridge, so fall back to the ``terminal.backend`` config key —
    the same rule ``_terminal_task_cwd`` uses.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if not backend or backend == "local":
        try:
            terminal_cfg = _load_cfg().get("terminal", {})
            if isinstance(terminal_cfg, dict):
                cfg_backend = str(terminal_cfg.get("backend") or "").strip().lower()
                if cfg_backend and cfg_backend != "local":
                    backend = cfg_backend
        except Exception:
            pass
    return backend or "local"


def _display_session_cwd(session: dict | None) -> str:
    """Session cwd for display/probe surfaces, healed past deleted worktrees.

    Persists the healed value back to the session row (best-effort, local only)
    so the next load is already coherent and the sidebar lane stops showing a
    session pinned to a vanished path.
    """
    cwd = _session_cwd(session)
    if not _is_local_terminal_backend():
        return cwd

    healed = _heal_dead_cwd(cwd)
    if healed and healed != cwd and session is not None:
        session["cwd"] = healed
        _persist_session_cwd_and_schedule_git_meta(session, healed)

    return healed


def _reconcile_session_cwd_from_terminal(session: dict | None) -> bool:
    """Re-anchor a session that SETTLED in another git checkout. Returns moved.

    An agent told to work in a fresh worktree does exactly that — `git worktree
    add`, `cd` into it, and every later command runs there — but the session
    stayed pinned to wherever it started, so the desktop kept labelling the chat
    with the primary checkout's branch while all the work landed elsewhere.

    A plain `cd` is deliberately NOT a workspace move (see
    ``_apply_project_workspace``): browsing to /tmp to read a log must not
    re-home the chat. What we adopt here is narrower — the session's recorded
    cwd is in a DIFFERENT working tree of the SAME repository (the shape
    ``git worktree add`` produces). Everything else — a non-git workspace
    stepping into a repo, or a git workspace visiting an unrelated repo — is
    a browsing visit, and a user's explicitly chosen workspace is never
    overridden at all.

    Local backends only: a remote/SSH cwd names a path on the host, which this
    gateway can neither stat nor probe with git.
    """
    if not session or not _is_local_terminal_backend():
        return False

    # A workspace the user (or GUI) explicitly chose is never overridden by
    # where the agent's terminal happened to settle — only another explicit
    # action (`_set_session_cwd`, a project switch) moves it. A cwd this very
    # function adopted is marked `cwd_from_settle` so a session can keep
    # following the agent through successive worktrees.
    if session.get("explicit_cwd") and not session.get("cwd_from_settle"):
        return False

    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(session.get("session_key") or "")
    except Exception:
        return False

    if not recorded:
        return False

    resolved = os.path.abspath(os.path.expanduser(str(recorded)))
    current = os.path.abspath(os.path.expanduser(_session_cwd(session)))
    if resolved == current or not os.path.isdir(resolved):
        return False

    # The worktree ROOT, not the common repo root: folding worktrees together
    # here is exactly what hides the move we're looking for.
    landed = _git_repo_root_for_cwd(resolved)
    current_root = _git_repo_root_for_cwd(current)
    # A relocation is a move between two DIFFERENT git working trees. When the
    # session's own workspace is not in a git repo, the agent stepping into one
    # to read a file or run a command is a browsing visit, not a re-home:
    # adopting it would hijack a non-git workspace onto whatever repo a tool
    # call touched first (e.g. a home-directory session pinned to the checkout
    # it read a file from).
    if not landed or not current_root or landed == current_root:
        return False

    # And only between checkouts of the SAME repository — the shape a real
    # `git worktree add` produces (linked worktrees share the common .git
    # dir). Settling in an UNRELATED repo (`cd ~/other-project && git log`)
    # is likewise a visit: adopting it would re-home the chat onto whatever
    # foreign repo the terminal last touched.
    landed_common = _git_common_repo_root_for_cwd(resolved)
    current_common = _git_common_repo_root_for_cwd(current)
    if not landed_common or landed_common != current_common:
        return False

    session["cwd"] = resolved
    # The session works here now, so this is its workspace — a desktop chat
    # whose cwd was an unpersisted launch artifact earns a real row. The
    # settle marker keeps this adoption overridable by the NEXT settle while
    # still yielding to a user's explicit choice (see the guard above).
    session["explicit_cwd"] = True
    session["cwd_from_settle"] = True
    _register_session_cwd(session)

    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    return True


def _emit_settled_session_info(sid: str, session: dict, agent) -> None:
    """Emit end-of-turn ``session.info``, reconciling a settled cwd first.

    The turn is over, so the agent has stopped moving: this is the one moment
    where its recorded cwd is a stable answer to "where does this session
    work". Reconciling before building the payload means the same event that
    already tells the desktop the turn ended also carries the new cwd/branch —
    the client follows it with no new event type and no extra round trip.
    """
    try:
        _reconcile_session_cwd_from_terminal(session)
    except Exception:
        logger.debug("failed to reconcile settled session cwd", exc_info=True)
    _emit("session.info", sid, _session_info(agent, session))


def _session_source(session: dict | None) -> str:
    if session:
        source = str(session.get("source") or "").strip()
        if source:
            return source
    return _resolve_session_platform()


def _register_session_cwd(session: dict | None) -> None:
    if not session:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides

        cwd, cwd_source = _terminal_task_cwd_with_source(session)
        register_task_env_overrides(
            session["session_key"], {"cwd": cwd, "cwd_source": cwd_source}
        )
    except Exception:
        pass


def _ensure_session_db_row(session: dict) -> bool:
    """Idempotently persist the session's DB row on first real activity.

    Called from prompt.submit so a row only exists once the user actually sends
    a message — abandoned drafts never leave an empty "Untitled" session behind.
    Uses INSERT OR IGNORE under the hood, so re-calls (and the AIAgent's own
    lazy create) are no-ops.
    Returns False only when the store is unavailable (no openable state.db);
    prompt.submit turns that into an RPC error so a send fails loudly with a
    toast instead of streaming into a store that will never save it (#98924).
    Every other outcome — no key, best-effort attempt, success — is True.

    A cwd the user *chose* is always persisted. When they made no explicit
    choice the launch directory stands in, and whether that is meaningful
    depends on how the session was started:

    * The desktop launches from wherever the app bundle was opened (often ``/``
      or the user's home), so stamping that would file every unpicked chat under
      a folder the user never chose. Those stay null and group under "No
      workspace", which is the desired default.
    * A terminal session (``hermes`` / ``hermes --tui`` / CLI) is started from a
      directory the user deliberately ``cd``'d into — that IS the workspace, and
      it is also where the agent's terminal actually runs. Dropping it stranded
      the session with no cwd AND no git_repo_root, so the sidebar could never
      place it under its project.
    """
    key = session.get("session_key")
    if not key:
        return
    # Persist into the session's own profile db (global remote mode), not the
    # launch profile's — otherwise the row lands in the wrong state.db, the
    # unified list mis-tags it, and resume 404s ("session not found").
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            from hermes_state import get_shared_session_db
            db = get_shared_session_db(Path(profile_home) / "state.db")
        except Exception:
            logger.debug("failed to open profile db for session row", exc_info=True)
            return False
        close_db = True
    else:
        db = _get_db()
        close_db = False
    if db is None:
        # Fail loud ONLY when the store actually failed to open (#98924):
        # _db_error records the SessionDB open exception. A None db with no
        # recorded error means "no store in this context" (degraded harness,
        # store deliberately absent) — that keeps the pinned best-effort
        # contract and stays True.
        return _db_error is None
    # The session's own model/effort/fast pick — the composer override shipped on
    # session.create, or a restored /model switch — must own the row's model +
    # model_config. The agent isn't built yet at first prompt.submit, so derive
    # the row from the live override dict; fall back to the global resolved model
    # only when this chat made no explicit pick. Writing the global default here
    # used to win the INSERT-OR-IGNORE race against the agent's own correct
    # lazy-create, so a reconnect/resume rebuilt from the global model and
    # silently reverted the chat (e.g. picked gpt-5.5, reconnect snapped back to
    # the profile default). model_config carries provider/reasoning/service_tier
    # so resume restores effort + fast too, not just the model name.
    override = session.get("model_override")
    override = override if isinstance(override, dict) else {}
    row_model = str(override.get("model") or "").strip() or _resolve_model()
    model_config: dict = {}
    for src_key, cfg_key in (
        ("model", "model"),
        ("provider", "provider"),
        ("base_url", "base_url"),
        ("api_mode", "api_mode"),
    ):
        if val := override.get(src_key):
            model_config[cfg_key] = str(val)
    # The composer override may carry the RESOLVED provider "custom" for a named
    # ``providers:`` / ``custom_providers:`` entry. Persisting bare "custom" here
    # (the very first DB write for a fresh desktop session, before the agent is
    # built) is the origin of the recurring "No LLM provider configured" rows:
    # on the next resume bare "custom" routes to OpenRouter with no key. Recover
    # the durable ``custom:<name>`` identity from the override's base_url, else
    # the configured provider, so a routable identity is persisted from the
    # start (matches _runtime_model_config's normalization).
    if str(model_config.get("provider") or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or row_model or None,
            )
            if healed:
                model_config["provider"] = healed
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (db row)", exc_info=True
            )
    if (reasoning := session.get("create_reasoning_override")) is not None:
        model_config["reasoning_config"] = reasoning
    create_service_tier_override = session.get("create_service_tier_override")
    if create_service_tier_override is not None:
        # Empty string is the in-memory sentinel for an explicit normal tier:
        # it bypasses _make_agent's profile fallback without sending a bogus
        # service_tier value to the provider. Persist a durable marker so resume
        # can distinguish that choice from an omitted/inherited tier.
        model_config["service_tier"] = create_service_tier_override or "normal"
    # Branch lineage: stamp the same ``_branched_from`` marker the TUI /branch
    # uses so list_sessions_rich keeps the branch listed and the desktop sidebar
    # can nest it under its parent.
    parent_session_id = session.get("parent_session_id") or None
    if parent_session_id:
        model_config["_branched_from"] = parent_session_id
    # Bot-Mode room plumbing sessions are per-member scratch conversations
    # inside a group chat: their runtime must ALWAYS follow the member profile's
    # CURRENT config, never the provider that was pinned when the row was first
    # written. Persist that contract explicitly so resume can distinguish room
    # plumbing from a normal user chat (whose stored model/provider must be
    # restored verbatim). See _stored_session_runtime_overrides.
    if session.get("room_plumbing"):
        model_config["room_plumbing"] = True
    # Bot-Mode canonical chats (the ONE forever DM per bot) and room plumbing
    # sessions are plugin-owned scratch conversations: their runtime must ALWAYS
    # follow the member profile's CURRENT config, never the model/provider that
    # was pinned when the row was first written. Persist that contract explicitly
    # so resume can distinguish them from a normal user chat (whose stored
    # model/provider must be restored verbatim). See
    # _stored_session_runtime_overrides.
    if session.get("follow_profile_config"):
        model_config["follow_profile_config"] = True
    try:
        db.create_session(
            key,
            source=_session_source(session),
            model=row_model,
            model_config=model_config or None,
            parent_session_id=parent_session_id,
            cwd=_persisted_session_cwd(session),
            # Self-describing rows: aggregators that merge multiple profile DBs
            # into one list can't rely on which file a row came from alone.
            # Stamp the launch profile explicitly instead of leaving NULL —
            # NULL is exactly what the #94724 legacy-owner backfill exists to
            # repair, and rows minted AFTER that one-shot backfill ran stayed
            # NULL forever: profile-keyed matching then drops them from the
            # sidebar and deep links can't resolve them (#99222).
            profile_name=(
                Path(profile_home).name if profile_home else _current_profile_name()
            ),
        )
        # A session can be born hidden (session.create hidden=true, or a
        # session.set_hidden that arrived before the row existed): apply the
        # deferred intent now that the row exists, mirroring pending_title.
        if session.get("pending_hidden"):
            try:
                db.set_session_hidden(key, True)
            except Exception:
                logger.debug("failed to apply pending hidden flag", exc_info=True)
    except Exception as exc:
        # Disk-full is not a soft failure: if we swallow it here, prompt.submit
        # returns {"status":"streaming"} and the user's message vanishes with
        # no toast. Re-raise so the submit handler can return a real RPC error.
        from hermes_state import is_disk_full_error

        if is_disk_full_error(exc):
            raise
        logger.debug("failed to persist desktop session row", exc_info=True)
    finally:
        if close_db:
            try:
                from hermes_state import release_or_close
                release_or_close(db)
            except Exception:
                pass
    return True


def _persist_branch_seed(session: dict) -> None:
    """First-turn persist of a branch's copied transcript.

    A branch is a draft until its first submit: the parent's messages live only
    in ``session["history"]`` (they ride into the agent as ``conversation_history``,
    which ``_flush_messages_to_session_db`` skips by identity). Without this the
    branch row would resume missing its pre-branch context. Runs once; the row +
    parent link are written by ``_ensure_session_db_row`` just before this.
    """
    if not session.get("parent_session_id") or session.get("_branch_seed_persisted"):
        return
    key = session.get("session_key")
    if not key:
        return
    with session["history_lock"]:
        seed = [dict(msg) for msg in (session.get("history") or [])]
    if not seed:
        return
    with _session_db(session) as db:
        if db is None:
            return
        try:
            # Bounded-chunk transactions (see #23254): a branch seed can be
            # hundreds of rows; chunking keeps each BEGIN IMMEDIATE short so
            # concurrent writers aren't starved. Recovery semantics match the
            # old per-row loop (mid-copy failure leaves a partial seed with
            # _branch_seed_persisted unset).
            db.append_messages_batch(
                key,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        "reasoning": msg.get("reasoning"),
                        "reasoning_content": msg.get("reasoning_content"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "codex_reasoning_items": msg.get("codex_reasoning_items"),
                        "codex_message_items": msg.get("codex_message_items"),
                        # Timeline markers (model_switch, personality_switch,
                        # auto_continue, …) ride as role=user; dropping the tag
                        # here re-planted them as bare user turns after a
                        # restart, corrupting the truncate ordinal address
                        # space the same way #82756 did.
                        "display_kind": msg.get("display_kind"),
                        "display_metadata": msg.get("display_metadata"),
                        # Preserve the parent's original message timestamps —
                        # append_message would otherwise stamp time.time() and the
                        # branch's copied history would all appear authored "now".
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in seed
                ],
                chunk_rows=500,
            )
            session["_branch_seed_persisted"] = True
        except Exception as exc:
            from hermes_state import is_disk_full_error

            if is_disk_full_error(exc):
                raise
            logger.debug("branch seed persist failed", exc_info=True)


@contextlib.contextmanager
def _session_db(session: dict):
    """Yield the SessionDB that owns this session's row (profile-aware).

    Mirrors :func:`_ensure_session_db_row`: a remote/profile session persists
    into its own profile's ``state.db`` (a fresh handle we close on exit);
    everything else borrows the shared ``_get_db()`` handle (left open). Yields
    None when the db is unavailable.
    """
    db, close_db = None, False
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            from hermes_state import get_shared_session_db
            db, close_db = get_shared_session_db(Path(profile_home) / "state.db"), True
        except Exception:
            logger.debug("failed to open profile db for session", exc_info=True)
    else:
        db = _get_db()
    try:
        yield db
    finally:
        if close_db and db is not None:
            with contextlib.suppress(Exception):
                from hermes_state import release_or_close
                release_or_close(db)


def _rewind_active_session_history(
    session: dict,
    user_ordinal: int,
    *,
    require_retryable: bool = False,
) -> tuple[list[dict], dict, int]:
    """Rewind one canonical user turn while retaining carrier scaffolding.

    The caller holds ``history_lock``.  Persistent sessions archive the target
    and tail; a composite carrier's own hidden handoff is inserted in that same
    transaction.  Memory is installed only after the durable commit and is
    built from the already-validated prefix plus the returned scaffold row id,
    so there is no fallible post-commit reload.
    """
    from agent.context_compressor import (
        history_before_user_originated_turn,
        retryable_user_text,
        split_user_originated_turn,
        user_originated_turn_view,
    )
    from agent.memory_manager import sanitize_context
    from agent.tool_dispatch_helpers import (
        _is_multimodal_tool_result,
        _multimodal_text_summary,
    )

    def _comparison_content(message: dict) -> Any:
        content = message.get("content")
        if _is_multimodal_tool_result(content):
            content = _multimodal_text_summary(content)
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif (
                    isinstance(part, dict)
                    and part.get("type") in {"image", "image_url", "input_image"}
                ):
                    text_parts.append("[screenshot]")
            content = "\n".join(text_parts) if text_parts else None
        if message.get("role") in {"user", "assistant"} and isinstance(content, str):
            return sanitize_context(content).strip()
        return content

    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    user_indices = [
        index
        for index, message in enumerate(history)
        if user_originated_turn_view(message) is not None
    ]
    if user_ordinal < 0 or user_ordinal >= len(user_indices):
        raise ValueError("target user message is no longer in session history")
    target_index = user_indices[user_ordinal]
    installed, live_view = history_before_user_originated_turn(history, target_index)
    rewound_count = len(history) - target_index

    session_key = str(session.get("session_key") or "").strip()
    persisted = False
    if session_key:
        with _session_db(session) as db:
            if db is None:
                raise RuntimeError("session database is unavailable")
            expected_active_ids = db.get_active_message_ids(session_key)
            durable = db.get_messages_as_conversation(
                session_key,
                include_row_ids=True,
            )
            durable_user_indices = [
                index
                for index, message in enumerate(durable)
                if user_originated_turn_view(message) is not None
            ]
            if len(durable_user_indices) != len(user_indices):
                raise RuntimeError(
                    "session history changed before the rewind could be persisted"
                )
            durable_target_index = durable_user_indices[user_ordinal]
            durable_target = durable[durable_target_index]
            durable_prefix, durable_live_view = history_before_user_originated_turn(
                durable, durable_target_index
            )
            if _comparison_content(durable_live_view) != _comparison_content(live_view):
                raise RuntimeError(
                    "session history changed before the rewind could be persisted"
                )
            target_row_id = durable_target.get("_row_id")
            if not isinstance(target_row_id, int):
                raise RuntimeError("rewind target has no durable row identity")
            if require_retryable:
                retryable_user_text(durable_live_view.get("content"))
            scaffold, _ = split_user_originated_turn(durable_target)
            result = db.rewind_to_message(
                session_key,
                target_row_id,
                preserve_compaction_handoff=scaffold is not None,
                expected_active_ids=expected_active_ids,
                expected_target_content=durable_live_view.get("content"),
            )
            if scaffold is not None:
                replacement_id = result.get("replacement_message_id")
                if not isinstance(replacement_id, int):
                    raise RuntimeError(
                        "rewind commit did not return the replacement scaffold id"
                    )
                durable_prefix[-1]["_row_id"] = replacement_id
                durable_prefix[-1]["_db_persisted"] = True
                installed[-1] = durable_prefix[-1]
            # Current clients address destructive follow-ups by durable row id.
            # Preserve the richer warm content (for example image parts), but
            # copy row identities when the retained warm/durable shapes align.
            if len(installed) == len(durable_prefix) and all(
                warm.get("role") == durable_message.get("role")
                and bool(warm.get("display_kind"))
                == bool(durable_message.get("display_kind"))
                and _comparison_content(warm)
                == _comparison_content(durable_message)
                for warm, durable_message in zip(installed, durable_prefix)
            ):
                for warm, durable_message in zip(installed, durable_prefix):
                    row_id = durable_message.get("_row_id")
                    if isinstance(row_id, int):
                        warm["_row_id"] = row_id
            live_view = durable_live_view
            rewound_count = int(result.get("rewound_count", 0))
            persisted = True
    elif require_retryable:
        retryable_user_text(live_view.get("content"))

    installed = [message.copy() for message in installed]
    session["history"] = installed
    session["history_version"] = int(session.get("history_version", 0)) + 1
    agent = session.get("agent")
    if agent is not None:
        agent._session_messages = installed
        if hasattr(agent, "_last_flushed_db_idx"):
            agent._last_flushed_db_idx = len(installed) if persisted else 0
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = installed[:] if persisted else None
    return installed, live_view, rewound_count


def _history_without_ephemeral_scaffolding(history: list[dict]) -> list[dict]:
    """Return the durable transcript shape without transient recovery rows."""
    from run_agent import _is_ephemeral_scaffolding

    return [
        message.copy()
        for message in history
        if not _is_ephemeral_scaffolding(message)
    ]


def _persist_session_git_meta(session: dict, cwd: str, generation: int) -> None:
    """Resolve + persist a session's git branch / repo root WITHOUT blocking.

    Branch and root come from ``git`` subprocess probes; running them inline on
    the session-init / cwd-set path would stall startup whenever ``cwd`` is slow
    or on an unreachable mount. Run them on a short-lived daemon thread instead
    and persist via the same profile-aware db the caller writes ``cwd`` to.

    Best-effort: ``cwd`` itself is persisted synchronously by the caller, so a
    probe failure just leaves these enrichment columns unset (the project tree
    falls back to its live resolver / lazy backfill). Daemon, so a mid-flight
    probe never delays gateway shutdown.
    """
    session_key = session.get("session_key", "")
    if (
        not session_key
        or not cwd
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        return
    # Snapshot the routing fields now; the live session dict may be gone by the
    # time the thread runs. `_session_db` reopens the profile-correct db inside.
    db_session = {"session_key": session_key, "profile_home": session.get("profile_home")}

    def _run() -> None:
        try:
            branch = _git_branch_for_cwd(cwd)
            root = _git_common_repo_root_for_cwd(cwd)
            if not (branch or root):
                return
            with _session_db(db_session) as db:
                if db is not None:
                    db.publish_session_git_metadata(
                        session_key,
                        cwd,
                        generation,
                        branch,
                        root,
                    )
        except Exception:
            logger.debug("failed to persist session git metadata", exc_info=True)

    threading.Thread(target=_run, name="git-meta", daemon=True).start()


def _persist_session_cwd_and_schedule_git_meta(
    session: dict,
    cwd: str,
    *,
    db=None,
) -> int | None:
    """Claim a DB-backed probe generation, then start Git enrichment."""
    try:
        if db is not None:
            generation = db.update_session_cwd(
                session.get("session_key", ""), cwd
            )
        else:
            with _session_db(session) as owner_db:
                if owner_db is None:
                    return None
                generation = owner_db.update_session_cwd(
                    session.get("session_key", ""), cwd
                )
    except Exception:
        logger.debug("failed to persist session cwd", exc_info=True)
        return None

    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        return None
    _persist_session_git_meta(session, cwd, generation)
    return generation


def _set_session_cwd(session: dict, cwd: str) -> str:
    from hermes_constants import translate_cwd_for_wsl_backend

    cwd = translate_cwd_for_wsl_backend(str(cwd))
    resolved = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    session["cwd"] = resolved
    # An explicit user choice — persist it as the workspace (and let a later
    # lazy row creation persist it too, not the launch-dir fallback).
    session["explicit_cwd"] = True
    # A user's choice supersedes any earlier settle-adopted cwd: from here on
    # the terminal wandering must not move the workspace again.
    session["cwd_from_settle"] = False
    _register_session_cwd(session)
    # The synchronous DB write claims ordering authority; Git subprocesses stay
    # off the hot path and may publish only for that exact generation.
    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    try:
        from tools.terminal_tool import cleanup_vm

        cleanup_vm(session["session_key"])
    except Exception:
        pass
    return resolved


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
