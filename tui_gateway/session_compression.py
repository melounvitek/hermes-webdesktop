"""Live compression: config hot-reload onto a running agent, pending model switch apply, /compress (CompressionLockHeld when a turn holds the lock), session-key sync after compress.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _tui_compression_config_signature(cfg: dict | None) -> tuple:
    """Stable snapshot of compression/context keys that must apply next turn.

    Reuses the messaging-gateway cache-busting extract so Desktop/TUI and
    messaging stay on the same key set. Adds ``idle_compact_after_seconds``
    and ``tail_mode``, which affect live TUI sessions but are not in the
    gateway tuple today.
    """
    from gateway.run import GatewayRunner

    keys = GatewayRunner._extract_cache_busting_config(cfg)
    picked = {
        key: value
        for key, value in keys.items()
        if key.startswith("compression.") or key == "model.context_length"
    }
    compression = cfg.get("compression") if isinstance(cfg, dict) and isinstance(cfg.get("compression"), dict) else {}
    for extra in ("idle_compact_after_seconds", "tail_mode"):
        picked[f"compression.{extra}"] = compression.get(extra)
    return tuple(sorted(picked.items()))


def _compressor_ctor_default(name: str, fallback: Any) -> Any:
    """Read a normalized default from ContextCompressor's REAL signature.

    Unset restoration must go through the same derivation the construction
    path uses (#94724 review finding on #95980) — pulling the default off
    ``ContextCompressor.__init__`` itself instead of hardcoding copies keeps
    the two from drifting.
    """
    try:
        import inspect

        from agent.context_compressor import ContextCompressor

        default = inspect.signature(ContextCompressor.__init__).parameters[
            name
        ].default
        if default is inspect.Parameter.empty:
            return fallback
        return default
    except Exception:
        return fallback


def _derived_default_threshold_percent(agent: Any, compression: dict) -> float:
    """Default compaction threshold when ``compression.threshold`` is unset.

    Mirrors agent_init exactly: the ctor's global default, then the per-model
    resolution (Codex gpt-5.4/5.5 + spark autoraise, Arcee Trinity, etc.)
    via the SAME ``_resolve_compression_threshold`` helper — so removing the
    key restores the model-derived value, not a bare constant.
    """
    try:
        pct = float(_compressor_ctor_default("threshold_percent", 0.50))
    except (TypeError, ValueError):
        pct = 0.50
    try:
        from agent.agent_init import _resolve_compression_threshold
        from agent.auxiliary_client import (
            _compression_threshold_for_model,
            _is_codex_gpt54_or_gpt55,
            _is_codex_spark,
        )

        model = getattr(agent, "model", "") or ""
        provider = getattr(agent, "provider", "") or ""
        autoraise_enabled = str(
            compression.get("codex_gpt55_autoraise", True)
        ).lower() in {"true", "1", "yes"}
        model_cthresh = _compression_threshold_for_model(
            model,
            provider,
            allow_codex_gpt55_autoraise=autoraise_enabled,
        )
        pct, _notice = _resolve_compression_threshold(
            pct,
            model_cthresh,
            model=model,
            is_codex_autoraise=(
                _is_codex_gpt54_or_gpt55(model, provider)
                or _is_codex_spark(model, provider)
            ),
        )
    except Exception:
        pass
    return pct


def _apply_live_compression_config(agent: Any, cfg: dict | None) -> None:
    """Update a live session's compressor from current config.yaml.

    Preserves the agent object, session identity, history, and callbacks.
    Recomputes the trigger from the ratio-based threshold and then applies
    ``compression.threshold_tokens`` so raising, lowering, or clearing the
    cap all take effect on the next preflight.

    Every adopted key has UNSET semantics (#94724 review finding on the
    merged #95980): removing a key from config.yaml restores the normalized
    default — or the model-derived value — on the next turn, through the
    same derivation the construction path uses (ContextCompressor ctor
    defaults read off its real signature, the Codex threshold autoraise via
    ``_resolve_compression_threshold``, context-length re-inference via the
    deferred ``get_model_context_length`` resolution). The old behavior
    acted only on PRESENT keys, leaving stale values active forever.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    compression_raw = cfg.get("compression")
    compression = compression_raw if isinstance(compression_raw, dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}

    enabled_raw = compression.get("enabled", True)
    if isinstance(enabled_raw, bool):
        agent.compression_enabled = enabled_raw
    else:
        agent.compression_enabled = str(enabled_raw).lower() in {"true", "1", "yes"}

    agent.codex_responses_native_compaction = is_truthy_value(
        compression.get("codex_responses_native", False)
    )
    native_threshold_raw = compression.get(
        "codex_responses_compact_threshold", 200_000
    )
    try:
        if isinstance(native_threshold_raw, bool):
            raise ValueError
        native_threshold = int(native_threshold_raw)
        if native_threshold <= 0:
            raise ValueError
    except (TypeError, ValueError):
        logger.warning(
            "Invalid compression.codex_responses_compact_threshold=%r; "
            "using 200000.",
            native_threshold_raw,
        )
        native_threshold = 200_000
    agent.codex_responses_compact_threshold = native_threshold

    # Absence restores the agent_init/config default (0 = disabled).
    idle_raw = compression.get("idle_compact_after_seconds", 0)
    try:
        agent.compression_idle_compact_after_seconds = max(0, int(idle_raw or 0))
    except (TypeError, ValueError):
        pass

    cc = getattr(agent, "context_compressor", None)
    if cc is None:
        return

    # tail_mode: ctor normalization — unknown/absent values land on "lean",
    # matching agent_init's default and the compressor's own fallback.
    default_tail = str(_compressor_ctor_default("tail_mode", "lean"))
    mode = str(compression.get("tail_mode", default_tail) or default_tail)
    mode = mode.strip().lower()
    cc.tail_mode = mode if mode in ("legacy", "lean") else default_tail

    def _assign_int(key: str, attr: str, default: int, min_value: int = 0) -> None:
        raw = compression.get(key, default)
        try:
            value = default if raw is None else int(raw)
        except (TypeError, ValueError):
            return
        setattr(cc, attr, max(min_value, value))

    _assign_int(
        "proactive_prune_tokens",
        "proactive_prune_tokens",
        int(_compressor_ctor_default("proactive_prune_tokens", 0)),
    )
    _assign_int(
        "proactive_prune_min_result_chars",
        "proactive_prune_min_result_chars",
        int(_compressor_ctor_default("proactive_prune_min_result_chars", 8000)),
    )
    _assign_int(
        "proactive_prune_min_reclaim_tokens",
        "proactive_prune_min_reclaim_tokens",
        int(_compressor_ctor_default("proactive_prune_min_reclaim_tokens", 4096)),
    )
    _assign_int(
        "protect_last_n",
        "protect_last_n",
        int(_compressor_ctor_default("protect_last_n", 20)),
    )
    _assign_int(
        "min_tail_user_messages",
        "min_tail_user_messages",
        int(_compressor_ctor_default("min_tail_user_messages", 1)),
        min_value=1,
    )

    try:
        ratio_raw = compression.get(
            "target_ratio", _compressor_ctor_default("summary_target_ratio", 0.20)
        )
        cc.summary_target_ratio = max(0.10, min(float(ratio_raw), 0.80))
    except (TypeError, ValueError):
        pass

    raw_thresholds = compression.get("model_thresholds")
    if isinstance(raw_thresholds, dict):
        cc.model_thresholds = {
            str(k): float(v)
            for k, v in raw_thresholds.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    else:
        # Absent (or invalid shape — agent_init treats both as empty):
        # stale per-model overrides must stop steering the live threshold.
        cc.model_thresholds = {}

    # threshold: present value wins; absence derives the default through
    # the same agent_init resolution (global default + per-model autoraise).
    pct: float | None = None
    if "threshold" in compression:
        try:
            pct = float(compression["threshold"])
        except (TypeError, ValueError):
            pct = None
    if pct is None:
        pct = _derived_default_threshold_percent(agent, compression)
    try:
        cc._config_threshold_percent = pct
        cc._configured_threshold_percent = pct
        base = pct
        model_thresholds = getattr(cc, "model_thresholds", None) or {}
        if model_thresholds:
            from agent.context_compressor import resolve_model_threshold

            base = resolve_model_threshold(
                getattr(agent, "model", "") or "",
                model_thresholds,
                pct,
            )
        cc._base_threshold_percent = base
        if hasattr(cc, "_effective_threshold_percent"):
            try:
                cc.threshold_percent = cc._effective_threshold_percent(
                    cc.context_length, base
                )
            except Exception:
                cc.threshold_percent = pct
        else:
            cc.threshold_percent = pct
    except (TypeError, ValueError):
        pass

    raw_ctx = model_cfg.get("context_length")
    if raw_ctx is not None:
        try:
            new_ctx = int(raw_ctx)
        except (TypeError, ValueError):
            new_ctx = 0
        if new_ctx > 0:
            cc._config_context_length = new_ctx
            try:
                cc.context_length = new_ctx
            except Exception:
                pass
    elif getattr(cc, "_config_context_length", None) is not None:
        # model.context_length removed: drop the config override and force
        # re-inference from model metadata on next access — the same
        # deferred get_model_context_length resolution agent construction
        # uses (#32221). The re-resolve also re-applies the small-context
        # threshold floor for the genuinely re-inferred window.
        cc._config_context_length = None
        cc._resolved_context_length = None

    coerce_cap = getattr(cc, "_coerce_threshold_tokens_cap", None)
    if callable(coerce_cap):
        cc.threshold_tokens_cap = coerce_cap(compression.get("threshold_tokens"))
    elif "threshold_tokens" in compression:
        try:
            cap = int(compression.get("threshold_tokens"))
            cc.threshold_tokens_cap = cap if cap > 0 else None
        except (TypeError, ValueError):
            cc.threshold_tokens_cap = None
    else:
        cc.threshold_tokens_cap = None

    # Invalidate cached trigger so the next preflight re-derives from the
    # current percent/window and then applies the (possibly new) cap.
    if hasattr(cc, "_threshold_tokens"):
        cc._threshold_tokens = None
        if hasattr(cc, "_tail_token_budget"):
            cc._tail_token_budget = None
    elif hasattr(cc, "_apply_threshold_tokens_cap"):
        compute = getattr(cc, "_compute_threshold_tokens", None)
        if callable(compute):
            cc.threshold_tokens = compute(
                getattr(cc, "context_length", 0) or 0,
                getattr(cc, "threshold_percent", 0.5),
                getattr(cc, "max_tokens", None),
            )
        cc._apply_threshold_tokens_cap()
    else:
        cap = getattr(cc, "threshold_tokens_cap", None)
        current = getattr(cc, "threshold_tokens", None)
        if cap and current:
            cc.threshold_tokens = min(int(current), int(cap))


def _sync_agent_compression_with_config(sid: str, session: dict) -> None:
    """Adopt compression.* / model.context_length edits at turn start.

    Messaging gateways already rebuild a cached agent when these keys change.
    Desktop/TUI only synced the model; the live compressor kept the threshold
    captured at agent creation (#95151).
    """
    agent = session.get("agent")
    if agent is None:
        return
    cfg = _load_cfg() or {}
    signature = _tui_compression_config_signature(cfg)
    seen = session.get("config_compression_seen")
    session["config_compression_seen"] = signature
    if signature == seen:
        return
    try:
        _apply_live_compression_config(agent, cfg)
    except Exception as e:
        logger.warning(
            "Could not apply live compression config for %s: %s", sid, e
        )


def _apply_pending_model_switch(sid: str, session: dict) -> None:
    """Apply a model switch queued while a turn was running.

    ``config.set model`` on a busy session doesn't mutate the live agent (the
    worker thread is reading model/client mid-request); it stashes the pick in
    ``session["pending_model_switch"]``.  This runs on the TURN thread at turn
    start — before the first model call, nothing in flight — so the in-place
    swap (client rebuild, the slow part) is safe here.  A failed switch keeps
    the current model and never blocks the turn, matching
    ``_sync_agent_model_with_config``.
    """
    pending = session.pop("pending_model_switch", None)
    if not pending or session.get("agent") is None:
        return
    try:
        result = _apply_model_switch(
            sid,
            session,
            pending["raw"],
            confirm_expensive_model=bool(pending.get("confirm_expensive_model")),
        )
        # A queued pick is a deliberate user action; honour the expensive-model
        # confirm by NOT applying it silently — surface the warning and drop the
        # switch rather than spend on a pricey model the user never confirmed.
        if result.get("confirm_required"):
            _emit(
                "error",
                sid,
                {"message": result.get("confirm_message") or result.get("warning") or ""},
            )
    except Exception as e:
        _emit(
            "error",
            sid,
            {"message": f"Could not switch model: {e}"},
        )


class CompressionLockHeld(Exception):
    """Raised by _compress_session_history when compression skipped due
    to a concurrent lock on the session's compression_locks row."""
    def __init__(self, holder: str | None = None):
        self.holder = holder
        super().__init__(f"Compression lock held: {holder or 'unknown'}")


def _compress_session_history(
    session: dict,
    focus_topic: str | None = None,
    approx_tokens: int | None = None,
    before_messages: list | None = None,
    history_version: int | None = None,
) -> tuple[int, dict]:
    """Compress a session's history — the single choke point shared by all
    three manual-compress routes (session.compress RPC, command.dispatch
    /compress|/compact, and the slash-exec mirror).

    ``focus_topic`` is the RAW argument string after ``/compress``. It is
    parsed here with :func:`parse_partial_compress_args` so boundary-aware
    forms (``here [N]``, ``up to here``, ``--keep N``) trigger a partial
    compress — head summarized, most recent ``keep_last`` exchanges kept
    verbatim — on EVERY route, mirroring cli.py's ``_manual_compress`` and
    gateway/slash_commands.py (PR #35252). Parsing at the choke point (not
    per-route) is what fixes #35533: previously "/compress here 3" reached
    this helper unparsed and ran a FULL compress focused on the literal
    text "here 3".
    """
    from agent.conversation_compression import (
        finalize_context_engine_compression_notification,
    )
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import (
        parse_partial_compress_args,
        rejoin_compressed_head_and_tail,
        split_history_for_partial_compress,
    )

    agent = session["agent"]
    # Snapshot history under the lock so the LLM-bound compression call
    # below does NOT hold history_lock for the duration of the request —
    # otherwise other handlers acquiring the lock (prompt.submit etc.)
    # block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
    history = before_messages
    if len(history) < 4:
        usage = _get_usage(agent)
        return 0, usage
    partial, keep_last, focus_topic = parse_partial_compress_args(focus_topic or "")
    # Boundary-aware split: only the head is summarized; the most recent
    # `keep_last` exchanges ride along verbatim. A degenerate split (empty
    # tail — everything would be kept, or no head left to compress) falls
    # back to full compression so the user still gets an action.
    tail: list = []
    head = history
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        if not tail:
            partial = False
            head = history
    if approx_tokens is None:
        # Include system prompt + tool schemas so the figure reflects real
        # request pressure, not a transcript-only underestimate (#6217).
        _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
        _tools = getattr(agent, "tools", None) or None
        approx_tokens = estimate_request_tokens_rough(
            history, system_prompt=_sys_prompt, tools=_tools
        )
    # Pass system_message=None so AIAgent._compress_context rebuilds the
    # system prompt cleanly via _build_system_prompt(None). Passing the
    # cached prompt (which already contains the agent identity block)
    # makes the rebuild append the identity a second time. Mirrors the
    # CLI's _manual_compress fix for issue #15281.
    # force=True: every caller of this helper is a manual /compress path
    # (session.compress RPC, slash compress/compact, slash-worker mirror) —
    # auto-compaction runs inside the agent loop, not here. Manual
    # compaction bypasses the summary-failure cooldown, matching the CLI
    # and gateway handlers.
    try:
        compressed, _ = agent._compress_context(
            head,
            None,
            approx_tokens=approx_tokens,
            # Partial compress has no focus topic (the modes are exclusive;
            # parse_partial_compress_args returns focus_topic=None for the
            # boundary-aware forms).
            focus_topic=focus_topic or None,
            force=True,
            defer_context_engine_notification=True,
        )
    except Exception:
        finalize_context_engine_compression_notification(
            agent,
            committed=False,
        )
        raise
    # If _compress_context returned unchanged because a concurrent
    # compression lock is held, raise so callers can surface a clear
    # message instead of the misleading "No changes from compression" text.
    # Type-pinned (is True / str): real values are None/True/holder-string;
    # bare truthiness is fooled by MagicMock auto-attrs on test doubles.
    _lock_skipped = getattr(agent, "_compression_skipped_due_to_lock", None)
    if _lock_skipped is True or isinstance(_lock_skipped, str):
        agent._compression_skipped_due_to_lock = None
        # No boundary was committed on a lock-skip; discard any pending
        # deferred context-engine notification (exactly-once, no-op safe).
        finalize_context_engine_compression_notification(
            agent,
            committed=False,
        )
        raise CompressionLockHeld(
            _lock_skipped if isinstance(_lock_skipped, str) else None
        )

    if partial and tail:
        compressed = rejoin_compressed_head_and_tail(compressed, tail)
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the compressed
            # result so we don't clobber concurrent edits.
            finalize_context_engine_compression_notification(
                agent,
                committed=False,
            )
            usage = _get_usage(agent)
            return 0, usage
        session["history"] = compressed
        session["history_version"] = history_version + 1
    usage = _get_usage(agent)
    return len(history) - len(compressed), usage


def _sync_session_key_after_compress(
    sid: str,
    session: dict,
    *,
    clear_pending_title: bool = True,
    restart_slash_worker: bool = True,
) -> None:
    """Re-anchor session_key when AIAgent._compress_context rotates session_id.

    AIAgent._compress_context ends the current SessionDB session and creates
    a new continuation session, rotating ``agent.session_id``.  The TUI
    gateway keeps the gateway-side ``session_key`` separate (used for
    approval routing, slash worker init, DB title/history lookups, yolo
    state).  Without this sync, those operations would target the ended
    parent session while the agent writes to the new continuation session.

    Policy flags:
        clear_pending_title: True for manual /compress (title belongs to old
            session). False for post-turn auto-compression (preserve user
            intent so pending_title can be applied to the continuation).
        restart_slash_worker: True for manual /compress and post-turn
            auto-compression (worker holds stale session key). False only
            if the caller manages the worker lifecycle separately.
    """
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    lease_reanchored = _transfer_active_session_slot(
        sid,
        session,
        new_session_id=new_session_id,
    )
    if not lease_reanchored:
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid,
            old_key,
            new_session_id,
        )

    try:
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
            register_gateway_notify,
            unregister_gateway_notify,
        )

        try:
            unregister_gateway_notify(old_key)
        except Exception:
            pass
        session["session_key"] = new_session_id
        try:
            yolo_was_on = is_session_yolo_enabled(old_key)
        except Exception:
            yolo_was_on = False
        if yolo_was_on:
            try:
                enable_session_yolo(new_session_id)
                disable_session_yolo(old_key)
            except Exception:
                pass
        try:
            register_gateway_notify(
                new_session_id,
                lambda data: _emit_approval_request(sid, data),
            )
        except Exception:
            pass
    except Exception:
        # Even if the approval module fails to import, still anchor the
        # session_key on the new continuation id so downstream lookups
        # don't keep targeting the ended row.
        session["session_key"] = new_session_id

    # #84417 (belt): invalidate any in-flight ``_drain_queued_prompt`` claim
    # that captured generation under the pre-rotation session_key. A raced
    # drain must not dispatch on the continuation with a stale claim; the
    # claimed envelope is restored to the queue (see ``_drain_queued_prompt``)
    # so legitimate follow-ups still survive. Complements self-duplicate
    # scrubbing on redirect.
    session["_queued_prompt_generation"] = int(
        session.get("_queued_prompt_generation", 0)
    ) + 1

    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        try:
            _restart_slash_worker(sid, session)
        except Exception:
            pass


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
