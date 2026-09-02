"""Live compression: config hot-reload onto a running agent, pending model switch apply, /compress
(CompressionLockHeld when a turn holds the lock), session-key sync after compress.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations


import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _tui_compression_config_signature(cfg: dict | None) -> tuple:
    """Stable snapshot of compression/context keys that must apply next turn: the messaging-gateway
    cache-busting extract (same key set as messaging) plus ``idle_compact_after_seconds`` and
    ``tail_mode`` (affect live TUI sessions, not in the gateway tuple)."""
    from gateway.run import GatewayRunner

    keys = GatewayRunner._extract_cache_busting_config(cfg)
    picked = {k: v for k, v in keys.items() if k.startswith("compression.") or k == "model.context_length"}
    compression = cfg.get("compression") if isinstance(cfg, dict) and isinstance(cfg.get("compression"), dict) else {}
    for extra in ("idle_compact_after_seconds", "tail_mode"):
        picked[f"compression.{extra}"] = compression.get(extra)
    return tuple(sorted(picked.items()))


def _compressor_ctor_default(name: str, fallback: Any) -> Any:
    """Default read off ContextCompressor.__init__'s REAL signature, so unset-key restoration uses the
    construction path's derivation instead of a hardcoded copy that could drift."""
    try:
        import inspect

        from agent.context_compressor import ContextCompressor

        default = inspect.signature(ContextCompressor.__init__).parameters[name].default
        return fallback if default is inspect.Parameter.empty else default
    except Exception:
        return fallback


def _derived_default_threshold_percent(agent: Any, compression: dict) -> float:
    """Default compaction threshold when ``compression.threshold`` is unset. Mirrors agent_init: ctor
    global default, then per-model resolution (Codex autoraise etc.) via the SAME
    ``_resolve_compression_threshold`` — removing the key restores the model-derived value."""
    try:
        pct = float(_compressor_ctor_default("threshold_percent", 0.50))
    except (TypeError, ValueError):
        pct = 0.50
    try:
        from agent.agent_init import _resolve_compression_threshold
        from agent.auxiliary_client import _compression_threshold_for_model, _is_codex_gpt54_or_gpt55, _is_codex_spark

        model = getattr(agent, "model", "") or ""
        provider = getattr(agent, "provider", "") or ""
        autoraise_enabled = str(compression.get("codex_gpt55_autoraise", True)).lower() in {"true", "1", "yes"}
        model_cthresh = _compression_threshold_for_model(
            model, provider, allow_codex_gpt55_autoraise=autoraise_enabled
        )
        pct, _notice = _resolve_compression_threshold(
            pct, model_cthresh, model=model,
            is_codex_autoraise=_is_codex_gpt54_or_gpt55(model, provider) or _is_codex_spark(model, provider),
        )
    except Exception:
        pass
    return pct


# (config key == compressor attr, ctor-default fallback, min_value)
_COMPRESSION_INT_KEYS = (
    ("proactive_prune_tokens", 0, 0),
    ("proactive_prune_min_result_chars", 8000, 0),
    ("proactive_prune_min_reclaim_tokens", 4096, 0),
    ("protect_last_n", 20, 0),
    ("min_tail_user_messages", 1, 1),
)


def _apply_live_compression_config(agent: Any, cfg: dict | None) -> None:
    """Update a live session's compressor from current config.yaml, preserving the agent object,
    session identity, history and callbacks. Recomputes the trigger from the ratio threshold, then
    applies ``compression.threshold_tokens`` so raising/lowering/clearing the cap lands next preflight.

    Every adopted key has UNSET semantics: a removed key restores the normalized default (or the
    model-derived value) through the construction path's own derivation (ctor signature defaults,
    Codex autoraise, deferred context-length re-inference). Acting only on PRESENT keys would leave
    stale values active forever.
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

    agent.codex_responses_native_compaction = is_truthy_value(compression.get("codex_responses_native", False))
    native_threshold_raw = compression.get("codex_responses_compact_threshold", 200_000)
    try:
        if isinstance(native_threshold_raw, bool):
            raise ValueError
        native_threshold = int(native_threshold_raw)
        if native_threshold <= 0:
            raise ValueError
    except (TypeError, ValueError):
        logger.warning("Invalid compression.codex_responses_compact_threshold=%r; using 200000.", native_threshold_raw)
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

    # tail_mode: unknown/absent values land on the ctor default ("lean"), matching agent_init.
    default_tail = str(_compressor_ctor_default("tail_mode", "lean"))
    mode = str(compression.get("tail_mode", default_tail) or default_tail).strip().lower()
    cc.tail_mode = mode if mode in ("legacy", "lean") else default_tail

    for key, fallback, min_value in _COMPRESSION_INT_KEYS:
        default = int(_compressor_ctor_default(key, fallback))
        raw = compression.get(key, default)
        try:
            value = default if raw is None else int(raw)
        except (TypeError, ValueError):
            continue
        setattr(cc, key, max(min_value, value))

    try:
        ratio_raw = compression.get("target_ratio", _compressor_ctor_default("summary_target_ratio", 0.20))
        cc.summary_target_ratio = max(0.10, min(float(ratio_raw), 0.80))
    except (TypeError, ValueError):
        pass

    raw_thresholds = compression.get("model_thresholds")
    if isinstance(raw_thresholds, dict):
        cc.model_thresholds = {
            str(k): float(v)
            for k, v in raw_thresholds.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    else:
        # Absent or invalid shape (agent_init treats both as empty): stale overrides must stop steering.
        cc.model_thresholds = {}

    # threshold: present value wins; absence derives via the agent_init resolution (default + autoraise).
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

            base = resolve_model_threshold(getattr(agent, "model", "") or "", model_thresholds, pct)
        cc._base_threshold_percent = base
        if hasattr(cc, "_effective_threshold_percent"):
            try:
                cc.threshold_percent = cc._effective_threshold_percent(cc.context_length, base)
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
        # model.context_length removed: drop the override and force re-inference from model metadata
        # on next access (construction's deferred resolution); re-applies the small-context floor too.
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

    # Invalidate the cached trigger so the next preflight re-derives from percent/window, then the cap.
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
    """Adopt compression.* / model.context_length edits at turn start (messaging gateways rebuild the
    agent on these keys; Desktop/TUI keeps the live compressor, so it must be updated in place)."""
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
        logger.warning("Could not apply live compression config for %s: %s", sid, e)


def _apply_pending_model_switch(sid: str, session: dict) -> None:
    """Apply a model switch queued (``session["pending_model_switch"]``) while a turn was running.

    Runs on the TURN thread at turn start — nothing in flight — so the in-place swap (client rebuild)
    is safe. A failed switch keeps the current model and never blocks the turn, matching
    ``_sync_agent_model_with_config``.
    """
    pending = session.pop("pending_model_switch", None)
    if not pending or session.get("agent") is None:
        return
    try:
        result = _apply_model_switch(
            sid, session, pending["raw"], confirm_expensive_model=bool(pending.get("confirm_expensive_model"))
        )
        # Honour the expensive-model confirm: surface the warning and drop the switch rather than
        # spend on a model the user never confirmed.
        if result.get("confirm_required"):
            _emit("error", sid, {"message": result.get("confirm_message") or result.get("warning") or ""})
    except Exception as e:
        _emit("error", sid, {"message": f"Could not switch model: {e}"})


class CompressionLockHeld(Exception):
    """Raised by _compress_session_history when a concurrent compression_locks row skipped compression."""

    def __init__(self, holder: str | None = None):
        self.holder = holder
        super().__init__(f"Compression lock held: {holder or 'unknown'}")


def _compress_session_history(
    session: dict, focus_topic: str | None = None, approx_tokens: int | None = None,
    before_messages: list | None = None, history_version: int | None = None,
) -> tuple[int, dict]:
    """Single choke point for all manual-compress routes (session.compress RPC, command.dispatch
    /compress|/compact, slash-exec mirror).

    ``focus_topic`` is the RAW argument string after ``/compress``, parsed HERE (not per-route) with
    :func:`parse_partial_compress_args` so boundary forms (``here [N]``, ``up to here``, ``--keep N``)
    trigger a partial compress on EVERY route — otherwise "/compress here 3" would run a FULL compress
    focused on the literal text "here 3". Mirrors cli.py ``_manual_compress`` / gateway slash_commands.
    """
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import (
        parse_partial_compress_args, rejoin_compressed_head_and_tail, split_history_for_partial_compress,
    )

    agent = session["agent"]
    # Snapshot under the lock so the LLM-bound compression call does NOT hold history_lock for the
    # request — otherwise prompt.submit etc. block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
    history = before_messages
    if len(history) < 4:
        return 0, _get_usage(agent)
    partial, keep_last, focus_topic = parse_partial_compress_args(focus_topic or "")
    # Only the head is summarized; the last `keep_last` exchanges ride along verbatim. A degenerate
    # split (empty tail) falls back to full compression so the user still gets an action.
    tail: list = []
    head = history
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        if not tail:
            partial = False
            head = history
    if approx_tokens is None:
        # Include system prompt + tool schemas so the figure reflects real request pressure.
        _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
        _tools = getattr(agent, "tools", None) or None
        approx_tokens = estimate_request_tokens_rough(history, system_prompt=_sys_prompt, tools=_tools)
    # system_message=None: _compress_context rebuilds the system prompt via _build_system_prompt(None);
    # passing the cached prompt (already holding the identity block) appends the identity twice.
    # force=True: every caller is a manual /compress path, which bypasses the summary-failure
    # cooldown like the CLI and gateway handlers. Partial compress has no focus topic (exclusive modes).
    try:
        compressed, _ = agent._compress_context(
            head, None, approx_tokens=approx_tokens, focus_topic=focus_topic or None, force=True,
            defer_context_engine_notification=True,
        )
    except Exception:
        finalize_context_engine_compression_notification(agent, committed=False)
        raise
    # Lock-skipped: raise so callers surface a clear message instead of "No changes from compression".
    # Type-pinned (is True / str) because bare truthiness is fooled by MagicMock auto-attrs.
    _lock_skipped = getattr(agent, "_compression_skipped_due_to_lock", None)
    if _lock_skipped is True or isinstance(_lock_skipped, str):
        agent._compression_skipped_due_to_lock = None
        # No boundary committed; discard the pending deferred notification (exactly-once, no-op safe).
        finalize_context_engine_compression_notification(agent, committed=False)
        raise CompressionLockHeld(_lock_skipped if isinstance(_lock_skipped, str) else None)

    if partial and tail:
        compressed = rejoin_compressed_head_and_tail(compressed, tail)
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the result so we don't clobber concurrent edits.
            finalize_context_engine_compression_notification(agent, committed=False)
            return 0, _get_usage(agent)
        session["history"] = compressed
        session["history_version"] = history_version + 1
    return len(history) - len(compressed), _get_usage(agent)


def _sync_session_key_after_compress(
    sid: str, session: dict, *, clear_pending_title: bool = True, restart_slash_worker: bool = True
) -> None:
    """Re-anchor the gateway-side ``session_key`` when _compress_context rotates ``agent.session_id``
    to a SessionDB continuation; otherwise approval routing, slash worker init, DB title/history
    lookups and yolo state keep targeting the ended parent.

    clear_pending_title: True for manual /compress (title belongs to the old session); False for
    post-turn auto-compression so pending_title applies to the continuation.
    restart_slash_worker: True unless the caller manages the worker (it holds the stale key).
    """
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    lease_reanchored = _transfer_active_session_slot(sid, session, new_session_id=new_session_id)
    if not lease_reanchored:
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid, old_key, new_session_id,
        )

    try:
        from tools.approval import (
            disable_session_yolo, enable_session_yolo, is_session_yolo_enabled, register_gateway_notify,
            unregister_gateway_notify,
        )

        with contextlib.suppress(Exception):
            unregister_gateway_notify(old_key)
        session["session_key"] = new_session_id
        try:
            yolo_was_on = is_session_yolo_enabled(old_key)
        except Exception:
            yolo_was_on = False
        if yolo_was_on:
            with contextlib.suppress(Exception):
                enable_session_yolo(new_session_id)
                disable_session_yolo(old_key)
        with contextlib.suppress(Exception):
            register_gateway_notify(new_session_id, lambda data: _emit_approval_request(sid, data))
    except Exception:
        # Even if the approval module fails to import, anchor session_key on the continuation id.
        session["session_key"] = new_session_id

    # Invalidate any in-flight ``_drain_queued_prompt`` claim taken under the pre-rotation key: a raced
    # drain must not dispatch on the continuation (its envelope is restored to the queue).
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1

    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        with contextlib.suppress(Exception):
            _restart_slash_worker(sid, session)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
