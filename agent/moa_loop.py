"""Mixture-of-Agents runtime helpers for /moa turns.

The slash command marks one user turn as MoA-enabled; the normal agent loop still
owns tool calling and turn termination, while this module gathers reference-model
context before each model iteration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
from dataclasses import KW_ONLY, dataclass, replace
from types import SimpleNamespace
from typing import Any

from agent.auxiliary_client import call_llm
from agent.message_content import flatten_message_text
from agent.transports import get_transport
from agent.usage_pricing import CanonicalUsage

logger = logging.getLogger(__name__)

# Privacy filter (moa.privacy_filter: '' | display | full, #59959). Secret shapes
# are handled by agent.redact; these two patterns add the PII classes it leaves
# alone (emails, formatted NA phones). The phone pattern requires explicit
# delimiters so line numbers, dates, times, SHAs, IPs and versions never match.
_MOA_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_MOA_PHONE_RE = re.compile(
    r"(?<![\w.+-])"                    # no leading word char / dot / + / - (kills IPs, IDs, versions)
    r"(?:\+?1[ .-])?"                  # optional NA country code
    r"(?:\(\d{3}\)[ .-]?|\d{3}[.-])"   # delimited area code: (555) or 555- / 555.
    r"\d{3}[.-]\d{4}"                  # exchange-subscriber with explicit separator
    r"(?![\w-])"                       # no trailing word char / hyphen
)


def _redact_reference_text(text: Any) -> Any:
    """Redact secrets (central redactor) then MoA PII patterns.

    force=True: the privacy filter is its own opt-in, independent of the global
    log-redaction toggle. code_file=True: advisory text is prose/code, so the
    ENV/JSON assignment heuristics that mangle source snippets stay off.
    """
    if not isinstance(text, str) or not text:
        return text
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(text, force=True, code_file=True)
    text = _MOA_EMAIL_RE.sub("[redacted email]", text)
    text = _MOA_PHONE_RE.sub("[redacted phone]", text)
    return text


def _moa_privacy_mode(moa_raw: Any) -> str:
    """Resolve the normalized privacy-filter mode from a raw ``moa`` config."""
    from hermes_cli.moa_config import coerce_privacy_filter

    raw = moa_raw if isinstance(moa_raw, dict) else {}
    return coerce_privacy_filter(raw.get("privacy_filter"))


def _redact_reference_outputs(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[tuple[str, str, Any]]:
    """Redact advisor text in reference-output tuples; accounting slot untouched."""
    return [
        (label, _redact_reference_text(text), acct)
        for label, text, acct in reference_outputs
    ]


def _redact_message_content(content: Any) -> Any:
    """Redact a message's content: a string, or the text parts of a content-part list."""
    if isinstance(content, str):
        return _redact_reference_text(content)
    if isinstance(content, list):
        return [
            {**p, "text": _redact_reference_text(p.get("text"))}
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            else p
            for p in content
        ]
    return content


def _redact_trace_messages(messages: Any) -> Any:
    """Redact message copies for trace persistence (string or content-part lists)."""
    if not isinstance(messages, list):
        return messages
    return [
        {**m, "content": _redact_message_content(m.get("content"))} if isinstance(m, dict) else m
        for m in messages
    ]


def _redact_trace_accounting(acct: Any) -> Any:
    """Copy a ``_RefAccounting`` with its trace text (messages/output) redacted."""
    if not isinstance(acct, _RefAccounting):
        return acct
    return replace(
        acct,
        messages=_redact_trace_messages(acct.messages),
        output=_redact_reference_text(acct.output),
    )


# Cold-start caches (#66793): the preset and each (provider, model) runtime are
# immutable for a turn, so avoid re-resolving them on every create() call.
_preset_cache_lock = threading.Lock()
_preset_cache: dict[tuple, Any] = {}


def _resolve_preset_cached(preset_name: str) -> tuple[dict[str, Any], Any]:
    """Return ``(preset, raw moa config)``, caching the resolved preset per config mtime.

    load_config() is (mtime_ns, size)-cached upstream; the saving here is skipping
    resolve_moa_preset's full validation of the moa block on every create().
    """
    from hermes_cli.config import get_config_path, load_config
    from hermes_cli.moa_config import resolve_moa_preset

    try:
        cfg_stamp = get_config_path().stat().st_mtime_ns
    except OSError:
        cfg_stamp = None
    moa_raw = load_config().get("moa") or {}
    key = (cfg_stamp, preset_name)
    preset = None
    if cfg_stamp is not None:
        with _preset_cache_lock:
            preset = _preset_cache.get(key)
    if preset is None:
        preset = resolve_moa_preset(moa_raw, preset_name)
        if cfg_stamp is not None:
            with _preset_cache_lock:
                _preset_cache.clear()  # one live config stamp at a time
                _preset_cache[key] = preset
    return preset, moa_raw

_runtime_cache_lock = threading.Lock()
_runtime_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

# Short TTL so rotated keys / base_url edits are picked up within 5 minutes.
_RUNTIME_CACHE_TTL_SECONDS = 300.0

# Cap on concurrent reference calls (guards pathologically large presets).
_MAX_REFERENCE_WORKERS = 8


@dataclass(slots=True)
class _RefAccounting:
    """Per-reference usage, cost and full trace (third slot of a reference-output tuple).

    Advisors may run on a different model than the aggregator, so cost is priced at
    the advisor's OWN rate and summed in dollars. ``messages``/``output``/``model``/
    ``provider``/``temperature`` carry the full input/output for trace persistence
    (only populated when tracing is on).
    """

    usage: Any
    cost_usd: Any = None
    cost_status: str | None = None
    cost_source: str | None = None
    _: KW_ONLY
    messages: Any = None
    output: str | None = None
    model: str | None = None
    provider: str | None = None
    temperature: Any = None


# Per-tool-result char budget for the advisory view: tool CALLS are kept in full,
# tool RESULTS are head+tail previewed. The aggregator always gets the full transcript.
_REFERENCE_TOOL_RESULT_BUDGET = 4000

# Reference system prompt: without this framing a reference assumes it is the acting
# agent and refuses ("I can't access repositories") or tries to call tools.
_REFERENCE_SYSTEM_PROMPT = (
    "You are a reference advisor in a Mixture of Agents (MoA) process. You are "
    "NOT the acting agent and you do NOT execute anything: you cannot call "
    "tools, run commands, browse, or access files, repositories, or URLs, and "
    "you should not try to or apologize for being unable to. A separate "
    "aggregator/orchestrator model holds those capabilities and will take the "
    "actual actions.\n\n"
    "CRITICAL: You must NEVER claim or imply that you have executed a command, "
    "downloaded a file, accessed a URL, or performed any action. You can only "
    "analyze and advise based on the conversation context. Examples of what to "
    "avoid:\n"
    "- Bad: \"I ran curl and got 404.\"\n"
    "- Bad: \"I downloaded the file successfully.\"\n"
    "- Bad: \"I checked the repository and found...\"\n"
    "- Good: \"Based on the error pattern, a curl request to that URL would likely return 404.\"\n"
    "- Good: \"The conversation suggests downloading this file may help.\"\n"
    "- Good: \"From the context, checking the repository would reveal...\"\n\n"
    "The conversation below is the current state of a task handled by that "
    "acting agent. Your job is to give your most intelligent analysis of that "
    "state: understand the goal, reason about the problem, and advise on what "
    "to do next. Surface the best approach, concrete next steps and tool-use "
    "strategy, likely pitfalls and risks, and anything the acting agent may "
    "have missed or gotten wrong. Assume any referenced files, URLs, or "
    "systems exist and reason about them from the context given rather than "
    "asking for access.\n\n"
    "Respond with your advice directly — no preamble, no disclaimers about "
    "tools or access. Your response is private guidance handed to the "
    "aggregator, not an answer shown to the user. NEVER claim to have executed "
    "anything."
)



def _slot_label(slot: dict[str, Any]) -> str:
    label = f"{(slot.get('provider') or '').strip()}:{(slot.get('model') or '').strip()}"
    effort = str(slot.get("reasoning_effort") or "").strip()
    return f"{label}[reasoning={effort}]" if effort else label


def _slot_reasoning_config(slot: dict[str, Any]) -> dict[str, Any] | None:
    """Translate optional per-MoA-slot reasoning_effort into runtime config."""
    effort = slot.get("reasoning_effort")
    try:
        from hermes_constants import parse_reasoning_effort

        return parse_reasoning_effort(effort)
    except Exception:  # pragma: no cover - defensive; bad config must not break MoA
        return None


def _aggregator_reasoning_config(aggregator: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the aggregator's reasoning config: slot > per-model > global.

    The aggregator is the ACTING model, so it falls back through the shared
    ``resolve_reasoning_config`` chokepoint (#64187). References deliberately do
    not: inheriting a global ``xhigh`` into every advisor would multiply cost.
    """
    cfg = _slot_reasoning_config(aggregator)
    if cfg is not None:
        return cfg
    try:
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config

        return resolve_reasoning_config(
            load_config() or {}, str(aggregator.get("model") or "")
        )
    except Exception:  # pragma: no cover - defensive; bad config must not break MoA
        return None


def _slot_runtime(slot: dict[str, Any]) -> dict[str, Any]:
    """Resolve a slot to ``call_llm`` kwargs via ``resolve_runtime_provider``.

    Gives the slot its provider's real api_mode/base_url/api_key instead of letting
    the auxiliary auto-detector guess. Falls back to bare provider/model on error
    (uncached). Cached per (provider, model) with a short TTL (#66793).
    """
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    cache_key = (provider, model)
    now = time.monotonic()
    with _runtime_cache_lock:
        entry = _runtime_cache.get(cache_key)
    if entry is not None:
        stamped_at, cached = entry
        if now - stamped_at < _RUNTIME_CACHE_TTL_SECONDS:
            return cached
    out: dict[str, Any] = {"provider": provider, "model": model}
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested=provider, target_model=model)
        out.update({k: rt[k] for k in ("base_url", "api_key", "api_mode") if rt.get(k)})
        request_overrides = rt.get("request_overrides")
        if isinstance(request_overrides, dict):
            extra_body = request_overrides.get("extra_body")
            if isinstance(extra_body, dict) and extra_body:
                out["extra_body"] = dict(extra_body)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MoA slot runtime resolution failed for %s: %s",
                     _slot_label(slot), exc)
        # Never cache a fallback result: a transient error would pin bare kwargs for a TTL.
        return out
    with _runtime_cache_lock:
        _runtime_cache[cache_key] = (now, out)
    return out


def _merge_slot_extra_body(
    slot_extra_body: Any,
    caller_extra_body: Any,
) -> Any:
    """Merge slot defaults with a caller override for ``call_llm``."""
    if isinstance(slot_extra_body, dict) and slot_extra_body:
        if isinstance(caller_extra_body, dict):
            return {**slot_extra_body, **caller_extra_body}
        if caller_extra_body:
            return caller_extra_body
        return dict(slot_extra_body)
    return caller_extra_body


def _maybe_apply_moa_cache_control(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
    *,
    cache_disabled: bool | None = None,
    cache_ttl: str | None = None,
) -> list[dict[str, Any]]:
    """Apply cache_control to an advisor/aggregator request when its route honors it.

    Same policy function and marker helper as the main loop; MoA has no static
    prefix so the legacy system-and-3 fallback is used. ``cache_disabled`` is
    stamped onto the stub so ``cache_ttl: off`` is honored (#76085); ``cache_ttl``
    threads the agent's tier, clamped per destination (#84733). Returns the
    messages unchanged on any error.
    """
    try:
        from agent.agent_runtime_helpers import (
            anthropic_prompt_cache_policy,
            blank_cache_policy_stub,
        )
        from agent.prompt_caching import (
            apply_anthropic_cache_control,
            effective_cache_ttl,
            envelope_tool_part_cache_markers_supported,
        )

        # Explicit kwarg > runtime snapshot (threaded from the live agent) > config.
        if cache_disabled is None and "_cache_disabled" in runtime:
            cache_disabled = runtime.get("_cache_disabled")

        # blank_cache_policy_stub is the only sanctioned stub (carries _cache_disabled, #76085).
        stub = blank_cache_policy_stub(cache_disabled)
        should_cache, native_layout = anthropic_prompt_cache_policy(
            stub,
            provider=runtime.get("provider") or "",
            base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "",
            model=runtime.get("model") or "",
        )
        if not should_cache:
            return messages
        return apply_anthropic_cache_control(
            messages,
            cache_ttl=effective_cache_ttl(
                # None → "5m"; cache-disabled routes already returned above.
                cache_ttl,
                provider=runtime.get("provider") or "",
                model=runtime.get("model") or "",
            ),
            native_anthropic=native_layout,
            # Envelope routes reject part-level markers in tool_result.content[] (#89886).
            tool_part_markers=envelope_tool_part_cache_markers_supported(
                runtime.get("provider") or "", runtime.get("base_url") or ""
            ),
        )
    except Exception as exc:  # pragma: no cover - decoration must never break a call
        logger.debug("MoA cache_control decoration skipped: %s", exc)
        return messages


def _price_reference_response(
    response: Any, slot: dict[str, Any], runtime: dict[str, Any]
) -> tuple[Any, Any, str | None, str | None]:
    """Normalize a reference response's usage with the slot's OWN provider/api_mode and
    price it at its own rate (hence fan-out cost is summed in dollars). Never raises."""
    from agent.usage_pricing import estimate_usage_cost, normalize_usage

    usage = CanonicalUsage()
    raw_usage = getattr(response, "usage", None)
    if raw_usage:
        try:
            usage = normalize_usage(
                raw_usage, provider=runtime.get("provider"), api_mode=runtime.get("api_mode"),
            )
        except Exception:  # pragma: no cover - defensive
            usage = CanonicalUsage()
    try:
        cost = estimate_usage_cost(
            slot.get("model") or "",
            usage,
            provider=runtime.get("provider"),
            base_url=runtime.get("base_url"),
            api_key=runtime.get("api_key"),
        )
        return usage, cost.amount_usd, cost.status, cost.source
    except Exception:  # pragma: no cover - defensive
        return usage, None, None, None


def _run_reference(
    slot: dict[str, Any],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reference_timeout: float | None = None,
    context_length_cache: Any = None,
    cache_disabled: bool | None = None,
    cache_ttl: str | None = None,
) -> tuple[str, str, Any]:
    """Call one reference model; return ``(label, text, accounting)``. Never raises.

    The slot is resolved to its provider's real runtime and called through the
    same ``call_llm`` path any model uses. Usage is normalized with the slot's OWN
    provider/api_mode and priced at its own rate. A failed reference becomes a
    labelled ``[failed: …]`` note. Runs inside a thread pool (call_llm is blocking).
    """
    label = _slot_label(slot)
    runtime = _slot_runtime(slot)
    trace_fields = {
        "model": slot.get("model"),
        "provider": runtime.get("provider") or slot.get("provider"),
        "temperature": temperature,
    }
    try:
        # The trimmed view already stripped the agent's system prompt; this is the only one.
        messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
        # Trim to THIS model's window (advisors may be smaller than the aggregator, #60345);
        # estimated after the system prompt is prepended so it counts too.
        messages = _trim_messages_for_reference(
            messages,
            slot,
            runtime,
            reserve_output_tokens=max_tokens,
            context_length_cache=context_length_cache,
        )
        # Anthropic-style caching is opt-in per request; the advisory view is append-only
        # across iterations, so decorating lets iteration N+1 replay N's cached prefix.
        # Pin the live agent disable onto the runtime (not a fresh config read, #76085).
        cache_runtime = runtime
        if cache_disabled is not None:
            cache_runtime = {**runtime, "_cache_disabled": cache_disabled}
        messages = _maybe_apply_moa_cache_control(
            messages, cache_runtime, cache_ttl=cache_ttl
        )
        # Per-slot max_tokens beats the preset-level reference_max_tokens.
        _slot_max_tokens: int | None = slot.get("max_tokens")
        _effective_max_tokens = _slot_max_tokens if _slot_max_tokens is not None else max_tokens
        extra_headers = None
        # Normalize provider aliases (github, github-copilot, ...) via the canonical table.
        from agent.auxiliary_client import _normalize_aux_provider

        if _normalize_aux_provider(str(runtime.get("provider") or "")) in (
            "copilot",
            "copilot-acp",
        ):
            # Copilot gates premium models on request attribution; MoA fan-out serves the
            # user's current turn, so mirror the main agent's x-initiator header.
            extra_headers = {"x-initiator": "user"}
        response = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=_effective_max_tokens,
            timeout=reference_timeout,
            reasoning_config=_slot_reasoning_config(slot),
            extra_headers=extra_headers,
            **runtime,
        )
        _output_text = _extract_text(response) or "(empty response)"
        acct = _RefAccounting(
            *_price_reference_response(response, slot, runtime),
            messages=messages,
            output=_output_text,
            **trace_fields,
        )
        return label, _output_text, acct
    except Exception as exc:
        logger.warning("MoA reference model %s failed: %s", label, exc)
        return label, f"[failed: {exc}]", _RefAccounting(
            CanonicalUsage(),
            messages=[{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages],
            output=f"[failed: {exc}]",
            **trace_fields,
        )


# Output headroom reserved in the reference window when reference_max_tokens is unset.
_REFERENCE_DEFAULT_OUTPUT_RESERVE = 8192

# Estimator slack: estimate_messages_tokens_rough is a rough chars/4 heuristic.
_REFERENCE_TRIM_SAFETY_FRACTION = 0.10


def _trim_messages_for_reference(
    messages: list[dict[str, Any]],
    slot: dict[str, str],
    runtime: dict[str, Any],
    *,
    reserve_output_tokens: int | None = None,
    context_length_cache: Any = None,
) -> list[dict[str, Any]]:
    """Trim an advisory request to fit a reference model's context window (#60345).

    ``messages`` is the full request (advisory system prompt included). Budget =
    window minus ``reserve_output_tokens`` (or a default) minus a safety fraction.
    Drops the OLDEST frames after the system prompt while keeping: the system
    prompt, a user-first body, and the trailing user turn plus one preceding turn
    (even if still over budget). ``context_length_cache`` memoizes the window per
    (provider, model) for the fan-out; unresolvable windows leave messages unchanged.
    """
    if not messages:
        return messages

    from agent.model_metadata import (
        estimate_messages_tokens_rough,
        get_model_context_length,
    )

    model = str(slot.get("model") or "")
    provider = str(runtime.get("provider") or slot.get("provider") or "")
    if not model:
        return messages

    cache_key = (provider, model)
    context_length: int | None = None
    if isinstance(context_length_cache, dict) and cache_key in context_length_cache:
        context_length = context_length_cache[cache_key]
    else:
        try:
            context_length = get_model_context_length(
                model=model,
                base_url=str(runtime.get("base_url") or ""),
                api_key=str(runtime.get("api_key") or ""),
                provider=provider,
            )
        except Exception:
            logger.debug(
                "MoA reference context-length resolution failed for %s",
                _slot_label(slot),
            )
            context_length = None
        if isinstance(context_length_cache, dict):
            # Cache failures too so a flaky metadata source is not re-probed per reference.
            context_length_cache[cache_key] = context_length

    if not isinstance(context_length, int) or context_length <= 0:
        return messages

    reserve = (
        int(reserve_output_tokens)
        if isinstance(reserve_output_tokens, int) and reserve_output_tokens > 0
        else _REFERENCE_DEFAULT_OUTPUT_RESERVE
    )
    budget = int(context_length * (1.0 - _REFERENCE_TRIM_SAFETY_FRACTION)) - reserve
    if budget <= 0:
        return messages

    estimated = estimate_messages_tokens_rough(messages)
    if estimated <= budget:
        return messages

    has_system = bool(messages) and messages[0].get("role") == "system"
    head = [messages[0]] if has_system else []
    body = list(messages[1:] if has_system else messages)

    # Keep the trailing user turn plus at least one preceding turn.
    while len(body) > 2 and estimate_messages_tokens_rough(head + body) > budget:
        body.pop(0)
        # Preserve the user-first invariant after each pop.
        while len(body) > 2 and body[0].get("role") == "assistant":
            body.pop(0)
    # Two frames left with an assistant first: still enforce user-first.
    while len(body) > 1 and body[0].get("role") == "assistant":
        body.pop(0)

    trimmed = head + body
    dropped = len(messages) - len(trimmed)
    if dropped:
        logger.info(
            "MoA reference %s: estimated %d tokens exceeds budget %d "
            "(window %d, output reserve %d); dropped %d oldest message(s).",
            _slot_label(slot),
            estimated,
            budget,
            context_length,
            reserve,
            dropped,
        )
    return trimmed


_REFERENCE_POLL_INTERVAL_S = 5.0

# Sentinel for a reference aborted by user interrupt; the facade must never cache it.
_INTERRUPTED_REFERENCE_NOTE = "[skipped: interrupted by user]"


def _run_references_parallel(
    reference_models: list[dict[str, Any]],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    progress_callback: Any = None,
    reference_timeout: float | None = None,
    agent: Any = None,
    late_accounting_sink: Any = None,
) -> list[tuple[str, str, Any]]:
    """Fan out all reference models in parallel; outputs are in ``reference_models`` order.

    Slots with ``provider == "moa"`` are skipped with a labelled note (recursion
    guard). ``progress_callback(refs_done, refs_total, label)`` fires per completion
    (best-effort). Each element is ``(label, text, _RefAccounting)``.

    With *agent*, the wait polls every ``_REFERENCE_POLL_INTERVAL_S`` so a user
    interrupt can abort it (like tool_executor's batch). In-flight calls cannot be
    killed; their eventual accounting goes to ``late_accounting_sink``.
    """
    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures: dict[Any, int] = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    # Executor threads start with an empty contextvars.Context; propagate the turn's
    # (approval callbacks + Nous Portal conversation tag) into each worker.
    from tools.thread_context import propagate_context_to_thread

    total = len(reference_models)
    completed = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    # Shared per-fan-out context-length cache (dict get/set is GIL-atomic).
    _ctx_len_cache: dict[tuple[str, str], int | None] = {}
    cache_disabled = (
        getattr(agent, "_cache_disabled", None) if agent is not None else None
    )
    # Agent's cache TTL for every advisor request (#84733); clamped in the decorator.
    cache_ttl = getattr(agent, "_cache_ttl", None) if agent is not None else None
    try:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = (
                    _slot_label(slot),
                    "[skipped: MoA presets cannot recursively reference MoA]",
                    _RefAccounting(CanonicalUsage()),
                )
                continue
            futures[
                executor.submit(
                    propagate_context_to_thread(_run_reference),
                    slot,
                    ref_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reference_timeout=reference_timeout,
                    context_length_cache=_ctx_len_cache,
                    cache_disabled=cache_disabled,
                    cache_ttl=cache_ttl,
                )
            ] = idx

        # Collect every reference (no early exit except a user interrupt); progress
        # callbacks fire per completion.
        pending = set(futures)
        while pending:
            done, pending = _futures_wait(pending, timeout=_REFERENCE_POLL_INTERVAL_S)
            for future in done:
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress_callback is not None:
                    try:
                        label = _slot_label(reference_models[idx])
                        progress_callback(completed, total, label)
                    except Exception as exc:  # pragma: no cover - display must never break
                        logger.debug("MoA progress_callback failed: %s", exc)
            if not pending:
                break
            if agent is not None and getattr(agent, "_interrupt_requested", False):
                interrupted = True
                break

        if interrupted:
            for future, idx in futures.items():
                if results[idx] is not None:
                    continue
                if future.cancel():
                    # Never dispatched — nothing was billed.
                    results[idx] = (
                        _slot_label(reference_models[idx]),
                        _INTERRUPTED_REFERENCE_NOTE,
                        _RefAccounting(CanonicalUsage()),
                    )
                elif future.done():
                    # Finished between the interrupt check and now: keep its real output/accounting.
                    results[idx] = future.result()
                else:
                    # Already running — cannot be killed; the provider WILL bill when it
                    # completes, so hand its eventual accounting to the caller's sink.
                    label = _slot_label(reference_models[idx])
                    results[idx] = (
                        label,
                        _INTERRUPTED_REFERENCE_NOTE,
                        _RefAccounting(CanonicalUsage()),
                    )
                    if late_accounting_sink is not None:
                        def _record_late(f: Any, _label: str = label) -> None:
                            try:
                                _lbl, _txt, _acct = f.result()
                            except Exception:  # pragma: no cover - defensive
                                return
                            try:
                                late_accounting_sink(_label, _acct)
                            except Exception:  # pragma: no cover - defensive
                                logger.debug(
                                    "MoA: late accounting sink failed for %s",
                                    _label,
                                )
                        future.add_done_callback(_record_late)
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    return [r for r in results if r is not None]


def _truncate_tool_result(text: str, budget: int = _REFERENCE_TOOL_RESULT_BUDGET) -> str:
    """Head+tail preview of a tool result with an ``[... N chars omitted ...]`` marker."""
    if not text or len(text) <= budget:
        return text
    half = budget // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} chars omitted ...]\n{text[-half:]}"


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict or an attribute object (tool calls arrive as either)."""
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _render_tool_calls(tool_calls: Any) -> str:
    """Render an assistant turn's tool_calls as ``[called tool: name(args)]`` lines.

    Tolerates dict- and SimpleNamespace-shaped entries (and nested ``function``).
    """
    lines: list[str] = []
    for tc in tool_calls or []:
        fn = _field(tc, "function")
        name = _field(fn, "name") or _field(tc, "name") or "tool"
        fn_args = _field(fn, "arguments")
        if isinstance(fn_args, str):
            args_text = fn_args
        elif fn_args is not None:
            try:
                args_text = json.dumps(fn_args, ensure_ascii=False)
            except Exception:
                args_text = str(fn_args)
        else:
            args_text = ""
        lines.append(f"[called tool: {name}({args_text})]" if args_text else f"[called tool: {name}]")
    return "\n".join(lines)


_ADVISORY_INSTRUCTION = (
    "[The conversation above is the current state of the task. Give your "
    "most intelligent judgement: what is going on, what should happen next, "
    "what risks or mistakes you see, and how the acting agent should "
    "proceed.]"
)


def _reference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the advisory (reference-model) view of the conversation.

    Flattens the transcript to plain user/assistant TEXT turns: the system prompt
    is dropped, tool_calls are rendered inline, and tool results are folded into
    the preceding assistant turn as ``[tool result: ...]`` previews. Zero tool-role
    messages / tool_calls arrays are emitted, so strict providers do not 400.
    The view always ends on a ``user`` turn (Anthropic treats a trailing assistant
    turn as prefill): a synthetic advisory request is APPENDED rather than
    deleting context. The aggregator always receives the full transcript.
    """
    rendered: list[dict[str, Any]] = []
    last_user_content: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        # Content may be a list (cache_control-decorated text parts, multimodal turns);
        # flatten_message_text extracts text so decorated and undecorated transcripts
        # yield a byte-identical view (stable advisory prefix for advisor caching).
        text = flatten_message_text(content)

        if role == "system":
            continue
        if role == "user":
            if not text.strip() and isinstance(content, list) and content:
                # Structured content with no text (e.g. image-only): an empty user
                # message is rejected by strict providers and skipping breaks alternation.
                text = "[user sent non-text content (e.g. an image attachment)]"
            if not text.strip():
                # Genuinely empty user turn: strict providers (Kimi, ZAI) 400 on it; dropping
                # is safe because the advisory view is not strictly alternating anyway.
                continue
            last_user_content = text
            rendered.append({"role": "user", "content": text})
        elif role == "assistant":
            parts: list[str] = []
            if text.strip():
                parts.append(text.strip())
            calls_text = _render_tool_calls(msg.get("tool_calls"))
            if calls_text:
                parts.append(calls_text)
            # Empty assistant turns (no text, no calls) carry nothing advisory.
            if parts:
                rendered.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            # Fold the tool result into the preceding assistant turn as text.
            result_text = _truncate_tool_result(text)
            block = f"[tool result: {result_text}]"
            if rendered and rendered[-1].get("role") == "assistant":
                rendered[-1]["content"] = rendered[-1]["content"] + "\n" + block
            else:
                # No assistant turn to attach to (e.g. a leading tool result).
                rendered.append({"role": "assistant", "content": block})
        # Any other role is ignored.

    # End on a user turn by appending a synthetic advisory request (Anthropic
    # rejects trailing assistant prefill); an existing trailing user turn is left as is.
    if rendered and rendered[-1].get("role") == "assistant":
        rendered.append({"role": "user", "content": _ADVISORY_INSTRUCTION})

    if not rendered:
        # Nothing rendered: fall back to the latest user turn.
        if last_user_content is not None:
            return [{"role": "user", "content": last_user_content}]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                fallback_text = flatten_message_text(msg.get("content"))
                if fallback_text.strip():
                    return [{"role": "user", "content": fallback_text}]
    return rendered



def _extract_text(response: Any) -> str:
    try:
        transport = get_transport("chat_completions")
        if transport is None:
            raise RuntimeError("chat_completions transport unavailable")
        normalized = transport.normalize_response(response)
        text = (normalized.content or "").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()
    except Exception:
        return ""


def _preset_temperature(preset: dict[str, Any], key: str) -> float | None:
    """Read an optional preset temperature; None (absent/empty/null) = provider default.

    A hardcoded fallback previously forced 0.6/0.4 even when unset.
    """
    value = preset.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("ignoring non-numeric %s=%r in MoA preset", key, value)
        return None


def _hash_messages(msgs: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\u0000".join(f"{m.get('role')}:{m.get('content')}" for m in msgs).encode("utf-8", "replace")
    ).hexdigest()


def _is_failed_reference(text: str) -> bool:
    """Whether a reference output is a ``[failed: …]`` / ``[skipped: …]`` sentinel."""
    sentinel = text.lstrip().lower()
    return sentinel.startswith("[failed:") or sentinel.startswith("[skipped:")


def _successful_references(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[tuple[str, str, Any]]:
    """Filter failed advice while preserving each accounting payload."""
    return [output for output in reference_outputs if not _is_failed_reference(output[1])]


def _failed_reference_labels(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[str]:
    return [label for label, text, _accounting in reference_outputs if _is_failed_reference(text)]


def _join_reference_outputs(outputs: list[tuple[str, str, Any]], degraded: str = "") -> str:
    """Render numbered reference blocks for the aggregator, appending any degraded notice."""
    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text, _acct) in enumerate(outputs, start=1)
    )
    if degraded:
        joined = f"{joined}\n\n{degraded}" if joined else degraded
    return joined


def _sum_reference_accounting(outputs: list[tuple[str, str, Any]]) -> tuple[Any, Any]:
    """Sum fan-out usage AND cost in dollars (each advisor priced at its OWN rate)."""
    usage = CanonicalUsage()
    cost: Any = None
    for _lbl, _txt, acct in outputs:
        if isinstance(acct, _RefAccounting):
            if isinstance(acct.usage, CanonicalUsage):
                usage = usage + acct.usage
            if acct.cost_usd is not None:
                cost = (cost or 0) + acct.cost_usd
    return usage, cost


def _degraded_notice(failed_labels: list[str], policy: str) -> str:
    if not failed_labels or policy.strip().lower() == "silent":
        return ""
    return f"[Reference models unavailable: {', '.join(failed_labels)}]"


def aggregate_moa_context(
    *,
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    reference_models: list[dict[str, Any]],
    aggregator: dict[str, Any],
    temperature: float | None = None,
    aggregator_temperature: float | None = None,
    reference_max_tokens: int | None = None,
    reference_timeout: float | None = None,
    degraded_reference_policy: str = "loud",
    agent: Any = None,
) -> str:
    """Run configured reference models and synthesize their advice (one-shot /moa).

    Failures become model-specific notes instead of aborting the loop.
    ``reference_max_tokens`` caps ONLY the fan-out — capping the aggregator
    truncated long syntheses (#53580). ``temperature`` / ``aggregator_temperature``
    default to None (provider default). ``agent`` makes the fan-out interruptible.
    """
    reference_models = [slot for slot in reference_models if slot.get("enabled", True)]
    reference_outputs: list[tuple[str, str, Any]] = []
    ref_messages = _reference_messages(api_messages)
    reference_outputs = _run_references_parallel(
        reference_models,
        ref_messages,
        temperature=temperature,
        max_tokens=reference_max_tokens,
        reference_timeout=reference_timeout,
        agent=agent,
    )

    successful_outputs = _successful_references(reference_outputs)
    failed_labels = _failed_reference_labels(reference_outputs)

    # 'full' privacy mode also redacts advisor text before it reaches this synthesizer
    # ('display' has no surface here). Failed refs are already filtered out.
    try:
        from hermes_cli.config import load_config as _load_config

        if _moa_privacy_mode((_load_config() or {}).get("moa")) == "full":
            successful_outputs = _redact_reference_outputs(successful_outputs)
    except Exception:  # pragma: no cover - privacy filter must never break a turn
        logger.debug("MoA privacy filter check failed", exc_info=True)

    degraded = _degraded_notice(failed_labels, degraded_reference_policy)
    joined = _join_reference_outputs(successful_outputs, degraded)

    # Every reference failed: skip the aggregator (synthesizing over nothing can block
    # for the full provider timeout) and return only the sanitized notice.
    if reference_outputs and not successful_outputs:
        logger.warning(
            "MoA: all %d reference(s) failed — skipping aggregator synthesis",
            len(reference_outputs),
        )
        notice = degraded or "[Reference models unavailable]"
        return (
            "[Mixture of Agents context — all reference models failed. "
            "Proceeding without aggregated guidance.]\n"
            f"References: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
            f"{notice}"
        )

    synth_prompt = (
        "You are the aggregator in a Mixture of Agents process. Synthesize the "
        "reference responses into concise, actionable guidance for the main "
        "Hermes agent. Focus on next steps, tool-use strategy, risks, and any "
        "disagreements. Do not answer the user directly unless that is all that "
        "is needed; produce context the main agent should use in its normal loop.\n\n"
        f"Original user prompt:\n{user_prompt}\n\n"
        f"Reference responses:\n{joined}"
    )

    agg_label = _slot_label(aggregator)
    agg_runtime = _slot_runtime(aggregator)
    # Pin the live agent disable onto synthesis decoration (#76085); None is a no-op.
    agg_cache_runtime = agg_runtime
    _agg_cache_disabled = (
        getattr(agent, "_cache_disabled", None) if agent is not None else None
    )
    if _agg_cache_disabled is not None:
        agg_cache_runtime = {
            **agg_runtime,
            "_cache_disabled": _agg_cache_disabled,
        }
    # Agent's cache TTL for the one-shot synthesis path (#84733).
    _agg_cache_ttl = getattr(agent, "_cache_ttl", None) if agent is not None else None
    try:
        # Same cache_control decoration as the advisor calls; this synthesis call is
        # a third independent MoA call path that otherwise re-bills its full input.
        agg_messages = _maybe_apply_moa_cache_control(
            [{"role": "user", "content": synth_prompt}],
            agg_cache_runtime,
            cache_ttl=_agg_cache_ttl,
        )
        response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            reasoning_config=_aggregator_reasoning_config(aggregator),
            **agg_runtime,
        )
        synthesis = _extract_text(response)
    except Exception as exc:
        logger.warning("MoA aggregator model %s failed: %s", agg_label, exc)
        synthesis = ""

    if not synthesis:
        synthesis = joined

    return (
        "[Mixture of Agents context — use this as private guidance for the "
        "normal Hermes agent loop. You may call tools, continue reasoning, or "
        "finish normally.]\n"
        f"Aggregator: {agg_label}\n"
        f"References: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
        f"{synthesis.strip()}"
    )


def _completed_response_as_stream_chunk(response: Any) -> Any:
    """Adapt a completed response (``choices[0].message``) into one delta stream chunk.

    Done at the MoA facade boundary so transports stay untouched.
    """

    choices = getattr(response, "choices", None)
    first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    message = getattr(first_choice, "message", None)
    raw_tool_calls = getattr(message, "tool_calls", None)
    tool_call_deltas = None
    if isinstance(raw_tool_calls, (list, tuple)) and raw_tool_calls:
        tool_call_deltas = []
        for index, tc in enumerate(raw_tool_calls):
            function = getattr(tc, "function", None)
            tool_call_deltas.append(SimpleNamespace(
                index=getattr(tc, "index", index),
                id=getattr(tc, "id", None),
                type=getattr(tc, "type", None) or "function",
                function=SimpleNamespace(
                    name=getattr(function, "name", None),
                    arguments=getattr(function, "arguments", None),
                ),
            ))
    delta = SimpleNamespace(
        content=getattr(message, "content", None),
        tool_calls=tool_call_deltas,
        reasoning_content=getattr(message, "reasoning_content", None),
        reasoning=getattr(message, "reasoning", None),
        reasoning_details=getattr(message, "reasoning_details", None),
    )
    choice = SimpleNamespace(
        index=getattr(first_choice, "index", 0),
        delta=delta,
        finish_reason=getattr(first_choice, "finish_reason", None) or "stop",
    )
    return SimpleNamespace(
        id=getattr(response, "id", None),
        model=getattr(response, "model", None),
        choices=[choice],
        usage=getattr(response, "usage", None),
    )


def _attach_reference_guidance(agg_messages: list[dict[str, Any]], guidance: str) -> None:
    """Attach the per-turn reference block at the END of the aggregator prompt.

    The block varies per iteration; merging it into the (early) original user
    message would diverge the prompt prefix and re-prefill the whole conversation
    each step. Appending keeps ``[system][task][tool-history]`` cache-stable. A
    trailing user turn is merged in place (string or content-part list — a new text
    part rides AFTER the cache_control-marked part); otherwise a user message is
    appended (two consecutive user turns would be rejected by strict providers).
    """
    last = agg_messages[-1] if agg_messages else None
    if last is not None and last.get("role") == "user":
        last_content = last.get("content")
        if isinstance(last_content, str):
            last["content"] = last_content + "\n\n" + guidance
            return
        if isinstance(last_content, list):
            last["content"] = [*last_content, {"type": "text", "text": "\n\n" + guidance}]
            return
    agg_messages.append({"role": "user", "content": guidance})


def peel_reference_guidance(
    messages: list[dict[str, Any]],
    guidance: Any,
) -> list[dict[str, Any]]:
    """Exact inverse of ``_attach_reference_guidance`` (the three attach shapes).

    Used by the failover redecoration chokepoint so a cache breakpoint never lands
    on the turn-varying guidance (#72626). Returns a new list; inputs are not mutated.
    """
    if not guidance or not messages:
        return messages
    guidance_text = str(guidance)
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return messages
    content = last.get("content")
    if content == guidance_text:
        # Attach shape (c): guidance was appended as its own user message.
        return list(messages[:-1])
    suffix = "\n\n" + guidance_text
    if isinstance(content, str) and content.endswith(suffix):
        # Attach shape (a): merged into a trailing string user turn.
        peeled = dict(last)
        peeled["content"] = content[: -len(suffix)]
        return [*messages[:-1], peeled]
    if isinstance(content, list) and content:
        last_part = content[-1]
        if isinstance(last_part, dict) and last_part.get("type", "text") == "text":
            text = last_part.get("text") or ""
            if text == suffix or text == guidance_text:
                # Attach shape (b): guidance rode as its own trailing part.
                peeled = dict(last)
                peeled["content"] = list(content[:-1])
                if not peeled["content"]:
                    # Guidance was the only content: drop the whole message (mirrors shape c).
                    return list(messages[:-1])
                return [*messages[:-1], peeled]
            if text.endswith(suffix):
                new_part = dict(last_part)
                new_part["text"] = text[: -len(suffix)]
                peeled = dict(last)
                peeled["content"] = [*content[:-1], new_part]
                return [*messages[:-1], peeled]
    return messages


class MoAChatCompletions:
    """OpenAI-chat-compatible facade where the aggregator is the acting model."""

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.preset_name = preset_name or "default"
        # Optional best-effort display hook, called as ``reference_callback(event, **kwargs)``:
        #   "moa.reference"   index, count, label, text
        #   "moa.progress"    refs_done, refs_total, label   (per reference completion)
        #   "moa.phase"       phase, refs_done, refs_total, aggregator
        #   "moa.aggregating" aggregator (label), ref_count
        self.reference_callback = reference_callback
        # Owning AIAgent (optional); lets the fan-out check agent._interrupt_requested.
        self._agent = agent
        # State-scoped reference cache keyed on the advisory-view signature: a new
        # user/tool message is a MISS (references re-run), a redundant create() with
        # identical state is a HIT (no re-run, no re-emit).
        self._ref_cache_key: tuple | None = None
        self._ref_cache_outputs: list[tuple[str, str, Any]] = []
        # Fan-out usage/cost from the latest cache-MISS create(), awaiting
        # consume_reference_usage (zero deposited on a HIT so spend counts once).
        self._pending_reference_usage: Any = CanonicalUsage()
        self._pending_reference_cost: Any = None
        # Guards pending usage/cost against late-accounting callbacks on worker threads.
        self._accounting_lock = threading.Lock()
        # Resolved aggregator slot from the latest create(); cost accounting prices the
        # acting turn at its real model instead of the virtual preset name.
        self.last_aggregator_slot: Any = None
        # Full-turn trace parts from a cache-MISS create(), flushed by consume_and_save_trace.
        self._pending_trace: Any = None
        # Per-advisor metrics for observability hooks; NOT consumed (post_api_request
        # fires on a different branch than consume_and_save_trace).
        self._last_reference_metrics: Any = None
        # every_n cadence state, scoped to a single USER TURN (resets on a new user
        # message) so iteration 1 of every turn is on-cadence.
        self._fanout_iteration_count = 0
        self._fanout_turn_sig: str | None = None
        self._fanout_last_state_sig: str | None = None
        # Normalized moa.privacy_filter mode ('' | 'display' | 'full'), refreshed per create().
        self._privacy_mode: str = ""

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending fan-out ``(CanonicalUsage, cost_usd_or_None)`` and reset both.

        Clearing prevents a streaming retry re-entering accounting from double-counting.
        """
        with self._accounting_lock:
            usage = self._pending_reference_usage or CanonicalUsage()
            cost = self._pending_reference_cost
            self._pending_reference_usage = CanonicalUsage()
            self._pending_reference_cost = None
        return usage, cost

    def last_reference_metrics(self) -> Any:
        """Per-advisor metrics from the most recent fan-out, or None (read-only)."""
        return self._last_reference_metrics

    def _record_late_reference_accounting(self, label: str, accounting: Any) -> None:
        """Fold a late-completing interrupted reference's real spend into pending totals.

        Registered as a done-callback on abandoned futures (they still bill). Thread-safe.
        """
        if not isinstance(accounting, _RefAccounting):
            return
        self._fold_pending_accounting(*_sum_reference_accounting([(label, "", accounting)]))
        logger.debug(
            "MoA: recorded late accounting for interrupted reference %s", label
        )

    def _fold_pending_accounting(self, usage: Any, cost: Any) -> None:
        """Add (never overwrite) fan-out spend so late interrupted-reference deposits survive."""
        with self._accounting_lock:
            self._pending_reference_usage = (
                self._pending_reference_usage or CanonicalUsage()
            ) + usage
            if cost is not None:
                self._pending_reference_cost = (self._pending_reference_cost or 0) + cost

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn trace to disk (no-op when nothing is pending).

        ``aggregator_output_fallback`` is the caller's resolved acting text: on the
        streaming path the output could not be captured at ``create()`` time, so it
        is folded in here. Clears the pending trace; never raises.
        """
        pending = self._pending_trace
        self._pending_trace = None
        if not pending or "aggregator_input_messages" not in pending:
            return
        try:
            from agent.moa_trace import save_moa_turn

            agg_slot = pending.get("aggregator_slot") or {}
            # Inline capture (non-streaming) beats the caller's streamed text.
            agg_output = pending.get("aggregator_output")
            if agg_output is None and aggregator_output_fallback:
                agg_output = aggregator_output_fallback
            save_moa_turn(
                session_id=session_id,
                preset_name=pending.get("preset", ""),
                reference_outputs=pending.get("reference_outputs", []),
                aggregator_label=pending.get("aggregator_label", ""),
                aggregator_model=agg_slot.get("model"),
                aggregator_provider=agg_slot.get("provider"),
                aggregator_temperature=pending.get("aggregator_temperature"),
                aggregator_input_messages=pending.get("aggregator_input_messages"),
                aggregator_output=agg_output,
                aggregator_streamed=bool(pending.get("aggregator_streamed")),
            )
        except Exception as exc:  # pragma: no cover - tracing must never break a turn
            logger.debug("MoA trace flush failed: %s", exc)

    def _emit(self, event: str, **kwargs: Any) -> None:
        cb = self.reference_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("MoA reference_callback failed for %s: %s", event, exc)

    def prepare(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run the advisor fan-out and return the exact aggregator request.

        The loop measures this augmented prompt before its compression gate, then
        hands the object back to ``create()`` so the fan-out is not repeated.
        """
        return self.create(messages=messages, _moa_prepare_only=True)

    def rebase_prepared_request(
        self, prepared: dict[str, Any], messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Re-attach already-generated guidance to a rebuilt (compressed) transcript."""
        guidance = prepared.get("guidance")
        agg_messages = [dict(message) for message in messages]
        if guidance:
            _attach_reference_guidance(agg_messages, str(guidance))
        return {**prepared, "messages": agg_messages}

    def _call_prepared_aggregator(
        self, prepared: dict[str, Any], api_kwargs: dict[str, Any]
    ) -> Any:
        """Send an already prepared MoA aggregator request exactly once."""
        agg_messages = prepared["messages"]
        aggregator = prepared["aggregator"]
        aggregator_temperature = prepared["aggregator_temperature"]
        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
        max_tokens: Any = agg_kwargs.get("max_tokens")
        tools: Any = agg_kwargs.get("tools")
        extra_body: Any = agg_kwargs.get("extra_body")
        agg_runtime = _slot_runtime(aggregator)
        try:
            from agent.agent_runtime_helpers import (
                plan_cache_sections_for_destination,
            )

            guidance = prepared.get("guidance")
            planning_messages = agg_messages
            if guidance:
                planning_messages = peel_reference_guidance(
                    agg_messages,
                    str(guidance),
                )
            # plan_cache_sections_for_destination returns request-local copies.
            # Tri-state cache_disabled: facades built via __new__ have no _agent; forcing
            # False would suppress the planner's config fallback (#76085).
            _agent = getattr(self, "_agent", None)
            _cache_disabled = (
                getattr(_agent, "_cache_disabled", None)
                if _agent is not None
                else None
            )
            agg_messages, tools = plan_cache_sections_for_destination(
                planning_messages,
                tools,
                provider=agg_runtime.get("provider") or "",
                base_url=agg_runtime.get("base_url") or "",
                api_mode=agg_runtime.get("api_mode") or "",
                model=agg_runtime.get("model") or "",
                cache_disabled=_cache_disabled,
                # Agent TTL + stable system prefix so MoA stops regressing 1h → 5m (#84733).
                cache_ttl=getattr(_agent, "_cache_ttl", None),
                static_system_prefix=getattr(
                    _agent, "_cached_system_prompt_static", None
                ),
            )
            if guidance:
                _attach_reference_guidance(agg_messages, str(guidance))
        except Exception as exc:  # pragma: no cover - cache planning must not block MoA
            # Warning, not debug: this is the aggregator's ONLY decoration path.
            logger.warning(
                "MoA aggregator cache plan failed — sending undecorated "
                "request (cache misses expected): %s", exc,
            )
        # Record the exact aggregator INPUT into the pending trace; the persisted COPY
        # is redacted under any privacy mode while the live input stays raw.
        if self._pending_trace is not None:
            self._pending_trace["aggregator_input_messages"] = (
                _redact_trace_messages([dict(m) for m in agg_messages])
                if getattr(self, "_privacy_mode", "")
                else agg_messages
            )
            self._pending_trace["aggregator_label"] = _slot_label(aggregator)
        # The aggregator is the acting model: call it through the same request path
        # any model uses, with max_tokens passed through (None → model maximum).
        # stream=True returns the RAW token stream (consumer reassembles + retries);
        # the non-streaming path forwards no stream/stream_options/timeout.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = (
                api_kwargs.get("stream_options") or {"include_usage": True}
            )
            # The consumer's stream-read timeout must govern the aggregator stream.
            if api_kwargs.get("timeout") is not None:
                stream_kwargs["timeout"] = api_kwargs["timeout"]
        # Pop the runtime's extra_body and merge with the caller's (caller wins) so the
        # explicit kwarg never collides with **agg_runtime.
        agg_extra_body = _merge_slot_extra_body(
            agg_runtime.pop("extra_body", None),
            extra_body,
        )
        _agg_response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra_body=agg_extra_body,
            # Same reasoning policy as the direct create() path (#64187).
            reasoning_config=_aggregator_reasoning_config(aggregator),
            **stream_kwargs,
            **agg_runtime,
        )
        # Non-streaming: capture the aggregator output inline. Streaming: the output
        # lands as the turn's assistant message; the trace marks it streamed.
        if self._pending_trace is not None:
            if stream:
                self._pending_trace["aggregator_streamed"] = True
                self._pending_trace["aggregator_output"] = None
            else:
                self._pending_trace["aggregator_streamed"] = False
                try:
                    self._pending_trace["aggregator_output"] = _extract_text(_agg_response)
                except Exception:  # pragma: no cover - defensive
                    self._pending_trace["aggregator_output"] = None
        if stream and hasattr(_agg_response, "choices"):
            # Some adapters (openai-codex Responses) return a completed response even
            # when streaming was requested; hand the loop a one-chunk iterator (#55933).
            return iter((_completed_response_as_stream_chunk(_agg_response),))
        return _agg_response

    def _fanout_cache_key(
        self,
        preset: dict[str, Any],
        ref_messages: list[dict[str, Any]],
        reference_models: list[dict[str, Any]],
    ) -> tuple:
        """Compute the turn-scoped reference cache key per the preset's fan-out cadence.

        Cadence (#67199, #63393). "user_turn" (default): advisors run once per user
        turn — the signature hashes only the prefix up to the LAST USER message, so later
        tool iterations are cache HITs. "per_iteration": re-run whenever the advisory
        view changes. "every_n:<N>": iteration 1 of a turn, then every Nth; in-between
        iterations reuse the last on-cadence guidance (key pinned to that run so the
        lookup is a HIT: no advisor calls, no double accounting, no re-emit).
        """
        fanout_mode = str(preset.get("fanout") or "user_turn").strip().lower()
        every_n = 0
        if fanout_mode.startswith("every_n:"):
            try:
                every_n = int(fanout_mode.split(":", 1)[1])
            except (TypeError, ValueError):
                every_n = 0
            if every_n < 2:
                # every_n:1 IS per-iteration (mirrors _coerce_fanout).
                fanout_mode = "per_iteration"
        sig_messages = ref_messages
        turn_prefix = ref_messages
        if fanout_mode == "user_turn" or every_n >= 2:
            # Last REAL user message: the synthetic _ADVISORY_INSTRUCTION marker must not
            # count or the prefix would grow (and re-sign) every iteration.
            for _i in range(len(ref_messages) - 1, -1, -1):
                _m = ref_messages[_i]
                if _m.get("role") == "user" and _m.get("content") != _ADVISORY_INSTRUCTION:
                    turn_prefix = ref_messages[: _i + 1]
                    break
            if fanout_mode == "user_turn":
                sig_messages = turn_prefix

        # every_n bookkeeping: advance the counter only when the advisory STATE changed
        # (a streaming retry must not consume a cadence slot); reset on a new turn prefix.
        if every_n >= 2:
            _turn_sig = _hash_messages(turn_prefix)
            if _turn_sig != self._fanout_turn_sig:
                self._fanout_turn_sig = _turn_sig
                self._fanout_iteration_count = 0
                self._fanout_last_state_sig = None
            _state_sig = _hash_messages(ref_messages)
            if _state_sig != self._fanout_last_state_sig:
                self._fanout_last_state_sig = _state_sig
                self._fanout_iteration_count += 1
            # Iteration 1 is on-cadence; then every Nth iteration after it.
            _on_cadence = (self._fanout_iteration_count - 1) % every_n == 0
            if not _on_cadence and self._ref_cache_outputs:
                return self._ref_cache_key

        return (
            self.preset_name,
            _hash_messages(sig_messages),
            tuple(_slot_label(s) for s in reference_models),
        )

    def create(self, **api_kwargs: Any) -> Any:
        prepared_request = api_kwargs.pop("_moa_prepared_request", None)
        if prepared_request is not None:
            if not isinstance(prepared_request, dict):
                raise TypeError("_moa_prepared_request must be a dict")
            return self._call_prepared_aggregator(prepared_request, api_kwargs)

        preset, _moa_raw = _resolve_preset_cached(self.preset_name)
        # Remembered on self so _call_prepared_aggregator redacts the trace consistently.
        privacy_mode = _moa_privacy_mode(_moa_raw)
        self._privacy_mode = privacy_mode
        messages = list(api_kwargs.get("messages") or [])
        reference_models = [
            slot for slot in (preset.get("reference_models") or [])
            if slot.get("enabled", True)
        ]
        aggregator = preset.get("aggregator") or {}
        # The MoA path's virtual model/provider have no pricing entry; expose the real slot.
        self.last_aggregator_slot = dict(aggregator) if aggregator else None
        # No output caps by default (None → call_llm omits max_tokens). A preset MAY cap
        # ADVISOR output (dominant MoA latency); the acting aggregator is never capped.
        reference_max_tokens = preset.get("reference_max_tokens")
        # None = provider default (see _preset_temperature).
        temperature = _preset_temperature(preset, "reference_temperature")
        aggregator_temperature = _preset_temperature(preset, "aggregator_temperature")
        # None = inherit auxiliary.moa_reference.timeout via call_llm.
        raw_reference_timeout = preset.get("reference_timeout")
        reference_timeout = (
            float(raw_reference_timeout) if raw_reference_timeout else None
        )
        degraded_reference_policy = str(
            preset.get("degraded_reference_policy") or "loud"
        )
        if aggregator_temperature is None and api_kwargs.get("temperature") is not None:
            # The acting agent's own temperature applies to the aggregator (the acting model).
            aggregator_temperature = api_kwargs.get("temperature")

        # A disabled preset = "use the aggregator directly".
        if not preset.get("enabled", True):
            reference_models = []

        reference_outputs: list[tuple[str, str, Any]] = []
        ref_messages = _reference_messages(messages)
        _cache_key = self._fanout_cache_key(preset, ref_messages, reference_models)
        _refs_from_cache = _cache_key == self._ref_cache_key and bool(self._ref_cache_outputs)

        if _refs_from_cache:
            reference_outputs = list(self._ref_cache_outputs)
            # Cache HIT: references already ran and were accounted this turn. Deposit
            # nothing, but do NOT zero pending totals (a late interrupted reference may
            # have deposited real spend). No trace either — a repeat iteration is not a turn.
            self._pending_trace = None
        else:
            # Per-reference progress (``moa.progress``) through the same display hook.
            def _progress(done: int, total: int, label: str) -> None:
                self._emit(
                    "moa.progress",
                    refs_done=done,
                    refs_total=total,
                    label=label,
                )

            reference_outputs = _run_references_parallel(
                reference_models,
                ref_messages,
                temperature=temperature,
                max_tokens=reference_max_tokens,
                progress_callback=_progress,
                reference_timeout=reference_timeout,
                agent=self._agent,
                late_accounting_sink=self._record_late_reference_accounting,
            )
            interrupted_any = any(
                text == _INTERRUPTED_REFERENCE_NOTE
                for _lbl, text, _acct in reference_outputs
            )
            if interrupted_any:
                # An interrupted fan-out is a partial snapshot: never cache it (a HIT
                # would replay placeholder notes every iteration).
                self._ref_cache_key = None
                self._ref_cache_outputs = []
            else:
                self._ref_cache_key = _cache_key
                self._ref_cache_outputs = list(reference_outputs)
            # Fold advisor spend into accounting exactly once per turn.
            self._fold_pending_accounting(*_sum_reference_accounting(reference_outputs))
            # Stash the fan-out for trace persistence (aggregator input/label filled in
            # later; output stitched in by consume_and_save_trace). Traces are persisted,
            # so ANY active privacy mode redacts advisor text and per-advisor input/output.
            if privacy_mode:
                _trace_refs = [
                    (label, _redact_reference_text(text), _redact_trace_accounting(acct))
                    for label, text, acct in reference_outputs
                ]
            else:
                _trace_refs = list(reference_outputs)
            self._pending_trace = {
                "preset": self.preset_name,
                "reference_outputs": _trace_refs,
                "aggregator_slot": aggregator,
                "aggregator_temperature": aggregator_temperature,
            }
            # Derived from the privacy-redacted _trace_refs.
            try:
                from agent.moa_trace import slot_metrics

                self._last_reference_metrics = [
                    slot_metrics(acct, label, output=text)
                    for label, text, acct in _trace_refs
                ]
            except Exception as exc:  # pragma: no cover - never break a turn
                logger.debug("MoA reference metrics render failed: %s", exc)
                self._last_reference_metrics = None

            # Surface each reference's answer BEFORE the aggregator acts (once per turn).
            # The cache keeps RAW text; redaction happens at each consuming surface.
            _ref_count = len(reference_outputs)
            for _idx, (_label, _text, _accounting) in enumerate(reference_outputs, start=1):
                self._emit(
                    "moa.reference",
                    index=_idx,
                    count=_ref_count,
                    label=_label,
                    text=_redact_reference_text(_text) if privacy_mode else _text,
                )
            if _ref_count:
                # Phase transition: fan-out complete, aggregator about to act.
                self._emit(
                    "moa.phase",
                    phase="aggregator",
                    refs_done=_ref_count,
                    refs_total=_ref_count,
                    aggregator=_slot_label(aggregator),
                )
                self._emit(
                    "moa.aggregating",
                    aggregator=_slot_label(aggregator),
                    ref_count=_ref_count,
                )

        agg_messages = [dict(m) for m in messages]
        successful_outputs = _successful_references(reference_outputs)
        failed_labels = _failed_reference_labels(reference_outputs)
        # 'full' privacy mode redacts advisor text reaching the AGGREGATOR too (#59959);
        # 'display' leaves it raw. Applied to a per-call copy — the cache holds raw text.
        _agg_refs = (
            _redact_reference_outputs(successful_outputs)
            if privacy_mode == "full"
            else successful_outputs
        )
        degraded = _degraded_notice(failed_labels, degraded_reference_policy)
        header = (
            "[Mixture of Agents reference context]\n"
            f"Preset: {self.preset_name}\n"
            f"Aggregator/acting model: {_slot_label(aggregator)}\n"
        )
        guidance: str | None = None
        if reference_outputs and not successful_outputs:
            # Every reference failed: the aggregator acts alone. Under the loud policy it
            # still gets the sanitized unavailability notice; under silent, nothing.
            logger.warning(
                "MoA: all %d reference(s) failed — acting aggregator-alone "
                "without reference guidance",
                len(reference_outputs),
            )
            if degraded:
                guidance = (
                    f"{header}\n"
                    "All reference models failed this turn — no advisory "
                    "guidance is available. Act on your own judgment.\n\n"
                    f"{degraded}"
                )
        elif _agg_refs or degraded:
            guidance = (
                f"{header}"
                f"References: {', '.join(label for label, _, _ in _agg_refs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{_join_reference_outputs(_agg_refs, degraded)}"
            )
        if guidance:
            _attach_reference_guidance(agg_messages, guidance)

        prepared_request = {
            "messages": agg_messages,
            "guidance": guidance,
            "aggregator": aggregator,
            "aggregator_temperature": aggregator_temperature,
        }
        if api_kwargs.pop("_moa_prepare_only", False):
            return prepared_request
        return self._call_prepared_aggregator(prepared_request, api_kwargs)


class MoAClient:
    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(
            preset_name, reference_callback=reference_callback, agent=agent,
        )

    def consume_reference_usage(self) -> Any:
        """Pop pending reference-fan-out usage + cost from the completions facade."""
        return self.chat.completions.consume_reference_usage()

    @property
    def last_aggregator_slot(self) -> Any:
        """Resolved aggregator slot from the most recent create(), or None."""
        return getattr(self.chat.completions, "last_aggregator_slot", None)

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn MoA trace via the completions facade."""
        return self.chat.completions.consume_and_save_trace(
            session_id, aggregator_output_fallback=aggregator_output_fallback
        )

    def last_reference_metrics(self) -> Any:
        """Per-advisor metrics from the most recent fan-out, or None (read-only)."""
        return self.chat.completions.last_reference_metrics()


# Display-event relay table for build_moa_facade: event -> (primary kwarg, secondary
# kwarg or None, {tool_progress_callback kwarg: emit kwarg}). The callback signature is
# ``cb(event, label, text, None, **moa_*)``; "moa.progress" is rendered by frontends as
# a status-bar ``MOA: N/M refs done``.
_RELAY_EVENTS: dict[str, tuple[str, str | None, dict[str, str]]] = {
    "moa.reference": ("label", "text", {"moa_index": "index", "moa_count": "count"}),
    "moa.progress": ("label", None, {"moa_refs_done": "refs_done", "moa_refs_total": "refs_total"}),
    "moa.phase": (
        "aggregator",
        None,
        {"moa_phase": "phase", "moa_refs_done": "refs_done", "moa_refs_total": "refs_total"},
    ),
    "moa.aggregating": ("aggregator", None, {"moa_ref_count": "ref_count"}),
}


def build_moa_facade(agent, preset_name: Any = None) -> MoAClient:
    """Build the MoA facade client for ``agent``, wiring the reference relay.

    Single construction point for ``MoAClient`` (agent_init, fallback restore,
    transport recovery, switch_model): a bare ``MoAClient(preset)`` would drop the
    ``reference_callback`` relay and silence display events for the session
    (#53802). The relay reads ``agent.tool_progress_callback`` at emit time.
    """
    def _moa_reference_relay(event: str, **kwargs: Any) -> None:
        cb = getattr(agent, "tool_progress_callback", None)
        spec = _RELAY_EVENTS.get(event)
        if cb is None or spec is None:
            return
        primary, secondary, extra_map = spec
        try:
            cb(
                event,
                str(kwargs.get(primary) or ""),
                str(kwargs.get(secondary) or "") if secondary else None,
                None,
                **{out: kwargs.get(src) for out, src in extra_map.items()},
            )
        except Exception:
            pass

    resolved_preset = preset_name
    if resolved_preset is None and getattr(agent, "provider", None) == "moa":
        resolved_preset = getattr(agent, "model", None)

    resolved_preset = str(resolved_preset or "default")
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        moa_cfg = normalize_moa_config(load_config().get("moa") or {})
        presets = moa_cfg.get("presets") or {}
        if resolved_preset not in presets:
            resolved_preset = moa_cfg.get("default_preset") or "default"
    except Exception:
        resolved_preset = "default"

    return MoAClient(
        resolved_preset,
        reference_callback=_moa_reference_relay,
        # Lets the fan-out wait be aborted on a user interrupt.
        agent=agent,
    )
