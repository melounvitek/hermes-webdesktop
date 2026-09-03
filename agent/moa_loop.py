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

# Privacy filter (moa.privacy_filter: '' | display | full): PII classes agent.redact
# leaves alone. The phone pattern requires explicit delimiters so line numbers,
# dates, times, SHAs, IPs and versions never match.
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
    log-redaction toggle. code_file=True: keeps the ENV/JSON assignment heuristics
    (which mangle source snippets) off advisory prose/code.
    """
    if not isinstance(text, str) or not text:
        return text
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(text, force=True, code_file=True)
    text = _MOA_EMAIL_RE.sub("[redacted email]", text)
    return _MOA_PHONE_RE.sub("[redacted phone]", text)


def _moa_privacy_mode(moa_raw: Any) -> str:
    """Normalized privacy-filter mode from a raw ``moa`` config."""
    from hermes_cli.moa_config import coerce_privacy_filter

    raw = moa_raw if isinstance(moa_raw, dict) else {}
    return coerce_privacy_filter(raw.get("privacy_filter"))


def _redact_reference_outputs(
    reference_outputs: list[tuple[str, str, Any]],
) -> list[tuple[str, str, Any]]:
    """Redact advisor text in reference-output tuples; accounting slot untouched."""
    return [(label, _redact_reference_text(text), acct) for label, text, acct in reference_outputs]


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


# Cold-start caches: preset and per-(provider, model) runtime are immutable for a turn.
_preset_cache_lock = threading.Lock()
_preset_cache: dict[tuple, Any] = {}


def _resolve_preset_cached(preset_name: str) -> tuple[dict[str, Any], Any]:
    """``(preset, raw moa config)``; the resolved preset is cached per config mtime
    (skips resolve_moa_preset's full validation of the moa block on every create())."""
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

    Cost is priced at the advisor's OWN rate and summed in dollars (advisors may run
    on a different model than the aggregator). Trace fields are only populated when
    tracing is on.
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
    except Exception:  # pragma: no cover - bad config must not break MoA
        return None


def _aggregator_reasoning_config(aggregator: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregator reasoning config: slot > per-model > global (shared chokepoint).

    References deliberately do NOT fall back: inheriting a global ``xhigh`` into
    every advisor would multiply cost.
    """
    cfg = _slot_reasoning_config(aggregator)
    if cfg is not None:
        return cfg
    try:
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config

        return resolve_reasoning_config(load_config() or {}, str(aggregator.get("model") or ""))
    except Exception:  # pragma: no cover - bad config must not break MoA
        return None


def _slot_runtime(slot: dict[str, Any]) -> dict[str, Any]:
    """Slot → ``call_llm`` kwargs with the provider's real api_mode/base_url/api_key.

    Cached per (provider, model) with a short TTL. Falls back to bare provider/model
    on error — never cached, or a transient error would pin bare kwargs for a TTL.
    """
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    cache_key = (provider, model)
    now = time.monotonic()
    with _runtime_cache_lock:
        entry = _runtime_cache.get(cache_key)
    if entry is not None and now - entry[0] < _RUNTIME_CACHE_TTL_SECONDS:
        return entry[1]
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
        logger.debug("MoA slot runtime resolution failed for %s: %s", _slot_label(slot), exc)
        return out
    with _runtime_cache_lock:
        _runtime_cache[cache_key] = (now, out)
    return out


def _merge_slot_extra_body(slot_extra_body: Any, caller_extra_body: Any) -> Any:
    """Merge slot defaults with a caller override (caller wins) for ``call_llm``."""
    if isinstance(slot_extra_body, dict) and slot_extra_body:
        if isinstance(caller_extra_body, dict):
            return {**slot_extra_body, **caller_extra_body}
        if caller_extra_body:
            return caller_extra_body
        return dict(slot_extra_body)
    return caller_extra_body


def _agent_cache_opts(agent: Any) -> tuple[Any, Any]:
    """The live agent's ``(_cache_disabled, _cache_ttl)``; ``(None, None)`` without an agent."""
    if agent is None:
        return None, None
    return getattr(agent, "_cache_disabled", None), getattr(agent, "_cache_ttl", None)


def _with_cache_disabled(runtime: dict[str, Any], cache_disabled: Any) -> dict[str, Any]:
    """Pin the live agent's cache disable onto a runtime snapshot (None is a no-op)."""
    if cache_disabled is None:
        return runtime
    return {**runtime, "_cache_disabled": cache_disabled}


def _maybe_apply_moa_cache_control(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
    *,
    cache_disabled: bool | None = None,
    cache_ttl: str | None = None,
) -> list[dict[str, Any]]:
    """Apply cache_control to an advisor/aggregator request when its route honors it.

    Same policy/marker helpers as the main loop; MoA has no static prefix so the
    legacy system-and-3 fallback is used. ``cache_disabled`` is stamped onto the
    stub so ``cache_ttl: off`` is honored; ``cache_ttl`` is clamped per destination.
    Returns the messages unchanged on any error.
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
        provider = runtime.get("provider") or ""
        model = runtime.get("model") or ""
        # blank_cache_policy_stub is the only sanctioned stub (carries _cache_disabled).
        should_cache, native_layout = anthropic_prompt_cache_policy(
            blank_cache_policy_stub(cache_disabled),
            provider=provider,
            base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "",
            model=model,
        )
        if not should_cache:
            return messages
        return apply_anthropic_cache_control(
            messages,
            # None → "5m"; cache-disabled routes already returned above.
            cache_ttl=effective_cache_ttl(cache_ttl, provider=provider, model=model),
            native_anthropic=native_layout,
            # Envelope routes reject part-level markers in tool_result.content[].
            tool_part_markers=envelope_tool_part_cache_markers_supported(
                provider, runtime.get("base_url") or ""
            ),
        )
    except Exception as exc:  # pragma: no cover - decoration must never break a call
        logger.debug("MoA cache_control decoration skipped: %s", exc)
        return messages


def _price_reference_response(
    response: Any, slot: dict[str, Any], runtime: dict[str, Any]
) -> tuple[Any, Any, str | None, str | None]:
    """Normalize a reference's usage with the slot's OWN provider/api_mode and price it
    at its own rate (hence fan-out cost is summed in dollars). Never raises."""
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
    """Call one reference model; return ``(label, text, accounting)``. Never raises:
    a failed reference becomes a labelled ``[failed: …]`` note. Runs in a thread pool."""
    label = _slot_label(slot)
    runtime = _slot_runtime(slot)
    trace_fields = {
        "model": slot.get("model"),
        "provider": runtime.get("provider") or slot.get("provider"),
        "temperature": temperature,
    }
    # The advisory view already stripped the agent's system prompt; this is the only one.
    messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
    try:
        # Trim to THIS model's window (advisors may be smaller than the aggregator).
        trimmed = _trim_messages_for_reference(
            messages,
            slot,
            runtime,
            reserve_output_tokens=max_tokens,
            context_length_cache=context_length_cache,
        )
        # The advisory view is append-only across iterations, so cache_control lets
        # iteration N+1 replay N's cached prefix.
        trimmed = _maybe_apply_moa_cache_control(
            trimmed, _with_cache_disabled(runtime, cache_disabled), cache_ttl=cache_ttl
        )
        # Per-slot max_tokens beats the preset-level reference_max_tokens.
        slot_max_tokens = slot.get("max_tokens")
        extra_headers = None
        # Normalize provider aliases (github, github-copilot, ...) via the canonical table.
        from agent.auxiliary_client import _normalize_aux_provider

        if _normalize_aux_provider(str(runtime.get("provider") or "")) in ("copilot", "copilot-acp"):
            # Copilot gates premium models on request attribution; MoA fan-out serves the
            # user's current turn, so mirror the main agent's x-initiator header.
            extra_headers = {"x-initiator": "user"}
        response = call_llm(
            task="moa_reference",
            messages=trimmed,
            temperature=temperature,
            max_tokens=slot_max_tokens if slot_max_tokens is not None else max_tokens,
            timeout=reference_timeout,
            reasoning_config=_slot_reasoning_config(slot),
            extra_headers=extra_headers,
            **runtime,
        )
        output_text = _extract_text(response) or "(empty response)"
        acct = _RefAccounting(
            *_price_reference_response(response, slot, runtime),
            messages=trimmed,
            output=output_text,
            **trace_fields,
        )
        return label, output_text, acct
    except Exception as exc:
        logger.warning("MoA reference model %s failed: %s", label, exc)
        note = f"[failed: {exc}]"
        return label, note, _RefAccounting(
            CanonicalUsage(), messages=messages, output=note, **trace_fields,
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
    """Trim an advisory request to fit a reference model's context window.

    Budget = window − ``reserve_output_tokens`` (or a default) − a safety fraction.
    Drops the OLDEST frames after the system prompt, always keeping a user-first
    body and the trailing user turn plus one preceding turn (even if still over
    budget). ``context_length_cache`` memoizes the window per (provider, model);
    unresolvable windows leave messages unchanged.
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
    has_cache = isinstance(context_length_cache, dict)
    if has_cache and cache_key in context_length_cache:
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
            logger.debug("MoA reference context-length resolution failed for %s", _slot_label(slot))
            context_length = None
        if has_cache:
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

    has_system = messages[0].get("role") == "system"
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
            _slot_label(slot), estimated, budget, context_length, reserve, dropped,
        )
    return trimmed


_REFERENCE_POLL_INTERVAL_S = 5.0

# Sentinel for a reference aborted by user interrupt; the facade must never cache it.
_INTERRUPTED_REFERENCE_NOTE = "[skipped: interrupted by user]"


def _placeholder_output(slot: dict[str, Any], note: str) -> tuple[str, str, Any]:
    """A reference-output tuple for a slot that was not (fully) run: zero accounting."""
    return _slot_label(slot), note, _RefAccounting(CanonicalUsage())


def _settle_interrupted(
    futures: dict[Any, int],
    results: list,
    reference_models: list[dict[str, Any]],
    late_accounting_sink: Any,
) -> None:
    """Fill every unfinished slot after a user interrupt: cancel never-dispatched
    futures (nothing billed), keep real output of ones that just finished, and hand
    running ones (cannot be killed, WILL bill) to ``late_accounting_sink``."""
    for future, idx in futures.items():
        if results[idx] is not None:
            continue
        slot = reference_models[idx]
        if future.cancel():
            results[idx] = _placeholder_output(slot, _INTERRUPTED_REFERENCE_NOTE)
        elif future.done():
            results[idx] = future.result()
        else:
            results[idx] = _placeholder_output(slot, _INTERRUPTED_REFERENCE_NOTE)
            if late_accounting_sink is not None:
                def _record_late(f: Any, _label: str = results[idx][0]) -> None:
                    try:
                        _lbl, _txt, _acct = f.result()
                    except Exception:  # pragma: no cover - defensive
                        return
                    try:
                        late_accounting_sink(_label, _acct)
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("MoA: late accounting sink failed for %s", _label)
                future.add_done_callback(_record_late)


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
    """Fan out all reference models in parallel; ``(label, text, _RefAccounting)`` per
    slot in ``reference_models`` order.

    ``provider == "moa"`` slots are skipped with a note (recursion guard).
    ``progress_callback(refs_done, refs_total, label)`` fires per completion. With
    *agent*, the wait polls every ``_REFERENCE_POLL_INTERVAL_S`` so a user interrupt
    can abort it; in-flight calls cannot be killed and bill via ``late_accounting_sink``.
    """
    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures: dict[Any, int] = {}
    # Propagate the turn's contextvars (approval callbacks, Nous conversation tag).
    from tools.thread_context import propagate_context_to_thread

    total = len(reference_models)
    completed = 0
    executor = ThreadPoolExecutor(max_workers=min(_MAX_REFERENCE_WORKERS, total))
    interrupted = False
    # Shared per-fan-out context-length cache (dict get/set is GIL-atomic).
    ctx_len_cache: dict[tuple[str, str], int | None] = {}
    # Agent's cache disable + TTL for every advisor request; clamped in the decorator.
    cache_disabled, cache_ttl = _agent_cache_opts(agent)
    try:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = _placeholder_output(
                    slot, "[skipped: MoA presets cannot recursively reference MoA]"
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
                    context_length_cache=ctx_len_cache,
                    cache_disabled=cache_disabled,
                    cache_ttl=cache_ttl,
                )
            ] = idx

        # Collect every reference (no early exit except a user interrupt).
        pending = set(futures)
        while pending:
            done, pending = _futures_wait(pending, timeout=_REFERENCE_POLL_INTERVAL_S)
            for future in done:
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress_callback is not None:
                    try:
                        progress_callback(completed, total, _slot_label(reference_models[idx]))
                    except Exception as exc:  # pragma: no cover - display must never break
                        logger.debug("MoA progress_callback failed: %s", exc)
            if not pending:
                break
            if agent is not None and getattr(agent, "_interrupt_requested", False):
                interrupted = True
                break

        if interrupted:
            _settle_interrupted(futures, results, reference_models, late_accounting_sink)
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

    Plain user/assistant TEXT turns only: system prompt dropped, tool_calls rendered
    inline, tool results folded into the preceding assistant turn as previews (no
    tool-role messages / tool_calls arrays, so strict providers do not 400). Always
    ends on a ``user`` turn (Anthropic treats a trailing assistant turn as prefill)
    by APPENDING a synthetic request. The aggregator always gets the full transcript.
    """
    rendered: list[dict[str, Any]] = []
    last_user_content: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        # Decorated (cache_control parts) and undecorated transcripts must yield a
        # byte-identical view so the advisory prefix stays cache-stable.
        text = flatten_message_text(content)

        if role == "user":
            if not text.strip() and isinstance(content, list) and content:
                # Image-only turn: empty user messages are rejected by strict providers
                # and skipping would break alternation.
                text = "[user sent non-text content (e.g. an image attachment)]"
            if not text.strip():
                # Genuinely empty user turn: strict providers 400 on it; safe to drop.
                continue
            last_user_content = text
            rendered.append({"role": "user", "content": text})
        elif role == "assistant":
            parts = [text.strip()] if text.strip() else []
            calls_text = _render_tool_calls(msg.get("tool_calls"))
            if calls_text:
                parts.append(calls_text)
            # Empty assistant turns (no text, no calls) carry nothing advisory.
            if parts:
                rendered.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            # Fold the tool result into the preceding assistant turn as text.
            block = f"[tool result: {_truncate_tool_result(text)}]"
            if rendered and rendered[-1].get("role") == "assistant":
                rendered[-1]["content"] = rendered[-1]["content"] + "\n" + block
            else:
                # No assistant turn to attach to (e.g. a leading tool result).
                rendered.append({"role": "assistant", "content": block})
        # system and any other role are ignored.

    # Anthropic rejects trailing assistant prefill: end on a synthetic user request.
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
    """Assistant text of a completed response: transport-normalized, else ``choices[0]``."""
    try:
        transport = get_transport("chat_completions")
        if transport is None:
            raise RuntimeError("chat_completions transport unavailable")
        text = (transport.normalize_response(response).content or "").strip()
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
    """Read an optional preset temperature; None (absent/empty/null) = provider default."""
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
    return text.lstrip().lower().startswith(("[failed:", "[skipped:"))


def _split_references(
    reference_outputs: list[tuple[str, str, Any]],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """``(successful outputs, failed labels)``; accounting payloads are preserved."""
    successful = [o for o in reference_outputs if not _is_failed_reference(o[1])]
    failed = [label for label, text, _acct in reference_outputs if _is_failed_reference(text)]
    return successful, failed


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


def _slot_labels(slots: list[dict[str, Any]]) -> str:
    return ", ".join(_slot_label(slot) for slot in slots)


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
    ``reference_max_tokens`` caps ONLY the fan-out (capping the aggregator truncated
    long syntheses). ``agent`` makes the fan-out interruptible.
    """
    reference_models = [slot for slot in reference_models if slot.get("enabled", True)]
    reference_outputs = _run_references_parallel(
        reference_models,
        _reference_messages(api_messages),
        temperature=temperature,
        max_tokens=reference_max_tokens,
        reference_timeout=reference_timeout,
        agent=agent,
    )
    successful_outputs, failed_labels = _split_references(reference_outputs)

    # 'full' privacy mode also redacts advisor text before it reaches the synthesizer.
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
        return (
            "[Mixture of Agents context — all reference models failed. "
            "Proceeding without aggregated guidance.]\n"
            f"References: {_slot_labels(reference_models)}\n\n"
            f"{degraded or '[Reference models unavailable]'}"
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
    cache_disabled, cache_ttl = _agent_cache_opts(agent)
    try:
        # Same cache_control decoration as the advisor calls; this synthesis call is
        # a third independent MoA call path that otherwise re-bills its full input.
        agg_messages = _maybe_apply_moa_cache_control(
            [{"role": "user", "content": synth_prompt}],
            _with_cache_disabled(agg_runtime, cache_disabled),
            cache_ttl=cache_ttl,
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

    return (
        "[Mixture of Agents context — use this as private guidance for the "
        "normal Hermes agent loop. You may call tools, continue reasoning, or "
        "finish normally.]\n"
        f"Aggregator: {agg_label}\n"
        f"References: {_slot_labels(reference_models)}\n\n"
        f"{(synthesis or joined).strip()}"
    )


def _completed_response_as_stream_chunk(response: Any) -> Any:
    """Adapt a completed response into one delta stream chunk (facade boundary only)."""
    choices = getattr(response, "choices", None)
    first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    message = getattr(first_choice, "message", None)
    raw_tool_calls = getattr(message, "tool_calls", None)
    tool_call_deltas = None
    if isinstance(raw_tool_calls, (list, tuple)) and raw_tool_calls:
        tool_call_deltas = [
            SimpleNamespace(
                index=getattr(tc, "index", index),
                id=getattr(tc, "id", None),
                type=getattr(tc, "type", None) or "function",
                function=SimpleNamespace(
                    name=getattr(getattr(tc, "function", None), "name", None),
                    arguments=getattr(getattr(tc, "function", None), "arguments", None),
                ),
            )
            for index, tc in enumerate(raw_tool_calls)
        ]
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

    The block varies per iteration; appending keeps ``[system][task][tool-history]``
    cache-stable. A trailing user turn is merged in place (string, or a new text part
    AFTER the cache_control-marked part); otherwise a user message is appended (two
    consecutive user turns would be rejected by strict providers).
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
    """Exact inverse of ``_attach_reference_guidance`` (the three attach shapes), so a
    cache breakpoint never lands on the turn-varying guidance. Inputs are not mutated."""
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
        return [*messages[:-1], {**last, "content": content[: -len(suffix)]}]
    if isinstance(content, list) and content:
        last_part = content[-1]
        if isinstance(last_part, dict) and last_part.get("type", "text") == "text":
            text = last_part.get("text") or ""
            if text == suffix or text == guidance_text:
                # Attach shape (b): guidance rode as its own trailing part. Guidance as
                # the only content drops the whole message (mirrors shape c).
                if len(content) == 1:
                    return list(messages[:-1])
                return [*messages[:-1], {**last, "content": list(content[:-1])}]
            if text.endswith(suffix):
                new_part = {**last_part, "text": text[: -len(suffix)]}
                return [*messages[:-1], {**last, "content": [*content[:-1], new_part]}]
    return messages


class MoAChatCompletions:
    """OpenAI-chat-compatible facade where the aggregator is the acting model.

    ``reference_callback(event, **kwargs)`` is an optional best-effort display hook
    (events: ``moa.reference``, ``moa.progress``, ``moa.phase``, ``moa.aggregating``;
    kwargs per ``_RELAY_EVENTS``). ``agent`` is the owning AIAgent; it lets the
    fan-out check ``_interrupt_requested``.
    """

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.preset_name = preset_name or "default"
        self.reference_callback = reference_callback
        self._agent = agent
        # Reference cache keyed on the advisory-view signature: new state = MISS
        # (references re-run), identical state = HIT (no re-run, no re-emit).
        self._ref_cache_key: tuple | None = None
        self._ref_cache_outputs: list[tuple[str, str, Any]] = []
        # Fan-out spend awaiting consume_reference_usage (nothing deposited on a HIT so
        # spend counts once); the lock guards late-accounting callbacks on worker threads.
        self._pending_reference_usage: Any = CanonicalUsage()
        self._pending_reference_cost: Any = None
        self._accounting_lock = threading.Lock()
        # Real aggregator slot so cost accounting prices the acting turn at its model.
        self.last_aggregator_slot: Any = None
        # Full-turn trace parts from a cache-MISS create(), flushed by consume_and_save_trace.
        self._pending_trace: Any = None
        # Per-advisor metrics for observability hooks; NOT consumed (post_api_request
        # fires on a different branch than consume_and_save_trace).
        self._last_reference_metrics: Any = None
        # every_n cadence state, scoped to one USER TURN so iteration 1 is on-cadence.
        self._fanout_iteration_count = 0
        self._fanout_turn_sig: str | None = None
        self._fanout_last_state_sig: str | None = None
        # Normalized moa.privacy_filter mode ('' | 'display' | 'full'), refreshed per create().
        self._privacy_mode: str = ""

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending fan-out ``(CanonicalUsage, cost_usd_or_None)`` and reset both
        (so a streaming retry re-entering accounting cannot double-count)."""
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
        """Done-callback for abandoned (still billing) futures: fold their real spend in."""
        if not isinstance(accounting, _RefAccounting):
            return
        self._fold_pending_accounting(*_sum_reference_accounting([(label, "", accounting)]))
        logger.debug("MoA: recorded late accounting for interrupted reference %s", label)

    def _fold_pending_accounting(self, usage: Any, cost: Any) -> None:
        """Add (never overwrite) fan-out spend so late interrupted-reference deposits survive."""
        with self._accounting_lock:
            self._pending_reference_usage = (self._pending_reference_usage or CanonicalUsage()) + usage
            if cost is not None:
                self._pending_reference_cost = (self._pending_reference_cost or 0) + cost

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn trace to disk (no-op when nothing is pending).

        ``aggregator_output_fallback`` is the caller's resolved acting text for the
        streaming path (not capturable at ``create()`` time). Never raises.
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
        """Run the advisor fan-out and return the exact aggregator request, which the
        loop measures before its compression gate and hands back to ``create()``."""
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

    def _plan_aggregator_cache(
        self,
        agg_messages: list[dict[str, Any]],
        tools: Any,
        guidance: Any,
        agg_runtime: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any]:
        """Cache-breakpoint the aggregator request for its destination.

        Guidance is peeled before planning and re-attached after so a breakpoint never
        lands on the turn-varying block. Any error → undecorated request (warning, not
        debug: this is the aggregator's ONLY decoration path).
        """
        try:
            from agent.agent_runtime_helpers import plan_cache_sections_for_destination

            planning_messages = agg_messages
            if guidance:
                planning_messages = peel_reference_guidance(agg_messages, str(guidance))
            # Tri-state cache_disabled: facades built via __new__ have no _agent; forcing
            # False would suppress the planner's config fallback.
            _agent = getattr(self, "_agent", None)
            cache_disabled, cache_ttl = _agent_cache_opts(_agent)
            agg_messages, tools = plan_cache_sections_for_destination(
                planning_messages,
                tools,
                provider=agg_runtime.get("provider") or "",
                base_url=agg_runtime.get("base_url") or "",
                api_mode=agg_runtime.get("api_mode") or "",
                model=agg_runtime.get("model") or "",
                cache_disabled=cache_disabled,
                # Agent TTL + stable system prefix so MoA does not regress 1h → 5m.
                cache_ttl=cache_ttl,
                static_system_prefix=getattr(_agent, "_cached_system_prompt_static", None),
            )
            if guidance:
                _attach_reference_guidance(agg_messages, str(guidance))
        except Exception as exc:  # pragma: no cover - cache planning must not block MoA
            logger.warning(
                "MoA aggregator cache plan failed — sending undecorated "
                "request (cache misses expected): %s", exc,
            )
        return agg_messages, tools

    def _call_prepared_aggregator(
        self, prepared: dict[str, Any], api_kwargs: dict[str, Any]
    ) -> Any:
        """Send an already prepared MoA aggregator request exactly once."""
        aggregator = prepared["aggregator"]
        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_runtime = _slot_runtime(aggregator)
        agg_messages, tools = self._plan_aggregator_cache(
            prepared["messages"], api_kwargs.get("tools"), prepared.get("guidance"), agg_runtime
        )
        # Trace the exact aggregator INPUT (persisted copy redacted; live input raw).
        if self._pending_trace is not None:
            self._pending_trace["aggregator_input_messages"] = (
                _redact_trace_messages([dict(m) for m in agg_messages])
                if getattr(self, "_privacy_mode", "")
                else agg_messages
            )
            self._pending_trace["aggregator_label"] = _slot_label(aggregator)
        # stream=True returns the RAW token stream (consumer reassembles + retries);
        # the non-streaming path forwards no stream/stream_options/timeout.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = api_kwargs.get("stream_options") or {"include_usage": True}
            # The consumer's stream-read timeout must govern the aggregator stream.
            if api_kwargs.get("timeout") is not None:
                stream_kwargs["timeout"] = api_kwargs["timeout"]
        # Pop the runtime's extra_body so the explicit kwarg never collides with **agg_runtime.
        agg_extra_body = _merge_slot_extra_body(
            agg_runtime.pop("extra_body", None), api_kwargs.get("extra_body"),
        )
        agg_response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=prepared["aggregator_temperature"],
            max_tokens=api_kwargs.get("max_tokens"),
            tools=tools,
            extra_body=agg_extra_body,
            # Same reasoning policy as the direct create() path.
            reasoning_config=_aggregator_reasoning_config(aggregator),
            **stream_kwargs,
            **agg_runtime,
        )
        # Streaming output lands as the turn's assistant message; the trace marks it.
        if self._pending_trace is not None:
            self._pending_trace["aggregator_streamed"] = stream
            output = None
            if not stream:
                try:
                    output = _extract_text(agg_response)
                except Exception:  # pragma: no cover - defensive
                    output = None
            self._pending_trace["aggregator_output"] = output
        if stream and hasattr(agg_response, "choices"):
            # Some adapters (openai-codex Responses) return a completed response even
            # when streaming was requested; hand the loop a one-chunk iterator.
            return iter((_completed_response_as_stream_chunk(agg_response),))
        return agg_response

    def _fanout_cache_key(
        self,
        preset: dict[str, Any],
        ref_messages: list[dict[str, Any]],
        reference_models: list[dict[str, Any]],
    ) -> tuple:
        """Turn-scoped reference cache key per the preset's fan-out cadence.

        "user_turn" (default) hashes only the prefix up to the LAST USER message, so
        later tool iterations are HITs. "per_iteration" re-runs whenever the advisory
        view changes. "every_n:<N>": iteration 1 of a turn, then every Nth; in-between
        iterations return the pinned last on-cadence key (HIT: no calls, no re-emit).
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
            for i in range(len(ref_messages) - 1, -1, -1):
                m = ref_messages[i]
                if m.get("role") == "user" and m.get("content") != _ADVISORY_INSTRUCTION:
                    turn_prefix = ref_messages[: i + 1]
                    break
            if fanout_mode == "user_turn":
                sig_messages = turn_prefix

        # every_n bookkeeping: advance the counter only when the advisory STATE changed
        # (a streaming retry must not consume a cadence slot); reset on a new turn prefix.
        if every_n >= 2:
            turn_sig = _hash_messages(turn_prefix)
            if turn_sig != self._fanout_turn_sig:
                self._fanout_turn_sig = turn_sig
                self._fanout_iteration_count = 0
                self._fanout_last_state_sig = None
            state_sig = _hash_messages(ref_messages)
            if state_sig != self._fanout_last_state_sig:
                self._fanout_last_state_sig = state_sig
                self._fanout_iteration_count += 1
            # Iteration 1 is on-cadence; then every Nth iteration after it.
            on_cadence = (self._fanout_iteration_count - 1) % every_n == 0
            if not on_cadence and self._ref_cache_outputs:
                return self._ref_cache_key

        return (
            self.preset_name,
            _hash_messages(sig_messages),
            tuple(_slot_label(s) for s in reference_models),
        )

    def _run_fanout(
        self,
        preset: dict[str, Any],
        ref_messages: list[dict[str, Any]],
        reference_models: list[dict[str, Any]],
        aggregator: dict[str, Any],
        aggregator_temperature: Any,
        cache_key: tuple,
    ) -> list[tuple[str, str, Any]]:
        """Cache-MISS path of ``create``: run the advisors, account, trace and emit.

        A preset MAY cap ADVISOR output (dominant MoA latency); the acting aggregator
        is never capped. None timeout = inherit auxiliary.moa_reference.timeout.
        """
        raw_reference_timeout = preset.get("reference_timeout")

        def _progress(done: int, total: int, label: str) -> None:
            self._emit("moa.progress", refs_done=done, refs_total=total, label=label)

        reference_outputs = _run_references_parallel(
            reference_models,
            ref_messages,
            temperature=_preset_temperature(preset, "reference_temperature"),
            max_tokens=preset.get("reference_max_tokens"),
            progress_callback=_progress,
            reference_timeout=float(raw_reference_timeout) if raw_reference_timeout else None,
            agent=self._agent,
            late_accounting_sink=self._record_late_reference_accounting,
        )
        if any(text == _INTERRUPTED_REFERENCE_NOTE for _lbl, text, _acct in reference_outputs):
            # An interrupted fan-out is a partial snapshot: never cache it (a HIT
            # would replay placeholder notes every iteration).
            self._ref_cache_key = None
            self._ref_cache_outputs = []
        else:
            self._ref_cache_key = cache_key
            self._ref_cache_outputs = list(reference_outputs)
        # Fold advisor spend into accounting exactly once per turn.
        self._fold_pending_accounting(*_sum_reference_accounting(reference_outputs))
        # Stash the fan-out for trace persistence (aggregator parts filled in later).
        # Traces are persisted, so ANY active privacy mode redacts them.
        privacy_mode = self._privacy_mode
        if privacy_mode:
            trace_refs = [
                (label, _redact_reference_text(text), _redact_trace_accounting(acct))
                for label, text, acct in reference_outputs
            ]
        else:
            trace_refs = list(reference_outputs)
        self._pending_trace = {
            "preset": self.preset_name,
            "reference_outputs": trace_refs,
            "aggregator_slot": aggregator,
            "aggregator_temperature": aggregator_temperature,
        }
        # Derived from the privacy-redacted trace_refs.
        try:
            from agent.moa_trace import slot_metrics

            self._last_reference_metrics = [
                slot_metrics(acct, label, output=text) for label, text, acct in trace_refs
            ]
        except Exception as exc:  # pragma: no cover - never break a turn
            logger.debug("MoA reference metrics render failed: %s", exc)
            self._last_reference_metrics = None

        # Surface each answer BEFORE the aggregator acts; the cache keeps RAW text.
        ref_count = len(reference_outputs)
        for idx, (label, text, _accounting) in enumerate(reference_outputs, start=1):
            self._emit(
                "moa.reference",
                index=idx,
                count=ref_count,
                label=label,
                text=_redact_reference_text(text) if privacy_mode else text,
            )
        if ref_count:
            # Phase transition: fan-out complete, aggregator about to act.
            agg_label = _slot_label(aggregator)
            self._emit(
                "moa.phase",
                phase="aggregator",
                refs_done=ref_count,
                refs_total=ref_count,
                aggregator=agg_label,
            )
            self._emit("moa.aggregating", aggregator=agg_label, ref_count=ref_count)
        return reference_outputs

    def _build_guidance(
        self,
        reference_outputs: list[tuple[str, str, Any]],
        aggregator: dict[str, Any],
        degraded_reference_policy: str,
    ) -> str | None:
        """Render the reference block attached to the aggregator prompt (None = nothing)."""
        successful_outputs, failed_labels = _split_references(reference_outputs)
        # 'full' privacy mode redacts advisor text reaching the AGGREGATOR too.
        agg_refs = (
            _redact_reference_outputs(successful_outputs)
            if self._privacy_mode == "full"
            else successful_outputs
        )
        degraded = _degraded_notice(failed_labels, degraded_reference_policy)
        header = (
            "[Mixture of Agents reference context]\n"
            f"Preset: {self.preset_name}\n"
            f"Aggregator/acting model: {_slot_label(aggregator)}\n"
        )
        if reference_outputs and not successful_outputs:
            # Every reference failed: the aggregator acts alone (loud policy → notice).
            logger.warning(
                "MoA: all %d reference(s) failed — acting aggregator-alone "
                "without reference guidance",
                len(reference_outputs),
            )
            if degraded:
                return (
                    f"{header}\n"
                    "All reference models failed this turn — no advisory "
                    "guidance is available. Act on your own judgment.\n\n"
                    f"{degraded}"
                )
            return None
        if agg_refs or degraded:
            return (
                f"{header}"
                f"References: {', '.join(label for label, _, _ in agg_refs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{_join_reference_outputs(agg_refs, degraded)}"
            )
        return None

    def create(self, **api_kwargs: Any) -> Any:
        prepared_request = api_kwargs.pop("_moa_prepared_request", None)
        if prepared_request is not None:
            if not isinstance(prepared_request, dict):
                raise TypeError("_moa_prepared_request must be a dict")
            return self._call_prepared_aggregator(prepared_request, api_kwargs)

        preset, moa_raw = _resolve_preset_cached(self.preset_name)
        # Remembered on self so _call_prepared_aggregator redacts the trace consistently.
        self._privacy_mode = _moa_privacy_mode(moa_raw)
        messages = list(api_kwargs.get("messages") or [])
        # A disabled preset = "use the aggregator directly".
        reference_models = [
            slot for slot in (preset.get("reference_models") or [])
            if slot.get("enabled", True)
        ] if preset.get("enabled", True) else []
        aggregator = preset.get("aggregator") or {}
        # The MoA path's virtual model/provider have no pricing entry; expose the real slot.
        self.last_aggregator_slot = dict(aggregator) if aggregator else None
        # None = provider default (see _preset_temperature); the acting agent's own
        # temperature applies to the aggregator (the acting model).
        aggregator_temperature = _preset_temperature(preset, "aggregator_temperature")
        if aggregator_temperature is None and api_kwargs.get("temperature") is not None:
            aggregator_temperature = api_kwargs.get("temperature")

        ref_messages = _reference_messages(messages)
        cache_key = self._fanout_cache_key(preset, ref_messages, reference_models)
        if cache_key == self._ref_cache_key and self._ref_cache_outputs:
            # HIT: already ran and accounted. Do NOT zero pending totals (a late
            # interrupted reference may have deposited) and no trace (not a new turn).
            reference_outputs = list(self._ref_cache_outputs)
            self._pending_trace = None
        else:
            reference_outputs = self._run_fanout(
                preset, ref_messages, reference_models, aggregator, aggregator_temperature, cache_key
            )

        agg_messages = [dict(m) for m in messages]
        guidance = self._build_guidance(
            reference_outputs, aggregator, str(preset.get("degraded_reference_policy") or "loud")
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
    """OpenAI-client-shaped wrapper: ``client.chat.completions`` is a ``MoAChatCompletions``.

    The accounting/trace surface (``consume_reference_usage``, ``last_aggregator_slot``,
    ``consume_and_save_trace``, ``last_reference_metrics``) is delegated to the facade.
    """

    _DELEGATED = (
        "consume_reference_usage", "last_aggregator_slot",
        "consume_and_save_trace", "last_reference_metrics",
    )

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(
            preset_name, reference_callback=reference_callback, agent=agent,
        )

    def __getattr__(self, name: str) -> Any:
        if name in MoAClient._DELEGATED:
            return getattr(self.chat.completions, name)
        raise AttributeError(name)


# Relay table: event -> (primary kwarg, secondary kwarg or None, {cb kwarg: emit kwarg}).
# The callback signature is ``cb(event, label, text, None, **moa_*)``.
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
    """Single construction point for ``MoAClient``: a bare ``MoAClient(preset)`` would
    drop the ``reference_callback`` relay and silence display events for the session.
    The relay reads ``agent.tool_progress_callback`` at emit time."""
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

    # ``agent`` lets the fan-out wait be aborted on a user interrupt.
    return MoAClient(resolved_preset, reference_callback=_moa_reference_relay, agent=agent)
