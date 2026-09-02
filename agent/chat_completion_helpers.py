"""Helper functions for the chat-completions code path.

Extracted from :class:`AIAgent` for cleanliness — bodies of the
non-streaming API call, request kwargs builder, assistant-message
materializer, provider-fallback activator, max-iterations handler,
and per-turn resource cleanup.

Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  :class:`AIAgent` keeps thin forwarder methods so call
sites unchanged.  Symbols that tests patch on ``run_agent`` (e.g.
``cleanup_vm`` / ``cleanup_browser`` in
``test_zombie_process_cleanup.py``) are resolved through
:func:`_ra` so the patch contract is preserved.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout
from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH
from agent.error_classifier import (
    FailoverReason,
    PROVIDER_STREAM_NON_JSON_ERROR_CODE,
)
from agent.errors import EmptyStreamError
from agent.fast_mode import effective_request_overrides
from agent.turn_context import substitute_api_content
from agent.gemini_native_adapter import is_native_gemini_base_url
from agent.model_metadata import is_local_endpoint
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import (
    _sanitize_surrogates,
    _repair_tool_call_arguments,
)
from agent.reasoning_summaries import separate_glued_reasoning_blocks
from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current
from tools.terminal_tool import is_persistent_env
from utils import base_url_host_matches, base_url_hostname, env_float, env_int

logger = logging.getLogger(__name__)
_OPENROUTER_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}
_PROVIDER_STREAM_ERROR_FINISH_REASONS = {"error", "error_finish"}
_PROVIDER_STREAM_SSE_FIELDS = {"event", "data", "id", "retry"}
_PROVIDER_STREAM_ERROR_TEXT_LIMIT = 4096

# When the fallback chain is fully exhausted on a non-rate-limit failure
# (e.g. every provider returns a non-retryable client error like HTTP 400),
# arm a short cooldown so the NEXT turn's restore_primary_runtime stays gated
# and does not reset _fallback_index=0 to replay the entire chain again.
# Without this, a client/gateway that re-submits immediately would re-marshal
# the full (potentially 80k-token) context once per provider every turn and
# can drive a constrained host into memory/swap exhaustion.  Rate-limit /
# billing reasons keep their own 60s cooldown (set above); this is the
# narrower non-rate-limit case.  See issue #24996.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0


def _context_thread_target(callback):
    """Bind a no-argument thread target to the caller's ContextVars."""
    context = contextvars.copy_context()
    return lambda: context.run(callback)


def _join_worker_for_relay_teardown(worker, *, label: str) -> None:
    """Bounded worker join before raising InterruptedError (#81521).

    Raising immediately lets turn teardown (finish_logical_calls /
    end_turn / close_session) race a still-open Relay physical LLM scope
    and corrupt the LIFO stack — "scope handle is not at the top of the
    stack" → CLI EIO / redraw storm.  Only joins when Relay managed
    execution is actually live: when no Relay consumers are registered
    there is no scope to unwind, and the join would just delay interrupt
    detection (tests/run_agent/test_interrupt_propagation.py).
    """
    try:
        from agent import relay_runtime

        runtime = relay_runtime.get_runtime(create=False)
        if runtime is None or not runtime.managed_execution_enabled():
            return
    except Exception:
        return
    worker.join(timeout=2.0)
    if worker.is_alive():
        logger.warning(
            "%s worker still alive after interrupt abort (2.0s join "
            "timeout); Relay teardown will best-effort drain orphaned "
            "scopes (#81521).",
            label,
        )


def _ra():
    """Lazy ``run_agent`` reference.

    Used to honor test patches like
    ``patch("run_agent.cleanup_vm")`` / ``patch("run_agent.cleanup_browser")``
    that target symbols imported into ``run_agent``'s namespace.
    """
    import run_agent
    return run_agent


class ProviderStreamError(Exception):
    """Provider encoded an API error as streaming content instead of an SDK error."""

    def __init__(
        self,
        *,
        status_code: Optional[int],
        body: dict,
        raw_text: str,
        headers: Any = None,
    ):
        self.status_code = status_code
        self.body = body
        self.raw_text = raw_text
        self.response = SimpleNamespace(headers=headers or {})
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        error_obj = self.body.get("error", {}) if isinstance(self.body, dict) else {}
        code = error_obj.get("code") if isinstance(error_obj, dict) else None
        message = error_obj.get("message") if isinstance(error_obj, dict) else None
        parts = ["Provider stream returned an error event"]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        if code:
            parts.append(str(code))
        text = " - ".join(parts)
        if message:
            text += f": {message}"
        return text


def _status_code_from_value(value: Any) -> Optional[int]:
    if isinstance(value, int) and 100 <= value < 600:
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:HTTP_STATUS/)?\b([1-5]\d\d)\b", value, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _status_code_from_payload(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    candidates = [
        payload.get("status_code"),
        payload.get("status"),
        payload.get("http_status"),
    ]
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        candidates.extend([
            error_obj.get("status_code"),
            error_obj.get("status"),
            error_obj.get("http_status"),
            error_obj.get("code"),
        ])
    candidates.append(payload.get("code"))

    for candidate in candidates:
        status_code = _status_code_from_value(candidate)
        if status_code is not None:
            return status_code
    return None


def _json_object_from_text(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return None
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _parse_provider_sse_events(text: str) -> list[dict]:
    """Parse provider text that looks like Server-Sent Events."""
    events: list[dict] = []
    current = {"event": None, "data": [], "comments": [], "fields": {}}

    def _has_event_data(event: dict) -> bool:
        return bool(
            event.get("event")
            or event.get("data")
            or event.get("comments")
            or event.get("fields")
        )

    def _flush_current():
        nonlocal current
        if _has_event_data(current):
            data_text = "\n".join(current["data"])
            status_candidates = list(current["comments"])
            for key in ("status", "status_code", "http_status"):
                if key in current["fields"]:
                    status_candidates.append(current["fields"][key])
            events.append({
                "event": current["event"],
                "data": data_text,
                "comments": list(current["comments"]),
                "fields": dict(current["fields"]),
                "status_code": next(
                    (
                        status
                        for status in (
                            _status_code_from_value(value)
                            for value in status_candidates
                        )
                        if status is not None
                    ),
                    None,
                ),
            })
        current = {"event": None, "data": [], "comments": [], "fields": {}}

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            _flush_current()
            continue
        if line.startswith(":"):
            current["comments"].append(line[1:].strip())
            continue

        field, sep, value = line.partition(":")
        if not sep:
            current["fields"][field.strip().lower()] = ""
            continue
        field = field.strip().lower()
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            current["event"] = value.strip()
        elif field == "data":
            current["data"].append(value)
        else:
            current["fields"][field] = value

    _flush_current()
    return events


def _provider_error_body(payload: dict, status_code: Optional[int]) -> dict:
    """Normalize common provider error payloads to OpenAI-style body.error."""
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            return payload
    else:
        payload = {}

    code = (
        payload.get("code")
        or payload.get("error_code")
        or payload.get("type")
        or (f"HTTP_{status_code}" if status_code else "provider_stream_error")
    )
    message = (
        payload.get("message")
        or payload.get("error_description")
        or payload.get("error")
        or "Provider stream returned an error event."
    )
    normalized_error = {"message": str(message)}
    if code:
        normalized_error["code"] = str(code)
    for key in ("request_id", "param", "type"):
        if payload.get(key):
            normalized_error[key] = payload[key]
    return {"error": normalized_error}


def _provider_stream_error_from_json_decode_error(
    error: json.JSONDecodeError,
    *,
    response: Any = None,
) -> ProviderStreamError:
    """Preserve plain-text SSE data rejected inside the OpenAI SDK.

    OpenAI-compatible providers occasionally send ``event: error`` with a
    non-JSON ``data:`` field.  The SDK raises from ``sse.json()`` before it can
    yield a completion chunk, but ``JSONDecodeError.doc`` still contains the
    provider's original message.
    """
    from agent.redact import redact_sensitive_text

    raw_text = str(getattr(error, "doc", "") or "").strip()
    safe_text = redact_sensitive_text(
        _sanitize_surrogates(raw_text),
        force=True,
    )
    safe_text = safe_text[:_PROVIDER_STREAM_ERROR_TEXT_LIMIT]
    message = safe_text or "Provider stream returned non-JSON SSE data."
    headers = getattr(response, "headers", None) if response is not None else None

    return ProviderStreamError(
        status_code=None,
        body=_provider_error_body(
            {
                "code": PROVIDER_STREAM_NON_JSON_ERROR_CODE,
                "message": message,
            },
            None,
        ),
        raw_text=safe_text,
        headers=headers,
    )


def _iter_provider_stream_chunks(stream, *, response: Any = None):
    """Yield SDK chunks while translating SDK-level SSE decode failures."""
    try:
        yield from stream
    except json.JSONDecodeError as error:
        stream_response = response() if callable(response) else response
        if stream_response is None:
            stream_response = getattr(stream, "response", None)
        raise _provider_stream_error_from_json_decode_error(
            error,
            response=stream_response,
        ) from error


def _payload_has_error_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("error"), (dict, str)):
        return True
    if payload.get("message") and (
        payload.get("code")
        or payload.get("error_code")
        or _status_code_from_payload(payload) is not None
    ):
        return True
    return False


def _provider_stream_text_may_be_sse(text: str) -> bool:
    """Return True while pending text still looks like an SSE control block."""
    stripped = (text or "").lstrip()
    if not stripped:
        return False

    lines = stripped.splitlines()
    trailing_newline = stripped.endswith(("\n", "\r"))
    saw_sse_field = False

    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r")
        if line == "":
            continue
        if line.startswith(":"):
            saw_sse_field = True
            continue

        field, sep, _value = line.partition(":")
        field_name = field.strip().lower()
        if sep and field_name in _PROVIDER_STREAM_SSE_FIELDS:
            saw_sse_field = True
            continue

        is_last_incomplete = index == len(lines) - 1 and not trailing_newline
        if is_last_incomplete and any(
            sse_field.startswith(field_name)
            for sse_field in _PROVIDER_STREAM_SSE_FIELDS
        ):
            return True
        return False

    return saw_sse_field


def _provider_stream_error_from_text(
    text: str,
    finish_reason: Optional[str],
    *,
    response: Any = None,
) -> Optional[ProviderStreamError]:
    """Convert provider-streamed error text into an exception for retry logic."""
    if not text:
        return None

    finish_reason_text = str(finish_reason or "").lower()
    has_error_finish = finish_reason_text in _PROVIDER_STREAM_ERROR_FINISH_REASONS
    if not has_error_finish:
        return None

    for event in _parse_provider_sse_events(text):
        event_name = str(event.get("event") or "").strip().lower()
        payload = _json_object_from_text(event.get("data") or "") or {}
        status_code = event.get("status_code") or _status_code_from_payload(payload)
        is_error_event = event_name == "error"
        is_http_error = status_code is not None and status_code >= 400
        is_error_payload = _payload_has_error_shape(payload)
        is_structured_error_event = is_error_event and (
            has_error_finish or is_http_error or is_error_payload
        )
        is_bare_error_finish_payload = (
            not is_error_event and has_error_finish and is_error_payload
        )

        if not (
            is_http_error
            or is_structured_error_event
            or is_bare_error_finish_payload
        ):
            continue

        headers = getattr(response, "headers", None) if response is not None else None
        return ProviderStreamError(
            status_code=status_code,
            body=_provider_error_body(payload, status_code),
            raw_text=text,
            headers=headers,
        )

    payload = _json_object_from_text(text)
    if payload is not None:
        status_code = _status_code_from_payload(payload)
        if has_error_finish or (status_code is not None and status_code >= 400):
            headers = getattr(response, "headers", None) if response is not None else None
            return ProviderStreamError(
                status_code=status_code,
                body=_provider_error_body(payload, status_code),
                raw_text=text,
                headers=headers,
            )

    if has_error_finish and text.strip():
        headers = getattr(response, "headers", None) if response is not None else None
        return ProviderStreamError(
            status_code=None,
            body=_provider_error_body({}, None),
            raw_text=text,
            headers=headers,
        )
    return None


def estimate_request_context_tokens(api_payload: Any) -> int:
    """Estimate context/load tokens from an API payload, dict or messages list.

    The stale-call detectors historically assumed a Chat Completions request:
    they pulled ``api_kwargs["messages"]`` and ran a cheap char/4 estimate.
    Codex / Responses API requests carry the conversational payload in
    ``input`` (with additional load in ``instructions`` and ``tools``), so the
    legacy estimator reported ~0 tokens for every Codex turn and the
    context-tier scaling never fired.

    This helper handles both shapes:
      - bare list -> treat as Chat Completions ``messages``
      - dict with ``messages`` -> Chat Completions (+ ``tools`` if present)
      - dict with ``input`` -> Responses API (+ ``instructions``/``tools``)
      - any other dict -> fall back to summing string values
    """

    def _chars(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        return len(str(value))

    def _message_chars(messages: Any) -> int:
        if not isinstance(messages, list):
            return _chars(messages)
        return sum(_chars(item) for item in messages)

    if isinstance(api_payload, list):
        return _message_chars(api_payload) // 4

    if isinstance(api_payload, dict):
        messages = api_payload.get("messages")
        if isinstance(messages, list):
            total_chars = _message_chars(messages)
            if "tools" in api_payload:
                total_chars += _chars(api_payload.get("tools"))
            return total_chars // 4

        if "input" in api_payload:
            total_chars = (
                _chars(api_payload.get("input"))
                + _chars(api_payload.get("instructions"))
                + _chars(api_payload.get("tools"))
            )
            return total_chars // 4

        return sum(_chars(value) for value in api_payload.values()) // 4

    return _chars(api_payload) // 4


def _is_openai_codex_backend(agent) -> bool:
    from agent.codex_responses_adapter import classify_responses_route

    return classify_responses_route(agent).is_codex_backend


def openai_codex_stale_timeout_floor(est_tokens: int) -> float:
    """Minimum wall-clock stale timeout for openai-codex by estimated context.

    Gateway/Telegram sessions routinely ship ~15–25k tokens of tools +
    instructions before the first user message. Subscription-backed Codex can
    legitimately spend several minutes in backend admission/prefill at that
    size; the generic 90s non-stream stale default aborts healthy calls. The
    floor engages above 10k estimated tokens so those gateway-scale payloads
    are covered; smaller requests keep the generic default.
    """
    if est_tokens > 100_000:
        return 1200.0
    if est_tokens > 50_000:
        return 900.0
    if est_tokens > 10_000:
        return 600.0
    return 0.0


def _validated_openrouter_provider_sort(raw_sort: Any) -> Optional[str]:
    """Return a normalized OpenRouter provider.sort value or None."""
    if not isinstance(raw_sort, str):
        return None
    sort_value = raw_sort.strip().lower()
    if not sort_value:
        return None
    if sort_value in _OPENROUTER_PROVIDER_SORT_VALUES:
        return sort_value
    logger.warning(
        "Ignoring invalid OpenRouter provider.sort value %r (allowed: %s)",
        raw_sort,
        ", ".join(sorted(_OPENROUTER_PROVIDER_SORT_VALUES)),
    )
    return None


def _provider_preferences_for_agent(agent) -> Dict[str, Any]:
    """Build the validated provider-routing object shared by request paths."""
    preferences: Dict[str, Any] = {}
    if agent.providers_allowed:
        preferences["only"] = agent.providers_allowed
    if agent.providers_ignored:
        preferences["ignore"] = agent.providers_ignored
    if agent.providers_order:
        preferences["order"] = agent.providers_order
    provider_sort = _validated_openrouter_provider_sort(agent.provider_sort)
    if provider_sort:
        preferences["sort"] = provider_sort
    if agent.provider_require_parameters:
        preferences["require_parameters"] = True
    if agent.provider_data_collection:
        preferences["data_collection"] = agent.provider_data_collection
    return preferences


def _prompt_cache_scope_for_agent(agent) -> "str | None":
    """Rotation-stable logical cache scope for *agent*, or None.

    Guarded-import wrapper over the never-raising
    ``agent.prompt_cache_scope.resolve_prompt_cache_scope_safe`` — the
    transports treat a None/empty value as "fall back to the physical
    session_id", so any resolution failure degrades to pre-#79017 behavior
    instead of blocking the request build.
    """
    try:
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe

        return resolve_prompt_cache_scope_safe(agent)
    except Exception:
        logger.debug("prompt-cache scope resolution failed", exc_info=True)
        return None


def _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs: dict) -> dict:
    """Merge Portal ``tags`` / ``session_id`` onto an Anthropic Messages kwargs dict.

    The Nous provider profile is only consulted by the OpenAI-wire transport;
    anthropic_messages callers must merge it themselves. Passes ``session_id``
    only — not ``provider_preferences`` (those become a top-level ``provider``
    routing object on the OpenAI wire). Never blocks a turn on tagging.
    """
    if getattr(agent, "provider", None) not in {"nous", "nous-portal", "nousresearch"}:
        return anthropic_kwargs
    try:
        from providers import get_provider_profile

        nous_profile = get_provider_profile("nous")
        if nous_profile is not None:
            anthropic_kwargs.setdefault("extra_body", {}).update(
                nous_profile.build_extra_body(
                    session_id=getattr(agent, "session_id", None)
                )
            )
    except Exception as exc:  # noqa: BLE001 — never block a turn on tagging
        logger.debug("Nous Portal extra_body merge failed: %s", exc)
    return anthropic_kwargs


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _estimate_chunk_bytes(chunk: Any) -> int:
    """Cheap per-chunk size estimate for the stream diagnostic counters.

    The previous implementation used ``len(repr(chunk))`` — a full recursive
    repr of a pydantic model on EVERY streaming chunk (5.5-8.8 µs each,
    ~20-30 ms of pure CPU on a 3,000-chunk response, in the hottest loop in
    the agent). The counter only feeds a retry-diagnostic log line, so an
    estimate based on the delta payload lengths is plenty (2.1-2.4 µs, ~3x
    cheaper, and independent of model/pydantic field count). Chat Completions
    chunks are sized from their delta content/reasoning/tool-argument strings
    plus a small framing constant; anything shape-unknown (Anthropic events,
    stub providers) falls back to a flat constant so `bytes` stays monotonic
    and roughly proportional to traffic.
    """
    size = 40  # SSE/JSON framing floor per chunk
    try:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                for attr in ("content", "reasoning_content", "reasoning"):
                    v = getattr(delta, attr, None)
                    if isinstance(v, str):
                        size += len(v)
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            args = getattr(fn, "arguments", None)
                            if isinstance(args, str):
                                size += len(args)
                            name = getattr(fn, "name", None)
                            if isinstance(name, str):
                                size += len(name)
        else:
            # Non-chat-completions shapes (Anthropic events etc.): try the
            # common text fields, else keep the framing floor.
            for attr in ("text", "partial_json"):
                v = getattr(getattr(chunk, "delta", None), attr, None)
                if isinstance(v, str):
                    size += len(v)
    except Exception:
        pass
    return size


def _codex_wait_notice_recovery(
    *,
    stale_timeout: float,
    ttfb_enabled: bool,
    ttfb_timeout: float,
    last_event_ts: Optional[float],
    call_start: float,
    idle_enabled: bool,
    idle_timeout: float,
    elapsed: float,
) -> str:
    """Describe the earliest enabled Codex watchdog on the call timeline."""
    deadlines: list[float] = []
    if math.isfinite(stale_timeout):
        deadlines.append(stale_timeout)
    if last_event_ts is None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(ttfb_timeout)
    elif idle_enabled and math.isfinite(idle_timeout):
        deadlines.append(max(0.0, last_event_ts - call_start) + idle_timeout)
    if not deadlines or min(deadlines) <= elapsed:
        return ""
    return f"; auto-reconnect at {int(min(deadlines))}s"


# ── Cross-turn stale-call circuit breaker (#58962) ─────────────────────
# A session wedged against an unresponsive provider hits the stale detector
# on every call and loops forever (observed: 494 consecutive failures over
# 3+ days, each burning the full stale timeout × retries with no response).
# The agent carries ``_consecutive_stale_streams``: incremented on every
# stale kill, reset only when a call actually completes (or when the
# provider is swapped — switch_model / try_activate_fallback /
# restore_primary_runtime — since the streak measured the OLD provider).
# Past the give-up threshold, calls abort immediately with an actionable
# error instead of re-waiting out the stale timeout.

def _stale_streak(agent) -> int:
    try:
        return int(getattr(agent, "_consecutive_stale_streams", 0) or 0)
    except Exception:
        return 0


def _bump_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = _stale_streak(agent) + 1
    except Exception:
        pass


def _reset_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = 0
    except Exception:
        pass


_INTERRUPTED_WAIT_STALE_SECONDS = 30.0


def _record_interrupted_provider_wait(
    agent,
    elapsed: float,
    *,
    response_started: bool,
) -> bool:
    """Count a user-aborted pre-response stall toward the stale breaker.

    Interactive users commonly send a follow-up while a provider is wedged.
    Once the same no-output interval that earns a wait notice has elapsed, that
    interrupt is evidence of an unresponsive attempt rather than a quick user
    cancellation. Mid-response and early interrupts remain neutral.
    """
    if response_started or elapsed < _INTERRUPTED_WAIT_STALE_SECONDS:
        return False
    _bump_stale_streak(agent)
    logger.warning(
        "Interrupted provider wait counted as stale after %.0fs with no output; "
        "consecutive stale attempts=%d.",
        elapsed,
        _stale_streak(agent),
    )
    return True


def _report_stale_nonstream_kill(
    agent,
    api_kwargs: dict,
    elapsed: float,
    stale_timeout: float,
    *,
    inline: bool = False,
    hint: Optional[str] = None,
) -> None:
    """Emit the user/operator-facing trio for a stale non-streaming kill.

    Shared by the interrupt-worker poll loop and the inline
    ``direct_api_call`` watchdog so the log line, status message, and
    activity token stay identical across both paths. Only reporting lives
    here — the kill/state sequences differ deliberately between the two
    callers (locking models are not the same).
    """
    model = api_kwargs.get("model", "unknown")
    logger.warning(
        "%son-streaming API call stale for %.0fs (threshold %.0fs). "
        "model=%s context=~%s tokens. Killing connection.",
        "Inline n" if inline else "N",
        elapsed,
        stale_timeout,
        model,
        f"{estimate_request_context_tokens(api_kwargs):,}",
    )
    try:
        agent._buffer_status(
            f"⚠️ No response from provider for {int(elapsed)}s "
            f"(non-streaming, model: {model}). {hint or 'Aborting call.'}"
        )
    except Exception:
        logger.debug("stale status buffering failed", exc_info=True)


def _touch_stale_kill_activity(agent, elapsed: float) -> None:
    try:
        agent._touch_activity(
            f"stale non-streaming call killed after {int(elapsed)}s"
        )
    except Exception:
        logger.debug("stale activity touch failed", exc_info=True)


def _check_stale_giveup(agent) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait."""
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup > 0 and _streak >= _giveup:
        raise RuntimeError(
            "Provider has been unresponsive (no response received) for "
            f"{_streak} consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Switch models or start a new "
            "session, then retry."
        )


def _derive_stream_stale_timeout(agent, api_kwargs: dict) -> float:
    """Stale-stream patience for a provider that is never a local endpoint.

    Mirrors the main streaming path's derivation — provider config → env base
    → context-size scaling → reasoning-model floor — minus the local-endpoint
    ``float('inf')``/900s disable branch, which cannot apply to Bedrock (its
    endpoint is always the AWS cloud). Factored so the Bedrock streaming
    watchdog shares the exact same patience budget as the OpenAI/Anthropic
    stale-stream detector below.
    """
    _cfg_stale = get_provider_stale_timeout(agent.provider, agent.model)
    if _cfg_stale is not None:
        _base = _cfg_stale
    else:
        _base = env_float("HERMES_STREAM_STALE_TIMEOUT", 180.0)
    _est_tokens = estimate_request_context_tokens(api_kwargs)
    if _est_tokens > 100_000:
        _timeout = max(_base, 300.0)
    elif _est_tokens > 50_000:
        _timeout = max(_base, 240.0)
    else:
        _timeout = _base
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    # Resolve the model id from BOTH the OpenAI/Anthropic key (``model``) and
    # the Bedrock key (``modelId``). OpenAI/Anthropic wins first via the ``or``
    # chain, so those paths are unchanged. Bedrock carries the model as a
    # dotted, region-prefixed inference-profile id (e.g.
    # ``us.anthropic.claude-opus-4-6-v1:0``) that the floor's start-of-slug
    # regex cannot match directly — normalize it to a canonical slug first.
    _model_id = api_kwargs.get("model") or api_kwargs.get("modelId") or ""
    _reasoning_floor = get_reasoning_stale_timeout_floor(_model_id)
    if _reasoning_floor is None and api_kwargs.get("modelId"):
        _reasoning_floor = _bedrock_reasoning_stale_floor(api_kwargs["modelId"])
    if _reasoning_floor is not None:
        _timeout = max(_timeout, _reasoning_floor)
    return _timeout


def _bedrock_reasoning_stale_floor(model_id: object) -> "float | None":
    """Map a Bedrock inference-profile id to its reasoning stale-timeout floor.

    Bedrock carries the model as a dotted, region-prefixed id such as
    ``us.anthropic.claude-opus-4-6-v1:0``, whereas
    :func:`get_reasoning_stale_timeout_floor` anchors its slug patterns at the
    start of a bare slug (``claude-opus-4``). Strip the region prefix
    (``us.``/``eu.``/``apac.``/...) and try two candidate slugs against the
    floor:

    * the segment after the provider namespace (``claude-opus-4-6-v1:0``) —
      matches Anthropic-style slugs whose floor key excludes the provider
      (``claude-opus-4``); and
    * the region-stripped id with the provider dot rewritten to a dash
      (``deepseek-r1-v1:0``) — matches provider-qualified floor keys
      (``deepseek-r1``).

    The floor's right-anchor (``$`` or ``-``/``.``/``_``) tolerates the
    trailing date-stamp / ``-v1:0`` version suffix, so no suffix stripping is
    needed. First non-None wins; returns None for unknown models.

    The floor table mixes version-separator conventions: some keys are
    keyed with a dashed version (``claude-opus-4``) while others embed a
    dotted version (``claude-sonnet-4.5``, ``claude-sonnet-4.6``). Bedrock
    always dashes the version (``claude-sonnet-4-5-v1:0``), so for every
    candidate slug we also try the alternate version-separator form —
    digit-dash-digit rewritten to digit-dot-digit and vice-versa — so a
    dashed Bedrock id matches a dotted floor key (and the reverse). The
    rewrite only touches version-number separators (a dash/dot flanked by
    digits), never other dashes in the slug, so ``claude-sonnet`` is left
    intact while ``4-5`` becomes ``4.5``.
    """
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    if not model_id or not isinstance(model_id, str):
        return None
    name = model_id.strip().lower()
    for prefix in (
        "global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.",
        "ca.", "sa.", "me.", "af.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    base_candidates = [name]
    if "." in name:
        base_candidates.append(name.rsplit(".", 1)[1])   # claude-opus-4-6-v1:0
        base_candidates.append(name.replace(".", "-", 1))  # deepseek-r1-v1:0
    candidates: list[str] = []
    for cand in base_candidates:
        # Try the slug as-is plus both alternate version-separator forms.
        # ``4-5`` <-> ``4.5`` only; a dash/dot not flanked by digits is
        # left alone (e.g. ``claude-sonnet`` stays dashed).
        dashed_to_dotted = re.sub(r"(?<=\d)-(?=\d)", ".", cand)
        dotted_to_dashed = re.sub(r"(?<=\d)\.(?=\d)", "-", cand)
        for form in (cand, dashed_to_dotted, dotted_to_dashed):
            if form not in candidates:
                candidates.append(form)
    for cand in candidates:
        floor = get_reasoning_stale_timeout_floor(cand)
        if floor is not None:
            return floor
    return None


def _dispatch_nonstreaming_api_request(agent, api_kwargs: dict, *, make_client):
    """Run one non-streaming LLM request for the active api_mode and return it.

    Shared by the interrupt-worker path (``interruptible_api_call``) and the
    inline path (``direct_api_call``) so the per-api_mode dispatch — codex /
    anthropic / bedrock / MoA / OpenAI-compatible — lives in exactly one place.

    ``make_client(reason, kind=...)`` builds the per-request client for the
    codex / OpenAI-compatible (``kind="openai"``) and anthropic
    (``kind="anthropic_messages"``) branches; the worker path uses it to
    register the client with its stranger-thread abort machinery, the inline
    path uses it to capture the client for its own ``finally`` close. The
    bedrock / MoA branches manage their own clients and never call it. All
    interrupt, abort, cancellation, and close semantics stay in the callers —
    this helper only issues the request.
    """
    if agent.api_mode == "codex_responses":
        request_client = make_client("codex_stream_request")
        return agent._run_codex_stream(
            api_kwargs,
            client=request_client,
            on_first_delta=getattr(agent, "_codex_on_first_delta", None),
        )
    if agent.api_mode == "anthropic_messages":
        # #67142: use a request-local Anthropic client so the stale/interrupt
        # watchdog aborts sockets from the stranger thread while the worker
        # owns the SDK close — never closing the shared client mid-flight.
        request_client = make_client(
            "anthropic_messages_request", kind="anthropic_messages"
        )
        return agent._anthropic_messages_create(api_kwargs, client=request_client)
    if agent.api_mode == "bedrock_converse":
        # Bedrock uses boto3 directly — no OpenAI client needed.
        # normalize_converse_response produces an OpenAI-compatible
        # SimpleNamespace so the rest of the agent loop can treat
        # bedrock responses like chat_completions responses.
        from agent.bedrock_adapter import (
            _get_bedrock_runtime_client,
            invalidate_runtime_client,
            is_stale_connection_error,
            normalize_converse_response,
            recover_from_cache_point_rejection,
        )
        region = api_kwargs.pop("__bedrock_region__", "us-east-1")
        api_kwargs.pop("__bedrock_converse__", None)
        client = _get_bedrock_runtime_client(region)
        try:
            raw_response = client.converse(**api_kwargs)
        except Exception as _bedrock_exc:
            # A model that refuses cachePoint in one section (Nova rejects it
            # inside toolConfig.tools, #97281) fails every turn otherwise —
            # drop that marker and resend before surfacing the error.
            _retry_kwargs = recover_from_cache_point_rejection(
                _bedrock_exc, api_kwargs
            )
            if _retry_kwargs is not None:
                raw_response = client.converse(**_retry_kwargs)
                return normalize_converse_response(raw_response)
            # Evict the cached client on stale-connection failures
            # so the outer retry loop builds a fresh client/pool.
            if is_stale_connection_error(_bedrock_exc):
                invalidate_runtime_client(region)
            raise
        return normalize_converse_response(raw_response)
    if agent.provider == "moa":
        # MoA is a virtual chat-completions provider backed by the
        # in-process MoAClient facade. Do not rebuild a request-local
        # OpenAI client from the virtual runtime metadata.
        #
        # After a client replacement (credential rotation /
        # dead-connection cleanup / fallback+restore), agent.client may
        # become a native OpenAI client while agent.provider stays
        # "moa".  Pop the MoA-internal key so the native SDK does not
        # reject it as an unexpected kwarg — but only when the live
        # client is NOT the facade: the facade consumes the key, and
        # stripping it there forces a wasteful duplicate reference
        # fan-out (the facade re-prepares from scratch).  Only the MoA
        # facade's completions object exposes ``prepare()``.  (#78382)
        _completions = getattr(getattr(agent.client, "chat", None), "completions", None)
        if not callable(getattr(_completions, "prepare", None)):
            api_kwargs.pop("_moa_prepared_request", None)
        return agent.client.chat.completions.create(**api_kwargs)
    request_client = make_client("chat_completion_request")
    return request_client.chat.completions.create(**api_kwargs)


def should_use_direct_api_call(agent) -> bool:
    """Whether an OpenAI-wire request should skip the interrupt worker.

    Two nested-pool contexts wedge before the socket opens when the request
    is pushed onto yet another daemon worker thread:

    - Gateway cron turns (#62151): gateway asyncio loop → cron thread →
      interrupt worker. Fixed by running inline.
    - Delegated children (#60203): gateway loop → async-delegation executor
      (module-lifetime daemon pool) → per-child timeout executor → interrupt
      worker. Same fingerprint after multi-day gateway uptime — children hang
      at their FIRST API call with zero stale-detector output (the worker
      never reaches dispatch), all providers, restart cures it. The cron fix
      originally excluded delegation "for lack of evidence"; #60203 is that
      evidence.

    Running inline drops the deepest thread layer (whose only job is
    interactive-interrupt responsiveness). Interrupts still work: the inline
    path registers ``agent._active_request_abort``, which ``interrupt()``
    invokes cross-thread to shut the active sockets — the same mechanism the
    async-delegation stall monitor (#72227) relies on.

    Keep native/Codex/Bedrock/MoA transports on their established workers:
    their cancellation and client ownership differ.
    """
    if getattr(agent, "api_mode", None) != "chat_completions":
        return False
    if getattr(agent, "provider", None) == "moa":
        return False
    if getattr(agent, "platform", None) == "cron":
        return True
    # Delegated child (delegate_task sync or background) — detected via the
    # execution ContextVar set by _run_single_child, with the agent's own
    # platform stamp as a fallback for callers that bypass the runner.
    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            return True
    except Exception:
        pass
    return getattr(agent, "platform", None) == "subagent"


# How often an in-flight direct_api_call refreshes last_activity_ts.
# Must stay well under the async-delegation idle stall threshold (450s) and
# the sync heartbeat idle window so a healthy slow model wait is never
# mistaken for a frozen child. Kept below the 30s monitor sweep interval so
# progress tokens change every sample while the request is open.
_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS = 15.0


def _managed_local_load_notice(agent, api_kwargs: dict) -> "Optional[str]":
    """A live phase notice while the managed local server works before the
    first token, or None when neither phase (nor the managed server) applies:

    - "⏳ loading <model> into memory — N%"  (weights streaming off disk;
      real per-tensor percent from the router's SSE stream)
    - "⚙ processing prompt — N of ~M tokens (P%)"  (prefill; live counter
      from /slots, denominator estimated from the request body)

    A cold local model spends ~tens of seconds loading and a long-context
    turn spends tens more in prefill; without this, both windows render as
    the generic "no output yet (provider may be slow or overloaded)" stall
    warning — alarming copy for healthy, expected phases.
    """
    try:
        base = str(getattr(agent, "base_url", "") or "")
        if not base:
            return None
        import json as _json
        from urllib.parse import urlparse

        from hermes_cli.local_runtime.load_progress import (
            get_loading_progress,
            get_prefill_progress,
        )
        from hermes_cli.local_runtime.supervisor import state_path

        state = _json.loads(state_path().read_text(encoding="utf-8"))
        managed = urlparse(str(state.get("base_url", ""))).netloc.lower()
        if not managed or urlparse(base).netloc.lower() != managed:
            return None
        model = str(api_kwargs.get("model", ""))
        progress = get_loading_progress().get(model)
        if progress is not None:
            return (
                f"⏳ loading {model} into memory — {progress['percent']}% "
                "(responses start once the model is loaded)"
            )
        prefill = get_prefill_progress(model)
        if prefill is not None:
            processed = int(prefill["processed"])
            total = estimate_request_context_tokens(api_kwargs)
            if total and total >= processed:
                pct = max(0, min(100, round(processed / total * 100)))
                return f"⚙ processing prompt — {pct}%"
            # Counter past the estimate (estimator undercounted): no honest
            # denominator, so no percent — the UI shows label-only.
            return "⚙ processing prompt"
        return None
    except Exception:  # noqa: BLE001 — a status nicety must never break a call
        return None


def _resolve_direct_stale_timeout(agent, api_kwargs: dict) -> float:
    """Stale budget for the inline non-streaming call.

    Same derivation the interrupt-worker path uses for its stale-call
    detector (provider ``stale_timeout_seconds`` →
    ``HERMES_API_CALL_STALE_TIMEOUT`` → reasoning-model floor → context-size
    scaling, ``inf`` for a local endpoint on the implicit default), so cron and
    delegated turns get exactly the patience every other non-streaming request
    already gets.

    A non-numeric result — an agent stub that never implements the resolver —
    leaves the watchdog disarmed rather than arming it on a bogus budget.
    A resolver that *raises* propagates, exactly as it does on the worker
    path's stale detector: swallowing it into ``inf`` would silently disarm
    the watchdog and reinstate the unbounded hang this exists to fix.
    """
    resolver = getattr(agent, "_compute_non_stream_stale_timeout", None)
    if not callable(resolver):
        return float("inf")
    value = resolver(api_kwargs)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("inf")
    return float(value)


def _inline_nonstream_hard_timeout(stale_timeout: float):
    """Socket-level backstop for inline non-streaming calls (#85252).

    The keepalive httpx client uses ``read=None`` so SSE streams can idle
    during reasoning. That same client serves cron/subagent non-streaming
    calls. Combined with a stranger-thread abort that must not ``close()``
    the FD (#29507), a hung provider then waits until TCP dies — observed
    5–11× past the stale threshold.

    Returns an ``httpx.Timeout`` whose read budget equals the stale
    watchdog, a float if httpx is unavailable, or ``None`` when the
    watchdog is disarmed (local endpoint / non-finite budget).
    """
    if not math.isfinite(stale_timeout) or stale_timeout <= 0:
        return None
    conn_cap = min(stale_timeout, 60.0)
    try:
        import httpx as _httpx

        return _httpx.Timeout(
            connect=conn_cap,
            read=stale_timeout,
            write=conn_cap,
            pool=conn_cap,
        )
    except Exception:
        return stale_timeout


def direct_api_call(agent, api_kwargs: dict):
    """Run a non-streaming LLM call inline on the conversation thread.

    Used when ``should_use_direct_api_call`` is True (cron turns, delegated
    children): no interrupt worker, so the nested-pool deadlock (#62151,
    #60203) cannot occur. An activity heartbeat keeps ``last_activity_ts``
    advancing or the stall monitor interrupts a slow-but-healthy wait at
    ~450s. A stale-call watchdog bounds the request (#80759): the keepalive
    client uses ``read=None``, so a silent provider never trips a read
    timeout — the timer aborts in-flight sockets via the registered hook, and
    a per-call ``timeout`` equal to the stale budget is the backstop when the
    abort finds nothing to shut down (#85252). Both surface a retryable
    ``TimeoutError`` for the outer retry loop.
    """
    _check_stale_giveup(agent)
    agent._touch_activity("waiting for non-streaming API response")
    # Lifecycle state, every transition under the lock (#75301): ``done``
    # stops a late timer bumping the stale streak after unwind; ``cancelled``
    # makes a user/monitor interrupt own the outcome so a racing timer can't
    # misclassify the kill as staleness; ``stale`` is the one-shot transition.
    request_state = {"client": None, "done": False, "stale": False, "cancelled": False}
    request_client_lock = threading.Lock()
    activity_hb_stop = threading.Event()

    def _abort_active_request(reason: str) -> bool:
        """Abort the inline request from a watchdog/interrupt thread.

        Returns True when this call owned the stale transition (so the
        timer callback only reports/bumps once, and never after an
        interrupt or a completed request).
        """
        # Abort under the lock (same contract as _RequestClientRegistry):
        # once released the finally may cache the client and the NEXT call
        # check it out, so a late abort would poison an innocent request.
        with request_client_lock:
            if request_state["done"]:
                return False
            if reason == "stale_call_kill" and request_state["cancelled"]:
                return False
            if reason != "stale_call_kill":
                # Interrupt wins the lock -> owns the outcome; a later timer
                # must not count it as staleness.
                request_state["cancelled"] = True
            newly_stale = reason == "stale_call_kill" and not request_state["stale"]
            if newly_stale:
                request_state["stale"] = True
                # Bump BEFORE releasing: a fast retry's reset must not be
                # overtaken by this older timer restoring the streak.
                _bump_stale_streak(agent)
            request_client = request_state["client"]
            if request_client is not None:
                try:
                    agent._abort_request_openai_client(request_client, reason=reason)
                except Exception:
                    logger.debug(
                        "Inline request abort failed (%s)", reason, exc_info=True
                    )
            return newly_stale

    def _make_client(reason: str, kind: str = "openai"):
        # Only OpenAI-wire requests reach direct_api_call; ``kind`` exists
        # for signature parity with the dispatch helper.
        client = agent._create_request_openai_client(reason=reason, api_kwargs=api_kwargs)
        stale_before_dispatch = False
        with request_client_lock:
            request_state["client"] = client
            if request_state["stale"]:
                # Timer fired during client construction: the abort found no
                # socket, so dispatching now would open one AFTER the only
                # watchdog fired. Fail here instead. (Residual ms-scale window
                # before httpx opens its socket is accepted.)
                stale_before_dispatch = True
                try:
                    agent._abort_request_openai_client(
                        client, reason="stale_call_kill"
                    )
                except Exception:
                    logger.debug(
                        "Inline abort after late client registration failed",
                        exc_info=True,
                    )
        if stale_before_dispatch:
            raise TimeoutError(
                "Non-streaming API call timed out before request dispatch "
                f"(threshold: {int(stale_timeout)}s)"
            )
        agent._active_request_abort = _abort_active_request
        return client

    def _activity_heartbeat() -> None:
        # Do not put the API call itself on another worker thread — that is
        # the nested-pool deadlock this path exists to avoid (#60203). This
        # ticker only refreshes the activity clock.
        while not activity_hb_stop.wait(_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS):
            try:
                agent._touch_activity("waiting for non-streaming API response")
            except Exception:
                pass

    activity_hb = threading.Thread(
        target=_activity_heartbeat,
        name="direct-api-activity-hb",
        daemon=True,
    )
    # Resolve the budget BEFORE start(): the resolver may raise (fail-closed),
    # and a leaked heartbeat thread would mask real stalls forever.
    call_start = time.time()
    stale_timeout = _resolve_direct_stale_timeout(agent, api_kwargs)
    # Never override an explicit per-call timeout; otherwise pin
    # read=stale_timeout so a no-op abort can't leave the read=None socket
    # hanging until TCP dies (#85252).
    hard_timeout = _inline_nonstream_hard_timeout(stale_timeout)
    if hard_timeout is not None and "timeout" not in api_kwargs:
        api_kwargs = dict(api_kwargs)
        api_kwargs["timeout"] = hard_timeout
    activity_hb.start()

    def _on_stale() -> None:
        # Timer thread: aborts sockets only, never issues a request (keeps
        # the no-worker property). False = request finished or an interrupt
        # owns the outcome; stay silent.
        if not _abort_active_request("stale_call_kill"):
            return
        elapsed = time.time() - call_start
        _report_stale_nonstream_kill(
            agent, api_kwargs, elapsed, stale_timeout, inline=True
        )
        _touch_stale_kill_activity(agent, elapsed)

    stale_watchdog = None
    if math.isfinite(stale_timeout) and stale_timeout > 0:
        stale_watchdog = threading.Timer(stale_timeout, _on_stale)
        stale_watchdog.name = "direct-api-stale-watchdog"
        stale_watchdog.daemon = True
        stale_watchdog.start()

    # Only a clean return reports the reuse reason; errors/interrupts really
    # close the client so the retry builds a fresh pool.
    succeeded = False
    try:
        response = _dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=_make_client
        )
    except Exception:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call") from None
        with request_client_lock:
            was_stale = request_state["stale"]
        if was_stale:
            # Our own abort caused the transport error: raise a retryable
            # TimeoutError, never InterruptedError ("the user wants to stop").
            raise TimeoutError(
                f"Non-streaming API call timed out after "
                f"{int(time.time() - call_start)}s with no response "
                f"(threshold: {int(stale_timeout)}s)"
            ) from None
        raise
    else:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call")
        # Mark ``done`` under the lock so a timer firing between response
        # arrival and unwind is a no-op and cannot overwrite the reset below.
        # If a timer already won, the request still completed: return it (the
        # reset undoes the bump; the finally discards the poisoned client).
        with request_client_lock:
            request_state["done"] = True
        _reset_stale_streak(agent)
        succeeded = True
        return response
    finally:
        if stale_watchdog is not None:
            stale_watchdog.cancel()
        with request_client_lock:
            request_state["done"] = True
        activity_hb_stop.set()
        activity_hb.join(timeout=2.0)
        if getattr(agent, "_active_request_abort", None) is _abort_active_request:
            agent._active_request_abort = None
        with request_client_lock:
            request_client = request_state["client"]
            request_state["client"] = None
        if request_client is not None:
            agent._close_request_openai_client(
                request_client,
                reason="request_complete" if succeeded else "request_error_cleanup",
            )


class _RequestClientRegistry:
    """Per-request client / stream-handle registry shared by the request worker
    and the stranger threads (interrupt-check loop, stale detector) that may
    need to abort it.

    ``kind`` is ``"openai"`` (default), ``"anthropic_messages"`` or ``"stream"``
    and routes :meth:`close_once` to the matching abort/close helpers (#67142).
    ``kind="stream"`` registers a per-request *stream handle* instead of a
    client — used under the MoA facade, whose singleton client has no
    per-request sockets to abort, so interrupts must close the stream object
    itself (#57354).

    Thread-ownership rule (#29507): the owning worker thread pops + fully
    closes on its way out. A *stranger* thread only aborts the sockets so the
    worker's blocked ``recv``/``send`` unwinds with EPIPE/EOF — never
    ``client.close()`` — avoiding the FD-recycling race where the kernel
    reassigned a just-closed TLS socket FD to ``kanban.db`` and the still-live
    SSL BIO wrote a TLS record into the SQLite header. The abort happens under
    the holder lock: once released, the worker's finally may pop + cache the
    client for reuse and the NEXT call check it out, so a late abort would
    poison an innocent in-flight request's sockets. A registered stream handle
    is safe to close from any thread (closing IS the abort), so the ownership
    carve-out only applies to real per-request clients.
    """

    def __init__(self, agent):
        self.agent = agent
        self.client = None
        self.kind = "openai"
        self.owner_tid = None
        self.diag = None  # per-attempt stream diagnostics (streaming path)
        self.lock = threading.Lock()

    def set_client(self, client, *, kind: str = "openai"):
        with self.lock:
            self.client = client
            self.kind = kind
            self.owner_tid = threading.get_ident()
        return client

    @staticmethod
    def _stream_close_callable(stream):
        close = getattr(stream, "close", None)
        if callable(close):
            return close
        response = getattr(stream, "response", None)
        close = getattr(response, "close", None)
        if callable(close):
            return close
        return None

    def set_stream_handle(self, stream):
        if self._stream_close_callable(stream) is None:
            return stream
        with self.lock:
            self.client = stream
            self.kind = "stream"
            self.owner_tid = threading.get_ident()
        return stream

    def _close_stream_handle(self, stream, reason: str) -> None:
        close = self._stream_close_callable(stream)
        if close is None:
            return
        try:
            close()
            logger.info("Streaming response handle closed (%s)", reason)
        except Exception as exc:
            logger.debug(
                "Streaming response handle close failed (%s): %s",
                reason,
                exc,
            )

    def close_once(self, reason: str) -> None:
        with self.lock:
            request_client = self.client
            request_kind = self.kind
            owner_tid = self.owner_tid
            stranger_thread = (
                request_kind != "stream"
                and request_client is not None
                and owner_tid is not None
                and owner_tid != threading.get_ident()
            )
            if stranger_thread:
                if request_kind == "anthropic_messages":
                    self.agent._abort_request_anthropic_client(
                        request_client, reason=reason
                    )
                else:
                    self.agent._abort_request_openai_client(request_client, reason=reason)
                return
            self.client = None
            self.owner_tid = None
        if request_client is None:
            return
        if request_kind == "stream":
            self._close_stream_handle(request_client, reason)
        elif request_kind == "anthropic_messages":
            self.agent._close_request_anthropic_client(request_client, reason=reason)
        else:
            self.agent._close_request_openai_client(request_client, reason=reason)


@dataclass
class _NonStreamWatchdogs:
    """Poll-loop thresholds for one non-streaming request."""
    stale_timeout: float
    codex: bool            # api_mode == codex_responses (codex watchdogs armed)
    est_tokens: int
    ttfb_enabled: bool
    ttfb_timeout: float
    idle_enabled: bool
    idle_timeout: float


def _resolve_nonstream_watchdogs(agent, api_kwargs: dict) -> _NonStreamWatchdogs:
    """Stale-call timeout plus the Codex Responses stream watchdogs.

    Non-streaming calls return nothing until the full response is ready, so a
    hung provider would block for the full httpx timeout (1800s) with zero
    feedback; the stale detector kills early so the main retry loop can apply
    credential rotation / provider fallback.

    Codex (chatgpt.com/backend-api/codex) has two extra failure modes: it
    accepts the connection and never emits a stream event (a fresh reconnect
    succeeds in ~2s, so waiting out the 180–900s stale timeout is wasteful),
    and it emits an opening SSE frame then stalls forever in SSL read. The
    no-byte TTFB cutoff covers the first; the event-idle gap (any valid SSE
    event is activity, as in Codex CLI's stream_idle_timeout) covers the
    second. Tunables: HERMES_CODEX_TTFB_TIMEOUT_SECONDS,
    HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS (0 disables each),
    HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS / HERMES_CODEX_TTFB_STRICT,
    HERMES_CODEX_TTFB_MAX_SECONDS, HERMES_CODEX_HARD_TIMEOUT_SECONDS.
    """
    stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
    codex = agent.api_mode == "codex_responses"
    openai_codex_backend = _is_openai_codex_backend(agent)
    est_tokens = estimate_request_context_tokens(api_kwargs)
    if codex and openai_codex_backend:
        # Raise the stale floor for large payloads so healthy gateway-scale
        # requests aren't aborted mid-prefill.
        codex_floor = openai_codex_stale_timeout_floor(est_tokens)
        if codex_floor:
            stale_timeout = max(stale_timeout, codex_floor)
        # Flat hard ceiling (#64507): a request that emits SOME bytes then
        # wedges is otherwise only reclaimed at the raised stale floor. The
        # default sits ABOVE the max floor (1200s) — a backstop against
        # unbounded hangs, never a tighter limit. 0 disables.
        hard_timeout = _env_float("HERMES_CODEX_HARD_TIMEOUT_SECONDS", 1500.0)
        if hard_timeout > 0:
            stale_timeout = min(stale_timeout, hard_timeout)

    if est_tokens > 100_000:
        idle_default = 180.0
    elif est_tokens > 50_000:
        idle_default = 120.0
    elif est_tokens > 10_000:
        idle_default = 60.0
    else:
        idle_default = 12.0

    # No-byte TTFB cutoff. Default 120s: the SDK's own read timeout is 600s,
    # and a tight 12s killed subscription-backed requests mid-prefill.
    ttfb_enabled = codex
    ttfb_timeout = _env_float("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", 120.0)
    if ttfb_timeout <= 0:
        ttfb_enabled = False
    elif openai_codex_backend:
        # Large requests legitimately spend tens of seconds in admission /
        # prefill before the first SSE event: scale the cutoff up to the idle
        # default unless HERMES_CODEX_TTFB_STRICT keeps the smaller one.
        disable_above = _env_float("HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", 10_000.0)
        strict = os.environ.get("HERMES_CODEX_TTFB_STRICT", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not strict and disable_above > 0 and est_tokens >= disable_above:
            if ttfb_timeout < idle_default:
                logger.info(
                    "Scaling openai-codex no-byte TTFB watchdog from %.0fs to %.0fs "
                    "for large request (context=~%s tokens >= %.0f). "
                    "Set HERMES_CODEX_TTFB_STRICT=1 to keep the smaller cutoff.",
                    ttfb_timeout,
                    idle_default,
                    f"{est_tokens:,}",
                    disable_above,
                )
                ttfb_timeout = idle_default
        ttfb_cap = _env_float("HERMES_CODEX_TTFB_MAX_SECONDS", 120.0)
        if ttfb_cap > 0 and ttfb_timeout > ttfb_cap:
            logger.info(
                "Capping openai-codex no-byte TTFB timeout from %.0fs to %.0fs "
                "(context=~%s tokens). Set HERMES_CODEX_TTFB_MAX_SECONDS to tune.",
                ttfb_timeout,
                ttfb_cap,
                f"{est_tokens:,}",
            )
            ttfb_timeout = ttfb_cap

    idle_timeout = _env_float("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", idle_default)
    idle_enabled = codex and idle_timeout > 0
    return _NonStreamWatchdogs(
        stale_timeout=stale_timeout,
        codex=codex,
        est_tokens=est_tokens,
        ttfb_enabled=ttfb_enabled,
        ttfb_timeout=ttfb_timeout,
        idle_enabled=idle_enabled,
        idle_timeout=idle_timeout,
    )


def _codex_silent_hang_hint(agent, api_kwargs: dict) -> Optional[str]:
    hint_fn = getattr(agent, "_codex_silent_hang_hint", None)
    if not callable(hint_fn):
        return None
    try:
        return hint_fn(model=api_kwargs.get("model"))
    except Exception:
        return None


def interruptible_api_call(agent, api_kwargs: dict):
    """
    Run the API call in a background thread so the main conversation loop
    can detect interrupts without waiting for the full HTTP round-trip.

    Each worker thread gets its own OpenAI client instance. Interrupts only
    close that worker-local client, so retries and other requests never
    inherit a closed transport.

    Includes a stale-call detector: if no response arrives within the
    configured timeout, the connection is killed and an error raised so
    the main retry loop can try again with backoff / credential rotation /
    provider fallback.
    """
    # Cron and other non-interactive, nested-pool contexts must not spawn the
    # interrupt worker — it wedges before the socket opens on the 2nd+ call
    # (#62151). Run inline instead. See should_use_direct_api_call.
    if should_use_direct_api_call(agent):
        return direct_api_call(agent, api_kwargs)

    result = {"response": None, "error": None}

    # Cross-turn stale-call circuit breaker (#58962) — non-streaming sibling
    # of the guard in interruptible_streaming_api_call.  Quiet-mode /
    # subagent / no-stream-consumer sessions take THIS path, and a wedged
    # unattended session here has the same infinite stale-retry class.
    _check_stale_giveup(agent)

    _clients = _RequestClientRegistry(agent)
    # Request-local cancel flag: agent._interrupt_requested is cleared at turn
    # boundaries but this daemon worker can outlive the turn, so the worker
    # needs to know THIS request was force-closed and not surface the
    # resulting transport error as a network bug (#6600).
    _request_cancelled = {"value": False}
    # Codex retirement token: the worker checks
    # ``agent._active_codex_stream_request_token`` to know it still owns the
    # turn; a watchdog kill clears it so a worker still draining SSE raises
    # instead of returning partial output as "completed"
    # (run_codex_stream._request_is_current). ``_codex_request_retired`` is
    # the request-local mirror used to swallow our own force-close error.
    _codex_request_token = object() if agent.api_mode == "codex_responses" else None
    _codex_request_retired = {"value": False}

    def _install_codex_request_token() -> None:
        if _codex_request_token is None:
            return
        if _codex_request_retired["value"]:
            # Already retired before the worker got going — do not re-publish.
            return
        agent._active_codex_stream_request_token = _codex_request_token

    def _retire_codex_request_token() -> None:
        if _codex_request_token is None:
            return
        _codex_request_retired["value"] = True
        if (
            getattr(agent, "_active_codex_stream_request_token", None)
            is _codex_request_token
        ):
            agent._active_codex_stream_request_token = None

    def _call():
        try:
            _install_codex_request_token()
            # Per-request clients are registered with the abort machinery so
            # the watchdogs force-close the worker's connection, never the
            # shared client (#67142).
            result["response"] = _dispatch_nonstreaming_api_request(
                agent,
                api_kwargs,
                make_client=lambda reason, kind="openai": _clients.set_client(
                    agent._create_request_anthropic_client(reason=reason)
                    if kind == "anthropic_messages"
                    else agent._create_request_openai_client(
                        reason=reason, api_kwargs=api_kwargs
                    ),
                    kind=kind,
                ),
            )
        except Exception as e:
            # Our own force-close caused this error: swallow it, the main
            # thread raises InterruptedError (#6600). Retirement logs at info
            # (a watchdog discarded output the provider already sent — what an
            # operator debugging a truncated reply needs); cancellation at debug.
            if _request_cancelled["value"] or _codex_request_retired["value"]:
                if _codex_request_retired["value"]:
                    logger.info(
                        "Codex worker caught %s after request retirement — "
                        "discarding the stale partial instead of surfacing it "
                        "as a completed response. %s",
                        type(e).__name__,
                        agent._client_log_context(),
                    )
                else:
                    logger.debug(
                        "Non-streaming worker caught %s after request "
                        "cancellation — exiting without surfacing a network "
                        "error.",
                        type(e).__name__,
                    )
                return
            result["error"] = e
        finally:
            # Retire first: close_once can raise, and a leaked token would let
            # a later worker mistake itself for the owning attempt.
            _retire_codex_request_token()
            # Reuse reason only on a clean response; error or cancel-swallow
            # really closes so the next attempt builds a fresh pool.
            _clients.close_once(
                "request_complete"
                if result["response"] is not None
                else "request_error_cleanup"
            )

    wd = _resolve_nonstream_watchdogs(agent, api_kwargs)
    _stale_timeout = wd.stale_timeout
    _codex_watchdog_enabled = wd.codex
    _est_tokens_for_codex_watchdog = wd.est_tokens
    _ttfb_enabled, _ttfb_timeout = wd.ttfb_enabled, wd.ttfb_timeout
    _codex_idle_enabled, _codex_idle_timeout = wd.idle_enabled, wd.idle_timeout
    if _codex_watchdog_enabled:
        # Reset before the worker starts so a marker left over from a previous
        # call on this agent can't be misread as first-byte for this one.
        agent._codex_stream_last_event_ts = None
        agent._codex_stream_last_progress_ts = None

    _call_start = time.time()
    agent._touch_activity("waiting for non-streaming API response")

    def _abort_request(reason: str) -> None:
        """Watchdog/interrupt kill: abort the request client and retire the codex
        token; the worker sees its own forced close via the cancel flags."""
        try:
            # #67142: routes by client kind — anthropic aborts the request-local
            # client's sockets from this poll (stranger) thread instead of
            # closing the shared _anthropic_client.
            _clients.close_once(reason)
        except Exception:
            pass
        _retire_codex_request_token()

    def _await_worker_after_kill(timeout_message: str) -> None:
        # Wait briefly for the worker to notice the closed connection.
        t.join(timeout=2.0)
        if result["error"] is None and result["response"] is None:
            result["error"] = TimeoutError(timeout_message)

    t = threading.Thread(target=_context_thread_target(_call), daemon=True)
    t.start()
    _poll_count = 0
    while t.is_alive():
        t.join(timeout=0.3)
        _poll_count += 1

        # Every ~30s: gateway inactivity heartbeat + rewrite the status line
        # so users see WHAT the wait is (the "infinite thinking" complaint).
        if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
            _elapsed = time.time() - _call_start
            try:
                _recovery = _codex_wait_notice_recovery(
                    stale_timeout=_stale_timeout,
                    ttfb_enabled=_ttfb_enabled,
                    ttfb_timeout=_ttfb_timeout,
                    last_event_ts=getattr(
                        agent, "_codex_stream_last_event_ts", None
                    ),
                    call_start=_call_start,
                    idle_enabled=_codex_idle_enabled,
                    idle_timeout=_codex_idle_timeout,
                    elapsed=_elapsed,
                )
                agent._emit_wait_notice(
                    f"⏳ waiting on {api_kwargs.get('model', 'the provider')} — "
                    f"{int(_elapsed)}s with no response yet (provider may be slow "
                    f"or overloaded{_recovery})"
                )
            except Exception:
                logger.debug("wait-notice construction failed", exc_info=True)

        _elapsed = time.time() - _call_start

        # TTFB detector: no Codex event past the first-byte cutoff — kill so
        # the retry loop reconnects instead of waiting out the stale timeout.
        if (
            _ttfb_enabled
            and _elapsed > _ttfb_timeout
            and getattr(agent, "_codex_stream_last_event_ts", None) is None
        ):
            _silent_hint = _codex_silent_hang_hint(agent, api_kwargs)
            logger.warning(
                "Codex stream produced no bytes within TTFB cutoff "
                "(%.0fs > %.0fs, model=%s). Backend accepted the connection "
                "but sent no stream events. Killing connection so the retry "
                "loop can reconnect.",
                _elapsed, _ttfb_timeout, api_kwargs.get("model", "unknown"),
            )
            if _silent_hint:
                agent._buffer_status(
                    f"⚠️ No first byte from provider in {int(_elapsed)}s "
                    f"(codex stream, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Reconnecting. {_silent_hint}"
                )
            else:
                agent._buffer_status(
                    f"⚠️ No first byte from provider in {int(_elapsed)}s "
                    f"(codex stream, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Reconnecting."
                )
            _abort_request("codex_ttfb_kill")
            agent._emit_wait_notice(
                f"⚠ no response from provider in {int(_elapsed)}s — "
                f"reconnecting..."
            )
            agent._touch_activity(
                f"codex stream killed after {int(_elapsed)}s with no first byte"
            )
            _await_worker_after_kill(
                f"Codex stream produced no bytes within {int(_elapsed)}s "
                f"(TTFB threshold: {int(_ttfb_timeout)}s)"
                + (f". {_silent_hint}" if _silent_hint else "")
            )
            break

        # Stream-idle detector: first byte arrived, then events stopped
        # (keepalive/in_progress frames refresh the timestamp and don't count).
        _last_codex_event_ts = getattr(agent, "_codex_stream_last_event_ts", None)
        if (
            _codex_idle_enabled
            and _last_codex_event_ts is not None
            and (time.time() - _last_codex_event_ts) > _codex_idle_timeout
        ):
            _event_stale_elapsed = time.time() - _last_codex_event_ts
            logger.warning(
                "Codex stream produced no SSE events for %.0fs after first byte "
                "(threshold %.0fs, model=%s, context=~%s tokens). Killing "
                "connection so the retry loop can reconnect.",
                _event_stale_elapsed,
                _codex_idle_timeout,
                api_kwargs.get("model", "unknown"),
                f"{_est_tokens_for_codex_watchdog:,}",
            )
            agent._buffer_status(
                f"⚠️ Codex stream sent no events for {int(_event_stale_elapsed)}s "
                f"after first byte (model: {api_kwargs.get('model', 'unknown')}). "
                f"Reconnecting."
            )
            _abort_request("codex_stream_idle_kill")
            agent._touch_activity(
                f"codex stream killed after {int(_event_stale_elapsed)}s with no SSE events"
            )
            _await_worker_after_kill(
                f"Codex stream produced no SSE events for {int(_event_stale_elapsed)}s "
                f"after first byte (threshold: {int(_codex_idle_timeout)}s)"
            )
            break

        # Stale-call detector: kill the connection if no response
        # arrives within the configured timeout.
        if _elapsed > _stale_timeout:
            _silent_hint = _codex_silent_hang_hint(agent, api_kwargs)
            _report_stale_nonstream_kill(
                agent, api_kwargs, _elapsed, _stale_timeout, hint=_silent_hint
            )
            _abort_request("stale_call_kill")
            # Circuit breaker (#58962): count the stale kill.  See the
            # canonical comment block above ``_stale_streak()``.
            _bump_stale_streak(agent)
            _touch_stale_kill_activity(agent, _elapsed)
            _await_worker_after_kill(
                f"Non-streaming API call timed out after {int(_elapsed)}s "
                f"with no response (threshold: {int(_stale_timeout)}s)"
                + (f". {_silent_hint}" if _silent_hint else "")
            )
            break

        if agent._interrupt_requested:
            _record_interrupted_provider_wait(
                agent,
                _elapsed,
                response_started=(
                    _codex_watchdog_enabled
                    and getattr(agent, "_codex_stream_last_event_ts", None) is not None
                ),
            )
            # Mark cancelled BEFORE force-closing so the worker treats the
            # transport error as a cancel (#6600).
            _request_cancelled["value"] = True
            logger.debug(
                "Force-closing httpx client due to interrupt (not a network error)."
            )
            # Force-close the worker-local connection (never the shared client:
            # releasing a TLS FD mid-SSL-BIO corrupted an unrelated SQLite DB,
            # #67142), then let the worker unwind Relay scopes before raising
            # (#81521).
            _abort_request("interrupt_abort")
            _join_worker_for_relay_teardown(t, label="Non-streaming")
            raise InterruptedError("Agent interrupted during API call")
    if result["error"] is not None:
        raise result["error"]
    # Success — clear the circuit breaker (#58962): the provider proved
    # responsive.  See the canonical comment block above ``_stale_streak()``.
    if result["response"] is not None:
        _reset_stale_streak(agent)
    return result["response"]



def _consume_ephemeral_reasoning_off(agent) -> bool:
    """Consume the one-shot "answer without thinking" continuation flag.

    Set by the length-continuation path when a request returned reasoning
    but NO visible content — the thinking phase consumed the entire output
    cap (GLM-5.3 on ollama-cloud with reasoning_effort=high: reported live as
    finish_reason="length", content="", completion_tokens == max_tokens).

    Continuation turns never replay the prior reasoning, so re-running with
    thinking ON re-derives — and re-burns — the whole thinking budget from
    scratch instead of writing the answer (observed: 4 futile continuations
    then "Response remained truncated after 4 continuation attempts").
    When True is returned the caller must override the wire reasoning_config
    with ``{"enabled": False, "effort": "none"}`` for exactly the next call.

    Prompt-cache cost (deliberate, bounded): the reasoning parameter is part
    of the provider's cache key on config-sensitive providers — Anthropic
    renders thinking/effort into the prompt, OpenAI lists reasoning.effort
    among prefix-affecting settings — so THAT one request misses the prefix
    cache and pays a cold write of the full prefix (1.25x input instead of
    the 0.1x read).  The next request goes out with the configured reasoning
    again and hits the thinking-on entry written by the truncated request
    (still within TTL), so the damage is exactly one write.  Template-tail
    providers (GLM/Qwen/Kimi-style, where thinking on/off is a chat-template
    switch at the tail) see no prefix change at all.  The system prompt bytes
    are never touched.  This is far cheaper than what the flag prevents: four
    full-output-budget requests that produce nothing and end the turn with an
    error.
    """
    if getattr(agent, "_ephemeral_reasoning_off", False):
        agent._ephemeral_reasoning_off = False
        return True
    return False


def _reasoning_config_for_wire(agent):
    """``agent.reasoning_config`` with the one-shot reasoning-off override applied."""
    if _consume_ephemeral_reasoning_off(agent):
        return {
            **(agent.reasoning_config or {}),
            "enabled": False,
            "effort": "none",
        }
    return agent.reasoning_config


def _alias_tool_search_bridge_for_xai(agent, transport, tools_for_api):
    """xAI chat-completions reserves the function name ``tool_search`` for its
    native tool and 400s when the client bridge declares it (#95003) — same
    reserved-name class the codex branch sanitizes (#27197). Rename the wire
    declaration to an alias; ``normalize_response`` maps calls back via the
    transport's request-local ``_last_wire_aliases`` provenance, which is
    reset here for THIS request so a stale map from an earlier request can't
    reverse-map a name this one never aliased. Deep-copy first (#27907):
    tools_for_api aliases agent.tools, so an in-place rename would corrupt
    the shared registry for every later non-xAI request."""
    if transport is not None and hasattr(transport, "_last_wire_aliases"):
        transport._last_wire_aliases = {}
    is_xai_chat = (
        agent.provider in {"xai", "xai-oauth"}
        or agent._base_url_hostname == "api.x.ai"
    )
    if not (is_xai_chat and tools_for_api):
        return tools_for_api
    try:
        import copy as _copy_xai

        from agent.transports.chat_completions import (
            _rename_tool_search_bridge_for_xai,
        )

        has_bridge = any(
            (t.get("function") or {}).get("name") == "tool_search"
            for t in tools_for_api
            if isinstance(t, dict)
        )
        if has_bridge:
            tools_for_api = _copy_xai.deepcopy(tools_for_api)
            tools_for_api, alias_map = _rename_tool_search_bridge_for_xai(tools_for_api)
            if transport is not None:
                transport._last_wire_aliases = alias_map
    except Exception as exc:
        logger.warning(
            "%s⚠️ Failed to alias tool_search bridge for xAI: %s",
            getattr(agent, "log_prefix", ""), exc,
        )
    return tools_for_api


def build_api_kwargs(agent, api_messages: list, tools_for_api: list | None = None) -> dict:
    """Build the keyword arguments dict for the active API mode."""
    # One-shot continuation override — consumed exactly once, on the FIRST
    # request this call builds (only one api_mode branch runs per invocation).
    _wire_reasoning_config = _reasoning_config_for_wire(agent)
    if tools_for_api is None:
        tools_for_api = agent.tools
    # The one place request_overrides are consumed: static /fast values are
    # already pinned in agent.request_overrides; auto/cold windows layer the
    # fast override here, per request, only while the window is open.
    _request_overrides = effective_request_overrides(agent)

    if agent.api_mode == "anthropic_messages":
        _transport = agent._get_transport()
        anthropic_messages = agent._prepare_anthropic_messages_for_api(api_messages)
        ctx_len = getattr(agent, "context_compressor", None)
        ctx_len = ctx_len.context_length if ctx_len else None
        ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
        if ephemeral_out is not None:
            agent._ephemeral_max_output_tokens = None  # consume immediately
        anthropic_kwargs = _transport.build_kwargs(
            model=agent.model,
            messages=anthropic_messages,
            tools=tools_for_api,
            max_tokens=ephemeral_out if ephemeral_out is not None else agent.max_tokens,
            reasoning_config=_wire_reasoning_config,
            is_oauth=agent._is_anthropic_oauth,
            preserve_dots=agent._anthropic_preserve_dots(),
            context_length=ctx_len,
            base_url=getattr(agent, "_anthropic_base_url", None),
            fast_mode=_request_overrides.get("speed") == "fast",
            drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)),
        )
        # Nous Portal reads ``tags`` and ``session_id`` as top-level body fields
        # on its Messages route the same way it does on /chat/completions, but
        # the profile hook that produces them is only consulted by the
        # OpenAI-wire transport. Merge them here so Messages traffic keeps
        # product attribution and sticky routing.
        return _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs)

    # AWS Bedrock native Converse API — bypasses the OpenAI client entirely.
    # The adapter handles message/tool conversion and boto3 calls directly.
    if agent.api_mode == "bedrock_converse":
        _bt = agent._get_transport()
        region = getattr(agent, "_bedrock_region", None) or "us-east-1"
        guardrail = getattr(agent, "_bedrock_guardrail_config", None)
        return _bt.build_kwargs(
            model=agent.model,
            messages=api_messages,
            tools=tools_for_api,
            max_tokens=agent.max_tokens or 4096,
            region=region,
            guardrail_config=guardrail,
        )

    # Rotation-stable logical cache scope, shared by every OpenAI-wire branch
    # below (codex + both chat_completions paths). Memoized on the agent —
    # cheap after the first call. Resolved after the anthropic/bedrock early
    # returns above, which don't use prompt_cache_key.
    _cache_scope_id = _prompt_cache_scope_for_agent(agent)

    if agent.api_mode == "codex_responses":
        _ct = agent._get_transport()
        from agent.codex_responses_adapter import classify_responses_route

        is_codex_backend, is_xai_responses, is_github_responses = (
            classify_responses_route(agent)
        )
        _msgs_for_codex = agent._prepare_messages_for_non_vision_model(api_messages)

        # Native server-side compaction (gpt-5.6 on direct OpenAI API /
        # ChatGPT Codex routes only) — None on every other route/model, in
        # which case the request is unchanged from pre-feature behavior.
        from agent.native_compaction import native_compaction_context_management
        _context_management = native_compaction_context_management(
            agent,
            is_codex_backend=is_codex_backend,
            is_xai_responses=is_xai_responses,
            is_github_responses=is_github_responses,
        )

        # xAI's /responses endpoint rejects ``pattern`` and ``format`` keywords
        # in tool schemas (HTTP 400 "Invalid arguments passed to the model").
        # Most commonly hit when MCP-derived tools carry JSON Schema validation
        # keywords through. Strip them before building kwargs. See #27197.
        # It also rejects ``enum`` values containing ``/`` (HuggingFace IDs
        # like ``Qwen/Qwen3.5-0.8B`` shipped by MCP servers) — same 400 with
        # the same opaque message; strip those enums too.
        #
        # Deep-copy ``tools_for_api`` before sanitizing: the sanitizers
        # mutate in place (documented contract on ``strip_slash_enum`` /
        # ``strip_pattern_and_format``), and ``tools_for_api`` is a direct
        # reference to ``agent.tools``.  Without the copy, the first xAI
        # request permanently strips constraints from the shared per-agent
        # tool registry — every subsequent non-xAI call from the same
        # agent (auxiliary task routed to Anthropic, OpenRouter fallback,
        # main-model swap) sees the already-stripped schema.  See #27907.
        if is_xai_responses:
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools_for_api = _copy.deepcopy(tools_for_api)
                tools_for_api, _ = strip_pattern_and_format(tools_for_api)
                tools_for_api, _ = strip_slash_enum(tools_for_api)
            except Exception as exc:
                logger.warning(
                    "%s⚠️ Failed to sanitize tool schemas for xAI: %s",
                    getattr(agent, "log_prefix", ""), exc,
                )

        return _ct.build_kwargs(
            model=agent.model,
            messages=_msgs_for_codex,
            tools=tools_for_api,
            reasoning_config=_wire_reasoning_config,
            session_id=getattr(agent, "session_id", None),
            cache_scope_id=_cache_scope_id,
            base_url=agent.base_url,
            max_tokens=agent.max_tokens,
            timeout=agent._resolved_api_call_timeout(),
            request_overrides=_request_overrides,
            provider=getattr(agent, "provider", None),
            is_github_responses=is_github_responses,
            is_codex_backend=is_codex_backend,
            is_xai_responses=is_xai_responses,
            github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
            replay_encrypted_reasoning=bool(
                getattr(agent, "_codex_reasoning_replay_enabled", True)
            ),
            context_management=_context_management,
        )

    # ── chat_completions (default) ─────────────────────────────────────
    _ct = agent._get_transport()

    tools_for_api = _alias_tool_search_bridge_for_xai(agent, _ct, tools_for_api)

    # Provider detection flags
    _is_qwen = agent._is_qwen_portal()
    _is_or = agent._is_openrouter_url()
    _is_gh = (
        base_url_host_matches(agent._base_url_lower, "models.github.ai")
        or base_url_host_matches(agent._base_url_lower, "githubcopilot.com")
    )
    _is_nous = base_url_host_matches(agent._base_url_lower, "nousresearch.com")
    _is_nvidia = base_url_host_matches(agent._base_url_lower, "integrate.api.nvidia.com")
    _is_kimi = (
        base_url_host_matches(agent.base_url, "api.kimi.com")
        or base_url_host_matches(agent.base_url, "moonshot.ai")
        or base_url_host_matches(agent.base_url, "moonshot.cn")
    )
    _is_tokenhub = base_url_host_matches(agent._base_url_lower, "tokenhub.tencentmaas.com")
    _is_lmstudio = (agent.provider or "").strip().lower() == "lmstudio"

    # Temperature: _fixed_temperature_for_model may return OMIT_TEMPERATURE
    # sentinel (temperature omitted entirely), a numeric override, or None.
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
        _ft = _fixed_temperature_for_model(agent.model, agent.base_url)
        _omit_temp = _ft is OMIT_TEMPERATURE
        _fixed_temp = _ft if not _omit_temp else None
    except Exception:
        _omit_temp = False
        _fixed_temp = None

    # Provider preferences (aggregator profile decides whether to emit them).
    _prefs = _provider_preferences_for_agent(agent)

    # Anthropic-compatible max-output fallback (last resort only — applied in
    # build_kwargs *after* ephemeral/user/profile max_tokens, never overriding
    # an explicit value).  Model-gated, not URL-gated: any chat-completions
    # proxy serving a Claude/MiniMax/Qwen3 model needs max_tokens, because the
    # Anthropic Messages API treats it as mandatory and proxies that omit it
    # (AWS Bedrock, NVIDIA, LiteLLM, vLLM, corporate gateways) default as low
    # as 4096 output tokens — easily exhausted by thinking + large tool calls
    # like write_file/patch.  OpenRouter/Nous were the only routes covered
    # before; gating on _ANTHROPIC_OUTPUT_LIMITS membership covers them all.
    _ant_max = None
    try:
        from agent.anthropic_adapter import (
            _get_anthropic_max_output,
            _ANTHROPIC_OUTPUT_LIMITS,
        )
        _model_norm = (agent.model or "").lower().replace(".", "-")
        if any(key in _model_norm for key in _ANTHROPIC_OUTPUT_LIMITS):
            _ant_max = _get_anthropic_max_output(agent.model)
    except Exception:
        pass

    # Qwen session metadata
    _qwen_meta = None
    if _is_qwen:
        _qwen_meta = {
            "sessionId": agent.session_id or "hermes",
            "promptId": str(uuid.uuid4()),
        }

    # ── Provider profile path (registered providers) vs legacy flag path ──
    try:
        from providers import get_provider_profile
        _profile = get_provider_profile(agent.provider)
    except Exception:
        _profile = None

    # One-shot ephemeral output cap is consumed by whichever path builds the request.
    _ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
    if _ephemeral_out is not None:
        agent._ephemeral_max_output_tokens = None
    # Strip image parts for non-vision models (no-op when vision-capable) —
    # on BOTH paths; registered providers with profiles used to bypass it.
    _msgs_for_chat = agent._prepare_messages_for_non_vision_model(api_messages)
    _common = dict(
        model=agent.model,
        messages=_msgs_for_chat,
        tools=tools_for_api,
        base_url=agent.base_url,
        timeout=agent._resolved_api_call_timeout(),
        max_tokens=agent.max_tokens,
        ephemeral_max_output_tokens=_ephemeral_out,
        max_tokens_param_fn=agent._max_tokens_param,
        reasoning_config=_wire_reasoning_config,
        request_overrides=_request_overrides,
        session_id=getattr(agent, "session_id", None),
        cache_scope_id=_cache_scope_id,
        ollama_num_ctx=agent._ollama_num_ctx,
        provider_preferences=_prefs or None,
        openrouter_min_coding_score=agent.openrouter_min_coding_score,
        anthropic_max_output=_ant_max,
        supports_reasoning=agent._supports_reasoning_extra_body(),
        qwen_session_metadata=_qwen_meta,
    )

    if _profile:
        # Profiles handle per-provider quirks via hooks fed the context above.
        return _ct.build_kwargs(provider_profile=_profile, **_common)

    # ── Legacy flag path ────────────────────────────────────────────
    # Reached only when get_provider_profile() returns None — i.e. a
    # completely unknown provider not in providers/ registry.
    return _ct.build_kwargs(
        **_common,
        model_lower=(agent.model or "").lower(),
        is_openrouter=_is_or,
        is_nous=_is_nous,
        is_qwen_portal=_is_qwen,
        is_github_models=_is_gh,
        is_nvidia_nim=_is_nvidia,
        is_kimi=_is_kimi,
        is_tokenhub=_is_tokenhub,
        is_lmstudio=_is_lmstudio,
        is_custom_provider=agent.provider == "custom",
        qwen_prepare_fn=agent._qwen_prepare_chat_messages if _is_qwen else None,
        qwen_prepare_inplace_fn=agent._qwen_prepare_chat_messages_inplace if _is_qwen else None,
        fixed_temperature=_fixed_temp,
        omit_temperature=_omit_temp,
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if _is_gh else None,
        lmstudio_reasoning_options=agent._lmstudio_reasoning_options_cached() if _is_lmstudio else None,
        provider_name=agent.provider,
    )



def build_assistant_message(agent, assistant_message, finish_reason: str) -> dict:
    """Build a normalized assistant message dict from an API response message.

    Handles reasoning extraction, reasoning_details, and optional tool_calls
    so both the tool-call path and the final-response path share one builder.
    """
    assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
    reasoning_text = agent._extract_reasoning(assistant_message)
    _from_structured = bool(reasoning_text)

    # Fallback: extract inline <think> blocks from content when no structured
    # reasoning fields are present (some models/providers embed thinking
    # directly in the content rather than returning separate API fields).
    if not reasoning_text:
        content = flatten_message_text(getattr(assistant_message, "content", None))
        think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        if think_blocks:
            combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
            reasoning_text = combined or None

    if reasoning_text and agent.verbose_logging:
        logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

    if reasoning_text and agent.reasoning_callback:
        # Skip callback when streaming is active — reasoning was already
        # displayed during the stream via one of two paths:
        #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
        #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
        # When streaming is NOT active, always fire so non-streaming modes
        # (gateway, batch, quiet) still get reasoning.
        # Any reasoning that wasn't shown during streaming is caught by the
        # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
        if not agent.stream_delta_callback and not agent._stream_callback:
            try:
                agent.reasoning_callback(reasoning_text)
            except Exception:
                pass

    # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
    # can return invalid surrogate code points that crash json.dumps() on persist.
    _raw_content = flatten_message_text(getattr(assistant_message, "content", None))
    _san_content = _sanitize_surrogates(_raw_content)
    if reasoning_text:
        reasoning_text = _sanitize_surrogates(reasoning_text)

    # Strip inline <think> tags at the storage boundary — reasoning is already
    # in ``reasoning_text``. Left in, they leaked to messaging platforms
    # (#8878, #9568), inflated context (#9306) and polluted session titles.
    if isinstance(_san_content, str) and _san_content:
        _san_content = agent._strip_think_blocks(_san_content).strip()

    # Redact credentials the model inlined in prose BEFORE the message enters
    # history / state.db / gateway delivery. No-op when HERMES_REDACT_SECRETS
    # is off (#19798).
    if isinstance(_san_content, str) and _san_content:
        from agent.redact import redact_sensitive_text
        _san_content = redact_sensitive_text(_san_content)

    # Textless turns are NOT padded here: ``repair_empty_non_final_messages``
    # (inside ``sanitize_api_messages``, the pre-send chokepoint) is the single
    # owner. Write-time padding was tried and rejected — it broke codex
    # commentary turns (content:'' is designed there) and cannot survive
    # ``_rows_to_conversation``'s whitespace strip.

    msg = stamp_message_timestamp({
        "role": "assistant",
        "content": _san_content,
        "reasoning": reasoning_text,
        "finish_reason": finish_reason,
    })

    raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
    if raw_reasoning_content is None and hasattr(assistant_message, "model_extra"):
        model_extra = getattr(assistant_message, "model_extra", None) or {}
        if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
            raw_reasoning_content = model_extra["reasoning_content"]
    if raw_reasoning_content is not None:
        msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
    elif assistant_tool_calls and agent._needs_thinking_reasoning_pad():
        # DeepSeek v4 / Kimi thinking modes 400 on a replayed tool-call
        # message without reasoning_content. Pad with a single space (empty
        # string is rejected too) without fabricating reasoning.
        # Refs #15250, #17400, #17341.
        msg["reasoning_content"] = reasoning_text or " "

    # Streaming-only providers accumulate reasoning via delta chunks and never
    # set it on the message, so neither branch above fires; replaying through
    # a thinking model then 400s (#16844, #16884). Promote streamed reasoning
    # ONLY when nothing set the field and text was captured: SDK-exposed
    # reasoning_content and the tool-call pad still win, and reasoning-less
    # turns leave the field absent so the replay-time leak guard (#15748)
    # and promotion tiers still apply.
    if "reasoning_content" not in msg and reasoning_text:
        msg["reasoning_content"] = reasoning_text

    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        # Preserve reasoning_details exactly (opaque signature /
        # encrypted_content fields) for cross-turn reasoning continuity.
        raw_details = assistant_message.reasoning_details
        preserved = []
        for d in raw_details:
            if isinstance(d, dict):
                preserved.append(d)
            elif hasattr(d, "__dict__"):
                preserved.append(d.__dict__)
            elif hasattr(d, "model_dump"):
                try:
                    # warnings=False: avoid pydantic serializer UserWarnings
                    # on generic-union SDK models leaking to the terminal.
                    preserved.append(d.model_dump(warnings=False))
                except TypeError:
                    preserved.append(d.model_dump())
        if preserved:
            msg["reasoning_details"] = preserved

    # Anthropic interleaved thinking: reasoning_details + tool_calls lose the
    # cross-type order and reconstruction reorders signed blocks (HTTP 400
    # "thinking blocks ... cannot be modified"). Carry the verbatim ordered
    # block list so the adapter replays the message unchanged.
    ordered_blocks = getattr(assistant_message, "anthropic_content_blocks", None)
    if ordered_blocks:
        msg["anthropic_content_blocks"] = ordered_blocks

    bedrock_blocks = getattr(assistant_message, "bedrock_content_blocks", None)
    if bedrock_blocks:
        msg["bedrock_content_blocks"] = bedrock_blocks

    # Codex Responses API: preserve encrypted reasoning items for
    # multi-turn continuity. These get replayed as input on the next turn.
    codex_items = getattr(assistant_message, "codex_reasoning_items", None)
    if codex_items:
        msg["codex_reasoning_items"] = codex_items

    # Codex Responses API: preserve exact assistant message items (with
    # id/phase) so follow-up turns can replay structured items instead of
    # flattening to plain text. This is required for prefix cache hits.
    codex_message_items = getattr(assistant_message, "codex_message_items", None)
    if codex_message_items:
        msg["codex_message_items"] = codex_message_items

    if assistant_tool_calls:
        tool_calls = []
        for tool_call in assistant_tool_calls:
            raw_id = getattr(tool_call, "id", None)
            call_id = getattr(tool_call, "call_id", None)
            if not isinstance(call_id, str) or not call_id.strip():
                embedded_call_id, _ = agent._split_responses_tool_id(raw_id)
                call_id = embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_id, str) and raw_id.strip():
                    call_id = raw_id.strip()
                else:
                    _fn = getattr(tool_call, "function", None)
                    _fn_name = getattr(_fn, "name", "") if _fn else ""
                    _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                    call_id = agent._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
            call_id = call_id.strip()

            response_item_id = getattr(tool_call, "response_item_id", None)
            if not isinstance(response_item_id, str) or not response_item_id.strip():
                _, embedded_response_item_id = agent._split_responses_tool_id(raw_id)
                response_item_id = embedded_response_item_id

            response_item_id = agent._derive_responses_function_call_id(
                call_id,
                response_item_id if isinstance(response_item_id, str) else None,
            )

            tc_dict = {
                "id": call_id,
                "call_id": call_id,
                "response_item_id": response_item_id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                },
            }
            # Tool-call arguments are deliberately NOT redacted: this dict is
            # replayed to the model every turn (and verbatim on resume), so a
            # `***` mask gets copied into the next call and breaks every
            # credential-dependent command (#43083). It also protected
            # nothing — the secret still leaks via tool OUTPUT.
            # Preserve extra_content (Gemini thought_signature) or Gemini 3
            # thinking models 400 on the next request.
            extra = getattr(tool_call, "extra_content", None)
            if extra is not None:
                if hasattr(extra, "model_dump"):
                    try:
                        extra = extra.model_dump(warnings=False)
                    except TypeError:
                        extra = extra.model_dump()
                tc_dict["extra_content"] = extra
            tool_calls.append(tc_dict)
        msg["tool_calls"] = tool_calls

    return msg



def rewrite_prompt_model_identity(agent, model: str, provider: str) -> None:
    """Point the cached system prompt's ``Model:``/``Provider:`` lines at
    the active runtime after a provider switch.

    The system prompt is session-stable and replayed verbatim for prefix-cache
    warmth, but after a failover the new backend's cache is cold anyway —
    while a stale identity line makes the agent misreport which model it is
    when asked.  Rewrite the lines in place WITHOUT persisting to the session
    DB: the stored row keeps the primary's labels, so when the primary is
    restored the prompt is byte-identical to the stored copy again and its
    prefix cache still matches.

    Only the LAST occurrence of each line is touched — the identity lines
    live in the volatile tail of the prompt, and earlier matches could be
    user content (memory snapshots, context files).
    """
    sp = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(sp, str) or not sp:
        return
    for label, value in (("Model", model), ("Provider", provider)):
        if not value:
            continue
        matches = list(re.finditer(rf"(?m)^{label}: .*$", sp))
        if matches:
            last = matches[-1]
            sp = f"{sp[:last.start()]}{label}: {value}{sp[last.end():]}"
    agent._cached_system_prompt = sp


def _fallback_entry_key(fb: dict) -> tuple[str, str, str]:
    return (
        str(fb.get("provider") or "").strip().lower(),
        str(fb.get("model") or "").strip(),
        str(fb.get("base_url") or "").strip().rstrip("/"),
    )


def _fallback_entry_unavailable_without_network(agent, fb: dict) -> Optional[str]:
    """Return a skip reason for fallback entries known to be unusable locally."""
    fb_provider = (fb.get("provider") or "").strip().lower()
    if fb_provider != "nous":
        return None
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception as exc:
        return f"nous_auth_unreadable:{type(exc).__name__}"
    access_value = state.get("access_token")
    refresh_value = state.get("refresh_token")
    has_access = isinstance(access_value, str) and bool(access_value.strip())
    has_refresh = isinstance(refresh_value, str) and bool(refresh_value.strip())
    if not (has_access or has_refresh):
        return "nous_token_missing"
    return None


def _fallback_reason_text(reason: "FailoverReason | None") -> str:
    """Return a concise operator-facing explanation for a fallback switch."""
    if reason is None:
        return "provider failure"
    labels = {
        FailoverReason.auth: "authentication failed",
        FailoverReason.auth_permanent: "authentication permanently failed",
        FailoverReason.billing: "billing or quota exhausted",
        FailoverReason.rate_limit: "rate limit",
        FailoverReason.upstream_rate_limit: "upstream model rate limit",
        FailoverReason.overloaded: "provider overloaded",
        FailoverReason.server_error: "provider server error",
        FailoverReason.timeout: "request timeout",
        FailoverReason.ssl_cert_verification: "TLS certificate verification failed",
        FailoverReason.context_overflow: "context window exceeded",
        FailoverReason.payload_too_large: "request payload too large",
        FailoverReason.image_too_large: "image payload too large",
        FailoverReason.model_not_found: "model not found",
        FailoverReason.provider_policy_blocked: "provider policy blocked the request",
        FailoverReason.content_policy_blocked: "content policy blocked the request",
        FailoverReason.format_error: "request format rejected",
        FailoverReason.invalid_encrypted_content: "encrypted reasoning state rejected",
        FailoverReason.multimodal_tool_content_unsupported: "multimodal tool content unsupported",
        FailoverReason.thinking_signature: "thinking signature rejected",
        FailoverReason.long_context_tier: "long-context tier unavailable",
        FailoverReason.oauth_long_context_beta_forbidden: "OAuth long-context beta unavailable",
        FailoverReason.llama_cpp_grammar_pattern: "grammar pattern rejected",
        FailoverReason.unknown: "provider failure",
    }
    label = labels.get(reason)
    if label:
        return label
    value = getattr(reason, "value", None)
    return str(value or reason or "provider failure").replace("_", " ")


def _fallback_api_mode_hint(fb: dict, fb_provider: str, fb_base_url_hint: Optional[str]) -> tuple[bool, str]:
    """(explicit, api_mode) for a fallback entry from its ORIGINAL base_url.

    resolve_provider_client() rewrites a dual-surface /anthropic base to /v1,
    losing the Anthropic wire signal, so detection runs on the URL the user
    configured (#79787). An explicit ``api_mode`` on the entry always wins —
    including "chat_completions" — and suppresses all later re-detection.
    ``provider: anthropic`` without a base_url uses the default endpoint and
    must still resolve to anthropic_messages.
    """
    explicit = bool(str(fb.get("api_mode") or "").strip())
    if explicit:
        return True, str(fb.get("api_mode")).strip()
    if fb_provider == "anthropic":
        return False, "anthropic_messages"
    if fb_base_url_hint and (
        fb_base_url_hint.rstrip("/").lower().endswith("/anthropic")
        or base_url_hostname(fb_base_url_hint) == "api.anthropic.com"
    ):
        return False, "anthropic_messages"
    return False, "chat_completions"


def _fallback_api_mode_resolved(agent, fb_provider: str, fb_model: str, fb_base_url: str) -> str:
    """Re-detect api_mode from provider / resolved base URL / model once the
    hint pass landed on the chat_completions default (never called when the
    entry pinned api_mode explicitly)."""
    if fb_provider == "openai-codex":
        return "codex_responses"
    if fb_provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire: anthropic/* must land on /v1/messages.
        # resolve_provider_client still returns an OpenAI client for Nous; the
        # anthropic_messages branch of the swap rebuilds the native client.
        from hermes_cli.providers import nous_api_mode

        return nous_api_mode(fb_model)
    if (
        fb_base_url.rstrip("/").lower().endswith("/anthropic")
        or base_url_hostname(fb_base_url) == "api.anthropic.com"
    ):
        # Named custom providers (e.g. cron-anthropic) resolve base_url from
        # config, so the hint pass never saw it. Same host match as
        # determine_api_mode() / _detect_api_mode_for_url(). (#32243, #49247)
        return "anthropic_messages"
    if agent._is_azure_openai_url(fb_base_url):
        # Azure serves gpt-5.x on /chat/completions — no Responses API.
        return "chat_completions"
    if agent._is_direct_openai_url(fb_base_url):
        return "codex_responses"
    if agent._provider_model_requires_responses_api(fb_model, provider=fb_provider):
        # GPT-5.x usually needs Responses; provider exceptions (Copilot
        # gpt-5-mini) stay on chat completions inside the predicate.
        return "codex_responses"
    if fb_provider == "bedrock" or (
        base_url_hostname(fb_base_url).startswith("bedrock-runtime.")
        and base_url_host_matches(fb_base_url, "amazonaws.com")
    ):
        return "bedrock_converse"
    return "chat_completions"


def _rebind_fallback_credential_pool(agent, fb_provider: str, fb_model: str) -> None:
    """Rebind the credential pool when the provider changes (#33163): keeping
    the primary pool would let rate_limit/billing/auth recovery mutate the
    wrong credential set and overwrite the fallback's base_url. A pool for the
    same provider (two openrouter entries) is preserved; otherwise the
    fallback provider's own pool is loaded so rotation keeps working."""
    existing_pool = getattr(agent, "_credential_pool", None)
    if existing_pool is not None:
        pool_provider = (getattr(existing_pool, "provider", "") or "").strip().lower()
        if pool_provider and pool_provider != fb_provider:
            logger.info(
                "Fallback to %s/%s: clearing primary credential pool "
                "(pool_provider=%s) to prevent cross-provider contamination",
                fb_provider, fb_model, pool_provider,
            )
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
    if getattr(agent, "_credential_pool", None) is None:
        try:
            from agent.credential_pool import load_pool

            fallback_pool = load_pool(fb_provider)
            if fallback_pool and fallback_pool.has_credentials():
                agent._credential_pool = fallback_pool
                logger.info(
                    "Fallback to %s/%s: attached fallback credential pool",
                    fb_provider, fb_model,
                )
        except Exception as exc:
            logger.debug(
                "Fallback to %s/%s: could not attach credential pool: %s",
                fb_provider, fb_model, exc,
            )


def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
    """Switch to the next fallback model/provider in the chain.

    Called when the current model is failing after retries.  Swaps the
    OpenAI client, model slug, and provider in-place so the retry loop
    can continue with the new backend.  Advances through the chain on
    each call; returns False when exhausted.

    Uses the centralized provider router (resolve_provider_client) for
    auth resolution and client construction — no duplicated provider→key
    mappings.
    """
    if reason in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}:
        # Only start cooldown when leaving the primary provider.  If we're
        # already on a fallback and chain-switching, the primary wasn't the
        # source of the 429 so the cooldown should not be reset/extended.
        fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
        if (not fallback_already_active) or (primary_provider and current_provider == primary_provider):
            # Exponential backoff: keep upstream's 60s first-hit cooldown and
            # escalate on CONSECUTIVE rate-limits: 60s → 2m → 4m → 8m → ... →
            # 4h cap. The first 429 must NOT bench the primary for half an
            # hour — fast primary restore is the common case; escalation only
            # punishes providers that keep 429ing.
            # Counter is reset by restore_primary_runtime on successful restore.
            backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
            agent._rate_limit_backoff_count = backoff_count + 1
            backoff_seconds = min(60 * (2 ** backoff_count), 14400)
            agent._rate_limited_until = time.monotonic() + backoff_seconds
            logging.info(
                "Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d)",
                backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1,
            )
    if agent._fallback_index >= len(agent._fallback_chain):
        # Chain exhausted.  If we actually walked a non-empty chain and the
        # failure was NOT a rate-limit/billing event (those already armed
        # their own 60s cooldown above), arm a short cooldown so the next
        # turn's restore_primary_runtime stays gated instead of resetting
        # _fallback_index=0 and re-marshaling the whole context across every
        # provider again.  Guards the cross-turn replay storm in #24996.
        if (
            len(agent._fallback_chain) > 0
            and reason not in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}
        ):
            _existing_cooldown = getattr(agent, "_rate_limited_until", 0) or 0
            agent._rate_limited_until = max(
                _existing_cooldown,
                time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S,
            )
        return False
    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    fb_key = _fallback_entry_key(fb)
    unavailable = getattr(agent, "_unavailable_fallback_keys", None)
    if unavailable is None:
        unavailable = set()
        agent._unavailable_fallback_keys = unavailable
    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return agent._try_activate_fallback(reason)
    fb_provider = (fb.get("provider") or "").strip().lower()
    fb_model = (fb.get("model") or "").strip()
    if not fb_provider or not fb_model:
        return agent._try_activate_fallback(reason)  # skip invalid, try next

    local_skip_reason = _fallback_entry_unavailable_without_network(agent, fb)
    if local_skip_reason:
        unavailable.add(fb_key)
        logger.warning(
            "Fallback skip: %s/%s is not locally usable (%s); suppressing for this session",
            fb_provider,
            fb_model,
            local_skip_reason,
        )
        return agent._try_activate_fallback(reason)

    # Skip entries that resolve to the same backend that just failed —
    # falling back to it loops the failure. Identity semantics (which axes
    # distinguish two backends, shim aliases, first-class credential
    # surfaces, multi-endpoint pools) are owned by agent.backend_identity —
    # see #22548, #70893, #62984. Do not re-implement comparisons here.
    from agent.backend_identity import BackendIdentity, should_skip_candidate

    current_ident = BackendIdentity.build(
        provider=getattr(agent, "provider", ""),
        model=getattr(agent, "model", ""),
        base_url=str(getattr(agent, "base_url", "") or ""),
    )
    fb_ident = BackendIdentity.build(
        provider=fb_provider,
        model=fb_model,
        base_url=(fb.get("base_url") or ""),
    )
    if should_skip_candidate(fb_ident, current_ident):
        logger.warning(
            "Fallback skip: chain entry %s/%s resolves to the same backend "
            "as the current one (%s)",
            fb_provider, fb_model, current_ident.base_url or current_ident.provider,
        )
        return agent._try_activate_fallback(reason)

    # Use centralized router for client construction.
    # raw_codex=True because the main agent needs direct responses.stream()
    # access for Codex providers.
    try:
        from agent.auxiliary_client import resolve_provider_client
        # Pass base_url and api_key from fallback config so custom
        # endpoints (e.g. Ollama Cloud) resolve correctly instead of
        # falling through to OpenRouter defaults.
        from hermes_cli.fallback_config import resolve_entry_api_key

        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = resolve_entry_api_key(fb)
        fb_api_mode_explicit, fb_api_mode = _fallback_api_mode_hint(fb, fb_provider, fb_base_url_hint)

        # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
        # when no explicit key is in the fallback config. Host match
        # (not substring) — see GHSA-76xc-57q6-vm5m.
        if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
            from agent.secret_scope import get_secret

            fb_api_key_hint = get_secret("OLLAMA_API_KEY") or None
        fb_client, _resolved_fb_model = resolve_provider_client(
            fb_provider, model=fb_model, raw_codex=True,
            explicit_base_url=fb_base_url_hint,
            explicit_api_key=fb_api_key_hint,
            api_mode=fb_api_mode)
        if fb_client is None:
            logger.warning(
                "Fallback to %s failed: provider not configured",
                fb_provider)
            unavailable.add(fb_key)
            return agent._try_activate_fallback(reason)  # try next in chain
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider

            fb_model = normalize_model_for_provider(fb_model, fb_provider)
        except Exception as _norm_err:
            logger.warning(
                "Could not normalize fallback model %r for provider %r: %s",
                fb_model, fb_provider, _norm_err,
            )

        fb_base_url = str(fb_client.base_url)
        if not fb_api_mode_explicit and fb_api_mode == "chat_completions":
            fb_api_mode = _fallback_api_mode_resolved(agent, fb_provider, fb_model, fb_base_url)

        old_model = agent.model
        old_provider = agent.provider
        old_base_url = agent.base_url

        # Clear the per-config context_length override so the fallback
        # model's actual context window is resolved instead of inheriting
        # the stale value from the previous model.  See #22387.
        agent._config_context_length = None
        agent.model = fb_model
        agent.provider = fb_provider
        agent.requested_provider = fb_provider
        agent.base_url = fb_base_url
        agent.api_mode = fb_api_mode
        # Per-provider reasoning_content echo opt-in (see _reasoning_echo_opt_in).
        # Read from the fallback entry so the flag travels with the active
        # provider; restore_primary_runtime will revert it from the snapshot.
        agent._reasoning_echo_flag = bool(fb.get("reasoning_echo", False))
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent._fallback_activated = True

        _rebind_fallback_credential_pool(agent, fb_provider, fb_model)

        # Honor per-provider / per-model request_timeout_seconds for the
        # fallback target (same knob the primary client uses).  None = use
        # SDK default.
        _fb_timeout = get_provider_request_timeout(fb_provider, fb_model)

        if fb_api_mode == "anthropic_messages":
            # Build native Anthropic client instead of using OpenAI client
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
            effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = fb_base_url
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url, timeout=_fb_timeout,
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if fb_provider == "anthropic" else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            # Swap OpenAI client and config in-place
            agent.api_key = fb_client.api_key
            agent.client = fb_client
            # Preserve provider-specific headers that
            # resolve_provider_client() may have baked into
            # fb_client via the default_headers kwarg.  The OpenAI
            # SDK stores these in _custom_headers.  Without this,
            # subsequent request-client rebuilds (via
            # _create_request_openai_client) drop the headers,
            # causing 403s from providers like Kimi Coding that
            # require a User-Agent sentinel.
            fb_headers = getattr(fb_client, "_custom_headers", None)
            if not fb_headers:
                fb_headers = getattr(fb_client, "default_headers", None)
            agent._client_kwargs = {
                "api_key": fb_client.api_key,
                "base_url": fb_base_url,
                **({"default_headers": dict(fb_headers)} if fb_headers else {}),
            }
            if _fb_timeout is not None:
                agent._client_kwargs["timeout"] = _fb_timeout
                # Rebuild the shared OpenAI client so the configured
                # timeout takes effect on the very next fallback request,
                # not only after a later credential-rotation rebuild.
                agent._replace_primary_openai_client(reason="fallback_timeout_apply")

        from agent.agent_runtime_helpers import sync_credential_pool_entry_id
        sync_credential_pool_entry_id(agent)

        # Re-evaluate prompt caching for the new provider/model
        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=fb_provider,
                base_url=fb_base_url,
                api_mode=fb_api_mode,
                model=fb_model,
            )
        )

        # LM Studio: preload before probing the fallback's context length.
        agent._ensure_lmstudio_runtime_loaded()

        # Update context compressor limits for the fallback model.
        # Without this, compression decisions use the primary model's
        # context window (e.g. 200K) instead of the fallback's (e.g. 32K),
        # causing oversized sessions to overflow the fallback.
        # Also pass _config_context_length so the explicit config override
        # (model.context_length in config.yaml) is respected — without this,
        # the fallback activation drops to 128K even when config says 204800.
        if hasattr(agent, 'context_compressor') and agent.context_compressor:
            from agent.model_metadata import get_model_context_length
            # ``agent.api_key`` may be callable (Entra ID); the
            # context-length resolver expects a string for live
            # probes. Foundry typically resolves via config/static
            # catalogs anyway, so coerce defensively.
            _fb_ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
            fb_context_length = get_model_context_length(
                agent.model, base_url=agent.base_url,
                api_key=_fb_ctx_api_key, provider=agent.provider,
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=fb_context_length,
                base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),  # callable preserved → call_llm
                provider=agent.provider,
                api_mode=agent.api_mode,
            )

        # Re-resolve reasoning_config for the new fallback model (Closes #21256).
        # Shared chokepoint: per-model override > global reasoning_effort
        # (YAML boolean False = disabled). Wrapped in try/except because a
        # config load failure must not kill the swap.
        try:
            from hermes_cli.config import load_config
            from hermes_constants import resolve_reasoning_config

            agent.reasoning_config = resolve_reasoning_config(
                load_config() or {}, agent.model
            )
            logger.info(
                "Fallback %s: reasoning_config resolved: %s",
                agent.model, agent.reasoning_config,
            )
        except Exception as _reasoning_err:
            logger.debug(
                "Failed to resolve reasoning_config for fallback %s; keeping current: %s",
                agent.model, _reasoning_err,
            )
            # Keep whatever reasoning_config was active — don't break the fallback swap.

        # Re-resolve extra_body for the fallback provider (Closes #75091).
        # The OLD provider's custom_providers-contributed extra_body (e.g. a
        # vendor-specific reasoning toggle) must not ride along onto the
        # fallback provider, which is a different API that may reject those
        # fields.  Removal is KEY-SCOPED: only keys the old provider's
        # custom_providers entry contributed (value unchanged since init)
        # are dropped; the fallback provider's own extra_body is then merged
        # back in.  Caller/profile-provided extra_body keys
        # (request_overrides passed at init, which win over provider config
        # per _merge_custom_provider_extra_body precedence) MUST survive the
        # swap untouched.
        try:
            from agent.agent_init import (
                _custom_provider_extra_body_for_agent,
                _merge_custom_provider_extra_body,
            )
            _custom_providers = getattr(agent, "_custom_providers", None) or []
            # What did the OLD provider's config contribute?
            _old_provider_eb = _custom_provider_extra_body_for_agent(
                provider=old_provider,
                model=old_model,
                base_url=old_base_url,
                custom_providers=_custom_providers,
            ) or {}
            _overrides = dict(getattr(agent, "request_overrides", {}) or {})
            _existing_eb = _overrides.get("extra_body")
            if isinstance(_existing_eb, dict) and _old_provider_eb:
                _scrubbed = dict(_existing_eb)
                for _k, _v in _old_provider_eb.items():
                    # Drop only keys the old provider contributed: the value
                    # must still match what its config injected — a caller
                    # override of the same key would have won at init and
                    # differ, so it survives.  Keys the new provider
                    # redefines are re-added with the NEW provider's value
                    # by the merge below.
                    if _k in _scrubbed and _scrubbed[_k] == _v:
                        _scrubbed.pop(_k)
                if _scrubbed:
                    _overrides["extra_body"] = _scrubbed
                else:
                    _overrides.pop("extra_body", None)
                agent.request_overrides = _overrides
            # Merge in the fallback provider's own extra_body (existing
            # caller-provided keys win on conflict inside the merge helper).
            _merge_custom_provider_extra_body(agent, _custom_providers)
            logger.info(
                "Fallback %s: extra_body resolved: %s",
                agent.model,
                (getattr(agent, "request_overrides", {}) or {}).get("extra_body"),
            )
        except Exception as _eb_err:
            logger.debug(
                "Failed to resolve extra_body for fallback %s; keeping current: %s",
                agent.model, _eb_err,
            )

        # Keep the prompt's self-identity in sync with the model actually
        # answering, so "what model are you?" doesn't report the primary.
        rewrite_prompt_model_identity(agent, fb_model, fb_provider)

        notice = (
            f"⚠️ Model fallback: {old_model} via {old_provider} unavailable "
            f"({_fallback_reason_text(reason)}); using {fb_model} via {fb_provider}."
        )
        # The buffered switch is surfaced on terminal failure. A successful
        # fallback clears retry chatter, so retain every switch as a durable
        # one-shot notice for _emit_pending_fallback_notice (run_agent.py).
        agent._buffer_status(notice)
        pending = getattr(agent, "_pending_fallback_notice", None)
        if isinstance(pending, list):
            pending.append(notice)
        elif pending:
            agent._pending_fallback_notice = [str(pending), notice]
        else:
            agent._pending_fallback_notice = [notice]
        # ``_fallback_activated`` is also reused by temporary `/model --once`
        # restoration. Keep separate provenance so the restore path only emits
        # a fallback-recovery notice after an actual provider fallback.
        agent._provider_fallback_active = True
        agent._provider_fallback_route = (str(fb_model), str(fb_provider))
        logger.info(
            "Fallback activated: %s → %s (%s)",
            old_model, fb_model, fb_provider,
        )
        # Reset the stale-call circuit breaker (#58962): the streak measured
        # the OLD provider's unresponsiveness.  Carrying it over would
        # short-circuit the freshly activated fallback before it gets a
        # single stream attempt.
        _reset_stale_streak(agent)
        from agent.native_compaction import resolve_native_compaction_capabilities
        agent.runtime_capabilities = resolve_native_compaction_capabilities(
            model=agent.model,
            base_url=agent.base_url,
            provider=fb_provider,
            is_codex_backend=fb_provider == "openai-codex",
        )
        return True
    except Exception as e:
        if fb_provider == "nous":
            unavailable.add(fb_key)
        logger.error("Failed to activate fallback %s: %s", fb_model, e)
        return agent._try_activate_fallback(reason)  # try next in chain



def handle_max_iterations(agent, messages: list, api_call_count: int) -> str:
    """Request a summary when max iterations are reached. Returns the final response text."""
    warning = f"⚠️  Reached maximum iterations ({agent.max_iterations}). Requesting summary..."
    if getattr(agent, "suppress_status_output", False):
        # Strict machine-readable mode (hermes chat -Q, oneshot, background
        # review): keep diagnostics out of stdout so wrappers receive only
        # the final assistant content (#93220 class). Note: plain quiet_mode
        # is NOT the right gate — the interactive CLI runs quiet_mode=True by
        # default and should still see this warning.
        logger.warning(warning)
    else:
        agent._safe_print(warning)

    summary_api_request_id = f"iteration-summary:{uuid.uuid4()}"
    summary_call_outcome = "failed"

    def _managed_summary_call(request, callback, *, retry_count: int):
        from agent import relay_llm

        return relay_llm.execute_current(
            request,
            callback,
            name=str(getattr(agent, "provider", "") or "provider"),
            model_name=str(getattr(agent, "model", "") or ""),
            metadata={
                "api_mode": str(
                    getattr(agent, "api_mode", "") or "chat_completions"
                ),
                "api_request_id": summary_api_request_id,
                "call_role": "iteration_summary",
                "retry_count": retry_count,
            },
            defer_logical_completion=True,
        )

    # Shared constant so compaction recognizers can identify this runtime nudge
    # by its stable content after SessionDB projection strips metadata flags
    # (see MAX_ITERATIONS_SUMMARY_REQUEST / _is_synthetic_compression_user_turn).
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

    summary_request = MAX_ITERATIONS_SUMMARY_REQUEST
    append_message(messages, {"role": "user", "content": summary_request})

    try:
        # Build API messages, stripping internal-only fields
        # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
        _needs_sanitize = agent._should_sanitize_tool_calls()
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            agent._copy_reasoning_content_for_api(msg, api_msg)
            for internal_field in ("reasoning", "finish_reason"):
                api_msg.pop(internal_field, None)
            # Strict OpenAI-compatible gateways (Fireworks-backed OpenCode Go,
            # Mistral, Moonshot/Kimi) reject any message key outside the Chat
            # Completions schema. The main loop drops these via
            # ChatCompletionsTransport.convert_messages(), but the summary path
            # hand-builds messages and calls chat.completions.create() directly,
            # bypassing the transport — so mirror that sanitization here:
            # tool_name (SQLite FTS bookkeeping), the codex_* reasoning carriers,
            # timestamp (preserved on gateway user replay entries for the
            # stale-confirmation expiry check — #47868 rejection class),
            # and every Hermes-internal underscore-prefixed scaffolding key.
            for schema_foreign in ("tool_name", "codex_reasoning_items", "codex_message_items", "timestamp", "platform_message_id"):
                api_msg.pop(schema_foreign, None)
            # api_content (the persist-what-you-send sidecar) carries the
            # exact bytes every main-loop call sent for this message —
            # substitute it before dropping the key (Hermes bookkeeping,
            # never a provider field), mirroring the loop's api_messages
            # build. Popping without substituting would send CLEAN content
            # here, diverging the summary request's prefix at the EARLIEST
            # sidecar-carrying message and re-prefilling the whole transcript
            # at exactly the moment the context is largest.
            substitute_api_content(api_msg)
            if _needs_sanitize:
                # In MoA mode, agent.model is the virtual preset name,
                # not the actual aggregator model.  Resolve the real
                # aggregator model so Gemini preserves thought_signature.
                _sanitize_model = agent.model
                if agent.provider == "moa":
                    _moa_client = getattr(agent, "client", None)
                    if _moa_client is not None:
                        _agg_slot = getattr(_moa_client, "last_aggregator_slot", None)
                        if _agg_slot and _agg_slot.get("model"):
                            _sanitize_model = _agg_slot["model"]
                agent._sanitize_tool_calls_for_strict_api(api_msg, model=_sanitize_model)
            api_messages.append(api_msg)

        effective_system = agent._cached_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages
        if agent.prefill_messages:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Same safety net as the main loop: repair tool-call/result
        # pairing before asking for a final summary.  Compression and
        # session resume can leave a tool result whose parent assistant
        # tool_call was summarized away; Responses API rejects that as
        # "No tool call found for function call output".
        api_messages = agent._sanitize_api_messages(api_messages)

        # Same safety net as the main loop: drop thinking-only assistant
        # turns so Anthropic-family providers don't 400 the summary call.
        # _thinking_prefill must survive until here so the drop pass can
        # recognize stubs after reasoning fields are stripped.
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        # Strip all remaining underscore-prefixed scaffolding keys before the
        # wire. The summary path calls chat.completions.create() directly,
        # bypassing the transport's universal underscore-key sweeper.
        for api_msg in api_messages:
            if isinstance(api_msg, dict):
                for internal_key in [k for k in api_msg if isinstance(k, str) and k.startswith("_")]:
                    api_msg.pop(internal_key, None)

        summary_extra_body = {}
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
        except Exception:
            _fixed_temperature_for_model = None
            _OMIT_TEMP = None
        _raw_summary_temp = (
            _fixed_temperature_for_model(agent.model, agent.base_url)
            if _fixed_temperature_for_model is not None
            else None
        )
        _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
        _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
        _is_nous = "nousresearch" in agent._base_url_lower
        # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
        # Mirror ChatCompletionsTransport.build_kwargs() so the summary path
        # — which calls chat.completions.create() directly without going
        # through the transport — sends the same shape the transport does.
        _is_lmstudio_summary = (
            (agent.provider or "").strip().lower() == "lmstudio"
            and agent._supports_reasoning_extra_body()
        )
        _lm_reasoning_effort: str | None = (
            agent._resolve_lmstudio_summary_reasoning_effort()
            if _is_lmstudio_summary else None
        )
        if not _is_lmstudio_summary and agent._supports_reasoning_extra_body():
            if agent.reasoning_config is not None:
                summary_extra_body["reasoning"] = agent.reasoning_config
            else:
                summary_extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": "medium"
                }
        if _is_nous:
            from agent.portal_tags import nous_portal_tags as _portal_tags
            summary_extra_body["tags"] = _portal_tags()

        if agent.api_mode == "codex_responses":
            def _attempt(retry_count: int) -> str:
                codex_kwargs = agent._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                response = agent._run_codex_stream(codex_kwargs)
                return (agent._get_transport().normalize_response(response).content or "").strip()
        elif agent.api_mode == "anthropic_messages":
            def _attempt(retry_count: int) -> str:
                transport = agent._get_transport()
                ant_kw = transport.build_kwargs(
                    model=agent.model,
                    messages=api_messages,
                    tools=None,
                    max_tokens=agent.max_tokens,
                    reasoning_config=agent.reasoning_config,
                    is_oauth=agent._is_anthropic_oauth,
                    preserve_dots=agent._anthropic_preserve_dots(),
                    base_url=getattr(agent, "_anthropic_base_url", None),
                )
                ant_kw = _merge_nous_portal_messages_extra_body(agent, ant_kw)
                response = _managed_summary_call(
                    ant_kw, agent._anthropic_messages_create, retry_count=retry_count,
                )
                result = transport.normalize_response(response, strip_tool_prefix=agent._is_anthropic_oauth)
                return (result.content or "").strip()
        else:
            summary_kwargs = {
                "model": agent.model,
                "messages": api_messages,
            }
            if _summary_temperature is not None:
                summary_kwargs["temperature"] = _summary_temperature
            if agent.max_tokens is not None:
                summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
            if _lm_reasoning_effort is not None:
                summary_kwargs["reasoning_effort"] = _lm_reasoning_effort

            # Merge the profile's canonical body even when routing is unset:
            # profiles may always emit required metadata such as Portal tags.
            provider_preferences = _provider_preferences_for_agent(agent)
            profile_extra_body = {}
            try:
                from providers import get_provider_profile

                provider_profile = get_provider_profile(agent.provider)
                if provider_profile is not None:
                    profile_extra_body = provider_profile.build_extra_body(
                        session_id=getattr(agent, "session_id", None),
                        provider_preferences=provider_preferences or None,
                        model=agent.model,
                        base_url=agent.base_url,
                        reasoning_config=agent.reasoning_config,
                    )
            except Exception:
                pass

            if profile_extra_body:
                summary_extra_body.update(profile_extra_body)
            if provider_preferences and "provider" not in profile_extra_body and (
                (agent.provider or "").strip().lower() == "openrouter"
                or agent._is_openrouter_url()
            ):
                summary_extra_body["provider"] = provider_preferences

            # Pareto Code router plugin — model-gated. Same shape as
            # the main-loop emission so summary calls on
            # openrouter/pareto-code respect the user's coding-score floor.
            if (
                agent.model == "openrouter/pareto-code"
                and (
                    (agent.provider or "").strip().lower() == "openrouter"
                    or agent._is_openrouter_url()
                )
                and agent.openrouter_min_coding_score is not None
                and agent.openrouter_min_coding_score != ""
            ):
                try:
                    _ps = float(agent.openrouter_min_coding_score)
                except (TypeError, ValueError):
                    _ps = None
                if _ps is not None and 0.0 <= _ps <= 1.0:
                    summary_extra_body["plugins"] = [
                        {"id": "pareto-router", "min_coding_score": _ps}
                    ]

            if summary_extra_body:
                summary_kwargs["extra_body"] = summary_extra_body

            def _attempt(retry_count: int) -> str:
                summary_client = agent._ensure_primary_openai_client(
                    reason="iteration_limit_summary_retry" if retry_count else "iteration_limit_summary"
                )
                response = _managed_summary_call(
                    summary_kwargs,
                    lambda request: summary_client.chat.completions.create(**request),
                    retry_count=retry_count,
                )
                return (agent._get_transport().normalize_response(response).content or "").strip()

        # One retry on an empty summary; a summary that is empty once its
        # <think> block is stripped is NOT retried (matches prior behavior).
        for retry_count in (0, 1):
            final_response = _attempt(retry_count)
            if not final_response:
                continue
            if "<think>" in final_response:
                final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
            if final_response:
                summary_call_outcome = "success"
                append_message(
                    messages,
                    {"role": "assistant", "content": final_response},
                )
            else:
                final_response = "I reached the iteration limit and couldn't generate a summary."
            break
        else:
            final_response = "I reached the iteration limit and couldn't generate a summary."

    except Exception as e:
        logger.warning("Failed to get summary response: %s", e)
        final_response = f"I reached the maximum iterations ({agent.max_iterations}) but couldn't summarize. Error: {str(e)}"
    finally:
        from agent import relay_llm

        relay_llm.complete_logical_call(
            summary_api_request_id,
            outcome=summary_call_outcome,
        )

    return final_response



def cleanup_task_resources(agent, task_id: str) -> None:
    """Clean up VM and browser resources for a given task.

    Skips ``cleanup_vm`` when the active terminal environment is marked
    persistent (``persistent_filesystem=True``) so that long-lived sandbox
    containers survive between turns. The idle reaper in
    ``terminal_tool._cleanup_inactive_envs`` still tears them down once
    ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
    torn down per-turn as before to prevent resource leakage (the original
    intent of this hook for the Morph backend, see commit fbd3a2fd).

    Skips ``cleanup_browser`` in headed mode so the browser window stays
    visible between turns. The inactivity reaper in
    ``browser_tool._cleanup_inactive_browser_sessions`` still handles
    idle sessions.
    """
    try:
        if is_persistent_env(task_id):
            if agent.verbose_logging:
                logging.debug(
                    f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                    f"idle reaper will handle it."
                )
        else:
            _ra().cleanup_vm(task_id)
    except Exception as e:
        if agent.verbose_logging:
            logger.warning("Failed to cleanup VM for task %s: %s", task_id, e)
    try:
        headed = False
        try:
            from tools.browser_tool import _is_headed_mode
            headed = _is_headed_mode()
        except Exception:
            headed = bool(os.environ.get("AGENT_BROWSER_HEADED"))
        if headed:
            if agent.verbose_logging:
                logging.debug(
                    f"Skipping per-turn cleanup_browser for headed session {task_id}; "
                    f"idle reaper will handle it."
                )
        else:
            _ra().cleanup_browser(task_id)
    except Exception as e:
        if agent.verbose_logging:
            logger.warning("Failed to cleanup browser for task %s: %s", task_id, e)


def _build_partial_stream_stub(
    role, full_content, full_reasoning, model_name, usage_obj, *,
    dropped_tool_names=None,
):
    """Build a partial-stream-stub response for mid-stream drop scenarios.

    Used when the SSE stream ends without a ``finish_reason`` after
    delivering content (text-only drops, tool-call-arg drops).  The stub
    is tagged ``PARTIAL_STREAM_STUB_ID`` with ``FINISH_REASON_LENGTH`` so
    the conversation loop enters its continuation/retry path instead of
    silently accepting truncated output as a complete turn (#32086).
    """
    mock_message = SimpleNamespace(
        role=role,
        content=full_content,
        tool_calls=None,
        reasoning_content=full_reasoning,
    )
    mock_choice = SimpleNamespace(
        index=0,
        message=mock_message,
        finish_reason=FINISH_REASON_LENGTH,
    )
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model=model_name,
        choices=[mock_choice],
        usage=usage_obj,
        _dropped_tool_names=dropped_tool_names or None,
    )


# SSE error events from proxies (e.g. OpenRouter's
# {"error":{"message":"Network connection lost."}}) are raised as APIError by
# the OpenAI SDK. They are semantically identical to httpx connection drops —
# the upstream stream died — and are retried with a fresh connection.
# Distinguished from HTTP errors by the missing status_code (APIStatusError
# for 4xx/5xx always carries one).
_SSE_CONN_PHRASES = (
    "connection lost",
    "connection reset",
    "connection closed",
    "connection terminated",
    "network error",
    "network connection",
    "terminated",
    "peer closed",
    "broken pipe",
    "upstream connect error",
)


def _is_sse_connection_error(exc: BaseException) -> bool:
    from openai import APIError as _APIError

    if not isinstance(exc, _APIError) or getattr(exc, "status_code", None):
        return False
    err_lower = str(exc).lower()
    return any(phrase in err_lower for phrase in _SSE_CONN_PHRASES)


def _relay_stream_identity(agent, name_default: str) -> dict:
    """``session_id``/``name``/``model_name`` kwargs for ``relay_llm.stream``."""
    return {
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "name": str(getattr(agent, "provider", "") or name_default),
        "model_name": str(getattr(agent, "model", "") or ""),
    }


def _relay_stream_metadata(agent, api_mode: str) -> dict:
    return {
        "api_mode": api_mode,
        "api_request_id": getattr(agent, "_current_api_request_id", None),
        "call_role": (
            "delegated"
            if getattr(agent, "is_subagent", False)
            else "fallback"
            if int(getattr(agent, "_fallback_index", 0) or 0) > 0
            else "primary"
        ),
    }


def _stream_final_text(response) -> str:
    try:
        choices = getattr(response, "choices", None)
        first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    except Exception:
        pass
    try:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
    except Exception:
        pass
    return ""


def _emit_stream_start(agent) -> None:
    emit = getattr(agent, "_emit_stream_start", None)
    if emit is not None:
        emit()


def _emit_stream_end(agent, *, final_text: str, finished: bool, error: str | None) -> None:
    emit = getattr(agent, "_emit_stream_end", None)
    if emit is not None:
        emit(final_text=final_text, finished=finished, error=error)


def _stream_codex_passthrough(agent, api_kwargs: dict, on_first_delta):
    """Codex streams internally via _run_codex_stream (reached through
    _interruptible_api_call); park ``on_first_delta`` on the agent so it can pick
    it up, and bracket the call with the stream start/end emitters."""
    agent._codex_on_first_delta = on_first_delta
    _emit_stream_start(agent)
    try:
        response = agent._interruptible_api_call(api_kwargs)
        _emit_stream_end(agent, final_text=_stream_final_text(response), finished=True, error=None)
        return response
    except Exception as exc:
        _emit_stream_end(agent, final_text="", finished=False, error=str(exc))
        raise
    finally:
        agent._codex_on_first_delta = None


def _stream_bedrock_converse(agent, api_kwargs: dict, on_first_delta):
    """Bedrock Converse: boto3 ``converse_stream()`` on a worker thread with
    real-time delta callbacks, polled by an interrupt / stale-event watchdog
    (same UX as the Anthropic and chat_completions streams)."""
    result = {"response": None, "error": None}
    first_delta_fired = {"done": False}
    deltas_were_sent = {"yes": False}
    # Liveness for the boto3 worker: ``for event in event_stream`` has NO
    # read timeout, so on_event stamps every Bedrock event and the poll loop
    # trips a watchdog when the gap exceeds the stale timeout.
    _bedrock_started_at = time.time()
    _bedrock_last_event = {"t": _bedrock_started_at}
    _bedrock_response_started = {"yes": False}
    # Read (not popped): the worker's own pop inside _bedrock_call must
    # still resolve the same region.
    _bedrock_region = api_kwargs.get("__bedrock_region__", "us-east-1")
    # Same patience budget as the OpenAI/Anthropic stale detector.
    _bedrock_stale_timeout = _derive_stream_stale_timeout(agent, api_kwargs)

    # Cross-turn stale-stream circuit breaker (#58962), as on the OpenAI/
    # Anthropic path.
    _check_stale_giveup(agent)

    def _fire_first():
        if not first_delta_fired["done"] and on_first_delta:
            first_delta_fired["done"] = True
            try:
                on_first_delta()
            except Exception:
                pass

    def _bedrock_call():
        stream = None
        try:
            from agent import relay_llm
            from agent.bedrock_adapter import (
                _get_bedrock_runtime_client,
                invalidate_runtime_client,
                is_stale_connection_error,
                is_streaming_access_denied_error,
                normalize_converse_response,
                recover_from_cache_point_rejection,
                stream_converse_with_callbacks,
            )
            intercepted_events = []
            writer_token = {"value": None}

            def _open_bedrock_stream(next_api_kwargs: dict[str, Any]):
                final_kwargs = dict(next_api_kwargs)
                region = final_kwargs.pop("__bedrock_region__", "us-east-1")
                final_kwargs.pop("__bedrock_converse__", None)
                client = _get_bedrock_runtime_client(region)
                try:
                    raw_response = client.converse_stream(**final_kwargs)
                except Exception as _bedrock_exc:
                    # Some families refuse a cachePoint block in one section
                    # (Nova: toolConfig.tools, #97281): drop it and reopen
                    # inside the same Relay attempt.
                    _retry_kwargs = recover_from_cache_point_rejection(
                        _bedrock_exc, final_kwargs
                    )
                    if _retry_kwargs is not None:
                        return client.converse_stream(**_retry_kwargs).get(
                            "stream", []
                        )
                    # InvokeModel-only IAM policies cannot stream; fall back
                    # inside the same Relay attempt (one lifecycle boundary).
                    if is_streaming_access_denied_error(_bedrock_exc):
                        agent._disable_streaming = True
                        agent._safe_print(
                            "\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream — "
                            "falling back to non-streaming InvokeModel.\n"
                            "   Grant that action to restore streaming output.\n"
                        )
                        logger.info(
                            "bedrock: converse_stream denied by IAM (%s) — "
                            "using non-streaming converse() for this session.",
                            type(_bedrock_exc).__name__,
                        )
                        return normalize_converse_response(
                            client.converse(**final_kwargs)
                        )
                    if is_stale_connection_error(_bedrock_exc):
                        invalidate_runtime_client(region)
                    raise
                return raw_response.get("stream", [])

            def _on_text(text):
                _bedrock_response_started["yes"] = True
                _fire_first()
                agent._fire_stream_delta(text)
                deltas_were_sent["yes"] = True

            def _on_tool(name):
                _bedrock_response_started["yes"] = True
                _fire_first()
                agent._fire_tool_gen_started(name)

            def _on_reasoning(text):
                _bedrock_response_started["yes"] = True
                _fire_first()
                agent._fire_reasoning_delta(text)

            def _finalize_bedrock_stream():
                return stream_converse_with_callbacks(
                    {"stream": list(intercepted_events)}
                )

            def _bedrock_stream_created(_stream: Any) -> None:
                writer_token["value"] = claim_stream_writer(agent)

            def _accept_bedrock_event(_event: Any) -> bool:
                token = writer_token["value"]
                return token is None or stream_writer_is_current(agent, token)

            try:
                from agent.plugin_stream_hooks import has_reasoning_stream_observer_hooks

                plugin_reasoning_observer = has_reasoning_stream_observer_hooks()
            except Exception:
                logger.debug("plugin reasoning stream observer check failed", exc_info=True)
                plugin_reasoning_observer = False

            stream = relay_llm.stream(
                dict(api_kwargs),
                _open_bedrock_stream,
                **_relay_stream_identity(agent, "bedrock"),
                finalizer=_finalize_bedrock_stream,
                on_stream_created=_bedrock_stream_created,
                on_chunk=intercepted_events.append,
                chunk_adapter=lambda chunk: chunk,
                accept_chunk=_accept_bedrock_event,
                completed_response_predicate=lambda response: bool(
                    getattr(response, "choices", None)
                ),
                metadata=_relay_stream_metadata(agent, "custom"),
                defer_logical_completion=True,
            )
            streamed_response = stream_converse_with_callbacks(
                {"stream": stream},
                on_text_delta=_on_text if agent._has_stream_consumers() else None,
                on_tool_start=_on_tool,
                on_reasoning_delta=_on_reasoning
                if agent.reasoning_callback or agent.stream_delta_callback or plugin_reasoning_observer
                else None,
                on_interrupt_check=lambda: agent._interrupt_requested,
                on_event=lambda: _bedrock_last_event.__setitem__("t", time.time()),
            )
            result["response"] = stream.final_response or streamed_response
        except Exception as e:
            result["error"] = e
        finally:
            if stream is not None:
                stream.close()

    _emit_stream_start(agent)
    try:
        t = threading.Thread(
            target=_context_thread_target(_bedrock_call), daemon=True
        )
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if agent._interrupt_requested:
                _record_interrupted_provider_wait(
                    agent,
                    time.time() - _bedrock_started_at,
                    response_started=_bedrock_response_started["yes"],
                )
                # Let the worker unwind Relay scopes before raising (#81521).
                _join_worker_for_relay_teardown(t, label="Bedrock streaming")
                raise InterruptedError("Agent interrupted during Bedrock API call")
            # Liveness watchdog: no event past the stale timeout = wedged
            # stream (the worker would block in the event loop forever).
            _stale_elapsed = time.time() - _bedrock_last_event["t"]
            if _stale_elapsed > _bedrock_stale_timeout:
                logger.warning(
                    "Bedrock stream stale for %.0fs (threshold %.0fs) — no events "
                    "received. region=%s model=%s. Aborting call.",
                    _stale_elapsed, _bedrock_stale_timeout,
                    _bedrock_region, api_kwargs.get("modelId", "unknown"),
                )
                agent._buffer_status(
                    f"⚠️ No events from Bedrock for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('modelId', 'unknown')}). Aborting..."
                )
                _bump_stale_streak(agent)
                # Evict the region's cached client so the NEXT call gets a
                # fresh pool. This does NOT abort the in-flight botocore
                # EventStream (no external cancellation exists); the daemon
                # worker keeps reading until its socket errors, so THIS call
                # ends via the TimeoutError below and the streak escalates.
                try:
                    from agent.bedrock_adapter import invalidate_runtime_client
                    invalidate_runtime_client(_bedrock_region)
                except Exception as _inval_exc:
                    logger.debug(
                        "bedrock: stale client eviction failed: %s", _inval_exc
                    )
                _bedrock_last_event["t"] = time.time()
                # Raises RuntimeError past HERMES_STREAM_STALE_GIVEUP; otherwise
                # end THIS call with a TimeoutError (break — we cannot abort the
                # worker) and let the streak carry forward.
                _check_stale_giveup(agent)
                result["error"] = TimeoutError(
                    f"Bedrock stream produced no events for {int(_stale_elapsed)}s "
                    f"(threshold {int(_bedrock_stale_timeout)}s) — aborting stalled "
                    f"stream so the retry/fallback path can recover."
                )
                break
        # The Bedrock callback returns a PARTIAL response on interrupt without
        # raising (on_interrupt_check), so the in-loop raise may never fire.
        # Re-check so /stop is not swallowed (#59999 area).
        if agent._interrupt_requested:
            _record_interrupted_provider_wait(
                agent,
                time.time() - _bedrock_started_at,
                response_started=_bedrock_response_started["yes"],
            )
            raise InterruptedError("Agent interrupted during Bedrock API call (post-worker)")
        if result["error"] is not None:
            raise result["error"]
        # Success clears the cross-turn breaker (#58962).
        if result["response"] is not None:
            _reset_stale_streak(agent)
        _emit_stream_end(agent, final_text=_stream_final_text(result["response"]), finished=True, error=None)
        return result["response"]
    except Exception as exc:
        _emit_stream_end(agent, final_text="", finished=False, error=str(exc))
        raise


class _ToolCallAccumulator:
    """Assemble streamed tool-call deltas into complete ``tool_calls`` entries.

    ``acc`` maps slot index -> ``{"id","type","function":{"name","arguments"},
    "extra_content"}``. Ollama-compatible endpoints reuse index 0 for every
    tool call in a parallel batch, distinguishing them only by id, so a new id
    arriving at an already-seen raw index is redirected to a fresh slot.
    """

    def __init__(self):
        self.acc: dict = {}
        self._notified: set = set()
        self._last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
        self._active_slot_by_idx: dict = {}  # raw_index -> current slot in acc

    def feed(self, tc_delta) -> Optional[str]:
        """Merge one delta; return the tool name the first time it is complete."""
        raw_index = getattr(tc_delta, "index", None)
        raw_idx = raw_index if raw_index is not None else 0
        tc_id = getattr(tc_delta, "id", None)
        delta_id = tc_id or ""
        if isinstance(tc_id, int):  # Poolside sends integer ids
            tc_id = str(tc_id)

        if raw_idx not in self._active_slot_by_idx:
            self._active_slot_by_idx[raw_idx] = raw_idx
        if (
            delta_id
            and raw_idx in self._last_id_at_idx
            and delta_id != self._last_id_at_idx[raw_idx]
        ):
            self._active_slot_by_idx[raw_idx] = max(self.acc, default=-1) + 1
        if delta_id:
            self._last_id_at_idx[raw_idx] = delta_id
        idx = self._active_slot_by_idx[raw_idx]

        entry = self.acc.setdefault(idx, {
            "id": tc_id or "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
            "extra_content": None,
        })
        if tc_id:
            entry["id"] = tc_id
        tc_function = getattr(tc_delta, "function", None)
        if tc_function:
            function_name = getattr(tc_function, "name", None)
            if function_name:
                # Assignment, not +=: names arrive complete (OpenAI spec) and
                # some providers (MiniMax M2.7 via NVIDIA NIM) resend the full
                # name in every chunk — concatenation gives "read_fileread_file".
                entry["function"]["name"] = function_name
            function_arguments = getattr(tc_function, "arguments", None)
            if function_arguments:
                entry["function"]["arguments"] += function_arguments
        extra = getattr(tc_delta, "extra_content", None)
        if extra is None and hasattr(tc_delta, "model_extra"):
            extra = (tc_delta.model_extra if isinstance(tc_delta.model_extra, dict) else {}).get("extra_content")
        if extra is not None:
            if hasattr(extra, "model_dump"):
                try:
                    extra = extra.model_dump(warnings=False)
                except TypeError:
                    extra = extra.model_dump()
            entry["extra_content"] = extra
        name = entry["function"]["name"]
        if name and idx not in self._notified:
            self._notified.add(idx)
            return name
        return None


class _StreamingCall:
    """One streaming request on the chat_completions / anthropic_messages wire.

    State shared between the request worker (``_call`` and the per-wire
    ``_call_chat_completions`` / ``_call_anthropic``) and the poll-loop monitor
    (heartbeat, stale-stream kill, interrupt abort) lives on the instance;
    the dict/lock holders are mutated in place from both threads.
    """

    def __init__(self, agent, api_kwargs: dict, on_first_delta):
        self.agent = agent
        self.api_kwargs = api_kwargs
        self.on_first_delta = on_first_delta
        self.worker = None  # request thread; None in inline mode
        self.result = {"response": None, "error": None, "partial_tool_names": []}

        self.clients = _RequestClientRegistry(agent)
        # Request-local cancellation flag — see interruptible_api_call for the full
        # rationale. The streaming retry loop is where the 7-minute cascading-
        # interrupt hang originated: a force-close raised RemoteProtocolError, the
        # loop classified it as a transient network error, and burned full retry
        # cycles (and emitted "reconnecting" noise) on a request the user already
        # cancelled. The token lets the worker recognize its own forced close and
        # exit immediately instead of retrying. (PR #6600.)
        self._request_cancelled = {"value": False}

        self.first_delta_fired = {"done": False}
        self.deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
        self.provider_tool_in_flight = {"yes": False}
        # Wall-clock timestamp of the last real streaming chunk.  The outer
        # poll loop uses this to detect stale connections that keep receiving
        # SSE keep-alive pings but no actual data.
        self.last_chunk_time = {"t": time.time()}
        # Stale-stream patience, shared between the httpx socket read timeout
        # (built in ``_call_chat_completions`` below) and the stale-stream detector
        # (computed further down, before the worker thread starts).  Initialized
        # here so the read-timeout builder can floor itself at the stale value and
        # never fire before the detector.  ``None`` until the detector value is
        # resolved, so the builder degrades to its plain default if it ever runs
        # first.
        self._stream_stale_timeout = None
        self.stream_attempt_lock = threading.Lock()
        self.stream_attempt_state = {
            "current": 0,
            "cancelled": set(),
            "discarded_chunks": 0,
            "discarded_bytes": 0,
        }
        self.managed_stream_holder = {"stream": None}

    def _set_managed_stream(self, stream: Any) -> Any:
        self.managed_stream_holder["stream"] = stream
        return stream

    def _close_managed_stream(self) -> None:
        stream = self.managed_stream_holder.pop("stream", None)
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Managed provider stream cleanup failed", exc_info=True)

    def _start_stream_attempt(self) -> int:
        with self.stream_attempt_lock:
            self.stream_attempt_state["current"] += 1
            attempt_id = int(self.stream_attempt_state["current"])
        self.provider_tool_in_flight["yes"] = False
        return attempt_id

    def _cancel_current_stream_attempt(self, reason: str) -> None:
        with self.stream_attempt_lock:
            current = int(self.stream_attempt_state.get("current") or 0)
            if current:
                self.stream_attempt_state["cancelled"].add(current)
        if current:
            logger.debug(
                "Marked stream attempt %s cancelled: %s",
                current,
                reason,
            )

    def _stream_attempt_is_active(self, stream_attempt_id: int) -> bool:
        with self.stream_attempt_lock:
            return (
                stream_attempt_id == int(self.stream_attempt_state.get("current") or 0)
                and stream_attempt_id not in self.stream_attempt_state["cancelled"]
            )

    def _stream_attempt_was_cancelled(self, stream_attempt_id: int) -> bool:
        with self.stream_attempt_lock:
            return stream_attempt_id in self.stream_attempt_state["cancelled"]

    def _discard_stale_stream_chunk(self, stream_attempt_id: int, chunk) -> None:
        try:
            chunk_bytes = len(repr(chunk))
        except Exception:
            chunk_bytes = 0
        with self.stream_attempt_lock:
            self.stream_attempt_state["discarded_chunks"] += 1
            self.stream_attempt_state["discarded_bytes"] += chunk_bytes
            discarded_chunks = self.stream_attempt_state["discarded_chunks"]
            discarded_bytes = self.stream_attempt_state["discarded_bytes"]
        if discarded_chunks == 1:
            logger.warning(
                "Discarding chunk from superseded stream attempt %s "
                "(discarded_chunks=%s discarded_bytes=%s)",
                stream_attempt_id,
                discarded_chunks,
                discarded_bytes,
            )
        else:
            logger.debug(
                "Discarded stale stream chunk from attempt %s "
                "(discarded_chunks=%s discarded_bytes=%s)",
                stream_attempt_id,
                discarded_chunks,
                discarded_bytes,
            )

    def _fire_first_delta(self):
        if not self.first_delta_fired["done"] and self.on_first_delta:
            self.first_delta_fired["done"] = True
            try:
                self.on_first_delta()
            except Exception:
                pass

    def _call_chat_completions(self, stream_attempt_id: int):
        """Stream a chat completions response."""
        import httpx as _httpx
        # Per-provider / per-model request_timeout_seconds (from config.yaml)
        # wins over the HERMES_API_TIMEOUT env default if the user set it.
        _provider_timeout_cfg = get_provider_request_timeout(self.agent.provider, self.agent.model)
        _base_timeout = (
            _provider_timeout_cfg
            if _provider_timeout_cfg is not None
            else env_float("HERMES_API_TIMEOUT", 1800.0)
        )
        # Read timeout: config wins; else HERMES_STREAM_READ_TIMEOUT (120s).
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = env_float("HERMES_STREAM_READ_TIMEOUT", 120.0)
            # Local providers prefill for minutes: raise the read timeout
            # unless the user overrode HERMES_STREAM_READ_TIMEOUT.
            if _stream_read_timeout == 120.0 and self.agent.base_url and is_local_endpoint(self.agent.base_url):
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    self.agent.base_url, _stream_read_timeout,
                )
            elif (
                _stream_read_timeout == 120.0
                and self._stream_stale_timeout is not None
                and self._stream_stale_timeout != float("inf")
                and self._stream_stale_timeout > _stream_read_timeout
            ):
                # Reasoning models pause mid-stream for minutes; the stale
                # detector (180–300s) tolerates that, so the raw 120s socket
                # read timeout must not fire first and preempt it.
                _stream_read_timeout = self._stream_stale_timeout
                logger.debug(
                    "Cloud reasoning stream — read timeout raised to %.0fs to "
                    "match stale-stream detector", _stream_read_timeout,
                )
        # connect/pool cover the TCP handshake, not inference: cap at 60s.
        _conn_cap = min(_base_timeout, 60.0) if _provider_timeout_cfg is not None else 30.0
        content_parts: list = []
        tool_calls = _ToolCallAccumulator()
        tool_calls_acc = tool_calls.acc
        finish_reason = None
        model_name = None
        role = "assistant"
        reasoning_parts: list = []
        usage_obj = None
        _diag = self.agent._stream_diag_init()
        self.clients.diag = _diag
        _writer_token = {"value": None}
        attempt_request_client = {"value": None}
        attempt_stream_response = {"value": None}

        def _open_stream(next_api_kwargs: dict[str, Any]):
            stream_kwargs = {
                **next_api_kwargs,
                "stream": True,
                "timeout": _httpx.Timeout(
                    connect=_conn_cap,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=_conn_cap,
                ),
            }
            # Native Gemini rejects OpenAI's usage-streaming extension.
            if not is_native_gemini_base_url(self.agent.base_url):
                stream_kwargs["stream_options"] = {"include_usage": True}
            request_client = self.clients.set_client(
                self.agent._create_request_openai_client(
                    reason="chat_completion_stream_request",
                    api_kwargs=stream_kwargs,
                )
            )
            attempt_request_client["value"] = request_client
            self.last_chunk_time["t"] = time.time()
            self.agent._touch_activity("waiting for provider response (streaming)")
            return request_client.chat.completions.create(**stream_kwargs)

        def _stream_created(raw_stream: Any) -> None:
            response = getattr(raw_stream, "response", None)
            attempt_stream_response["value"] = response
            self.agent._capture_rate_limits(response)
            self.agent._capture_credits(response)
            self.agent._stream_diag_capture_response(_diag, response)
            self.agent._check_openrouter_cache_status(response)
            _writer_token["value"] = claim_stream_writer(self.agent)

        def _accept_stream_chunk(_chunk: Any) -> bool:
            # A stale-attempt fence can win while Relay hands back a received
            # tool-call chunk: record only that a tool call was in flight (so
            # retry policy doesn't see a partial text response); the chunk is
            # still rejected below.
            try:
                choices = getattr(_chunk, "choices", None)
                delta = getattr(choices[0], "delta", None) if choices else None
                if getattr(delta, "tool_calls", None):
                    self.provider_tool_in_flight["yes"] = True
            except Exception:
                pass
            # Marker-only finish chunk (finish_reason, no writable delta)
            # always passes: the fence only stops a superseded stream writing
            # MORE text, and fending the completion signal would make the
            # drop-guard mislabel a clean end as a mid-stream drop.
            try:
                _choices = getattr(_chunk, "choices", None)
                if _choices:
                    _choice = _choices[0]
                    if getattr(_choice, "finish_reason", None):
                        _delta = getattr(_choice, "delta", None)
                        _has_write = bool(
                            getattr(_delta, "content", None)
                            or getattr(_delta, "tool_calls", None)
                            or getattr(_delta, "reasoning_content", None)
                            or getattr(_delta, "reasoning", None)
                        )
                        if not _has_write:
                            return True
            except Exception:
                pass
            if not self._stream_attempt_is_active(stream_attempt_id):
                return False
            token = _writer_token["value"]
            if token is not None and not stream_writer_is_current(self.agent, token):
                logger.warning(
                    "Streaming attempt superseded by a newer stream; stopping "
                    "consumption to preserve the single-writer invariant "
                    "(model=%s).",
                    self.api_kwargs.get("model", "unknown"),
                )
                return False
            # Stamp activity BEFORE Relay processes the chunk so the watchdog
            # can't cancel a live stream mid-interceptor.
            self.last_chunk_time["t"] = time.time()
            return True

        def _relay_final_response() -> dict[str, Any]:
            tool_calls = [tool_calls_acc[index] for index in sorted(tool_calls_acc)]
            return {
                "model": model_name,
                "choices": [
                    {
                        "message": {
                            "role": role,
                            "content": "".join(content_parts) or None,
                            "reasoning_content": "".join(reasoning_parts) or None,
                            "tool_calls": tool_calls or None,
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage_obj,
            }

        from agent import relay_llm

        stream = self._set_managed_stream(
            relay_llm.stream(
                self.api_kwargs,
                _open_stream,
                **_relay_stream_identity(self.agent, "provider"),
                finalizer=_relay_final_response,
                on_stream_created=_stream_created,
                accept_chunk=_accept_stream_chunk,
                completed_response_predicate=lambda value: hasattr(value, "choices"),
                metadata=_relay_stream_metadata(self.agent, "chat_completions"),
                defer_logical_completion=True,
            )
        )
        if self.agent.provider == "moa":
            # Hermes interrupts the managed stream; Relay retains sole
            # ownership of closing the underlying provider stream.
            self.clients.set_stream_handle(stream)
        pending_text_parts: list[str] = []

        def _flush_pending_stream_text():
            if not pending_text_parts:
                return
            pending_parts = list(pending_text_parts)
            pending_text_parts.clear()
            if not tool_calls_acc:
                for text in pending_parts:
                    self._fire_first_delta()
                    self.agent._fire_stream_delta(text)
                    self.deltas_were_sent["yes"] = True
                return
            if self.agent.stream_delta_callback:
                for text in pending_parts:
                    try:
                        self.agent.stream_delta_callback(text)
                        self.agent._record_streamed_assistant_text(text)
                    except Exception:
                        pass

        for chunk in _iter_provider_stream_chunks(
            stream,
            response=lambda: attempt_stream_response["value"],
        ):
            self.last_chunk_time["t"] = time.time()
            self.agent._touch_activity("receiving stream response")

            # Best-effort diagnostics; never interrupt the hot path.
            try:
                _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                if _diag.get("first_chunk_at") is None:
                    _diag["first_chunk_at"] = self.last_chunk_time["t"]
                # Delta-length estimate: ~3x cheaper than repr() per chunk.
                try:
                    _diag["bytes"] = int(_diag.get("bytes", 0)) + _estimate_chunk_bytes(chunk)
                except Exception:
                    pass
            except Exception:
                pass

            if self.agent._interrupt_requested:
                # A half-read SSE response stays checked out of the httpx pool,
                # and the partial response below makes the finally cache the
                # client WITH the leaked connection (one per interrupt until
                # the pool exhausts). Close on the owning thread first.
                try:
                    stream.close()
                except Exception:
                    # Still checked out: poison the slot so the finally really
                    # closes the pool instead of caching it.
                    request_client = attempt_request_client["value"]
                    if request_client is not None:
                        self.agent._abort_request_openai_client(
                            request_client,
                            reason="interrupt_stream_close_failed",
                        )
                break

            if not self._stream_attempt_is_active(stream_attempt_id):
                self._discard_stale_stream_chunk(stream_attempt_id, chunk)
                continue

            if not chunk.choices:
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model
                # Usage comes in the final chunk with empty choices
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage
                # Some providers (DeepInfra) send validation errors as
                # in-stream chunks (choices=None + error_type/error_message);
                # otherwise they'd surface as a misleading EmptyStreamError
                # and pointless retries (#65631).
                _err_type = getattr(chunk, "error_type", None)
                _err_msg = getattr(chunk, "error_message", None)
                if _err_type or _err_msg:
                    _status = _status_code_from_payload(
                        {"code": _err_type, "message": _err_msg}
                    ) or _status_code_from_value(_err_type)
                    raise ProviderStreamError(
                        status_code=_status,
                        body=_provider_error_body(
                            {
                                "code": _err_type or "provider_in_stream_error",
                                "message": str(_err_msg or chunk),
                            },
                            _status,
                        ),
                        raw_text=f"{_err_type}: {_err_msg}",
                    )
                # Nous Portal usage frames often have choices=[] plus
                # lastOne=true and no [DONE]. Treat that as a clean
                # terminal, not a mid-stream drop (#90848).
                last_one = getattr(chunk, "lastOne", None)
                if last_one is None:
                    extra = getattr(chunk, "model_extra", None)
                    if isinstance(extra, dict):
                        last_one = extra.get("lastOne")
                # Integer/string-truthy sentinels included — relabelled
                # upstreams have been seen sending 1 / "true".
                if last_one in (True, 1, "true") and finish_reason is None:
                    finish_reason = "stop"
                continue

            delta = chunk.choices[0].delta
            if hasattr(chunk, "model") and chunk.model:
                model_name = chunk.model

            # Read finish_reason/usage BEFORE any content-shape `continue`:
            # the SSE-echo guard below can swallow a merged finish chunk
            # (vLLM emitting standalone ':' tokens), falsely flagging
            # truncation (#94614).
            chunk_finish_reason = getattr(chunk.choices[0], "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                usage_obj = chunk.usage

            # Accumulate reasoning content
            reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning_text:
                # Summary-part models send one markdown block per delta with no
                # separator; re-insert it (see agent/reasoning_summaries.py).
                reasoning_text = separate_glued_reasoning_blocks(
                    reasoning_parts[-1] if reasoning_parts else "",
                    reasoning_text,
                )
                reasoning_parts.append(reasoning_text)
                self._fire_first_delta()
                self.agent._fire_reasoning_delta(reasoning_text)

            # Text content (list-of-blocks deltas flattened once); callbacks
            # fire only when no tool calls.
            delta_content = flatten_message_text(getattr(delta, "content", None), sep="")
            if delta_content:
                content_parts.append(delta_content)
                if not tool_calls_acc:
                    if pending_text_parts or _provider_stream_text_may_be_sse(delta_content):
                        pending_text_parts.append(delta_content)
                        pending_text = "".join(pending_text_parts)
                        if _provider_stream_text_may_be_sse(pending_text):
                            continue
                        _flush_pending_stream_text()
                        continue
                    self._fire_first_delta()
                    self.agent._fire_stream_delta(delta_content)
                    self.deltas_were_sent["yes"] = True
                # Tool calls suppress content streaming (no chatty "I'll use
                # the tool..." preamble), but reasoning tags inside that
                # content must still reach the display or the reasoning box
                # only appears post-response. Route it through the delta
                # callback for tag extraction; the CLI drops non-reasoning
                # text once the stream box is closed.
                elif self.agent.stream_delta_callback:
                    try:
                        self.agent.stream_delta_callback(delta_content)
                        self.agent._record_streamed_assistant_text(delta_content)
                    except Exception:
                        pass

            # Accumulate tool call deltas — notify display on first name
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                _flush_pending_stream_text()
                for tc_delta in delta_tool_calls:
                    name = tool_calls.feed(tc_delta)
                    if name is not None:
                        self._fire_first_delta()
                        self.agent._fire_tool_gen_started(name)
                        # Record the partial tool-call name so the outer
                        # stub-builder can surface a user-visible warning
                        # if streaming dies before this tool's arguments
                        # are fully delivered; otherwise a stall during
                        # tool-call JSON generation lets the stub return
                        # ``tool_calls=None`` and silently discard the action.
                        self.result["partial_tool_names"].append(name)



        self._close_managed_stream()

        if self._stream_attempt_was_cancelled(stream_attempt_id):
            raise _httpx.RemoteProtocolError(
                f"stream attempt {stream_attempt_id} was superseded"
            )

        # Some adapters accept ``stream=True`` but return a completed
        # response: switch this session to non-streaming.
        if stream.final_response is not None:
            final_response = stream.final_response
            logger.info(
                "Streaming request returned a final response object instead of "
                "an iterator; switching %s/%s to non-streaming for this session.",
                self.agent.provider or "unknown",
                self.agent.model or "unknown",
            )
            self.agent._disable_streaming = True
            choices = final_response.choices
            first_choice = (
                choices[0]
                if isinstance(choices, (list, tuple)) and choices
                else None
            )
            message = getattr(first_choice, "message", None)
            if message is not None:
                reasoning_text = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                )
                if isinstance(reasoning_text, str) and reasoning_text:
                    self._fire_first_delta()
                    self.agent._fire_reasoning_delta(reasoning_text)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    self._fire_first_delta()
                    self.agent._fire_stream_delta(content)
            return final_response

        # Build mock response matching non-streaming shape
        full_content = "".join(content_parts) or None
        mock_tool_calls = None
        has_truncated_tool_args = False
        if tool_calls_acc:
            mock_tool_calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                arguments = tc["function"]["arguments"]
                tool_name = tc["function"]["name"] or "?"
                if arguments and arguments.strip():
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        # Repair before flagging (GLM via Ollama: trailing
                        # commas, unclosed brackets, Python None); "{}" means
                        # unrepairable -> truncation handling.
                        repaired = _repair_tool_call_arguments(arguments, tool_name)
                        if repaired != "{}":
                            arguments = repaired
                        else:
                            has_truncated_tool_args = True
                elif finish_reason is None:
                    # Name arrived, zero argument bytes, no finish_reason:
                    # unflagged this becomes a "stop" turn whose empty args
                    # are coerced to "{}" and executed with no retry (#80498).
                    has_truncated_tool_args = True
                mock_tool_calls.append(SimpleNamespace(
                    id=tc["id"],
                    type=tc["type"],
                    extra_content=tc.get("extra_content"),
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=arguments,
                    ),
                ))

        # Zero-chunk guard: nothing usable = upstream error / malformed SSE,
        # not a legitimate empty completion.
        if (
            finish_reason is None
            and not content_parts
            and not reasoning_parts
            and not tool_calls_acc
        ):
            raise EmptyStreamError(
                "Provider returned an empty stream with no finish_reason "
                "(possible upstream error or malformed SSE response)."
            )

        # Partial/unparseable tool args WITH finish_reason="length" is a real
        # output-cap truncation (boost max_tokens on retry). With NO
        # finish_reason the upstream dropped/stalled mid tool-call (some
        # dedicated endpoints close cleanly after minutes of stalling); the
        # model never hit a cap, so stamping "length" would burn 3 useless
        # max_tokens retries and report a misleading truncation. Route it
        # through the partial-stream stub so the loop fails fast and honestly.
        _tool_args_dropped_no_finish = has_truncated_tool_args and finish_reason is None
        if _tool_args_dropped_no_finish:
            _dropped_names = [
                (tool_calls_acc[idx]["function"]["name"] or "?")
                for idx in sorted(tool_calls_acc)
            ]
            logger.warning(
                "Stream ended with no finish_reason while a tool call's "
                "arguments were still incomplete (tools=%s); treating as a "
                "mid-tool-call stream drop, not an output-length truncation.",
                _dropped_names,
            )
            return _build_partial_stream_stub(
                role, full_content,
                "".join(reasoning_parts) or None,
                model_name, usage_obj,
                dropped_tool_names=_dropped_names or None,
            )

        # Text-only drop: no finish_reason after text but no tool calls.
        # Without this the partial text is stamped "stop" and the model's next
        # step is lost (#32086). A usage object proves the provider finished
        # (include_usage sends a final usage-only chunk with empty choices and
        # no finish_reason, #91373), so that is not a drop.
        _text_only_dropped_no_finish = (
            finish_reason is None
            and content_parts
            and not tool_calls_acc
            and usage_obj is None
        )
        if _text_only_dropped_no_finish:
            logger.warning(
                "Stream ended with no finish_reason after delivering text "
                "with no tool calls; treating as a mid-stream drop."
            )
            return _build_partial_stream_stub(
                role, full_content,
                "".join(reasoning_parts) or None,
                model_name, usage_obj,
            )

        effective_finish_reason = finish_reason or "stop"
        if has_truncated_tool_args:
            effective_finish_reason = "length"

        provider_stream_error = _provider_stream_error_from_text(
            full_content or "",
            effective_finish_reason,
            response=getattr(stream, "response", None),
        )
        if provider_stream_error is not None:
            raise provider_stream_error
        _flush_pending_stream_text()

        full_reasoning = "".join(reasoning_parts) or None
        mock_message = SimpleNamespace(
            role=role,
            content=full_content,
            tool_calls=mock_tool_calls,
            reasoning_content=full_reasoning,
        )
        mock_choice = SimpleNamespace(
            index=0,
            message=mock_message,
            finish_reason=effective_finish_reason,
        )
        return SimpleNamespace(
            id="stream-" + str(uuid.uuid4()),
            model=model_name,
            choices=[mock_choice],
            usage=usage_obj,
        )

    def _call_anthropic(self, request_client):
        """Stream an Anthropic Messages API response.

        Fires delta callbacks but returns the native Message from
        get_final_message() so the rest of the loop is unchanged. Runs on the
        per-request ``request_client`` (registered with the abort machinery)
        so the watchdog can abort this socket without closing the shared
        client mid-flight (#67142).
        """
        has_tool_use = False
        # Eventless stream: the real SDK's get_final_message() raises
        # AssertionError (no message_start); shims may fabricate a contentless
        # Message with no stop_reason, or return None under ``python -O``.
        # All are normalized to EmptyStreamError so _call() retries.
        saw_stream_event = False

        self.last_chunk_time["t"] = time.time()
        _diag = self.agent._stream_diag_init()
        self.clients.diag = _diag
        _writer_token = {"value": None}
        _stream_context = {"manager": None, "stream": None}
        base_final_message = None

        from agent import relay_llm
        from agent.anthropic_adapter import sanitize_anthropic_kwargs

        accumulator = relay_llm.AnthropicStreamAccumulator()

        def _open_anthropic_stream(next_api_kwargs: dict[str, Any]):
            final_kwargs = dict(next_api_kwargs)
            sanitize_anthropic_kwargs(
                final_kwargs,
                log_prefix=getattr(self.agent, "log_prefix", ""),
            )
            manager = request_client.messages.stream(**final_kwargs)
            _stream_context["manager"] = manager
            return manager.__enter__()

        def _anthropic_stream_created(raw_stream: Any) -> None:
            _stream_context["stream"] = raw_stream
            # Snapshot ``stream.response`` diagnostics now so they survive a
            # stream that dies before the first event.
            try:
                self.agent._stream_diag_capture_response(
                    _diag,
                    getattr(raw_stream, "response", None),
                )
            except Exception:
                pass
            _writer_token["value"] = claim_stream_writer(self.agent)

        def _accept_anthropic_event(_event: Any) -> bool:
            token = _writer_token["value"]
            if token is None or stream_writer_is_current(self.agent, token):
                return True
            logger.warning(
                "Anthropic streaming attempt superseded by a newer stream; "
                "stopping consumption to preserve the single-writer "
                "invariant (model=%s).",
                self.api_kwargs.get("model", "unknown"),
            )
            return False

        stream = self._set_managed_stream(
            relay_llm.stream(
                self.api_kwargs,
                _open_anthropic_stream,
                **_relay_stream_identity(self.agent, "anthropic"),
                finalizer=accumulator.finalize,
                on_stream_created=_anthropic_stream_created,
                on_chunk=accumulator.observe,
                accept_chunk=_accept_anthropic_event,
                metadata=_relay_stream_metadata(self.agent, "anthropic_messages"),
                defer_logical_completion=True,
            )
        )
        try:
            for event in stream:
                saw_stream_event = True
                self.last_chunk_time["t"] = time.time()
                self.agent._touch_activity("receiving stream response")
                try:
                    _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                    if _diag.get("first_chunk_at") is None:
                        _diag["first_chunk_at"] = self.last_chunk_time["t"]
                    _diag["bytes"] = int(_diag.get("bytes", 0)) + _estimate_chunk_bytes(event)
                except Exception:
                    pass
                if self.agent._interrupt_requested:
                    break

                event_type = getattr(event, "type", None)
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        has_tool_use = True
                        tool_name = getattr(block, "name", None)
                        if tool_name:
                            self._fire_first_delta()
                            self.agent._fire_tool_gen_started(tool_name)
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text and not has_tool_use:
                                self._fire_first_delta()
                                self.agent._fire_stream_delta(text)
                                self.deltas_were_sent["yes"] = True
                        elif delta_type == "thinking_delta":
                            thinking_text = getattr(delta, "thinking", "")
                            if thinking_text:
                                self._fire_first_delta()
                                self.agent._fire_reasoning_delta(thinking_text)
            if not self.agent._interrupt_requested:
                raw_stream = _stream_context["stream"]
                if raw_stream is not None:
                    try:
                        base_final_message = raw_stream.get_final_message()
                    except AssertionError:
                        if not saw_stream_event:
                            raise EmptyStreamError(
                                "Provider returned an empty stream with no events "
                                "(possible upstream error or malformed event stream)."
                            ) from None
                        raise
        finally:
            try:
                self._close_managed_stream()
            finally:
                manager = _stream_context["manager"]
                if manager is not None:
                    manager.__exit__(None, None, None)

        if self.agent._interrupt_requested:
            return None

        def _tool_use_dropped_mid_stream(message) -> bool:
            """True when the stream died mid tool call (#80498 sibling).

            A legitimate completion always has a ``stop_reason``; a
            ``tool_use`` block with none means the SSE closed between
            ``content_block_start`` and ``message_delta`` and its ``input`` is
            a partial snapshot (usually ``{}``). Raising EmptyStreamError
            blocks the empty-args execution on every path: no streamed text
            -> bounded stream retry; text already streamed -> partial-stream
            stub / continuation.
            """
            if getattr(message, "stop_reason", None) is not None:
                return False
            for block in getattr(message, "content", None) or []:
                if getattr(block, "type", None) == "tool_use":
                    return True
            return False

        if (
            base_final_message is not None
            and not getattr(base_final_message, "content", None)
            and getattr(base_final_message, "stop_reason", None) is None
        ):
            raise EmptyStreamError(
                "Provider returned an empty stream with no stop_reason "
                "(possible upstream error or malformed event stream)."
            )
        if base_final_message is not None and not stream.output_modified:
            if _tool_use_dropped_mid_stream(base_final_message):
                raise EmptyStreamError(
                    "Stream ended with no stop_reason while a tool_use "
                    "block was still incomplete; treating as a "
                    "mid-tool-call stream drop (#80498)."
                )
            return base_final_message
        final_message = accumulator.response(base_final_message)
        if (
            not getattr(final_message, "content", None)
            and getattr(final_message, "stop_reason", None) is None
        ):
            raise EmptyStreamError(
                "Provider returned an empty stream with no stop_reason "
                "(possible upstream error or malformed event stream)."
            )
        if _tool_use_dropped_mid_stream(final_message):
            raise EmptyStreamError(
                "Stream ended with no stop_reason while a tool_use "
                "block was still incomplete; treating as a "
                "mid-tool-call stream drop (#80498)."
            )
        return final_message

    def _call(self):
        import httpx as _httpx

        _max_stream_retries = env_int("HERMES_STREAM_RETRIES", 2)

        try:
            for _stream_attempt in range(_max_stream_retries + 1):
                stream_attempt_id = self._start_stream_attempt()
                # Interrupt check before each retry: otherwise /stop closes
                # the connection and the retry opens a FRESH one, blocking up
                # to a full read timeout per attempt.
                if self.agent._interrupt_requested:
                    self._cancel_current_stream_attempt("interrupt_before_stream_retry")
                    raise InterruptedError("Agent interrupted before stream retry")
                _emit_stream_start(self.agent)
                try:
                    if self.agent.api_mode == "anthropic_messages":
                        # Per-request client (credential refresh inside) so the
                        # watchdog aborts its socket, not the shared client (#67142).
                        request_client = self.clients.set_client(
                            self.agent._create_request_anthropic_client(
                                reason="anthropic_stream_request"
                            ),
                            kind="anthropic_messages",
                        )
                        self.result["response"] = self._call_anthropic(request_client)
                    else:
                        self.result["response"] = self._call_chat_completions(stream_attempt_id)
                    _emit_stream_end(self.agent, 
                        final_text=_stream_final_text(self.result["response"]),
                        finished=True,
                        error=None,
                    )
                    return  # success
                except Exception as e:
                    _emit_stream_end(self.agent, final_text="", finished=False, error=str(e))
                    self._close_managed_stream()
                    # Our own interrupt force-close caused this error: exit
                    # with no retry/fallback/"reconnecting" (the poll loop
                    # raises InterruptedError). Fix for the cascading-interrupt
                    # hang (#6600).
                    if self._request_cancelled["value"]:
                        logger.debug(
                            "Streaming worker caught %s after request "
                            "cancellation — exiting without retry.",
                            type(e).__name__,
                        )
                        return
                    _is_timeout = isinstance(
                        e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                    )
                    _is_conn_err = isinstance(
                        e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                    )
                    _is_stream_parse_err = self.agent._is_provider_stream_parse_error(e)
                    _is_empty_stream = isinstance(e, EmptyStreamError)

                    # Stream died AFTER tokens were delivered: normally no
                    # retry (would duplicate text the user saw). Exception: a
                    # tool call in flight — aborting discards it, so retry on
                    # TRANSIENT connection errors only (a "reconnecting"
                    # marker + duplicated preamble beats a failed action). No
                    # tool has executed yet in this call, so this is safe.
                    if self.deltas_were_sent["yes"]:
                        _partial_tool_in_flight = bool(
                            self.result.get("partial_tool_names")
                        ) or self.provider_tool_in_flight["yes"]
                        _is_sse_conn_err_preview = (
                            not _is_timeout and not _is_conn_err and _is_sse_connection_error(e)
                        )
                        _is_transient = (
                            _is_timeout
                            or _is_conn_err
                            or _is_sse_conn_err_preview
                            or _is_stream_parse_err
                        )
                        _can_silent_retry = (
                            _partial_tool_in_flight
                            and _is_transient
                            and _stream_attempt < _max_stream_retries
                        )
                        if not _can_silent_retry:
                            # Either no tool call was in-flight (so the
                            # turn was a pure text response — current
                            # stub-with-recovered-text behaviour is
                            # correct), or retries are exhausted, or the
                            # error isn't transient.  Fall through to the
                            # stub path.
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            self.result["error"] = e
                            return
                        # Retry silently: "reconnecting" marker (explains the
                        # re-streamed preamble), then reset per-attempt state.
                        # ``_emit_stream_drop`` below emits the WARNING.
                        try:
                            self.agent._fire_stream_delta(
                                "\n\n⚠ Connection dropped mid tool-call; "
                                "reconnecting…\n\n"
                            )
                        except Exception:
                            pass
                        # Reset the streamed-text buffer so the retry's preamble
                        # isn't double-recorded in _current_streamed_assistant_text.
                        try:
                            self.agent._reset_stream_delivery_tracking()
                        except Exception:
                            pass
                        # Fresh accumulators: don't concat onto the dead stream's partial JSON.
                        self.result["partial_tool_names"] = []
                        self.deltas_were_sent["yes"] = False
                        self.first_delta_fired["done"] = False
                        self.agent._emit_stream_drop(
                            error=e,
                            attempt=_stream_attempt + 2,
                            max_attempts=_max_stream_retries + 1,
                            mid_tool_call=True,
                            diag=self.clients.diag,
                        )
                        self._cancel_current_stream_attempt("stream_mid_tool_retry_cleanup")
                        self.clients.close_once("stream_mid_tool_retry_cleanup")
                        # Shared clients are never closed from inside a request
                        # (#67142/#70773 FD-recycle): the request-local client
                        # was worker-closed above; the next attempt builds fresh.
                        continue

                    _is_sse_conn_err = (
                        not _is_timeout and not _is_conn_err and _is_sse_connection_error(e)
                    )

                    if (
                        _is_timeout
                        or _is_conn_err
                        or _is_sse_conn_err
                        or _is_stream_parse_err
                        or _is_empty_stream
                    ):
                        # Transient network / timeout error. Retry the
                        # streaming request with a fresh connection first.
                        if _stream_attempt < _max_stream_retries:
                            self.agent._emit_stream_drop(
                                error=e,
                                attempt=_stream_attempt + 2,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=False,
                                diag=self.clients.diag,
                            )
                            self._cancel_current_stream_attempt("stream_retry_cleanup")
                            self.clients.close_once("stream_retry_cleanup")
                            # Shared clients are never closed from inside a request
                            # (#67142/#70773); _ensure_primary_openai_client
                            # replaces the OpenAI primary lazily on the next attempt.
                            continue
                        # Retries exhausted: log with full diagnostics (chain,
                        # headers, bytes/elapsed); subagent lines carry log_prefix.
                        self.agent._log_stream_retry(
                            kind="exhausted",
                            error=e,
                            attempt=_max_stream_retries + 1,
                            max_attempts=_max_stream_retries + 1,
                            mid_tool_call=False,
                            diag=self.clients.diag,
                        )
                        if _is_stream_parse_err:
                            _exhausted_msg = (
                                "❌ Provider returned malformed streaming data after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                        elif _is_empty_stream:
                            # Stream opened but no chunks: don't say "connection
                            # failed" — that sends users chasing network issues.
                            _exhausted_msg = (
                                "❌ Provider returned an empty response stream "
                                f"after {_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                        else:
                            _exhausted_msg = (
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                        self.agent._buffer_status(_exhausted_msg)
                    else:
                        _err_lower = str(e).lower()
                        _is_stream_unsupported = (
                            "stream" in _err_lower
                            and "not supported" in _err_lower
                        )
                        # AnthropicBedrock: IAM without InvokeModelWithResponseStream
                        # rejects messages.stream() for the whole session ->
                        # flip to non-streaming (messages.create = InvokeModel).
                        _is_bedrock_stream_denied = False
                        if (
                            not _is_stream_unsupported
                            and "invokemodelwithresponsestream" in _err_lower
                        ):
                            # Message pre-check first: importing bedrock_adapter
                            # triggers a lazy boto3 install.
                            from agent.bedrock_adapter import (
                                is_streaming_access_denied_error,
                            )
                            _is_bedrock_stream_denied = (
                                is_streaming_access_denied_error(e)
                            )
                        if _is_stream_unsupported or _is_bedrock_stream_denied:
                            self.agent._disable_streaming = True
                            self.agent._safe_print(
                                "\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream. "
                                "Switching to non-streaming.\n"
                                "   Grant that action to restore streaming output.\n"
                                if _is_bedrock_stream_denied else
                                "\n⚠  Streaming is not supported for this "
                                "model/provider. Switching to non-streaming.\n"
                                "   To avoid this delay, set display.streaming: false "
                                "in config.yaml\n"
                            )
                        logger.exception(
                            "Streaming failed before delivery: %s",
                            e,
                        )

                    # Propagate to the main retry loop (credential rotation,
                    # fallback, backoff; _disable_streaming flips the next attempt).
                    self.result["error"] = e
                    return
        except InterruptedError as e:
            # A fast pre-retry interrupt noticed on the worker surfaces
            # through the normal result channel.
            self.result["error"] = e
            return
        finally:
            self._close_managed_stream()
            # Reuse reason only on a clean stream; otherwise really close so
            # the next attempt builds a fresh pool.
            self.clients.close_once(
                "stream_request_complete"
                if self.result["response"] is not None
                else "stream_error_cleanup"
            )

    def _run_call(self):
        try:
            self._call()
        finally:
            self._call_done.set()

    def _call_alive(self) -> bool:
        return not self._call_done.is_set()

    def _wait_call(self, timeout: float) -> None:
        self._call_done.wait(timeout=timeout)

    def _monitor_loop(self) -> None:
        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        # Managed local server: surface a cold model's weight-load progress
        # immediately instead of the 30s "provider may be slow" copy. Polled
        # ~1s only while no chunks have arrived; in-memory read, no network.
        _last_load_poll = 0.0
        _load_notice_shown = False
        _load_notice_misses = 0
        _is_local_base = bool(self.agent.base_url) and is_local_endpoint(self.agent.base_url)
        while self._call_alive():
            self._wait_call(0.3)

            _hb_now = time.time()
            # Cold-load window: last_chunk_time is touched only by REAL chunks,
            # so "no chunk for 2s+" holds through a model load and never during
            # healthy token flow — keeps this probe off the hot path.
            if (
                _is_local_base
                and _hb_now - self.last_chunk_time["t"] >= 2.0
                and _hb_now - _last_load_poll >= 1.0
            ):
                _last_load_poll = _hb_now
                _load_notice = _managed_local_load_notice(self.agent, self.api_kwargs)
                if _load_notice is not None:
                    self.agent._emit_wait_notice(_load_notice)
                    self.agent._touch_activity("local model loading")
                    _load_notice_shown = True
                    _load_notice_misses = 0
                    # Loading IS liveness for the heartbeat; the stale detector
                    # needs no help — the local floor (900s) dwarfs any load.
                    _last_heartbeat = _hb_now
                    continue
                if _load_notice_shown:
                    # One missed sample is routine (probe timeout under load);
                    # clearing on it strobed the status line. Require 3 misses.
                    _load_notice_misses += 1
                    if _load_notice_misses >= 3:
                        _load_notice_shown = False
                        _load_notice_misses = 0
                        self.agent._emit_wait_notice("")

            # Heartbeat for the gateway inactivity monitor: the worker touches
            # activity per chunk, but the start-to-first-chunk gap (thinking,
            # local prefill) can exceed the gateway timeout.
            if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _last_heartbeat = _hb_now
                _waiting_secs = int(_hb_now - self.last_chunk_time["t"])
                if _waiting_secs >= _HEARTBEAT_INTERVAL:
                    # No chunks for 30s+: say WHAT the wait is and WHEN recovery kicks in.
                    if (
                        self._stream_stale_timeout is not None
                        and self._stream_stale_timeout != float("inf")
                    ):
                        _recovery = f"; auto-reconnect at {int(self._stream_stale_timeout)}s"
                    else:
                        _recovery = ""
                    self.agent._emit_wait_notice(
                        f"⏳ waiting on {self.api_kwargs.get('model', 'the provider')} — "
                        f"{_waiting_secs}s with no output yet (provider may be "
                        f"slow or overloaded, or the model is thinking{_recovery})"
                    )
                else:
                    # Chunks are flowing — keep the activity tracker fresh but
                    # leave the live display alone.
                    self.agent._touch_activity(
                        f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
                    )

            # Detect stale streams: connections kept alive by SSE pings
            # but delivering no real chunks.  Kill the client so the
            # inner retry loop can start a fresh connection.
            _stale_elapsed = time.time() - self.last_chunk_time["t"]
            if _stale_elapsed > self._stream_stale_timeout:
                _est_ctx = estimate_request_context_tokens(self.api_kwargs)
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, self._stream_stale_timeout,
                    self.api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self.agent._buffer_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {self.api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    self._cancel_current_stream_attempt("stale_stream_kill")
                    self.clients.close_once("stale_stream_kill")
                except Exception:
                    pass
                # Circuit breaker (#58962): count the stale kill.  See the
                # canonical comment block above ``_stale_streak()``.
                _bump_stale_streak(self.agent)
                # The shared client (anthropic or OpenAI) must NOT be closed
                # from this poll (stranger) thread: worker threads from earlier
                # stale-killed attempts may still be unwinding SSL BIOs — the
                # FD-recycle corruption vector (#67142/#70773). The request-
                # local client was aborted above (keeps the #28161 no-hang
                # guarantee); the OpenAI primary is replaced lazily.
                # Reset the timer so we don't kill repeatedly while
                # the inner thread processes the closure.
                self.last_chunk_time["t"] = time.time()
                self.agent._emit_wait_notice(
                    f"⚠ no output from provider for {int(_stale_elapsed)}s — "
                    f"reconnecting..."
                )
                self.agent._touch_activity(
                    f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
                )

            if self.agent._interrupt_requested:
                # The stale branch above already counted this iteration when its
                # deadline won the race; do not double-count a simultaneous stop.
                if _stale_elapsed <= self._stream_stale_timeout:
                    _record_interrupted_provider_wait(
                        self.agent,
                        _stale_elapsed,
                        response_started=self.deltas_were_sent["yes"],
                    )
                # Mark THIS request cancelled before force-closing so the worker's
                # exception handler recognizes the forced transport error as a
                # cancel and exits without retrying or surfacing a network error.
                # (#6600)
                self._request_cancelled["value"] = True
                logger.debug(
                    "Force-closing streaming httpx client due to interrupt "
                    "(not a network error)."
                )
                try:
                    self._cancel_current_stream_attempt("stream_interrupt_abort")
                    # #67142: kind-aware — anthropic aborts the request-local
                    # client's socket from this poll thread; the shared
                    # _anthropic_client is never closed here.
                    self.clients.close_once("stream_interrupt_abort")
                except Exception:
                    pass
                # Let the worker unwind Relay-managed scopes before raising
                # InterruptedError; raising first lets turn teardown race a
                # still-open physical scope and corrupt the LIFO stack (#81521).
                # No-op without Relay; inline mode has no worker to wait for.
                if self.worker is not None:
                    _join_worker_for_relay_teardown(self.worker, label="Streaming")
                self._monitor_interrupted["yes"] = True
                return

    def run(self):
        """Resolve the stale timeout, run the request (worker thread or inline),
        drive the heartbeat/stale/interrupt monitor, then translate the outcome."""

        # Provider-configured stale timeout takes priority over env default.
        _cfg_stale = get_provider_stale_timeout(self.agent.provider, self.agent.model)
        if _cfg_stale is not None:
            _stream_stale_timeout_base = _cfg_stale
        else:
            _stream_stale_timeout_base = env_float("HERMES_STREAM_STALE_TIMEOUT", 180.0)
        # Local providers can prefill for 300s+, so tolerate much longer
        # silence — but finite (an infinite timeout stalled sessions on a
        # crashed endpoint forever). 900s default from config
        # ``agent.local_stream_stale_timeout``; HERMES_LOCAL_STREAM_STALE_TIMEOUT
        # overrides. Skipped when the user set HERMES_STREAM_STALE_TIMEOUT.
        if _stream_stale_timeout_base == 180.0 and self.agent.base_url and is_local_endpoint(self.agent.base_url):
            _local_default = 900.0
            try:
                from hermes_cli.config import load_config_readonly

                _cfg = load_config_readonly()  # read-only consumer — no deepcopy
                _agent_cfg = _cfg.get("agent") if isinstance(_cfg, dict) else None
                if isinstance(_agent_cfg, dict):
                    _v = _agent_cfg.get("local_stream_stale_timeout")
                    if isinstance(_v, (int, float)):
                        _local_default = float(_v)
            except Exception:
                pass
            self._stream_stale_timeout = env_float("HERMES_LOCAL_STREAM_STALE_TIMEOUT", _local_default)
            logger.debug(
                "Local provider detected (%s) — stale stream timeout set to %.0fs",
                self.agent.base_url, self._stream_stale_timeout,
            )
        else:
            # Large contexts: slow models think for minutes before the first
            # token; scale the threshold or the detector kills healthy streams.
            _est_tokens = estimate_request_context_tokens(self.api_kwargs)
            if _est_tokens > 100_000:
                self._stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                self._stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                self._stream_stale_timeout = _stream_stale_timeout_base
            # Known reasoning models exceed the 180s chat threshold while
            # thinking (surfaces as BrokenPipeError from the gateway). Floor
            # only — explicit user config (get_provider_stale_timeout) wins.
            from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
            _reasoning_floor = get_reasoning_stale_timeout_floor(self.api_kwargs.get("model"))
            if _reasoning_floor is not None:
                self._stream_stale_timeout = max(self._stream_stale_timeout, _reasoning_floor)

        # Delegated children and cron turns run the request INLINE: a worker
        # thread inside their nested pools wedges before the socket opens
        # (#62151, #60203). They must still STREAM — a silent non-streaming
        # POST is killed by edge proxies during thinking (#90202) and our
        # stale watchdog can't tell thinking from a hang (#100260). Only the
        # poll loop (heartbeat/stale/interrupt) moves to a monitor thread,
        # which never issues a request, so the no-worker deadlock fix holds.
        self._inline = should_use_direct_api_call(self.agent)
        self._call_done = threading.Event()
        self._monitor_interrupted = {"yes": False}

        if self._inline:
            self.worker = None
        else:
            self.worker = threading.Thread(target=_context_thread_target(self._run_call), daemon=True)
            self.worker.start()

        if self._inline:
            # Request on THIS thread; heartbeat / stale / interrupt monitor on a
            # side thread that only ever aborts sockets (never dispatches).
            monitor = threading.Thread(
                target=_context_thread_target(self._monitor_loop),
                name="stream-inline-monitor",
                daemon=True,
            )
            monitor.start()
            try:
                self._run_call()
            finally:
                monitor.join(timeout=2.0)
        else:
            self._monitor_loop()
        if self._monitor_interrupted["yes"]:
            raise InterruptedError("Agent interrupted during streaming API call")
        # The worker may return early on interrupt (e.g. _call_anthropic ->
        # None) before the poll loop saw the flag; re-check so /stop is not
        # swallowed (#59999 area).
        if self.agent._interrupt_requested:
            raise InterruptedError("Agent interrupted during streaming API call (post-worker)")
        if self.result["error"] is not None:
            if self.deltas_were_sent["yes"]:
                # Tokens already reached the platform: return a
                # finish_reason="length" stub so the continuation machinery
                # fires; tool_calls=None blocks executing incomplete calls.
                _partial_text = (
                    getattr(self.agent, "_current_streamed_assistant_text", "") or ""
                ).strip() or None

                # Append a user-visible warning if tool calls were dropped so
                # the user and model both know what was attempted.
                _partial_names = list(self.result.get("partial_tool_names") or [])
                if _partial_names:
                    _name_str = ", ".join(_partial_names[:3])
                    if len(_partial_names) > 3:
                        _name_str += f", +{len(_partial_names) - 3} more"
                    _warn = (
                        f"\n\n⚠ Stream stalled mid tool-call "
                        f"({_name_str}); the action was not executed. "
                        f"Ask me to retry if you want to continue."
                    )
                    _partial_text = (_partial_text or "") + _warn
                    # Fire as streaming delta so the user sees it immediately.
                    try:
                        self.agent._fire_stream_delta(_warn)
                    except Exception:
                        pass
                    logger.warning(
                        "Partial stream dropped tool call(s) %s after %s chars "
                        "of text; surfaced warning to user: %s",
                        _partial_names, len(_partial_text or ""), self.result["error"],
                    )
                    _stub_finish_reason = FINISH_REASON_LENGTH
                else:
                    logger.warning(
                        "Partial stream delivered before error; returning "
                        "length-truncated stub with %s chars of recovered "
                        "content so the loop can continue from where the "
                        "stream died: %s",
                        len(_partial_text or ""),
                        self.result["error"],
                    )
                    _stub_finish_reason = FINISH_REASON_LENGTH
                # The stub may carry EMPTY content on purpose: the loop's
                # truncation path skips appending an empty PARTIAL_STREAM_STUB_ID
                # stub and only sends the continuation nudge. Placeholder text
                # here was tried and reverted — it defeats that guard and leaks
                # into the stitched final response. Persisted empty turns are
                # healed by ``repair_empty_non_final_messages`` (single owner).
                _stub_msg = SimpleNamespace(
                    role="assistant", content=_partial_text, tool_calls=None,
                    reasoning_content=None,
                )
                # Classify output-layer content filtering (MiniMax 1027, Azure
                # content_filter, Anthropic refusal) HERE, before the raw error is
                # swallowed into the length stub: the loop reads the tag and falls
                # back instead of re-hitting a deterministic filter (#32421).
                _content_filter_terminated = False
                try:
                    from agent.error_classifier import classify_api_error, FailoverReason
                    _cls = classify_api_error(
                        self.result["error"],
                        provider=str(getattr(self.agent, "provider", "") or ""),
                        model=str(getattr(self.agent, "model", "") or ""),
                    )
                    _content_filter_terminated = (
                        _cls.reason == FailoverReason.content_policy_blocked
                    )
                except Exception:
                    _content_filter_terminated = False
                _stub = SimpleNamespace(
                    id=PARTIAL_STREAM_STUB_ID,
                    model=getattr(self.agent, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason=_stub_finish_reason,
                    )],
                    usage=None,
                    _dropped_tool_names=_partial_names or None,
                )
                if _content_filter_terminated:
                    _stub._content_filter_terminated = True
                # Deltas fired => provider responsive: clear the breaker (#58962).
                _reset_stale_streak(self.agent)
                return _stub
            raise self.result["error"]
        # Success — clear the circuit breaker (#58962): the provider proved
        # responsive.  See the canonical comment block above ``_stale_streak()``.
        if self.result["response"] is not None:
            _reset_stale_streak(self.agent)
        # Propagate first-chunk timing for the ``post_api_request`` hook.
        _diag_last = self.clients.diag
        if isinstance(_diag_last, dict) and _diag_last.get("first_chunk_at"):
            self.agent._last_api_first_chunk_at = float(_diag_last["first_chunk_at"])
        return self.result["response"]


def interruptible_streaming_api_call(agent, api_kwargs: dict, *, on_first_delta=None):
    """Streaming variant of _interruptible_api_call for real-time token delivery.

    Handles all api_modes:
    - chat_completions: stream=True on OpenAI-compatible endpoints
    - anthropic_messages: client.messages.stream() via Anthropic SDK
    - bedrock_converse: boto3 converse_stream() with delta callbacks
    - codex_responses: delegates to _run_codex_stream (already streaming)

    Fires stream_delta_callback and _stream_callback for each text token.
    Tool-call turns suppress the callback — only text-only final responses
    stream to the consumer.  Returns a SimpleNamespace that mimics the
    non-streaming response shape so the rest of the agent loop is unchanged.

    Cron turns and delegated children (should_use_direct_api_call) stay on
    this streaming path and run the request inline (see ``_StreamingCall.run``);
    only the codex branch detours through _interruptible_api_call.
    """
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")
    if agent.api_mode == "codex_responses":
        return _stream_codex_passthrough(agent, api_kwargs, on_first_delta)
    if agent.api_mode == "bedrock_converse":
        return _stream_bedrock_converse(agent, api_kwargs, on_first_delta)
    # Cross-turn stale-stream circuit breaker (#58962) — see the canonical
    # comment block above ``_stale_streak()``.  Raises past the give-up
    # threshold instead of burning another stale-timeout×retries cycle.
    _check_stale_giveup(agent)
    return _StreamingCall(agent, api_kwargs, on_first_delta).run()

# ── Provider fallback ──────────────────────────────────────────────────



__all__ = [
    "interruptible_api_call",
    "build_api_kwargs",
    "build_assistant_message",
    "try_activate_fallback",
    "handle_max_iterations",
    "cleanup_task_resources",
    "interruptible_streaming_api_call",
]
