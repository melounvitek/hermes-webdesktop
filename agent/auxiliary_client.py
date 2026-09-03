"""Shared auxiliary client router for side tasks (compression, search, vision, ...).

Resolution order for text tasks (auto mode):
  1. User's main provider + main model (used regardless of provider type —
     aggregators, direct API-key providers, native Anthropic, Codex, etc.)
  2. OpenRouter  (OPENROUTER_API_KEY)
  3. Nous Portal (~/.hermes/auth.json active provider)
  4. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
  5. Native Anthropic
  6. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
  7. None

``auxiliary.free_only: true`` restricts the step-2 OpenRouter fallback to
``:free`` SKUs; ``auxiliary.openrouter_model`` overrides the default.

Resolution order for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is one of the supported vision backends below
  2. OpenRouter
  3. Nous Portal
  4. Native Anthropic
  5. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  6. None

Codex OAuth is deliberately in neither chain (undocumented, shifting model
allow-list); it is used only as the main provider or via explicit
auxiliary.<task>.provider + auxiliary.<task>.model. Per-task overrides live
under ``auxiliary:`` in config.yaml. HTTP 402 / credit errors in call_llm()
fall through to the next provider in the chain.
"""

import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path  # noqa: F401 — used by test mocks
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, parse_qs, urlunparse

from agent.codex_headers import (
    CODEX_AUX_BASE_URL as _CODEX_AUX_BASE_URL,
    apply_required_codex_headers as _apply_required_codex_headers,
    codex_cloudflare_headers as _codex_cloudflare_headers,
    is_official_codex_base_url as _is_official_codex_base_url,
)

# `openai.OpenAI` is imported lazily (~240 ms cold); `OpenAI` below is a proxy
# so in-module calls, `auxiliary_client.OpenAI` reads and
# `patch("agent.auxiliary_client.OpenAI")` all keep working.
if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401 — type hints only

_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Lazy stand-in for ``openai.OpenAI``: forwards calls and isinstance checks, importing on first use."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


OpenAI = _OpenAIProxy()


# Availability probe mode: check_fns only need to know whether a client is
# RESOLVABLE, so inside `aux_probe_mode()` constructors return a stub instead of
# importing openai + building httpx/SSL (~0.3s on CLI startup). Resolution policy
# is unchanged; stubs are never cached (see _store_cached_client).
_aux_probe_state = threading.local()


class _AuxProbeClientStub:
    """Non-functional placeholder returned while `aux_probe_mode` is active."""

    __slots__ = ("api_key", "base_url")

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def __getattr__(self, name: str) -> Any:
        # Loud failure if a probe stub ever leaks into a runtime call path.
        raise RuntimeError(
            f"_AuxProbeClientStub used as a real client (attribute {name!r}); "
            "aux_probe_mode is for availability checks only"
        )

    def __repr__(self) -> str:
        return "<aux availability-probe client stub>"


def _aux_probe_active() -> bool:
    return bool(getattr(_aux_probe_state, "active", False))


@contextlib.contextmanager
def aux_probe_mode():
    """Resolve provider availability without constructing real SDK clients."""
    prev = getattr(_aux_probe_state, "active", False)
    _aux_probe_state.active = True
    try:
        yield
    finally:
        _aux_probe_state.active = prev

from agent.credential_pool import load_pool
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    strip_codex_context_variant_suffix as _strip_codex_ctx_variant,
)
from hermes_cli.config import get_hermes_home
from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches, base_url_hostname, env_float, is_truthy_value, model_forces_max_completion_tokens, normalize_proxy_env_vars

logger = logging.getLogger(__name__)


# resolve_provider_client fall-through dedup: misconfigured-provider warnings fire
# on every retry, so only the first per process surfaces. Separate sets let tests
# clear each branch independently.
_LOGGED_UNKNOWN_PROVIDER_KEYS: set = set()
_LOGGED_UNHANDLED_AUTHTYPE_KEYS: set = set()
_LOGGED_UNSUPPORTED_EXTPROC_KEYS: set = set()
_LOGGED_UNSUPPORTED_OAUTH_KEYS: set = set()


def _resolve_aux_verify(base_url: Optional[str]) -> Any:
    """httpx ``verify`` for an aux base_url, mirroring the main client (per-provider
    ``ssl_ca_cert`` / ``ssl_verify``, ``HERMES_CA_BUNDLE`` / ``SSL_CERT_FILE``);
    any failure falls back to the httpx/certifi default (``True``)."""
    try:
        from agent.ssl_verify import resolve_httpx_verify
        from hermes_cli.config import get_custom_provider_tls_settings, load_config_readonly

        tls = get_custom_provider_tls_settings(str(base_url or ""), config=load_config_readonly())
        return resolve_httpx_verify(
            ca_bundle=tls.get("ssl_ca_cert"),
            ssl_verify=tls.get("ssl_verify"),
            base_url=str(base_url or ""),
        )
    except Exception:
        return True


_WARNED_KEEPALIVE_IMPORT_SKEW = False


def _openai_http_client_kwargs(
    base_url: Optional[str],
    *,
    async_mode: bool = False,
) -> Dict[str, Any]:
    """Inject keepalive httpx client with env-only proxy (not macOS system proxy)."""
    try:
        from agent.process_bootstrap import build_keepalive_http_client
        client = build_keepalive_http_client(
            str(base_url or ""),
            async_mode=async_mode,
            verify=_resolve_aux_verify(base_url),
        )
    except (ImportError, AttributeError):
        # Version-skewed install (Desktop runtime lagging a git tree): older
        # process_bootstrap lacks this helper. Degrade to the SDK default httpx
        # client rather than kill the job; warn once.
        global _WARNED_KEEPALIVE_IMPORT_SKEW
        if not _WARNED_KEEPALIVE_IMPORT_SKEW:
            _WARNED_KEEPALIVE_IMPORT_SKEW = True
            logger.warning(
                "agent.process_bootstrap.build_keepalive_http_client is "
                "unavailable — mixed/stale install detected (#64333). Falling "
                "back to the SDK default HTTP client. Run `hermes update` (or "
                "reinstall the Desktop app) to resync the runtime."
            )
        client = None
    return {"http_client": client} if client is not None else {}


def _create_openai_client(*, api_key: str, base_url: str, **kwargs: Any) -> Any:
    if _aux_probe_active():
        # Availability probe: resolved credentials/base_url are the answer.
        return _AuxProbeClientStub(api_key=api_key, base_url=base_url)
    kwargs = {**_openai_http_client_kwargs(base_url), **kwargs}
    # OpenCode Zen free tier: the keyless placeholder must never hit the wire
    # (relay 401s any unrecognized bearer) — blank the Authorization header.
    try:
        from hermes_cli.models import (
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
            opencode_zen_free_headers,
        )
        if api_key == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER:
            merged = dict(kwargs.get("default_headers") or {})
            merged.update(opencode_zen_free_headers())
            kwargs["default_headers"] = merged
    except Exception:
        pass
    _apply_required_codex_headers(kwargs, access_token=api_key, base_url=base_url)
    # Hermes owns aux retry/fallback policy; the SDK default (max_retries=2) would
    # triple wall time on a hung endpoint before Hermes sees one failure.
    kwargs.setdefault("max_retries", 0)
    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)


# Interrupt protection for atomic aux tasks: a compression summary killed by an
# ordinary gateway interrupt degrades to a static marker, so a thread-local flag
# marks such calls protected. Explicit host cancel (Ctrl+C, /stop) still
# overrides it, timeouts still fire, every other aux task stays interruptible.
_aux_interrupt_protection = threading.local()


class AuxiliaryExplicitCancellation(BaseException):
    """Frozen signal that an auxiliary attempt was explicitly hard-cancelled.

    ``BaseException`` so broad ``except Exception`` retry/fallback code never treats a
    host stop as a transport failure; ``cause`` is immutable class data so nothing
    re-queries a mutable host Event after the transport has unwound.
    """

    cause = "explicit_host_cancel"

    def __init__(self) -> None:
        super().__init__("auxiliary request explicitly cancelled by host")


def _aux_interrupt_protected() -> bool:
    return bool(getattr(_aux_interrupt_protection, "active", False))


def _aux_interrupt_cancel_requested() -> bool:
    """Return whether an explicit host cancel overrides aux protection."""
    check = _capture_aux_cancel_check()
    return _captured_aux_cancel_requested(check) if check is not None else False


@contextlib.contextmanager
def aux_interrupt_protection(
    active: bool = True,
    cancel_check=None,
    cancel_event=None,
):
    """Mark this thread's aux LLM call interrupt-protected (re-entrant-safe).

    ``cancel_check`` / ``cancel_event`` keep an explicit host hard-cancel path
    (``cancel_event`` preferred when the host owns an Event); nested scopes inherit both.
    """
    prev = getattr(_aux_interrupt_protection, "active", False)
    prev_cancel_check = getattr(_aux_interrupt_protection, "cancel_check", None)
    prev_cancel_event = getattr(_aux_interrupt_protection, "cancel_event", None)
    _aux_interrupt_protection.active = active
    if callable(cancel_check):
        _aux_interrupt_protection.cancel_check = cancel_check
    if cancel_event is not None and callable(getattr(cancel_event, "is_set", None)):
        _aux_interrupt_protection.cancel_event = cancel_event
    try:
        yield
    finally:
        _aux_interrupt_protection.active = prev
        _aux_interrupt_protection.cancel_check = prev_cancel_check
        _aux_interrupt_protection.cancel_event = prev_cancel_event


def _capture_aux_cancel_check() -> Optional[Callable[[], Any]]:
    """Capture the current explicit-cancel source on the owning request thread."""
    event = getattr(_aux_interrupt_protection, "cancel_event", None)
    is_set = getattr(event, "is_set", None)
    if callable(is_set):
        return is_set
    check = getattr(_aux_interrupt_protection, "cancel_check", None)
    if callable(check):
        # Preserve callable identity so attempt-local decision objects keep
        # methods like begin_timeout_cleanup() when captured by adapters.
        return check
    return None


def _captured_aux_cancel_requested(cancel_check: Callable[[], Any]) -> bool:
    """Read a request-thread cancellation source without leaking its failures."""
    try:
        return bool(cancel_check())
    except Exception:
        logger.debug("captured aux cancel check failed", exc_info=True)
        return False


class _AuxiliaryCancellationDecision:
    """Atomically choose explicit cancellation or provider timeout per attempt."""

    def __init__(self, source_cancel_check: Callable[[], Any]) -> None:
        self._source_cancel_check = source_cancel_check
        self._lock = threading.Lock()
        self._outcome = "active"

    def __call__(self) -> bool:
        with self._lock:
            if self._outcome == "cancelled":
                return True
            if self._outcome == "timed_out":
                return False
            if _captured_aux_cancel_requested(self._source_cancel_check):
                self._outcome = "cancelled"
                return True
            return False

    def begin_timeout_cleanup(self) -> bool:
        """Return whether timeout won and destructive cleanup is permitted."""
        with self._lock:
            if self._outcome == "active":
                if _captured_aux_cancel_requested(self._source_cancel_check):
                    self._outcome = "cancelled"
                else:
                    self._outcome = "timed_out"
            return self._outcome == "timed_out"


# Forward-progress hooks for streamed aux calls: a fixed host deadline kills a
# SLOW model streaming a big summary as hard as a HUNG one, so wire consumers
# tick the progress hook only for non-empty payloads and the host extends its
# deadline while tokens move. Thread-local: the call and its stream consumption
# run on the installing thread.
_aux_progress = threading.local()
_aux_dispatch = threading.local()
_aux_provider_response = threading.local()
# Absolute monotonic deadline of the waiting HOST. The stream's own ceiling
# (_aux_stream_total_ceiling, >= the host's and started later) would otherwise
# leave an orphaned stream still billing after every host-ceiling timeout.
_aux_stream_deadline = threading.local()


def _tick_hook(local: threading.local, label: str) -> None:
    """Call the thread-local hook installed on ``local``, if any. Never raises."""
    hook = getattr(local, "hook", None)
    if hook is None:
        return
    try:
        hook()
    except Exception:
        logger.debug("aux %s hook failed", label, exc_info=True)


def _notify_aux_progress() -> None:
    """Tick the installed forward-progress hook, if any."""
    _tick_hook(_aux_progress, "progress")


def _notify_aux_dispatch() -> None:
    """Record an actual provider dispatch without claiming response progress."""
    _tick_hook(_aux_dispatch, "dispatch")


def _notify_aux_timing_response() -> None:
    """Record a content-free frame (keepalive/empty delta): counts toward
    ``time_to_first_progress_ms`` but must not reset a compression inactivity fence."""
    _tick_hook(_aux_provider_response, "provider response")


def _notify_aux_provider_response() -> None:
    """Record a provider response/chunk, then preserve the liveness signal."""
    _notify_aux_timing_response()
    _notify_aux_progress()


def _aux_progress_active() -> bool:
    return getattr(_aux_progress, "hook", None) is not None


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Field access for wire objects that may be dicts or SDK/SimpleNamespace objects."""
    val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if val is None else val


def _anthropic_event_has_content(event: Any) -> bool:
    """Whether an Anthropic stream event carries a non-empty payload."""
    event_type = _field(event, "type")
    if event_type == "content_block_delta":
        delta = _field(event, "delta")
        return any(
            bool(_field(delta, field))
            for field in ("text", "thinking", "partial_json", "signature", "citation")
        )
    if event_type == "content_block_start":
        block = _field(event, "content_block")
        return _field(block, "type") == "tool_use" and any(
            bool(_field(block, field)) for field in ("id", "name")
        )
    return False


def _anthropic_aux_stream_event_hook() -> Callable[[Any], None]:
    """Per-event callback for the Anthropic aux wire: progress only for substantive
    payloads (keepalives must not keep a stalled summary alive), stop at the host
    deadline or explicit cancel. The ``TimeoutError`` text must say "timed out"
    so ``_is_timeout_error`` classifies it."""
    host_deadline = _current_aux_stream_deadline()
    started = time.monotonic()

    def _on_event(event: Any) -> None:
        if _anthropic_event_has_content(event):
            _notify_aux_provider_response()
        else:
            _notify_aux_timing_response()
        if _aux_interrupt_cancel_requested():
            raise AuxiliaryExplicitCancellation()
        if host_deadline is not None and time.monotonic() >= host_deadline:
            raise TimeoutError(
                "Anthropic auxiliary stream timed out at the host compression "
                f"deadline after {time.monotonic() - started:.0f}s "
                "(the caller already stopped waiting)"
            )

    return _on_event


_CODEX_PROGRESS_DELTA_TYPES = frozenset({
    "response.output_text.delta",
    "response.reasoning_summary_text.delta",
    "response.text.delta",
    "response.audio.delta",
    "response.function_call_arguments.delta",
    "response.reasoning_text.delta",
})

# A dead stream fails at the no-progress window (first token AND between
# tokens); a live stream re-arms per event, bounded by _aux_stream_total_ceiling().
_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS = 60.0


def _codex_event_has_content(event: Any) -> bool:
    """Whether a Codex Responses event carries a non-empty payload."""
    event_type = _field(event, "type")
    if event_type in _CODEX_PROGRESS_DELTA_TYPES:
        return bool(_field(event, "delta"))
    if event_type == "response.output_item.added":
        item = _field(event, "item")
        return "function_call" in str(_field(item, "type") or "") and any(
            bool(_field(item, field))
            for field in ("id", "call_id", "name", "arguments")
        )
    return False


@contextlib.contextmanager
def _aux_thread_local_hook(local: threading.local, hook):
    """Install one thread-local hook, restoring the prior on exit (non-callable = passthrough)."""
    previous = getattr(local, "hook", None)
    local.hook = hook if callable(hook) else previous
    try:
        yield
    finally:
        local.hook = previous


@contextlib.contextmanager
def aux_progress_hook(hook):
    """Install *hook* as the current thread's aux forward-progress callback (None = passthrough)."""
    with _aux_thread_local_hook(_aux_progress, hook):
        yield


def _current_aux_stream_deadline() -> Optional[float]:
    """The waiting host's absolute monotonic deadline, if one is installed."""
    return getattr(_aux_stream_deadline, "value", None)


@contextlib.contextmanager
def aux_stream_deadline(deadline: Optional[float]):
    """Publish the host's absolute ``time.monotonic()`` deadline to the stream consumer.

    ``None`` is a passthrough; re-entrant-safe. This is the host->worker return leg
    of the progress hook: without it the isolated provider daemon streams to its own
    ceiling after the host stopped waiting, billing a summary the commit fence refuses.
    """
    previous = getattr(_aux_stream_deadline, "value", None)
    _aux_stream_deadline.value = deadline if isinstance(deadline, (int, float)) else previous
    try:
        yield
    finally:
        _aux_stream_deadline.value = previous


# Back-compat alias — the timing hooks were introduced with this name.
_aux_timing_hook = _aux_thread_local_hook


def _run_protected_sync_provider_call(
    callback: Callable[[dict[str, Any]], Any],
    kwargs: dict[str, Any],
) -> Any:
    """Run one protected provider callback in an attempt-isolated daemon thread.

    Aux clients are process-shared and cannot be closed to wake one request, so
    the callback (incl. stream aggregation) runs in a daemon while the owner polls
    cancellation; on cancel the owner unwinds at once and the daemon finishes under
    the provider timeout in ``kwargs`` (it owns no transcript/commit state and never
    holds the session lock). Unprotected calls, or no cancel source: direct sync path.
    """
    source_cancel_check = _capture_aux_cancel_check()
    if not _aux_interrupt_protected() or not callable(source_cancel_check):
        return callback(kwargs)
    # One linearized outcome per attempt: the host Event is reused/cleared on later
    # turns and the Codex timeout Timer may race owner polling — same lock for both.
    cancel_check = _AuxiliaryCancellationDecision(source_cancel_check)
    if cancel_check():
        raise AuxiliaryExplicitCancellation()
    # Thread-locals do not cross into the daemon: timing hooks fire from the thread
    # running the callback, and the host deadline is inert unless carried along.
    progress_hook = getattr(_aux_progress, "hook", None)
    dispatch_hook = getattr(_aux_dispatch, "hook", None)
    provider_response_hook = getattr(_aux_provider_response, "hook", None)
    host_deadline = _current_aux_stream_deadline()
    provider_context = contextvars.copy_context()
    done = threading.Event()
    outcome: dict[str, Any] = {}

    def _provider_worker() -> None:
        try:
            with (
                aux_progress_hook(progress_hook),
                _aux_thread_local_hook(_aux_dispatch, dispatch_hook),
                _aux_thread_local_hook(_aux_provider_response, provider_response_hook),
                aux_stream_deadline(host_deadline),
                aux_interrupt_protection(cancel_check=cancel_check),
            ):
                outcome["result"] = callback(kwargs)
        except BaseException as exc:
            outcome["exception"] = exc
        finally:
            done.set()

    threading.Thread(
        target=provider_context.run,
        args=(_provider_worker,),
        name="hermes-protected-aux-provider",
        daemon=True,
    ).start()

    while True:
        # Check cancel before AND after each wait so it wins when result publication
        # and the host Event land in the same polling interval.
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        if not done.wait(0.02):
            continue
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        exception = outcome.get("exception")
        if exception is not None:
            raise exception
        return outcome.get("result")


def _client_declares(client_obj: Any, flag: str) -> bool:
    """Whether ``client_obj`` (or its class) sets ``flag`` truthy.

    Capability declaration instead of isinstance so an out-of-tree provider client
    can opt out of the transport/async wrappers without being imported here
    (mirrors ``SUPPORTS_HERMES_TOOL_CALLS``). Absent attribute → False.
    """
    if client_obj is None:
        return False
    try:
        return bool(getattr(client_obj, flag, False))
    except Exception:
        return False


def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False


def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=""))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return clean, params
    return url, None


# Warn only once per process about stale OPENAI_BASE_URL.
_stale_base_url_warned = False

_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "actual-computer": "actual",
    "actualcomputer": "actual",
    "aci": "actual",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-model": "copilot",
    "github-models": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan",
    "tencent-lkeap": "tencent-tokenplan",
}


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        # Resolve to the actual main provider so named custom providers work.
        main_prov = (_read_main_provider() or "").strip().lower()
        if not main_prov or main_prov in {"auto", "main"}:
            return "custom"
        normalized = main_prov
    return _PROVIDER_ALIASES.get(normalized, normalized)


# Sentinel from _fixed_temperature_for_model(): callers strip ``temperature``
# entirely. Kimi/Moonshot manage it server-side — any value can conflict with
# gateway mode selection (thinking → 1.0, non-thinking → 0.6).
OMIT_TEMPERATURE: object = object()


def _bare_model(model: Optional[str]) -> str:
    """Lowercased model slug with any ``vendor/`` prefix stripped."""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_kimi_model(model: Optional[str]) -> bool:
    """True for any Kimi / Moonshot model that manages temperature server-side."""
    bare = _bare_model(model)
    return bare.startswith("kimi-") or bare == "kimi"


def _is_arcee_trinity_thinking(model: Optional[str]) -> bool:
    """True for Arcee Trinity Large Thinking (direct or via OpenRouter)."""
    return _bare_model(model) == "trinity-large-thinking"


# Codex OAuth hard-caps gpt-5.4/5.5/5.6 at 272K (raw API/OpenRouter expose 1.05M);
# the default 50% trigger would compact at ~136K, so raise to 85% (~231K).
_CODEX_GPT54_GPT55_COMPACTION_THRESHOLD = 0.85
# gpt-5.3-codex-spark: Codex-OAuth-only, native 128K; 70% (~90K) leaves summary headroom.
_CODEX_SPARK_COMPACTION_THRESHOLD = 0.70


def _is_codex_gpt54_or_gpt55(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for gpt-5.4/5.5/5.6 (and the Daybreak Sol alias) on the Codex OAuth route only.

    Other routes expose a larger window for the same slug and keep the user's
    threshold. Prefix-matched so ``-pro`` and dated snapshots track every 272K-capped
    family; ``-900k`` picker variants are excluded (no small window to protect).
    Name kept for the ``compression.codex_gpt55_autoraise`` config key.
    """
    bare = _codex_route_bare_model(model, provider)
    if bare is None:
        return False
    from agent.model_metadata import is_codex_context_variant
    if is_codex_context_variant(bare):
        return False
    if bare == "gpt-daybreak-blue-latest":
        return True
    return any(
        bare == fam or bare.startswith(fam + "-") or bare.startswith(fam + ".")
        for fam in ("gpt-5.4", "gpt-5.5", "gpt-5.6")
    )


def _codex_route_bare_model(model: Optional[str], provider: Optional[str]) -> Optional[str]:
    """Lowercased bare model slug when ``provider`` is the Codex OAuth route, else None."""
    if (provider or "").strip().lower() != "openai-codex":
        return None
    return _bare_model(model)


def _is_codex_spark(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for ``gpt-5.3-codex-spark`` on the Codex OAuth route (the slug exists nowhere else)."""
    return _codex_route_bare_model(model, provider) == "gpt-5.3-codex-spark"


def _fixed_temperature_for_model(
    model: Optional[str],
    base_url: Optional[str] = None,
) -> "Optional[float] | object":
    """``OMIT_TEMPERATURE`` (drop the key; Kimi/Moonshot), a fixed ``float``, or ``None``."""
    if _is_kimi_model(model):
        logger.debug("Omitting temperature for Kimi model %r (server-managed)", model)
        return OMIT_TEMPERATURE
    if _is_arcee_trinity_thinking(model):
        return 0.5
    return None


def _compression_threshold_for_model(
    model: Optional[str],
    provider: Optional[str] = None,
    *,
    allow_codex_gpt55_autoraise: bool = True,
) -> Optional[float]:
    """Per-model/route compression threshold override (fraction of context used), or None.

    Arcee Trinity Large Thinking → 0.75 (preserve reasoning context); Codex-route
    gpt-5.4/5.5/5.6 → 0.85, gated by ``allow_codex_gpt55_autoraise``; Codex-route
    gpt-5.3-codex-spark → 0.70, ungated (unambiguously correct for a 128K window).
    """
    if _is_arcee_trinity_thinking(model):
        return 0.75
    if allow_codex_gpt55_autoraise and _is_codex_gpt54_or_gpt55(model, provider):
        return _CODEX_GPT54_GPT55_COMPACTION_THRESHOLD
    if _is_codex_spark(model, provider):
        return _CODEX_SPARK_COMPACTION_THRESHOLD
    return None


# Aux "fast tier" families, fastest first (measured p50 titling latency). Matched
# as substrings against the LIVE /v1/models catalog because pinned ids rot;
# rolling "-latest" aliases lead as the only structurally rot-proof ids.
_FAST_MODEL_FAMILIES: tuple = (
    "gpt-mini-latest",
    "gpt-nano-latest",
    "claude-haiku-latest",
    "gemini-flash-latest",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "haiku-4.5",
    "gemini-3.6-flash",
    "flash-lite",
    "-nano",
    "-mini",
    "-flash",
    "haiku",
)

# Disqualifiers: reasoning variants think before answering; ":batch" is a queue;
# ":free" tiers are rate-limited and slowest; embedders/modality endpoints
# ("all-minilm", "gpt-4o-mini-tts") match a family rung but cannot answer a prompt.
_FAST_MODEL_EXCLUDE: tuple = (
    "thinking", "reason", "-r1", "minilm", ":batch", ":free",
    "o1-", "o3-", "o4-", "codex", "audio", "-vl", "embed",
    "-tts", "-transcribe", "-realtime", "-image", "-search-preview",
)


_VERSION_CHUNK_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _model_recency_key(model_id: str) -> tuple:
    """Sort key putting a family's newest release first: digit runs compare numerically
    (plain string order picks ``gpt-3.5-mini`` over ``gpt-5.4-mini`` and breaks at 9 vs 10)."""
    chunks = []
    for index, part in enumerate(_VERSION_CHUNK_RE.split(model_id.lower())):
        if not part:
            continue
        # re.split with one capturing group alternates text, number, text, …
        chunks.append((1, float(part), "") if index % 2 else (0, 0.0, part))
    return tuple(chunks)


def _fast_model_from_catalog(provider_id: str) -> str:
    """Newest ``_FAST_MODEL_FAMILIES`` match from the provider's live (cached) catalog.

    "" when the catalog is unavailable or holds no small model (caller falls through
    to the curated default). Never raises; the fetch is memory+disk cached.
    """
    is_nous = provider_id.strip().lower() == "nous"
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials
        from hermes_cli.models import fetch_models_with_pricing
        from providers import get_provider_profile

        # Most /v1/models endpoints are authenticated; an anonymous 401 would read
        # as "no small model" and pin the curated default forever.
        api_key, base_url = "", ""
        try:
            creds = resolve_api_key_provider_credentials(provider_id) or {}
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
        except Exception:
            # Not an API-key provider, or nothing configured; anonymous fetch may still work.
            logger.debug("No credentials for %s catalog", provider_id, exc_info=True)

        if not api_key and is_nous:
            # Nous is OAuth (resolver raises); anonymous reads return the full catalog.
            try:
                from hermes_cli.models import _resolve_nous_pricing_credentials

                api_key, base_url = _resolve_nous_pricing_credentials()
            except Exception:
                logger.debug("No Nous credentials for catalog", exc_info=True)

        if not base_url:
            base_url = str(getattr(get_provider_profile(provider_id), "base_url", "") or "")
        base_url = base_url.rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/v1"):  # fetch_models_with_pricing appends /v1/models
            base_url = base_url[:-3]
        # Nous-only args must match the pickers' or the seeded cache loses sale
        # chrome and policy-catalog expiry.
        _nous_kwargs = {}
        if is_nous:
            from hermes_cli.models import _NOUS_CATALOG_TTL_SECONDS

            _nous_kwargs = {
                "include_sale_original": True,
                "cache_ttl_seconds": _NOUS_CATALOG_TTL_SECONDS,
            }
        catalog = fetch_models_with_pricing(
            api_key=api_key or None, base_url=base_url, timeout=3.0, **_nous_kwargs
        ) or {}
    except Exception:
        logger.debug("Fast-model catalog lookup failed for %s", provider_id, exc_info=True)
        return ""

    ids = sorted((str(m) for m in catalog), key=_model_recency_key, reverse=True)
    if is_nous:
        # Narrow catalog ids by org policy, as the pickers do.
        try:
            from hermes_cli.models import (
                nous_policy_allowed_ids,
                restrict_to_nous_policy,
            )

            ids = restrict_to_nous_policy(ids, nous_policy_allowed_ids())
        except Exception:
            logger.debug("Nous policy filter unavailable", exc_info=True)
    for family in _FAST_MODEL_FAMILIES:
        for model_id in ids:
            lowered = model_id.lower()
            if family in lowered and not any(x in lowered for x in _FAST_MODEL_EXCLUDE):
                return model_id
    return ""


def _nous_policy_blocks(model_id: str) -> bool:
    """True when the org's model policy does not admit *model_id*."""
    try:
        from hermes_cli.models import nous_policy_allowed_ids, restrict_to_nous_policy

        allowed = nous_policy_allowed_ids()
        return bool(allowed) and not restrict_to_nous_policy([model_id], allowed)
    except Exception:
        logger.debug("Nous policy check unavailable", exc_info=True)
        return False


# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
def _get_aux_model_for_provider(provider_id: str, *, prefer_fast: bool = False) -> str:
    """Cheap auxiliary model for a provider.

    Ladder: (``prefer_fast`` only) live-catalog family match, then
    ``ProviderProfile.resolve_aux_model``; then ``default_aux_model`` (curated);
    then the legacy dict. ``prefer_fast`` is opt-in (titling) so other callers
    keep their static behaviour and cache keys.
    """
    profile = None
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider_id)
    except Exception:
        pass

    picked = ""
    if prefer_fast:
        picked = _fast_model_from_catalog(provider_id)
        if not picked and profile is not None:
            try:
                picked = profile.resolve_aux_model() or ""
            except Exception:
                logger.debug("resolve_aux_model failed for %s", provider_id, exc_info=True)

    if not picked and profile is not None and profile.default_aux_model:
        picked = profile.default_aux_model
    if not picked:
        picked = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, "")

    # Rungs 2-4 are policy-blind; a blocked pick is refused at request time, so
    # drop it and let the caller keep the main model.
    if picked and provider_id.strip().lower() == "nous" and _nous_policy_blocks(picked):
        return ""
    return picked


# Fallback for providers without ProviderProfile.default_aux_model (plus some
# pinned here). New providers should set default_aux_model instead.
_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: Dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "stepfun": "step-3.5-flash",
    "kimi-coding-cn": "kimi-k2-turbo-preview",
    "gmi": "google/gemini-3.1-flash-lite-preview",
    "anthropic": "claude-haiku-4-5-20251001",
    "ai-gateway": "google/gemini-3-flash",
    "opencode-zen": "gemini-3-flash",
    "opencode-go": "glm-5",
    "kilocode": "google/gemini-3.6-flash",
    "ollama-cloud": "nemotron-3-nano:30b",
    "tencent-tokenhub": "hy4-preview",
    "tencent-tokenplan": "hy4-preview",
    # No "deepinfra": its aux model lives on the ProviderProfile (read first).
}

# Legacy alias for callers not yet using _get_aux_model_for_provider().
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK

# Tasks that may opt into ``auxiliary.<task>.prefer_fast_model``.
_FAST_MODEL_TASKS: frozenset = frozenset({"title_generation"})


def _task_prefers_fast_model(task: Optional[str]) -> bool:
    """Return whether an eligible task explicitly opts into fast-model routing."""
    if task not in _FAST_MODEL_TASKS:
        return False
    task_config = _get_auxiliary_task_config(task)
    return is_truthy_value(task_config.get("prefer_fast_model"), default=False)


# Dedicated vision models for direct providers whose main chat model differs.
_PROVIDER_VISION_MODELS: Dict[str, str] = {
    "xiaomi": "mimo-v2.5",
    "zai": "glm-5v-turbo",
}


def _resolve_provider_vision_default(provider: str) -> Optional[str]:
    """Provider default vision model id, or None: static ``_PROVIDER_VISION_MODELS``
    (vision-only names absent from any catalog) win, else the
    ``ProviderProfile.default_vision_model()`` hook resolves a live default."""
    static = _PROVIDER_VISION_MODELS.get(provider)
    if static:
        return static
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        return profile.default_vision_model() if profile is not None else None
    except Exception:
        return None


# Endpoints that reject image input: vision auto-detect skips these to the
# aggregator chain instead of returning a client that 404s (the Kimi Coding Plan
# Anthropic wire has no image_in; vision lives on api.moonshot.ai).
_PROVIDERS_WITHOUT_VISION: frozenset = frozenset({"kimi-coding", "kimi-coding-cn"})

# OpenRouter app attribution (always sent). `X-Title` is what the dashboard reads.
_OR_HEADERS_BASE = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}

# Truthy values for boolean env-var parsing.
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _apply_user_default_headers(headers: dict | None) -> dict | None:
    """Merge user ``model.default_headers`` onto resolved headers (user wins).

    Mirrors ``AIAgent._apply_user_default_headers`` so a custom endpoint behind a
    WAF that rejects the SDK's ``User-Agent`` / ``X-Stainless-*`` works for aux
    calls too. ``model.extra_headers`` is an alias that wins over default_headers.
    SECURITY: values may carry credentials — never log them.
    """
    try:
        from hermes_cli.config import cfg_get, load_config
        _cfg = load_config()
        user_headers = cfg_get(_cfg, "model", "default_headers")
        alias_headers = cfg_get(_cfg, "model", "extra_headers")
        if isinstance(alias_headers, dict) and alias_headers:
            merged_user: dict = {}
            if isinstance(user_headers, dict):
                merged_user.update(user_headers)
            merged_user.update(alias_headers)
            user_headers = merged_user
    except Exception:
        return headers
    if not isinstance(user_headers, dict) or not user_headers:
        return headers
    merged = dict(headers or {})
    merged.update({str(k): str(v) for k, v in user_headers.items() if v is not None})
    return merged or headers


def build_or_headers(or_config: dict | None = None) -> dict:
    """OpenRouter headers, plus response-cache headers when enabled.

    Precedence env > config > default: ``HERMES_OPENROUTER_CACHE`` overrides
    ``openrouter.response_cache``; ``HERMES_OPENROUTER_CACHE_TTL`` (1-86400 s)
    overrides ``openrouter.response_cache_ttl``. ``or_config=None`` reads from disk.
    """
    headers = dict(_OR_HEADERS_BASE)
    if or_config is None:
        try:
            from hermes_cli.config import load_config_readonly
            or_config = load_config_readonly().get("openrouter", {})
        except Exception:
            or_config = {}
    env_cache = os.environ.get("HERMES_OPENROUTER_CACHE", "").strip().lower()
    cache_enabled = env_cache in _TRUTHY_ENV_VALUES if env_cache else or_config.get("response_cache", False)
    if not cache_enabled:
        return headers
    headers["X-OpenRouter-Cache"] = "true"
    env_ttl = os.environ.get("HERMES_OPENROUTER_CACHE_TTL", "").strip()
    if env_ttl:
        if env_ttl.isdigit():
            ttl = int(env_ttl)
            if 1 <= ttl <= 86400:
                headers["X-OpenRouter-Cache-TTL"] = str(ttl)
    else:
        ttl = or_config.get("response_cache_ttl", 300)
        if isinstance(ttl, (int, float)) and 1 <= ttl <= 86400:
            headers["X-OpenRouter-Cache-TTL"] = str(int(ttl))
    return headers


# NVIDIA NIM cloud billing attribution; host-gated because NVIDIA_BASE_URL may
# point at a local/on-prem NIM.
_NVIDIA_NIM_CLOUD_HEADERS = {"X-BILLING-INVOKE-ORIGIN": "HermesAgent"}


def build_nvidia_nim_headers(base_url: str | None) -> dict:
    """Return NVIDIA NIM cloud attribution headers for build.nvidia.com traffic."""
    if base_url_host_matches(str(base_url or ""), "integrate.api.nvidia.com"):
        return dict(_NVIDIA_NIM_CLOUD_HEADERS)
    return {}


# Vercel AI Gateway attribution (HTTP-Referer → referrerUrl, X-Title → appName).
from hermes_cli import __version__ as _HERMES_VERSION

_AI_GATEWAY_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}

# Nous Portal attribution extra_body. Tags come from agent.portal_tags so the
# client= marker tracks hermes_cli.__version__ — never inline a literal here.
from agent.portal_tags import nous_portal_tags as _nous_portal_tags


def _nous_extra_body() -> dict:
    """Fresh Nous Portal ``extra_body`` (per call, so a hot-reloaded version is reflected)."""
    return {"tags": _nous_portal_tags()}


# Back-compat snapshot; tests/plugins read ``NOUS_EXTRA_BODY`` directly.
NOUS_EXTRA_BODY = _nous_extra_body()

# Set at resolve time — True if the auxiliary client points to Nous Portal
auxiliary_is_nous: bool = False

# _OPENROUTER_MODEL MUST stay a :free SKU (matching the free_only warning): this
# lane engages silently, and a paid default meant spend the user never opted
# into. User-configured values are honored untouched (_warn_paid_lane_once fires).
_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
_NOUS_MODEL = "google/gemini-3.6-flash"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"

# Hosts exposing BOTH ``…/anthropic`` and a sibling OpenAI ``…/v1``. Matched on
# the URL *host* only: unconditional rewrites break Anthropic-only gateways.
_DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES = ("minimax.io", "minimax.chat", "minimaxi.com")
_DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES = ("api.minimax.",)


def _is_dual_surface_anthropic_host(url: str) -> bool:
    """True when the URL's host is a known dual-surface (MiniMax-family) host."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in _DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES
    ) or any(host.startswith(prefix) for prefix in _DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES)


def _to_openai_base_url(base_url: str) -> str:
    """Normalize dual-surface Anthropic URLs to their OpenAI-compatible sibling.

    MiniMax-family: ``/anthropic`` → ``/v1``; ZAI Coding Plan → ``/coding/paas/v4``
    (the general endpoint bills separately); Kimi Code ``/coding`` → ``/coding/v1``
    (the OpenAI SDK path 404s without it). Anthropic-only gateways keep their path.
    """
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/anthropic"):
        if base_url_host_matches(url, "open.bigmodel.cn") or base_url_host_matches(url, "api.z.ai"):
            rewritten = url[: -len("/anthropic")] + "/coding/paas/v4"
            logger.debug("Auxiliary client: rewrote ZAI base URL %s → %s", url, rewritten)
            return rewritten
        if _is_dual_surface_anthropic_host(url):
            rewritten = url[: -len("/anthropic")] + "/v1"
            logger.debug("Auxiliary client: rewrote dual-surface base URL %s → %s", url, rewritten)
            return rewritten
        # Anthropic-only gateway: leave the /anthropic path alone.
        logger.debug(
            "Auxiliary client: keeping Anthropic-only base URL %s (no dual-surface host match)",
            url,
        )
        return url
    if base_url_host_matches(url, "api.kimi.com") and url.endswith("/coding"):
        rewritten = url + "/v1"
        logger.debug("Auxiliary client: rewrote Kimi base URL %s → %s", url, rewritten)
        return rewritten
    return url


def _load_pool_with_credentials(provider: str, note: str = "") -> Optional[Any]:
    """``load_pool(provider)`` when it has credentials, else None (never raises)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s%s: %s", provider, note, exc)
        return None
    return pool if pool and pool.has_credentials() else None


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    pool = _load_pool_with_credentials(provider)
    if pool is None:
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _peek_pool_entry(provider: str) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order."""
    pool = _load_pool_with_credentials(provider, " (peek)")
    if pool is None:
        return None
    try:
        current_fn = getattr(pool, "current", None)
        if callable(current_fn):
            current = current_fn()
            if current is not None:
                return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    # runtime_api_key handles provider-specific fallback (e.g. agent_key for nous).
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    if getattr(entry, "provider", None) == "nous":
        # Canonical auth-layer reader so the env override shares one normalization path.
        from hermes_cli.auth import _nous_inference_env_override

        env_url = _nous_inference_env_override()
        if env_url:
            return env_url
    # runtime_base_url is provider-aware; fall back for non-PooledCredential entries.
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


# Hosts the aux Anthropic path may be pointed at via model.base_url; anything
# else falls back to the Anthropic default so a foreign host never leaks in.
_ANTHROPIC_COMPATIBLE_HOSTS = frozenset({"api.anthropic.com"})


def _is_anthropic_compatible_host(url: str) -> bool:
    """True for native Anthropic hosts and gateways serving Messages under a
    ``/anthropic`` path (same convention as runtime_provider / ``_wrap_if_needed``),
    so a configured ``model.base_url`` whose gateway holds auth is not discarded.
    A bare non-Anthropic base_url is False."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if host in _ANTHROPIC_COMPATIBLE_HOSTS:
            return True
        path = (parsed.path or "").rstrip("/").lower()
        return path.endswith("/anthropic") or path.endswith("/anthropic/v1")
    except Exception:
        return False


def _nous_min_key_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800")))
    except (TypeError, ValueError):
        return 1800


def _scoped_key_env(name: str) -> str:
    """Read a provider API key env var through the profile secret scope.

    In agent turns the scope's verdict is authoritative (a scoped miss must not
    borrow another profile's key); unscoped startup/CLI paths fall back to os.environ.
    """
    if not name:
        return ""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return (get_secret(name) or "").strip()
        except UnscopedSecretError:
            pass
    except Exception:
        pass
    return (os.getenv(name) or "").strip()


# Codex Responses → chat.completions adapter, so aux consumers need no changes.
def _parse_codex_final_response(final: Any) -> Tuple[List[str], List[Any], Any]:
    """Split a completed Responses object into (text_parts, tool_calls, usage) in chat.completions shape."""
    text_parts: List[str] = []
    tool_calls_raw: List[Any] = []
    for item in (getattr(final, "output", None) or []):
        item_type = _field(item, "type")
        if item_type == "message":
            for part in (_field(item, "content") or []):
                if _field(part, "type") in {"output_text", "text"}:
                    text_parts.append(_field(part, "text", ""))
        elif item_type == "function_call":
            tool_calls_raw.append(SimpleNamespace(
                id=_field(item, "call_id", ""),
                type="function",
                function=SimpleNamespace(
                    name=_field(item, "name", ""),
                    arguments=_field(item, "arguments", "{}"),
                ),
            ))
    usage = None
    resp_usage = getattr(final, "usage", None)
    if resp_usage:
        def _u(key: str) -> int:
            return getattr(resp_usage, key, 0) or (
                resp_usage.get(key, 0) if isinstance(resp_usage, dict) else 0
            )
        usage = SimpleNamespace(
            prompt_tokens=_u("input_tokens"),
            completion_tokens=_u("output_tokens"),
            total_tokens=_u("total_tokens"),
        )
    return text_parts, tool_calls_raw, usage


class _CodexCompletionsAdapter:
    """Drop-in shim routing chat.completions.create() kwargs through Codex Responses streaming."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def _build_responses_kwargs(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Any]:
        """Translate chat.completions kwargs into Responses API kwargs; returns ``(resp_kwargs, model, timeout)``."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)

        # Split system/instructions from replayable messages, then use the
        # SINGLE shared chat->Responses converter (agent/transports/codex.py).
        # A private loop here let role="tool" leak into Responses input[],
        # which the API rejects; the shared converter encodes tool history
        # as function_call/function_call_output so all paths stay identical.
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        from utils import base_url_host_matches

        instructions = "You are a helpful assistant."
        replay_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                replay_messages.append(msg)

        # Copilot binds replayed codex_message_items ids to a backend connection
        # that doesn't survive credential rotation (HTTP 401 on replay); this
        # adapter bypasses build_kwargs so it needs the same guard.
        _host_for_input = str(getattr(self._client, "base_url", "") or "")
        _is_github_for_input = base_url_host_matches(_host_for_input, "githubcopilot.com")
        # Aux calls never send ``context_management`` (native compaction is a
        # main-turn feature), so never replay or emit a compaction checkpoint.
        input_items = _chat_messages_to_responses_input(
            replay_messages,
            is_github_responses=_is_github_for_input,
            native_compaction_eligible=False,
        )

        resp_kwargs: Dict[str, Any] = {
            # Codex only knows the base slug; strip the Hermes ``-900k`` picker suffix.
            "model": _strip_codex_ctx_variant(model),
            "instructions": instructions,
            "input": input_items or [{"role": "user", "content": ""}],
            "store": False,
        }

        # Forward the chat.completions timeout; otherwise a Codex stream can
        # sit behind a dead-looking CLI until the user force-interrupts.
        timeout = kwargs.get("timeout")
        if timeout is not None:
            resp_kwargs["timeout"] = timeout

        # The Codex endpoint rejects max_output_tokens/temperature (400) — omit.

        # Translate extra_body.reasoning into Responses top-level reasoning +
        # include, mirroring agent/transports/codex.py::build_kwargs().
        extra_body = kwargs.get("extra_body") or {}
        if isinstance(extra_body, dict):
            # service_tier (fast mode) is a top-level Responses field; xAI's
            # Responses endpoint rejects it — same xAI-only guard as main transport.
            service_tier = extra_body.get("service_tier")
            client_base_url = str(getattr(self._client, "base_url", "") or "")
            is_xai_responses = (
                base_url_host_matches(client_base_url, "x.ai")
                or base_url_host_matches(client_base_url, "api.x.ai")
            )
            if (
                isinstance(service_tier, str)
                and service_tier.strip()
                and not is_xai_responses
            ):
                resp_kwargs["service_tier"] = service_tier.strip()

            reasoning_cfg = extra_body.get("reasoning")
            if isinstance(reasoning_cfg, dict):
                if reasoning_cfg.get("enabled") is False:
                    # Explicitly disabled — leave reasoning/include unset. Codex
                    # still thinks by default; we honor intent where the API allows.
                    pass
                else:
                    # Truthy-only (mirrors build_kwargs): Codex 400s on
                    # e.g. {"effort": null}, so falsy falls back to default.
                    effort = reasoning_cfg.get("effort") or "medium"
                    # Shared per-model clamp with the main Codex transport
                    # ("max" is gpt-5.6-only; "minimal"/"ultra" always rejected).
                    from agent.reasoning_effort import (
                        clamp_effort,
                        codex_supported_efforts,
                    )

                    effort = clamp_effort(effort, codex_supported_efforts(model))
                    resp_kwargs["reasoning"] = {
                        "effort": effort,
                        "summary": "auto",
                    }
                    resp_kwargs["include"] = ["reasoning.encrypted_content"]

        # Tools for auxiliary callers (e.g. skills_hub) that pass function schemas
        tools = kwargs.get("tools")
        if tools:
            # xAI Responses rejects ``pattern``/``format`` JSON Schema keywords
            # (400); strip to match chat_completion_helpers.py parity. Deep-copy
            # first — sanitizers mutate inner dicts in place and would
            # permanently strip the caller's tool registry.
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools = _copy.deepcopy(list(tools))
                tools, _ = strip_pattern_and_format(tools)
                tools, _ = strip_slash_enum(tools)
            except Exception as exc:
                logger.warning(
                    "Auxiliary client: failed to sanitize tool schemas for "
                    "Codex/xAI Responses path: %s", exc,
                )
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                converted.append({
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            if converted:
                resp_kwargs["tools"] = converted

        # Stable prompt-cache routing, mirroring agent/transports/codex.py::build_kwargs;
        # without it aux Responses calls (MoA aggregator etc.) stay cache-cold while the
        # main transport is warm. Key is content-addressed from the static prefix
        # (instructions + tool schemas) so it survives across turns. Skip the top-level
        # field where the main transport does: xAI takes it in extra_body, GitHub/Copilot opts out.
        try:
            from agent.transports.codex import (
                _cache_scope_from_session_id,
                _content_cache_key,
                _default_prompt_cache_retention_for_request,
            )
            from utils import base_url_host_matches

            _host_src = str(getattr(self._client, "base_url", "") or "")
            _is_xai = base_url_host_matches(_host_src, "x.ai") or base_url_host_matches(_host_src, "api.x.ai")
            _is_github = (
                base_url_host_matches(_host_src, "githubcopilot.com")
                or base_url_host_matches(_host_src, "models.github.ai")
            )
            if not _is_xai and not _is_github and "prompt_cache_key" not in resp_kwargs:
                # Scope by the owning conversation so unrelated sessions with the
                # same instructions/tools don't share a cache slot; prefer the
                # rotation-stable logical scope, fall back to the physical session id.
                _scope = _cache_scope_from_session_id(
                    _runtime_main_value("cache_scope")
                    or _runtime_main_value("session_id")
                )
                _cache_key = _content_cache_key(instructions, resp_kwargs.get("tools"), _scope)
                if _cache_key:
                    resp_kwargs["prompt_cache_key"] = _cache_key
            if "prompt_cache_retention" not in resp_kwargs:
                _cache_retention = _default_prompt_cache_retention_for_request(
                    model,
                    _host_src,
                )
                if _cache_retention:
                    resp_kwargs["prompt_cache_retention"] = _cache_retention
        except Exception:
            logger.debug(
                "Codex auxiliary: prompt_cache_key derivation skipped", exc_info=True
            )
        return resp_kwargs, model, timeout

    def create(self, **kwargs) -> Any:
        resp_kwargs, model, timeout = self._build_responses_kwargs(kwargs)

        # Stream and collect the response
        total_timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        # Progress-aware deadlines, three regimes: (1) first substantive payload must
        # arrive within ``no_progress_timeout`` or we fail fast into the caller's
        # retry/fallback chain — a dead or keepalive-only zombie stream must not hold
        # the whole compression budget; (2) each substantive event re-arms that window
        # (keepalive/lifecycle frames do NOT, mirroring commit-fence gating), so a live
        # stream producing tokens is never killed by an absolute total; (3) a hard
        # ceiling from ``_aux_stream_total_ceiling`` still terminates a pathological drip.
        _start_monotonic = time.monotonic()
        no_progress_timeout = _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
        if total_timeout is not None:
            no_progress_timeout = min(no_progress_timeout, float(total_timeout))
        hard_deadline = _start_monotonic + _aux_stream_total_ceiling(total_timeout)
        # The waiting host's absolute deadline (aux_stream_deadline) clamps the hard
        # ceiling so the watchdog Timer severs the socket the instant the host stops
        # waiting — a stream blocked between events can't be stopped by a per-event check.
        _host_deadline = _current_aux_stream_deadline()
        if isinstance(_host_deadline, (int, float)) and _host_deadline < hard_deadline:
            hard_deadline = float(_host_deadline)
        deadline_lock = threading.Lock()
        progress_deadline = [_start_monotonic + no_progress_timeout]
        saw_content = threading.Event()
        timed_out = threading.Event()
        # Set only when the timeout WON (not when the owner hard-cancelled first):
        # tells the owner's ``finally`` the shared client's FDs still need a real close.
        timeout_release_pending = threading.Event()
        stream_finished = threading.Event()
        timeout_timer: List[Optional[threading.Timer]] = [None]
        # A protected provider call may outlive its owning compression attempt
        # (owner returns on hard cancel while this adapter is still blocked in the
        # SDK stream). Timer threads don't inherit this worker's thread-local
        # protection state, so freeze the hard-cancel source before creating the timer.
        protected_cancel_check = (
            _capture_aux_cancel_check() if _aux_interrupt_protected() else None
        )
        attempt_stream_lock = threading.Lock()
        attempt_stream: List[Any] = []
        # The request-driving thread owns the transport FDs — see _close_client_on_timeout.
        owner_tid = threading.get_ident()

        def _effective_deadline() -> float:
            with deadline_lock:
                return min(hard_deadline, progress_deadline[0])

        def _close_shared_client(failure_note: str) -> None:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Codex auxiliary: %s", failure_note, exc_info=True)

        def _close_attempt_stream(failure_note: str) -> None:
            # Closes only this attempt's stream — never the process-shared client.
            with attempt_stream_lock:
                stream = attempt_stream[0] if attempt_stream else None
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    logger.debug("Codex auxiliary: %s", failure_note, exc_info=True)

        def _record_stream_progress() -> None:
            # Substantive payload re-arms the no-progress window; hard ceiling never moves.
            with deadline_lock:
                progress_deadline[0] = time.monotonic() + no_progress_timeout

        def _timeout_message() -> str:
            elapsed = time.monotonic() - _start_monotonic
            if time.monotonic() >= hard_deadline:
                return (
                    "Codex auxiliary Responses stream exceeded "
                    f"{hard_deadline - _start_monotonic:.1f}s hard ceiling"
                )
            if not saw_content.is_set():
                return (
                    "Codex auxiliary Responses stream produced no output "
                    f"within {float(no_progress_timeout):.1f}s "
                    f"(no-progress timeout, {elapsed:.1f}s elapsed)"
                )
            return (
                "Codex auxiliary Responses stream stalled: no new output "
                f"for {float(no_progress_timeout):.1f}s "
                f"({elapsed:.1f}s elapsed)"
            )

        def _close_client_on_timeout() -> None:
            begin_timeout_cleanup = getattr(
                protected_cancel_check, "begin_timeout_cleanup", None
            )
            if callable(begin_timeout_cleanup):
                timeout_won = bool(begin_timeout_cleanup())
            else:
                timeout_won = not (
                    callable(protected_cancel_check)
                    and _captured_aux_cancel_requested(protected_cancel_check)
                )
            # Publish transport timeout only after the attempt-local decision is
            # fixed, so owner polling cannot observe completion in between.
            timed_out.set()
            if not timeout_won:
                # Owner already hard-cancelled. The OpenAI client is process-shared,
                # so never close/evict it here (would disrupt unrelated sessions);
                # wake only this attempt's stream if responses.create() returned one,
                # otherwise rely on the bounded SDK/provider timeout.
                _close_attempt_stream("cancelled attempt stream close during timeout failed")
                return
            # FD-ownership contract: only the thread driving the request may
            # ``close()`` this client's FDs. From a stranger thread (the watchdog
            # Timer) only ``shutdown()`` is FD-safe — ``close()`` releases the raw
            # TLS fd while the owner's OpenSSL BIO still caches it, the kernel
            # recycles it (e.g. into a SQLite handle), and the owner's TLS flush
            # corrupts that file. The owner does the real close in its ``finally``.
            timeout_release_pending.set()
            if threading.get_ident() == owner_tid:
                _close_shared_client("client close during timeout failed")
            else:
                try:
                    from agent.agent_runtime_helpers import force_close_tcp_sockets

                    shutdown_count = force_close_tcp_sockets(self._client)
                    logger.info(
                        "Codex auxiliary client aborted (timeout, tcp_force_closed=%d, "
                        "deferred_close=stranger_thread)",
                        shutdown_count,
                    )
                except Exception:
                    logger.debug("Codex auxiliary: client abort during timeout failed", exc_info=True)
                # Socket shutdown only wakes a reader on a REAL transport; the owner
                # may be blocked inside the SDK's event stream (or a socketless test
                # double). Closing the attempt-owned stream releases it without
                # touching the shared client's FDs.
                _close_attempt_stream("attempt stream close during stranger-thread timeout failed")
            # The aux client cache wraps this same ``self._client``; drop the entry
            # so the next aux call doesn't reuse the dead transport and fail fast.
            try:
                _evict_cached_client_instance(self._client)
            except Exception:
                logger.debug("Codex auxiliary: cache eviction on timeout failed", exc_info=True)

        def _check_cancelled() -> None:
            if total_timeout is not None and time.monotonic() >= _effective_deadline():
                if not timed_out.is_set():
                    _close_client_on_timeout()
                raise TimeoutError(_timeout_message())
            try:
                from tools.interrupt import is_interrupted
                # Protected atomic aux tasks (compression) must not abort on a
                # mid-flight gateway interrupt (would trigger a degraded fallback
                # marker). Explicit host cancellation has its own exception; timeouts
                # still fire and unprotected aux tasks remain interruptible.
                if _aux_interrupt_cancel_requested():
                    raise AuxiliaryExplicitCancellation()
                if is_interrupted() and not _aux_interrupt_protected():
                    raise InterruptedError("Codex auxiliary Responses stream interrupted")
            except (InterruptedError, AuxiliaryExplicitCancellation):
                raise
            except Exception:
                # Interrupt state is best-effort UX; never a new failure mode.
                pass

        def _watchdog_fire() -> None:
            # Re-armable: if progress moved the deadline forward, reschedule
            # instead of killing a live stream.
            remaining = _effective_deadline() - time.monotonic()
            if remaining > 0:
                if timed_out.is_set() or stream_finished.is_set():
                    return
                t = threading.Timer(remaining, _watchdog_fire)
                t.daemon = True
                timeout_timer[0] = t
                t.start()
                return
            _close_client_on_timeout()

        try:
            if total_timeout:
                timeout_timer[0] = threading.Timer(
                    max(_effective_deadline() - time.monotonic(), 0.0),
                    _watchdog_fire,
                )
                timeout_timer[0].daemon = True
                timeout_timer[0].start()
            _check_cancelled()

            # Use low-level ``responses.create(stream=True)`` and assemble the final
            # response ourselves from ``response.output_item.done``: the high-level
            # ``responses.stream()`` helper reconstructs from
            # ``response.completed.response.output``, which the Codex backend has
            # returned as ``null`` (crashing the SDK with a NoneType TypeError).
            from agent.codex_runtime import (
                _bypass_sdk_request_transform,
                _consume_codex_event_stream,
            )

            stream_kwargs = dict(resp_kwargs)
            stream_kwargs["stream"] = True
            # Keep bulk wire payload out of the SDK's GIL-holding request transform.
            stream_kwargs = _bypass_sdk_request_transform(stream_kwargs)

            def _on_each_event(_event: Any) -> None:
                # Per event: TTFP telemetry records every frame, but forward
                # progress (compression commit fence, no-progress window) counts
                # only substantive payloads — keepalives must not re-arm, so a
                # zombie stream dies at the same window as a dead connection.
                if _codex_event_has_content(_event):
                    _record_stream_progress()
                    saw_content.set()
                    _notify_aux_provider_response()
                else:
                    _notify_aux_timing_response()
                _check_cancelled()

            event_stream = self._client.responses.create(**stream_kwargs)
            with attempt_stream_lock:
                attempt_stream.append(event_stream)
            # The timer may fire while responses.create() is blocked; if the
            # cancelled attempt had no stream to close then, close it now that it
            # is attempt-owned — never touch the shared client.
            if (
                timed_out.is_set()
                and callable(protected_cancel_check)
                and _captured_aux_cancel_requested(protected_cancel_check)
            ):
                close_fn = getattr(event_stream, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        logger.debug(
                            "Codex auxiliary: late cancelled attempt stream close failed",
                            exc_info=True,
                        )
            try:
                # Some Codex-compatible hosts accept ``stream=True`` but return a
                # completed Responses object (not iterable) — don't hand it to the consumer.
                if hasattr(event_stream, "output"):
                    final = event_stream
                else:
                    final = _consume_codex_event_stream(
                        event_stream,
                        model=str(resp_kwargs.get("model") or model),
                        on_event=_on_each_event,
                    )
            finally:
                close_fn = getattr(event_stream, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                with attempt_stream_lock:
                    attempt_stream.clear()

            if final is None:
                raise RuntimeError("Codex auxiliary Responses stream did not return a final response")

            text_parts, tool_calls_raw, usage = _parse_codex_final_response(final)
        except Exception as exc:
            if timed_out.is_set():
                raise TimeoutError(_timeout_message()) from exc
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise
        finally:
            stream_finished.set()
            _t = timeout_timer[0]
            if _t is not None:
                _t.cancel()
            # A stranger-thread timeout only shut sockets down; the owning thread
            # releases the FDs here. Gated on timeout_release_pending, NOT timed_out:
            # after a hard-cancel the shared client must stay usable for other sessions.
            if timeout_release_pending.is_set():
                _close_shared_client("owner-thread close after timeout failed")

        content = "".join(text_parts).strip() or None

        # Build a response that looks like chat.completions
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls_raw or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="stop" if not tool_calls_raw else "tool_calls",
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _ChatShim:
    """Exposes ``client.chat.completions.create()`` over a sync or async adapter."""

    def __init__(self, adapter: Any):
        self.completions = adapter


class _AsyncCompletionsAdapter:
    """Async adapter: runs the sync adapter's ``create`` via asyncio.to_thread()."""

    def __init__(self, sync_adapter: Any):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncAuxiliaryClientBase:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create().

    Mirrors ``_real_client`` (when the sync wrapper has one) so cache eviction by
    leaf OpenAI client drops this async entry too; otherwise it keeps reusing a
    closed transport.
    """

    def __init__(self, sync_wrapper: Any):
        self.chat = _ChatShim(_AsyncCompletionsAdapter(sync_wrapper.chat.completions))
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        if hasattr(sync_wrapper, "_real_client"):
            self._real_client = sync_wrapper._real_client


_AsyncAnthropicCompletionsAdapter = _AsyncCompletionsAdapter  # imported by tests


class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper routing through the Codex Responses API.

    Exposes .api_key/.base_url for introspection by async wrappers.
    """

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        self.chat = _ChatShim(_CodexCompletionsAdapter(real_client, model))
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class AsyncCodexAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


def _translate_anthropic_response_format(
    anthropic_kwargs: Dict[str, Any], response_format: Any,
) -> None:
    """Merge an OpenAI response format into Anthropic ``output_config``."""
    if not isinstance(response_format, dict):
        return

    format_type = response_format.get("type")
    if format_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict) or "schema" not in json_schema:
            return
        native_format = {
            "type": "json_schema",
            "schema": json_schema["schema"],
        }
    elif format_type == "json_object":
        # Anthropic SDK has no schema-less JSON mode; only ``json_schema``.
        native_format = {
            "type": "json_schema",
            "schema": {"type": "object"},
        }
    else:
        return

    output_config = anthropic_kwargs.get("output_config")
    if not isinstance(output_config, dict):
        output_config = {}
        anthropic_kwargs["output_config"] = output_config
    output_config["format"] = native_format


class _AnthropicCompletionsAdapter:
    """OpenAI-client-compatible adapter for Anthropic Messages API."""

    def __init__(
        self,
        real_client: Any,
        model: str,
        is_oauth: bool = False,
        base_url: str | None = None,
    ):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth
        # Prefer the caller-supplied URL; fall back to the SDK client's host only
        # for Nous Portal — a blanket fallback would flip MiniMax/Zhipu aux
        # adapters to third-party handling (stripping thinking signatures).
        self._base_url = base_url or None
        if not self._base_url:
            candidate = str(getattr(real_client, "base_url", "") or "") or None
            if candidate:
                try:
                    from agent.anthropic_adapter import _is_nous_portal_endpoint

                    if _is_nous_portal_endpoint(candidate):
                        self._base_url = candidate
                except Exception:
                    pass

    def create(self, **kwargs) -> Any:
        from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
        from agent.transports import get_transport

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        reasoning_config = kwargs.get("_reasoning_config")
        # ZAI's Anthropic endpoint rejects max_tokens on vision models (code 1210);
        # callers signal this via _skip_zai_max_tokens.
        _skip_mt = kwargs.pop("_skip_zai_max_tokens", False)
        if _skip_mt:
            max_tokens = None
        else:
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        temperature = kwargs.get("temperature")

        normalized_tool_choice = None
        if isinstance(tool_choice, str):
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type", "")).lower()
            if choice_type == "function":
                normalized_tool_choice = tool_choice.get("function", {}).get("name")
            elif choice_type in {"auto", "required", "none"}:
                normalized_tool_choice = choice_type

        # Reasoning priority: explicit per-call _reasoning_config (MoA per-slot)
        # wins over extra_body.reasoning; build_anthropic_kwargs translates to ``thinking``.
        _reasoning_cfg = reasoning_config
        if _reasoning_cfg is None:
            _eb = kwargs.get("extra_body")
            if isinstance(_eb, dict):
                _rc = _eb.get("reasoning")
                if isinstance(_rc, dict):
                    _reasoning_cfg = _rc

        anthropic_kwargs = build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_config=_reasoning_cfg,
            tool_choice=normalized_tool_choice,
            is_oauth=self._is_oauth,
            # Portal routes on ``anthropic/<slug>`` ids and replays signed thinking
            # keyed off base_url; omitting it breaks Portal model resolution.
            base_url=self._base_url,
        )
        # Opus 4.7+ rejects non-default temperature/top_p/top_k; build_anthropic_kwargs
        # also strips these as a safety net — keep both layers.
        if temperature is not None:
            from agent.anthropic_adapter import _forbids_sampling_params
            if not _forbids_sampling_params(model):
                anthropic_kwargs["temperature"] = temperature

        # Pass caller extra_body through (documented Anthropic SDK passthrough for
        # vendor fields), merged over build_anthropic_kwargs' own extra_body.
        # Excluded: ``reasoning`` and ``response_format`` (already TRANSLATED to
        # native fields — forwarding raw would 400 on strict gateways) and
        # ``_``-prefixed private Hermes plumbing.
        caller_extra_body = kwargs.get("extra_body")
        # A top-level ``response_format`` kwarg gets the same translation as the
        # extra_body form (previously silently dropped by the kwarg allow-list);
        # when both are present the extra_body form wins.
        top_level_response_format = kwargs.get("response_format")
        if top_level_response_format is not None:
            _translate_anthropic_response_format(
                anthropic_kwargs, top_level_response_format,
            )
        if caller_extra_body and isinstance(caller_extra_body, dict):
            _translate_anthropic_response_format(
                anthropic_kwargs, caller_extra_body.get("response_format"),
            )
            passthrough = {
                k: v for k, v in caller_extra_body.items()
                if k not in {"reasoning", "response_format"}
                and not str(k).startswith("_")
            }
            if passthrough:
                existing = anthropic_kwargs.get("extra_body") or {}
                if not isinstance(existing, dict):
                    existing = {}
                anthropic_kwargs["extra_body"] = {**existing, **passthrough}

        response = create_anthropic_message(
            self._client,
            anthropic_kwargs,
            # Record provider-response timing every event, but tick forward
            # progress only for substantive payloads so keepalives can't hold a
            # stalled summary open. None keeps the fast get_final_message path.
            on_stream_event=(
                _anthropic_aux_stream_event_hook()
                if _aux_progress_active()
                else None
            ),
        )
        _transport = get_transport("anthropic_messages")
        _nr = _transport.normalize_response(
            response, strip_tool_prefix=self._is_oauth
        )

        # ToolCall already duck-types as OpenAI shape via properties.
        assistant_message = SimpleNamespace(
            content=_nr.content,
            tool_calls=_nr.tool_calls,
            reasoning=_nr.reasoning,
        )
        finish_reason = _nr.finish_reason

        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        choice = SimpleNamespace(
            index=0,
            message=assistant_message,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class AnthropicAuxiliaryClient:
    """OpenAI-client-compatible wrapper over a native Anthropic client."""

    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool = False):
        self._real_client = real_client
        self.chat = _ChatShim(_AnthropicCompletionsAdapter(
            real_client, model, is_oauth=is_oauth, base_url=base_url,
        ))
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, "close", None)
        if callable(close_fn):
            close_fn()


class AsyncAnthropicAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


class _BedrockCompletionsAdapter:
    """Translates ``chat.completions.create(**kwargs)`` into Bedrock Converse."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model

    def create(self, **kwargs) -> Any:
        from agent.bedrock_adapter import call_converse

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        # OpenAI accepts ``stop`` as str or list; Converse requires a list.
        stop = kwargs.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if kwargs.get("tool_choice") is not None:
            # Converse toolChoice isn't wired through call_converse(); surface the drop.
            logger.debug(
                "BedrockAuxiliaryClient: tool_choice=%r not supported by the "
                "Converse shim — ignored.", kwargs.get("tool_choice"),
            )
        if kwargs.get("stream"):
            # Converse streaming isn't wired here; call_llm's streaming consumer
            # detects a final object and downgrades to non-live output.
            logger.debug(
                "BedrockAuxiliaryClient: stream=True requested for %s — "
                "returning a complete response (Converse shim does not "
                "stream); caller downgrades to non-streaming.",
                model,
            )
        response = call_converse(
            region=self._region,
            model=model,
            messages=messages,
            tools=kwargs.get("tools"),
            # Omitted/None cap → None so Bedrock uses the model max, matching the
            # no-cap-by-default policy of every other aux wire. Truthiness (not
            # ``is None``) is deliberate: it mirrors the Anthropic shim, so an
            # explicit 0 means "no cap" on both wires.
            max_tokens=int(max_tokens) if max_tokens else None,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stop_sequences=stop,
        )
        # Converse is complete-response here: mark provider progress only after
        # return so TTFP reflects real Bedrock latency, not dispatch/setup.
        _notify_aux_provider_response()
        return response


class BedrockAuxiliaryClient:
    """OpenAI-client-compatible wrapper over AWS Bedrock Converse API."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model
        self.chat = _ChatShim(_BedrockCompletionsAdapter(region, model))
        self.api_key = "aws-sdk"
        self.base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    def close(self):
        pass


class AsyncBedrockAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


def _endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """True if ``base_url`` speaks Anthropic Messages instead of OpenAI chat.completions.

    Mirrors ``hermes_cli.runtime_provider._detect_api_mode_for_url`` so aux and
    main agent agree on transport. Covers any ``/anthropic`` URL (MiniMax, Zhipu,
    LiteLLM gateways), ``api.kimi.com/coding`` (only speaks the Anthropic shape;
    chat.completions 404s on its model aliases), and ``api.anthropic.com``.
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith("/anthropic") or path.endswith("/anthropic/v1"):
        return True
    hostname = base_url_hostname(normalized)
    if hostname == "api.anthropic.com":
        return True
    return bool(hostname == "api.kimi.com" and "/coding" in normalized)


def _is_specialized_aux_client(client_obj: Any) -> bool:
    """True for clients that must never be re-dispatched through a wire adapter.

    Anthropic/Bedrock/Codex wrappers, plus any client declaring
    ``HERMES_SKIP_TRANSPORT_WRAP`` (native/ACP shims, in-tree or from a provider
    plugin) — a class-attribute declaration rather than isinstance, so this hot
    path never imports those client modules just to type-test.
    """
    return (
        _safe_isinstance(client_obj, (AnthropicAuxiliaryClient, BedrockAuxiliaryClient, CodexAuxiliaryClient))
        or _client_declares(client_obj, "HERMES_SKIP_TRANSPORT_WRAP")
    )


def _maybe_wrap_anthropic(
    client_obj: Any,
    model: str,
    api_key: str,
    base_url: str,
    api_mode: Optional[str] = None,
) -> Any:
    """Rewrap a plain OpenAI client in ``AnthropicAuxiliaryClient`` when the
    endpoint actually speaks Anthropic Messages.

    Single chokepoint for aux transport correction, run at the end of every
    ``resolve_provider_client`` branch so api_key providers, ``custom``, and
    future /anthropic gateways land on the right wire regardless of branch.
    Returns ``client_obj`` unchanged if already a complete client (an
    Anthropic/Codex wrapper, or any client declaring
    ``HERMES_SKIP_TRANSPORT_WRAP`` — native/ACP shims, in-tree or plugin), the
    endpoint is OpenAI-wire, ``api_mode`` is explicitly non-Anthropic, or the
    ``anthropic`` SDK is missing (falls back to OpenAI wire).
    """
    # Probe stubs only signal resolvability (skipping also avoids importing
    # adapter modules on the probe path); specialized adapters are never re-dispatched.
    if isinstance(client_obj, _AuxProbeClientStub) or _is_specialized_aux_client(client_obj):
        return client_obj
    # Explicit non-anthropic api_mode wins over URL heuristics.
    if api_mode and api_mode != "anthropic_messages":
        return client_obj
    if api_mode != "anthropic_messages" and not _endpoint_speaks_anthropic_messages(base_url):
        return client_obj

    try:
        from agent.anthropic_adapter import build_anthropic_client
    except ImportError:
        logger.warning(
            "Endpoint %s speaks Anthropic Messages but the anthropic SDK is "
            "not installed — falling back to OpenAI-wire (will likely 404).",
            base_url,
        )
        return client_obj

    try:
        real_client = build_anthropic_client(api_key, base_url)
    except Exception as exc:
        logger.warning(
            "Failed to build Anthropic client for %s (%s) — falling back to "
            "OpenAI-wire client.", base_url, exc,
        )
        return client_obj

    logger.debug(
        "Auxiliary transport: wrapping client in AnthropicAuxiliaryClient "
        "(model=%s, base_url=%s, api_mode=%s)",
        model, base_url[:60] if base_url else "", api_mode or "auto-detected",
    )
    return AnthropicAuxiliaryClient(
        real_client, model, api_key, base_url, is_oauth=False,
    )


def _read_nous_auth() -> Optional[dict]:
    """Read ~/.hermes/auth.json (or the credential pool) for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens, else None.
    """
    pool_present, entry = _select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }

    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text(encoding="utf-8-sig"))
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        # Must have at least an access_token or agent_key
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    """Extract a usable Nous inference JWT from stored auth state."""
    from hermes_cli.auth import _nous_invoke_jwt_is_usable

    for token_key, expiry_key in (
        ("agent_key", "agent_key_expires_at"),
        ("access_token", "expires_at"),
    ):
        token = provider.get(token_key)
        if not isinstance(token, str) or not token.strip():
            continue
        if _nous_invoke_jwt_is_usable(
            token,
            scope=provider.get("scope"),
            expires_at=provider.get(expiry_key),
        ):
            return token
    return ""


def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


def _resolve_nous_pool_runtime_api(*, force_refresh: bool = False) -> Optional[tuple[str, str]]:
    """Resolve Nous auxiliary credentials from the selected pool entry."""
    try:
        from hermes_cli.auth import _agent_key_is_usable

        pool = load_pool("nous")
    except Exception as exc:
        logger.debug("Auxiliary Nous pool credential resolution failed: %s", exc)
        return None

    if not pool or not pool.has_credentials():
        return None

    try:
        entry = pool.select()
    except Exception as exc:
        logger.debug("Auxiliary Nous pool selection failed: %s", exc)
        return None

    if entry is None:
        return None

    def _entry_state(e: Any) -> Dict[str, Any]:
        return {
            k: getattr(e, k, None)
            for k in ("agent_key", "agent_key_expires_at", "access_token", "expires_at", "scope")
        }

    if force_refresh or not _agent_key_is_usable(_entry_state(entry), _nous_min_key_ttl_seconds()):
        try:
            refreshed = pool.try_refresh_current()
        except Exception as exc:
            logger.debug("Auxiliary Nous pool refresh failed: %s", exc)
            refreshed = None
        if refreshed is None:
            return None
        entry = refreshed

    api_key = _nous_api_key(_entry_state(entry))
    base_url = _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL)
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_nous_runtime_api(*, force_refresh: bool = False) -> Optional[tuple[str, str]]:
    """Return fresh Nous runtime credentials (pool first, then auth store + JWT refresh).

    Mirrors the main agent's 401 recovery path rather than trusting raw auth.json tokens.
    """
    pooled = _resolve_nous_pool_runtime_api(force_refresh=force_refresh)
    if pooled is not None:
        return pooled

    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials(
            timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logger.debug("Auxiliary Nous runtime credential resolution failed: %s", exc)
        return None

    return _creds_pair(creds)


def _creds_pair(creds: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """``(api_key, base_url)`` from a runtime-credentials dict, or None when either is missing."""
    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_xai_oauth_for_aux() -> Optional[Tuple[str, str]]:
    """Resolve a fresh xAI OAuth (api_key, base_url) for auxiliary clients, or None.

    Pool first (some xAI OAuth logins exist only as pool entries), then the
    singleton auth-store resolver for older logins.
    """
    try:
        from hermes_cli.auth import (
            DEFAULT_XAI_OAUTH_BASE_URL,
            _xai_validate_inference_base_url,
        )

        pool = load_pool("xai-oauth")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = str(
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                    or ""
                ).strip()
                base_url = _xai_validate_inference_base_url(
                    os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
                    or os.getenv("XAI_BASE_URL", "").strip().rstrip("/")
                    or str(getattr(entry, "runtime_base_url", None) or "").strip().rstrip("/")
                    or str(getattr(entry, "base_url", None) or "").strip().rstrip("/"),
                    fallback=DEFAULT_XAI_OAUTH_BASE_URL,
                )
                if api_key and base_url:
                    return api_key, base_url
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth pool credential resolution failed: %s", exc)

    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth runtime credential resolution failed: %s", exc)
        return None
    return _creds_pair(creds)


def _read_codex_access_token() -> Optional[str]:
    """Read a valid, non-expired Codex OAuth access token from Hermes auth store.

    A present-but-exhausted pool falls back to the profile's auth.json token
    instead of hard-failing.
    """
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token

    try:
        from hermes_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        # Expired JWTs would block the auto chain and prevent fallback to working providers.
        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass  # Non-JWT token or decode error — use as-is

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order; (client, model) or (None, None)."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        if _is_provider_unhealthy(provider_id):
            logger.debug("Auxiliary api-key chain: %s is unhealthy, skipping", provider_id)
            continue
        if provider_id == "anthropic":
            # Gate on explicit config so Claude Code credentials aren't silently
            # used as auxiliary fallback.
            try:
                from hermes_cli.auth import is_provider_explicitly_configured
                if not is_provider_explicitly_configured("anthropic"):
                    continue
            except ImportError:
                pass
            return _try_anthropic()

        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue

            raw_base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            via = " via pool"
        else:
            creds = resolve_api_key_provider_credentials(provider_id)
            api_key = str(creds.get("api_key", "")).strip()
            if not api_key:
                continue
            raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
            via = ""

        model = _get_aux_model_for_provider(provider_id) or None
        if model is None:
            continue  # skip provider if we don't know a valid aux model
        logger.debug("Auxiliary text client: %s (%s)%s", pconfig.name, model, via)
        return _build_api_key_chain_client(provider_id, api_key, raw_base_url, model)

    return None, None


def _endpoint_default_headers(
    base_url: str, provider: str, *, is_vision: bool = False, xai: bool = False,
) -> Optional[dict]:
    """Provider-specific client headers by endpoint host, merged with user ``model.default_headers``.

    Kimi Code needs the claude-code User-Agent; Copilot needs its request headers
    (``is_vision`` adds Copilot-Vision-Request); NVIDIA NIM and (optionally) xAI have
    their own fingerprints; anything else falls back to the provider profile.
    """
    if base_url_host_matches(base_url, "api.kimi.com"):
        headers: dict = {"User-Agent": "claude-code/0.1.0"}
    elif base_url_host_matches(base_url, "githubcopilot.com"):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = dict(copilot_request_headers(is_agent_turn=True, is_vision=is_vision))
    elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
        headers = dict(build_nvidia_nim_headers(base_url))
    elif xai and base_url_host_matches(base_url, "x.ai"):
        from tools.xai_http import hermes_xai_default_headers

        headers = dict(hermes_xai_default_headers())
    else:
        headers = _profile_default_headers(provider) or {}
    return _apply_user_default_headers(headers or None) or None


def _profile_default_headers(provider: str) -> Optional[dict]:
    """Client-level attribution headers from the provider profile (e.g. GMI User-Agent), or None."""
    if not provider:
        return None
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        if profile and profile.default_headers:
            return dict(profile.default_headers)
    except Exception:
        pass
    return None


def _build_api_key_chain_client(
    provider_id: str, api_key: str, raw_base_url: str, model: str,
) -> Tuple[Any, str]:
    """Build the auto-chain client for one API-key provider (native Gemini, else OpenAI-wire + Anthropic rewrap)."""
    base_url = _to_openai_base_url(raw_base_url)
    if provider_id == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        if is_native_gemini_base_url(base_url):
            return GeminiNativeClient(api_key=api_key, base_url=base_url), model
    if base_url_host_matches(base_url, "api.kimi.com"):
        headers = {"User-Agent": "claude-code/0.1.0"}
    elif base_url_host_matches(base_url, "githubcopilot.com"):
        from hermes_cli.models import copilot_default_headers

        headers = copilot_default_headers()
    elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
        headers = build_nvidia_nim_headers(base_url)
    else:
        headers = _profile_default_headers(provider_id)
    extra = {}
    if headers:
        extra["default_headers"] = headers
    merged = _apply_user_default_headers(extra.get("default_headers"))
    if merged:
        extra["default_headers"] = merged
    client = _create_openai_client(api_key=api_key, base_url=base_url, **extra)
    return _maybe_wrap_anthropic(client, model, api_key, raw_base_url), model


# ── Provider resolution helpers ─────────────────────────────────────────────


_paid_lane_warned: set = set()


def _is_free_model(model: Optional[str]) -> bool:
    """True when ``model`` is a free SKU (``:free`` suffix or ``stealth/`` prefix) — naming-convention trust."""
    if not model:
        return False
    normalized = str(model).strip()
    return normalized.endswith(":free") or normalized.startswith("stealth/")


def _aux_openrouter_settings() -> Tuple[bool, str]:
    """Read (free_only, openrouter_model) from config; (False, _OPENROUTER_MODEL) on failure."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
        free_only = bool(cfg_get(cfg, "auxiliary", "free_only", default=False))
        val = cfg_get(cfg, "auxiliary", "openrouter_model")
        model = val.strip() if isinstance(val, str) and val.strip() else _OPENROUTER_MODEL
        return free_only, model
    except Exception:
        return False, _OPENROUTER_MODEL


def _warn_paid_lane_once(model: str) -> None:
    """Log a WARNING the first time a non-free OpenRouter model is engaged."""
    if model in _paid_lane_warned:
        return
    _paid_lane_warned.add(model)
    logger.warning(
        "Auxiliary client: PAID lane engaged for auxiliary task — OpenRouter "
        "fallback model %r is not a :free SKU and may incur real spend. Set "
        "auxiliary.free_only: true to restrict auxiliary fallbacks to free "
        "models, or auxiliary.openrouter_model to a :free model.",
        model,
    )


def _try_openrouter(explicit_api_key: str = None, model: str = None) -> Tuple[Optional[OpenAI], Optional[str]]:
    free_only, cfg_model = _aux_openrouter_settings()
    or_model = model or cfg_model
    if free_only and not _is_free_model(or_model):
        logger.warning(
            "Auxiliary client: auxiliary.free_only is enabled but the "
            "OpenRouter fallback model %r is not a :free SKU — skipping the "
            "OpenRouter fallback. Set auxiliary.openrouter_model to a :free "
            "model (e.g. nvidia/nemotron-3-ultra-550b-a55b:free) or disable "
            "auxiliary.free_only.",
            or_model,
        )
        return None, None
    if not _is_free_model(or_model):
        _warn_paid_lane_once(or_model)

    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        or_key = explicit_api_key or _pool_runtime_api_key(entry)
        if or_key:
            base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
            logger.debug("Auxiliary client: OpenRouter via pool")
            return _create_openai_client(api_key=or_key, base_url=base_url,
                           default_headers=build_or_headers()), or_model
        # Exhausted pool: fall through to OPENROUTER_API_KEY rather than fail.
        logger.debug("Auxiliary client: OpenRouter pool exhausted, trying OPENROUTER_API_KEY")

    or_key = explicit_api_key or _scoped_key_env("OPENROUTER_API_KEY")
    if not or_key:
        _mark_provider_unhealthy("openrouter", ttl=60)
        return None, None
    logger.debug("Auxiliary client: OpenRouter")
    return _create_openai_client(api_key=or_key, base_url=OPENROUTER_BASE_URL,
                   default_headers=build_or_headers()), or_model


def _describe_openrouter_unavailable(model: str = None) -> str:
    """Return the policy or credential reason OpenRouter was unavailable."""
    free_only, cfg_model = _aux_openrouter_settings()
    or_model = model or cfg_model
    if free_only and not _is_free_model(or_model):
        return (
            f"auxiliary.free_only rejected non-free model {or_model!r}; "
            "the request was skipped before provider availability checks"
        )
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        if entry is None:
            return "OpenRouter credential pool has no usable entries (credentials may be exhausted)"
        if not _pool_runtime_api_key(entry):
            return "OpenRouter credential pool entry is missing a runtime API key"
    if not _scoped_key_env("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY not set"
    return "no usable OpenRouter credentials found"


def _try_nous(vision: bool = False) -> Tuple[Optional[OpenAI], Optional[str]]:
    # Cross-session rate guard: if another session recorded a 429, skip Nous
    # rather than pile onto the tapped RPH bucket.
    try:
        from agent.nous_rate_guard import nous_rate_limit_remaining
        _remaining = nous_rate_limit_remaining()
        if _remaining is not None and _remaining > 0:
            logger.debug(
                "Auxiliary: skipping Nous Portal (rate-limited, resets in %.0fs)",
                _remaining,
            )
            _mark_provider_unhealthy("nous", ttl=_remaining)
            return None, None
    except Exception:
        pass

    nous = _read_nous_auth()
    runtime = _resolve_nous_runtime_api(force_refresh=False)
    if runtime is None and not nous:
        logger.warning(
            "Auxiliary Nous client unavailable: no Nous authentication found "
            "(run: hermes auth)."
        )
        _mark_provider_unhealthy("nous", ttl=60)
        return None, None
    if runtime is None and nous:
        logger.debug(
            "Auxiliary Nous: runtime JWT refresh failed; checking stored "
            "auth.json token."
        )
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug("Auxiliary client: Nous Portal")

    # Portal /api/nous/recommended-models is authoritative (tier-aware); fall
    # back to _NOUS_MODEL when unreachable or null.
    model = _NOUS_MODEL
    if not _aux_probe_active():
        # Probes skip the lookup: exact model is irrelevant and it hits the network.
        try:
            from hermes_cli.models import get_nous_recommended_aux_model
            recommended = get_nous_recommended_aux_model(vision=vision)
            if recommended:
                model = recommended
                logger.debug(
                    "Auxiliary/%s: using Portal-recommended model %s",
                    "vision" if vision else "text", model,
                )
            else:
                logger.debug(
                    "Auxiliary/%s: no Portal recommendation, falling back to %s",
                    "vision" if vision else "text", model,
                )
        except Exception as exc:
            logger.debug(
                "Auxiliary/%s: recommended-models lookup failed (%s); "
                "falling back to %s",
                "vision" if vision else "text", exc, model,
            )

    if runtime is not None:
        api_key, base_url = runtime
    else:
        api_key = _nous_api_key(nous or {})
        if not api_key:
            logger.warning(
                "Auxiliary Nous client unavailable: no usable inference JWT found "
                "(run: hermes auth add nous)."
            )
            _mark_provider_unhealthy("nous", ttl=60)
            return None, None
        base_url = str((nous or {}).get("inference_base_url") or _nous_base_url()).rstrip("/")
    return (
        _create_openai_client(
            api_key=api_key,
            base_url=base_url,
        ),
        model,
    )


def _refresh_nous_recommended_model(
    *, vision: bool, stale_model: Optional[str]
) -> Optional[str]:
    """Force a fresh Portal recommended-model fetch after a stale-model 404.

    Long-lived processes cache the Portal payload and can pin a model that was
    later dropped from the catalog. Returns the fresh recommendation, else
    ``_NOUS_MODEL``, whichever differs from ``stale_model``; ``None`` if neither.
    """
    stale = (stale_model or "").strip().lower()
    fresh: Optional[str] = None
    try:
        from hermes_cli.models import get_nous_recommended_aux_model

        fresh = get_nous_recommended_aux_model(vision=vision, force_refresh=True)
    except Exception as exc:
        logger.debug(
            "Nous recommended-model refresh failed (%s); using default %s",
            exc, _NOUS_MODEL,
        )
    if fresh and fresh.strip().lower() != stale:
        return fresh
    # Fall back to the known-good default only if it actually differs.
    if _NOUS_MODEL.strip().lower() != stale:
        return _NOUS_MODEL
    return None


def _read_main_field(field: str, *, readonly: bool, lower: bool = False) -> str:
    """Main ``model.<field>``: process-local runtime override (``set_runtime_main``) first, then config.yaml.

    The override wins so tools gating on "the active main model" see the live
    CLI/gateway runtime, not the persisted default. ``readonly`` picks
    ``load_config_readonly`` (model/provider) vs ``load_config`` (api_key/base_url).
    """
    override = _runtime_main_value(field)
    if isinstance(override, str) and override.strip():
        value = override.strip()
        return value.lower() if lower else value
    try:
        from hermes_cli import config as _cfg_mod

        cfg = (_cfg_mod.load_config_readonly if readonly else _cfg_mod.load_config)()
        model_cfg = cfg.get("model", {})
        if field == "model" and isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            value = model_cfg.get("default" if field == "model" else field, "")
            if isinstance(value, str) and value.strip():
                value = value.strip()
                return value.lower() if lower else value
    except Exception:
        pass
    return ""


def _read_main_model() -> str:
    """Active main model (runtime override, else config.yaml ``model.default``), or ""."""
    return _read_main_field("model", readonly=True)


def _read_main_provider() -> str:
    """Lowercase main provider id (runtime override first, then config.yaml), or ""."""
    return _read_main_field("provider", readonly=True, lower=True)


def _read_main_api_key() -> str:
    """Main model API key; lets ``custom`` aux tasks with a base_url but empty api_key inherit main creds."""
    return _read_main_field("api_key", readonly=False)


def _read_main_base_url() -> str:
    """Main model base_url: runtime override first, then config.yaml."""
    return _read_main_field("base_url", readonly=False)


def _resolve_moa_aggregator(preset_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a MoA preset to its aggregator (provider, model); (None, None) if unresolvable.

    "moa" is virtual — aux tasks skip the reference fan-out and use the
    aggregator slot. Single shared helper so preset lookup can't drift between
    ``_resolve_auto``, ``_resolve_task_provider_model`` and ``resolve_provider_client``.
    ``preset_name`` None/"" resolves the user's default preset.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import resolve_moa_preset

        preset = resolve_moa_preset(load_config().get("moa") or {}, preset_name or None)
        agg = preset.get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_provider and agg_model and agg_provider.lower() != "moa":
            return agg_provider, agg_model
    except Exception:
        logger.debug(
            "MoA aggregator resolution failed for preset %r", preset_name, exc_info=True
        )
    return None, None


def _read_main_model_for_aux() -> str:
    """Main model with MoA presets unwrapped to the aggregator's model.

    A MoA preset name is never a valid wire model id; aux chains pre-filling
    from the main model must use this. Returns "" when the preset can't be
    resolved — sending nothing beats sending a name that 400s.
    """
    model = _read_main_model()
    if (_read_main_provider() or "").strip().lower() == "moa":
        _, agg_model = _resolve_moa_aggregator(model)
        return agg_model or ""
    return model


def _read_main_api_key_if_same_host(aux_base_url: str) -> str:
    """Return the main api_key only when *aux_base_url* shares the main base_url's host.

    Unconditional inheritance would leak the main credential to any host a
    misconfigured aux base_url names; a mismatch keeps ``no-key-required`` → 401.
    """
    aux_host = base_url_hostname(aux_base_url)
    if not aux_host:
        return ""
    main_host = base_url_hostname(_read_main_base_url())
    if not main_host or aux_host != main_host:
        return ""
    return _read_main_api_key()


# Compatibility mirrors for older readers/tests; the ContextVar below is
# authoritative (overlapping gateway sessions make a process-global unsafe).
_RUNTIME_MAIN_PROVIDER: str = ""
_RUNTIME_MAIN_MODEL: str = ""
_RUNTIME_MAIN_BASE_URL: str = ""
_RUNTIME_MAIN_API_KEY: Any = ""
_RUNTIME_MAIN_API_MODE: str = ""
_RUNTIME_MAIN_AUTH_MODE: str = ""
_RUNTIME_MAIN_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_runtime_main", default=None)
)

_RELAY_AUX_CALL_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_relay_call", default=None)
)


def _new_relay_aux_call_context(args: tuple, kwargs: dict) -> Dict[str, Any]:
    task = args[0] if args else kwargs.get("task")
    return {
        "task": str(task or "unknown"),
        "request_id": f"aux-{uuid.uuid4().hex}",
        "attempt_count": 0,
        "provider": "",
        "model": "",
        "response_model": None,
        "api_mode": "chat_completions",
    }


def _relay_auxiliary_call(callback):
    """Give every physical retry in one auxiliary call a shared Relay identity."""

    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        token = _RELAY_AUX_CALL_CONTEXT.set(_new_relay_aux_call_context(args, kwargs))
        try:
            return callback(*args, **kwargs)
        except BaseException:
            _fail_relay_auxiliary_call()
            raise
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    return wrapped


def _relay_auxiliary_call_async(callback):
    """Async counterpart to :func:`_relay_auxiliary_call`."""

    @functools.wraps(callback)
    async def wrapped(*args, **kwargs):
        token = _RELAY_AUX_CALL_CONTEXT.set(_new_relay_aux_call_context(args, kwargs))
        try:
            return await callback(*args, **kwargs)
        except BaseException:
            _fail_relay_auxiliary_call()
            raise
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    return wrapped


def _set_relay_auxiliary_route(
    provider: str | None,
    model: str | None,
    api_mode: str | None,
) -> None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    context["provider"] = str(provider or "auxiliary")
    context["model"] = str(model or "unknown")
    context["response_model"] = None
    context["api_mode"] = str(api_mode or "chat_completions")


def _record_route_info(
    route_info: Optional[Dict[str, str]],
    provider: Optional[str],
    model: Optional[str],
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


def _relay_auxiliary_metadata(
    *,
    provider: str | None = None,
    api_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return None
    attempt_count = int(context.get("attempt_count") or 0)
    context["attempt_count"] = attempt_count + 1
    provider_name = str(provider or context.get("provider") or "auxiliary")
    model_name = str(context.get("model") or "unknown")
    return provider_name, model_name, {
        "api_mode": str(api_mode or context.get("api_mode") or "chat_completions"),
        "api_request_id": str(context["request_id"]),
        "call_role": f"auxiliary:{context['task']}",
        "retry_count": attempt_count,
        "auxiliary_task": str(context["task"]),
    }


def _relay_sync_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    # Isolate only the provider callback so the owning thread can unwind its
    # lease/DB transaction on hard cancel without touching the shared client.
    if route is None:
        return _run_protected_sync_provider_call(callback, kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return relay_llm.execute_current(
        kwargs,
        lambda request: _run_protected_sync_provider_call(callback, request),
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata,
        defer_logical_completion=True,
    )


async def _relay_async_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return await callback(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return await relay_llm.execute_current_async(
        kwargs,
        callback,
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata,
        defer_logical_completion=True,
    )


def _relay_sync_stream(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
) -> Any:
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return client.chat.completions.create(**kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return relay_llm.stream_current(
        kwargs,
        lambda request: client.chat.completions.create(**request),
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        finalizer=dict,
        metadata=metadata,
        completed_response_predicate=lambda value: hasattr(value, "choices"),
    )
_RUNTIME_MAIN_COMPAT_SNAPSHOT: Tuple[Any, ...] = ("", "", "", "", "", "")
_RUNTIME_MAIN_COMPAT_LOCK = threading.Lock()


def _compat_runtime_main() -> Optional[Dict[str, Any]]:
    """Expose deliberately patched legacy globals as a main context.

    Mirrors must never become runtime inputs: a direct patch counts only when
    it differs from the mirrored snapshot and only on the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    values = (
        _RUNTIME_MAIN_PROVIDER,
        _RUNTIME_MAIN_MODEL,
        _RUNTIME_MAIN_BASE_URL,
        _RUNTIME_MAIN_API_KEY,
        _RUNTIME_MAIN_API_MODE,
        _RUNTIME_MAIN_AUTH_MODE,
    )
    if values == _RUNTIME_MAIN_COMPAT_SNAPSHOT:
        return None
    return dict(zip(_MAIN_RUNTIME_FIELDS, values))


def _runtime_main_value(field: str) -> Any:
    """Read one runtime field through context-local/controlled legacy state."""
    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if runtime is None:
        runtime = _compat_runtime_main()
    if isinstance(runtime, dict):
        value = runtime.get(field)
        if value:
            return value
    return ""


def set_runtime_main(
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
    base_url: str = "",
    api_key: Any = "",
    api_mode: str = "",
    auth_mode: str = "",
    session_id: str = "",
    cache_scope: str = "",
) -> contextvars.Token:
    """Record the current context's live main runtime for auxiliary routing.

    Context-local so concurrent gateway sessions don't clobber each other;
    legacy mirrors are updated for old readers. ``cache_scope`` is the
    rotation-stable logical cache scope, preferred over ``session_id`` for
    prompt_cache_key derivation.
    """
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    global _RUNTIME_MAIN_AUTH_MODE, _RUNTIME_MAIN_COMPAT_SNAPSHOT
    runtime = {
        "provider": (provider or "").strip().lower(),
        "requested_provider": (requested_provider or "").strip().lower(),
        "model": (model or "").strip(),
        "base_url": (base_url or "").strip(),
        "api_key": (
            api_key.strip()
            if isinstance(api_key, str)
            else api_key if callable(api_key) else ""
        ),
        "api_mode": (api_mode or "").strip(),
        "auth_mode": (auth_mode or "").strip().lower(),
        "session_id": (session_id or "").strip(),
        "cache_scope": (cache_scope or "").strip(),
    }
    # Publish authoritative context before updating the locked mirrors.
    token = _RUNTIME_MAIN_CONTEXT.set(runtime)
    with _RUNTIME_MAIN_COMPAT_LOCK:
        (
            _RUNTIME_MAIN_PROVIDER,
            _RUNTIME_MAIN_MODEL,
            _RUNTIME_MAIN_BASE_URL,
            _RUNTIME_MAIN_API_KEY,
            _RUNTIME_MAIN_API_MODE,
            _RUNTIME_MAIN_AUTH_MODE,
        ) = (runtime[field] for field in _MAIN_RUNTIME_FIELDS)
        _RUNTIME_MAIN_COMPAT_SNAPSHOT = tuple(
            runtime[field] for field in _MAIN_RUNTIME_FIELDS
        )
    return token


def reset_runtime_main(token: contextvars.Token) -> None:
    """Restore the runtime binding that preceded one scoped turn."""
    if token is None:
        return
    try:
        _RUNTIME_MAIN_CONTEXT.reset(token)
    except (RuntimeError, ValueError):
        # Tokens can't be reset from a copied Context (background workers
        # inherit values, not token ownership).
        pass


@contextlib.contextmanager
def scoped_runtime_main(main_runtime: Optional[Dict[str, Any]]):
    """Temporarily bind an explicit runtime without touching legacy mirrors."""
    runtime = _normalize_main_runtime(main_runtime)
    token = _RUNTIME_MAIN_CONTEXT.set(runtime or None)
    try:
        yield runtime
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)


def clear_runtime_main() -> None:
    """Clear the runtime override in the current context."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    global _RUNTIME_MAIN_AUTH_MODE, _RUNTIME_MAIN_COMPAT_SNAPSHOT
    _RUNTIME_MAIN_CONTEXT.set(None)
    with _RUNTIME_MAIN_COMPAT_LOCK:
        _RUNTIME_MAIN_PROVIDER = ""
        _RUNTIME_MAIN_MODEL = ""
        _RUNTIME_MAIN_BASE_URL = ""
        _RUNTIME_MAIN_API_KEY = ""
        _RUNTIME_MAIN_API_MODE = ""
        _RUNTIME_MAIN_AUTH_MODE = ""
        _RUNTIME_MAIN_COMPAT_SNAPSHOT = ("", "", "", "", "", "")


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint like the main CLI (env OPENAI_BASE_URL or config-saved)."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None

    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        openai_key = _scoped_key_env("OPENAI_API_KEY")
        if not openai_base:
            return None, None, None
        runtime = {
            "base_url": openai_base,
            "api_key": openai_key,
        }

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None

    custom_base = custom_base.strip().rstrip("/")
    if base_url_host_matches(custom_base, "openrouter.ai"):
        # requested='custom' falls back to OpenRouter when unconfigured; treat as "no custom endpoint".
        return None, None, None

    # Local servers (Ollama, vLLM, ...) ignore auth but the SDK needs a non-empty key.
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None

    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""


def _validate_proxy_env_urls() -> None:
    """Fail fast with a clear error when proxy env vars have malformed URLs.

    A shell typo like ``HTTP_PROXY=http://127.0.0.1:6153export NEXT=...`` otherwise
    surfaces as a cryptic httpx ``Invalid port`` that doesn't name the env var.
    """
    from urllib.parse import urlparse

    normalize_proxy_env_vars()

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `hermes setup` or `hermes model` and enter a valid http(s) base URL."
        ) from exc


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    if len(runtime) == 2:
        custom_base, custom_key = runtime
        custom_mode = None
    else:
        custom_base, custom_key, custom_mode = runtime
    if not custom_base or not custom_key:
        return None, None
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return None, None
    model = _read_main_model_for_aux() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s, api_mode=%s)", model, custom_mode or "chat_completions")
    _clean_base, _dq = _extract_url_query_params(custom_base)
    _extra = {"default_query": _dq} if _dq else {}
    # User model.default_headers override the SDK fingerprint headers here too,
    # matching the main client so the whole session passes a strict gateway/WAF.
    _custom_headers = _apply_user_default_headers(None)
    if _custom_headers:
        _extra["default_headers"] = _custom_headers
    if custom_mode == "codex_responses":
        real_client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)
        return CodexAuxiliaryClient(real_client, model), model
    if custom_mode == "anthropic_messages":
        # Third-party Anthropic-compatible gateway — never OAuth (that's api.anthropic.com only).
        try:
            from agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning(
                "Custom endpoint declares api_mode=anthropic_messages but the "
                "anthropic SDK is not installed — falling back to OpenAI-wire."
            )
            return _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra), model
        return (
            AnthropicAuxiliaryClient(real_client, model, custom_key, custom_base, is_oauth=False),
            model,
        )
    # URL-based anthropic detection for custom endpoints without explicit api_mode.
    _fallback_client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)
    _fallback_client = _maybe_wrap_anthropic(
        _fallback_client, model, custom_key, custom_base, custom_mode,
    )
    return _fallback_client, model


def _build_xai_oauth_aux_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for xAI Grok OAuth (Responses API); (None, None) if not authed.

    Caller must pass an explicit model — a pinned Grok default would rot as
    xAI's allowlist drifts.
    """
    if not model:
        logger.warning(
            "Auxiliary client: xai-oauth requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    resolved = _resolve_xai_oauth_for_aux()
    if resolved is None:
        return None, None
    api_key, base_url = resolved
    logger.debug("Auxiliary client: xAI OAuth (%s via Responses API)", model)
    from tools.xai_http import hermes_xai_default_headers

    real_client = _create_openai_client(
        api_key=api_key,
        base_url=base_url,
        default_headers=hermes_xai_default_headers(),
    )
    return CodexAuxiliaryClient(real_client, model), model


def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an explicit model; (None, None) without a Codex OAuth token.

    No auto-selected default: the Codex endpoint's model allow-list is
    undocumented and drifts, so any hardcoded default goes stale.
    """
    if not model:
        logger.warning(
            "Auxiliary client: openai-codex requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if codex_token:
            base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
        else:
            codex_token = _read_codex_access_token()
            if not codex_token:
                return None, None
            base_url = _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", model)
    real_client = _create_openai_client(
        api_key=codex_token,
        base_url=base_url,
        default_headers=_codex_cloudflare_headers(codex_token, base_url=base_url),
    )
    return CodexAuxiliaryClient(real_client, model), model


def _try_azure_foundry(
    *,
    model: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve an Azure Foundry auxiliary client via the main agent's runtime resolver.

    Delegating to ``_resolve_azure_foundry_runtime`` gives api_key vs Entra ID
    (callable bearer provider), per-model api_mode routing, entra config and
    base_url overrides for free. Returns ``(client, model)`` or ``(None, None)``.
    """
    try:
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        from hermes_cli.auth import AuthError
        from hermes_cli.config import load_config_readonly
    except ImportError:
        return None, None

    try:
        cfg = load_config_readonly()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
    except Exception:
        model_cfg = {}

    try:
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=model_cfg,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=model,
        )
    except AuthError as exc:
        logger.debug("Auxiliary azure-foundry: %s", exc)
        return None, None
    except Exception as exc:
        logger.debug("Auxiliary azure-foundry runtime error: %s", exc)
        return None, None

    api_key = runtime.get("api_key")
    base_url = str(runtime.get("base_url", "") or "")
    runtime_api_mode = api_mode or runtime.get("api_mode") or "chat_completions"

    # api_key may be a callable token provider (truthy); bail only on None/"".
    _has_key = bool(api_key) if not callable(api_key) else True
    if not _has_key or not base_url:
        return None, None

    final_model = _normalize_resolved_model(
        model or str(model_cfg.get("default") or ""),
        "azure-foundry",
    )
    if not final_model:
        # No fallback aux model for Azure (needs a deployment name) — return
        # "no client" so the auto chain falls through instead of 404ing.
        logger.debug(
            "Auxiliary azure-foundry: no model resolved (model=%r, default=%r)",
            model, model_cfg.get("default"),
        )
        return None, None

    # The SDK drops api-version query params from the base URL; pass via default_query.
    extra: Dict[str, Any] = {}
    _clean_base, _dq = _extract_url_query_params(base_url)
    if _dq:
        extra["default_query"] = _dq

    client = _create_openai_client(api_key=api_key, base_url=_clean_base, **extra)

    if runtime_api_mode == "codex_responses":
        # Responses-API-only models: translate chat.completions.create() to /responses.
        return CodexAuxiliaryClient(client, final_model), final_model

    if runtime_api_mode == "anthropic_messages":
        # Forward api_key verbatim (string or Entra callable); build_anthropic_client
        # installs the bearer-injecting hook for callables.
        return _maybe_wrap_anthropic(
            client, final_model, api_key,
            base_url, runtime_api_mode,
        ), final_model

    return client, final_model


def _try_anthropic(explicit_api_key: str = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    except ImportError:
        return None, None

    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present and entry is not None:
        token = explicit_api_key or _pool_runtime_api_key(entry)
    else:
        # Pool absent or has no usable entry: fall through to the legacy resolver
        # (like openrouter/codex) so a dead pool entry can't wedge aux tasks when
        # a valid standalone credential exists.
        entry = None
        token = explicit_api_key or resolve_anthropic_token()
    if not token:
        return None, None

    # Honor config.yaml model.base_url only when provider is anthropic AND the
    # URL is Anthropic-compatible; otherwise a foreign host (Codex, OpenRouter
    # accepting Anthropic-format requests) would 401 every aux side-channel call.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url and _is_anthropic_compatible_host(cfg_base_url):
                    base_url = cfg_base_url
    except Exception:
        pass

    from agent.anthropic_adapter import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _get_aux_model_for_provider("anthropic") or "claude-haiku-4-5-20251001"
    if _aux_probe_active():
        # Probe: token + adapter import resolved; skip real client construction.
        return _AuxProbeClientStub(api_key="", base_url=base_url), model
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        # Adapter imports fine but the anthropic SDK itself is missing.
        return None, None
    return AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model


_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode")
_MAIN_RUNTIME_CONTEXT_FIELDS = _MAIN_RUNTIME_FIELDS + ("requested_provider",)


def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override.

    ``api_key`` may be a zero-arg callable (Entra ID token provider, accepted
    by the OpenAI SDK) — preserved as-is so aux clients share main-agent auth.
    """
    if main_runtime is None:
        # Context-local state first; compat mirrors may hold another
        # concurrent session's endpoint/key.
        main_runtime = _RUNTIME_MAIN_CONTEXT.get()
        if main_runtime is None:
            main_runtime = _compat_runtime_main()
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _MAIN_RUNTIME_CONTEXT_FIELDS:
        value = main_runtime.get(field)
        if field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
            continue
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    for identity_field in ("provider", "requested_provider"):
        identity = normalized.get(identity_field)
        if isinstance(identity, str):
            normalized[identity_field] = identity.lower()
    return normalized


def _get_provider_chain() -> List[tuple]:
    """Return the ordered provider detection chain.

    Built at call time so test patches on ``_try_*`` are picked up.
    ``openai-codex`` is deliberately absent: its shifting model allow-list
    makes guessed-model fallback fail; it is used only as main provider or
    when explicitly requested with a model.
    """
    return [
        ("openrouter", _try_openrouter),
        ("nous", _try_nous),
        ("local/custom", _try_custom_endpoint),
        ("api-key", _resolve_api_key_provider),
    ]


# ── Auxiliary "recently 402'd" unhealthy-provider cache ────────────────────
#
# A 402'd provider stays depleted for hours; re-trying it first on every aux
# call burns an RTT each time. Mark it unhealthy for a TTL so the chain skips
# it; entries auto-expire. In-process only (profiles may use different keys).

_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}

# resolved_provider / explicit-config names → chain labels.
_AUX_UNHEALTHY_LABEL_ALIASES = {
    "openrouter": "openrouter",
    "nous": "nous",
    "custom": "local/custom",
    "local/custom": "local/custom",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
}


def _normalize_chain_label(provider: str) -> str:
    """Normalize a resolved_provider value to a chain label; unknown
    direct API-key providers fall back to the lowercased input."""
    if not provider:
        return ""
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)


def _mark_provider_unhealthy(provider: str, ttl: Optional[float] = None) -> None:
    """Hide ``provider`` from chain iteration until the TTL expires (after a confirmed payment error)."""
    label = _normalize_chain_label(provider)
    if not label:
        return
    ttl = _AUX_UNHEALTHY_TTL_SECONDS if ttl is None else ttl
    expires_at = time.time() + ttl
    _aux_unhealthy_until[label] = expires_at
    logger.warning(
        "Auxiliary: marking %s unhealthy for %ds (payment / credit error). "
        "Subsequent auxiliary calls will skip it until %s.",
        label, int(ttl), time.strftime("%H:%M:%S", time.localtime(expires_at)),
    )


def _is_provider_unhealthy(label: str) -> bool:
    """True iff ``label`` is unhealthy and unexpired; lazily evicts expired entries."""
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True


def _log_skip_unhealthy(label: str, task: Optional[str] = None) -> None:
    """Log a skipped unhealthy provider at most once per minute per label."""
    now = time.time()
    last = _aux_unhealthy_logged_at.get(label, 0.0)
    if now - last >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info(
            "Auxiliary %s: skipping %s (recently returned payment error, retry in %ds)",
            task or "call", label, max(0, int(expires_at - now)),
        )


def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache (tests / explicit user reset)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    True for HTTP 402, and for 429/other codes whose message indicates billing
    or daily-quota exhaustion (functionally credit exhaustion) rather than
    transient rate limiting.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # Providers sometimes wrap credit errors in 429/403/404 bodies.
    if status in {402, 403, 404, 429, None} and any(kw in err_lower for kw in (
        "credits", "insufficient funds",
        "can only afford", "billing",
        "payment required",
        "out of funds", "run out of funds",
        "balance_depleted", "no usable credits",
        "model_not_supported_on_free_tier",
        "not available on the free tier",
        "requires a subscription", "upgrade for access",
        "upgrade for higher limits", "reached your session usage limit",
        # Daily / monthly / weekly quota exhaustion keywords
        "quota exceeded", "quota_exceeded",
        "too many tokens per day", "daily limit",
        "tokens per day", "daily quota",
        "resource exhausted",  # Vertex AI / gRPC quota errors
        "weekly usage limit", "weekly limit",  # OpenCode Go weekly subscription cap
    )):
        return True
    return False


def _nous_portal_account_has_fresh_paid_access() -> bool:
    """Return True only when the fresh Nous account API says paid access is allowed."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        return account_info.paid_service_access is True
    except Exception as exc:
        logger.debug("Auxiliary Nous paid-entitlement refresh check failed: %s", exc)
        return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect 429 rate-limit errors (not billing/quota, which _is_payment_error owns)."""
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()

    # OpenAI SDK's RateLimitError may omit .status_code — match by class name.
    if type(exc).__name__ == "RateLimitError":
        return True

    if status == 429:
        if any(kw in err_lower for kw in (
            "rate limit", "rate_limit", "too many requests",
            "try again", "retry after", "resets in",
        )):
            return True
        # Generic 429 without billing keywords = rate limit.
        if not any(kw in err_lower for kw in (
            "credits", "insufficient funds", "billing",
            "payment required", "can only afford",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
        )):
            return True
    return False


def _is_timeout_error(exc: Exception) -> bool:
    """Detect a full-budget request timeout, distinct from a fast connection drop.

    A timeout burns the whole ``timeout`` budget, so a same-provider retry on
    the compression path doubles wall time; fast drops stay on the retry path.
    """
    try:
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    except ImportError:
        pass
    if "Timeout" in type(exc).__name__:
        return True
    return "timed out" in str(exc).lower()


def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors (endpoint unreachable), as opposed to 4xx/5xx API errors."""
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
        # httpcore/httpx premature stream close — transient, retry/reroute.
        "incomplete chunked read",
        "peer closed connection",
        "response ended prematurely",
        "unexpected eof",
        "remoteprotocolerror",
        "localprotocolerror",
    )):
        return True
    return False


def _is_transient_transport_error(exc: Exception) -> bool:
    """True for a one-off transport blip worth retrying on the SAME provider.

    Connection/stream-close errors (via ``_is_connection_error``) plus pure
    5xx/408. Deliberately narrow: payment/auth/rate-limit errors are handled
    by switching provider, refreshing creds, or rotating the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


_DEFAULT_TRANSIENT_RETRIES = 2
# Backoff base (seconds); overridable so tests can zero it out.
_TRANSIENT_RETRY_BACKOFF_BASE = 1.0


def _transient_retry_count() -> int:
    """Same-provider retries for a transient blip: ``auxiliary.transient_retries``
    (default 2), clamped to [0, 6]; config-read failures fall back to default."""
    try:
        from hermes_cli.config import cfg_get, load_config

        val = cfg_get(load_config(), "auxiliary", "transient_retries")
        if val is None:
            return _DEFAULT_TRANSIENT_RETRIES
        n = int(val)
        return max(0, min(n, 6))
    except Exception:
        return _DEFAULT_TRANSIENT_RETRIES


def _is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    # xAI returns 403 "unauthenticated:bad-credentials" for expired OAuth tokens
    # — semantically a 401.
    if status == 403 and "bad-credentials" in err_lower:
        return True
    return bool("unauthenticated" in err_lower and "bad-credentials" in err_lower)


def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Detect provider 400s for an unsupported request parameter.

    Matches on both the parameter name and a generic unsupported/unknown/
    unrecognized marker (endpoints phrase this several ways) so call sites can
    retry without the offending key.
    """
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    if param_lower not in err_lower:
        return False
    return any(marker in err_lower for marker in (
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized request argument",
        "unrecognized parameter",
        "invalid parameter",
    ))


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Back-compat wrapper for ``temperature``; kept as a named symbol because tests/call sites import it."""
    return _is_unsupported_parameter_error(exc, "temperature")


def _is_structured_output_rejection(exc: Exception) -> bool:
    """Detect provider 400s that reject the structured-output request field.

    Covers both wires: OpenAI ``response_format`` (incl. vLLM translating it to
    ``guided_grammar`` and failing without xgrammar) and Anthropic
    ``output_config.format`` (older gateways: "Extra inputs are not permitted").
    Callers tolerate an unconstrained reply, so the reaction is one retry
    without the field.
    """
    status = getattr(exc, "status_code", None)
    if status is not None and status not in {400, 422}:
        return False
    err_lower = str(exc).lower()
    # vLLM grammar-backend failures name the translated parameter, not ours.
    if "guided_grammar" in err_lower or "xgrammar" in err_lower or (
        "compile_grammar_error" in err_lower
    ):
        return True
    if "extra inputs are not permitted" in err_lower and (
        "response_format" in err_lower or "output_config" in err_lower
    ):
        return True
    if "response_format" in err_lower and "unavailable" in err_lower:
        return True
    return (
        _is_unsupported_parameter_error(exc, "response_format")
        or _is_unsupported_parameter_error(exc, "output_config")
    )


def _without_structured_output_format(kwargs: dict) -> Optional[dict]:
    """Copy *kwargs* without ``response_format`` (top-level and ``extra_body``).

    Returns None when nothing was removed, so call sites don't retry an unchanged request.
    """
    changed = False
    retry_kwargs = dict(kwargs)
    if retry_kwargs.pop("response_format", None) is not None:
        changed = True
    extra_body = retry_kwargs.get("extra_body")
    if isinstance(extra_body, dict) and "response_format" in extra_body:
        remaining = {
            k: v for k, v in extra_body.items() if k != "response_format"
        }
        if remaining:
            retry_kwargs["extra_body"] = remaining
        else:
            retry_kwargs.pop("extra_body", None)
        changed = True
    return retry_kwargs if changed else None


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detect "the requested model doesn't exist" errors (404 / invalid model).

    Typically a long-lived process pinned a model since dropped from the
    catalog. Keys on "does not exist / not found" phrasing and excludes billing
    keywords, which :func:`_is_payment_error` owns.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "free tier", "free-tier",
        "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return any(kw in err_lower for kw in (
        "model does not exist",
        "does not exist in our configuration",
        "openrouter catalog",
        "is not a valid model",
        "no such model",
        "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found",
        "unknown model",
    ))


def _is_model_incompatible_error(exc: Exception) -> bool:
    """Detect "this route cannot serve this model" 400s (capability mismatch).

    The model exists but the current provider/account cannot run it (e.g. a
    Codex/ChatGPT-account fallback asked to compress a non-OpenAI model). Auth
    and payment predicates don't fire, so without this the whole aux task would
    abort; treating it as fallback-worthy lets the chain continue. Excludes
    billing 400s (payment path) and not-found 400s (_is_model_not_found_error).
    """
    status = getattr(exc, "status_code", None)
    if status not in {400, None}:
        return False
    err_lower = str(exc).lower()
    # Key on billing keywords directly: _is_payment_error is status-gated and
    # would not recognise a 400-coded billing body.
    if _is_model_not_found_error(exc):
        return False
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "payment required",
        "free tier", "free-tier", "not available on the free tier",
        "model_not_supported_on_free_tier", "quota",
    )):
        return False
    return any(kw in err_lower for kw in (
        "is not supported when using",   # codex/ChatGPT-account model gating
        "model is not supported",
        "not supported with this",
        "not supported for this account",
        "model_not_supported",
        "does not support this model",
        "unsupported model",
    ))


def _is_invalid_aux_response_error(exc: Exception) -> bool:
    """Detect HTTP-200 empty/malformed ChatCompletions — a capability failure
    that should follow the same fallback path as model-incompatibility errors."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "auxiliary " in msg
        and "llm returned invalid response" in msg
        and "choices[0].message" in msg
    )


# Auxiliary tasks that sit on a user-visible critical path. A same-provider
# retry after a full-budget timeout costs another whole ``timeout`` window
# before the fallback chain is reached, so these skip it and fall through
# immediately. Fast blips (a streaming-close or a 5xx) still retry, since
# those are cheap. See issue #54465 for the compression case.
_TIMEOUT_NO_RETRY_TASKS = frozenset({"compression", "vision"})


def _should_skip_same_provider_retry(task: Optional[str], exc: Exception) -> bool:
    """True when a transient error should go straight to fallback.

    Compression is on the critical preflight path: a user cannot continue or
    resume an oversized session until it compacts. Vision is on the
    interactive path: the turn holding the image cannot answer, and because
    turns are serialised the following user messages stall behind it. For
    those tasks a same-provider retry on a full-budget timeout means another
    whole ``timeout`` of wall-clock before the fallback chain runs, doubling
    the user-visible stall (#54465).

    Carve-out: a fast first-token fail (dead stream detected within the 60s
    no-progress window, zero output seen — see ``_timeout_message``) is cheap,
    so it keeps the normal same-provider retry; the provider is often fine
    and only that one stream was stillborn. Mid-stream stalls and hard-ceiling
    timeouts skip to fallback.
    """
    return (
        task in _TIMEOUT_NO_RETRY_TASKS
        and _is_timeout_error(exc)
        and "no-progress timeout" not in str(exc)
    )


def _evict_cached_clients(provider: str) -> None:
    """Drop cached auxiliary clients for a provider so fresh creds are used."""
    normalized = _normalize_aux_provider(provider)
    with _client_cache_lock:
        stale_keys = [
            key for key in _client_cache
            if _normalize_aux_provider(str(key[0])) == normalized
        ]
        for key in stale_keys:
            client = _client_cache.get(key, (None, None, None))[0]
            if client is not None:
                _close_cached_client(client)
            _client_cache.pop(key, None)


def _evict_cached_client_instance(target: Any) -> bool:
    """Drop the cache entry whose stored client (or its ``_real_client``) is *target*.

    Used when a cached client is poisoned (closed transport after a timeout).
    Async wrappers must expose the same ``_real_client`` as their sync sibling,
    or the async entry survives and keeps reusing the dead transport.
    Returns True when at least one entry was evicted.
    """
    if target is None:
        return False
    evicted = False
    with _client_cache_lock:
        for key in list(_client_cache.keys()):
            entry = _client_cache.get(key)
            if entry is None:
                continue
            cached = entry[0]
            if cached is None:
                continue
            real = getattr(cached, "_real_client", None)
            if cached is target or real is target:
                del _client_cache[key]
                evicted = True
    return evicted


def _pool_cache_hint(
    provider: str,
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a stable cache discriminator for pooled providers."""
    normalized = _normalize_aux_provider(provider)
    if normalized == "auto":
        runtime = _normalize_main_runtime(main_runtime)
        normalized = _normalize_aux_provider(runtime.get("provider") or _read_main_provider())
    if normalized in {"", "auto", "custom"}:
        return ""
    entry = _peek_pool_entry(normalized)
    if entry is None:
        return ""
    entry_id = str(getattr(entry, "id", "") or "").strip()
    if not entry_id:
        return ""
    return f"{normalized}:{entry_id}"


def _pool_error_context(exc: Exception) -> Dict[str, Any]:
    status = getattr(exc, "status_code", None)
    payload: Dict[str, Any] = {"message": str(exc)}
    if status is not None:
        payload["status_code"] = status
    return payload


def _recoverable_pool_provider(
    resolved_provider: str,
    client: Any,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Infer which provider pool can recover the current auxiliary client."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized not in {"", "auto", "custom"}:
        return normalized
    base = str(getattr(client, "base_url", "") or "")
    if base_url_host_matches(base, "chatgpt.com"):
        return "openai-codex"
    if base_url_host_matches(base, "openrouter.ai"):
        return "openrouter"
    if base_url_host_matches(base, "inference-api.nousresearch.com"):
        return "nous"
    if base_url_host_matches(base, "api.anthropic.com"):
        return "anthropic"
    if base_url_host_matches(base, "githubcopilot.com"):
        return "copilot"
    if base_url_host_matches(base, "api.kimi.com"):
        return "kimi-coding"
    if base_url_host_matches(base, "api.x.ai"):
        return "xai-oauth"
    # Providers outside the hardcoded list (e.g. opencode-go): match base URL
    # against registered api_key providers so pool rotation works for them too.
    if main_runtime:
        rt = _normalize_main_runtime(main_runtime)
        rt_provider = rt.get("provider", "")
        if rt_provider and rt_provider not in {"", "auto", "custom"}:
            try:
                from hermes_cli.auth import PROVIDER_REGISTRY
                pconfig = PROVIDER_REGISTRY.get(rt_provider)
                if pconfig and getattr(pconfig, "auth_type", None) == "api_key":
                    rt_base = str(getattr(pconfig, "inference_base_url", "") or "").rstrip("/")
                    if rt_base and base_url_host_matches(base, base_url_hostname(rt_base)):
                        return rt_provider
            except Exception:
                pass
    return None


def _recover_provider_pool(provider: str, exc: Exception, *, failed_api_key: str = "") -> bool:
    """Try same-provider credential-pool recovery for auxiliary calls.

    ``failed_api_key`` lets mark_exhausted_and_rotate identify the right pool
    entry even if another process already rotated (current() would be None).
    """
    normalized = _normalize_aux_provider(provider)
    try:
        pool = load_pool(normalized)
    except Exception as load_exc:
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc)
        return False
    if not pool or not pool.has_credentials():
        return False

    status_code = getattr(exc, "status_code", None)
    error_context = _pool_error_context(exc)
    hint = failed_api_key or None

    if _is_auth_error(exc):
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _evict_cached_clients(normalized)
            return True
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else 401,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
        return False

    if _is_payment_error(exc) or _is_rate_limit_error(exc):
        fallback_status = 402 if _is_payment_error(exc) else 429
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
    return False


def _prepare_same_provider_retry(
    *,
    task: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    main_runtime: Optional[Dict[str, Any]],
    final_model: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
    async_mode: bool,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild (client, request kwargs) for a same-provider retry after credential recovery."""
    if task == "vision":
        effective_provider, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider,
            model=final_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            async_mode=async_mode,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=async_mode,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
        )
        effective_provider = _effective_provider_for_client(
            retry_client, resolved_provider,
        )
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )

    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        effective_provider or resolved_provider,
        retry_model or final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=effective_timeout,
        extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=retry_base or resolved_base_url,
        task=task,
    )
    # Preserve per-request attribution headers (e.g. Copilot ``x-initiator``)
    # so the rebuilt-client retry doesn't lose capability gating.
    if extra_headers:
        retry_kwargs["extra_headers"] = dict(extra_headers)
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs["messages"] = _convert_openai_images_to_anthropic(retry_kwargs["messages"])
    return retry_client, retry_kwargs


def _retry_same_provider_sync(*, resolved_provider: str, resolved_api_mode: Optional[str], task: Optional[str], **prep) -> Any:
    retry_client, retry_kwargs = _prepare_same_provider_retry(
        task=task, resolved_provider=resolved_provider, resolved_api_mode=resolved_api_mode,
        async_mode=False, **prep,
    )
    return _validate_llm_response(
        _relay_sync_completion(
            retry_client, retry_kwargs, provider=resolved_provider, api_mode=resolved_api_mode,
        ),
        task,
    )


async def _retry_same_provider_async(*, resolved_provider: str, resolved_api_mode: Optional[str], task: Optional[str], **prep) -> Any:
    retry_client, retry_kwargs = _prepare_same_provider_retry(
        task=task, resolved_provider=resolved_provider, resolved_api_mode=resolved_api_mode,
        async_mode=True, **prep,
    )
    return _validate_llm_response(
        await _relay_async_completion(
            retry_client, retry_kwargs, provider=resolved_provider, api_mode=resolved_api_mode,
        ),
        task,
    )




def _refresh_provider_credentials(provider: str) -> bool:
    """Refresh short-lived credentials for OAuth-backed auxiliary providers."""
    normalized = _normalize_aux_provider(provider)
    try:
        if normalized == "copilot":
            from hermes_cli.copilot_auth import (
                _jwt_cache,
                _token_fingerprint,
                exchange_copilot_token,
                resolve_copilot_token,
            )

            raw_token, _source = resolve_copilot_token()
            if not str(raw_token or "").strip():
                return False
            _jwt_cache.pop(_token_fingerprint(raw_token), None)
            exchange_copilot_token(raw_token)
            _evict_cached_clients(normalized)
            return True
        if normalized == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=True)
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "nous":
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
                force_refresh=True,
            )
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "anthropic":
            from agent.anthropic_credentials import read_claude_code_credentials, _refresh_oauth_token, resolve_anthropic_token

            creds = read_claude_code_credentials()
            token = _refresh_oauth_token(creds) if isinstance(creds, dict) and creds.get("refreshToken") else None
            if not str(token or "").strip():
                token = resolve_anthropic_token()
            if not str(token or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "xai-oauth":
            # Prefer pool-level refresh, then the singleton auth-store resolver.
            pool = load_pool(normalized)
            if pool and pool.has_credentials():
                pool.select()
                refreshed = pool.try_refresh_current()
                if refreshed is not None and str(getattr(refreshed, "runtime_api_key", "") or "").strip():
                    _evict_cached_clients(normalized)
                    return True
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            creds = resolve_xai_oauth_runtime_credentials(force_refresh=True)
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
        if normalized == "vertex":
            # Mirrors run_agent's Vertex refresh. The cache key ignores the
            # rotating bearer token, so without eviction here a ~1h-expired
            # aux Vertex client 401s forever.
            from agent.vertex_adapter import get_vertex_config

            token, base_url = get_vertex_config()
            if not isinstance(token, str) or not token.strip():
                return False
            if not isinstance(base_url, str) or not base_url.strip():
                return False
            _evict_cached_clients(normalized)
            return True
    except Exception as exc:
        logger.debug("Auxiliary provider credential refresh failed for %s: %s", normalized, exc)
        return False
    return False


def _auth_refresh_provider_for_route(
    resolved_provider: Optional[str],
    client_base_url: str,
) -> str:
    """Return the provider whose short-lived credentials should be refreshed.

    Auto-routed calls keep ``resolved_provider == "auto"``; infer the backend
    from the client's base URL so refresh works for auto routes too.
    """
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized and normalized != "auto":
        return normalized
    if base_url_host_matches(client_base_url, "api.githubcopilot.com"):
        return "copilot"
    if base_url_host_matches(client_base_url, "chatgpt.com"):
        return "openai-codex"
    if base_url_host_matches(client_base_url, "api.anthropic.com"):
        return "anthropic"
    if base_url_host_matches(client_base_url, "inference-api.nousresearch.com"):
        return "nous"
    return normalized


def _fallback_chain_entry(task: Optional[str], fb_label: str) -> Optional[Dict[str, Any]]:
    """Resolve the ``fallback_chain`` entry a ``fallback_chain[<i>](<provider>)`` label points at.

    Returns ``None`` when the label is not a configured-chain candidate or the index no longer resolves.
    """
    if not task or not fb_label:
        return None
    m = re.match(r"fallback_chain\[(\d+)\]", fb_label)
    if not m:
        return None
    try:
        chain = _get_auxiliary_task_config(task).get("fallback_chain")
        entry = chain[int(m.group(1))] if isinstance(chain, list) else None
    except Exception:
        return None
    return entry if isinstance(entry, dict) else None


def _coerce_positive_timeout(raw: Any) -> Optional[float]:
    """Coerce a config ``timeout`` to a positive float, or None (rejects bools, which are ints)."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return None


def _fallback_entry_timeout(task: Optional[str], fb_label: str) -> Optional[float]:
    """Resolve a per-entry ``timeout`` for a configured fallback candidate.

    Fallbacks used to inherit the primary's deadline, which killed healthy but
    slower fallbacks. Returns ``None`` (caller keeps the task-level timeout)
    when the entry has no valid ``timeout``.
    """
    entry = _fallback_chain_entry(task, fb_label)
    raw = entry.get("timeout") if entry else None
    return _coerce_positive_timeout(raw)


def _fallback_provider_from_label(label: str) -> str:
    """Recover the provider identifier from a fallback display label."""
    match = re.match(
        r"(?:fallback_chain\[\d+\]|fallback_providers\[\d+\]|main-agent)\(([^)]+)\)$",
        label or "",
    )
    return match.group(1).strip() if match else str(label or "").strip()


class _FallbackDestination(NamedTuple):
    provider: str
    base_url: str
    api_mode: Optional[str]
    model: Optional[str]


def _complete_fallback_destination(
    provider: str,
    base_url: str,
    api_mode: Optional[str],
    model: Optional[str],
) -> _FallbackDestination:
    if not api_mode:
        if _endpoint_speaks_anthropic_messages(base_url):
            api_mode = "anthropic_messages"
        else:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider

                runtime = resolve_runtime_provider(
                    requested=provider,
                    explicit_base_url=base_url or None,
                    target_model=model or "",
                )
                api_mode = str(runtime.get("api_mode") or "").strip() or None
            except Exception:
                pass
    return _FallbackDestination(provider, base_url, api_mode, model)


def _fallback_destination_from_entry(
    entry: Dict[str, Any],
    fb_client: Any,
    fb_model: Optional[str],
) -> _FallbackDestination:
    provider = str(entry.get("provider") or "").strip()
    base_url = str(
        entry.get("base_url") or getattr(fb_client, "base_url", "") or ""
    ).strip()
    api_mode = str(
        entry.get("api_mode") or entry.get("transport") or ""
    ).strip() or None
    model = fb_model or str(entry.get("model") or "").strip() or None
    return _complete_fallback_destination(provider, base_url, api_mode, model)


def _fallback_destination(
    task: Optional[str],
    fb_client: Any,
    fb_model: Optional[str],
    fb_label: str,
) -> _FallbackDestination:
    """Return the resolved route identity used by a fallback request."""
    attached = getattr(fb_client, "_hermes_fallback_destination", None)
    if isinstance(attached, _FallbackDestination):
        return attached
    entry = _fallback_chain_entry(task, fb_label)
    if entry is not None:
        return _fallback_destination_from_entry(entry, fb_client, fb_model)
    return _complete_fallback_destination(
        _fallback_provider_from_label(fb_label),
        str(getattr(fb_client, "base_url", "") or ""),
        None,
        fb_model,
    )


def _replan_synchronous_cache_sections(
    messages: list,
    tools: Optional[list],
    *,
    destination: _FallbackDestination,
) -> tuple[list, list]:
    """Strip source decoration and plan one synchronous destination locally."""
    from agent.agent_runtime_helpers import (
        configured_cache_ttl,
        plan_cache_sections_for_destination,
    )

    return plan_cache_sections_for_destination(
        messages,
        tools,
        provider=destination.provider,
        base_url=destination.base_url,
        api_mode=destination.api_mode or "",
        model=destination.model or "",
        # Thread the operator's configured TTL so fallback requests don't regress
        # a configured 1h to the 5m default; no live agent here, so read config.
        cache_ttl=configured_cache_ttl(),
    )


def _fallback_request_kwargs(
    destination: _FallbackDestination,
    *,
    task: Optional[str],
    messages: list,
    tools: Optional[list],
    temperature: Optional[float],
    max_tokens: Optional[int],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
    fallback_entry: dict,
    task_config: dict,
    apply_fast_lane: bool,
) -> Dict[str, Any]:
    """Build request kwargs for one fallback destination (cache-section replan + fast-lane cap)."""
    fallback_max_tokens, fallback_extra_body = max_tokens, effective_extra_body
    if apply_fast_lane:
        fallback_max_tokens, fallback_extra_body = _compression_fast_lane_controls(
            task,
            actual_provider=destination.provider,
            actual_model=destination.model,
            requested_provider=fallback_entry.get("provider"),
            requested_model=fallback_entry.get("model"),
            route_config=fallback_entry,
            leak_guard_config=task_config,
            max_tokens=max_tokens,
            extra_body=effective_extra_body,
        )
    fallback_messages, fallback_tools = _replan_synchronous_cache_sections(
        messages,
        tools,
        destination=destination,
    )
    fb_kwargs = _build_call_kwargs(
        destination.provider, destination.model, fallback_messages,
        temperature=temperature, max_tokens=fallback_max_tokens,
        tools=fallback_tools, timeout=effective_timeout,
        extra_body=fallback_extra_body, reasoning_config=reasoning_config,
        base_url=destination.base_url, task=task)
    if apply_fast_lane and fallback_max_tokens is not None and max_tokens is None:
        fb_kwargs.update(
            auxiliary_max_tokens_param(fallback_max_tokens, model=destination.model)
        )
    return fb_kwargs


def _plan_fallback_candidate(
    fb_client: Any,
    fb_model: Optional[str],
    fb_label: str,
    *,
    task: Optional[str],
    effective_timeout: float,
    apply_fast_lane: bool,
    **request,
) -> Tuple[_FallbackDestination, Dict[str, Any], Callable[[str, Any, Optional[str]], Dict[str, Any]]]:
    """Resolve the destination + first-attempt kwargs for a fallback candidate.

    Returns ``(destination, kwargs, rebuild)`` where ``rebuild(provider, client, model)``
    produces kwargs for the credential-refreshed retry destination. A configured-chain
    entry's own ``timeout`` overrides ``effective_timeout``.
    """
    fb_timeout = _fallback_entry_timeout(task, fb_label)
    if fb_timeout is not None and fb_timeout != effective_timeout:
        logger.info(
            "Auxiliary %s: %s using its configured timeout %.0fs "
            "(task-level was %.0fs)",
            task or "call", fb_label, fb_timeout, effective_timeout,
        )
        effective_timeout = fb_timeout
    destination = _fallback_destination(task, fb_client, fb_model, fb_label)
    task_config = _get_auxiliary_task_config(task) if task == "compression" else {}
    fallback_entry = _fallback_chain_entry(task, fb_label) or {}
    common = dict(
        task=task, effective_timeout=effective_timeout,
        fallback_entry=fallback_entry, task_config=task_config,
        apply_fast_lane=apply_fast_lane, **request,
    )

    def _rebuild(provider: str, client: Any, model: Optional[str]) -> Tuple[_FallbackDestination, Dict[str, Any]]:
        retry_destination = _FallbackDestination(
            provider,
            destination.base_url or str(getattr(client, "base_url", "") or ""),
            destination.api_mode,
            model or destination.model,
        )
        return retry_destination, _fallback_request_kwargs(retry_destination, **common)

    return destination, _fallback_request_kwargs(destination, **common), _rebuild


def _quarantine_fallback_candidate(task: Optional[str], fb_label: str, fb_provider: str, fb_err: Exception, *, tag: str = "") -> None:
    """Refresh unavailable or still 401s: token is dead. Quarantine the candidate so the caller moves on."""
    _mark_provider_unhealthy(fb_provider or fb_label)
    logger.warning(
        "Auxiliary %s%s: fallback candidate %s has a stale/unrefreshable "
        "credential (%s) — skipping to next fallback",
        task or "call", tag, fb_label, fb_err,
    )


def _call_fallback_candidate_sync(
    fb_client: Any,
    fb_model: Optional[str],
    fb_label: str,
    *,
    task: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
) -> Optional[Any]:
    """Call one fallback candidate with stale-credential recovery.

    On an auth error: refresh the candidate's credentials and retry once with a
    rebuilt client; if that also auth-fails, mark the provider unhealthy and
    return ``None`` so the caller moves to the next layer instead of aborting
    the task. Non-auth errors raise.
    """
    destination, fb_kwargs, rebuild = _plan_fallback_candidate(
        fb_client, fb_model, fb_label, task=task, effective_timeout=effective_timeout,
        apply_fast_lane=True, messages=messages, tools=tools, temperature=temperature,
        max_tokens=max_tokens, effective_extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
    )

    def _send(client: Any, request_kwargs: Dict[str, Any], dest: _FallbackDestination) -> Any:
        return _validate_llm_response(
            _relay_sync_completion(
                client,
                request_kwargs,
                provider=dest.provider,
                api_mode=dest.api_mode,
                create=lambda request: _create_with_progress(
                    client,
                    request,
                    task,
                    force_stream=_provider_requires_stream(dest.provider, dest.base_url),
                ),
            ),
            task,
        )

    try:
        return _send(fb_client, fb_kwargs, destination)
    except Exception as fb_err:
        if not _is_auth_error(fb_err):
            raise
        fb_provider = _auth_refresh_provider_for_route(
            destination.provider, destination.base_url
        )
        if fb_provider not in {"auto", "", None} and _refresh_provider_credentials(fb_provider):
            retry_client, retry_model = _get_cached_client(
                fb_provider,
                destination.model,
                base_url=destination.base_url or None,
                api_mode=destination.api_mode,
            )
            if retry_client is not None:
                retry_destination, retry_kwargs = rebuild(fb_provider, retry_client, retry_model)
                try:
                    return _send(retry_client, retry_kwargs, retry_destination)
                except Exception as retry_err:
                    if not _is_auth_error(retry_err):
                        raise
        _quarantine_fallback_candidate(task, fb_label, fb_provider, fb_err)
        return None


async def _call_fallback_candidate_async(
    fb_client: Any,
    fb_model: Optional[str],
    fb_label: str,
    *,
    task: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
) -> Optional[Any]:
    """Async mirror of :func:`_call_fallback_candidate_sync` (no fast-lane cap on this wire)."""
    destination, fb_kwargs, rebuild = _plan_fallback_candidate(
        fb_client, fb_model, fb_label, task=task, effective_timeout=effective_timeout,
        apply_fast_lane=False, messages=messages, tools=tools, temperature=temperature,
        max_tokens=max_tokens, effective_extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
    )

    async def _send(client: Any, request_kwargs: Dict[str, Any], dest: _FallbackDestination) -> Any:
        return _validate_llm_response(
            await _relay_async_completion(
                client,
                request_kwargs,
                provider=dest.provider,
                api_mode=dest.api_mode,
            ),
            task,
        )

    try:
        return await _send(fb_client, fb_kwargs, destination)
    except Exception as fb_err:
        if not _is_auth_error(fb_err):
            raise
        fb_provider = _auth_refresh_provider_for_route(
            destination.provider, destination.base_url
        )
        if fb_provider not in {"auto", "", None} and _refresh_provider_credentials(fb_provider):
            retry_client, retry_model = _get_cached_client(
                fb_provider,
                destination.model,
                async_mode=True,
                base_url=destination.base_url or None,
                api_mode=destination.api_mode,
            )
            if retry_client is not None:
                retry_destination, retry_kwargs = rebuild(fb_provider, retry_client, retry_model)
                try:
                    return await _send(retry_client, retry_kwargs, retry_destination)
                except Exception as retry_err:
                    if not _is_auth_error(retry_err):
                        raise
        _quarantine_fallback_candidate(task, fb_label, fb_provider, fb_err, tag=" (async)")
        return None




def _try_payment_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "payment error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the auto-detection chain after a payment/credit or connection error, skipping the failed provider.

    Returns (client, model, provider_label) or (None, None, "").
    """
    skip = failed_provider.lower().strip()
    # Also skip the main-provider path if it maps to the same backend.
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    skip_chain_labels = {_normalize_chain_label(s) for s in skip_labels}

    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — falling back to %s (%s)",
                task or "call", reason, failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: %s on %s and no fallback available (tried: %s)",
        task or "call", reason, failed_provider, ", ".join(tried),
    )
    return None, None, ""


def _try_main_agent_model_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "error",
    failed_model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Last-resort fallback to the main agent provider + model after the configured chain is exhausted.

    ``failed_model`` narrows the skip to the exact (provider, model) pair (model-scoped failures:
    timeout/connection/rate-limit); None keeps the whole-provider skip (auth/payment failures, where
    shared credentials are broken). Same-URL custom endpoints serve many models, so a hung aux model
    says nothing about the main model's health. Returns (client, model, label) or (None, None, "").
    """
    main_provider = (_read_main_provider() or "").strip()
    main_model = (_read_main_model() or "").strip()
    if main_provider.lower() == "moa":
        # MoA virtual provider: fall back to the preset's aggregator — the
        # acting model — instead of the unreachable "moa"/<preset-name> pair.
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if not _agg_provider or not _agg_model:
            return None, None, ""
        main_provider, main_model = _agg_provider, _agg_model
    if not main_provider or not main_model or main_provider.lower() in {"auto", ""}:
        return None, None, ""

    # Scope semantics owned by agent.backend_identity: model-scoped failures skip
    # only the exact deployment; provider-wide (no failed_model) skip the credential surface.
    from agent.backend_identity import (
        BackendIdentity,
        FailureScope,
        should_skip_candidate,
    )

    skip_model = (failed_model or "").strip().lower() or None
    if should_skip_candidate(
        BackendIdentity.build(provider=main_provider, model=main_model),
        BackendIdentity.build(provider=failed_provider, model=skip_model),
        FailureScope.MODEL if skip_model else FailureScope.CREDENTIAL,
    ):
        return None, None, ""
    if _is_provider_unhealthy(main_provider):
        _log_skip_unhealthy(main_provider, task)
        return None, None, ""

    try:
        client, resolved_model = resolve_provider_client(
            provider=main_provider, model=main_model,
        )
    except Exception:
        client, resolved_model = None, None

    if client is None:
        return None, None, ""

    label = f"main-agent({main_provider})"
    logger.info(
        "Auxiliary %s: %s on %s — falling back to main agent model %s (%s)",
        task or "call", reason, failed_provider, label, resolved_model or main_model,
    )
    return client, resolved_model or main_model, label


# ── Context-window screening for runtime fallback chains ──
#
# The startup feasibility check filters too-small aux models, but the runtime
# fallback chains did not, so compression could stop at a reachable-but-too-small
# candidate. Helpers below screen by context window; ``None`` (unknown) passes through.

def _task_minimum_context_length(task: Optional[str]) -> Optional[int]:
    """Minimum context length for an auxiliary task; None = no floor.

    Only ``compression`` has one (the same MINIMUM_CONTEXT_LENGTH floor the startup feasibility
    check enforces); other tasks return None so the runtime chain stays permissive for them.
    """
    if not task:
        return None
    if task == "compression":
        return MINIMUM_CONTEXT_LENGTH
    return None


def _candidate_context_window(
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> Optional[int]:
    """Best-effort context window for a fallback candidate; ``None`` = unknown (never raises).

    Callers treat ``None`` as pass-through so custom/unregistered models keep their fallback surface.
    """
    if not model:
        return None
    try:
        ctx = get_model_context_length(
            model,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )
    except Exception as exc:
        logger.debug(
            "Auxiliary fallback: could not resolve context window for %s/%s: %s",
            provider, model, exc,
        )
        return None
    # Propagate None explicitly in case the resolver ever returns Optional[int].
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return None


def _context_too_small(
    entry: Dict[str, Any],
    provider: str,
    model: str,
    min_ctx: Optional[int],
    *,
    task: Optional[str],
    label: str,
    name_model: bool = False,
) -> Optional[str]:
    """Screen one fallback candidate by context window; returns the ``tried`` note when it is too small."""
    if min_ctx is None:
        return None
    fb_ctx = _candidate_context_window(
        provider,
        model,
        base_url=str(entry.get("base_url") or ""),
        api_key=_fallback_entry_api_key(entry) or "",
    )
    if fb_ctx is None or fb_ctx >= min_ctx:
        return None
    if name_model:
        logger.info(
            "Auxiliary %s: skipping %s (%s context=%d < min=%d), continuing chain",
            task, label, model, fb_ctx, min_ctx,
        )
    else:
        logger.info(
            "Auxiliary %s: skipping %s (context=%d < min=%d), continuing chain",
            task or "call", label, fb_ctx, min_ctx,
        )
    return f"{label} (context too small: {fb_ctx}<{min_ctx})"


def _try_configured_fallback_chain(
    task: str,
    failed_provider: str,
    reason: str = "error",
    failed_model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try auxiliary.<task>.fallback_chain entries in order (each needs ``provider``; model/base_url/api_key optional).

    ``failed_model`` narrows the skip to the exact (provider, model) pair so sibling models on the
    same provider still run after a model-scoped failure (timeout/connection/rate-limit). None keeps
    the whole provider skipped (auth/payment: shared credentials are broken).
    Returns (client, model, provider_label) or (None, None, "").
    """
    if not task:
        return None, None, ""

    task_config = _get_auxiliary_task_config(task)
    chain = task_config.get("fallback_chain")
    if not chain or not isinstance(chain, list):
        return None, None, ""

    skip_model = (failed_model or "").strip().lower() or None
    # Scope semantics owned by agent.backend_identity: failed_model → model-scoped
    # (exact deployment skipped); none → provider-wide (credential surface skipped).
    from agent.backend_identity import (
        BackendIdentity,
        FailureScope,
        should_skip_candidate,
    )

    failed_ident = BackendIdentity.build(
        provider=failed_provider, model=skip_model,
    )
    failure_scope = (
        FailureScope.MODEL if skip_model else FailureScope.CREDENTIAL
    )
    tried = []
    min_ctx = _task_minimum_context_length(task)

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider", "")).strip()
        if not fb_provider:
            continue
        fb_model_raw = str(entry.get("model", "")).strip()
        if should_skip_candidate(
            BackendIdentity.build(
                provider=fb_provider,
                model=fb_model_raw,
                base_url=str(entry.get("base_url") or ""),
            ),
            failed_ident,
            failure_scope,
        ):
            continue
        fb_model = fb_model_raw or None

        label = f"fallback_chain[{i}]({fb_provider})"

        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception:
            fb_client, resolved_model = None, None

        if fb_client is not None:
            too_small = _context_too_small(
                entry, fb_provider, resolved_model, min_ctx, task=task, label=label, name_model=True,
            ) if resolved_model else None
            if too_small:
                tried.append(too_small)
                continue
            logger.info(
                "Auxiliary %s: %s on %s — configured fallback to %s (%s)",
                task, reason, failed_provider, label, resolved_model or fb_model or "default",
            )
            return fb_client, resolved_model or fb_model, label
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: configured fallback_chain exhausted (tried: %s)",
            task, ", ".join(tried),
        )
    return None, None, ""


def _try_configured_fallback_for_unavailable_client(
    task: Optional[str],
    failed_provider: str,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the task fallback_chain when an explicit aux provider cannot build a client (no key/OAuth/pool creds).

    Deliberately stops at the per-task chain; the main-agent model stays the runtime last resort.
    """
    explicit = (failed_provider or "").strip().lower()
    if not task or not explicit or explicit in {"auto"}:
        return None, None, ""
    return _try_configured_fallback_chain(
        task,
        explicit,
        reason="provider unavailable",
    )


def _fallback_entry_api_key(entry: Dict[str, Any]) -> Optional[str]:
    """Resolve inline or env-backed API key via the secret-scope-aware resolver (no raw os.getenv under multiplexing)."""
    from hermes_cli.fallback_config import resolve_entry_api_key

    return resolve_entry_api_key(entry)


def _resolve_fallback_entry(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve one fallback entry through the central provider router."""
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip() or None
    if not provider or not model:
        return None, None
    base_url = str(entry.get("base_url") or "").strip() or None
    api_key = _fallback_entry_api_key(entry)
    api_mode = str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    client, resolved_model = resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
    )
    if client is not None:
        try:
            client._hermes_fallback_destination = _fallback_destination_from_entry(
                entry, client, resolved_model
            )
        except Exception:
            pass
    return client, resolved_model


def _try_main_fallback_chain(
    task: Optional[str],
    failed_provider: str = "",
    reason: str = "error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the top-level main-agent fallback chain for a ``provider: auto`` auxiliary call.

    Auto tasks should honour the user's declared main fallback policy before Hermes' built-in
    discovery chain. Read via ``get_fallback_chain`` so ``fallback_providers`` and legacy
    ``fallback_model`` participate in the same order as the main agent.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.fallback_config import get_fallback_chain

        chain = get_fallback_chain(load_config_readonly())
    except Exception as exc:
        logger.debug("Auxiliary %s: could not load main fallback chain: %s", task or "call", exc)
        return None, None, ""

    if not chain:
        return None, None, ""

    failed_norm = (failed_provider or "").strip().lower()
    main_norm = (_read_main_provider() or "").strip().lower()
    skip = {p for p in (failed_norm, main_norm, "auto") if p}
    tried: List[str] = []
    min_ctx = _task_minimum_context_length(task)

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider") or "").strip()
        fb_model = str(entry.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        fb_norm = fb_provider.lower()
        label = f"fallback_providers[{i}]({fb_provider})"
        if fb_norm in skip:
            tried.append(f"{label} (skipped)")
            continue
        if _is_provider_unhealthy(fb_norm):
            _log_skip_unhealthy(fb_norm, task)
            tried.append(f"{label} (unhealthy)")
            continue
        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception as exc:
            logger.debug("Auxiliary %s: main fallback %s failed to resolve: %s", task or "call", label, exc)
            fb_client, resolved_model = None, None
        if fb_client is not None:
            too_small = _context_too_small(
                entry, fb_provider, resolved_model or fb_model, min_ctx, task=task, label=label,
            )
            if too_small:
                tried.append(too_small)
                continue
            logger.info(
                "Auxiliary %s: %s on %s — main fallback chain to %s (%s)",
                task or "call", reason, failed_provider or "auto", label,
                resolved_model or fb_model,
            )
            return fb_client, resolved_model or fb_model, fb_provider
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: main fallback chain exhausted (tried: %s)",
            task or "call", ", ".join(tried),
        )
    return None, None, ""


def _resolve_auto_route(
    main_runtime: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
) -> Tuple[Optional[OpenAI], Optional[str], str]:
    """Full auto-detection chain, including the selected provider identity.

    Priority: (1) main provider + main model, regardless of provider type, so aux tasks stay on the
    model the user chose; (2) configured fallback policy; (3) OpenRouter → Nous → custom → Codex →
    API-key providers, only when the main provider has no working client.
    """
    global auxiliary_is_nous, _stale_base_url_warned
    auxiliary_is_nous = False  # Reset — _try_nous() will set True if it wins
    runtime = _normalize_main_runtime(main_runtime)
    runtime_provider = runtime.get("provider", "")
    runtime_model = str(runtime.get("model") or "")
    runtime_base_url = str(runtime.get("base_url") or "")
    runtime_api_key = runtime.get("api_key", "")
    runtime_api_mode = str(runtime.get("api_mode") or "")

    # ── Warn once if OPENAI_BASE_URL is set but config.yaml uses a named provider:
    #    a stale OPENAI_BASE_URL in ~/.hermes/.env after `hermes model` poisons routing. ──
    if not _stale_base_url_warned:
        _env_base = os.getenv("OPENAI_BASE_URL", "").strip()
        _cfg_provider = runtime_provider or _read_main_provider()
        if (_env_base and _cfg_provider
                and _cfg_provider != "custom"
                and not _cfg_provider.startswith("custom:")):
            logger.warning(
                "OPENAI_BASE_URL is set (%s) but model.provider is '%s'. "
                "Auxiliary clients may route to the wrong endpoint. "
                "Run: hermes model to reconfigure, or remove "
                "OPENAI_BASE_URL from ~/.hermes/.env",
                _env_base, _cfg_provider,
            )
            _stale_base_url_warned = True

    # ── Step 1: main provider + main model → use them directly ──
    # "auto" means "use my main chat model for side tasks too", including aggregator users.
    # Explicit per-task overrides (auxiliary.<task>.provider) still win.
    main_provider = str(runtime_provider or _read_main_provider() or "")
    main_model = str(runtime_model or _read_main_model() or "")

    # Latency-critical tasks may opt in to the provider's fast model. Titling is the only eligible
    # task (~8 tokens naming a sidebar row; seconds on a reasoning model). Opt-in only, because
    # every settings surface defines "auto" as the main model — overriding silently makes it cosmetic.
    if _task_prefers_fast_model(task) and main_provider and main_provider not in {"auto", ""}:
        fast_model = _get_aux_model_for_provider(main_provider, prefer_fast=True)
        if fast_model and fast_model != main_model:
            logger.debug(
                "Auxiliary task %s: preferring fast model %s over main model %s",
                task, fast_model, main_model,
            )
            main_model = fast_model

    # MoA virtual provider: "model" is a preset name with no real HTTP endpoint (provider 400s on it).
    # Aux tasks don't need the fan-out — run on the aggregator (the preset's acting model).
    if main_provider == "moa":
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if _agg_provider and _agg_model:
            main_provider = _agg_provider
            main_model = _agg_model
            # Drop the facade's "moa://local" base_url / placeholder key so the
            # aggregator resolves through its own provider credentials.
            runtime_base_url = ""
            runtime_api_key = ""
            runtime_api_mode = ""

    if (main_provider and main_model
            and main_provider not in {"auto", ""}):
        resolved_provider = main_provider
        explicit_base_url = runtime_base_url or None
        explicit_api_key = None
        if runtime_base_url and main_provider == "custom":
            # Anonymous custom endpoint — pass through explicit base_url + api_key.
            resolved_provider = "custom"
            explicit_base_url = runtime_base_url
            explicit_api_key = runtime_api_key or None
        elif main_provider.startswith("custom:"):
            # Named custom provider (custom_providers / providers dict entry).
            _has_named_entry = False
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider
                _has_named_entry = _get_named_custom_provider(main_provider) is not None
            except ImportError:
                pass
            if _has_named_entry:
                # KEEP the full ``custom:<name>`` so resolve_provider_client hits the named arm,
                # which honours the entry's api_mode (e.g. anthropic_messages). Collapsing to plain
                # "custom" strips /anthropic and routes via chat.completions (404s on some proxies).
                # base_url/api_key come from the entry, so leave explicit_* unset.
                resolved_provider = main_provider
                explicit_base_url = None
            elif runtime_base_url:
                # Config-less named custom provider (exists only in live runtime):
                # collapse to the anonymous custom arm with the runtime endpoint + key.
                resolved_provider = "custom"
                explicit_base_url = runtime_base_url
                explicit_api_key = runtime_api_key or None
            elif runtime_api_key:
                explicit_api_key = runtime_api_key
        elif runtime_api_key:
            # Pin aux to the main session's working key instead of re-selecting
            # from the pool (which might pick an exhausted key).
            explicit_api_key = runtime_api_key
        # Skip Step-1 if the main provider was recently 402'd; the unhealthy TTL bounds
        # the bypass so a topped-up account recovers. Avoids one doomed 402 RTT per aux call.
        main_chain_label = _normalize_chain_label(resolved_provider)
        if main_chain_label and _is_provider_unhealthy(main_chain_label):
            _log_skip_unhealthy(main_chain_label)
        else:
            client, resolved = resolve_provider_client(
                resolved_provider,
                main_model,
                explicit_base_url=explicit_base_url,
                explicit_api_key=explicit_api_key,
                api_mode=runtime_api_mode or None,
            )
            if client is not None:
                logger.info("Auxiliary auto-detect: using main provider %s (%s)",
                            main_provider, resolved or main_model)
                return client, resolved or main_model, resolved_provider

    # ── Step 2: user-configured fallback policy ─────────────────────────
    # Task-specific chain first, then the main agent's top-level fallback chain;
    # the hardcoded discovery chain below is only the default for users with no policy.
    if task:
        fb_client, fb_model, fb_label = _try_configured_fallback_chain(
            task, main_provider or "auto", reason="main provider unavailable")
        if fb_client is not None:
            return fb_client, fb_model, _fallback_provider_from_label(fb_label)
    fb_client, fb_model, fb_label = _try_main_fallback_chain(
        task, main_provider or "auto", reason="main provider unavailable")
    if fb_client is not None:
        return fb_client, fb_model, fb_label

    # ── Step 3: aggregator / fallback chain ──────────────────────────────
    tried = []
    for label, try_fn in _get_provider_chain():
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info("Auxiliary auto-detect: using %s (%s) — skipped: %s",
                            label, model or "default", ", ".join(tried))
            else:
                logger.info("Auxiliary auto-detect: using %s (%s)", label, model or "default")
            return client, model, label
        tried.append(label)
    logger.warning("Auxiliary auto-detect: no provider available (tried: %s). "
                   "Compression, summarization, and memory flush will not work. "
                   "Set OPENROUTER_API_KEY or configure a local model in config.yaml.",
                   ", ".join(tried))
    return None, None, ""


def _resolve_auto(
    main_runtime: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Backward-compatible auto resolver for callers that only need client/model."""
    client, model, _provider = _resolve_auto_route(main_runtime=main_runtime, task=task)
    return client, model


def _tag_effective_provider(client: Any, provider: str) -> None:
    """Retain auto-routing identity on the client that survives cache reuse."""
    if client is None or not provider:
        return
    try:
        setattr(client, "_hermes_aux_effective_provider", provider)
    except (AttributeError, TypeError):
        logger.debug(
            "Auxiliary client %s cannot retain effective provider %s",
            type(client).__name__, provider,
        )


def _effective_provider_for_client(client: Any, fallback: str) -> str:
    """Return the concrete provider selected for an auto-routed client."""
    effective_provider = getattr(client, "_hermes_aux_effective_provider", "")
    if isinstance(effective_provider, str) and effective_provider:
        return effective_provider
    return str(fallback or "")


# ── Centralized Provider Router ─────────────────────────────────────────────
#
# resolve_provider_client() is the single entry point for building a configured client
# from a (provider, model) pair: auth, base URL, headers, API format (Chat vs Responses).
# Consumers must go through it or the public helpers below — never read auth env vars ad-hoc.


def _to_async_client(sync_client, model: str, is_vision: bool = False):
    """Convert a sync client to its async counterpart, preserving Codex routing.

    ``is_vision=True`` on Copilot adds the ``Copilot-Vision-Request`` header (vision payloads time out without it).
    """
    from openai import AsyncOpenAI

    if isinstance(sync_client, _AuxProbeClientStub):
        return sync_client, model
    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return AsyncAnthropicAuxiliaryClient(sync_client), model
    if isinstance(sync_client, BedrockAuxiliaryClient):
        return AsyncBedrockAuxiliaryClient(sync_client), model
    try:
        from agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient

        if isinstance(sync_client, GeminiNativeClient):
            return AsyncGeminiNativeClient(sync_client), model
    except ImportError:
        pass
    # Clients that are already usable from async code (the ACP shims drive a
    # subprocess, not an HTTP connection pool) opt out of the async wrapper.
    if _client_declares(sync_client, "HERMES_SKIP_ASYNC_WRAP"):
        return sync_client, model

    sync_base_url = str(sync_client.base_url)
    async_kwargs = {"api_key": sync_client.api_key, "base_url": sync_base_url}
    if base_url_host_matches(sync_base_url, "openrouter.ai"):
        headers = _apply_user_default_headers(build_or_headers())
    elif _is_official_codex_base_url(sync_base_url):
        headers = _apply_user_default_headers(
            _codex_cloudflare_headers(sync_client.api_key, base_url=sync_base_url)
        )
    else:
        # Provider for the profile-header fallback is inferred from the hostname.
        try:
            from agent.model_metadata import _infer_provider_from_url
            inferred = _infer_provider_from_url(sync_base_url) or ""
        except Exception:
            inferred = ""
        headers = _endpoint_default_headers(sync_base_url, inferred, is_vision=is_vision, xai=True)
    if headers:
        async_kwargs["default_headers"] = headers
    _apply_required_codex_headers(
        async_kwargs, access_token=sync_client.api_key, base_url=sync_base_url,
    )
    async_kwargs = {
        **_openai_http_client_kwargs(sync_base_url, async_mode=True),
        **async_kwargs,
    }
    # Hermes owns the auxiliary retry/timeout budget; disable SDK-internal retries.
    async_kwargs.setdefault("max_retries", 0)
    return AsyncOpenAI(**async_kwargs), model


def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name


def _named_custom_api_key(custom_entry: Dict[str, Any], provider: str, custom_base: str) -> Any:
    """Credential for a named custom provider: inline api_key → key_env → key_cmd → credential pool → placeholder.

    Aux resolves named custom providers here, not via _resolve_named_custom_runtime,
    so key_cmd must be honoured at the same precedence or every aux call 401s.
    """
    custom_key: Any = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
    if not custom_key and custom_key_env:
        custom_key = _scoped_key_env(custom_key_env)
    custom_key_cmd = str(custom_entry.get("key_cmd", "") or "").strip()
    if custom_key_cmd:
        from agent.command_token_source import build_command_token_provider
        custom_key = build_command_token_provider(
            custom_key_cmd, custom_entry.get("name") or provider
        ) or custom_key
    if not custom_key:
        try:
            from agent.credential_pool import custom_provider_pool_key_candidates

            pool_name = custom_entry.get("provider_key") or custom_entry.get("name") or provider
            for pool_key in custom_provider_pool_key_candidates(custom_base, pool_name):
                try:
                    pool = load_pool(pool_key)
                except Exception:
                    continue
                if not pool.has_credentials():
                    continue
                pool_entry = pool.select()
                if pool_entry is None:
                    continue
                pool_api_key = (
                    getattr(pool_entry, "runtime_api_key", None)
                    or getattr(pool_entry, "access_token", "")
                    or ""
                )
                if str(pool_api_key).strip():
                    custom_key = str(pool_api_key).strip()
                    break
        except Exception:
            pass
    return custom_key or "no-key-required"


def _build_bedrock_client(provider: str, model: Optional[str], *, raw_codex: bool) -> Tuple[Optional[Any], Optional[str]]:
    """AWS Bedrock: Claude → Anthropic Bedrock SDK (prompt caching, thinking); OpenAI models
    (GPT-5.5/5.6) → Bedrock Mantle's OpenAI Responses endpoint; everything else → Converse API."""
    try:
        from agent.bedrock_adapter import (
            has_aws_credentials,
            is_anthropic_bedrock_model,
            resolve_bedrock_runtime_region,
            is_openai_bedrock_model,
            bedrock_openai_base_url,
            resolve_bedrock_bearer_token,
            configure_bedrock_openai_client_kwargs,
        )
        from agent.anthropic_adapter import build_anthropic_bedrock_client
    except ImportError:
        logger.warning("resolve_provider_client: bedrock requested but "
                       "boto3, httpx/openai, or anthropic SDK not installed")
        return None, None

    if not has_aws_credentials():
        logger.debug("resolve_provider_client: bedrock requested but "
                     "no AWS credentials found")
        return None, None

    # Region must match the main runtime's resolution (bedrock.region in config first, then
    # env/profile) so aux calls never leave the primary runtime's configured region.
    region = resolve_bedrock_runtime_region()
    default_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    final_model = _normalize_resolved_model(model or default_model, provider) or default_model

    if is_openai_bedrock_model(final_model):
        # Module-level lazy ``OpenAI`` proxy on purpose so tests can patch("agent.auxiliary_client.OpenAI").
        client_kwargs: Dict[str, Any] = {
            "api_key": resolve_bedrock_bearer_token() or "aws-sdk",
            "base_url": bedrock_openai_base_url(region),
        }
        configure_bedrock_openai_client_kwargs(client_kwargs)
        client = OpenAI(**client_kwargs)
        logger.debug("resolve_provider_client: bedrock-openai (%s, %s)", final_model, region)
        if raw_codex:
            return client, final_model
        return CodexAuxiliaryClient(client, final_model), final_model

    base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
    if is_anthropic_bedrock_model(final_model):
        try:
            real_client = build_anthropic_bedrock_client(region)
        except ImportError as exc:
            logger.warning("resolve_provider_client: cannot create Bedrock "
                           "client: %s", exc)
            return None, None
        client = AnthropicAuxiliaryClient(
            real_client, final_model, api_key="aws-sdk",
            base_url=base_url,
        )
        logger.debug("resolve_provider_client: bedrock anthropic (%s, %s)",
                     final_model, region)
    else:
        client = BedrockAuxiliaryClient(region, final_model)
        logger.debug("resolve_provider_client: bedrock converse (%s, %s)",
                     final_model, region)
    return client, final_model


def _build_vertex_client(provider: str, model: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Google Vertex AI: Gemini via the OpenAI-compatible endpoint with an OAuth2 bearer (standard OpenAI client)."""
    try:
        from agent.vertex_adapter import get_vertex_config, has_vertex_credentials
    except ImportError:
        logger.warning("resolve_provider_client: vertex requested but "
                       "google-auth not installed")
        return None, None

    if not has_vertex_credentials():
        logger.debug("resolve_provider_client: vertex requested but "
                     "no GCP credentials found")
        return None, None

    token, base_url = get_vertex_config()
    if not token or not base_url:
        logger.warning("resolve_provider_client: vertex requested but "
                       "could not mint token / resolve project")
        return None, None

    final_model = _normalize_resolved_model(model or "google/gemini-3-flash-preview", provider)
    try:
        # Aliased import: a bare `from openai import OpenAI` would shadow the module-level lazy proxy.
        from openai import OpenAI as _VertexOpenAI
        client = _VertexOpenAI(api_key=token, base_url=base_url)
    except Exception as exc:
        logger.warning("resolve_provider_client: cannot create Vertex "
                       "client: %s", exc)
        return None, None
    logger.debug("resolve_provider_client: vertex (%s)", final_model)
    return client, final_model


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: return a configured client (auth, base URL, API format) for a provider + optional model.

    The client always exposes ``.chat.completions.create()``; Codex/Responses providers get an adapter.
    ``provider`` accepts built-in names, ``custom:<name>``, "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
    or "auto" (full auto-detection chain). ``model=None`` uses the provider's default aux model.
    ``raw_codex`` returns the bare OpenAI client for callers needing ``responses.stream()``.
    ``api_mode`` forces "codex_responses"/"chat_completions"/"anthropic_messages" instead of auto-detect.
    Returns (client, resolved_model) or (None, None) if auth is unavailable.
    """
    _validate_proxy_env_urls()
    # Keep the pre-alias name so a custom_providers entry named like a built-in
    # alias (e.g. "kimi" → "kimi-coding") is still reachable via the named-custom branch.
    original_provider = (provider or "").strip().lower()
    provider = _normalize_aux_provider(provider)

    # MoA chokepoint: "moa" is not an HTTP provider; direct callers (vision auto-detect,
    # main-agent fallback, plugins) would dead-end in the unknown-provider branch. If the
    # preset can't be resolved, leave the call untouched for the normal missing-provider diagnostic.
    if provider == "moa":
        _agg_provider, _agg_model = _resolve_moa_aggregator(model)
        if _agg_provider and _agg_model:
            original_provider = _agg_provider.strip().lower()
            provider = _normalize_aux_provider(_agg_provider)
            model = _agg_model
            # The moa:// facade endpoint/key belong to the virtual runtime, not the aggregator.
            if explicit_base_url and str(explicit_base_url).lower().startswith("moa://"):
                explicit_base_url = None
                explicit_api_key = None

    # Model resolution for concrete providers: caller ``model`` → provider catalog default
    # (empty for OAuth-gated providers whose model lists drift, so no default can rot) → main model
    # from config (MoA main → aggregator model, since a preset NAME is never a wire model id).
    # The main-model step is load-bearing for OAuth providers: aux tasks run on the configured
    # model instead of silently dropping to the Step-2 fallback. Every branch below thus sees a
    # non-empty ``model`` whenever anything is configured; with nothing configured the branches
    # still hit their missing-credentials returns and _resolve_auto falls through to Step 2.
    # ``auto`` is excluded: pre-filling from the process-global main model can pair a stale slug
    # with the provider _resolve_auto actually selects (e.g. Claude slug sent to Codex).
    # Nous + vision is also excluded: its model comes from the Portal's tier-aware vision
    # recommendation, and a pre-filled text-only chat model would win and 404.
    _nous_portal_vision = provider == "nous" and is_vision
    if not model and provider != "auto" and not _nous_portal_vision:
        model = _get_aux_model_for_provider(provider) or _read_main_model_for_aux() or model

    def _needs_codex_wrap(client_obj, base_url_str: str, model_str: str) -> bool:
        """True if a plain OpenAI client needs the Responses API wrapper (explicit api_mode or api.openai.com + codex model)."""
        if isinstance(client_obj, CodexAuxiliaryClient):
            return False
        if raw_codex:
            return False
        if provider == "actual":
            return True
        if api_mode == "codex_responses":
            return True
        if api_mode and api_mode != "codex_responses":
            return False  # explicit non-codex mode
        if base_url_hostname(base_url_str) == "api.openai.com":
            model_lower = (model_str or "").lower()
            if "codex" in model_lower:
                return True
        return False

    def _wrap_if_needed(client_obj, final_model_str: str, base_url_str: str = "",
                        api_key_str: str = ""):
        """Wrap a plain OpenAI client in the right transport adapter; specialized wrappers pass through.

        Codex (Responses API): explicit ``api_mode=codex_responses`` or api.openai.com + codex model.
        Anthropic (Messages): ``api_mode=anthropic_messages``, any ``/anthropic`` suffix,
        ``api.kimi.com/coding``, or ``api.anthropic.com``.
        """
        if _needs_codex_wrap(client_obj, base_url_str, final_model_str):
            logger.debug(
                "resolve_provider_client: wrapping client in CodexAuxiliaryClient "
                "(api_mode=%s, model=%s, base_url=%s)",
                api_mode or "auto-detected", final_model_str,
                base_url_str[:60] if base_url_str else "")
            return CodexAuxiliaryClient(client_obj, final_model_str)
        return _maybe_wrap_anthropic(
            client_obj, final_model_str, api_key_str, base_url_str, api_mode,
        )

    def _route(client_obj, final_model_str):
        """Return (client, model), converting to the async wrapper when ``async_mode``."""
        if async_mode:
            return _to_async_client(client_obj, final_model_str, is_vision=is_vision)
        return client_obj, final_model_str

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved, effective_provider = _resolve_auto_route(
            main_runtime=main_runtime,
            task=task,
        )
        if client is None:
            return None, None
        # An OpenRouter-format model override won't work on a non-OpenRouter
        # provider (e.g. local server); drop it for the provider's default.
        if model and "/" in model and resolved and "/" not in resolved:
            logger.debug(
                "Dropping OpenRouter-format model %r for non-OpenRouter "
                "auxiliary provider (using %r instead)", model, resolved)
            model = None
        routed_client, routed_model = _route(client, model or resolved)
        _tag_effective_provider(routed_client, effective_provider)
        return routed_client, routed_model

    # ── OpenRouter ───────────────────────────────────────────
    if provider == "openrouter":
        client, default = _try_openrouter(
            explicit_api_key=explicit_api_key,
            model=model,
        )
        if client is None:
            logger.warning(
                "resolve_provider_client: openrouter requested but %s",
                _describe_openrouter_unavailable(model=model),
            )
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return _route(client, final_model)

    # ── Nous Portal (OAuth) ──────────────────────────────────────────
    if provider == "nous":
        # Vision: caller flag, _PROVIDER_VISION_MODELS override, or a known vision id.
        _is_vision = (
            is_vision
            or model in _PROVIDER_VISION_MODELS.values()
            or (model or "").strip().lower() == "mimo-v2-omni"
        )
        client, default = _try_nous(vision=_is_vision)
        if client is None:
            logger.warning("resolve_provider_client: nous requested "
                           "but Nous Portal not configured (run: hermes auth)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        # Dual-wire: anthropic/* → /v1/messages, else /chat/completions. Derive from
        # the catalog id (not a stale api_mode) so aux matches the main agent.
        from hermes_cli.providers import nous_api_mode

        portal_mode = nous_api_mode(final_model)
        api_key_str = str(getattr(client, "api_key", "") or "")
        base_url_str = str(getattr(client, "base_url", "") or "")
        client = _maybe_wrap_anthropic(
            client, final_model, api_key_str, base_url_str, portal_mode,
        )
        return _route(client, final_model)

    # ── OpenAI Codex (OAuth → Responses API) ─────────────────────────
    if provider == "openai-codex":
        if not model:
            logger.warning(
                "resolve_provider_client: openai-codex requested without a "
                "model; pass model explicitly (e.g. model.model in config.yaml "
                "or auxiliary.<task>.model for per-task aux routing)."
            )
            return None, None
        if raw_codex:
            # Raw OpenAI client for callers needing responses.stream() (main agent loop).
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning("resolve_provider_client: openai-codex requested "
                               "but no Codex OAuth token found (run: hermes model)")
                return None, None
            final_model = _normalize_resolved_model(model, provider)
            raw_client = _create_openai_client(
                api_key=codex_token,
                base_url=_CODEX_AUX_BASE_URL,
                default_headers=_codex_cloudflare_headers(codex_token),
            )
            return (raw_client, final_model)
        # Standard path: wrap in CodexAuxiliaryClient adapter
        client, default = _build_codex_client(model)
        if client is None:
            logger.warning("resolve_provider_client: openai-codex requested "
                           "but no Codex OAuth token found (run: hermes model)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return _route(client, final_model)

    # ── xAI Grok OAuth (device code → Responses API) ───────────────
    # Without this branch xai-oauth falls to the generic oauth_external arm, returns (None, None),
    # and silently re-routes every aux task to the user's Step-2 fallback — surprise
    # OpenRouter/Nous bills for side tasks they expected on their xAI subscription.
    if provider == "xai-oauth":
        client, default = _build_xai_oauth_aux_client(model)
        if client is None:
            logger.warning(
                "resolve_provider_client: xai-oauth requested but no xAI "
                "OAuth token found (run: hermes model -> xAI Grok OAuth — SuperGrok / Premium+)"
            )
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return _route(client, final_model)

    # ── Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY) ───────────
    if provider == "custom":
        custom_base = ""
        custom_key = ""
        # Base for the Anthropic-wrap decision. anthropic_messages must keep the raw /anthropic
        # base while the plain OpenAI client uses the /v1-rewritten custom_base (never
        # /anthropic/chat/completions). Empty means "use custom_base".
        wrap_base = ""
        if explicit_base_url:
            custom_base = _to_openai_base_url(explicit_base_url).strip()
            if api_mode == "anthropic_messages":
                wrap_base = (explicit_base_url or "").strip().rstrip("/")
            custom_key = (
                (explicit_api_key or "").strip()
                or _scoped_key_env("OPENAI_API_KEY")
                or _read_main_api_key_if_same_host(custom_base)
                or "no-key-required"  # local servers don't need auth
            )
            if not custom_base:
                logger.warning(
                    "resolve_provider_client: explicit custom endpoint requested "
                    "but base_url is empty"
                )
                return None, None
        elif main_runtime:
            # Reuse main_runtime's concrete base_url + api_key for a named custom provider;
            # re-resolving from bare "custom" loses the name and lands on the wrong provider.
            _main_base = str(main_runtime.get("base_url") or "").strip().rstrip("/")
            _main_key = str(main_runtime.get("api_key") or "").strip()
            if _main_base and _main_key:
                custom_base = _main_base
                custom_key = _main_key
        if custom_base and custom_key:
            final_model = _normalize_resolved_model(
                model or (main_runtime.get("model") if main_runtime else None) or "gpt-4o-mini",
                provider,
            )
            extra = {}
            _clean_base, _dq = _extract_url_query_params(custom_base)
            if _dq:
                extra["default_query"] = _dq
            _custom_headers = _endpoint_default_headers(custom_base, provider, is_vision=is_vision)
            if _custom_headers:
                extra["default_headers"] = _custom_headers
            client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **extra)
            client = _wrap_if_needed(client, final_model, wrap_base or custom_base, custom_key)
            return _route(client, final_model)
        # Try custom first, then API-key providers (Codex excluded here:
        # falling through to Codex with no model is a stale-constant trap).
        for try_fn in (_try_custom_endpoint, _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = _normalize_resolved_model(model or default, provider)
                _cbase = str(getattr(client, "base_url", "") or "")
                # ``client.api_key`` may be a callable (Azure Entra bearer provider);
                # wrapping decisions only need base_url + api_mode.
                _raw_ckey = getattr(client, "api_key", "")
                _ckey = "" if (callable(_raw_ckey) and not isinstance(_raw_ckey, str)) else str(_raw_ckey or "")
                client = _wrap_if_needed(client, final_model, _cbase, _ckey)
                return _route(client, final_model)
        logger.warning("resolve_provider_client: custom/main requested "
                       "but no endpoint credentials found")
        return None, None

    # ── Named custom providers (config.yaml providers dict / custom_providers list) ───
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider
        # If the raw name is an alias (``kimi`` → ``kimi-coding``) and a custom_providers entry
        # exists under it, the custom entry wins over alias rewriting. Only for aliases, so
        # entries matching a canonical name (e.g. ``nous``) still defer to the built-in.
        custom_entry = None
        if original_provider and original_provider != provider:
            custom_entry = _get_named_custom_provider(original_provider)
        if custom_entry is None:
            custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = (custom_entry.get("base_url") or "").strip()
            custom_key = _named_custom_api_key(custom_entry, provider, custom_base)
            if custom_key == "no-key-required":
                logger.warning(
                    "resolve_provider_client: named custom provider %r has no resolvable "
                    "api_key — request will be sent with placeholder no-key-required "
                    "and will 401 on auth-required endpoints",
                    custom_entry.get("name") or provider,
                )
            # Explicit per-task api_mode override wins over the provider entry's.
            entry_api_mode = (api_mode or custom_entry.get("api_mode") or "").strip()
            if custom_base:
                final_model = _normalize_resolved_model(
                    model
                    or custom_entry.get("model")
                    or (main_runtime.get("model") if main_runtime else None)
                    or _read_main_model_for_aux()
                    or "gpt-4o-mini",
                    provider,
                )
                logger.debug(
                    "resolve_provider_client: named custom provider %r (%s, api_mode=%s)",
                    provider, final_model, entry_api_mode or "chat_completions")

                def _openai_wire_client():
                    # OpenAI-wire paths need the /v1 equivalent of the configured base.
                    _clean_base2, _dq2 = _extract_url_query_params(_to_openai_base_url(custom_base))
                    _extra2 = {"default_query": _dq2} if _dq2 else {}
                    _headers2 = _apply_user_default_headers(None)
                    if _headers2:
                        _extra2["default_headers"] = _headers2
                    return _create_openai_client(api_key=custom_key, base_url=_clean_base2, **_extra2)

                # anthropic_messages: route via AnthropicAuxiliaryClient (mirrors _try_custom_endpoint);
                # the Anthropic SDK sees the original (un-rewritten) URL.
                if entry_api_mode == "anthropic_messages":
                    try:
                        from agent.anthropic_adapter import build_anthropic_client
                        real_client = build_anthropic_client(custom_key, custom_base)
                    except ImportError:
                        logger.warning(
                            "Named custom provider %r declares api_mode="
                            "anthropic_messages but the anthropic SDK is not "
                            "installed — falling back to OpenAI-wire.",
                            provider,
                        )
                        return _route(_openai_wire_client(), final_model)
                    return _route(AnthropicAuxiliaryClient(
                        real_client, final_model, custom_key, custom_base, is_oauth=False,
                    ), final_model)
                client = _openai_wire_client()
                # codex_responses, or auto-detect via _wrap_if_needed (which reads the
                # closed-over task-level `api_mode`).
                if entry_api_mode == "codex_responses":
                    client = CodexAuxiliaryClient(client, final_model)
                else:
                    client = _wrap_if_needed(client, final_model, custom_base, custom_key)
                return _route(client, final_model)
            logger.warning(
                "resolve_provider_client: named custom provider %r has no base_url",
                provider)
            return None, None
    except ImportError:
        pass

    # ── Azure Foundry (delegates to runtime resolver for auth_mode-aware routing) ─
    # The generic PROVIDER_REGISTRY path only knows the static AZURE_FOUNDRY_API_KEY env var,
    # missing ``auth_mode: entra_id`` (callable bearer token) and config-driven base_url
    # overrides. Delegate to the main agent's runtime resolver so aux inherits the full Azure config.
    if provider == "azure-foundry":
        client, default_model = _try_azure_foundry(
            model=model,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            api_mode=api_mode,
        )
        if client is None:
            logger.warning(
                "resolve_provider_client: azure-foundry requested but "
                "runtime resolution failed (run: hermes doctor for "
                "diagnostics)"
            )
            return None, None
        final_model = _normalize_resolved_model(model or default_model, provider)
        return _route(client, final_model)

    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            resolve_api_key_provider_credentials,
            resolve_external_process_provider_credentials,
        )
    except ImportError:
        logger.debug("hermes_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        # Debug-level, deduped per provider name so repeated retries stay silent.
        if provider not in _LOGGED_UNKNOWN_PROVIDER_KEYS:
            _LOGGED_UNKNOWN_PROVIDER_KEYS.add(provider)
            logger.debug("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        if provider == "anthropic":
            client, default_model = _try_anthropic(explicit_api_key=explicit_api_key)
            if client is None:
                logger.warning("resolve_provider_client: anthropic requested but no Anthropic credentials found")
                return None, None
            final_model = _normalize_resolved_model(model or default_model, provider)
            return _route(client, final_model)

        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get("api_key", "")).strip()
        # Explicit api_key override (fallback_model / custom_providers entry) lets callers
        # authenticate where no built-in credential is registered for this alias.
        if explicit_api_key:
            api_key = explicit_api_key.strip() or api_key
        raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        if explicit_base_url:
            raw_base_url = explicit_base_url.strip().rstrip("/")
        # OpenCode Zen free tier (*-free slugs) is served anonymously on the Zen relay only;
        # any bearer (even a Go subscription key) is rejected, so route keyless regardless of creds.
        try:
            from hermes_cli.models import opencode_zen_free_runtime as _oc_free_rt
            _free_rt = _oc_free_rt(provider, model)
        except Exception:
            _free_rt = None
        if _free_rt is not None:
            api_key = _free_rt["api_key"]
            raw_base_url = str(_free_rt["base_url"]).rstrip("/")
        if provider == "actual":
            try:
                from hermes_cli.auth import (
                    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
                    is_actual_local_base_url,
                    normalize_actual_base_url,
                )

                raw_base_url = normalize_actual_base_url(raw_base_url)
                if not api_key and is_actual_local_base_url(raw_base_url):
                    api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
            except Exception:
                pass
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            if provider == "copilot":
                tried_sources.append("gh auth token")
            logger.debug("resolve_provider_client: provider %s has no API "
                         "key configured (tried: %s)",
                         provider, ", ".join(tried_sources))
            return None, None

        base_url = _to_openai_base_url(raw_base_url)
        # Explicit base_url override: a fallback_model/custom_providers entry routing a
        # built-in provider name to a user-specified endpoint.
        if explicit_base_url:
            base_url = _to_openai_base_url(explicit_base_url.strip().rstrip("/"))

        default_model = _get_aux_model_for_provider(provider)
        final_model = _normalize_resolved_model(model or default_model, provider)

        if provider == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            if is_native_gemini_base_url(base_url):
                client = GeminiNativeClient(api_key=api_key, base_url=base_url)
                logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
                return _route(client, final_model)

        headers = _endpoint_default_headers(base_url, provider, is_vision=is_vision, xai=True)
        client = _create_openai_client(api_key=api_key, base_url=base_url,
                        **({"default_headers": headers} if headers else {}))

        # Copilot GPT-5+ models (except gpt-5-mini) are only reachable via the Responses API;
        # wrap so call_llm() transparently routes through responses.stream().
        if provider == "copilot" and final_model and not raw_codex:
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                if _should_use_copilot_responses_api(final_model):
                    logger.debug(
                        "resolve_provider_client: copilot model %s needs "
                        "Responses API — wrapping with CodexAuxiliaryClient",
                        final_model)
                    client = CodexAuxiliaryClient(client, final_model)
            except ImportError:
                pass

        # General api_mode handling for any API-key provider (e.g. direct OpenAI + codex model);
        # also rewraps Anthropic-wire endpoints (api.kimi.com/coding, /anthropic gateways) so
        # providers like kimi-coding land on the right transport without per-provider branches.
        client = _wrap_if_needed(client, final_model, raw_base_url, api_key)

        logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
        return _route(client, final_model)

    if pconfig.auth_type == "external_process":
        creds = resolve_external_process_provider_credentials(provider)
        final_model = _normalize_resolved_model(
            model
            or (main_runtime.get("model") if main_runtime else None)
            or _read_main_model_for_aux(),
            provider,
        )
        # Any external-process provider whose registered profile supplies a
        # client is served here — keyed on the profile, not on a provider name,
        # so an out-of-tree ACP provider reaches the auxiliary path (compression,
        # vision, background review) exactly like the in-tree one.
        _extproc_profile = None
        try:
            from providers import get_provider_profile as _get_provider_profile

            _extproc_profile = _get_provider_profile(provider)
        except Exception:
            _extproc_profile = None
        if _extproc_profile is not None:
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
            command = str(creds.get("command", "")).strip() or None
            args = list(creds.get("args") or [])
            if not final_model:
                logger.warning(
                    "resolve_provider_client: %s requested but no model "
                    "was provided or configured",
                    provider,
                )
                return None, None
            if not api_key or not base_url:
                logger.warning(
                    "resolve_provider_client: %s requested but external "
                    "process credentials are incomplete",
                    provider,
                )
                return None, None
            try:
                client = _extproc_profile.create_client(
                    api_key=api_key,
                    base_url=base_url,
                    command=command,
                    args=args,
                )
            except Exception:
                logger.warning(
                    "resolve_provider_client: profile %r failed to create an "
                    "external-process client",
                    provider,
                    exc_info=True,
                )
                client = None
            if client is not None:
                logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
                return _route(client, final_model)
        if provider not in _LOGGED_UNSUPPORTED_EXTPROC_KEYS:
            _LOGGED_UNSUPPORTED_EXTPROC_KEYS.add(provider)
            logger.debug("resolve_provider_client: external-process provider %s not "
                         "directly supported", provider)
        return None, None

    elif pconfig.auth_type == "vertex":
        client, final_model = _build_vertex_client(provider, model)
        if client is None:
            return None, None
        return _route(client, final_model)

    elif pconfig.auth_type == "aws_sdk":
        client, final_model = _build_bedrock_client(provider, model, raw_codex=raw_codex)
        if client is None:
            return None, None
        return _route(client, final_model)

    elif pconfig.auth_type in {"oauth_device_code", "oauth_external"}:
        # OAuth providers — route through their specific try functions
        if provider == "nous":
            return resolve_provider_client("nous", model, async_mode)
        if provider == "openai-codex":
            return resolve_provider_client("openai-codex", model, async_mode)
        if provider == "xai-oauth":
            return resolve_provider_client("xai-oauth", model, async_mode)
        # Other OAuth providers not directly supported
        if provider not in _LOGGED_UNSUPPORTED_OAUTH_KEYS:
            _LOGGED_UNSUPPORTED_OAUTH_KEYS.add(provider)
            logger.debug("resolve_provider_client: OAuth provider %s not "
                         "directly supported, try 'auto'", provider)
        return None, None

    # Debug-level, deduped on (auth_type, provider): the first occurrence surfaces a real
    # schema-drift bug, per-call retries stay silent.
    _auth_dedup_key = (pconfig.auth_type, provider)
    if _auth_dedup_key not in _LOGGED_UNHANDLED_AUTHTYPE_KEYS:
        _LOGGED_UNHANDLED_AUTHTYPE_KEYS.add(_auth_dedup_key)
        logger.debug("resolve_provider_client: unhandled auth_type %s for %s",
                     pconfig.auth_type, provider)
    return None, None


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(
    task: str = "",
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    ``task`` selects a config.yaml ``auxiliary.<task>`` provider/model override.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


_VISION_AUTO_PROVIDER_ORDER = (
    "openrouter",
    "nous",
    "deepinfra",
)


def _main_model_supports_vision(provider: str, model: Optional[str]) -> bool:
    """Return True when ``provider``/``model`` is known to accept image input.

    Lets vision auto-detect skip a text-only main provider instead of surfacing
    a cryptic provider-side error. Unknown capability → True (attempt the call).
    """
    try:
        from agent.image_routing import _lookup_supports_vision
        from hermes_cli.config import load_config_readonly
    except ImportError:
        return True
    try:
        supports = _lookup_supports_vision(provider, model, load_config_readonly())
    except Exception:  # pragma: no cover - defensive
        return True
    if supports is None:
        # No capability data — attempt the call rather than silently skipping.
        return True
    return bool(supports)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)


def _resolve_strict_vision_backend(
    provider: str,
    model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "copilot":
        return resolve_provider_client("copilot", model, is_vision=True)
    if provider == "openrouter":
        return _try_openrouter(model=model)
    if provider == "nous":
        # Must go through resolve_provider_client so anthropic/* vision picks
        # wrap onto /v1/messages; a bare _try_nous client 404s.
        return resolve_provider_client("nous", model, is_vision=True)
    if provider == "openai-codex":
        # No safe default Codex model (shifting allow-list); callers must
        # specify via auxiliary.<task>.model.
        return resolve_provider_client("openai-codex", model, is_vision=True)
    if provider == "anthropic":
        return _try_anthropic()
    if provider == "deepinfra":
        # Default vision model is discovered live via the profile's
        # default_vision_model() hook so no hardcoded id can rot.
        vision_model = model or _resolve_provider_vision_default("deepinfra")
        if not vision_model:
            logger.debug(
                "Vision auto-detect: deepinfra catalog unreachable or "
                "returned no vision-tagged models — skipping"
            )
            return None, None
        return resolve_provider_client("deepinfra", vision_model, is_vision=True)
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def get_available_vision_backends() -> List[str]:
    """Return available vision backends in auto-selection order (active provider → OpenRouter → Nous).

    Single source of truth for setup, tool gating, and runtime auto-routing.
    """
    available: List[str] = []
    main_provider = _read_main_provider()
    if main_provider and main_provider not in {"auto", ""}:
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            if _strict_vision_backend_available(main_provider):
                available.append(main_provider)
        else:
            client, _ = resolve_provider_client(main_provider, _read_main_model())
            if client is not None:
                available.append(main_provider)
    # 2. OpenRouter, 3. Nous — skip if already covered by main provider.
    for p in _VISION_AUTO_PROVIDER_ORDER:
        if p not in available and _strict_vision_backend_available(p):
            available.append(p)
    return available


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides beat provider selection; explicit providers may
    force experimental backends; auto mode only tries backends known to work.
    """
    runtime = _normalize_main_runtime(main_runtime)
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model, is_vision=True)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        provider_for_base_override = (
            requested if requested and requested not in {"", "auto"} else "custom"
        )
        client, final_model = resolve_provider_client(
            provider_for_base_override,
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=runtime,
        )
        return provider_for_base_override, client, (final_model if client is not None else None)

    if requested == "auto":
        # Auto-detect order: 1. main provider + model (per-provider vision
        # overrides / live DeepInfra discovery; Nous uses its own strict backend),
        # 2. OpenRouter, 3. Nous Portal, 4. DeepInfra, 5. stop.
        main_provider = str(runtime.get("provider") or _read_main_provider())
        main_model = str(runtime.get("model") or _read_main_model())
        if main_provider.strip().lower() == "moa":
            # MoA main_model is a preset NAME, not a wire model — unwrap to the
            # preset's aggregator slot so capability probes target a real pair.
            _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
            if _agg_provider and _agg_model:
                main_provider, main_model = _agg_provider, _agg_model
                # The moa:// facade endpoint belongs to the virtual provider, not
                # the aggregator's real provider.
                runtime = dict(runtime)
                runtime["base_url"] = ""
                runtime["api_key"] = ""
                runtime["api_mode"] = ""
        if main_provider and main_provider not in {"auto", "", "moa"}:
            # A provider vision default (static override or catalog discovery)
            # is a *known* multimodal model; the pinned chat model usually isn't,
            # so only fall back to it when no provider default exists.
            provider_vision_default = _resolve_provider_vision_default(main_provider)
            vision_model = provider_vision_default or main_model
            if main_provider == "nous":
                # Nous picks its vision model from Portal tier-aware slots inside
                # _try_nous(vision=True); passing the chat model would override
                # that and 404. Only explicit auxiliary.vision.model may override.
                sync_client, default_model = _resolve_strict_vision_backend(
                    main_provider, resolved_model or provider_vision_default
                )
                if sync_client is not None:
                    logger.info(
                        "Vision auto-detect: using main provider %s (%s)",
                        main_provider, default_model or resolved_model or main_model,
                    )
                    return _finalize(main_provider, sync_client, default_model)
            elif main_provider in _PROVIDERS_WITHOUT_VISION:
                # Provider endpoint rejects image input entirely (e.g. Kimi
                # Coding Plan); fall through to aggregators instead of 404ing.
                logger.debug(
                    "Vision auto-detect: skipping main provider %s (no "
                    "vision support) — falling through to aggregator chain",
                    main_provider,
                )
            elif not _main_model_supports_vision(main_provider, vision_model):
                # Known text-only model; sending an image yields a cryptic
                # provider error. Log only the provider name (CodeQL
                # clear-text-logging false positives on multi-value logs).
                logger.debug(
                    "Vision auto-detect: skipping main provider %s "
                    "(reports no vision capability) — falling through to "
                    "aggregator chain",
                    main_provider,
                )
            else:
                # Custom endpoints carry no built-in base_url/api_key, so recover
                # the live main endpoint from set_runtime_main() (or the
                # configured custom endpoint) to build a working client.
                rpc_base_url = None
                rpc_api_key = None
                rpc_api_mode = resolved_api_mode
                if main_provider == "custom" or main_provider.startswith("custom:"):
                    if runtime.get("base_url"):
                        custom_base, custom_key, custom_mode = (
                            runtime.get("base_url"), runtime.get("api_key") or None, runtime.get("api_mode"),
                        )
                    else:
                        # Non-gateway caller: no live runtime recorded.
                        custom_base, custom_key, custom_mode = _resolve_custom_runtime()
                    if custom_base:
                        rpc_base_url = custom_base
                        rpc_api_key = custom_key
                        rpc_api_mode = resolved_api_mode or custom_mode or None
                rpc_client, rpc_model = resolve_provider_client(
                    main_provider, vision_model,
                    api_mode=rpc_api_mode,
                    explicit_base_url=rpc_base_url,
                    explicit_api_key=rpc_api_key,
                    main_runtime=runtime,
                    is_vision=True)
                if rpc_client is not None:
                    logger.info(
                        "Vision auto-detect: using main provider %s (%s)",
                        main_provider, rpc_model or vision_model,
                    )
                    return _finalize(
                        main_provider, rpc_client, rpc_model or vision_model)

        # Fall back through aggregators (their dedicated vision model, not the
        # user's main model).
        for candidate in _VISION_AUTO_PROVIDER_ORDER:
            if candidate == main_provider:
                continue  # already tried above
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)

        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(
            requested, resolved_model
        )
        return _finalize(requested, sync_client, default_model)

    # ZAI vision must use the OpenAI-compatible endpoint: the Anthropic wire
    # rejects max_tokens on multimodal calls (error 1210).
    if requested == "zai" and not resolved_base_url:
        zai_openai_urls = [
            "https://open.bigmodel.cn/api/paas/v4",
            "https://api.z.ai/api/paas/v4",
        ]
        for _zai_url in zai_openai_urls:
            client, final_model = _get_cached_client(
                requested, resolved_model, async_mode,
                base_url=_zai_url,
                api_key=resolved_api_key or None,
                api_mode="chat_completions",
                main_runtime=runtime,
                is_vision=True,
            )
            if client is not None:
                return _finalize(requested, client, final_model)
        # Fallback: try without explicit base_url (old behavior)

    client, final_model = _get_cached_client(requested, resolved_model, async_mode,
                                             api_mode=resolved_api_mode,
                                             main_runtime=runtime,
                                             is_vision=True)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs (Nous Portal product tags when Nous-backed, else {})."""
    return _nous_extra_body() if auxiliary_is_nous else {}


def auxiliary_max_tokens_param(value: int, *, model: Optional[str] = None) -> dict:
    """Return the correct max-tokens kwarg for the auxiliary client's provider.

    Direct OpenAI/Copilot and newer OpenAI-family models (by ``model`` name, so
    custom endpoints fronting e.g. gpt-5.x are caught) need max_completion_tokens.
    """
    custom_base = _current_custom_base_url()
    or_key = _scoped_key_env("OPENROUTER_API_KEY")
    _custom_host = base_url_hostname(custom_base) or ""
    if (not or_key
            and _read_nous_auth() is None
            and (
                _custom_host == "api.openai.com"
                or _custom_host == "api.githubcopilot.com"
                or _custom_host.endswith(".githubcopilot.com")
            )):
        return {"max_completion_tokens": value}
    if model_forces_max_completion_tokens(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


# ── Centralized LLM Call API ────────────────────────────────────────────────
# call_llm()/async_call_llm() own the full lifecycle: resolve provider+model,
# get a cached client, shape request args, call, return. Every auxiliary LLM
# consumer should use these rather than hand-building clients.

# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# Loop identity is NOT part of the key: stale-loop entries are replaced in
# place on async hits, bounding growth to one entry per provider config
# (avoids fd accumulation in long-running gateways).
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


class _CallableCacheDiscriminator:
    """Hash a credential callback by identity without exposing its state."""

    __slots__ = ("_callback",)

    def __init__(self, callback: Any) -> None:
        # Retain the callback so its id cannot be reused while cached.
        self._callback = callback

    def __hash__(self) -> int:
        return id(self._callback)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _CallableCacheDiscriminator)
            and self._callback is other._callback
        )

    def __repr__(self) -> str:
        return "<callable-api-key>"


def _runtime_cache_discriminator(field: str, value: Any) -> Any:
    """Return a hashable, secret-safe runtime cache-key component."""
    if field == "api_key" and callable(value):
        return _CallableCacheDiscriminator(value)
    if field == "api_key" and isinstance(value, str) and value:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
        return ("api-key-digest", digest)
    return value


def _client_cache_key(
    provider: str,
    *,
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple:
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple(
        _runtime_cache_discriminator(field, runtime.get(field, ""))
        for field in _MAIN_RUNTIME_FIELDS
    ) if provider == "auto" else ()
    # `auto` resolves through task-specific policy, so the task joins the key.
    task_key = (
        (task or "", _task_prefers_fast_model(task))
        if provider == "auto"
        else ""
    )
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    # Model MUST be in the key: concurrent calls to the same endpoint with
    # different models would otherwise share an entry, and the second builder's
    # _store_cached_client would close the first's client mid-request.
    model_key = model or runtime.get("model", "")
    api_key_key = _runtime_cache_discriminator("api_key", api_key or "")
    return (provider, async_mode, base_url or "", api_key_key, api_mode or "", runtime_key, is_vision, task_key, pool_hint, model_key)


def _current_event_loop() -> Any:
    """``asyncio.get_event_loop()`` or None when no loop can be obtained (async cache-key binding)."""
    try:
        import asyncio as _aio
        return _aio.get_event_loop()
    except RuntimeError:
        return None


def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any = None) -> None:
    if isinstance(client, _AuxProbeClientStub):
        # Probe stubs must never be cached — the next hit would get a dud client.
        return
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _close_cached_client(old_entry[0])
        _client_cache[cache_key] = (client, default_model, bound_loop)


def _refresh_nous_auxiliary_client(
    *,
    cache_provider: str,
    model: Optional[str],
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    lookup_model: Optional[str] = None,
    lookup_task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Refresh Nous runtime creds, rebuild the client, and replace the cache entry.

    ``model`` is the resolved model actually sent on the wire (e.g. the provider
    default ``"Hermes-4-405B"``); it is stored as the entry's usable model and
    returned to the caller. ``lookup_model`` is the model as it was passed to
    ``_get_cached_client`` when the (now stale) client was acquired -- ``None``
    on the default Nous config, where ``call_llm`` looks up with
    ``resolved_model=None``. The cache KEY MUST be built from ``lookup_model`` so
    the fresh client overwrites the exact entry the stale client is served from.
    Keying on the resolved ``model`` instead stored under a different key (model
    element ``"Hermes-4-405B"`` vs the lookup's ``""``), leaving the expired
    client immortal so every auxiliary call 401s forever (#56889).

    ``lookup_task`` is the task the stale client was acquired under. For
    ``provider == "auto"`` the task participates in the cache key (task-specific
    fallback policy), so it MUST be carried into the key here for the same
    reason as ``lookup_model``; otherwise an auto-provider client refreshed on a
    401 lands under the ``task=""`` key while the stale entry survives under the
    task-scoped key (#58894).
    """
    runtime = _resolve_nous_runtime_api(force_refresh=True)
    if runtime is None:
        return None, model

    fresh_key, fresh_base_url = runtime
    sync_client = _create_openai_client(api_key=fresh_key, base_url=fresh_base_url)
    final_model = model

    current_loop = _current_event_loop() if async_mode else None
    if async_mode:
        client, final_model = _to_async_client(sync_client, final_model or "", is_vision=is_vision)
    else:
        client = sync_client

    cache_key = _client_cache_key(
        cache_provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        task=lookup_task,
        model=lookup_model,
    )
    _store_cached_client(cache_key, client, final_model, bound_loop=current_loop)
    return client, final_model


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The SDK's ``__del__`` schedules ``aclose()`` on the *running* loop, but the
    transport is bound to the loop the client was created on; when that loop is
    closed/dead this raises "Event loop is closed" into prompt_toolkit's loop.
    Safe because cached clients are closed explicitly and the OS reaps the rest.
    Call once at CLI startup, before any ``AsyncOpenAI`` client is created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    Stops ``__del__`` scheduling ``aclose()`` on a dead loop. Deliberately skips
    the full async close — the OS drops the connections at exit.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def _schedule_async_close(close_result: Any, client: Any) -> None:
    """Finish an async close without leaking an unawaited coroutine."""
    async def _await_close() -> None:
        try:
            await close_result
        except Exception:
            pass
        finally:
            _force_close_async_httpx(client)

    runner = _await_close()
    try:
        import asyncio as _aio

        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            _aio.run(runner)
        else:
            task = loop.create_task(runner)

            def _consume(completed_task) -> None:
                try:
                    completed_task.exception()
                except BaseException:
                    pass

            task.add_done_callback(_consume)
            runner = None
    except Exception:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        _force_close_async_httpx(client)


def _close_cached_client(client: Any, *, close_async: bool = False) -> None:
    """Close one cached client, awaiting async transports only when safe."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if not callable(close_fn):
        _force_close_async_httpx(client)
        return
    try:
        close_result = close_fn()
    except Exception:
        _force_close_async_httpx(client)
        return
    if inspect.isawaitable(close_result):
        if close_async:
            _schedule_async_close(close_result, client)
        else:
            # Never await a client owned by another live loop; close the
            # coroutine (no unawaited warning) and neuter the transport.
            try:
                close_result.close()
            except Exception:
                pass
            _force_close_async_httpx(client)
        return
    _force_close_async_httpx(client)


def shutdown_cached_clients() -> None:
    """Close all cached clients; call at CLI shutdown *before* the loop closes.

    Snapshot+clear under the lock, close outside it: async teardown can block
    while an owner loop drains, and holding the lock would convoy every caller.
    """
    with _client_cache_lock:
        clients = [
            (entry[0], entry[2])
            for entry in _client_cache.values()
            if entry[0] is not None
        ]
        _client_cache.clear()
    try:
        import asyncio as _aio

        running_loop = _aio.get_running_loop()
    except RuntimeError:
        running_loop = None
    for client, owner_loop in clients:
        # A live foreign loop owns its transport — neuter only and let it finish
        # teardown. Closed loops and the current loop are safe to drain here.
        close_async = owner_loop is not None and (
            owner_loop.is_closed() or owner_loop is running_loop
        )
        _close_cached_client(client, close_async=close_async)


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call after each agent turn; defense-in-depth behind ``neuter_async_httpx_del``.
    """
    stale_clients = []
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                stale_keys.append(key)
                stale_clients.append(client)
        for key in stale_keys:
            del _client_cache[key]
    for client in stale_clients:
        _close_cached_client(client, close_async=True)


def _is_openrouter_client(client: Any) -> bool:
    for obj in (client, getattr(client, "_client", None), getattr(client, "client", None)):
        if obj and base_url_host_matches(str(getattr(obj, "base_url", "") or ""), "openrouter.ai"):
            return True
    return False


def _cached_client_accepts_slash_models(client: Any, cached_default: Optional[str]) -> bool:
    """Best-effort check for cached clients that accept ``vendor/model`` IDs."""
    if _is_openrouter_client(client):
        return True
    return bool(cached_default and "/" in cached_default)


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that support them.

    Mirrors the resolve_provider_client() guard, which cache hits skip.
    """
    if model and "/" in model and not _cached_client_accepts_slash_models(client, cached_default):
        return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients bind to the loop they were created on, so every async hit
    validates the cached loop is the current, open loop; stale entries are
    replaced in place (bounded cache, no cross-loop reuse).
    """
    current_loop = _current_event_loop() if async_mode else None
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(
        provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        task=task,
        model=model,
    )
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # Cached client must be bound to the CURRENT, OPEN loop.
                loop_ok = (
                    cached_loop is not None
                    and cached_loop is current_loop
                    and not cached_loop.is_closed()
                )
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return cached_client, effective
                # Stale — evict. Only a closed owner loop may be awaited here;
                # a live foreign loop stays force-neutered.
                owner_loop_closed = (
                    cached_loop is not None and cached_loop.is_closed()
                )
                _close_cached_client(cached_client, close_async=owner_loop_closed)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return cached_client, effective
    # Build outside the lock. For pool-backed providers derive the key from the
    # pool entry: resolve_api_key_provider_credentials prefers env vars, which
    # would bypass pool rotation and retry an exhausted key.
    effective_api_key = api_key
    if not effective_api_key:
        _pe = _peek_pool_entry(_normalize_aux_provider(provider))
        if _pe is not None:
            _pk = _pool_runtime_api_key(_pe)
            if _pk:
                effective_api_key = _pk
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=effective_api_key,
        api_mode=api_mode,
        main_runtime=runtime,
        is_vision=is_vision,
        task=task,
    )
    if client is not None:
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # FIFO safety-belt eviction. Do NOT close evicted clients:
                # another caller may be mid-request on one; refcount/GC handles it.
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key = next(iter(_client_cache))
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                built_client = client
                client, default_model, _ = _client_cache[cache_key]
                # Race loser was never exposed to a caller — safe to close now.
                _close_cached_client(built_client, close_async=async_mode)
    return client, model or default_model


# Aliases for direct REST APIs not modeled in PROVIDER_REGISTRY, so
# ``auxiliary.<task>.provider: openai`` resolves to a working ``custom``
# endpoint (OPENAI_API_KEY + api.openai.com) instead of silently falling
# back to the main provider and sending OpenAI model names elsewhere.
_AUX_DIRECT_API_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine (provider, model, base_url, api_key, api_mode) for a call.

    Priority: explicit args > config auxiliary.{task}.* > "auto". A bare
    base_url means custom, but a first-class provider + base_url keeps the
    provider identity so its auth/transport shaping still applies.
    api_mode is "chat_completions", "codex_responses", or None (auto-detect).
    """
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None
    cfg_api_mode = None

    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        # Resolve key_env → env var when api_key is not set directly
        if not cfg_api_key:
            cfg_key_env = str(
                task_config.get("key_env") or task_config.get("api_key_env") or ""
            ).strip()
            if cfg_key_env:
                cfg_api_key = _scoped_key_env(cfg_key_env) or None
        cfg_api_mode = str(task_config.get("api_mode", "")).strip() or None

    # 'auto' is a sentinel ("inherit / auto-detect"), not a model id — leaking it
    # to the wire yields a 200 with an error-text body that consumers accept as
    # output. The explicit `model` kwarg needs the same normalization because
    # MoA slots forward preset `model:` fields through it, not via config.
    if model and model.lower() == "auto":
        model = None
    if cfg_model and cfg_model.lower() == "auto":
        cfg_model = None

    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode

    # An *explicit* `provider: moa` (arg or config) bypasses _resolve_auto(),
    # which only unwraps the implicit case; "moa" isn't in PROVIDER_REGISTRY and
    # would dead-end. Resolve to the preset's aggregator slot instead.
    def _unwrap_moa_provider(prov: str, mdl: Optional[str]) -> Tuple[str, Optional[str]]:
        if prov.strip().lower() != "moa":
            return prov, mdl
        agg_provider, agg_model = _resolve_moa_aggregator(mdl)
        if agg_provider and agg_model:
            return agg_provider, agg_model
        return prov, mdl

    if provider and str(provider).strip().lower() == "moa":
        provider, resolved_model = _unwrap_moa_provider(provider, resolved_model)
        # Any moa:// facade endpoint belongs to the facade, not the aggregator's
        # real provider — drop it (mirrors _resolve_auto()).
        if provider and provider.lower() != "moa":
            base_url = None
            api_key = None
    elif cfg_provider and str(cfg_provider).strip().lower() == "moa":
        cfg_provider, cfg_model = _unwrap_moa_provider(cfg_provider, resolved_model)
        if cfg_provider and cfg_provider.lower() != "moa":
            resolved_model = cfg_model
            cfg_base_url = None
            cfg_api_key = None

    # Direct API-key aliases (``provider: openai`` → custom + api.openai.com/v1).
    # A user-supplied base_url is kept, but the provider still becomes ``custom``
    # so resolution avoids the PROVIDER_REGISTRY-only path.
    def _expand_direct_api_alias(prov: Optional[str], existing_base: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not prov:
            return prov, existing_base
        target_base = _AUX_DIRECT_API_BASE_URLS.get(prov.strip().lower())
        if target_base is None:
            return prov, existing_base
        return "custom", existing_base or target_base

    def _preserve_provider_with_base_url(prov: Optional[str]) -> bool:
        normalized = str(prov or "").strip().lower()
        if normalized in {"", "auto", "custom"} or normalized.startswith("custom:"):
            return False
        try:
            from hermes_cli.providers import get_provider

            return get_provider(normalized) is not None
        except Exception:
            # Keep provider-backed routes safe when the catalog can't load.
            return normalized in {
                "anthropic",
                "copilot",
                "copilot-acp",
                "minimax-oauth",
                "nous",
                "openai-codex",
                "qwen-oauth",
                "xai-oauth",
            }

    if provider:
        provider, base_url = _expand_direct_api_alias(provider, base_url)
    if cfg_provider:
        cfg_provider, cfg_base_url = _expand_direct_api_alias(cfg_provider, cfg_base_url)

    # An explicit provider without base_url adopts the task's configured
    # endpoint (same or unnamed provider) so the early return below carries it.
    # Explicit "auto" is excluded — it must keep flowing through auto-resolution.
    if provider and provider != "auto" and not base_url and cfg_base_url and cfg_provider in (None, provider):
        base_url = cfg_base_url
        if not api_key:
            api_key = cfg_api_key

    if base_url and _preserve_provider_with_base_url(provider):
        return provider, resolved_model, base_url, api_key, resolved_api_mode
    if base_url:
        return "custom", resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode

    if task:
        if cfg_base_url and cfg_api_key:
            return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
        if cfg_base_url and cfg_provider and cfg_provider != "auto":
            # base_url without api_key: keep the provider so it can resolve
            # credentials from env vars instead of locking into "custom".
            return cfg_provider, resolved_model, cfg_base_url, None, resolved_api_mode
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode

        return "auto", resolved_model, None, None, resolved_api_mode

    return "auto", resolved_model, None, None, resolved_api_mode


_DEFAULT_AUX_TIMEOUT = 30.0

# Reasoning compression models can exceed the default 120 s config timeout,
# falling back to the deterministic marker. Bounded *floor* for config-derived
# compression timeouts only; never overrides an explicit per-call timeout.
_COMPRESSION_TIMEOUT_FLOOR_SECONDS = 300.0


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Return the config dict for auxiliary.<task>, or {} when unavailable.

    Plugin-registered tasks get their declared defaults layered under user
    config (user wins); built-in tasks' defaults live in DEFAULT_CONFIG.
    """
    if not task:
        return {}
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        task_config = {}

    # Layer plugin defaults under user config so register_auxiliary_task(defaults=…)
    # works without config.yaml entries.
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    merged = dict(_defaults)
                    merged.update(task_config)
                    return merged
                break
    except Exception:
        # Plugin discovery failure must not break aux task config reads.
        pass

    return task_config


class CompressionFastLane(NamedTuple):
    """Explicit, non-reasoning compression route safe for a bounded summary."""

    certified_non_reasoning: bool
    max_tokens: Optional[int]
    reasoning_config: Optional[Dict[str, Any]]


def _fast_lane_config_fields(
    config: Dict[str, Any],
) -> tuple[str, str, bool, Optional[int]]:
    """``(provider, model, non_reasoning, cap)`` from one task config.

    ``non_reasoning`` only when ``reasoning_effort`` EXPLICITLY disables thinking
    (unset is NOT non-reasoning); ``cap`` is a positive int ``max_output_tokens``
    or None — booleans are config drift, never a cap (``int(True) == 1``).
    """
    from hermes_constants import parse_reasoning_effort

    provider = str(config.get("provider") or "").strip().lower()
    model = str(config.get("model") or "").strip()
    parsed_effort = parse_reasoning_effort(config.get("reasoning_effort"))
    non_reasoning = parsed_effort is not None and parsed_effort.get("enabled") is False
    raw_cap = config.get("max_output_tokens")
    try:
        cap = 0 if isinstance(raw_cap, bool) else int(raw_cap or 0)
    except (TypeError, ValueError):
        cap = 0
    return provider, model, non_reasoning, (cap if cap > 0 else None)


def resolve_compression_fast_lane(
    actual_provider: str,
    actual_model: Optional[str],
    *,
    requested_provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    route_config: Optional[Dict[str, Any]] = None,
) -> CompressionFastLane:
    """Certify the opt-in fast lane: capped only when an explicit, operator-certified
    non-reasoning provider/model exactly matches the route actually called."""
    config = route_config if route_config is not None else _get_auxiliary_task_config("compression")
    cfg_provider, cfg_model, non_reasoning, cap = _fast_lane_config_fields(config)
    provider = str(requested_provider or "").strip().lower() or cfg_provider
    model = str(requested_model or "").strip() or cfg_model
    explicit_route = provider not in {"", "auto"} and model.lower() not in {"", "auto"}
    provider_matches = _normalize_aux_provider(
        _fallback_provider_from_label(str(actual_provider or ""))
    ) == _normalize_aux_provider(provider)
    model_matches = str(actual_model or "").strip().lower() == model.lower()
    certified = explicit_route and provider_matches and model_matches and non_reasoning
    if not certified:
        return CompressionFastLane(False, None, None)
    return CompressionFastLane(True, cap, {"enabled": False, "effort": "none"})


def _compression_config_claims_fast_lane(config: Dict[str, Any]) -> bool:
    """Whether task config declares fast-only controls that cannot leak."""
    provider, model, non_reasoning, cap = _fast_lane_config_fields(config)
    return (
        provider not in {"", "auto"}
        and model.lower() not in {"", "auto"}
        and non_reasoning
        and cap is not None
    )


def _compression_fast_lane_controls(
    task: str | None,
    *,
    actual_provider: str,
    actual_model: str | None,
    requested_provider: str | None,
    requested_model: str | None,
    route_config: Dict[str, Any],
    leak_guard_config: Dict[str, Any],
    max_tokens: int | None,
    extra_body: Dict[str, Any],
) -> tuple[int | None, Dict[str, Any]]:
    """Apply the certified compression controls to one resolved route."""
    if task != "compression" or max_tokens is not None:
        return max_tokens, extra_body
    body = dict(extra_body)
    lane = resolve_compression_fast_lane(
        actual_provider,
        actual_model,
        requested_provider=requested_provider,
        requested_model=requested_model,
        route_config=route_config,
    )
    if lane.reasoning_config is not None:
        if "reasoning" not in body:
            body["reasoning"] = lane.reasoning_config
    elif _compression_config_claims_fast_lane(leak_guard_config):
        body.pop("reasoning", None)
    return lane.max_tokens, body


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """``auxiliary.<task>.timeout`` from config, else *default*."""
    if not task:
        return default
    raw = _get_auxiliary_task_config(task).get("timeout")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _effective_aux_timeout(task: str, timeout: Optional[float]) -> float:
    """Explicit ``timeout`` wins, else config; compression gets a floor so a
    reasoning model summarising a large context isn't cut off."""
    effective = timeout if timeout is not None else _get_task_timeout(task)
    if timeout is None and task == "compression":
        effective = max(effective, _COMPRESSION_TIMEOUT_FLOOR_SECONDS)
    return effective


def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Shallow copy of ``auxiliary.<task>.extra_body`` with ``reasoning_effort`` folded
    into ``reasoning`` unless one is configured (more specific wins). MoA tasks are
    excluded: their reasoning depth is per-slot in the preset."""
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("extra_body")
    result = dict(raw) if isinstance(raw, dict) else {}
    if "reasoning" not in result:
        effort = task_config.get("reasoning_effort")
        if effort is not None and effort != "":
            if task in ("moa_reference", "moa_aggregator"):
                logger.warning(
                    "auxiliary.%s.reasoning_effort is not supported — MoA "
                    "reasoning depth is per-slot: set reasoning_effort on the "
                    "preset's reference_models entries / aggregator instead "
                    "(moa.presets.<name>...). Ignoring.",
                    task,
                )
                return result
            from hermes_constants import parse_reasoning_effort
            parsed = parse_reasoning_effort(effort)
            if parsed is not None:
                result["reasoning"] = parsed
            else:
                logger.warning(
                    "auxiliary.%s.reasoning_effort %r is not a valid level "
                    "(none, minimal, low, medium, high, xhigh, max, ultra) — ignoring",
                    task, effort,
                )
    return result


# Per-task concurrency limiting: many sessions can spawn unbounded background aux
# calls, each retrying across the fallback chain during incidents.
_aux_sync_semaphores: Dict[str, Tuple[int, threading.BoundedSemaphore]] = {}
_aux_async_semaphores: Dict[Tuple[str, int], Tuple[int, Any]] = {}
_aux_sem_lock = threading.Lock()


def _get_task_max_concurrency(task: Optional[str]) -> Optional[int]:
    """Return ``auxiliary.<task>.max_concurrency`` as a positive int, or None."""
    if not task or task == "vision":
        # Vision uses this key for its encode/resize CPU pool; its LLM calls stay concurrent.
        return None
    raw = _get_auxiliary_task_config(task).get("max_concurrency")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _cached_semaphore(store: dict, key: Any, limit: int, factory: Callable[[int], Any]) -> Any:
    """Return the cached semaphore for ``key``, rebuilding it when the limit changed."""
    with _aux_sem_lock:
        entry = store.get(key)
        if entry is None or entry[0] != limit:
            store[key] = entry = (limit, factory(limit))
        return entry[1]


def _acquire_sync_aux_semaphore(task: Optional[str]) -> Optional[threading.BoundedSemaphore]:
    """Get a per-task sync semaphore, rebuilding it after a config change."""
    limit = _get_task_max_concurrency(task)
    if limit is None:
        return None
    return _cached_semaphore(_aux_sync_semaphores, task, limit, threading.BoundedSemaphore)


def _acquire_async_aux_semaphore(task: Optional[str]):
    """Get a per-task, per-event-loop async semaphore after config lookup."""
    limit = _get_task_max_concurrency(task)
    if limit is None:
        return None
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return _cached_semaphore(_aux_async_semaphores, (task, id(loop)), limit, asyncio.Semaphore)


def _reset_aux_semaphores() -> None:
    """Drop cached semaphores (test helper)."""
    with _aux_sem_lock:
        _aux_sync_semaphores.clear()
        _aux_async_semaphores.clear()


# Anthropic-compatible endpoints reached via the OpenAI SDK wrapper; their image
# content blocks must use Anthropic format.
_ANTHROPIC_COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})


def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """True for known Anthropic-compatible providers or any ``/anthropic`` URL path."""
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower


def _convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url``/``video_url`` blocks to Anthropic ``image``/``video``.

    Only list-content messages with such blocks change; everything else passes through.
    """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get("type") == "image_url":
                image_url_val = (block.get("image_url") or {}).get("url", "")
                if image_url_val.startswith("data:"):
                    header, _, b64data = image_url_val.partition(",")
                    media_type = "image/png"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        },
                    })
                changed = True
            elif block.get("type") == "video_url":
                # MiniMax's Anthropic-compatible endpoint expects type="video" (not
                # "video_url"/"input_video"); source shape mirrors the "image" block.
                # https://platform.minimax.io/docs/api-reference/text-anthropic-api
                video_url_val = (block.get("video_url") or {}).get("url", "")
                if video_url_val.startswith("data:"):
                    header, _, b64data = video_url_val.partition(",")
                    media_type = "video/mp4"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    new_content.append({
                        "type": "video",
                        "source": {
                            "type": "url",
                            "url": video_url_val,
                        },
                    })
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted


_PROFILE_REASONING_KEYS = {
    "reasoning",
    "reasoning_effort",
    "thinking",
    "thinking_config",
    "thinkingconfig",
    "thinking_budget",
    "thinkingbudget",
    "enable_thinking",
    "think",
    "verbosity",
}


def _contains_profile_reasoning_fields(value: Any) -> bool:
    """Return whether a profile payload contains a reasoning wire control."""
    if not isinstance(value, dict):
        return False
    for key, nested in value.items():
        normalized = str(key).strip().lower()
        if normalized in _PROFILE_REASONING_KEYS:
            return True
        if _contains_profile_reasoning_fields(nested):
            return True
    return False


_NOUS_PROVIDER_NAMES = frozenset({"nous", "nous-portal", "nousresearch"})


def _nous_on_messages_wire(provider_norm: str, model: str) -> bool:
    """True when a Nous Portal route serves ``model`` over /v1/messages (dual-wire catalog)."""
    if provider_norm not in _NOUS_PROVIDER_NAMES:
        return False
    from hermes_cli.providers import nous_api_mode

    return nous_api_mode(model) == "anthropic_messages"


def _forwards_max_tokens(
    provider: str, provider_norm: str, model: str, effective_base: str, task: Optional[str],
) -> bool:
    """Whether an explicit max_tokens is forwarded on this route.

    No default output cap: omitted max_tokens means "model's max output" on most
    providers and sidesteps wire quirks (max_completion_tokens on GPT-5/Copilot,
    ZAI vision rejecting it). Forward only where mandatory or meaningfully honored:
    Anthropic Messages wire (hard 400 without it); NVIDIA NIM (some models return
    200 with empty choices[] when omitted); MoA reference slots; Gemini native
    (fixed 65,535 ceiling when omitted, so MoA reference_max_tokens needs the cap);
    OpenRouter (budgets credit against the FULL output window when omitted → 402
    on low-credit accounts); managed local llama-server (uncapped decode with no
    EOS burns the GPU to the full context window).
    """
    if _is_anthropic_compat_endpoint(provider, effective_base):
        return True
    if _nous_on_messages_wire(provider_norm, model):
        return True
    if (
        provider_norm in {"nvidia", "nvidia-nim", "nim", "build-nvidia", "nemotron"}
        or base_url_host_matches(effective_base, "integrate.api.nvidia.com")
    ):
        return True
    if bool(task) and str(task) == "moa_reference":
        return True
    is_gemini_native = provider_norm in {"gemini", "google", "google-gemini", "google-ai-studio"}
    if not is_gemini_native and effective_base:
        try:
            from agent.gemini_native_adapter import is_native_gemini_base_url
            is_gemini_native = is_native_gemini_base_url(effective_base)
        except Exception:
            pass
    if is_gemini_native:
        return True
    if provider_norm == "openrouter" or base_url_host_matches(effective_base, "openrouter.ai"):
        return True
    return _is_managed_local_endpoint(effective_base)


def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    reasoning_config: Optional[dict] = None,
    base_url: Optional[str] = None,
    task: Optional[str] = None,
) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }

    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        temperature = None  # strip — let server choose
    elif fixed_temperature is not None:
        temperature = fixed_temperature

    # Opus 4.7+ rejects any non-default temperature/top_p/top_k; drop silently so
    # aux callers that hardcode temperature don't 400 when the aux model flips.
    if temperature is not None:
        from agent.anthropic_adapter import _forbids_sampling_params
        if _forbids_sampling_params(model):
            temperature = None

    if temperature is not None:
        kwargs["temperature"] = temperature

    effective_base = base_url or (
        _current_custom_base_url() if provider == "custom" else ""
    )
    provider_norm = str(provider or "").strip().lower()

    if max_tokens is not None and _forwards_max_tokens(
        provider, provider_norm, model, effective_base, task,
    ):
        # auxiliary_max_tokens_param() picks max_completion_tokens where needed.
        kwargs.update(auxiliary_max_tokens_param(max_tokens, model=model))

    if tools:
        # Vertex/Azure/Bedrock 400 on duplicate tool names; upstream dedups
        # already, this turns a regression into a warning instead of a hard fail.
        _seen: set = set()
        _deduped: list = []
        for _t in tools:
            _tname = (_t.get("function") or {}).get("name", "")
            if _tname and _tname in _seen:
                logger.warning(
                    "_build_call_kwargs: duplicate tool name '%s' removed "
                    "(provider=%s model=%s)",
                    _tname, provider, model,
                )
                continue
            if _tname:
                _seen.add(_tname)
            _deduped.append(_t)
        kwargs["tools"] = _deduped

    # Provider profiles are the source of truth for reasoning wire shapes
    # (top-level, nested body, or extra_body.reasoning); providers without a
    # reasoning-aware profile keep the generic ``extra_body.reasoning`` fallback.
    profile_body: Dict[str, Any] = {}
    profile_reasoning_extra: Dict[str, Any] = {}
    profile_top_level: Dict[str, Any] = {}
    profile_handles_reasoning = False
    try:
        from providers import get_provider_profile
        from providers.base import ProviderProfile

        profile = get_provider_profile(provider_norm)
        if profile is not None:
            profile_body = profile.build_extra_body(
                model=model,
                base_url=effective_base,
                reasoning_config=reasoning_config,
            ) or {}
            profile_reasoning_extra, profile_top_level = (
                profile.build_api_kwargs_extras(
                    reasoning_config=reasoning_config,
                    supports_reasoning=reasoning_config is not None,
                    model=model,
                    base_url=effective_base,
                )
            )
            profile_reasoning_extra = profile_reasoning_extra or {}
            profile_top_level = profile_top_level or {}
            profile_handles_reasoning = (
                type(profile).build_api_kwargs_extras
                is not ProviderProfile.build_api_kwargs_extras
                or _contains_profile_reasoning_fields(profile_body)
                or _contains_profile_reasoning_fields(profile_reasoning_extra)
                or _contains_profile_reasoning_fields(profile_top_level)
            )
    except Exception as exc:
        logger.debug(
            "_build_call_kwargs: provider profile projection failed for %s: %s",
            provider,
            exc,
        )

    kwargs.update(profile_top_level)
    merged_extra = dict(extra_body or {})
    merged_extra.update(profile_body)
    merged_extra.update(profile_reasoning_extra)
    if (
        reasoning_config
        and isinstance(reasoning_config, dict)
        and not profile_handles_reasoning
    ):
        if reasoning_config.get("enabled") is False:
            merged_extra["reasoning"] = {"enabled": False}
        else:
            effort = reasoning_config.get("effort") or "medium"
            merged_extra["reasoning"] = {"enabled": True, "effort": effort}
    # Portal tags + sticky session_id fallback when the profile didn't supply
    # them; session_id keeps aux calls on the main turn's upstream instance
    # (cache warmth) — tags alone are not enough on /v1/messages.
    if provider_norm in _NOUS_PROVIDER_NAMES:
        if "tags" not in merged_extra:
            merged_extra["tags"] = _nous_portal_tags()
        if "session_id" not in merged_extra:
            try:
                from agent.portal_tags import get_conversation_context

                sticky_key = get_conversation_context()
            except Exception:
                sticky_key = None
            if sticky_key:
                merged_extra["session_id"] = sticky_key
    if merged_extra:
        kwargs["extra_body"] = merged_extra

    # Anthropic Messages adapters take reasoning via a private kwarg that plain
    # OpenAI SDK clients would reject; Portal Claude is dual-wire, so include it
    # only when the catalog id selects /v1/messages.
    if reasoning_config and isinstance(reasoning_config, dict):
        raw_base = base_url or ""
        if (
            provider_norm == "anthropic"
            or _nous_on_messages_wire(provider_norm, model)
            or _endpoint_speaks_anthropic_messages(raw_base)
            or _is_anthropic_compat_endpoint(provider_norm, raw_base)
        ):
            kwargs["_reasoning_config"] = dict(reasoning_config)

    return kwargs


def _validate_llm_response(
    response: Any,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Validate that an LLM response has the expected .choices[0].message shape.

    Fails fast instead of letting malformed payloads crash downstream with a
    misleading AttributeError. Also the single accounting chokepoint for aux
    usage (``agent.aux_accounting``): every successful non-streaming response
    passes here exactly once. Recording is best-effort; *provider*/*base_url*
    are optional accounting hints.
    """
    if response is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned None response"
        )
    from agent.aux_accounting import record_aux_usage
    record_aux_usage(response, task, provider=provider, base_url=base_url)
    # Adapter SimpleNamespace responses are fine — they have .choices[0].message.
    try:
        choices = response.choices
        if not choices or not hasattr(choices[0], "message"):
            raise AttributeError("missing choices[0].message")
    except (AttributeError, TypeError, IndexError) as exc:
        recovered = _recover_aux_response_message(response)
        if recovered is not None:
            _record_relay_auxiliary_response_model(response)
            _complete_relay_auxiliary_call()
            return recovered
        response_type = type(response).__name__
        response_preview = str(response)[:120]
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned invalid response "
            f"(type={response_type}): {response_preview!r}. "
            f"Expected object with .choices[0].message — check provider "
            f"adapter or custom endpoint compatibility."
        ) from exc
    _record_relay_auxiliary_response_model(response)
    _complete_relay_auxiliary_call()
    return response


def _complete_relay_auxiliary_call(*, outcome: str = "success") -> None:
    """Close one auxiliary logical call after acceptance or terminal failure."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    from agent import relay_llm

    relay_llm.complete_logical_call(
        str(context.get("request_id") or ""),
        outcome=outcome,
        model_name=str(context.get("model") or "unknown"),
        provider_name=str(context.get("provider") or "auxiliary"),
        response_model_name=context.get("response_model"),
    )


def _record_relay_auxiliary_response_model(response: Any) -> None:
    """Retain the provider-reported model for terminal route attribution."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    if isinstance(response, dict):
        model = response.get("model")
    else:
        model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        context["response_model"] = model


def _fail_relay_auxiliary_call() -> None:
    """Close a terminally failed call without replacing its original error."""
    try:
        _complete_relay_auxiliary_call(outcome="failed")
    except Exception:
        logger.warning(
            "Relay auxiliary failure finalization failed",
            exc_info=True,
        )


def _recover_aux_response_message(response: Any) -> Optional[Any]:
    """Synthesize chat-completions shape from Responses-style text fields.

    Some compatible endpoints return text outside ``choices`` (``output_text``,
    ``output`` items); preserve it before declaring the response malformed.
    """
    text = _extract_aux_response_text(response)
    if not text:
        return None

    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=getattr(response, "finish_reason", None) or "stop",
    )
    try:
        response.choices = [choice]
        return response
    except Exception:
        return SimpleNamespace(
            id=getattr(response, "id", ""),
            model=getattr(response, "model", ""),
            object=getattr(response, "object", "chat.completion"),
            choices=[choice],
            usage=getattr(response, "usage", None),
        )


def _extract_aux_response_text(response: Any) -> str:
    output_text = _field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _field(response, "output")
    if not isinstance(output, list):
        return ""

    parts: List[str] = []
    for item in output:
        item_type = _field(item, "type")
        if item_type and item_type != "message":
            continue
        for part in (_field(item, "content") or []):
            part_type = _field(part, "type")
            if part_type in {"output_text", "text", None}:
                text = _field(part, "text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n".join(parts).strip()


# ── Streamed aggregation for progress-hooked auxiliary calls ─────────────
# With a progress hook installed (today: context compression), the primary
# attempt streams and re-aggregates: ``timeout`` becomes an inter-chunk idle
# timeout (httpx read timeout is per read) and each chunk ticks outer watchdogs.
# _aux_stream_total_ceiling() still bounds a 1-token-per-idle-window stream.

_AUX_STREAM_CEILING_FLOOR_SECONDS = 600.0
_AUX_STREAM_CEILING_MULTIPLIER = 4.0


def _aux_stream_total_ceiling(effective_timeout: Optional[float]) -> float:
    """Absolute wall-clock bound for a progress-hooked streamed aux call.

    Generous by design: the idle timeout is the real guard; this only stops a
    stream trickling one token per idle window forever.
    """
    try:
        timeout = float(effective_timeout) if effective_timeout is not None else 0.0
    except (TypeError, ValueError):
        timeout = 0.0
    return max(_AUX_STREAM_CEILING_FLOOR_SECONDS,
               _AUX_STREAM_CEILING_MULTIPLIER * timeout)


def _client_streams_internally(client: Any) -> bool:
    """Adapters that stream inside .create() tick the hook themselves (Codex,
    Anthropic) or cannot stream (Bedrock); none accept ``stream=True`` from us."""
    return isinstance(client, (
        CodexAuxiliaryClient,
        AnthropicAuxiliaryClient,
        BedrockAuxiliaryClient,
    ))


_MANAGED_LOCAL_STATE_TTL_S = 15.0
_managed_local_cache: "tuple[float, str]" = (0.0, "")


def _managed_local_netloc() -> str:
    """host:port of the managed local llama-server, or "" when none.

    Read from the supervisor's state file with a short TTL; same source provider
    resolution uses, so the match is exact (no false positives on localhost).
    """
    global _managed_local_cache
    now = time.monotonic()
    ts, cached = _managed_local_cache
    if now - ts < _MANAGED_LOCAL_STATE_TTL_S:
        return cached
    netloc = ""
    try:
        from hermes_cli.local_runtime.supervisor import state_path

        raw = state_path().read_text(encoding="utf-8")
        base = str((json.loads(raw) or {}).get("base_url", ""))
        netloc = urlparse(base).netloc.lower()
    except Exception:
        netloc = ""
    _managed_local_cache = (now, netloc)
    return netloc


def _is_managed_local_endpoint(base_url: Optional[str]) -> bool:
    """True when *base_url* targets the llama-server this Hermes manages."""
    if not base_url:
        return False
    managed = _managed_local_netloc()
    if not managed:
        return False
    try:
        return urlparse(str(base_url)).netloc.lower() == managed
    except Exception:
        return False


def _provider_requires_stream(provider: str, base_url: Optional[str]) -> bool:
    """Detect providers that only accept streaming (non-stream = HTTP 400).

    Known hosts (Tencent Copilot) plus any URL substring listed in
    ``auxiliary.stream_only_base_urls``. The managed local llama-server is
    streamed for cancellation: it only notices a dead client on socket write, so
    a non-streamed abandoned request decodes to the end of the context window.
    """
    _url = str(base_url or "").lower()
    if not _url:
        return False
    if base_url_host_matches(_url, "copilot.tencent.com"):
        return True
    if _is_managed_local_endpoint(_url):
        return True
    try:
        from hermes_cli.config import load_config
        aux_cfg = (load_config() or {}).get("auxiliary", {})
        markers = aux_cfg.get("stream_only_base_urls") or []
        if isinstance(markers, (list, tuple)):
            for marker in markers:
                if isinstance(marker, str) and marker.strip() and marker.strip().lower() in _url:
                    return True
    except Exception:
        # Config read is best-effort; never break an aux call over it.
        pass
    return False


_AFFORDABLE_TOKENS_RE = re.compile(
    r"can only afford\s+([0-9][0-9,]*)", re.IGNORECASE
)

# Below this the affordable budget can't fit a useful aux output — treat as exhaustion.
_AFFORDABLE_RETRY_FLOOR_TOKENS = 512
# Headroom so provider-side token-count rounding can't 402 the retry.
_AFFORDABLE_RETRY_MARGIN_TOKENS = 64


def _affordable_max_tokens_from_error(exc: Exception) -> Optional[int]:
    """Extract the affordable output budget from a credit-limited 402.

    OpenRouter's rejection states it ("...but can only afford 7117"): the account
    HAS credit, the cap was just too large. Returns affordable minus a margin, or
    ``None`` when no count is present or the budget is too small to be useful.
    """
    if not _is_payment_error(exc):
        return None
    match = _AFFORDABLE_TOKENS_RE.search(str(exc))
    if not match:
        return None
    try:
        affordable = int(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None
    capped = affordable - _AFFORDABLE_RETRY_MARGIN_TOKENS
    if capped < _AFFORDABLE_RETRY_FLOOR_TOKENS:
        return None
    return capped


def _create_with_progress(
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str] = None,
    *,
    force_stream: bool = False,
) -> Any:
    """Credit-aware wrapper over :func:`_create_with_progress_once`.

    A 402 naming an affordable budget is not terminal exhaustion: retry ONCE with
    the provider-stated cap (only ever lowering an existing cap). Anything else
    re-raises for the normal recovery chains.
    """
    try:
        return _create_with_progress_once(
            client, kwargs, task, force_stream=force_stream,
        )
    except Exception as exc:
        affordable = _affordable_max_tokens_from_error(exc)
        if affordable is None:
            raise
        existing_cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if isinstance(existing_cap, (int, float)) and 0 < existing_cap <= affordable:
            # Already within budget — the error is something else; don't spin.
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("max_tokens", None)
        retry_kwargs.pop("max_completion_tokens", None)
        retry_kwargs.update(
            auxiliary_max_tokens_param(
                affordable, model=str(kwargs.get("model") or "") or None,
            )
        )
        logger.info(
            "Auxiliary %s: credit-limited 402 (affordable=%d tokens); "
            "retrying once with a clamped output cap instead of failing: %s",
            task or "call", affordable, exc,
        )
        return _create_with_progress_once(
            client, retry_kwargs, task, force_stream=force_stream,
        )


def _create_with_progress_once(
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str] = None,
    *,
    force_stream: bool = False,
) -> Any:
    """chat.completions.create() that streams when a progress hook is active
    or the provider only accepts streamed requests.

    Identical to plain ``create(**kwargs)`` when neither trigger applies or the
    adapter streams internally. Otherwise sends ``stream=True`` and aggregates,
    ticking the hook for substantive chunks. Streaming rejections fall back to a
    plain call — except under ``force_stream``, where the original error surfaces.
    """
    _notify_aux_dispatch()
    _notify_aux_progress()  # Preserve the watchdog's historical dispatch tick.
    if (not _aux_progress_active() and not force_stream) or _client_streams_internally(client):
        response = client.chat.completions.create(**kwargs)
        if not _client_streams_internally(client):
            _notify_aux_provider_response()
        return response

    total_ceiling = _aux_stream_total_ceiling(kwargs.get("timeout"))
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    try:
        chunks = client.chat.completions.create(**stream_kwargs)
    except Exception as exc:
        # Genuine provider failures aren't streaming's fault — surface unchanged
        # so the existing recovery chains see the same error as a plain call.
        if (
            force_stream
            or _is_transient_transport_error(exc)
            or _is_auth_error(exc)
            or _is_payment_error(exc)
            or _is_rate_limit_error(exc)
        ):
            raise
        # Possibly a streaming-specific rejection: retry non-streaming once; a
        # genuinely bad request reproduces the real error for the except-chains.
        logger.debug(
            "Auxiliary %s: streamed request failed (%s); retrying "
            "non-streaming", task or "call", exc,
        )
        _notify_aux_dispatch()
        response = client.chat.completions.create(**kwargs)
        _notify_aux_provider_response()
        return response

    # Some shims (MoA quiet mode, defensive adapters) return a complete response
    # despite stream=True; it counts as provider response + forward progress.
    if hasattr(chunks, "choices"):
        _notify_aux_provider_response()
        return chunks
    return _aggregate_chat_stream(
        chunks, model=str(kwargs.get("model") or ""), total_ceiling=total_ceiling,
    )


def _aggregate_chat_stream(
    chunks: Any,
    *,
    model: str = "",
    total_ceiling: Optional[float] = None,
) -> Any:
    """Consume a chat.completions chunk stream into a complete response.

    Ticks the aux progress hook only for substantive fragments. Raises
    TimeoutError (phrased "timed out" so ``_is_timeout_error`` matches) when
    *total_ceiling* elapses. Accumulation shared via :class:`_ChatStreamAccumulator`.
    """
    acc = _ChatStreamAccumulator(
        model=model,
        total_ceiling=total_ceiling,
        host_deadline=_current_aux_stream_deadline(),
    )
    try:
        for chunk in chunks:
            acc.feed(chunk)
    finally:
        close_fn = getattr(chunks, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    return acc.finish()


class _ChatStreamAccumulator:
    """Shared per-chunk accumulation so sync and async aggregation cannot drift."""

    def __init__(
        self,
        model: str = "",
        total_ceiling: Optional[float] = None,
        host_deadline: Optional[float] = None,
    ):
        self._started = time.monotonic()
        self._total_ceiling = total_ceiling
        # Absolute instant the waiting host gives up; checked alongside (not
        # instead of) the ceiling, and unaffected by pre-construction dispatch/TTFT.
        self._host_deadline = host_deadline
        self.content_parts: List[str] = []
        self.reasoning_parts: List[str] = []
        self.reasoning_details: List[Any] = []
        self.tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        self.finish_reason = None
        self.usage = None
        self.resp_id = ""
        self.resp_model = model or ""

    def feed(self, chunk: Any) -> None:
        # Every frame records transport timing (TTFP); only a substantive
        # payload ticks the forward-progress hook that keeps compression alive.
        _notify_aux_timing_response()
        made_progress = False
        if (
            self._total_ceiling is not None
            and (time.monotonic() - self._started) >= self._total_ceiling
        ):
            raise TimeoutError(
                f"Auxiliary streamed call timed out after {self._total_ceiling:.0f}s "
                "total ceiling (stream still open but over budget)"
            )
        if (
            self._host_deadline is not None
            and time.monotonic() >= self._host_deadline
        ):
            raise TimeoutError(
                "Auxiliary streamed call timed out at the host compression "
                f"deadline after {time.monotonic() - self._started:.0f}s "
                "(the caller already stopped waiting; streaming on would only "
                "pin its session lease)"
            )
        self.resp_id = getattr(chunk, "id", None) or self.resp_id
        self.resp_model = getattr(chunk, "model", None) or self.resp_model
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            self.usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        piece = getattr(delta, "content", None)
        if piece:
            self.content_parts.append(piece)
            made_progress = True
        reasoning_piece = (
            getattr(delta, "reasoning", None)
            or getattr(delta, "reasoning_content", None)
        )
        if reasoning_piece and isinstance(reasoning_piece, str):
            self.reasoning_parts.append(reasoning_piece)
            made_progress = True
        # OpenRouter-style models may stream thinking via ``reasoning_details``;
        # only details with actual text count as progress, so structural/signed
        # envelopes can't keep a stalled compression alive.
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details is None:
            model_extra = getattr(delta, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_details = model_extra.get("reasoning_details")
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                self.reasoning_details.append(detail)
                if isinstance(detail, dict) and any(
                    isinstance(detail.get(field), str) and detail[field]
                    for field in ("summary", "thinking", "content", "text")
                ):
                    made_progress = True
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            acc = self.tool_calls_acc.setdefault(
                idx, {"id": "", "name": "", "arguments": []}
            )
            tool_fragment = False
            if getattr(tc, "id", None):
                acc["id"] = tc.id
                tool_fragment = True
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                    tool_fragment = True
                if getattr(fn, "arguments", None):
                    acc["arguments"].append(fn.arguments)
                    tool_fragment = True
            made_progress = made_progress or tool_fragment

        if made_progress:
            _notify_aux_progress()

    def finish(self) -> Any:
        tool_calls = None
        if self.tool_calls_acc:
            tool_calls = [
                SimpleNamespace(
                    id=acc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=acc["name"],
                        arguments="".join(acc["arguments"]),
                    ),
                )
                for _idx, acc in sorted(self.tool_calls_acc.items())
            ]
        message = SimpleNamespace(
            role="assistant",
            content="".join(self.content_parts),
            tool_calls=tool_calls,
            reasoning="".join(self.reasoning_parts) or None,
            reasoning_details=self.reasoning_details or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason=self.finish_reason or "stop",
        )
        return SimpleNamespace(
            id=self.resp_id,
            model=self.resp_model,
            object="chat.completion",
            choices=[choice],
            usage=self.usage,
        )


async def _aggregate_chat_stream_async(
    chunks: Any,
    *,
    model: str = "",
    total_ceiling: Optional[float] = None,
) -> Any:
    """Async mirror of :func:`_aggregate_chat_stream` (AsyncOpenAI streams need ``async for``)."""
    acc = _ChatStreamAccumulator(
        model=model,
        total_ceiling=total_ceiling,
        host_deadline=_current_aux_stream_deadline(),
    )
    try:
        async for chunk in chunks:
            acc.feed(chunk)
    finally:
        close_fn = getattr(chunks, "close", None) or getattr(chunks, "aclose", None)
        if callable(close_fn):
            try:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
    return acc.finish()


async def _acreate_with_stream(
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str] = None,
) -> Any:
    """Async chat.completions.create() for stream-only providers: sends
    ``stream=True`` and aggregates the async chunk stream."""
    total_ceiling = _aux_stream_total_ceiling(kwargs.get("timeout"))
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    chunks = await client.chat.completions.create(**stream_kwargs)
    # Defensive: shims may hand back a complete response despite stream=True.
    if hasattr(chunks, "choices"):
        return chunks
    return await _aggregate_chat_stream_async(
        chunks, model=str(kwargs.get("model") or ""), total_ceiling=total_ceiling,
    )


# ── Shared request head + recovery ladder for call_llm / async_call_llm ────────
# The sync and async entry points differ only in how a provider request is
# awaited. Route resolution (``_resolve_call_client``) and the ordered recovery
# ladder (``_aux_recovery_ladder``) are therefore written once; the ladder is a
# generator that yields ``_LadderStep`` requests and receives the response (or
# has the exception thrown back in), so rung ORDER and each rung's
# accept/re-raise contract are identical on both wires by construction.

class _ResolvedAuxRoute(NamedTuple):
    client: Any
    final_model: Optional[str]
    resolved_provider: str
    effective_provider: str


def _resolve_call_client(
    task: Optional[str],
    *,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    main_runtime: Optional[Dict[str, Any]],
    async_mode: bool,
) -> _ResolvedAuxRoute:
    """Resolve the client for one aux call: vision chain, or cached text client with
    the explicit-provider fallback_chain / auto-chain rescue. Raises RuntimeError
    with the user-facing setup hint when nothing is configured."""
    effective_provider = resolved_provider
    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model,
            base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key,
            async_mode=async_mode,
            main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=async_mode,
                main_runtime=main_runtime,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=async_mode,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
            task=task,
        )
        effective_provider = _effective_provider_for_client(
            client, resolved_provider,
        )
        if client is None:
            # Explicit provider with no credentials: honor the task fallback_chain
            # before raising (fallback entries may use OAuth / credential-pool auth).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                    task, _explicit,
                )
                if fb_client is not None:
                    client, final_model = fb_client, fb_model
                    if async_mode:
                        client, final_model = _to_async_client(
                            fb_client, fb_model or "", is_vision=(task == "vision")
                        )
                    resolved_provider = fb_label or resolved_provider
                    effective_provider = resolved_provider
                else:
                    raise RuntimeError(
                        f"Provider '{_explicit}' is set in config.yaml but no API key "
                        f"was found. Set the {_explicit.upper()}_API_KEY environment "
                        f"variable, or switch to a different provider with `hermes model`."
                    )
            # Auto/custom with no credentials: walk the full auto chain (not just
            # OpenRouter). model=None so each provider uses its own default.
            if client is None and not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client(
                    "auto", async_mode=async_mode, main_runtime=main_runtime, task=task,
                )
                effective_provider = _effective_provider_for_client(
                    client, "auto",
                )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: hermes setup")

    return _ResolvedAuxRoute(client, final_model, resolved_provider, effective_provider)


class _PreparedAuxRequest(NamedTuple):
    client: Any
    final_model: Optional[str]
    kwargs: Dict[str, Any]
    resolved_provider: str
    request_provider: str
    resolved_model: Optional[str]
    resolved_base_url: Optional[str]
    resolved_api_key: Optional[str]
    resolved_api_mode: Optional[str]
    effective_timeout: float
    effective_extra_body: Dict[str, Any]
    base_info: str


def _prepare_aux_request(
    task: Optional[str],
    *,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    main_runtime: Dict[str, Any],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    timeout: Optional[float],
    extra_body: Optional[dict],
    reasoning_config: Optional[dict],
    extra_headers: Optional[Dict[str, str]],
    api_mode: Optional[str],
    route_info: Optional[Dict[str, str]],
    async_mode: bool,
) -> _PreparedAuxRequest:
    """Shared head of call_llm/async_call_llm: resolve route + client, publish it, build request kwargs.

    The sync wire additionally applies the certified compression fast lane and
    per-request ``extra_headers``; ``base_info`` is the client's base_url (sync
    falls back to the resolved base_url when the client exposes none).
    """
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = api_mode
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    client, final_model, resolved_provider, effective_provider = _resolve_call_client(
        task,
        provider=provider, model=model, base_url=base_url, api_key=api_key,
        resolved_provider=resolved_provider, resolved_model=resolved_model,
        resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
        resolved_api_mode=resolved_api_mode, main_runtime=main_runtime,
        async_mode=async_mode,
    )

    effective_timeout = _effective_aux_timeout(task, timeout)
    request_provider = effective_provider or resolved_provider
    fast_compression_cap = None
    if not async_mode:
        compression_config = (
            _get_auxiliary_task_config("compression") if task == "compression" else {}
        )
        fast_compression_cap, effective_extra_body = _compression_fast_lane_controls(
            task,
            actual_provider=request_provider,
            actual_model=final_model,
            requested_provider=provider,
            requested_model=model,
            route_config=compression_config,
            leak_guard_config=compression_config,
            max_tokens=max_tokens,
            extra_body=effective_extra_body,
        )
    _set_relay_auxiliary_route(request_provider, final_model, resolved_api_mode)
    _record_route_info(
        route_info, _fallback_provider_from_label(request_provider), final_model
    )

    if async_mode:
        base_info = str(getattr(client, "base_url", "") or "")
    else:
        base_info = str(getattr(client, "base_url", resolved_base_url) or "")
        if task:
            logger.info("Auxiliary %s: using %s (%s)%s",
                         task, request_provider or "auto", final_model or "default",
                         f" at {base_info}" if base_info and "openrouter" not in base_info else "")

    # Pass the client's actual base_url so endpoint-specific temperature overrides
    # work on auto-detected routes (api.moonshot.ai vs api.kimi.com/coding).
    kwargs = _build_call_kwargs(
        request_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=base_info or resolved_base_url, task=task)
    if fast_compression_cap is not None and max_tokens is None:
        # Narrow exception to "no cap" on aux calls: the compression route is
        # certified non-reasoning, so a bounded summary is intentional. Only fires
        # when the caller passed no max_tokens (explicit caps pass through untouched).
        kwargs.update(auxiliary_max_tokens_param(fast_compression_cap, model=final_model))
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)

    # Convert image blocks for Anthropic-compatible endpoints (e.g. MiniMax)
    client_base = str(getattr(client, "base_url", "") or "")
    if _is_anthropic_compat_endpoint(request_provider, client_base):
        kwargs["messages"] = _convert_openai_images_to_anthropic(kwargs["messages"])

    return _PreparedAuxRequest(
        client, final_model, kwargs, resolved_provider, request_provider,
        resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode,
        effective_timeout, effective_extra_body, base_info,
    )


class _LadderStep(NamedTuple):
    """A provider request the ladder asks its driver to perform.

    kind: "call" (client, kwargs) | "retry_same_provider" (provider, model) |
    "fallback" (fb_client, fb_model, fb_label).
    """
    kind: str
    args: tuple


_RERAISE_ORIGINAL = object()


# Ordered (predicate, reason) pairs for the provider-fallback rung: first match
# wins, so a payment-flavoured 429 reads as "payment error", not "rate limit".
_FALLBACK_REASONS: Tuple[Tuple[Callable[[Exception], bool], str], ...] = (
    (_is_auth_error, "auth error"),
    (_is_payment_error, "payment error"),
    (_is_rate_limit_error, "rate limit"),
    (_is_model_incompatible_error, "model incompatible with route"),
    (_is_invalid_aux_response_error, "invalid provider response"),
    (_is_connection_error, "connection error"),
)


def _fallback_reason(exc: Exception) -> Optional[str]:
    """Human-readable reason when ``exc`` warrants trying another provider, else None."""
    for predicate, reason in _FALLBACK_REASONS:
        if predicate(exc):
            return reason
    return None


def _rung(step: "_LadderStep", accept: Callable[[Exception], bool]):
    """One ladder rung: perform ``step``; yields ``(response, None)`` on success,
    ``(None, exc)`` when ``accept(exc)`` lets the next rung handle it, else re-raises."""
    try:
        result = yield step
    except Exception as exc:
        if not accept(exc):
            raise
        return None, exc
    return result, None


def _aux_recovery_ladder(
    first_err: Exception,
    *,
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str],
    async_mode: bool,
    base_info: str,
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    final_model: Optional[str],
    max_tokens: Optional[int],
    main_runtime: Optional[Dict[str, Any]],
    route_info: Optional[Dict[str, str]],
):
    """Ordered recovery rungs after the primary request failed (generator).

    Rungs, in order: temperature strip → structured-output strip → max_tokens
    strip → Nous stale-model self-heal → Nous paid/401 credential refresh →
    OAuth credential refresh + same-provider retry → credential-pool rotation →
    provider fallback (per-task chain, main fallback chain, discovery chain /
    main-agent-model net). Each rung either returns a response, narrows
    ``first_err`` and falls through, or re-raises. Returns ``_RERAISE_ORIGINAL``
    when every rung is exhausted (after evicting a connection-poisoned client).
    """
    tag = " (async)" if async_mode else ""

    def _call(target_client: Any, request_kwargs: Dict[str, Any]) -> _LadderStep:
        return _LadderStep("call", (target_client, request_kwargs))


    def _param_rung_accepts(exc: Exception) -> bool:
        # Fall through to the max_tokens/payment/auth chains with the stripped
        # kwargs; re-raise anything those chains won't handle.
        return (
            _is_payment_error(exc)
            or _is_connection_error(exc)
            or _is_auth_error(exc)
            or "max_tokens" in str(exc)
            or "unsupported_parameter" in str(exc)
        )

    def _capacity_rung_accepts(exc: Exception) -> bool:
        return _is_payment_error(exc) or _is_connection_error(exc) or _is_rate_limit_error(exc)

    def _credential_rung_accepts(exc: Exception) -> bool:
        return _is_auth_error(exc) or _is_payment_error(exc) or _is_rate_limit_error(exc)

    if "temperature" in kwargs and _is_unsupported_temperature_error(first_err):
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("temperature", None)
        logger.info(
            "Auxiliary %s%s: provider rejected temperature; retrying once without it",
            task or "call", tag,
        )
        resp, first_err = yield from _rung(_call(client, retry_kwargs), _param_rung_accepts)
        if first_err is None:
            return resp
        kwargs = retry_kwargs

    if _is_structured_output_rejection(first_err):
        retry_kwargs = _without_structured_output_format(kwargs)
        if retry_kwargs is not None:
            logger.info(
                "Auxiliary %s%s: provider rejected the structured-output "
                "format field; retrying once without it (schema "
                "enforcement degrades to prompt compliance): %s",
                task or "call", tag, first_err,
            )
            resp, first_err = yield from _rung(_call(client, retry_kwargs), _param_rung_accepts)
            if first_err is None:
                return resp
            kwargs = retry_kwargs

    err_str = str(first_err)
    # ZAI vision models reject max_tokens with code 1210 and a message that
    # never mentions "max_tokens", so detect it explicitly.
    _is_zai_param_error = (
        "1210" in err_str
        and "bigmodel" in str(getattr(client, "base_url", ""))
    )
    if max_tokens is not None and (
        "max_tokens" in err_str
        or "unsupported_parameter" in err_str
        or _is_unsupported_parameter_error(first_err, "max_tokens")
        or _is_zai_param_error
    ):
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        resp, first_err = yield from _rung(_call(client, kwargs), _capacity_rung_accepts)
        if first_err is None:
            return resp

    # ── Stale-model self-heal (Nous Portal recommendation drift) ───
    # A long-lived process can pin a Portal model since dropped from the catalog
    # (every call 404s); force a fresh Portal fetch and retry once. Nous-only.
    _heal_is_nous = (
        resolved_provider == "nous"
        or base_url_host_matches(base_info, "inference-api.nousresearch.com")
    )
    if _is_model_not_found_error(first_err) and _heal_is_nous:
        healed_model = _refresh_nous_recommended_model(
            vision=(task == "vision"), stale_model=kwargs.get("model"))
        if healed_model and healed_model != kwargs.get("model"):
            logger.warning(
                "Auxiliary %s%s: model %r no longer in Nous catalog; "
                "retrying with refreshed recommendation %r",
                task or "call", tag, kwargs.get("model"), healed_model,
            )
            kwargs["model"] = healed_model
            resp, first_err = yield from _rung(_call(client, kwargs), lambda exc: True)
            if first_err is None:
                return resp

    # ── Nous auth refresh parity with main agent ──────────────────
    client_is_nous = (
        resolved_provider == "nous"
        or base_url_host_matches(base_info, "inference-api.nousresearch.com")
    )
    if (
        _is_payment_error(first_err)
        and client_is_nous
        and _nous_portal_account_has_fresh_paid_access()
    ):
        refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
            cache_provider=resolved_provider or "nous",
            model=final_model,
            lookup_model=resolved_model,
            lookup_task=task,
            async_mode=async_mode,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
            is_vision=(task == "vision"),
        )
        if refreshed_client is not None:
            logger.info(
                "Auxiliary %s%s: refreshed Nous runtime credentials after paid account check, retrying",
                task or "call", tag,
            )
            if refreshed_model and refreshed_model != kwargs.get("model"):
                kwargs["model"] = refreshed_model
            resp, first_err = yield from _rung(
            _call(refreshed_client, kwargs),
            lambda exc: _credential_rung_accepts(exc) or _is_connection_error(exc),
        )
            if first_err is None:
                return resp

    if _is_auth_error(first_err) and client_is_nous:
        refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
            cache_provider=resolved_provider or "nous",
            model=final_model,
            lookup_model=resolved_model,
            lookup_task=task,
            async_mode=async_mode,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
            is_vision=(task == "vision"),
        )
        if refreshed_client is not None:
            logger.info("Auxiliary %s%s: refreshed Nous runtime credentials after 401, retrying",
                        task or "call", tag)
            if refreshed_model and refreshed_model != kwargs.get("model"):
                kwargs["model"] = refreshed_model
            return (yield _call(refreshed_client, kwargs))

    # ── Auth refresh retry ───────────────────────────────────────
    auth_refresh_provider = _auth_refresh_provider_for_route(
        resolved_provider, base_info)
    if (_is_auth_error(first_err)
            and auth_refresh_provider not in {"auto", "", None}
            and not client_is_nous):
        if _refresh_provider_credentials(auth_refresh_provider):
            if auth_refresh_provider != _normalize_aux_provider(resolved_provider):
                # The stale client is cached under the route label
                # (e.g. "auto"), not the concrete backend we refreshed.
                _evict_cached_clients(resolved_provider)
            logger.info(
                "Auxiliary %s%s: refreshed %s credentials after auth error, retrying",
                task or "call", tag, auth_refresh_provider,
            )
            return (yield _LadderStep("retry_same_provider", (auth_refresh_provider, resolved_model or final_model)))

    # ── Same-provider credential-pool recovery ─────────────────────
    pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime)
    # Capture the exact key used so recovery finds the right pool entry even if
    # another process rotated the pool meanwhile (current() would be None).
    _client_api_key = str(getattr(client, "api_key", "") or "")
    if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
        recovery_err = first_err
        # Skip the extra retry for clear payment/quota errors — the endpoint
        # won't accept another request with the same exhausted key.
        if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
            resp, recovery_err = yield from _rung(_call(client, kwargs), _credential_rung_accepts)
            if recovery_err is None:
                return resp
        if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
            logger.info(
                "Auxiliary %s%s: recovered %s via credential-pool rotation after %s",
                task or "call", tag, pool_provider, type(recovery_err).__name__,
            )
            try:
                return (yield _LadderStep("retry_same_provider", (resolved_provider, resolved_model)))
            except Exception as retry2_err:
                # Rotated key also hit a wall: mark it now so concurrent processes
                # skip it, then fall through to the payment fallback below.
                if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                        or _is_rate_limit_error(retry2_err)):
                    _recover_provider_pool(pool_provider, retry2_err)
                    first_err = retry2_err
                else:
                    raise

    # ── Payment / connection / rate-limit / auth fallback ─────────
    # Try alternative providers when the resolved one returns 402/credit
    # exhaustion, is unreachable, is rate-limited (429), or 401s past the
    # refresh paths. Auth is NOT a capacity error: it only bypasses the
    # explicit-provider gate in auto mode.
    # Capacity errors (payment/quota, connection, exhausted 429, model incompatible
    # with route, malformed response) bypass the explicit-provider gate: the
    # provider cannot serve this request regardless of user intent. Auth errors
    # are NOT capacity errors: they only fall back in auto mode.
    is_auto = resolved_provider in {"auto", "", None}
    reason = _fallback_reason(first_err)
    is_capacity_error = any(
        predicate(first_err) for predicate, label in _FALLBACK_REASONS if label != "auth error"
    )
    if reason is not None and (is_auto or is_capacity_error):
        if reason == "payment error":
            # Mark the concrete backend (not the "auto" label) unhealthy so
            # later aux calls skip it instead of paying another doomed RTT.
            _mark_provider_unhealthy(
                _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime) or resolved_provider
            )
        logger.info("Auxiliary %s%s: %s on %s (%s), trying fallback",
                    task or "call", tag, reason, resolved_provider, first_err)

        # Skip only the failed model for model-specific failures; 401/402 are
        # provider-wide, so keep skipping the whole provider.
        _chain_failed_model = (
            None if reason in ("auth error", "payment error") else final_model
        )
        # Fallback order: per-task fallback_chain; then for auto: main
        # fallback_providers, then built-in discovery chain; for explicit
        # providers: main agent model safety net.
        fb_client, fb_model, fb_label = (None, None, "")
        if is_auto:
            fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                task, resolved_provider or "auto", reason=reason,
                failed_model=_chain_failed_model)
            if fb_client is None:
                fb_client, fb_model, fb_label = _try_main_fallback_chain(
                    task, resolved_provider or "auto", reason=reason)
            if fb_client is None:
                fb_client, fb_model, fb_label = _try_payment_fallback(
                    resolved_provider, task, reason=reason)
        else:
            fb_client, fb_model, fb_label = _try_configured_fallback_chain(
                task, resolved_provider or "auto", reason=reason,
                failed_model=_chain_failed_model)
            if fb_client is None:
                fb_client, fb_model, fb_label = _try_main_agent_model_fallback(
                    resolved_provider, task, reason=reason,
                    failed_model=_chain_failed_model)

        if fb_client is not None:
            # Second pass: the candidate credential was stale and quarantined — walk
            # the discovery chain once more (unhealthy entries are skipped).
            for _pass in range(2):
                _record_route_info(
                    route_info, _fallback_provider_from_label(fb_label), fb_model
                )
                fb_resp = yield _LadderStep("fallback", (fb_client, fb_model, fb_label))
                if fb_resp is not None:
                    return fb_resp
                if _pass == 0:
                    fb_client, fb_model, fb_label = _try_payment_fallback(
                        resolved_provider, task, reason="stale fallback credential")
                    if fb_client is None:
                        break
        # All fallback layers exhausted — one user-visible warning, then re-raise.
        logger.warning(
            "Auxiliary %s%s: %s on %s and all fallbacks exhausted "
            "(fallback_chain + main agent model). Raising original error.",
            task or "call", tag, reason, resolved_provider,
        )
    # Connection/timeout errors poison the cached client (closed transport,
    # half-read stream); evict so the next aux call rebuilds a fresh one.
    if _is_connection_error(first_err):
        try:
            _evict_cached_client_instance(client)
        except Exception:
            logger.debug("Auxiliary%s: cache eviction after connection error failed",
                         tag, exc_info=True)
    return _RERAISE_ORIGINAL


def _drive_ladder(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Run a ladder generator, feeding each step's result (or exception) back in."""
    try:
        step = next(ladder)
        while True:
            try:
                result = perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value


async def _drive_ladder_async(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Async twin of :func:`_drive_ladder` (``perform`` is awaited)."""
    try:
        step = next(ladder)
        while True:
            try:
                result = await perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value


@_relay_auxiliary_call
def call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    api_mode: str = None,
    stream: bool = False,
    stream_options: dict = None,
    route_info: Optional[Dict[str, str]] = None,
    latency_info: Optional[Dict[str, int]] = None,
) -> Any:
    """Run an auxiliary LLM request, applying the configured task limit."""
    queue_started_at = time.monotonic()
    semaphore = _acquire_sync_aux_semaphore(task)
    if semaphore is not None:
        semaphore.acquire()
    request_started_at = time.monotonic()
    if latency_info is not None:
        latency_info["queue_wait_ms"] = max(
            0, int((request_started_at - queue_started_at) * 1000)
        )
    prior_progress_hook = getattr(_aux_progress, "hook", None)

    def _timed_response() -> None:
        if latency_info is not None and "time_to_first_progress_ms" not in latency_info:
            latency_info["time_to_first_progress_ms"] = max(
                0, int((time.monotonic() - request_started_at) * 1000)
            )

    def _timed_dispatch() -> None:
        if latency_info is not None and "provider_dispatch_ms" not in latency_info:
            latency_info["provider_dispatch_ms"] = max(
                0, int((time.monotonic() - request_started_at) * 1000)
            )

    try:
        with (
            aux_progress_hook(
                prior_progress_hook
                if callable(prior_progress_hook)
                else ((lambda: None) if latency_info is not None else None)
            ),
            _aux_timing_hook(_aux_dispatch, _timed_dispatch),
            _aux_timing_hook(_aux_provider_response, _timed_response),
        ):
            response = _call_llm_impl(
                task=task,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                main_runtime=main_runtime,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                timeout=timeout,
                extra_body=extra_body,
                reasoning_config=reasoning_config,
                extra_headers=extra_headers,
                api_mode=api_mode,
                stream=stream,
                stream_options=stream_options,
                route_info=route_info,
            )
        if stream and semaphore is not None:
            stream_semaphore = semaphore
            semaphore = None
            return _release_sync_semaphore_after_stream(response, stream_semaphore)
        return response
    finally:
        if latency_info is not None:
            latency_info["summary_generation_ms"] = max(
                0, int((time.monotonic() - request_started_at) * 1000)
            )
        if semaphore is not None:
            semaphore.release()


def _release_sync_semaphore_after_stream(
    stream: Any, semaphore: threading.BoundedSemaphore,
):
    """Release a permit only after a streaming response is consumed or closed."""
    try:
        yield from stream
    finally:
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        finally:
            semaphore.release()


def _call_llm_impl(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    api_mode: str = None,
    stream: bool = False,
    stream_options: dict = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized synchronous LLM call: resolve provider/model, auth, kwargs, fallbacks.

    task: aux task name whose provider:model is read from config (ignored if provider set).
    api_mode overrides task config; timeout=None reads auxiliary.{task}.timeout;
    extra_headers override client defaults (e.g. Copilot ``x-initiator``).
    stream=True returns the raw SDK stream iterator (caller consumes/falls back) instead
    of a validated response. Raises RuntimeError if no provider is configured.
    """
    # One immutable runtime snapshot for keying/resolution/retries/fallbacks, so a
    # concurrent /model switch can't mix key and client from different runtimes.
    main_runtime = _normalize_main_runtime(main_runtime)
    req = _prepare_aux_request(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        main_runtime=main_runtime, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
        reasoning_config=reasoning_config, extra_headers=extra_headers,
        api_mode=api_mode, route_info=route_info, async_mode=False,
    )
    client, final_model, kwargs = req.client, req.final_model, req.kwargs
    resolved_provider, request_provider = req.resolved_provider, req.request_provider
    resolved_model, resolved_base_url = req.resolved_model, req.resolved_base_url
    resolved_api_key, resolved_api_mode = req.resolved_api_key, req.resolved_api_mode
    effective_timeout, effective_extra_body = req.effective_timeout, req.effective_extra_body
    _base_info = req.base_info

    # Streaming path (MoA aggregator): return the raw SDK stream, deliberately
    # skipping validation and the fallback chain below — those assume a complete
    # response. The caller owns reassembly, stale-stream detection and fallback.
    if stream:
        kwargs["stream"] = True
        if stream_options:
            kwargs["stream_options"] = stream_options
        if task == "moa_aggregator" and isinstance(client, CodexAuxiliaryClient):
            # Responses-shim clients consume the stream internally and return a
            # completed object; Relay's managed stream would iterate that object
            # itself. Return directly — the MoA facade wraps it as a one-chunk stream.
            return client.chat.completions.create(**kwargs)
        return _relay_sync_stream(
            client,
            kwargs,
            provider=request_provider,
            api_mode=resolved_api_mode,
        )

    def _primary(**validate_kw: Any) -> Any:
        return _validate_llm_response(
            _relay_sync_completion(
                client,
                kwargs,
                provider=request_provider,
                api_mode=resolved_api_mode,
                create=lambda request: _create_with_progress(
                    client,
                    request,
                    task,
                    force_stream=_provider_requires_stream(
                        request_provider, _base_info or resolved_base_url,
                    ),
                ),
            ),
            task,
            **validate_kw,
        )

    try:
        # Bounded same-provider retry (exponential backoff, count from
        # auxiliary.transient_retries) for transient transport blips before the
        # except-chain escalates to fallback — a dropped connection shouldn't
        # abandon a healthy provider (matters for pinned MoA advisors).
        try:
            return _primary(provider=request_provider, base_url=_base_info)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            # Critical-path tasks skip the same-provider retry on a
            # full-budget timeout; see _should_skip_same_provider_retry.
            if _should_skip_same_provider_retry(task, transient_err):
                logger.info(
                    "Auxiliary %s: timeout on the critical path; "
                    "skipping same-provider retry and falling back: %s",
                    task, transient_err,
                )
                raise
            _max_transient_retries = _transient_retry_count()
            _last_transient = transient_err
            for _attempt in range(1, _max_transient_retries + 1):
                _backoff = min(_TRANSIENT_RETRY_BACKOFF_BASE * (2.0 ** (_attempt - 1)), 8.0)
                logger.info(
                    "Auxiliary %s: transient transport error (attempt %d/%d); "
                    "retrying same provider after %.1fs before fallback: %s",
                    task or "call", _attempt, _max_transient_retries, _backoff,
                    _last_transient,
                )
                time.sleep(_backoff)
                try:
                    return _primary()
                except Exception as retry_transient:
                    if not _is_transient_transport_error(retry_transient):
                        raise
                    _last_transient = retry_transient
            raise _last_transient
    except Exception as first_err:
        def _perform(step: _LadderStep) -> Any:
            if step.kind == "call":
                target_client, request_kwargs = step.args
                return _validate_llm_response(
                    _relay_sync_completion(
                        target_client, request_kwargs,
                        provider=resolved_provider, api_mode=resolved_api_mode,
                    ), task)
            if step.kind == "retry_same_provider":
                retry_provider, retry_model = step.args
                return _retry_same_provider_sync(
                    task=task,
                    resolved_provider=retry_provider,
                    resolved_model=retry_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    main_runtime=main_runtime,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                    reasoning_config=reasoning_config,
                    extra_headers=extra_headers,
                )
            fb_client, fb_model, fb_label = step.args
            return _call_fallback_candidate_sync(
                fb_client, fb_model, fb_label,
                task=task, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                tools=tools, effective_timeout=effective_timeout,
                effective_extra_body=effective_extra_body,
                reasoning_config=reasoning_config)

        result = _drive_ladder(
            _aux_recovery_ladder(
                first_err,
                client=client, kwargs=kwargs, task=task, async_mode=False,
                base_info=_base_info,
                resolved_provider=resolved_provider, resolved_model=resolved_model,
                resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
                resolved_api_mode=resolved_api_mode, final_model=final_model,
                max_tokens=max_tokens, main_runtime=main_runtime, route_info=route_info,
            ),
            _perform,
        )
        if result is _RERAISE_ORIGINAL:
            raise
        return result


def _coerce_llm_message(response):
    """Pull a message (dict, object, or str) out of a response-or-message value.

    Accepts dict-shaped responses/bare messages (compression, proxies) and
    ChatCompletion objects; MagicMock ``reasoning_*`` attrs are deliberately not strings.
    """
    if response is None or isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" not in response:
            return response
        choices = response.get("choices") or []
        if not choices:
            return None
        first = choices[0]
        return first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    choices = getattr(response, "choices", None)
    if not choices:
        return response
    first = choices[0]
    return first.get("message") if isinstance(first, dict) else getattr(first, "message", None)


def _message_field(msg, name):
    if isinstance(msg, dict):
        return msg.get(name)
    return getattr(msg, name, None)


def extract_content_or_reasoning(response, *, max_reasoning_chars: int | None = None) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.

    Order: ``content`` (inline think blocks stripped) → ``reasoning``/
    ``reasoning_content`` → ``reasoning_details`` (OpenRouter array). Accepts a
    response or bare message; ``max_reasoning_chars`` bounds a reasoning
    fallback so unbounded chain-of-thought can't become the compaction summary.
    Returns ``""`` if nothing found.
    """
    import re

    msg = _coerce_llm_message(response)
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()

    raw = _message_field(msg, "content")
    if not isinstance(raw, str):
        raw = str(raw) if raw else ""
    content = raw.strip()

    if content:
        # Mirrors _strip_think_blocks
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>"
            r".*?"
            r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned

    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = _message_field(msg, field)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())

    details = _message_field(msg, "reasoning_details")
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = (
                    detail.get("summary")
                    or detail.get("content")
                    or detail.get("text")
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))

    if not reasoning_parts:
        return ""

    text = "\n\n".join(reasoning_parts)
    if max_reasoning_chars is not None and len(text) > max_reasoning_chars:
        logger.warning(
            "fell back to reasoning fields (%d chars); truncating to %d",
            len(text),
            max_reasoning_chars,
        )
        return text[:max_reasoning_chars]
    return text


@_relay_auxiliary_call_async
async def async_call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Run an asynchronous auxiliary LLM request under the configured limit."""
    semaphore = _acquire_async_aux_semaphore(task)
    if semaphore is not None:
        await semaphore.acquire()
    try:
        return await _async_call_llm_impl(
            task=task,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            main_runtime=main_runtime,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            extra_body=extra_body,
            reasoning_config=reasoning_config,
            route_info=route_info,
        )
    finally:
        if semaphore is not None:
            semaphore.release()


async def _async_call_llm_impl(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized asynchronous LLM call; see call_llm() for full documentation."""
    # Keep every async phase on one runtime identity across awaits (concurrent /model switch).
    main_runtime = _normalize_main_runtime(main_runtime)
    extra_headers = None  # async entry point has no per-request header override
    req = _prepare_aux_request(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        main_runtime=main_runtime, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
        reasoning_config=reasoning_config, extra_headers=None,
        api_mode=None, route_info=route_info, async_mode=True,
    )
    client, final_model, kwargs = req.client, req.final_model, req.kwargs
    resolved_provider, request_provider = req.resolved_provider, req.request_provider
    resolved_model, resolved_base_url = req.resolved_model, req.resolved_base_url
    resolved_api_key, resolved_api_mode = req.resolved_api_key, req.resolved_api_mode
    effective_timeout, effective_extra_body = req.effective_timeout, req.effective_extra_body
    _client_base = req.base_info

    try:
        # Retry ONCE on the same provider for a transient blip before escalating
        # to fallback — see call_llm() for the rationale.
        _force_stream_async = (
            _provider_requires_stream(
                request_provider, _client_base or resolved_base_url,
            )
            and not isinstance(client, (
                AsyncCodexAuxiliaryClient,
                AsyncAnthropicAuxiliaryClient,
                AsyncBedrockAuxiliaryClient,
            ))
        )

        async def _acreate(_kwargs: Dict[str, Any]) -> Any:
            if _force_stream_async:
                return await _acreate_with_stream(client, _kwargs, task)
            return await client.chat.completions.create(**_kwargs)

        async def _primary(**validate_kw: Any) -> Any:
            return _validate_llm_response(
                await _relay_async_completion(
                    client,
                    kwargs,
                    provider=request_provider,
                    api_mode=resolved_api_mode,
                    create=_acreate,
                ),
                task,
                **validate_kw,
            )

        try:
            return await _primary(provider=request_provider, base_url=_client_base)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            # Same rule as call_llm(); the async Codex adapter wraps the sync
            # stream via to_thread, so the same TimeoutError reaches here.
            if _should_skip_same_provider_retry(task, transient_err):
                logger.info(
                    "Auxiliary %s (async): timeout on the critical "
                    "path; skipping same-provider retry and falling back: %s",
                    task, transient_err,
                )
                raise
            logger.info(
                "Auxiliary %s (async): transient transport error; retrying "
                "once on the same provider before fallback: %s",
                task or "call", transient_err,
            )
            return await _primary()
    except Exception as first_err:
        async def _perform(step: _LadderStep) -> Any:
            if step.kind == "call":
                target_client, request_kwargs = step.args
                return _validate_llm_response(
                    await _relay_async_completion(
                        target_client, request_kwargs,
                        provider=resolved_provider, api_mode=resolved_api_mode,
                    ), task)
            if step.kind == "retry_same_provider":
                retry_provider, retry_model = step.args
                return await _retry_same_provider_async(
                    task=task,
                    resolved_provider=retry_provider,
                    resolved_model=retry_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    main_runtime=main_runtime,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                    reasoning_config=reasoning_config,
                    extra_headers=extra_headers,
                )
            fb_client, fb_model, fb_label = step.args
            fb_client, _ = _to_async_client(
                fb_client, fb_model or "", is_vision=(task == "vision")
            )
            return await _call_fallback_candidate_async(
                fb_client, fb_model, fb_label,
                task=task, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                tools=tools, effective_timeout=effective_timeout,
                effective_extra_body=effective_extra_body,
                reasoning_config=reasoning_config)

        result = await _drive_ladder_async(
            _aux_recovery_ladder(
                first_err,
                client=client, kwargs=kwargs, task=task, async_mode=True,
                base_info=_client_base,
                resolved_provider=resolved_provider, resolved_model=resolved_model,
                resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
                resolved_api_mode=resolved_api_mode, final_model=final_model,
                max_tokens=max_tokens, main_runtime=main_runtime, route_info=route_info,
            ),
            _perform,
        )
        if result is _RERAISE_ORIGINAL:
            raise
        return result
