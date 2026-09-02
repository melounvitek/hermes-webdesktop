"""Structured error-surface descriptors for UI clients (Desktop/TUI).

Maps the internal failure taxonomy (``FailoverReason`` values carried in turn
results as ``failure_reason``, or raw exceptions from the turn dispatcher)
onto a small, stable wire descriptor::

    {"layer": <ui layer>, "code": <specific code>, "retryable": <bool>}

Layers (wire values): provider (model API rejected/failed the call), endpoint
(user-configured custom/local endpoint transport failure), streaming (SSE
dropped mid-turn), auth, billing (fallback signal; clients usually have a
richer ``billing_block``), gateway (local runtime errored), disk (disk full /
persistence failure).

Dependency-light and NEVER raises: surfacing diagnostics must not break the
error path it describes. Descriptors are advisory — clients fall back to
string sniffing when absent or partial.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

LAYER_PROVIDER = "provider"
LAYER_ENDPOINT = "endpoint"
LAYER_STREAMING = "streaming"
LAYER_AUTH = "auth"
LAYER_BILLING = "billing"
LAYER_GATEWAY = "gateway"
LAYER_DISK = "disk"

# failure_reason → UI layer. Unlisted reasons fall back to LAYER_PROVIDER:
# every FailoverReason comes from classifying a provider call.
_REASON_TO_LAYER = {
    "auth": LAYER_AUTH,
    "auth_permanent": LAYER_AUTH,
    "billing": LAYER_BILLING,
    "billing_unverified": LAYER_BILLING,
}

# Failures between us and the base_url (not a provider verdict); on a
# custom/local endpoint they point at the user's endpoint config.
_TRANSPORT_REASONS = {"timeout", "ssl_cert_verification"}

# Deterministic for the request — a bare "Retry" repeats the failure. Fallback
# only: current backends stamp the classifier's verdict in ``failure_retryable``.
# Kept in sync with ``classify_api_error``'s retryable=False verdicts.
_NON_RETRYABLE_REASONS = {
    "auth", "auth_permanent", "billing", "billing_unverified", "content_policy_blocked",
    "provider_policy_blocked", "model_not_found", "format_error", "ssl_cert_verification",
}

# Providers whose base_url is user-supplied rather than a known vendor.
_CUSTOM_ENDPOINT_PROVIDERS = {"custom", "local", "llama.cpp", "llamacpp", "ollama", "lmstudio", "vllm"}

# Mid-stream drop markers. Deliberately narrow: our own retry-exhaustion
# summaries plus the OpenAI SDK's stream-abort errors.
_STREAM_DROP_FRAGMENTS = (
    "stream connection", "peer closed connection", "incomplete chunked read",
    "connection broken", "stream ended prematurely", "sse", "mid-stream",
)

# Exception top-level modules that mean "API/transport call failed" (vs. a bug
# in our dispatcher = gateway layer): every SDK family our adapters raise from
# plus raw transports.
_API_EXC_MODULE_PREFIXES = (
    "openai", "httpx", "httpcore", "anthropic", "botocore", "boto3", "google",
    "grpc", "requests", "aiohttp", "ssl", "socket", "urllib",
)


def _is_custom_endpoint(provider: Optional[str]) -> bool:
    p = (provider or "").strip().lower()
    return p in _CUSTOM_ENDPOINT_PROVIDERS or p.startswith("custom:")


def _looks_like_stream_drop(message: str) -> bool:
    msg = message.lower()
    return any(fragment in msg for fragment in _STREAM_DROP_FRAGMENTS)


def _surface(layer: str, code: str, retryable: bool, provider: str = "", model: str = "") -> dict:
    out = {"layer": layer, "code": code, "retryable": bool(retryable)}
    # Identity captured at classification time, so clients report the session
    # that actually failed — not whatever the composer points at later.
    if provider:
        out["provider"] = provider
    if model:
        out["model"] = model
    return out


def _disk_full(candidate: Any) -> bool:
    try:
        from hermes_state import is_disk_full_error

        return bool(is_disk_full_error(candidate))
    except Exception:  # pragma: no cover - defensive import guard
        return False


def build_error_surface_from_result(result: Any, provider: str = "", model: str = "") -> Optional[dict]:
    """Descriptor for a returned-error turn result (``failed=True`` dicts).

    Uses the stamped ``failure_reason`` plus error text. None when the result
    carries no failure signal.
    """
    try:
        if not isinstance(result, dict):
            return None
        error_text = str(result.get("error") or "")
        reason = str(result.get("failure_reason") or "").strip()
        if not error_text and not reason:
            return None

        # Disk-full wins outright: the fix (free space) is unrelated to the
        # provider stack; hermes_state owns the pattern list.
        if error_text and _disk_full(error_text):
            return _surface(LAYER_DISK, "disk_full", False, provider, model)

        if result.get("billing_block") or reason in ("billing", "billing_unverified"):
            return _surface(LAYER_BILLING, reason or "billing", False, provider, model)

        if not reason:
            # Failed result without a classified reason (legacy paths).
            if _looks_like_stream_drop(error_text):
                return _surface(LAYER_STREAMING, "stream_drop", True, provider, model)
            return _surface(LAYER_PROVIDER, "unknown", True, provider, model)

        layer = _REASON_TO_LAYER.get(reason)
        if layer is None:
            if reason in _TRANSPORT_REASONS and _is_custom_endpoint(provider):
                layer = LAYER_ENDPOINT
            elif _looks_like_stream_drop(error_text):
                layer = LAYER_STREAMING
            else:
                layer = LAYER_PROVIDER
        # Prefer the classifier's own verdict (``failure_retryable``); the
        # reason-set fallback covers older results.
        retryable = result.get("failure_retryable")
        if not isinstance(retryable, bool):
            retryable = reason not in _NON_RETRYABLE_REASONS
        return _surface(layer, reason, retryable, provider, model)
    except Exception:  # pragma: no cover — never break the error path
        logger.debug("error_surface: result classification failed", exc_info=True)
        return None


def build_error_surface_from_exception(exc: BaseException, provider: str = "", model: str = "") -> Optional[dict]:
    """Descriptor for an exception that escaped the turn dispatcher.

    API/transport exceptions go through ``classify_api_error`` (same taxonomy
    as the retry loop); anything else is a gateway-layer failure.
    """
    try:
        message = str(exc) or type(exc).__name__

        if _disk_full(exc):
            return _surface(LAYER_DISK, "disk_full", False, provider, model)

        exc_module = type(exc).__module__ or ""
        api_like = exc_module.split(".")[0] in _API_EXC_MODULE_PREFIXES or hasattr(exc, "status_code")

        if not api_like or not isinstance(exc, Exception):
            return _surface(LAYER_GATEWAY, type(exc).__name__, True, provider, model)

        from agent.error_classifier import classify_api_error

        classified = classify_api_error(exc, provider=provider, model=model)
        synthetic = {"error": classified.message or message, "failure_reason": classified.reason.value}
        surface = build_error_surface_from_result(synthetic, provider=provider, model=model)
        if surface is not None:
            surface["retryable"] = bool(classified.retryable)
        return surface
    except Exception:  # pragma: no cover — never break the error path
        logger.debug("error_surface: exception classification failed", exc_info=True)
        return None
