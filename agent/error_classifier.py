"""API error classification for smart failover and recovery.

A priority-ordered pipeline maps an API exception to a ``ClassifiedError``
whose recovery hints (retry, rotate credential, fallback, compress, abort) the
retry loop in run_agent.py consults instead of re-matching strings itself.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Synthetic code for the OpenAI SDK rejecting a provider's SSE ``data:`` field
# before any completion chunk arrives; distinct from generic JSON parse errors.
PROVIDER_STREAM_NON_JSON_ERROR_CODE = "provider_stream_non_json_data"


# ── Error taxonomy ──────────────────────────────────────────────────────

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""

    auth = "auth"                        # Transient auth (401/403) — refresh/rotate
    auth_permanent = "auth_permanent"    # Auth failed after refresh — abort

    billing = "billing"                  # 402 or confirmed credit exhaustion — rotate immediately
    rate_limit = "rate_limit"            # 429 or quota-based throttling — backoff then rotate
    upstream_rate_limit = "upstream_rate_limit"  # Aggregator's upstream model 429 — fallback model, key is healthy

    overloaded = "overloaded"            # 503/529 — provider overloaded, backoff
    server_error = "server_error"        # 500/502 — internal server error, retry

    timeout = "timeout"                  # Connection/read timeout — rebuild client + retry
    ssl_cert_verification = "ssl_cert_verification"  # Deterministic TLS chain failure — fail fast with guidance

    context_overflow = "context_overflow"  # Context too large — compress, not failover
    payload_too_large = "payload_too_large"  # 413 — compress payload
    image_too_large = "image_too_large"   # Native image part exceeds provider's per-image limit — shrink and retry
    image_corrupt = "image_corrupt"       # Provider can't decode image bytes — strip and retry (shrinking won't help)

    model_not_found = "model_not_found"  # 404 or invalid model — fallback to different model
    provider_policy_blocked = "provider_policy_blocked"  # Aggregator account data/privacy policy excluded the only endpoint
    content_policy_blocked = "content_policy_blocked"  # Provider safety filter rejected this prompt — don't retry unchanged

    format_error = "format_error"        # 400 bad request — abort or strip + retry
    invalid_encrypted_content = "invalid_encrypted_content"  # Responses replay blob rejected — strip replay state and retry
    multimodal_tool_content_unsupported = "multimodal_tool_content_unsupported"  # Provider rejected list content in tool messages — downgrade to text

    thinking_signature = "thinking_signature"  # Anthropic thinking block sig invalid
    long_context_tier = "long_context_tier"    # Anthropic "extra usage" tier gate
    oauth_long_context_beta_forbidden = "oauth_long_context_beta_forbidden"  # Anthropic OAuth rejects 1M beta — disable beta and retry
    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"  # llama.cpp grammar rejects regex `pattern`/`format` — strip from tools and retry

    unknown = "unknown"                  # Unclassifiable — retry with backoff


# ── Classification result ───────────────────────────────────────────────

@dataclass
class ClassifiedError:
    """Structured classification of an API error with recovery hints."""

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    # Recovery hints — the retry loop checks these instead of re-classifying.
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in {FailoverReason.auth, FailoverReason.auth_permanent}

    @property
    def billing_unverified(self) -> bool:
        """True when a ``billing`` verdict rests on an ambiguous body (#82154)."""
        return bool(self.error_context.get("billing_unverified"))


# ── Provider-specific patterns ──────────────────────────────────────────

# Billing exhaustion (not transient rate limit).
_BILLING_PATTERNS = [
    "insufficient credits",
    "insufficient_quota",
    "insufficient balance",
    "credit balance",
    "credits exhausted",
    "credits have been exhausted",
    "requires available credits",
    "account balance is too low",
    "no usable credits",
    "top up your credits",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
    "out of extra usage",  # Anthropic OAuth Pro/Max overage bucket depleted (HTTP 400)
    "out of funds",
    "run out of funds",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    "not available on the free tier",
]

# Billing matches that are NOT proof of exhaustion: Anthropic returns the same
# "out of extra usage" body for a content-filter rejection (#82154). Verdict
# stays ``billing`` but error_context marks it unverified so surfaces hedge
# and the credential pool uses a short cooldown instead of the 1h bench.
_UNVERIFIED_BILLING_PATTERNS = ("out of extra usage",)


def _billing_ambiguity_context(error_msg: str) -> Dict[str, Any]:
    """error_context marking a billing verdict as unverified (see above)."""
    if any(p in error_msg for p in _UNVERIFIED_BILLING_PATTERNS):
        return {"billing_unverified": True, "possible_content_filter": True}
    return {}


# xAI's explicit Grok credit-exhaustion code, returned as HTTP 403 rather than
# 402. The 403 special case stays provider-scoped: other providers' billing
# codes on a 403 remain auth failures.
_XAI_SPENDING_LIMIT_ERROR_CODE = "personal-team-blocked:spending-limit"

# Structured codes meaning the account cannot serve paid traffic.
_BILLING_ERROR_CODES = frozenset({
    "insufficient_quota",
    "billing_not_active",
    "payment_required",
    "insufficient_credits",
    "no_usable_credits",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    "member_spend_cap_exceeded",
    _XAI_SPENDING_LIMIT_ERROR_CODE,
})

# Rate limiting (transient, will resolve).
_RATE_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "rate increased too quickly",  # Alibaba/DashScope throttling
    # AWS Bedrock throttling
    "throttlingexception",
    "too many concurrent requests",
    "servicequotaexceededexception",
    # Bedrock "Throttling error: Too many tokens..." also contains the overflow
    # phrase "too many tokens"; rate limit is matched first so throttle wins.
    "throttling",
]

# Provider-side overload: the credential is valid, the server is busy, so back
# off and retry the same key — never rotate. Some providers (Z.AI/Zhipu) reuse
# HTTP 429 for this, so the 429 path checks these first. Kept narrow so a
# normal "you have been rate-limited" doesn't land here. (#14038, #15297)
_OVERLOADED_PATTERNS = [
    "overloaded",
    "temporarily overloaded",
    "service is temporarily overloaded",
    "service may be temporarily overloaded",
    "server is overloaded",
    "server overloaded",
    "service overloaded",
    "service is overloaded",
    "upstream overloaded",
    "currently overloaded",
    "at capacity",
    "over capacity",
]

# Usage-limit patterns that need disambiguation (billing OR rate_limit).
_USAGE_LIMIT_PATTERNS = [
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
]

# Signals that a usage limit is transient (periodic quota, not billing).
_USAGE_LIMIT_TRANSIENT_SIGNALS = [
    "try again",
    "retry",
    "resets at",
    "reset in",
    "resets in",
    "reset after",
    "available in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
    "per minute",
    "per second",
]

# Payload-too-large detected from message text (proxies embed the status).
_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large",
    "payload too large",
    "error code: 413",
    "request_too_large",  # Anthropic's 413 type, re-wrapped by proxies without a status
    "request exceeds the maximum size",
]

# Image-size rejections. Matched on 400 bodies (not 413): providers return a
# specific 400 before the whole request hits the size limit (Anthropic: hard
# 5 MB per image, "image exceeds 5 MB maximum").
_IMAGE_TOO_LARGE_PATTERNS = [
    "image exceeds",        # Anthropic: "image exceeds 5 MB maximum"
    "image too large",      # generic
    "image_too_large",      # error_code variant
    "image size exceeds",   # variant
    "image dimensions exceed",  # Anthropic: "image dimensions exceed max allowed size: 8000 pixels"
    "dimensions exceed max allowed size",  # Anthropic dimension-cap (wording variant)
    "max allowed size: 8000",  # Anthropic dimension-cap (explicit pixel ceiling)
    # MiniMax Anthropic-compat: "media exceeds size limit: max 10485760 bytes"
    # (#76039). A non-image media rejection landing here is harmless: the
    # shrink pass finds no image parts and the original error surfaces.
    "media exceeds",
    "media too large",
]

# Image bytes undecodable (e.g. re-serialized history lost data). Shrinking
# can't fix corruption, so these route to strip-and-retry, never shrink.
# xAI wordings (#69078); the last is the full sentence on purpose — shorter
# fragments also match non-image download failures.
_IMAGE_CORRUPT_PATTERNS = [
    "invalid png image",
    "invalid jpeg image",
    "base64 string of provided image cannot be decoded",
    "downloaded response does not contain a valid jpg, png, webp, or ico image",
]

# Providers that reject list-type ``content`` in tool messages with a 400
# (Xiaomi MiMo, some Alibaba endpoints, OpenAI-compat long tail). Recovery:
# strip image parts from tool messages, remember (provider, model), retry. (#27344)
_MULTIMODAL_TOOL_CONTENT_PATTERNS = [
    # Xiaomi MiMo: {"error":{"code":"400","message":"Param Incorrect","param":"text is not set"}}
    "text is not set",
    "tool message content must be a string",
    "tool content must be a string",
    "tool message must be a string",
    # OpenAI-compat schema-validation shapes
    "expected string, got list",
    "expected string, got array",
    # Alibaba/DashScope variant
    "tool_call.content must be string",
]

_CONTEXT_OVERFLOW_PATTERNS = [
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    # Bare "max_tokens" is load-bearing: the output-cap-retry path keys off it.
    # Empty-response advisories mentioning it are intercepted earlier by
    # _EMPTY_PROVIDER_RESPONSE_PATTERNS, so they never route into compression.
    "max_tokens",
    "maximum number of tokens",
    # vLLM / local inference server patterns
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",             # "engine prompt length X exceeds"
    "input is too long",
    "maximum model length",
    # Ollama patterns
    "context length exceeded",
    "truncating input",
    # llama.cpp / llama-server patterns
    "slot context",              # "slot context: N tokens, prompt N tokens"
    "n_ctx_slot",
    # Chinese error messages (some providers return these)
    "超过最大长度",
    "上下文长度",
    # Z.AI / Zhipu GLM pattern (English form; error code 1210)
    "tokens in request more than max tokens allowed",
    # AWS Bedrock Converse API error patterns
    "input is too long",
    "max input token",
    "input token",
    "exceeds the maximum number of input tokens",
    # Together/Fireworks: "Input length N exceeds the maximum allowed input length of M tokens."
    "maximum allowed input length",
]

_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
    # OpenRouter 404 when no endpoint for the model supports tool calling;
    # model_not_found triggers fallback instead of burning retries (#58446).
    "no endpoints found that support tool use",
]


def _model_id_missing_known_prefix(model: str, provider: str) -> bool:
    """True when a bare model id is only known to the provider as ``vendor/id``.

    NVIDIA NIM answers a bare id with a naked ``404 page not found``; the
    curated catalogue tells that apart from a bad endpoint. Never guesses: an
    id absent from the catalogue returns False so real endpoint problems keep
    their retryable ``unknown`` classification.
    """
    name = (model or "").strip()
    if not name or "/" in name:
        return False
    try:
        from hermes_cli.model_normalize import suggest_prefixed_model_id

        return bool(suggest_prefixed_model_id((provider or "").strip(), name))
    except Exception:
        return False


# Qwen/vLLM chat-template raise_exception("No user query found in messages").
# Shared by _INVALID_MESSAGE_BODY_PATTERNS (→ format_error) and the llama.cpp
# grammar exclusion guard so the two sites cannot drift.
_NO_USER_QUERY_SIGNAL = "no user query found"

# Malformed-message-array 400s: deterministic rejections of the *transcript*
# (e.g. a content-less assistant stub after a dead stream). NOT context
# overflow — the input may be tiny — so they must fail fast as format_error
# instead of thrashing the compression loop.
_INVALID_MESSAGE_BODY_PATTERNS = [
    "must have non-empty content",
    "messages must have non-empty",
    "invalid_request_body",
    "text content blocks must be non-empty",
    "content field is required",
    "messages: at least one message is required",
    # Qwen/vLLM templates: no surviving non-empty user turn. Compression
    # cannot invent one, and local engines may wrap this as a grammar error.
    _NO_USER_QUERY_SIGNAL,
]

# Request-validation signals: malformed request, identical on every retry.
# Some gateways (codex.nekos.me) return these as 5xx, so the 5xx path also
# checks them to avoid a retry flood on a deterministic rejection.
_REQUEST_VALIDATION_PATTERNS = [
    "unknown parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "invalid_request_error",
    "unknown_parameter",
    "unsupported_parameter",
]

# Parameters Hermes sends on SOME routes only → hosts where sending them is
# deliberate. A rejection from any other host means the provider's own gateway
# injected the field, so the 400 is a server-side flake, not our request shape.
# ``prompt_cache_retention``: only sent for api.meta.ai / bedrock-mantle
# (agent/transports/codex.py); the Codex OAuth backend rejects it spontaneously.
_SERVER_INJECTED_PARAM_SENDERS: Dict[str, tuple] = {
    "prompt_cache_retention": ("meta", "muse", "msl", "model-api", "bedrock", "mantle"),
}

_PARAM_REJECTION_WORDS = ("not supported", "unsupported", "unknown", "unrecognized")


def _is_server_injected_param_rejection(error_msg: str, provider: str) -> bool:
    """True when a 400 blames a one-route-only parameter this route never sends.

    Conservative: fires only for known parameters AND only when ``provider``
    is not a route that sends them, so a genuine client-side bad parameter
    (``max_tokens`` on GPT-5) still fails fast as ``format_error``.
    """
    if not error_msg:
        return False
    provider_slug = (provider or "").strip().lower()
    for param, senders in _SERVER_INJECTED_PARAM_SENDERS.items():
        if param not in error_msg or not any(w in error_msg for w in _PARAM_REJECTION_WORDS):
            continue
        return not any(sender in provider_slug for sender in senders)
    return False


# OpenRouter 404 when the account privacy setting (or per-request
# ``provider.data_collection: deny``) excludes the only endpoint for a model.
# Not model_not_found: the model exists, fallback can't help (account-level),
# and the body already carries the fix URL.
_PROVIDER_POLICY_BLOCKED_PATTERNS = [
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
]

# Per-prompt provider safety-filter blocks (distinct from the account-level
# provider_policy_blocked). Deterministic for the unchanged request, so
# fallback immediately. Each phrase is verbatim from a specific provider —
# never a generic word like "policy" that could collide with billing/auth.
_CONTENT_POLICY_BLOCKED_PATTERNS = [
    # OpenAI Codex (#18028) — message may arrive without an HTTP status
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
    # OpenAI moderation — chat completions / responses
    "violates our usage policies",
    "violates openai's usage policies",
    "your request was flagged by",
    # Anthropic safety system
    "prompt was flagged by our safety",
    "responses cannot be generated due to safety",
    # OpenAI-standard token / Azure error code. Deliberately NOT the space
    # variant "content filter", which appears in benign echoed config text.
    "content_filter",
    "responsibleaipolicyviolation",
    # MiniMax output-layer safety filter, "output new_sensitive (1027)" (#32421)
    "new_sensitive",
]

# Auth patterns (non-status-code signals)
_AUTH_PATTERNS = [
    "invalid api key",
    "invalid_api_key",
    "gateway_auth_failed",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
]

# Provider empty-response advisories (OpenRouter / nano-gpt / similar). Checked
# before context-overflow matching because the text often mentions
# "max_tokens", which used to send healthy sessions into a compression spiral.
_EMPTY_PROVIDER_RESPONSE_PATTERNS = [
    "returned an empty response",
    "empty response despite retries",
    "provider returned an empty response",
    "model returning empty responses",
    "empty response stream",
]

# Timeout wording from generic exception types (RuntimeError from a shim
# wrapping a subprocess timeout) that the type-based heuristics would miss.
_TIMEOUT_MESSAGE_PATTERNS = [
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
]

# Connect/DNS failures surfaced by generic exception types with no status, so
# _TRANSPORT_ERROR_TYPES never fires. Deliberately EXCLUDES mid-stream
# disconnect strings — those belong to _SERVER_DISCONNECT_PATTERNS, which may
# route large sessions to compression; a never-established connection cannot
# be an overflow rejection.
_CONNECTION_MESSAGE_PATTERNS = [
    # TCP connect failures
    "connection refused",
    "econnrefused",
    "no route to host",
    "network is unreachable",
    "network unreachable",
    # DNS resolution failures (Python, glibc, macOS, Node bridge phrasings)
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "getaddrinfo failed",
    "getaddrinfo enotfound",
    "eai_again",
    # Node/undici bridge generic network failure (MCP servers, local shims)
    "fetch failed",
    "failed to fetch",
    # Envoy/proxy upstream connect failure (cloud gateways)
    "upstream connect error",
]

_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    # SSL type names listed so provider-wrapped SSL errors (chain lost) still
    # classify as transport instead of unknown.
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    # OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
})

# Ambiguous disconnects (no status): transient hiccup OR a gateway dropping an
# oversized request. A large session + one of these → context-overflow path.
_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
]

# SSL certificate verification failures are deterministic (proxy, missing CA,
# expired/self-signed cert) — fail fast. Checked BEFORE _SSL_TRANSIENT_PATTERNS
# because these messages usually also contain "[SSL:".
_SSL_CERT_VERIFY_PATTERNS = [
    "certificate verify failed",       # Python ssl module canonical text
    "certificate_verify_failed",       # OpenSSL error token
    "unable to get local issuer certificate",
    "self-signed certificate",
    "self signed certificate",
    "certificate has expired",
    "hostname mismatch, certificate is not valid",
    "unable to verify the first certificate",  # Node/undici phrasing (MCP bridges)
]

# Transient SSL alerts: retry but NOT compression (kept apart from
# _SERVER_DISCONNECT_PATTERNS). Matched on stable substrings because OpenSSL 3
# changed token separators (SSLV3_ALERT_... → SSL/TLS_ALERT_...).
_SSL_TRANSIENT_PATTERNS = [
    # Space-separated (human-readable form, Python ssl module, most SDKs)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    # Underscore-separated OpenSSL tokens
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    # Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
]


# ── Verdicts and rule tables ────────────────────────────────────────────
#
# A verdict is ``(reason, hint_overrides)``; overrides not listed take the
# ClassifiedError defaults (retryable=True, everything else False/empty).
# Rule tables are ordered ``(patterns, reason, hints)`` triples matched
# first-hit; ``hints`` may be a callable of the error message.

Verdict = Tuple[FailoverReason, Dict[str, Any]]
_ROTATE_FALLBACK = {"should_rotate_credential": True, "should_fallback": True}

_V_BILLING: Verdict = (FailoverReason.billing, {"retryable": False, **_ROTATE_FALLBACK})
_V_RATE_LIMIT: Verdict = (FailoverReason.rate_limit, dict(_ROTATE_FALLBACK))
_V_OVERLOADED: Verdict = (FailoverReason.overloaded, {})
_V_SERVER_ERROR: Verdict = (FailoverReason.server_error, {})
_V_CONTEXT_OVERFLOW: Verdict = (FailoverReason.context_overflow, {"should_compress": True})
_V_PAYLOAD_TOO_LARGE: Verdict = (FailoverReason.payload_too_large, {"should_compress": True})
_V_MODEL_NOT_FOUND: Verdict = (FailoverReason.model_not_found, {"retryable": False, "should_fallback": True})
_V_POLICY_BLOCKED: Verdict = (FailoverReason.provider_policy_blocked, {"retryable": False})
_V_FORMAT_ERROR: Verdict = (FailoverReason.format_error, {"retryable": False, "should_fallback": True})
_V_AUTH_ROTATE: Verdict = (FailoverReason.auth, {"retryable": False, **_ROTATE_FALLBACK})
_V_AUTH_FALLBACK: Verdict = (FailoverReason.auth, {"retryable": False, "should_fallback": True})
_V_TIMEOUT: Verdict = (FailoverReason.timeout, {})
_V_IMAGE_TOO_LARGE: Verdict = (FailoverReason.image_too_large, {})
_V_IMAGE_CORRUPT: Verdict = (FailoverReason.image_corrupt, {})
_V_MULTIMODAL: Verdict = (FailoverReason.multimodal_tool_content_unsupported, {})
_V_INVALID_ENCRYPTED: Verdict = (FailoverReason.invalid_encrypted_content, {})


def _billing_hints(error_msg: str) -> Dict[str, Any]:
    """Billing verdict carrying the #82154 ambiguity marker when applicable."""
    return {**_V_BILLING[1], "error_context": _billing_ambiguity_context(error_msg)}


def _rule(patterns: Sequence[str], verdict: Verdict, hints=None) -> tuple:
    return (patterns, verdict[0], verdict[1] if hints is None else hints)


def _emit(result_fn: Callable[..., ClassifiedError], verdict: Verdict) -> ClassifiedError:
    return result_fn(verdict[0], **verdict[1])


def _first_match(error_msg: str, rules: Iterable[tuple], result_fn) -> Optional[ClassifiedError]:
    """Return the verdict of the first rule whose pattern list hits ``error_msg``."""
    for patterns, reason, hints in rules:
        if any(p in error_msg for p in patterns):
            overrides: Dict[str, Any] = hints(error_msg) if callable(hints) else hints
            return result_fn(reason, **overrides)
    return None


# Image/tool-content 400s, ordered: multimodal recovery differs from image
# shrink; corrupt bytes need strip-and-retry, not shrink; image-shrink is a
# cheaper recovery than context compression for "exceeds" + "image" bodies.
_IMAGE_TOOL_RULES = (
    _rule(_MULTIMODAL_TOOL_CONTENT_PATTERNS, _V_MULTIMODAL),
    _rule(_IMAGE_CORRUPT_PATTERNS, _V_IMAGE_CORRUPT),
    _rule(_IMAGE_TOO_LARGE_PATTERNS, _V_IMAGE_TOO_LARGE),
)

# Overflow signals arriving as 5xx (llama.cpp reports overflow as 500; busy /
# model-load OOM as 503). Empty-response advisories must not enter compression.
_OVERFLOW_AS_5XX_RULES = (
    _rule(_EMPTY_PROVIDER_RESPONSE_PATTERNS, _V_SERVER_ERROR),
    _rule(_CONTEXT_OVERFLOW_PATTERNS, _V_CONTEXT_OVERFLOW),
)

# 404: Nous API surfaces credit depletion as a paid model vanishing from the
# Free Tier (billing, not missing model); OpenRouter policy block before
# model_not_found.
_404_RULES = (
    _rule(_BILLING_PATTERNS, _V_BILLING),
    _rule(_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED),
    _rule(_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
)

# 400 tail after the deterministic request-shape checks. Some providers return
# model-not-found / rate-limit / billing as 400 instead of 404/429/402.
_400_TAIL_RULES = _OVERFLOW_AS_5XX_RULES + (
    _rule(_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED),
    _rule(_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
    _rule(_RATE_LIMIT_PATTERNS, _V_RATE_LIMIT),
    _rule(_BILLING_PATTERNS, _V_BILLING, _billing_hints),
)

# Status-less message path, head (before usage-limit disambiguation).
_MESSAGE_HEAD_RULES = (_rule(_PAYLOAD_TOO_LARGE_PATTERNS, _V_PAYLOAD_TOO_LARGE),) + _IMAGE_TOOL_RULES

# Status-less message path, tail. Overload before rate_limit/billing so a
# message-only "overloaded" backs off instead of rotating; auth is
# non-retryable (same key always fails); policy block before model_not_found;
# timeout/connection wording last, classified as transport (never compression).
_MESSAGE_TAIL_RULES = (
    _rule(_OVERLOADED_PATTERNS, _V_OVERLOADED),
    _rule(_BILLING_PATTERNS, _V_BILLING, _billing_hints),
    _rule(_RATE_LIMIT_PATTERNS, _V_RATE_LIMIT),
    _rule(_EMPTY_PROVIDER_RESPONSE_PATTERNS, _V_SERVER_ERROR),
    _rule(_CONTEXT_OVERFLOW_PATTERNS, _V_CONTEXT_OVERFLOW),
    _rule(_AUTH_PATTERNS, _V_AUTH_ROTATE),
    _rule(_PROVIDER_POLICY_BLOCKED_PATTERNS, _V_POLICY_BLOCKED),
    _rule(_MODEL_NOT_FOUND_PATTERNS, _V_MODEL_NOT_FOUND),
    _rule(_TIMEOUT_MESSAGE_PATTERNS, _V_TIMEOUT),
    _rule(_CONNECTION_MESSAGE_PATTERNS, _V_TIMEOUT),
)

# Structured error code → verdict. The error-code rate_limit verdict rotates
# but does not set should_fallback (unlike the message/status paths).
_ERROR_CODE_VERDICTS: Dict[str, Verdict] = {
    **dict.fromkeys(("resource_exhausted", "throttled", "rate_limit_exceeded"),
                    (FailoverReason.rate_limit, {"should_rotate_credential": True})),
    **dict.fromkeys(_BILLING_ERROR_CODES, _V_BILLING),
    **dict.fromkeys(("model_not_found", "model_not_available", "invalid_model"), _V_MODEL_NOT_FOUND),
    **dict.fromkeys(("context_length_exceeded", "max_tokens_exceeded"), _V_CONTEXT_OVERFLOW),
    "invalid_encrypted_content": _V_INVALID_ENCRYPTED,
}

_5XX_VALIDATION_CODES = {"invalid_request_error", "unknown_parameter", "unsupported_parameter"}
_400_VALIDATION_CODES = {"unknown_parameter", "unsupported_parameter"}
_400_VALIDATION_PATTERNS = [p for p in _REQUEST_VALIDATION_PATTERNS if p != "invalid_request_error"]


# ── Classification pipeline ─────────────────────────────────────────────

def _openrouter_wrapped_message(err_obj: dict) -> str:
    """Lowercased inner message from OpenRouter's ``error.metadata.raw`` JSON wrapper."""
    metadata = err_obj.get("metadata", {})
    raw = metadata.get("raw") or "" if isinstance(metadata, dict) else ""
    if not (isinstance(raw, str) and raw.strip()):
        return ""
    try:
        inner = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    inner_err = inner.get("error", {}) if isinstance(inner, dict) else None
    if isinstance(inner_err, dict):
        return str(inner_err.get("message") or "").lower()
    return ""


def _build_error_msg(error: Exception, body: Any) -> str:
    """Lowercased str(error) + body message + OpenRouter-wrapped upstream message.

    str(error) alone may omit the body (OpenAI SDK's APIStatusError.__str__
    returns only the first arg), so body text is appended for pattern matching.
    """
    raw_msg = str(error).lower()
    body_msg = metadata_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            body_msg = str(err_obj.get("message") or "").lower()
            metadata_msg = _openrouter_wrapped_message(err_obj)
        if not body_msg:
            body_msg = str(body.get("message") or "").lower()
    parts = [raw_msg]
    if body_msg and body_msg not in raw_msg:
        parts.append(body_msg)
    if metadata_msg and metadata_msg not in raw_msg and metadata_msg not in body_msg:
        parts.append(metadata_msg)
    return " ".join(parts)


def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation.

    Priority order: plugin hooks → provider-specific special cases → HTTP
    status → structured error code → message patterns → SSL → disconnect +
    large session → transport types → unknown (retryable with backoff).
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)
    response_headers = _extract_response_headers(error)
    error_msg = _build_error_msg(error, body)
    provider_lower = (provider or "").strip().lower()
    model_lower = (model or "").strip().lower()

    def _result(reason: FailoverReason, **overrides) -> ClassifiedError:
        defaults = {
            "reason": reason,
            "status_code": status_code,
            "provider": provider,
            "model": model,
            "message": _extract_message(error, body),
        }
        defaults.update(overrides)
        return ClassifiedError(**defaults)

    # ── 0. Plugin classifiers (first valid result wins) ─────────────
    # Runs before the built-in pipeline so a provider plugin can add or correct
    # classifications. invoke_hook isolates callback failures; this guard only
    # covers import/dispatch failure.
    try:
        from hermes_cli.plugins import get_plugin_error_classification
        plugin_classification = get_plugin_error_classification(
            provider=provider,
            model=model,
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            error_message=error_msg,
            error_body=body,
            error=error,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
        )
    except Exception as exc:
        logger.debug("Plugin error classification unavailable: %s", exc)
        plugin_classification = None
    if plugin_classification is not None:
        reason = plugin_classification.pop("reason")
        logger.info(
            "API error classified by plugin hook: %s (provider=%s, status=%s)",
            reason.value, provider, status_code,
        )
        return _result(reason, **plugin_classification)

    # ── 1. Provider-specific patterns (highest priority) ────────────

    # Deterministic per-prompt safety refusal. Before status classification so
    # a 400 block isn't downgraded to format_error and a status-less block
    # isn't left retryable (#18028).
    if any(p in error_msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _result(FailoverReason.content_policy_blocked, retryable=False, should_fallback=True)

    # Anthropic thinking-block 400s: signature mismatch after any transcript
    # mutation, or "blocks in the latest assistant message cannot be modified".
    # Not gated on provider — OpenRouter proxies Anthropic errors.
    if (
        status_code == 400
        and "thinking" in error_msg
        and (
            "signature" in error_msg
            or "cannot be modified" in error_msg
            or "must remain as they were" in error_msg
        )
    ):
        return _result(FailoverReason.thinking_signature, retryable=True, should_compress=False)

    # Anthropic long-context tier gate (429 "extra usage" + "long context")
    if status_code == 429 and "extra usage" in error_msg and "long context" in error_msg:
        return _result(FailoverReason.long_context_tier, retryable=True, should_compress=True)

    # Anthropic OAuth subscription rejects the 1M-context beta header (400
    # "The long context beta is not yet available for this subscription");
    # run_agent rebuilds the client without the beta and retries once.
    if status_code == 400 and "long context beta" in error_msg and "not yet available" in error_msg:
        return _result(FailoverReason.oauth_long_context_beta_forbidden, retryable=True, should_compress=False)

    # llama.cpp json-schema-to-grammar rejects regex escapes / ``format`` in
    # tool schemas (400); the retry loop strips pattern/format and retries.
    # Exclude the Qwen/vLLM "No user query found" template error that local
    # engines wrap as "Unable to generate parser for this template" — that is
    # a poisoned transcript (→ format_error), not a grammar problem.
    llama_cpp_grammar_hit = status_code == 400 and (
        "error parsing grammar" in error_msg
        or "json-schema-to-grammar" in error_msg
        or ("unable to generate parser" in error_msg and "template" in error_msg)
    )
    if llama_cpp_grammar_hit and _NO_USER_QUERY_SIGNAL not in error_msg:
        return _result(FailoverReason.llama_cpp_grammar_pattern, retryable=True, should_compress=False)

    # xAI Grok subscription entitlement. As HTTP 403 the status path handles
    # it; as an SSE ``type=error`` frame there is no status and the message
    # matches no pattern list, so it would burn max_retries as ``unknown``.
    if (
        "do not have an active grok subscription" in error_msg
        or ("out of available resources" in error_msg and "grok" in error_msg)
    ):
        return _result(FailoverReason.auth, retryable=False, should_fallback=True)

    # ── 2. HTTP status code classification ──────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code, error_msg, error_code, body,
            provider=provider_lower, model=model_lower,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
            response_headers=response_headers,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # Local MoA streaming adapter-shape bugs are not a provider outage; falling
    # back would silently replace the MoA route with a single model (#55933).
    if provider_lower == "moa" and (
        "'types.SimpleNamespace' object is not iterable" in str(error)
        or "'types.SimpleNamespace' object has no attribute 'index'" in str(error)
    ):
        return _result(FailoverReason.format_error, retryable=False, should_fallback=False)

    # Persisted MoA preset name that was renamed/deleted — deterministic config error.
    from agent.errors import MoAPresetNotFoundError

    if isinstance(error, MoAPresetNotFoundError):
        return _result(FailoverReason.model_not_found, retryable=False)

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(
        error_msg, error_type,
        approx_tokens=approx_tokens,
        context_length=context_length,
        result_fn=_result,
    )
    if classified is not None:
        return classified

    # ── 5. SSL: deterministic cert failure → fail fast; transient alert → retry
    # Cert-verify first: those messages also contain "[ssl:". Transient alerts
    # are classified before the disconnect check so a large session doesn't
    # compress on a flaky TLS handshake.
    if any(p in error_msg for p in _SSL_CERT_VERIFY_PATTERNS):
        return _result(FailoverReason.ssl_cert_verification, retryable=False, should_fallback=False)
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Before the generic transport catch: a disconnect on a large session is
    # more likely an overflow rejection than a transport hiccup.
    if any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS) and not status_code:
        # Reasoning models: a disconnect is far more likely the gateway
        # idle-killing a long thinking stream than overflow — never compress
        # (and silently drop history) on a phantom overflow (#52310).
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute thresholds only proxy for smaller context windows.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(FailoverReason.context_overflow, retryable=True, should_compress=True)
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7. Stale-call circuit breaker → failover immediately ────────
    # _check_stale_giveup() raises RuntimeError before any network call; as
    # ``unknown`` it would burn every retry instantly against the dead provider.
    if (
        error_type == "RuntimeError"
        and "consecutive stale attempts" in error_msg
        and "aborting this call" in error_msg
    ):
        return _result(FailoverReason.timeout, retryable=False, should_fallback=True)

    # ── 8. Transport / timeout heuristics ───────────────────────────
    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 9. Fallback: unknown ────────────────────────────────────────
    return _result(FailoverReason.unknown, retryable=True)


# ── Status code classification ──────────────────────────────────────────

def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    response_headers=None,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on HTTP status code with message-aware refinement."""

    if status_code == 401:
        # Not retryable on its own: credential rotation / provider refresh run
        # before the retryability check; if they fail, the client-error abort
        # path (fallback first) is correct.
        return result_fn(FailoverReason.auth, retryable=False, **_ROTATE_FALLBACK)

    if status_code == 403:
        # OpenRouter 403 "key limit exceeded" and similar plan/credit exhaustion are billing.
        if (
            (provider == "xai-oauth" and error_code.lower() == _XAI_SPENDING_LIMIT_ERROR_CODE)
            or "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return _emit(result_fn, _V_BILLING)
        return _emit(result_fn, _V_AUTH_FALLBACK)

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        classified = _first_match(error_msg, _404_RULES, result_fn)
        if classified is not None:
            return classified
        # Bare id the catalogue only knows prefixed → malformed id (NVIDIA NIM
        # "404 page not found"), deterministic (#78796).
        if _model_id_missing_known_prefix(model, provider):
            return _emit(result_fn, _V_MODEL_NOT_FOUND)
        # Generic 404 (wrong endpoint path, proxy glitch): model_not_found would
        # silently fall back and misreport; stay unknown so the real error surfaces.
        return result_fn(FailoverReason.unknown, retryable=True)

    if status_code == 413:
        return result_fn(FailoverReason.payload_too_large, retryable=True, should_compress=True)

    if status_code == 429:
        # Z.AI/Zhipu reuse 429 for server-wide overload: back off on the same
        # key instead of burning the pool (#14038).
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(FailoverReason.overloaded, retryable=True)
        # OpenRouter-wrapped upstream 429: the user's key is healthy, so
        # fall back to another model rather than rotating/benching the key.
        if _is_openrouter_upstream_error(body, provider):
            upstream_provider = _extract_upstream_provider_name(body)
            ctx = {"upstream_provider": upstream_provider} if upstream_provider else {}
            return result_fn(
                FailoverReason.upstream_rate_limit,
                retryable=True,
                should_rotate_credential=False,
                should_fallback=True,
                error_context=ctx,
            )
        # Quota walls returned as 429 (Anthropic ``usage_limit_reached``, other
        # providers' "quota"/"limit exceeded", explicit billing phrases) are
        # billing — but ONLY when the body is not itself an explicit rate-limit
        # phrase ("Rate limit exceeded" contains "limit exceeded") and carries
        # no reset/retry signal (#93419, #39441).
        has_usage_limit = (
            error_code.lower() == "usage_limit_reached"
            or "usage_limit_reached" in error_msg
            or any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
        )
        has_billing = any(p in error_msg for p in _BILLING_PATTERNS)
        if (
            (has_billing or has_usage_limit)
            and not any(p in error_msg for p in _RATE_LIMIT_PATTERNS)
            and not _has_usage_limit_transient_signal(error_msg, body, response_headers)
        ):
            return _emit(result_fn, _V_BILLING)
        return _emit(result_fn, _V_RATE_LIMIT)

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider, model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Deterministic request-validation errors returned as 5xx
        # (codex.nekos.me) must fail fast, not retry-flood — unless the
        # rejected parameter was injected server-side (see _classify_400).
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in _5XX_VALIDATION_CODES
        ) and not _is_server_injected_param_rejection(error_msg, provider):
            return _emit(result_fn, _V_FORMAT_ERROR)
        classified = _first_match(error_msg, _OVERFLOW_AS_5XX_RULES, result_fn)
        return classified if classified is not None else result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        classified = _first_match(error_msg, _OVERFLOW_AS_5XX_RULES, result_fn)
        return classified if classified is not None else result_fn(FailoverReason.overloaded, retryable=True)

    # 408 Request Timeout is retry-safe (RFC 9110 §15.5.9) — proxies in front
    # of self-hosted backends emit it when generation outruns the read window.
    if status_code == 408:
        return result_fn(FailoverReason.timeout, retryable=True)

    if 400 <= status_code < 500:
        return _emit(result_fn, _V_FORMAT_ERROR)

    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


_RESET_FIELDS = ("resets_in_seconds", "resets_at", "reset_at", "retry_after")
_RESET_HEADERS = ("retry-after", "Retry-After", "x-ratelimit-reset", "X-RateLimit-Reset")


def _has_usage_limit_transient_signal(error_msg: str, body: dict, response_headers) -> bool:
    """Return whether a usage-limit response identifies a reset window."""
    if any(pattern in error_msg for pattern in _USAGE_LIMIT_TRANSIENT_SIGNALS):
        return True
    payloads = [body]
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        payloads.append(body["error"])
    for payload in payloads:
        if isinstance(payload, dict) and any(payload.get(f) not in (None, "") for f in _RESET_FIELDS):
            return True
    if response_headers and hasattr(response_headers, "get"):
        return any(response_headers.get(h) not in (None, "") for h in _RESET_HEADERS)
    return False


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: "usage limit, try again in 5 minutes" is a periodic quota, not billing."""
    if (
        any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
        and any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
    ):
        return _emit(result_fn, _V_RATE_LIMIT)
    return _emit(result_fn, _V_BILLING)


def _body_message_candidates(body: dict) -> Iterator[Any]:
    """Body message fields in priority order (OpenAI, flat, litellm/Bedrock proxy shapes)."""
    err_obj = body.get("error", {})
    yield err_obj.get("message") if isinstance(err_obj, dict) else None
    yield body.get("message")
    yield body.get("errorMessage")
    args = body.get("errorArgs")
    yield args.get("reason") if isinstance(args, dict) else None


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic."""
    classified = _first_match(error_msg, _IMAGE_TOOL_RULES, result_fn)
    if classified is not None:
        return classified

    # Invalid encrypted reasoning replay blob (OpenAI Responses). Before
    # context_overflow: "encrypted content … could not be verified" can trip
    # the overflow heuristics.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or ("encrypted content for item" in error_msg and "could not be verified" in error_msg)
        or "could not decrypt the provided encrypted_content" in error_msg
    ):
        return result_fn(FailoverReason.invalid_encrypted_content, retryable=True, should_fallback=False)

    # A 400 blaming a field this route never sent (Codex OAuth backend injects
    # and then rejects prompt_cache_retention ~20% of the time): transient,
    # retry the identical request; never compress. Before the validation branch.
    if _is_server_injected_param_rejection(error_msg, provider):
        return result_fn(FailoverReason.server_error, retryable=True, should_compress=False)

    # Unsupported/unknown parameter before context_overflow: GPT-5's
    # "Unsupported parameter: 'max_tokens'…" contains the overflow pattern
    # "max_tokens". Generic ``invalid_request_error`` is deliberately NOT used
    # here — OpenAI stamps it on genuine overflow 400s too.
    if (
        any(p in error_msg for p in _400_VALIDATION_PATTERNS)
        or error_code_lower in _400_VALIDATION_CODES
    ):
        return _emit(result_fn, _V_FORMAT_ERROR)

    # Malformed message array (empty-content assistant stub, etc.) before
    # context_overflow: the input can be tiny and compression cannot fix it.
    # Proxies (litellm/Bedrock) surface it as errorCode=INVALID_REQUEST_BODY.
    if (
        any(p in error_msg for p in _INVALID_MESSAGE_BODY_PATTERNS)
        or error_code_lower == "invalid_request_body"
    ):
        logger.warning(
            "Malformed message array 400 (invalid request body) classified as "
            "format_error, NOT context overflow — failing fast + falling back "
            "instead of entering the compression loop. This usually means an "
            "empty-content assistant stub is in the transcript; num_messages=%s "
            "approx_tokens=%s. error=%.200s",
            num_messages, approx_tokens, error_msg,
        )
        return _emit(result_fn, _V_FORMAT_ERROR)

    classified = _first_match(error_msg, _400_TAIL_RULES, result_fn)
    if classified is not None:
        return classified

    # Generic 400 + large session → probable context overflow (Anthropic can
    # return a bare "Error"). Proxy shapes are recognised so a long, descriptive
    # rejection is not mistaken for a bare error.
    err_body_msg = ""
    if isinstance(body, dict):
        err_body_msg = next(
            (m for m in (str(c or "").strip().lower() for c in _body_message_candidates(body)) if m),
            "",
        )
    is_generic = len(err_body_msg) < 30 or err_body_msg in {"error", ""}
    # Absolute thresholds only proxy for smaller context windows.
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256000 and (approx_tokens > 80000 or num_messages > 80)
    )
    if is_generic and is_large:
        return result_fn(FailoverReason.context_overflow, retryable=True, should_compress=True)

    return _emit(result_fn, _V_FORMAT_ERROR)


# ── Error code classification ───────────────────────────────────────────

def _classify_by_error_code(
    error_code: str, error_msg: str, result_fn,
) -> Optional[ClassifiedError]:
    """Classify by structured error codes from the response body."""
    code_lower = error_code.lower()

    # Deterministic request-validation failures encoded as plain-text
    # ``event: error`` SSE data behind HTTP 200: retrying cannot succeed, a
    # configured fallback still may.
    if code_lower == PROVIDER_STREAM_NON_JSON_ERROR_CODE and "request validation failed:" in error_msg:
        return _emit(result_fn, _V_FORMAT_ERROR)

    verdict = _ERROR_CODE_VERDICTS.get(code_lower)
    return _emit(result_fn, verdict) if verdict is not None else None


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(
    error_msg: str,
    error_type: str,
    *,
    approx_tokens: int,
    context_length: int,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on error message patterns when no status code is available."""
    classified = _first_match(error_msg, _MESSAGE_HEAD_RULES, result_fn)
    if classified is not None:
        return classified

    # Status-less usage limits need the same disambiguation as 402.
    if any(p in error_msg for p in _USAGE_LIMIT_PATTERNS):
        if any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS):
            return _emit(result_fn, _V_RATE_LIMIT)
        return _emit(result_fn, _V_BILLING)

    return _first_match(error_msg, _MESSAGE_TAIL_RULES, result_fn)


# ── Helpers ─────────────────────────────────────────────────────────────

def _cause_chain(error: Exception) -> Iterator[Any]:
    """Yield the error and its __cause__/__context__ chain, at most 5 deep."""
    current = error
    for _ in range(5):
        yield current
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            return
        current = cause


def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
    for current in _cause_chain(error):
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)  # some SDKs use .status
        if isinstance(code, int) and 100 <= code < 600:
            return code
    return None


def _extract_error_body(error: Exception) -> dict:
    """Extract the structured error body from an SDK exception or its cause chain."""
    for current in _cause_chain(error):
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            return body
        response = getattr(current, "response", None)
        if response is not None:
            try:
                json_body = response.json()
                if isinstance(json_body, dict):
                    return json_body
            except Exception:
                pass
    return {}


def _extract_response_headers(error: Exception):
    """Walk the error and its cause chain to find response headers."""
    for current in _cause_chain(error):
        headers = getattr(getattr(current, "response", None), "headers", None)
        if headers and hasattr(headers, "get"):
            return headers
    return {}


def _code_from_payload(payload: Any, top_keys: Sequence[str], peek_message: bool) -> str:
    """Code/type from ``payload.error`` or a top-level key; ``"400"`` is not a code.

    With ``peek_message``, a JSON string in ``error.message`` is parsed for a
    nested code (Responses API surfaces ``invalid_encrypted_content`` this way).
    """
    if not isinstance(payload, dict):
        return ""
    error_obj = payload.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip() and code.strip() != "400":
            return code.strip()
        message = error_obj.get("message")
        if peek_message and isinstance(message, str) and message.strip().startswith("{"):
            try:
                inner = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                inner = None
            nested_code = _code_from_payload(inner, ("code", "error_code"), False)
            if nested_code:
                return nested_code
    code = next((payload.get(k) for k in top_keys if payload.get(k)), "")
    if isinstance(code, (str, int)):
        text = str(code).strip()
        if text and text != "400":
            return text
    return ""


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    return _code_from_payload(body, ("code", "error_code", "errorCode"), True) if body else ""


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message (structured body first)."""
    for msg in _body_message_candidates(body or {}):
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    return str(error)[:500]


def _is_openrouter_upstream_error(body: Any, provider: str) -> bool:
    """Detect OpenRouter's "Provider returned error" wrapper around an upstream failure.

    The user's OpenRouter key is healthy — the upstream provider failed — so
    credential rotation is the wrong recovery.
    """
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    if str(err.get("message") or "").strip().lower() != "provider returned error":
        return False
    if (provider or "").strip().lower() == "openrouter":
        return True
    # Otherwise require the metadata shape only OpenRouter produces.
    metadata = err.get("metadata")
    return isinstance(metadata, dict) and ("raw" in metadata or "provider_name" in metadata)


def _extract_upstream_provider_name(body: Any) -> Optional[str]:
    """Pull the upstream provider name out of OpenRouter's error metadata."""
    err = body.get("error") if isinstance(body, dict) else None
    metadata = err.get("metadata") if isinstance(err, dict) else None
    name = metadata.get("provider_name") if isinstance(metadata, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None
