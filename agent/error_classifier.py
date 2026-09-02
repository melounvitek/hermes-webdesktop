"""API error classification for smart failover and recovery.

A structured taxonomy of API errors plus a priority-ordered classification
pipeline that picks the recovery action (retry, rotate credential, fall back
to another provider, compress context, or abort). The retry loop in
run_agent.py consults it for every API failure instead of inline matching.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Synthetic error code for the OpenAI SDK rejecting a provider's SSE ``data:``
# field before any completion chunk arrives. Distinct from generic JSON parse
# failures so stream-specific recovery needs no invented HTTP status.
PROVIDER_STREAM_NON_JSON_ERROR_CODE = "provider_stream_non_json_data"


# ── Error taxonomy ──────────────────────────────────────────────────────

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""

    # Authentication / authorization
    auth = "auth"                        # Transient auth (401/403) — refresh/rotate
    auth_permanent = "auth_permanent"    # Auth failed after refresh — abort

    # Billing / quota
    billing = "billing"                  # 402 or confirmed credit exhaustion — rotate immediately
    rate_limit = "rate_limit"            # 429 or quota-based throttling — backoff then rotate
    # Aggregator 429 from the upstream model: fall back to another model, NOT
    # credential rotation — the user's key is healthy.
    upstream_rate_limit = "upstream_rate_limit"

    # Server-side
    overloaded = "overloaded"            # 503/529 — provider overloaded, backoff
    server_error = "server_error"        # 500/502 — internal server error, retry

    # Transport
    timeout = "timeout"                  # Connection/read timeout — rebuild client + retry
    # TLS cert verification failure is deterministic for the host (inspecting
    # proxy, missing/expired CA, self-signed) — fail fast with guidance.
    ssl_cert_verification = "ssl_cert_verification"

    # Context / payload
    context_overflow = "context_overflow"  # Context too large — compress, not failover
    payload_too_large = "payload_too_large"  # 413 — compress payload
    image_too_large = "image_too_large"   # Native image part exceeds provider's per-image limit — shrink and retry
    image_corrupt = "image_corrupt"       # Image bytes undecodable — shrinking won't help, strip and retry

    # Model / provider policy
    model_not_found = "model_not_found"  # 404 or invalid model — fallback to different model
    provider_policy_blocked = "provider_policy_blocked"  # Aggregator (e.g. OpenRouter) blocked the only endpoint via account data/privacy policy
    content_policy_blocked = "content_policy_blocked"  # Provider safety filter rejected this prompt — deterministic, don't retry unchanged

    # Request format
    format_error = "format_error"        # 400 bad request — abort or strip + retry
    invalid_encrypted_content = "invalid_encrypted_content"  # Responses replay blob rejected — strip replay state and retry
    multimodal_tool_content_unsupported = "multimodal_tool_content_unsupported"  # Provider rejected list-type tool content (e.g. Xiaomi MiMo) — downgrade to text and retry

    # Provider-specific
    thinking_signature = "thinking_signature"  # Anthropic thinking block sig invalid
    long_context_tier = "long_context_tier"    # Anthropic "extra usage" tier gate
    oauth_long_context_beta_forbidden = "oauth_long_context_beta_forbidden"  # Anthropic OAuth subscription rejects 1M context beta — disable beta and retry
    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"  # llama.cpp grammar converter rejects regex escapes in `pattern` / `format` — strip from tools and retry

    # Catch-all
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
        """True when a ``billing`` verdict rests on an ambiguous body.

        Anthropic's "out of extra usage" 400 can also be a content-filter
        rejection (#82154); surfaces must hedge rather than assert exhaustion.
        """
        return bool(self.error_context.get("billing_unverified"))


# ── Provider-specific patterns ──────────────────────────────────────────

# Billing exhaustion (not transient rate limit)
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

# Billing matches that are NOT proof of exhaustion. Anthropic returns the same
# "out of extra usage" body on a subscription OAuth token both when the overage
# bucket is depleted AND when its content filter rejects the request (#82154).
# Classification stays ``billing`` (rotation + fallback are right either way),
# but ``error_context`` carries the ambiguity so surfaces hedge and the pool
# applies a short cooldown instead of the one-hour billing bench.
_UNVERIFIED_BILLING_PATTERNS = ("out of extra usage",)


def _billing_ambiguity_context(error_msg: str) -> Dict[str, Any]:
    """error_context marking a billing verdict as unverified (see above)."""
    if any(p in error_msg for p in _UNVERIFIED_BILLING_PATTERNS):
        return {"billing_unverified": True, "possible_content_filter": True}
    return {}


# xAI's explicit Grok credit-exhaustion code, returned as HTTP 403 not 402.
# Kept provider-scoped: other providers' billing codes stay auth on 403.
_XAI_SPENDING_LIMIT_ERROR_CODE = "personal-team-blocked:spending-limit"

# Structured codes meaning the account cannot serve paid traffic until
# credits/subscription capacity is restored.
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

# Rate limiting (transient, will resolve)
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
    # Generic throttle prefix — Bedrock/proxies say "Throttling error: Too many
    # tokens...". Without it the message hits the context-overflow list ("too
    # many tokens") and a healthy session gets compressed instead of backing
    # off. Rate-limit runs BEFORE overflow in the message-only path so the
    # throttle wins. (port of anomalyco/opencode#37848)
    "throttling",
]

# Provider-side overload, NOT a per-credential rate limit or billing problem.
# The key is valid, so recover by "back off and retry the same key", never
# rotate (rotation exhausts the pool while the endpoint is busy; single-key
# users have nothing to rotate to). Z.AI/Zhipu reuse HTTP 429 for overload, so
# the 429 path checks this list before defaulting to rate_limit. Phrases are
# kept overload-flavoured so "you have been rate-limited" doesn't land here.
# (#14038, #15297)
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

# Usage-limit patterns needing disambiguation (billing OR rate_limit)
_USAGE_LIMIT_PATTERNS = [
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
]

# Signals that a usage limit is transient (not billing)
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

# Payload-too-large from message text (no status_code attr) — proxies and some
# backends embed the HTTP status in the message.
_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large",
    "payload too large",
    "error code: 413",
    # Anthropic's structured 413 type; proxies may re-wrap it into a plain
    # message with no status — same compression recovery. (port of anomalyco/opencode#37848)
    "request_too_large",
    "request exceeds the maximum size",
]

# Image-size patterns, matched against 400 bodies (not 413): most providers
# return a specific image-too-big 400 before the request hits the 413 limit.
# Anthropic's hard 5 MB per-image cap matters most.
_IMAGE_TOO_LARGE_PATTERNS = [
    "image exceeds",        # Anthropic: "image exceeds 5 MB maximum"
    "image too large",      # generic
    "image_too_large",      # error_code variant
    "image size exceeds",   # variant
    "image dimensions exceed",  # Anthropic: "image dimensions exceed max allowed size: 8000 pixels"
    "dimensions exceed max allowed size",  # Anthropic dimension-cap (wording variant)
    "max allowed size: 8000",  # Anthropic dimension-cap (explicit pixel ceiling)
    # MiniMax's Anthropic-compatible endpoint: "media exceeds size limit: max
    # 10485760 bytes (2013)" for an oversized native image (#76039). A
    # non-image media rejection landing here is safe: the shrink pass finds no
    # image parts and the caller surfaces the original error unchanged.
    "media exceeds",
    "media too large",
    # "request_too_large" on a request with an image → image is the likely
    # culprit; the shrink path is still tried first.
]

# Image corruption — the provider decodes the request but not the image bytes
# (e.g. a re-serialized part in replayed history). Shrinking can't fix that,
# so this routes to strip-and-retry (image_corrupt), never the shrink path.
# xAI has two wordings for truncated bytes ("Invalid PNG image." aligned,
# "base64 string ... cannot be decoded" unaligned) and a third for URL images
# it downloads itself. The last is matched as the full sentence on purpose —
# shorter fragments also match non-image download failures. (#69078)
_IMAGE_CORRUPT_PATTERNS = [
    "invalid png image",
    "invalid jpeg image",
    "base64 string of provided image cannot be decoded",
    "downloaded response does not contain a valid jpg, png, webp, or ico image",
]

# Strict OpenAI-spec providers require tool message ``content`` to be a string;
# some (Anthropic native, Codex, Gemini, first-party OpenAI) accept a parts
# list so computer_use screenshots survive, others (Xiaomi MiMo, some Alibaba
# endpoints, a long tail) reject it with a 400. Recovery: strip image parts
# from tool messages, remember (provider, model) for the session, retry. (#27344)
_MULTIMODAL_TOOL_CONTENT_PATTERNS = [
    # Xiaomi MiMo: {"error":{"code":"400","message":"Param Incorrect","param":"text is not set"}}
    "text is not set",
    # Generic "tool message must be string" shapes
    "tool message content must be a string",
    "tool content must be a string",
    "tool message must be a string",
    # OpenAI-compat schema-validation messages for list-type tool content
    "expected string, got list",
    "expected string, got array",
    # Alibaba/DashScope variant
    "tool_call.content must be string",
]

# Context overflow patterns
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
    # Bare "max_tokens" is load-bearing — the output-cap-retry path keys off it
    # ("max_tokens: 65536 > context_window: 200000"). Do NOT remove. Empty-
    # response advisories also say "very low max_tokens" but are intercepted by
    # _EMPTY_PROVIDER_RESPONSE_PATTERNS BEFORE this list is consulted.
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
    # Together/Fireworks: "Input length 131393 exceeds the maximum allowed
    # input length of 131040 tokens." (port of anomalyco/opencode#37848)
    "maximum allowed input length",
]

# Model not found patterns
_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
    # OpenRouter 404 when no endpoint for the model supports tool calling.
    # model_not_found triggers fallback to a model/provider that does; without
    # it the error is ``unknown``/retryable and burns every attempt on the
    # same deterministic rejection. (PR #58446)
    "no endpoints found that support tool use",
]


def _model_id_missing_known_prefix(model: str, provider: str) -> bool:
    """True when a bare model id is only known to the provider as ``vendor/id``.

    Some providers answer a malformed id with a naked 404 (NVIDIA NIM: ``404
    page not found`` for a bare ``nemotron-3-ultra-550b-a55b``), indistinguishable
    from a bad endpoint path. If the id has no ``/`` but the curated catalogue
    has exactly one entry ending in ``/<id>``, the prefix was dropped and the
    failure is deterministic. Never guesses: an id absent from the catalogue
    (local NIM container, proxied model) returns False so genuine endpoint
    problems keep their retryable ``unknown`` classification.
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
# shape, not a parameter. Canonical case: a stream dies mid-response, Hermes
# persists a content-less assistant stub, and next turn Anthropic (or the
# litellm/Bedrock proxies in front of it) reject "all messages must have
# non-empty content ..." / errorCode INVALID_REQUEST_BODY. NOT context overflow
# — the input may be tiny — but the "400 + large session" heuristic used to
# route them into compression, ending in "Cannot compress further" every retry.
# Match explicitly and fail fast as format_error. (Stub creation is fixed in
# chat_completion_helpers; this covers transcripts already poisoned.)
_INVALID_MESSAGE_BODY_PATTERNS = [
    "must have non-empty content",
    "messages must have non-empty",
    "invalid_request_body",
    "text content blocks must be non-empty",
    "content field is required",
    "messages: at least one message is required",
    # Qwen/vLLM templates raise this when no non-empty user turn survives
    # (truncation, compression that dropped the only user message, a lineage
    # opening with assistant/tool). Compression cannot invent a user query, so
    # fail fast rather than thrash — and don't mis-route into llama.cpp grammar
    # recovery when local engines wrap it as "Unable to generate parser ...".
    _NO_USER_QUERY_SIGNAL,
]

# Request-validation patterns — malformed request, fails identically on every
# retry. Some OpenAI-compatible gateways (codex.nekos.me) return these as 5xx,
# so the generic "5xx → retryable server_error" rule would hammer the same
# rejection, reset the counter via transport recovery, and flood. A 5xx body
# carrying one of these is classified as non-retryable format_error.
_REQUEST_VALIDATION_PATTERNS = [
    "unknown parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "invalid_request_error",
    "unknown_parameter",
    "unsupported_parameter",
]

# Params Hermes sends on SOME routes only → the providers/hosts where sending
# them is deliberate. When a host NOT in the set rejects one, the client never
# sent it — the provider's own gateway injected it — so the 400 is a server
# flake, not a deterministic request-shape error (see
# _is_server_injected_param_rejection and its branch in _classify_400).
# ``prompt_cache_retention`` is only sent for api.meta.ai and bedrock-mantle
# (agent/transports/codex.py::_default_prompt_cache_retention_for_request);
# the Codex OAuth backend rejects it spontaneously on requests that never
# carried it.
_SERVER_INJECTED_PARAM_SENDERS: Dict[str, tuple] = {
    "prompt_cache_retention": ("meta", "muse", "msl", "model-api", "bedrock", "mantle"),
}


def _is_server_injected_param_rejection(error_msg: str, provider: str) -> bool:
    """True when a 400 blames a parameter this route never sends.

    ``error_msg`` is lowercased concatenated text; ``provider`` the lowercased
    slug. A match means the rejection isn't our request shape, so retrying
    the identical request is the correct recovery. Deliberately conservative:
    fires only for known one-route-only params AND only when the provider is
    not a route that sends them, so a genuine bad parameter (``max_tokens``
    on GPT-5) still fails fast as ``format_error``.
    """
    if not error_msg:
        return False
    provider_slug = (provider or "").strip().lower()
    for param, senders in _SERVER_INJECTED_PARAM_SENDERS.items():
        if param not in error_msg:
            continue
        # Must be a rejection of that parameter, not an incidental mention.
        if not (
            "not supported" in error_msg
            or "unsupported" in error_msg
            or "unknown" in error_msg
            or "unrecognized" in error_msg
        ):
            continue
        # This route sends the field on purpose — a real request error.
        return not any(sender in provider_slug for sender in senders)
    return False


# OpenRouter aggregator policy block: an account privacy setting (or a
# per-request ``provider.data_collection: deny``) excludes the only endpoint
# serving a model → 404 "No endpoints available matching your guardrail
# restrictions and data policy. Configure: https://openrouter.ai/settings/privacy".
# Classified ``provider_policy_blocked`` not ``model_not_found``: the model
# exists, fallback won't help (account-level setting applies to every call),
# and the body already carries the fix URL.
_PROVIDER_POLICY_BLOCKED_PATTERNS = [
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
]

# Provider content-policy / safety-filter blocks — *per-prompt* decisions by
# the upstream model provider (unlike the OpenRouter account-level block
# above). Deterministic for the unchanged request, so retrying burns paid
# attempts on a refusal; switch to a fallback immediately or surface guidance.
# Patterns are verbatim strings from specific safety pipelines, never generic
# words ("policy", "violation") that collide with billing/auth/format errors.
_CONTENT_POLICY_BLOCKED_PATTERNS = [
    # OpenAI Codex (#18028) — message may arrive without an HTTP status
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
    # OpenAI moderation — chat completions / responses ("usage policies"
    # disambiguates from billing's "exceeded ... policy")
    "violates our usage policies",
    "violates openai's usage policies",
    "your request was flagged by",
    # Anthropic safety system
    "prompt was flagged by our safety",
    "responses cannot be generated due to safety",
    # Azure / OpenAI Responses: ``content_filter`` is the OpenAI-standard
    # error/finish token; ``responsibleaipolicyviolation`` is Azure's code.
    # Deliberately NOT the space variant ("content filter") — it appears in
    # benign config/tooltip text that providers echo back.
    "content_filter",
    "responsibleaipolicyviolation",
    # MiniMax output-layer safety filter, surfaced verbatim as "output
    # new_sensitive (1027)" when the model's *output* (often a big tool-call
    # block) trips the filter and the SSE stream is truncated. Narrow enough
    # not to collide with billing/format/auth strings. (#32421)
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

# Provider empty-response advisories (OpenRouter / nano-gpt / similar).
# Checked before context-overflow matching: the advisory often mentions
# "max_tokens" as a cause, which sent healthy sessions into a compression
# death spiral ending in "Cannot compress further".
_EMPTY_PROVIDER_RESPONSE_PATTERNS = [
    "returned an empty response",
    "empty response despite retries",
    "provider returned an empty response",
    "model returning empty responses",
    "empty response stream",
]

# Provider-side timeout wording when the exception type is generic (e.g.
# RuntimeError from a local shim wrapping a subprocess timeout). Checked
# before the type-based transport heuristics so these don't fall into the
# unknown bucket and get misreported as empty responses.
_TIMEOUT_MESSAGE_PATTERNS = [
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
]

# Connection-establishment / DNS failure wording for generic exception types
# (local shim, MCP bridge, SDK re-raising without chaining) where
# _TRANSPORT_ERROR_TYPES never fires and there is no HTTP status. Without
# message matching these become ``unknown`` and miss the transport
# eager-fallback path. (ported from anomalyco/opencode#40707)
# Deliberately EXCLUDES mid-stream disconnect strings ("connection reset by
# peer", "unexpected eof", ...) — those belong to _SERVER_DISCONNECT_PATTERNS,
# which runs later and routes large sessions to compression. A connection
# never established cannot be a server-side overflow rejection.
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

# Transport error type names
_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    # Transient mid-stream SSL/TLS failures. ssl.SSLError is an OSError (caught
    # by isinstance) but the names are listed so provider-wrapped SSL errors
    # that lose the exception chain still classify as transport.
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    # OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
})

# Server disconnect patterns (no status code, transport-level). Ambiguous: a
# plain close may be a transient hiccup OR a server-side context-overflow
# rejection (gateways disconnect instead of returning an HTTP error for
# oversized requests). Large session + one of these → compression recovery.
_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
]

# SSL certificate verification failures — deterministic, NOT transient (TLS-
# inspecting proxy, missing CA, expired/self-signed cert). Burning the retry
# budget hides the actionable fix for minutes. Must be checked BEFORE
# _SSL_TRANSIENT_PATTERNS: these messages usually also contain "[SSL:", which
# would otherwise match the transient list and retry forever.
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

# SSL/TLS transient failures — distinct from _SERVER_DISCONNECT_PATTERNS. A
# mid-stream SSL alert is a transport hiccup (flaky network, renegotiation,
# LB drop), NOT an overflow signal: retry but never compress. OpenSSL 3.x
# changed the code separator (SSLV3_ALERT_BAD_RECORD_MAC → SSL/TLS_ALERT_...),
# so match stable substrings that survive format churn.
_SSL_TRANSIENT_PATTERNS = [
    # Space-separated (human-readable form, Python ssl module, most SDKs)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    # Underscore-separated OpenSSL tokens (ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC)
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    # Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
]


# ── Classification pipeline ─────────────────────────────────────────────

def _billing_verdict(result_fn, **overrides) -> ClassifiedError:
    """Non-retryable billing: rotate credential and fall back."""
    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
        **overrides,
    )


def _rate_limit_verdict(result_fn) -> ClassifiedError:
    """Retryable rate limit: rotate credential and fall back."""
    return result_fn(
        FailoverReason.rate_limit,
        retryable=True,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _overflow_or_empty_response(error_msg: str, result_fn) -> Optional[ClassifiedError]:
    """Shared body-refinement for 400/5xx: empty-response advisory vs overflow.

    Empty-provider-response advisories often mention "max_tokens" and must
    NOT enter compression (they used to thrash until "Cannot compress
    further" on a healthy session); an explicit overflow signal routes to
    compress-and-retry. Order matters: advisories first.
    """
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(FailoverReason.server_error, retryable=True, should_compress=False)
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(FailoverReason.context_overflow, retryable=True, should_compress=True)
    return None


def _policy_or_model_not_found(error_msg: str, result_fn) -> Optional[ClassifiedError]:
    """Aggregator policy block (checked first so it isn't mislabelled as a
    missing model), then model-not-found."""
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(FailoverReason.provider_policy_blocked, retryable=False, should_fallback=False)
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
    return None


def _body_error_message(body: dict) -> str:
    """Lowercased message from ``error.message``, then flat ``message``.

    Also reads litellm/Bedrock proxy shapes ({"errorMessage", "errorArgs":
    {"reason"}}): without them a long descriptive rejection looks like a bare
    "generic" error and, on a large session, mis-routes into compression.
    """
    if not isinstance(body, dict):
        return ""
    err_obj = body.get("error", {})
    msg = ""
    if isinstance(err_obj, dict):
        msg = str(err_obj.get("message") or "").strip().lower()
    if not msg:
        msg = str(body.get("message") or "").strip().lower()
    if not msg:
        msg = str(body.get("errorMessage") or "").strip().lower()
    if not msg:
        args = body.get("errorArgs")
        if isinstance(args, dict):
            msg = str(args.get("reason") or "").strip().lower()
    return msg


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

    Priority-ordered pipeline:
      0. Plugin ``transform_api_error_classification`` hooks (first valid result wins)
      1. Special-case provider-specific patterns (thinking sigs, tier gates)
      2. HTTP status code + message-aware refinement
      3. Error code classification (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL cert-verify → fail fast; SSL/TLS transient alerts → retry as timeout
      6. Server disconnect + large session → context overflow
      7. Stale-call circuit breaker → failover; transport error heuristics
      8. Fallback: unknown (retryable with backoff)

    ``approx_tokens``/``context_length``/``num_messages`` feed the large-session
    heuristics; ``provider``/``model`` are echoed into the result.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429
    # so rate-limit handling (reason, pool rotation, fallback gating) fires.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)
    response_headers = _extract_response_headers(error)

    # Build the pattern-matching text. str(error) alone may omit the body
    # message (OpenAI SDK APIStatusError.__str__ returns the first arg), so
    # append it; also unwrap OpenRouter's metadata.raw, where the real upstream
    # message ("context length exceeded") lives inside an inner JSON string.
    _raw_msg = str(error).lower()
    _body_msg = ""
    _metadata_msg = ""
    if isinstance(body, dict):
        _err_obj = body.get("error", {})
        if isinstance(_err_obj, dict):
            _body_msg = str(_err_obj.get("message") or "").lower()
            _metadata = _err_obj.get("metadata", {})
            if isinstance(_metadata, dict):
                _raw_json = _metadata.get("raw") or ""
                if isinstance(_raw_json, str) and _raw_json.strip():
                    try:
                        import json
                        _inner = json.loads(_raw_json)
                        if isinstance(_inner, dict):
                            _inner_err = _inner.get("error", {})
                            if isinstance(_inner_err, dict):
                                _metadata_msg = str(_inner_err.get("message") or "").lower()
                    except (json.JSONDecodeError, TypeError):
                        pass
        if not _body_msg:
            _body_msg = str(body.get("message") or "").lower()
    parts = [_raw_msg]
    if _body_msg and _body_msg not in _raw_msg:
        parts.append(_body_msg)
    if _metadata_msg and _metadata_msg not in _raw_msg and _metadata_msg not in _body_msg:
        parts.append(_metadata_msg)
    error_msg = " ".join(parts)
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
    # Consulted BEFORE the built-in pipeline so a provider plugin can add or
    # correct classifications (contract: ``transform_api_error_classification``
    # in hermes_cli.plugins.VALID_HOOKS). Callback exceptions and malformed
    # returns are handled inside the helper; this guard only covers
    # import/dispatch failure, so a broken plugin never breaks classification.
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

    # Content-policy / safety block: deterministic refusal of THIS prompt.
    # Runs before status classification so a 400 block isn't downgraded to
    # ``format_error`` and a status-less block (OpenAI Codex SDK) isn't left
    # retryable in ``unknown``. (#18028)
    if any(p in error_msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _result(FailoverReason.content_policy_blocked, retryable=False, should_fallback=True)

    # Anthropic thinking-block 400s. Two failure modes, one recovery (strip
    # reasoning_details and retry without thinking blocks — see the
    # thinking_signature handler in conversation_loop.py):
    #   1. Signature mismatch — blocks are signed against the full turn; any
    #      mutation (compression, truncation, merging) invalidates it.
    #      Pattern: "signature" + "thinking".
    #   2. Frozen-block mutation — "`thinking` ... blocks in the latest
    #      assistant message cannot be modified ... must remain as they were".
    #      No "signature" token, so it used to hard-abort as a client error.
    # Not gated on provider: OpenRouter proxies Anthropic errors verbatim.
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

    # Anthropic OAuth subscription rejects the 1M-context beta header (400 "The
    # long context beta is not yet available for this subscription."). The
    # recovery in run_agent.py rebuilds the client without the beta and retries
    # once. Different status + phrase from the 429 tier gate above.
    if status_code == 400 and "long context beta" in error_msg and "not yet available" in error_msg:
        return _result(FailoverReason.oauth_long_context_beta_forbidden, retryable=True, should_compress=False)

    # llama.cpp's json-schema-to-grammar rejects regex escapes (``\d``/``\w``)
    # and most ``format`` values that MCP servers routinely emit in ``pattern``.
    # On match the retry loop strips pattern/format from the tools and retries
    # once; cloud providers accept these keywords and never hit this branch.
    # Exclude the Qwen/vLLM "No user query found" template failure that local
    # engines wrap as "Unable to generate parser for this template" — that is a
    # poisoned transcript (→ _INVALID_MESSAGE_BODY_PATTERNS / format_error),
    # and stripping keywords would retry uselessly.
    if status_code == 400:
        _llama_cpp_grammar_hit = (
            "error parsing grammar" in error_msg
            or "json-schema-to-grammar" in error_msg
            or ("unable to generate parser" in error_msg and "template" in error_msg)
        )
    else:
        _llama_cpp_grammar_hit = False
    if _llama_cpp_grammar_hit and _NO_USER_QUERY_SIGNAL not in error_msg:
        return _result(FailoverReason.llama_cpp_grammar_pattern, retryable=True, should_compress=False)

    # xAI Grok subscription entitlement ("run out of available resources or do
    # not have an active Grok subscription") arrives two ways: HTTP 403, which
    # _classify_by_status routes to auth and _is_entitlement_failure then
    # stops the refresh loop; or an SSE ``type=error`` frame with
    # status_code=None, which skips the status path and matches no message
    # list — without this guard it falls to retryable ``unknown`` and
    # _is_entitlement_failure never runs (it only fires under auth).
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

    # Local MoA streaming compatibility errors are adapter-shape bugs, not a
    # provider outage; falling back would silently swap the MoA route for a
    # single-model answer (#55933 follow-up).
    if provider_lower == "moa" and (
        "'types.SimpleNamespace' object is not iterable" in str(error)
        or "'types.SimpleNamespace' object has no attribute 'index'" in str(error)
    ):
        return _result(FailoverReason.format_error, retryable=False, should_fallback=False)

    # Local MoA config drift (persisted session names a renamed/deleted preset)
    # is deterministic — retrying makes a config error look like an outage.
    from agent.errors import MoAPresetNotFoundError

    if isinstance(error, MoAPresetNotFoundError):
        return _result(FailoverReason.model_not_found, retryable=False)

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(error_msg, _result)
    if classified is not None:
        return classified

    # ── 5. SSL certificate verification failures → fail fast ────────
    # Deterministic for the host; checked BEFORE the transient-SSL patterns
    # because cert-verify messages also contain "[ssl:".
    if any(p in error_msg for p in _SSL_CERT_VERIFY_PATTERNS):
        return _result(FailoverReason.ssl_cert_verification, retryable=False, should_fallback=False)

    # ── 5b. SSL/TLS transient errors → retry as timeout (not compression) ──
    # Mid-stream SSL alerts are transport hiccups, not overflow signals; checked
    # before the disconnect step so a large session isn't compressed for a
    # flaky handshake. Also catches SDK re-raises that lose the ssl.SSLError type.
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Must precede the generic transport catch: a disconnect on a large
    # session is more likely overflow than a hiccup; otherwise
    # RemoteProtocolError always maps to timeout regardless of size.
    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and not status_code:
        # Reasoning-model override: a disconnect here is far more likely the
        # upstream proxy idle-killing a multi-minute thinking stream (NVIDIA NIM
        # ~120s, NVIDIA/NemoClaw#4846; OpenAI/Anthropic similar) than a true
        # overflow, even on large sessions — compressing would silently delete
        # history on a phantom context-length error. The per-model stale-timeout
        # floor in agent/reasoning_timeouts.py makes a real transport failure
        # recoverable via retry, so reclassify as timeout. (Part 1 of #52310)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute token/message thresholds only proxy for smaller windows;
        # large-context sessions can hold hundreds of messages well under budget.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(FailoverReason.context_overflow, retryable=True, should_compress=True)
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7b. Stale-call circuit breaker → failover immediately ──────
    # _check_stale_giveup() (agent/chat_completion_helpers.py) raises
    # RuntimeError after N consecutive stale attempts, *before* any network
    # call. Not a transport timeout: as ``unknown``/retryable it would burn
    # all max_retries instantly against the same dead provider before
    # fallback. Non-retryable + should_fallback activates fallback on first hit.
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
        # Not retryable on its own — pool rotation and provider refresh (Codex,
        # Anthropic, Nous) run before the retryability check in run_agent.py
        # and ``continue`` on success; on failure retryable=False hits the
        # client-error abort path (which tries fallback first).
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # OpenRouter 403 "key limit exceeded" is billing; other providers also
        # use 403 for account-plan or credit exhaustion.
        if (
            (provider == "xai-oauth" and error_code.lower() == _XAI_SPENDING_LIMIT_ERROR_CODE)
            or "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return _billing_verdict(result_fn)
        return result_fn(FailoverReason.auth, retryable=False, should_fallback=True)

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        # Nous API surfaces HA/NAS credit depletion as a paid model becoming
        # unavailable on the Free Tier (404, not 402) — billing, not a missing
        # model, so the loop can show top-up guidance.
        if any(p in error_msg for p in _BILLING_PATTERNS):
            return _billing_verdict(result_fn)
        # OpenRouter policy-block 404 first (model exists; fallback won't help;
        # body carries the fix URL), then model-not-found.
        classified = _policy_or_model_not_found(error_msg, result_fn)
        if classified is not None:
            return classified
        # A bare id the catalogue only knows prefixed is a malformed model id:
        # NVIDIA NIM answers with a naked "404 page not found", so the generic
        # branch below would burn retries and report an outage (#78796).
        if _model_id_missing_known_prefix(model, provider):
            return result_fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)
        # Generic 404 with no model signal: wrong endpoint path (local
        # llama.cpp/Ollama/vLLM misconfig), proxy glitch, or transient backend
        # issue. model_not_found would silently fall back and tell the model
        # it is missing — wrong and wastes a turn. Let the loop surface it.
        return result_fn(FailoverReason.unknown, retryable=True)

    if status_code == 413:
        return result_fn(FailoverReason.payload_too_large, retryable=True, should_compress=True)

    if status_code == 429:
        # long_context_tier already handled upstream. Z.AI/Zhipu reuse 429 for
        # server-wide overload: the credential is valid, so back off and retry
        # the same key rather than rotating (burns the pool, useless for a
        # single-key user). (#14038)
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(FailoverReason.overloaded, retryable=True)
        # OpenRouter-wrapped upstream 429 ("Provider returned error"): the
        # upstream model is throttled, the user's key is healthy — rotating
        # would bench it ~24min for nothing. Fall back to a different model.
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
        # Account/subscription usage exhaustion is a quota wall, not a rate
        # throttle; Anthropic returns it as 429 (``usage_limit_reached``,
        # #93419) and the generic branch used to retry it. Also covers
        # _USAGE_LIMIT_PATTERNS for other providers' hard quota walls, but ONLY
        # when the message isn't an explicit rate-limit phrase — otherwise
        # "Rate limit exceeded" ("limit exceeded") would promote to billing.
        # Periodic quotas with an explicit reset/retry signal stay rate_limit.
        # Explicit billing phrases in a 429 body are a hard wall too (#39441).
        has_usage_limit = (
            error_code.lower() == "usage_limit_reached"
            or "usage_limit_reached" in error_msg
            or any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
        )
        has_billing = any(p in error_msg for p in _BILLING_PATTERNS)
        has_explicit_rate_limit = any(p in error_msg for p in _RATE_LIMIT_PATTERNS)
        has_transient_signal = _has_usage_limit_transient_signal(error_msg, body, response_headers)
        if (has_billing or has_usage_limit) and not has_explicit_rate_limit and not has_transient_signal:
            return _billing_verdict(result_fn)
        return _rate_limit_verdict(result_fn)

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Some OpenAI-compatible gateways (codex.nekos.me → 502) return
        # request-validation errors as 5xx. Deterministic, so the generic
        # "5xx → retryable" rule would turn one bad request into a retry
        # flood: detect the signals (message or structured code) and fail fast.
        # Exception: a param WE never sent on this route was injected by the
        # provider itself, so retrying is correct (mirrors _classify_400).
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in {"invalid_request_error", "unknown_parameter", "unsupported_parameter"}
        ) and not _is_server_injected_param_rejection(error_msg, provider):
            return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)
        # llama.cpp/llama-server report context overflow as HTTP 500 instead of
        # 400/413; route explicit overflow into compression (mirroring
        # _classify_400) instead of blind server_error retries that drop the turn.
        classified = _overflow_or_empty_response(error_msg, result_fn)
        if classified is not None:
            return classified
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        # Same overflow-as-5xx variant (server busy / model-load OOM, or a
        # Cloudflare/Tailscale hop relabeling the status); otherwise transient
        # overload.
        classified = _overflow_or_empty_response(error_msg, result_fn)
        if classified is not None:
            return classified
        return result_fn(FailoverReason.overloaded, retryable=True)

    # 408 Request Timeout is retry-safe by definition (RFC 9110 §15.5.9), not a
    # malformed request; reverse proxies in front of llama.cpp/Ollama/vLLM emit
    # it when a long generation outruns the read window. Route to ``timeout``
    # rather than the generic 4xx abort below.
    if status_code == 408:
        return result_fn(FailoverReason.timeout, retryable=True)

    # Other 4xx — non-retryable
    if 400 <= status_code < 500:
        return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)

    # Other 5xx — retryable
    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _has_usage_limit_transient_signal(error_msg: str, body: dict, response_headers) -> bool:
    """Return whether a usage-limit response identifies a reset window."""
    if any(pattern in error_msg for pattern in _USAGE_LIMIT_TRANSIENT_SIGNALS):
        return True

    payloads = [body]
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        payloads.append(body["error"])
    reset_fields = ("resets_in_seconds", "resets_at", "reset_at", "retry_after")
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if any(payload.get(f) is not None and payload.get(f) != "" for f in reset_fields):
            return True

    if response_headers and hasattr(response_headers, "get"):
        for header in ("retry-after", "Retry-After", "x-ratelimit-reset", "X-RateLimit-Reset"):
            value = response_headers.get(header)
            if value is not None and value != "":
                return True
    return False


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: billing exhaustion vs transient usage limit.

    Some 402s are rate limits disguised as payment errors — "Usage limit, try
    again in 5 minutes" is a periodic quota that resets, not a billing problem.
    """
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
    if has_usage_limit and has_transient_signal:
        return _rate_limit_verdict(result_fn)
    return _billing_verdict(result_fn)


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic.

    Ordering is load-bearing; every early branch below explains why it must
    precede context_overflow.
    """

    # Multimodal tool content rejected. BEFORE image_too_large (different
    # recovery: strip image parts from tool messages, mark the model
    # no-list-tool-content for the session) and BEFORE context_overflow
    # because "text is not set" is ambiguous alone but specific for a 400 on a
    # request known to carry multimodal tool content.
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(FailoverReason.multimodal_tool_content_unsupported, retryable=True)

    # Image corruption (xAI's undecodable-image check). BEFORE image_too_large:
    # corrupt bytes need strip-and-retry — shrinking can't repair a bad PNG.
    if any(p in error_msg for p in _IMAGE_CORRUPT_PATTERNS):
        return result_fn(FailoverReason.image_corrupt, retryable=True)

    # Image too large (Anthropic's 5 MB per-image check). BEFORE context_overflow:
    # messages can trip both ("exceeds" + "image") and shrinking is cheaper.
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(FailoverReason.image_too_large, retryable=True)

    # Invalid encrypted reasoning replay blob (OpenAI Responses). BEFORE
    # context_overflow: "encrypted content ... could not be verified" can trip
    # the overflow heuristics. ``error_msg`` is lowercased upstream.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or ("encrypted content for item" in error_msg and "could not be verified" in error_msg)
        or "could not decrypt the provided encrypted_content" in error_msg
    ):
        return result_fn(FailoverReason.invalid_encrypted_content, retryable=True, should_fallback=False)

    # Server-injected parameter rejection: a 400 blaming a field the client
    # never sent. MUST precede the request-validation branch, which would abort
    # the turn as a deterministic format_error. Observed live on the Codex
    # OAuth backend: it intermittently adds ``prompt_cache_retention`` to its
    # own upstream call and rejects it (~20% over n=20 on a 1-message request
    # with no cache params); a byte-identical retry succeeds.
    if _is_server_injected_param_rejection(error_msg, provider):
        # The request shape was fine — never route this into compression.
        return result_fn(FailoverReason.server_error, retryable=True, should_compress=False)

    # Request-validation (unsupported/unknown parameter) MUST precede
    # context_overflow: GPT-5 rejecting max_tokens says "Unsupported parameter:
    # 'max_tokens' ... Use 'max_completion_tokens'", and "max_tokens" is an
    # overflow pattern — without this guard the 400 enters the compression
    # loop and ends in "Cannot compress further". Deterministic → fail fast.
    # NOTE: deliberately NOT keyed off generic ``invalid_request_error`` —
    # OpenAI stamps that code on genuine overflow 400s too.
    if (
        any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS if p != "invalid_request_error")
        or error_code_lower in {"unknown_parameter", "unsupported_parameter"}
    ):
        return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)

    # Malformed message array (empty-content assistant stub, etc.). BEFORE
    # context_overflow: the input can be tiny, so the "400 + large session"
    # heuristic would thrash compression on an unchanged request. Checked
    # against message text AND the structured code (litellm/Bedrock surface
    # errorCode=INVALID_REQUEST_BODY).
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
        return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)

    # Empty-response advisory (no compression) / explicit context overflow.
    classified = _overflow_or_empty_response(error_msg, result_fn)
    if classified is not None:
        return classified

    # Some providers return policy-block / model-not-found as 400 instead of 404.
    classified = _policy_or_model_not_found(error_msg, result_fn)
    if classified is not None:
        return classified

    # Some providers return rate limit / billing as 400 instead of 429/402.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return _rate_limit_verdict(result_fn)
    if any(p in error_msg for p in _BILLING_PATTERNS):
        # "out of extra usage" on a 400 may be a content-filter rejection
        # (#82154): mark unverified so surfaces hedge and the pool skips the bench.
        return _billing_verdict(result_fn, error_context=_billing_ambiguity_context(error_msg))

    # Generic 400 + large session → probable context overflow (Anthropic can
    # return a bare "Error" message when context is too large).
    err_body_msg = _body_error_message(body)
    is_generic = len(err_body_msg) < 30 or err_body_msg in {"error", ""}
    # Absolute token/message thresholds only proxy for smaller windows;
    # large-context sessions can hold many messages well under budget.
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256000 and (approx_tokens > 80000 or num_messages > 80)
    )
    if is_generic and is_large:
        return result_fn(FailoverReason.context_overflow, retryable=True, should_compress=True)

    # Non-retryable format error
    return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)


# ── Error code classification ───────────────────────────────────────────

def _classify_by_error_code(error_code: str, error_msg: str, result_fn) -> Optional[ClassifiedError]:
    """Classify by structured error codes from the response body."""
    code_lower = error_code.lower()

    # Some OpenAI-compatible endpoints encode deterministic request-validation
    # failures as plain-text ``event: error`` SSE data behind HTTP 200. The
    # unchanged request cannot succeed, but a configured fallback still may.
    if code_lower == PROVIDER_STREAM_NON_JSON_ERROR_CODE and "request validation failed:" in error_msg:
        return result_fn(FailoverReason.format_error, retryable=False, should_fallback=True)

    if code_lower in {"resource_exhausted", "throttled", "rate_limit_exceeded"}:
        return result_fn(FailoverReason.rate_limit, retryable=True, should_rotate_credential=True)

    if code_lower in _BILLING_ERROR_CODES:
        return _billing_verdict(result_fn)

    if code_lower in {"model_not_found", "model_not_available", "invalid_model"}:
        return result_fn(FailoverReason.model_not_found, retryable=False, should_fallback=True)

    if code_lower in {"context_length_exceeded", "max_tokens_exceeded"}:
        return result_fn(FailoverReason.context_overflow, retryable=True, should_compress=True)

    if code_lower == "invalid_encrypted_content":
        return result_fn(FailoverReason.invalid_encrypted_content, retryable=True, should_fallback=False)

    return None


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(error_msg: str, result_fn) -> Optional[ClassifiedError]:
    """Classify on message patterns when no status code is available.

    Mirrors the 400 path's ordering (payload → multimodal → image corrupt →
    image too large → usage limit → overloaded → billing → rate limit →
    empty-response/overflow → auth → policy/model → timeout → connection).
    """
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(FailoverReason.payload_too_large, retryable=True, should_compress=True)

    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(FailoverReason.multimodal_tool_content_unsupported, retryable=True)

    if any(p in error_msg for p in _IMAGE_CORRUPT_PATTERNS):
        return result_fn(FailoverReason.image_corrupt, retryable=True)

    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(FailoverReason.image_too_large, retryable=True)

    # Usage-limit needs the same disambiguation as 402: a transient signal
    # ("try again", "resets at") means a periodic quota, not exhaustion.
    if any(p in error_msg for p in _USAGE_LIMIT_PATTERNS):
        if any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS):
            return _rate_limit_verdict(result_fn)
        return _billing_verdict(result_fn)

    # Overloaded BEFORE rate_limit/billing so a message-only "overloaded" (no
    # 503/529, e.g. Anthropic-compatible proxies) backs off instead of falling
    # to ``unknown`` or rotating credentials.
    if any(p in error_msg for p in _OVERLOADED_PATTERNS):
        return result_fn(FailoverReason.overloaded, retryable=True)

    if any(p in error_msg for p in _BILLING_PATTERNS):
        # Adapters can strip the status from Anthropic's "out of extra usage"
        # 400, so the same ambiguity marking applies here (#82154).
        return _billing_verdict(result_fn, error_context=_billing_ambiguity_context(error_msg))

    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return _rate_limit_verdict(result_fn)

    classified = _overflow_or_empty_response(error_msg, result_fn)
    if classified is not None:
        return classified

    # Auth is never retried directly — the key is invalid and will fail again;
    # retryable=False makes the caller rotate or fall back instead.
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    classified = _policy_or_model_not_found(error_msg, result_fn)
    if classified is not None:
        return classified

    # Timeout wording from generic exception types (local shims / custom
    # providers wrapping a subprocess/HTTP timeout): transport timeout so the
    # loop rebuilds the client instead of treating the turn as empty output.
    if any(p in error_msg for p in _TIMEOUT_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    # Connection/DNS failure wording — same generic-type problem; classified
    # as timeout (the transport bucket) so eager transport fallback and client
    # rebuild apply. Never compression: nothing was ever sent.
    if any(p in error_msg for p in _CONNECTION_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    return None


# ── Helpers ─────────────────────────────────────────────────────────────

def _cause_chain(error: Exception, depth: int = 5):
    """Yield ``error`` then its __cause__/__context__ chain, bounded to ``depth``."""
    current = error
    for _ in range(depth):
        yield current
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause


def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
    for current in _cause_chain(error):
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs use .status instead of .status_code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
    return None


def _extract_error_body(error: Exception) -> dict:
    """Extract the structured error body from an SDK exception or its cause chain."""
    for current in _cause_chain(error):
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            return body
        # Some errors have .response.json()
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


def _clean_code(code) -> str:
    """Stringify a code/type value; drop blanks and the useless literal "400"."""
    if isinstance(code, (str, int)):
        text = str(code).strip()
        if text and text != "400":
            return text
    return ""


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    if not body:
        return ""

    def _code_from_payload(payload) -> str:
        """Code/type from a nested error payload dict (defensive)."""
        if not isinstance(payload, dict):
            return ""
        payload_error = payload.get("error", {})
        if isinstance(payload_error, dict):
            nested = payload_error.get("code") or payload_error.get("type") or ""
            if isinstance(nested, str) and _clean_code(nested):
                return _clean_code(nested)
        return _clean_code(payload.get("code") or payload.get("error_code") or "")

    error_obj = body.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and _clean_code(code):
            return _clean_code(code)

        # Some providers wrap the real JSON error body as a string inside
        # error.message — peek for a nested code (Responses API surfaces
        # ``invalid_encrypted_content`` this way).
        message = error_obj.get("message")
        if isinstance(message, str) and message.strip().startswith("{"):
            import json
            try:
                inner = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                inner = None
            nested_code = _code_from_payload(inner)
            if nested_code:
                return nested_code

    # Top-level code
    return _clean_code(body.get("code") or body.get("error_code") or body.get("errorCode") or "")


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message (structured body first)."""
    if body:
        error_obj = body.get("error", {})
        candidates = []
        if isinstance(error_obj, dict):
            candidates.append(error_obj.get("message", ""))
        candidates.append(body.get("message", ""))
        # litellm / Bedrock proxy shape: {"errorMessage": ..., "errorArgs": {"reason": ...}}
        candidates.append(body.get("errorMessage", ""))
        args = body.get("errorArgs")
        if isinstance(args, dict):
            candidates.append(args.get("reason", ""))
        for msg in candidates:
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
    return str(error)[:500]


def _is_openrouter_upstream_error(body: Any, provider: str) -> bool:
    """Detect OpenRouter's aggregator-wrapped upstream provider errors.

    OpenRouter wraps upstream errors (DeepSeek, Anthropic, ...) with the outer
    message "Provider returned error" and the real error in ``metadata.raw``.
    The user's OpenRouter key is healthy, so credential rotation is wrong.
    """
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    if str(err.get("message") or "").strip().lower() != "provider returned error":
        return False
    # Require the explicit OpenRouter provider OR the metadata shape only
    # OpenRouter produces (metadata.raw / metadata.provider_name).
    if (provider or "").strip().lower() == "openrouter":
        return True
    metadata = err.get("metadata")
    return isinstance(metadata, dict) and ("raw" in metadata or "provider_name" in metadata)


def _extract_upstream_provider_name(body: Any) -> Optional[str]:
    """Pull the upstream provider name out of OpenRouter's error metadata."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    metadata = err.get("metadata")
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("provider_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None
