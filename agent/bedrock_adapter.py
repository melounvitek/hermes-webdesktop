"""AWS Bedrock Converse API adapter for Hermes Agent.

Talks to Bedrock through the native Converse API (boto3) instead of an
OpenAI-compatible endpoint, so the AWS credential chain (IAM roles, SSO
profiles, env vars, instance metadata), cross-region inference profiles,
guardrails and control-plane model discovery all work without API keys.

Same layout as ``anthropic_adapter.py``: all Bedrock-specific logic lives
here; OpenAI-format messages/tools are converted to Converse format on the
way in and responses are normalized back to OpenAI-shaped objects for the
agent loop. Requires ``boto3`` (optional dependency).
"""

import base64
import json
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# boto3 is not in the [all] extras; lazy_deps installs it on demand so the
# provider works without baking boto3 into the base image.
try:
    from tools.lazy_deps import ensure
    ensure("provider.bedrock", prompt=False)
except Exception:
    pass  # lazy_deps unavailable or install failed — let downstream imports surface the real error


_bedrock_runtime_client_cache: Dict[str, Any] = {}
_bedrock_control_client_cache: Dict[str, Any] = {}

# Bedrock-hosted OpenAI GPT-5.x models are not exposed through the Converse
# runtime; AWS serves them from the Bedrock Mantle OpenAI-compatible Responses
# endpoint (https://bedrock-mantle.<region>.api.aws/openai/v1). Keep the
# allowlist narrow so Converse-capable GPT-OSS models stay on the native path.
BEDROCK_OPENAI_RESPONSES_MODEL_IDS: Tuple[str, ...] = (
    "openai.gpt-5.5",
    "openai.gpt-5.6-sol",
    "openai.gpt-5.6-terra",
    "openai.gpt-5.6-luna",
)
_BEDROCK_OPENAI_HOST_RE = re.compile(r"^bedrock-mantle\.([a-z0-9-]+)\.api\.aws$", re.IGNORECASE)


_MIN_BOTO3_VERSION = (1, 34, 59)


def _require_boto3():
    """Import boto3, raising a clear error if not installed or too old.

    converse()/converse_stream() need boto3 >= 1.34.59; a system boto3 (e.g.
    Ubuntu's 1.34.46) can shadow the venv pin under editable installs.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "The 'boto3' package is required for the AWS Bedrock provider. "
            "Install it with: pip install boto3\n"
            "Or install Hermes with Bedrock support: pip install -e '.[bedrock]'"
        )
    try:
        version = tuple(int(x) for x in boto3.__version__.split(".")[:3])
    except (AttributeError, ValueError):
        return boto3  # can't parse — don't block on version check
    if version < _MIN_BOTO3_VERSION:
        raise RuntimeError(
            f"boto3 {boto3.__version__} does not support converse_stream "
            f"(minimum 1.34.59 required). Upgrade with: "
            f"pip install --upgrade boto3"
        )
    return boto3


def _cached_client(cache: Dict[str, Any], service: str, region: str):
    """Get or create a per-region boto3 client using the default credential chain."""
    if region not in cache:
        cache[region] = _require_boto3().client(service, region_name=region)
    return cache[region]


def _get_bedrock_runtime_client(region: str):
    return _cached_client(_bedrock_runtime_client_cache, "bedrock-runtime", region)


def _get_bedrock_control_client(region: str):
    return _cached_client(_bedrock_control_client_cache, "bedrock", region)


def reset_client_cache():
    """Clear cached boto3 clients. Used in tests and profile switches."""
    _bedrock_runtime_client_cache.clear()
    _bedrock_control_client_cache.clear()


def invalidate_runtime_client(region: str) -> bool:
    """Evict one region's cached ``bedrock-runtime`` client (stale HTTP pool).

    Returns True if an entry was evicted.
    """
    existed = region in _bedrock_runtime_client_cache
    _bedrock_runtime_client_cache.pop(region, None)
    return existed


# --- Bedrock Mantle / OpenAI Responses support ---


def is_openai_bedrock_model(model_id: str) -> bool:
    """True for Bedrock-hosted OpenAI models that require Mantle (GPT-OSS excluded)."""
    normalized = str(model_id or "").strip().lower()
    return normalized in {m.lower() for m in BEDROCK_OPENAI_RESPONSES_MODEL_IDS}


def merge_bedrock_openai_model_ids(model_ids: List[str]) -> List[str]:
    """Append Mantle-only OpenAI models, which control-plane discovery never lists."""
    merged = list(model_ids or [])
    seen = {str(m).lower() for m in merged}
    for model_id in BEDROCK_OPENAI_RESPONSES_MODEL_IDS:
        if model_id.lower() not in seen:
            merged.append(model_id)
            seen.add(model_id.lower())
    return merged


def bedrock_openai_base_url(region: str) -> str:
    """Return Bedrock Mantle's OpenAI-compatible base URL for *region*."""
    resolved = (region or "").strip() or resolve_bedrock_runtime_region()
    return f"https://bedrock-mantle.{resolved}.api.aws/openai/v1"


def bedrock_openai_region_from_base_url(base_url: str) -> Optional[str]:
    """Extract the AWS region from a Bedrock Mantle OpenAI base URL."""
    host = urlparse(str(base_url or "")).hostname or ""
    match = _BEDROCK_OPENAI_HOST_RE.match(host)
    return match.group(1) if match else None


def is_bedrock_openai_base_url(base_url: str) -> bool:
    """True for Bedrock Mantle endpoints (bare host or /openai[/v1] path)."""
    parsed = urlparse(str(base_url or ""))
    if not _BEDROCK_OPENAI_HOST_RE.match(parsed.hostname or ""):
        return False
    path = (parsed.path or "").rstrip("/").lower()
    return path in {"", "/openai", "/openai/v1"}


def resolve_bedrock_bearer_token(env: Optional[Dict[str, str]] = None) -> str:
    """Return AWS_BEARER_TOKEN_BEDROCK when Bedrock API-key auth is configured."""
    env = env if env is not None else os.environ
    return (env.get("AWS_BEARER_TOKEN_BEDROCK", "") or "").strip()


class BedrockOpenAISigV4Auth(httpx.Auth):
    """httpx auth hook that SigV4-signs Bedrock Mantle OpenAI requests."""

    requires_request_body = True

    def __init__(self, region: str, service: str = "bedrock"):
        self.region = (region or "").strip() or resolve_bedrock_runtime_region()
        self.service = service

    def auth_flow(self, request):  # pragma: no cover - exercised by live call
        import botocore.session
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            raise RuntimeError(
                "No AWS credentials available for Bedrock OpenAI Responses. "
                "Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
                "SSO, or an instance/task role."
            )
        # SigV4 must own Authorization: drop the SDK's placeholder bearer header
        # but keep every other SDK header (content-type, accept, request IDs).
        headers = {
            str(k): str(v)
            for k, v in request.headers.items()
            if str(k).lower() not in {"authorization", "x-amz-date", "x-amz-security-token"}
        }
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
            headers=headers,
        )
        SigV4Auth(credentials.get_frozen_credentials(), self.service, self.region).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers.items()))
        yield request


def build_bedrock_openai_http_client(region: str, *, timeout: Optional[float] = None):
    """Build an httpx client that SigV4-signs Bedrock OpenAI requests."""
    kwargs: Dict[str, Any] = {"auth": BedrockOpenAISigV4Auth(region)}
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        kwargs["timeout"] = timeout
    return httpx.Client(**kwargs)


def configure_bedrock_openai_client_kwargs(
    client_kwargs: Dict[str, Any], *, timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Install SigV4 auth on OpenAI SDK kwargs for Bedrock Mantle.

    Real API keys (``AWS_BEARER_TOKEN_BEDROCK``) keep the SDK's bearer auth;
    the ``aws-sdk``/``no-key-required`` placeholders mean IAM credential-chain
    auth, so we attach a per-request SigV4 httpx client.
    """
    base_url = str(client_kwargs.get("base_url") or "")
    if not is_bedrock_openai_base_url(base_url):
        return client_kwargs
    api_key = client_kwargs.get("api_key")
    if isinstance(api_key, str) and api_key.strip() and api_key not in {"aws-sdk", "no-key-required"}:
        return client_kwargs
    region = bedrock_openai_region_from_base_url(base_url) or resolve_bedrock_runtime_region()
    client_kwargs["api_key"] = "aws-sdk"
    client_kwargs["http_client"] = build_bedrock_openai_http_client(region, timeout=timeout)
    return client_kwargs


# --- Stale-connection detection ---
# boto3 caches its HTTPS pool inside the client. A pooled connection killed
# out from under us (NAT timeout, VPN flap, RST, proxy idle cull) surfaces as
# botocore ConnectionClosedError / urllib3 ProtocolError, or as a bare
# AssertionError from urllib3's internal pool-state checks. Retrying with the
# same client reproduces the failure, so the fix is to evict the client.

_STALE_LIB_MODULE_PREFIXES = ("urllib3.", "botocore.", "boto3.")


def _traceback_frames_modules(exc: BaseException):
    """Yield the ``__name__`` of each frame in exc's traceback."""
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        yield tb.tb_frame.f_globals.get("__name__", "") or ""
        tb = tb.tb_next


def _stale_error_types() -> tuple:
    """botocore + urllib3 transport-failure exception classes (best-effort import)."""
    types: list = []
    try:
        from botocore.exceptions import ConnectionError as BotoConnectionError, HTTPClientError
        types += [BotoConnectionError, HTTPClientError]
    except ImportError:  # pragma: no cover — botocore always present with boto3
        pass
    try:
        from urllib3.exceptions import ConnectionError as Urllib3ConnectionError, NewConnectionError, ProtocolError
        types += [ProtocolError, NewConnectionError, Urllib3ConnectionError]
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


def is_stale_connection_error(exc: BaseException) -> bool:
    """True if ``exc`` indicates a dead/stale Bedrock HTTP connection.

    Matches botocore ConnectionError/HTTPClientError subclasses, urllib3
    transport errors, and bare ``AssertionError``s raised from inside
    urllib3/botocore/boto3. Application-level AssertionErrors are NOT matched.
    """
    if isinstance(exc, _stale_error_types()):
        return True
    if isinstance(exc, AssertionError):
        for module in _traceback_frames_modules(exc):
            if module.startswith(_STALE_LIB_MODULE_PREFIXES):
                return True
    return False


def is_streaming_access_denied_error(exc: BaseException) -> bool:
    """True when IAM denied ``bedrock:InvokeModelWithResponseStream``.

    InvokeModel-only policies reject converse_stream() permanently, so callers
    should fall back to non-streaming converse() instead of retrying. Detection
    is message-based because the AnthropicBedrock SDK wraps the same AWS
    response in its own exception types but preserves the action name.
    """
    msg = str(exc).lower()
    if "invokemodelwithresponsestream" not in msg:
        return False
    try:
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover — botocore always present with boto3
        ClientError = None  # type: ignore[assignment]
    if ClientError is not None and isinstance(exc, ClientError):
        code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")
        return code in ("AccessDeniedException", "UnauthorizedException")
    return "not authorized" in msg or "accessdenied" in msg


# --- AWS credential detection ---

# Priority order; the first group whose vars are ALL set names the auth source.
_AWS_AUTH_ENV_CHAIN: Tuple[Tuple[str, ...], ...] = (
    ("AWS_BEARER_TOKEN_BEDROCK",),                    # Bedrock bearer token
    ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),   # explicit IAM key pair
    ("AWS_PROFILE",),                                 # named profile (SSO, assume-role)
    ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",),      # ECS / CodeBuild
    ("AWS_WEB_IDENTITY_TOKEN_FILE",),                 # EKS IRSA
)


def _boto3_chain_has_credentials() -> bool:
    """True if boto3's default chain resolves credentials (IMDS, task role, ...)."""
    try:
        import botocore.session
        credentials = botocore.session.get_session().get_credentials()
        if credentials is not None:
            resolved = credentials.get_frozen_credentials()
            return bool(resolved and resolved.access_key)
    except Exception:
        pass
    return False


def resolve_aws_auth_env_var(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return the name of the active AWS auth source, or None.

    Env vars first (fast, no I/O), then ``"iam-role"`` if boto3's chain
    resolves implicit credentials (EC2 IMDS, ECS task role, Lambda).
    """
    env = env if env is not None else os.environ
    for group in _AWS_AUTH_ENV_CHAIN:
        if all(env.get(var, "").strip() for var in group):
            return group[0]
    return "iam-role" if _boto3_chain_has_credentials() else None


def has_aws_credentials(env: Optional[Dict[str, str]] = None) -> bool:
    """True if any AWS credential source (env vars or boto3 chain) is detected."""
    return resolve_aws_auth_env_var(env) is not None or _boto3_chain_has_credentials()


def resolve_bedrock_region(env: Optional[Dict[str, str]] = None) -> str:
    """AWS_REGION → AWS_DEFAULT_REGION → botocore configured region → us-east-1.

    The botocore fallback matters for EU/AP users who set the region in
    ~/.aws/config via a named profile; without it discovery returns us.* IDs.
    """
    env = env if env is not None else os.environ
    explicit = env.get("AWS_REGION", "").strip() or env.get("AWS_DEFAULT_REGION", "").strip()
    if explicit:
        return explicit
    try:
        import botocore.session
        region = botocore.session.get_session().get_config_variable("region")
        if region:
            return region
    except Exception:
        pass
    return "us-east-1"


def resolve_bedrock_runtime_region(config: Optional[Dict[str, Any]] = None) -> str:
    """``bedrock.region`` from config.yaml, else :func:`resolve_bedrock_region`.

    Matches ``hermes_cli/runtime_provider.py``. Every non-runtime Bedrock
    endpoint (auxiliary clients, picker discovery) must use this, otherwise
    auxiliary calls can leave the primary runtime's region when the config
    and ambient AWS env/profile disagree. Pass *config* to avoid a disk read.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            config = {}
    cfg_region = str(((config or {}).get("bedrock") or {}).get("region") or "").strip()
    return cfg_region or resolve_bedrock_region()


def bedrock_model_ids_or_none() -> Optional[List[str]]:
    """Live-discover Bedrock model IDs; None on failure/empty so callers use the static list."""
    try:
        discovered = discover_bedrock_models(resolve_bedrock_runtime_region())
        if discovered:
            return merge_bedrock_openai_model_ids([m["id"] for m in discovered])
    except Exception:
        pass
    return None


# --- Tool-calling / prompt-cache capability detection ---

# Models known to reject toolConfig with a ValidationException. Conservative:
# unknown models are assumed to support tools; add here when one fails.
_NON_TOOL_CALLING_PATTERNS = [
    "deepseek.r1",          # DeepSeek R1 — reasoning only
    "deepseek-r1",          # Alternate ID format
    "stability.",           # Image generation
    "cohere.embed",         # Embeddings
    "amazon.titan-embed",   # Embeddings
]


def _model_supports_tool_use(model_id: str) -> bool:
    """False for denylisted models; unknown models default to True."""
    model_lower = model_id.lower()
    return not any(pattern in model_lower for pattern in _NON_TOOL_CALLING_PATTERNS)


# Converse API cachePoint allowlist — inverted policy vs tools: an unsupported
# model rejects cachePoint with a ValidationException, so unknown models get
# NO cache markers until confirmed. Claude normally uses the AnthropicBedrock
# SDK path and only reaches build_converse_kwargs under bearer-token auth.
# Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
_CACHE_POINT_PATTERNS = [
    "anthropic.claude",  # bearer-token fallback path
    "amazon.nova",
]


def _model_supports_prompt_cache(model_id: str) -> bool:
    """True if the model accepts a Converse API cachePoint block."""
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in _CACHE_POINT_PATTERNS)


# --- Server-verdict cachePoint suppression ---
# Bedrock's real cachePoint rule is per-family AND per-field (Nova accepts it
# in system/messages but hard-fails on toolConfig.tools), and any static table
# drifts as AWS ships new families — failing 100% of turns with no workaround.
# So when Bedrock names a placement as unpermitted we record that verdict,
# drop the marker from that placement for the rest of the process, and retry
# the rejected request once without it.

CACHE_POINT_PLACEMENTS = ("tools", "system", "messages")

# model_id (lowercased) → placements Bedrock has rejected this process.
_CACHE_POINT_REJECTIONS: Dict[str, set] = {}

# "#/toolConfig/tools/18: extraneous key [cachePoint] is not permitted"
_CACHE_POINT_PATH_PATTERN = re.compile(r"#/(?P<path>[A-Za-z0-9_./\[\]-]*)", re.IGNORECASE)


def cache_point_rejection_placement(exc: BaseException) -> Optional[str]:
    """Return the Converse section whose cachePoint Bedrock refused, or None.

    Message-based on purpose: the JSON pointer in the ValidationException is
    the only thing that says *which* section was rejected, and the same
    wording arrives both as a raw botocore ClientError and wrapped by SDKs.
    An unlocalisable rejection maps to "tools" — the only placement any
    supported family is known to refuse while accepting the others.
    """
    msg = str(exc)
    lowered = msg.lower()
    if "cachepoint" not in lowered:
        return None
    if "not permitted" not in lowered and "extraneous" not in lowered:
        return None
    match = _CACHE_POINT_PATH_PATTERN.search(msg)
    path = (match.group("path") if match else "").lower()
    if "toolconfig" in path or "tools" in path:
        return "tools"
    if "system" in path:
        return "system"
    if "messages" in path:
        return "messages"
    return "tools"


def note_cache_point_rejection(model_id: str, placement: str) -> None:
    """Record that ``model_id`` refuses cachePoint blocks in ``placement``."""
    if placement not in CACHE_POINT_PLACEMENTS:
        return
    _CACHE_POINT_REJECTIONS.setdefault(model_id.lower(), set()).add(placement)


def cache_point_allowed(model_id: str, placement: str) -> bool:
    """False once Bedrock has refused this placement for this model."""
    return placement not in _CACHE_POINT_REJECTIONS.get(model_id.lower(), ())


def reset_cache_point_rejections() -> None:
    """Clear recorded cachePoint rejections. Used in tests."""
    _CACHE_POINT_REJECTIONS.clear()


def _is_cache_point_block(block: Any) -> bool:
    return isinstance(block, dict) and set(block.keys()) == {"cachePoint"}


def _without_cache_points(blocks: Any) -> Optional[list]:
    """Return ``blocks`` minus cachePoint entries, or None if not a list / nothing removed."""
    if not isinstance(blocks, list):
        return None
    cleaned = [b for b in blocks if not _is_cache_point_block(b)]
    return None if len(cleaned) == len(blocks) else cleaned


def strip_cache_points(kwargs: Dict[str, Any], placement: str) -> Dict[str, Any]:
    """Copy of Converse kwargs with ``placement``'s cachePoint removed.

    Returns the input unchanged (same object) when there was nothing to strip,
    which is what callers use to decide a retry cannot help.
    """
    if placement == "system":
        cleaned = _without_cache_points(kwargs.get("system"))
        return kwargs if cleaned is None else {**kwargs, "system": cleaned}

    if placement == "tools":
        tool_config = kwargs.get("toolConfig")
        cleaned = _without_cache_points((tool_config or {}).get("tools"))
        if cleaned is None:
            return kwargs
        return {**kwargs, "toolConfig": {**tool_config, "tools": cleaned}}

    if placement == "messages":
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return kwargs
        changed = False
        cleaned_messages = []
        for msg in messages:
            content = _without_cache_points(msg.get("content") if isinstance(msg, dict) else None)
            if content is not None:
                changed = True
                cleaned_messages.append({**msg, "content": content})
            else:
                cleaned_messages.append(msg)
        return {**kwargs, "messages": cleaned_messages} if changed else kwargs

    return kwargs


def recover_from_cache_point_rejection(exc: BaseException, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Record Bedrock's cachePoint verdict and return retry kwargs, or None.

    None means the error was not a cachePoint rejection, or the marker was
    already absent — retrying cannot change the outcome; caller must re-raise.
    """
    placement = cache_point_rejection_placement(exc)
    if placement is None:
        return None
    retry_kwargs = strip_cache_points(kwargs, placement)
    if retry_kwargs is kwargs:
        return None
    model_id = str(kwargs.get("modelId", ""))
    note_cache_point_rejection(model_id, placement)
    logger.warning(
        "bedrock: %s rejected a cachePoint block in %s — dropping that cache "
        "marker for this model and retrying. Prompt caching stays active for "
        "the remaining sections.",
        model_id or "model", placement,
    )
    return retry_kwargs


_REGIONAL_PREFIXES = ("global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af.")


def is_anthropic_bedrock_model(model_id: str) -> bool:
    """True for Claude on Bedrock (``anthropic.claude-*`` with any regional prefix).

    These use the AnthropicBedrock SDK path for full feature parity (prompt
    caching, thinking budgets); non-Claude models use the Converse path.
    """
    model_lower = model_id.lower()
    for prefix in _REGIONAL_PREFIXES:
        if model_lower.startswith(prefix):
            model_lower = model_lower[len(prefix):]
            break
    return model_lower.startswith("anthropic.claude")


# --- Message format conversion: OpenAI → Bedrock Converse ---

def convert_tools_to_converse(tools: List[Dict]) -> List[Dict]:
    """OpenAI ``{"function": {...}}`` tool defs → Converse ``{"toolSpec": {...}}``."""
    result = []
    for t in tools or []:
        fn = t.get("function", {})
        result.append({
            "toolSpec": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "inputSchema": {"json": fn.get("parameters", {"type": "object", "properties": {}})},
            }
        })
    return result


# Converse rejects empty OR whitespace-only text blocks ("text content blocks
# must contain non-whitespace text"), so the placeholder must be non-whitespace.
_EMPTY_TEXT_PLACEHOLDER = "(empty)"


def _safe_text(text) -> str:
    """Return ``text`` if it has non-whitespace content, else the placeholder (None/non-str ok)."""
    if text is None:
        return _EMPTY_TEXT_PLACEHOLDER
    if not isinstance(text, str):
        text = str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


def _image_block_from_data_url(url: str) -> Dict:
    """``data:<mime>;base64,...`` → Converse image block with RAW bytes.

    boto3 base64-encodes at the wire layer, so passing the base64 string
    through double-encodes and Bedrock rejects it ("Failed to sanitize image").
    """
    header, _, data = url.partition(",")
    media_type = "image/jpeg"
    if header.startswith("data:"):
        mime_part = header[5:].split(";")[0]
        if mime_part:
            media_type = mime_part
    try:
        raw_bytes = base64.b64decode(data)
    except Exception:
        raw_bytes = data.encode("utf-8")
    return {
        "image": {
            "format": media_type.split("/")[-1] if "/" in media_type else "jpeg",
            "source": {"bytes": raw_bytes},
        }
    }


def _convert_content_to_converse(content) -> List[Dict]:
    """OpenAI message content (str or parts list) → Converse content blocks.

    Empty/whitespace-only text becomes the placeholder; remote image URLs are
    not supported by Converse and are included as a text reference instead.
    """
    if not isinstance(content, list):
        return [{"text": _safe_text(content)}]
    blocks = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"text": _safe_text(part)})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")
        if part_type == "text":
            blocks.append({"text": _safe_text(part.get("text", ""))})
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if url.startswith("data:"):
                blocks.append(_image_block_from_data_url(url))
            else:
                blocks.append({"text": f"[Image: {url}]"})
    return blocks if blocks else [{"text": _EMPTY_TEXT_PLACEHOLDER}]


def _system_blocks(content) -> List[Dict]:
    """System content → text blocks; blank parts are dropped, not placeholder-filled."""
    if isinstance(content, str):
        return [{"text": content}] if content.strip() else []
    blocks: List[Dict] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text.strip():
                    blocks.append({"text": text})
            elif isinstance(part, str) and part.strip():
                blocks.append({"text": part})
    return blocks


def _tool_use_block(tool_use_id, name, input_dict) -> Dict:
    return {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_dict}}


def _replay_ordered_blocks(ordered_blocks: List) -> List[Dict]:
    """Rebuild the exact Bedrock block sequence captured at normalization time.

    Redacted reasoning bytes are stored base64-encoded (JSON-safe sidecar) and
    decoded here; undecodable entries are skipped.
    """
    content_blocks: List[Dict] = []
    for block in ordered_blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block and isinstance(block["text"], str):
            content_blocks.append({"text": block["text"]})
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if not isinstance(reasoning, dict):
                continue
            replay = {}
            if isinstance(reasoning.get("text"), str):
                replay["text"] = reasoning["text"]
            encoded = reasoning.get("redactedContentBase64")
            if isinstance(encoded, str) and encoded:
                try:
                    replay["redactedContent"] = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    continue
            if replay:
                content_blocks.append({"reasoningContent": replay})
        elif "toolUse" in block and isinstance(block["toolUse"], dict):
            tu = block["toolUse"]
            content_blocks.append(_tool_use_block(tu.get("toolUseId", ""), tu.get("name", ""), tu.get("input", {})))
    return content_blocks


def _assistant_blocks(msg: Dict, content) -> List[Dict]:
    """Assistant message → Converse blocks.

    An ordered ``bedrock_content_blocks`` sidecar is authoritative and replayed
    verbatim. Otherwise: redacted thinking from ``reasoning_details`` (so
    opaque encrypted reasoning round-trips byte-for-byte), then text, then
    tool calls.
    """
    ordered_blocks = msg.get("bedrock_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        content_blocks = _replay_ordered_blocks(ordered_blocks)
        if content_blocks:
            return content_blocks

    content_blocks = []
    for detail in (msg.get("reasoning_details") or []):
        if not isinstance(detail, dict) or detail.get("type") != "redacted_thinking":
            continue
        encoded = detail.get("data") or detail.get("redactedContentBase64")
        if not isinstance(encoded, str) or not encoded:
            continue
        try:
            redacted = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            continue
        content_blocks.append({"reasoningContent": {"redactedContent": redacted}})

    if isinstance(content, str) and content.strip():
        content_blocks.append({"text": content})
    elif isinstance(content, list):
        content_blocks.extend(_convert_content_to_converse(content))

    for tc in (msg.get("tool_calls", []) or []):
        fn = tc.get("function", {})
        args_str = fn.get("arguments", "{}")
        try:
            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args_dict = {}
        content_blocks.append(_tool_use_block(tc.get("id", ""), fn.get("name", ""), args_dict))
    return content_blocks


def _append_turn(converse_msgs: List[Dict], role: str, blocks: List[Dict]) -> None:
    """Append blocks as a new turn, merging into the previous turn of the same role (strict alternation)."""
    if converse_msgs and converse_msgs[-1]["role"] == role:
        converse_msgs[-1]["content"].extend(blocks)
    else:
        converse_msgs.append({"role": role, "content": blocks})


def convert_messages_to_converse(messages: List[Dict]) -> Tuple[Optional[List[Dict]], List[Dict]]:
    """OpenAI messages → ``(system_blocks_or_None, converse_messages)``.

    System messages become the system prompt; tool results become
    ``toolResult`` blocks in a user turn. Converse requires strict
    user/assistant alternation with a user turn first and last, so same-role
    neighbours are merged and placeholder user turns are inserted at the ends.
    """
    system_blocks: List[Dict] = []
    converse_msgs: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            system_blocks.extend(_system_blocks(content))
        elif role == "tool":
            result_content = content if isinstance(content, str) else json.dumps(content)
            _append_turn(converse_msgs, "user", [{
                "toolResult": {
                    "toolUseId": msg.get("tool_call_id", ""),
                    "content": [{"text": _safe_text(result_content)}],
                }
            }])
        elif role == "assistant":
            content_blocks = _assistant_blocks(msg, content) or [{"text": _EMPTY_TEXT_PLACEHOLDER}]
            _append_turn(converse_msgs, "assistant", content_blocks)
        elif role == "user":
            _append_turn(converse_msgs, "user", _convert_content_to_converse(content))

    if converse_msgs and converse_msgs[0]["role"] != "user":
        converse_msgs.insert(0, {"role": "user", "content": [{"text": _EMPTY_TEXT_PLACEHOLDER}]})
    if converse_msgs and converse_msgs[-1]["role"] != "user":
        converse_msgs.append({"role": "user", "content": [{"text": _EMPTY_TEXT_PLACEHOLDER}]})

    return (system_blocks if system_blocks else None, converse_msgs)


# --- Response format conversion: Bedrock Converse → OpenAI ---

# Bedrock stopReason → OpenAI finish_reason (unknown → "stop").
_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


def _encode_redacted(redacted) -> Optional[str]:
    """Redacted reasoning payload → base64 str (bytes encoded, str passed through, else None)."""
    if isinstance(redacted, (bytes, bytearray)):
        return base64.b64encode(bytes(redacted)).decode("ascii")
    if isinstance(redacted, str):
        return redacted
    return None


def _tool_call_ns(tool_use_id: str, name: str, input_dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_use_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(input_dict)),
    )


def _build_openai_response(
    text_parts: List[str], reasoning_parts: List[str], reasoning_details: List[Dict[str, Any]],
    tool_calls: List[SimpleNamespace], ordered_blocks: List[Dict[str, Any]],
    usage_data: Dict[str, int], stop_reason: str, model: str,
) -> SimpleNamespace:
    """Assemble the OpenAI-shaped response shared by the sync and streaming paths.

    Converse's inputTokens EXCLUDES cache read/write tokens (OpenAI's
    prompt_tokens includes them), so they are added back here and the
    Anthropic-named cache fields are surfaced for downstream normalize_usage().
    """
    msg = SimpleNamespace(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
        reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
        reasoning_details=reasoning_details or None,
        bedrock_content_blocks=ordered_blocks or None,
    )

    input_tokens = usage_data.get("inputTokens", 0)
    cache_read_tokens = usage_data.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage_data.get("cacheWriteInputTokens", 0)
    output_tokens = usage_data.get("outputTokens", 0)
    usage = SimpleNamespace(
        prompt_tokens=input_tokens + cache_read_tokens + cache_write_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + cache_read_tokens + cache_write_tokens + output_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_input_tokens=cache_write_tokens,
    )

    finish_reason = _STOP_REASON_TO_FINISH_REASON.get(stop_reason, "stop")
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)],
        usage=usage,
        model=model,
    )


def normalize_converse_response(response: Dict) -> SimpleNamespace:
    """Bedrock Converse response → OpenAI ``ChatCompletion``-shaped SimpleNamespace.

    Exposes ``.choices[0].message.{content,tool_calls,reasoning_content,
    reasoning_details,bedrock_content_blocks}``, ``.choices[0].finish_reason``
    and ``.usage``.
    """
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])

    text_parts = []
    reasoning_parts = []
    reasoning_details = []
    ordered_blocks = []
    tool_calls = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
            ordered_blocks.append({"text": block["text"]})
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if not isinstance(reasoning, dict):
                continue
            thinking_text = reasoning.get("text", "")
            encoded = _encode_redacted(reasoning.get("redactedContent"))
            ordered_reasoning = {}
            if thinking_text:
                reasoning_parts.append(str(thinking_text))
                ordered_reasoning["text"] = str(thinking_text)
            if encoded:
                reasoning_details.append({"type": "redacted_thinking", "data": encoded})
                ordered_reasoning["redactedContentBase64"] = encoded
            if ordered_reasoning:
                ordered_blocks.append({"reasoningContent": ordered_reasoning})
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_use_id, name, tool_input = tu.get("toolUseId", ""), tu.get("name", ""), tu.get("input", {})
            ordered_blocks.append(_tool_use_block(tool_use_id, name, tool_input))
            tool_calls.append(_tool_call_ns(tool_use_id, name, tool_input))

    return _build_openai_response(
        text_parts, reasoning_parts, reasoning_details, tool_calls, ordered_blocks,
        response.get("usage", {}), response.get("stopReason", "end_turn"), response.get("modelId", ""),
    )


# --- Streaming response conversion ---

def normalize_converse_stream_events(event_stream) -> SimpleNamespace:
    """Consume a ConverseStream event stream (no callbacks) → same shape as ``normalize_converse_response()``."""
    return stream_converse_with_callbacks(event_stream)


def stream_converse_with_callbacks(
    event_stream,
    on_text_delta=None,
    on_tool_start=None,
    on_reasoning_delta=None,
    on_interrupt_check=None,
    on_event=None,
) -> SimpleNamespace:
    """Process a boto3 ``converse_stream()`` response with real-time callbacks.

    ``on_text_delta`` only fires while no toolUse block has been seen (same
    semantics as the Anthropic/chat_completions paths). ``on_interrupt_check``
    runs per event; True stops streaming. ``on_event`` is a wire-level liveness
    signal fired for EVERY event before any branching (so a watchdog can tell
    "still receiving" from "wedged"); its exceptions are swallowed so it can
    never abort the stream. Returns the ``normalize_converse_response()`` shape.
    """
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    reasoning_details: List[Dict[str, Any]] = []
    tool_calls: List[SimpleNamespace] = []
    stream_blocks: Dict[int, Dict[str, Any]] = {}
    current_block_index: Optional[int] = None
    current_tool: Optional[Dict] = None
    current_text_buffer: List[str] = []
    has_tool_use = False
    stop_reason = "end_turn"
    usage_data: Dict[str, int] = {}

    def _current_block(default: Dict[str, Any]) -> Dict[str, Any]:
        idx = current_block_index if current_block_index is not None else len(stream_blocks)
        return stream_blocks.setdefault(idx, default)

    def _flush_text() -> None:
        nonlocal current_text_buffer
        if current_text_buffer:
            text_parts.append("".join(current_text_buffer))
            current_text_buffer = []

    for event in event_stream.get("stream", []):
        if on_event is not None:
            try:
                on_event()
            except Exception:
                pass
        if on_interrupt_check and on_interrupt_check():
            break

        if "contentBlockStart" in event:
            start_event = event["contentBlockStart"]
            current_block_index = start_event.get("contentBlockIndex", len(stream_blocks))
            start = start_event.get("start", {})
            if "toolUse" in start:
                has_tool_use = True
                _flush_text()
                current_tool = {
                    "toolUseId": start["toolUse"].get("toolUseId", ""),
                    "name": start["toolUse"].get("name", ""),
                    "input_json": "",
                }
                stream_blocks[current_block_index] = _tool_use_block(current_tool["toolUseId"], current_tool["name"], {})
                if on_tool_start:
                    on_tool_start(current_tool["name"])

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                text = delta["text"]
                block = _current_block({"text": ""})
                block["text"] = block.get("text", "") + text
                current_text_buffer.append(text)
                if on_text_delta and not has_tool_use:
                    on_text_delta(text)
            elif "toolUse" in delta:
                if current_tool is not None:
                    current_tool["input_json"] += delta["toolUse"].get("input", "")
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if isinstance(reasoning, dict):
                    thinking_text = reasoning.get("text", "")
                    if thinking_text:
                        reasoning_parts.append(str(thinking_text))
                        if on_reasoning_delta:
                            on_reasoning_delta(thinking_text)
                        block = _current_block({"reasoningContent": {}})
                        block.setdefault("reasoningContent", {})["text"] = block["reasoningContent"].get("text", "") + str(thinking_text)
                    encoded = _encode_redacted(reasoning.get("redactedContent"))
                    if encoded:
                        reasoning_details.append({"type": "redacted_thinking", "data": encoded})
                        block = _current_block({"reasoningContent": {}})
                        block.setdefault("reasoningContent", {})["redactedContentBase64"] = encoded

        elif "contentBlockStop" in event:
            if current_tool is not None:
                try:
                    input_dict = json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                except (json.JSONDecodeError, TypeError):
                    input_dict = {}
                tool_calls.append(_tool_call_ns(current_tool["toolUseId"], current_tool["name"], input_dict))
                if current_block_index is not None and current_block_index in stream_blocks:
                    stream_blocks[current_block_index]["toolUse"]["input"] = input_dict
                current_tool = None
            else:
                _flush_text()

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")

        elif "metadata" in event:
            meta_usage = event["metadata"].get("usage", {})
            usage_data = {
                "inputTokens": meta_usage.get("inputTokens", 0),
                "outputTokens": meta_usage.get("outputTokens", 0),
                "cacheReadInputTokens": meta_usage.get("cacheReadInputTokens", 0),
                "cacheWriteInputTokens": meta_usage.get("cacheWriteInputTokens", 0),
            }

    _flush_text()

    return _build_openai_response(
        text_parts, reasoning_parts, reasoning_details, tool_calls,
        [stream_blocks[i] for i in sorted(stream_blocks)], usage_data, stop_reason, "",
    )


# --- High-level API: call Bedrock Converse ---

def _cache_point() -> Dict[str, Any]:
    return {"cachePoint": {"type": "default"}}


def build_converse_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    guardrail_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``bedrock-runtime.converse()`` / ``converse_stream()``.

    ``max_tokens=None`` omits ``inferenceConfig.maxTokens`` so Bedrock uses the
    model's maximum output (callers wanting the full budget, e.g. uncapped
    auxiliary vision calls, pass None explicitly; default stays 4096).
    cachePoint markers go on system, tools and the second-newest message
    (so the marker survives as only the tail grows — mirrors the Anthropic
    system_and_3 strategy) — each only if the model supports caching and
    Bedrock has not rejected that placement.
    """
    system_prompt, converse_messages = convert_messages_to_converse(messages)
    cache_enabled = _model_supports_prompt_cache(model)

    inference_config: Dict[str, Any] = {}
    if max_tokens is not None:
        inference_config["maxTokens"] = max_tokens

    kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": converse_messages,
        "inferenceConfig": inference_config,
    }

    if system_prompt:
        if cache_enabled and cache_point_allowed(model, "system"):
            system_prompt = system_prompt + [_cache_point()]
        kwargs["system"] = system_prompt

    from agent.anthropic_adapter import _forbids_sampling_params

    if not _forbids_sampling_params(model):
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p

    if stop_sequences:
        inference_config["stopSequences"] = stop_sequences

    if tools:
        converse_tools = convert_tools_to_converse(tools)
        if converse_tools:
            # Non-tool-calling models (e.g. DeepSeek R1) reject toolConfig with a
            # ValidationException → retry loop → failure. Strip tools and warn.
            if _model_supports_tool_use(model):
                if cache_enabled and cache_point_allowed(model, "tools"):
                    converse_tools = converse_tools + [_cache_point()]
                kwargs["toolConfig"] = {"tools": converse_tools}
            else:
                logger.warning(
                    "Model %s does not support tool calling — tools stripped. "
                    "The agent will operate in text-only mode.", model
                )

    if (
        cache_enabled
        and cache_point_allowed(model, "messages")
        and len(converse_messages) >= 2
    ):
        content = converse_messages[-2].get("content")
        if isinstance(content, list) and content:
            content.append(_cache_point())

    if guardrail_config:
        kwargs["guardrailConfig"] = guardrail_config

    if not inference_config:
        del kwargs["inferenceConfig"]  # optional on the wire; don't send {}

    return kwargs


def call_converse(
    region: str,
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    guardrail_config: Optional[Dict] = None,
) -> SimpleNamespace:
    """Non-streaming Converse call → OpenAI-compatible response.

    Retries once without the rejected cachePoint placement; evicts the cached
    client on stale-connection errors before re-raising.
    """
    client = _get_bedrock_runtime_client(region)
    kwargs = build_converse_kwargs(
        model, messages, tools, max_tokens, temperature, top_p, stop_sequences, guardrail_config,
    )

    try:
        response = client.converse(**kwargs)
    except Exception as exc:
        retry_kwargs = recover_from_cache_point_rejection(exc, kwargs)
        if retry_kwargs is not None:
            return normalize_converse_response(client.converse(**retry_kwargs))
        if is_stale_connection_error(exc):
            logger.warning(
                "bedrock: stale-connection error on converse(region=%s, model=%s): "
                "%s — evicting cached client so the next call reconnects.",
                region, model, type(exc).__name__,
            )
            invalidate_runtime_client(region)
        raise
    return normalize_converse_response(response)


# --- Model discovery ---

_discovery_cache: Dict[str, Any] = {}
_DISCOVERY_CACHE_TTL_SECONDS = 3600


def reset_discovery_cache():
    """Clear the model discovery cache. Used in tests."""
    _discovery_cache.clear()


def _list_foundation_models(client, filter_set: set, models: List[Dict[str, Any]]) -> None:
    """Append active, streaming-capable, text-output foundation models (optionally provider-filtered)."""
    for summary in client.list_foundation_models().get("modelSummaries", []):
        model_id = (summary.get("modelId") or "").strip()
        if not model_id:
            continue
        if filter_set:
            provider_name = (summary.get("providerName") or "").lower()
            model_prefix = model_id.split(".")[0].lower() if "." in model_id else ""
            if provider_name not in filter_set and model_prefix not in filter_set:
                continue
        if summary.get("modelLifecycle", {}).get("status", "").upper() != "ACTIVE":
            continue
        if not summary.get("responseStreamingSupported", False):
            continue
        output_mods = summary.get("outputModalities", [])
        if "TEXT" not in output_mods:
            continue
        models.append({
            "id": model_id,
            "name": (summary.get("modelName") or model_id).strip(),
            "provider": (summary.get("providerName") or "").strip(),
            "input_modalities": summary.get("inputModalities", []),
            "output_modalities": output_mods,
            "streaming": True,
        })


def _list_inference_profiles(client, filter_set: set, models: List[Dict[str, Any]]) -> None:
    """Append active cross-region inference profiles whose IDs are not already present (paginated)."""
    profiles = []
    next_token = None
    while True:
        kwargs = {"nextToken": next_token} if next_token else {}
        response = client.list_inference_profiles(**kwargs)
        profiles.extend(response.get("inferenceProfileSummaries", []))
        next_token = response.get("nextToken")
        if not next_token:
            break

    seen_ids = {m["id"].lower() for m in models}
    for profile in profiles:
        profile_id = (profile.get("inferenceProfileId") or "").strip()
        if not profile_id or profile.get("status") != "ACTIVE" or profile_id.lower() in seen_ids:
            continue
        if filter_set and not any(
            _extract_provider_from_arn(m.get("modelArn", "")).lower() in filter_set
            for m in profile.get("models", [])
        ):
            continue
        models.append({
            "id": profile_id,
            "name": (profile.get("inferenceProfileName") or profile_id).strip(),
            "provider": "inference-profile",
            "input_modalities": ["TEXT"],
            "output_modalities": ["TEXT"],
            "streaming": True,
        })
        seen_ids.add(profile_id.lower())


def discover_bedrock_models(region: str, provider_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Discover Bedrock foundation models + inference profiles (cached 1h per region/filter).

    Each entry has ``id``, ``name``, ``provider`` ("inference-profile" for
    profiles), ``input_modalities``, ``output_modalities``, ``streaming``.
    Sorted with ``global.`` cross-region profiles first, then by name.
    Returns [] when the client cannot be built.
    """
    import time

    cache_key = f"{region}:{','.join(sorted(provider_filter or []))}"
    cached = _discovery_cache.get(cache_key)
    if cached and (time.time() - cached["timestamp"]) < _DISCOVERY_CACHE_TTL_SECONDS:
        return cached["models"]

    try:
        client = _get_bedrock_control_client(region)
    except Exception as e:
        logger.warning("Failed to create Bedrock client for model discovery: %s", e)
        return []

    models = []
    filter_set = {f.lower() for f in (provider_filter or [])}

    try:
        _list_foundation_models(client, filter_set, models)
    except Exception as e:
        logger.warning("Failed to list Bedrock foundation models: %s", e)

    try:
        _list_inference_profiles(client, filter_set, models)
    except Exception as e:
        logger.debug("Skipping inference profile discovery: %s", e)

    models.sort(key=lambda m: (
        0 if m["id"].startswith("global.") else 1,
        m["name"].lower(),
    ))

    _discovery_cache[cache_key] = {"timestamp": time.time(), "models": models}
    return models


def _extract_provider_from_arn(arn: str) -> str:
    """``arn:aws:bedrock:...:foundation-model/anthropic.claude-v2`` → ``"anthropic"``."""
    match = re.search(r"foundation-model/([^.]+)", arn)
    return match.group(1) if match else ""


# --- Bedrock model context lengths ---
# Static fallback table for when the live probe is unavailable; used by
# agent/model_metadata.py. Keys are matched by longest substring, so versioned
# entries (opus-4-6/4-7/4-8) win over the generic "anthropic.claude-opus-4".

BEDROCK_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude (per Anthropic's models overview). Fable / Sonnet 5 /
    # Opus 4.8-4.6 / Sonnet 4.6 are 1M GA; Sonnet 4.5 / Sonnet 4 lost their 1M
    # beta and are 200K; Haiku 4.5 is 200K. The 1M entries must match
    # agent/model_metadata.py DEFAULT_CONTEXT_LENGTHS or context compresses early.
    "anthropic.claude-fable-5":      1_000_000,
    "anthropic.claude-fable":        1_000_000,
    "anthropic.claude-sonnet-5":     1_000_000,
    "anthropic.claude-opus-4-8":     1_000_000,
    "anthropic.claude-opus-4-7":     1_000_000,
    "anthropic.claude-opus-4-6":     1_000_000,
    "anthropic.claude-sonnet-4-6":   1_000_000,
    "anthropic.claude-sonnet-4-5":   200_000,
    "anthropic.claude-haiku-4-5":    200_000,
    "anthropic.claude-opus-4":       200_000,
    "anthropic.claude-sonnet-4":     200_000,
    "anthropic.claude-3-5-sonnet":   200_000,
    "anthropic.claude-3-5-haiku":    200_000,
    "anthropic.claude-3-opus":       200_000,
    "anthropic.claude-3-sonnet":     200_000,
    "anthropic.claude-3-haiku":      200_000,
    # Amazon Nova
    "amazon.nova-pro":               300_000,
    "amazon.nova-lite":              300_000,
    "amazon.nova-micro":             128_000,
    # Meta Llama
    "meta.llama4-maverick":          128_000,
    "meta.llama4-scout":             128_000,
    "meta.llama3-3-70b-instruct":    128_000,
    # Mistral
    "mistral.mistral-large":         128_000,
    # DeepSeek
    "deepseek.v3":                   128_000,
    # OpenAI on Bedrock (Mantle/Responses route)
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
    "openai.gpt-5.5":                272_000,
    "openai.gpt-5.6-sol":            272_000,
    "openai.gpt-5.6-terra":          272_000,
    "openai.gpt-5.6-luna":           272_000,
}

# Default for unknown Bedrock models
BEDROCK_DEFAULT_CONTEXT_LENGTH = 128_000

# Probe padding targets (tokens). Tiered rather than one giant request because
# a wildly oversized payload (5M tokens) yields an opaque InternalServerException
# instead of a clean ValidationException, and stepping up discovers 2M+
# windows without over-padding smaller ones. The error reports the true maximum.
_BEDROCK_PROBE_TIERS = (1_300_000, 2_200_000)
_WORDS_PER_TOKEN = 0.9  # conservative: ensures the padded prompt clears the tier


def _static_bedrock_context_length(model_id: str) -> int:
    """Longest-substring match against the static table (versioned IDs resolve correctly)."""
    model_lower = model_id.lower()
    matches = [key for key in BEDROCK_CONTEXT_LENGTHS if key in model_lower]
    return BEDROCK_CONTEXT_LENGTHS[max(matches, key=len)] if matches else BEDROCK_DEFAULT_CONTEXT_LENGTH


def probe_bedrock_context_length(model_id: str, region: str) -> Optional[int]:
    """Discover a model's real context window by provoking a length error.

    No Bedrock metadata API exposes the window; the only authoritative source
    is the ValidationException for an oversized prompt ("prompt is too long:
    1300032 tokens > 1000000 maximum"). Length validation happens before
    inference, so the probe is rejected immediately and costs nothing. If a
    tier is accepted, that tier is returned as a safe lower bound.

    Returns None if the probe could not run (no credentials, network error,
    no parseable limit) so the caller falls back to the static table.
    """
    try:
        from agent.model_metadata import parse_context_limit_from_error
    except ImportError:  # pragma: no cover — same package
        return None

    try:
        client = _get_bedrock_runtime_client(region)
    except Exception as exc:  # boto3 missing / credential resolution failure
        logger.debug("Bedrock context probe skipped for %s: %s", model_id, exc)
        return None

    last_error = ""
    for tier_tokens in _BEDROCK_PROBE_TIERS:
        oversized = "data " * int(tier_tokens / _WORDS_PER_TOKEN)
        try:
            client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": oversized}]}],
                inferenceConfig={"maxTokens": 8},
            )
            logger.debug(
                "Bedrock context probe for %s accepted ~%s-token prompt; "
                "window is at least that", model_id, f"{tier_tokens:,}",
            )
            return tier_tokens
        except Exception as exc:
            last_error = str(exc)
            limit = parse_context_limit_from_error(last_error)
            if limit and limit >= 1024:
                logger.info(
                    "Probed Bedrock context window for %s: %s tokens",
                    model_id, f"{limit:,}",
                )
                return limit
            # Opaque server error / auth / throttle at this tier — try the next.

    logger.debug(
        "Bedrock context probe for %s returned no parseable limit: %s",
        model_id, last_error[:200],
    )
    return None


def get_bedrock_context_length(model_id: str, region: str = "", probe: bool = True) -> int:
    """Context window: live probe (if ``probe`` and ``region``) → static table → default.

    The static table is a fallback only: AWS ships new versions faster than
    the table tracks, and a stale substring match silently caps the window
    (e.g. a 1M Opus pinned to 200K via "opus-4"). ``probe=False`` or an empty
    region skips the network call for offline/display paths.
    """
    if probe and region:
        probed = probe_bedrock_context_length(model_id, region)
        if probed:
            return probed
    return _static_bedrock_context_length(model_id)
