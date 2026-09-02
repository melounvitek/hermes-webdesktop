"""Native OpenAI Responses server-side compaction — gpt-5.6 on direct OpenAI routes only.

Including ``context_management=[{"type": "compaction", "compact_threshold": N}]``
in a ``/v1/responses`` request makes the server summarize older context into an
opaque ``compaction`` item (``encrypted_content``, sealed to the issuing
endpoint) once the input crosses N tokens; replaying that item stands in for
the pruned history. Docs: https://developers.openai.com/api/docs/guides/compaction

Support is deliberately narrow (live-verified):
* gpt-5.6 family only — gpt-5.1/5.2 fail server-side (HTTP 500 blocking, a
  permanent stall streaming) with no structured "unsupported" rejection, so an
  explicit model-family check is the only safe gate.
* Direct OpenAI routes only (api.openai.com or the ChatGPT Codex backend) —
  other Responses surfaces would 400 on the field and cannot mint/decrypt the blob.

Hermes' local compressor stays armed as fallback owner: the native threshold is
clamped below the local trigger so the server compacts first, and captured
compaction items ride the existing ``codex_reasoning_items`` sidecar (persistence,
replay, cross-issuer stamping, kill switch). This module stays free of
transport/adapter imports so transport, adapter, and loop share the gate
without cycles; ``context_compressor`` and ``message_content`` sit below it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.context_compressor import is_compaction_summary_message
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# Native compaction fires this many tokens below the local compressor's
# trigger so the server always gets the first shot.
LOCAL_TRIGGER_SAFETY_MARGIN = 8_192
# Fallback when automatic mode has no local trigger to follow.
DEFAULT_COMPACT_THRESHOLD = 200_000
# Substring match so dated snapshots and variants (gpt-5.6-mini) stay eligible.
_ELIGIBLE_MODEL_MARKER = "gpt-5.6"


def is_native_compaction_model(model: Optional[str]) -> bool:
    """True when the model is in the gpt-5.6 family."""
    return _ELIGIBLE_MODEL_MARKER in (model or "").lower()


def resolve_native_compaction_capabilities(
    *,
    model: Optional[str],
    base_url: Optional[str],
    provider: Optional[str] = None,
    is_codex_backend: bool = False,
) -> Dict[str, bool]:
    """Resolve the native-compaction capability for a runtime destination.

    A resolved ``False`` is distinct from "unresolved" and must survive model
    switches unchanged.
    """
    direct_default = (provider or "").strip().lower() == "openai" and not base_url
    eligible = is_native_compaction_model(model) and (
        direct_default
        or is_direct_openai_route(base_url, is_codex_backend=is_codex_backend)
    )
    return {"native_compaction": eligible}


def is_direct_openai_route(
    base_url: Optional[str],
    *,
    is_codex_backend: bool = False,
) -> bool:
    """True for api.openai.com or the ChatGPT Codex backend — nothing else."""
    if is_codex_backend:
        return True
    try:
        hostname = (urlsplit(base_url or "").hostname or "").lower()
    except ValueError:
        return False
    return hostname == "api.openai.com"


def resolve_compact_threshold(
    configured_threshold: Any,
    local_trigger_tokens: Any = None,
) -> int:
    """Resolve automatic mode or clamp an explicit native threshold.

    An omitted/invalid setting follows the local compressor trigger
    (``ContextCompressor.threshold_tokens``) minus the safety margin. An
    explicit positive integer is absolute unless it must be clamped so native
    compaction fires first. Booleans are never thresholds.
    """
    local = None
    try:
        if local_trigger_tokens is not None and not isinstance(local_trigger_tokens, bool):
            local = int(local_trigger_tokens)
    except (TypeError, ValueError):
        local = None
    if local is not None and local <= 0:
        local = None

    upper = None
    if local is not None:
        if local > LOCAL_TRIGGER_SAFETY_MARGIN:
            upper = max(1_024, local - LOCAL_TRIGGER_SAFETY_MARGIN)
        else:
            upper = max(1_024, int(local * 0.8))

    try:
        configured = (
            None
            if isinstance(configured_threshold, (bool, float))
            else int(configured_threshold)
        )
    except (TypeError, ValueError):
        configured = None
    if configured is None or configured <= 0:
        return upper if upper is not None else DEFAULT_COMPACT_THRESHOLD
    if upper is None:
        return configured
    return max(1_024, min(configured, upper))


_checkpoint_suppression_logged = False


def _warn_native_compaction_suppressed_by_checkpoint_gate() -> None:
    """Log once per process; the suppression itself is re-evaluated per request."""
    global _checkpoint_suppression_logged
    if _checkpoint_suppression_logged:
        return
    _checkpoint_suppression_logged = True
    logger.warning(
        "compression.checkpoint_required is enabled: server-side native "
        "compaction (context_management) is disabled for this agent so the "
        "checkpoint-aware Hermes compressor stays authoritative."
    )


def native_compaction_context_management(
    agent: Any,
    *,
    is_codex_backend: bool,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Return the ``context_management`` payload for this request, or None.

    None means "do not send the field" (request byte-identical to pre-feature).
    Every gate is re-checked per request so a mid-session model switch or the
    in-session kill switch (``agent.codex_responses_native_compaction = False``,
    set by rejection recovery) takes effect on the next call.
    """
    capabilities = getattr(agent, "runtime_capabilities", None)
    if isinstance(capabilities, dict) and not capabilities.get("native_compaction", False):
        return None
    if not getattr(agent, "codex_responses_native_compaction", False):
        return None
    # compression.enabled: false disables ALL automatic compaction, native included.
    if not getattr(agent, "compression_enabled", True):
        return None
    # Server-side compaction is a lossy boundary the provider owns — no
    # pre-compress checkpoint can run first — so the checkpoint-aware Hermes
    # compressor stays authoritative. Explicit-True matches compress_context().
    if getattr(agent, "compression_checkpoint_required", False) is True:
        _warn_native_compaction_suppressed_by_checkpoint_gate()
        return None
    if is_xai_responses or is_github_responses:
        return None
    if not is_native_compaction_model(getattr(agent, "model", None)):
        return None
    trusted_proxy = bool(
        getattr(agent, "capabilities", {}).get("openai_native_compaction", False)
    )
    if not trusted_proxy and not is_direct_openai_route(
        getattr(agent, "base_url", None), is_codex_backend=is_codex_backend
    ):
        return None

    compressor = getattr(agent, "context_compressor", None)
    threshold = resolve_compact_threshold(
        getattr(agent, "codex_responses_compact_threshold", None),
        getattr(compressor, "threshold_tokens", None) if compressor is not None else None,
    )
    return [{"type": "compaction", "compact_threshold": threshold}]


# Retention budgets for plaintext user messages / local compression summaries
# carried across a native compaction boundary (mirrors Codex CLI's
# RETAINED_MESSAGE_TOKEN_BUDGET; the summary budget prevents summary inflation).
RETAINED_USER_MESSAGE_TOKEN_BUDGET = 64_000
RETAINED_SUMMARY_TOKEN_BUDGET = 32_000


def _approx_tokens(text: str) -> int:
    """Cheap chars//4 token estimate — same shape Codex uses for retention."""
    return max(1, len(text) // 4)


def _extract_item_text(item: Any) -> Optional[str]:
    """Measurable text from a Responses item (string/multipart/metadata), or None."""
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if content is None and "output_text" in item:
        content = item.get("output_text")

    if isinstance(content, str):
        return content if content.strip() else None

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part.strip())
            elif isinstance(part, dict):
                part_text = part.get("text") or part.get("input_text") or part.get("output_text")
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
                part_meta = part.get("metadata")
                if isinstance(part_meta, dict) and isinstance(part_meta.get("text"), str) and part_meta["text"].strip():
                    parts.append(part_meta["text"].strip())
        text = " ".join(parts)
        return text if text.strip() else None

    return None


def _has_retainable_image_content(item: Any) -> bool:
    """True for a converted Responses message with a valid ``input_image`` part.

    Only the adapter-owned ``input_image`` shape counts: unknown or empty
    multipart placeholders must not become durable history for being non-empty.
    """
    if not isinstance(item, dict):
        return False
    content = item.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if str(part.get("type") or "").strip().lower() != "input_image":
            continue
        image_url = part.get("image_url")
        if isinstance(image_url, str) and image_url.strip():
            return True
    return False


# Canonical provenance check (metadata marker, then canonical prefix classifier).
# Deliberately NOT a second heuristic: no underscore-key scan, no matching on
# ad-hoc headings — either could promote ordinary or adversarial content to
# durable retained history.
_is_summary_item = is_compaction_summary_message


def prune_pre_checkpoint_items(
    items: List[Dict[str, Any]],
    retained_user_token_budget: int = RETAINED_USER_MESSAGE_TOKEN_BUDGET,
    retained_summary_token_budget: int = RETAINED_SUMMARY_TOKEN_BUDGET,
    enable_summary_retention: bool = True,
    item_sources: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Restructure Responses input around the newest compaction checkpoint.

    The server drops every input item preceding a replayed ``compaction`` item,
    which silently erases the user's plaintext asks and any local-compression
    summary (``role="assistant"``). With a checkpoint present, rebuild as::

        [checkpoint run] + [retained user & summary messages (newest-first budget)] + [post]

    - The NEWEST contiguous run of checkpoints wins.
    - User messages are kept verbatim within ``retained_user_token_budget``;
      the boundary message is head-truncated when it only partially fits
      (string content only — goals are stated up front). A recognized
      image-only user message is retained whole at one-token cost.
    - Summaries are retained whole within ``retained_summary_token_budget`` and
      never sliced (their structural framing would corrupt); one that doesn't
      fit is dropped. Identical summary text is never retained twice.
    - Relative order between user messages and summaries is preserved.
    - ``item_sources`` (parallel to ``items``) is the raw chat message each item
      was converted from. Conversion can be lossy for summaries (a
      merge-into-tail carrier becomes a typed ``function_call_output``, or an
      assistant carrier is shadowed by a stale exact replay), so when a source
      is itself a canonical summary carrier its content is read from the
      SOURCE and retained as a synthesized ``role="assistant"`` message.
    - ``enable_summary_retention`` is a function-level override for tests, not
      a config surface.
    """
    if not isinstance(items, list) or not items:
        return items

    last_cp = None
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("type") == "compaction":
            last_cp = i
    if last_cp is None:
        return items

    first_cp = last_cp
    while (
        first_cp > 0
        and isinstance(items[first_cp - 1], dict)
        and items[first_cp - 1].get("type") == "compaction"
    ):
        first_cp -= 1

    pre = items[:first_cp]
    checkpoint_run = items[first_cp : last_cp + 1]
    post = items[last_cp + 1 :]

    if isinstance(item_sources, list) and len(item_sources) == len(items):
        pre_sources: List[Any] = item_sources[:first_cp]
    else:
        pre_sources = [None] * len(pre)

    retained_reversed: List[Dict[str, Any]] = []
    user_remaining = max(0, int(retained_user_token_budget))
    summary_remaining = max(0, int(retained_summary_token_budget))
    seen_summary_texts: set = set()

    def _retain_summary(text: Optional[str], retained_item: Dict[str, Any]) -> None:
        """Retain a summary whole when it fits the budget and is not a duplicate."""
        nonlocal summary_remaining
        if not text or summary_remaining <= 0 or text in seen_summary_texts:
            return
        cost = _approx_tokens(text)
        if cost > summary_remaining:
            return  # never slice a summary's structural framing
        seen_summary_texts.add(text)
        retained_reversed.append(retained_item)
        summary_remaining -= cost

    for item, source in zip(reversed(pre), reversed(pre_sources)):
        if not isinstance(item, dict):
            continue

        # Source-based detection sees past a lossy conversion; it only fires
        # when the source itself is a provenance-tagged summary carrier.
        if enable_summary_retention and isinstance(source, dict) and _is_summary_item(source):
            text = flatten_message_text(source.get("content"))
            _src_role = source.get("role")
            _retain_summary(text if text.strip() else None, {
                "role": _src_role if _src_role in ("user", "assistant") else "assistant",
                "content": text,
            })
            continue

        # Typed non-message items never carry role=user or a summary flag.
        if "type" in item and item.get("type") != "message":
            continue

        is_summary = enable_summary_retention and _is_summary_item(item)
        is_user = item.get("role") == "user"

        if not is_user and not is_summary:
            continue

        text = _extract_item_text(item)
        has_retainable_image = is_user and _has_retainable_image_content(item)
        if text is None and not has_retainable_image:
            continue
        if text is None:
            text = ""

        if is_summary:
            _retain_summary(text, item)
        elif is_user:
            if user_remaining <= 0:
                continue
            cost = _approx_tokens(text)
            if cost <= user_remaining:
                retained_reversed.append(item)
                user_remaining -= cost
            elif isinstance(item.get("content"), str):
                truncated = dict(item)
                truncated["content"] = item["content"][: user_remaining * 4]
                if truncated["content"].strip():
                    retained_reversed.append(truncated)
                user_remaining = 0

    result = checkpoint_run + list(reversed(retained_reversed)) + post

    logger.debug(
        "Pruned pre-checkpoint items: %d input -> %d retained (user_rem=%d, summary_rem=%d)",
        len(items), len(result), user_remaining, summary_remaining,
    )
    return result


_REJECTION_MARKERS = (
    "unknown", "unsupported", "invalid", "unexpected", "not permitted",
    "not allowed", "unrecognized", "extra field", "no such", "bad request",
    "not supported",
)


def is_native_compaction_rejection(error: Any, status_code: Any = None) -> bool:
    """True when a provider error is a STRUCTURED rejection of ``context_management``.

    Drives the loop's one-shot recovery (strip the field, disable for the
    session, retry), so matching is narrow: a transient 5xx whose body merely
    ECHOES the request must not permanently downgrade native compaction. Requires
    ``status_code`` 400 (or unknown — some transports surface only a message)
    AND the field name alongside rejection language.
    """
    text = str(error or "").lower()
    if "context_management" not in text and "compact_threshold" not in text:
        return False
    if status_code is not None:
        try:
            if int(status_code) != 400:
                return False
        except (TypeError, ValueError):
            pass
    return any(marker in text for marker in _REJECTION_MARKERS)


def has_compaction_checkpoint(items: Any) -> bool:
    """Does this ``codex_reasoning_items`` sidecar carry a compaction checkpoint?

    A ``type: "compaction"`` item is cumulative context, not per-turn
    reasoning, and exists in exactly one place: anything that rewrites or
    discards the sidecar must ask this first or lose the compacted history.
    """
    return any(
        isinstance(item, dict) and item.get("type") == "compaction"
        for item in (items if isinstance(items, list) else ())
    )


def merge_interim_reasoning_items(
    prior_items: Any,
    new_items: Any,
) -> List[Dict[str, Any]]:
    """Merge ``codex_reasoning_items`` across Codex incomplete-continuation dedup.

    A checkpoint captured on the EARLIER response is not re-emitted by the
    continuation, so a blind overwrite drops the only copy. Rule: newer items
    win, but prior checkpoints are prepended unless the newer payload has its own.
    """
    kept_checkpoints = [
        item
        for item in (prior_items if isinstance(prior_items, list) else [])
        if isinstance(item, dict) and item.get("type") == "compaction"
    ]
    new_list = list(new_items) if isinstance(new_items, list) else []
    if has_compaction_checkpoint(new_list) or not kept_checkpoints:
        return new_list
    return kept_checkpoints + new_list
