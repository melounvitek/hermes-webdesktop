"""Per-attempt recovery bookkeeping (``TurnRetryState``) for the conversation turn loop.

Each one-shot recovery branch of the inner retry loop is guarded by a flag here so it
fires at most once per attempt. Loop-control (``retry_count``, ``max_retries``) stays
as plain locals. Dependency-free so it imports without a cycle."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class TurnRetryState:
    """One-shot recovery guards + restart signals for a single API-call attempt.

    A fresh instance is created per ``api_call_count`` iteration; each guard fires at
    most once, and ``restart_with_*`` signals tell the loop to rebuild and retry."""

    # ── Per-provider OAuth / credential refresh guards ───────────────────
    codex_auth_retry_attempted: bool = False
    anthropic_auth_retry_attempted: bool = False
    nous_auth_retry_attempted: bool = False
    nous_paid_entitlement_refresh_attempted: bool = False
    copilot_auth_retry_attempted: bool = False
    # Copilot surfaces a stale/degraded credential as a 400
    # ``model_not_available_for_integrator`` / ``model_not_supported``, not a 401.
    # Single-shot forced re-exchange + rebuild, separate from the 401 guard.
    copilot_stale_cred_retry_attempted: bool = False
    vertex_auth_retry_attempted: bool = False

    # ── Format / payload recovery guards ─────────────────────────────────
    thinking_sig_retry_attempted: bool = False
    invalid_encrypted_content_retry_attempted: bool = False
    native_compaction_reject_retry_attempted: bool = False
    image_shrink_retry_attempted: bool = False
    multimodal_tool_content_retry_attempted: bool = False
    oauth_1m_beta_retry_attempted: bool = False
    llama_cpp_grammar_retry_attempted: bool = False

    # ── Transport / rate-limit recovery ──────────────────────────────────
    primary_recovery_attempted: bool = False
    has_retried_429: bool = False

    # ── Auth-failure provider failover ───────────────────────────────────
    # Set once a persistent 401/403 has been escalated to the fallback chain, so
    # we don't loop on the same auth failover within one attempt.
    auth_failover_attempted: bool = False

    # ── Restart signals (read by the outer loop after the attempt) ───────
    restart_with_compressed_messages: bool = False
    restart_with_length_continuation: bool = False
    # Set when a content-filter stream stall (e.g. MiniMax "new_sensitive") was
    # escalated to the fallback chain: partial content was rolled back off
    # ``messages``; re-issue the call against the new provider (#32421).
    restart_with_rebuilt_messages: bool = False
    # A user correction cancelled the in-flight request: append a role-safe checkpoint +
    # user message, rebuild the payload, and retry the same logical iteration.
    restart_with_redirected_messages: bool = False

    def __iter__(self):
        # Convenience for debugging / tests: iterate (name, value) pairs.
        for f in fields(self):
            yield f.name, getattr(self, f.name)
