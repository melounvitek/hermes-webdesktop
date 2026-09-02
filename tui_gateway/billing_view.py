"""Billing / usage / subscription serializers for the TUI RPC surface.

STRUCTURED envelopes (result.ok / result.error) rather than JSON-RPC errors, so
rpc() always resolves and the client branches on the typed billing code.
Data-building lives in agent/billing_view.py + hermes_cli/nous_billing.py.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so tests may still monkeypatch e.g.
``server._usage_payload``.
"""

from __future__ import annotations

from typing import Optional

from .method_ctx import bind_module


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRemoteSpendingRevoked, BillingScopeRequired, BillingSessionRevoked, BillingTransient,
    )
    kind = "error"
    if isinstance(exc, BillingRemoteSpendingRevoked):
        kind = "remote_spending_revoked"
    elif isinstance(exc, BillingSessionRevoked):
        kind = "session_revoked"
    elif isinstance(exc, BillingScopeRequired):
        kind = "insufficient_scope"
    elif isinstance(exc, BillingTransient):
        kind = str(exc.error) if getattr(exc, "error", None) else "rate_limited"
    elif getattr(exc, "error", None):
        kind = str(exc.error)
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
        # Remote-Spending contract extras (threaded so the TUI can render
        # actor-aware copy + route recovery without re-parsing the message).
        "actor": getattr(exc, "actor", None),
        "code": getattr(exc, "code", None),
        "recovery": getattr(exc, "recovery", None),
    }


def _serialize_billing_state(state) -> dict:
    """Serialize a BillingState for the wire (Decimals → strings, money-safe)."""
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)
    card = None
    if state.card is not None:
        card = {
            "brand": state.card.brand,
            "last4": state.card.last4,
            "masked": state.card.masked,
            # None/False on older NAS payloads; resolved_via is the resolution
            # rung for rung-gated surfaces (/subscription confirm).
            "display": state.card.display,
            "resolved_via": state.card.resolved_via,
        }
    payment_method = None
    if state.payment_method is not None:
        pm = state.payment_method
        # Each kind sends only its own fields. Emitting every key with nulls
        # would contradict the shared type — a client checking `'brand' in pm`
        # would read every Link method as a card.
        if pm.kind == "card":
            payment_method = {
                "kind": "card", "brand": pm.brand, "last4": pm.last4, "wallet": pm.wallet,
                "resolved_via": pm.resolved_via,
            }
        elif pm.kind == "link":
            payment_method = {"kind": "link", "email": pm.email, "resolved_via": pm.resolved_via}
        else:
            payment_method = {
                "kind": "unknown", "raw_kind": pm.raw_kind, "resolved_via": pm.resolved_via,
            }
    monthly_cap = None
    if state.monthly_cap is not None:
        mc = state.monthly_cap
        monthly_cap = {
            "limit_usd": _s(mc.limit_usd), "limit_display": format_money(mc.limit_usd),
            "spent_this_month_usd": _s(mc.spent_this_month_usd),
            "spent_display": format_money(mc.spent_this_month_usd),
            "is_default_ceiling": mc.is_default_ceiling,
        }
    auto_reload = None
    if state.auto_reload is not None:
        ar = state.auto_reload
        card_out = None
        if ar.card is not None:
            if ar.card.kind == "distinct":
                card_out = {
                    "kind": "distinct", "payment_method_id": ar.card.payment_method_id,
                    "brand": ar.card.brand, "last4": ar.card.last4,
                }
            else:
                card_out = {"kind": ar.card.kind}
        auto_reload = {
            "enabled": ar.enabled, "threshold_usd": _s(ar.threshold_usd),
            "threshold_display": format_money(ar.threshold_usd),
            "reload_to_usd": _s(ar.reload_to_usd),
            "reload_to_display": format_money(ar.reload_to_usd), "card": card_out,
        }
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "org_name": state.org_name,
        "org_slug": state.org_slug,
        "role": state.role,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "can_charge": state.can_charge,
        "balance_usd": _s(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_s(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _s(state.min_usd),
        "max_usd": _s(state.max_usd),
        "card": card,
        "payment_method": payment_method,
        "monthly_cap": monthly_cap,
        "auto_reload": auto_reload,
        "portal_url": state.portal_url,
        "error": state.error,
        # Shared two-bar dollar usage model so /topup matches /usage and
        # /subscription from one fetch; fail-open.
        "usage": _usage_payload(state),
    }


def _usage_payload(state) -> dict:
    """Best-effort shared usage model for the /topup + /subscription overlay bars.

    Only fetched when logged in; fail-open to {available:false} so the overview
    still renders if the account-info path is down.
    """
    if not getattr(state, "logged_in", False):
        return {"available": False}
    try:
        from agent.billing_usage import build_usage_model
        return _serialize_usage_model(build_usage_model())
    except Exception:
        return {"available": False}


def _serialize_usage_bar(bar) -> Optional[dict]:
    """Serialize a UsageBar (dollar magnitudes → display strings + fractions)."""
    if bar is None:
        return None
    from agent.billing_usage import _fmt_usd
    return {
        "kind": bar.kind, "remaining_display": _fmt_usd(bar.remaining_usd),
        "total_display": _fmt_usd(bar.total_usd), "spent_display": _fmt_usd(bar.spent_usd),
        "pct_used": bar.pct_used, "fill_fraction": bar.fill_fraction,
    }


def _serialize_usage_model(model) -> dict:
    """Serialize a UsageModel for the wire — the shared two-bar dollar view.

    Dollars-only (no 'credits'); fail-open shape mirrors the other billing RPCs
    ({ok, available:false} when logged out / unreachable).
    """
    from agent.billing_usage import _fmt_usd, format_renews
    if model is None or not getattr(model, "available", False):
        return {"ok": True, "available": False}
    return {
        "ok": True,
        "available": True,
        "status": model.status,
        "plan_name": model.plan_name,
        "renews_at": model.renews_at,
        "renews_display": getattr(model, "renews_display", None) or format_renews(model.renews_at),
        "subscription_remaining_display": (
            None if model.subscription_remaining_usd is None else _fmt_usd(model.subscription_remaining_usd)
        ),
        "topup_remaining_display": (
            None if model.topup_remaining_usd is None else _fmt_usd(model.topup_remaining_usd)
        ),
        "total_spendable_display": (
            None if model.total_spendable_usd is None else _fmt_usd(model.total_spendable_usd)
        ),
        "has_topup": model.has_topup,
        "plan_bar": _serialize_usage_bar(model.plan_bar),
        "topup_bar": _serialize_usage_bar(model.topup_bar),
    }


def _serialize_subscription_state(state) -> dict:
    """Serialize a SubscriptionState for the wire (Decimals → strings)."""
    from agent.billing_usage import format_renews
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)
    current = None
    if state.current is not None:
        c = state.current
        current = {
            "tier_id": c.tier_id, "tier_name": c.tier_name,
            "monthly_credits": _s(c.monthly_credits), "credits_remaining": _s(c.credits_remaining),
            "cycle_ends_at": c.cycle_ends_at,
            "pending_downgrade_tier_name": c.pending_downgrade_tier_name,
            "pending_downgrade_at": c.pending_downgrade_at,
            "pending_downgrade_display": format_renews(c.pending_downgrade_at),
            "cancel_at_period_end": c.cancel_at_period_end,
            "cancellation_effective_at": c.cancellation_effective_at,
            "cancellation_effective_display": format_renews(c.cancellation_effective_at),
        }
    # Selectable catalog for the in-terminal tier picker; price is pre-formatted
    # ($X / $X.YY) so the TUI renders it directly.
    tiers = [
        {
            "tier_id": t.tier_id, "name": t.name, "tier_order": t.tier_order,
            "dollars_per_month_display": format_money(t.dollars_per_month),
            "monthly_credits": _s(t.monthly_credits), "is_current": t.is_current,
            "is_enabled": t.is_enabled,
        }
        for t in state.tiers
    ]
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "org_name": state.org_name,
        "org_id": state.org_id,
        "role": state.role,
        "context": state.context,
        "current": current,
        "tiers": tiers,
        "portal_url": state.portal_url,
        "error": state.error,
        # Shared two-bar usage model (account-info is the only source with
        # top-up dollars); fail-open → {available:false}; lazy when logged out.
        "usage": _usage_payload(state),
    }


def _serialize_subscription_preview(p) -> dict:
    """Serialize a SubscriptionChangePreview for the wire (Decimal → string)."""
    return {
        "ok": True,
        "effect": p.effect,
        "reason": p.reason,
        "current_tier_id": p.current_tier_id,
        "current_tier_name": p.current_tier_name,
        "target_tier_id": p.target_tier_id,
        "target_tier_name": p.target_tier_name,
        "monthly_credits_delta": (
            None if p.monthly_credits_delta is None else str(p.monthly_credits_delta)
        ),
        "amount_due_now_cents": p.amount_due_now_cents,
        "effective_at": p.effective_at,
    }


def register(server) -> None:
    """Publish this module's serializers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
