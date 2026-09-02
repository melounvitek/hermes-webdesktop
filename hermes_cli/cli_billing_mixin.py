"""Billing and subscription handlers for the interactive CLI, lifted out of ``cli.py``.

``HermesCLI`` inherits ``CLIBillingMixin`` so every ``self.<handler>`` resolves via the MRO.
cli.py-internal symbols (``_cprint``/``_b``/``_d``, display constants) are imported LAZILY
inside each method — the mixin never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

_RULE = "─" * 41

# Poll `failed` reasons → copy (default: generic line carrying the raw reason).
_CHARGE_FAILED_COPY = {
    "authentication_required": "  🔴 Your bank requires verification (3DS). Complete it on the portal to finish this purchase.",
    "payment_method_expired": "  🔴 Your card has expired. Update it on the portal.",
    "card_declined": "  🔴 Your card was declined. Try another card on the portal.",
}

# Submit-time BillingError codes with a fixed copy (no payload/type inspection).
_CHARGE_ERROR_COPY = {
    "no_payment_method": "  💳 No card on file — top up and manage billing on the portal.",
    "cli_billing_disabled": "  Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.",
    "role_required": "  Adding funds needs an org admin/owner. Ask an admin, or manage on the portal.",
    "idempotency_conflict": "  🔴 That charge key was already used for a different amount. Start a fresh top-up.",
}
_CHARGE_ERROR_COPY["remote_spending_disabled"] = _CHARGE_ERROR_COPY["cli_billing_disabled"]

# Upgrade 2xx `status` → (line, echo recoveryUrl as "Portal:"). Missing status → ambiguous.
_UPGRADE_STATUS_COPY = {
    "requires_action": ("  🟡 This upgrade needs extra verification (3DS). Finish it on the portal.", True),
    "payment_failed": ("  🔴 Your card was declined. Update your payment method on the portal and try again.", True),
}

_ALLOW_REMOTE_SPENDING_CHOICES = [
    ("yes", "Allow Remote Spending", "open your browser to authorize"),
    ("no", "Not now", "cancel"),
]


class CLIBillingMixin:
    """Mixin holding interactive-CLI billing and subscription handlers."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _modal_choice(self, title, detail, choices):
        """Run the choice modal and return the normalized choice value."""
        raw = self._prompt_text_input_modal(title=title, detail=detail, choices=choices)
        return self._normalize_slash_confirm_choice(raw, choices)

    def _print_org_line(self, state) -> None:
        """Dim ``Org: <name> · <Role>`` line (skipped when there is no org)."""
        from cli import _cprint, _d
        if state.org_name:
            role = (state.role or "").title()
            _org_line = f"Org: {state.org_name}{f' · {role}' if role else ''}"
            _cprint(f"  {_d(_org_line)}")

    def _try_usage_model(self):
        """Shared dollar usage model (the only source with top-up dollars); None on any failure."""
        try:
            from agent.billing_usage import build_usage_model
            return build_usage_model()
        except Exception:
            return None

    def _usage_bar_lines(self, usage, plan_name) -> list:
        """The plan + top-up dollar bars as ready-to-print lines (filled = remaining); [] when
        nothing to draw. The caller picks its print fn — block ordering differs per surface
        (``_cprint`` vs ``print`` under patch_stdout). One source of truth for /usage, /subscription, /topup."""
        lines: list = []
        pb = getattr(usage, "plan_bar", None) if usage else None
        if pb is not None and pb.total_usd > 0:
            filled = max(0, min(10, round(pb.fill_fraction * 10)))
            bar = ("█" * filled) + ("░" * (10 - filled))
            pct_s = f" · {pb.pct_used}% used" if pb.pct_used is not None else ""
            label = (plan_name or "plan").ljust(8)[:8]
            lines.append(f"  {label}[{bar}]  ${pb.remaining_usd:,.2f} left of ${pb.total_usd:,.2f}{pct_s}")
        tb = getattr(usage, "topup_bar", None) if usage else None
        if tb is not None and tb.remaining_usd > 0:
            lines.append(f"  {'top-up'.ljust(8)}[{'█' * 10}]  ${tb.remaining_usd:,.2f} · never expires")
        return lines

    def _print_total_spendable(self, usage, print_fn) -> None:
        if usage and getattr(usage, "has_topup", False) and getattr(usage, "total_spendable_usd", None) is not None:
            print_fn(f"  Total spendable: ${usage.total_spendable_usd:,.2f}")

    def _step_up_remote_spending(self, *, explain, noninteractive_msg, declined_msg, not_granted_msg) -> bool:
        """The "! One-time setup" step-up: explain, confirm in the modal, then run the browser
        device-flow via ``step_up_nous_billing_scope``. Returns True only when the scope was granted;
        every refusal/failure path has already printed its own line."""
        from cli import _cprint, _d
        print()
        print("  ! One-time setup")
        _cprint(f"  {_d(explain)}")
        if not self._app:
            print(noninteractive_msg)
            return False
        choice = self._modal_choice("Allow Remote Spending", "Opens your browser to authorize this terminal.", _ALLOW_REMOTE_SPENDING_CHOICES)
        if choice != "yes":
            print(declined_msg)
            return False
        print("  Opening your browser to allow Remote Spending…")
        try:
            from hermes_cli.auth import step_up_nous_billing_scope
            granted = step_up_nous_billing_scope(open_browser=True)
        except Exception as exc:
            print(f"  Couldn't allow Remote Spending: {exc}")
            return False
        if not granted:
            print(not_granted_msg)
            return False
        return True

    def _print_portal_line(self, exc) -> None:
        """``Portal: <url>`` via _cprint when the error carries a portal deep-link."""
        from cli import _cprint
        _url = getattr(exc, "portal_url", None) if exc is not None else None
        if _url:
            _cprint(f"  Portal: {_url}")

    # ------------------------------------------------------------------
    # /usage — Nous balance block
    # ------------------------------------------------------------------

    def _print_nous_credits_block(self) -> bool:
        """Print the Nous dollar balance block (two-bar view) when a Nous account is logged in;
        True if anything printed. Prefers the shared dollar model (``agent.billing_usage``, the
        /usage + /subscription source of truth), falling back to legacy ``nous_credits_lines`` text.
        Agent-independent so the TUI slash-worker (no live agent) still shows it. Fail-open."""
        from cli import _cprint, _b, _d
        try:
            from agent.billing_usage import build_usage_model, format_renews
            usage = build_usage_model()
        except Exception:
            usage = None
            format_renews = None  # type: ignore

        if usage is not None and usage.available and format_renews is not None:
            printed_any = False
            plan = usage.plan_name or ("Free" if usage.status == "free" else None)
            renews_display = getattr(usage, "renews_display", None) or format_renews(usage.renews_at)
            renews = f" · renews {renews_display}" if renews_display else ""
            if plan:
                print()
                _cprint(f"  {_b(f'Plan: {plan}{renews}')}")
                printed_any = True
            # Everything below goes through _cprint like the Plan line: raw print() and _cprint()
            # flush to different buffers under patch_stdout and would interleave nondeterministically.
            for _bar_ln in self._usage_bar_lines(usage, usage.plan_name):
                _cprint(_bar_ln)
                printed_any = True
            self._print_total_spendable(usage, _cprint)
            if usage.status == "free":
                _cprint(f"  {_d('> Free · free models only. Run /subscription to reach paid models.')}")
                printed_any = True
            elif usage.status == "low":
                _amt = f"${usage.total_spendable_usd:,.2f}" if usage.total_spendable_usd is not None else "under $5"
                _cprint(f"  ! Low balance · {_amt} left. Run /topup or /subscription.")
                printed_any = True
            if printed_any:
                return True

        from agent.account_usage import nous_credits_lines
        lines = nous_credits_lines()
        if not lines:
            return False
        print()
        for line in lines:
            print(f"  {line}")
        return True

    def _print_usage_cta(self) -> None:
        """The `/usage` call-to-action; mirrors the TUI's ``USAGE_CTA``. Only printed when a Nous
        account is logged in (both commands are Nous-account only)."""
        from cli import _cprint, _d
        _cprint(f"  {_d('Run /subscription to change plan · /topup to add to your balance')}")

    # ------------------------------------------------------------------
    # /subscription — view plan + change it (CLI surface)
    # ------------------------------------------------------------------

    def _show_subscription(self):
        """`/subscription` (alias `/upgrade`) — CLI mirror of the TUI ``SubscriptionOverlay``.
        Deep-links to NAS's own ``/manage-subscription`` page (NOT the Stripe portal). The terminal
        NEVER charges for a subscription. Fail-open: logged-out / portal hiccup → clear message."""
        from cli import _cprint, _b, _d
        from agent.subscription_view import build_subscription_state, subscription_manage_url

        state = build_subscription_state()
        if not state.logged_in:
            print()
            if state.error:
                _cprint(f"  💳 {_d(f'Could not load subscription: {state.error}')}")
            else:
                _cprint(f"  💳 {_d('Not logged into Nous Portal.')}")
                print("  Run `hermes portal` to log in, then /subscription.")
            return

        if state.context == "team":  # no personal plan — teams run on a shared balance
            print()
            _cprint(f"  ⚕ {_b('Team subscription')}")
            print(f"  {_RULE}")
            self._print_org_line(state)
            org = state.org_name or "a team org"
            print(f"  This terminal is connected to {org}. Teams run on a shared")
            print("  balance · use /topup to add funds.")
            _cprint(f"  {_d('Personal subscriptions live on your personal account.')}")
            return

        self._subscription_overview(state, subscription_manage_url(state))

    def _subscription_overview(self, state, manage_url):
        """Print the plan read block (dollars-only, two-bar view, state-matched nudges), then the
        action: portal hand-off for members / non-interactive, catalog for Free, change menu for paid admins."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews

        usage = self._try_usage_model()
        c = state.current
        is_free = not (c and c.tier_id)
        can_change = state.can_change_plan
        plan_name = (c.tier_name or c.tier_id) if c else (usage.plan_name if usage else None)
        u_status = getattr(usage, "status", None) if usage else None
        renews_display = getattr(usage, "renews_display", None) if usage else None
        if not renews_display and c and c.cycle_ends_at:
            renews_display = format_renews(c.cycle_ends_at)

        # Status line carries a pending change ("→ Plus" / "→ cancels") so the headline itself flags it.
        _flip = ""
        if c and c.cancel_at_period_end:
            _flip = " → cancels"
        elif c and c.pending_downgrade_tier_name:
            _flip = f" → {c.pending_downgrade_tier_name}"
        if not plan_name:
            status = "Plan: Free · free models only"
        elif usage is not None and u_status == "low" and usage.total_spendable_usd is not None:
            status = f"Plan: {plan_name}{_flip} · ${usage.total_spendable_usd:,.2f} left"
        else:
            _spend = getattr(usage, "total_spendable_usd", None) if usage else None
            _left = f" · ${_spend:,.2f} left" if _spend is not None else ""
            _tail = " · view only" if not can_change else (f" · renews {renews_display}" if renews_display else "")
            status = f"Plan: {plan_name}{_flip}{_left}{_tail}"

        # Lead with the scheduled change (cancel > downgrade) so it can't read as "nothing happened".
        # All-_cprint (blanks included) so the block orders deterministically even when piped.
        _trans = None
        if c and c.cancel_at_period_end:
            _trans = ((c.tier_name or "your plan"), "cancels", format_renews(c.cancellation_effective_at) or "the end of the billing period")
        elif c and c.pending_downgrade_tier_name:
            _trans = ((c.tier_name or "your plan"), c.pending_downgrade_tier_name, format_renews(c.pending_downgrade_at) or "the end of the cycle")
        _cprint("")
        if _trans:
            _from, _to, _when = _trans
            _cprint(f"  ⏳ {_b('Scheduled change')}")
            _cprint(f"  {_from} ──▶ {_to}  {_d('· ' + _when)}")
            _cprint(f"  {_d(f'You keep {_from} (and its credits) until then.')}")
            _cprint("")

        _cprint(f"  ⚕ {_b(status)}")
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, plan_name):
            print(_bar_ln)
        self._print_total_spendable(usage, print)
        if is_free:
            _cprint(f"  {_d('> Paid models need a subscription. Start one to reach them.')}")
        elif u_status == "low":
            _amt = f"${usage.total_spendable_usd:,.2f}" if usage is not None and usage.total_spendable_usd is not None else "under $5"
            _cprint(f"  ! Low balance · {_amt} left. Top up or upgrade before a mid-run cutoff.")
        self._print_org_line(state)
        print(f"  {_RULE}")

        if not can_change:
            print()
            _cprint(f"  {_d('Plan changes need an org admin/owner.')}")
            if manage_url:
                print(f"  Manage on portal: {manage_url}")
            return
        if not self._app:  # non-interactive (TUI slash-worker / piped): the modal can't run
            print()
            if manage_url:
                print(f"  Manage your subscription: {manage_url}")
                print("  Open it in your browser, then re-run /subscription.")
            return
        if is_free:  # a NEW subscription needs a fresh card → catalog + portal deep-link only
            self._subscription_free_catalog(state, manage_url)
            return
        self._subscription_change_menu(state, manage_url)

    def _open_url_in_browser(self, url: str) -> bool:
        """Open ``url`` in a REAL graphical browser; the one opener behind every portal hand-off.
        Applies the console-browser / remote-session guard from ``hermes_cli.auth``:
        ``webbrowser.open()`` returns True even for a text-mode browser (w3m/lynx over SSH) that
        hijacks the TTY, so those are refused and the caller prints the URL instead."""
        if not url:
            return False
        try:
            from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session
            if _is_remote_session() or not _can_open_graphical_browser():
                return False
        except Exception:
            pass  # guard unavailable → plain best-effort open
        try:
            import webbrowser
            return bool(webbrowser.open(url))
        except Exception:
            return False

    def _subscription_free_catalog(self, state, manage_url):
        """Free + admin/owner + interactive: print the plan catalog, pick one, open the portal
        manage-subscription deep-link with ``plan=<tier_id>`` so it preselects the plan.
        Monthly credits are DOLLARS. The terminal never charges here (a new sub needs a fresh card)."""
        from cli import _cprint, _b, _d
        from agent.subscription_view import format_tier_row, selectable_tiers, subscription_manage_url

        tiers = selectable_tiers(state)
        if not tiers:
            self._subscription_open_portal(state, manage_url, verb="Start a subscription")
            return

        print()
        _cprint(f"  ⚕ {_b('Choose a plan')}")
        print(f"  {_RULE}")
        for i, t in enumerate(tiers, 1):
            print(f"  {i}. {format_tier_row(t)}")
        _cprint(f"  {_d('Starting a subscription opens the portal to add your card.')}")

        choices = [(t.tier_id, format_tier_row(t), f"start {t.name} on the portal") for t in tiers]
        choices.append(("cancel", "Cancel", "do nothing"))
        raw = self._prompt_text_input_modal(title="Start a subscription", detail="Pick a plan to open it on the portal.", choices=choices)
        # Rows are printed numbered → accept a bare number as a pick (the shared normalizer
        # only knows the confirm-dialog digit aliases).
        _digit = (raw or "").strip()
        if _digit.isdigit() and 1 <= int(_digit) <= len(tiers):
            choice = tiers[int(_digit) - 1].tier_id
        else:
            choice = self._normalize_slash_confirm_choice(raw, choices)
        if not choice or choice == "cancel":
            print("  🟡 Cancelled. No plan started.")
            return
        tier_url = subscription_manage_url(state, tier_id=choice) or manage_url
        if not tier_url:
            _cprint(f"  {_d('No manage URL available — is your portal configured?')}")
            return
        picked = next((t for t in tiers if t.tier_id == choice), None)
        label = picked.name if picked else "your plan"
        if self._open_url_in_browser(tier_url):
            print(f"  Opening the portal to start {label}…")
        else:
            print(f"  Open this URL to start {label}: {tier_url}")
        print("  Finish in your browser, then re-run /subscription.")

    def _subscription_open_portal(self, state, manage_url, *, verb="Manage your subscription"):
        """Open / copy the manage-subscription URL — the portal hand-off."""
        from cli import _cprint, _d
        if not manage_url:
            print()
            _cprint(f"  {_d('No manage URL available — is your portal configured?')}")
            return
        print()
        choices = [
            ("open", verb, "open the subscription page in your browser"),
            ("copy", "Copy link", "copy the manage-subscription URL to your clipboard"),
            ("cancel", "Cancel", "do nothing"),
        ]
        choice = self._modal_choice(verb, "", choices)
        if choice == "open":
            if not self._open_url_in_browser(manage_url):
                print(f"  Open this URL: {manage_url}")
            print()
            print("  Finish in your browser, then re-run /subscription.")
        elif choice == "copy":
            try:
                self._write_osc52_clipboard(manage_url)
                print(f"  📋 Copied: {manage_url}")
            except Exception:
                print(f"  Manage URL: {manage_url}")
        else:
            print("  🟡 Cancelled.")

    def _subscription_change_menu(self, state, manage_url):
        """The in-terminal change menu for a paid admin/owner (interactive)."""
        c = state.current
        has_pending = bool(c and (c.cancel_at_period_end or c.pending_downgrade_tier_name))
        keep_name = (c.tier_name if c else None) or "your plan"
        # A scheduled change makes undo the likeliest intent → promote it first. The Close row uses
        # value "close" (not "cancel") so typing "cancel" can't be confused with "Cancel subscription".
        if has_pending:
            choices = [
                ("keep", f"Keep {keep_name} (undo the scheduled change)", "cancel the pending change"),
                ("change", "Change plan", "upgrade or downgrade in the terminal"),
            ]
        else:
            choices = [
                ("change", "Change plan", "upgrade or downgrade in the terminal"),
                ("cancel_sub", "Cancel subscription", "schedule cancellation at period end"),
            ]
        choices.append(("portal", "Manage on portal", "open the billing page in your browser"))
        choices.append(("close", "Close", "do nothing"))
        choice = self._modal_choice("Manage your subscription", "", choices)
        if choice == "change":
            self._subscription_pick_tier(state)
        elif choice == "keep":
            self._subscription_apply(state, ("resume", None))
        elif choice == "cancel_sub":
            self._subscription_confirm_cancel(state)
        elif choice == "portal":
            self._subscription_open_portal(state, manage_url)
        else:
            print("  🟡 Closed. No plan change.")

    def _subscription_pick_tier(self, state):
        """Tier picker → preview → confirm (mirrors the TUI picker screen). Selectable = enabled paid
        tiers other than current (dropping to free is a cancellation, on the change menu)."""
        from agent.subscription_view import format_tier_row, is_upgrade, selectable_tiers

        c = state.current
        selectable = selectable_tiers(state)
        if not selectable:
            print("  No other plans are available to switch to right now.")
            return
        choices = []
        for t in selectable:
            direction = "upgrade" if is_upgrade(state, t.tier_id) else "downgrade"
            choices.append((t.tier_id, f"{format_tier_row(t)} · {direction}", f"switch to {t.name}"))
        choices.append(("cancel", "Back", "do nothing"))
        choice = self._modal_choice("Change plan", f"Current: {c.tier_name if c else 'Free'}. Pick a plan to preview the effect.", choices)
        if not choice or choice == "cancel":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_preview_and_confirm(state, choice)

    def _subscription_preview_and_confirm(self, state, tier_id, *, allow_stepup=True):
        """Preview the change (chargeless quote), show the effect, then confirm+apply.
        ``allow_stepup=False`` (a post-grant replay) declines a second step-up on a repeated scope
        denial so the flow can't re-prompt/re-open the browser in a loop."""
        from cli import _cprint, _b, _d
        from agent.subscription_view import is_upgrade, subscription_change_preview_from_payload, subscription_manage_url
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_subscription_preview

        _cprint(f"  {_d('Checking the change…')}")
        try:
            payload = post_subscription_preview(subscription_type_id=tier_id)
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=("preview", tier_id))
            else:
                print("  Remote Spending still isn't active for this terminal — the authorization didn't take. Retry, or make this change on the portal.")
            return
        except BillingError as exc:
            self._subscription_render_error(state, exc)
            return
        p = subscription_change_preview_from_payload(payload)
        effect = p.effect
        target = p.target_tier_name or "the selected plan"
        print()
        if effect == "no_op":
            _cprint(f"  {_d(f'You are already on {target} — nothing to change.')}")
            return
        if effect not in ("charge_now", "scheduled"):
            # blocked OR unknown effect → fail SAFE (never schedule a real change on an unrecognized
            # string) and re-offer the portal. plan= rides along only for an UPGRADE hand-off —
            # downgrades stay native, so a blocked downgrade keeps the generic manage link.
            _plan = tier_id if is_upgrade(state, tier_id) else None
            _cprint(f"  🟡 {p.reason or 'This change cannot be confirmed here — manage it on the portal.'}")
            _mu = subscription_manage_url(state, tier_id=_plan)
            if _mu:
                print(f"  Manage on portal: {_mu}")
            return
        if effect == "charge_now":
            _amt = f"${p.amount_due_now_cents / 100:.2f}" if p.amount_due_now_cents is not None else None
            _cprint(f"  {_b('Confirm plan change')}  {_d('· charged now')}")
            if _amt:
                _cprint(f"  Upgrade to {target}. You will be charged {_amt} now (prorated).")
            else:
                _cprint(f"  Upgrade to {target}. You will be charged the prorated amount now.")
            # Best-effort: name the exact card, but only when the resolver rung matches what a
            # subscription charge actually uses (subPin / customerDefault — Stripe's precedence).
            _card_line = "The card on your subscription will be charged."
            try:
                from agent.billing_view import build_billing_state
                _bs = build_billing_state(timeout=6.0)
                _c = _bs.card if _bs.logged_in else None
                if _c is not None and _c.resolved_via in ("subPin", "customerDefault"):
                    _card_line = f"{_c.masked} — the card on your subscription — will be charged."
            except Exception:
                pass
            _cprint(f"  {_d(_card_line)}")
            pay_label = f"Pay {_amt} & upgrade now" if _amt else "Upgrade now (prorated charge)"
            action = ("upgrade", tier_id)
            # The money-moving row is NOT the default — a bare Enter hits "Go back", so a stray keystroke can't charge.
            confirm_choices = [
                ("cancel", "Go back", "do not charge"),
                ("yes", pay_label, "charge + upgrade now"),
            ]
        else:  # scheduled (whitelisted above)
            _when = p.effective_at[:10] if (p.effective_at and len(p.effective_at) >= 10) else "the end of the billing period"
            _cprint(f"  {_b('Confirm plan change')}  {_d('· scheduled · not today')}")
            _cprint(f"  Change to {target} — takes effect {_when}. No charge now; you keep your current plan until then.")
            pay_label = f"Schedule change to {target}"
            action = ("schedule", tier_id)
            confirm_choices = [
                ("yes", pay_label, "apply this change"),
                ("cancel", "Go back", "do not change"),
            ]
        if p.monthly_credits_delta:
            _cprint(f"  {_d(f'Monthly credits change: {p.monthly_credits_delta}.')}")
        if self._modal_choice(pay_label, "", confirm_choices) != "yes":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_apply(state, action, allow_stepup=allow_stepup)

    def _subscription_confirm_cancel(self, state):
        """Confirm, then schedule a cancellation at period end."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews

        c = state.current
        _end = (format_renews(c.cycle_ends_at) if (c and c.cycle_ends_at) else None) or "the end of the billing period"
        print()
        _cprint(f"  {_b('Confirm cancellation')}  {_d('· scheduled · not today')}")
        _cprint(f"  Cancel {(c.tier_name if c else 'your plan')} — it stays active until {_end}, then won't renew.")
        _cprint(f"  {_d('You keep your remaining credits for this period. You can resume before it ends.')}")
        confirm_choices = [
            ("yes", "Cancel subscription", "schedule cancellation at period end"),
            ("cancel", "Go back", "keep your plan"),
        ]
        if self._modal_choice("Cancel subscription?", "", confirm_choices) != "yes":
            print("  🟡 Cancelled. Your plan is unchanged.")
            return
        self._subscription_apply(state, ("cancel", None))

    def _subscription_apply(self, state, action, idempotency_key=None, *, allow_stepup=True):
        """Run the mutation for `action` — ("upgrade"|"schedule", tier_id) / ("cancel"|"resume", None) —
        handling the scope step-up + the result. insufficient_scope routes to the step-up and replays;
        the upgrade idempotency key is reused across the replay. ``allow_stepup=False`` (post-grant
        replay) declines a second step-up so the flow can't re-prompt/re-open the browser in a loop."""
        from cli import _cprint, _d, _DIM, _RST
        from hermes_cli.nous_billing import (
            BillingError, BillingTransient, BillingRemoteSpendingRevoked, BillingScopeRequired, BillingSessionRevoked,
            delete_subscription_pending_change, post_subscription_upgrade, put_subscription_pending_change,
        )

        kind, arg = action
        key = None
        if kind == "upgrade":
            from agent.billing_view import new_idempotency_key
            key = idempotency_key or new_idempotency_key()
        try:
            if kind == "upgrade":
                try:
                    res = post_subscription_upgrade(subscription_type_id=arg, idempotency_key=key) or {}
                except BillingScopeRequired:
                    raise  # a scope denial rejects BEFORE charging → route to the step-up
                except (BillingTransient, BillingSessionRevoked, BillingRemoteSpendingRevoked) as exc:
                    # Deterministic PRE-charge rejections (429/401/403) never reached Stripe →
                    # the correct recovery copy, NOT the "maybe charged" ambiguity.
                    self._subscription_render_error(state, exc)
                    return
                except BillingError as exc:
                    _status = getattr(exc, "status", None)
                    _code = getattr(exc, "error", None)
                    if _code in ("network_error", "endpoint_unavailable") or _status is None or _status >= 500:
                        # Genuinely INDETERMINATE (transport / unparseable 2xx / 5xx): NAS may already have
                        # charged. Steer to a re-check, never a blind retry (a fresh key can't dedup).
                        self._subscription_render_upgrade_ambiguous(exc)
                    else:  # deterministic 4xx (role_required / no_payment_method / …)
                        self._subscription_render_error(state, exc)
                    return
                status = res.get("status")
                name = res.get("targetTierName") or "your new plan"
                _url = res.get("recoveryUrl")
                if status == "already_on_tier":
                    _cprint(f"  {_DIM}✓ You are already on {name}.{_RST}")
                elif status == "upgraded":
                    _cprint(f"  {_DIM}✓ Upgraded to {name}. Your new monthly credits land in a moment.{_RST}")
                elif status in _UPGRADE_STATUS_COPY:
                    line, echo_url = _UPGRADE_STATUS_COPY[status]
                    _cprint(line)
                    if echo_url and _url:
                        _cprint(f"  Portal: {_url}")
                else:  # unknown / absent 2xx status → also ambiguous, not a flat failure
                    self._subscription_render_upgrade_ambiguous(None)
                return
            if kind == "schedule":
                put_subscription_pending_change(subscription_type_id=arg)
                _cprint(f"  {_DIM}✓ Scheduled — your plan doesn't change today. You keep it until the end of the billing period, then it switches.{_RST}")
            elif kind == "cancel":
                put_subscription_pending_change(cancel=True)
                _cprint(f"  {_DIM}✓ Scheduled — your plan stays active until the end of the billing period, then it cancels. Nothing changes today.{_RST}")
            elif kind == "resume":
                delete_subscription_pending_change()
                _cprint(f"  {_DIM}✓ Undone — you stay on your current plan.{_RST}")
            _cprint(f"  {_d('Re-run /subscription anytime to review it.')}")
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=action, idempotency_key=key)
            else:
                print("  Remote Spending still isn't active for this terminal — the authorization didn't take. Retry, or make this change on the portal.")
        except BillingError as exc:
            self._subscription_render_error(state, exc)

    def _subscription_handle_scope_required(self, state, *, retry, idempotency_key=None):
        """insufficient_scope → allow remote spending (step-up), then replay `retry` ONCE so the
        user never re-runs the command."""
        from cli import _cprint, _DIM, _RST
        granted = self._step_up_remote_spending(
            explain="To change your plan from the terminal, allow Remote Spending once. It opens your browser to authorize, then your change picks up right here.",
            noninteractive_msg="  Run `hermes portal` and allow Remote Spending, then re-run /subscription.",
            declined_msg="  No change made. Allow Remote Spending when you're ready.",
            not_granted_msg="  Couldn't allow Remote Spending — an org admin or owner has to approve it for this org.",
        )
        if not granted:
            return
        _cprint(f"  {_DIM}✓ Remote Spending allowed.{_RST}")
        # Bust the 30s token cache: it still holds the pre-grant unscoped token and _request only
        # busts on a 401 (not a 403 scope denial) — without this the replay would 403 again.
        try:
            from hermes_cli import nous_billing as _nb
            _nb.invalidate_cached_token()
        except Exception:
            pass
        # Re-fetch fresh state, then replay the held action ONCE (allow_stepup=False).
        from agent.subscription_view import build_subscription_state
        try:
            fresh = build_subscription_state()
        except Exception:
            fresh = state
        rkind, rarg = retry
        if rkind == "preview":
            self._subscription_preview_and_confirm(fresh, rarg, allow_stepup=False)
        else:
            self._subscription_apply(fresh, retry, idempotency_key=idempotency_key, allow_stepup=False)

    def _subscription_render_error(self, state, exc):
        """Render a subscription BillingError (a lighter _billing_render_charge_error)."""
        from cli import _cprint
        code = getattr(exc, "error", None)
        msg = str(exc) or "Something went wrong."
        if code == "insufficient_scope":  # defensive: the flow routes scope to the step-up before here
            _cprint("  🟡 Remote Spending isn't allowed yet. Allow it, then retry.")
        elif code in ("subscription_mutation_rejected", "preview_rejected"):
            _cprint(f"  🟡 {msg}")
        else:
            _cprint(f"  🔴 {msg}")
        self._print_portal_line(exc)

    def _subscription_render_upgrade_ambiguous(self, exc):
        """A charge-route failure (transport / timeout / 500 / unknown status) is AMBIGUOUS — NAS may
        already have charged. Steer to a re-check, never a flat failure that invites a blind retry
        (the CLI can't persist the key across a command re-run)."""
        from cli import _cprint, _d
        _cprint("  🟡 Couldn't confirm the upgrade — your card may or may not have been charged.")
        _cprint(f"  {_d('Re-run /subscription to check your plan before trying again.')}")
        self._print_portal_line(exc)

    # ------------------------------------------------------------------
    # /topup — Remote Spending (CLI surface, all 5 screens)
    # ------------------------------------------------------------------

    def _show_billing(self, command: str = "/topup"):
        """`/topup` — Remote Spending for Nous. ZERO sub-commands: any argument is ignored and the
        Overview menu is the only way to reach Buy / Auto-reload / Monthly-limit. Non-interactive
        contexts render text + the portal deep-link, never prompting. All money is Decimal end-to-end;
        the terminal never collects card details."""
        from cli import _cprint, _d
        from agent.billing_view import build_billing_state

        state = build_billing_state()
        if not state.logged_in:
            print()
            if state.error:
                _msg = f"Couldn't load billing: {state.error}"
                _cprint(f"  💳 {_d(_msg)}")
            else:
                _cprint(f"  💳 {_d('Not logged into Nous Portal.')}")
                print("  Run `hermes portal` to log in, then /topup.")
            return
        self._billing_overview(state)

    def _billing_portal_hint(self, state, *, reason: str = "") -> None:
        """Print a portal deep-link line (the funnel for portal-only actions)."""
        url = getattr(state, "portal_url", None)
        if not url:
            return
        if reason:
            print(f"  {reason}")
        print(f"  Manage on portal: {url}")

    def _billing_overview(self, state):
        """Screen 1 — balance in title, two-bar dollar usage, action menu (Add funds first).
        No scope preflight — remote spending is discovered reactively when a charge 403s. A missing
        card does NOT gate the overview either: it only matters at CHARGE time (the buy flow hands off)."""
        from cli import _cprint, _b, _d
        from agent.billing_view import format_money

        usage = self._try_usage_model()
        print()
        _cprint(f"  💳 {_b(f'Top up · balance {format_money(state.balance_usd)}')}")
        self._print_org_line(state)
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, getattr(usage, "plan_name", None)):
            print(_bar_ln)

        ar = state.auto_reload
        if ar is not None:
            if ar.enabled:
                print(f"  Auto-reload: on — below {format_money(ar.threshold_usd)} → reload to {format_money(ar.reload_to_usd)}")
            else:
                print("  Auto-reload: off")
        if state.can_change_plan and state.cli_billing_enabled:  # card at a glance, full-menu case only
            if state.card is not None:
                print(f"  Card: {state.card.display}")
            else:
                _cprint(f"  {_d('No saved card on file — “Add funds” walks you through adding one.')}")
        print(f"  {_RULE}")

        # Action gating: admin + kill-switch for charge/auto-reload; everyone gets portal.
        if not state.can_change_plan:
            _cprint(f"  {_d('Billing actions require an org admin/owner.')}")
            self._billing_portal_hint(state)
            return
        if not state.cli_billing_enabled:
            _cprint(f"  {_d('Remote spending is off for this org.')}")
            self._billing_portal_hint(state, reason="A billing admin can turn it on from the portal's Hermes Agent page to add funds here.")
            return
        if not self._app:  # non-interactive: no modal, just the portal funnel
            self._billing_portal_hint(state)
            return

        # One-time vs automatic — the distinction stated up front in each first sentence.
        _cprint(f"  {_d('Add funds now — a single charge, added to your balance today.')}")
        if (
            ar is not None and ar.enabled
            and ar.reload_to_usd is not None and ar.reload_to_usd.is_finite()
            and ar.threshold_usd is not None and ar.threshold_usd.is_finite()
        ):
            _auto_line = f"Refill when low — charges {format_money(ar.reload_to_usd)} automatically when your balance falls below {format_money(ar.threshold_usd)}."
        else:
            _auto_line = "Refill when low — charges your card automatically when your balance falls below the amount you set."
        _cprint(f"  {_d(_auto_line)}")
        print(f"  {_RULE}")

        # No "Allow Remote Spending" item — discovered at pay time. "Add funds" charges in-terminal
        # against the org's portal-saved card (server-held; no card ref leaves the client).
        choices = [
            ("buy", "Add funds", "a single charge, added to your balance today"),
            ("auto", "Auto-reload", "refill automatically when your balance runs low"),
            ("limit", "Monthly limit", "show the monthly spend cap (read-only)"),
            ("portal", "Manage on portal", "open the billing page in your browser"),
            ("cancel", "Cancel", "do nothing"),
        ]
        choice = self._modal_choice("Top up your balance", "", choices)
        if choice == "buy":
            self._billing_buy_flow(state)
        elif choice == "auto":
            self._billing_auto_reload_flow(state)
        elif choice == "limit":
            self._billing_limit_screen(state)
        elif choice == "portal":
            self._billing_open_portal(state)
        else:
            print("  Cancelled.")

    def _billing_open_portal(self, state):
        url = getattr(state, "portal_url", None)
        if not url:
            print("  No portal URL available.")
            return
        if not self._open_url_in_browser(url):
            print(f"  Open this URL: {url}")
        print("  Complete billing changes in the browser.")

    def _billing_require_admin(self, state) -> bool:
        """Guard charge/auto-reload entry points; print + return False if blocked."""
        from cli import _cprint, _d
        if not state.can_change_plan:
            print()
            _cprint(f"  💳 {_d('Billing actions require an org admin/owner.')}")
            self._billing_portal_hint(state)
            return False
        if not state.cli_billing_enabled:
            print()
            _cprint(f"  💳 {_d('Remote spending is off for this org.')}")
            self._billing_portal_hint(state, reason="A billing admin can turn it on from the portal's Hermes Agent page before adding funds.")
            return False
        return True

    def _billing_add_card_flow(self, state):
        """No saved card → guide adding one on the portal (never in-terminal), with a bounded re-check
        loop so the purchase continues right here once the card is saved (also recovers a transient
        miss — the card display is best-effort server-side). Returns refreshed state, or None to abandon."""
        from cli import _cprint, _b, _d, _DIM, _RST
        print()
        _cprint(f"  💳 {_b('Add a card first')}")
        _cprint("  No saved card on file.")
        _cprint(f"  {_d('Add a card once on the portal billing page — after that you can top up right from the terminal.')}")
        choices = [
            ("portal", "Add a card on the portal", "opens the billing page in your browser"),
            ("recheck", "I've added it — check again", "re-check for the card and continue"),
            ("cancel", "Back", "do nothing"),
        ]
        for _ in range(8):  # bounded: portal-open plus a handful of re-checks
            choice = self._modal_choice("Add a card", "", choices)
            if choice == "portal":
                self._billing_open_portal(state)
                _cprint(f"  {_d('Add the card on the billing page, then pick “check again” here.')}")
                continue
            if choice == "recheck":
                from agent.billing_view import build_billing_state
                try:
                    fresh = build_billing_state()
                except Exception:
                    fresh = None
                if fresh is not None and fresh.logged_in:
                    state = fresh
                if state.card is not None:
                    _cprint(f"  {_DIM}✓ Card found: {state.card.display} — continuing.{_RST}")
                    return state
                print("  Still no card on file — finish adding it on the portal, then check again.")
                continue
            break
        print("  Cancelled. No funds added.")
        return None

    def _billing_buy_flow(self, state):
        """Screen 2 (preset select) → Screen 3 (confirm + charge + poll). No scope preflight: the
        charge flies and we react to the server's 403 order (insufficient_scope → in-flight reauth,
        no_payment_method → portal hand-off)."""
        from cli import _cprint, _b
        from agent.billing_view import format_money, validate_charge_amount

        if not self._billing_require_admin(state):
            return
        if not self._app:
            presets = ", ".join(format_money(p) for p in state.charge_presets)
            print()
            _cprint(f"  💳 {_b('Add funds')}")
            print(f"  Presets: {presets}")
            print("  Run this in the interactive CLI to complete a purchase.")
            self._billing_portal_hint(state)
            return
        if state.card is None:  # guided add-card path first, so the amount pick can't 403
            state = self._billing_add_card_flow(state)
            if state is None or state.card is None:
                return

        preset_choices = [(str(p), format_money(p), "one-time credit purchase") for p in state.charge_presets]
        preset_choices.append(("custom", "Custom amount…", "enter your own amount"))
        preset_choices.append(("cancel", "Cancel", "do nothing"))
        card = state.card
        choice = self._modal_choice("Add funds", f"Payment: {card.display}" if card else "No saved card on file", preset_choices)
        if not choice or choice == "cancel":
            print("  Cancelled. No funds added.")
            return

        from decimal import Decimal
        if choice == "custom":
            entered = self._prompt_text_input("  Amount (USD): ")
            if entered is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print("  Cancelled. No funds added.")
                return
            v = validate_charge_amount(entered or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok:
                print(f"  🔴 {v.error}")
                return
            amount = v.amount
        else:
            try:
                amount = Decimal(choice)
            except Exception:
                print("  🔴 Invalid selection.")
                return
        self._billing_confirm_and_charge(state, amount)

    def _billing_confirm_and_charge(self, state, amount):
        """Screen 3 — confirm total + consent, charge, then poll to settlement."""
        from cli import _cprint, _b, _d
        from agent.billing_view import format_money, new_idempotency_key
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_charge

        card = state.card
        print()
        _cprint(f"  💳 {_b('Confirm purchase')}")
        print(f"  {_RULE}")
        print(f"  Total: {format_money(amount)}")
        if card:
            print(f"  Payment: {card.display}")
            if card.provenance is None:  # older NAS without provenance → generic line
                _cprint(f"  {_d('Your card saved on the portal will be charged.')}")
        print(f"  {_RULE}")
        _cprint(f"  {_d('By confirming, you allow Nous Research to charge your card.')}")

        confirm_choices = [
            ("pay", f"Pay {format_money(amount)} now", "submit the charge"),
            ("portal", "Manage on portal", "manage your card / billing in the browser"),
            ("cancel", "Go back", "do not charge"),
        ]
        if not self._app:
            print("  Run in the interactive CLI to confirm a purchase.")
            return
        choice = self._modal_choice(f"Pay {format_money(amount)}?", card.display if card else "no saved card", confirm_choices)
        if choice == "portal":
            self._billing_open_portal(state)
            return
        if choice != "pay":
            print("  Cancelled. No funds added.")
            return

        key = new_idempotency_key()  # reused on the post-step-up resume so a double-submit collapses
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingScopeRequired:
            self._billing_handle_scope_required(state, amount=amount, idempotency_key=key)
            return
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return
        charge_id = result.get("chargeId")
        if not charge_id:
            print("  🔴 No charge id returned; please check the portal.")
            return
        _cprint(f"  {_d('Charge submitted — confirming settlement…')}")
        self._billing_poll_charge(state, charge_id, amount)

    def _billing_poll_charge(self, state, charge_id, amount):
        """Poll loop: 2s interval, 5-min cap, cancellable. settled = ledger truth."""
        import time as _time
        from agent.billing_view import format_money, parse_money
        from hermes_cli.nous_billing import BillingError, BillingTransient, get_charge_status

        deadline = _time.time() + 300
        while _time.time() < deadline:
            try:
                status = get_charge_status(charge_id)
            except BillingTransient as exc:  # retry-after, NOT a failure — back off and keep polling
                _time.sleep(min(exc.retry_after or 5, 30))
                continue
            except BillingError as exc:
                print(f"  🔴 Could not check the charge: {exc}")
                return
            state_str = status.get("status")
            if state_str == "settled":
                amt = status.get("amountUsd")
                shown = format_money(parse_money(amt)) if amt else format_money(amount)
                print(f"  ✓ {shown} added to your balance.")
                return
            if state_str == "failed":
                self._billing_render_charge_failed(state, status.get("reason"))
                return
            _time.sleep(2.0)  # pending
        print("  🟡 Still processing after 5 minutes — this is a timeout, not a failure. Check /billing or the portal shortly.")
        self._billing_portal_hint(state)

    def _billing_render_charge_failed(self, state, reason):
        """Poll `failed` reasons → the right copy + portal funnel."""
        reason = (reason or "").strip()
        print(_CHARGE_FAILED_COPY.get(reason) or f"  🔴 The charge didn't go through ({reason or 'processing_error'}).")
        self._billing_portal_hint(state)

    def _billing_render_charge_error(self, state, exc):
        """Render a typed BillingError at submit time (pre-poll). Order matters: revoked/session
        checks precede code lookups, and Transient precedes the insufficient_scope fallback."""
        from hermes_cli.nous_billing import BillingTransient, BillingRemoteSpendingRevoked, BillingSessionRevoked

        code = getattr(exc, "error", None)
        portal_url = getattr(exc, "portal_url", None) or getattr(state, "portal_url", None)
        if isinstance(exc, BillingRemoteSpendingRevoked) or code == "remote_spending_revoked":
            # This terminal's spend was revoked; recovery is reconnect.
            who = "An admin stopped this terminal's spending." if getattr(exc, "actor", None) == "admin" else "You stopped this terminal's spending."
            print(f"  🔴 {who} Reconnect to restore — run `hermes portal` to re-authorize.")
        elif isinstance(exc, BillingSessionRevoked) or code == "session_revoked":
            print("  🔴 Your session was logged out. Run `hermes portal` to log in again.")
        elif code == "no_payment_method":
            print(_CHARGE_ERROR_COPY[code])
        elif code in ("cli_billing_disabled", "remote_spending_disabled") or getattr(exc, "code", None) == "remote_spending_disabled":
            print(_CHARGE_ERROR_COPY["cli_billing_disabled"])  # dual error/code gate payload
        elif code in _CHARGE_ERROR_COPY:  # role_required / idempotency_conflict
            print(_CHARGE_ERROR_COPY[code])
        elif code == "monthly_cap_exceeded":
            remaining = (getattr(exc, "payload", {}) or {}).get("remainingUsd")
            print(f"  🔴 Monthly spend cap reached — ${remaining} headroom left." if remaining is not None else "  🔴 Monthly spend cap reached.")
        elif isinstance(exc, BillingTransient):
            wait = getattr(exc, "retry_after", None)
            mins = f" (try again in ~{max(1, round(wait / 60))} min)" if wait else ""
            print(f"  🟡 Too many charges right now{mins}. This isn't a payment failure.")
        elif code == "insufficient_scope":
            # Never leak the raw billing:manage scope (a raced post-grant replay can re-raise it).
            print("  🔴 Remote Spending needs approval — run /topup to allow it, then retry.")
        else:
            print(f"  🔴 {exc}")
        if portal_url:
            print(f"  Portal: {portal_url}")

    def _billing_handle_scope_required(self, state, *, amount=None, idempotency_key=None):
        """403 insufficient_scope → in-flight reauth, then resume the held ``amount`` on an explicit
        press-Enter confirm, reusing ``idempotency_key`` so the resumed charge collapses with the
        original. Never leaks the raw billing:manage scope."""
        from cli import _cprint, _d
        from agent.billing_view import build_billing_state, format_money, new_idempotency_key

        amount_str = format_money(amount) if amount is not None else "your top-up"
        granted = self._step_up_remote_spending(
            explain=f"To charge from this terminal, allow Remote Spending once. It opens your browser to authorize, then {amount_str} picks up right here.",
            noninteractive_msg="  Run `hermes portal` and allow Remote Spending, then retry.",
            declined_msg="  No charge made. Run /topup when you want to allow Remote Spending.",
            not_granted_msg="  Couldn't allow Remote Spending — an org admin or owner has to approve it. Your card was not charged.",
        )
        if not granted:
            return

        # The token now carries the scope, but the ORG kill-switch (cli_billing_enabled) is a separate
        # gate — re-fetch /state so we don't over-promise.
        fresh = build_billing_state()
        if not (fresh.logged_in and fresh.cli_billing_enabled):
            print("  Remote Spending is allowed for this terminal, but it's still off for this org. A billing admin can turn it on from the portal's Hermes Agent page, then run /topup again.")
            self._billing_portal_hint(fresh)
            return
        if fresh.card is None:  # half-done state: say so rather than a bare "✓ enabled"
            print("  ✓ Remote Spending allowed — but there's no card on file yet.")
            _cprint(f"  {_d('Top up and manage billing on the portal to continue.')}")
            self._billing_portal_hint(fresh)
            return
        if amount is None:  # scope-required hit outside a charge (e.g. auto-reload config)
            print("  ✓ Remote Spending allowed. Run /topup to continue.")
            return

        print("  ✓ Remote Spending allowed.")
        resume_choices = [
            ("resume", f"Resume {format_money(amount)} top-up", "finish the held purchase"),
            ("cancel", "Cancel", "do not charge"),
        ]
        if self._modal_choice("Resume your top-up", f"{format_money(amount)} is ready to finish — press Enter to resume.", resume_choices) != "resume":
            print("  Cancelled. No funds added.")
            return

        from hermes_cli.nous_billing import BillingError, post_charge
        key = idempotency_key or new_idempotency_key()
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingError as exc:
            self._billing_render_charge_error(fresh, exc)
            return
        charge_id = result.get("chargeId")
        if not charge_id:
            print("  No charge id returned; please check the portal.")
            return
        _cprint(f"  {_d('Resuming your top-up — confirming settlement…')}")
        self._billing_poll_charge(fresh, charge_id, amount)

    def _billing_auto_reload_flow(self, state):
        """Screen 4 — auto-reload config: threshold + reload-to → PATCH. Prefills current values;
        validates both (2dp, within bounds, ``reload_to > threshold``); offers "Turn off" when already on."""
        from cli import _cprint, _b, _d
        from agent.billing_view import format_money, validate_charge_amount

        if not self._billing_require_admin(state):
            return
        card = state.card
        ar = state.auto_reload
        currently_on = bool(ar and ar.enabled)

        print()
        _cprint(f"  💳 {_b('Auto-reload')}")
        print(f"  {_RULE}")
        _cprint(f"  {_d('Automatically add funds when your balance is low.')}")
        if card:
            print(f"  Card on file: {card.masked}")
        else:
            print("  No saved card — manage billing on the portal.")
            self._billing_portal_hint(state)
            return
        if currently_on:
            print(f"  Currently: below {format_money(ar.threshold_usd)} → reload to {format_money(ar.reload_to_usd)}")
        if not self._app:
            print("  Run in the interactive CLI to configure auto-reload.")
            self._billing_portal_hint(state)
            return

        if currently_on:  # let the user turn it off without re-entering values
            top_choices = [
                ("edit", "Edit thresholds", "change when / how much to reload"),
                ("off", "Turn off", "disable auto-reload"),
                ("cancel", "Cancel", "do nothing"),
            ]
            top = self._modal_choice("Auto-reload", f"On — below {format_money(ar.threshold_usd)} → reload to {format_money(ar.reload_to_usd)}", top_choices)
            if top == "off":
                self._billing_auto_reload_disable(state)
                return
            if top != "edit":
                print("  🟡 Cancelled.")
                return

        _CANCELLED = object()

        def _ask_amount(label, current):
            """Prompt one amount; empty input keeps `current` when editing. Returns the Decimal (or the
            kept current value, possibly None), or _CANCELLED after printing the cancel/validation message."""
            cur = format_money(current) if currently_on else None
            raw = self._prompt_text_input(f"  {label} (USD)" + (f" [{cur}]: " if cur else ": "))
            if raw is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print("  🟡 Cancelled.")
                return _CANCELLED
            if not (raw or "").strip() and currently_on:
                return current
            v = validate_charge_amount(raw or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok or v.amount is None:
                print(f"  🔴 {v.error}")
                return _CANCELLED
            return v.amount

        threshold_amt = _ask_amount("When balance falls below", ar.threshold_usd if currently_on else None)
        if threshold_amt is _CANCELLED:
            return
        reload_amt = _ask_amount("Reload balance to", ar.reload_to_usd if currently_on else None)
        if reload_amt is _CANCELLED:
            return
        if reload_amt is None or threshold_amt is None or reload_amt <= threshold_amt:
            print("  🔴 Reload-to amount must be greater than the threshold.")
            return

        print()
        _cprint(f"  {_d(f'By confirming, you authorize Nous Research to charge {card.masked} whenever your balance reaches {format_money(threshold_amt)}. Turn off any time here or on the portal.')}")
        confirm_choices = [
            ("agree", "Agree and turn on", "enable auto-reload"),
            ("cancel", "Cancel", "do nothing"),
        ]
        if self._modal_choice("Turn on auto-reload?", f"Below {format_money(threshold_amt)} → reload to {format_money(reload_amt)}", confirm_choices) != "agree":
            print("  🟡 Cancelled.")
            return
        if self._billing_patch_auto_top_up(state, enabled=True, threshold=float(threshold_amt), top_up_amount=float(reload_amt)):
            print(f"  ✅ Auto-reload on: below {format_money(threshold_amt)} → reload to {format_money(reload_amt)}.")

    def _billing_patch_auto_top_up(self, state, **kwargs) -> bool:
        """PATCH auto-top-up, routing scope denials to the step-up and other errors to the renderer.
        True on success."""
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, patch_auto_top_up
        try:
            patch_auto_top_up(**kwargs)
        except BillingScopeRequired:
            self._billing_handle_scope_required(state)
            return False
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return False
        return True

    def _billing_auto_reload_disable(self, state):
        """Turn off auto-reload (PATCH ``enabled:false``). The endpoint requires ``threshold``/
        ``topUpAmount`` even when disabling, so echo back the current values (fallback 0)."""
        ar = state.auto_reload
        thr = float(ar.threshold_usd) if ar and ar.threshold_usd is not None else 0.0
        rel = float(ar.reload_to_usd) if ar and ar.reload_to_usd is not None else 0.0
        if self._billing_patch_auto_top_up(state, enabled=False, threshold=thr, top_up_amount=rel):
            print("  ✅ Auto-reload turned off.")

    def _billing_limit_screen(self, state):
        """Screen 5 — monthly spend limit (read-only; cap is portal-only)."""
        from cli import _cprint, _b, _d
        from agent.billing_view import format_money

        print()
        _cprint(f"  💳 {_b('Monthly spend limit')}")
        print(f"  {_RULE}")
        cap = state.monthly_cap
        if cap is None or cap.limit_usd is None:
            _cprint(f"  {_d('No monthly cap visible (managed on the portal).')}")
        else:
            ceiling = " (default ceiling)" if cap.is_default_ceiling else ""
            print(f"  {format_money(cap.spent_this_month_usd)} of {format_money(cap.limit_usd)} used this month{ceiling}")
        _cprint(f"  {_d('The monthly limit is set on the portal — the terminal shows it read-only.')}")
        self._billing_portal_hint(state)
