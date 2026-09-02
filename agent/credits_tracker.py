"""Credits tracking for Nous inference API responses: parses x-nous-credits-*
(and optional x-nous-tool-pool-*) headers into a validated CreditsState, with
depletion detection (paid_access), subscription-cap used_fraction, and warn-once
schema-version gating. The hardened parser used by all live consumers.

Header schema (x-nous-credits-*; each *-micros balance has a *-usd twin holding
the server's formatted USD string):
    version                    contract/schema version
    remaining-micros/-usd      total remaining balance
    subscription-micros/-usd   subscription balance (SIGNED; may be negative/debt)
    subscription-limit-*       subscription cap (PAIRED/optional)
    rollover-micros            rolled-over balance
    purchased-micros/-usd      purchased balance
    denominator-kind           "subscription_cap" | "none"
    paid-access                "true" | "false" (STRING!)
    disabled-reason            reason string (header omitted when null)
    as-of-ms                   server-side timestamp (ms epoch)
Tool-pool headers use a SEPARATE prefix: x-nous-tool-pool-micros (balance) and
x-nous-tool-pool-gated-off ("true" | "false" STRING!).

Money is handled as micros ints only; *_usd values are preserved verbatim as
the raw strings the server sent (never re-parsed to float).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)

# Warn-once latch: emit the version-unsupported warning at most once per process.
_version_warning_emitted: bool = False

# Valid denominator kinds (exhaustive set from the API contract).
_VALID_DENOMINATOR_KINDS = frozenset({"subscription_cap", "none"})

# USD format: optional leading minus, one-or-more digits, dot, exactly 2 digits.
_USD_RE = re.compile(r"^-?\d+\.\d{2}$")

_SENTINEL = object()  # singleton sentinel for "parse failed"


def _safe_int(value: Any) -> Any:
    """Exact int (money-safe) or ``_SENTINEL``. ``int()`` directly, NOT ``int(float())``
    (precision loss above 2**53 corrupts money); float-shaped strings fail."""
    if value is None:
        return _SENTINEL
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return _SENTINEL


def _validate_usd(value: Optional[str]) -> bool:
    """Return True iff value is a non-None string matching ^-?\\d+\\.\\d{2}$."""
    return value is not None and bool(_USD_RE.match(value))


@dataclass
class CreditsState:
    """Full credits state parsed from x-nous-credits-* response headers."""

    version: int = 0
    remaining_micros: int = 0
    remaining_usd: str = ""
    subscription_micros: int = 0  # SIGNED — may be negative (debt). ONLY field allowed negative.
    subscription_usd: str = ""
    subscription_limit_micros: Optional[int] = None  # PAIRED + OPTIONAL (only when subscription_cap)
    subscription_limit_usd: Optional[str] = None
    rollover_micros: int = 0
    purchased_micros: int = 0
    purchased_usd: str = ""
    tool_pool_micros: int = 0
    tool_pool_gated_off: bool = False
    denominator_kind: str = "none"  # "subscription_cap" | "none"
    paid_access: bool = True  # depletion keys off THIS == False, NEVER remaining==0
    disabled_reason: Optional[str] = None  # header omitted entirely when null
    as_of_ms: int = 0
    captured_at: float = 0.0  # time.time() when this was captured
    from_header: bool = False  # True only when populated by parse_credits_headers()

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")

    @property
    def depleted(self) -> bool:
        """Keyed off ``paid_access == False`` ONLY — never ``remaining_micros == 0``,
        a false positive when the balance is zero but access is live (renewal pending)."""
        return not self.paid_access

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction of the subscription cap consumed, in [0.0, 1.0]; None without a
        computable denominator. Guarded on the LIMIT FIELD (the real denominator),
        not ``denominator_kind`` (metadata)."""
        if not isinstance(self.subscription_limit_micros, int) or self.subscription_limit_micros <= 0:
            return None
        used = self.subscription_limit_micros - self.subscription_micros
        return max(0.0, min(1.0, used / self.subscription_limit_micros))


# ── Credits policy constants. Switching notices sticky→TTL later also needs a
# paired *_TTL_MS per notice kind (AgentNotice has the field; not plumbed yet).
CREDITS_NOTICE_KIND = "sticky"      # v1: credits notices are sticky
CREDITS_RESTORED_TTL_MS = 8000     # the only TTL notice in v1 (depletion-recovery confirmation)

# Usage-gauge bands (ascending): (threshold_fraction, level, label_pct). One
# escalating line showing the HIGHEST band reached (50 → 75 → 90); crossing up
# replaces it, recovering steps it down. The policy derives everything from it.
CREDITS_USAGE_BANDS: tuple[tuple[float, str, int], ...] = ((0.50, "info", 50), (0.75, "warn", 75), (0.90, "warn", 90))
CREDITS_USAGE_KEY = "credits.usage"  # single key for the escalating usage notice

# Min subscription balance counting as "grant not yet spent" for the grant_spent
# gate. 1¢: portal-seeded states (float dollars → micros) can carry sub-cent
# residue where headers report 0 — without the floor such a seed opens the gate
# and the first header re-creates the at-open nag.
GRANT_UNSPENT_MIN_MICROS = 10_000


def new_credits_latch() -> dict:
    """Fresh notice latch for :func:`evaluate_credits_notices`. Every producer must
    build it here so a new gate key lands everywhere at once."""
    return {"active": set(), "seen_below_90": False, "usage_band": None, "seen_grant_unspent": False}


@dataclass
class AgentNotice:
    """Driver-agnostic out-of-band notice, fired via ``AIAgent.notice_callback``
    (cleared via ``notice_clear_callback``); each driver renders its own way.
    ``kind``/``ttl_ms`` stay expressive so a future config can switch v1's
    sticky credits notices to TTL without touching the policy."""

    text: str
    level: str = "info"            # info | warn | error | success
    kind: str = "sticky"           # sticky | ttl
    ttl_ms: Optional[int] = None   # honored only when kind == "ttl"
    key: Optional[str] = None      # dedupe / fired-once-latch / clear key
    id: Optional[str] = None


def _sticky_notice(text: str, level: str, key: str) -> AgentNotice:
    return AgentNotice(text=text, level=level, kind=CREDITS_NOTICE_KIND, key=key, id=key)


def is_free_tier_model(model: str, base_url: str = "") -> bool:
    """Return True when *model* is a Nous free-tier model, using ONLY local data.

    Zero-network signals: (1) ``:free`` suffix — canonical Nous free SKU marker;
    (2) ``stealth/`` prefix — stealth-preview SKUs are free without the suffix
    (naming-convention trust: a PAID ``stealth/`` model would wrongly suppress
    the banner); (3) a PEEK into ``hermes_cli.models``' pricing cache (filled by
    the model picker; a miss never fetches — gateway sessions never run the
    picker, so there only 1-2 apply).

    Fail-open to False (depleted notice still shows): a wrong warning is
    recoverable noise; hiding it on a paid model masks a real block.
    """
    if not model:
        return False
    if model.endswith(":free") or model.startswith("stealth/"):
        return True
    if not base_url:
        return False
    try:
        from hermes_cli.models import _is_model_free, peek_cached_pricing

        # peek_cached_pricing owns the /v1-suffix and auth-state key details.
        pricing = peek_cached_pricing(base_url)
        if not pricing:
            return False
        return _is_model_free(model, pricing)
    except Exception:
        return False


def evaluate_credits_notices(
    state: CreditsState, latch: dict, *, model_is_free: bool = False,
) -> tuple[list[AgentNotice], list[str]]:
    """Reconcile credits notices against the latch (see :func:`new_credits_latch`).
    Mutates ``latch`` IN PLACE. Pure — no I/O, no agent/run_agent imports.

    ``model_is_free`` (see :func:`is_free_tier_model`) suppresses
    ``credits.depleted`` — a depleted account on a free model keeps inferencing,
    so the banner is noise. Suppression does NOT emit "restored"; that fires
    only on a genuine ``paid_access`` flip back to True.

    Returns ``(to_show, to_clear)``; caller emits to_clear FIRST, then to_show.
    """
    to_show: list[AgentNotice] = []
    to_clear: list[str] = []
    uf = state.used_fraction

    # Crossing latch: band notices fire only once uf was observed below the LOWEST
    # band, so a session opening mid-range doesn't fire on its first observation
    # (the cold-start seed primes this when it WANTS an open-high warning).
    if uf is not None and uf < CREDITS_USAGE_BANDS[0][0]:
        latch["seen_below_90"] = True

    # Grant-spent gate: fires only after this session OBSERVED the grant unspent
    # (≥1¢). Opening at grant-spent is a steady STATE (/usage carries it), not an
    # event. Unlike seen_below_90, seeds must NOT prime this gate.
    if uf is not None and uf < 1.0 and state.subscription_micros >= GRANT_UNSPENT_MIN_MICROS:
        latch["seen_grant_unspent"] = True
    active = latch["active"]

    # Highest band reached (ascending → last match wins); None below all.
    current_band: Optional[tuple[float, str, int]] = None
    if uf is not None:
        for band in CREDITS_USAGE_BANDS:
            if uf >= band[0]:
                current_band = band
    # Top-up suppression: with purchased credits the cap gauge is the wrong
    # denominator ("90% used" on $50 of top-up is noise; it used to stick
    # PERMANENTLY beside grant_spent at >=100%). grant_spent below covers the
    # cap-reached case; a mid-session top-up flips current_band → None and the
    # clear path removes the band line.
    if state.purchased_micros > 0:
        current_band = None
    grant_cond = (
        state.denominator_kind == "subscription_cap" and uf is not None and uf >= 1.0 and state.purchased_micros > 0
    )
    depleted_cond = not state.paid_access

    # ── usage gauge: highest crossed band only; replace on band change (climb or
    # step-down); clear below the lowest band or when the denominator vanishes.
    shown_band = latch.get("usage_band")
    target_band = current_band[2] if (current_band and latch["seen_below_90"]) else None
    if target_band != shown_band:
        if CREDITS_USAGE_KEY in active:
            to_clear.append(CREDITS_USAGE_KEY)
            active.discard(CREDITS_USAGE_KEY)
        if target_band is not None:
            # Absolute dollars used (a bare "N%" is only meaningful against a Nous
            # cap): cap − remaining in micros, clamped [0, cap]; "$?" if a producer
            # set the limit without its *_usd. Re-emits on band change only.
            _cap_usd = state.subscription_limit_usd or "?"
            _level = current_band[1]  # type: ignore[index]  (current_band set when target_band set)
            _lim = state.subscription_limit_micros or 0
            _used_micros = max(0, min(_lim, _lim - state.subscription_micros))
            _used_usd = f"{_used_micros / 1_000_000:.2f}" if _lim else "?"
            _glyph = "⚠" if _level == "warn" else "•"
            to_show.append(
                _sticky_notice(f"{_glyph} You've used ${_used_usd} of your ${_cap_usd} cap", _level, CREDITS_USAGE_KEY)
            )
            active.add(CREDITS_USAGE_KEY)
        latch["usage_band"] = target_band

    # ── grant_spent: the gate guards only the SHOW and is CONSUMED by it — one
    # announcement per crossing. A header flicker (uf → None → 1.0) clears the
    # line but cannot re-announce; only a renewal re-opening the gate (fresh ≥1¢
    # observation) arms the next. .get(): default closed for hand-built latches.
    if grant_cond and "credits.grant_spent" not in active and latch.get("seen_grant_unspent", False):
        to_show.append(
            _sticky_notice(f"• Grant spent · ${state.purchased_usd} top-up left", "info", "credits.grant_spent")
        )
        active.add("credits.grant_spent")
        latch["seen_grant_unspent"] = False
    elif "credits.grant_spent" in active and not grant_cond:
        to_clear.append("credits.grant_spent")
        active.discard("credits.grant_spent")

    # ── depleted: suppressed while the model is free (inference still works).
    show_depleted = depleted_cond and not model_is_free
    if show_depleted and "credits.depleted" not in active:
        to_show.append(_sticky_notice("✕ Credit access paused · run /topup to top up", "error", "credits.depleted"))
        active.add("credits.depleted")
    elif "credits.depleted" in active and not show_depleted:
        to_clear.append("credits.depleted")
        active.discard("credits.depleted")
        if not depleted_cond:
            # Genuine recovery only — switching to a free model while still
            # depleted must NOT claim access was restored.
            to_show.append(AgentNotice(
                text="✓ Credit access restored", level="success", kind="ttl",
                ttl_ms=CREDITS_RESTORED_TTL_MS, key="credits.restored", id="credits.restored",
            ))
    return (to_show, to_clear)


# (field, header, signed) — required micros fields; only subscription may be negative.
_MICROS_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("remaining_micros", "x-nous-credits-remaining-micros", False),
    ("subscription_micros", "x-nous-credits-subscription-micros", True),
    ("rollover_micros", "x-nous-credits-rollover-micros", False),
    ("purchased_micros", "x-nous-credits-purchased-micros", False), ("as_of_ms", "x-nous-credits-as-of-ms", False),
)
_USD_FIELDS: tuple[tuple[str, str], ...] = (
    ("remaining_usd", "x-nous-credits-remaining-usd"), ("subscription_usd", "x-nous-credits-subscription-usd"),
    ("purchased_usd", "x-nous-credits-purchased-usd"),
)
# (field, header, default-when-absent) — "true"/"false" (case-insensitive) STRING flags.
_BOOL_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("paid_access", "x-nous-credits-paid-access", True),  # absent → fail-open (assume access)
    ("tool_pool_gated_off", "x-nous-tool-pool-gated-off", False),
)


def parse_credits_headers(headers: Mapping[str, str], provider: str = "") -> Optional[CreditsState]:
    """Parse x-nous-credits-* (and x-nous-tool-pool-*) headers into a CreditsState.

    Returns None (miss) on ANY of: no ``x-nous-credits-version`` header; version
    != 1 (> 1 also warns once); a ``*_micros``/``as_of_ms`` field non-integer or
    negative (subscription excepted); a ``*_usd`` not matching
    ``^-?\\d+\\.\\d{2}$``; ``denominator_kind`` outside {"subscription_cap",
    "none"}; ``paid_access``/``tool_pool_gated_off`` not exactly "true"/"false";
    any unexpected exception.

    Fail-open on the subscription_limit pair: a half-pair (only -micros or only
    -usd) is treated as both-absent — the parse STILL SUCCEEDS with both None.
    """
    global _version_warning_emitted
    try:
        # Cheap probe before the lowercase copy (header names are case-insensitive):
        # bail when the version header is absent — the hot path for non-Nous providers.
        if not any(k.lower() == "x-nous-credits-version" for k in headers):
            return None
        lowered = {k.lower(): v for k, v in headers.items()}
        version_val = _safe_int(lowered.get("x-nous-credits-version"))
        if version_val is _SENTINEL:
            return None
        if version_val != 1:
            if version_val > 1 and not _version_warning_emitted:
                _version_warning_emitted = True
                logger.warning("credits header version %d unsupported, ignoring — update Hermes", version_val)
            return None
        fields: dict[str, Any] = {}
        for name, key, signed in _MICROS_FIELDS:
            val = _safe_int(lowered.get(key))
            if val is _SENTINEL or (not signed and val < 0):
                return None
            fields[name] = val

        # tool_pool_micros is OPTIONAL: absent → 0; present-but-invalid → miss.
        _tp_raw = lowered.get("x-nous-tool-pool-micros")
        _tp_val = 0 if _tp_raw is None else _safe_int(_tp_raw)
        if _tp_val is _SENTINEL or _tp_val < 0:
            return None
        fields["tool_pool_micros"] = _tp_val
        for name, key in _USD_FIELDS:
            val = lowered.get(key, "")
            if not _validate_usd(val):
                return None
            fields[name] = val

        # subscription_limit_* PAIRED + OPTIONAL: both present → validate both
        # (any invalid → miss); half-pair or both absent → both None, parse continues.
        sub_limit_micros_raw = lowered.get("x-nous-credits-subscription-limit-micros")
        sub_limit_usd_raw = lowered.get("x-nous-credits-subscription-limit-usd")
        if sub_limit_micros_raw is not None and sub_limit_usd_raw is not None:
            lm = _safe_int(sub_limit_micros_raw)
            if lm is _SENTINEL or lm < 0 or not _validate_usd(sub_limit_usd_raw):
                return None
            fields["subscription_limit_micros"] = lm
            fields["subscription_limit_usd"] = sub_limit_usd_raw
        denominator_kind = lowered.get("x-nous-credits-denominator-kind", "none")
        if denominator_kind not in _VALID_DENOMINATOR_KINDS:
            return None
        for name, key, default in _BOOL_FIELDS:
            if key not in lowered:
                fields[name] = default
                continue
            raw = lowered[key].strip().lower()
            if raw not in ("true", "false"):
                return None
            fields[name] = raw == "true"
        return CreditsState(
            version=version_val,
            denominator_kind=denominator_kind,
            disabled_reason=lowered.get("x-nous-credits-disabled-reason"),  # None if absent (omitted when null)
            captured_at=time.time(),
            from_header=True,
            **fields,
        )
    except Exception:
        # Fail-open → miss; breadcrumb distinguishes a parser regression from a
        # legitimate no-headers response.
        logger.debug("credits ▸ parse_credits_headers raised (fail-open miss)", exc_info=True)
        return None


# ── Dev fixtures (HERMES_DEV_CREDITS_FIXTURE): throwaway scaffolding to trigger
# any notice state without real spend. Value is a state NAME or a FILE PATH whose
# contents are a name (re-read every turn → `echo depleted > /tmp/cf` flips live).
# Drives the per-turn capture/notice path, the cold-start seed, and /usage.
def _fixture(remaining: str, subscription: str, limit: Optional[str] = None, purchased: Optional[str] = None,
             *, paid: bool = True, reason: Optional[str] = None) -> dict:
    """Fixture spec from *_usd strings; micros derived exactly (Decimal)."""
    from decimal import Decimal
    d: dict = {}
    for field, usd in (("remaining", remaining), ("subscription", subscription),
                       ("subscription_limit", limit), ("purchased", purchased)):
        if usd is not None:
            d[f"{field}_micros"], d[f"{field}_usd"] = int(Decimal(usd) * 1_000_000), usd
    if limit is not None:
        d["denominator_kind"] = "subscription_cap"
    d["paid_access"] = paid
    if reason is not None:
        d["disabled_reason"] = reason
    return d


_DEV_FIXTURES: dict[str, dict] = {
    "healthy": _fixture("30.34", "18.00", "20.00", "12.34"),  # used_fraction ~0.1, paid → no notice (recovery target)
    "sub_50pct": _fixture("10.00", "10.00", "20.00"),  # used_fraction == 0.5 → credits.usage band 50 (info)
    "sub_75pct": _fixture("5.00", "5.00", "20.00"),  # used_fraction == 0.75 → band 75 (warn)
    "sub_90pct": _fixture("2.00", "2.00", "20.00"),  # used_fraction == 0.9 → band 90 (warn)
    # uf == 1.0 + purchased>0 → SILENT at open (crossing-gated); flip healthy →
    # grant_exhausted via the fixture-file path to see credits.grant_spent
    "grant_exhausted": _fixture("12.34", "0.00", "20.00", "12.34"),
    "depleted": _fixture("0.00", "0.00", None, "0.00", paid=False, reason="out_of_credits"),  # → credits.depleted
    # subscription in debt (negative, the only signed field) → depleted
    "debt": _fixture("0.00", "-5.00", "20.00", "0.00", paid=False, reason="out_of_credits"),
}


def dev_fixture_credits_state() -> Optional[CreditsState]:
    """Return a fixture CreditsState for HERMES_DEV_CREDITS_FIXTURE, or None
    (unknown name / "clear" / "none" / unset → None).

    Hard prod-leak guard: applies ONLY when HERMES_DEV_CREDITS is also on, so a
    stray fixture env var can never surface fabricated balances on a real account.
    """
    if not is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        return None
    raw = os.environ.get("HERMES_DEV_CREDITS_FIXTURE", "").strip()
    if not raw:
        return None
    name = raw
    if os.path.sep in raw or "/" in raw:  # looks like a path → read the name from the file
        try:
            with open(raw, "r", encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            return None
    spec = _DEV_FIXTURES.get(name.lower())
    if not spec:
        return None
    # Stamp what the REAL parser always guarantees so a fixture is field-identical
    # to a parse_credits_headers() result (differential test): version 1, and a
    # valid purchased_usd (a zero-top-up account still carries "0.00").
    merged = {"version": 1, "purchased_usd": "0.00", **spec}
    return CreditsState(**merged, from_header=True, captured_at=time.time())


def _credits_state_from_account(info) -> Optional[CreditsState]:
    """Map a NousPortalAccountInfo into a header-shaped CreditsState for the seed.
    Float account dollars → micros plus a DISPLAY *_usd (formatting account floats
    is allowed; parsing a server *_usd is not). Fail-open → None."""
    try:
        _acc = getattr(info, "paid_service_access_info", None)
        _sub = getattr(info, "subscription", None)

        def _money(dollars) -> tuple[int, str]:  # (micros, display usd); (0, "") when absent
            if isinstance(dollars, (int, float)):
                return int(round(dollars * 1_000_000)), f"{dollars:.2f}"
            return 0, ""
        _remaining = _money(getattr(_acc, "total_usable_credits", None))
        _sub_rem = _money(getattr(_acc, "subscription_credits_remaining", None))
        _purchased = _money(getattr(_acc, "purchased_credits_remaining", None))
        _monthly = getattr(_sub, "monthly_credits", None)
        _cap = _money(_monthly) if isinstance(_monthly, (int, float)) and _monthly > 0 else (None, None)
        _paid = getattr(info, "paid_service_access", None)
        return CreditsState(
            remaining_micros=_remaining[0], remaining_usd=_remaining[1], subscription_micros=_sub_rem[0],
            subscription_usd=_sub_rem[1], subscription_limit_micros=_cap[0], subscription_limit_usd=_cap[1],
            purchased_micros=_purchased[0], purchased_usd=_purchased[1],
            rollover_micros=_money(getattr(_sub, "rollover_credits", None))[0],
            denominator_kind="subscription_cap" if _cap[0] is not None else "none",
            paid_access=_paid if isinstance(_paid, bool) else True, from_header=False, captured_at=time.time(),
        )
    except Exception:
        logger.debug("credits ▸ seed account→state mapping failed", exc_info=True)
        return None


def _hydrate_seed_state(agent, state) -> None:
    """Install a seed CreditsState on the agent and fire the notice policy once.
    Primes the crossing gate: the cold-start snapshot IS the first observation, so
    a session opening in a band warns immediately. Safe from a worker thread."""
    agent._credits_state = state
    if getattr(agent, "_credits_session_start_micros", None) is None:
        agent._credits_session_start_micros = state.remaining_micros
    _latch = getattr(agent, "_credits_latch", None)
    if isinstance(_latch, dict) and state.used_fraction is not None:
        # Prime ONLY seen_below_90. Never prime seen_grant_unspent: a seed
        # observing grant-spent is a steady state; priming revives the nag.
        _latch["seen_below_90"] = True
    emit = getattr(agent, "_emit_credits_notices", None)
    if callable(emit):
        emit()


def seed_credits_at_session_start(agent) -> bool:
    """Hydrate agent._credits_state from the portal account (or a dev fixture) and
    fire the notice policy so warnings show at session OPEN. Shared by the
    TUI/desktop build ("ready") and first-turn setup (plain-CLI fallback).
    Idempotent once a seed or real header populated _credits_state.

    Returns True iff it seeded this call. Never raises — credits must never block startup.
    """
    try:
        if getattr(agent, "provider", "") != "nous":
            return False
        if getattr(agent, "_credits_state", None) is not None:
            return False
        try:
            fixture = dev_fixture_credits_state()
        except Exception:
            fixture = None
        if fixture is not None:
            # Synchronous: a fixture is instant, and tests rely on the state +
            # notice landing before this returns.
            _hydrate_seed_state(agent, fixture)
            return True

        # FIRE-AND-FORGET: a slow portal must never delay "ready". The daemon
        # thread re-checks idempotency (a live header may land first).
        import threading

        def _bg_seed() -> None:
            try:
                from hermes_cli.nous_account import get_nous_portal_account_info
                info = get_nous_portal_account_info(force_fresh=True)
                if getattr(agent, "_credits_state", None) is not None:
                    return  # a live inference header beat us — don't clobber it
                state = _credits_state_from_account(info)
                if state is not None:
                    _hydrate_seed_state(agent, state)
            except Exception:
                logger.debug("credits ▸ session-start seed (background) failed", exc_info=True)
        threading.Thread(target=_bg_seed, name="credits-seed", daemon=True).start()
        return True
    except Exception:
        # Innermost log across all call sites so a dead seed is diagnosable.
        logger.debug("credits ▸ session-start seed failed (fail-open)", exc_info=True)
        return False
