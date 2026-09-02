"""Single owner for backend identity and failure-scoped skip decisions.

Every fallback / dedup / skip / quarantine decision asks one question: **"is
this candidate the same backend as the one that failed, along the axis that
failure invalidated?"** Answering it inline at each call site (comparing
whatever string was locally convenient) repeatedly reintroduced the same bugs:
same-shim aliases treated as distinct, sibling models skipped for one model's
timeout, dedup ignoring ``base_url`` and stranding multi-endpoint pools.

"provider" conflates three independent identity axes, each invalidated by a
different failure class:

* **credential surface** — auth 401 / payment 402 kill everything sharing the
  credential (every model, every host reached with that key/token).
* **endpoint** — DNS failure / connection refused kill everything behind the
  URL, regardless of model or credential.
* **model deployment** — timeout / overload / rate limit / model-incompatible
  kill ONE model's deployment. A sibling model behind the same URL is an
  independent deployment (one model hung while another on the identical
  endpoint kept serving).

Call sites build :class:`BackendIdentity` values and ask
:func:`should_skip_candidate`. Do not re-implement any comparison inline —
extend THIS module instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FailureScope(Enum):
    """Which identity axis a failure invalidates."""

    #: Timeout, overload/429, connection blip, model-incompatible, invalid
    #: response: evidence against ONE model deployment only.
    MODEL = "model"
    #: Auth 401 / payment 402: evidence against the shared credential.
    CREDENTIAL = "credential"
    #: DNS / connection-refused / unreachable host: evidence against the endpoint.
    ENDPOINT = "endpoint"


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


@dataclass(frozen=True)
class BackendIdentity:
    """Normalized identity of one (provider, model, endpoint) deployment.

    Empty fields mean "unknown" — an unknown axis can neither prove sameness
    nor difference on its own; the remaining axes decide."""

    provider: str = ""
    model: str = ""
    base_url: str = ""

    @classmethod
    def build(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "BackendIdentity":
        return cls(
            provider=_norm(provider),
            model=_norm(model),
            base_url=_norm(base_url).rstrip("/"),
        )


def _both_first_class(a: BackendIdentity, b: BackendIdentity) -> bool:
    """True when both providers are distinct registered first-class providers.

    Two different registry providers have distinct credential surfaces even
    when they share an inference host (xai-oauth vs xai, openai-codex vs
    openai-api). Custom/shim aliases are NOT in the registry, so two aliases
    pointing at one URL still count as the same backend."""
    if not a.provider or not b.provider or a.provider == b.provider:
        return False
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        return a.provider in PROVIDER_REGISTRY and b.provider in PROVIDER_REGISTRY
    except Exception:
        return False


def same_credential_surface(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities share the credential a 401/402 just invalidated?

    Conservative on purpose: an unprovable axis answers "different" (one wasted
    RTT) rather than "same" (stranded failover). Same label = same configured
    credential; different labels = different credential config (custom entries
    can each carry their own api_key, so a shared URL alone never proves a
    shared credential — it is only a weak signal when a label is missing)."""
    if a.provider and b.provider:
        return a.provider == b.provider
    return bool(a.base_url and a.base_url == b.base_url)


def same_endpoint(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Do two identities sit behind the endpoint that just went unreachable?
    An unknown base_url inherits the provider default, so a shared provider
    label implies the same default endpoint."""
    if a.base_url and b.base_url:
        return a.base_url == b.base_url
    return bool(a.provider and a.provider == b.provider)


def same_deployment(a: BackendIdentity, b: BackendIdentity) -> bool:
    """Are these the exact same model deployment (the thing a timeout kills)?

    Provider+model must match; base_url distinguishes only when BOTH sides carry
    an explicit URL (same provider+model on two explicit URLs is a pool, not a
    dup). Different labels with the same URL + model are still one deployment
    (same-host shim aliases) — unless both labels are first-class registry
    providers."""
    if not (a.provider and b.provider and a.provider == b.provider):
        return bool(
            a.base_url
            and a.base_url == b.base_url
            and a.model
            and a.model == b.model
            and not _both_first_class(a, b)
        )
    if not (a.model and b.model and a.model == b.model):
        return False
    return not (a.base_url and b.base_url and a.base_url != b.base_url)


def should_skip_candidate(
    candidate: BackendIdentity,
    failed: BackendIdentity,
    scope: FailureScope = FailureScope.MODEL,
) -> bool:
    """THE skip predicate: would trying ``candidate`` just repeat the failure?
    True when it is the same backend as ``failed`` along the axis ``scope``
    invalidated. Every fallback/dedup/skip site must call this."""
    if scope is FailureScope.CREDENTIAL:
        return same_credential_surface(candidate, failed)
    if scope is FailureScope.ENDPOINT:
        return same_endpoint(candidate, failed)
    return same_deployment(candidate, failed)
