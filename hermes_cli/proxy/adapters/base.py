"""Abstract base for proxy upstream adapters.

The proxy server is otherwise provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class UpstreamCredential:
    """A resolved bearer + base URL ready to forward to."""

    bearer: str
    """Authorization header value to send upstream (token only, no ``Bearer`` prefix)."""

    base_url: str
    """Upstream base URL, e.g. ``https://inference-api.nousresearch.com/v1``."""

    token_type: str = "Bearer"
    """Auth scheme — currently always ``Bearer`` for supported providers."""

    expires_at: Optional[str] = None
    """ISO-8601 expiry timestamp for the bearer, when known. Informational."""


class UpstreamAdapter(ABC):
    """Contract for an upstream provider the proxy can forward to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter key used on the CLI (e.g. ``"nous"``)."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name for logs and ``proxy status``."""

    @property
    @abstractmethod
    def allowed_paths(self) -> FrozenSet[str]:
        """Set of relative request paths the upstream accepts.

        Paths are relative to the proxy's ``/v1`` mount (``"/chat/completions"`` ⇒
        ``/v1/chat/completions``). Requests outside this set get a 404 with a helpful body.
        """

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Return True if the user has usable credentials for this upstream.

        Should be cheap — no network calls. Used by ``proxy start`` for a clear up-front error
        before binding a port.
        """

    @abstractmethod
    def get_credential(self) -> UpstreamCredential:
        """Return a fresh credential, refreshing or rotating if necessary.

        Implementations refresh a near-expiry access token, rotate a near-expiry upstream bearer key
        and persist refreshed state to disk. Raises RuntimeError when unauthenticated or refresh
        fails; the proxy then returns 401 to the client.
        """

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        """Return an alternate credential after an upstream auth failure.

        The default is no retry. Providers can override this for one-shot fallback paths after the
        upstream rejects the first request.
        """
        _ = failed_credential, status_code
        return None

    def describe(self) -> str:
        """One-line status summary for ``proxy status``."""
        try:
            cred = self.get_credential()
        except Exception as exc:  # pragma: no cover - defensive
            return f"{self.display_name}: not ready ({exc})"
        ttl = f" (expires {cred.expires_at})" if cred.expires_at else ""
        return f"{self.display_name}: {cred.base_url}{ttl}"


__all__ = ["UpstreamAdapter", "UpstreamCredential"]
