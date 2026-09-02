"""SelfHostedOIDCProvider — generic self-hosted OpenID Connect dashboard auth.

A standards-compliant OIDC Relying Party for the ``hermes dashboard`` gate.
Unlike the ``nous`` provider (Nous Portal's bespoke contract), this speaks
plain OIDC so it works against Authentik, Keycloak, Zitadel, Authelia, Auth0,
Okta, Google, … The HTTP round trip, cookies, CSRF ``state`` check and
``redirect_uri`` reconstruction are owned by ``hermes_cli/dashboard_auth/
routes.py``; this provider only:

  1. discovers endpoints from ``{issuer}/.well-known/openid-configuration``,
  2. builds the ``/authorize`` URL with PKCE (S256),
  3. exchanges the code at the discovered ``token_endpoint``,
  4. verifies the **ID token** against the discovered ``jwks_uri`` with
     ``iss``/``aud`` pinned, mapping ``sub``/``email``/``name`` onto a Session.

Why the ID token, not the access token: OIDC guarantees the ID token is a
signed JWT carrying identity claims; the access token's format is opaque per
spec (many IDPs issue random strings). The ``nous`` provider verifies its
access token only because Portal mints a custom JWT there.

Public (PKCE-only) and confidential (PKCE + ``client_secret``) clients are
both supported. With a secret, the client additionally authenticates at the
token endpoint via ``client_secret_basic`` or ``client_secret_post`` chosen
from ``token_endpoint_auth_methods_supported``. PKCE is sent in both modes —
the secret is layered on top, never a replacement (OAuth 2.1 / RFC 9700).

Configuration (env wins over config.yaml when set non-empty)::

    dashboard:
      oauth:
        provider: self-hosted
        self_hosted:
          issuer: https://auth.example.com/application/o/hermes/   # required
          client_id: hermes-dashboard                              # required
          scopes: "openid profile email"                           # optional
          # client_secret: confidential clients only — prefer the env var.

    HERMES_DASHBOARD_OIDC_ISSUER
    HERMES_DASHBOARD_OIDC_CLIENT_ID
    HERMES_DASHBOARD_OIDC_SCOPES         # optional
    HERMES_DASHBOARD_OIDC_CLIENT_SECRET  # optional; .env is its canonical home

On skip (missing issuer / client_id) the module-level :data:`LAST_SKIP_REASON`
carries a human-readable reason for the gate's fail-closed error.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import urllib.parse
from typing import Any, Dict, Optional

import httpx

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCodeError,
    LoginStart,
    ProviderError,
    RefreshExpiredError,
    Session,
)
from plugins.dashboard_auth._shared import (
    JSON_HEADERS,
    TOKEN_ENDPOINT_TIMEOUT_SEC as _TOKEN_ENDPOINT_TIMEOUT_SEC,
    exchange_token,
    load_config_section,
    make_jwks_client,
    parse_json_body,
    pkce_login_start,
    refresh_token_from,
    resolve_env_or_cfg,
    session_from_claims,
    validate_redirect_uri,
    verify_jwt,
)

logger = logging.getLogger(__name__)

# ``openid`` is mandatory (no ID token without it); profile/email populate
# display_name/email.
_DEFAULT_SCOPES = "openid profile email"

# RS256 is the OIDC default; ES256 is common on modern IDPs (Zitadel, newer
# Keycloak). HS256 is deliberately excluded: it implies a shared secret we
# don't hold in the public-client model and is a JWT algorithm-confusion footgun.
_ALLOWED_ID_TOKEN_ALGS = ("RS256", "ES256", "RS384", "RS512", "ES384", "ES512")

_DISCOVERY_TIMEOUT_SEC = 10.0
# Discovery is effectively static; a soft TTL lets a long-running dashboard
# pick up an IDP endpoint migration within the hour.
_DISCOVERY_CACHE_TTL_SEC = 3600

LAST_SKIP_REASON: str = ""


def _require_https_or_loopback(url: str, *, field: str) -> str:
    """Reject non-HTTPS endpoint URLs (loopback http allowed) so a misconfigured
    issuer can't ship auth codes / refresh tokens in cleartext."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" or (
        parsed.scheme == "http"
        and (parsed.hostname or "") in ("localhost", "127.0.0.1", "::1")
    ):
        return url
    raise ProviderError(
        f"OIDC {field} must be https:// (or http on localhost), got {url!r}"
    )


class SelfHostedOIDCProvider(DashboardAuthProvider):
    """Generic self-hosted OpenID Connect provider (authorization-code + PKCE)."""

    name = "self-hosted"
    display_name = "Self-Hosted OIDC"

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        scopes: str = _DEFAULT_SCOPES,
        client_secret: str = "",
    ) -> None:
        if not issuer:
            raise ValueError("issuer is required")
        if not client_id:
            raise ValueError("client_id is required")
        # Trailing slash normalised for stable compares; ``iss`` is pinned
        # against the *discovered* issuer so a config/IDP slash mismatch is tolerated.
        self._issuer = issuer.rstrip("/")
        _require_https_or_loopback(self._issuer, field="issuer")
        self._client_id = client_id
        self._scopes = scopes.strip() or _DEFAULT_SCOPES
        # Empty/whitespace secret ⇒ public client, so a provisioned-but-blank
        # secret can't flip us into a broken confidential mode.
        self._client_secret = (client_secret or "").strip()

        # Discovery + JWKS resolve lazily so registration never hits the
        # network (the IDP may be down at boot; fail per-request instead).
        self._discovery: Dict[str, Any] | None = None
        self._discovery_fetched_at: float = 0.0
        self._discovery_lock = threading.Lock()
        self._jwks_client: Any = None

    # ---- public API (DashboardAuthProvider) -------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        # Validate the redirect before discovery so a bad redirect_uri
        # surfaces even when the IDP is unreachable.
        validate_redirect_uri(redirect_uri)
        disco = self._get_discovery()
        return pkce_login_start(
            disco["authorization_endpoint"], client_id=self._client_id,
            scope=self._scopes, redirect_uri=redirect_uri,
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        # ``state`` is verified by the auth-route layer before this call.
        return self._exchange(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self._client_id,
                "code_verifier": code_verifier,
            },
            bad_request_exc=InvalidCodeError,
        )

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        return self._exchange(
            {
                "grant_type": "refresh_token",
                "client_id": self._client_id,
                "refresh_token": refresh_token,
                # Re-request the same scopes so the rotated ID token keeps its
                # identity claims (some IDPs narrow scope on refresh otherwise).
                "scope": self._scopes,
            },
            bad_request_exc=RefreshExpiredError,
            previous_refresh_token=refresh_token,
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # The session cookie carries the ID token in the access-token slot
        # (see _session_from_tokens) so this per-request check verifies a real
        # JWT. None on expiry/invalidity; ProviderError if IDP/JWKS unreachable.
        try:
            claims = self._verify_id_token(access_token)
        except InvalidCodeError:
            return None
        return self._session(access_token, "", claims)

    def revoke_session(self, *, refresh_token: str) -> None:
        # Best-effort RFC 7009 revocation when the IDP advertises an endpoint.
        # Must never raise — logout is client-side cookie clearing regardless.
        if not refresh_token:
            return None
        try:
            disco = self._get_discovery()
        except ProviderError:
            return None
        endpoint = str(disco.get("revocation_endpoint") or "").strip()
        if not endpoint:
            return None
        # Confidential clients must authenticate on revocation too (RFC 7009 §2.1).
        extra_data, extra_headers = self._token_endpoint_auth(disco)
        data = {"token": refresh_token, "token_type_hint": "refresh_token", "client_id": self._client_id, **extra_data}
        try:
            httpx.post(endpoint, data=data, headers={**JSON_HEADERS, **extra_headers}, timeout=_TOKEN_ENDPOINT_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("self-hosted OIDC: revoke failed (ignored): %s", exc)
        return None

    # ---- internals: token exchange ----------------------------------------

    def _token_endpoint_auth(self, disco: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, str]]:
        """Return ``(extra_data, extra_headers)`` for token-endpoint client auth.

        Public client → ``({}, {})`` (PKCE alone). Confidential client →
        ``client_secret_post`` when the IDP advertises it *without*
        ``client_secret_basic``, else HTTP Basic (the OIDC default and the
        fallback when nothing is advertised). RFC 6749 §2.3.1.
        """
        if not self._client_secret:
            return {}, {}
        methods = disco.get("token_endpoint_auth_methods_supported") or []
        if "client_secret_post" in methods and "client_secret_basic" not in methods:
            return {"client_secret": self._client_secret}, {}
        # Both halves must be form-url-encoded *before* base64 (RFC 6749
        # §2.3.1) or a secret containing ':' / reserved chars corrupts the header.
        userpass = f"{urllib.parse.quote(self._client_id, safe='')}:{urllib.parse.quote(self._client_secret, safe='')}"
        return {}, {"Authorization": f"Basic {base64.b64encode(userpass.encode('utf-8')).decode('ascii')}"}

    def _exchange(
        self,
        data: Dict[str, str],
        *,
        bad_request_exc: type[Exception],
        previous_refresh_token: str = "",
    ) -> Session:
        """POST the discovered token endpoint and turn the response into a Session.

        Confidential-client auth (body field or Basic header) is added here
        for both grants — the IDP rejects an unauthenticated refresh with
        ``invalid_client``. For a public client the request is PKCE-only.
        """
        disco = self._get_discovery()
        extra_data, extra_headers = self._token_endpoint_auth(disco)
        id_token, payload = exchange_token(
            disco["token_endpoint"],
            {**data, **extra_data},
            headers=extra_headers,
            bad_request_exc=bad_request_exc,
            idp="IDP",
            endpoint="OIDC token endpoint",
            token_key="id_token",
            missing_msg=(
                "OIDC token response missing id_token — ensure the 'openid' "
                "scope is configured and the client is allowed to receive an "
                "ID token."
            ),
        )
        claims = self._verify_id_token(id_token)
        # Prefer a freshly-issued RT, else keep the previous (some IDPs don't rotate).
        return self._session(id_token, refresh_token_from(payload, previous_refresh_token), claims)

    # ---- internals: discovery ---------------------------------------------

    def _fresh_discovery(self) -> Dict[str, Any] | None:
        if self._discovery is not None and time.time() - self._discovery_fetched_at < _DISCOVERY_CACHE_TTL_SEC:
            return self._discovery
        return None

    def _get_discovery(self) -> Dict[str, Any]:
        """Return the cached OIDC discovery document, fetching if stale (double-checked lock)."""
        disco = self._fresh_discovery()
        if disco is None:
            with self._discovery_lock:
                disco = self._fresh_discovery()
                if disco is None:
                    disco = self._discovery = self._fetch_discovery()
                    self._discovery_fetched_at = time.time()
                    # New issuer/keys → rebind the JWKS client to the fresh jwks_uri.
                    self._jwks_client = None
        return disco

    def _fetch_discovery(self) -> Dict[str, Any]:
        url = f"{self._issuer}/.well-known/openid-configuration"
        try:
            # follow_redirects=True: many IDPs answer discovery with a 3xx
            # (Authentik canonicalises .well-known; proxies upgrade http→https)
            # and httpx defaults to not following. Safe because the issuer pin
            # and HTTPS checks below validate the *resolved* document, so a
            # redirect to a hostile location can't smuggle in a bad issuer or a
            # cleartext endpoint. The token/revocation POSTs deliberately do
            # NOT follow redirects (they carry an auth code / refresh token).
            response = httpx.get(url, headers=JSON_HEADERS, timeout=_DISCOVERY_TIMEOUT_SEC, follow_redirects=True)
        except httpx.RequestError as exc:
            raise ProviderError(f"OIDC discovery unreachable: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"OIDC discovery returned {response.status_code} for {url!r}")
        payload = parse_json_body(response)
        if not payload:
            raise ProviderError("OIDC discovery returned a non-JSON body")

        def field(key: str) -> str:
            return str(payload.get(key, "") or "").strip()

        endpoints = {k: field(k) for k in ("authorization_endpoint", "token_endpoint", "jwks_uri")}
        if not all(endpoints.values()):
            raise ProviderError(
                "OIDC discovery missing one of authorization_endpoint / "
                "token_endpoint / jwks_uri"
            )

        # Issuer pin: a mismatch means the document came from the wrong place
        # (proxy/MITM/misconfig). Only a trailing-slash difference is tolerated.
        advertised_issuer = field("issuer")
        if advertised_issuer and advertised_issuer.rstrip("/") != self._issuer:
            raise ProviderError(
                f"OIDC discovery issuer mismatch: document advertises "
                f"{advertised_issuer!r} but configured issuer is "
                f"{self._issuer!r}"
            )
        for key, url in endpoints.items():
            _require_https_or_loopback(url, field=key)

        # Absent/garbage auth-methods → [] → OIDC default (basic) applies.
        auth_methods_raw = payload.get("token_endpoint_auth_methods_supported")
        return {
            "issuer": advertised_issuer or self._issuer,
            **endpoints,
            "revocation_endpoint": field("revocation_endpoint"),
            "token_endpoint_auth_methods_supported": (
                [str(m) for m in auth_methods_raw] if isinstance(auth_methods_raw, list) else []
            ),
        }

    # ---- internals: JWT verification + mapping ----------------------------

    def _get_jwks_client(self) -> Any:
        if self._jwks_client is None:
            self._jwks_client = make_jwks_client(self._get_discovery()["jwks_uri"])
        return self._jwks_client

    def _verify_id_token(self, id_token: str) -> Dict[str, Any]:
        issuer = self._get_discovery()["issuer"]
        return verify_jwt(
            id_token, self._get_jwks_client(), algorithms=list(_ALLOWED_ID_TOKEN_ALGS),
            audience=self._client_id, issuer=issuer, label="ID token",
        )

    def _session(self, id_token: str, refresh_token: str, claims: Dict[str, Any]) -> Session:
        """Map verified OIDC claims onto a Session.

        The verified ID token is stored in ``Session.access_token`` so the
        per-request ``verify_session`` re-verifies a real JWT; the opaque OAuth
        access token is not kept — the dashboard only needs identity.
        """
        email = str(claims.get("email", "") or "")
        # Org/tenant is non-standard: accept common spellings, else join
        # ``groups`` so multi-tenant IDPs surface *something* (free-form string).
        org_id = claims.get("org_id") or claims.get("organization") or ""
        groups = claims.get("groups")
        if not org_id and isinstance(groups, list) and groups:
            org_id = ",".join(str(g) for g in groups)
        return session_from_claims(
            self.name, claims, access_token=id_token, refresh_token=refresh_token,
            label="ID token", email=email,
            display_name=str(
                claims.get("name") or claims.get("preferred_username")
                or claims.get("nickname") or email or ""
            ),
            org_id=str(org_id or ""),
        )


# ---- Plugin entry point ----

def _load_config_oauth_section() -> dict:
    return load_config_section(logger, "dashboard-auth-self-hosted", "dashboard", "oauth", "self_hosted")


def register(ctx) -> None:
    """Register :class:`SelfHostedOIDCProvider` when issuer + client_id are set.

    On skip, :data:`LAST_SKIP_REASON` names BOTH configuration surfaces so
    operators don't guess wrong about which to set.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    oidc_cfg = _load_config_oauth_section()

    def setting(env_name: str, cfg_key: str) -> str:
        return resolve_env_or_cfg(env_name, oidc_cfg.get(cfg_key))

    issuer = setting("HERMES_DASHBOARD_OIDC_ISSUER", "issuer")
    client_id = setting("HERMES_DASHBOARD_OIDC_CLIENT_ID", "client_id")
    scopes = setting("HERMES_DASHBOARD_OIDC_SCOPES", "scopes") or _DEFAULT_SCOPES
    # Credential: canonical home is the env var / ~/.hermes/.env. Empty ⇒ public client.
    client_secret = setting("HERMES_DASHBOARD_OIDC_CLIENT_SECRET", "client_secret")

    if not issuer or not client_id:
        LAST_SKIP_REASON = (
            "Self-hosted OIDC dashboard auth is not configured. Set both an issuer and "
            "a client_id — either as env vars (HERMES_DASHBOARD_OIDC_ISSUER + "
            "HERMES_DASHBOARD_OIDC_CLIENT_ID) or under "
            "dashboard.oauth.self_hosted.{issuer,client_id} in config.yaml — or pass "
            "--insecure to skip the OAuth gate entirely. (issuer set: %s; client_id set: %s)"
            % (bool(issuer), bool(client_id))
        )
        logger.debug("dashboard-auth-self-hosted: %s", LAST_SKIP_REASON)
        return

    try:
        provider = SelfHostedOIDCProvider(
            issuer=issuer, client_id=client_id, scopes=scopes, client_secret=client_secret
        )
    except (ValueError, ProviderError) as exc:
        LAST_SKIP_REASON = f"SelfHostedOIDCProvider construction failed: {exc}"
        logger.warning("dashboard-auth-self-hosted: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-self-hosted: registered provider "
        "(issuer=%s, client_id=%s, scopes=%r, confidential=%s)",
        issuer, client_id, scopes, bool(client_secret),  # never log the secret itself
    )
