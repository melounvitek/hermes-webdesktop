"""
Email platform adapter for the Hermes gateway.

Allows users to interact with Hermes by sending emails.
Uses IMAP to receive and SMTP to send messages.

Environment variables:
    EMAIL_IMAP_HOST     — IMAP server host (e.g., imap.gmail.com)
    EMAIL_IMAP_PORT     — IMAP server port (default: 993)
    EMAIL_IMAP_SECURITY — IMAP transport: tls, starttls, or plain (default: tls)
    EMAIL_IMAP_TLS_VERIFY — Verify the IMAP TLS certificate (default: true)
    EMAIL_SMTP_HOST     — SMTP server host (e.g., smtp.gmail.com)
    EMAIL_SMTP_PORT     — SMTP server port (default: 587)
    EMAIL_SMTP_SECURITY — SMTP transport: tls, starttls, or plain (port-based default)
    EMAIL_SMTP_TLS_VERIFY — Verify the SMTP TLS certificate (default: true)
    EMAIL_ADDRESS       — Email address for the agent
    EMAIL_PASSWORD      — Email password or app-specific password
    EMAIL_POLL_INTERVAL — Seconds between mailbox checks (default: 15)
    EMAIL_ALLOWED_USERS — Comma-separated list of allowed sender addresses
"""

import asyncio
import email as email_lib
from contextlib import contextmanager
import imaplib
import logging
import os
import re
import smtplib
import socket
import ssl
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.config import Platform, PlatformConfig
from utils import is_truthy_value
from gateway.platforms._shared import get_scoped_secret as _get_esecret, coerce_port

logger = logging.getLogger(__name__)


# Backwards-compatible alias.
_get_secret = _get_esecret


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read (``env_int`` variant of ``_get_esecret``)."""
    return coerce_port(str(_get_esecret(name, "")).strip() or default, default)


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read (``env_bool`` variant of ``_get_esecret``)."""
    raw = str(_get_esecret(name, "")).strip()
    return is_truthy_value(raw, default=default) if raw else default


_SECURITY_ALIASES = {
    "tls": "tls", "ssl": "tls", "implicit": "tls",
    "starttls": "starttls",
    "plain": "plain", "none": "plain",
}


def _normalize_security(value: Any, default: str = "tls") -> str:
    """Map an IMAP/SMTP security setting to ``tls`` | ``starttls`` | ``plain``.

    Unknown values warn and fall back to *default* so a typo never silently
    downgrades to plaintext.
    """
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if not raw:
        return default
    mode = _SECURITY_ALIASES.get(raw)
    if mode is None:
        logger.warning("Unknown email security mode %r; using %r", value, default)
        return default
    return mode


def _tls_context(verify: bool, host: str) -> ssl.SSLContext:
    """Verified context by default; unverified only when explicitly opted out."""
    if verify:
        return ssl.create_default_context()
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("TLS verification disabled for non-loopback host %s", host)
    return ssl._create_unverified_context()


# Automated sender patterns — emails from these are silently ignored
_NOREPLY_PATTERNS = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounce", "notifications@",
    "automated@", "auto-confirm", "auto-reply", "automailer",
)

# RFC headers that indicate bulk/automated mail
_AUTOMATED_HEADERS = {
    "Auto-Submitted": lambda v: v.lower() != "no",
    "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}

# Gmail-safe max length per email body
MAX_MESSAGE_LENGTH = 50_000

SMTP_CONNECT_TIMEOUT = 30

_TRUTHY = {"true", "1", "yes"}


def _close_imap(imap: "imaplib.IMAP4") -> None:
    """Best-effort teardown that guarantees the socket is closed.

    ``IMAP4.logout()`` only guards ``OSError``; ``IMAP4.abort`` on a broken
    connection escapes before its ``shutdown()``, leaking one fd per failed poll
    (fatal on macOS's 256 soft limit). Chase it with an unconditional ``shutdown()``.
    """
    try:
        imap.logout()
    except Exception:
        try:
            imap.shutdown()
        except Exception:
            pass


def _create_ipv4_connection(host: str, port: int, timeout: float, source_address: Any = None) -> socket.socket:
    """``socket.create_connection`` constrained to ``AF_INET`` (no process-global
    socket mutation — email sends run in executor threads)."""
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(host, port, timeout, source_address=self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        raw_sock = _create_ipv4_connection(host, port, timeout, source_address=self.source_address)
        return self.context.wrap_socket(raw_sock, server_hostname=getattr(self, "_host", host))


def _open_smtp(host: str, port: int, security: str, ctx: ssl.SSLContext,
               smtp_cls: type, smtp_ssl_cls: type, **kwargs: Any) -> smtplib.SMTP:
    """Open one SMTP connection with TLS established per *security*; *kwargs* go to the constructor."""
    if security == "tls":
        return smtp_ssl_cls(host, port, context=ctx, **kwargs)
    smtp = smtp_cls(host, port, **kwargs)
    if security == "starttls":
        try:
            smtp.starttls(context=ctx)
        except Exception:
            smtp.close()
            raise
    return smtp


# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _send_imap_id(imap: "imaplib.IMAP4") -> None:
    """Send RFC 2971 IMAP ID. 163/NetEase require it after LOGIN (else every UID
    command returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed."""
    try:
        try:
            from hermes_cli import __version__ as _hermes_version
        except Exception:  # noqa: BLE001 — keep ID best-effort if import fails
            _hermes_version = "0"
        imap.xatom(
            "ID",
            f'("name" "hermes-agent" "version" "{_hermes_version}" '
            '"vendor" "NousResearch" '
            '"support-email" "noreply@nousresearch.com")',
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """Return True if this email is from an automated/noreply source."""
    addr = address.lower()
    if any(pattern in addr for pattern in _NOREPLY_PATTERNS):
        return True
    for header, check in _AUTOMATED_HEADERS.items():
        value = headers.get(header, "")
        if value and check(value):
            return True
    return False


def check_email_requirements() -> bool:
    """True when all email settings are present and non-blank (blank ``EMAIL_*``
    keys left by an abandoned setup must not enable the platform)."""
    return all(
        _get_secret(name, "").strip()
        for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST")
    )


_CHARSET_ALIASES = {
    # Aliases seen in the wild that Python's codec registry doesn't know.
    # "unknown-8bit" / "x-unknown" are RFC 1428 placeholders some MTAs (QQ
    # Mail among them) emit when the original charset was lost.
    "unknown-8bit": "utf-8",
    "unknown": "utf-8",
    "x-unknown": "utf-8",
    "default": "utf-8",
    "ansi_x3.110-1983": "latin-1",
    "cp-850": "cp850",
    "gb2312": "gb18030",  # superset; avoids failures on GBK extensions
    "gbk": "gb18030",
    "ks_c_5601-1987": "cp949",
}


def _safe_decode(payload: bytes, charset: "Optional[str]") -> str:
    """Decode *payload* without ever raising: ``errors="replace"`` does not guard a
    missing codec (``LookupError``), so fall back via alias table → UTF-8 → latin-1."""
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    label = _CHARSET_ALIASES.get(label, label)
    for candidate in (label, "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 header into a plain string; never raises."""
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    return " ".join(
        _safe_decode(part, charset) if isinstance(part, bytes) else part
        for part, charset in parts
    )


def _first_body_part(msg: email_lib.message.Message, content_type: str) -> str:
    """Decoded text of the first non-attachment part of *content_type*, or ''."""
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")):
            continue
        if part.get_content_type() == content_type:
            payload = part.get_payload(decode=True)
            if payload:
                return _safe_decode(payload, part.get_content_charset())
    return ""


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        text = _first_body_part(msg, "text/plain")
        if text:
            return text
        html = _first_body_part(msg, "text/html")
        return _strip_html(html) if html else ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return ""
    text = _safe_decode(payload, msg.get_content_charset())
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


# Ordered (pattern, replacement) substitutions for _strip_html.
_HTML_SUBS = (
    (re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"), (re.compile(r"<p[^>]*>", re.IGNORECASE), "\n"),
    (re.compile(r"</p>", re.IGNORECASE), "\n"), (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"&nbsp;"), " "), (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"), (re.compile(r"\n{3,}"), "\n\n"),
)


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    for pattern, repl in _HTML_SUBS:
        html = pattern.sub(repl, html)
    return html.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    return (match.group(1) if match else raw).strip().lower()


def _domain_of(address: str) -> str:
    """Return the lowercased domain part of an email address, or ''."""
    _, _, domain = address.rpartition("@")
    return domain.strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Relaxed DMARC alignment: equal, or one is a dot-suffix of the other."""
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


# "method=result" tokens (``dmarc=pass``) and property values
# (``header.from=example.com``) in an Authentication-Results header.
_AUTH_METHOD_RE = re.compile(r"\b(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
_AUTH_PROP_RE = re.compile(
    r"\b(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*([^\s;]+)",
    re.IGNORECASE,
)


def _verify_sender_authentication(msg: email_lib.message.Message, from_addr: str, *,
                                  authserv_id: str = "") -> Tuple[bool, str]:
    """Verify that the message's ``From:`` domain is authenticated.

    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy
    signal is the ``Authentication-Results`` header stamped by the *receiving*
    server. It prepends, so the FIRST instance is trusted and any injected copy
    sorts below it. Pinned to *authserv_id* when given.

    Returns ``(authenticated, reason)``; True on a DMARC pass, an aligned SPF
    pass, or an aligned DKIM (``header.d``) pass. No header → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``).
    """
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return False, "no Authentication-Results header"
    trusted = None
    for raw in headers:
        value = " ".join(str(raw).split())
        if authserv_id:
            # authserv-id is the first token before the first ';'
            serv = value.split(";", 1)[0].strip().lower()
            if not _domains_aligned(serv, authserv_id) and serv != authserv_id.lower():
                continue
        trusted = value
        break
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"

    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}
    # DMARC already enforces From alignment, so a pass is sufficient.
    if methods.get("dmarc") == "pass":
        return True, "dmarc=pass"
    # SPF pass: envelope/MAIL FROM domain must align with From.
    if methods.get("spf") == "pass":
        spf_domain = (_domain_of(props.get("smtp.mailfrom", "")) or props.get("smtp.from", "")
                      or props.get("envelope-from", ""))
        spf_domain = _domain_of(spf_domain) if "@" in spf_domain else spf_domain
        if _domains_aligned(spf_domain, from_domain):
            return True, "spf=pass aligned"
    # DKIM pass: signing domain header.d must align with From.
    if methods.get("dkim") == "pass":
        dkim_domain = props.get("header.d", "") or _domain_of(props.get("header.from", ""))
        if _domains_aligned(dkim_domain, from_domain):
            return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(msg: email_lib.message.Message, skip_attachments: bool = False) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally.

    When *skip_attachments* is True, all attachment/inline parts are ignored.
    """
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", ""))
        if skip_attachments or ("attachment" not in disposition and "inline" not in disposition):
            continue
        # Skip text/plain and text/html body parts
        content_type = part.get_content_type()
        if content_type in {"text/plain", "text/html"} and "attachment" not in disposition:
            continue
        filename = part.get_filename()
        filename = _decode_header_value(filename) if filename else f"attachment.{part.get_content_subtype() or 'bin'}"
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        ext = Path(filename).suffix.lower()
        if ext in _IMAGE_EXTS:
            try:
                cached_path = cache_image_from_bytes(payload, ext)
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
            kind = "image"
        else:
            cached_path = cache_document_from_bytes(payload, filename)
            kind = "document"
        attachments.append({"path": cached_path, "filename": filename, "type": kind, "media_type": content_type})

    return attachments


def _attach_file(msg: MIMEMultipart, path: Path, filename: str) -> None:
    """Attach *path* to *msg* as base64 application/octet-stream."""
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    # Per-account seen-UID snapshot surviving adapter recreation: the reconnect
    # watcher builds a FRESH adapter per retry, and without this
    # connect(is_reconnect=True) would re-mark the mailbox seen and skip mail
    # that arrived during the outage. Keyed by address (multiplex gateways run
    # several accounts). Same-process only — a full restart re-baselines.
    _seen_uids_snapshot: Dict[str, set] = {}

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        # Env vars first, then PlatformConfig.extra so a config.yaml-only setup
        # works. Host/address are stripped: a stray newline made IMAP4_SSL raise
        # ``[Errno 8] nodename nor servname`` instead of "host not set".
        extra = config.extra or {}

        def setting(env: str, key: str) -> str:
            return _get_secret(env, "") or extra.get(key, "")

        def tls_verify(env: str, key: str) -> bool:
            return _esecret_bool(env, is_truthy_value(extra.get(key), default=True))

        self._address = setting("EMAIL_ADDRESS", "address").strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = setting("EMAIL_IMAP_HOST", "imap_host").strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        self._imap_security = _normalize_security(setting("EMAIL_IMAP_SECURITY", "imap_security"))
        self._imap_tls_verify = tls_verify("EMAIL_IMAP_TLS_VERIFY", "imap_tls_verify")
        self._smtp_host = setting("EMAIL_SMTP_HOST", "smtp_host").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._smtp_security = _normalize_security(setting("EMAIL_SMTP_SECURITY", "smtp_security"),
                                                  default="tls" if self._smtp_port == 465 else "starttls")
        self._smtp_tls_verify = tls_verify("EMAIL_SMTP_TLS_VERIFY", "smtp_tls_verify")
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)

        # config.yaml: platforms.email.skip_attachments: true
        self._skip_attachments = extra.get("skip_attachments", False)

        # Require an authenticated From: domain (SPF/DKIM/DMARC) before trusting
        # it for authorization (GHSA-rxqh-5572-8m77). Default ON; opt out via
        # platforms.email.require_authenticated_sender: false or EMAIL_TRUST_FROM_HEADER=true.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        elif _esecret_bool("EMAIL_TRUST_FROM_HEADER", False):
            self._require_authenticated_sender = False
        else:
            self._require_authenticated_sender = True

        # Optional authserv-id pinning Authentication-Results to the operator's
        # own receiving server (defends against an injected header sorting first).
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()

        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        # Distinguish "checked, nothing new" from "the check itself failed".
        self._last_fetch_failed: bool = False
        self._last_fetch_error: str = ""
        # Map chat_id (sender email) -> last subject + message-id for threading
        self._thread_context: Dict[str, Dict[str, str]] = {}
        logger.info("[Email] Adapter initialized for %s", self._address)

    def _trim_seen_uids(self) -> None:
        """Keep only the highest half of UIDs once over the cap (UIDs are monotonic; UNSEEN prevents re-delivery)."""
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            # UIDs are bytes like b'1234' — sort numerically and keep top half
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))
            keep = self._seen_uids_max // 2
            self._seen_uids = set(sorted_uids[-keep:])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            # Fallback: just clear old entries if sort fails
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2:])

    def _connect_imap(self) -> imaplib.IMAP4:
        """Create an IMAP connection using implicit TLS, STARTTLS, or plaintext."""
        if self._imap_security == "tls":
            return imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30,
                                     ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))

        imap = imaplib.IMAP4(self._imap_host, self._imap_port, timeout=30)
        if self._imap_security == "starttls":
            try:
                imap.starttls(ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
            except Exception:
                _close_imap(imap)
                raise
        return imap

    @contextmanager
    def _inbox(self):
        """Logged-in IMAP handle on INBOX; always ``_close_imap``-ed on exit (a
        login/select failure used to leak one fd per reconnect attempt)."""
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            _close_imap(imap)

    def _connect_smtp(self) -> smtplib.SMTP:
        """SMTP connection with TLS established (callers go straight to ``login()``).

        An unreachable IPv6 address can hang until the socket timeout, so
        connection-level failures retry through an IPv4-only socket path (no
        global resolver mutation). TLS verification errors are not retried.
        """
        host, port, security = self._smtp_host, self._smtp_port, self._smtp_security
        ctx = _tls_context(self._smtp_tls_verify, host)
        try:
            return _open_smtp(host, port, security, ctx, smtplib.SMTP, smtplib.SMTP_SSL,
                              timeout=SMTP_CONNECT_TIMEOUT)
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            # Connection-level failure (may be unreachable IPv6): retry IPv4 only.
            return _open_smtp(host, port, security, ctx, _IPv4SMTP, _IPv4SMTP_SSL,
                              timeout=SMTP_CONNECT_TIMEOUT)

    def _fail(self, log_fmt: str, err: object, code: str, detail: str, *, retryable: bool) -> bool:
        """Log *err*, record a fatal error for the gateway's reconnect machinery, return False."""
        logger.error(log_fmt, err)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    def _probe_imap(self, is_reconnect: bool) -> bool:
        """Connection test + seen-UID baseline. Sets a fatal error and returns False on failure."""
        try:
            with self._inbox() as imap:
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # Same-process reconnect: restore the previous adapter's
                    # baseline so mail that arrived during the outage stays
                    # eligible for the next poll instead of being skipped.
                    self._seen_uids = set(snapshot)
                    self._trim_seen_uids()
                    logger.info(
                        "[Email] IMAP reconnect test passed. Restored %d seen UIDs; "
                        "messages received during the outage will be processed.",
                        len(self._seen_uids),
                    )
                else:
                    # First connect (or no snapshot): mark all existing messages seen.
                    status, data = imap.uid("search", None, "ALL")
                    if status == "OK" and data and data[0]:
                        self._seen_uids.update(data[0].split())
                    self._trim_seen_uids()
                    logger.info("[Email] IMAP connection test passed. %d existing messages skipped.", len(self._seen_uids))
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
            return True
        except Exception as e:
            # Always set an explicit fatal code, else the gateway treats every
            # failure as transient with zero owner signal. retryable=True because
            # imaplib raises the same generic IMAP4.error for bad credentials AND
            # transient NOs (Gmail "too many simultaneous connections"); long-lived
            # loops surface via the reconnect watcher's NEEDS_ATTENTION escalation.
            return self._fail("[Email] IMAP connection failed: %s", e, "email_imap_connect_error",
                              f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}", retryable=True)

    def _probe_smtp(self) -> bool:
        """SMTP connect + login test. Sets a fatal error and returns False on failure."""
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # Typed auth failure (535 & friends) can never self-heal, so drop out
            # of the reconnect queue — unambiguous, unlike IMAP4.error above.
            return self._fail(
                "[Email] SMTP authentication failed: %s", e, "email_auth_error",
                f"SMTP authentication failed for {self._address}: {e}. "
                "Check EMAIL_PASSWORD (for Gmail/Outlook this must be an "
                "app password, not the account password).",
                retryable=False,
            )
        except Exception as e:
            return self._fail("[Email] SMTP connection failed: %s", e, "email_smtp_connect_error",
                              f"SMTP connection to {self._smtp_host} failed: {e}", retryable=True)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host is an actionable config error, not
        # IMAP4_SSL("") raising ``[Errno 8] nodename nor servname provided``.
        required = (("EMAIL_ADDRESS", self._address), ("EMAIL_PASSWORD", self._password),
                    ("EMAIL_IMAP_HOST", self._imap_host), ("EMAIL_SMTP_HOST", self._smtp_host))
        missing = [name for name, value in required if not value]
        if missing:
            message = ("Not configured — missing " + ", ".join(missing)
                       + ". Set it via `hermes gateway setup` (env) or platforms.email in config.yaml.")
            # Non-retryable: a blank-but-present env var (``EMAIL_IMAP_HOST=``)
            # used to drive an indefinite retry loop that leaked until OOM.
            return self._fail("[Email] %s", message, "email_missing_configuration", message, retryable=False)
        if not self._probe_imap(is_reconnect) or not self._probe_smtp():
            return False
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("[Email] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages at regular intervals."""
        while self._running:
            try:
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, self._fetch_new_messages)
        # Dispatch partial results BEFORE escalating a failure — a mid-batch
        # exception returns what was fetched, and those are already marked seen.
        for msg_data in messages:
            await self._dispatch_message(msg_data)
        if self._last_fetch_failed:
            # The IMAP check itself failed (not an empty inbox): route through the
            # fatal-error hook so the gateway's reconnect/backoff re-establishes the
            # mailbox. The handler runs in a detached task (gateway/run.py), so
            # awaiting it from our own poll task is safe despite teardown cancelling us.
            self._last_fetch_failed = False
            self._set_fatal_error("email_imap_fetch_failed", self._last_fetch_error or "IMAP fetch failed", retryable=True)
            await self._notify_fatal_error()

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        try:
            with self._inbox() as imap:
                status, data = imap.uid("search", None, "UNSEEN")
                uids = data[0].split() if status == "OK" and data and data[0] else []
                for uid in uids:
                    if uid in self._seen_uids:
                        continue
                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        # Transient per-UID refusal: leave UID unseen so the next poll retries.
                        continue
                    # Mark seen once a response arrived (even malformed) so garbage
                    # is skipped once, not retried forever — but NOT before the
                    # fetch: a connection failure must leave the rest of the batch
                    # eligible for the next poll.
                    self._seen_uids.add(uid)
                    if len(self._seen_uids) > self._seen_uids_max:
                        self._trim_seen_uids()
                    try:
                        raw_email = msg_data[0][1]
                    except (IndexError, TypeError):
                        logger.warning("[Email] Unexpected IMAP response structure for UID %s, skipping", uid)
                        continue
                    if not isinstance(raw_email, (bytes, bytearray)):
                        logger.warning("[Email] Non-bytes IMAP payload for UID %s, skipping", uid)
                        continue
                    # One poison message (unparseable headers, pathological
                    # attachment, DNS hiccup in SPF/DKIM) must not abort the batch
                    # or escalate to a reconnect — it is already marked seen.
                    try:
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error("[Email] Failed to process message UID %s, skipping: %s", uid, parse_exc)
                        continue
                    if parsed is not None:
                        results.append(parsed)
        except Exception as e:
            logger.error("[Email] IMAP fetch error: %s", e)
            self._last_fetch_failed = True
            self._last_fetch_error = str(e)
        # Keep the reconnect snapshot current so a mid-outage adapter recreation
        # does not re-dispatch messages this instance already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _parse_fetched_message(self, uid: bytes, raw_email: "bytes | bytearray") -> Optional[Dict[str, Any]]:
        """Parse one fetched RFC822 payload into a dispatchable dict.

        Returns ``None`` for automated/noreply senders. Raises on pathological
        input — the caller's per-message guard logs the UID and continues.
        """
        msg = email_lib.message_from_bytes(raw_email)

        sender_raw = msg.get("From", "")
        sender_addr = _extract_email_address(sender_raw)
        sender_name = _decode_header_value(sender_raw)
        # Remove email from name if present
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')

        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        # Skip automated/noreply senders before any processing
        if _is_automated_sender(sender_addr, dict(msg.items())):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None

        # Verify From: authentication while the raw message (and its trusted
        # Authentication-Results header) is in scope; the verdict is consumed
        # at dispatch where authorization is decided (GHSA-rxqh-5572-8m77).
        sender_authenticated, auth_reason = _verify_sender_authentication(msg, sender_addr, authserv_id=self._authserv_id)
        return {
            "uid": uid,
            "sender_addr": sender_addr,
            "sender_name": sender_name,
            "subject": subject,
            "message_id": msg.get("Message-ID", ""),
            "in_reply_to": msg.get("In-Reply-To", ""),
            "body": _extract_text_body(msg),
            "attachments": _extract_attachments(msg, skip_attachments=self._skip_attachments),
            "date": msg.get("Date", ""),
            "sender_authenticated": sender_authenticated,
            "auth_reason": auth_reason,
        }

    @staticmethod
    def _allow_all_senders() -> bool:
        """True when the operator opted into any sender (EMAIL_ or GATEWAY_ALLOW_ALL_USERS)."""
        return (_get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY
                or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY)

    @staticmethod
    def _allowlist_in_effect() -> bool:
        """True when EMAIL_ALLOWED_USERS or GATEWAY_ALLOWED_USERS gates access.

        Without one the gateway default-denies every sender, so the spoofable
        From: identity grants nothing and the authentication gate is moot.
        """
        return bool(_get_secret("EMAIL_ALLOWED_USERS", "").strip() or os.getenv("GATEWAY_ALLOWED_USERS", "").strip())

    def _sender_accepted(self, sender_addr: str, msg_data: Dict[str, Any]) -> bool:
        """Pre-dispatch sender gate: self, automated, allowlist, From: authentication."""
        if sender_addr == self._address.lower():
            return False
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return False

        # Drop senders the gateway would never authorize before a MessageEvent
        # (and thread context) exists — otherwise a race between dispatch and
        # authorization can send a reply even though the handler returned None.
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        if not allowed_raw:
            if not self._allow_all_senders():
                logger.debug("[Email] Dropping sender at dispatch — EMAIL_ALLOWED_USERS is unset "
                             "and open access is not opted in: %s", sender_addr)
                return False
        elif sender_addr.lower() not in {a.strip().lower() for a in allowed_raw.split(",") if a.strip()}:
            logger.debug("[Email] Dropping non-allowlisted sender at dispatch: %s", sender_addr)
            return False

        # Reject spoofed senders (GHSA-rxqh-5572-8m77): the allowlist keys on the
        # attacker-controlled From:. Only matters when an allowlist GRANTS access
        # and allow-all is off; fail-closed before matching against the allowlist.
        if (self._require_authenticated_sender and self._allowlist_in_effect()
                and not self._allow_all_senders() and not msg_data.get("sender_authenticated", False)):
            logger.warning(
                "[Email] Dropping sender with unauthenticated From: %s (%s). "
                "If your mail server does not stamp Authentication-Results, set "
                "platforms.email.require_authenticated_sender: false (or "
                "EMAIL_TRUST_FROM_HEADER=true) to accept the risk.",
                sender_addr, msg_data.get("auth_reason", "no verdict"),
            )
            return False
        return True

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        if not self._sender_accepted(sender_addr, msg_data):
            return

        subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]
        # Include subject as context unless it is a reply
        text = f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body

        msg_type = MessageType.TEXT
        for att in attachments:
            if att["type"] == "image" and msg_type == MessageType.TEXT:
                msg_type = MessageType.PHOTO
            elif att["type"] == "document":
                # Document wins over PHOTO for mixed attachments: run.py keys
                # image handling off the per-path mime type regardless of
                # message_type, but document-context injection gates strictly
                # on MessageType.DOCUMENT — so DOCUMENT surfaces both.
                msg_type = MessageType.DOCUMENT

        # Store thread context for reply threading
        self._thread_context[sender_addr] = {"subject": subject, "message_id": msg_data["message_id"]}
        name = msg_data["sender_name"] or sender_addr
        event = MessageEvent(
            text=text or "(empty email)",
            message_type=msg_type,
            source=self.build_source(chat_id=sender_addr, chat_name=name, chat_type="dm",
                                     user_id=sender_addr, user_name=name),
            message_id=msg_data["message_id"],
            media_urls=[att["path"] for att in attachments],
            media_types=[att["media_type"] for att in attachments],
            reply_to_message_id=msg_data["in_reply_to"] or None,
        )
        logger.info("[Email] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking SMTP sender in the executor; wrap its Message-ID in a SendResult."""
        try:
            message_id = await asyncio.get_running_loop().run_in_executor(None, fn, *args)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an email reply to the given address."""
        return await self._run_send(self._send_email, (chat_id, content, reply_to),
                                    "[Email] Send failed to %s: %s", chat_id)

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None,
                   *, attach_empty_body: bool = False) -> Tuple[MIMEMultipart, str, str]:
        """Build a threaded reply skeleton. Returns ``(msg, msg_id, subject)``."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        if body or attach_empty_body:
            msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg, msg_id, subject

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Login, send, and always release the SMTP connection (quit, else close)."""
        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, attach_empty_body=True)
        self._smtp_send(msg)
        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image URL as part of an email body (``metadata`` unused)."""
        text = caption or ""
        text += f"\n\nImage: {image_url}"
        return await self.send(chat_id, text.strip(), reply_to)

    async def send_multiple_images(self, chat_id: str, images: List[Tuple[str, str]],
                                   metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """Send a batch of images as one email with multiple MIME attachments.

        Local files are attached; URL images are linked in the body (no remote
        download). No hard cap beyond SMTP message size limits.
        """
        if not images:
            return

        from urllib.parse import unquote as _unquote

        body_parts: List[str] = []
        local_paths: List[str] = []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if image_url.startswith("file://"):
                local_path = _unquote(image_url[7:])
                if Path(local_path).exists():
                    local_paths.append(local_path)
                else:
                    logger.warning("[Email] Skipping missing image: %s", local_path)
            else:
                # Remote URLs just get linked in the body (parity with send_image)
                body_parts.append(f"Image: {image_url}")
        if not local_paths and not body_parts:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._send_email_with_attachments, chat_id, "\n\n".join(body_parts), local_paths)
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: List[str]) -> str:
        """Send an email with multiple file attachments via SMTP (unattachable files are skipped)."""
        msg, msg_id, _ = self._new_reply(to_addr, body)
        for file_path in file_paths:
            p = Path(file_path)
            try:
                _attach_file(msg, p, p.name)
            except Exception as e:
                logger.warning("[Email] Failed to attach %s: %s", file_path, e)
        self._smtp_send(msg)
        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        return msg_id

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        """Send a file as an email attachment."""
        return await self._run_send(self._send_email_with_attachment, (chat_id, caption or "", file_path, file_name),
                                    "[Email] Send document failed: %s")

    def _send_email_with_attachment(self, to_addr: str, body: str, file_path: str,
                                    file_name: Optional[str] = None) -> str:
        """Send an email with a file attachment via SMTP."""
        msg, msg_id, _ = self._new_reply(to_addr, body)
        p = Path(file_path)
        _attach_file(msg, p, file_name or p.name)
        self._smtp_send(msg)
        return msg_id

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        ctx = self._thread_context.get(chat_id, {})
        return {"name": chat_id, "type": "dm", "chat_id": chat_id, "subject": ctx.get("subject", "")}


# Plugin glue: register() exposes the platform via the registry (replacing the
# former Platform.EMAIL branches in gateway/run.py, gateway/config.py,
# hermes_cli/gateway.py and tools/send_message_tool.py). EMAIL_* env →
# PlatformConfig seeding stays in core.


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery via SMTP (one-shot); standalone_sender_fn contract."""
    extra = getattr(pconfig, "extra", {}) or {}
    address = extra.get("address") or _get_secret("EMAIL_ADDRESS", "")
    password = _get_secret("EMAIL_PASSWORD", "")
    smtp_host = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", "")
    smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
    smtp_security = _normalize_security(
        _get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"),
        default="tls" if smtp_port == 465 else "starttls",
    )
    smtp_tls_verify = _esecret_bool(
        "EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True)
    )

    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}

    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["From"] = address
        msg["To"] = chat_id
        msg["Subject"] = "Hermes Agent"
        msg["Date"] = formatdate(localtime=True)

        ctx = _tls_context(smtp_tls_verify, smtp_host)
        server = _open_smtp(smtp_host, smtp_port, smtp_security, ctx, smtplib.SMTP, smtplib.SMTP_SSL)
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e
            return _e(f"Email send failed: {e}")
        except Exception:
            return {"error": f"Email send failed: {e}"}


def _is_connected(config) -> bool:
    """Connected when an address is configured (PlatformConfig.extra or EMAIL_ADDRESS)."""
    extra = getattr(config, "extra", {}) or {}
    if extra.get("address"):
        return True
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs EmailAdapter from a PlatformConfig."""
    return EmailAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="email",
        label="Email",
        adapter_factory=_build_adapter,
        check_fn=check_email_requirements,
        is_connected=_is_connected,
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib) — no extra deps",
        allowed_users_env="EMAIL_ALLOWED_USERS",
        allow_all_env="EMAIL_ALLOW_ALL_USERS",
        cron_deliver_env_var="EMAIL_HOME_ADDRESS",
        standalone_sender_fn=_standalone_send,
        max_message_length=50_000,
        pii_safe=True,
        emoji="📧",
        allow_update_command=True,
    )
