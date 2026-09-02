"""
Google Chat platform adapter.

Inbound: authenticated HTTP callbacks or a Cloud Pub/Sub pull subscription
(for no-public-URL deployments). Outbound: Google Chat REST API.

Concurrency model: the Pub/Sub SubscriberClient invokes its callback in a
background thread; ``handle_message`` must run on the asyncio loop, so the
callback schedules it thread-safely and never blocks on ``.result()`` (that
would saturate the Pub/Sub executor). All outbound Chat REST calls go through
``asyncio.to_thread`` because googleapiclient is synchronous.

Event routing: only MESSAGE events dispatch to the agent. ADDED_TO_SPACE caches
the bot's resource name; CARD_CLICKED is only ACK'd (interactivity deferred).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path as _Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.secret_scope import is_multiplex_active
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

from .cards import card_spec_to_cards_v2, format_message as _format_message  # noqa: F401  (re-exported)


def _adc_would_borrow_foreign_credentials() -> bool:
    """True when ADC would silently read another profile's SA from process env.

    ``google.auth.default()`` consults ``os.environ`` directly; under multiplexing
    a scoped profile reaching the ADC branch would authenticate as the default
    profile's identity if the process env carries one. Fail closed instead.
    """
    return is_multiplex_active() and bool(
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )

# Heavy google-cloud + googleapiclient imports are deferred to first adapter use
# (eager import cost ~110ms / ~33MB on every CLI invocation via plugin discovery).
# ``_load_google_modules()`` rebinds these globals. ``HttpError = Exception`` is
# the placeholder: ``except HttpError`` clauses bind the *current* global at
# evaluation time, and ``__init__`` always loads the real class first.
GOOGLE_CHAT_AVAILABLE: bool = False
httplib2: Any = None  # type: ignore
pubsub_v1: Any = None  # type: ignore
gax_exceptions: Any = None  # type: ignore
service_account: Any = None  # type: ignore
AuthorizedHttp: Any = None  # type: ignore
build_service: Any = None  # type: ignore
HttpError: Any = Exception  # type: ignore
MediaFileUpload: Any = None  # type: ignore

_google_modules_loaded: bool = False
# (global name, module, attribute-or-None) rebound by ``_load_google_modules``.
_GOOGLE_IMPORTS = (
    ("httplib2", "httplib2", None),
    ("pubsub_v1", "google.cloud.pubsub_v1", None),
    ("gax_exceptions", "google.api_core.exceptions", None),
    ("service_account", "google.oauth2.service_account", None),
    ("AuthorizedHttp", "google_auth_httplib2", "AuthorizedHttp"),
    ("build_service", "googleapiclient.discovery", "build"),
    ("HttpError", "googleapiclient.errors", "HttpError"),
    ("MediaFileUpload", "googleapiclient.http", "MediaFileUpload"),
)
_GOOGLE_ID_TOKEN_CERTS_TTL_SECONDS = 300
_google_id_token_request: Any = None
_google_id_token_request_lock = threading.Lock()


class _CachedGoogleAuthRequest:
    def __init__(self, request: Any, ttl_seconds: int = _GOOGLE_ID_TOKEN_CERTS_TTL_SECONDS) -> None:
        self._request = request
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}

    def __call__(self, url: str, method: str = "GET", **kwargs: Any) -> Any:
        cache_key = (method.upper(), url)
        if cache_key[0] != "GET":
            return self._request(url=url, method=method, **kwargs)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > now:
                return cached[1]
        response = self._request(url=url, method=method, **kwargs)
        if getattr(response, "status", None) == 200:
            with self._lock:
                self._cache[cache_key] = (now + self._ttl_seconds, response)
        return response


def _get_google_id_token_request() -> Any:
    global _google_id_token_request
    with _google_id_token_request_lock:
        if _google_id_token_request is None:
            try:
                from google.auth.transport import requests as google_requests
            except ImportError as exc:
                raise RuntimeError("google-auth is required for Google Chat HTTP callbacks") from exc
            _google_id_token_request = _CachedGoogleAuthRequest(google_requests.Request())
        return _google_id_token_request


def _verify_google_id_token(token: str, audience: str) -> Dict[str, Any]:
    try:
        from google.oauth2 import id_token
    except ImportError as exc:
        raise RuntimeError("google-auth is required for Google Chat HTTP callbacks") from exc
    return id_token.verify_oauth2_token(token, _get_google_id_token_request(), audience)


def _load_google_modules() -> bool:
    """Lazily import the google-cloud + googleapiclient stack; idempotent.

    Returns True when the optional deps imported; rebinds the module globals so
    code using ``pubsub_v1``, ``service_account``, ``HttpError`` sees real classes.
    """
    global GOOGLE_CHAT_AVAILABLE, _google_modules_loaded
    if _google_modules_loaded:
        return GOOGLE_CHAT_AVAILABLE
    _google_modules_loaded = True
    try:
        loaded = {
            name: getattr(importlib.import_module(module), attr) if attr else importlib.import_module(module)
            for name, module, attr in _GOOGLE_IMPORTS
        }
    except ImportError:
        GOOGLE_CHAT_AVAILABLE = False
        return False
    globals().update(loaded)
    GOOGLE_CHAT_AVAILABLE = True
    return True

from gateway.config import Platform, PlatformConfig

# Register the dynamic ``google_chat`` enum member at import time: ``_missing_()``
# caches the pseudo-member so ``Platform.GOOGLE_CHAT`` attribute access works
# before any adapter instance is constructed.
Platform("google_chat")
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult,
    cache_audio_from_bytes, cache_document_from_bytes, cache_image_from_bytes, cache_video_from_bytes,
)


# Pinned to the legacy module path so operator log filters keep matching after
# the in-tree → plugin migration (``__name__`` is namespaced by the plugin loader).
logger = logging.getLogger("gateway.platforms.google_chat")

_SUBSCRIPTION_PATH_RE = re.compile(r"^projects/(?P<project>[^/]+)/subscriptions/(?P<sub>[^/]+)$")

# chat.bot covers the bot's own messaging ops (create/patch/delete, spaces,
# memberships, media.download). The bot CANNOT call media.upload — Google
# requires user OAuth for it; see ``oauth.py`` for the per-user consent flow.
_CHAT_SCOPES = ["https://www.googleapis.com/auth/chat.bot", "https://www.googleapis.com/auth/pubsub"]

# Google Chat text-message size limit is 4096; leave margin.
_MAX_TEXT_LENGTH = 4000
# Per-space rate-limit hit counter threshold; warn if exceeded.
_RATE_LIMIT_WARN_THRESHOLD = 5

# Outbound retry: transient 429/5xx would otherwise drop user-visible messages;
# backoff stays bounded so a true outage surfaces quickly.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 8.0
_RETRY_JITTER = 0.3
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def _http_status(exc: BaseException) -> Any:
    """``resp.status`` of a googleapiclient HttpError, or None."""
    return getattr(getattr(exc, "resp", None), "status", None)


def _is_retryable_error(exc: BaseException) -> bool:
    """True for transient failures (429, 5xx, transport errors); auth/4xx are permanent."""
    status = _http_status(exc)
    if isinstance(status, int):
        return status in _RETRYABLE_HTTP_STATUSES
    # SSL/socket errors carry no HTTP status: match common transport wording.
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return True
    if "connection" in text and ("reset" in text or "refused" in text or "aborted" in text):
        return True
    return "broken pipe" in text or "remote disconnected" in text

# Left in ``_typing_messages`` after ``send()`` patches the typing marker into the
# real response: keeps ``_keep_typing`` from creating a fresh card before the base
# class cancels its typing task, and tells ``stop_typing`` to skip the API delete
# (which would tombstone the response we just patched).
_TYPING_CONSUMED_SENTINEL = "<consumed>"


def check_google_chat_requirements() -> bool:
    """Canonical "are the optional deps available" probe; triggers the lazy import."""
    return _load_google_modules()


# Only these hosts may serve attachment download URIs: anything else is rejected
# to block SSRF (e.g. downloadUri pointing at the GCE metadata service with the
# SA bearer token attached).
_TRUSTED_ATTACHMENT_HOSTS = (
    "googleapis.com", "chat.google.com", "drive.google.com", "docs.google.com",
    "lh3.googleusercontent.com", "lh4.googleusercontent.com", "lh5.googleusercontent.com", "lh6.googleusercontent.com",
)


def _is_google_owned_host(url: str) -> bool:
    """Return True iff *url* is https and targets a Google-owned domain."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return any(host == h or host.endswith("." + h) for h in _TRUSTED_ATTACHMENT_HOSTS)


def _redact_sensitive(text: str) -> str:
    """Redact Pub/Sub resource paths and SA emails from user-facing error strings."""
    if not text:
        return text
    text = re.sub(r"projects/[^/\s]+/subscriptions/[^/\s]+", "projects/<redacted>/subscriptions/<redacted>", text)
    text = re.sub(r"projects/[^/\s]+/topics/[^/\s]+", "projects/<redacted>/topics/<redacted>", text)
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.iam\.gserviceaccount\.com",
        "<sa>@<project>.iam.gserviceaccount.com",
        text,
    )
    return text


_MIME_MESSAGE_TYPES = (("image/", MessageType.PHOTO), ("audio/", MessageType.AUDIO), ("video/", MessageType.VIDEO))


def _mime_for_message_type(mime: str) -> MessageType:
    """Map a MIME string to a MessageType; non-media falls through to DOCUMENT."""
    for prefix, message_type in _MIME_MESSAGE_TYPES:
        if mime and mime.startswith(prefix):
            return message_type
    return MessageType.DOCUMENT


class _SACredentialError(Exception):
    """Internal: classified SA-credential load failure (``kind`` + optional cause)."""

    def __init__(self, kind: str, detail: Optional[BaseException] = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.detail = detail


def _load_sa_credentials_from(sa_value: Optional[str]) -> Any:
    """Build SA credentials from a path / inline JSON, or fall back to ADC.

    Raises ``_SACredentialError`` with kind in {inline_invalid, not_found,
    file_invalid, adc_foreign, adc_no_auth, adc_failed}; callers map kinds to
    their own user-facing messages.
    """
    if sa_value:
        if sa_value.lstrip().startswith("{"):
            try:
                info = json.loads(sa_value)
            except json.JSONDecodeError as exc:
                raise _SACredentialError("inline_invalid", exc) from exc
        else:
            if not os.path.exists(sa_value):
                raise _SACredentialError("not_found")
            try:
                with open(sa_value, "r", encoding="utf-8") as fh:
                    info = json.load(fh)
            except json.JSONDecodeError as exc:
                raise _SACredentialError("file_invalid", exc) from exc
        return service_account.Credentials.from_service_account_info(info, scopes=_CHAT_SCOPES)
    # No explicit SA — ADC (Cloud Run / GCE workload identity, or gcloud ADC login).
    if _adc_would_borrow_foreign_credentials():
        raise _SACredentialError("adc_foreign")
    try:
        import google.auth as google_auth
    except ImportError:
        raise _SACredentialError("adc_no_auth")
    try:
        credentials, _project = google_auth.default(scopes=_CHAT_SCOPES)
    except Exception as exc:
        raise _SACredentialError("adc_failed", exc) from exc
    return credentials


class _ThreadCountStore:
    """Per-(chat_id, thread_name) inbound message counter, persisted to disk.

    Drives the DM main-flow vs side-thread heuristic: prev_count == 0 means Chat
    just auto-created the thread for a top-level message (shared DM session, reply
    top-level); prev_count >= 1 means the user engaged an existing thread (isolate
    session, reply in-thread). Persistence matters: a restart that wiped counts
    would demote active side-threads to main flow and leak context.

    File format: ``{"<chat_id>": {"<thread_name>": <int_count>}}``. Missing or
    corrupt file resets to empty (warned); write-through on every ``incr``.
    """

    def __init__(self, path: _Path):
        self._path = path
        self._counts: Dict[str, Dict[str, int]] = {}

    def load(self) -> None:
        """Load counts from disk; missing file → empty, corrupt JSON → empty + warn."""
        if not self._path.exists():
            self._counts = {}
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            logger.warning("[GoogleChat] thread-count store at %s is corrupt; starting fresh: %s", self._path, exc)
            self._counts = {}
            return
        except OSError as exc:
            logger.warning("[GoogleChat] could not read thread-count store at %s: %s", self._path, exc)
            self._counts = {}
            return
        # Validate shape — anything off-schema gets dropped silently.
        clean: Dict[str, Dict[str, int]] = {}
        if isinstance(data, dict):
            for chat_id, threads in data.items():
                if not isinstance(chat_id, str) or not isinstance(threads, dict):
                    continue
                clean_threads = {t: c for t, c in threads.items() if isinstance(t, str) and isinstance(c, int)}
                if clean_threads:
                    clean[chat_id] = clean_threads
        self._counts = clean

    def get(self, chat_id: str, thread_name: str) -> int:
        return self._counts.get(chat_id, {}).get(thread_name, 0)

    def incr(self, chat_id: str, thread_name: str) -> int:
        """Increment and write through; returns the PRE-increment value."""
        chat_counts = self._counts.setdefault(chat_id, {})
        prev = chat_counts.get(thread_name, 0)
        chat_counts[thread_name] = prev + 1
        self._save()
        return prev

    def _save(self) -> None:
        """Atomic write; failure is non-fatal (in-memory counts stay consistent)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._counts, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning("[GoogleChat] could not persist thread-count store to %s: %s", self._path, exc)


_SA_ERROR_MESSAGES = {
    "inline_invalid": "Inline SA JSON is not valid JSON: {exc}",
    "not_found": "Service Account JSON file not found at configured path.",
    "file_invalid": "Service Account JSON file is not valid JSON: {exc}",
    "adc_foreign": (
        "Google Chat ADC skipped for this profile: service-account credentials are set in the process "
        "environment but not in this profile's secret scope. Set GOOGLE_CHAT_SERVICE_ACCOUNT_JSON in this "
        "profile's .env."
    ),
    "adc_no_auth": (
        "No Service Account credentials configured. Set GOOGLE_CHAT_SERVICE_ACCOUNT_JSON or "
        "GOOGLE_APPLICATION_CREDENTIALS, or install google-auth to use Application Default Credentials."
    ),
    "adc_failed": (
        "No Service Account credentials configured and Application Default Credentials are unavailable. Set "
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON or run ``gcloud auth application-default login``. ADC error: {exc}"
    ),
}


class GoogleChatAdapter(BasePlatformAdapter):
    """
    Google Chat bot adapter using Pub/Sub pull + Chat REST API.

    Required environment (see gateway/config.py Google Chat block):
      GOOGLE_CHAT_PROJECT_ID           (or GOOGLE_CLOUD_PROJECT fallback)
      GOOGLE_CHAT_SUBSCRIPTION_NAME    (or GOOGLE_CHAT_SUBSCRIPTION fallback)
      GOOGLE_CHAT_SERVICE_ACCOUNT_JSON (or GOOGLE_APPLICATION_CREDENTIALS)

    Optional:
      GOOGLE_CHAT_ALLOWED_USERS, GOOGLE_CHAT_ALLOW_ALL_USERS
      GOOGLE_CHAT_HOME_CHANNEL
      GOOGLE_CHAT_MAX_MESSAGES (FlowControl, default 1)
      GOOGLE_CHAT_MAX_BYTES    (FlowControl, default 16_777_216 = 16 MiB)
    """

    MAX_MESSAGE_LENGTH = _MAX_TEXT_LENGTH
    # Pub/Sub supervisor configuration.
    _MAX_RECONNECT_ATTEMPTS = 10
    _RECONNECT_BASE_DELAY = 2.0
    _RECONNECT_MAX_DELAY = 120.0

    def __init__(self, config: PlatformConfig):
        # ``Platform("google_chat")`` resolves via ``_missing_()``; bundled platform
        # plugins are looked up by value, not enum attribute (matches Teams, IRC).
        super().__init__(config, Platform("google_chat"))
        # Load the google stack here (not only in connect()) so code paths that
        # construct the adapter and call methods directly see real classes for
        # ``MediaFileUpload`` / ``HttpError`` / etc. Idempotent.
        _load_google_modules()
        self._subscriber: Optional[Any] = None
        self._chat_api: Optional[Any] = None
        # User-authed Chat clients for native ``media.upload`` (bot identity is
        # rejected there). Multi-user: each user's token lives under their email
        # and ``_send_file`` picks the requesting user's via ``_last_sender_by_chat``.
        # ``_user_credentials`` / ``_user_chat_api`` hold the LEGACY single-user
        # token as a last-ditch fallback for pre-multi-user installs.
        self._user_chat_api: Optional[Any] = None
        self._user_credentials: Optional[Any] = None
        self._user_creds_by_email: Dict[str, Any] = {}
        self._user_chat_api_by_email: Dict[str, Any] = {}
        # chat_id → most-recent inbound sender email (drives per-user token lookup).
        self._last_sender_by_chat: Dict[str, str] = {}
        self._credentials: Optional[Any] = None
        self._project_id: Optional[str] = None
        self._subscription_path: Optional[str] = None
        self._streaming_pull_future: Optional[Any] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot_user_id: Optional[str] = None  # users/{id}
        self._dedup = MessageDeduplicator()
        self._typing_messages: Dict[str, str] = {}
        self._clarify_state: Dict[str, str] = {}
        self._shutting_down = False
        self._rate_limit_hits: Dict[str, int] = {}
        # Last inbound thread per space. DMs create a NEW thread per top-level
        # message but users see one conversation: thread_id is dropped from the
        # source (stable session key) and cached here so replies still land in
        # the right visual thread.
        self._last_inbound_thread: Dict[str, str] = {}
        # Persisted per-(chat, thread) inbound counts for the side-thread heuristic
        # (survives gateway restarts).
        try:
            from hermes_constants import get_hermes_home as _get_hermes_home
            _hermes_home = _get_hermes_home()
        except (ModuleNotFoundError, ImportError):
            _hermes_home = _Path.home() / ".hermes"
        self._thread_count_store = _ThreadCountStore(_hermes_home / "google_chat_thread_counts.json")
        # In-flight typing-card creates per chat_id: send_typing() reserves an Event
        # BEFORE the API call so concurrent _keep_typing calls wait instead of
        # duplicating cards.
        self._typing_card_inflight: Dict[str, asyncio.Event] = {}
        # Typing cards that lost a race with send(); patched away at end of turn.
        self._orphan_typing_messages: Dict[str, List[str]] = {}
        # Snapshot profile-scoped settings now: Pub/Sub callbacks run on threads
        # where the ContextVar secret scope is unavailable.
        extra = self.config.extra
        self._max_messages = self._int_setting(extra, "max_messages", "GOOGLE_CHAT_MAX_MESSAGES", 1)
        self._max_bytes = self._int_setting(extra, "max_bytes", "GOOGLE_CHAT_MAX_BYTES", 16 * 1024 * 1024)
        self._bootstrap_spaces = str(
            extra.get("bootstrap_spaces") or _get_scoped_secret("GOOGLE_CHAT_BOOTSTRAP_SPACES", "") or ""
        ).strip()
        self._debug_raw = bool(extra.get("debug_raw") or _get_scoped_secret("GOOGLE_CHAT_DEBUG_RAW"))
        self._http_events_url = self._str_setting(extra, "http_events_url", "GOOGLE_CHAT_HTTP_EVENTS_URL")
        self._http_events_audience = self._str_setting(
            extra, "http_events_audience", "GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE", self._http_events_url
        )
        self._http_events_service_account_email = self._str_setting(
            extra, "http_events_service_account_email", "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL"
        ).lower()

    @staticmethod
    def _int_setting(extra: Dict[str, Any], key: str, env_name: str, default: int) -> int:
        try:
            return int(extra.get(key) or _get_scoped_secret(env_name, str(default)))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _str_setting(extra: Dict[str, Any], key: str, env_name: str, fallback: str = "") -> str:
        return (extra.get(key) or _get_scoped_secret(env_name, "") or fallback).strip()

    # ------------------------------------------------------------------
    # Configuration loading and validation
    # ------------------------------------------------------------------
    def _load_sa_credentials(self) -> Any:
        """Load SA credentials: ``extra['service_account_json']`` (path or inline
        JSON) → ``GOOGLE_APPLICATION_CREDENTIALS`` → Application Default Credentials."""
        sa_path = self.config.extra.get("service_account_json") or _get_scoped_secret("GOOGLE_APPLICATION_CREDENTIALS")
        try:
            credentials = _load_sa_credentials_from(sa_path)
        except _SACredentialError as err:
            message = _SA_ERROR_MESSAGES[err.kind].format(exc=err.detail)
            if err.kind == "not_found":
                raise FileNotFoundError(message)
            raise ValueError(message) from err.detail
        if not sa_path:
            logger.info("[GoogleChat] No SA JSON configured; using Application Default Credentials")
        return credentials

    def _validate_config(self) -> Tuple[str, Optional[str]]:
        """Return (project_id, subscription_path); the latter is None for HTTP inbound.

        Raises ValueError with a sanitized message on any config problem.
        """
        project_id = (self.config.extra.get("project_id") or "").strip()
        subscription = (self.config.extra.get("subscription_name") or "").strip()
        http_events_url = (self.config.extra.get("http_events_url") or "").strip()

        if subscription:
            match = _SUBSCRIPTION_PATH_RE.match(subscription)
            if not match:
                raise ValueError("GOOGLE_CHAT_SUBSCRIPTION_NAME must match 'projects/<project>/subscriptions/<sub>'.")
            subscription_project = match.group("project")
            if project_id and subscription_project != project_id:
                raise ValueError(
                    "project_id in GOOGLE_CHAT_PROJECT_ID does not match the "
                    "project embedded in GOOGLE_CHAT_SUBSCRIPTION_NAME."
                )
            return project_id or subscription_project, subscription
        if http_events_url:
            return project_id, None
        if not project_id:
            raise ValueError("GOOGLE_CHAT_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set.")
        raise ValueError(
            "GOOGLE_CHAT_SUBSCRIPTION_NAME (or GOOGLE_CHAT_SUBSCRIPTION) is not set. "
            "Set GOOGLE_CHAT_HTTP_EVENTS_URL for HTTP callback mode."
        )

    # ------------------------------------------------------------------
    # Loop bridge helpers (thread -> asyncio loop)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_background_failure(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("[GoogleChat] Background inbound processing failed")

    @staticmethod
    def _loop_accepts_callbacks(loop: Optional[asyncio.AbstractEventLoop]) -> bool:
        return loop is not None and not bool(getattr(loop, "is_closed", lambda: False)())

    def _submit_on_loop(self, coro: Any) -> None:
        """Schedule a coroutine on the adapter loop from a Pub/Sub callback thread."""
        loop = self._loop
        if not self._loop_accepts_callbacks(loop):
            # Shutdown race: safe to drop, Pub/Sub redelivers on next reconnect.
            logger.warning("[GoogleChat] Loop not accepting callbacks; dropping event")
            return
        try:
            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(
                coro, loop,
                logger=logger,
                log_message="[GoogleChat] Failed to schedule background callback",
                log_level=logging.WARNING,
            )
        except RuntimeError:
            logger.warning("[GoogleChat] Loop closed between check and submit")
            return
        if future is None:
            return
        future.add_done_callback(self._log_background_failure)

    # ------------------------------------------------------------------
    # Bot identity resolution
    # ------------------------------------------------------------------
    def _bot_id_cache_path(self) -> _Path:
        """Cache location for the resolved bot user_id; resolved at call time so
        multiplexed profiles (connect() runs in-scope) don't share one file."""
        from hermes_constants import get_hermes_home as _get_hermes_home

        return _get_hermes_home() / "google_chat_bot_id.json"

    def _load_cached_bot_id(self) -> Optional[str]:
        path = self._bot_id_cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("bot_user_id") or None
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cached_bot_id(self, bot_user_id: str) -> None:
        try:
            path = self._bot_id_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"bot_user_id": bot_user_id}), encoding="utf-8")
        except OSError:
            logger.debug("[GoogleChat] Could not persist bot_user_id cache", exc_info=True)

    async def _resolve_bot_user_id(self) -> Optional[str]:
        """Resolve ``users/{id}`` via members.list on the home channel, then bootstrap spaces.

        Returns None when no space is known; the self-filter then falls back to
        ``sender.type == 'BOT'`` (safe but less precise).
        """
        candidate_spaces: List[str] = []
        if self.config.home_channel and self.config.home_channel.chat_id:
            candidate_spaces.append(self.config.home_channel.chat_id)
        if self._bootstrap_spaces:
            candidate_spaces.extend(s.strip() for s in self._bootstrap_spaces.split(",") if s.strip())
        for space in candidate_spaces:
            try:
                members = await asyncio.to_thread(
                    lambda s=space: self._chat_api.spaces()
                    .members()
                    .list(parent=s, pageSize=50)
                    .execute(http=self._new_authed_http())
                )
            except HttpError as exc:
                logger.debug("[GoogleChat] members.list failed on %s: %s", space, _redact_sensitive(str(exc)))
                continue
            for member in members.get("memberships", []):
                if member.get("member", {}).get("type") == "BOT":
                    name = member.get("member", {}).get("name")
                    if name:
                        return name
        return None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Validate config, authenticate, start Pub/Sub pull, resolve bot id."""
        # connect() is the gate: everything after sees the real google classes.
        if not _load_google_modules():
            self._set_fatal_error(
                code="missing_deps",
                message="google-cloud-pubsub / google-api-python-client not installed",
                retryable=False,
            )
            return False

        self._loop = asyncio.get_running_loop()
        try:
            project_id, subscription_path = self._validate_config()
            credentials = self._load_sa_credentials()
        except (ValueError, FileNotFoundError) as exc:
            msg = _redact_sensitive(str(exc))
            logger.error("[GoogleChat] Config validation failed: %s", msg)
            self._set_fatal_error(code="config_invalid", message=msg, retryable=False)
            return False

        self._project_id = project_id
        self._subscription_path = subscription_path
        self._credentials = credentials

        try:
            self._chat_api = await asyncio.to_thread(
                lambda: build_service("chat", "v1", credentials=credentials, cache_discovery=False)
            )
        except Exception as exc:
            msg = _redact_sensitive(str(exc))
            logger.error("[GoogleChat] Failed to build Chat API client: %s", msg)
            self._set_fatal_error(code="chat_api_init", message=msg, retryable=False)
            return False

        # Legacy single-user OAuth (per-user tokens load lazily on first send).
        # Failure is NON-fatal: only attachments degrade to a text notice.
        try:
            from .oauth import (
                load_user_credentials as _load_user_creds,
                build_user_chat_service as _build_user_chat,
                list_authorized_emails as _list_emails,
            )
            user_creds = await asyncio.to_thread(_load_user_creds)
            if user_creds is not None:
                self._user_credentials = user_creds
                self._user_chat_api = await asyncio.to_thread(lambda: _build_user_chat(user_creds))
                logger.info("[GoogleChat] Legacy user OAuth loaded — fallback attachment delivery enabled")
            authorized = await asyncio.to_thread(_list_emails)
            if authorized:
                logger.info("[GoogleChat] %d per-user OAuth tokens on disk: %s", len(authorized), ", ".join(authorized))
            elif user_creds is None:
                logger.info(
                    "[GoogleChat] No user OAuth tokens at setup — file "
                    "attachments will degrade to text-only fallback. "
                    "Each user runs /setup-files once in their own DM "
                    "to enable native attachments."
                )
        except Exception as exc:
            logger.warning(
                "[GoogleChat] User OAuth load failed (attachments will degrade to text-only fallback): %s",
                _redact_sensitive(str(exc)),
            )
            self._user_credentials = None
            self._user_chat_api = None

        try:
            await asyncio.to_thread(self._thread_count_store.load)
        except Exception:
            logger.warning("[GoogleChat] thread-count store load failed (treating all threads as fresh)", exc_info=True)

        if subscription_path is not None:
            # Sanity check: subscription exists / SA has access.
            subscription_fatals = {
                gax_exceptions.NotFound: dict(
                    code="subscription_not_found", message="Pub/Sub subscription not found at configured path"
                ),
                gax_exceptions.PermissionDenied: dict(
                    code="subscription_permission",
                    message="Service Account lacks roles/pubsub.subscriber on the subscription",
                ),
            }
            self._subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
            try:
                await asyncio.to_thread(
                    lambda: self._subscriber.get_subscription(request={"subscription": subscription_path})
                )
            except (gax_exceptions.NotFound, gax_exceptions.PermissionDenied) as exc:
                self._set_fatal_error(**subscription_fatals[type(exc)], retryable=False)
                return False
            except Exception as exc:
                msg = _redact_sensitive(str(exc))
                logger.error("[GoogleChat] subscription.get failed: %s", msg)
                self._set_fatal_error(code="subscription_check", message=msg, retryable=True)
                return False

        # Resolve bot user_id (eager): cache first, then members.list.
        self._bot_user_id = self._load_cached_bot_id()
        if not self._bot_user_id:
            self._bot_user_id = await self._resolve_bot_user_id()
            if self._bot_user_id:
                self._save_cached_bot_id(self._bot_user_id)
            else:
                logger.info("[GoogleChat] bot_user_id not yet resolved; will resolve on first addedToSpace or member lookup")

        if subscription_path is not None:
            self._supervisor_task = asyncio.create_task(self._run_supervisor())
            inbound = "pubsub"
        else:
            self._supervisor_task = None
            inbound = "http"

        self._mark_connected()
        logger.info(
            "[GoogleChat] Connected; project=%s, inbound=%s, subscription=%s, "
            "bot_user_id=%s, flow_control(msgs=%s, bytes=%s)",
            project_id or "<unset>",
            inbound,
            "<redacted>" if subscription_path else "<none>",
            self._bot_user_id or "<unresolved>",
            self._max_messages,
            self._max_bytes,
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Clean shutdown: stop accepting new messages, wait in-flight, close clients."""
        self._shutting_down = True
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._supervisor_task, timeout=5.0)
        if self._streaming_pull_future is not None:
            with contextlib.suppress(Exception):
                self._streaming_pull_future.cancel()
                await asyncio.to_thread(self._streaming_pull_future.result, 10.0)
            self._streaming_pull_future = None
        if self._subscriber is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._subscriber.close)
            self._subscriber = None
        self._mark_disconnected()
        logger.info("[GoogleChat] Disconnected")

    # ------------------------------------------------------------------
    # Pub/Sub supervisor (reconnect loop)
    # ------------------------------------------------------------------
    async def _run_supervisor(self) -> None:
        """Run streaming_pull with exponential backoff + full jitter; fatal after N attempts.

        ``subscribe()`` returns a Future that resolves when the stream dies; we
        await ``future.result()`` in a worker thread and react to exceptions.
        """
        attempt = 0
        while not self._shutting_down:
            flow = pubsub_v1.types.FlowControl(max_messages=self._max_messages, max_bytes=self._max_bytes)
            try:
                future = self._subscriber.subscribe(
                    self._subscription_path,
                    callback=self._on_pubsub_message,
                    flow_control=flow,
                )
                self._streaming_pull_future = future
                if attempt > 0:
                    logger.info("[GoogleChat] Pub/Sub stream reconnected after %d attempts", attempt)
                attempt = 0
                # Blocks until stream dies or cancel(); normal completion = disconnect.
                await asyncio.to_thread(future.result)
                if self._shutting_down:
                    return
            except asyncio.CancelledError:
                return
            except gax_exceptions.Unauthenticated:
                self._set_fatal_error(
                    code="pubsub_auth", message="Pub/Sub authentication failed (SA key invalid/revoked)", retryable=False
                )
                return
            except gax_exceptions.PermissionDenied:
                self._set_fatal_error(
                    code="pubsub_permission", message="SA lacks pubsub.subscriber on the subscription", retryable=False
                )
                return
            except Exception as exc:
                attempt += 1
                msg = _redact_sensitive(str(exc))
                logger.warning(
                    "[GoogleChat] Pub/Sub stream died (attempt %d/%d): %s", attempt, self._MAX_RECONNECT_ATTEMPTS, msg
                )
                if attempt >= self._MAX_RECONNECT_ATTEMPTS:
                    self._set_fatal_error(
                        code="pubsub_reconnect_exhausted",
                        message=f"Pub/Sub reconnect failed {attempt} times; giving up",
                        retryable=False,
                    )
                    return
                delay = min(self._RECONNECT_MAX_DELAY, self._RECONNECT_BASE_DELAY * (2 ** (attempt - 1)))
                try:
                    await asyncio.sleep(random.uniform(0, delay))
                except asyncio.CancelledError:
                    return

    # ------------------------------------------------------------------
    # Inbound event handling (Pub/Sub callback runs in a thread)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_message_payload(
        envelope: Dict[str, Any], ce_type: str = ""
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]:
        """Detect the envelope format and return ``(message, space, format_name)``.

        Returns None for unrecognized envelopes or non-MESSAGE events. Formats:
          1. Workspace Add-ons: ``{"chat": {"messagePayload": {"message", "space"}}}``
          2. Native Chat API Pub/Sub: ``{"type": "MESSAGE", "message", "space"}``
          3. Relay / flat: ``{"event_type", "sender_email", "text", "space_name",
             "thread_name", "message_name"}`` — a Chat-API-shaped ``message`` is
             synthesized so downstream code needs no branching.
        """
        # Format 1: membership/card payloads are handled by _on_pubsub_message first.
        chat_block = envelope.get("chat") or {}
        msg_payload_wrapper = chat_block.get("messagePayload") if chat_block else None
        if msg_payload_wrapper:
            msg = msg_payload_wrapper.get("message") or {}
            space = msg_payload_wrapper.get("space") or msg.get("space") or {}
            return msg, space, "workspace_addons"

        # Format 2.
        if isinstance(envelope.get("message"), dict):
            if envelope.get("type", "") != "MESSAGE":
                return None
            msg = envelope["message"]
            space = envelope.get("space") or msg.get("space") or {}
            return msg, space, "native_chat_api"

        # Format 3.
        if "event_type" in envelope or "sender_email" in envelope:
            if envelope.get("event_type", "MESSAGE") != "MESSAGE":
                return None
            sender_email = (envelope.get("sender_email") or "").strip()
            sender_display = envelope.get("sender_display_name") or sender_email or "Unknown"
            # No Chat resource name for relay events: synthesize a stable surrogate
            # from the email so dedup keys / session IDs are deterministic.
            sender_name_surrogate = "users/relay-" + (sender_email or "unknown").replace("@", "_at_").replace(".", "_")
            text = envelope.get("text", "") or ""
            # Honor the relay's ``sender_type`` so the BOT self-filter fires for
            # forwarded bot replies (relays MUST forward sender.type); default to
            # HUMAN for backward compatibility when absent.
            sender_type = str(envelope.get("sender_type") or "HUMAN").strip().upper() or "HUMAN"
            if sender_type not in {"HUMAN", "BOT"}:
                sender_type = "HUMAN"
            msg: Dict[str, Any] = {
                "name": envelope.get("message_name", "") or "",
                "sender": {
                    "name": sender_name_surrogate,
                    "email": sender_email,
                    "displayName": sender_display,
                    "type": sender_type,
                },
                "text": text,
                "argumentText": text,
            }
            thread_name = envelope.get("thread_name") or ""
            if thread_name:
                msg["thread"] = {"name": thread_name}
            space = {
                "name": envelope.get("space_name", "") or "",
                "spaceType": envelope.get("space_type", "SPACE"),
            }
            return msg, space, "relay_flat"

        return None

    def _prepare_inbound(
        self, envelope: Dict[str, Any], ce_type: Optional[str] = None
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Extract + self-filter + dedup an inbound envelope.

        Returns ``(msg_with_space, enriched_envelope)`` ready for
        ``_dispatch_message``, or None when the event must be dropped. Debug
        logs only on the Pub/Sub path (``ce_type`` given).
        """
        extracted = self._extract_message_payload(envelope, ce_type or "")
        if extracted is None:
            if ce_type is not None:
                logger.debug(
                    "[GoogleChat] Envelope did not match a known message format; "
                    "ce-type=%s, keys=%s", ce_type, list(envelope.keys())
                )
            return None
        msg, space, _fmt = extracted
        # Self-filter: drop bot-sourced messages (own replies and other bots).
        if (msg.get("sender") or {}).get("type") == "BOT":
            return None
        # Dedup guard — Pub/Sub is at-least-once.
        msg_name = msg.get("name") or ""
        if msg_name and self._dedup.is_duplicate(msg_name):
            if ce_type is not None:
                logger.debug("[GoogleChat] Dedup drop for %s", msg_name)
            return None
        # Give both dicts a top-level "space" so the dispatch side has one shape.
        msg_with_space = dict(msg)
        if "space" not in msg_with_space and space:
            msg_with_space["space"] = space
        enriched_env = dict(envelope)
        if "space" not in enriched_env and space:
            enriched_env["space"] = space
        return msg_with_space, enriched_env

    def _on_pubsub_message(self, message: Any) -> None:
        """Pub/Sub callback — parse envelope and dispatch to the asyncio loop.

        Runs in a SubscriberClient worker thread: never block, never raise (that
        triggers nack + infinite redelivery). Event type comes from the ``ce-type``
        attribute; the body's ``chat`` object carries ``messagePayload`` or
        ``membershipPayload`` accordingly.
        """
        if self._shutting_down:
            message.nack()
            return
        try:
            envelope = json.loads(message.data.decode("utf-8"))
        except Exception:
            logger.exception("[GoogleChat] Could not parse Pub/Sub envelope")
            message.ack()
            return

        attrs = dict(getattr(message, "attributes", {}) or {})
        ce_type = attrs.get("ce-type") or ""
        logger.debug("[GoogleChat] Envelope keys=%s, ce-type=%s", list(envelope.keys()), ce_type)
        if self._debug_raw:
            # Contains message text + sender email: redact and gate at DEBUG so
            # default log configs never surface it.
            try:
                from agent.redact import redact_sensitive_text

                dump = redact_sensitive_text(json.dumps(envelope))
            except Exception:
                dump = "<redact filter unavailable>"
            logger.debug("[GoogleChat] RAW envelope (redacted): %s", dump[:2000])

        try:
            chat_block = envelope.get("chat") or {}
            if "membership" in ce_type or "MEMBERSHIP" in ce_type:
                mpl = chat_block.get("membershipPayload") or {}
                space = mpl.get("space") or {}
                membership = mpl.get("membership") or {}
                if "created" in ce_type:
                    # ADDED_TO_SPACE for this bot — resolve self user_id.
                    member = membership.get("member") or {}
                    if member.get("type") == "BOT" and not self._bot_user_id:
                        name = member.get("name")
                        if name:
                            self._bot_user_id = name
                            self._save_cached_bot_id(name)
                    logger.info("[GoogleChat] ADDED_TO_SPACE %s", space.get("name", "?"))
                else:
                    logger.info("[GoogleChat] REMOVED_FROM_SPACE %s", space.get("name", "?"))
                message.ack()
                return
            if "widget" in ce_type or "card" in ce_type.lower():
                logger.info("[GoogleChat] Card/widget event ack'd (v2 feature, deferred)")
                message.ack()
                return
            prepared = self._prepare_inbound(envelope, ce_type)
            if prepared is not None:
                self._submit_on_loop(self._dispatch_message(*prepared))
            message.ack()
        except Exception:
            logger.exception("[GoogleChat] Error in _on_pubsub_message")
            with contextlib.suppress(Exception):
                message.ack()

    async def dispatch_http_event(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        prepared = self._prepare_inbound(envelope)
        if prepared is not None:
            await self._dispatch_message(*prepared)
        return {}

    def verify_http_event_request(self, auth_header: str) -> Tuple[bool, str]:
        if not self._http_events_audience or not self._http_events_service_account_email:
            return False, "google_chat_http_events_not_configured"
        if not auth_header.startswith("Bearer "):
            return False, "missing_google_bearer"
        token = auth_header[7:].strip()
        if not token:
            return False, "missing_google_bearer"
        try:
            claims = _verify_google_id_token(token, self._http_events_audience)
        except Exception as exc:
            logger.warning("[GoogleChat] HTTP event bearer verification failed: %s", _redact_sensitive(str(exc)))
            return False, "invalid_google_bearer"
        expected = {item.strip().lower() for item in self._http_events_service_account_email.split(",") if item.strip()}
        claim_email = str(claims.get("email") or "").strip().lower()
        if not claim_email or claim_email not in expected:
            return False, "unexpected_google_bearer_identity"
        return True, ""

    async def _dispatch_message(self, msg: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        """Translate a Chat message to a MessageEvent and hand off.

        ``/setup-files`` is intercepted BEFORE the agent sees it (bot-local OAuth flow).
        """
        try:
            event = await self._build_message_event(msg, envelope)
            if event is None:
                return
            text = (event.text or "").strip()
            if text.startswith("/setup-files") and event.source is not None:
                # The sender email (user_id_alt) is the per-user OAuth token key.
                sender_email = event.source.user_id_alt if event.source and event.source.user_id_alt else None
                handled = await self._handle_setup_files_command(
                    chat_id=event.source.chat_id,
                    thread_id=event.source.thread_id,
                    raw_text=text,
                    sender_email=sender_email,
                )
                if handled:
                    return
            await self.handle_message(event)
        except Exception:
            logger.exception("[GoogleChat] _dispatch_message failed")

    async def _handle_setup_files_command(
        self, chat_id: str, thread_id: Optional[str], raw_text: str, sender_email: Optional[str] = None
    ) -> bool:
        """In-chat OAuth setup flow; see ``setup_files.handle_setup_files_command``."""
        from .setup_files import handle_setup_files_command

        return await handle_setup_files_command(self, chat_id, thread_id, raw_text, sender_email)

    async def _build_message_event(self, msg: Dict[str, Any], envelope: Dict[str, Any]) -> Optional[MessageEvent]:
        """Parse a Chat API message into a hermes MessageEvent."""
        space = envelope.get("space") or msg.get("space") or {}
        space_name = space.get("name") or ""  # "spaces/XXX"
        space_type = (space.get("type") or space.get("spaceType") or "").upper()
        thread = msg.get("thread") or {}
        thread_name = thread.get("name") or None
        sender = msg.get("sender") or {}
        sender_name = sender.get("name") or ""
        sender_display = sender.get("displayName") or sender.get("email") or sender_name
        sender_email = sender.get("email") or ""

        # Cache the asker's email per space so _send_file picks the right per-user
        # OAuth token (lower-cased to match the sanitized token-file lookup).
        if sender_email and space_name:
            self._last_sender_by_chat[space_name] = sender_email.strip().lower()

        chat_type = "dm" if space_type in {"DIRECT_MESSAGE", "DM"} else "group"
        text = (msg.get("argumentText") or msg.get("text") or "").strip()

        # Slash command: emit MessageType.COMMAND with normalized text.
        slash = msg.get("slashCommand") or {}
        is_slash = bool(slash)
        if is_slash:
            command_id = str(slash.get("commandId") or "")
            if command_id and not text.startswith("/"):
                text = f"/cmd_{command_id} {text}".strip()

        media_urls: List[str] = []
        media_types: List[str] = []
        message_type = MessageType.TEXT
        for att in msg.get("attachment") or []:
            try:
                local_path, mime = await self._download_attachment(att)
            except Exception:
                logger.exception("[GoogleChat] attachment download failed")
                continue
            if not local_path:
                continue
            media_urls.append(local_path)
            media_types.append(mime or "application/octet-stream")
            # Prefer the first-seen type for MessageType if no text present.
            if message_type == MessageType.TEXT and not text:
                message_type = _mime_for_message_type(mime or "")
        if is_slash:
            message_type = MessageType.COMMAND

        # PRE-increment count (persisted) drives the main-flow-vs-side-thread heuristic.
        prev_thread_count = 0
        if thread_name and space_name:
            prev_thread_count = self._thread_count_store.incr(space_name, thread_name)

        # DMs: prev_count == 0 → Chat auto-created this thread for a top-level
        # message; share one DM session and reply top-level (passing thread.name
        # would render the pair as an expandable thread). prev_count >= 1 → user
        # engaged an existing thread; isolate the session and reply in-thread.
        # Groups: threads are real containers; always isolate AND reply in-thread.
        if chat_type == "dm":
            is_side_thread = prev_thread_count > 0
            session_thread_id = thread_name if is_side_thread else None
            # Outbound cache only for side-threads so main-flow replies land top-level.
            if thread_name and space_name and is_side_thread:
                self._last_inbound_thread[space_name] = thread_name
            elif space_name:
                self._last_inbound_thread.pop(space_name, None)
        else:
            session_thread_id = thread_name
            if thread_name and space_name:
                self._last_inbound_thread[space_name] = thread_name

        source = self.build_source(
            chat_id=space_name,
            chat_name=space.get("displayName") or space.get("name") or "",
            chat_type=chat_type,
            # Email is the canonical id (allowlists are configured with emails);
            # the ``users/{id}`` resource name moves to user_id_alt. Falls back to
            # the resource name when the sender has no email.
            user_id=(sender_email or sender_name),
            user_name=sender_display,
            thread_id=session_thread_id,
            user_id_alt=(sender_name or None),
        )
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=msg,
            message_id=msg.get("name") or None,
            media_urls=media_urls,
            media_types=media_types,
        )

    async def _download_attachment(self, attachment: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Download an inbound attachment to the local cache; return (path, mime).

        Bot SA path is ``media.download`` via ``attachmentDataRef.resourceName``.
        Drive-picker shares without a resourceName need user OAuth + Drive scope
        (skipped). ``downloadUri`` is a last resort — it is meant for user OAuth
        tokens and usually 401s for SA tokens, kept for forward-compat.
        """
        mime = attachment.get("contentType") or ""
        source = attachment.get("source") or ""
        name = attachment.get("name") or ""
        resource_name = (attachment.get("attachmentDataRef") or {}).get("resourceName") or ""
        download_uri = attachment.get("downloadUri") or ""

        # Chat tags BOTH drag-and-drop uploads AND Drive-picker shares as DRIVE_FILE;
        # only the former carry a resourceName the bot path can use.
        if source == "DRIVE_FILE" and not resource_name:
            logger.info("[GoogleChat] Skipping Drive-picker attachment (no resourceName, would need user-OAuth Drive scope)")
            return None, mime

        data: Optional[bytes] = None
        if resource_name:
            def _fetch_media() -> bytes:
                req = self._chat_api.media().download_media(resourceName=resource_name)
                from googleapiclient.http import MediaIoBaseDownload
                import io

                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()
                return buf.getvalue()

            try:
                data = await asyncio.to_thread(_fetch_media)
            except HttpError as exc:
                logger.warning("[GoogleChat] media.download_media failed: %s", _redact_sensitive(str(exc)))
                data = None

        if data is None and download_uri:
            if not _is_google_owned_host(download_uri):
                logger.warning("[GoogleChat] Rejecting attachment fetch: non-Google host")
                return None, mime

            def _fetch_uri() -> bytes:
                import google.auth.transport.requests as gar

                authed_session = gar.AuthorizedSession(self._credentials)
                resp = authed_session.get(download_uri, timeout=30)
                resp.raise_for_status()
                return resp.content

            try:
                data = await asyncio.to_thread(_fetch_uri)
            except Exception as exc:
                logger.warning(
                    "[GoogleChat] downloadUri fetch failed (SA tokens often "
                    "lack access here; this is expected for user-uploaded "
                    "content): %s",
                    _redact_sensitive(str(exc)),
                )
                return None, mime

        if data is None:
            return None, mime

        # cache_* helpers take ``ext`` for media and a positional filename for docs.
        filename = name.split("/")[-1] if name else "attachment"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if mime.startswith("image/"):
            local = cache_image_from_bytes(data, ext=ext or ".jpg")
        elif mime.startswith("audio/"):
            local = cache_audio_from_bytes(data, ext=ext or ".ogg")
        elif mime.startswith("video/"):
            local = cache_video_from_bytes(data, ext=ext or ".mp4")
        else:
            local = cache_document_from_bytes(data, filename)
        return local, mime

    # ------------------------------------------------------------------
    # Outbound send paths
    # ------------------------------------------------------------------
    def _note_rate_limit(self, chat_id: str) -> int:
        self._rate_limit_hits[chat_id] = self._rate_limit_hits.get(chat_id, 0) + 1
        return self._rate_limit_hits[chat_id]

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a text message; ``metadata`` may carry ``thread_id``.

        A tracked typing card is transformed in-place via ``messages.patch`` (delete
        would leave a "Message deleted" tombstone). The base class's ``_keep_typing``
        is paused for this chat meanwhile. Over-long content: the first chunk patches
        the typing card, subsequent chunks are new messages.
        """
        thread_id = self._resolve_thread_id(reply_to, metadata, chat_id=chat_id)
        self.pause_typing_for_chat(chat_id)
        try:
            # Format BEFORE chunking so the size limit applies to the rendered form.
            chunks = self._chunk_text(self.format_message(content))
            if not chunks:
                return SendResult(success=False, error="empty message")

            last_result: Optional[SendResult] = None
            typing_msg_name = self._typing_messages.pop(chat_id, None)
            if typing_msg_name == _TYPING_CONSUMED_SENTINEL:
                typing_msg_name = None
            patched_typing = False

            for idx, chunk in enumerate(chunks):
                body: Dict[str, Any] = {"text": chunk}
                # Only set thread on the create path; patch inherits.
                if thread_id and (idx > 0 or not typing_msg_name):
                    body["thread"] = {"name": thread_id}
                try:
                    if idx == 0 and typing_msg_name:
                        result = await self._patch_message(typing_msg_name, body)
                        patched_typing = True
                    else:
                        result = await self._create_message(chat_id, body)
                    last_result = result
                except HttpError as exc:
                    status = _http_status(exc)
                    if status == 403:
                        self._set_fatal_error(
                            code="chat_forbidden",
                            message="Bot lacks access (removed from space or perms revoked)",
                            retryable=False,
                        )
                        return SendResult(success=False, error=str(exc))
                    if status == 404:
                        # Typing card deleted under us: fall back to a fresh message.
                        if idx == 0 and typing_msg_name:
                            logger.info("[GoogleChat] Typing card disappeared; creating new message")
                            typing_msg_name = None
                            result = await self._create_message(chat_id, body)
                            last_result = result
                            continue
                        logger.info("[GoogleChat] send target 404; skipping")
                        return SendResult(success=False, error="target not found")
                    if status == 429:
                        hits = self._note_rate_limit(chat_id)
                        if hits >= _RATE_LIMIT_WARN_THRESHOLD:
                            logger.warning("[GoogleChat] Rate limit hit %d times on chat; throttling", hits)
                        raise
                    raise
            if last_result is None:
                return SendResult(success=False, error="empty message")
            # Sentinel keeps a trailing _keep_typing tick from posting a fresh marker
            # that stop_typing would then delete and tombstone. Cleared in
            # on_processing_complete.
            if patched_typing:
                self._typing_messages[chat_id] = _TYPING_CONSUMED_SENTINEL
            return last_result
        finally:
            self.resume_typing_for_chat(chat_id)

    async def send_card(self, chat_id: str, card: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        body: Dict[str, Any] = {"cardsV2": [card]}
        thread_id = self._resolve_thread_id(None, metadata, chat_id=chat_id)
        if thread_id:
            body["thread"] = {"name": thread_id}
        try:
            result = await self._create_message(chat_id, body)
            result.raw_response = result.raw_response or {"cardsV2": body["cardsV2"]}
            return result
        except HttpError as exc:
            return SendResult(
                success=False,
                error=_redact_sensitive(str(exc)),
                retryable=_http_status(exc) in _RETRYABLE_HTTP_STATUSES,
            )
        except Exception as exc:
            logger.debug("[GoogleChat] send_card failed", exc_info=True)
            return SendResult(success=False, error=_redact_sensitive(str(exc)), retryable=_is_retryable_error(exc))

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str, session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not choices:
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata)

        buttons: List[Dict[str, Any]] = []
        for choice in choices:
            choice_text = str(choice).strip()
            if not choice_text:
                continue
            label = choice_text if len(choice_text) <= 80 else choice_text[:77] + "..."
            buttons.append({
                "text": label,
                "action": "hermes_clarify",
                "parameters": {"clarify_id": clarify_id, "choice": choice_text},
            })
        buttons.append({
            "text": "Other / type answer",
            "action": "hermes_clarify",
            "parameters": {"clarify_id": clarify_id, "choice": "__other__"},
        })
        card = card_spec_to_cards_v2({
            "card_id": f"clarify-{clarify_id}",
            "header": {"title": "Question"},
            "sections": [{
                "widgets": [
                    {"type": "text", "text": f"❓ {question}"},
                    {"type": "buttons", "buttons": buttons},
                ]
            }],
        })
        result = await self.send_card(chat_id, card, metadata=metadata)
        if result.success:
            self._clarify_state[clarify_id] = session_key
            return result
        return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        """Edit a sent message via ``messages.patch``.

        Required by the tool-progress + token-streaming pipeline (``GatewayStreamConsumer``
        gates on this override). ``finalize`` is unused: Chat's patch has no
        streaming lifecycle. 404/403 are reported as non-success so the gateway
        falls back to ``send()``.
        """
        if not message_id:
            return SendResult(success=False, error="missing message_id")
        if len(content) > _MAX_TEXT_LENGTH:
            content = content[: _MAX_TEXT_LENGTH - 1] + "…"
        try:
            return await self._patch_message(message_id, {"text": content})
        except HttpError as exc:
            if _http_status(exc) == 429:
                self._note_rate_limit(chat_id)
            return SendResult(success=False, error=_redact_sensitive(str(exc)))
        except Exception as exc:
            logger.debug("[GoogleChat] edit_message failed", exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message. Prefer ``edit_message`` internally — deletion leaves a
        "Message deleted by its author" tombstone; provided for stream-consumer
        fallback paths that genuinely need removal."""
        if not message_id:
            return False

        def _do_delete() -> None:
            self._chat_api.spaces().messages().delete(name=message_id).execute(http=self._new_authed_http())

        try:
            await asyncio.to_thread(_do_delete)
            return True
        except HttpError as exc:
            if _http_status(exc) in {403, 404}:
                return False
            logger.debug("[GoogleChat] delete_message failed: %s", _redact_sensitive(str(exc)))
            return False
        except Exception:
            logger.debug("[GoogleChat] delete_message failed", exc_info=True)
            return False

    async def _patch_message(self, message_name: str, body: Dict[str, Any]) -> SendResult:
        """Update a message's text (and optionally cards) in-place."""
        update_mask = ",".join(k for k in ("text", "cardsV2") if k in body) or "text"
        # Patch body cannot carry thread (immutable).
        patch_body = {k: v for k, v in body.items() if k not in {"thread",}}

        def _do_patch() -> Dict[str, Any]:
            return (
                self._chat_api.spaces()
                .messages()
                .patch(name=message_name, updateMask=update_mask, body=patch_body)
                .execute(http=self._new_authed_http())
            )

        resp = await asyncio.to_thread(_do_patch)
        return SendResult(success=True, message_id=resp.get("name", message_name))

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        if len(text) <= _MAX_TEXT_LENGTH:
            return [text]
        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= _MAX_TEXT_LENGTH:
                chunks.append(remaining)
                break
            # Split on a newline near the cutoff when one exists past the midpoint.
            cut = remaining.rfind("\n", 0, _MAX_TEXT_LENGTH)
            if cut < _MAX_TEXT_LENGTH // 2:
                cut = _MAX_TEXT_LENGTH
            chunks.append(remaining[:cut])
            remaining = remaining[cut:].lstrip()
        return chunks

    @classmethod
    def format_message(cls, content: str) -> str:
        """Convert standard Markdown to Google Chat's dialect (see ``cards.format_message``)."""
        return _format_message(content)

    def _resolve_thread_id(
        self, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], chat_id: Optional[str] = None
    ) -> Optional[str]:
        """Return the thread resource name to reply under, or None.

        Priority: ``metadata['thread_id']`` (gateway session plumbing) →
        ``thread_name`` / ``thread_ts`` aliases → ``reply_to`` when it is already a
        ``spaces/X/threads/Y`` name → ``_last_inbound_thread[chat_id]`` (DM replies
        would otherwise land top-level, disconnected from the user's question).
        Cron deliveries (``job_id`` in metadata) skip the last fallback so output
        starts a fresh top-level message instead of burying itself in a stale thread.
        """
        if metadata:
            for key in ("thread_id", "thread_name", "thread_ts"):
                value = metadata.get(key)
                if value:
                    return str(value)
        if reply_to and "/threads/" in reply_to and "/messages/" not in reply_to:
            return reply_to
        if metadata and metadata.get("job_id"):
            return None
        if chat_id:
            cached = self._last_inbound_thread.get(chat_id)
            if cached:
                return cached
        return None

    def _new_authed_http(self) -> Any:
        """Fresh AuthorizedHttp per call: httplib2 shares SSL state, so the discovery
        client is not thread-safe across ``asyncio.to_thread`` workers."""
        return AuthorizedHttp(self._credentials, http=httplib2.Http(timeout=30))

    async def _call_with_retry(self, sync_fn: Callable[[], Any], *, op_name: str = "chat-api-call") -> Any:
        """Run ``sync_fn`` in a thread with bounded retry + jittered backoff.

        Only transient failures (see ``_is_retryable_error``) are retried; permanent
        ones bubble up on the first attempt. Cancellation propagates immediately.
        """
        delay = _RETRY_BASE_DELAY
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                return await asyncio.to_thread(sync_fn)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_retryable_error(exc) or attempt >= _RETRY_MAX_ATTEMPTS:
                    raise
                jitter = delay * _RETRY_JITTER * random.random()
                wait = min(delay + jitter, _RETRY_MAX_DELAY + _RETRY_JITTER)
                logger.warning(
                    "[GoogleChat] %s attempt %d/%d failed (%s); retrying in %.2fs",
                    op_name, attempt, _RETRY_MAX_ATTEMPTS, _redact_sensitive(str(exc)), wait,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, _RETRY_MAX_DELAY)

    def _track_outbound_thread(self, chat_id: str, resp: Dict[str, Any]) -> None:
        """Count the outbound destination thread so a later user "Reply in thread"
        on the bot's message resolves as a known side-thread (prev_count >= 1)
        instead of looking fresh and being misclassified as main flow."""
        resp_thread = (resp.get("thread") or {}).get("name") or ""
        if chat_id and resp_thread:
            try:
                self._thread_count_store.incr(chat_id, resp_thread)
            except Exception:
                logger.debug("[GoogleChat] outbound thread-count incr failed", exc_info=True)

    async def _create_message(self, chat_id: str, body: Dict[str, Any]) -> SendResult:
        """POST spaces/{space}/messages via REST, returning SendResult.

        With ``thread.name`` in the body we MUST pass
        ``messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD`` — the default
        silently ignores thread.name and starts a new thread. FALLBACK (vs OR_FAIL)
        still delivers when a stale thread no longer exists.
        """
        kwargs: Dict[str, Any] = {"parent": chat_id, "body": body}
        if (body.get("thread") or {}).get("name"):
            kwargs["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

        def _do_create() -> Dict[str, Any]:
            return self._chat_api.spaces().messages().create(**kwargs).execute(http=self._new_authed_http())

        resp = await self._call_with_retry(_do_create, op_name="messages.create")
        self._track_outbound_thread(chat_id, resp)
        return SendResult(success=True, message_id=resp.get("name"))

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Post a visible 'Hermes is thinking…' marker message (Chat has no typing API).

        ``send()`` PATCHes the marker with the real reply; ``on_processing_complete``
        reaps it otherwise. The card must be created in the user's thread because
        ``messages.patch`` cannot move a message between threads.

        Cancellation safety: ``_keep_typing`` wraps this in ``wait_for(timeout=1.5)``.
        A cancelled ``to_thread`` create would still land a card we never recorded,
        and the next tick would create a second one. So the slot is reserved with
        an in-flight Event and the create runs in a shielded background task that
        records the msg_id regardless of our own cancellation.
        """
        # Already have a card (real msg_id, sentinel, or in-flight) — bail.
        if chat_id in self._typing_messages:
            return
        if chat_id in self._typing_card_inflight:
            # Bounded wait for the running create so "the card is up when we return".
            with contextlib.suppress(asyncio.TimeoutError, KeyError):
                await asyncio.wait_for(self._typing_card_inflight[chat_id].wait(), timeout=5.0)
            return

        thread_id = self._resolve_thread_id(reply_to=None, metadata=metadata, chat_id=chat_id)
        body: Dict[str, Any] = {"text": getattr(self.config, "typing_status_text", None) or "Hermes is thinking…"}
        if thread_id:
            body["thread"] = {"name": thread_id}

        completed = asyncio.Event()
        self._typing_card_inflight[chat_id] = completed

        async def _create_and_record() -> None:
            try:
                result = await self._create_message(chat_id, body)
                if result.success and result.message_id:
                    if chat_id not in self._typing_messages:
                        self._typing_messages[chat_id] = result.message_id
                    else:
                        # send() or another create claimed the slot first: this card
                        # is an orphan; on_processing_complete cleans it up.
                        self._orphan_typing_messages.setdefault(chat_id, []).append(result.message_id)
            except Exception:
                logger.debug("[GoogleChat] send_typing background create failed", exc_info=True)
            finally:
                self._typing_card_inflight.pop(chat_id, None)
                completed.set()

        task = asyncio.create_task(_create_and_record())
        # The shielded task keeps running if our awaiter is cancelled.
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise

    async def stop_typing(self, chat_id: str) -> None:
        """NO-OP when a live card is tracked: the marker is a real message that
        ``send()`` patches in place, and upstream calls ``stop_typing`` BEFORE
        ``send()`` (deleting would tombstone and force a fresh message). Only the
        SENTINEL is popped so the next turn starts clean; stranded cards on
        error/cancel paths are reaped by ``on_processing_complete``."""
        current = self._typing_messages.get(chat_id)
        if not current:
            return
        if current == _TYPING_CONSUMED_SENTINEL:
            self._typing_messages.pop(chat_id, None)
            return
        # Real message_name — leave it for send() to patch. Deliberate no-op.
        return

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Reap typing card(s) after the turn: pop the SENTINEL on success; on
        failure/cancel patch a still-tracked card to a final label (no tombstone);
        patch orphan cards (background creates that lost a race with send()) to "·"."""
        if event.source is None:
            return
        chat_id = event.source.chat_id
        try:
            current = self._typing_messages.pop(chat_id, None)
            if current and current != _TYPING_CONSUMED_SENTINEL:
                label = "(interrupted)" if outcome == ProcessingOutcome.CANCELLED else "(no reply)"
                try:
                    await self._patch_message(current, {"text": label})
                except Exception:
                    logger.debug("[GoogleChat] on_processing_complete patch fallback failed", exc_info=True)
            for orphan_id in self._orphan_typing_messages.pop(chat_id, []):
                try:
                    await self._patch_message(orphan_id, {"text": "·"})
                except Exception:
                    logger.debug("[GoogleChat] orphan typing-card patch failed: %s", orphan_id, exc_info=True)
        except Exception:
            logger.debug("[GoogleChat] cleanup in on_processing_complete failed", exc_info=True)

    # ------------------------------------------------------------------
    # Attachment send paths
    # ------------------------------------------------------------------
    async def _consume_typing_card_with_text(self, chat_id: str, text: str) -> Optional[SendResult]:
        """Patch the tracked typing card with ``text`` (no tombstone).

        Returns None when there is no real card (caller creates a new message) —
        the SENTINEL is left in place so ``_keep_typing`` doesn't post a fresh card
        during a subsequent attachment send. Raises transient HttpErrors.
        """
        current = self._typing_messages.get(chat_id)
        if not current or current == _TYPING_CONSUMED_SENTINEL:
            return None
        self._typing_messages.pop(chat_id, None)
        try:
            result = await self._patch_message(current, {"text": text})
            self._typing_messages[chat_id] = _TYPING_CONSUMED_SENTINEL
            return result
        except HttpError as exc:
            if _http_status(exc) == 404:
                return None  # card disappeared — caller creates a new message
            raise

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline image via URL (no upload); patches the typing card when tracked."""
        thread_id = self._resolve_thread_id(reply_to, metadata, chat_id=chat_id)
        text = "\n".join(([caption] if caption else []) + [image_url])
        try:
            patched = await self._consume_typing_card_with_text(chat_id, text)
            if patched is not None:
                return patched
            body: Dict[str, Any] = {"text": text}
            if thread_id:
                body["thread"] = {"name": thread_id}
            return await self._create_message(chat_id, body)
        except HttpError as exc:
            return SendResult(success=False, error=_redact_sensitive(str(exc)))

    async def _send_file_reply(
        self, chat_id: str, path: str, caption: Optional[str], reply_to: Optional[str], kwargs: Dict[str, Any],
        mime_hint: Optional[str], override_filename: Optional[str] = None,
    ) -> SendResult:
        thread_id = self._resolve_thread_id(reply_to, kwargs.get("metadata"), chat_id=chat_id)
        return await self._send_file(
            chat_id, path, caption, mime_hint=mime_hint, thread_id=thread_id, override_filename=override_filename
        )

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs: Any
    ) -> SendResult:
        return await self._send_file_reply(chat_id, image_path, caption, reply_to, kwargs, "image/*")

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs: Any,
    ) -> SendResult:
        return await self._send_file_reply(chat_id, file_path, caption, reply_to, kwargs, None, file_name)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs: Any
    ) -> SendResult:
        return await self._send_file_reply(chat_id, audio_path, caption, reply_to, kwargs, "audio/ogg")

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs: Any
    ) -> SendResult:
        return await self._send_file_reply(chat_id, video_path, caption, reply_to, kwargs, "video/mp4")

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Google Chat has no native animation type; fall back to send_image."""
        return await self.send_image(chat_id, animation_url, caption=caption, reply_to=reply_to, metadata=metadata)

    # ------------------------------------------------------------------
    # Native attachment delivery via user OAuth
    #
    # media.upload hard-rejects SA authentication, so the user grants the bot
    # chat.messages.create ONCE via ``/setup-files`` and the bot uploads AS the
    # user. See https://developers.google.com/chat/api/guides/auth/users and
    # ``oauth.py``.
    # ------------------------------------------------------------------
    _LEGACY_USER_IDENTITY = "__legacy__"

    async def _load_per_user_chat_api(self, email: str) -> Optional[Any]:
        """Get (or build + cache) a user-authed Chat client for ``email``.

        Cache hit → refresh creds (eviction on refresh failure so the next request
        goes back through disk / the text-notice fallback). Miss → load token from
        disk, build the client, cache both.
        """
        from .oauth import (
            load_user_credentials as _load,
            build_user_chat_service as _build,
            refresh_or_none as _refresh,
        )

        cached_api = self._user_chat_api_by_email.get(email)
        cached_creds = self._user_creds_by_email.get(email)
        if cached_api is not None and cached_creds is not None:
            try:
                refreshed = await asyncio.to_thread(_refresh, cached_creds, email)
            except Exception:
                logger.debug("[GoogleChat] cached per-user refresh raised", exc_info=True)
                refreshed = None
            if refreshed is None:
                self._user_chat_api_by_email.pop(email, None)
                self._user_creds_by_email.pop(email, None)
                return None
            self._user_creds_by_email[email] = refreshed
            return cached_api

        try:
            creds = await asyncio.to_thread(_load, email)
            if creds is None:
                return None
            api = await asyncio.to_thread(lambda: _build(creds))
        except Exception:
            logger.debug("[GoogleChat] per-user creds load/build failed for %s", email, exc_info=True)
            return None

        self._user_creds_by_email[email] = creds
        self._user_chat_api_by_email[email] = api
        return api

    async def _acquire_user_chat_api(self, sender_email: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve the user-authed Chat client for an outbound attachment.

        Order: per-user token for ``sender_email`` → legacy single-user fallback →
        ``(None, None)`` (caller posts the setup notice). The identity label
        (email or ``"__legacy__"``) tells ``_invalidate_user_creds`` which slot to evict.
        """
        if sender_email:
            api = await self._load_per_user_chat_api(sender_email)
            if api is not None:
                return api, sender_email

        if self._user_chat_api is not None:
            try:
                from .oauth import refresh_or_none as _refresh
                refreshed = await asyncio.to_thread(_refresh, self._user_credentials, None)
            except Exception:
                logger.debug("[GoogleChat] legacy creds refresh raised", exc_info=True)
                refreshed = None
            if refreshed is None:
                logger.warning("[GoogleChat] legacy user-OAuth refresh returned None — evicting fallback creds")
                self._user_credentials = None
                self._user_chat_api = None
                return None, None
            self._user_credentials = refreshed
            return self._user_chat_api, self._LEGACY_USER_IDENTITY

        return None, None

    def _invalidate_user_creds(self, identity: Optional[str]) -> None:
        """Drop creds for ``identity`` (email or ``__legacy__``) after an auth failure."""
        if not identity:
            return
        if identity == self._LEGACY_USER_IDENTITY:
            self._user_credentials = None
            self._user_chat_api = None
            return
        self._user_creds_by_email.pop(identity, None)
        self._user_chat_api_by_email.pop(identity, None)

    async def _send_file(
        self, chat_id: str, path: str, caption: Optional[str], mime_hint: Optional[str],
        thread_id: Optional[str] = None, override_filename: Optional[str] = None,
    ) -> SendResult:
        """Native Chat attachment: user-authed ``media.upload`` then ``messages.create``.

        Uses the most recent inbound sender's OAuth token for this chat (legacy
        single-user token as fallback; text notice when neither exists). Since
        ``messages.patch`` cannot add an attachment, the typing card is patched
        with the caption (or a single space) so it retires without a tombstone.
        """
        if not os.path.exists(path):
            return SendResult(success=False, error=f"file not found: {path}")

        filename = override_filename or os.path.basename(path) or "upload.bin"
        mime = mime_hint or "application/octet-stream"
        sender_email = self._last_sender_by_chat.get(chat_id)
        chat_api, identity = await self._acquire_user_chat_api(sender_email)
        if chat_api is None:
            return await self._post_attachment_fallback(chat_id, path, filename, caption, thread_id)

        try:
            await self._consume_typing_card_with_text(chat_id, caption or " ")
        except Exception:
            logger.debug("[GoogleChat] _send_file pre-patch typing-card failed", exc_info=True)

        def _upload() -> Dict[str, Any]:
            media = MediaFileUpload(path, mimetype=mime, resumable=False)
            return chat_api.media().upload(parent=chat_id, body={"filename": filename}, media_body=media).execute()

        try:
            upload_resp = await asyncio.to_thread(_upload)
        except HttpError as exc:
            status = _http_status(exc)
            if status in {401, 403}:
                logger.warning(
                    "[GoogleChat] media.upload auth failure for identity=%s "
                    "(token revoked or scope missing) — falling back to "
                    "text notice. Status=%s", identity, status,
                )
                self._invalidate_user_creds(identity)
                return await self._post_attachment_fallback(chat_id, path, filename, caption, thread_id)
            return SendResult(success=False, error=_redact_sensitive(str(exc)))

        attachment_ref = upload_resp.get("attachmentDataRef")
        if not attachment_ref:
            return SendResult(success=False, error="upload returned no attachmentDataRef")

        body: Dict[str, Any] = {"attachment": [{"attachmentDataRef": attachment_ref}]}
        if caption:
            body["text"] = caption
        if thread_id:
            body["thread"] = {"name": thread_id}
        # The attachmentDataRef is bound to the uploading principal, so this create
        # also needs user auth. messageReplyOption: see _create_message.
        create_kwargs: Dict[str, Any] = {"parent": chat_id, "body": body}
        if thread_id:
            create_kwargs["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

        def _create_with_attachment() -> Dict[str, Any]:
            return chat_api.spaces().messages().create(**create_kwargs).execute()

        try:
            resp = await asyncio.to_thread(_create_with_attachment)
            self._track_outbound_thread(chat_id, resp)
            return SendResult(success=True, message_id=resp.get("name"))
        except HttpError as exc:
            return SendResult(success=False, error=_redact_sensitive(str(exc)))

    async def _post_attachment_fallback(
        self, chat_id: str, path: str, filename: str, caption: Optional[str], thread_id: Optional[str]
    ) -> SendResult:
        """Post the ``/setup-files`` notice (plus host path) when native delivery is
        unavailable. Always returns ``success=False``."""
        lines = []
        if caption:
            lines.append(caption)
        lines.extend([
            f"⚠️ No he podido adjuntar **{filename}**.",
            "Google Chat sólo permite adjuntar archivos cuando el bot tiene "
            "permiso explícito tuyo (OAuth de usuario). Es un consentimiento "
            "único que se hace desde este chat.",
            "**Para activarlo:** envía `/setup-files` y sigue las instrucciones.",
            f"Mientras tanto el archivo está en el host: `{path}`",
        ])
        body: Dict[str, Any] = {"text": "\n".join(lines)}
        if thread_id:
            body["thread"] = {"name": thread_id}
        try:
            await self._create_message(chat_id, body)
        except Exception:
            logger.debug("[GoogleChat] attachment fallback notice send failed", exc_info=True)
        return SendResult(
            success=False,
            error="google_chat: native attachment requires user OAuth — run /setup-files in chat",
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return {name, type, chat_id} for a space."""
        try:
            info = await asyncio.to_thread(
                lambda: self._chat_api.spaces().get(name=chat_id).execute(http=self._new_authed_http())
            )
        except HttpError as exc:
            logger.debug("[GoogleChat] get_chat_info failed: %s", _redact_sensitive(str(exc)))
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        space_type = (info.get("spaceType") or info.get("type") or "").upper()
        return {
            "name": info.get("displayName") or chat_id,
            "type": "dm" if space_type in {"DIRECT_MESSAGE", "DM"} else "group",
            "chat_id": chat_id,
        }


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _validate_config(config: PlatformConfig) -> bool:
    """Plugin-side config gate for HTTP callback or Pub/Sub inbound modes."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("http_events_url") or (extra.get("project_id") and extra.get("subscription_name")))


def _env_inbound_settings() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(project, subscription, http_events_url) from the scoped env, with legacy fallbacks."""
    project = _get_scoped_secret("GOOGLE_CHAT_PROJECT_ID") or _get_scoped_secret("GOOGLE_CLOUD_PROJECT")
    subscription = _get_scoped_secret("GOOGLE_CHAT_SUBSCRIPTION_NAME") or _get_scoped_secret("GOOGLE_CHAT_SUBSCRIPTION")
    return project, subscription, _get_scoped_secret("GOOGLE_CHAT_HTTP_EVENTS_URL")


def _check_for_registry() -> bool:
    """``check_fn`` for the registry pass: deps installed AND minimum inbound env set,
    so an unconfigured user never sees ``google_chat`` auto-enabled."""
    if not check_google_chat_requirements():
        return False
    project, subscription, http_events_url = _env_inbound_settings()
    return bool(http_events_url or (project and subscription))


def _is_connected(config: PlatformConfig) -> bool:
    """``GatewayConfig.get_connected_platforms()`` polls this."""
    return bool(getattr(config, "enabled", False)) and _validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed ``PlatformConfig.extra`` from env vars during ``_apply_env_overrides``
    (before the adapter exists, so ``gateway status`` reflects env-only config).

    Returns None when the minimum inbound settings are absent. ``home_channel`` is
    turned into a ``HomeChannel`` by the core hook rather than merged into extra.
    """
    project, subscription, http_events_url = _env_inbound_settings()
    if not (http_events_url or (project and subscription)):
        return None
    seed: Dict[str, Any] = {}
    for extra_name, value in (
        ("project_id", project),
        ("subscription_name", subscription),
        ("http_events_url", http_events_url),
        ("http_events_audience", _get_scoped_secret("GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE")),
        ("http_events_service_account_email", _get_scoped_secret("GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL")),
        ("max_messages", _get_scoped_secret("GOOGLE_CHAT_MAX_MESSAGES")),
        ("max_bytes", _get_scoped_secret("GOOGLE_CHAT_MAX_BYTES")),
        ("bootstrap_spaces", _get_scoped_secret("GOOGLE_CHAT_BOOTSTRAP_SPACES")),
        ("debug_raw", _get_scoped_secret("GOOGLE_CHAT_DEBUG_RAW")),
        (
            "service_account_json",
            _get_scoped_secret("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON") or _get_scoped_secret("GOOGLE_APPLICATION_CREDENTIALS"),
        ),
    ):
        if value:
            seed[extra_name] = value
    home = _get_scoped_secret("GOOGLE_CHAT_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {"chat_id": home, "name": _get_scoped_secret("GOOGLE_CHAT_HOME_CHANNEL_NAME", "Home")}
    return seed


_SETUP_WALKTHROUGH = """Google Chat needs a GCP project, a Pub/Sub topic + subscription,
and a Service Account with Pub/Sub Subscriber on the subscription.
Walkthrough:
  1. Create or select a GCP project; enable Google Chat API + Cloud Pub/Sub API.
  2. Create a Service Account (no project-level IAM role needed).
  3. Create a Pub/Sub topic (e.g. hermes-chat-events) and a Pull subscription.
  4. On the TOPIC: add chat-api-push@system.gserviceaccount.com as Pub/Sub Publisher.
  5. On the SUBSCRIPTION: grant your Service Account Pub/Sub Subscriber.
  6. Download the Service Account JSON key.
  7. Google Chat API console → Configuration: connection = Cloud Pub/Sub,
     point at the topic, enable 1:1 + group, restrict visibility.
  8. Install the bot in a space (fires ADDED_TO_SPACE and resolves its user_id).

Full guide: website/docs/user-guide/messaging/google_chat.md
"""


def interactive_setup() -> None:
    """``hermes setup`` wizard for plugin platforms: print GCP instructions, prompt
    for env vars, persist them to ``~/.hermes/.env``."""
    from hermes_cli.cli_output import print_info, print_success, print_warning, prompt, prompt_yes_no
    from hermes_cli.config import get_env_value, save_env_value

    existing_sub = get_env_value("GOOGLE_CHAT_SUBSCRIPTION_NAME")
    if existing_sub:
        print_info(f"Google Chat: already configured (subscription: {existing_sub})")
        if not prompt_yes_no("Reconfigure Google Chat?", False):
            return
    for line in _SETUP_WALKTHROUGH.splitlines():
        print_info(line)

    for question, env_name, required_label, password in (
        ("GCP project ID (e.g. my-project)", "GOOGLE_CHAT_PROJECT_ID", "Project ID", False),
        ("Pub/Sub subscription (projects/<proj>/subscriptions/<sub>)", "GOOGLE_CHAT_SUBSCRIPTION_NAME", "Subscription", False),
        ("Path to Service Account JSON (or inline JSON)", "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", None, True),
    ):
        value = prompt(question, default=get_env_value(env_name) or "", password=password)
        if not value:
            if required_label is None:
                continue
            print_warning(f"{required_label} is required — skipping Google Chat setup")
            return
        save_env_value(env_name, value.strip())

    if prompt_yes_no("Restrict access to specific users? (recommended)", True):
        allowed = prompt("Allowed user emails (comma-separated)", default=get_env_value("GOOGLE_CHAT_ALLOWED_USERS") or "")
        if allowed:
            save_env_value("GOOGLE_CHAT_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("GOOGLE_CHAT_ALLOWED_USERS", "")
    else:
        save_env_value("GOOGLE_CHAT_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — anyone who can DM the bot can command it.")

    home = prompt(
        "Home space for cron/notification delivery (e.g. spaces/AAAA, or empty)",
        default=get_env_value("GOOGLE_CHAT_HOME_CHANNEL") or "",
    )
    if home:
        save_env_value("GOOGLE_CHAT_HOME_CHANNEL", home.strip())

    print()
    print_success("Google Chat configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway: hermes gateway restart")


# Strict resource-name pattern: anything outside Chat's documented character set
# is a tampered chat_id trying to break out of the REST URL path.
_GCHAT_CHAT_ID_RE = re.compile(r"^(?:spaces|users)/[A-Za-z0-9_-]+$")

_STANDALONE_SA_ERRORS = {
    "inline_invalid": "Google Chat standalone send: inline SA JSON is invalid: {exc}",
    "not_found": "Google Chat standalone send: SA JSON file not found at {path}",
    "file_invalid": "Google Chat standalone send: SA JSON file is invalid: {exc}",
    "adc_foreign": (
        "Google Chat standalone send: ADC skipped for this profile: service-account credentials are set in "
        "the process environment but not in this profile's secret scope"
    ),
    "adc_no_auth": (
        "Google Chat standalone send: no SA credentials configured and google-auth is not installed for ADC fallback"
    ),
    "adc_failed": (
        "Google Chat standalone send: no SA credentials configured and Application Default Credentials are "
        "unavailable: {exc}"
    ),
}


async def _standalone_send(
    pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """POST a single Chat message via REST without the SDK.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway runner
    is not in this process (e.g. ``hermes cron``). Needs SA credentials
    (``GOOGLE_CHAT_SERVICE_ACCOUNT_JSON`` / ``GOOGLE_APPLICATION_CREDENTIALS`` /
    ADC) and a validated space resource name. ``media_files`` / ``force_document``
    are accepted for signature parity only; attachments send as text.
    """
    if not chat_id:
        return {"error": "Google Chat standalone send: chat_id (space resource) is required"}
    if not _GCHAT_CHAT_ID_RE.match(chat_id):
        return {"error": (
            f"Google Chat standalone send: chat_id {chat_id!r} must match "
            f"'spaces/<id>' or 'users/<id>' with only [A-Za-z0-9_-] in the id"
        )}
    if thread_id is not None and not re.match(r"^spaces/[A-Za-z0-9_-]+/threads/[A-Za-z0-9_-]+$", thread_id):
        return {"error": (
            f"Google Chat standalone send: thread_id {thread_id!r} must match "
            f"'spaces/<id>/threads/<id>'"
        )}

    extra = getattr(pconfig, "extra", {}) or {}
    sa_value = (
        extra.get("service_account_json")
        or _get_scoped_secret("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON")
        or _get_scoped_secret("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if service_account is None:
        return {"error": "Google Chat standalone send: google-auth not installed"}
    try:
        from google.auth.transport.requests import Request as _GoogleAuthRequest
    except Exception as e:
        return {"error": f"Google Chat standalone send: google-auth import failed: {e}"}

    try:
        creds = _load_sa_credentials_from(sa_value)
    except _SACredentialError as err:
        return {"error": _STANDALONE_SA_ERRORS[err.kind].format(exc=err.detail, path=sa_value)}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"error": f"Google Chat standalone send: credential load failed: {e}"}

    # Bound the synchronous token refresh so a hung STS endpoint can't stall cron.
    try:
        await asyncio.wait_for(asyncio.to_thread(creds.refresh, _GoogleAuthRequest()), timeout=10.0)
    except asyncio.TimeoutError:
        return {"error": "Google Chat standalone send: token refresh timed out"}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return {"error": f"Google Chat standalone send: token refresh failed: {e}"}

    token = getattr(creds, "token", None)
    if not token:
        return {"error": "Google Chat standalone send: refreshed credentials have no token"}

    body: Dict[str, Any] = {"text": message}
    if thread_id:
        body["thread"] = {"name": thread_id}
    url = f"https://chat.googleapis.com/v1/{chat_id}/messages"
    try:
        import aiohttp as _aiohttp
    except ImportError:
        return {"error": "Google Chat standalone send: aiohttp not installed"}

    try:
        async with _aiohttp.ClientSession(timeout=_aiohttp.ClientTimeout(total=30.0), trust_env=gateway_trust_env()) as session:
            async with session.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    return {"error": f"Google Chat standalone send: API returned {resp.status}: {text[:300]}"}
                payload = await resp.json()
        return {"success": True, "message_id": payload.get("name")}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Google Chat standalone send raised", exc_info=True)
        return {"error": f"Google Chat standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point: registers the ``google_chat`` adapter with the platform
    registry (consulted by the gateway's ``_create_adapter`` before its built-ins)."""
    ctx.register_platform(
        name="google_chat",
        label="Google Chat",
        adapter_factory=lambda cfg: GoogleChatAdapter(cfg),
        check_fn=_check_for_registry,
        validate_config=_validate_config,
        is_connected=_is_connected,
        required_env=["GOOGLE_CHAT_SERVICE_ACCOUNT_JSON"],
        install_hint="Run `hermes setup` to install Google Chat support.",
        setup_fn=interactive_setup,
        # Seeds PlatformConfig.extra + home_channel from env during _apply_env_overrides.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery for ``deliver=google_chat`` jobs.
        cron_deliver_env_var="GOOGLE_CHAT_HOME_CHANNEL",
        # Out-of-process cron delivery via the Chat REST API.
        standalone_sender_fn=_standalone_send,
        allowed_users_env="GOOGLE_CHAT_ALLOWED_USERS",
        allow_all_env="GOOGLE_CHAT_ALLOW_ALL_USERS",
        # Chat caps text at 4096; margin for typing-marker patches and edit overhead.
        max_message_length=4000,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are on Google Chat. Limited markdown subset is rendered: "
            "*bold*, _italic_, ~strike~, `code`. No headings or lists. "
            "Message size limit: 4000 characters; longer responses are split "
            "across multiple messages. You are in a space (DM or group). "
            "Images render inline; audio, video, and document attachments "
            "render as download cards (no native voice/video UI). To send "
            "files, include MEDIA:/absolute/path/to/file in your response. "
            "Native file attachments require the user to run /setup-files "
            "once in their own DM — until they do, file requests fall back "
            "to a text notice with the host path. Do NOT generate interactive "
            "Card v2 buttons — Google Chat interactivity is not yet supported "
            "by this gateway; ask for typed confirmations instead. While you "
            "are generating a response, a 'Hermes is thinking…' marker message "
            "appears in the space and is deleted once your response is ready. "
            "You do NOT have access to Google Chat-specific APIs — you cannot "
            "search space history, list space members, or manage spaces. Do "
            "not promise to perform these actions; explain that you can only "
            "read messages sent directly to you and respond in the same "
            "space/thread."
        ),
    )
