"""Local / self-hosted model servers: Ollama (native /api/tags probe, headers, base-url resolution), LM Studio (/api/v1/models, load-on-demand), Ollama Cloud (merged live + models.dev catalog with disk cache).

Split out of ``hermes_cli.models``; every moved name is re-imported there, so
``hermes_cli.models.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple, Optional
from hermes_cli.urllib_security import url_origin


def _root_for_ollama_native_api(base_url: str) -> str:
    """Convert an OpenAI-style Ollama base URL to the native API root."""
    root = str(base_url or "").strip().rstrip("/")
    if root.startswith(":"):
        root = "http://127.0.0.1" + root
    elif root and "://" not in root:
        root = "http://" + root
    for suffix in ("/api/tags", "/v1/models", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root


def _normalize_openai_base_url(base_url: Optional[str]) -> str:
    """Add a usable HTTP scheme without changing an OpenAI API path."""
    value = str(base_url or "").strip()
    if value.startswith(":"):
        return "http://127.0.0.1" + value
    if value and "://" not in value:
        return "http://" + value
    return value


def _get_ollama_base_url() -> str:
    """Resolve the local Ollama-compatible endpoint URL.

    Prefer explicit config under ``providers.ollama.base_url`` because this is how local Ollama-
    compatible endpoints can be wired without changing the active model provider. Fall back to
    active ``model.base_url`` only when the active provider is ollama/custom, then to Ollama's local
    default.
    """
    from hermes_cli.models import _get_model_config_dict, _get_provider_config_dict, should_use_ollama_native_catalog
    provider_cfg = _get_provider_config_dict("ollama")
    configured = (
        provider_cfg.get("base_url", "")
        or provider_cfg.get("api", "")
        or provider_cfg.get("url", "")
        or ""
    )
    if configured:
        return str(configured).strip()

    model_cfg = _get_model_config_dict()
    model_provider = str(model_cfg.get("provider", "") or "").strip().lower()
    model_base = str(model_cfg.get("base_url", "") or "").strip()
    if model_provider == "ollama" and model_base:
        return model_base
    if model_provider == "custom" and model_base:
        # Only reuse the active bare custom endpoint when it is actually
        # Ollama-compatible. Otherwise a user working against an unrelated
        # OpenAI-compatible endpoint would make the Ollama picker probe that
        # endpoint's /api/tags and hide their local Ollama catalog.
        try:
            if should_use_ollama_native_catalog("custom", model_base):
                return model_base
        except (OSError, RuntimeError, TypeError, ValueError):
            pass

    env_host = os.getenv("OLLAMA_HOST", "").strip()
    if env_host:
        if env_host.startswith(":") and not env_host.startswith("::"):
            env_host = "127.0.0.1" + env_host
        elif env_host.startswith("[") and env_host.endswith("]"):
            env_host = f"{env_host}:11434"
        elif "://" in env_host:
            try:
                parsed = urllib.parse.urlsplit(env_host)
                if parsed.hostname and parsed.port is None:
                    hostname = parsed.hostname
                    if ":" in hostname and not hostname.startswith("["):
                        hostname = f"[{hostname}]"
                    userinfo = (
                        parsed.netloc.rsplit("@", 1)[0] + "@"
                        if "@" in parsed.netloc
                        else ""
                    )
                    env_host = parsed._replace(
                        netloc=f"{userinfo}{hostname}:11434"
                    ).geturl()
            except ValueError:
                pass
        elif env_host.count(":") > 1 and not env_host.startswith("["):
            env_host = f"[{env_host}]:11434"
        elif ":" not in env_host:
            env_host = f"{env_host}:11434"
        return env_host
    return "http://localhost:11434"


def _get_ollama_request_headers() -> dict[str, str]:
    """Return configured headers and credentials for native Ollama requests."""
    from hermes_cli.models import _get_provider_config_dict
    entry = _get_provider_config_dict("ollama")
    raw = entry.get("extra_headers")
    try:
        from hermes_cli.config import normalize_extra_headers

        result = normalize_extra_headers(raw)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        result = {}

    api_key = str(entry.get("api_key") or "").strip()
    if not api_key:
        key_env = str(
            entry.get("key_env") or entry.get("api_key_env") or ""
        ).strip()
        api_key = os.getenv(key_env, "").strip() if key_env else ""
    if api_key:
        if not any(key.lower() == "authorization" for key in result):
            result["Authorization"] = f"Bearer {api_key}"
    return result


def _get_ollama_native_headers(
    base_url: Optional[str],
    *,
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Resolve Ollama credentials and headers for one endpoint origin."""
    from hermes_cli.models import _get_ollama_request_headers, _get_provider_config_dict
    entry = _get_provider_config_dict("ollama")
    configured_base = str(
        entry.get("base_url") or entry.get("api") or entry.get("url") or ""
    ).strip()
    explicit_key = str(api_key or "").strip()
    configured_matches = bool(
        configured_base
        and base_url
        and _same_ollama_native_root(base_url, configured_base)
    )
    if not configured_matches and not explicit_key:
        return {}
    headers = _get_ollama_request_headers() if configured_matches else {}
    if explicit_key:
        # A provider-specific key must not inherit any configured Authorization
        # variant from the Ollama origin when both share a native root.
        for key in tuple(headers):
            if key.lower() == "authorization":
                del headers[key]
        headers["Authorization"] = f"Bearer {explicit_key}"
    return headers


_OLLAMA_LOCAL_MODELS_CACHE_TTL: int = 300  # seconds (5 minutes)


_OLLAMA_LOCAL_MODELS_CACHE: dict[str, tuple[tuple[str, ...], float]] = {}


_OLLAMA_LOCAL_PROBE_FAILURE_CACHE: dict[str, float] = {}


_OLLAMA_LOCAL_PROBE_REACHABLE: dict[str, bool] = {}


_OLLAMA_LOCAL_PROBE_FAILURE_TTL: int = 30


_OLLAMA_LOCAL_CACHE_MAX_ENTRIES: int = 256


def _evict_related_ollama_cache_entries(key: str) -> None:
    _OLLAMA_LOCAL_MODELS_CACHE.pop(key, None)
    _OLLAMA_LOCAL_PROBE_REACHABLE.pop(key, None)
    for failure_key in list(_OLLAMA_LOCAL_PROBE_FAILURE_CACHE):
        if failure_key == key or failure_key.startswith(f"{key}|timeout:"):
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)


def _remember_ollama_cache(cache: dict[str, Any], key: str, value: Any) -> None:
    if key not in cache and len(cache) >= _OLLAMA_LOCAL_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(cache))
        _evict_related_ollama_cache_entries(
            oldest_key.split("|timeout:", 1)[0]
        )
    cache[key] = value


def _ollama_probe_cache_key(root: str, headers: Optional[dict[str, str]]) -> str:
    cache_key = root
    if headers:
        import hashlib

        normalized_headers = sorted(
            (str(key).lower(), str(value)) for key, value in headers.items()
        )
        header_blob = json.dumps(
            normalized_headers, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8", errors="replace")
        header_fingerprint = hashlib.blake2b(header_blob, digest_size=8).hexdigest()
        cache_key = f"{root}|headers:{header_fingerprint}"
    return cache_key


def probe_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Probe local Ollama-compatible models from native ``/api/tags``.

    Returns ``None`` when the endpoint cannot be reached or returns malformed data, and a list
    (possibly empty) when ``/api/tags`` was reachable. Stock Ollama exposes its authoritative local
    model catalog at ``/api/tags``; OpenAI-compatible ``/v1/models`` is not required for local
    Ollama servers.
    """
    from hermes_cli.models import _HERMES_USER_AGENT, _get_ollama_base_url, _urlopen_model_catalog_request
    root = _root_for_ollama_native_api(base_url or _get_ollama_base_url())
    if not root:
        return None
    cache_key = _ollama_probe_cache_key(root, headers)
    failure_key = f"{cache_key}|timeout:{float(timeout):.3f}"
    cached = _OLLAMA_LOCAL_MODELS_CACHE.get(cache_key)
    if cached is not None:
        cached_models, cached_at = cached
        if time.monotonic() - cached_at < _OLLAMA_LOCAL_MODELS_CACHE_TTL:
            return list(cached_models)
    failed_at = _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.get(failure_key)
    if failed_at is not None:
        if time.monotonic() - failed_at < _OLLAMA_LOCAL_PROBE_FAILURE_TTL:
            return None
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)

    try:
        url = root.rstrip("/") + "/api/tags"
        request_headers = {"User-Agent": _HERMES_USER_AGENT}
        request_headers.update(headers or {})
        req = urllib.request.Request(url, headers=request_headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (
        ValueError,
        OSError,
        TimeoutError,
        http.client.HTTPException,
        urllib.error.URLError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None

    models: list[str] = []
    seen: set[str] = set()
    for item in raw_models:
        if isinstance(item, dict):
            model_id = str(item.get("model") or item.get("name") or "").strip()
        else:
            _remember_ollama_cache(
                _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
            )
            _remember_ollama_cache(
                _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
            )
            return None
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    if raw_models and not models:
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None
    _remember_ollama_cache(_OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, True)
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)
    _remember_ollama_cache(
        _OLLAMA_LOCAL_MODELS_CACHE,
        cache_key,
        (tuple(models), time.monotonic()),
    )
    return models


def fetch_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch local Ollama-compatible models, preserving probe failure as ``None``."""
    from hermes_cli.models import probe_ollama_local_models
    return probe_ollama_local_models(base_url, timeout, headers=headers)


def _same_ollama_native_root(left: str, right: str) -> bool:
    """Return True when two Ollama/OpenAI-style base URLs share an API root."""
    left_root = _root_for_ollama_native_api(left).rstrip("/")
    right_root = _root_for_ollama_native_api(right).rstrip("/")
    if not left_root or not right_root:
        return False
    try:
        left_parts = urllib.parse.urlsplit(left_root)
        right_parts = urllib.parse.urlsplit(right_root)
        return (
            url_origin(left_root) == url_origin(right_root)
            and left_parts.path.rstrip("/") == right_parts.path.rstrip("/")
        )
    except (AttributeError, ValueError):
        return False


def should_use_ollama_native_catalog(
    provider: Optional[str],
    base_url: Optional[str],
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """Return True when model discovery should use local Ollama ``/api/tags``.

    Bare ``ollama`` is normalized to ``custom`` elsewhere so runtime paths share the OpenAI-
    compatible client, but local Ollama's authoritative model list is ``/api/tags``. Use it when
    the caller asked for Ollama explicitly, the base URL matches ``providers.ollama.base_url``,
    or an ambiguous custom URL on Ollama's default port actually serves ``/api/tags``; other
    custom endpoints keep the ``/models`` probe.
    """
    from hermes_cli.models import _get_provider_config_dict, probe_ollama_local_models
    requested = str(provider or "").strip().lower()
    root = _root_for_ollama_native_api(base_url or "")
    if root:
        try:
            host = (urllib.parse.urlparse(root).hostname or "").lower()
            if host == "ollama.com" or host.endswith(".ollama.com"):
                return False
        except ValueError:
            pass

    known_non_local_providers = {
        "openrouter",
        "nous",
        "anthropic",
        "openai",
        "openai-codex",
        "gemini",
        "ollama-cloud",
    }
    if requested in known_non_local_providers:
        return False

    if requested == "ollama":
        if not root:
            return False
        configured = _get_provider_config_dict("ollama")
        configured_base = str(
            configured.get("base_url")
            or configured.get("api")
            or configured.get("url")
            or ""
        ).strip()
        if configured_base and not _same_ollama_native_root(root, configured_base):
            return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None
        return True

    provider_cfg = _get_provider_config_dict("ollama")
    configured_ollama_base_url = str(
        provider_cfg.get("base_url", "")
        or provider_cfg.get("api", "")
        or provider_cfg.get("url", "")
        or ""
    ).strip()
    if configured_ollama_base_url and _same_ollama_native_root(root, configured_ollama_base_url):
        return True

    if not root:
        return False

    local_like_providers = {"", "custom", "local", "llamacpp", "llama.cpp", "llama-cpp", "vllm"}
    if requested not in local_like_providers and not requested.startswith("custom:"):
        return False

    if requested == "custom:ollama" or requested.endswith("-ollama"):
        return True

    try:
        parsed = urllib.parse.urlparse(root)
        if parsed.port != 11434:
            return False
    except ValueError:
        return False

    return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None


def _ollama_local_catalog(force_refresh: bool) -> list[str]:
    """Catalog for the raw ``ollama`` provider: native ``/api/tags`` when the endpoint is a real
    Ollama server, else the OpenAI-style ``/v1/models`` of the configured gateway."""
    from hermes_cli.models import _get_ollama_base_url, _get_ollama_native_headers, _get_provider_config_dict, fetch_api_models, fetch_ollama_local_models, should_use_ollama_native_catalog
    if force_refresh:
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
    base_url = _get_ollama_base_url()
    headers = _get_ollama_native_headers(base_url)
    if should_use_ollama_native_catalog("ollama", base_url, headers=headers):
        if headers:
            native_models = fetch_ollama_local_models(base_url, headers=headers)
        else:
            native_models = fetch_ollama_local_models(base_url)
        native_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers or None)
        if native_models or _OLLAMA_LOCAL_PROBE_REACHABLE.get(native_key) is True:
            return native_models or []
    # Non-native Ollama-compatible endpoints (incl. Ollama Cloud) and gateways exposing only
    # OpenAI-style /v1/models.
    config = _get_provider_config_dict("ollama")
    fallback_key = str(config.get("api_key") or "").strip()
    if not fallback_key:
        key_env = str(config.get("key_env") or "").strip()
        fallback_key = os.getenv(key_env, "").strip() if key_env else ""
    fallback_base = _normalize_openai_base_url(config.get("base_url") or base_url)
    fallback_headers = _get_ollama_native_headers(fallback_base, api_key=fallback_key)
    return fetch_api_models(fallback_key, fallback_base, headers=fallback_headers or None) or []


def _lmstudio_server_root(base_url: Optional[str]) -> Optional[str]:
    """Return the LM Studio server root for native ``/api/v1`` endpoints.

    Users commonly copy either the OpenAI-compatible runtime URL (``.../v1``) or the native API
    prefix (``.../api`` / ``.../api/v1``). Native probes append ``/api/v1/...`` themselves, so
    normalize all accepted forms back to the bare server root to avoid ``/api/api/v1`` requests.
    """
    root = (base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root or None


def _lmstudio_request_headers(api_key: Optional[str] = None) -> dict:
    """Build HTTP headers for LM Studio native API requests."""
    from hermes_cli.models import _HERMES_USER_AGENT
    headers = {"User-Agent": _HERMES_USER_AGENT}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _lmstudio_fetch_raw_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[dict]]:
    """Fetch the raw model list from LM Studio's ``/api/v1/models``."""
    from hermes_cli.models import _urlopen_model_catalog_request
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    headers = _lmstudio_request_headers(api_key)
    request = urllib.request.Request(server_root + "/api/v1/models", headers=headers)
    try:
        with _urlopen_model_catalog_request(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            from hermes_cli.auth import AuthError
            raise AuthError(
                f"LM Studio rejected the request with HTTP {exc.code}.",
                provider="lmstudio",
                code="auth_rejected",
            ) from exc
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed with HTTP %s", server_root, exc.code,
        )
        return None
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed: %s", server_root, exc,
        )
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s returned malformed payload (no `models` list)",
            server_root,
        )
        return None
    return raw_models


def probe_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Probe LM Studio's model listing.

    Returns chat-capable model keys, including a valid empty list when the server is reachable
    but has no non-embedding models; returns ``None`` on network errors, malformed responses, or
    bad base URLs. Raises ``AuthError`` on HTTP 401/403 so token issues surface separately from
    reachability.
    """
    from hermes_cli.models import _lmstudio_fetch_raw_models
    raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    if raw_models is None:
        return None

    keys: list[str] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("type") or "").strip().lower() == "embedding":
            continue
        key = str(raw.get("key") or raw.get("id") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def fetch_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Fetch LM Studio chat-capable model keys from native ``/api/v1/models``.

    Embedding models are filtered out; network errors, malformed responses, and bad base URLs
    yield an empty list. Raises ``AuthError`` on HTTP 401/403 so callers can distinguish a
    missing or wrong ``LM_API_KEY`` from an unreachable server — the most common LM Studio
    support case.
    """
    from hermes_cli.models import probe_lmstudio_models
    models = probe_lmstudio_models(api_key=api_key, base_url=base_url, timeout=timeout)
    return models or []


class LMStudioLoadResult(NamedTuple):
    """Verified LM Studio runtime plus load-attempt provenance."""

    context_length: Optional[int]
    load_attempted: bool = False
    rejected: bool = False


def ensure_lmstudio_model_loaded(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    target_context_length: Optional[int],
    timeout: float = 120.0,
    *,
    return_load_result: bool = False,
) -> Optional[int] | LMStudioLoadResult:
    """Ensure ``model`` is loaded and return verified runtime context.

    Existing loaded-instance context is authoritative. Cold loads omit ``context_length`` unless the
    caller supplied an explicit override; the returned context must come from LM Studio's echoed or
    refreshed state.
    """
    from hermes_cli.models import _lmstudio_fetch_raw_models, _urlopen_model_catalog_request

    def _result(
        context_length: Optional[int],
        *,
        load_attempted: bool = False,
        rejected: bool = False,
    ) -> Optional[int] | LMStudioLoadResult:
        value = LMStudioLoadResult(context_length, load_attempted, rejected)
        return value if return_load_result else context_length

    def _positive_int(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    def _loaded_context(entry: dict) -> Optional[int]:
        instances = entry.get("loaded_instances")
        if not isinstance(instances, list):
            return None
        for instance in instances:
            config = instance.get("config") if isinstance(instance, dict) else None
            context = config.get("context_length") if isinstance(config, dict) else None
            parsed = _positive_int(context)
            if parsed is not None:
                return parsed
        return None

    def _find_entry(raw_models: list[dict]) -> Optional[dict]:
        for raw in raw_models:
            if isinstance(raw, dict) and (raw.get("key") == model or raw.get("id") == model):
                return raw
        return None

    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return _result(None)

    explicit_context = _positive_int(target_context_length)
    if target_context_length is not None and explicit_context is None:
        return _result(None)

    headers = _lmstudio_request_headers(api_key)

    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=10)
    except Exception:
        raw_models = None
    if raw_models is None:
        return _result(None)

    target_entry = _find_entry(raw_models)
    if target_entry is None:
        return _result(None)

    max_ctx = _positive_int(target_entry.get("max_context_length"))
    if explicit_context is not None and max_ctx is not None and explicit_context > max_ctx:
        return _result(None, rejected=True)

    current_context = _loaded_context(target_entry)
    if current_context is not None:
        return _result(current_context)

    loaded_instances = target_entry.get("loaded_instances")
    if not isinstance(loaded_instances, list) or loaded_instances:
        return _result(None)

    load_payload: dict[str, Any] = {"model": model, "echo_load_config": True}
    if explicit_context is not None:
        load_payload["context_length"] = explicit_context
    body = json.dumps(load_payload).encode()
    load_headers = dict(headers)
    load_headers["Content-Type"] = "application/json"
    try:
        load_request = urllib.request.Request(
            server_root + "/api/v1/models/load",
            data=body,
            headers=load_headers,
            method="POST",
        )
        with _urlopen_model_catalog_request(load_request, timeout=timeout) as resp:
            response_body = resp.read()
    except Exception:
        return _result(None, load_attempted=True)

    try:
        response_payload = json.loads(response_body.decode())
    except Exception:
        response_payload = None
    load_config = response_payload.get("load_config") if isinstance(response_payload, dict) else None
    applied_context = (
        _positive_int(load_config.get("context_length"))
        if isinstance(load_config, dict)
        else None
    )
    if applied_context is not None:
        return _result(applied_context, load_attempted=True)

    try:
        refreshed_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=10)
    except Exception:
        refreshed_models = None
    if refreshed_models is None:
        return _result(None, load_attempted=True)
    refreshed_entry = _find_entry(refreshed_models)
    refreshed_context = _loaded_context(refreshed_entry) if refreshed_entry is not None else None
    return _result(refreshed_context, load_attempted=True)


def lmstudio_model_reasoning_options(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Return the reasoning ``allowed_options`` LM Studio publishes for ``model``.

    Reads ``capabilities.reasoning.allowed_options`` from ``/api/v1/models``; returns ``[]``
    when the model is unknown, the endpoint is unreachable, or no reasoning capability is
    declared.
    """
    from hermes_cli.models import _lmstudio_fetch_raw_models
    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception:
        raw_models = None
    if not raw_models:
        return []

    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") != model and raw.get("id") != model:
            continue
        caps = raw.get("capabilities")
        reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
        opts = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
        if isinstance(opts, list):
            return [str(o).strip().lower() for o in opts if isinstance(o, str)]
        return []
    return []


def ollama_model_supports_thinking(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[bool]:
    """Return True if an Ollama (Cloud or local) model advertises ``thinking``.

    Probes native ``/api/show`` and checks ``capabilities`` — the authoritative source, since
    the OpenAI-compat ``/v1/models`` endpoint omits it. Tri-state: True when ``thinking`` is
    declared, False when the probe succeeded without it, None when the probe failed so the
    caller picks the fallback (treated as "don't emit").
    """
    import httpx

    server_url = (base_url or "").strip().rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    if not server_url:
        return None

    bare_model = _strip_ollama_cloud_suffix((model or "").strip())
    if not bare_model:
        return None

    token = str(api_key or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            caps = resp.json().get("capabilities")
            if isinstance(caps, list):
                return "thinking" in caps
    except Exception:
        return None
    return None


_OLLAMA_CLOUD_CACHE_TTL = 3600  # 1 hour


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    """Strip :cloud / -cloud suffixes that models.dev appends to Ollama Cloud IDs.

    The live API uses clean IDs (e.g. 'kimi-k2.6') while models.dev sometimes returns them as
    'kimi-k2.6:cloud'. Normalising before the dedup merge prevents duplicate entries in the merged
    model list.
    """
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _ollama_cloud_cache_path() -> Path:
    """Return the path for the Ollama Cloud model cache."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "ollama_cloud_models_cache.json"


def _load_ollama_cloud_cache(*, ignore_ttl: bool = False) -> Optional[dict]:
    """Load cached Ollama Cloud models from disk (None when missing, empty, or stale)."""
    from hermes_cli.models import _read_json_cache

    try:
        data = _read_json_cache(_ollama_cloud_cache_path())
        if data is None:
            return None
        models = data.get("models")
        if not (isinstance(models, list) and models):
            return None
        if not ignore_ttl and (time.time() - data.get("cached_at", 0)) > _OLLAMA_CLOUD_CACHE_TTL:
            return None  # stale
        return data
    except Exception:
        return None


def _save_ollama_cloud_cache(models: list[str]) -> None:
    """Persist the merged Ollama Cloud model list to disk. Best-effort."""
    from hermes_cli.models import _write_json_cache

    try:
        _write_json_cache(_ollama_cloud_cache_path(), {"models": models, "cached_at": time.time()}, indent=None)
    except Exception:
        pass


def fetch_ollama_cloud_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Fetch Ollama Cloud models by merging live API + models.dev, with disk cache.

    Resolution order: 1. Disk cache (if fresh, < 1 hour, and not force_refresh) 2. Live
    ``/v1/models`` endpoint (primary — freshest source) 3. models.dev registry (secondary — fills
    gaps for unlisted models) 4. Merge: live models first, then models.dev additions (deduped)

    Returns a list of model IDs (never None — empty list on total failure).
    """
    from hermes_cli.models import fetch_api_models
    # 1. Check disk cache
    if not force_refresh:
        cached = _load_ollama_cloud_cache()
        if cached is not None:
            return cached["models"]

    # 2. Live API probe
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OLLAMA_BASE_URL", "") or "https://ollama.com/v1"

    live_models: list[str] = []
    if api_key:
        result = fetch_api_models(api_key, base_url, timeout=8.0)
        if result:
            live_models = result

    # 3. models.dev registry
    mdev_models: list[str] = []
    try:
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models("ollama-cloud")
    except Exception:
        pass

    # 4. Merge: live first, then models.dev additions (deduped, order-preserving)
    if live_models or mdev_models:
        seen: set[str] = set()
        merged: list[str] = []
        for m in live_models:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        for m in mdev_models:
            normalized = _strip_ollama_cloud_suffix(m)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
        if merged:
            _save_ollama_cloud_cache(merged)
            return merged

    # Total failure — return stale cache if available (ignore TTL)
    stale = _load_ollama_cloud_cache(ignore_ttl=True)
    if stale is not None:
        return stale["models"]

    return []
