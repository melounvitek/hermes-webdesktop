"""Shared model-switching logic for CLI and gateway /model commands.

Both the CLI (cli.py) and gateway (gateway/run.py) /model handlers
share the same core pipeline:

  parse flags -> alias resolution -> provider resolution ->
  credential resolution -> normalize model name ->
  metadata lookup -> build result

This module ties together the foundation layers:

- ``agent.models_dev``            -- models.dev catalog, ModelInfo, ProviderInfo
- ``hermes_cli.providers``        -- canonical provider identity + overlays
- ``hermes_cli.model_normalize``  -- per-provider name formatting

Provider switching uses the ``--provider`` flag exclusively.
No colon-based ``provider:model`` syntax — colons are reserved for
OpenRouter variant suffixes (``:free``, ``:extended``, ``:fast``).
"""

from __future__ import annotations

import http.client
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, NamedTuple, Optional

from hermes_cli.providers import (
    ProviderDef,
    custom_provider_aliases,
    custom_provider_slug,
    determine_api_mode,
    get_label,
    host_mandated_api_mode,
    is_aggregator,
    resolve_provider_full,
)
from hermes_cli.model_normalize import (
    normalize_model_for_provider,
)
from agent.models_dev import (
    ModelCapabilities,
    ModelInfo,
    get_model_capabilities,
    get_model_info,
    list_provider_models,
)
from utils import base_url_host_matches, base_url_hostname, base_url_origin

# Providers whose picker model list should NOT be capped by max_models.
# OpenCode Zen / Go are aggregators whose full catalogs (70+ models each) must
# be visible so users can pick any model they have access to.
_UNCAPPED_PICKER_PROVIDERS: frozenset[str] = frozenset({"opencode-zen", "opencode-go"})

logger = logging.getLogger(__name__)


def _declared_model_ids(value: Any) -> list[str]:
    """Return configured model IDs from supported config shapes.

    Accepts:
    - ``{"model-id": {...}}``
    - ``["model-a", "model-b"]``
    - ``[{"id": "model-a"}, {"name": "model-b"}]``
    - ``"model-a"``
    """
    ids: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        model_id = candidate.strip()
        if not model_id:
            return
        lowered = model_id.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        ids.append(model_id)

    if isinstance(value, str):
        _add(value)
        return ids

    if isinstance(value, dict):
        for model_id in value:
            # Backward compat: pre-fix Hermes wrote sentinel keys inside the
            # user-facing ``models`` mapping. Never list them as model IDs.
            if model_id in {
                "__explicit_model_allowlist__",
                "__discovered_model_catalog__",
            }:
                continue
            _add(model_id)
        return ids

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add(item)
                continue
            if isinstance(item, dict):
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    model_id = item.get("name")
                _add(model_id)
        return ids

    return ids


def _entry_models_discovered(entry: Any) -> bool:
    """True when the entry's ``models`` mapping was auto-discovered by Hermes.

    The current shape is an entry-level ``models_discovered: true`` sibling of
    ``models``. Older Hermes versions wrote an in-mapping
    ``__discovered_model_catalog__: true`` sentinel instead — accept that on
    read for backward compatibility (the next discovery save migrates the
    entry to the clean shape).
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("models_discovered") is True:
        return True
    models = entry.get("models")
    return (
        isinstance(models, dict)
        and models.get("__discovered_model_catalog__") is True
    )


def _models_config_is_allowlist(value: Any, discovered: bool = False) -> bool:
    """Return True when ``models:`` is an intentional ID allowlist.

    A mapping like ``{model_id: {context_length: N}}`` is per-model *metadata*
    written by ``_save_custom_provider`` / the ``hermes model`` wizard — not a
    catalog narrow. Treating that shape as an allowlist made Desktop/Telegram
    pickers show only the saved default for local Ollama (no ``api_key``),
    while ``hermes model`` still live-probed the full ``/v1/models`` list.
    Refresh could not help because the same gate skipped probing.

    List/string shapes remain allowlists for no-key endpoints. To pin a
    dict-shaped catalog, set ``discover_models: false``.

    ``discovered`` is the entry-level ``models_discovered`` flag (see
    ``_entry_models_discovered``): a catalog Hermes itself persisted after a
    successful probe is never a user pin, whatever its shape.
    """
    if discovered:
        return False
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return False
    if isinstance(value, (list, tuple)):
        return bool(_declared_model_ids(value))
    return False


def _save_discovered_models_to_config(
    api_url: str,
    model_ids: list[str],
    *,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Persist discovered models into ``custom_providers`` in config.yaml.

    Called after a successful ``/v1/models`` probe so that the next read
    with ``discover_models: false`` uses the cached list instead of a stale
    or minimal manually-configured subset.

    Matches entries by ``base_url`` (trailing-slash-normalised).  A failed
    config write is swallowed — the picker still shows the live models for
    this session.
    """
    if not api_url or not model_ids:
        return
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "")).strip()
            if entry_url.rstrip("/").lower() != norm_url:
                continue
            entry_mode = str(
                entry.get("api_mode") or entry.get("transport") or ""
            ).strip().lower() or None
            if entry_mode != api_mode:
                continue
            if headers is not None:
                entry_headers = _extra_headers_from_config(entry)
                if entry_headers != headers:
                    continue
            existing = entry.get("models")
            legacy_discovered = (
                isinstance(existing, dict)
                and existing.get("__discovered_model_catalog__") is True
            )
            entry_discovered = (
                entry.get("models_discovered") is True or legacy_discovered
            )
            # Preserve per-model metadata: when ``models`` is a mapping
            # (e.g. ``{"model-a": {"context_length": 8192}}``) or a list of
            # dicts (e.g. ``[{"id": "model-a", "context_length": 8192}]``),
            # the user has curated metadata per model — do not replace it.
            # A mapping Hermes itself discovered (``models_discovered: true``
            # or the legacy in-mapping sentinel) is ours to refresh.
            if isinstance(existing, dict) and not entry_discovered:
                continue
            if isinstance(existing, list) and any(
                isinstance(m, dict) for m in existing
            ):
                continue
            # Only update when models are stale — avoids unnecessary
            # config writes on every picker open.  A legacy-shape entry
            # (sentinel inside ``models``) is always rewritten so the next
            # save migrates it to the clean entry-level flag.
            if isinstance(existing, list) and existing == model_ids:
                continue
            if (
                isinstance(existing, dict)
                and entry_discovered
                and not legacy_discovered
                and list(existing) == model_ids
            ):
                continue
            entry["models"] = {model_id: {} for model_id in model_ids}
            entry["models_discovered"] = True
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


def _bare_custom_provider_def(current_base_url: str) -> Optional[ProviderDef]:
    """ProviderDef for a direct ``model.provider: custom`` endpoint."""
    base_url = str(current_base_url or "").strip()
    if not base_url:
        return None
    return ProviderDef(
        id="custom",
        name="Custom endpoint",
        transport="openai_chat",
        api_key_env_vars=(),
        base_url=base_url,
        is_aggregator=False,
        auth_type="api_key",
        source="model-config",
    )


_MODEL_DISCOVERY_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    http.client.HTTPException,
)


class _NativePickerModelList(list[str]):
    """A successful native catalog, including an authoritative empty one."""


def _fetch_picker_live_models(
    api_key: str,
    api_url: str,
    native_catalog_provider: str,
    preserve_native_models: bool,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
    api_mode: str | None = None,
) -> list[str] | None:
    """Fetch picker models with native Ollama and cached generic discovery."""
    from hermes_cli.models import (
        _get_ollama_native_headers,
        _normalize_openai_base_url,
        cached_fetch_api_models,
        fetch_ollama_local_models,
        should_use_ollama_native_catalog,
    )

    candidate_headers = _get_ollama_native_headers(api_url, api_key=api_key)
    caller_has_authorization = any(
        key.lower() == "authorization" for key in (headers or {})
    )
    if caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
    if headers:
        for key in tuple(candidate_headers):
            if any(key.lower() == existing.lower() for existing in headers):
                del candidate_headers[key]
        candidate_headers.update(headers)
    if api_key and not caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(
        native_catalog_provider, api_url, headers=candidate_headers or None
    )
    resolved_headers = candidate_headers or None if use_native else headers

    if use_native:
        if preserve_native_models:
            return None
        native_models = fetch_ollama_local_models(
            api_url, timeout=timeout, headers=resolved_headers
        )
        if native_models is not None:
            return _NativePickerModelList(native_models)
        # A failed native probe is not authoritative: retry the cached generic
        # OpenAI-compatible catalog before reporting no models.
        return cached_fetch_api_models(
            api_key,
            _normalize_openai_base_url(api_url),
            timeout=timeout,
            headers=resolved_headers,
            api_mode=api_mode,
        )
    generic_models = cached_fetch_api_models(
        api_key,
        api_url,
        timeout=timeout,
        headers=resolved_headers,
        api_mode=api_mode,
    )
    return generic_models if generic_models else None


# ---------------------------------------------------------------------------
# Non-agentic model warning
# ---------------------------------------------------------------------------

_HERMES_MODEL_WARNING = (
    "Nous Research Hermes 3 & 4 models are NOT agentic and are not designed "
    "for use with Hermes Agent. They lack the tool-calling capabilities "
    "required for agent workflows. Consider using an agentic model instead "
    "(Claude, GPT, Gemini, DeepSeek, etc.)."
)

# Match only the real Nous Research Hermes 3 / Hermes 4 chat families.
# The previous substring check (`"hermes" in name.lower()`) false-positived on
# unrelated local Modelfiles like ``hermes-brain:qwen3-14b-ctx16k`` that just
# happen to carry "hermes" in their tag but are fully tool-capable.
#
# Positive examples the regex must match:
#   NousResearch/Hermes-3-Llama-3.1-70B, hermes-4-405b, openrouter/hermes3:70b
# Negative examples it must NOT match:
#   hermes-brain:qwen3-14b-ctx16k, qwen3:14b, claude-opus-4-6
_NOUS_HERMES_NON_AGENTIC_RE = re.compile(
    r"(?:^|[/:])hermes[-_ ]?[34](?:[-_.:]|$)",
    re.IGNORECASE,
)


# Opaque internal model-ID display
# ---------------------------------------------------------------------------
# Some proxies (notably Palantir Foundry's LLM-proxy) identify models by
# resource-instance IDs that are deeply nested, verbose, and pure noise to
# read in CLI status output, e.g.:
#
#   ri.language-model-service..language-model.anthropic-claude-4-7-opus
#
# The provider_label (e.g. "palantir-claude46") already carries the routing
# context, so the only useful information left in the opaque ID is the
# trailing slug. Strip the boilerplate prefix for *display* — never for
# wire-side comparison, persistence, config writes, alias lookup, or
# anything that round-trips back into the API.
#
# Match by substring on a known prefix so we never accidentally truncate
# a legitimate model name that happens to contain dots.

_OPAQUE_MODEL_PREFIXES: tuple[str, ...] = (
    "ri.language-model-service..language-model.",
)


def format_model_for_display(model_name: str) -> str:
    """Return a human-friendly form of *model_name* for CLI status output.

    Strips known opaque proxy prefixes (Palantir Foundry's
    ``ri.language-model-service..language-model.*``) and returns the
    trailing slug. Falls through to the original string for everything
    else, so real model IDs (``claude-4-7-opus-20260101``,
    ``gpt-5-4``, ``meta-llama/Llama-3.3-70B-Instruct``) are untouched.

    This is a DISPLAY-ONLY helper. Do NOT use the return value for any
    wire-side operation — the proxy expects the full opaque ID, and
    callers that compare or persist must keep the original.
    """
    if not model_name:
        return model_name
    for prefix in _OPAQUE_MODEL_PREFIXES:
        if model_name.startswith(prefix):
            tail = model_name[len(prefix):]
            return tail if tail else model_name
    return model_name


# ---------------------------------------------------------------------------
def is_nous_hermes_non_agentic(model_name: str) -> bool:
    """Return True if *model_name* is a real Nous Hermes 3/4 chat model.

    Used to decide whether to surface the non-agentic warning at startup.
    Callers in :mod:`cli.py` and here should go through this single helper
    so the two sites don't drift.
    """
    if not model_name:
        return False
    return bool(_NOUS_HERMES_NON_AGENTIC_RE.search(model_name))


def _check_hermes_model_warning(model_name: str) -> str:
    """Return a warning string if *model_name* is a Nous Hermes 3/4 chat model."""
    if is_nous_hermes_non_agentic(model_name):
        return _HERMES_MODEL_WARNING
    return ""


# ---------------------------------------------------------------------------
# Model aliases -- short names -> (vendor, family) with NO version numbers.
# Resolved dynamically against the live models.dev catalog.
# ---------------------------------------------------------------------------

class ModelIdentity(NamedTuple):
    """Vendor slug and family prefix used for catalog resolution."""
    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelIdentity] = {
    # Anthropic
    "sonnet":    ModelIdentity("anthropic", "claude-sonnet"),
    "opus":      ModelIdentity("anthropic", "claude-opus"),
    "haiku":     ModelIdentity("anthropic", "claude-haiku"),
    "claude":    ModelIdentity("anthropic", "claude"),

    # OpenAI
    "gpt5":      ModelIdentity("openai", "gpt-5"),
    "gpt":       ModelIdentity("openai", "gpt"),
    "codex":     ModelIdentity("openai", "codex"),
    "o3":        ModelIdentity("openai", "o3"),
    "o4":        ModelIdentity("openai", "o4"),

    # Google
    "gemini":    ModelIdentity("google", "gemini"),

    # DeepSeek
    "deepseek":  ModelIdentity("deepseek", "deepseek-chat"),

    # X.AI
    "grok":      ModelIdentity("x-ai", "grok"),

    # Meta
    "llama":     ModelIdentity("meta-llama", "llama"),

    # Qwen / Alibaba
    "qwen":      ModelIdentity("qwen", "qwen"),

    # MiniMax
    "minimax":   ModelIdentity("minimax", "minimax"),

    # Nvidia
    "nemotron":  ModelIdentity("nvidia", "nemotron"),

    # Moonshot / Kimi
    "kimi":      ModelIdentity("moonshotai", "kimi"),

    # Z.AI / GLM
    "glm":       ModelIdentity("z-ai", "glm"),

    # Step Plan (StepFun)
    "step":      ModelIdentity("stepfun", "step"),

    # Xiaomi
    "mimo":      ModelIdentity("xiaomi", "mimo"),

    # Arcee
    "trinity":   ModelIdentity("arcee-ai", "trinity"),
}


# ---------------------------------------------------------------------------
# Direct aliases — exact model+provider+base_url for endpoints that aren't
# in the models.dev catalog (e.g. Ollama Cloud, local servers).
# Checked BEFORE catalog resolution.  Format:
#   alias -> (model_id, provider, base_url)
# These can also be loaded from config.yaml ``model_aliases:`` section.
# ---------------------------------------------------------------------------

class DirectAlias(NamedTuple):
    """Exact model mapping that bypasses catalog resolution.

    ``api_key`` / ``key_env`` carry the alias endpoint's OWN credential.
    Without them the switch keeps whatever key the *default* provider
    resolved, which 401s against the alias host and sends that provider's
    secret to an unrelated third party (#83612).
    """
    model: str
    provider: str
    base_url: str
    # Defaulted so existing positional construction —
    # ``DirectAlias(model, provider, base_url)`` — keeps working for callers
    # and for the string-format aliases built below.
    api_key: str = ""
    key_env: str = ""


# Built-in direct aliases (can be extended via config.yaml model_aliases:)
_BUILTIN_DIRECT_ALIASES: dict[str, DirectAlias] = {}

# Merged dict (builtins + user config); populated by _load_direct_aliases()
DIRECT_ALIASES: dict[str, DirectAlias] = {}


def _load_direct_aliases() -> dict[str, DirectAlias]:
    """Load direct aliases from config.yaml ``model_aliases:`` section.

    Config format::

        model_aliases:
          qwen:
            model: "qwen3.5:397b"
            provider: custom
            base_url: "https://ollama.com/v1"
          minimax:
            model: "minimax-m2.7"
            provider: custom
            base_url: "https://ollama.com/v1"
          theta:
            model: "theta-1"
            provider: custom
            base_url: "https://theta.example.com/v1"
            api_key: "sk-..."          # literal, or "${THETA_API_KEY}"
            key_env: "THETA_API_KEY"   # read from the environment instead

    ``api_key``/``key_env`` are the alias endpoint's own credential. When
    neither is set the key is resolved from the alias HOST, never from the
    previously active provider (#83612).

    Also reads ``model.aliases`` (set by ``hermes config set model.aliases.xxx``
    or hand-written). String entries (``ds-flash: deepseek/deepseek-v4-flash``)
    are converted into DirectAlias objects with the provider parsed from the
    ``provider/`` prefix in the value; if no slash, the current provider is
    used. Dict entries use the same shape as ``model_aliases:`` (``model``,
    ``provider``, ``base_url`` keys).
    """
    merged = dict(_BUILTIN_DIRECT_ALIASES)
    try:
        from hermes_cli.config import load_config
        cfg = load_config()

        # --- model_aliases (dict-based format) ---
        user_aliases = cfg.get("model_aliases")
        if isinstance(user_aliases, dict):
            for name, entry in user_aliases.items():
                if not isinstance(entry, dict):
                    continue
                model = entry.get("model", "")
                provider = entry.get("provider", "custom")
                base_url = entry.get("base_url", "")
                if model:
                    merged[name.strip().lower()] = DirectAlias(
                        model=model, provider=provider, base_url=base_url,
                        api_key=str(entry.get("api_key", "") or "").strip(),
                        key_env=str(entry.get("key_env", "") or "").strip(),
                    )

        # --- model.aliases (from config set / hand-written config) ---
        model_section = cfg.get("model", {})
        if isinstance(model_section, dict):
            simple_aliases = model_section.get("aliases")
            if isinstance(simple_aliases, dict):
                current_provider = model_section.get("provider", "")
                for name, value in simple_aliases.items():
                    key = name.strip().lower()
                    if not key or key in merged:
                        continue  # don't override explicit model_aliases entries
                    if isinstance(value, dict):
                        # Dict form mirrors the ``model_aliases:`` shape:
                        # localqwen: {model: qwen3.5:4b, provider: custom}.
                        # Hand-written configs already use it; honoring it
                        # here keeps aliases with an explicit provider from
                        # being silently dropped (#87189).
                        model = str(value.get("model") or "").strip()
                        if not model:
                            continue
                        provider = str(value.get("provider") or "").strip()
                        merged[key] = DirectAlias(
                            model=model,
                            provider=provider or current_provider or "custom",
                            base_url=str(value.get("base_url") or "").strip(),
                        )
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    val = value.strip()
                    if "/" in val:
                        provider, model = val.split("/", 1)
                    else:
                        provider = current_provider
                        model = val
                    merged[key] = DirectAlias(
                        model=model.strip(),
                        provider=provider.strip() or current_provider,
                        base_url="",
                    )
    except Exception:
        pass
    return merged


# Identity of the config the cached aliases were built from. The cache is
# process-global but its source is profile-local, so it must be keyed or the
# first profile to resolve an alias pins its definitions — and, since entries
# carry `api_key`, its credentials — for every later profile in the process.
# Same shape `load_config()` already keys its own cache on, so a profile
# switch (HERMES_HOME moves, so the path moves) and a config/key rotation
# (mtime/size move) both invalidate.
_DIRECT_ALIAS_IDENTITY: Optional[tuple] = None
# A copy of what this loader last produced. Callers and tests seed
# DIRECT_ALIASES both by rebinding the module attribute AND by editing it in
# place, so neither the object's identity nor a "did we load" flag can tell
# our own stale cache from someone else's contents. Comparing against what we
# actually wrote does: if the dict no longer holds it, the entries are not
# ours to discard.
_DIRECT_ALIAS_LOADED: Optional[dict] = None


def _direct_alias_source_identity() -> Optional[tuple]:
    """Identity of the active profile's alias source, or None if unknowable.

    None means "do not reuse the cache" — a source we cannot identify must
    not be assumed to be the one already loaded.
    """
    try:
        from hermes_constants import get_config_path

        path = get_config_path()
        try:
            stat = path.stat()
        except OSError:
            # A missing config is still a definite identity for this profile.
            return (str(path), None, None)
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except Exception:
        return None


def _ensure_direct_aliases() -> None:
    """Load direct aliases for the ACTIVE profile, caching per config identity.

    Mutates the existing DIRECT_ALIASES dict in place rather than rebinding
    the module attribute. This keeps `from hermes_cli.model_switch import
    DIRECT_ALIASES` references valid in callers — rebinding would leave them
    pointing at a stale empty dict.
    """
    global _DIRECT_ALIAS_IDENTITY, _DIRECT_ALIAS_LOADED
    identity = _direct_alias_source_identity()
    if DIRECT_ALIASES and (
        # Contents are not what we loaded — seeded or edited by a caller.
        # Not ours to discard.
        DIRECT_ALIASES != _DIRECT_ALIAS_LOADED
        # Ours, and still the same config file at the same signature.
        or (identity is not None and identity == _DIRECT_ALIAS_IDENTITY)
    ):
        return
    loaded = _load_direct_aliases()
    # clear()+update() rather than a rebind: callers hold this exact dict.
    DIRECT_ALIASES.clear()
    DIRECT_ALIASES.update(loaded)
    _DIRECT_ALIAS_IDENTITY = identity
    _DIRECT_ALIAS_LOADED = dict(loaded)


def direct_alias_api_key(alias: DirectAlias) -> str:
    """Resolve a direct alias's own credential, or "" when it has none.

    Precedence, highest first — ``api_key`` always wins over ``key_env``, so
    an entry carrying both is not ambiguous:

    1. ``api_key: "${VAR}"`` — indirection, read from the environment.
    2. ``api_key: "sk-..."`` — literal.
    3. ``key_env: VAR`` — read from the environment.
    4. otherwise "" — the caller resolves from the alias host instead.
    Environment reads go through the per-profile secret scope for the same
    reason the user-provider branch does: a raw ``os.environ`` read hands
    this profile whatever key the process env holds — another profile's,
    under the multiplexed gateway.
    """
    raw = (alias.api_key or "").strip()
    if raw.startswith("${") and raw.endswith("}"):
        return _scoped_key_env(raw[2:-1].strip())
    if raw:
        return raw
    return _scoped_key_env((alias.key_env or "").strip())


def direct_alias_runtime_request(alias: DirectAlias) -> tuple[str, Optional[str]]:
    """Return ``(requested_provider, explicit_api_key)`` for resolving *alias*.

    Single owner of the invariant that a URL-bearing direct alias resolves its
    credential for the alias HOST, never for its provider label. A label like
    ``anthropic`` on an unrelated URL would otherwise reach that provider's
    explicit-runtime branch, keep the foreign URL, and fall back to the live
    vendor token. Bare ``custom`` is host-gated (#28660), so an authoritative
    URL still resolves its vendor key and a foreign one resolves none.

    An alias with no base_url keeps its label: there is no foreign host to
    protect against, and the label is the only routing information there is.
    """
    key = direct_alias_api_key(alias) or None
    if alias.base_url:
        return "custom", key
    return (alias.provider or "custom"), key


# Hosts where plaintext HTTP is not a downgrade — a local server has no
# network hop to intercept.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _may_reuse_session_credential(session_base_url: str, alias_base_url: str) -> bool:
    """Whether the session's key may follow a switch to *alias_base_url*.

    Same hostname is NOT sufficient to authorise handing a bearer secret to a
    new URL. ``http://h`` and ``https://h:8443`` are different origins and
    different trust boundaries, so an alias that keeps the hostname but drops
    the scheme would otherwise put a live session credential on the wire in
    the clear. Require an identical (scheme, host, port), and refuse plaintext
    outside loopback.
    """
    session = base_url_origin(session_base_url)
    alias = base_url_origin(alias_base_url)
    if not session[1] or session != alias:
        return False
    scheme, hostname, _ = alias
    return scheme == "https" or hostname in _LOOPBACK_HOSTS


class StartupModelRoute(NamedTuple):
    """Model/provider pair resolved before an agent is constructed."""

    model: str
    provider: str = ""
    base_url: str = ""
    api_key: str = ""


def resolve_startup_model_route(
    raw_model: str,
    *,
    explicit_provider: str = "",
    current_provider: str = "",
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None,
) -> Optional[StartupModelRoute]:
    """Resolve aliases and configured ``provider/model`` input at startup.

    ``HermesCLI`` is constructed before the interactive ``/model`` pipeline
    runs.  Keeping this small resolver at the same boundary as
    ``DIRECT_ALIASES`` prevents startup from attaching the configured default
    provider to an explicitly requested model. Provider/model strings are
    consumed only for providers present in user configuration; aggregator
    namespaces remain untouched.

    ``current_provider`` is the provider the session would otherwise use
    (config ``model.provider`` / ``--provider``). When it is a routing
    aggregator and the raw string is an aggregator-native slug
    (``anthropic/claude-opus-4.6`` on OpenRouter), the input stays on the
    aggregator — bare vendor slugs resolve WITHIN the aggregator first and a
    ``providers:`` block for the same vendor must not steal the route.
    """
    raw = str(raw_model or "").strip()
    if not raw:
        return None

    _ensure_direct_aliases()
    direct = DIRECT_ALIASES.get(raw.lower())
    if direct is not None:
        if explicit_provider:
            # An explicit --provider wins over the alias's own label; the
            # alias contributes model/base_url only.
            return StartupModelRoute(
                model=direct.model,
                provider=explicit_provider,
                base_url=direct.base_url,
            )
        # Resolve through the SAME owner the interactive /model and oneshot
        # paths use: a URL-bearing alias must resolve its credential for the
        # alias HOST, never for its provider label — a label like
        # ``anthropic`` on a foreign URL would otherwise reach that
        # provider's explicit-runtime branch and put the live vendor token
        # on the foreign wire (#28660).
        alias_provider, alias_key = direct_alias_runtime_request(direct)
        return StartupModelRoute(
            model=direct.model,
            provider=alias_provider,
            base_url=direct.base_url,
            api_key=alias_key or "",
        )

    if explicit_provider or "/" not in raw:
        return None
    prefix, model = (part.strip() for part in raw.split("/", 1))
    if not prefix or not model:
        return None

    # Aggregator-native slugs stay on the aggregator. A user on OpenRouter
    # whose config also has a ``providers.anthropic`` block must NOT have
    # ``anthropic/claude-opus-4.6`` silently rerouted to native Anthropic.
    if current_provider:
        try:
            from hermes_cli.providers import (
                is_routing_aggregator as _is_routing_agg,
                normalize_provider as _norm_prov,
            )

            if _is_routing_agg(_norm_prov(current_provider)):
                from hermes_cli.models import _find_openrouter_slug

                if _find_openrouter_slug(raw):
                    return None
        except Exception:
            pass

    configured = {
        str(name).strip().lower()
        for name in (user_providers or {})
        if str(name).strip()
    }
    configured.update(
        f"custom:{entry.get('name', '').strip().lower()}"
        for entry in (custom_providers or [])
        if isinstance(entry, dict) and str(entry.get("name") or "").strip()
    )
    try:
        from hermes_cli.models import normalize_provider

        canonical = normalize_provider(prefix)
    except Exception:
        canonical = prefix.lower()

    if prefix.lower() in configured:
        provider = prefix
    elif canonical.lower() in configured:
        provider = canonical
    else:
        return None

    if is_aggregator(canonical):
        return None
    return StartupModelRoute(model=model, provider=provider)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""

    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    request_overrides: Optional[dict] = None
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Optional[ModelCapabilities] = None
    runtime_capabilities: Optional[dict[str, bool]] = None
    model_info: Optional[ModelInfo] = None
    is_global: bool = False


@dataclass(frozen=True)
class ModelFlagParseResult:
    """Parsed flags for a /model command."""

    model_input: str
    explicit_provider: str = ""
    is_global: bool = False
    force_refresh: bool = False
    is_session: bool = False
    is_once: bool = False
# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

def parse_model_flags_detailed(raw_args: str) -> ModelFlagParseResult:
    """Parse flags from /model command args.

    Returns a :class:`ModelFlagParseResult`. ``--once`` is intentionally
    parsed here but interpreted by each caller because each frontend has its
    own live-session restore hook.

    ``is_global`` and ``is_session`` are independent flag presences; the
    *effective* persistence decision is resolved by
    :func:`resolve_persist_behavior` so the config-gated default
    (``model.persist_switch_by_default``) is applied in one place.

    Examples::

        "sonnet"                         -> ("sonnet", "", False, False, False)
        "sonnet --global"                -> ("sonnet", "", True, False, False)
        "sonnet --session"               -> ("sonnet", "", False, False, True)
        "sonnet --once"                  -> is_once=True
        "sonnet --provider anthropic"    -> ("sonnet", "anthropic", False, False, False)
        "--provider my-ollama"           -> ("", "my-ollama", False, False, False)
        "--refresh"                      -> ("", "", False, True, False)
        "sonnet --provider anthropic --global" -> ("sonnet", "anthropic", True, False, False)
    """
    is_global = False
    explicit_provider = ""
    force_refresh = False
    is_session = False
    is_once = False

    # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
    # A single Unicode dash before a flag keyword becomes "--"
    import re as _re
    raw_args = _re.sub(r'[\u2012\u2013\u2014\u2015](provider|global|session|refresh|once)', r'--\1', raw_args)

    # Keep this hand-rolled because model IDs may contain colons/slashes and
    # the historical parser did not require shell quoting.
    parts = raw_args.split()
    i = 0
    filtered: list[str] = []
    while i < len(parts):
        if parts[i] == "--global":
            is_global = True
            i += 1
        elif parts[i] == "--session":
            is_session = True
            i += 1
        elif parts[i] == "--refresh":
            force_refresh = True
            i += 1
        elif parts[i] == "--once":
            is_once = True
            i += 1
        elif parts[i] == "--provider" and i + 1 < len(parts):
            explicit_provider = parts[i + 1]
            i += 2
        else:
            filtered.append(parts[i])
            i += 1

    model_input = " ".join(filtered).strip()
    return ModelFlagParseResult(
        model_input=model_input,
        explicit_provider=explicit_provider,
        is_global=is_global,
        force_refresh=force_refresh,
        is_session=is_session,
        is_once=is_once,
    )


def parse_model_flags(raw_args: str) -> tuple[str, str, bool, bool, bool]:
    """Parse legacy /model flags and return the historical 5-tuple.

    New call sites that care about ``--once`` should use
    :func:`parse_model_flags_detailed`.
    """
    parsed = parse_model_flags_detailed(raw_args)
    return (
        parsed.model_input,
        parsed.explicit_provider,
        parsed.is_global,
        parsed.force_refresh,
        parsed.is_session,
    )


def resolve_persist_behavior(
    is_global: bool,
    is_session: bool,
    is_once: bool = False,
    explicit_provider: str = "",
) -> bool:
    """Decide whether a ``/model`` switch should persist to ``config.yaml``.

    Resolution order:

    1. ``--once`` explicitly opts out → ``False`` (next turn only).
    2. ``--session`` explicitly opts out → ``False`` (this session only).
    3. ``--global`` explicitly opts in → ``True``.
    4. No default configured yet (neither ``model.default`` nor
       ``model.provider`` set — a fresh install whose first-ever pick this
       is) → ``True``.  Without a persisted provider, ``resolve_provider``
       falls through to whatever ``*_API_KEY`` env var is lying around on
       the next launch (#86414), so the first pick becomes the default
       instead of evaporating.  Applies to every surface (CLI, gateway,
       Desktop picker) so no client has to hardcode ``--global``.
    5. ``--provider`` given without an explicit persist flag → ``False``
       (session only).  Provider switches are typically exploratory — the
       user is trying a different backend for this conversation, not
       reconfiguring the default.  ``--global`` can still force persist.
    6. Otherwise defer to ``model.persist_switch_by_default`` in
       ``config.yaml`` (defaults to ``False``: a plain ``/model <name>``
       affects only the current session).  Users who want the old
       persist-by-default behavior can set the key to ``true``; a one-off
       ``--global`` always persists.

    The config read is defensive: on a fresh install ``model`` may be a
    flat string rather than a dict, in which case the built-in default
    (``False``) applies.
    """
    if is_once:
        return False
    if is_session:
        return False
    if is_global:
        return True
    try:
        from hermes_cli.config import load_config

        model_cfg = load_config().get("model")
    except Exception:
        return False
    if isinstance(model_cfg, dict):
        if not (model_cfg.get("default") or model_cfg.get("provider")):
            return True
        if explicit_provider:
            return False
        return bool(model_cfg.get("persist_switch_by_default", False))
    # Flat-string form: a non-empty string IS a configured default.
    return not model_cfg


# ---------------------------------------------------------------------------
# Single-owner /model request parsing + effective-model resolution
# ---------------------------------------------------------------------------
#
# Historically each surface (cli.py, gateway/slash_commands.py,
# tui_gateway/server.py) re-implemented flag parsing + conflict checks, and
# each resolution surface (gateway/run.py, gateway/platforms/api_server.py)
# re-implemented the session-override > channel/session > global precedence.
# Commit 7dd00bb47d had to re-fix the api_server discarding session-persisted
# models precisely because the precedence rule lived in two places.  The
# helpers below are the ONE owner; surfaces map error codes to their own
# user-facing copy but never re-derive the semantics.

# Error codes emitted by parse_model_switch_args().
MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL = "once_with_global"
MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET = "once_requires_target"

# Canonical (surface-neutral) error copy.  Surfaces prepend their own
# decoration ("  ✗ " in the CLI, "❌ " in the gateway) but MUST NOT change
# the core sentence — it is shared user-visible copy.
MODEL_SWITCH_ERROR_TEXT = {
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL: "/model --once cannot be combined with --global",
    MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET: "/model --once requires a model or provider.",
}


@dataclass(frozen=True)
class ModelSwitchRequest:
    """A fully parsed /model command request.

    ``scope`` is the *requested* persistence scope derived purely from the
    flags: ``"once"`` | ``"session"`` | ``"global"`` | ``"default"`` (no
    explicit scope flag; the effective decision then belongs to
    :func:`resolve_persist_behavior`, which also reads config).

    ``errors`` carries error *codes* (see ``MODEL_SWITCH_ERR_*``); surfaces
    render them via :data:`MODEL_SWITCH_ERROR_TEXT` plus their own prefix.
    """

    raw: str
    target: str
    explicit_provider: str = ""
    is_global: bool = False
    is_session: bool = False
    is_once: bool = False
    force_refresh: bool = False
    scope: str = "default"
    errors: tuple = ()

    # Compat properties so a ModelSwitchRequest can be passed anywhere a
    # ModelFlagParseResult was accepted (e.g. tui_gateway._apply_model_switch).
    @property
    def model_input(self) -> str:
        return self.target

    @property
    def flags(self) -> "ModelFlagParseResult":
        return ModelFlagParseResult(
            model_input=self.target,
            explicit_provider=self.explicit_provider,
            is_global=self.is_global,
            force_refresh=self.force_refresh,
            is_session=self.is_session,
            is_once=self.is_once,
        )

    def error_messages(self) -> list:
        """Canonical (undercorated) error strings for this request."""
        return [MODEL_SWITCH_ERROR_TEXT[code] for code in self.errors]


def parse_model_switch_args(raw: str) -> ModelSwitchRequest:
    """Parse a raw /model argument string into a :class:`ModelSwitchRequest`.

    The ONE parser for every /model surface.  Wraps
    :func:`parse_model_flags_detailed` (tokenization + Unicode-dash
    normalization) and layers on the flag-conflict validation that cli.py,
    gateway/slash_commands.py, and tui_gateway/server.py each used to
    re-implement:

    * ``--once`` + ``--global``  → ``MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL``
    * ``--once`` with no model and no ``--provider``
      → ``MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET``

    Model targets pass through untouched: bare names (``sonnet``),
    aggregator slugs (``vendor/model``), and colon forms (``vendor:model``)
    are all resolved later by :func:`switch_model` (aggregator-aware — bare
    names resolve WITHIN the current aggregator first).
    """
    raw = str(raw or "")
    parsed = parse_model_flags_detailed(raw)

    errors: list = []
    if parsed.is_once and parsed.is_global:
        errors.append(MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL)
    if parsed.is_once and not parsed.model_input and not parsed.explicit_provider:
        errors.append(MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET)

    if parsed.is_once:
        scope = "once"
    elif parsed.is_session:
        scope = "session"
    elif parsed.is_global:
        scope = "global"
    else:
        scope = "default"

    return ModelSwitchRequest(
        raw=raw,
        target=parsed.model_input,
        explicit_provider=parsed.explicit_provider,
        is_global=parsed.is_global,
        is_session=parsed.is_session,
        is_once=parsed.is_once,
        force_refresh=parsed.force_refresh,
        scope=scope,
        errors=tuple(errors),
    )


def _effective_model_candidate(value: Any) -> str:
    """Extract a model-name candidate from a str / dict / attr-object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("model") or "").strip()
    model_attr = getattr(value, "model", None)
    if model_attr is not None:
        return str(model_attr or "").strip()
    return ""


def resolve_effective_model(
    session_overrides: Any = None,
    channel_config: Any = None,
    global_config: Any = "",
) -> str:
    """Resolve the effective model: session override > channel > global.

    The single owner of the precedence rule that gateway/run.py
    (``_resolve_model_for_channel`` / ``_apply_session_model_override``) and
    gateway/platforms/api_server.py (``_create_agent``'s session-override /
    session-persisted-model branches) each encoded independently — the
    divergence commit 7dd00bb47d had to close.  A user-issued ``/model``
    (session override) always wins over per-channel/session-persisted
    configuration, which wins over the global default.

    Each argument may be a plain model string, a dict with a ``"model"``
    key (a gateway ``_session_model_overrides`` entry), or an object with a
    ``.model`` attribute (a ``ChannelOverride``).  Empty/None entries fall
    through to the next tier.
    """
    for tier in (session_overrides, channel_config, global_config):
        candidate = _effective_model_candidate(tier)
        if candidate:
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def _model_sort_key(model_id: str, prefix: str) -> tuple:
    """Sort key for model version preference.

    Extracts version numbers after the family prefix and returns a sort key
    that prefers higher versions.  Suffix tokens (``pro``, ``omni``, etc.)
    are used as tiebreakers, with common quality indicators ranked.

    Examples (with prefix ``"mimo"``)::

        mimo-v2.5-pro   → (-2.5, 0, 'pro')     # highest version wins
        mimo-v2.5       → (-2.5, 1, '')          # no suffix = lower than pro
        mimo-v2-pro     → (-2.0, 0, 'pro')
        mimo-v2-omni    → (-2.0, 1, 'omni')
        mimo-v2-flash   → (-2.0, 1, 'flash')
    """
    # Strip the prefix (and optional "/" separator for aggregator slugs)
    rest = model_id[len(prefix):]
    if rest.startswith("/"):
        rest = rest[1:]
    rest = rest.lstrip("-").strip()

    # Parse version and suffix from the remainder.
    # "v2.5-pro" → version [2.5], suffix "pro"
    # "-omni"    → version [],    suffix "omni"
    # State machine: start → in_version → between → in_suffix
    nums: list[float] = []
    suffix_buf = ""
    state = "start"
    num_buf = ""

    def _flush() -> None:
        nonlocal num_buf
        if num_buf:
            try:
                nums.append(float(num_buf.rstrip(".")))
            except ValueError:
                pass
            num_buf = ""

    for ch in rest:
        if state == "in_suffix":
            suffix_buf += ch
        elif state == "in_version":
            if ch.isdigit():
                num_buf += ch
            elif ch == ".":
                if "." in num_buf:
                    _flush()  # second dot: start a new version component
                else:
                    num_buf += ch
            else:
                _flush()
                if ch in "-_":
                    state = "between"
                else:
                    state = "in_suffix"
                    suffix_buf += ch
        else:  # "start" / "between": skip separators, enter version on v/digit, else suffix
            if ch in "vV":
                state = "in_version"
            elif ch.isdigit():
                state = "in_version"
                num_buf = ch
            elif ch not in "-_.":
                state = "in_suffix"
                suffix_buf += ch

    # Flush remaining buffer (strip trailing dots — "5.4." → "5.4")
    if state == "in_version":
        _flush()

    suffix = suffix_buf.lower().strip("-_.")
    suffix = suffix.strip()

    # Split out YYYYMMDD date stamps (e.g. claude-opus-4-20250514): they are
    # snapshot markers, not version components, and would otherwise dwarf
    # real point versions (20250514 > 8).  Kept as a trailing tiebreaker so
    # bare IDs sort before their dated snapshots, and newer snapshots before
    # older ones.  The 19_000_101 threshold reclassifies only 8-digit stamps,
    # so shorter numeric components (mistral-large-2411, gpt-4-0613) keep
    # their current behavior.
    version_nums: list[float] = []
    date_stamp = 0.0
    for n in nums:
        if n >= 19_000_101:
            date_stamp = max(date_stamp, n)
        else:
            version_nums.append(n)

    # Negate versions so higher → sorts first
    version_key = tuple(-n for n in version_nums)
    date_key = (0.0, 0.0) if date_stamp == 0.0 else (1.0, -date_stamp)

    # Suffix quality ranking: pro/max > (no suffix) > omni/flash/mini/lite
    # Lower number = preferred
    # "sol" is the flagship tier of the GPT-5.6 series (sol > terra > luna);
    # without it, alias resolution would tiebreak alphabetically and pick
    # luna (the cheapest) for `/model gpt`. Unlike pro/max/plus/turbo it is a
    # series codename, not a generic quality word — revisit if another vendor
    # ever ships a "-sol" suffix that isn't a flagship.
    _SUFFIX_RANK = {"pro": 0, "max": 0, "plus": 0, "turbo": 0, "sol": 0}
    suffix_rank = _SUFFIX_RANK.get(suffix, 1)

    return version_key + (suffix_rank, suffix) + date_key


class AmbiguousAliasError(Exception):
    """Alias family-matches multiple catalog models; caller must disambiguate.

    Raised by :func:`resolve_alias` instead of silently picking one candidate
    via version-sort heuristics. ``candidates`` is sorted best-guess-first
    (see :func:`_model_sort_key`) for display purposes only.
    """

    def __init__(self, alias: str, provider: str, candidates: list[str]):
        self.alias = alias
        self.provider = provider
        self.candidates = candidates
        super().__init__(
            f"alias {alias!r} matches {len(candidates)} models on {provider}"
        )


def _ambiguous_alias_message(err: "AmbiguousAliasError") -> str:
    """User-facing disambiguation list for an ambiguous alias."""
    shown = err.candidates[:10]
    lines = "\n".join(f"  {i}. {m}" for i, m in enumerate(shown, 1))
    more = ""
    if len(err.candidates) > len(shown):
        more = f"\n  … and {len(err.candidates) - len(shown)} more"
    return (
        f"'{err.alias}' matches {len(err.candidates)} models on "
        f"{err.provider} — not switching automatically:\n{lines}{more}\n"
        f"Pick one with /model <exact-model-name>."
    )


def resolve_alias(
    raw_input: str,
    current_provider: str,
) -> Optional[tuple[str, str, str]]:
    """Resolve a short alias against the current provider's catalog.

    Looks up *raw_input* in :data:`MODEL_ALIASES`, then searches the
    current provider's models.dev catalog for the model whose ID starts
    with ``vendor/family`` (or just ``family`` for non-aggregator
    providers) and has the **highest version**.

    Returns:
        ``(provider, resolved_model_id, alias_name)`` if a match is
        found on the current provider, or ``None`` if the alias doesn't
        exist or no matching model is available.
    """
    key = raw_input.strip().lower()

    # Check direct aliases first (exact model+provider+base_url mappings)
    _ensure_direct_aliases()
    direct = DIRECT_ALIASES.get(key)
    if direct is not None:
        return (direct.provider, direct.model, key)

    # Reverse lookup: match by model ID so full names (e.g. "kimi-k2.5",
    # "glm-4.7") route through direct aliases instead of falling through
    # to the catalog/OpenRouter.
    for alias_name, da in DIRECT_ALIASES.items():
        if da.model.lower() == key:
            return (da.provider, da.model, alias_name)

    identity = MODEL_ALIASES.get(key)
    if identity is None:
        return None

    vendor, family = identity

    # Build catalog from models.dev, then merge in static _PROVIDER_MODELS
    # entries that models.dev may be missing (e.g. newly added models not
    # yet synced to the registry).
    catalog = list_provider_models(current_provider)
    try:
        from hermes_cli.models import _PROVIDER_MODELS
        static = _PROVIDER_MODELS.get(current_provider, [])
        if static:
            seen = {m.lower() for m in catalog}
            for m in static:
                if m.lower() not in seen:
                    catalog.append(m)
    except Exception:
        pass

    # For aggregators, models are vendor/model-name format
    aggregator = is_aggregator(current_provider)

    if aggregator:
        prefix = f"{vendor}/{family}".lower()
        matches = [
            mid for mid in catalog
            if mid.lower().startswith(prefix)
        ]
    else:
        family_lower = family.lower()
        matches = [
            mid for mid in catalog
            if mid.lower().startswith(family_lower)
        ]

    if not matches:
        return None

    # Sort by version descending (best guess first) for display, but NEVER
    # silently pick among multiple candidates: version-sort heuristics have
    # repeatedly guessed wrong (dated snapshots outranking point releases,
    # suffix tiebreaks landing on the cheapest tier). One match = resolve;
    # several = make the user choose.
    prefix_for_sort = f"{vendor}/{family}" if aggregator else family
    matches.sort(key=lambda m: _model_sort_key(m, prefix_for_sort))
    if len(matches) > 1:
        raise AmbiguousAliasError(key, current_provider, matches)
    return (current_provider, matches[0], key)


def get_authenticated_provider_slugs(
    current_provider: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
) -> list[str]:
    """Return slugs of providers that have credentials.

    Uses ``list_authenticated_providers()`` which is backed by the models.dev
    in-memory cache (1 hr TTL) — no extra network cost.
    """
    try:
        providers = list_authenticated_providers(
            current_provider=current_provider,
            user_providers=user_providers,
            custom_providers=custom_providers,
            max_models=0,
        )
        return [p["slug"] for p in providers]
    except Exception:
        return []


def _resolve_alias_fallback(
    raw_input: str,
    authenticated_providers: list[str] = (),
) -> Optional[tuple[str, str, str]]:
    """Try to resolve an alias on the user's authenticated providers.

    Falls back to ``("openrouter", "nous")`` only when no authenticated
    providers are supplied (backwards compat for non-interactive callers).
    """
    providers = authenticated_providers or ("openrouter", "nous")
    for provider in providers:
        # AmbiguousAliasError propagates: the alias exists on this provider,
        # the user just has to choose — trying the next provider instead
        # would silently switch them somewhere they didn't ask to go.
        result = resolve_alias(raw_input, provider)
        if result is not None:
            return result
    return None


def resolve_display_context_length(
    model: str,
    provider: str,
    base_url: str = "",
    api_key: str = "",
    model_info: Optional[ModelInfo] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
) -> Optional[int]:
    """Resolve the context length to show in /model output.

    models.dev reports per-vendor context (e.g. gpt-5.5 = 1.05M on openai)
    but provider-enforced limits can be lower (e.g. Codex OAuth caps the
    same slug at 272k). The authoritative source is
    ``agent.model_metadata.get_model_context_length`` which already knows
    about Codex OAuth, Copilot, Nous, and falls back to models.dev for the
    rest.

    When ``custom_providers`` is provided, per-model ``context_length``
    overrides from ``custom_providers[].models.<id>.context_length`` are
    honored — this closes #15779 where ``/model`` switch ignored user-set
    overrides.

    Prefer the provider-aware value; fall back to ``model_info.context_window``
    only if the resolver returns nothing.
    """
    if config_context_length is not None and (
        configured_model or configured_provider or configured_base_url
    ):
        try:
            from hermes_cli.route_identity import should_clear_context_pin

            if should_clear_context_pin(
                configured_model,
                model,
                configured_base_url,
                base_url,
                configured_provider,
                provider,
            ):
                config_context_length = None
        except Exception:
            config_context_length = None

    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(
            model,
            base_url=base_url or "",
            api_key=api_key or "",
            provider=provider or None,
            custom_providers=custom_providers,
            config_context_length=config_context_length,
        )
        if ctx:
            return int(ctx)
    except Exception:
        pass
    if model_info is not None and model_info.context_window:
        return int(model_info.context_window)
    return None


async def resolve_display_context_length_async(
    model: str,
    provider: str,
    base_url: str = "",
    api_key: str = "",
    model_info: Optional[ModelInfo] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
) -> Optional[int]:
    """Async variant of :func:`resolve_display_context_length`.

    The sync version runs two blocking chains: the route comparison in
    ``should_clear_context_pin`` and the full provider probe ladder in
    ``get_model_context_length`` (blocking ``requests`` calls to Anthropic
    ``/v1/models``, Copilot, Nous, Codex, GMI, Ollama, models.dev and
    OpenRouter).  Async gateway handlers must not run either on the event
    loop — see ``agent.model_metadata.get_model_context_length_async`` and
    ``hermes_cli.route_identity.should_clear_context_pin_async``, which
    offload the same chains for the message path.

    Shares all logic with the sync version — no code duplication.
    """
    import asyncio

    return await asyncio.to_thread(
        resolve_display_context_length,
        model,
        provider,
        base_url=base_url,
        api_key=api_key,
        model_info=model_info,
        custom_providers=custom_providers,
        config_context_length=config_context_length,
        configured_model=configured_model,
        configured_provider=configured_provider,
        configured_base_url=configured_base_url,
    )


# ---------------------------------------------------------------------------
# Configured-provider detection for typed model names
# ---------------------------------------------------------------------------


def _configured_provider_matches(
    model_name: str,
    user_providers: Optional[dict],
    custom_providers: Optional[list],
) -> dict[str, str]:
    """Return ``{provider_slug: canonical_model_id}`` for every configured
    provider whose declared models contain an exact (case-insensitive) match
    for ``model_name``.

    Used by :func:`switch_model` to route a *typed* model name to the provider
    that actually declares it in user/custom provider config, instead of
    leaving it on the current provider.  Without this, a model declared under
    ``providers.<slug>`` / ``custom_providers`` but typed while the current
    provider is ``openai-codex`` stays on Codex and is soft-accepted as an
    unknown hidden Codex model (#45006).

    Matching is exact (case-insensitive); the configured spelling is returned
    so the downstream validation/override path sees the canonical id.  Only the
    explicitly-declared model collections are scanned (``models``, the singular
    ``model``, and ``default_model``) — never fuzzy/family matching.
    """
    if not model_name or not model_name.strip():
        return {}
    target = model_name.strip().lower()

    def _match(value) -> Optional[str]:
        """Canonical id if ``value`` (a model collection or scalar) declares
        ``target``, else None."""
        for model_id in _declared_model_ids(value):
            if model_id.lower() == target:
                return model_id
        return None

    matches: dict[str, str] = {}

    if isinstance(user_providers, dict):
        for slug, cfg in user_providers.items():
            if not isinstance(slug, str) or not isinstance(cfg, dict):
                continue
            for key in ("models", "model", "default_model"):
                hit = _match(cfg.get(key))
                if hit:
                    matches[slug] = hit
                    break

    if isinstance(custom_providers, list):
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            slug = f"custom:{name}"
            if slug in matches:
                continue
            for key in ("models", "model", "default_model"):
                hit = _match(entry.get(key))
                if hit:
                    matches[slug] = hit
                    break

    return matches


def _resolve_named_custom_model_id(
    model_name: str,
    target_provider: str,
    custom_providers: Optional[list],
) -> str:
    """Map a picker-prefixed custom model selection to its configured ID."""
    provider = str(target_provider or "").strip().lower()
    if not provider.startswith("custom:") or "/" not in model_name:
        return model_name

    prefix, candidate = model_name.split("/", 1)
    prefix = prefix.strip().lower()
    candidate = candidate.strip()
    if not prefix or not candidate:
        return model_name

    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        entry_slugs = custom_provider_aliases(
            str(entry.get("name") or ""),
            str(entry.get("provider_key") or ""),
        )
        if provider not in entry_slugs or f"custom:{prefix}" not in entry_slugs:
            continue
        for model_id in _declared_model_ids(entry.get("models")):
            if model_id.lower() == candidate.lower():
                return model_id
    return model_name


# ---------------------------------------------------------------------------
# Core model-switching pipeline
# ---------------------------------------------------------------------------

def _switch_fail(is_global: bool, message: str, **fields) -> ModelSwitchResult:
    return ModelSwitchResult(success=False, is_global=is_global, error_message=message, **fields)


def _runtime_creds(fallback_headers: dict, **kwargs) -> tuple[str, str, str, dict, dict]:
    """``resolve_runtime_provider`` unpacked as ``(api_key, base_url, api_mode,
    capabilities, extra_headers)``; ``extra_headers`` falls back to *fallback_headers*."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(**kwargs)
    return (
        runtime.get("api_key", ""),
        runtime.get("base_url", ""),
        runtime.get("api_mode", ""),
        runtime.get("capabilities") or {},
        runtime.get("extra_headers") or fallback_headers,
    )


def _entry_configured_key(cfg: dict, read_env) -> str:
    """Inline ``api_key`` (a ``${VAR}`` template resolves via *read_env*), else
    ``key_env``/``api_key_env`` via *read_env*."""
    key = str(cfg.get("api_key", "") or "").strip()
    if key.startswith("${") and key.endswith("}"):
        key = read_env(key[2:-1])
    if not key:
        key_env = str(cfg.get("key_env") or cfg.get("api_key_env") or "").strip()
        key = read_env(key_env) if key_env else ""
    return key


def _ollama_configured_base() -> tuple[dict, str]:
    from hermes_cli.models import _get_provider_config_dict

    cfg = _get_provider_config_dict("ollama")
    return cfg, str(cfg.get("base_url") or cfg.get("api") or cfg.get("url") or "").strip()


def _unknown_provider_message(explicit_provider: str) -> str:
    msg = (
        f"Unknown provider '{explicit_provider}'. "
        f"Check 'hermes model' for available providers, or define it "
        f"in config.yaml under 'providers:'."
    )
    # Surface common config issues that cause provider resolution failures
    try:
        from hermes_cli.config import validate_config_structure
        issues = validate_config_structure()
        if issues:
            msg += "\n\nRun 'hermes doctor' — config issues detected:"
            for ci in issues[:3]:
                msg += f"\n  • {ci.message}"
    except Exception:
        pass
    return msg


def _aggregator_alias_error(
    explicit_provider: str, target_provider: str, current_provider: str, user_providers, custom_providers,
) -> str:
    """Guard against silent aggregator hops: a vendor alias like bare "openai"
    resolves to an aggregator ("openrouter"); if that aggregator has no
    credentials, refuse instead of switching the user onto an unauthed endpoint
    (HTTP 401) and point at the real direct provider."""
    from hermes_cli.models import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import ALIASES

    explicit_norm = explicit_provider.strip().lower()
    alias_target = ALIASES.get(explicit_norm)
    if not (
        alias_target
        and alias_target == target_provider
        and target_provider != explicit_norm
        and target_provider in _AGGREGATOR_PROVIDERS
    ):
        return ""
    authed = get_authenticated_provider_slugs(
        current_provider=current_provider, user_providers=user_providers, custom_providers=custom_providers,
    )
    if target_provider in authed:
        return ""
    suggestions = [s for s in authed if s.startswith(explicit_norm) and s != explicit_norm]
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return (
        f"Provider '{explicit_norm}' is an alias that routes "
        f"through {get_label(target_provider)}, which "
        f"has no credentials configured.{hint}"
    )


def _aggregator_catalog_match(new_model: str, catalog: list) -> str | None:
    """Exact (case-insensitive) match on full id, then on the bare part after ``vendor/``."""
    new_model_lower = new_model.lower()
    for mid in catalog:
        if mid.lower() == new_model_lower:
            return mid
    for mid in catalog:
        if "/" in mid and mid.split("/", 1)[1].lower() == new_model_lower:
            return mid
    return None


def _config_declares_model(
    new_model: str, target_provider: str, base_url: str, user_providers, custom_providers,
) -> bool:
    """A model declared in the user's ``providers:``/``custom_providers:`` config
    is accepted even when the remote /v1/models does not list it (cloud/aliased
    models). Custom entries match by slug alias or by base_url."""
    if user_providers:
        from hermes_cli.config import is_provider_enabled
        for slug, cfg in user_providers.items():
            if not is_provider_enabled(cfg):
                continue
            if slug == target_provider and new_model in _declared_model_ids(cfg.get("models", {})):
                return True
    if custom_providers and isinstance(custom_providers, list):
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            entry_aliases = custom_provider_aliases(
                str(entry.get("name", "") or ""), str(entry.get("provider_key") or ""),
            )
            if (target_provider.lower() in entry_aliases or entry.get("base_url", "") == base_url) and (
                new_model == entry.get("model", "") or new_model in _declared_model_ids(entry.get("models", {}))
            ):
                return True
    return False


def _apply_direct_alias_endpoint(
    da: DirectAlias, target_provider: str, new_model: str, api_key: str, base_url: str,
) -> tuple[str, str, dict | None, bool]:
    """Route a direct alias to its own base_url and decide its credential.

    Returns ``(api_key, base_url, validation_headers_override, suppress_ollama_headers)``
    where a ``None`` headers override means "leave as is".

    Credentials were resolved against the DEFAULT provider; carrying that key
    onto the alias's endpoint both 401s and ships the default provider's secret
    to an unrelated host. The alias's own endpoint decides: its declared key when
    it has one; the session key only when the alias points at the SAME ORIGIN
    it was resolved for; otherwise a fresh resolution against the alias
    base_url (whose env-key fallbacks are gated on authoritative hosts, so
    OLLAMA_API_KEY still resolves for ollama.com while OPENROUTER_API_KEY never
    reaches an unrelated host).
    """
    from hermes_cli.models import _same_ollama_native_root
    from hermes_cli.runtime_provider import resolve_runtime_provider

    alias_key = direct_alias_api_key(da)
    if alias_key:
        base_url, api_key = da.base_url, alias_key
    elif api_key and api_key != "no-key-required" and _may_reuse_session_credential(base_url, da.base_url):
        # Same origin: the key is host-appropriate and re-resolving would only
        # repeat the work (incl. a second local-endpoint /models probe).
        base_url = da.base_url
    else:
        try:
            req, explicit = direct_alias_runtime_request(da)
            alias_runtime = resolve_runtime_provider(
                requested=req, explicit_api_key=explicit, explicit_base_url=da.base_url, target_model=new_model,
            )
        except Exception:
            alias_runtime = {}
        same_host = _may_reuse_session_credential(base_url, da.base_url)
        base_url = alias_runtime.get("base_url", "") or da.base_url
        # The resolver reports "no key found" as the `no-key-required`
        # placeholder; normalise so a same-host credential still outranks it.
        resolved_key = alias_runtime.get("api_key", "")
        if resolved_key == "no-key-required":
            resolved_key = ""
        api_key = resolved_key or (api_key if same_host else "") or "no-key-required"

    headers_override = None
    suppress = False
    # providers.ollama refinement: pick up the configured key only for the
    # configured native root; drop key and provider-level headers for any other
    # origin. Skipped when the alias declared its own credential (explicit
    # api_key/key_env outranks a provider-level config key).
    if not alias_key and target_provider.strip().lower() == "ollama":
        ollama_cfg, ollama_cfg_base = _ollama_configured_base()
        if ollama_cfg_base and _same_ollama_native_root(base_url, ollama_cfg_base):
            configured_key = _entry_configured_key(ollama_cfg, lambda n: os.environ.get(n, "").strip())
            if configured_key:
                api_key = configured_key
        else:
            # Different origin, or no configured root to safely associate the
            # provider-level headers with.
            headers_override, suppress, api_key = {}, True, "no-key-required"
    return api_key or "no-key-required", base_url, headers_override, suppress


def switch_model(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str = "",
    current_api_key: str = "",
    is_global: bool = False,
    explicit_provider: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Resolution chain:

      If --provider given:
        a. Resolve provider via resolve_provider_full()
        b. Resolve credentials
        c. If model given, resolve alias on target provider or use as-is
        d. If no model, auto-detect from endpoint

      If no --provider:
        a. Try alias resolution on current provider
        b. If alias exists but not on current provider -> fallback
        c. On aggregator, try vendor/model slug conversion
        d. Aggregator catalog search
        e. detect_provider_for_model() as last resort
        f. Resolve credentials
        g. Normalize model name for target provider

      Finally:
        h. Get full model metadata from models.dev
        i. Build result

    ``explicit_provider`` comes from the --provider flag (empty = none);
    ``user_providers`` / ``custom_providers`` are the ``providers:`` dict and
    ``custom_providers:`` list from config.yaml.
    """
    from hermes_cli.models import (
        copilot_model_api_mode,
        detect_provider_for_model,
        validate_requested_model,
        opencode_model_api_mode,
        _get_ollama_request_headers,
        _same_ollama_native_root,
    )

    resolved_alias = ""
    new_model = raw_input.strip()
    target_provider = current_provider
    resolved_moa_preset = False

    # =================================================================
    # PATH A: Explicit --provider given
    # =================================================================
    if explicit_provider:
        pdef = resolve_provider_full(explicit_provider, user_providers, custom_providers)
        if pdef is None and explicit_provider.strip().lower() == "custom":
            pdef = _bare_custom_provider_def(current_base_url)
        if pdef is None:
            return _switch_fail(is_global, _unknown_provider_message(explicit_provider))

        target_provider = pdef.id
        if target_provider == "moa" and not new_model:
            try:
                from hermes_cli.config import load_config
                from hermes_cli.moa_config import normalize_moa_config

                new_model = normalize_moa_config(load_config().get("moa") or {})["default_preset"]
            except Exception:
                new_model = "default"

        agg_err = _aggregator_alias_error(
            explicit_provider, target_provider, current_provider, user_providers, custom_providers,
        )
        if agg_err:
            return _switch_fail(is_global, agg_err, target_provider=target_provider, provider_label=pdef.name)

        # No model specified: auto-detect from the endpoint
        if not new_model:
            if not pdef.base_url:
                return _switch_fail(
                    is_global,
                    f"Provider '{pdef.name}' has no base URL configured. "
                    f"Specify a model: /model <model-name> --provider {explicit_provider}",
                    target_provider=target_provider, provider_label=pdef.name,
                )
            from hermes_cli.runtime_provider import _auto_detect_local_model
            new_model = _auto_detect_local_model(pdef.base_url)
            if not new_model:
                return _switch_fail(
                    is_global,
                    f"No model detected on {pdef.name} ({pdef.base_url}). "
                    f"Specify the model explicitly: /model <model-name> --provider {explicit_provider}",
                    target_provider=target_provider, provider_label=pdef.name,
                )

        # Resolve alias on the TARGET provider
        try:
            alias_result = resolve_alias(new_model, target_provider)
        except AmbiguousAliasError as err:
            return _switch_fail(is_global, _ambiguous_alias_message(err), target_provider=target_provider)
        if alias_result is not None:
            _, new_model, resolved_alias = alias_result

    # =================================================================
    # PATH B: No explicit provider — resolve from model input
    # =================================================================
    else:
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import exact_moa_preset_name, normalize_moa_config

            moa_match = exact_moa_preset_name(normalize_moa_config(load_config().get("moa") or {}), raw_input)
            if moa_match:
                target_provider, new_model, resolved_alias = "moa", moa_match, ""
                resolved_moa_preset = True
                alias_result = None
            else:
                alias_result = resolve_alias(raw_input, current_provider)
        except AmbiguousAliasError as err:
            return _switch_fail(is_global, _ambiguous_alias_message(err))
        except Exception:
            try:
                alias_result = resolve_alias(raw_input, current_provider)
            except AmbiguousAliasError as err:
                return _switch_fail(is_global, _ambiguous_alias_message(err))

        # --- Step a: alias on current provider ---
        if resolved_moa_preset:
            pass
        elif alias_result is not None:
            target_provider, new_model, resolved_alias = alias_result
            logger.debug("Alias '%s' resolved to %s on %s", resolved_alias, new_model, target_provider)
        else:
            # --- Step b: alias exists but not on current provider -> fallback ---
            key = raw_input.strip().lower()
            if key in MODEL_ALIASES:
                authed = get_authenticated_provider_slugs(
                    current_provider=current_provider, user_providers=user_providers, custom_providers=custom_providers,
                )
                try:
                    fallback_result = _resolve_alias_fallback(raw_input, authed)
                except AmbiguousAliasError as err:
                    return _switch_fail(is_global, _ambiguous_alias_message(err))
                if fallback_result is None:
                    identity = MODEL_ALIASES[key]
                    return _switch_fail(
                        is_global,
                        f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
                        f"but no matching model was found in any provider catalog. "
                        f"Try specifying the full model name.",
                    )
                target_provider, new_model, resolved_alias = fallback_result
                logger.debug(
                    "Alias '%s' resolved via fallback to %s on %s", resolved_alias, new_model, target_provider,
                )
            else:
                # --- Step c: on an aggregator, vendor:model -> vendor/model ---
                # Only without a slash: with one, the colon is a variant tag
                # (:free, :extended, :fast) that must be preserved.
                colon_pos = raw_input.find(":")
                cur_norm = str(current_provider).strip().lower()
                if (
                    colon_pos > 0
                    and "/" not in raw_input
                    and is_aggregator(current_provider)
                    and not cur_norm.startswith("custom")
                    and cur_norm != "ollama"
                ):
                    left = raw_input[:colon_pos].strip().lower()
                    right = raw_input[colon_pos + 1:].strip()
                    if left and right:
                        new_model = f"{left}/{right}"
                        logger.debug("Converted vendor:model '%s' to aggregator slug '%s'", raw_input, new_model)

        # --- Step d: aggregator catalog search ---
        # If the CURRENT provider's live catalog resolved the model, step e must
        # not second-guess and switch providers — flat-namespace resellers
        # (opencode-go/zen) return bare ids that coincidentally match native
        # providers' static catalogs.
        resolved_in_current_catalog = False
        if is_aggregator(target_provider) and not resolved_alias:
            catalog = list_provider_models(target_provider)
            if catalog:
                matched = _aggregator_catalog_match(new_model, catalog)
                if matched is not None:
                    new_model, resolved_in_current_catalog = matched, True

        # --- Step d.5: configured-provider exact match ---
        # A model declared in user/custom provider config routes there BEFORE
        # detect_provider_for_model() guesses from static catalogs and before a
        # soft-accepting current provider (openai-codex) can swallow it as an
        # unknown hidden model. Deliberately NOT gated on ``not is_custom``.
        config_routed = False
        if not resolved_alias and not resolved_in_current_catalog and target_provider == current_provider:
            cfg_matches = _configured_provider_matches(new_model, user_providers, custom_providers)
            if cfg_matches:
                if current_provider in cfg_matches:
                    new_model = cfg_matches[current_provider]
                    config_routed = True
                else:
                    match_slugs = sorted(cfg_matches)
                    if len(match_slugs) > 1:
                        return _switch_fail(
                            is_global,
                            f"'{new_model}' is declared by multiple configured "
                            f"providers ({', '.join(match_slugs)}). Re-run with "
                            f"--provider <slug> to choose which one to use.",
                        )
                    target_provider = match_slugs[0]
                    new_model = cfg_matches[target_provider]
                    config_routed = True
                    logger.debug("Configured-provider detection routed '%s' to %s", new_model, target_provider)
                    # providers.<slug> endpoints resolve in the credential block
                    # via resolve_user_provider(), which is gated on
                    # explicit_provider; custom:* slugs resolve at runtime directly.
                    if isinstance(user_providers, dict) and target_provider in user_providers:
                        explicit_provider = target_provider

        # --- Step e: detect_provider_for_model() as last resort ---
        is_custom = (
            current_provider in {"custom", "local"}
            or current_provider.startswith("custom:")
            or base_url_hostname(current_base_url or "") in ("localhost", "127.0.0.1")
        )
        if (
            target_provider == current_provider
            and not is_custom
            and not resolved_alias
            and not resolved_in_current_catalog
            and not config_routed
        ):
            detected = detect_provider_for_model(new_model, current_provider)
            if detected:
                target_provider, new_model = detected

    # =================================================================
    # COMMON PATH: Resolve credentials, normalize, get metadata
    # =================================================================
    provider_changed = target_provider != current_provider
    provider_label = get_label(target_provider)
    if target_provider == "custom" and current_base_url:
        provider_label = "Custom endpoint"
    if target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(target_provider, user_providers, custom_providers)
        if custom_pdef is not None:
            provider_label = custom_pdef.name

    # --- Resolve credentials ---
    api_key = current_api_key
    base_url = current_base_url
    api_mode = ""
    runtime_capabilities: dict[str, bool] = {}
    ollama_headers: dict[str, str] = {}
    validation_headers: dict[str, str] = {}
    suppress_ollama_headers = False

    if provider_changed or explicit_provider:
        # providers.<name> blocks carry their own base_url + transport + key
        # reference; resolve_runtime_provider() resolves by provider NAME and
        # would re-resolve a block named "openai" from scratch (or hop to an
        # aggregator), so use the pdef's endpoint directly.
        user_pdef = None
        if explicit_provider and user_providers:
            from hermes_cli.providers import resolve_user_provider
            user_pdef = resolve_user_provider(explicit_provider.strip().lower(), user_providers)
            if user_pdef is None:
                user_pdef = resolve_user_provider(target_provider, user_providers)
        if user_pdef is not None and user_pdef.base_url:
            ucfg = (user_providers or {}).get(explicit_provider.strip().lower()) \
                or (user_providers or {}).get(target_provider) or {}
            # Key reads go through the per-profile secret scope: a raw
            # os.environ read would hand this profile another profile's key
            # under the multiplexed gateway.
            ukey = _entry_configured_key(ucfg, _scoped_key_env)
            validation_headers = _extra_headers_from_config(ucfg)
            try:
                api_key, base_url, api_mode, runtime_capabilities, validation_headers = _runtime_creds(
                    validation_headers,
                    requested=target_provider,
                    explicit_api_key=ukey or None,
                    explicit_base_url=user_pdef.base_url,
                    target_model=new_model,
                )
                api_key = api_key or ukey
                base_url = base_url or user_pdef.base_url
            except Exception:
                api_key, base_url, api_mode = ukey, user_pdef.base_url, ""
        elif target_provider == "custom" and current_base_url:
            api_key, base_url = current_api_key, current_base_url
            api_mode = determine_api_mode(target_provider, base_url)
        else:
            try:
                api_key, base_url, api_mode, runtime_capabilities, validation_headers = _runtime_creds(
                    validation_headers, requested=target_provider, target_model=new_model,
                )
            except Exception as e:
                return _switch_fail(
                    is_global,
                    f"Could not resolve credentials for provider '{provider_label}': {e}",
                    target_provider=target_provider, provider_label=provider_label,
                )
    else:
        keep_current_ollama_endpoint = False
        if current_provider == "custom" and current_base_url:
            try:
                from hermes_cli.models import should_use_ollama_native_catalog
                ollama_headers = _get_ollama_request_headers()
                _, configured_ollama_base = _ollama_configured_base()
                # Provider-level Ollama headers only belong to the configured
                # native root; without one there is no safe origin for them.
                if not configured_ollama_base or not _same_ollama_native_root(current_base_url, configured_ollama_base):
                    ollama_headers = {}
                    suppress_ollama_headers = True
                keep_current_ollama_endpoint = should_use_ollama_native_catalog(
                    current_provider, current_base_url, headers=ollama_headers,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                keep_current_ollama_endpoint = False
        if keep_current_ollama_endpoint:
            # Mid-session `/model <name>` on a local Ollama-compatible endpoint
            # keeps the endpoint in use; re-resolving bare `custom` from config
            # can fall through to an unrelated default provider.
            api_key = current_api_key or "no-key-required"
            base_url = current_base_url
            api_mode = determine_api_mode(current_provider, base_url)
            validation_headers = ollama_headers
        else:
            try:
                api_key, base_url, api_mode, runtime_capabilities, validation_headers = _runtime_creds(
                    validation_headers, requested=current_provider, target_model=new_model,
                )
            except Exception:
                pass

    # --- Direct alias override: use the alias's exact base_url if set ---
    if resolved_alias:
        _ensure_direct_aliases()
        da = DIRECT_ALIASES.get(resolved_alias)
        if da is not None and da.base_url:
            api_key, base_url, headers_override, suppress = _apply_direct_alias_endpoint(
                da, target_provider, new_model, api_key, base_url,
            )
            api_mode = ""  # clear so determine_api_mode re-detects from URL
            if headers_override is not None:
                validation_headers = headers_override
            if suppress:
                suppress_ollama_headers = True

    # --- api_mode from the final (provider, base_url) before validation ---
    # Fills an empty mode (alias cleared it) and overrides a STALE mode carried
    # from previous session state when the host mandates one wire protocol
    # (e.g. gpt-5.x on api.openai.com would otherwise 400 on tools+reasoning).
    mandated_mode = host_mandated_api_mode(base_url)
    if mandated_mode is not None:
        api_mode = mandated_mode
    elif not api_mode:
        api_mode = determine_api_mode(target_provider, base_url)

    # --- Normalize model name for target provider ---
    new_model = _resolve_named_custom_model_id(new_model, target_provider, custom_providers)
    new_model = normalize_model_for_provider(new_model, target_provider)

    # --- Validate ---
    if target_provider.strip().lower() == "ollama":
        headers = {} if suppress_ollama_headers else (validation_headers or _get_ollama_request_headers())
    else:
        headers = validation_headers or (
            _extra_headers_from_config(user_providers.get(target_provider))
            if user_providers and target_provider in user_providers
            else None
        )
    try:
        validation = validate_requested_model(
            new_model, target_provider, api_key=api_key, base_url=base_url, api_mode=api_mode or None, headers=headers,
        )
    except Exception as e:
        validation = {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": f"Could not validate `{new_model}`: {e}",
        }

    if not validation.get("accepted"):
        if _config_declares_model(new_model, target_provider, base_url, user_providers, custom_providers):
            validation = {"accepted": True, "persist": True, "recognized": False, "message": validation.get("message", "")}
        else:
            return _switch_fail(
                is_global, validation.get("message", "Invalid model"),
                new_model=new_model, target_provider=target_provider, provider_label=provider_label,
            )

    if validation.get("corrected_model"):
        new_model = validation["corrected_model"]

    # --- Per-provider api_mode overrides ---
    if target_provider in {"copilot", "github-copilot"}:
        api_mode = copilot_model_api_mode(new_model, api_key=api_key)
    if target_provider in {"opencode-zen", "opencode-go", "opencode"}:
        api_mode = opencode_model_api_mode(target_provider, new_model)
    if target_provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal serves anthropic/* on /v1/messages and everything else on
        # /chat/completions; re-derive from the FINAL model so alias clears /
        # empty fallbacks cannot leave Claude on the OpenAI wire.
        from hermes_cli.providers import nous_api_mode

        api_mode = nous_api_mode(new_model)
    if not api_mode:
        api_mode = determine_api_mode(target_provider, base_url, model=new_model)

    # OpenCode base URLs end with /v1 for OpenAI-compatible models but the
    # Anthropic SDK prepends its own /v1/messages: strip for anthropic_messages,
    # re-append for chat_completions/codex_responses (mirrors
    # resolve_runtime_provider; either direction alone breaks the other family).
    from hermes_cli.models import opencode_provider_family
    if opencode_provider_family(target_provider) is not None and isinstance(base_url, str):
        from hermes_cli.models import normalize_opencode_base_url
        base_url = normalize_opencode_base_url(target_provider, api_mode, base_url)

    capabilities = get_model_capabilities(target_provider, new_model, allow_network=True)
    from agent.native_compaction import resolve_native_compaction_capabilities
    runtime_capabilities = resolve_native_compaction_capabilities(
        model=new_model,
        base_url=base_url,
        provider=target_provider,
        is_codex_backend=target_provider.strip().lower() == "openai-codex",
    )
    model_info = get_model_info(target_provider, new_model, allow_network=True)

    warnings: list[str] = []
    if validation.get("message"):
        warnings.append(validation["message"])
    hermes_warn = _check_hermes_model_warning(new_model)
    if hermes_warn:
        warnings.append(hermes_warn)

    # Carry the switched provider's request_overrides (custom_providers
    # ``extra_body`` such as chat_template_kwargs) so the gateway applies them
    # like the default-provider path does.
    request_overrides = None
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider, _custom_provider_request_overrides
        cp_for_ro = _get_named_custom_provider(target_provider)
        if cp_for_ro:
            request_overrides = _custom_provider_request_overrides(cp_for_ro) or None
    except Exception:
        request_overrides = None

    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        request_overrides=dict(request_overrides or {}),
        warning_message=" | ".join(warnings) if warnings else "",
        provider_label=provider_label,
        resolved_via_alias=resolved_alias,
        capabilities=capabilities,
        runtime_capabilities={
            key: value
            for key, value in runtime_capabilities.items()
            if isinstance(key, str) and isinstance(value, bool)
        },
        model_info=model_info,
        is_global=is_global,
    )


# ---------------------------------------------------------------------------
# Authenticated providers listing (for /model no-args display)
# ---------------------------------------------------------------------------

# Process-level guard so the picker prewarm thread is spawned at most once per
# process — mirrors run_agent's _openrouter_prewarm_done. Without a guard a
# long-lived process (or repeated triggers) would leak one OS thread per call.
import threading as _threading  # noqa: E402

_picker_prewarm_done = _threading.Event()


def _credential_pool_is_usable(provider: str, *, raw_pool_present: bool = False) -> bool:
    """Return whether *provider* has a credential that can be selected now.

    ``auth.json`` historically allowed opaque token-style pool values that do
    not deserialize into ``PooledCredential`` entries. Preserve visibility for
    those legacy values, but when a real pool exists its availability state is
    authoritative: an all-exhausted/dead pool is not authenticated.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        if pool.has_credentials():
            return pool.has_available()
    except Exception:
        pass
    return raw_pool_present


def _extra_headers_from_config(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    from hermes_cli.config import normalize_extra_headers

    return normalize_extra_headers(entry.get("extra_headers"))


def prewarm_picker_cache_async() -> Optional["_threading.Thread"]:
    """Warm the provider-models disk cache in a background daemon thread.

    The no-args ``/model`` picker calls ``list_authenticated_providers()``,
    which fetches each authenticated provider's live ``/v1/models`` list on a
    cold/stale cache. Those fetches are independent HTTP round-trips but run
    serially, so the first ``/model`` open in a session (or any open after the
    1h cache TTL expires) blocks ~1-2s on the user's critical path.

    This pre-warms that exact path off-thread during idle session time: it
    runs ``list_authenticated_providers()`` once, which populates
    ``provider_models_cache.json`` for every authed provider. By the time the
    user types ``/model``, the picker hits the warm disk cache and renders in
    ~100ms.

    Fire-and-forget. Process-level Event guard ensures it runs at most once.
    Fully exception-isolated — a slow or offline provider can never affect the
    session. Returns the spawned thread (for tests) or None if already warmed.
    """
    if _picker_prewarm_done.is_set():
        return None
    _picker_prewarm_done.set()

    def _warm() -> None:
        try:
            from hermes_cli.inventory import load_picker_context

            ctx = load_picker_context()
            # Calling this is what populates cached_provider_model_ids() ->
            # provider_models_cache.json for each authed provider. We discard
            # the result; the side effect (warm disk cache) is the point.
            list_authenticated_providers(
                current_provider=ctx.current_provider,
                current_base_url=ctx.current_base_url,
                current_model=ctx.current_model,
                user_providers=ctx.user_providers,
                custom_providers=ctx.custom_providers,
                excluded_providers=ctx.excluded_providers or [],
            )
        except Exception:
            # Best-effort warmup — never surface errors into the session.
            logger.debug("picker cache prewarm failed", exc_info=True)

    t = _threading.Thread(target=_warm, daemon=True, name="picker-cache-prewarm")
    t.start()
    return t


def _scoped_key_env(name: str) -> str:
    """Read a provider key env var through the per-profile secret scope.

    The multiplexed gateway installs a secret scope per turn; a raw
    ``os.environ`` read hands the current profile whatever key happens to be
    in the process environment — another profile's, in a multiplexer. That is
    the class swept in 854007d1c for the fallback/aux key reads; the picker's
    ``key_env`` reads were not covered.

    Identical to ``os.getenv`` when multiplexing is off. A fail-closed
    ``UnscopedSecretError`` (multiplexing on, no scope installed) means "no
    credential visible for this profile here", which is exactly how the picker
    already treats a missing key.
    """
    if not name:
        return ""
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name, "") or "").strip()
    except Exception:
        return ""


# --- Parallel prefetch for provider model catalogs -----------------------
#
# When the 1h disk cache lapses (or on first cold open), list_authenticated_providers()
# calls cached_provider_model_ids() serially for each authed provider.  Each call
# that misses the cache blocks on a live /v1/models HTTP round-trip (1-8s per
# provider depending on endpoint latency).  With 10+ authed providers the
# cumulative serial blocking time is 15-30+ seconds.
#
# This prefetch function runs those same cached_provider_model_ids() calls in
# parallel via ThreadPoolExecutor before the main picker build loop starts.
# The main loop then hits warm cache entries instead of blocking on live
# fetches.  Providers whose cache was already fresh (SWR or within TTL) are
# skipped entirely — no wasted network calls.
#
# Net effect on a 13-provider setup with an expired cache:
#   Before: ~20s serial blocking (sum of all provider latencies)
#   After:  ~8s parallel (max single provider latency), rest served from cache

_PARALLEL_PREFETCH_WORKERS = 8


def _prefetch_provider_models_parallel(provider_slugs: list[str]) -> None:
    """Fetch model catalogs for multiple providers in parallel.

    Only providers whose cache entry is stale or missing are fetched; fresh
    entries are skipped to avoid unnecessary network calls.  Each worker uses
    :func:`update_provider_cache_entry` (thread-safe) to persist its result,
    so concurrent writes to ``provider_models_cache.json`` don't clobber each
    other.

    :param provider_slugs: Hermes provider IDs to prefetch (e.g. ``["openrouter",
        "anthropic", "deepseek"]``).  Unknown providers are silently skipped.
    """
    from hermes_cli.models import cached_provider_model_ids

    # Quick-stale-check: skip providers whose cache is already fresh so we
    # don't waste network calls on a warm cache.  We check staleness the same
    # way cached_provider_model_ids does internally: load the cache, compare
    # age to TTL.  This is a read-only check — if the cache file changes
    # between this check and the actual fetch, cached_provider_model_ids will
    # still do the right thing (it re-reads the cache internally).
    from hermes_cli.models import (
        _load_provider_models_cache,
        _credential_fingerprint,
        _PROVIDER_MODELS_CACHE_TTL,
        normalize_provider,
    )

    now = time.time()
    stale_slugs: list[str] = []
    cache = _load_provider_models_cache()
    for slug in provider_slugs:
        normalized = normalize_provider(slug) or (slug or "")
        if not normalized:
            continue
        entry = cache.get(normalized)
        fp = _credential_fingerprint(normalized)
        if (
            isinstance(entry, dict)
            and entry.get("fp") == fp
            and isinstance(entry.get("models"), list)
            and entry["models"]
        ):
            age = now - float(entry.get("at", 0))
            if age < _PROVIDER_MODELS_CACHE_TTL:
                continue  # fresh, skip
        stale_slugs.append(normalized)

    if not stale_slugs:
        return

    import concurrent.futures

    def _fetch_one(slug: str) -> None:
        try:
            models = cached_provider_model_ids(slug, force_refresh=True)
            # cached_provider_model_ids already persists the result, but in a
            # non-locked read-modify-write.  Re-persist via the thread-safe
            # path to guarantee no lost writes under concurrency.
            if models:
                from hermes_cli.models import update_provider_cache_entry
                update_provider_cache_entry(slug, models)
        except Exception:
            pass  # best-effort; picker falls back to curated list

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_PARALLEL_PREFETCH_WORKERS, len(stale_slugs)),
        thread_name_prefix="model-cache-prefetch",
    ) as executor:
        list(executor.map(_fetch_one, stale_slugs))


# --- Provider-row discovery shared by the picker and the prefetch scan -------
#
# ``list_authenticated_providers`` builds picker rows in sections:
#   1  built-in providers mapped to models.dev (PROVIDER_TO_MODELS_DEV)
#   2  Hermes-only overlays (nous, openai-codex, copilot, opencode-go, ...)
#   2b canonical providers missed by 1/2 (keeps /model in sync with `hermes model`)
#   3  ``providers:`` dict entries from config, 3b the bare active custom endpoint
#   4  ``custom_providers:`` list entries
# ``_collect_authed_provider_slugs`` mirrors the credential checks of 1/2/2b
# without fetching model lists. The helpers below are the single copy of each
# check; every ``from hermes_cli.auth/models import`` stays lazy so tests can
# patch those modules.


def _iter_builtin_candidates(models_dev_data: dict, excluded: set, seen: set):
    """Yield ``(hermes_id, mdev_id, pconfig, env_vars)`` for section-1 rows.

    Skips vendor names that are aliases routing through an aggregator (bare
    "openai" -> "openrouter": emitting them would silently switch a user onto an
    endpoint they may have no key for), hermes_ids that are aliases of another
    canonical profile ("kimi" -> "kimi-coding"), non-api_key auth types (section
    2 handles them with auth-store checks), and providers Hermes cannot route.
    PROVIDER_REGISTRY env var names win over models.dev's (which can be wrong).
    """
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY, is_runtime_provider_routable
    from hermes_cli.models import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import ALIASES

    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        alias_target = ALIASES.get(hermes_id)
        if alias_target and alias_target != hermes_id and alias_target in _AGGREGATOR_PROVIDERS:
            continue
        canonical = hermes_id
        try:
            from providers import get_provider_profile
            prof = get_provider_profile(hermes_id)
            if prof is not None:
                canonical = prof.name
        except Exception:
            pass
        if canonical != hermes_id or hermes_id.lower() in seen:
            continue
        if hermes_id.lower() in excluded or mdev_id.lower() in excluded:
            continue
        pdata = models_dev_data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue
        pconfig = PROVIDER_REGISTRY.get(hermes_id)
        if pconfig and pconfig.auth_type != "api_key":
            continue
        if not is_runtime_provider_routable(hermes_id):
            continue
        if pconfig and pconfig.api_key_env_vars:
            env_vars = list(pconfig.api_key_env_vars)
        else:
            env_vars = pdata.get("env", [])
            if not isinstance(env_vars, list):
                continue
        yield hermes_id, mdev_id, pconfig, env_vars


def _auth_store_has_provider(*keys: str) -> bool:
    """True when ``auth.json`` has a ``providers`` entry under any of *keys*."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        providers_store = store.get("providers", {})
        return bool(store and any(k in providers_store for k in keys))
    except Exception as exc:
        logger.debug("Auth store check failed for %s: %s", keys[0] if keys else "", exc)
        return False


def _raw_pool_usable(hermes_id: str) -> bool:
    """Section-1 pool check: only consult the pool when auth.json lists a raw entry."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        if store and store.get("credential_pool", {}).get(hermes_id):
            return _credential_pool_is_usable(hermes_id, raw_pool_present=True)
    except Exception:
        pass
    return False


def _pool_usable(slug: str) -> bool:
    try:
        return _credential_pool_is_usable(slug)
    except Exception as exc:
        logger.debug("Credential pool check failed for %s: %s", slug, exc)
        return False


def _overlay_has_env_creds(pid: str, hermes_slug: str, overlay, read_env) -> bool:
    """Section-2 env/SDK credential check shared by the picker and the prefetch scan.

    Vertex authenticates via OAuth2 (service-account JSON / ADC), not an API
    key, so it gets its own probe; otherwise the provider is hidden from the
    picker even when fully configured.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    has_creds = False
    if overlay.auth_type == "vertex":
        try:
            from agent.vertex_adapter import has_vertex_credentials
            has_creds = has_vertex_credentials()
        except Exception as exc:
            logger.debug("Vertex credential check failed: %s", exc)
    elif overlay.extra_env_vars:
        has_creds = any(read_env(ev) for ev in overlay.extra_env_vars)
    if not has_creds and overlay.auth_type == "api_key":
        for key in (pid, hermes_slug):
            pcfg = PROVIDER_REGISTRY.get(key)
            if pcfg and pcfg.api_key_env_vars and any(read_env(ev) for ev in pcfg.api_key_env_vars):
                return True
    return has_creds


def _has_fast_aws_sdk_signal() -> bool:
    """True when explicit AWS auth config is present in the environment.

    Deliberately avoids botocore's full credential chain: picker discovery runs
    for non-Bedrock providers too, and botocore may probe EC2 IMDS
    (169.254.169.254) on local machines before returning no credentials.
    """
    env = os.environ
    if env.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        return True
    if env.get("AWS_ACCESS_KEY_ID", "").strip() and env.get("AWS_SECRET_ACCESS_KEY", "").strip():
        return True
    return any(
        env.get(name, "").strip()
        for name in (
            "AWS_PROFILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
    )


def _has_aws_sdk_creds_for_listing(slug: str, current_provider: str) -> bool:
    """Credential check for AWS SDK providers in non-runtime discovery.

    The full boto3 chain is only consulted for the *current* provider.
    """
    if _has_fast_aws_sdk_signal():
        return True
    if str(slug or "").strip().lower() != str(current_provider or "").strip().lower():
        return False
    try:
        from agent.bedrock_adapter import has_aws_credentials
        return bool(has_aws_credentials())
    except Exception:
        return False


def _is_aws_sdk(pconfig) -> bool:
    return bool(pconfig) and getattr(pconfig, "auth_type", "") == "aws_sdk"


def _live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str, merge_models_dev: bool = True) -> list:
    """Unified pathway: ``cached_provider_model_ids`` so the /model picker sees the
    SAME list ``hermes model`` builds (disk-cached), falling back to the curated
    static list (merged with models.dev for preferred providers) when live is empty.
    """
    from hermes_cli.models import _MODELS_DEV_PREFERRED, _merge_with_models_dev, cached_provider_model_ids

    model_ids = cached_provider_model_ids(slug)
    if not model_ids:
        for key in fallback_keys or (slug,):
            model_ids = curated.get(key, [])
            if model_ids:
                break
        if merge_models_dev and slug in _MODELS_DEV_PREFERRED:
            model_ids = _merge_with_models_dev(slug, model_ids)
    return model_ids


def _aws_live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str) -> list:
    """Bedrock: live discovery reflects the active region (eu.*, ap.*) rather than
    the static us.* list; any failure falls back to the curated list."""
    from hermes_cli.models import cached_provider_model_ids

    fallback_keys = fallback_keys or (slug,)
    try:
        ids = cached_provider_model_ids(slug)
        if ids:
            return ids
    except Exception:
        pass
    for key in fallback_keys:
        ids = curated.get(key, [])
        if ids:
            return ids
    return []


def _nous_picker_model_ids(curated: dict, force_fresh_nous_tier: bool) -> list:
    """Nous serves a huge alphabetical live catalog; the picker shows ONLY the
    curated agentic list, augmented with the Portal's free/paid recommendations
    (so newly launched models surface without a CLI release) and narrowed by org
    policy. Mirrors ``_model_flow_nous`` so GUI pickers match the CLI. A failed
    recommendation fetch still yields a policy-filtered curated list.
    """
    model_ids = curated.get("nous", [])
    try:
        from hermes_cli.models import (
            get_pricing_for_provider,
            check_nous_free_tier,
            union_with_portal_free_recommendations,
            union_with_portal_paid_recommendations,
        )
        from hermes_cli.auth import get_provider_auth_state

        pricing = get_pricing_for_provider("nous") or {}
        try:
            portal = (get_provider_auth_state("nous") or {}).get("portal_base_url", "") or ""
        except Exception:
            portal = ""
        if check_nous_free_tier(force_fresh=force_fresh_nous_tier):
            model_ids, _ = union_with_portal_free_recommendations(model_ids, pricing, portal)
        else:
            model_ids, _ = union_with_portal_paid_recommendations(model_ids, pricing, portal)
    except Exception:
        pass
    try:
        from hermes_cli.models import nous_policy_allowed_ids, restrict_to_nous_policy

        model_ids = restrict_to_nous_policy(model_ids, nous_policy_allowed_ids(), rescue_empty=True)
    except Exception:
        pass
    return model_ids


def _cap_models(model_ids: list, max_models: int | None, slug: str = "") -> list:
    """Apply ``max_models``; aggregators in ``_UNCAPPED_PICKER_PROVIDERS`` show everything."""
    if slug in _UNCAPPED_PICKER_PROVIDERS or max_models is None:
        return model_ids
    return model_ids[:max_models]


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _entry_base_url(entry: dict, keys: tuple = ("base_url", "url", "api")) -> str:
    for key in keys:
        value = entry.get(key, "")
        if value:
            return value
    return ""


def _entry_api_mode(entry: dict) -> str | None:
    return str(entry.get("api_mode") or entry.get("transport") or "").strip().lower() or None


def _credential_identity(inline_api_key: str, key_env: str) -> str:
    return inline_api_key if inline_api_key else (f"env:{key_env}" if key_env else "")


def _discover_flag(entry: dict):
    """``discover_models`` (default True); ``"false"/"no"/"0"`` strings mean False."""
    discover = entry.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    return discover


def _display_prefix(name: str) -> str:
    """Text before the per-model separator Hermes's own writer uses ("—" / " - ")."""
    for sep in ("—", " - "):
        if sep in name:
            return name.split(sep)[0].strip()
    return name


def _discover_endpoint_models(
    api_key: str,
    api_url: str,
    native_catalog_provider: str,
    has_explicit_models: bool,
    *,
    headers: dict | None,
    api_mode: str | None,
    probe_live: bool,
    discovery_allowed: bool,
    for_picker: bool,
) -> tuple[list | None, bool]:
    """Return ``(models, native_catalog_empty)`` for a custom endpoint row.

    ``probe_live`` runs the native-aware picker fetch; otherwise, when discovery
    is allowed, a warm same-fingerprint cache entry still serves the full catalog
    with no round-trip. ``has_explicit_models`` gates the *probe* (a network-cost
    guard for keyless endpoints that declare a catalog), never the cache read —
    applying it to the read re-pins the endpoint to its declared subset. Returns
    ``(None, False)`` when nothing usable was found.
    """
    timeout = 1.5 if for_picker else 5.0
    if probe_live:
        try:
            live_models = _fetch_picker_live_models(
                api_key, api_url, native_catalog_provider, has_explicit_models,
                headers=headers, timeout=timeout, api_mode=api_mode,
            )
            is_native = isinstance(live_models, _NativePickerModelList)
            if live_models is not None and (live_models or not has_explicit_models or is_native):
                return live_models, (is_native and not live_models)
        except Exception:
            pass
    elif discovery_allowed:
        try:
            from hermes_cli.models import cached_fetch_api_models

            cached_models = cached_fetch_api_models(
                api_key, api_url, cache_only=True, timeout=timeout, headers=headers, api_mode=api_mode,
            )
            if cached_models:
                return cached_models, False
        except _MODEL_DISCOVERY_ERRORS:
            pass
    return None, False


def _collect_authed_provider_slugs(
    models_dev_data: dict,
    curated: dict[str, list[str]],
    excluded: list[str],
) -> list[str]:
    """Quick-scan which providers have credentials, without fetching model lists.

    Mirrors the credential checks of sections 1, 2 and 2b of
    :func:`list_authenticated_providers` but never calls
    ``cached_provider_model_ids``; the result feeds
    :func:`_prefetch_provider_models_parallel`. Env vars are read through the
    per-profile secret scope. AWS SDK providers are skipped (heavier detection).
    """
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import HERMES_OVERLAYS
    from hermes_cli.models import CANONICAL_PROVIDERS

    excluded_set = {str(p).strip().lower() for p in excluded if p}
    slugs: list[str] = []
    seen: set[str] = set()

    for hermes_id, _mdev_id, _pconfig, env_vars in _iter_builtin_candidates(models_dev_data, excluded_set, seen):
        if any(_scoped_key_env(ev) for ev in env_vars) or _raw_pool_usable(hermes_id):
            slugs.append(hermes_id)
            seen.add(hermes_id.lower())

    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if pid.lower() in seen or hermes_slug.lower() in seen:
            continue
        if pid.lower() in excluded_set or hermes_slug.lower() in excluded_set:
            continue
        if overlay.auth_type == "aws_sdk":
            continue
        if (
            _overlay_has_env_creds(pid, hermes_slug, overlay, _scoped_key_env)
            or _auth_store_has_provider(pid, hermes_slug)
            or _pool_usable(hermes_slug)
        ):
            slugs.append(hermes_slug)
            seen.add(pid.lower())
            seen.add(hermes_slug.lower())

    for cp in CANONICAL_PROVIDERS:
        if cp.slug.lower() in seen or cp.slug.lower() in excluded_set:
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = bool(
            cp_config and cp_config.api_key_env_vars and any(_scoped_key_env(ev) for ev in cp_config.api_key_env_vars)
        )
        if has_creds or _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug):
            slugs.append(cp.slug)
            seen.add(cp.slug.lower())

    # Nous excluded: its picker branch builds from the curated list and never
    # reads the api_key-only cache entry a prefetch would write.
    return [s for s in slugs if s != "nous"]


@dataclass
class _PickerBuild:
    """Mutable state threaded through the ``list_authenticated_providers`` sections."""

    current_provider: str
    current_base_url: str
    current_model: str
    max_models: int | None
    for_picker: bool
    force_fresh_nous_tier: bool
    probe_custom_providers: bool
    probe_current_custom_provider: bool
    refresh: bool
    excluded: set
    curated: dict
    results: list = field(default_factory=list)
    seen_slugs: set = field(default_factory=set)  # lowercase-normalized to catch case variants
    # Effective base URLs of every built-in row, so section 4 hides
    # ``custom_providers`` entries that duplicate a built-in endpoint.
    builtin_endpoints: set = field(default_factory=set)
    # (display_name, base_url) pairs emitted by section 3 so section 4 skips
    # overlapping ``custom_providers`` rows (callers often pass both).
    section3_pairs: set = field(default_factory=set)
    current_provider_norm: str = field(init=False)
    current_base_url_norm: str = field(init=False)

    def __post_init__(self):
        self.current_provider_norm = self.current_provider.lower()
        self.current_base_url_norm = self.current_base_url.rstrip("/").lower()

    def can_probe_custom(self, *, row_is_current: bool) -> bool:
        return bool(self.probe_custom_providers or (self.probe_current_custom_provider and row_is_current))

    def record_builtin_endpoint(self, slug: str) -> None:
        """Prefer the live env override (e.g. DASHSCOPE_BASE_URL) over the static
        inference_base_url so dedup matches what a user typing that URL into
        custom_providers would actually hit."""
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
        except Exception:
            return
        pcfg = PROVIDER_REGISTRY.get(slug)
        if not pcfg:
            return
        url = os.environ.get(pcfg.base_url_env_var, "") if getattr(pcfg, "base_url_env_var", "") else ""
        normed = _norm_url(url or getattr(pcfg, "inference_base_url", "") or "")
        if normed:
            self.builtin_endpoints.add(normed)

    def add_builtin_row(self, slug: str, name: str, is_current: bool, model_ids: list, source: str, *, uncapped_ok: bool = True) -> None:
        self.results.append({
            "slug": slug,
            "name": name,
            "is_current": is_current,
            "is_user_defined": False,
            "models": _cap_models(model_ids, self.max_models, slug if uncapped_ok else ""),
            "total_models": len(model_ids),
            "source": source,
        })
        self.seen_slugs.add(slug.lower())
        self.record_builtin_endpoint(slug)


def _lap_builtin_rows(b: _PickerBuild, data: dict, user_providers: dict) -> None:
    """Section 1: models.dev-mapped providers with api_key auth."""
    from agent.models_dev import get_provider_info

    for hermes_id, mdev_id, pconfig, env_vars in _iter_builtin_candidates(data, b.excluded, b.seen_slugs):
        if not (any(os.environ.get(ev) for ev in env_vars) or _raw_pool_usable(hermes_id)):
            continue
        model_ids = _live_or_curated_ids(hermes_id, b.curated)
        # A providers.<built-in>.models block extends the discovered catalog;
        # section 3 cannot emit it later because this row owns the slug.
        configured = user_providers.get(hermes_id) if isinstance(user_providers, dict) else None
        configured_models = _declared_model_ids(configured.get("models")) if isinstance(configured, dict) else []
        model_ids = list(dict.fromkeys([*configured_models, *model_ids]))
        pinfo = get_provider_info(mdev_id)
        display_name = pconfig.name if pconfig and pconfig.name else (pinfo.name if pinfo else mdev_id)
        b.add_builtin_row(
            hermes_id, display_name, b.current_provider in (hermes_id, mdev_id), model_ids, "built-in",
        )


def _lap_overlay_rows(b: _PickerBuild, data: dict) -> None:
    """Section 2: Hermes-only providers (nous, openai-codex, copilot, opencode-go, ...)."""
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.providers import HERMES_OVERLAYS

    # HERMES_OVERLAYS keys may be models.dev IDs ("github-copilot") while
    # config.yaml uses Hermes IDs ("copilot").
    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if pid.lower() in b.seen_slugs or hermes_slug.lower() in b.seen_slugs:
            continue
        if pid.lower() in b.excluded or hermes_slug.lower() in b.excluded:
            continue

        if getattr(overlay, "keyless", False):
            has_creds = True  # served anonymously (opencode-free)
        elif overlay.auth_type == "aws_sdk":
            has_creds = _has_aws_sdk_creds_for_listing(hermes_slug, b.current_provider)
        else:
            has_creds = _overlay_has_env_creds(pid, hermes_slug, overlay, os.environ.get)
        # External-process providers (copilot-acp) hold no key/token/pool entry by
        # design — the spawned ACP subprocess brings its own auth. "Configured"
        # means the executable resolves, which is what get_auth_status() reports;
        # without this the has_creds filter hides the provider from every picker.
        if not has_creds and overlay.auth_type == "external_process":
            try:
                from hermes_cli.auth import get_auth_status
                _ext_status = get_auth_status(hermes_slug) or {}
                has_creds = bool(_ext_status.get("logged_in") or _ext_status.get("configured"))
            except Exception as exc:
                logger.debug("External-process check failed for %s: %s", pid, exc)
        # Auth store / credential pool cover OAuth providers AND api_key providers
        # that also support OAuth (anthropic via Claude Code credential files).
        if not has_creds:
            has_creds = _auth_store_has_provider(pid, hermes_slug)
        if not has_creds:
            # Full auto-seeding pool check catches external stores (Codex CLI
            # ~/.codex/auth.json) not yet in auth.json.
            try:
                if _credential_pool_is_usable(hermes_slug):
                    has_creds = True
                elif b.for_picker:
                    # Show providers whose pool is entirely in cooldown: limits are
                    # per-model for many providers, so another model may work.
                    try:
                        from agent.credential_pool import load_pool
                        has_creds = load_pool(hermes_slug).has_credentials()
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Credential pool check failed for %s: %s", hermes_slug, exc)
        if not has_creds and hermes_slug == "anthropic":
            # The pool gates anthropic behind is_provider_explicitly_configured()
            # (aux tasks must not consume Claude Code tokens); the picker is
            # discovery-oriented, so read the external credential files directly.
            try:
                from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials
                hermes_creds = read_hermes_oauth_credentials()
                cc_creds = read_claude_code_credentials()
                if (hermes_creds and hermes_creds.get("accessToken")) or (cc_creds and cc_creds.get("accessToken")):
                    has_creds = True
            except Exception as exc:
                logger.debug("Anthropic external creds check failed: %s", exc)
        if not has_creds:
            continue

        if hermes_slug in {"openai-codex", "copilot", "copilot-acp"}:
            # Live OAuth-backed discovery so Pro-only Codex slugs not in the static
            # catalog appear; falls back to curated when unreachable.
            from hermes_cli.models import cached_provider_model_ids
            model_ids = cached_provider_model_ids(hermes_slug)
        elif overlay.auth_type == "aws_sdk":
            model_ids = _aws_live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        elif hermes_slug == "nous":
            model_ids = _nous_picker_model_ids(b.curated, b.force_fresh_nous_tier)
        else:
            model_ids = _live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        b.add_builtin_row(
            hermes_slug, get_label(hermes_slug), b.current_provider in (hermes_slug, pid), model_ids, "hermes",
        )
        b.seen_slugs.add(pid.lower())


def _lap_canonical_rows(b: _PickerBuild) -> None:
    """Section 2b: CANONICAL_PROVIDERS missed by sections 1/2."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    try:
        from hermes_cli.models import CANONICAL_PROVIDERS
    except ImportError:
        CANONICAL_PROVIDERS = []

    for cp in CANONICAL_PROVIDERS:
        if cp.slug.lower() in b.seen_slugs or cp.slug.lower() in b.excluded:
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = False
        if cp_config and cp_config.api_key_env_vars:
            lit = {ev for ev in cp_config.api_key_env_vars if os.environ.get(ev)}
            has_creds = bool(lit)
            # A regional "-cn" twin lit only by key vars shared with its non-CN
            # sibling is a phantom row: hide it unless it is the current provider,
            # and only when it has a dedicated var of its own the user could set.
            sib = PROVIDER_REGISTRY.get(cp.slug[:-3]) if cp.slug.endswith("-cn") else None
            sib_vars = set(sib.api_key_env_vars) if sib else set()
            if lit and lit <= sib_vars < set(cp_config.api_key_env_vars) and cp.slug != b.current_provider:
                continue
        if not has_creds:
            has_creds = _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug)
        if not has_creds and _is_aws_sdk(cp_config):
            has_creds = _has_aws_sdk_creds_for_listing(cp.slug, b.current_provider)
        if not has_creds:
            continue
        if _is_aws_sdk(cp_config):
            model_ids = _aws_live_or_curated_ids(cp.slug, b.curated)
        else:
            model_ids = _live_or_curated_ids(cp.slug, b.curated, merge_models_dev=False)
        b.add_builtin_row(
            cp.slug, cp.label, cp.slug == b.current_provider, model_ids, "canonical", uncapped_ok=False,
        )


def _lap_user_provider_rows(b: _PickerBuild, user_providers: dict) -> None:
    """Section 3: ``providers:`` dict entries, grouped by (api_url, credential,
    api_mode, extra_headers) so keyed providers on one endpoint with the same
    wire protocol collapse into one row (e.g. two Palantir Claude entries ->
    one "Palantir Claude" row); a different key_env/api_mode/headers keeps
    distinct rows since the wire protocol or tenant differs."""
    from collections import OrderedDict
    from hermes_cli.config import coerce_provider_id, is_provider_enabled

    ep_groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for ep_name, ep_cfg in user_providers.items():
        if not isinstance(ep_cfg, dict) or not is_provider_enabled(ep_cfg):
            continue
        if ep_name.lower() in b.seen_slugs:
            continue
        display_name = coerce_provider_id(ep_cfg.get("name")) or ep_name
        api_url = _entry_base_url(ep_cfg, ("base_url", "api", "url"))
        key_env = str(ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or "").strip()
        inline_api_key = str(ep_cfg.get("api_key", "") or "").strip()
        api_mode = _entry_api_mode(ep_cfg)
        headers_identity = tuple(sorted(_extra_headers_from_config(ep_cfg).items()))
        group_key = (_norm_url(api_url), _credential_identity(inline_api_key, key_env), api_mode, headers_identity)

        # ``default_model`` is the legacy key; ``model`` matches custom_providers.
        default_model = ep_cfg.get("default_model", "") or ep_cfg.get("model", "")
        entry_models = [default_model] if default_model else []
        for model_id in _declared_model_ids(ep_cfg.get("models", [])):
            if model_id not in entry_models:
                entry_models.append(model_id)

        if group_key not in ep_groups:
            # Strip the per-model suffix and trailing version tokens ("Palantir
            # Claude 4.7 Opus" -> "Palantir Claude"): cut at the first token with
            # a digit, only when >=2 words remain (avoids over-trimming).
            grp_display = _display_prefix(display_name)
            toks = grp_display.split()
            cut_at = next((i for i, t in enumerate(toks) if any(c.isdigit() for c in t.strip(".,()"))), None)
            if cut_at is not None and cut_at >= 2:
                grp_display = " ".join(toks[:cut_at]).strip()
            ep_groups[group_key] = {
                "slug": ep_name,  # first ep_name encountered
                "name": grp_display or display_name,
                "api_url": api_url,
                "models": [],
                "has_explicit_models": False,
                "ep_cfg": ep_cfg,
                "raw_names": [],
                "aliases": set(),
            }
        grp = ep_groups[group_key]
        for m in entry_models:
            if m and m not in grp["models"]:
                grp["models"].append(m)
        # A singular default_model/model is only the active selection and must
        # not suppress discovery; dict-shaped ``models:`` is context_length
        # metadata, not an allowlist — see ``_models_config_is_allowlist``.
        if _models_config_is_allowlist(ep_cfg.get("models"), _entry_models_discovered(ep_cfg)):
            grp["has_explicit_models"] = True
        grp["raw_names"].append(display_name)
        grp["aliases"].update(custom_provider_aliases(display_name, str(ep_name)))

    for grp in ep_groups.values():
        ep_cfg, ep_name, display_name, api_url = grp["ep_cfg"], grp["slug"], grp["name"], grp["api_url"]
        models_list = list(grp["models"])
        # Official OpenAI rows often have base_url but no models: dict — avoid a
        # misleading zero count.
        if not models_list and base_url_host_matches(str(api_url).strip().lower(), "api.openai.com"):
            models_list = list(b.curated.get("openai") or [])

        # Probe policy (mirrors section 4): with an api_key always probe; without
        # one, skip only when an allowlist-shaped ``models:`` narrows the endpoint.
        api_key = str(ep_cfg.get("api_key", "") or "").strip()
        if not api_key:
            key_env = str(ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or "").strip()
            api_key = _scoped_key_env(key_env) if key_env else ""
        has_explicit_models = bool(grp.get("has_explicit_models"))
        ep_url_norm = _norm_url(api_url)
        ep_aliases = {str(alias).lower() for alias in grp.get("aliases", set())}
        is_current = (
            str(ep_name).strip().lower() == b.current_provider_norm
            or b.current_provider_norm in ep_aliases
            or (
                b.current_provider_norm == "custom"
                and bool(b.current_base_url_norm)
                and ep_url_norm == b.current_base_url_norm
            )
        )
        discovery_allowed = bool(api_url) and _discover_flag(ep_cfg)
        discovered, native_catalog_empty = _discover_endpoint_models(
            api_key,
            api_url,
            ep_name if str(ep_name).strip().lower() in {"ollama", "custom:ollama"} else "custom",
            has_explicit_models,
            headers=_extra_headers_from_config(ep_cfg) or None,
            api_mode=ep_cfg.get("api_mode"),
            probe_live=(
                discovery_allowed
                and (bool(api_key) or not has_explicit_models)
                and b.can_probe_custom(row_is_current=is_current)
            ),
            discovery_allowed=discovery_allowed,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            models_list = discovered

        b.results.append({
            "slug": ep_name,
            "name": display_name,
            "is_current": is_current,
            "is_user_defined": True,
            "models": models_list,
            "total_models": len(models_list) if models_list else 0,
            "source": "user-config",
            "api_url": api_url,
            "native_catalog_empty": native_catalog_empty,
        })
        b.seen_slugs.add(ep_name.lower())
        b.seen_slugs.update(ep_aliases)
        # Record every raw member name so section 4 can match per-model
        # custom_providers rows even though the group label was collapsed.
        for raw_name in grp.get("raw_names") or [display_name]:
            pair = (str(raw_name).strip().lower(), ep_url_norm)
            if pair[0] and pair[1]:
                b.section3_pairs.add(pair)
                b.seen_slugs.add(custom_provider_slug(raw_name).lower())
        pair = (str(display_name).strip().lower(), ep_url_norm)
        if pair[0] and pair[1]:
            b.section3_pairs.add(pair)


def _lap_bare_custom_row(b: _PickerBuild, custom_providers: list | None) -> None:
    """Section 3b: ``model.provider: custom`` + ``model.base_url`` with no named
    providers:/custom_providers row — surface it so /model does not look like it
    ignored config.yaml."""
    if not (b.current_provider_norm == "custom" and b.current_base_url and "custom" not in b.seen_slugs):
        return
    if any(
        isinstance(cp, dict) and _norm_url(_entry_base_url(cp)) == _norm_url(b.current_base_url)
        for cp in (custom_providers or [])
    ):
        return
    api_url = str(b.current_base_url).strip().rstrip("/")
    models = [b.current_model] if b.current_model else []
    native_catalog_empty = False
    try:
        discovered, native_catalog_empty = _discover_endpoint_models(
            "", api_url, "custom", False,
            headers=None, api_mode=None,
            probe_live=bool(b.refresh or b.probe_current_custom_provider),
            discovery_allowed=True,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            models = discovered
    except Exception:
        pass
    b.results.append({
        "slug": "custom",
        "name": "Custom endpoint",
        "is_current": True,
        "is_user_defined": True,
        "models": _cap_models(models, b.max_models),
        "total_models": len(models),
        "source": "model-config",
        "api_url": api_url,
        "native_catalog_empty": native_catalog_empty,
    })
    b.seen_slugs.add("custom")


def _lap_custom_provider_rows(b: _PickerBuild, custom_providers: list) -> None:
    """Section 4: ``custom_providers:`` entries (one model each) grouped into one
    row per (endpoint, credential identity, api_mode, extra_headers, display
    prefix). Four "Ollama — X" entries on one host become one "Ollama" row;
    distinct prefixes sharing a proxy URL keep their own rows."""
    from collections import OrderedDict
    from hermes_cli.config import coerce_provider_id

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        raw_name = coerce_provider_id(entry.get("name"))
        api_url = str(_entry_base_url(entry) or "").strip().rstrip("/")
        if not raw_name or not api_url:
            continue
        inline_api_key = str(entry.get("api_key") or "").strip()
        key_env = str(entry.get("key_env") or "").strip()
        api_key = inline_api_key or _scoped_key_env(key_env)
        api_mode = _entry_api_mode(entry)
        discover = _discover_flag(entry)
        entry_extra_headers = _extra_headers_from_config(entry)
        prefix = _display_prefix(raw_name)
        group_key = (
            api_url, _credential_identity(inline_api_key, key_env), api_mode,
            tuple(sorted(entry_extra_headers.items())), prefix.lower(),
        )
        if group_key not in groups:
            display_name = prefix or raw_name
            groups[group_key] = {
                "slug": custom_provider_slug(display_name, str(entry.get("provider_key") or "").strip()),
                "name": display_name,
                "api_url": api_url,
                "api_key": api_key,
                "models": [],
                "has_explicit_models": False,
                "discover_models": discover,
                "api_mode": api_mode,
                "extra_headers": entry_extra_headers,
                "aliases": set(),
            }
        else:
            if api_key and not groups[group_key].get("api_key"):
                groups[group_key]["api_key"] = api_key
            if not discover:  # one opt-out pins the whole grouped row
                groups[group_key]["discover_models"] = False
        grp = groups[group_key]
        grp["aliases"].update(custom_provider_aliases(raw_name, str(entry.get("provider_key") or "")))
        # ``model:`` is only the active selection; every configured model lives
        # under ``models:`` (dict written by _save_custom_provider).
        default_model = (entry.get("model") or "").strip()
        if default_model and default_model not in grp["models"]:
            grp["models"].append(default_model)
        models_field = entry.get("models", {})
        if _models_config_is_allowlist(models_field, _entry_models_discovered(entry)):
            grp["has_explicit_models"] = True
        for model_id in _declared_model_ids(models_field):
            if model_id not in grp["models"]:
                grp["models"].append(model_id)

    section4_slugs: set = set()
    current_url_group_count = sum(
        1 for grp in groups.values()
        if b.current_base_url_norm and _norm_url(grp["api_url"]) == b.current_base_url_norm
    )
    for grp in groups.values():
        api_url, api_key, slug = grp["api_url"], grp.get("api_key", ""), grp["slug"]
        # Slug claimed by a built-in/overlay/providers: row -> skip (don't shadow).
        if slug.lower() in b.seen_slugs and slug.lower() not in section4_slugs:
            continue
        # Two custom endpoints with the same cleaned name: suffix a counter so
        # both stay visible.
        if slug.lower() in section4_slugs:
            base_slug, n = slug, 2
            while f"{base_slug}-{n}".lower() in b.seen_slugs:
                n += 1
            slug = f"{base_slug}-{n}"
            grp["slug"] = slug
        grp_url_norm = _norm_url(api_url)
        pair_key = (str(grp["name"]).strip().lower(), grp_url_norm)
        if pair_key[0] and pair_key[1] and pair_key in b.section3_pairs:
            continue
        # A built-in row already represents this endpoint (e.g. "my-dashscope"
        # vs the alibaba-coding-plan row): keep the built-in, hide the shadow.
        if grp_url_norm and grp_url_norm in b.builtin_endpoints:
            continue
        is_current = (
            slug.lower() == b.current_provider_norm
            or b.current_provider_norm in {str(alias).lower() for alias in grp.get("aliases", set())}
        ) or (
            b.current_provider_norm == "custom"
            and bool(b.current_base_url_norm)
            and grp_url_norm == b.current_base_url_norm
            and current_url_group_count == 1
        )
        # Probe policy: with an api_key live /models is the source of truth (replace
        # the partial ``models:`` subset); without one, an allowlist-shaped
        # ``models:`` narrows a public endpoint and skips the probe. A dict-shaped
        # ``models:`` is metadata, so still probe; pin with discover_models: false.
        has_explicit_models = bool(grp.get("has_explicit_models"))
        discovery_allowed = bool(api_url) and grp.get("discover_models", True)
        probe_live = (
            discovery_allowed
            and (bool(api_key) or not has_explicit_models)
            and b.can_probe_custom(row_is_current=is_current)
        )
        discovered, native_catalog_empty = _discover_endpoint_models(
            api_key,
            api_url,
            "ollama" if "ollama" in {str(slug).strip().lower(), str(grp.get("name") or "").strip().lower()} else "custom",
            has_explicit_models,
            headers=grp.get("extra_headers") or None,
            api_mode=grp.get("api_mode"),
            probe_live=probe_live,
            discovery_allowed=discovery_allowed,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            grp["models"] = discovered
            if probe_live:
                # A successful live probe persists the catalog for no-probe surfaces.
                try:
                    _save_discovered_models_to_config(
                        api_url, discovered, api_mode=grp.get("api_mode"), headers=grp.get("extra_headers") or None,
                    )
                except Exception:
                    pass
        b.results.append({
            "slug": slug,
            "name": grp["name"],
            "is_current": is_current,
            "is_user_defined": True,
            "models": grp["models"],
            "total_models": len(grp["models"]),
            "source": "user-config",
            "api_url": grp["api_url"],
            "native_catalog_empty": native_catalog_empty,
        })
        b.seen_slugs.add(slug.lower())
        section4_slugs.add(slug.lower())


def _build_curated_lists(current_provider: str, current_base_url: str, current_model: str) -> dict[str, list[str]]:
    """Curated model lists keyed by hermes provider id, plus the dynamic ones
    (nous manifest, Ollama Cloud, LM Studio live probe)."""
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS, get_curated_nous_model_ids

    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)
    curated["openrouter"] = [mid for mid, _ in OPENROUTER_MODELS]
    # Remote model-catalog manifest so new Portal models surface without a
    # release; falls back to the in-repo snapshot when unreachable.
    curated["nous"] = get_curated_nous_model_ids()
    if "ollama-cloud" not in curated:
        from hermes_cli.models import fetch_ollama_cloud_models
        curated["ollama-cloud"] = fetch_ollama_cloud_models()
    # LM Studio has no static catalog: probe its native endpoint live. Base URL
    # precedence: LM_BASE_URL > active config base_url (when current) > default.
    # On auth rejection / unreachable, fall back to the current model so the
    # picker still shows something offline.
    is_current_lmstudio = current_provider.strip().lower() == "lmstudio"
    if "lmstudio" not in curated and (os.environ.get("LM_API_KEY") or os.environ.get("LM_BASE_URL") or is_current_lmstudio):
        from hermes_cli.models import fetch_lmstudio_models
        from hermes_cli.auth import AuthError
        lm_base = (
            os.environ.get("LM_BASE_URL")
            or (current_base_url if is_current_lmstudio and current_base_url else None)
            or "http://127.0.0.1:1234/v1"
        )
        try:
            live = fetch_lmstudio_models(api_key=os.environ.get("LM_API_KEY", ""), base_url=lm_base, timeout=1.5)
        except AuthError:
            live = []
        if not live and is_current_lmstudio and current_model:
            live = [current_model]
        curated["lmstudio"] = live
    return curated


def list_authenticated_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    *,
    force_fresh_nous_tier: bool = False,
    max_models: int | None = None,
    current_model: str = "",
    refresh: bool = False,
    probe_custom_providers: bool = True,
    probe_current_custom_provider: bool = False,
    for_picker: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Detect which providers have credentials and list their curated models.

    Uses the curated lists from hermes_cli/models.py (OPENROUTER_MODELS,
    _PROVIDER_MODELS) — hand-picked agentic models, NOT the full models.dev
    catalog. Only providers with API keys set or user-defined endpoints appear.

    Returns a list of dicts: ``slug`` (the --provider value), ``name``,
    ``is_current``, ``is_user_defined``, ``models`` (up to max_models),
    ``total_models``, ``source`` ("built-in", "hermes", "canonical",
    "user-config", "model-config").

    ``force_fresh_nous_tier`` bypasses the short Nous tier cache for explicit
    account-sensitive flows; picker opens should leave it false.
    ``refresh`` busts the per-provider model-id disk cache up front so every row
    re-fetches live — for an explicit user "refresh models" action only.
    ``probe_custom_providers`` controls live ``/models`` discovery for saved
    custom endpoints (default true for CLI parity; GUI opens pass false).
    ``probe_current_custom_provider`` probes only the currently-selected custom
    endpoint so its list matches without blocking on offline ones.
    """
    from agent.models_dev import fetch_models_dev
    from hermes_cli.config import coerce_provider_id, stringify_provider_map

    # Explicit refresh: drop every cached model-id list so the calls below all
    # re-fetch live. A stale cache can fall back to the curated static list when
    # its live fetch fails, silently dropping live-only models the user had seen.
    if refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
        except Exception:
            pass

    # PyYAML parses unquoted numeric names (`provider: 2070`) as int.
    current_provider = coerce_provider_id(current_provider)
    current_base_url = str(current_base_url or "").strip()
    current_model = str(current_model or "").strip()
    user_providers = stringify_provider_map(user_providers)
    data = fetch_models_dev()

    b = _PickerBuild(
        current_provider=current_provider,
        current_base_url=current_base_url,
        current_model=current_model,
        max_models=max_models,
        for_picker=for_picker,
        force_fresh_nous_tier=force_fresh_nous_tier,
        probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_current_custom_provider,
        refresh=refresh,
        # A single entry like ``copilot`` hides the provider under every key it
        # surfaces as (hermes_id / mdev_id / canonical slug).
        excluded={str(p).strip().lower() for p in (excluded_providers or []) if p},
        curated=_build_curated_lists(current_provider, current_base_url, current_model),
    )

    # Warm the disk cache in parallel before the serial section loops, which
    # otherwise stack 15-30s of live /v1/models round-trips on a cold cache.
    # Skipped when refresh=True (serial path force-refreshes) and for <=3
    # providers (serial is fast enough; avoids thread-pool overhead).
    prefetch_slugs = [] if refresh else _collect_authed_provider_slugs(data, b.curated, excluded_providers or [])
    if len(prefetch_slugs) > 3:
        try:
            _prefetch_provider_models_parallel(prefetch_slugs)
        except Exception:
            pass  # best-effort; serial path still works

    _lap_builtin_rows(b, data, user_providers)
    _lap_overlay_rows(b, data)
    _lap_canonical_rows(b)
    if user_providers and isinstance(user_providers, dict):
        _lap_user_provider_rows(b, user_providers)
    _lap_bare_custom_row(b, custom_providers)
    if custom_providers and isinstance(custom_providers, list):
        _lap_custom_provider_rows(b, custom_providers)
    results = b.results

    # ``providers.<name>.enabled: false`` post-filter covers built-in rows
    # (sections 1-2) that bypass the per-section gate; matched by slug and
    # ``provider_id``.
    try:
        from hermes_cli.config import is_provider_enabled
        if isinstance(user_providers, dict):
            disabled = {
                str(name).strip().lower()
                for name, cfg in user_providers.items()
                if isinstance(cfg, dict) and not is_provider_enabled(cfg)
            }
            if disabled:
                results = [
                    r for r in results
                    if str(r.get("provider_id", "")).strip().lower() not in disabled
                    and str(r.get("slug", "")).strip().lower() not in disabled
                ]
    except Exception:
        pass

    # A custom/uncurated model set via `/model <provider>/<name>` would be
    # invisible in every picker (main and MoA slot pickers read these rows);
    # inject it at the front of the current provider's row as a uniform post-pass.
    if current_model:
        for row in results:
            if not row.get("is_current") or row.get("native_catalog_empty"):
                continue
            models = row.get("models") or []
            if current_model not in models:
                row["models"] = [current_model, *models]
                row["total_models"] = row.get("total_models", len(models)) + 1
            break

    # Current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))
    return results


def _prepend_moa_picker_provider(providers: List[dict], current_provider: str = "") -> List[dict]:
    """Add the virtual MoA provider row used by interactive model pickers.

    ``list_authenticated_providers()`` only returns real/auth-backed providers.
    The CLI model inventory adds MoA separately so named presets appear next to
    normal providers; gateway pickers call ``list_picker_providers()`` directly,
    so they need the same virtual row here. Reuse the inventory's single row
    builder so the row shape stays defined in one place.
    """
    try:
        from hermes_cli.inventory import _moa_provider_row

        moa_row = _moa_provider_row(current_provider)
        if moa_row is None:
            return providers
        return [moa_row] + [p for p in providers if str(p.get("slug", "")).lower() != "moa"]
    except Exception:
        return providers


def list_picker_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    max_models: int | None = None,
    current_model: str = "",
    include_moa: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Interactive-picker variant of :func:`list_authenticated_providers`.

    Post-processes the base list so the ``/model`` picker (Telegram/Discord
    inline keyboards) only surfaces models that are actually callable in the
    current install:

    - OpenRouter's model list is replaced with the output of
      :func:`hermes_cli.models.fetch_openrouter_models`, which filters the
      curated ``OPENROUTER_MODELS`` snapshot against the live OpenRouter
      catalog.  IDs the live catalog no longer carries drop out, so the
      picker never offers a model the user can't call.
    - Provider rows whose model list ends up empty are dropped, except
      custom endpoints (``is_user_defined=True`` with an ``api_url``) where
      the user may supply their own model set through config.

    All other providers and metadata fields are passed through unchanged.
    The typed ``/model <name>`` path is unaffected -- only the interactive
    picker payload is narrowed.
    """
    from hermes_cli.models import fetch_openrouter_models

    providers = list_authenticated_providers(
        current_provider=current_provider,
        current_base_url=current_base_url,
        user_providers=user_providers,
        custom_providers=custom_providers,
        max_models=max_models,
        current_model=current_model,
        for_picker=True,
        excluded_providers=excluded_providers,
    )
    if include_moa:
        providers = _prepend_moa_picker_provider(providers, current_provider=current_provider)

    filtered: List[dict] = []
    for p in providers:
        slug = str(p.get("slug", "")).lower()
        if slug == "openrouter":
            try:
                live = fetch_openrouter_models()
                live_ids = [mid for mid, _ in live]
            except Exception:
                live_ids = list(p.get("models", []))
            p = dict(p)
            p["models"] = live_ids[:max_models] if max_models is not None else live_ids
            p["total_models"] = len(live_ids)

        has_models = bool(p.get("models"))
        is_custom_endpoint = bool(p.get("is_user_defined")) and bool(p.get("api_url"))
        if not has_models and not is_custom_endpoint:
            continue
        filtered.append(p)

    return filtered
