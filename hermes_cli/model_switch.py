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

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

from hermes_cli.providers import (
    ProviderDef,
    custom_provider_aliases,
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
from utils import base_url_hostname, base_url_origin
from hermes_cli.model_switch_providers import (  # noqa: F401  (re-exported; tests patch hermes_cli.model_switch.<name>)
    _MODEL_DISCOVERY_ERRORS,
    _NativePickerModelList,
    _PARALLEL_PREFETCH_WORKERS,
    _PickerBuild,
    _UNCAPPED_PICKER_PROVIDERS,
    _auth_store_has_provider,
    _aws_live_or_curated_ids,
    _build_curated_lists,
    _cap_models,
    _collect_authed_provider_slugs,
    _credential_identity,
    _credential_pool_is_usable,
    _discover_endpoint_models,
    _discover_flag,
    _display_prefix,
    _entry_api_mode,
    _entry_base_url,
    _fetch_picker_live_models,
    _has_aws_sdk_creds_for_listing,
    _has_fast_aws_sdk_signal,
    _is_aws_sdk,
    _iter_builtin_candidates,
    _lap_bare_custom_row,
    _lap_builtin_rows,
    _lap_canonical_rows,
    _lap_custom_provider_rows,
    _lap_overlay_rows,
    _lap_user_provider_rows,
    _live_or_curated_ids,
    _norm_url,
    _nous_picker_model_ids,
    _overlay_has_env_creds,
    _picker_prewarm_done,
    _pool_usable,
    _prefetch_provider_models_parallel,
    _prepend_moa_picker_provider,
    _raw_pool_usable,
    _save_discovered_models_to_config,
    list_authenticated_providers,
    list_picker_providers,
    prewarm_picker_cache_async,
)


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


def _runtime_creds(fallback_headers: dict, **kwargs) -> tuple[str, str, str, dict]:
    """``resolve_runtime_provider`` unpacked as ``(api_key, base_url, api_mode,
    extra_headers)``; ``extra_headers`` falls back to *fallback_headers*."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(**kwargs)
    return (
        runtime.get("api_key", ""),
        runtime.get("base_url", ""),
        runtime.get("api_mode", ""),
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


def _moa_default_preset() -> str:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        return normalize_moa_config(load_config().get("moa") or {})["default_preset"]
    except Exception:
        return "default"


@dataclass
class _Switch:
    """Mutable state threaded through the ``switch_model`` steps.

    The routing steps settle ``target_provider`` / ``new_model`` /
    ``resolved_alias`` (and may promote a config-routed ``providers.<slug>`` to
    ``explicit_provider`` so the credential step resolves its block); the
    credential step fills ``api_key`` / ``base_url`` / ``api_mode`` /
    ``validation_headers``.
    """

    raw_input: str
    current_provider: str
    current_model: str
    current_base_url: str
    current_api_key: str
    is_global: bool
    explicit_provider: str
    user_providers: Optional[dict]
    custom_providers: Optional[list]
    new_model: str = ""
    target_provider: str = ""
    resolved_alias: str = ""
    provider_label: str = ""
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    validation_headers: dict = field(default_factory=dict)
    suppress_ollama_headers: bool = False
    validation: dict = field(default_factory=dict)

    def fail(self, message: str, **fields) -> ModelSwitchResult:
        return _switch_fail(self.is_global, message, **fields)

    @property
    def provider_changed(self) -> bool:
        return self.target_provider != self.current_provider


def _route_explicit_provider(st: _Switch) -> Optional[ModelSwitchResult]:
    """PATH A (``--provider`` given): resolve the provider, auto-detect a model
    from a local endpoint when none was typed, then resolve the alias on the
    TARGET provider."""
    pdef = resolve_provider_full(st.explicit_provider, st.user_providers, st.custom_providers)
    if pdef is None and st.explicit_provider.strip().lower() == "custom":
        pdef = _bare_custom_provider_def(st.current_base_url)
    if pdef is None:
        return st.fail(_unknown_provider_message(st.explicit_provider))

    st.target_provider = pdef.id
    if st.target_provider == "moa" and not st.new_model:
        st.new_model = _moa_default_preset()

    agg_err = _aggregator_alias_error(
        st.explicit_provider, st.target_provider, st.current_provider, st.user_providers, st.custom_providers,
    )
    if agg_err:
        return st.fail(agg_err, target_provider=st.target_provider, provider_label=pdef.name)

    if not st.new_model:
        if not pdef.base_url:
            return st.fail(
                f"Provider '{pdef.name}' has no base URL configured. "
                f"Specify a model: /model <model-name> --provider {st.explicit_provider}",
                target_provider=st.target_provider, provider_label=pdef.name,
            )
        from hermes_cli.runtime_provider import _auto_detect_local_model
        st.new_model = _auto_detect_local_model(pdef.base_url)
        if not st.new_model:
            return st.fail(
                f"No model detected on {pdef.name} ({pdef.base_url}). "
                f"Specify the model explicitly: /model <model-name> --provider {st.explicit_provider}",
                target_provider=st.target_provider, provider_label=pdef.name,
            )

    try:
        alias_result = resolve_alias(st.new_model, st.target_provider)
    except AmbiguousAliasError as err:
        return st.fail(_ambiguous_alias_message(err), target_provider=st.target_provider)
    if alias_result is not None:
        _, st.new_model, st.resolved_alias = alias_result
    return None


def _route_alias_fallback(st: _Switch, key: str) -> Optional[ModelSwitchResult]:
    """Step b: the alias exists but not on the current provider -> try the
    user's authenticated providers."""
    authed = get_authenticated_provider_slugs(
        current_provider=st.current_provider, user_providers=st.user_providers, custom_providers=st.custom_providers,
    )
    try:
        fallback_result = _resolve_alias_fallback(st.raw_input, authed)
    except AmbiguousAliasError as err:
        return st.fail(_ambiguous_alias_message(err))
    if fallback_result is None:
        identity = MODEL_ALIASES[key]
        return st.fail(
            f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
            f"but no matching model was found in any provider catalog. "
            f"Try specifying the full model name.",
        )
    st.target_provider, st.new_model, st.resolved_alias = fallback_result
    logger.debug(
        "Alias '%s' resolved via fallback to %s on %s", st.resolved_alias, st.new_model, st.target_provider,
    )
    return None


def _convert_vendor_colon_slug(st: _Switch) -> None:
    """Step c: on an aggregator, ``vendor:model`` -> ``vendor/model``. Only
    without a slash: with one, the colon is a variant tag (:free, :extended,
    :fast) that must be preserved."""
    raw_input = st.raw_input
    colon_pos = raw_input.find(":")
    cur_norm = str(st.current_provider).strip().lower()
    if (
        colon_pos > 0
        and "/" not in raw_input
        and is_aggregator(st.current_provider)
        and not cur_norm.startswith("custom")
        and cur_norm != "ollama"
    ):
        left = raw_input[:colon_pos].strip().lower()
        right = raw_input[colon_pos + 1:].strip()
        if left and right:
            st.new_model = f"{left}/{right}"
            logger.debug("Converted vendor:model '%s' to aggregator slug '%s'", raw_input, st.new_model)


def _route_configured_provider(st: _Switch) -> Optional[ModelSwitchResult] | bool:
    """Step d.5: a model declared in user/custom provider config routes there
    BEFORE detect_provider_for_model() guesses from static catalogs and before a
    soft-accepting current provider (openai-codex) can swallow it as an unknown
    hidden model. Returns a failure result, ``True`` when routed, else ``False``."""
    cfg_matches = _configured_provider_matches(st.new_model, st.user_providers, st.custom_providers)
    if not cfg_matches:
        return False
    if st.current_provider in cfg_matches:
        st.new_model = cfg_matches[st.current_provider]
        return True
    match_slugs = sorted(cfg_matches)
    if len(match_slugs) > 1:
        return st.fail(
            f"'{st.new_model}' is declared by multiple configured "
            f"providers ({', '.join(match_slugs)}). Re-run with "
            f"--provider <slug> to choose which one to use.",
        )
    st.target_provider = match_slugs[0]
    st.new_model = cfg_matches[st.target_provider]
    logger.debug("Configured-provider detection routed '%s' to %s", st.new_model, st.target_provider)
    # providers.<slug> endpoints resolve in the credential block via
    # resolve_user_provider(), which is gated on explicit_provider; custom:*
    # slugs resolve at runtime directly.
    if isinstance(st.user_providers, dict) and st.target_provider in st.user_providers:
        st.explicit_provider = st.target_provider
    return True


def _route_from_model_input(st: _Switch) -> Optional[ModelSwitchResult]:
    """PATH B (no ``--provider``): MoA preset / alias on the current provider
    (a) -> alias fallback (b) or ``vendor:model`` conversion (c) -> aggregator
    catalog search (d) -> configured-provider match (d.5) ->
    detect_provider_for_model() as last resort (e)."""
    from hermes_cli.models import detect_provider_for_model

    raw_input, current_provider = st.raw_input, st.current_provider
    resolved_moa_preset = False
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import exact_moa_preset_name, normalize_moa_config

        moa_match = exact_moa_preset_name(normalize_moa_config(load_config().get("moa") or {}), raw_input)
        if moa_match:
            st.target_provider, st.new_model, st.resolved_alias = "moa", moa_match, ""
            resolved_moa_preset = True
            alias_result = None
        else:
            alias_result = resolve_alias(raw_input, current_provider)
    except AmbiguousAliasError as err:
        return st.fail(_ambiguous_alias_message(err))
    except Exception:
        try:
            alias_result = resolve_alias(raw_input, current_provider)
        except AmbiguousAliasError as err:
            return st.fail(_ambiguous_alias_message(err))

    if resolved_moa_preset:
        pass
    elif alias_result is not None:
        st.target_provider, st.new_model, st.resolved_alias = alias_result
        logger.debug("Alias '%s' resolved to %s on %s", st.resolved_alias, st.new_model, st.target_provider)
    elif raw_input.strip().lower() in MODEL_ALIASES:
        fail = _route_alias_fallback(st, raw_input.strip().lower())
        if fail is not None:
            return fail
    else:
        _convert_vendor_colon_slug(st)

    # Step d: if the CURRENT provider's live catalog resolved the model, step e
    # must not second-guess and switch providers — flat-namespace resellers
    # (opencode-go/zen) return bare ids that coincidentally match native
    # providers' static catalogs.
    resolved_in_current_catalog = False
    if is_aggregator(st.target_provider) and not st.resolved_alias:
        catalog = list_provider_models(st.target_provider)
        if catalog:
            matched = _aggregator_catalog_match(st.new_model, catalog)
            if matched is not None:
                st.new_model, resolved_in_current_catalog = matched, True

    # Step d.5 — deliberately NOT gated on ``not is_custom``.
    config_routed = False
    if not st.resolved_alias and not resolved_in_current_catalog and st.target_provider == current_provider:
        config_routed = _route_configured_provider(st)
        if isinstance(config_routed, ModelSwitchResult):
            return config_routed

    # Step e
    is_custom = (
        current_provider in {"custom", "local"}
        or current_provider.startswith("custom:")
        or base_url_hostname(st.current_base_url or "") in ("localhost", "127.0.0.1")
    )
    if (
        st.target_provider == current_provider
        and not is_custom
        and not st.resolved_alias
        and not resolved_in_current_catalog
        and not config_routed
    ):
        detected = detect_provider_for_model(st.new_model, current_provider)
        if detected:
            st.target_provider, st.new_model = detected
    return None


def _switch_provider_label(st: _Switch) -> str:
    label = get_label(st.target_provider)
    if st.target_provider == "custom" and st.current_base_url:
        label = "Custom endpoint"
    if st.target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(st.target_provider, st.user_providers, st.custom_providers)
        if custom_pdef is not None:
            label = custom_pdef.name
    return label


def _creds_for_switched_provider(st: _Switch) -> Optional[ModelSwitchResult]:
    """Credentials when the provider changed or ``--provider`` was given.

    ``providers.<name>`` blocks carry their own base_url + transport + key
    reference; resolve_runtime_provider() resolves by provider NAME and would
    re-resolve a block named "openai" from scratch (or hop to an aggregator),
    so use the pdef's endpoint directly.
    """
    user_pdef = None
    if st.explicit_provider and st.user_providers:
        from hermes_cli.providers import resolve_user_provider
        user_pdef = resolve_user_provider(st.explicit_provider.strip().lower(), st.user_providers)
        if user_pdef is None:
            user_pdef = resolve_user_provider(st.target_provider, st.user_providers)
    if user_pdef is not None and user_pdef.base_url:
        ucfg = (st.user_providers or {}).get(st.explicit_provider.strip().lower()) \
            or (st.user_providers or {}).get(st.target_provider) or {}
        # Key reads go through the per-profile secret scope: a raw os.environ
        # read would hand this profile another profile's key under the
        # multiplexed gateway.
        ukey = _entry_configured_key(ucfg, _scoped_key_env)
        st.validation_headers = _extra_headers_from_config(ucfg)
        try:
            api_key, base_url, st.api_mode, st.validation_headers = _runtime_creds(
                st.validation_headers,
                requested=st.target_provider,
                explicit_api_key=ukey or None,
                explicit_base_url=user_pdef.base_url,
                target_model=st.new_model,
            )
            st.api_key = api_key or ukey
            st.base_url = base_url or user_pdef.base_url
        except Exception:
            st.api_key, st.base_url, st.api_mode = ukey, user_pdef.base_url, ""
    elif st.target_provider == "custom" and st.current_base_url:
        st.api_key, st.base_url = st.current_api_key, st.current_base_url
        st.api_mode = determine_api_mode(st.target_provider, st.base_url)
    else:
        try:
            st.api_key, st.base_url, st.api_mode, st.validation_headers = _runtime_creds(
                st.validation_headers, requested=st.target_provider, target_model=st.new_model,
            )
        except Exception as e:
            return st.fail(
                f"Could not resolve credentials for provider '{st.provider_label}': {e}",
                target_provider=st.target_provider, provider_label=st.provider_label,
            )
    return None


def _creds_for_current_provider(st: _Switch) -> None:
    """Credentials when staying on the current provider. Mid-session
    ``/model <name>`` on a local Ollama-compatible endpoint keeps the endpoint
    in use; re-resolving bare ``custom`` from config can fall through to an
    unrelated default provider."""
    from hermes_cli.models import _get_ollama_request_headers, _same_ollama_native_root

    keep_current_ollama_endpoint = False
    ollama_headers: dict[str, str] = {}
    if st.current_provider == "custom" and st.current_base_url:
        try:
            from hermes_cli.models import should_use_ollama_native_catalog
            ollama_headers = _get_ollama_request_headers()
            _, configured_ollama_base = _ollama_configured_base()
            # Provider-level Ollama headers only belong to the configured
            # native root; without one there is no safe origin for them.
            if not configured_ollama_base or not _same_ollama_native_root(st.current_base_url, configured_ollama_base):
                ollama_headers = {}
                st.suppress_ollama_headers = True
            keep_current_ollama_endpoint = should_use_ollama_native_catalog(
                st.current_provider, st.current_base_url, headers=ollama_headers,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            keep_current_ollama_endpoint = False
    if keep_current_ollama_endpoint:
        st.api_key = st.current_api_key or "no-key-required"
        st.base_url = st.current_base_url
        st.api_mode = determine_api_mode(st.current_provider, st.base_url)
        st.validation_headers = ollama_headers
    else:
        try:
            st.api_key, st.base_url, st.api_mode, st.validation_headers = _runtime_creds(
                st.validation_headers, requested=st.current_provider, target_model=st.new_model,
            )
        except Exception:
            pass


def _resolve_switch_credentials(st: _Switch) -> Optional[ModelSwitchResult]:
    """COMMON PATH part 1: credentials, direct-alias endpoint override, and the
    api_mode for the final (provider, base_url) before validation."""
    st.provider_label = _switch_provider_label(st)
    st.api_key, st.base_url = st.current_api_key, st.current_base_url
    if st.provider_changed or st.explicit_provider:
        fail = _creds_for_switched_provider(st)
        if fail is not None:
            return fail
    else:
        _creds_for_current_provider(st)

    # Direct alias override: use the alias's exact base_url if set.
    if st.resolved_alias:
        _ensure_direct_aliases()
        da = DIRECT_ALIASES.get(st.resolved_alias)
        if da is not None and da.base_url:
            st.api_key, st.base_url, headers_override, suppress = _apply_direct_alias_endpoint(
                da, st.target_provider, st.new_model, st.api_key, st.base_url,
            )
            st.api_mode = ""  # clear so determine_api_mode re-detects from URL
            if headers_override is not None:
                st.validation_headers = headers_override
            if suppress:
                st.suppress_ollama_headers = True

    # Fills an empty mode (alias cleared it) and overrides a STALE mode carried
    # from previous session state when the host mandates one wire protocol
    # (e.g. gpt-5.x on api.openai.com would otherwise 400 on tools+reasoning).
    mandated_mode = host_mandated_api_mode(st.base_url)
    if mandated_mode is not None:
        st.api_mode = mandated_mode
    elif not st.api_mode:
        st.api_mode = determine_api_mode(st.target_provider, st.base_url)
    return None


def _validate_switch(st: _Switch) -> Optional[ModelSwitchResult]:
    """COMMON PATH part 2: normalize the model name for the target provider,
    validate it, and accept config-declared models the remote catalog lacks."""
    from hermes_cli.models import _get_ollama_request_headers, validate_requested_model

    st.new_model = _resolve_named_custom_model_id(st.new_model, st.target_provider, st.custom_providers)
    st.new_model = normalize_model_for_provider(st.new_model, st.target_provider)

    if st.target_provider.strip().lower() == "ollama":
        headers = {} if st.suppress_ollama_headers else (st.validation_headers or _get_ollama_request_headers())
    else:
        headers = st.validation_headers or (
            _extra_headers_from_config(st.user_providers.get(st.target_provider))
            if st.user_providers and st.target_provider in st.user_providers
            else None
        )
    try:
        validation = validate_requested_model(
            st.new_model, st.target_provider, api_key=st.api_key, base_url=st.base_url,
            api_mode=st.api_mode or None, headers=headers,
        )
    except Exception as e:
        validation = {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": f"Could not validate `{st.new_model}`: {e}",
        }

    if not validation.get("accepted"):
        if _config_declares_model(st.new_model, st.target_provider, st.base_url, st.user_providers, st.custom_providers):
            validation = {"accepted": True, "persist": True, "recognized": False, "message": validation.get("message", "")}
        else:
            return st.fail(
                validation.get("message", "Invalid model"),
                new_model=st.new_model, target_provider=st.target_provider, provider_label=st.provider_label,
            )
    if validation.get("corrected_model"):
        st.new_model = validation["corrected_model"]
    st.validation = validation
    return None


def _copilot_api_mode(provider: str, model: str, api_key: str) -> str:
    from hermes_cli.models import copilot_model_api_mode

    return copilot_model_api_mode(model, api_key=api_key)


def _opencode_api_mode(provider: str, model: str, api_key: str) -> str:
    from hermes_cli.models import opencode_model_api_mode

    return opencode_model_api_mode(provider, model)


def _nous_api_mode(provider: str, model: str, api_key: str) -> str:
    # Portal serves anthropic/* on /v1/messages and everything else on
    # /chat/completions; re-derive from the FINAL model so alias clears /
    # empty fallbacks cannot leave Claude on the OpenAI wire.
    from hermes_cli.providers import nous_api_mode

    return nous_api_mode(model)


# Per-provider api_mode overrides applied after validation, keyed on the final
# target provider (the key sets are disjoint, so exactly one — or none — fires).
_PROVIDER_API_MODE_OVERRIDES: dict[str, Any] = {
    **dict.fromkeys(("copilot", "github-copilot"), _copilot_api_mode),
    **dict.fromkeys(("opencode-zen", "opencode-go", "opencode"), _opencode_api_mode),
    **dict.fromkeys(("nous", "nous-portal", "nousresearch"), _nous_api_mode),
}


def _build_switch_result(st: _Switch) -> ModelSwitchResult:
    """COMMON PATH part 3: final api_mode / base_url shaping, metadata, warnings."""
    override = _PROVIDER_API_MODE_OVERRIDES.get(st.target_provider)
    if override is not None:
        st.api_mode = override(st.target_provider, st.new_model, st.api_key)
    if not st.api_mode:
        st.api_mode = determine_api_mode(st.target_provider, st.base_url, model=st.new_model)

    # OpenCode base URLs end with /v1 for OpenAI-compatible models but the
    # Anthropic SDK prepends its own /v1/messages: strip for anthropic_messages,
    # re-append for chat_completions/codex_responses (mirrors
    # resolve_runtime_provider; either direction alone breaks the other family).
    from hermes_cli.models import normalize_opencode_base_url, opencode_provider_family
    if opencode_provider_family(st.target_provider) is not None and isinstance(st.base_url, str):
        st.base_url = normalize_opencode_base_url(st.target_provider, st.api_mode, st.base_url)

    capabilities = get_model_capabilities(st.target_provider, st.new_model, allow_network=True)
    from agent.native_compaction import resolve_native_compaction_capabilities
    runtime_capabilities = resolve_native_compaction_capabilities(
        model=st.new_model,
        base_url=st.base_url,
        provider=st.target_provider,
        is_codex_backend=st.target_provider.strip().lower() == "openai-codex",
    )
    model_info = get_model_info(st.target_provider, st.new_model, allow_network=True)

    warnings: list[str] = []
    if st.validation.get("message"):
        warnings.append(st.validation["message"])
    hermes_warn = _check_hermes_model_warning(st.new_model)
    if hermes_warn:
        warnings.append(hermes_warn)

    # Carry the switched provider's request_overrides (custom_providers
    # ``extra_body`` such as chat_template_kwargs) so the gateway applies them
    # like the default-provider path does.
    request_overrides = None
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider, _custom_provider_request_overrides
        cp_for_ro = _get_named_custom_provider(st.target_provider)
        if cp_for_ro:
            request_overrides = _custom_provider_request_overrides(cp_for_ro) or None
    except Exception:
        request_overrides = None

    return ModelSwitchResult(
        success=True,
        new_model=st.new_model,
        target_provider=st.target_provider,
        provider_changed=st.provider_changed,
        api_key=st.api_key,
        base_url=st.base_url,
        api_mode=st.api_mode,
        request_overrides=dict(request_overrides or {}),
        warning_message=" | ".join(warnings) if warnings else "",
        provider_label=st.provider_label,
        resolved_via_alias=st.resolved_alias,
        capabilities=capabilities,
        runtime_capabilities={
            key: value
            for key, value in runtime_capabilities.items()
            if isinstance(key, str) and isinstance(value, bool)
        },
        model_info=model_info,
        is_global=st.is_global,
    )


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

    Resolution chain: route the request (:func:`_route_explicit_provider` when
    ``--provider`` was given, else :func:`_route_from_model_input`) ->
    :func:`_resolve_switch_credentials` -> :func:`_validate_switch` ->
    :func:`_build_switch_result`. Each step returns a failure
    :class:`ModelSwitchResult` to stop the chain, or ``None`` to continue.

    ``explicit_provider`` comes from the --provider flag (empty = none);
    ``user_providers`` / ``custom_providers`` are the ``providers:`` dict and
    ``custom_providers:`` list from config.yaml.
    """
    st = _Switch(
        raw_input=raw_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=is_global,
        explicit_provider=explicit_provider,
        user_providers=user_providers,
        custom_providers=custom_providers,
        new_model=raw_input.strip(),
        target_provider=current_provider,
    )
    route = _route_explicit_provider if explicit_provider else _route_from_model_input
    for step in (route, _resolve_switch_credentials, _validate_switch):
        fail = step(st)
        if fail is not None:
            return fail
    return _build_switch_result(st)


def _extra_headers_from_config(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    from hermes_cli.config import normalize_extra_headers

    return normalize_extra_headers(entry.get("extra_headers"))


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


