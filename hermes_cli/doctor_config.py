"""Configuration-file checks for hermes doctor: .env, config.yaml validation, drift, deprecations.

Split out of ``hermes_cli/doctor.py``; every moved name is re-imported there, so
``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import os
import shutil
from hermes_cli.doctor_report import (
    Finding,
    _fail_and_issue,
    _section,
    check_fail,
    check_info,
    check_ok,
    check_warn,
)


def _has_provider_env_config(content: str) -> bool:
    """Return True when ~/.hermes/.env contains provider auth/base URL settings."""
    from hermes_cli.doctor import _PROVIDER_ENV_HINTS
    return any(key in content for key in _PROVIDER_ENV_HINTS)


# Deprecated / legacy config keys still read for back-compat. Doctor surfaces
# them as non-failing warnings with the modern replacement — it does not
# auto-migrate or delete (migrations live in config.py version steps).
_DEPRECATED_CONFIG_KEYS: tuple[tuple[str, str, str], ...] = (
    # (section, key, replacement)
    ("display", "tool_progress_overrides", "display.platforms"),
    ("delegation", "max_async_children", "delegation.max_concurrent_children"),
)


# compression.summary_* → auxiliary.compression (model/provider/base_url)
_DEPRECATED_COMPRESSION_SUMMARY_KEYS: tuple[str, ...] = (
    "summary_model",
    "summary_provider",
    "summary_base_url",
)


# Deprecated env vars (checked in the .env file, not process env, so config→env
# bridges like terminal.cwd → TERMINAL_CWD do not false-positive).
_DEPRECATED_ENV_VARS: tuple[tuple[str, str], ...] = (
    # HERMES_TOOL_PROGRESS is fully unsupported since the v12 config support
    # floor removed its only consumer (the v3→4 migration) — it is silently
    # ignored. HERMES_TOOL_PROGRESS_MODE is still read by the gateway as a
    # back-compat fallback but remains deprecated.
    ("HERMES_TOOL_PROGRESS", "display.tool_progress in config.yaml — ignored/unsupported since config floor v12"),
    ("HERMES_TOOL_PROGRESS_MODE", "display.tool_progress in config.yaml"),
    ("TERMINAL_CWD", "terminal.cwd in config.yaml"),
    ("MESSAGING_CWD", "terminal.cwd in config.yaml"),
    ("QQ_HOME_CHANNEL", "QQBOT_HOME_CHANNEL"),
    ("QQ_HOME_CHANNEL_NAME", "QQBOT_HOME_CHANNEL_NAME"),
)


def collect_deprecated_config_keys(raw_config: dict | None) -> list[tuple[str, str]]:
    """Return ``(legacy_path, replacement)`` for deprecated keys present in *raw_config*.

    Only keys that appear in the on-disk YAML are reported (raw file load, not
    merged defaults). Empty containers still count — presence of the legacy
    key is the signal that the user should migrate.
    """
    findings: list[tuple[str, str]] = []
    if not isinstance(raw_config, dict):
        return findings

    for section, key, replacement in _DEPRECATED_CONFIG_KEYS:
        section_val = raw_config.get(section)
        if isinstance(section_val, dict) and key in section_val:
            findings.append((f"{section}.{key}", replacement))

    compression = raw_config.get("compression")
    if isinstance(compression, dict):
        for key in _DEPRECATED_COMPRESSION_SUMMARY_KEYS:
            if key in compression:
                findings.append((f"compression.{key}", "auxiliary.compression"))

    return findings


def collect_deprecated_env_vars(env_map: dict | None) -> list[tuple[str, str]]:
    """Return ``(legacy_env, replacement)`` for deprecated vars present in *env_map*.

    *env_map* should come from the on-disk ``.env`` (e.g. ``load_env()``), not
    ``os.environ``, so bridged runtime vars do not trigger false positives.
    """
    findings: list[tuple[str, str]] = []
    if not isinstance(env_map, dict):
        return findings
    for name, replacement in _DEPRECATED_ENV_VARS:
        val = env_map.get(name)
        if val is not None and str(val).strip() != "":
            findings.append((name, replacement))
    return findings


def collect_relay_plugin_cutover_findings(
    raw_config: dict | None,
    env_map: dict | None,
) -> list[tuple[str, str]]:
    """Return actionable findings for the removed Hermes Relay plugin."""
    from hermes_cli.relay_plugin_cutover import (
        LEGACY_RELAY_EXPORT_ENV_VARS,
        RELAY_PLUGINS_CONFIG_ENV,
        configured_legacy_relay_env_vars,
        legacy_relay_plugin_keys,
    )

    findings: list[tuple[str, str]] = []
    if isinstance(raw_config, dict):
        plugins = raw_config.get("plugins")
        if isinstance(plugins, dict):
            for key in legacy_relay_plugin_keys(plugins.get("enabled")):
                findings.append(
                    (
                        f"plugins.enabled: {key}",
                        f"remove it and configure {RELAY_PLUGINS_CONFIG_ENV}",
                    )
                )

    effective_env = dict(env_map or {})
    # Fall through to process-level env ONLY when no explicit env_map was
    # given: run_doctor passes None and wants live-process vars included, but
    # callers (and tests) that hand in an explicit map are describing a
    # complete environment — merging os.environ on top breaks hermeticity on
    # any box that exports legacy relay vars (10-vs-2 findings, Aug 2026).
    if env_map is None:
        for name in (*LEGACY_RELAY_EXPORT_ENV_VARS, RELAY_PLUGINS_CONFIG_ENV):
            if name not in effective_env and os.environ.get(name) is not None:
                effective_env[name] = os.environ[name]
    if not str(effective_env.get(RELAY_PLUGINS_CONFIG_ENV, "")).strip():
        for name in configured_legacy_relay_env_vars(effective_env):
            findings.append(
                (
                    name,
                    f"move exporter settings to {RELAY_PLUGINS_CONFIG_ENV}; "
                    "this variable is now ignored",
                )
            )
    return findings


def report_deprecated_config_and_env(
    raw_config: dict | None = None,
    env_map: dict | None = None,
) -> list[tuple[str, str]]:
    """Emit non-failing doctor warnings for deprecated config keys and env vars.

    Returns the list of ``(legacy, replacement)`` findings that were reported
    (empty when nothing deprecated is present). Does not mutate config/env and
    does not append to the blocking ``issues`` list.
    """
    deprecated = collect_deprecated_config_keys(raw_config)
    deprecated.extend(collect_deprecated_env_vars(env_map))
    relay_cutover = collect_relay_plugin_cutover_findings(raw_config, env_map)
    findings = deprecated + relay_cutover
    if not findings:
        check_ok("No deprecated config keys or env vars")
        return findings

    for legacy, replacement in deprecated:
        check_warn(
            f"Deprecated: {legacy}",
            f"(use {replacement} instead)",
        )
        check_info(f"Replace {legacy} → {replacement} (warn-only; not auto-migrated here)")
    for legacy, replacement in relay_cutover:
        check_warn(
            f"Breaking Relay migration: {legacy}",
            f"({replacement})",
        )
        check_info(f"Migrate {legacy}: {replacement}")
    return findings


def managed_scope_check() -> None:
    """Report the active managed scope (resolved dir + pinned key counts).

    Silent when no managed scope is present. When the managed directory was
    resolved from the HERMES_MANAGED_DIR override (rather than the system
    default), that is surfaced too — a redirected scope is the documented
    foot-gun (see docs/design/managed-scope.md §7) and an operator should see it.
    """
    try:
        from hermes_cli import managed_scope
        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — diagnostics must never crash
        return
    if managed_dir is None:
        return
    n_cfg = len(managed_scope.managed_config_keys())
    n_env = len(managed_scope.load_managed_env())
    check_ok(
        f"Managed scope active: {n_cfg} config key(s), {n_env} env key(s) "
        f"pinned by {managed_dir}"
    )
    if os.environ.get("HERMES_MANAGED_DIR", "").strip():
        check_info(f"managed dir set via HERMES_MANAGED_DIR={managed_dir}")


def _check_mcp_security(should_fix: bool) -> Finding:
    """Flag mcp_servers entries with suspicious stdio commands."""
    f = Finding()
    manual_issues = f.manual_issues
    try:
        from hermes_cli.config import load_config
        from hermes_cli.mcp_security import validate_mcp_server_entry

        servers = load_config().get("mcp_servers") or {}
        suspicious = 0
        if isinstance(servers, dict):
            for name, entry in sorted(servers.items()):
                if not isinstance(entry, dict):
                    continue
                issues_found = validate_mcp_server_entry(name, entry)
                if not issues_found:
                    continue
                suspicious += 1
                check_warn(f"MCP server '{name}' has suspicious stdio command", "; ".join(issues_found))
                manual_issues.append(
                    f"Review/remove mcp_servers.{name} in config.yaml; rotate any credentials that may have been exposed."
                )
        if suspicious == 0:
            check_ok("No suspicious MCP stdio commands")
    except Exception as e:
        check_warn(f"MCP security check failed: {e}")
    return f


def _check_env_file(should_fix: bool) -> Finding:
    """Managed scope plus ~/.hermes/.env presence and provider credentials."""
    from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT, _DHH
    f = Finding()
    issues = f.issues
    managed_scope_check()
    # Check ~/.hermes/.env (primary location for user config)
    env_path = HERMES_HOME / '.env'
    if env_path.exists():
        check_ok(f"{_DHH}/.env file exists")
        
        # Prefer UTF-8 (.env is written as UTF-8 elsewhere). Fall back to
        # latin-1 for Windows Notepad/cp1252 files that are not valid UTF-8 —
        # matches hermes_cli.env_loader._load_dotenv_with_fallback.
        try:
            content = env_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = env_path.read_text(encoding="latin-1")
        if _has_provider_env_config(content):
            check_ok("API key or custom endpoint configured")
        else:
            check_warn(f"No API key found in {_DHH}/.env")
            issues.append("Run 'hermes setup' to configure API keys")
    else:
        # Also check project root as fallback
        fallback_env = PROJECT_ROOT / '.env'
        if fallback_env.exists():
            check_ok(".env file exists (in project directory)")
        else:
            check_fail(f"{_DHH}/.env file missing")
            if should_fix:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                env_path.touch()
                # .env holds API keys — restrict to owner-only access from
                # creation. touch() obeys umask which is commonly 0o022,
                # leaving the file world-readable; tighten explicitly.
                try:
                    os.chmod(str(env_path), 0o600)
                except OSError:
                    pass
                check_ok(f"Created empty {_DHH}/.env")
                check_info("Run 'hermes setup' to configure API keys")
                f.fixed += 1
            else:
                check_info("Run 'hermes setup' to create one")
                issues.append("Run 'hermes setup' to create .env")
    return f


def _check_config_file(should_fix: bool) -> Finding:
    """config.yaml presence; validate model.provider / model.default and credentials."""
    from hermes_cli.doctor import HERMES_HOME, PROJECT_ROOT, _DHH
    f = Finding()
    issues = f.issues
    # Check ~/.hermes/config.yaml (primary) or project cli-config.yaml (fallback)
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        check_ok(f"{_DHH}/config.yaml exists")

        # Validate model.provider and model.default values
        try:
            # Raw-file diagnostic: inspects what the user actually wrote.
            from hermes_cli.config import read_user_config_raw
            cfg = read_user_config_raw(config_path)
            model_section = cfg.get("model") or {}
            provider_raw = (model_section.get("provider") or "").strip()
            provider = provider_raw.lower()
            default_model = (model_section.get("default") or model_section.get("model") or "").strip()

            known_providers: set = set()
            try:
                from hermes_cli.auth import (
                    PROVIDER_REGISTRY,
                    resolve_provider as _resolve_auth_provider,
                )
                known_providers = set(PROVIDER_REGISTRY.keys()) | {"openrouter", "custom", "auto", "moa"}
            except Exception:
                _resolve_auth_provider = None
                pass
            try:
                from hermes_cli.config import get_compatible_custom_providers as _compatible_custom_providers
                from hermes_cli.providers import (
                    custom_provider_aliases as _custom_provider_aliases,
                    normalize_provider as _normalize_catalog_provider,
                    resolve_provider_full as _resolve_provider_full,
                )
            except Exception:
                _compatible_custom_providers = None
                _custom_provider_aliases = None
                _normalize_catalog_provider = None
                _resolve_provider_full = None

            custom_providers = []
            if _compatible_custom_providers is not None:
                try:
                    custom_providers = _compatible_custom_providers(cfg)
                except Exception:
                    custom_providers = []

            user_providers = cfg.get("providers")
            if isinstance(user_providers, dict):
                from hermes_cli.config import is_provider_enabled
                known_providers.update(
                    str(name).strip().lower()
                    for name, prov_cfg in user_providers.items()
                    if str(name).strip() and is_provider_enabled(prov_cfg)
                )
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                provider_key = str(entry.get("provider_key") or "").strip()
                if name and _custom_provider_aliases is not None:
                    known_providers.update(
                        _custom_provider_aliases(name, provider_key)
                    )

            valid_provider_ids = set(known_providers)
            provider_ids_to_accept = {provider} if provider else set()
            if _normalize_catalog_provider is not None:
                for known_provider in known_providers:
                    try:
                        valid_provider_ids.add(_normalize_catalog_provider(known_provider))
                    except Exception:
                        continue

            runtime_provider = provider
            if (
                provider
                and _resolve_auth_provider is not None
                and provider not in {"auto", "custom"}
            ):
                try:
                    runtime_provider = _resolve_auth_provider(provider)
                    provider_ids_to_accept.add(runtime_provider)
                except Exception:
                    runtime_provider = provider

            catalog_provider = provider
            if (
                provider
                and _resolve_provider_full is not None
                and provider not in {"auto", "custom"}
            ):
                provider_def = _resolve_provider_full(provider, user_providers, custom_providers)
                catalog_provider = provider_def.id if provider_def is not None else None
                if catalog_provider is not None:
                    provider_ids_to_accept.add(catalog_provider)

            if provider and provider != "auto":
                if catalog_provider is None or (
                    known_providers
                    and not (provider_ids_to_accept & valid_provider_ids)
                ):
                    known_list = ", ".join(sorted(known_providers)) if known_providers else "(unavailable)"
                    _fail_and_issue(
                        f"model.provider '{provider_raw}' is not a recognised provider",
                        f"(known: {known_list})",
                        (
                            f"model.provider '{provider_raw}' is unknown. "
                            f"Valid providers: {known_list}. "
                            f"Fix: run 'hermes config set model.provider <valid_provider>'"
                        ),
                        issues,
                    )

            # Warn if model is set to a provider-prefixed name on a provider that doesn't use them.
            # Vendor/model slugs are valid on aggregator-style providers and on any custom
            # provider — bare "custom" or a named "custom:<name>" that fronts an OpenAI-compatible
            # aggregator (e.g. custom:hpc-ai serving deepseek/deepseek-v4-flash) requires the prefix.
            provider_for_policy = runtime_provider or catalog_provider
            provider_policy_id = str(provider_for_policy or "").strip().lower()
            providers_accepting_vendor_slugs = {
                "openrouter",
                "auto",
                "ai-gateway",
                "kilocode",
                "opencode-zen",
                "huggingface",
                "lmstudio",
                "nous",
                "nvidia",
                # Fireworks' native model IDs are slash-form
                # (accounts/fireworks/models/... and .../routers/...), so a "/"
                # is expected, not an aggregator vendor prefix.
                "fireworks",
                # DeepInfra is an aggregator-style gateway: its catalog
                # is exclusively ``vendor/model`` slugs (Qwen/Qwen3.5-…,
                # meta-llama/Llama-3-…, anthropic/claude-opus-4-7, …).
                "deepinfra",
            }
            provider_accepts_vendor_slug = (
                provider_policy_id in providers_accepting_vendor_slugs
                or provider_policy_id == "custom"
                or provider_policy_id.startswith("custom:")
            )
            if (
                default_model
                and "/" in default_model
                and provider_policy_id
                and not provider_accepts_vendor_slug
            ):
                check_warn(
                    f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'",
                    "(vendor-prefixed slugs belong to aggregators like openrouter)",
                )
                issues.append(
                    f"model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. "
                    "Either set model.provider to 'openrouter', or drop the vendor prefix."
                )

            # Check credentials for the configured provider.
            # Limit to API-key providers in PROVIDER_REGISTRY — other provider
            # types (OAuth, SDK, anthropic/custom/auto) have their own env-var
            # checks elsewhere in doctor, and get_auth_status() returns a bare
            # {logged_in: False} for anything it doesn't explicitly dispatch,
            # which would produce false positives.
            if runtime_provider and runtime_provider not in ("auto", "custom"):
                try:
                    if runtime_provider == "openrouter":
                        from hermes_cli.config import get_env_value

                        configured = bool(
                            str(get_env_value("OPENROUTER_API_KEY") or "").strip()
                            or str(get_env_value("OPENAI_API_KEY") or "").strip()
                        )
                    else:
                        from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status

                        pconfig = PROVIDER_REGISTRY.get(runtime_provider)
                        configured = True
                        if pconfig and getattr(pconfig, "auth_type", "") == "api_key":
                            status = get_auth_status(runtime_provider) or {}
                            configured = bool(
                                status.get("configured")
                                or status.get("logged_in")
                                or status.get("api_key")
                            )
                    if not configured:
                        _fail_and_issue(
                            f"model.provider '{runtime_provider}' is set but no API key is configured",
                            "(check ~/.hermes/.env or run 'hermes setup')",
                            (
                                f"No credentials found for provider '{runtime_provider}'. "
                                f"Run 'hermes setup' or set the provider's API key in {_DHH}/.env, "
                                f"or switch providers with 'hermes config set model.provider <name>'"
                            ),
                            issues,
                        )
                except Exception:
                    pass

        except Exception as e:
            check_warn("Could not validate model/provider config", f"({e})")
    else:
        fallback_config = PROJECT_ROOT / 'cli-config.yaml'
        if fallback_config.exists():
            check_ok("cli-config.yaml exists (in project directory)")
        else:
            if should_fix:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                example_config = PROJECT_ROOT / 'cli-config.yaml.example'
                if example_config.exists():
                    shutil.copy2(str(example_config), str(config_path))
                    check_ok(f"Created {_DHH}/config.yaml from cli-config.yaml.example")
                else:
                    from hermes_cli.config import DEFAULT_CONFIG, save_config
                    save_config(DEFAULT_CONFIG)
                    check_ok(f"Created {_DHH}/config.yaml from defaults")
                f.fixed += 1
            else:
                check_warn("config.yaml not found", "(using defaults)")
    return f


def _check_config_drift(should_fix: bool) -> Finding:
    """Config version, stale root keys, HERMES_MAX_ITERATIONS ghost, deprecations, structure."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    f = Finding()
    issues, manual_issues = f.issues, f.manual_issues
    # Check config version and stale keys
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        try:
            from hermes_cli.config import check_config_version, migrate_config
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                check_warn(
                    f"Config version outdated (v{current_ver} → v{latest_ver})",
                    "(new settings available)"
                )
                if should_fix:
                    try:
                        migrate_config(interactive=False, quiet=False)
                        check_ok("Config migrated to latest version")
                        f.fixed += 1
                    except Exception as mig_err:
                        check_warn(f"Auto-migration failed: {mig_err}")
                        issues.append("Run 'hermes setup' to migrate config")
                else:
                    issues.append("Run 'hermes doctor --fix' or 'hermes setup' to migrate config")
            else:
                check_ok(f"Config version up to date (v{current_ver})")
        except Exception:
            pass

        # Detect stale root-level model keys (known bug source — PR #4329)
        try:
            # Raw-file diagnostic: stale-key detection must see the raw file.
            from hermes_cli.config import read_user_config_raw
            raw_config = read_user_config_raw(config_path)
            stale_root_keys = [k for k in ("provider", "base_url") if k in raw_config and isinstance(raw_config[k], str)]
            if stale_root_keys:
                check_warn(
                    f"Stale root-level config keys: {', '.join(stale_root_keys)}",
                    "(should be under 'model:' section)"
                )
                if should_fix:
                    # Coerce scalar/None ``model:`` into a dict before mutation —
                    # ``setdefault("model", {})`` would return an existing scalar
                    # and then ``model_section[k] = ...`` would raise TypeError.
                    raw_model = raw_config.get("model")
                    if isinstance(raw_model, dict):
                        model_section = raw_model
                    elif isinstance(raw_model, str) and raw_model.strip():
                        model_section = {"default": raw_model.strip()}
                        raw_config["model"] = model_section
                    else:
                        model_section = {}
                        raw_config["model"] = model_section
                    for k in stale_root_keys:
                        if not model_section.get(k):
                            model_section[k] = raw_config.pop(k)
                        else:
                            raw_config.pop(k)
                    from hermes_cli.config import atomic_config_write
                    atomic_config_write(config_path, raw_config)
                    check_ok("Migrated stale root-level keys into model section")
                    f.fixed += 1
                else:
                    issues.append("Stale root-level provider/base_url in config.yaml — run 'hermes doctor --fix'")
        except Exception:
            pass

        # Detect stale HERMES_MAX_ITERATIONS ghost in .env shadowing
        # agent.max_turns in config.yaml (issue #17534). The setup wizard
        # used to dual-write the iteration budget to both stores; users who
        # later edit only config.yaml are left with a .env ghost. The gateway
        # bridge normally derives HERMES_MAX_ITERATIONS from agent.max_turns
        # at startup, but if that bridge bails (any earlier config-parse
        # error), the stale .env value silently wins and the agent runs at the
        # wrong budget — e.g. config says 400 but the activity line reads N/90.
        # Read the .env FILE directly (load_env), not get_env_value/os.environ,
        # which the startup bridge may already have overridden.
        try:
            from hermes_cli.config import load_env, read_user_config_raw, remove_env_value
            # Raw-file diagnostic: drift check against the raw file.
            raw_config = read_user_config_raw(config_path)
            agent_cfg = raw_config.get("agent")
            cfg_max_turns = (
                agent_cfg.get("max_turns")
                if isinstance(agent_cfg, dict)
                else None
            )
            # Legacy root-level key counts too.
            if cfg_max_turns is None:
                cfg_max_turns = raw_config.get("max_turns")
            env_ghost = load_env().get("HERMES_MAX_ITERATIONS")
            drift = (
                cfg_max_turns is not None
                and env_ghost is not None
                and str(cfg_max_turns).strip() != str(env_ghost).strip()
            )
            if drift:
                check_warn(
                    f"HERMES_MAX_ITERATIONS={env_ghost} in .env shadows "
                    f"agent.max_turns={cfg_max_turns} in config.yaml",
                    "(stale ghost from an earlier `hermes setup` run)",
                )
                if should_fix:
                    if remove_env_value("HERMES_MAX_ITERATIONS"):
                        check_ok(
                            "Removed stale HERMES_MAX_ITERATIONS from .env "
                            f"(config.yaml agent.max_turns={cfg_max_turns} is now authoritative)"
                        )
                        f.fixed += 1
                    else:
                        check_warn("Could not remove HERMES_MAX_ITERATIONS from .env")
                        manual_issues.append(
                            "Manually delete the HERMES_MAX_ITERATIONS line from "
                            f"{_DHH}/.env — config.yaml agent.max_turns is authoritative."
                        )
                else:
                    issues.append(
                        "Stale HERMES_MAX_ITERATIONS in .env shadows config.yaml — "
                        "run 'hermes doctor --fix'"
                    )
        except Exception:
            pass

        # Surface deprecated/legacy config keys and env vars (warn-only).
        # Migrations may still live in config.py version steps; doctor does
        # not auto-delete here — only tells the user the modern replacement.
        try:
            from hermes_cli.config import load_env as _load_env_depr
            from hermes_cli.config import read_user_config_raw as _read_raw_depr

            # Raw-file diagnostic: deprecation sweep inspects the raw file.
            _raw_for_depr = _read_raw_depr(config_path)
            # Prefer the on-disk .env so bridged process env (e.g. TERMINAL_CWD
            # from terminal.cwd) does not false-positive.
            try:
                _env_for_depr = _load_env_depr()
            except Exception:
                _env_for_depr = {}
            report_deprecated_config_and_env(_raw_for_depr, _env_for_depr)
        except Exception:
            pass

        # Validate config structure (catches malformed custom_providers, etc.)
        try:
            from hermes_cli.config import validate_config_structure
            config_issues = validate_config_structure()
            if config_issues:
                _section("Config Structure")
                for ci in config_issues:
                    if ci.severity == "error":
                        check_fail(ci.message)
                    else:
                        check_warn(ci.message)
                    # Show the hint indented
                    for hint_line in ci.hint.splitlines():
                        check_info(hint_line)
                    issues.append(ci.message)
        except Exception:
            pass

    if not config_path.exists():
        # No config.yaml — still surface deprecated env vars from .env.
        try:
            from hermes_cli.config import load_env as _load_env_depr

            try:
                _env_for_depr = _load_env_depr()
            except Exception:
                _env_for_depr = {}
            report_deprecated_config_and_env({}, _env_for_depr)
        except Exception:
            pass
    return f


def _check_xai_retirement(should_fix: bool) -> Finding:
    f = Finding()
    manual_issues = f.manual_issues
    try:
        from hermes_cli.config import load_config
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            find_retired_xai_refs,
            format_issue,
        )

        _xai_cfg = load_config()
        retired_refs = find_retired_xai_refs(_xai_cfg)
        if not retired_refs:
            check_ok("No retired xAI models in config")
        else:
            for ref in retired_refs:
                check_warn(format_issue(ref))
            check_info(f"Migration guide: {MIGRATION_GUIDE_URL}")
            manual_issues.append(
                f"Update {len(retired_refs)} retired xAI model reference(s) "
                f"in config.yaml — see {MIGRATION_GUIDE_URL}"
            )
    except Exception as _xai_check_err:
        check_warn("xAI retirement check skipped", f"({_xai_check_err})")
    return f
