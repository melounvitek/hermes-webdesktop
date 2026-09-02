"""Table-driven config migration registry.

Each step is a function ``_migrate_to_N(results, quiet)`` whose body is copied verbatim from the
original block; only the shared skeleton (the version gate and the strict ascending ordering) lives
in the :func:`run_migrations` driver.

Semantics preserved exactly from the original ladder:
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

#: Auto-migration support floor. Configs whose on-disk ``_config_version`` is
#: below this are NOT auto-migrated any more (policy decision, July 2026):
#: v12 predates roughly two years of releases, and carrying the sub-v12
#: migration steps (plus the env bridges they consumed, e.g.
#: HERMES_TOOL_PROGRESS*) forever is not worth it. Below-floor configs are
#: left byte-for-byte untouched — the process continues with the config as-is
#: (defaults deep-merged at read time, matching the non-fatal posture used
#: for unparseable configs) and a clear message tells the user how to
#: proceed. The removed steps were the <12 targets: v4 (tool-progress .env →
#: config.yaml), v5 (timezone seed), v9 (clear ANTHROPIC_TOKEN).
SUPPORT_FLOOR_VERSION = 12


def support_floor_message() -> str:
    """Human-facing explanation shown when a config is below the floor."""
    from hermes_constants import display_hermes_home

    return (
        f"This config predates version {SUPPORT_FLOOR_VERSION} (~2 years old) "
        "and can no longer be auto-migrated. Back up "
        f"{display_hermes_home()}/config.yaml and run `hermes setup` to "
        f"regenerate, or manually set _config_version: {SUPPORT_FLOOR_VERSION} "
        "after reviewing the changelog."
    )


def _cfg():
    """Return the live ``hermes_cli.config`` module (lazy, cycle-free)."""
    from hermes_cli import config

    return config


def read_raw_config():
    return _cfg().read_raw_config()


def _persist_migration(config):
    _cfg()._persist_migration(config)


def _rewrite_stale_default(
    results: Dict[str, Any],
    quiet: bool,
    *,
    section: str,
    key: str,
    old: Any,
    new: Any,
    added: str,
    message: str,
    extra_guard: Callable[[Dict[str, Any]], bool] = lambda _m: True,
) -> None:
    """Shared step shape: rewrite ``<section>.<key>`` only when it still equals the OLD default.

    Never clobbers a value the user deliberately customized; unset keys inherit the new default at
    read time. ``new=None`` deletes the key instead of assigning.
    """
    config = read_raw_config()
    raw = config.get(section)
    if isinstance(raw, dict) and raw.get(key) == old and extra_guard(raw):
        if new is None:
            del raw[key]
        else:
            raw[key] = new
        config[section] = raw
        _persist_migration(config)
        results["config_added"].append(added)
        if not quiet:
            print(message)


def _migrate_to_12(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 11 → 12: migrate custom_providers list → providers dict ──
    _custom_provider_entry_to_provider_config = _cfg()._custom_provider_entry_to_provider_config

    config = read_raw_config()
    custom_list = config.get("custom_providers")
    if isinstance(custom_list, list) and custom_list:
        providers_dict = config.get("providers", {})
        if not isinstance(providers_dict, dict):
            providers_dict = {}
        migrated_count = 0
        for entry in custom_list:
            if not isinstance(entry, dict):
                continue
            old_name = entry.get("name", "")
            old_url = entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or ""
            if not old_url:
                continue  # skip entries with no URL

            # Generate a kebab-case key from the display name
            key = old_name.strip().lower().replace(" ", "-").replace("(", "").replace(")", "")
            # Remove consecutive hyphens and trailing hyphens
            while "--" in key:
                key = key.replace("--", "-")
            key = key.strip("-")
            if not key:
                # Fallback: derive from URL hostname
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(old_url)
                    key = (parsed.hostname or "endpoint").replace(".", "-")
                except Exception:
                    key = f"endpoint-{migrated_count}"

            # Don't overwrite existing entries
            base_key = key
            suffix = migrated_count
            while key in providers_dict:
                key = f"{base_key}-{suffix}"
                suffix += 1

            new_entry = _custom_provider_entry_to_provider_config(
                entry,
                provider_key=key,
            )
            if new_entry is None:
                continue
            if not old_name:
                new_entry.pop("name", None)
            if new_entry.get("api_key") in {"no-key", "no-key-required", ""}:
                new_entry.pop("api_key", None)

            providers_dict[key] = new_entry
            migrated_count += 1

        if migrated_count > 0:
            config["providers"] = providers_dict
            # Remove the old list — runtime reads via get_compatible_custom_providers()
            config.pop("custom_providers", None)
            _persist_migration(config)
            if not quiet:
                print(f"  ✓ Migrated {migrated_count} custom provider(s) to providers: section")
                for key in list(providers_dict.keys())[-migrated_count:]:
                    ep = providers_dict[key]
                    print(f"    → {key}: {ep.get('api', '')}")


def _migrate_to_13(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 12 → 13: clear dead LLM_MODEL / OPENAI_MODEL from .env ──
    # Written by the old setup wizard; nothing reads them (config.yaml is the
    # sole source of truth). Stale entries only confuse users.
    _c = _cfg()
    for dead_var in ("LLM_MODEL", "OPENAI_MODEL"):
        try:
            old_val = _c.get_env_value(dead_var)
            if old_val:
                _c.save_env_value(dead_var, "")
                if not quiet:
                    print(f"  ✓ Cleared {dead_var} from .env (no longer used — config.yaml is source of truth)")
        except Exception:
            pass


def _migrate_to_14(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 13 → 14: migrate legacy flat stt.model to provider section ──
    # A provider-agnostic `stt.model` fed OpenAI names to faster-whisper
    # ("Invalid model size"). Move it into the provider's section; drop the flat key.

    # Read raw config (no defaults merged) to check what the user actually
    # wrote, then apply changes to the merged config for saving.
    raw = read_raw_config()
    raw_stt = raw.get("stt", {})
    if isinstance(raw_stt, dict) and "model" in raw_stt:
        legacy_model = raw_stt["model"]
        provider = raw_stt.get("provider", "local")
        config = read_raw_config()
        stt = config.get("stt", {})
        # Remove the legacy flat key
        stt.pop("model", None)
        # Place it in the appropriate provider section only if the
        # user didn't already set a model there
        if provider in {"local", "local_command"}:
            # Don't migrate an OpenAI model name into the local section
            _local_models = {
                "tiny.en", "tiny", "base.en", "base", "small.en", "small",
                "medium.en", "medium", "large-v1", "large-v2", "large-v3",
                "large", "distil-large-v2", "distil-medium.en",
                "distil-small.en", "distil-large-v3", "distil-large-v3.5",
                "large-v3-turbo", "turbo",
            }
            if legacy_model in _local_models:
                # Check raw config — only set if user didn't already
                # have a nested local.model
                raw_local = raw_stt.get("local", {})
                if not isinstance(raw_local, dict) or "model" not in raw_local:
                    local_cfg = stt.setdefault("local", {})
                    local_cfg["model"] = legacy_model
            # else: drop it — it was an OpenAI model name, local section
            # already defaults to "base" via DEFAULT_CONFIG
        else:
            # Cloud provider — put it in that provider's section only
            # if user didn't already set a nested model
            raw_provider = raw_stt.get(provider, {})
            if not isinstance(raw_provider, dict) or "model" not in raw_provider:
                provider_cfg = stt.setdefault(provider, {})
                provider_cfg["model"] = legacy_model
        config["stt"] = stt
        _persist_migration(config)
        if not quiet:
            print("  ✓ Migrated legacy stt.model to provider-specific config")


def _migrate_to_16(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 15 → 16: migrate tool_progress_overrides into display.platforms ──

    config = read_raw_config()
    display = config.get("display", {})
    if not isinstance(display, dict):
        display = {}
    old_overrides = display.get("tool_progress_overrides")
    if isinstance(old_overrides, dict) and old_overrides:
        platforms = display.get("platforms", {})
        if not isinstance(platforms, dict):
            platforms = {}
        for plat, mode in old_overrides.items():
            if plat not in platforms:
                platforms[plat] = {}
            if "tool_progress" not in platforms[plat]:
                platforms[plat]["tool_progress"] = mode
        display["platforms"] = platforms
        config["display"] = display
        _persist_migration(config)
        if not quiet:
            migrated = ", ".join(f"{p}={m}" for p, m in old_overrides.items())
            print(f"  ✓ Migrated tool_progress_overrides → display.platforms: {migrated}")
        results["config_added"].append("display.platforms (migrated from tool_progress_overrides)")


def _migrate_to_17(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 16 → 17: remove legacy compression.summary_* keys ──

    config = read_raw_config()
    comp = config.get("compression", {})
    if isinstance(comp, dict):
        legacy = {k: comp.pop(f"summary_{k}", None) for k in ("model", "provider", "base_url")}
        migrated_keys = []
        # Migrate non-empty, non-default values to auxiliary.compression, never
        # overriding an explicit (non-"auto") aux value.
        for k, raw in legacy.items():
            val = str(raw).strip() if raw else ""
            if not val or (k == "provider" and val == "auto"):
                continue
            aux_comp = config.setdefault("auxiliary", {}).setdefault("compression", {})
            cur = aux_comp.get(k)
            if not cur or (k == "provider" and cur == "auto"):
                aux_comp[k] = val
                migrated_keys.append(f"{k}={raw}")
        if migrated_keys or any(v is not None for v in legacy.values()):
            config["compression"] = comp
            _persist_migration(config)
            if not quiet:
                if migrated_keys:
                    print(f"  ✓ Migrated compression.summary_* → auxiliary.compression: {', '.join(migrated_keys)}")
                else:
                    print("  ✓ Removed unused compression.summary_* keys")


def _migrate_to_21(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 20 → 21: plugins are now opt-in; grandfather existing user plugins ──
    # The loader requires ``plugins.enabled``; existing installs loaded every
    # discovered plugin (minus ``plugins.disabled``). Populate the allow-list
    # with installed user plugins not already disabled. Bundled plugins are NOT
    # grandfathered — they ship off for everyone and need explicit opt-in.
    _c = _cfg()

    config = read_raw_config()
    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    # Only migrate if the enabled allow-list hasn't been set yet.
    if "enabled" not in plugins_cfg:
        disabled = plugins_cfg.get("disabled", []) or []
        if not isinstance(disabled, list):
            disabled = []
        disabled_set = set(disabled)

        # Scan ``$HERMES_HOME/plugins/`` for currently installed user plugins.
        grandfathered: List[str] = []
        try:
            user_plugins_dir = _c.get_hermes_home() / "plugins"
            if user_plugins_dir.is_dir():
                for child in sorted(user_plugins_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    manifest_file = child / "plugin.yaml"
                    if not manifest_file.exists():
                        manifest_file = child / "plugin.yml"
                    if not manifest_file.exists():
                        continue
                    try:
                        with open(manifest_file, encoding="utf-8") as _mf:
                            manifest = _c.fast_safe_load(_mf) or {}
                    except Exception:
                        manifest = {}
                    name = manifest.get("name") or child.name
                    if name in disabled_set:
                        continue
                    grandfathered.append(name)
        except Exception:
            grandfathered = []

        plugins_cfg["enabled"] = grandfathered
        config["plugins"] = plugins_cfg
        _persist_migration(config)
        results["config_added"].append(
            f"plugins.enabled (opt-in allow-list, {len(grandfathered)} grandfathered)"
        )
        if not quiet:
            if grandfathered:
                print(
                    f"  ✓ Plugins now opt-in: grandfathered "
                    f"{len(grandfathered)} existing plugin(s) into plugins.enabled"
                )
            else:
                print(
                    "  ✓ Plugins now opt-in: no existing plugins to grandfather. "
                    "Use `hermes plugins enable <name>` to activate."
                )


def _migrate_to_23(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 22 → 23: seed curator defaults + create logs/curator/ ──
    # Older configs never wrote the curator section; deep-merge makes it
    # function, but users could not see/edit it and `hermes curator status`
    # had no stable logs dir. Writes `curator` and the `auxiliary.curator`
    # aux-task slot (only keys the user hasn't overridden) and mkdirs
    # logs/curator/ (belt-and-suspenders over ensure_hermes_home()).
    _c = _cfg()
    DEFAULT_CONFIG = _c.DEFAULT_CONFIG

    try:
        curator_dir = _c.get_hermes_home() / "logs" / "curator"
        curator_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        results["warnings"].append(f"Could not create {curator_dir}: {e}")

    config = read_raw_config()

    def _seed_missing(section: Dict[str, Any], defaults: Dict[str, Any]) -> List[str]:
        added = [k for k in defaults if k not in section]
        for k in added:
            section[k] = copy.deepcopy(defaults[k])
        return added

    # (1) Top-level curator section — only add missing keys
    raw_curator = config.get("curator")
    if not isinstance(raw_curator, dict):
        raw_curator = {}
    added_curator = _seed_missing(raw_curator, DEFAULT_CONFIG.get("curator", {}))
    if added_curator:
        config["curator"] = raw_curator

    # (2) auxiliary.curator task slot
    raw_aux = config.get("auxiliary")
    if not isinstance(raw_aux, dict):
        raw_aux = {}
    raw_aux_curator = raw_aux.get("curator")
    if not isinstance(raw_aux_curator, dict):
        raw_aux_curator = {}
    added_aux = _seed_missing(raw_aux_curator, DEFAULT_CONFIG.get("auxiliary", {}).get("curator", {}))
    if added_aux:
        raw_aux["curator"] = raw_aux_curator
        config["auxiliary"] = raw_aux

    if added_curator or added_aux:
        _persist_migration(config)
        for label, added in (("curator", added_curator), ("auxiliary.curator", added_aux)):
            if not added:
                continue
            results["config_added"].append(f"{label} ({len(added)} default key(s))")
            if not quiet:
                print(
                    f"  ✓ {'Curator' if label == 'curator' else label} settings now available "
                    f"({', '.join(added)}) — edit via `hermes config set`"
                )


def _migrate_to_25(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 24 → 25: lower model_catalog TTL 24h → 1h (only the OLD default 24) ──

    _rewrite_stale_default(
        results, quiet, section="model_catalog", key="ttl_hours", old=24, new=1,
        added="model_catalog.ttl_hours 24→1",
        message="  ✓ Lowered model_catalog.ttl_hours to 1 (hourly picker refresh)",
    )


def _migrate_to_29(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 28 → 29: rename memory/skills write_mode → write_approval ──
    # Tri-state write_mode (on|off|approve) → boolean write_approval. Only
    # "approve" carried gating intent → true; everything else → false (the old
    # "off = block writes" mode is dropped; memory_enabled: false disables
    # memory). Only rewrite a key the user actually persisted.

    config = read_raw_config()
    touched = False
    for subsystem in ("memory", "skills"):
        sub = config.get(subsystem)
        if not isinstance(sub, dict) or "write_mode" not in sub:
            continue
        old = sub.pop("write_mode")
        old_norm = old.strip().lower() if isinstance(old, str) else old
        sub["write_approval"] = (old_norm == "approve")
        config[subsystem] = sub
        touched = True
        results["config_added"].append(
            f"{subsystem}.write_mode → write_approval={sub['write_approval']}"
        )
    if touched:
        _persist_migration(config)
        if not quiet:
            print("  ✓ Renamed write_mode → write_approval (boolean gate)")


# ── Version 29 → 30: curator.consolidate defaults to false ──
# Schema-default-only change (deep-merge supplies it at read time; persisting
# a default would only bloat a lean config). No registry entry.


def _migrate_to_31(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 30 → 31: switch verify_on_stop OFF (one-time) ──
    # The "auto" sentinel (surface-aware) was more noise than signal; the new
    # default is OFF. Rewrite only when missing or still "auto" — an explicit
    # true/false the user set is preserved.

    config = read_raw_config()
    raw_agent = config.get("agent")
    if not isinstance(raw_agent, dict):
        raw_agent = {}
    cur = raw_agent.get("verify_on_stop")
    is_auto_sentinel = (
        isinstance(cur, str) and cur.strip().lower() == "auto"
    )
    # Only flip the non-committal states; leave explicit bool/on/off alone.
    if cur is None or is_auto_sentinel:
        raw_agent["verify_on_stop"] = False
        config["agent"] = raw_agent
        _persist_migration(config)
        results["config_added"].append("agent.verify_on_stop=false")
        if not quiet:
            print(
                "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false). "
                "Set it to true to re-enable, or \"auto\" for the legacy "
                "surface-aware behavior."
            )


def _migrate_to_32(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 31 → 32: flip the BAKED-IN literal true to OFF (one-time) ──
    # v31 only caught missing/"auto". The first ship (v30) defaulted
    # verify_on_stop to a literal True and migrate_config persisted defaults,
    # so every install that updated through v30 has `verify_on_stop: true`
    # written literally — never a user choice (there was no off-switch until
    # v31). Flip it once; a true set AFTER v32 is never touched.

    _rewrite_stale_default(
        results, quiet, section="agent", key="verify_on_stop", old=True, new=False,
        added="agent.verify_on_stop=false",
        message=(
            "  ✓ Turned off verify-on-stop (agent.verify_on_stop: false) — "
            "the old default was written into your config as a literal "
            "true. Set it to true again to re-enable, or \"auto\" for the "
            "legacy surface-aware behavior."
        ),
        extra_guard=lambda raw: raw.get("verify_on_stop") is True,
    )


def _migrate_to_33(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 32 → 33: unify delegation concurrency caps ──
    # max_async_children is deprecated; fold a raised value into
    # max_concurrent_children (take the max so nobody loses headroom), drop it.

    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if isinstance(raw_deleg, dict) and "max_async_children" in raw_deleg:
        old_async = raw_deleg.pop("max_async_children")
        try:
            old_async_i = int(old_async)
        except (TypeError, ValueError):
            old_async_i = None
        if old_async_i is not None and old_async_i > 3:
            try:
                cur_children = int(raw_deleg.get("max_concurrent_children", 3))
            except (TypeError, ValueError):
                cur_children = 3
            if old_async_i > cur_children:
                raw_deleg["max_concurrent_children"] = old_async_i
                results["config_added"].append(
                    f"delegation.max_concurrent_children={old_async_i} "
                    f"(folded from deprecated max_async_children)"
                )
        config["delegation"] = raw_deleg
        _persist_migration(config)
        if not quiet:
            print(
                "  ✓ Removed deprecated delegation.max_async_children — "
                "delegation.max_concurrent_children now caps background "
                "delegations too."
            )


def _migrate_to_34(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 33 → 34: one-time personality reset (post-unification) ──
    # Persistence used to be split: TUI/desktop wrote the NAME to
    # display.personality, CLI/gateway wrote rendered TEXT to agent.system_prompt
    # (and "/personality none" only blanked the text). Once display.personality
    # became authoritative, stale names resurrected personalities users had
    # turned off. Neither field is trustworthy, so reset once:
    # 1. display.personality → "" (announce the old name).
    # 2. agent.system_prompt → "" ONLY when it verbatim-equals a known
    #    personality's rendered text (written by the old /personality, never
    #    typed by hand). Any other text is a user-owned prompt, never touched.

    from hermes_cli.personality import (
        available_personalities,
        normalize_personality_name,
        prompt_text,
        render_personality_prompt,
    )

    config = read_raw_config()
    touched = False

    raw_display = config.get("display")
    old_name = ""
    if isinstance(raw_display, dict):
        old_name = normalize_personality_name(raw_display.get("personality", ""))
        if old_name:
            raw_display["personality"] = ""
            config["display"] = raw_display
            touched = True

    raw_agent = config.get("agent")
    scrubbed_text = False
    if isinstance(raw_agent, dict):
        manual = prompt_text(raw_agent.get("system_prompt", ""))
        if manual:
            rendered = {
                render_personality_prompt(defn)
                for defn in available_personalities(config).values()
            }
            if manual in rendered:
                raw_agent["system_prompt"] = ""
                config["agent"] = raw_agent
                touched = True
                scrubbed_text = True

    if touched:
        _persist_migration(config)
        results["config_added"].append("display.personality=none (one-time reset)")
        if not quiet:
            if old_name:
                print(
                    f"  ✓ Personality reset to none (was '{old_name}'). Personality "
                    "state was previously saved inconsistently across surfaces and "
                    "could re-enable a personality you had turned off. "
                    f"Run /personality {old_name} to turn it back on."
                )
            if scrubbed_text:
                print(
                    "  ✓ Removed personality text from agent.system_prompt (written "
                    "by an older /personality). That field is now reserved for "
                    "manual system prompts; personalities live in display.personality."
                )


def _migrate_to_35(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 34 → 35: background process notifications → concise ──
    # 'all' (old implicit default, rarely chosen on purpose) dumped raw output
    # walls into chat; 'concise' is the new default. Move 'all' → 'concise';
    # explicit result/error/off choices are preserved; unset inherits at read.

    config = read_raw_config()
    raw_display = config.get("display")
    if isinstance(raw_display, dict):
        raw_val = raw_display.get("background_process_notifications")
        if isinstance(raw_val, str) and raw_val.strip().lower() == "all":
            raw_display["background_process_notifications"] = "concise"
            config["display"] = raw_display
            _persist_migration(config)
            results["config_added"].append(
                "display.background_process_notifications=concise (was: all)"
            )
            if not quiet:
                print(
                    "  ✓ Background process notifications switched from 'all' to "
                    "'concise' — completions now show a one-line status message "
                    "instead of the raw output dump. Set "
                    "display.background_process_notifications: all to restore "
                    "the old behavior."
                )


def _migrate_to_36(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 35 → 36: raise the subagent iteration cap default 50 → 250 ──
    # 50 truncated substantial delegated work; only a value still pinned at
    # exactly the old default is lifted. Other explicit values are preserved.

    _rewrite_stale_default(
        results, quiet, section="delegation", key="max_iterations", old=50, new=250,
        added="delegation.max_iterations=250 (was: 50)",
        message=(
            "  ✓ Raised delegation.max_iterations from 50 to 250 — subagents "
            "now get a larger per-child tool-call budget so delegated work "
            "finishes instead of truncating. Set delegation.max_iterations "
            "back to 50 to restore the old cap."
        ),
    )


def _migrate_to_37(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 36 → 37: raise the delegation concurrency default 3 → 10 ──
    # 3 serialized independent fan-outs; 10 stays at/below the high-cost
    # warning threshold. Only a value still pinned at exactly 3 is lifted.

    _rewrite_stale_default(
        results, quiet, section="delegation", key="max_concurrent_children", old=3, new=10,
        added="delegation.max_concurrent_children=10 (was: 3)",
        message=(
            "  ✓ Raised delegation.max_concurrent_children from 3 to 10 — "
            "independent delegated children now fan out wider in parallel. "
            "Each child consumes API tokens independently; set "
            "delegation.max_concurrent_children back to 3 to restore the old cap."
        ),
    )


def _migrate_to_38(results: Dict[str, Any], quiet: bool) -> None:
    # Version 37 → 38: the bundled observability/nemo_relay plugin was
    # removed when Relay lifecycle ownership moved into the agent core.

    from hermes_cli.relay_plugin_cutover import legacy_relay_plugin_keys

    config = read_raw_config()
    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        return
    enabled = plugins.get("enabled")
    removed = legacy_relay_plugin_keys(enabled)
    if not removed or not isinstance(enabled, list):
        return

    plugins["enabled"] = [value for value in enabled if value not in removed]
    config["plugins"] = plugins
    _persist_migration(config)
    message = (
        "Removed legacy Relay plugin from plugins.enabled: "
        f"{', '.join(removed)}. Configure native Relay plugins with "
        "HERMES_NEMO_RELAY_PLUGINS_TOML."
    )
    results["warnings"].append(message)
    if not quiet:
        print(f"  ⚠ {message}")


def _migrate_to_39(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 38 → 39: remove the retired `bfl` toolset from saved lists ──
    # The FLUX 3 promo tools were removed in favor of the video_gen provider
    # surface; strip the key wherever a backfill/picker save wrote it so stale
    # config can't resurrect an unknown toolset.

    config = read_raw_config()
    changed = False
    for section in ("platform_toolsets", "known_builtin_toolsets"):
        mapping = config.get(section)
        if not isinstance(mapping, dict):
            continue
        for platform, toolsets in mapping.items():
            if isinstance(toolsets, list) and "bfl" in toolsets:
                mapping[platform] = [ts for ts in toolsets if ts != "bfl"]
                changed = True
        if changed:
            config[section] = mapping
    if changed:
        _persist_migration(config)
        results["config_added"].append("removed retired 'bfl' toolset from saved toolset lists")
        if not quiet:
            print(
                "  ✓ Removed the retired BFL FLUX 3 toolset from saved toolset "
                "lists — video generation now lives under `hermes tools` → "
                "Video Generation (Nous Subscription or FAL)."
            )


def _migrate_to_40(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 39 → 40: model_catalog.ttl_hours → ttl_minutes (default 20) ──
    # Only the OLD default (ttl_hours: 1, written by v25) is dropped; any other
    # explicit ttl_hours is a deliberate choice the loader still honours.

    _rewrite_stale_default(
        results, quiet, section="model_catalog", key="ttl_hours", old=1, new=None,
        added="model_catalog.ttl_hours 1 → ttl_minutes 20 (default)",
        message="  ✓ Model catalog now refreshes every 20 minutes (model_catalog.ttl_minutes)",
        extra_guard=lambda raw: "ttl_minutes" not in raw,
    )


#: Registry of (target_version, migration_fn), strictly ascending. The driver
#: applies every entry whose target version is greater than the on-disk
#: observe earlier steps' writes via read_raw_config() (filesystem state).
MIGRATIONS: Tuple[Tuple[int, Callable[[Dict[str, Any], bool], None]], ...] = (
    # v12 is the support floor: configs already AT v12 (or newer) still get
    # every remaining step below. Only configs BELOW 12 are refused by the
    # floor gate in run_migrations().
    (12, _migrate_to_12),
    (13, _migrate_to_13),
    (14, _migrate_to_14),
    # v15 only added a schema default; runtime merging supplies it without a
    # write. Registering a migration would falsely report or materialise it.
    (16, _migrate_to_16),
    (17, _migrate_to_17),
    (21, _migrate_to_21),
    (23, _migrate_to_23),
    (25, _migrate_to_25),
    (29, _migrate_to_29),
    (31, _migrate_to_31),
    (32, _migrate_to_32),
    (33, _migrate_to_33),
    (34, _migrate_to_34),
    (35, _migrate_to_35),
    (36, _migrate_to_36),
    (37, _migrate_to_37),
    (38, _migrate_to_38),
    (39, _migrate_to_39),
    (40, _migrate_to_40),
)


def run_migrations(current_ver: int, results: Dict[str, Any], quiet: bool) -> None:
    """Apply every registered migration whose target version exceeds *current_ver*.

    Replicates the original ladder's semantics exactly: *current_ver* is the on-disk schema version
    captured ONCE (via ``check_config_version()``) before any step runs, and it does not advance
    between steps — each step is gated on the same initial value, exactly like the original
    sequential ``if current_ver < N:`` blocks.
    """
    for target_ver, migration_fn in MIGRATIONS:
        if current_ver < target_ver:
            migration_fn(results, quiet)
