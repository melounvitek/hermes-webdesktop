"""Memory-provider setup dashboard routes (schema, existing values, external dependency install).

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
import math
import re
import shlex
import asyncio
from fastapi import APIRouter
from fastapi import HTTPException
from hermes_cli.web_models import MemoryProviderConfigUpdate, MemoryProviderSetupRequest
from plugins.memory.config_schema import get_provider_config_schema
from typing import Any, Dict, List, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


def _install_memory_provider_external_dependencies(
    dependencies: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    from hermes_cli.web_server import _command_result, _run_setup_command
    results: List[Dict[str, Any]] = []
    for dep in dependencies:
        name = dep.get("name") or "dependency"
        check_cmd = dep.get("check") or ""
        install_cmd = dep.get("install") or ""

        if check_cmd:
            try:
                check = _run_setup_command(
                    shlex.split(check_cmd),
                    display=check_cmd,
                    timeout=20,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        error=str(exc),
                    )
                )
            else:
                if check.returncode == 0:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="already_installed",
                            command=check_cmd,
                            completed=check,
                        )
                    )
                    continue
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        completed=check,
                    )
                )

            if not install_cmd:
                continue

        if install_cmd:
            try:
                install = _run_setup_command(
                    install_cmd,
                    display=install_cmd,
                    shell=True,
                    timeout=300,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_install",
                        name=name,
                        status="failed",
                        command=install_cmd,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                _command_result(
                    kind="external_install",
                    name=name,
                    status="installed" if install.returncode == 0 else "failed",
                    command=install_cmd,
                    completed=install,
                )
            )

            if check_cmd and install.returncode == 0:
                try:
                    post_check = _run_setup_command(
                        shlex.split(check_cmd),
                        display=check_cmd,
                        timeout=20,
                    )
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="verified" if post_check.returncode == 0 else "failed",
                            command=check_cmd,
                            completed=post_check,
                        )
                    )
                except Exception as exc:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="failed",
                            command=check_cmd,
                            error=str(exc),
                        )
                    )

    return results


def _install_memory_provider_setup(name: str) -> Dict[str, Any]:
    from hermes_cli.web_server import (
        _command_result,
        _discover_memory_provider_statuses,
        _install_memory_provider_pip_dependencies,
        _load_memory_provider,
        _memory_provider_manifest,
        _memory_provider_setup_manifest,
    )
    provider = _load_memory_provider(name)
    manifest = _memory_provider_manifest(name)
    if provider is None and not manifest:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    setup = _memory_provider_setup_manifest(name)
    results = []
    results.extend(_install_memory_provider_pip_dependencies(setup["pip_dependencies"]))
    results.extend(
        _install_memory_provider_external_dependencies(setup["external_dependencies"])
    )

    if not results:
        results.append(
            _command_result(
                kind="setup",
                name=name,
                status="no_declared_steps",
            )
        )

    ok = all(result["status"] not in {"failed"} for result in results)
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    return {
        "ok": ok,
        "provider": name,
        "results": results,
        "status": statuses.get(name),
    }


def _public_memory_provider_field(field: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.web_server import _field_is_set, _field_value
    entry = {
        "key": field["key"],
        "label": field["label"],
        "kind": field["kind"],
        "description": field["description"],
        "placeholder": field["placeholder"],
        "required": field["required"],
        "value": "" if field["kind"] == "secret" else _field_value(field, data),
        "is_set": _field_is_set(field, data),
        "options": field.get("options", []),
        "url": field.get("url", ""),
        "when": field.get("when"),
        "minimum": field.get("minimum"),
        "maximum": field.get("maximum"),
        "step": field.get("step"),
    }
    return entry


def _memory_provider_payload(name: str, provider: Any) -> Dict[str, Any]:
    from hermes_cli.web_server import (
        _memory_provider_label,
        _memory_provider_setup_info,
        _normalize_memory_provider_schema,
        _read_memory_provider_existing_values,
    )
    data = _read_memory_provider_existing_values(name)
    fields = [
        _public_memory_provider_field(field, data)
        for field in _normalize_memory_provider_schema(name, provider)
    ]
    return {
        "name": name,
        "label": _memory_provider_label(name),
        "fields": fields,
        "setup": _memory_provider_setup_info(name),
    }


def _coerce_schema_field(field: Dict[str, Any], raw: Any) -> Any:
    from hermes_cli.web_server import _coerce_bool, _field_default
    if field["kind"] == "boolean":
        return _coerce_bool(raw, default=_coerce_bool(_field_default(field), default=False))

    if field["kind"] in {"integer", "number"}:
        value = raw if raw is not None and raw != "" else _field_default(field)
        try:
            if isinstance(value, bool):
                raise ValueError
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError
            if field["kind"] == "integer":
                if not parsed.is_integer():
                    raise ValueError
                result: int | float = int(parsed)
            else:
                result = parsed
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid numeric value for '{field['key']}'") from exc

        minimum = field.get("minimum")
        maximum = field.get("maximum")
        if minimum is not None and result < minimum:
            raise ValueError(f"'{field['key']}' must be at least {minimum}")
        if maximum is not None and result > maximum:
            raise ValueError(f"'{field['key']}' must be at most {maximum}")
        return result

    value = str(raw if raw is not None else "").strip()
    if field["kind"] == "select":
        if not value:
            value = str(_field_default(field))
        allowed = {opt["value"] for opt in field.get("options", [])}
        if value not in allowed:
            raise ValueError(f"Invalid value for '{field['key']}'")
        return value

    return value or _field_default(field)


def _save_memory_provider_native_config(name: str, provider: Any, values: Dict[str, Any]) -> None:
    from hermes_cli.web_server import get_hermes_home, load_config, save_config
    if provider is not None and hasattr(provider, "save_config"):
        try:
            from agent.memory_provider import MemoryProvider as _BaseMemoryProvider
        except Exception:
            provider.save_config(values, str(get_hermes_home()))
            return
        if type(provider).save_config is not _BaseMemoryProvider.save_config:
            provider.save_config(values, str(get_hermes_home()))
            return

    cfg = load_config()
    memory_cfg = cfg.get("memory")
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
        cfg["memory"] = memory_cfg
    current = memory_cfg.get(name)
    if not isinstance(current, dict):
        current = {}
    current.update(values)
    memory_cfg[name] = current
    save_config(cfg)


def _write_memory_provider_config_values(
    name: str,
    provider: Any,
    values: Dict[str, Any],
) -> None:
    from hermes_cli.web_server import (
        _field_default,
        _field_visible,
        _normalize_memory_provider_schema,
        _read_memory_provider_existing_values,
        save_env_value,
    )
    existing = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    config_values: Dict[str, Any] = {}
    secrets: Dict[str, str] = {}

    for field in fields:
        if not _field_visible(field, {**existing, **config_values}, fields_by_key):
            continue

        if field["kind"] == "secret":
            submitted = str(values.get(field["key"]) or "").strip()
            if submitted and field.get("_env_key"):
                secrets[str(field["_env_key"])] = submitted
            continue

        raw = (
            values[field["key"]]
            if field["key"] in values
            else existing.get(field["key"], _field_default(field))
        )
        config_values[field["key"]] = _coerce_schema_field(field, raw)

    _save_memory_provider_native_config(name, provider, config_values)

    for env_key, secret in secrets.items():
        save_env_value(env_key, secret)


_MEMORY_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _require_valid_memory_provider_name(name: str) -> None:
    """Reject provider names that could traverse outside the plugin dirs.

    ``name`` is interpolated into filesystem paths by ``find_provider_dir()``
    and gates which plugin manifest's setup commands run. A strict charset
    allowlist (no path separators, no dots) makes traversal impossible
    regardless of how the downstream lookup evolves.
    """
    if not _MEMORY_PROVIDER_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")


@router.get("/api/memory/providers/{name}/config")
async def get_memory_provider_config(name: str, surface: Optional[str] = None, profile: Optional[str] = None):
    from hermes_cli.web_server import (
        _declared_provider_payload,
        _load_memory_provider,
        _memory_provider_setup_info,
        _profile_scope,
    )
    _require_valid_memory_provider_name(name)

    def _run():
        with _profile_scope(profile):
            if surface == "declared":
                declared = get_provider_config_schema(name)
                if declared is None:
                    # Undeclared providers (e.g. builtin) have no desktop
                    # config surface; the generic panel renders nothing.
                    return {"name": name, "label": name, "docs_url": "", "fields": []}
                return _declared_provider_payload(declared)

            provider = _load_memory_provider(name)
            if provider is None:
                # Undeclared providers (e.g. builtin) have no config surface. Return an
                # empty schema so the generic panel simply renders nothing.
                return {"name": name, "label": name, "fields": [], "setup": _memory_provider_setup_info(name)}
            return _memory_provider_payload(name, provider)

    return await asyncio.to_thread(_run)


@router.post("/api/memory/providers/{name}/setup")
async def setup_memory_provider(name: str, body: MemoryProviderSetupRequest):
    from hermes_cli.web_server import (
        _invalidate_plugins_hub_cache,
        _load_memory_provider,
        _memory_provider_manifest,
    )
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None and not _memory_provider_manifest(name):
        # No discoverable plugin directory → nothing whose manifest could
        # legitimately declare setup commands. Refuse before the
        # command-running path. (provider may be None with a manifest present
        # when its pip deps aren't installed yet — that's the setup use case.)
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")
    if provider is not None and body.values:
        try:
            _write_memory_provider_config_values(name, provider, body.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            _log.exception("Failed to persist memory provider setup values for %s", name)
            raise HTTPException(status_code=500, detail="Internal server error")
    _invalidate_plugins_hub_cache()
    return _install_memory_provider_setup(name)


@router.put("/api/memory/providers/{name}/config")
async def update_memory_provider_config(
    name: str, body: MemoryProviderConfigUpdate, surface: Optional[str] = None, profile: Optional[str] = None
):
    from hermes_cli.web_server import (
        _invalidate_plugins_hub_cache,
        _load_memory_provider,
        _profile_scope,
        _require_memory_provider_ready,
        _stringify_submitted_values,
        _update_memory_provider_config,
        load_config,
        save_config,
    )
    _require_valid_memory_provider_name(name)
    values = body.values or {}

    def _run():
        with _profile_scope(profile):
            if surface == "declared":
                declared = get_provider_config_schema(name)
                if declared is None:
                    raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")
                _update_memory_provider_config(declared, _stringify_submitted_values(values))
                _invalidate_plugins_hub_cache()
                return {"ok": True}

            provider = _load_memory_provider(name)
            if provider is None:
                raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")
            _write_memory_provider_config_values(name, provider, values)
            _require_memory_provider_ready(name)
            config = load_config()
            memory_config = config.get("memory")
            if not isinstance(memory_config, dict):
                memory_config = {}
                config["memory"] = memory_config
            memory_config["provider"] = name
            save_config(config)
            _invalidate_plugins_hub_cache()
            return {"ok": True, "active": name}

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/memory/providers/%s/config failed", name)
        raise HTTPException(status_code=500, detail="Internal server error")
