"""Memory-provider setup dashboard routes (schema, existing values, external dependency install).

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import logging
import math
import re
import shlex
from fastapi import APIRouter
from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import scoped_to_thread
from fastapi import HTTPException
from hermes_cli.web_models import MemoryProviderConfigUpdate, MemoryProviderSetupRequest
from plugins.memory.config_schema import get_provider_config_schema
from typing import Any, Dict, List, Optional
from plugins.memory.config_schema import ProviderConfigSchema, ProviderField, STORAGE_HONCHO_HOST_BLOCK
import subprocess
import json
from pathlib import Path

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# web_server helpers, late-bound so monkeypatch.setattr(web_server, ...) stays authoritative.
_coerce_bool = late("_coerce_bool")
_discover_memory_provider_statuses = late("_discover_memory_provider_statuses")
_field_default = late("_field_default")
_field_is_set = late("_field_is_set")
_field_value = late("_field_value")
_field_visible = late("_field_visible")
_invalidate_plugins_hub_cache = late("_invalidate_plugins_hub_cache")
_load_memory_provider = late("_load_memory_provider")
_memory_provider_manifest = late("_memory_provider_manifest")
_memory_provider_setup_info = late("_memory_provider_setup_info")
_memory_provider_setup_manifest = late("_memory_provider_setup_manifest")
_normalize_memory_provider_schema = late("_normalize_memory_provider_schema")
_read_memory_provider_existing_values = late("_read_memory_provider_existing_values")
_require_memory_provider_ready = late("_require_memory_provider_ready")
_run_setup_command = late("_run_setup_command")
get_hermes_home = late("get_hermes_home")
load_config = late("load_config")
save_config = late("save_config")
save_env_value = late("save_env_value")
_dependency_importable = late("_dependency_importable")
load_env = late("load_env")


# Sentinel: remove this key so it falls back to the host or built-in default.
_UNSET: Any = object()


def _coerce_field_value(field: ProviderField, raw: str) -> Any:
    """Coerce a submitted non-secret string to its native JSON type.

    A bool is stored as JSON ``false`` rather than ``"false"`` (truthy). Returns
    ``_UNSET`` when the field should be removed; raises ``ValueError`` on
    malformed input.
    """
    value = (raw or "").strip()
    kind = field.kind

    if kind == "select":
        if not value:
            value = field.default
        if value not in field.allowed_values():
            raise ValueError(f"Invalid value for '{field.key}'")
        return value

    if kind == "bool":
        from utils import is_truthy_value

        return is_truthy_value(value)

    if not value:
        # number / json / text / secret — blank clears the key.
        return _UNSET

    if kind == "number":
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid number for '{field.key}'") from exc
        return int(number) if number.is_integer() else number

    if kind == "json":
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid JSON for '{field.key}'") from exc
        if not isinstance(parsed, (dict, list)):
            raise ValueError(f"'{field.key}' must be a JSON object or array")
        return parsed

    return value


def _read_json_dict(path: Path, log_msg: str) -> Dict[str, Any]:
    """Read a JSON object from ``path``; missing/unreadable/non-dict -> ``{}``."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.warning(log_msg, path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


# — flat-json backend (default; reusable for simple providers) —


def _flat_json_path(provider: ProviderConfigSchema) -> Path:
    return get_hermes_home() / provider.name / "config.json"


def _read_flat_json(provider: ProviderConfigSchema) -> Dict[str, Any]:
    return _read_json_dict(_flat_json_path(provider), "Failed to read memory provider config from %s")


# — honcho host-block backend —


def _honcho_resolvers():
    """Lazily import the Honcho plugin's resolvers (optional plugin)."""
    from plugins.memory.honcho.client import _host_block, resolve_active_host, resolve_config_path

    return resolve_active_host, resolve_config_path, _host_block


def _read_honcho_config(path: Path) -> Dict[str, Any]:
    return _read_json_dict(path, "Failed to read Honcho config from %s")


def _apply_field_values(provider: ProviderConfigSchema, values: Dict[str, str], target_for) -> None:
    """Apply submitted non-secret fields to their backend dict, in place.

    Only keys present in ``values`` are touched, so a partial save never
    clobbers fields owned by another surface. ``_UNSET`` clears the key (and
    its aliases) so it falls back to the host/default mapping.
    """
    for field in provider.fields:
        if field.is_secret or field.key not in values:
            continue
        target = target_for(field)
        coerced = _coerce_field_value(field, values[field.key])
        if coerced is _UNSET:
            for key in (field.key, *field.aliases):
                target.pop(key, None)
        else:
            target[field.key] = coerced


def _trim_setup_output(value: Optional[str], limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated ..."


# ── Memory provider config: one generic GET/PUT pair, dispatching on storage ──


def _provider_field_entry(field: ProviderField) -> Dict[str, Any]:
    """Static, storage-independent shape of one field for the UI payload."""
    return {
        "key": field.key,
        "label": field.label,
        "kind": field.kind,
        "description": field.description,
        "info": field.info,
        "placeholder": field.placeholder,
        "inline": field.inline,
        "group": field.group,
        "options": [
            {"value": opt.value, "label": opt.label, "description": opt.description}
            for opt in field.options
        ],
    }


def _serialize_field_value(field: ProviderField, value: Any) -> str:
    """Render a stored native value as the string the generic UI edits.

    ``None`` (key absent) yields the field's declared default; bools become
    ``"true"``/``"false"``, JSON objects/arrays are re-encoded, numbers are
    stringified.
    """
    if value is None:
        return field.default
    if field.kind == "bool":
        from utils import is_truthy_value

        return "true" if is_truthy_value(value) else "false"
    if field.kind == "json" and isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _read_field(field: ProviderField, sources: tuple, env: Dict[str, str]) -> Any:
    """Return the stored native value from the first source holding it, or ``None``.

    Presence (``key in source``) decides, not truthiness, so a stored ``False``
    or ``0`` survives instead of being mistaken for "unset".
    """
    for source in sources:
        for source_key in (field.key, *field.aliases):
            if source_key in source and source[source_key] is not None:
                return source[source_key]
    for env_key in field.env_fallbacks:
        value = env.get(env_key)
        if value:
            return value
    return None


def _declared_field_is_set(field: ProviderField, sources: tuple, env: Dict[str, str]) -> bool:
    for env_key in (field.env_key, *field.env_fallbacks):
        if env_key and env.get(env_key):
            return True
    return any(source.get(k) for source in sources for k in (field.key, *field.aliases))


def _honcho_read_sources() -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Return (root config, active host key, host block) for the current profile."""
    resolve_active_host, resolve_config_path, host_block_of = _honcho_resolvers()
    host = resolve_active_host()
    raw = _read_honcho_config(resolve_config_path())
    return raw, host, host_block_of(raw, host)


def _write_provider_flat(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    from utils import atomic_json_write

    existing = _read_flat_json(provider)

    for field in provider.fields:
        if field.is_secret:
            submitted = (values.get(field.key) or "").strip()
            if submitted and field.env_key:
                save_env_value(field.env_key, submitted)

    _apply_field_values(provider, values, lambda field: existing)

    path = _flat_json_path(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, existing, mode=0o600)


def _write_provider_honcho(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    """Persist submitted fields to Honcho's real config for the active host.

    Only keys present in ``values`` are touched, so a partial save (e.g. the
    inline panel) never clobbers fields owned by the full-config editor. Blank
    text clears a key so it falls back to the host/default mapping.
    """
    from plugins.memory.honcho.oauth import ACCESS_TOKEN_PREFIX, _config_refresh_lock
    from utils import atomic_json_write

    resolve_active_host, resolve_config_path, host_block_of = _honcho_resolvers()
    host = resolve_active_host()
    # Write the file reads resolve, or a save shadows it with a sparse copy.
    path = resolve_config_path()

    # OAuth rotation is single-use; an unlocked RMW here can revoke the grant.
    with _config_refresh_lock(path):
        cfg = _read_honcho_config(path)
        hosts = cfg.get("hosts")
        cfg["hosts"] = hosts = hosts if isinstance(hosts, dict) else {}
        # Update the block reads resolve (legacy dot-form included), never shadow it.
        existing = host_block_of(cfg, host)
        host_key = next((k for k, v in hosts.items() if v is existing), host) if existing else host
        host_block = hosts.setdefault(host_key, existing)

        for field in provider.fields:
            if not field.is_secret:
                continue
            submitted = (values.get(field.key) or "").strip()
            if not submitted:
                continue
            if field.env_key:
                save_env_value(field.env_key, submitted)
            # Persist where the client reads first; an OAuth token owns that slot.
            stored = host_block.get(field.key)
            if not (isinstance(stored, str) and stored.startswith(ACCESS_TOKEN_PREFIX)):
                host_block[field.key] = submitted

        _apply_field_values(provider, values, lambda field: host_block if field.scope == "host" else cfg)

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, cfg, mode=0o600)


def _command_result(
    *, kind: str, name: str, status: str, command: str = "",
    completed: Optional[subprocess.CompletedProcess] = None, error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "status": status,
        "command": command,
        "returncode": None if completed is None else completed.returncode,
        "stdout": "" if completed is None else _trim_setup_output(completed.stdout),
        "stderr": _trim_setup_output(error or ("" if completed is None else completed.stderr)),
    }


def _declared_provider_payload(provider: ProviderConfigSchema) -> Dict[str, Any]:
    fields: List[Dict[str, Any]] = []
    env = load_env()
    is_honcho = provider.storage == STORAGE_HONCHO_HOST_BLOCK

    if is_honcho:
        raw, host, host_block = _honcho_read_sources()

        def sources_for(field: ProviderField) -> tuple:
            return (host_block, raw) if field.scope == "host" else (raw,)
    else:
        host = ""
        data = _read_flat_json(provider)

        def sources_for(field: ProviderField) -> tuple:
            return (data,)

    for field in provider.fields:
        entry = _provider_field_entry(field)
        sources = sources_for(field)

        if field.is_secret:
            entry["value"] = ""  # secrets are write-only over the API
            entry["is_set"] = _declared_field_is_set(field, sources, env)
            fields.append(entry)
            continue

        native = _read_field(field, sources, env)
        if is_honcho and not field.placeholder and field.key in {"workspace", "aiPeer"}:
            # Blank fields surface the resolved host Honcho will actually use.
            entry["placeholder"] = host

        value = _serialize_field_value(field, native)
        if field.kind == "select" and value not in field.allowed_values():
            value = field.default
        entry["value"] = value
        # Presence, not truthiness — a stored False/0 is still "set".
        entry["is_set"] = native is not None if is_honcho else bool(value)
        fields.append(entry)

    return {"name": provider.name, "label": provider.label, "docs_url": provider.docs_url, "fields": fields}


def _stringify_submitted_values(values: Dict[str, Any]) -> Dict[str, str]:
    """The declared-schema path edits strings; the dashboard may send natives."""
    out: Dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, str):
            out[key] = value
        elif isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            out[key] = json.dumps(value)
        else:
            out[key] = str(value)
    return out


def _memory_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``config["memory"]`` as a dict, creating/replacing a non-dict value."""
    memory_config = config.get("memory")
    if not isinstance(memory_config, dict):
        memory_config = {}
        config["memory"] = memory_config
    return memory_config


def _update_memory_provider_config(provider: ProviderConfigSchema, values: Dict[str, str]) -> None:
    if provider.storage == STORAGE_HONCHO_HOST_BLOCK:
        _write_provider_honcho(provider, values)
    else:
        _write_provider_flat(provider, values)

    config = load_config()
    memory_config = _memory_section(config)
    if memory_config.get("provider") != provider.name:
        memory_config["provider"] = provider.name
        save_config(config)


def _memory_provider_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _install_memory_provider_pip_dependencies(dependencies: List[str]) -> List[Dict[str, Any]]:
    missing = [dep for dep in dependencies if not _dependency_importable(dep)]
    if not dependencies:
        return []
    if not missing:
        return [_command_result(kind="pip", name=", ".join(dependencies), status="already_installed")]

    # Route through the lazy-install pipeline rather than pip against
    # sys.executable: on hosted/immutable images the agent venv is sealed
    # read-only and installs must go to HERMES_LAZY_INSTALL_TARGET, which
    # install_specs also activates on sys.path so the recheck sees the packages.
    name = ", ".join(missing)
    try:
        from tools.lazy_deps import install_specs

        outcome = install_specs(missing, timeout=240)
    except Exception as exc:
        return [_command_result(kind="pip", name=name, status="failed", error=str(exc))]

    if outcome.blocked:
        return [_command_result(kind="pip", name=name, status="failed", command=outcome.command, error=outcome.reason)]

    return [
        _command_result(
            kind="pip",
            name=name,
            status="installed" if outcome.ok else "failed",
            command=outcome.command,
            completed=subprocess.CompletedProcess(
                args=outcome.command, returncode=0 if outcome.ok else 1, stdout=outcome.stdout,
                stderr=outcome.stderr,
            ),
        )
    ]


def _install_memory_provider_external_dependencies(
    dependencies: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dep in dependencies:
        name = dep.get("name") or "dependency"
        check_cmd = dep.get("check") or ""
        install_cmd = dep.get("install") or ""

        def _check(status_of) -> bool:
            """Run the check command; append its result; True when it passed."""
            try:
                check = _run_setup_command(shlex.split(check_cmd), display=check_cmd, timeout=20)
            except Exception as exc:
                results.append(_command_result(kind="external_check", name=name, status=status_of(None), command=check_cmd, error=str(exc)))
                return False
            ok = check.returncode == 0
            results.append(_command_result(kind="external_check", name=name, status=status_of(ok), command=check_cmd, completed=check))
            return ok

        if check_cmd:
            if _check(lambda ok: "already_installed" if ok else ("missing" if install_cmd else "failed")):
                continue
            if not install_cmd:
                continue

        if install_cmd:
            try:
                install = _run_setup_command(install_cmd, display=install_cmd, shell=True, timeout=300)
            except Exception as exc:
                results.append(_command_result(kind="external_install", name=name, status="failed", command=install_cmd, error=str(exc)))
                continue

            results.append(
                _command_result(
                    kind="external_install", name=name,
                    status="installed" if install.returncode == 0 else "failed",
                    command=install_cmd, completed=install,
                )
            )

            if check_cmd and install.returncode == 0:
                _check(lambda ok: "verified" if ok else "failed")

    return results


def _install_memory_provider_setup(name: str) -> Dict[str, Any]:
    provider = _load_memory_provider(name)
    manifest = _memory_provider_manifest(name)
    if provider is None and not manifest:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    setup = _memory_provider_setup_manifest(name)
    results = _install_memory_provider_pip_dependencies(setup["pip_dependencies"])
    results.extend(_install_memory_provider_external_dependencies(setup["external_dependencies"]))

    if not results:
        results.append(_command_result(kind="setup", name=name, status="no_declared_steps"))

    ok = all(result["status"] not in {"failed"} for result in results)
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    return {
        "ok": ok,
        "provider": name,
        "results": results,
        "status": statuses.get(name),
    }


def _public_memory_provider_field(field: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    return {
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


def _memory_provider_payload(name: str, provider: Any) -> Dict[str, Any]:
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
    memory_cfg = _memory_section(cfg)
    current = memory_cfg.get(name)
    if not isinstance(current, dict):
        current = {}
    current.update(values)
    memory_cfg[name] = current
    save_config(cfg)


def _write_memory_provider_config_values(
    name: str, provider: Any, values: Dict[str, Any],
) -> None:
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
    and gates which plugin manifest's setup commands run; a strict charset
    allowlist (no path separators, no dots) makes traversal impossible.
    """
    if not _MEMORY_PROVIDER_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")


@router.get("/api/memory/providers/{name}/config")
async def get_memory_provider_config(name: str, surface: Optional[str] = None, profile: Optional[str] = None):
    _require_valid_memory_provider_name(name)

    def _run():
        # Undeclared providers (e.g. builtin) have no config surface; an
        # empty schema makes the generic panel render nothing.
        if surface == "declared":
            declared = get_provider_config_schema(name)
            if declared is None:
                return {"name": name, "label": name, "docs_url": "", "fields": []}
            return _declared_provider_payload(declared)

        provider = _load_memory_provider(name)
        if provider is None:
            return {"name": name, "label": name, "fields": [], "setup": _memory_provider_setup_info(name)}
        return _memory_provider_payload(name, provider)

    return await scoped_to_thread(profile, _run)


@router.post("/api/memory/providers/{name}/setup")
async def setup_memory_provider(name: str, body: MemoryProviderSetupRequest):
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None and not _memory_provider_manifest(name):
        # No discoverable plugin directory -> no manifest that could declare
        # setup commands; refuse before the command-running path. (provider
        # may be None with a manifest present when its pip deps aren't
        # installed yet — that's the setup use case.)
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
    _require_valid_memory_provider_name(name)
    values = body.values or {}

    def _run():
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
        _memory_section(config)["provider"] = name
        save_config(config)
        _invalidate_plugins_hub_cache()
        return {"ok": True, "active": name}

    try:
        return await scoped_to_thread(profile, _run)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/memory/providers/%s/config failed", name)
        raise HTTPException(status_code=500, detail="Internal server error")
