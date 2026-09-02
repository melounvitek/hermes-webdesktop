"""Skill readiness: required env vars, secret capture, and setup notes.

Split out of ``tools.skills_tool``; every name is re-imported there so
``from tools.skills_tool import X`` / ``patch("tools.skills_tool.X")`` keep
working. Module state (``_secret_capture_callback``, ``load_env``) stays in
``tools.skills_tool`` and is read lazily at call time so test patches on the
origin module are honored.
"""

import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Tuple

from hermes_constants import display_hermes_home
from utils import env_var_enabled

logger = logging.getLogger("tools.skills_tool")

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REMOTE_ENV_BACKENDS = frozenset({"docker", "singularity", "modal", "ssh", "daytona", "vercel_sandbox"})


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def _is_remote_env_backend(backend: str) -> bool:
    """Built-in remote backends plus plugin backends declaring is_remote."""
    if backend in _REMOTE_ENV_BACKENDS:
        return True
    if not backend or backend == "local":
        return False
    try:
        from agent.terminal_env_registry import provider_flag

        return bool(provider_flag(backend, "is_remote", False))
    except Exception:
        return False


def _normalize_prerequisite_values(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(frontmatter: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _as_dict_list(raw: Any) -> list:
    """Accept a single mapping or a list; anything else is treated as empty."""
    if isinstance(raw, dict):
        return [raw]
    return raw if isinstance(raw, list) else []


def _clean_str(value: Any) -> str | None:
    """Stripped string when *value* is a non-blank str, else None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}
    collect_secrets: List[Dict[str, Any]] = []
    for item in _as_dict_list(setup.get("collect_secrets")):
        if not isinstance(item, dict):
            continue
        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue
        entry: Dict[str, Any] = {
            "env_var": env_var,
            "prompt": str(item.get("prompt") or f"Enter value for {env_var}").strip(),
            "secret": bool(item.get("secret", True)),
        }
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)
    return {"help": _clean_str(setup.get("help")), "collect_secrets": collect_secrets}


def _get_required_environment_variables(
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Merge required_environment_variables, setup.collect_secrets and legacy
    prerequisites.env_vars into one deduped, validated list (first entry wins)."""
    setup = _normalize_setup_metadata(frontmatter)
    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen or not _ENV_VAR_NAME_RE.match(env_name):
            return
        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }
        help_text = _clean_str(
            entry.get("help") or entry.get("provider_url") or entry.get("url") or setup.get("help")
        )
        if help_text:
            normalized["help"] = help_text
        required_for = _clean_str(entry.get("required_for"))
        if required_for:
            normalized["required_for"] = required_for
        if entry.get("optional"):
            normalized["optional"] = True
        seen.add(env_name)
        required.append(normalized)

    for item in _as_dict_list(frontmatter.get("required_environment_variables")):
        if isinstance(item, str):
            _append_required({"name": item})
        elif isinstance(item, dict):
            _append_required(item)
    for item in setup["collect_secrets"]:
        _append_required({
            "name": item.get("env_var"),
            "prompt": item.get("prompt"),
            "help": item.get("provider_url") or setup.get("help"),
        })
    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})
    return required


def _capture_result(missing_names, setup_skipped=False, gateway_setup_hint=None):
    return {
        "missing_names": missing_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": gateway_setup_hint,
    }


def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prompt for missing secrets via the registered capture callback (if any)."""
    from tools import skills_tool as _st

    if not missing_entries:
        return _capture_result([])
    missing_names = [entry["name"] for entry in missing_entries]
    # Most gateway surfaces (messaging platforms) can't prompt for a secret, so
    # they short-circuit to the "unsupported" hint. Interactive gateway surfaces
    # (desktop app / TUI) set HERMES_INTERACTIVE — the same flag tools/approval.py
    # uses — and register a callback routing to a secure secret.request overlay,
    # so they fall through and actually prompt.
    if _is_gateway_surface() and not env_var_enabled("HERMES_INTERACTIVE"):
        return _capture_result(missing_names, gateway_setup_hint=_gateway_setup_hint())
    callback = _st._secret_capture_callback
    if callback is None:
        return _capture_result(missing_names)

    setup_skipped = False
    remaining_names: List[str] = []
    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        for k in ("help", "required_for"):
            if entry.get(k):
                metadata[k] = entry[k]
        try:
            callback_result = callback(entry["name"], entry["prompt"], metadata)
        except Exception:
            logger.warning(f"Secret capture callback failed for {entry['name']}", exc_info=True)
            callback_result = {"success": False, "stored_as": entry["name"], "validated": False, "skipped": True}
        ok = isinstance(callback_result, dict)
        if ok and callback_result.get("success") and not callback_result.get("skipped"):
            continue
        setup_skipped = True
        remaining_names.append(entry["name"])
    return _capture_result(remaining_names, setup_skipped)


def _is_gateway_surface() -> bool:
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_SESSION_PLATFORM"))


def _get_terminal_backend_name() -> str:
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _env_snapshot_or_load(env_snapshot):
    if env_snapshot is None:
        from tools import skills_tool as _st

        return _st.load_env()
    return env_snapshot


def _is_env_var_persisted(var_name: str, env_snapshot: Dict[str, str] | None = None) -> bool:
    env_snapshot = _env_snapshot_or_load(env_snapshot)
    if var_name in env_snapshot:
        return bool(env_snapshot.get(var_name))
    return bool(os.getenv(var_name))


def _remaining_required_environment_names(
    required_env_vars: List[Dict[str, Any]],
    capture_result: Dict[str, Any],
    *,
    env_snapshot: Dict[str, str] | None = None,
) -> List[str]:
    missing_names = set(capture_result["missing_names"])
    env_snapshot = _env_snapshot_or_load(env_snapshot)
    return [
        e["name"]
        for e in required_env_vars
        if not e.get("optional")
        and (e["name"] in missing_names or not _is_env_var_persisted(e["name"], env_snapshot))
    ]


def _gateway_setup_hint() -> str:
    try:
        from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE

        return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
    except Exception:
        return f"Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to {display_hermes_home()}/.env manually."


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    if readiness_status != SkillReadinessStatus.SETUP_NEEDED:
        return None
    note = f"Setup needed before using this skill: missing {', '.join(missing) if missing else 'required prerequisites'}."
    return f"{note} {setup_help}" if setup_help else note
