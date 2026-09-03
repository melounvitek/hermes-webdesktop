"""Local-embedded Hindsight runtime: import probe, install hint, the per-profile env
file the standalone ``hindsight-embed`` daemon consumes, and the health-grace export."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from agent.secret_scope import get_secret

from .settings import _DEFAULT_IDLE_TIMEOUT, _daemon_llm_provider, _parse_int_setting

logger = logging.getLogger(__name__.rpartition(".")[0])

# Read by hindsight_embed.daemon_embed_manager AT IMPORT TIME: how long to wait
# for a slow /health before killing the daemon as stale. Busy hosts exceed the
# upstream 2s check and get needlessly restarted, so it's plugin config.
_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"

# Stale embedded-daemon connection markers (client recreated, operation retried once).
_RETRIABLE_CONNECTION_MARKERS = (
    "cannot connect to host",
    "connection refused",
    "connect call failed",
    "clientconnectorerror",
)


def _export_port_health_grace_timeout(config: dict[str, Any]) -> None:
    """Export the daemon health grace timeout BEFORE ``daemon_embed_manager`` is
    imported. Only when configured; ``setdefault`` so an explicit env override wins."""
    raw = config.get("port_health_grace_timeout")
    if raw is None or raw == "":
        return
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = None
    if seconds is None or seconds < 0:
        logger.warning("%s Hindsight port_health_grace_timeout %r; ignoring.",
                       "Invalid" if seconds is None else "Negative", raw)
        return
    os.environ.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))


def _check_local_runtime() -> tuple[bool, str | None]:
    """Whether the local embedded stack imports cleanly (older CPUs: NumPy can raise
    at import, so Hermes degrades instead of retrying a broken backend).
    ``sentence_transformers`` is probed too: ``hindsight`` imports fine with a broken
    embedding stack, and the daemon would then abort on every retain/recall."""
    try:
        for module in ("hindsight", "hindsight_embed.daemon_embed_manager", "sentence_transformers"):
            importlib.import_module(module)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _local_runtime_hint(reason: str | None) -> str:
    """Install guidance when the local_embedded runtime is missing: ``plugin.yaml``
    declares only ``hindsight-client``, so a hand-written config, the legacy
    ``"mode": "local"`` alias or a restored backup hits ``No module named 'hindsight'``."""
    text = (reason or "").lower()
    if "no module named" in text and ("hindsight'" in text or 'hindsight"' in text
                                      or "hindsight_embed" in text):
        return (
            f" Install the embedded runtime with: uv pip install --python "
            f"{sys.executable} hindsight-all — or run 'hermes memory setup'. "
            "(local_embedded needs the 'hindsight-all' package, which provides the "
            "top-level 'hindsight' module; 'hindsight-client' alone only covers "
            "cloud / local_external.)"
        )
    return ""


def _load_simple_env(path) -> dict[str, str]:
    """Parse a KEY=VALUE env file (comments/blank lines ignored). utf-8-sig: also used
    on the Hermes .env during post_setup, where a Notepad BOM would stick to the first key."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _embedded_profile_env_path(config: dict[str, Any]) -> Path:
    profile = str(config.get("profile", "hermes") or "hermes")
    return Path.home() / ".hindsight" / "profiles" / f"{profile}.env"


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env that standalone hindsight-embed consumes."""
    if llm_api_key is None:
        llm_api_key = (
            config.get("llmApiKey")
            or config.get("llm_api_key")
            or get_secret("HINDSIGHT_LLM_API_KEY", "")
        )
    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(_daemon_llm_provider(config.get("llm_provider", ""))),
        "HINDSIGHT_API_LLM_API_KEY": str(llm_api_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(config.get("llm_model", "")),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    base_url = config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")
    if base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(base_url)
    idle_timeout = config.get("idle_timeout")
    if idle_timeout is None:
        idle_timeout = os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(
            _parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT)
        )
    return env_values


def _secure_write_profile_env(profile_env: Path, content: str) -> None:
    """Create/overwrite *profile_env* owner-only (0600); a pre-existing file is
    tightened BEFORE the plaintext LLM API key is written."""
    if profile_env.exists():
        try:
            os.chmod(profile_env, 0o600)
        except OSError:
            pass
    fd = os.open(str(profile_env), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)


def _validate_profile_env_permissions(profile_env: Path) -> None:
    """Post-write check: owner-only on POSIX (Windows ACLs aren't mode bits; skipped)."""
    if os.name != "posix":
        return
    import stat

    mode = stat.S_IMODE(profile_env.stat().st_mode)
    if mode != 0o600:
        try:
            os.chmod(profile_env, 0o600)
        except OSError:
            pass
        if stat.S_IMODE(profile_env.stat().st_mode) != 0o600:
            raise PermissionError(
                f"Embedded Hindsight profile environment is not owner-only: {profile_env}"
            )


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> Path:
    """Write the profile env file; never leave a plaintext key in a file whose
    permissions could not be verified."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    content = "".join(f"{key}={value}\n" for key, value in env_values.items())
    try:
        _secure_write_profile_env(profile_env, content)
        _validate_profile_env_permissions(profile_env)
    except BaseException:
        try:
            profile_env.unlink()
        except OSError:
            pass
        raise
    return profile_env
