"""Local-embedded Hindsight runtime helpers: import probe, install hint, the
per-profile env file the standalone ``hindsight-embed`` daemon consumes, and the
daemon health-grace env export."""

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

# Read by hindsight_embed.daemon_embed_manager AT IMPORT TIME (module-level
# constant): how long to wait for a slow /health before declaring the daemon
# stale and killing it. On resource-contended hosts a busy daemon can exceed
# the upstream 2s check and get needlessly restarted, so it's plugin config.
_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"

# Markers of a stale embedded-daemon connection (the client is recreated and
# the operation retried once).
_RETRIABLE_CONNECTION_MARKERS = (
    "cannot connect to host",
    "connection refused",
    "connect call failed",
    "clientconnectorerror",
)


def _export_port_health_grace_timeout(config: dict[str, Any]) -> None:
    """Export the daemon health grace timeout to the process env.

    Must run BEFORE ``hindsight_embed.daemon_embed_manager`` is imported. Only
    set when the user configured a value; ``setdefault`` so an explicit env
    override always wins.
    """
    raw = config.get("port_health_grace_timeout")
    if raw is None or raw == "":
        return
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid Hindsight port_health_grace_timeout %r; ignoring.", raw)
        return
    if seconds < 0:
        logger.warning("Negative Hindsight port_health_grace_timeout %r; ignoring.", raw)
        return
    os.environ.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))


def _check_local_runtime() -> tuple[bool, str | None]:
    """Return whether the local embedded Hindsight stack imports cleanly.

    On older CPUs NumPy can raise at import before the daemon starts; report
    "unavailable" so Hermes degrades instead of retrying a broken backend.
    ``sentence_transformers`` is imported too: ``hindsight``/``hindsight_embed``
    import fine even when the embedding stack is broken, and without this the
    probe (and ``hermes memory status``) would stay green while the daemon
    aborts on every retain/recall.
    """
    try:
        importlib.import_module("hindsight")
        importlib.import_module("hindsight_embed.daemon_embed_manager")
        importlib.import_module("sentence_transformers")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _local_runtime_hint(reason: str | None) -> str:
    """Install guidance when the local_embedded runtime is missing.

    The top-level ``hindsight`` module ships only with ``hindsight-all``;
    ``plugin.yaml`` declares just ``hindsight-client`` (cloud/local_external),
    so a hand-written config, the legacy ``"mode": "local"`` alias, or a
    restored backup hits ``ModuleNotFoundError: No module named 'hindsight'``.
    """
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
    """Parse a KEY=VALUE env file, ignoring comments and blank lines.

    utf-8-sig, not utf-8: also used on the Hermes .env during post_setup, where
    a Notepad BOM would otherwise stick to the first key.
    """
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
    """Create/overwrite *profile_env* owner-only (0600). The file carries the
    daemon's plaintext LLM API key, so a pre-existing file is tightened BEFORE
    the new secret bytes are written."""
    if profile_env.exists():
        try:
            os.chmod(profile_env, 0o600)
        except OSError:
            pass
    fd = os.open(str(profile_env), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)


def _validate_profile_env_permissions(profile_env: Path) -> None:
    """Post-write check: the secret file must be owner-only on POSIX (Windows
    ACLs aren't modelled by mode bits, so skipped there)."""
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
    """Write the profile env file; never leave a plaintext key behind in a file
    whose permissions could not be verified."""
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
