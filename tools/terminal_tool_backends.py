"""Execution-environment backends for the terminal tool: per-backend builders (local/docker/singularity/modal/daytona/vercel/ssh/plugin), the config-to-kwargs shapers, and the per-backend requirement checkers, both routed by dispatch table.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import logging
import importlib.util
import inspect
import shutil
import subprocess
from typing import Any, Dict, Optional
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.managed_modal import (
    ManagedModalEnvironment as _ManagedModalEnvironment,
)
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.singularity import (
    SingularityEnvironment as _SingularityEnvironment,
)
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.tool_backend_helpers import (
    has_direct_modal_credentials,
    nous_tool_gateway_unavailable_message,
    resolve_modal_backend_state,
)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"


_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = (config.get("vercel_runtime") or "").strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error(
            "Vercel Sandbox runtime %r is not supported. "
            "Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
            runtime,
            supported,
        )
        return False

    disk = config.get("container_disk", 51200)
    if disk not in {0, 51200}:
        logger.error(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
            "Use the default shared setting (51200 MB).",
            disk,
        )
        return False

    if importlib.util.find_spec("vercel") is None:
        logger.error(
            "vercel is required for the Vercel Sandbox terminal backend: pip install vercel"
        )
        return False

    from agent.secret_scope import get_secret

    has_oidc = bool(get_secret("VERCEL_OIDC_TOKEN"))
    has_token = bool(get_secret("VERCEL_TOKEN"))
    has_project = bool(get_secret("VERCEL_PROJECT_ID"))
    has_team = bool(get_secret("VERCEL_TEAM_ID"))

    if has_oidc:
        return True

    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error(
            "Vercel Sandbox backend selected with token auth, but "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all "
            "be set together. VERCEL_OIDC_TOKEN is supported for one-off "
            "local development only."
        )
        return False

    logger.error(
        "Vercel Sandbox backend selected but no supported auth configuration "
        "was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID "
        "for normal use. VERCEL_OIDC_TOKEN is supported for one-off local "
        "development only."
    )
    return False


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    from tools.terminal_tool import is_managed_tool_gateway_ready
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _ssh_config_from_config(config: Dict[str, Any]) -> dict:
    """``ssh_config`` for :func:`_create_environment` (shared by terminal_tool
    and the lazy :func:`ensure_task_env` bring-up)."""
    return {
        "host": config.get("ssh_host", ""),
        "user": config.get("ssh_user", ""),
        "port": config.get("ssh_port", 22),
        "key": config.get("ssh_key", ""),
        "persistent": config.get("ssh_persistent", False),
    }


def _container_config_from_config(config: Dict[str, Any]) -> dict:
    """``container_config`` for :func:`_create_environment` (shared by
    terminal_tool and the lazy :func:`ensure_task_env` bring-up)."""
    return {
        "container_cpu": config.get("container_cpu", 1),
        "container_memory": config.get("container_memory", 5120),
        "container_disk": config.get("container_disk", 51200),
        "container_persistent": config.get("container_persistent", True),
        "modal_mode": config.get("modal_mode", "auto"),
        "vercel_runtime": config.get("vercel_runtime", ""),
        "docker_volumes": config.get("docker_volumes", []),
        "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
        "docker_forward_env": config.get("docker_forward_env", []),
        "docker_env": config.get("docker_env", {}),
        "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
        "docker_extra_args": config.get("docker_extra_args", []),
        "docker_shm_size": config.get("docker_shm_size", "1g"),
        "docker_network": config.get("docker_network", True),
        "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
        "docker_shared_container_key": config.get("docker_shared_container_key", ""),
        "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
    }


def _resources(cc: Dict[str, Any]) -> dict:
    """Common sandbox resource kwargs (cpu/memory in MB/disk in MB/persistence)."""
    return {
        "cpu": cc.get("container_cpu", 1),
        "memory": cc.get("container_memory", 5120),
        "disk": cc.get("container_disk", 51200),
        "persistent_filesystem": cc.get("container_persistent", True),
    }


def _build_local_env(*, cwd, timeout, **_):
    return _LocalEnvironment(cwd=cwd, timeout=timeout)


def _build_docker_env(*, image, cwd, timeout, cc, task_id, host_cwd, **_):
    # One-shot orphan reaper for labeled containers left behind by prior
    # Hermes processes that died before atexit (SIGKILL / OOM / closed
    # terminal); once per process, ``terminal.docker_orphan_reaper: false``
    # disables it.
    from tools.terminal_tool import _DockerEnvironment, _docker_session_isolation_enabled, _has_isolation_overrides, _maybe_reap_docker_orphans
    _maybe_reap_docker_orphans(cc)
    # Per-session container isolation: a session-keyed container must not
    # outlive its session, so cross-process reuse/persist is disabled for it —
    # cleanup_vm()/the idle reaper stop+rm it. The shared "default" container
    # and RL/benchmark override sandboxes keep their existing lifecycle.
    session_scoped = (
        _docker_session_isolation_enabled()
        and task_id != "default"
        and not _has_isolation_overrides(task_id)
    )
    docker_env_obj = _DockerEnvironment(
        image=image, cwd=cwd, timeout=timeout, task_id=task_id,
        **_resources(cc),
        volumes=cc.get("docker_volumes", []),
        host_cwd=host_cwd,
        auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
        forward_env=cc.get("docker_forward_env", []),
        env=cc.get("docker_env", {}),
        run_as_host_user=cc.get("docker_run_as_host_user", False),
        network=cc.get("docker_network", True),
        extra_args=cc.get("docker_extra_args", []),
        persist_across_processes=(
            False if session_scoped
            else cc.get("docker_persist_across_processes", True)
        ),
        shared_container_key=cc.get("docker_shared_container_key", ""),
        shm_size=cc.get("docker_shm_size", "1g"),
    )
    # Marker read by is_persistent_env(): a session-scoped container survives
    # BETWEEN turns (skip per-turn teardown) but is removed at session close /
    # idle timeout. Guarded: test doubles may not accept attributes.
    if session_scoped:
        try:
            docker_env_obj._session_scoped = True
        except AttributeError:
            pass
    return docker_env_obj


def _build_singularity_env(*, image, cwd, timeout, cc, task_id, **_):
    return _SingularityEnvironment(
        image=image, cwd=cwd, timeout=timeout, task_id=task_id, **_resources(cc),
    )


def _build_modal_env(*, image, cwd, timeout, cc, task_id, **_):
    from tools.terminal_tool import managed_nous_tools_enabled
    res = _resources(cc)
    persistent = res["persistent_filesystem"]
    sandbox_kwargs = {k: res[k] for k in ("cpu", "memory") if res[k] > 0}
    if res["disk"] > 0:
        try:
            import modal
            if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                sandbox_kwargs["ephemeral_disk"] = res["disk"]
        except Exception:
            pass

    modal_state = _get_modal_backend_state(cc.get("modal_mode"))

    if modal_state["selected_backend"] == "managed":
        return _ManagedModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )

    if modal_state["selected_backend"] != "direct":
        if modal_state["managed_mode_blocked"]:
            raise ValueError(
                "Modal backend is configured for managed mode, but "
                "Nous Tool Gateway access is not currently available and no direct "
                "Modal credentials/config were found. "
                + nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                )
                + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
            )
        if modal_state["mode"] == "managed":
            raise ValueError(
                "Modal backend is configured for managed mode, but the managed tool gateway is unavailable. "
                + nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                )
            )
        if modal_state["mode"] == "direct":
            raise ValueError(
                "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."
            )
        message = "Modal backend selected but no direct Modal credentials/config was found."
        if managed_nous_tools_enabled():
            message = (
                "Modal backend selected but no direct Modal credentials/config or managed tool gateway was found."
            )
        raise ValueError(message)

    return _ModalEnvironment(
        image=image, cwd=cwd, timeout=timeout,
        modal_sandbox_kwargs=sandbox_kwargs,
        persistent_filesystem=persistent, task_id=task_id,
    )


def _build_daytona_env(*, image, cwd, timeout, cc, task_id, **_):
    # Lazy import so daytona SDK is only required when backend is selected.
    from tools.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment
    res = _resources(cc)
    res["cpu"] = int(res["cpu"])
    return _DaytonaEnvironment(image=image, cwd=cwd, timeout=timeout, task_id=task_id, **res)


def _build_vercel_env(*, cwd, timeout, cc, task_id, **_):
    from tools.environments.vercel_sandbox import (
        VercelSandboxEnvironment as _VercelSandboxEnvironment,
    )
    return _VercelSandboxEnvironment(
        runtime=cc.get("vercel_runtime") or None,
        cwd=cwd, timeout=timeout, task_id=task_id, **_resources(cc),
    )


def _build_ssh_env(*, cwd, timeout, ssh_config, **_):
    if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
        raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
    return _SSHEnvironment(
        host=ssh_config["host"],
        user=ssh_config["user"],
        port=ssh_config.get("port", 22),
        key_path=ssh_config.get("key", ""),
        cwd=cwd,
        timeout=timeout,
    )


def _build_plugin_env(*, env_type, image, cwd, timeout, cc, task_id, **_):
    from tools.terminal_tool import _get_plugin_env_provider
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        env_obj = provider.create_environment(
            cwd=cwd, timeout=timeout, task_id=task_id,
            image=image, container_config=cc,
        )
        # Stamp the backend name so path-resolution and progress surfaces
        # can identify plugin backends without class-name sniffing.
        try:
            env_obj._hermes_backend_name = provider.name.strip().lower()
        except AttributeError:
            pass  # test doubles may reject attributes
        return env_obj
    try:
        from agent.terminal_env_registry import plugin_backend_names

        plugin_names = plugin_backend_names()
    except Exception:
        plugin_names = []
    extra = (
        ", " + ", ".join(f"'{n}'" for n in plugin_names) if plugin_names else ""
    )
    raise ValueError(
        f"Unknown environment type: {env_type}. Use 'local', 'docker', "
        f"'singularity', 'modal', 'daytona', 'vercel_sandbox', 'ssh'{extra}"
    )


# Built-in backend -> builder. Anything else is looked up in the plugin registry.
_ENV_BUILDERS = {
    "local": _build_local_env,
    "docker": _build_docker_env,
    "singularity": _build_singularity_env,
    "modal": _build_modal_env,
    "daytona": _build_daytona_env,
    "vercel_sandbox": _build_vercel_env,
    "ssh": _build_ssh_env,
}


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: Optional[str] = None):
    """Create an execution environment (instance with ``execute()``) for *env_type*.

    ``image`` is ignored for local/ssh/vercel; ``container_config`` carries the
    container_*/docker_* resource keys; ``host_cwd`` is the host directory to
    bind into Docker when cwd mounting is explicitly enabled. Unknown
    ``env_type`` values fall through to plugin-registered backends.
    """
    builder = _ENV_BUILDERS.get(env_type, _build_plugin_env)
    return builder(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        cc=container_config or {}, task_id=task_id,
        ssh_config=ssh_config, host_cwd=host_cwd,
    )


def _check_docker_requirements(config: Dict[str, Any]) -> bool:
    from tools.environments.docker import find_docker
    docker = find_docker()
    if not docker:
        logger.error("Docker executable not found in PATH or common install locations")
        return False
    result = subprocess.run([docker, "version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
    return result.returncode == 0


def _check_singularity_requirements(config: Dict[str, Any]) -> bool:
    executable = shutil.which("apptainer") or shutil.which("singularity")
    if executable:
        result = subprocess.run([executable, "--version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
        return result.returncode == 0
    return False


def _check_ssh_requirements(config: Dict[str, Any]) -> bool:
    if not config.get("ssh_host") or not config.get("ssh_user"):
        logger.error(
            "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
            "are not both set. Configure both or switch TERMINAL_ENV to 'local'."
        )
        return False
    return True


def _check_modal_requirements(config: Dict[str, Any]) -> bool:
    from tools.terminal_tool import managed_nous_tools_enabled
    modal_state = _get_modal_backend_state(config.get("modal_mode"))
    if modal_state["selected_backend"] == "managed":
        return True

    if modal_state["selected_backend"] != "direct":
        if modal_state["managed_mode_blocked"]:
            logger.error(
                "Modal backend selected with TERMINAL_MODAL_MODE=managed, but "
                "Nous Tool Gateway access is not currently available and no direct "
                "Modal credentials/config were found. %s Choose "
                "TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.",
                nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                ),
            )
            return False
        if modal_state["mode"] == "managed":
            logger.error(
                "Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed "
                "tool gateway is unavailable. %s",
                nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                ),
            )
            return False
        elif modal_state["mode"] == "direct":
            if managed_nous_tools_enabled():
                logger.error(
                    "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                    "Modal credentials/config were found. Configure Modal or choose "
                    "TERMINAL_MODAL_MODE=managed/auto."
                )
            else:
                logger.error(
                    "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                    "Modal credentials/config were found. Configure Modal or choose "
                    "TERMINAL_MODAL_MODE=auto."
                )
            return False
        else:
            if managed_nous_tools_enabled():
                logger.error(
                    "Modal backend selected but no direct Modal credentials/config or managed "
                    "tool gateway was found. Configure Modal, set up the managed gateway, "
                    "or choose a different TERMINAL_ENV."
                )
            else:
                logger.error(
                    "Modal backend selected but no direct Modal credentials/config was found. "
                    "Configure Modal or choose a different TERMINAL_ENV."
                )
            return False

    if importlib.util.find_spec("modal") is None:
        logger.error("modal is required for direct modal terminal backend: pip install modal")
        return False

    return True


def _check_daytona_requirements(config: Dict[str, Any]) -> bool:
    from daytona import Daytona  # noqa: F401 — SDK presence check
    from agent.secret_scope import get_secret
    return get_secret("DAYTONA_API_KEY") is not None


def _check_plugin_requirements(config: Dict[str, Any]) -> bool:
    from tools.terminal_tool import _get_plugin_env_provider
    env_type = config["env_type"]
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        return bool(provider.check_requirements(config))
    logger.error(
        "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
        "modal, daytona, vercel_sandbox, ssh, or a plugin-registered backend.",
        env_type,
    )
    return False


# Built-in backend -> requirements checker; unknown backends go to the plugin registry.
_REQUIREMENT_CHECKERS = {
    "local": lambda config: True,
    "docker": _check_docker_requirements,
    "singularity": _check_singularity_requirements,
    "ssh": _check_ssh_requirements,
    "modal": _check_modal_requirements,
    "vercel_sandbox": _check_vercel_sandbox_requirements,
    "daytona": _check_daytona_requirements,
}
