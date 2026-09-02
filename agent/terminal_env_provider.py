"""
Terminal Environment Provider ABC
=================================

Pluggable-backend interface for terminal execution environments (cloud
sandboxes, remote runners). Providers register via
:meth:`PluginContext.register_terminal_environment_provider`;
:func:`tools.terminal_tool._create_environment` consults the registry for any
``TERMINAL_ENV`` / ``terminal.backend`` value that is not a built-in backend.
Built-ins stay in-tree under ``tools/environments/``; this extension point
exists so third-party sandbox vendors do NOT have to live in core.

Classification contract
-----------------------
A backend participates in core policy decisions that were historically
frozensets of built-in names. Each is a declarative attribute so a new backend
cannot silently miss a classification site:

* ``is_remote`` — commands run off-host: suppresses host OS/home/cwd hints in
  the system prompt, the host Python probe, and remote-aware skill env handling.
* ``is_container`` — own filesystem rooted away from the host: container
  resource config is passed through, host-looking cwds are sanitized, file
  tools use container path resolution.
* ``skip_container_guards`` — sandbox is disposable enough to skip
  dangerous-command approval prompts. Defaults to ``is_container``; backends
  that can mount host paths should override to ``False``.
* ``cache_path_base`` — where auto-synced ``~/.hermes/cache`` files land inside
  the backend (``"~/.hermes"``, ``"/root/.hermes"``), or ``None`` when host
  paths remain correct.
* ``strip_env_keys`` — vendor credential env vars, stripped from every
  subprocess the agent spawns so a model-authored command can never read them.
* ``session_isolated_when_nonpersistent`` — non-persistent mode gives each
  session its own sandbox identity; opt in when a shared name would let two
  ephemeral runs attach to and destroy each other's sandbox.

:meth:`create_environment` returns any object satisfying the
:class:`tools.environments.base.BaseEnvironment` duck-typed interface
(``execute()``, ``cleanup()`` …); the registry does not isinstance-check it.
The factory stamps ``_hermes_backend_name`` on the result so file-path
resolution can identify plugin backends without class-name sniffing.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

from agent.provider_base import ProviderBase


class TerminalEnvironmentProvider(ProviderBase):
    """Abstract base class for a pluggable terminal execution backend.

    :attr:`name` is the ``terminal.backend`` / ``TERMINAL_ENV`` value
    (``[a-z0-9_]``); the registry rejects built-in backend names.
    """

    @property
    def description(self) -> str:
        """One-line description shown in backend pickers."""
        return f"Run commands in a {self.display_name} environment."

    # -- Classification flags (see module docstring) -----------------------

    is_remote: bool = True
    is_container: bool = True
    session_isolated_when_nonpersistent: bool = False

    @property
    def skip_container_guards(self) -> bool:
        """Whether dangerous-command approval prompts are skipped."""
        return self.is_container

    @property
    def cache_path_base(self) -> Optional[str]:
        """Base dir for synced Hermes cache files inside the backend."""
        return None

    @property
    def strip_env_keys(self) -> frozenset:
        """Backend-owned credential env var names to strip from subprocesses."""
        return frozenset()

    @property
    def env_description(self) -> str:
        """Prompt-builder fallback for where commands run when the live backend
        probe fails at system-prompt build time (e.g. ``"a Daytona workspace (Linux)"``)."""
        return f"a {self.display_name} environment (likely Linux)"

    # -- Availability / setup UX -------------------------------------------

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this backend can service commands. Cheap check only — must
        NOT make network calls; runs during requirement checks and UI paints."""

    def check_requirements(self, config: Dict[str, Any]) -> bool:
        """Full requirements check for :func:`check_terminal_requirements` with the
        merged terminal env config. Default defers to :meth:`is_available`; log
        actionable errors before returning False."""
        return self.is_available()

    def probe(self) -> Tuple[str, str]:
        """Dashboard picker health probe ``(status, detail)`` with status in
        ``ready`` / ``needs_setup`` / ``unavailable``. Must never raise; stay fast (<~2s)."""
        if self.is_available():
            return ("ready", "")
        return ("needs_setup", f"{self.display_name} is not configured.")

    def setup_instructions(self) -> List[str]:
        """Lines printed by ``hermes setup`` after this backend is selected. The
        wizard persists ``terminal.backend`` itself; interactive flows go in
        :meth:`post_setup`."""
        return []

    def post_setup(self) -> None:
        """Optional interactive hook run by ``hermes setup`` after selection
        (prompt for tokens, install SDKs). Default no-op."""

    def doctor_checks(self) -> List[Tuple[bool, str, str]]:
        """``hermes doctor`` rows ``(ok, label, detail)``; default reflects :meth:`is_available`."""
        ok = False
        try:
            ok = bool(self.is_available())
        except Exception:
            ok = False
        detail = "(configured)" if ok else "(not configured — see setup instructions)"
        return [(ok, f"{self.display_name} backend", detail)]

    # -- The factory -------------------------------------------------------

    @abc.abstractmethod
    def create_environment(
        self,
        *,
        cwd: str,
        timeout: int,
        task_id: str = "default",
        image: Optional[str] = None,
        container_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Create and return an execution environment (``BaseEnvironment`` duck type).

        MUST accept ``**kwargs`` and ignore unknown keys so the factory signature
        can evolve without breaking older plugins. ``task_id`` keys environment
        reuse/persistence; ``container_config`` carries ``container_cpu`` /
        ``container_memory`` / ``container_disk`` / ``container_persistent`` when
        :attr:`is_container` is True.
        """
