"""Shared plumbing for the extracted dashboard routers.

Everything here is a thin wrapper over the late-binding seam in
:mod:`hermes_cli.web_deps`: web_server still owns the helpers and state, and
every access resolves at call time so ``monkeypatch.setattr(web_server, ...)``
stays authoritative.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable, Optional

from fastapi import HTTPException

from hermes_cli.web_deps import LateState, late

# Same logger the handlers used before extraction (identical logger object).
log = logging.getLogger("hermes_cli.web_server")

_profile_scope = late("_profile_scope")
_spawn_hermes_action = late("_spawn_hermes_action")
_profile_cli_args = late("_profile_cli_args")
# Config read-modify-write serialization for off-loop handlers (live lock —
# LateState supports ``with``-blocks).
_CONFIG_MUTATION_LOCK = LateState("_CONFIG_MUTATION_LOCK")


@contextlib.contextmanager
def config_write_scope(profile: Optional[str]):
    """Profile scope, then the config mutation lock — the write-path nesting
    every config-mutating handler uses."""
    with _profile_scope(profile):
        with _CONFIG_MUTATION_LOCK:
            yield


async def scoped_to_thread(profile: Optional[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` inside ``_profile_scope(profile)`` on a worker thread."""

    def _run():
        with _profile_scope(profile):
            return fn()

    return await asyncio.to_thread(_run)


@contextlib.contextmanager
def http_failure(log_msg: str, status: int, prefix: str):
    """Map unexpected exceptions to ``HTTPException(status, f"{prefix}: {exc}")``.

    ``HTTPException`` passes through untouched; anything else is logged with
    ``log_msg`` (full traceback) first.
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(log_msg)
        raise HTTPException(status_code=status, detail=f"{prefix}: {exc}")


def spawn_profile_action(
    profile: Optional[str], argv: list, name: str, *, log_msg: str, prefix: str
) -> dict:
    """Spawn a background ``hermes -p <profile> <argv>`` action; a spawn
    failure is logged and becomes ``500 "<prefix>: <exc>"``."""
    with http_failure(log_msg, 500, prefix):
        proc = _spawn_hermes_action(_profile_cli_args(profile) + argv, name)
    return {"ok": True, "pid": proc.pid, "name": name}


def require(value: Optional[str], detail: str) -> str:
    """Strip ``value``; 400 with ``detail`` when empty."""
    stripped = (value or "").strip()
    if not stripped:
        raise HTTPException(status_code=400, detail=detail)
    return stripped
