"""Tolerant env-var knob parsing shared by the gateway entry points.

A bare ``float(os.environ[...])`` would raise at import time on a typo
(``HERMES_SLASH_WATCHDOG_POLL_S=2s``) and kill a worker before it serves a
single command; these fall back to ``default`` on absent/empty/malformed values.
"""

from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
