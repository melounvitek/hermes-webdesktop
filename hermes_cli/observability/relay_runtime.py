"""Compatibility alias for the core Hermes Relay runtime."""

from __future__ import annotations

import sys

from agent import relay_runtime as _core_relay_runtime

sys.modules[__name__] = _core_relay_runtime
