"""Synthetic GIL-heavy turn driver for the AC-4 isolation certify harness.

The regime under test is interpreter-wide GIL starvation: concurrent heavy agent
turns run compute in threads of the SERVING process and starve the event loop
that flushes WebSocket frames (loop thread parked in ``take_gil`` — NOT blocked
on I/O).  To certify the isolation fix without spending real tokens, the turn
driver must reproduce THAT: sustained pure-Python CPU holding the GIL for the
turn's duration.  A network/sleep stub is WRONG — it releases the GIL during I/O
and never reproduces ``take_gil`` contention, so a green off it is fake.

This module is a **test seam**: dead unless ``HERMES_ISO_CERTIFY_SYNTH_TURN=1``.
When armed, ``tui_gateway.server._make_agent`` returns a
:class:`SyntheticHeavyAgent` instead of a real ``AIAgent``.  Both the in-process
``_pool`` path (isolation OFF) and the compute-host child path (isolation ON)
build through ``_make_agent``, so the isolation boundary is the only variable.

Per-turn intensity (wall duration, CPU chunk size, delta cadence, token
accounting) rides in the prompt text as a JSON object; any other prompt falls
back to env / built-in defaults.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Optional

from tui_gateway._env import env_float as _env_float, env_int as _env_int


def synth_turn_armed() -> bool:
    """True when the synthetic-turn test seam is armed via env."""
    return os.environ.get("HERMES_ISO_CERTIFY_SYNTH_TURN") == "1"


class SyntheticHeavyAgent:
    """An AIAgent-shaped object whose turn is a GIL-holding CPU burn.

    Presents only the surface ``tui_gateway.server``'s turn path and status
    helpers read (``run_conversation``/``interrupt``/``clear_interrupt`` plus the
    ``model``/``provider``/``session_*`` attributes consumed by ``_get_usage`` and
    ``_session_info``).  Never opens a socket or spawns a subprocess.
    """

    def __init__(self, session_id: str, *, model: str = "synthetic-heavy") -> None:
        self.session_id = session_id
        self.model = model
        self.provider = "synthetic"
        self.api_mode = "chat_completions"
        self.base_url = ""
        self.api_key = ""
        self.platform = ""
        self.tools: list[Any] = []
        self.reasoning_config: dict | None = None
        self.service_tier: str | None = None
        self.context_compressor = None
        self._config_context_length = 200_000
        self._cached_system_prompt = ""
        # Cumulative session counters (read by _get_usage → status bar).
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.history: list[dict[str, str]] = []
        self._interrupt = threading.Event()

    # ── interrupt contract (mirrors AIAgent) ───────────────────────────
    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def interrupt(self) -> None:
        self._interrupt.set()

    def _has_stream_consumers(self) -> bool:  # defensive; not used by our loop
        return True

    def close(self) -> None:
        """No-op teardown (session lifecycle calls agent.close() on some paths)."""
        self._interrupt.set()

    @staticmethod
    def _parse_spec(message: Any) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if isinstance(message, str) and message.strip().startswith("{"):
            try:
                parsed = json.loads(message.strip())
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                spec = parsed
        return {
            # Wall-clock seconds of GIL-holding compute.
            "duration_s": float(spec.get("duration_s", _env_float("HERMES_ISO_CERTIFY_DURATION_S", 8.0))),
            # Pure-Python ops per interrupt-check chunk: small enough that an
            # interrupt lands within ms, large enough to stay hot on the GIL.
            "chunk": int(spec.get("chunk", _env_int("HERMES_ISO_CERTIFY_CHUNK", 20_000))),
            # Streamed-delta cadence: each delta is a loop wakeup marshalling a frame.
            "delta_interval_s": float(spec.get("delta_interval_s", _env_float("HERMES_ISO_CERTIFY_DELTA_S", 0.05))),
            # Notional output tokens per delta (drives the 100K+-token heavy-turn proxy).
            "tokens_per_delta": int(spec.get("tokens_per_delta", _env_int("HERMES_ISO_CERTIFY_TPD", 512))),
            # Optional per-chunk sleep for a mixed regime (0 = pure burn).  --dry-run
            # uses a short duration, NOT a sleep, so it still exercises the real seam.
            "sleep_s": float(spec.get("sleep_s", 0.0)),
        }

    def run_conversation(
        self,
        message: Any,
        *,
        conversation_history: Optional[list[dict[str, str]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        task_id: Optional[str] = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        spec = self._parse_spec(message)
        duration = max(0.0, spec["duration_s"])
        chunk = max(1, spec["chunk"])
        interval = max(0.001, spec["delta_interval_s"])
        tokens_per_delta = max(0, spec["tokens_per_delta"])
        sleep_s = max(0.0, spec["sleep_s"])

        base_history = list(conversation_history if conversation_history is not None else self.history)
        start = last_delta = time.monotonic()
        acc = deltas = 0
        interrupted = False

        while True:
            if self._interrupt.is_set():
                interrupted = True
                break
            now = time.monotonic()
            if now - start >= duration:
                break
            # A tight integer loop never releases the GIL — the exact contention
            # that starves the serving loop.
            for _ in range(chunk):
                acc = (acc * 1_000_003 + 12_345) & 0xFFFFFFFFFFFFFFFF
            if sleep_s:
                time.sleep(sleep_s)
            if now - last_delta >= interval:
                deltas += 1
                self.session_output_tokens += tokens_per_delta
                self.session_completion_tokens += tokens_per_delta
                self.session_total_tokens += tokens_per_delta
                if stream_callback is not None:
                    stream_callback(f"synthtok-{deltas:05d} ")
                last_delta = now

        self.session_api_calls += 1
        # Fold the checksum into the reply so the loop can't be eliminated and
        # the turn is deterministic and inspectable.
        final = (
            f"[synthetic heavy turn] deltas={deltas} "
            f"out_tokens={self.session_output_tokens} "
            f"interrupted={interrupted} checksum={acc & 0xFFFF:04x}"
        )
        messages = [
            *base_history,
            {"role": "user", "content": str(message)[:200]},
            {"role": "assistant", "content": final},
        ]
        self.history = messages
        return {
            "final_response": final,
            "messages": messages,
            "interrupted": interrupted,
            "error": None,
            "last_reasoning": None,
        }


def maybe_build_synthetic_agent(session_id: str, model_override: Any = None) -> SyntheticHeavyAgent | None:
    """Return a :class:`SyntheticHeavyAgent` when the seam is armed, else ``None``.

    ``model_override`` (dict or str) only influences the reported ``model`` label;
    it never changes the compute.
    """
    if not synth_turn_armed():
        return None
    model = "synthetic-heavy"
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override["model"])
    elif isinstance(model_override, str) and model_override:
        model = model_override
    return SyntheticHeavyAgent(session_id, model=model)


__all__ = [
    "SyntheticHeavyAgent",
    "maybe_build_synthetic_agent",
    "synth_turn_armed",
]
