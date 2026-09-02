"""Pure tool-call loop guardrail primitives.

The controller is side-effect free: it tracks per-turn tool-call observations
and returns decisions. Runtime code decides whether a decision becomes warning
guidance, a synthetic tool result, or a controlled turn halt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "session_search", "skill_view",
    "skills_list", "browser_snapshot", "browser_console", "browser_get_images",
    "mcp_filesystem_read_file", "mcp_filesystem_read_text_file",
    "mcp_filesystem_read_multiple_files", "mcp_filesystem_list_directory",
    "mcp_filesystem_list_directory_with_sizes", "mcp_filesystem_directory_tree",
    "mcp_filesystem_get_file_info", "mcp_filesystem_search_files",
})

MUTATING_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "write_file", "patch", "todo_list", "memory", "skill_manage",
    "browser_click", "browser_type", "browser_press", "browser_scroll", "browser_navigate",
    "send_message", "cronjob_manage", "delegate_task", "process_manage",
})

# Tools legitimately re-invoked with identical args while waiting on external
# progress (pollers). The identical-call NOTICE never fires for these.
STALL_GUARD_REPEATABLE_TOOLS = frozenset({"process_manage"})
# Poller naming conventions on generated / MCP surfaces (``<vendor>_get_result``).
_STALL_GUARD_REPEATABLE_SUFFIXES = ("_get_result", "_poll")

# Notice fires on the Nth consecutive identical (tool, args, result) call; 3
# tolerates one legitimate double-check while catching observed re-issue loops.
STALL_GUARD_IDENTICAL_CALL_THRESHOLD = 3

# Result-reference stubbing: from the 2nd consecutive identical call whose fresh
# result is byte-identical, the duplicate payload is replaced by a reference
# stub. Results under this size aren't worth stubbing; errors are never stubbed.
IDENTICAL_RESULT_STUB_MIN_CHARS = 512
# Canonical-args preview kept in the stub so the model still knows WHAT the call
# was if compression later evicts the referenced result.
_RESULT_STUB_ARGS_PREVIEW_CHARS = 120

# Tools whose "failure" is normal work output (red test run, empty grep, page
# timeout). same_tool_failure (DIFFERENT commands) never halts these; only an
# exact-args replay with no intervening change, or an identical-result streak, can.
FAILURE_TOLERANT_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "process_manage", "process", "browser_navigate", "web_extract",
})

# A successful call to one of these marks progress for every failing signature
# still counted this turn: the next retry is a new experiment (edit -> re-run), not a replay.
PROGRESS_RESET_TOOL_NAMES = frozenset({
    "write_file", "patch", "terminal", "execute_code", "browser_click", "browser_type",
    "browser_press", "browser_navigate", "process_manage", "process", "delegate_task",
    "send_message", "cronjob", "cronjob_manage", "todo", "todo_list", "memory", "skill_manage",
})


def is_stall_guard_repeatable(tool_name: str) -> bool:
    """Whether a tool is exempt from the identical-call loop notice."""
    return tool_name in STALL_GUARD_REPEATABLE_TOOLS or tool_name.endswith(
        _STALL_GUARD_REPEATABLE_SUFFIXES
    )


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings never prevent execution. Hard stops are opt-in for interactive
    platforms but default on for unattended gateway/cron platforms where nobody
    can interrupt a model that ignores loop warnings.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    non_interactive_hard_stop_enabled: bool = True
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)
    loop_caps: "LoopCapConfig" = field(default_factory=lambda: LoopCapConfig())

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        *,
        platform: str | None = None,
    ) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section.

        Nested ``warn_after`` / ``hard_stop_after`` keys win over the flat legacy keys.
        """
        if not isinstance(data, Mapping):
            data = {}
        d = cls()
        hard_stop_enabled = _as_bool(data.get("hard_stop_enabled"), d.hard_stop_enabled)
        non_interactive_hard_stop_enabled = _as_bool(
            data.get("non_interactive_hard_stop_enabled"), d.non_interactive_hard_stop_enabled,
        )
        if _is_non_interactive_platform(platform) and non_interactive_hard_stop_enabled:
            hard_stop_enabled = True

        thresholds: dict[str, int] = {}
        for field_name, (section_name, key) in _THRESHOLD_SOURCES.items():
            section = data.get(section_name)
            if not isinstance(section, Mapping):
                section = {}
            thresholds[field_name] = _int_at_least(
                section.get(key, data.get(field_name)), getattr(d, field_name), 1,
            )

        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), d.warnings_enabled),
            hard_stop_enabled=hard_stop_enabled,
            non_interactive_hard_stop_enabled=non_interactive_hard_stop_enabled,
            loop_caps=LoopCapConfig.from_mapping(data.get("loop_caps")),
            **thresholds,
        )


# Threshold field -> (nested section, nested key). The flat legacy key is the field name itself.
_THRESHOLD_SOURCES: dict[str, tuple[str, str]] = {
    "exact_failure_warn_after": ("warn_after", "exact_failure"),
    "same_tool_failure_warn_after": ("warn_after", "same_tool_failure"),
    "no_progress_warn_after": ("warn_after", "idempotent_no_progress"),
    "exact_failure_block_after": ("hard_stop_after", "exact_failure"),
    "same_tool_failure_halt_after": ("hard_stop_after", "same_tool_failure"),
    "no_progress_block_after": ("hard_stop_after", "idempotent_no_progress"),
}


# Per-turn caps on runaway-prone tools; counters reset in reset_for_turn at the
# start of every agent loop, so the limit is per turn, not per session. Dozens
# of searches / subagent spawns in one loop is already pathological.
_DEFAULT_MAX_WEB_SEARCHES_PER_TURN = 50
_DEFAULT_MAX_SUBAGENTS_PER_TURN = 50


@dataclass(frozen=True)
class LoopCapConfig:
    """Per-turn hard ceilings on web_search calls / subagent spawns.

    Unlike the loop detector (keyed on repeated identical/failing calls) these
    count total calls within the turn and fire regardless of
    ``hard_stop_enabled``. ``0`` disables a cap.
    """

    max_web_searches: int = _DEFAULT_MAX_WEB_SEARCHES_PER_TURN
    max_subagents: int = _DEFAULT_MAX_SUBAGENTS_PER_TURN

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LoopCapConfig":
        """Build config from the ``tool_loop_guardrails.loop_caps`` section."""
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        return cls(
            max_web_searches=_int_at_least(data.get("max_web_searches"), defaults.max_web_searches, 0),
            max_subagents=_int_at_least(data.get("max_subagents"), defaults.max_subagents, 0),
        )


_INTERACTIVE_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp"})
# Not chat gateways, but bounded supervised task loops (a subagent is stopped by
# its parent; api_server has a live client). Both do real edit -> re-run work,
# so they keep the interactive warn-only default.
_SUPERVISED_TASK_PLATFORMS = frozenset({"subagent", "api_server"})


def _is_non_interactive_platform(platform: str | None) -> bool:
    """True for gateway/cron sessions where tool loops are unattended."""
    if not isinstance(platform, str) or not platform.strip():
        return False
    key = platform.strip().lower()
    return key not in _INTERACTIVE_PLATFORMS and key not in _SUPERVISED_TASK_PLATFORMS


@dataclass(frozen=True)
class IdenticalCallObservation:
    """Outcome of observing one completed call: ``notice`` is appended after the
    result, ``stub`` replaces a byte-identical duplicate result. Both may be set."""

    notice: str | None = None
    stub: str | None = None


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        return cls(tool_name=tool_name, args_hash=_sha256(canonical_tool_args(args or {})))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return _canonical_json(args)


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag.
    """
    if result is None or file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict) and data.get("success") is False and "exceed the limit" in data.get("error", ""):
            return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        # signature -> a mutating call succeeded since its last failure
        self._progress_since_failure: dict[ToolCallSignature, bool] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None
        # Identical-call streak: CONSECUTIVE identical (tool, args) calls with
        # identical results. Any different call or result resets it, so re-reads
        # after edits and varied polling are never flagged.
        self._identical_streak_sig: ToolCallSignature | None = None
        self._identical_streak_result_hash: str = ""
        self._identical_streak_count: int = 0
        # tool_call_id of the streak's FIRST call, so a stub can point at the full payload.
        self._identical_streak_first_call_id: str = ""
        # tool_call_id -> spillover path, so a stub referencing a result that
        # entered context only as a persisted-output preview can't dangle.
        self._persisted_result_paths: dict[str, str] = {}
        self._turn_web_search_count = 0
        self._turn_subagent_count = 0

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def _halt(
        self, action: str, code: str, message: str,
        tool_name: str, count: int, signature: ToolCallSignature,
    ) -> ToolGuardrailDecision:
        """Build a block/halt decision and record it as the turn's halt decision."""
        self._halt_decision = ToolGuardrailDecision(
            action=action, code=code, message=message,
            tool_name=tool_name, count=count, signature=signature,
        )
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        allow = ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # Loop caps apply regardless of hard_stop_enabled (which only governs the detector).
        cap_block = self._check_loop_cap(tool_name, args, signature)
        if cap_block is not None:
            return cap_block

        if not self.config.hard_stop_enabled:
            return allow

        exact_count = self._exact_failure_counts.get(signature, 0)
        if self._progress_since_failure.get(signature):
            # Something landed since this call last failed — let it run; the
            # streak restarts in after_call if it fails again.
            exact_count = 0
        if exact_count >= self.config.exact_failure_block_after:
            return self._halt(
                "block", "repeated_exact_failure_block",
                f"Blocked {tool_name}: the same tool call failed {exact_count} "
                "times with identical arguments. Stop retrying it unchanged; "
                "change strategy or explain the blocker.",
                tool_name, exact_count, signature,
            )

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None and record[1] >= self.config.no_progress_block_after:
                repeat_count = record[1]
                return self._halt(
                    "block", "idempotent_no_progress_block",
                    f"Blocked {tool_name}: this read-only call returned the same "
                    f"result {repeat_count} times. Stop repeating it unchanged; "
                    "use the result already provided or try a different query.",
                    tool_name, repeat_count, signature,
                )

        return allow

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        def warn(code: str, message: str, count: int) -> ToolGuardrailDecision:
            return ToolGuardrailDecision(
                action="warn", code=code, message=message,
                tool_name=tool_name, count=count, signature=signature,
            )

        if failed:
            # An identical failing call is only a REPLAY if nothing landed in
            # between; a mutation since the last identical failure makes the
            # retry a new experiment, so the exact-args streak restarts.
            if self._progress_since_failure.pop(signature, False):
                self._exact_failure_counts.pop(signature, None)
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            # same_tool_failure counts DIFFERENT args on one tool; for
            # failure-tolerant tools a run of distinct red commands is diagnosis,
            # not a loop — warn, never halt (exact-args replay still applies).
            if (
                self.config.hard_stop_enabled
                and tool_name not in FAILURE_TOLERANT_TOOL_NAMES
                and same_count >= self.config.same_tool_failure_halt_after
            ):
                return self._halt(
                    "halt", "same_tool_failure_halt",
                    f"Stopped {tool_name}: it failed {same_count} times this turn. "
                    "Stop retrying the same failing tool path and choose a different approach.",
                    tool_name, same_count, signature,
                )

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return warn(
                    "repeated_exact_failure_warning",
                    f"{tool_name} has failed {exact_count} times with identical arguments. "
                    "This looks like a loop; inspect the error and change strategy "
                    "instead of retrying it unchanged.",
                    exact_count,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return warn(
                    "same_tool_failure_warning",
                    _tool_failure_recovery_hint(tool_name, same_count),
                    same_count,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        # A successful mutation is progress for every failing signature still
        # counted this turn (next identical retry runs against changed state).
        # Pure loops never mutate between attempts, so the replay detector keeps its teeth.
        if tool_name in PROGRESS_RESET_TOOL_NAMES or file_mutation_result_landed(tool_name, result):
            for sig in list(self._exact_failure_counts):
                self._progress_since_failure[sig] = True
            self._same_tool_failure_counts.clear()

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = previous[1] + 1 if previous is not None and previous[0] == result_hash else 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return warn(
                "idempotent_no_progress_warning",
                f"{tool_name} returned the same result {repeat_count} times. "
                "Use the result already provided or change the query instead of "
                "repeating it unchanged.",
                repeat_count,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        return tool_name not in self.config.mutating_tools and tool_name in self.config.idempotent_tools

    def observe_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        tool_call_id: str = "",
        failed: bool = False,
    ) -> "IdenticalCallObservation":
        """Track consecutive identical calls; return notice + dedupe stub info.

        ``notice`` fires from the ``STALL_GUARD_IDENTICAL_CALL_THRESHOLD``-th
        consecutive identical (tool, args, result) call; purely observational,
        pollers exempt. ``stub`` replaces the CURRENT result from the 2nd
        byte-identical repeat — the tool still executed, only the context
        representation is deduplicated, so polling semantics survive (a changed
        result flows through whole and resets the streak). Pollers are NOT
        exempt from stubbing: an unchanged poll is exactly when the stub saves
        the most. Short results, failed results and non-string results are never
        stubbed. Callers substitute at result construction time, which is cache-safe.
        """
        is_plain_str = isinstance(result, str)
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        result_hash = _result_hash(result) if is_plain_str else ""

        if (
            is_plain_str
            and self._identical_streak_sig == signature
            and self._identical_streak_result_hash == result_hash
        ):
            self._identical_streak_count += 1
        else:
            # New streak; non-string (multimodal) results never form one.
            self._identical_streak_sig = signature if is_plain_str else None
            self._identical_streak_result_hash = result_hash
            self._identical_streak_count = 1 if is_plain_str else 0
            self._identical_streak_first_call_id = tool_call_id or ""

        count = self._identical_streak_count

        notice = None
        if not is_stall_guard_repeatable(tool_name) and count >= STALL_GUARD_IDENTICAL_CALL_THRESHOLD:
            ordinal = f"{count}{'th' if 11 <= count % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(count % 10, 'th')}"
            notice = (
                f"[hermes note: this is the {ordinal} consecutive identical call to "
                f"{tool_name} with identical arguments returning the same result. "
                "Do not repeat it — change arguments, use a different tool, or "
                "proceed with what you have.]"
            )
            # The no-progress BLOCK in before_call only covers idempotent_tools; this
            # streak is tool-agnostic, so with hard stops on, halt at the same threshold
            # (a model replaying a successful `terminal` call otherwise runs to the budget).
            if (
                self.config.hard_stop_enabled
                and count >= self.config.no_progress_block_after
                and self._halt_decision is None
            ):
                self._halt(
                    "halt", "identical_call_streak_halt",
                    f"Stopped {tool_name}: the same call with identical arguments "
                    f"returned the same result {count} times in a row. Stop "
                    "repeating it unchanged; use the result already provided or "
                    "change strategy.",
                    tool_name, count, signature,
                )

        stub = None
        if is_plain_str and count >= 2 and not failed and len(result) >= IDENTICAL_RESULT_STUB_MIN_CHARS:
            stub = self._build_result_reference_stub(tool_name, args)

        return IdenticalCallObservation(notice=notice, stub=stub)

    def record_persisted_result(self, tool_call_id: str, file_path: str) -> None:
        """Remember the spillover path a persisted result was saved to."""
        if tool_call_id and file_path:
            self._persisted_result_paths[tool_call_id] = file_path

    def _build_result_reference_stub(self, tool_name: str, args: Mapping[str, Any] | None) -> str:
        """Reference stub for a byte-identical duplicate result (tool + args preview)."""
        try:
            args_preview = canonical_tool_args(_coerce_args(args))
        except TypeError:
            args_preview = "{}"
        if len(args_preview) > _RESULT_STUB_ARGS_PREVIEW_CHARS:
            args_preview = args_preview[:_RESULT_STUB_ARGS_PREVIEW_CHARS] + "…"
        first_id = self._identical_streak_first_call_id
        ref = f" (tool_call_id {first_id})" if first_id else ""
        stub = (
            f"[hermes note: this result is byte-identical to the {tool_name} "
            f"result earlier this turn{ref}. Refer to that result; it has not "
            f"changed. Args: {args_preview}]"
        )
        spill_path = self._persisted_result_paths.get(first_id) if first_id else None
        if spill_path:
            stub += (
                f"\n[The referenced result was persisted to: {spill_path} — "
                "page through it with read_file if you need the full content.]"
            )
        return stub

    def _check_loop_cap(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Block once a per-turn cap is reached, else advance the counter and return None.

        A cap of 0 disables that limit. Blocking happens BEFORE the call when
        the count is already at the cap, so the (cap+1)-th call is refused.
        """
        caps = self.config.loop_caps
        if tool_name == "web_search":
            cap, count, increment = caps.max_web_searches, self._turn_web_search_count, 1
            code, message = "loop_web_search_cap", (
                f"Blocked web_search: this turn has already made {cap} "
                "web searches, the per-turn limit. This looks like a "
                "runaway search loop. Work with the results you already "
                "have and give the user your answer."
            )
        elif tool_name == "delegate_task":
            cap, count = caps.max_subagents, self._turn_subagent_count
            # Control actions (list/steer/stop) spawn nothing and must keep working after the cap is hit.
            increment = _subagent_spawn_count(args) if cap else 0
            if increment == 0:
                return None
            code, message = "loop_subagent_cap", (
                f"Blocked delegate_task: this turn has already spawned "
                f"{count} subagents (limit {cap}). "
                "This looks like a runaway delegation loop. Finish the "
                "work with the results you have and answer the user."
            )
        else:
            return None

        if cap and count >= cap:
            return self._halt("block", code, message, tool_name, count, signature)
        if tool_name == "web_search":
            self._turn_web_search_count += increment
        else:
            self._turn_subagent_count += increment
        return None


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {"error": decision.message, "guardrail": decision.to_metadata()},
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    return (result or "") + f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = _canonical_json(parsed)
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _int_at_least(value: Any, default: int, minimum: int) -> int:
    """Int parser: junk/None/below-minimum fall back to default (caps use minimum 0 so 0 = disabled)."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _subagent_spawn_count(args: Mapping[str, Any]) -> int:
    """Subagents one delegate_task call spawns: batch size when ``tasks`` is a
    non-empty list, else 1; control actions (list/steer/stop) spawn 0."""
    if str(args.get("action") or "").strip().lower() in ("list", "steer", "stop"):
        return 0
    tasks = args.get("tasks")
    return len(tasks) if isinstance(tasks, list) and tasks else 1


def _sha256(value: str) -> str:
    # surrogatepass: web-scraped results can carry unpaired UTF-16 surrogates; a
    # strict encode would raise and take down the conversation loop.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
