"""File-mutation verification footers and turn-completion explanations for ``AIAgent``.

The footer tells the model (and user) when a claimed file mutation did not land; the explainer
summarises why a turn ended without a final answer.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import re
from typing import Any, Dict, Optional

from agent.tool_dispatch_helpers import (
    _extract_error_preview,
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
)
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS,
    file_mutation_result_landed,
)


class TurnExplainersMixin:
    """File-mutation failure footer + turn-completion explainer (see module docstring)."""

    def _record_file_mutation_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        is_error: bool,
    ) -> None:
        """Record a ``write_file`` / ``patch`` outcome for the turn-end verifier.

        Failures store ``{path: {error_preview, tool}}``; a later success on the same path removes the entry.
        No-op when the per-turn state dict is not initialised (tool dispatched outside ``run_conversation``).
        """
        if tool_name not in _FILE_MUTATING_TOOLS:
            return
        state = getattr(self, "_turn_failed_file_mutations", None)
        if state is None:
            return
        targets = _extract_file_mutation_targets(tool_name, args)
        if not targets:
            return
        landed = file_mutation_result_landed(tool_name, result)
        if landed:
            landed_paths = _extract_landed_file_mutation_paths(tool_name, args, result)
            changed = getattr(self, "_turn_file_mutation_paths", None)
            if changed is not None:
                changed.update(landed_paths)
            # Feed the checkpoint agent-write ledger so /rollback's safe mode
            # can tell Hermes-authored content from later user hand-edits.
            mgr = getattr(self, "_checkpoint_mgr", None)
            if mgr is not None and getattr(mgr, "enabled", False):
                for _p in landed_paths:
                    try:
                        mgr.record_agent_write(_p)
                    except Exception:
                        pass
        if is_error and not landed:
            preview = _extract_error_preview(result)
            for path in targets:
                # Keep the FIRST error per path unless a later success replaces it.
                if path not in state:
                    state[path] = {
                        "tool": tool_name,
                        "error_preview": preview,
                    }
        else:
            for path in targets:
                state.pop(path, None)

    def _file_mutation_verifier_enabled(self) -> bool:
        """Check whether the per-turn file-mutation verifier footer is on.

        ``display.file_mutation_verifier`` (default True), cached per agent; ``HERMES_FILE_MUTATION_VERIFIER``
        overrides on every call and is never cached. A method so tests can patch one seam.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_FILE_MUTATION_VERIFIER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_file_mutation_verifier_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "file_mutation_verifier" in _display:
                enabled = bool(_display.get("file_mutation_verifier"))
            else:
                enabled = True  # safe default: verifier on
            self._file_mutation_verifier_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: verifier on

    # Bare absolute / home / Windows-drive paths in a footer line. Mirrors the gateway's
    # extract_local_files detector so anything it WOULD auto-attach is backticked first (#35584).
    _FOOTER_PATH_RE = re.compile(
        r"(?<![/:\w.`])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.[\w]+",
    )

    @classmethod
    def _neutralize_footer_paths(cls, text: str) -> str:
        """Wrap bare file paths in backticks so the gateway's ``extract_local_files`` never auto-attaches
        them.

        The extractor skips paths inside inline-code spans. Already-backticked paths are left alone (no
        double-wrap).
        """
        if not text:
            return text
        return cls._FOOTER_PATH_RE.sub(lambda m: f"`{m.group(0)}`", text)

    @classmethod
    def _format_file_mutation_failure_footer(cls, failed: Dict[str, Dict[str, Any]]) -> str:
        """Render the per-turn failed-mutation dict as a user-facing footer.

        Up to 10 paths with their first error preview, then an overflow count; empty string when nothing
        failed.
        Every path is backtick-wrapped via ``_neutralize_footer_paths`` so protected files cannot be auto-
        delivered.
        """
        if not failed:
            return ""
        lines = [
            "⚠️ File-mutation verifier: "
            f"{len(failed)} file(s) were NOT modified this turn despite any "
            "wording above that may suggest otherwise. Run `git status` or "
            "`read_file` to confirm."
        ]
        shown = 0
        for path, info in failed.items():
            if shown >= 10:
                break
            preview = (info.get("error_preview") or "").strip()
            tool = info.get("tool") or "patch"
            if preview:
                lines.append(f"  • `{path}` — [{tool}] {preview}")
            else:
                lines.append(f"  • `{path}` — [{tool}] failed")
            shown += 1
        remaining = len(failed) - shown
        if remaining > 0:
            lines.append(f"  • … and {remaining} more")
        # Neutralize paths the preview echoed; the lookbehind prevents double-wrapping the bullet path.
        return cls._neutralize_footer_paths("\n".join(lines))

    def _turn_completion_explainer_enabled(self) -> bool:
        """Check whether the end-of-turn completion explainer footer is on.

        ``display.turn_completion_explainer`` (default True), cached per agent;
        ``HERMES_TURN_COMPLETION_EXPLAINER``
        overrides on every call and is never cached. Mirrors ``_file_mutation_verifier_enabled``.
        """
        try:
            import os as _os
            env = _os.environ.get("HERMES_TURN_COMPLETION_EXPLAINER")
            if env is not None:
                return env.strip().lower() not in {"0", "false", "no", "off"}
            cached = getattr(self, "_turn_completion_explainer_enabled_cache", None)
            if cached is not None:
                return cached
            # Read from the persisted config.yaml so gateway and CLI share
            # the same setting.  Import lazily to avoid a startup-time cycle.
            try:
                from hermes_cli.config import load_config as _load_config
                _cfg = _load_config() or {}
            except Exception:
                _cfg = {}
            _display = _cfg.get("display") if isinstance(_cfg, dict) else None
            if isinstance(_display, dict) and "turn_completion_explainer" in _display:
                enabled = bool(_display.get("turn_completion_explainer"))
            else:
                enabled = True  # safe default: explainer on
            self._turn_completion_explainer_enabled_cache = enabled
            return enabled
        except Exception:
            pass
        return True  # safe default: explainer on

    @staticmethod
    def _format_turn_completion_explanation(
        turn_exit_reason: str, persistence_cause: Optional[str] = None
    ) -> str:
        """Render a user-facing explanation for an abnormal turn ending.

        Maps ``turn_exit_reason`` to an actionable message so a turn with no usable reply is never silent.
        ``persistence_cause`` refines ``session_persistence_failed`` wording (lock contention ≠ disk full).
        Returns "" for non-abnormal reasons so callers can concatenate unconditionally.
        """
        if not turn_exit_reason:
            return ""
        reason = str(turn_exit_reason)

        # Normal completion — stay quiet.  ``text_response(...)`` is the
        # healthy terminal; anything that produced a real reply is fine.
        if reason.startswith("text_response"):
            return ""

        prefix = "⚠️ No reply: "
        if reason == "empty_response_exhausted":
            return (
                prefix
                + "the model returned empty content after retries and any "
                "fallback providers. Try `continue`, switch model/provider, "
                "or inspect the tool output above."
            )
        if reason == "all_retries_exhausted_no_response":
            return (
                prefix
                + "all API retries were exhausted before a response was "
                "produced (provider errors / rate limits). Try `continue` "
                "or switch provider."
            )
        if reason == "partial_stream_recovery":
            return (
                prefix
                + "streaming stopped early and only a partial response was "
                "recovered. Send `continue` to resume from where it stopped."
            )
        if reason == "fallback_prior_turn_content":
            return (
                prefix
                + "no new content was produced this turn; showing recovered "
                "prior context. Send `continue` to retry."
            )
        if reason == "interrupted_during_api_call":
            return (
                prefix
                + "the request was interrupted mid-call before a reply was "
                "received. Send `continue` to retry."
            )
        if reason == "budget_exhausted":
            return (
                prefix
                + "the per-turn iteration/cost budget was exhausted before a "
                "final answer. Send `continue` to keep going."
            )
        if reason == "ollama_runtime_context_too_small":
            return (
                prefix
                + "the local model's context window was too small to finish. "
                "Increase the context size or use a larger model."
            )
        if reason.startswith("max_iterations_reached"):
            return (
                prefix
                + "the maximum tool-iteration limit was reached before a "
                "final answer. Send `continue` to keep going, or raise "
                "`max_iterations`."
            )
        if reason.startswith("error_near_max_iterations"):
            return (
                prefix
                + "an error occurred near the iteration limit before a final "
                "answer. Check the tool output above, then send `continue`."
            )
        if reason.startswith("repeated_outer_errors"):
            return (
                prefix
                + "the turn kept failing with repeated errors and was stopped "
                "early instead of retrying forever. Check the errors above, "
                "then send `continue` to retry."
            )
        if reason == "pending_tool_result":
            return (
                prefix
                + "the turn stopped while a tool result was still pending and "
                "the model produced no follow-up text. Send `continue` to "
                "let it summarize."
            )
        if reason == "session_persistence_failed":
            cause = persistence_cause or "unknown"
            if cause == "compression":
                return (
                    prefix
                    + "the turn was stopped because another process was "
                    "compressing this session. Your message should already be "
                    "saved — please send it again after compression completes."
                )
            if cause == "compression_closed":
                return (
                    prefix
                    + "the turn was stopped because this session was rotated "
                    "by context compression and its live continuation could "
                    "not be adopted. The storage itself is healthy — refresh "
                    "the client (or start a new turn) so it picks up the new "
                    "session id, then send your message again."
                )
            if cause == "turn_lease":
                return (
                    prefix
                    + "the turn was stopped because another Hermes process "
                    "took over this session. Your reply was not saved — wait "
                    "for the other process to finish, then send your message "
                    "again."
                )
            if cause == "locked":
                return (
                    prefix
                    + "the turn was stopped because session storage was busy "
                    "(another Hermes process was writing to the state "
                    "database). Your message should already be saved — "
                    "please send it again in a moment."
                )
            if cause == "replaced":
                return (
                    prefix
                    + "the turn was stopped because the state database file "
                    "was replaced underneath this process. Do not run "
                    "`hermes doctor --fix` or in-place FTS repair — stop "
                    "the process, restore the intended state.db, then "
                    "restart. Unwritten messages were diverted to "
                    "sessions/<session_id>.jsonl and, on the gateway, "
                    "pending_messages/pending-*.json."
                )
            if cause == "corrupt":
                return (
                    prefix
                    + "the turn was stopped because the state database "
                    "reported structural corruption (the transcript would "
                    "have been lost on restart). Freeing disk space will "
                    "not help. Recovery options:\n"
                    "1. Run `hermes doctor --fix`\n"
                    "2. Salvage with: sqlite3 ~/.hermes/state.db \".recover\" "
                    "(then replace state.db)\n"
                    "3. Restore from a backup in ~/.hermes/backups/\n"
                    "Then send your message again."
                )
            if cause == "disk":
                return (
                    prefix
                    + "the turn was stopped because session storage could not "
                    "be written (the transcript would have been lost on "
                    "restart). This is often a full disk — free some space "
                    "(or fix state.db permissions), then send your message "
                    "again."
                )
            return (
                prefix
                + "the turn was stopped because session storage could not be "
                "written (the transcript would have been lost on restart). "
                "Check the state database health (`hermes doctor`), then "
                "send your message again."
            )
        # Unknown/diagnostic-only reasons (e.g. "unknown", guardrail_halt
        # which already surfaces its own message) — don't second-guess.
        return ""
