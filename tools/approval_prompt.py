"""Human prompt surfaces for :mod:`tools.approval`.

The interactive CLI prompt (callback panel or ``input()`` fallback) and the
operator-selected plugin approval transport. Detection, allowed scopes,
persistence, timeout policy and the final authorization stay host-owned in
``tools.approval``; this module only asks and reports the answer.
"""

import logging
import os
import sys
import threading
import time
from tools.approval_human_wait import human_wait_window
from tools.interrupt import is_interrupted

logger = logging.getLogger("tools.approval")


def prompt_dangerous_approval(command: str, description: str,
                              timeout_seconds: int | None = None,
                              allow_permanent: bool = True,
                              approval_callback=None,
                              *, allow_session: bool = True,
                              smart_denied: bool = False) -> str:
    """Prompt the user to approve a dangerous command (CLI only).

    Args:
        allow_permanent: When False, hide [a]lways (tirith warnings present:
            broad permanent allowlisting is wrong for content-level findings).
        allow_session: When False, hide [s]ession too — the caller grants one
            operation and re-asks next time (the protected agent-instruction
            gate in ``tools/file_tools.py``). Offering a scope the caller
            discards makes every later write re-prompt and reads as broken.
        smart_denied: Owner override of a Smart DENY: offer only once/deny.
        approval_callback: CLI prompt_toolkit callback,
            ``(command, description, *, allow_permanent=True,
            allow_session=True, smart_denied=False) -> str``. Legacy
            signatures keep working while both keywords hold their defaults.

    Returns: 'once', 'session', 'always', 'deny', or 'timeout'. 'timeout'
        means no user response — still blocked (fail-closed), but callers
        report "no response" rather than an explicit denial.
    """
    from tools import approval as _a
    if timeout_seconds is None:
        timeout_seconds = _a._get_approval_timeout()

    # Everything below is a human prompt (callback panel or input() fallback,
    # both bounded by the approval deadline): record it as human-wait time so
    # the concurrent batch deadline excludes it (#79719).
    with human_wait_window():
        return _prompt_dangerous_approval_inner(
            command,
            description,
            timeout_seconds,
            allow_permanent,
            approval_callback,
            allow_session=allow_session,
            smart_denied=smart_denied,
        )


_CLI_CHOICE_ALIASES = {
    "o": "once", "once": "once",
    "s": "session", "session": "session",
    "a": "always", "always": "always",
}

_CLI_CHOICE_I18N = {
    "once": "approval.allowed_once",
    "session": "approval.allowed_session",
    "always": "approval.allowed_always",
    "deny": "approval.denied",
}


def _prompt_dangerous_approval_inner(command: str, description: str,
                                     timeout_seconds: int,
                                     allow_permanent: bool = True,
                                     approval_callback=None,
                                     *, allow_session: bool = True,
                                     smart_denied: bool = False) -> str:
    # Redact before any user-visible rendering; the original `command` is
    # still what executes after approval. Same redactor as memory/log
    # sanitization so tokens mask consistently across surfaces.
    from agent.redact import redact_sensitive_text
    display_command = redact_sensitive_text(command)
    display_description = redact_sensitive_text(description)

    # Smart DENY and a session-less gate both reduce the menu to once/deny.
    once_only = smart_denied or not allow_session

    if approval_callback is not None:
        try:
            callback_kwargs = {"allow_permanent": allow_permanent}
            if not allow_session:
                callback_kwargs["allow_session"] = False
            if smart_denied:
                callback_kwargs["smart_denied"] = True
            return approval_callback(
                display_command, display_description, **callback_kwargs
            )
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # Fail-closed guard: when prompt_toolkit owns the terminal and no callback
    # is registered on this thread, the input() fallback would spawn a daemon
    # thread whose read never sees Enter (keystrokes go to prompt_toolkit) —
    # an invisible deadlock (#15216). Deny loudly instead; threads needing
    # interactive approval must install a callback via
    # tools.terminal_tool.set_approval_callback() first.
    try:
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning(
                "Dangerous-command approval requested on a thread with no "
                "approval callback while prompt_toolkit is active; denying "
                "to avoid stdin deadlock. command=%r description=%r",
                command, description,
            )
            return "deny"
    except Exception:
        pass  # prompt_toolkit absent or detection failed: legacy input() path is safe

    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    try:
        from agent.i18n import t
        if once_only:
            prompt = t("approval.prompt_smart_deny")
        else:
            prompt = t("approval.prompt_long") if allow_permanent else t("approval.prompt_short")
        while True:
            print()
            print(f"  {t('approval.dangerous_header', description=display_description)}")
            print(f"      {display_command}")
            print()
            if once_only:
                print(t("approval.choose_smart_deny"))
            elif allow_permanent:
                print(t("approval.choose_long"))
            else:
                print(t("approval.choose_short"))
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n" + t("approval.timeout"))
                return "timeout"  # distinct from deny: the user never answered

            choice = result["choice"]
            if once_only:
                choice_map = {
                    **{
                        value: "once"
                        for value in t("approval.smart_deny_once_inputs").split(",")
                    },
                    **{
                        value: "deny"
                        for value in t("approval.smart_deny_deny_inputs").split(",")
                    },
                }
                decision = choice_map.get(choice, "deny")
            else:
                decision = _CLI_CHOICE_ALIASES.get(choice, "deny")
                if decision == "always" and not allow_permanent:
                    decision = "session"
            print(t(_CLI_CHOICE_I18N[decision]))
            return decision

    except (EOFError, KeyboardInterrupt):
        print("\n" + t("approval.cancelled"))
        return "deny"
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


def get_plugin_manager():
    """Lazy plugin-manager seam used by tests and early tool-only imports."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager as _get_manager

    # Approval can be imported before model_tools (which triggers discovery);
    # make an explicitly selected transport available on the first approval
    # instead of treating the undiscovered registry as unavailable.
    discover_plugins()
    return _get_manager()


def _present_with_selected_transport(
    *,
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    allow_session: bool,
    allow_permanent: bool,
) -> dict:
    """Present through an explicitly selected plugin transport, if any.

    A selected transport replaces every built-in prompt surface; detection,
    allowed scopes, persistence, timeout, and final authorization stay
    host-owned. A failed transport reaches a built-in surface only under the
    explicit ``transport_fallback: builtin`` opt-in.
    """
    from tools import approval as _a
    name, fallback = _a._get_approval_transport_config()
    if name == "builtin":
        return {"selected": False}

    try:
        registered = _a.get_plugin_manager().get_approval_transport(name)
    except Exception:
        # Plugin/discovery exception text may contain plugin-owned secrets.
        logger.warning("Could not resolve selected approval transport %r", name)
        registered = None
    if registered is None:
        logger.warning("Selected approval transport %r is unavailable", name)
        return {
            "selected": True,
            "choice": "deny",
            "failure": "unavailable",
            "fallback": fallback,
            "name": name,
        }

    try:
        from agent.redact import redact_sensitive_text
        from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport

        timeout_seconds = _a._get_approval_timeout()
        request = ApprovalRequest.create(
            command=redact_sensitive_text(command, force=True),
            description=redact_sensitive_text(description, force=True),
            pattern_key=pattern_key,
            pattern_keys=tuple(pattern_keys),
            session_key=session_key,
            surface=surface,
            allow_session=allow_session,
            allow_permanent=allow_permanent,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        # Never fall back to raw text if redaction or request construction
        # fails: fail closed without calling the plugin or leaking the
        # unredacted payload to logs/hooks.
        logger.warning("Could not build redacted plugin approval request")
        return {
            "selected": True,
            "choice": "deny",
            "failure": "error",
            "fallback": None,
            "name": name,
        }
    hook_kwargs = dict(
        command=request.command,
        description=request.description,
        pattern_key=pattern_key,
        pattern_keys=list(pattern_keys),
        session_key=session_key,
        surface=f"transport:{name}",
        request_id=request.request_id,
        request_digest=request.digest,
    )
    _a._fire_approval_hook("pre_approval_request", **hook_kwargs)
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - minimal tool-only environments
        touch_activity_if_due = None
    now = time.monotonic()
    activity_state = {"last_touch": now, "start": now}

    def _poll() -> None:
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for plugin approval transport")

    with human_wait_window(session_key):
        result = invoke_approval_transport(
            registered.present,
            request,
            timeout_seconds=timeout_seconds,
            on_poll=_poll,
            is_interrupted=is_interrupted,
        )
    hook_choice = result.choice if result.failure is None else f"transport_{result.failure}"
    _a._fire_approval_hook("post_approval_response", **hook_kwargs, choice=hook_choice)
    return {
        "selected": True,
        "choice": result.choice,
        "failure": result.failure,
        "fallback": fallback,
        "name": name,
    }


def _transport_denied_result(
    *, pattern_key: str, description: str, failure: str
) -> dict:
    from tools import approval as _a
    breaker_addendum = _a._denial_breaker_addendum(_a.get_current_session_key())
    return _a._denied(
        f"BLOCKED: Selected approval transport failed ({failure}); the user "
        "has NOT consented to this action. Do NOT retry this command or "
        "attempt the same outcome through another route."
        f"{breaker_addendum}",
        pattern_key=pattern_key, description=description,
        outcome=f"transport_{failure}",
    )


def _transport_choice(attempt: dict, *, pattern_key: str, description: str):
    """Interpret a ``_present_with_selected_transport`` attempt.

    Returns ``(choice, denied_result)``: both None when the built-in surfaces
    should run (no transport selected, or a failure with the explicit builtin
    fallback); a denied result for any other failure; else the user's choice.
    """
    if not attempt.get("selected"):
        return None, None
    failure = attempt.get("failure")
    if failure and attempt.get("fallback") == "builtin":
        logger.warning(
            "Approval transport %r failed (%s); using explicit builtin fallback",
            attempt.get("name"),
            failure,
        )
        return None, None
    if failure:
        return None, _transport_denied_result(
            pattern_key=pattern_key, description=description, failure=failure,
        )
    return attempt.get("choice"), None
