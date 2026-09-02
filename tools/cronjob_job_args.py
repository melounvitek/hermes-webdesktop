"""Cron job argument normalization, validation and result shaping
(extracted from tools/cronjob_tools.py; re-exported there)."""

import logging
from typing import Any, Dict, List, Optional, Union

from cron.jobs import effective_job_state

# Logger parity with the origin module.
logger = logging.getLogger("tools.cronjob_tools")


def _origin_from_env() -> Optional[Dict[str, str]]:
    from gateway.session_context import get_session_env
    origin_platform = get_session_env("HERMES_SESSION_PLATFORM")
    origin_chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    if not (origin_platform and origin_chat_id):
        return None
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID") or None
    # Slack stamps every TOP-LEVEL message's own id as the session thread (a
    # per-message session KEY, not a conversation location). Persisting it
    # would pin all future deliveries inside an ephemeral thread, so a thread
    # id equal to the creating message's id is synthetic and dropped; a real
    # in-thread creation (thread == parent's id != this message) keeps it.
    if thread_id and origin_platform == "slack":
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID") or None
        if message_id and str(thread_id) == str(message_id):
            logger.debug(
                "Cron origin: dropping synthetic per-message Slack "
                "thread_id=%s (== creation message id)", thread_id,
            )
            thread_id = None
    if thread_id:
        logger.debug(
            "Cron origin captured thread_id=%s for %s:%s",
            thread_id, origin_platform, origin_chat_id,
        )
    return {
        "platform": origin_platform,
        "chat_id": origin_chat_id,
        "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
        "thread_id": thread_id,
        # Lets an opt-in delivery mirror resolve the exact participant's
        # session in per-user-isolated group chats (parity with send_message).
        "user_id": get_session_env("HERMES_SESSION_USER_ID") or None,  # harmless for DMs
        # Workspace/server scope (Slack team, Discord guild...). Slack session
        # keys embed it, so a continuable cron seed built without it would
        # create a row no scoped reply ever resolves to.
        "scope_id": get_session_env("HERMES_SESSION_SCOPE_ID") or None,
    }


def _local_delivery_notice(job: Dict[str, Any], user_deliver: Optional[str]) -> Optional[str]:
    """Notice when a created job won't deliver anywhere.

    CLI/TUI sessions have no capturable origin, so deliver='origin' (or an
    omitted deliver) yields a job whose output is saved but never delivered.
    Surface that at create time rather than silently dropping the user's
    "tell me when it runs" intent. None when the user explicitly asked for
    ``local`` or the job resolves to a real target.
    """
    if (user_deliver or "").strip().lower() == "local":
        return None
    try:
        from cron.scheduler import _resolve_delivery_targets

        if _resolve_delivery_targets(job):
            return None
    except Exception:
        # Resolution unavailable — fall back to the origin signal.
        if job.get("origin"):
            return None
    return (
        "This is a local-only cron job: its output is saved (view it with "
        "cronjob(action='list')) but will NOT be delivered back into this "
        "session — CLI/TUI sessions have no live-delivery channel. To be "
        "notified when it runs, recreate or update the job with deliver set to "
        "a gateway-connected platform, e.g. deliver='telegram' or deliver='all'."
    )


def _mode_guidance_notes(job: Dict[str, Any], user_deliver: Optional[str]) -> List[str]:
    """Mode-specific guidance echoed once in the create/update response
    (instead of in the schema, which is paid for on every API call)."""
    notes: List[str] = []
    if job.get("monitor_script") or job.get("monitor_url"):
        notes.append(
            "Monitor mode: the source runs first each tick and its output is "
            "hashed as exact bytes — unchanged output suppresses the agent run "
            "(silent no_change tick), changed output injects a MONITOR CHANGE "
            "DETECTED diff into the prompt. The first tick always runs as "
            "baseline. The source must emit STABLE output (no timestamps, no "
            "random ordering) or every tick will look changed."
        )
    if job.get("no_agent"):
        notes.append(
            "no_agent mode: stdout is delivered verbatim; EMPTY stdout sends "
            "nothing at all (watchdog pattern — script should stay quiet when "
            "there is nothing to report). Non-zero exit or timeout sends an "
            "error alert. prompt/skills are ignored."
        )
    _deliver = (user_deliver or "").strip().lower()
    if _deliver:
        if "all" in _deliver.split(","):
            notes.append(
                "deliver='all' resolves at fire time and never includes "
                "bot-chat targets — channels connected later are picked up "
                "automatically."
            )
        if _deliver.startswith("bot-chat:"):
            notes.append(
                "Targeting another profile's Bot Chat costs that bot an agent "
                "turn per run."
            )
        # platform:chat_id with no thread segment loses topic targeting.
        for target in _deliver.split(","):
            parts = target.strip().split(":")
            if (
                len(parts) == 2
                and parts[0] not in ("bot-chat", "sms")
                and parts[1]
                and not parts[1].startswith("#")
            ):
                notes.append(
                    f"deliver target '{target.strip()}' has no :thread_id "
                    "segment — on thread/topic platforms the delivery lands in "
                    "the main chat, not a topic."
                )
                break
    return notes


def _split_monitor_arg(
    monitor: Optional[str],
    monitor_script: Optional[str],
    monitor_url: Optional[str],
) -> tuple:
    """Resolve the single model-facing ``monitor`` field into the stored
    ``(monitor_script, monitor_url)`` pair.

    Shape decides transport: http(s):// is a URL, anything else a script path
    (a legal script path can never start with a URL scheme). Storage keeps the
    two fields separate — interface merge, not a storage migration.
    Update semantics: None = unchanged, '' = clear; setting one source clears
    the other so switching transports never trips mutual exclusion. An
    explicit ``monitor`` wins over the legacy alias fields.
    """
    if monitor is None:
        return monitor_script, monitor_url
    value = monitor.strip()
    if not value:
        return "", ""
    if value.lower().startswith(("http://", "https://")):
        return "", value
    return value, ""


def _repeat_display(job: Dict[str, Any]) -> str:
    rep = job.get("repeat") or {}
    times, completed = rep.get("times"), rep.get("completed", 0)
    if times is None:
        return "forever"
    if times == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _clean_str_list(items: Any) -> List[str]:
    """Stripped, non-empty ``str(item)`` values from a str-or-iterable (order kept)."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    return [s for s in (str(i).strip() for i in items) if s]


def _canonical_skills(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    if skills is None:
        skills = [skill] if skill else []
    elif isinstance(skills, str):
        skills = [skills]
    # `item or ""`: a None entry must drop out, not stringify to "None".
    return list(dict.fromkeys(_clean_str_list(item or "" for item in skills)))


def _normalize_optional_job_value(value: Optional[Any], *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _normalize_deliver_param(value: Any) -> Optional[str]:
    """Canonical string form of ``deliver``; None for None/empty.

    MCP clients / scripts may pass a list (``["telegram"]``); stored as-is the
    scheduler's ``str(deliver).split(",")`` would yield the literal
    ``"['telegram']"``. Flatten at the API boundary.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(_clean_str_list(value)) or None
    return str(value).strip() or None


def _validate_bot_chat_deliver(deliver: Optional[str]) -> Optional[str]:
    """Validate ``bot-chat[:<profile>]`` deliver elements at create time.

    Bot Chat delivery is machine-local: the profile must exist where the
    scheduler fires (Desktop multi-gateway rosters may show same-named profiles
    from other machines). Fail loudly here rather than as a per-run delivery error.
    Returns an error string or None.
    """
    if not deliver:
        return None
    try:
        from cron.scheduler import parse_bot_chat_deliver_token
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None  # best-effort; resolution re-checks at fire time
    for part in str(deliver).split(","):
        profile_arg = parse_bot_chat_deliver_token(part.strip())
        if not profile_arg:
            continue  # not a bot-chat token, or bare token (own profile)
        try:
            canon = normalize_profile_name(profile_arg)
        except Exception:
            return f"invalid bot-chat profile name '{profile_arg}'"
        if not profile_exists(canon):
            return (
                f"bot-chat delivery profile '{profile_arg}' not found on this "
                "gateway's machine. Bot Chat delivery is machine-local — use a "
                "profile that exists here (hermes profile list), or omit the "
                "name (deliver='bot-chat') for the job's own profile."
            )
    return None


def _resolve_cron_context_deliver(deliver: Optional[str]) -> Optional[str]:
    """Resolve ``origin`` to a concrete target for creates made FROM a cron run.

    The creating session is ephemeral, so by fire time there is no origin to
    resolve. Non-cron sessions: returned unchanged. Cron sessions: ``origin``
    (or an omitted value) becomes the creating run's ``platform:chat_id[:thread]``
    from the HERMES_CRON_AUTO_DELIVER_* contextvars, or ``local`` when the
    creating run has no concrete target; other elements pass through verbatim.
    Without this the scheduler would fall back to guessing a home channel.
    """
    from gateway.session_context import get_session_env
    from utils import is_truthy_value

    if not is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")):
        return deliver

    def _creator_target() -> str:
        platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip()
        chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
        if not platform or not chat_id:
            return "local"
        thread_id = get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip()
        return f"{platform}:{chat_id}:{thread_id}" if thread_id else f"{platform}:{chat_id}"

    if deliver is None:
        return _creator_target()
    resolved = [_creator_target() if p.lower() == "origin" else p for p in _clean_str_list(str(deliver).split(","))]
    # Order-preserving de-dup: 'origin,local' with a local creator -> 'local'.
    return ",".join(dict.fromkeys(resolved)) or None


def _validate_cron_base_url(
    provider: Optional[Any], base_url: Optional[Any]
) -> Optional[str]:
    """Reject pairing a named provider's stored credential with an off-host base_url.

    A prompt-injected job could name a real provider plus an attacker
    base_url; at fire time the provider's stored key would be sent there
    (credential exfil). Allowed: no override; bare 'custom' (pure BYOK, key
    derived from the base_url itself); an override whose host matches the
    named provider's own endpoint. Everything else fails closed.
    Returns an error string if blocked, else None.
    """
    bu = _normalize_optional_job_value(base_url, strip_trailing_slash=True)
    if not bu:
        return None
    prov = _normalize_optional_job_value(provider)
    if not prov:
        # No provider inherits the default provider's stored key — same primitive.
        return (
            "base_url override requires an explicit provider. Set provider to a "
            "configured custom provider to use a custom endpoint."
        )
    try:
        from hermes_cli.runtime_provider import (
            has_named_custom_provider,
            resolve_requested_provider,
            _get_named_custom_provider,
        )
        from hermes_cli.auth import PROVIDER_REGISTRY
        from utils import base_url_host_matches, base_url_hostname
    except Exception:
        return f"Unable to validate base_url override for provider {prov!r}; refused."

    if prov.lower() == "custom":
        # Pure BYOK: key comes from a pool keyed by THIS base_url or host-gated
        # env vars, never an arbitrary stored secret.
        return None
    if has_named_custom_provider(prov):
        # A NAMED custom provider carries a STORED key that the runtime still
        # sends to an override base_url — require the configured host.
        try:
            cp = _get_named_custom_provider(prov)
        except Exception:
            cp = None
        cfg_host = base_url_hostname((cp or {}).get("base_url", "")) if cp else ""
        if cfg_host and base_url_host_matches(bu, cfg_host):
            return None
        return (
            f"base_url {bu!r} is not allowed for provider {prov!r}. A named "
            f"custom provider's stored credential may only be sent to its own "
            f"configured endpoint ({cfg_host or 'unknown'})."
        )
    try:
        resolved = resolve_requested_provider(prov)
    except Exception:
        resolved = prov
    pconfig = PROVIDER_REGISTRY.get(resolved) if isinstance(resolved, str) else None
    known_host = base_url_hostname(getattr(pconfig, "inference_base_url", "") if pconfig else "")
    if known_host and base_url_host_matches(bu, known_host):
        return None
    # Fail closed: covers named providers with stored credentials AND
    # aliases/unknown names we cannot host-match.
    return (
        f"base_url {bu!r} is not allowed for provider {prov!r}. A named "
        f"provider's stored credential may only be sent to its own endpoint; "
        f'use a configured custom provider (provider="custom") for a custom base_url.'
    )


def _validate_cron_script_path(script: Optional[str]) -> Optional[str]:
    """Scripts must be relative paths resolving within HERMES_HOME/scripts/
    (absolute / ~ / drive-letter paths rejected — prompt-injection guard).
    Returns an error string if blocked, else None; empty = clearing, OK."""
    if not script or not script.strip():
        return None

    from hermes_constants import get_hermes_home

    raw = script.strip()
    if raw.startswith(("/", "~")) or (len(raw) >= 2 and raw[1] == ":"):
        return (
            f"Script path must be relative to ~/.hermes/scripts/. "
            f"Got absolute or home-relative path: {raw!r}. "
            f"Place scripts in ~/.hermes/scripts/ and use just the filename."
        )

    from tools.path_security import validate_within_dir

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if validate_within_dir(scripts_dir / raw, scripts_dir):
        return f"Script path escapes the scripts directory via traversal: {raw!r}"
    return None


def _apply_continuity(
    context_from: Optional[Union[str, List[str]]],
    continuity: bool,
) -> Optional[List[str]]:
    """continuity=True ensures "self" is in context_from; False removes any
    "self" entry. Other entries are preserved untouched."""
    refs = _clean_str_list(context_from)
    has_self = any(r.lower() == "self" for r in refs)
    if continuity and not has_self:
        refs.append("self")
    elif not continuity and has_self:
        refs = [r for r in refs if r.lower() != "self"]
    return refs or None


def _validate_context_from_refs(refs: List[Any]) -> Optional[str]:
    """Error string if any non-"self" ref names a job that doesn't exist.
    ("self" resolves to the job's own id at run time, so it can't be checked
    against the store — the job doesn't exist yet at create time.)"""
    from cron.jobs import get_job as _get_job
    for ref_id in refs:
        if isinstance(ref_id, str) and ref_id.strip().lower() == "self":
            continue
        if not _get_job(ref_id):
            return (
                f"context_from job '{ref_id}' not found. "
                "Use cronjob(action='list') to see available jobs."
            )
    return None


# Optional fields echoed by _format_job only when truthy on the job record
# (order matters: it is the JSON key order).
_FORMAT_JOB_OPTIONAL_KEYS = (
    "script", "reasoning_effort", "monitor_script", "monitor_url",
    "monitor_state", "no_agent", "enabled_toolsets", "workdir",
)


def _format_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(job.get("prompt") or "")
    skills = _canonical_skills(job.get("skill"), job.get("skills"))
    job_id = str(job.get("id") or "unknown")
    name = str(job.get("name") or prompt[:50] or (skills[0] if skills else "") or job_id or "cron job")
    result = {
        "job_id": job_id,
        "name": name,
        "skill": skills[0] if skills else None,
        "skills": skills,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "model": job.get("model"),
        "provider": job.get("provider"),
        "base_url": job.get("base_url"),
        "schedule": job.get("schedule_display") or "?",
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_delivery_error": job.get("last_delivery_error"),
        "last_delivery_unverified": job.get("last_delivery_unverified"),
        "last_fire_error": job.get("last_fire_error"),
        "enabled": job.get("enabled", True),
        # Derive from enabled so half-paused records never render as paused.
        "state": effective_job_state(job),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
    }
    for key in _FORMAT_JOB_OPTIONAL_KEYS:
        if job.get(key):
            result[key] = True if key == "no_agent" else job[key]
    stored_refs = job.get("context_from") or []
    if isinstance(stored_refs, str):
        stored_refs = [stored_refs]
    is_self = lambda r: str(r).strip().lower() == "self" or r == job.get("id")  # noqa: E731
    if any(is_self(r) for r in stored_refs):
        result["continuity"] = True
    external_refs = [r for r in stored_refs if not is_self(r)]
    if external_refs:
        result["context_from"] = external_refs
    if isinstance(job.get("attach_to_session"), bool):
        result["attach_to_session"] = job["attach_to_session"]
    return result


def _gateway_liveness_notice(plural: bool = False) -> dict:
    """``gateway_running``/``warning`` payload via the shared CLI helper so the
    CLI and this tool agree on what "scheduler active" means. False -> warning
    (builtin ticker has no gateway process), None -> probe failed."""
    try:
        from hermes_cli.cron import _builtin_gateway_liveness

        _gw = _builtin_gateway_liveness()
    except Exception:
        return {"gateway_running": None}
    if _gw is False:
        subject = "these jobs are saved" if plural else "this job is saved"
        return {
            "gateway_running": False,
            "warning": (
                f"The Hermes gateway is not running — {subject} "
                "but will NOT fire until the gateway is started "
                "(hermes gateway install / hermes gateway start). "
                "Tell the user the task is scheduled but not active yet."
            ),
        }
    return {"gateway_running": None if _gw is None else True}
