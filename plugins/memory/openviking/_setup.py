"""Interactive ``hermes memory setup`` wizard for the OpenViking provider.

Pure UI flow: prompts, menus, and the persistence of the chosen connection
(Hermes ``.env`` only, or mirrored to an ``ovcli.conf.<name>`` profile that
Hermes then links). Network validation and file writers live in the package
``__init__`` and are looked up there at call time so tests can monkeypatch
them on the plugin module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_SETUP_CANCELLED = object()


def _ov():
    """The plugin module — resolved lazily so monkeypatched validators are honored."""
    return sys.modules[__package__]


def _retry_or_cancel_manual_setup(select, title: str, message: str, cancelled):
    print(f"  {message}")
    choice = select(
        title,
        [("Retry", "try this step again"), ("Cancel setup", "no changes saved")],
        default=0,
        cancel_returns=cancelled,
    )
    return True if choice == 0 else _SETUP_CANCELLED


def _print_validation_progress(message: str) -> None:
    print(f"  {message}", flush=True)


def _reachability_failure_allows_local_autostart(message: str) -> bool:
    return not (message or "").startswith(_ov()._OPENVIKING_RESPONDED_FAILURE_PREFIX)


def _handle_unreachable_endpoint(endpoint: str, message: str, select, cancelled, *, allow_local_autostart: bool = True):
    """-> True (reachable now) / False (re-prompt URL) / _SETUP_CANCELLED."""
    ov = _ov()
    is_local = ov._is_local_openviking_url(endpoint)
    if is_local and allow_local_autostart:
        print(f"  {message}")
        choice = select(
            "  Local OpenViking server is down",
            [
                ("Start local OpenViking", "run openviking-server and retry"),
                ("Retry URL", "enter the server URL again"),
                ("Cancel setup", "no changes saved"),
            ],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == 1:
            return False
        if choice != 0:
            return _SETUP_CANCELLED
        start_state, start_message = ov._start_local_openviking_server(endpoint)
        print(f"  {start_message}")
        if start_state != ov._LOCAL_SERVER_STARTED:
            return False
        print("  Waiting for OpenViking server to become reachable...", flush=True)
        if ov._wait_for_openviking_health(endpoint, timeout_seconds=ov._LOCAL_OPENVIKING_AUTOSTART_TIMEOUT):
            print("  OpenViking server is reachable.")
            return True
        print("  OpenViking server did not become reachable.")
        return False

    return _retry_or_cancel_manual_setup(
        select,
        "  OpenViking server unhealthy" if is_local else "  OpenViking server unreachable",
        message,
        cancelled,
    )


def _prompt_profile_name(prompt, select, cancelled) -> str | object:
    ov = _ov()
    while True:
        name = ov._clean_config_value(prompt("OpenViking profile name"))
        if ov._is_valid_ovcli_profile_name(name):
            return name
        retry = _retry_or_cancel_manual_setup(
            select,
            "  Invalid OpenViking profile name",
            "Profile names can only contain letters, numbers, '-' and '_'.",
            cancelled,
        )
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _confirm_replace_existing_profile(path: Path, values: dict, select, cancelled):
    ov = _ov()
    if not path.exists():
        return True
    try:
        existing_data = ov._load_ovcli_config(path)
    except Exception:
        existing_data = {}
    if existing_data == ov._ovcli_data_from_connection_values(values):
        return True
    choice = select(
        "  OpenViking profile already exists",
        [
            ("Choose another name", "leave the existing profile unchanged"),
            ("Replace profile", "overwrite this saved OpenViking profile"),
            ("Cancel setup", "no changes saved"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if choice == 1:
        return True
    if choice == 0:
        return False
    return _SETUP_CANCELLED


def _prompt_endpoint(prompt, select, cancelled) -> str | object:
    """Ask for a custom server URL until it normalizes and answers /health."""
    ov = _ov()
    while True:
        try:
            endpoint = ov._normalize_openviking_url(prompt("OpenViking server URL", default=ov._DEFAULT_ENDPOINT))
        except ov._OpenVikingEndpointError as exc:
            if _retry_or_cancel_manual_setup(select, "  Invalid OpenViking endpoint", str(exc), cancelled) is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue
        _print_validation_progress("Checking OpenViking server...")
        reachable, message = ov._validate_openviking_reachability(endpoint)
        if reachable:
            print("  OpenViking server is reachable.")
            return endpoint
        retry = _handle_unreachable_endpoint(
            endpoint, message, select, cancelled,
            allow_local_autostart=_reachability_failure_allows_local_autostart(message),
        )
        if retry is True:
            return endpoint
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


_KEY_ROLES = {"user": "User API key", "root": "Root API key"}
# When the entered key turns out to have the other role: (note, menu title, switch option).
_REROUTE = {
    "root": ("That key is valid, but it is a user API key.", "  OpenViking key is a user key",
             ("Use as User API key", "server derives account/user automatically")),
    "user": ("That key is valid, but it has root access.", "  OpenViking user API key is root key",
             ("Configure as Root API key", "provide account and user IDs")),
}
_OTHER_ROLE = {"root": "user", "user": "root"}


def _reroute_key_type(select, cancelled, current: str):
    """Offer to use the key as the other role (prefill), re-enter, or cancel.
    -> (api_key_type, prefill) or _SETUP_CANCELLED."""
    note, title, switch_option = _REROUTE[current]
    print(f"  {note}")
    route_choice = select(
        title,
        [switch_option, (f"Re-enter {_KEY_ROLES[current]}", f"try another {current} key"), ("Cancel setup", "no changes saved")],
        default=0,
        cancel_returns=cancelled,
    )
    if route_choice == 0:
        return _OTHER_ROLE[current], True
    if route_choice == 1:
        return current, False
    return _SETUP_CANCELLED


def _prompt_manual_connection_values(prompt, select, cancelled, *, service: bool = False):
    ov = _ov()
    if service:
        endpoint = ov._OPENVIKING_SERVICE_ENDPOINT
        print(f"  OpenViking Service endpoint: {endpoint}")
    else:
        endpoint = _prompt_endpoint(prompt, select, cancelled)
        if endpoint is _SETUP_CANCELLED:
            return _SETUP_CANCELLED

    is_local = ov._is_local_openviking_url(endpoint)
    api_key_type = "user" if service else ""
    prefilled_api_key = ""

    def retry(title: str, message: str):
        """True to loop again, _SETUP_CANCELLED to abort."""
        return _retry_or_cancel_manual_setup(select, title, message, cancelled)

    while True:
        values = {"endpoint": endpoint, "api_key": "", "root_api_key": "", "account": "", "user": "", "agent": ""}
        if not api_key_type:
            options = [
                ("User API key", "recommended; server derives account/user automatically" if is_local else "server derives account/user automatically"),
                ("Root API key", "requires account and user IDs"),
            ]
            if is_local:
                options.append(("No API key", "only for explicitly unauthenticated local development"))
            credential_choice = select(
                "  OpenViking credential" if is_local else "  OpenViking API key type",
                options, default=0, cancel_returns=cancelled,
            )
            if credential_choice == cancelled:
                return _SETUP_CANCELLED
            if is_local and credential_choice == 2:
                _print_validation_progress("Validating OpenViking local dev access...")
                valid, message, _role = ov._validate_openviking_setup_values(values)
                if valid:
                    print("  OpenViking local dev access validated.")
                    return values
                if retry("  OpenViking credential failed", message) is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            api_key_type = "root" if credential_choice == 1 else "user"

        values["api_key_type"] = api_key_type
        api_key_label = "OpenViking API key" if service else f"OpenViking {api_key_type} API key"
        if prefilled_api_key:
            values["api_key"], prefilled_api_key = prefilled_api_key, ""
        else:
            values["api_key"] = ov._clean_config_value(prompt(api_key_label, secret=True))
        if not values["api_key"]:
            if retry("  OpenViking API key required", f"{api_key_label} is required.") is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue

        if api_key_type == "root":
            _print_validation_progress("Validating OpenViking root API key...")
            valid, message, role = ov._validate_openviking_setup_values(values, require_api_key=True)
            if not (valid and role == "root"):
                if valid and role == "user":
                    routed = _reroute_key_type(select, cancelled, "root")
                    if routed is _SETUP_CANCELLED:
                        return _SETUP_CANCELLED
                    api_key_type, prefill = routed
                    prefilled_api_key = values["api_key"] if prefill else ""
                    continue
                if retry("  OpenViking root API key failed", message) is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            print("  OpenViking root API key validated.")
            values["root_api_key"] = values["api_key"]
            account_ok, account_message, values["account"] = ov._validate_openviking_identity_value(
                prompt("OpenViking account"), field="account",
            )
            user_ok, user_message, values["user"] = ov._validate_openviking_identity_value(
                prompt("OpenViking user"), field="user",
            )
            if not account_ok or not user_ok:
                message = account_message if not account_ok else user_message
                if retry("  OpenViking tenant identity required", message) is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                prefilled_api_key = values["api_key"]
                continue

        _print_validation_progress("Validating OpenViking API access...")
        valid, message, role = ov._validate_openviking_setup_values(values, require_api_key=service or not is_local)
        if not valid:
            if retry("  OpenViking API access failed", message) is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue
        if api_key_type == "user" and role == "root":
            routed = _reroute_key_type(select, cancelled, "user")
            if routed is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            api_key_type, prefill = routed
            prefilled_api_key = values["api_key"] if prefill else ""
            continue
        if api_key_type == "root" and role != "root":
            if retry("  OpenViking root API key failed", "The supplied key was not accepted as a root API key.") is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue
        print("  OpenViking API access validated.")
        return values


def _set_openviking_provider(config: dict, provider_config: dict) -> None:
    config["memory"]["provider"] = "openviking"
    config["memory"]["openviking"] = provider_config


def _link_ovcli_profile(*, config: dict, provider_config: dict, env_path: Path, ovcli_path: Path) -> None:
    ov = _ov()
    for key in ("endpoint", "api_key", "root_api_key", "account", "user", "agent", "api_key_type"):
        provider_config.pop(key, None)
    provider_config["use_ovcli_config"] = True
    ov._remember_ovcli_path(provider_config, ovcli_path)
    _set_openviking_provider(config, provider_config)
    ov._write_env_vars(env_path, {}, remove_keys=ov._OPENVIKING_ENV_KEYS)
    for key in ov._OPENVIKING_ENV_KEYS:
        os.environ.pop(key, None)


def _save_hermes_only_config(*, config: dict, provider_config: dict, env_path: Path, values: dict) -> None:
    ov = _ov()
    provider_config["use_ovcli_config"] = False
    provider_config.pop("ovcli_config_path", None)
    # A newly selected connection must not inherit the previous YAML peer; a
    # non-empty peer, if supplied, is saved with the connection below.
    provider_config.pop("agent", None)
    _set_openviking_provider(config, provider_config)
    # Publish the file writer's cleaned values to the current process as well.
    writes = {key: ov._env_line_safe(value) for key, value in ov._env_writes_from_connection_values(values).items()}
    ov._write_env_vars(env_path, writes, remove_keys=ov._OPENVIKING_ENV_KEYS)
    os.environ.update(writes)
    for key in set(ov._OPENVIKING_ENV_KEYS) - set(writes):
        os.environ.pop(key, None)


def _profile_display_name(profile) -> str:
    if profile.source == "env":
        return _ov()._OVCLI_CONFIG_ENV
    if profile.source == "active":
        return "ovcli.conf"
    return profile.name


def _profile_description(profile) -> str:
    ov = _ov()
    endpoint = ov._clean_config_value(profile.values.get("endpoint")) or ov._DEFAULT_ENDPOINT
    return f"{endpoint} ({profile.path})"


def _validate_profile_for_setup(profile) -> tuple[bool, str, Optional[str]]:
    ov = _ov()
    require_api_key = not ov._is_local_openviking_url(profile.values.get("endpoint", ""))
    return ov._validate_openviking_setup_values(profile.values, require_api_key=require_api_key)


def _print_openviking_ready(message: str, path: Optional[Path] = None) -> None:
    print("\n  OpenViking memory is ready")
    print(f"  {message}")
    if path is not None:
        print(f"  Config file: {path}")
    print("  Start a new Hermes session to activate.\n")


def _run_existing_profile_setup(*, profiles: list, select, cancelled, config: dict, provider_config: dict, env_path: Path) -> bool | object:
    while True:
        choice = select(
            "  OpenViking profile",
            [(_profile_display_name(profile), _profile_description(profile)) for profile in profiles],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == cancelled or choice < 0 or choice >= len(profiles):
            return _SETUP_CANCELLED
        profile = profiles[choice]

        for attempt in (0, 1):
            _print_validation_progress("Validating OpenViking profile...")
            ok, message, _role = _validate_profile_for_setup(profile)
            if ok:
                _link_ovcli_profile(config=config, provider_config=provider_config, env_path=env_path, ovcli_path=profile.path)
                _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
                return True
            print(f"  {message}")
            if attempt == 1:
                break  # second failure returns to the profile picker
            retry = select(
                "  OpenViking profile validation failed",
                [
                    ("Choose another profile", "select a different OpenViking profile"),
                    ("Retry validation", "try this profile again"),
                    ("Cancel setup", "no changes saved"),
                ],
                default=0,
                cancel_returns=cancelled,
            )
            if retry == 0:
                break
            if retry != 1:
                return _SETUP_CANCELLED


def _mirror_manual_config_to_openviking_store(*, prompt, select, cancelled, values: dict) -> Path | object:
    ov = _ov()
    while True:
        name = _prompt_profile_name(prompt, select, cancelled)
        if name is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        path = ov._ovcli_config_dir() / f"{ov._OVCLI_SAVED_PREFIX}{name}"
        replace = _confirm_replace_existing_profile(path, values, select, cancelled)
        if replace is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        if replace is False:
            continue
        ov._write_ovcli_config(path, values)
        return path


def _run_create_profile_setup(*, prompt, select, cancelled, config: dict, provider_config: dict, env_path: Path) -> bool | object:
    source_choice = select(
        "  OpenViking connection",
        [
            ("OpenViking Service (VolcEngine Cloud)", "use the managed OpenViking endpoint"),
            ("Custom", "use a local, VPS, or self-hosted OpenViking server"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if source_choice == cancelled:
        return _SETUP_CANCELLED

    values = _prompt_manual_connection_values(prompt, select, cancelled, service=(source_choice == 0))
    if values is _SETUP_CANCELLED:
        return _SETUP_CANCELLED
    if values is None:
        return False

    save_choice = select(
        "  Save OpenViking config",
        [
            ("Keep in Hermes only", "write values only to Hermes .env"),
            ("Mirror to OpenViking store", "write ~/.openviking/ovcli.conf.<name> and link it"),
        ],
        default=1,
        cancel_returns=cancelled,
    )
    if save_choice == cancelled:
        return _SETUP_CANCELLED

    if save_choice == 1:
        ovcli_path = _mirror_manual_config_to_openviking_store(prompt=prompt, select=select, cancelled=cancelled, values=values)
        if ovcli_path is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        _link_ovcli_profile(config=config, provider_config=provider_config, env_path=env_path, ovcli_path=ovcli_path)
        _print_openviking_ready("Created and linked OpenViking profile.", ovcli_path)
        return True

    _save_hermes_only_config(config=config, provider_config=provider_config, env_path=env_path, values=values)
    _print_openviking_ready("Connection saved to Hermes .env.")
    return True


def run_setup(hermes_home: str, config: dict) -> None:
    """Entry point for ``OpenVikingMemoryProvider.post_setup``."""
    from hermes_cli.config import save_config
    from hermes_cli.memory_setup import _CANCELLED, _curses_select, _print_cancelled_setup, _prompt

    env_path = Path(hermes_home) / ".env"
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    provider_config = config["memory"].get("openviking", {})
    if not isinstance(provider_config, dict):
        provider_config = {}
    common = dict(select=_curses_select, cancelled=_CANCELLED, config=config, provider_config=provider_config, env_path=env_path)

    print("\n  OpenViking memory setup\n")

    profiles = _ov()._discover_ovcli_profiles()
    if profiles:
        choice = _curses_select(
            "  OpenViking config source",
            [
                ("Use existing OpenViking profile", "choose from detected ovcli.conf profiles"),
                ("Create new OpenViking profile", "enter a new URL/API key"),
            ],
            default=0,
            cancel_returns=_CANCELLED,
        )
        if choice == _CANCELLED:
            _print_cancelled_setup()
            return
        if choice == 0:
            result = _run_existing_profile_setup(profiles=profiles, **common)
            if result is _SETUP_CANCELLED:
                _print_cancelled_setup()
            elif result:
                save_config(config)
            return
    else:
        print("  No existing OpenViking CLI profiles found. Creating a new config.")

    result = _run_create_profile_setup(prompt=_prompt, **common)
    if result is _SETUP_CANCELLED:
        _print_cancelled_setup()
    elif result:
        save_config(config)
