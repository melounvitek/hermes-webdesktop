"""hermes fallback — manage the fallback provider chain.

Fallback providers are tried in order when the primary model fails with rate-limit, overload, or
connection errors. See: https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-
providers

Subcommands: hermes fallback [list] Show the current fallback chain (default when no subcommand)
hermes fallback add Pick provider + model via the same picker as `hermes model`, then append the
selection to the chain hermes fallback remove Pick an entry to delete from the chain hermes fallback
clear Remove all fallback entries
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from hermes_cli.fallback_config import get_fallback_chain


def _identity(entry: Dict[str, Any]):
    """BackendIdentity for a ``{provider, model, base_url?}`` entry."""
    from agent.backend_identity import BackendIdentity

    return BackendIdentity.build(
        provider=entry.get("provider"),
        model=entry.get("model"),
        base_url=entry.get("base_url"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_chain(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the normalized fallback chain as a list of dicts.

    Accepts both the new list format (``fallback_providers``) and the legacy ``fallback_model``
    format. When both are present, the effective chain is merged with ``fallback_providers`` entries
    kept first. The returned list is always a fresh copy — callers can mutate without touching the
    config dict.
    """
    return get_fallback_chain(config)


def _write_chain(config: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
    """Persist the chain to ``fallback_providers`` and clear legacy key."""
    config["fallback_providers"] = chain
    # Drop the legacy single-dict key on write so there's only one source of truth.
    config.pop("fallback_model", None)


def _format_entry(entry: Dict[str, Any]) -> str:
    """One-line human-readable rendering of a fallback entry."""
    base = entry.get("base_url")
    return f"{entry.get('model', '?')}  (via {entry.get('provider', '?')}){f'  [{base}]' if base else ''}"


def _extract_fallback_from_model_cfg(model_cfg: Any) -> Optional[Dict[str, Any]]:
    """Pull the ``{provider, model, base_url?, api_mode?}`` dict from a ``config["model"]`` snapshot."""
    if not isinstance(model_cfg, dict):
        return None
    provider = (model_cfg.get("provider") or "").strip()
    # The picker writes the selected model to ``model.default``.
    model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()
    if not provider or not model:
        return None
    entry: Dict[str, Any] = {"provider": provider, "model": model}
    for key in ("base_url", "api_mode"):
        value = (model_cfg.get(key) or "").strip()
        if value:
            entry[key] = value
    return entry


def _snapshot_auth_active_provider() -> Any:
    """Return the current ``active_provider`` in auth.json, or a sentinel if unavailable."""
    try:
        from hermes_cli.auth import _load_auth_store
        return _load_auth_store().get("active_provider")
    except Exception:
        return None


def _restore_auth_active_provider(value: Any) -> None:
    """Write back a previously snapshotted ``active_provider`` value."""
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
        with _auth_store_lock():
            store = _load_auth_store()
            store["active_provider"] = value
            _save_auth_store(store)
    except Exception:
        # Best-effort — if auth.json can't be restored, the user's primary
        # provider may have been deactivated by the picker.  They can re-run
        # `hermes model` to fix it.  Don't fail the fallback add.
        pass


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _entries(n: int) -> str:
    return f"{n} {'entry' if n == 1 else 'entries'}"


def _print_chain(heading: str, chain: List[Dict[str, Any]]) -> None:
    print(f"  {heading} ({_entries(len(chain))}):")
    for i, entry in enumerate(chain, 1):
        print(f"    {i}. {_format_entry(entry)}")
    print()


def _load_chain(empty_message: str):
    """Load config + chain; print ``empty_message`` block and return ``(config, None)`` when empty."""
    from hermes_cli.config import load_config

    config = load_config()
    chain = _read_chain(config)
    if not chain:
        print()
        print(empty_message)
        print()
        return config, None
    return config, chain


def cmd_fallback_list(args) -> None:  # noqa: ARG001
    """Print the current fallback chain."""
    config, chain = _load_chain("  No fallback providers configured.")
    if chain is None:
        print("  Add one with:  hermes fallback add")
        print()
        return

    print()
    primary = _describe_primary(config)
    if primary:
        print(f"  Primary:   {primary}")
        print()
    _print_chain("Fallback chain", chain)
    print("  Tried in order when the primary fails (rate-limit, 5xx, connection errors).")
    print("  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers")
    print()


def _describe_primary(config: Dict[str, Any]) -> Optional[str]:
    """One-line description of the primary model for display purposes."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "?").strip() or "?"
        model = (model_cfg.get("default") or model_cfg.get("model") or "?").strip() or "?"
        return f"{model}  (via {provider})"
    if isinstance(model_cfg, str) and model_cfg.strip():
        return model_cfg.strip()
    return None


def cmd_fallback_add(args) -> None:
    """Launch the same picker as `hermes model`, then append the selection to the chain."""
    from hermes_cli.main import _require_tty, select_provider_and_model
    from hermes_cli.config import load_config, save_config

    _require_tty("fallback add")

    # Snapshot BEFORE the picker runs so we can distinguish "user actually
    # picked something" from "user cancelled" by comparing before/after.
    before_cfg = load_config()
    model_before = copy.deepcopy(before_cfg.get("model"))
    active_provider_before = _snapshot_auth_active_provider()

    print()
    print("  Adding a fallback provider.  The picker below is the same one used by")
    print("  `hermes model` — select the provider + model you want as a fallback.")
    print()

    def _restore() -> None:
        _restore_model_cfg(model_before)
        _restore_auth_active_provider(active_provider_before)

    try:
        select_provider_and_model(args=args)
    except SystemExit:
        # Some provider flows exit on auth failure — restore state and re-raise.
        _restore()
        raise

    # Read the post-picker state to see what the user selected.
    after_cfg = load_config()
    model_after = after_cfg.get("model")

    new_entry = _extract_fallback_from_model_cfg(model_after)
    if not new_entry:
        # Picker didn't complete (user cancelled or flow bailed).  Nothing to do.
        _restore()
        print()
        print("  No fallback added.")
        return

    # Picker picked the same thing that's already the primary → nothing changed,
    # and there's nothing useful to add as a fallback to itself. Identity
    # semantics owned by agent.backend_identity (#54250/#57584/#62984): same
    # provider+model on a DIFFERENT explicit base_url is a different backend
    # (multi-endpoint pool) and is a legitimate fallback.
    from agent.backend_identity import same_deployment

    new_ident = _identity(new_entry)
    primary_entry = _extract_fallback_from_model_cfg(model_before)
    if primary_entry and same_deployment(_identity(primary_entry), new_ident):
        _restore()
        print()
        print(f"  Selected model matches the current primary ({_format_entry(new_entry)}).")
        print("  A provider cannot be a fallback for itself — no change.")
        return

    # Reload the config with the primary restored, then append the new entry
    # to ``fallback_providers``.  We deliberately re-load (rather than mutating
    # ``after_cfg``) because the picker may have touched other top-level keys
    # (custom_providers, providers credentials) that we want to keep.
    _restore()

    final_cfg = load_config()
    chain = _read_chain(final_cfg)

    # Reject exact-duplicate fallback entries (same deployment; a different
    # explicit base_url is a different endpoint and NOT a duplicate).
    if any(same_deployment(_identity(existing), new_ident) for existing in chain):
        print()
        print(f"  {_format_entry(new_entry)} is already in the fallback chain — skipped.")
        return

    chain.append(new_entry)
    _write_chain(final_cfg, chain)
    save_config(final_cfg)

    print()
    print(f"  Added fallback: {_format_entry(new_entry)}")
    print(f"  Chain is now {_entries(len(chain))} long.")
    print()
    print("  Run `hermes fallback list` to view, or `hermes fallback remove` to delete.")


def _restore_model_cfg(model_before: Any) -> None:
    """Restore ``config["model"]`` to a previously-captured snapshot."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    if model_before is None:
        cfg.pop("model", None)
    else:
        cfg["model"] = copy.deepcopy(model_before)
    save_config(cfg)


def cmd_fallback_remove(args) -> None:  # noqa: ARG001
    """Pick an entry from the chain and remove it."""
    from hermes_cli.config import save_config

    config, chain = _load_chain("  No fallback providers configured — nothing to remove.")
    if chain is None:
        return

    choices = [_format_entry(e) for e in chain] + ["Cancel"]

    try:
        from hermes_cli.setup import _curses_prompt_choice
        idx = _curses_prompt_choice("Select a fallback to remove:", choices, 0)
    except Exception:
        idx = _numbered_pick("Select a fallback to remove:", choices)

    if idx is None or idx < 0 or idx >= len(chain):
        print()
        print("  Cancelled — no change.")
        return

    removed = chain.pop(idx)
    _write_chain(config, chain)
    save_config(config)

    print()
    print(f"  Removed fallback: {_format_entry(removed)}")
    print(f"  Chain is now {_entries(len(chain))} long." if chain else "  Fallback chain is now empty.")
    print()


def cmd_fallback_clear(args) -> None:  # noqa: ARG001
    """Remove all fallback entries (with confirmation)."""
    from hermes_cli.config import save_config

    config, chain = _load_chain("  No fallback providers configured — nothing to clear.")
    if chain is None:
        return

    print()
    _print_chain("Current fallback chain", chain)
    try:
        resp = input("  Clear all entries? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("  Cancelled.")
        return
    if resp not in {"y", "yes"}:
        print("  Cancelled — no change.")
        return

    _write_chain(config, [])
    save_config(config)
    print()
    print("  Fallback chain cleared.")
    print()


def _numbered_pick(question: str, choices: List[str]) -> Optional[int]:
    """Fallback numbered-list picker when curses is unavailable."""
    print(question)
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    print()
    while True:
        try:
            val = input(f"Choice [1-{len(choices)}]: ").strip()
            if not val:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"Please enter 1-{len(choices)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def cmd_fallback(args) -> None:
    """Top-level dispatcher for ``hermes fallback [subcommand]``."""
    sub = getattr(args, "fallback_command", None)
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"Unknown fallback subcommand: {sub}")
        print("Use one of: list, add, remove, clear")
        raise SystemExit(2)
    handler(args)


_SUBCOMMANDS = {
    None: cmd_fallback_list, "": cmd_fallback_list, "list": cmd_fallback_list, "ls": cmd_fallback_list,
    "add": cmd_fallback_add,
    "remove": cmd_fallback_remove, "rm": cmd_fallback_remove,
    "clear": cmd_fallback_clear,
}
