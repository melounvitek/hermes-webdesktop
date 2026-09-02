"""Skin + config-change watcher: signatures for skin/pet/cron/sessions/platforms/pairing/bot-relay state and the broadcast loop that pushes *.changed events.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin

        init_skin_from_config(_load_cfg())
        skin = get_active_skin()
        return {
            "name": skin.name,
            "colors": skin.colors,
            # Paired palettes: the TUI detects the terminal's polarity and
            # prefers the matching hand-tuned block over adapting `colors`.
            "light_colors": skin.light_colors,
            "dark_colors": skin.dark_colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


# Signature of the last skin broadcast: (name, active user-file mtime). Lets the
# per-tool reconcile fire ``skin.changed`` on any real move — a name switch OR a
# live color edit to the active skin — and nothing else.
_last_skin_sig: tuple[str, float | None] | None = None


def _skin_sig() -> tuple[str, float | None]:
    """(active skin name, its user-file mtime). Built-ins have no file, so only
    their name moves; a user skin's mtime lets an in-place color edit repaint too."""
    name = str((_load_cfg().get("display") or {}).get("skin") or "default")
    override = get_hermes_home_override()
    home = override if isinstance(override, str) and override else _hermes_home
    try:
        mtime: float | None = (Path(home) / "skins" / f"{name}.yaml").stat().st_mtime
    except OSError:
        mtime = None
    return name, mtime


def _note_skin_broadcast() -> None:
    """Sync the reconcile baseline after the /skin RPC emits, so the per-tool
    check doesn't re-broadcast the skin /skin just applied."""
    global _last_skin_sig
    try:
        _last_skin_sig = _skin_sig()
    except Exception:
        pass


def _broadcast_skin_if_changed() -> None:
    """Emit ``skin.changed`` when the active skin moved — the agent switched it
    (``hermes config set display.skin``) OR edited the active skin's colors in
    place ("I don't like that coral" → tweak the YAML).

    Routes through the SAME live path as ``/skin`` so every surface (TUI + desktop)
    repaints, no slash command. The signature check is a dict lookup + one stat,
    so polling it is ~free.
    """
    global _last_skin_sig
    try:
        sig = _skin_sig()
    except Exception:
        return
    if sig == _last_skin_sig:
        return
    _last_skin_sig = sig
    try:
        _broadcast_global_event("skin.changed", resolve_skin())
    except Exception:
        pass


def _watcher_home() -> Path:
    """Active profile home for the change watcher's signature probes."""
    override = get_hermes_home_override()
    return Path(override if isinstance(override, str) and override else _hermes_home)


def _pet_sig() -> tuple:
    """(slug, spritesheet revision, scale) of the active pet — ("off",) when none.

    Cheap by construction: config comes from the mtime-cached ``_load_cfg`` and
    the sheet revision is one stat. Moves when ``/pet`` (de)activates a pet, the
    hatch flow rebuilds a sheet, or the scale changes."""
    display = _load_cfg().get("display") or {}
    pet_cfg = display.get("pet") if isinstance(display.get("pet"), dict) else {}
    if not pet_cfg or not is_truthy_value(pet_cfg.get("enabled"), default=False):
        return ("off",)
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return ("off",)
        return (pet.slug, _pet_sheet_revision(pet.spritesheet), scale)
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return ("off",)


def _pet_changed_payload() -> dict:
    """``pet.info.meta``-shaped payload for ``pet.changed`` — enough for the
    renderer to decide whether the heavy sprite payload needs a refetch."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return {"enabled": False}
        return {
            "enabled": True,
            "slug": pet.slug,
            "displayName": pet.display_name,
            "scale": scale,
            "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        }
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return {"enabled": False}


def _cron_sig():
    """mtime of the profile's cron/jobs.json — moves on create/edit/pause/
    remove AND on scheduler tick bookkeeping (last_run/next_run)."""
    try:
        return (_watcher_home() / "cron" / "jobs.json").stat().st_mtime_ns
    except OSError:
        return None


def _sessions_sig():
    """Newest mtime across state.db and its WAL — the cross-process change
    signal. Messaging-gateway turns and cron runs are written by OTHER
    processes that never touch this gateway's transports; the shared SQLite
    file is the one thing they all move (#58671). A backend serving several
    profiles owns one store per profile, so every served sibling home is
    probed too — otherwise a routed profile's Bot Chat never refreshes."""
    sig = None
    for root in (_watcher_home(), *_served_profile_homes):
        for name in ("state.db", "state.db-wal"):
            try:
                mtime = (root / name).stat().st_mtime_ns
            except OSError:
                continue
            sig = mtime if sig is None else max(sig, mtime)
    return sig


def _platforms_sig():
    """mtime of gateway_state.json — the messaging gateway process persists
    platform connect/disconnect/health there, so its movement is the
    "connection status changed" signal for the Messaging page."""
    try:
        return (_watcher_home() / "gateway_state.json").stat().st_mtime_ns
    except OSError:
        return None


def _pairing_sig():
    """Newest mtime across every profile's pairing store.

    An unknown DMer's pending code is written by the messaging gateway — a
    DIFFERENT process that never touches this gateway's transports — so the
    files are the only shared signal. ``platforms.changed`` cannot stand in
    for this: it tracks connect/disconnect/health, and a pairing request
    moves nothing in gateway_state.json.
    """
    home = _watcher_home()
    sig = None
    # Global store (legacy `pairing/` and consolidated `platforms/pairing/`)
    # plus every named profile's own — the Messaging page can be scoped to any
    # of them, and a request landing in a profile store must still tick.
    roots = [home / "pairing", home / "platforms" / "pairing"]
    try:
        for profile_dir in (home / "profiles").iterdir():
            roots.append(profile_dir / "pairing")
            roots.append(profile_dir / "platforms" / "pairing")
    except OSError:
        pass

    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            # Only the pending/approved ledgers — _rate_limits.json moves on
            # every unauthorized DM, including ones that produce no new row.
            if not entry.name.endswith(("-pending.json", "-approved.json")):
                continue
            try:
                mtime = entry.stat().st_mtime_ns
            except OSError:
                continue
            sig = mtime if sig is None else max(sig, mtime)
    return sig


# Newest outbox-envelope mtime the watcher has EVER seen (monotone). A drain
# empties the outbox (rename → claimed/), and letting the signature fall back
# to None on empty would fire a spurious pending event right after every
# drain — so the signature only moves forward, on genuinely new envelopes.
_bot_relay_outbox_seen = 0


def _bot_relay_outbox_sig():
    """Newest mtime across pending bot-relay outbox envelopes (monotone).

    Envelopes are written by the AGENT process (``message_agent`` →
    ``tools.bot_relay.enqueue_envelope``) — a different process that never
    touches this gateway's transports — so the files are the only shared
    signal, exactly like the pairing store. The Desktop reacts to
    ``bot_relay.outbox.pending`` with an immediate (debounced) drain instead
    of waiting out its poll interval (#93091, motivated by #92760).
    """
    global _bot_relay_outbox_seen
    home = _watcher_home()
    root = home.parent.parent if home.parent.name == "profiles" else home
    newest = 0
    try:
        for entry in (root / "bot_relay" / "outbox").iterdir():
            if not entry.name.endswith(".json"):
                continue
            try:
                newest = max(newest, entry.stat().st_mtime_ns)
            except OSError:
                continue
    except OSError:
        pass
    if newest > _bot_relay_outbox_seen:
        _bot_relay_outbox_seen = newest
    return _bot_relay_outbox_seen or None


# Watched change signals: event → (check interval, signature fn, payload fn).
# Signatures are stat/dict-lookup cheap, same bar as the skin watcher; the
# check interval keeps the pricier probes (pet resolves the active sheet off
# disk) off the 0.5s tick.
_CHANGE_WATCHES: dict[str, tuple[float, Any, Any]] = {
    "pet.changed": (2.0, _pet_sig, _pet_changed_payload),
    "cron.changed": (1.0, _cron_sig, lambda: {}),
    "sessions.changed": (0.5, _sessions_sig, lambda: {}),
    "platforms.changed": (2.0, _platforms_sig, lambda: {}),
    "pairing.changed": (2.0, _pairing_sig, lambda: {}),
    # Cross-connection DM latency: 1s check so a queued envelope reaches the
    # Desktop's push-triggered drain fast; the Desktop's poll stays backstop.
    "bot_relay.outbox.pending": (1.0, _bot_relay_outbox_sig, lambda: {}),
}

# state.db moves on every message append during a streaming turn, and the
# gateway rewrites gateway_state.json for in-flight-count bookkeeping; the
# floor coalesces those bursts to one broadcast per window (trailing edge
# included — a floored change keeps its old signature and re-fires next tick).
_CHANGE_BROADCAST_FLOOR_S = {"sessions.changed": 2.0, "platforms.changed": 5.0}

_change_sigs: dict[str, Any] = {}
_change_checked_at: dict[str, float] = {}
_change_broadcast_at: dict[str, float] = {}


def _broadcast_watched_changes(now: float | None = None) -> None:
    """One pass over ``_CHANGE_WATCHES``: recompute due signatures, broadcast
    the events whose signature moved. First sighting seeds silently so a
    gateway boot never fires a spurious refresh storm."""
    now = time.monotonic() if now is None else now
    for event, (interval, sig_fn, payload_fn) in _CHANGE_WATCHES.items():
        if now - _change_checked_at.get(event, -interval) < interval:
            continue
        _change_checked_at[event] = now
        try:
            sig = sig_fn()
        except Exception:  # noqa: BLE001 - a broken probe must not kill the loop
            continue
        if event not in _change_sigs:
            _change_sigs[event] = sig
            continue
        if sig == _change_sigs[event]:
            continue
        floor = _CHANGE_BROADCAST_FLOOR_S.get(event, 0.0)
        if floor and now - _change_broadcast_at.get(event, -floor) < floor:
            # Floored: leave the old signature in place so the change re-fires
            # once the window opens (the trailing edge of the burst).
            continue
        _change_sigs[event] = sig
        _change_broadcast_at[event] = now
        try:
            _broadcast_global_event(event, payload_fn())
        except Exception:  # noqa: BLE001
            pass


_skin_watcher_started = False


def _ensure_skin_watcher() -> None:
    """Watch cheap on-disk signatures and broadcast change events — so a skin
    Hermes activates, a pet ``/pet`` adopts, a cron the scheduler fires, or a
    messaging turn another process writes goes live on every surface within a
    couple seconds, on its own, with no client-side poll in the loop.
    Idempotent; started at gateway.ready. (Named for its original skin-only
    duty; it is the process's one change watcher.)"""
    global _skin_watcher_started
    if _skin_watcher_started:
        return
    _skin_watcher_started = True
    _note_skin_broadcast()  # seed the baseline so only a real change repaints

    def _loop() -> None:
        while True:
            time.sleep(0.5)
            _broadcast_skin_if_changed()
            _broadcast_watched_changes()

    threading.Thread(target=_loop, name="hermes-change-watcher", daemon=True).start()


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
