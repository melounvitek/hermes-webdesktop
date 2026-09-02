"""Transport-neutral browser-control broker core.

Binds an identity-scoped *controller* (the party driving a browser) to
*callers* (agents talking to it over any transport) without knowing HTTP,
WebSocket, or any wire format. The browser is a stateful single-owner
resource while the agent side is multi-tenant and multi-transport, so every
rule here makes a class of violation structurally impossible:

- Registration tickets are short-lived, single-use, identity-bound, and
  ``secrets``-random; ``consume_ticket`` exchanges one exactly once. The
  ticket is the only cross-transport credential minted here; transports
  decide how to carry it.
- ``select`` matches on *every* stable identity field (principal, profile,
  session, controller id, browser profile id, transport family) plus the
  controller's currently negotiated capability set; partial matches yield
  ``None``. A same-identity reconnect renegotiates capabilities in place.
- One pending command per command id; ``complete`` is single-shot (late
  completion after cancel/detach returns ``False``).
- ``cancel`` aborts only the command matching scope + tool_call_id.
- ``detach`` fails closed: pending work of that scope raises
  :class:`ControllerCancelled` instead of racing a late ``complete``.
- ``disconnect`` marks a transport offline without cancelling running work;
  re-attaching the same identity refreshes the callback and may still
  deliver the original result. Detach, cancel, identity replacement, and
  timeout remain terminal.

Thread-safety: all state transitions happen under one reentrant lock; the
send callback is invoked *outside* it so a controller may synchronously
``complete`` from inside its own send, and waiters park on per-command
events, never on the broker lock.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)
_OWNER_UNSET = object()

#: Default lifetime of a minted registration ticket, in clock seconds.
DEFAULT_TICKET_TTL = 30.0
#: Default wall time a dispatch waits for the controller to complete.
DEFAULT_COMMAND_TIMEOUT = 30.0
#: Maximum cancel frames retained while a same-identity controller is offline.
MAX_DEFERRED_CANCELS = 512

#: Current wire protocol version. Registration requires this exact integer;
#: booleans are rejected even though ``bool`` subclasses ``int``.
BROWSER_CONTROL_PROTOCOL_VERSION = 1

#: Exact controller capability allowlist shared by every transport. Raw CDP,
#: script evaluation, console access, and other privileged surfaces stay out.
BROWSER_CONTROL_CAPABILITIES = frozenset(
    {
        "controller.noop",
        "browser_back",
        "browser_click",
        "browser_navigate",
        "browser_press",
        "browser_screenshot",
        "browser_scroll",
        "browser_snapshot",
        "browser_tab_activate",
        "browser_tabs",
        "browser_type",
    }
)

#: Privileged capabilities (JS eval, raw CDP): fail-closed unless Developer
#: Mode (``browser.extension_control.developer_mode``) is on AND the
#: controller explicitly negotiated them.
BROWSER_CONTROL_DEVELOPER_CAPABILITIES = frozenset({"browser_cdp", "browser_evaluate"})

#: Artifact transport capabilities. Non-developer because payloads never ride
#: in controller frames: only a store-validated ``artifact_id`` is dispatched.
BROWSER_CONTROL_ARTIFACT_CAPABILITIES = frozenset(
    {"browser_artifact_download", "browser_artifact_upload"}
)

#: Wire method names for controller frames; transports carry them verbatim.
FRAME_COMMAND = "browser.controller.command"
FRAME_CANCEL = "browser.controller.cancel"


def browser_control_protocol_supported(value: Any) -> bool:
    """Return whether ``value`` names the exact supported wire version."""
    return type(value) is int and value == BROWSER_CONTROL_PROTOCOL_VERSION


def _extension_control_flag(config: Optional[dict], key: str) -> bool:
    """Read ``browser.extension_control.<key>`` as a literal ``True`` (default off)."""
    if config is None:
        try:
            # Hot path (every browser tool call / check_fn): the read-only
            # loader skips load_config()'s deepcopy; we never mutate.
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            return False
    if not isinstance(config, dict):
        return False
    browser = config.get("browser")
    if not isinstance(browser, dict):
        return False
    extension_control = browser.get("extension_control")
    if not isinstance(extension_control, dict):
        return False
    return extension_control.get(key, False) is True


def browser_control_developer_mode(config: Optional[dict] = None) -> bool:
    """Explicit Developer Mode flag; gates ``browser_evaluate``/raw CDP only."""
    return _extension_control_flag(config, "developer_mode")


def browser_control_enabled(config: Optional[dict] = None) -> bool:
    """Return the explicit browser-control feature flag (disabled by default)."""
    return _extension_control_flag(config, "enabled")


def filter_browser_control_capabilities(
    value: Any,
    *,
    developer_mode: Optional[bool] = None,
) -> frozenset:
    """Return the permitted subset of a JSON/RPC capability list.

    Non-list input has no capabilities; unknown/non-string entries are
    dropped (registration rejects an empty returned set). Developer
    capabilities pass only when Developer Mode is on (passed in, else read
    from live config).
    """
    if not isinstance(value, list):
        return frozenset()
    allowed = BROWSER_CONTROL_CAPABILITIES | BROWSER_CONTROL_ARTIFACT_CAPABILITIES
    if developer_mode is None:
        developer_mode = browser_control_developer_mode()
    if developer_mode is True:
        allowed = allowed | BROWSER_CONTROL_DEVELOPER_CAPABILITIES
    return frozenset(c for c in value if isinstance(c, str) and c in allowed)


class BrowserControlError(Exception):
    """Base class for broker contract failures."""


class ControllerTicketInvalid(BrowserControlError):
    """A registration ticket is unknown, already consumed, or expired."""


class ControllerUnavailable(BrowserControlError):
    """No attached controller exactly matches the requested scope/capability."""


class ControllerCancelled(BrowserControlError):
    """A pending command was cancelled (explicitly or by detach)."""


class ControllerTimeout(BrowserControlError):
    """The controller did not complete the command before the timeout."""


class ControllerRejected(BrowserControlError):
    """The controller completed the command with ``ok=False``."""


@dataclass(frozen=True)
class ControllerScope:
    """Exact controller identity plus capability set; equality is over all fields."""

    principal_id: Optional[str] = None
    profile_id: Optional[str] = None
    session_id: Optional[str] = None
    controller_id: Optional[str] = None
    browser_profile_id: Optional[str] = None
    transport_family: Optional[str] = None
    capabilities: frozenset = frozenset()


def _scope_identity(scope: ControllerScope) -> tuple:
    """Stable identity fields, excluding negotiated capabilities."""
    return (
        scope.principal_id,
        scope.profile_id,
        scope.session_id,
        scope.controller_id,
        scope.browser_profile_id,
        scope.transport_family,
    )


def _same_scope_identity(first: ControllerScope, second: ControllerScope) -> bool:
    return _scope_identity(first) == _scope_identity(second)


@dataclass(frozen=True)
class Ticket:
    """Opaque, single-use registration credential."""

    value: str
    expires_at: float


@dataclass
class _TicketRecord:
    scope: ControllerScope
    expires_at: float
    consumed: bool = False


@dataclass
class _Controller:
    scope: ControllerScope
    send: Callable[[dict], None]
    owner: Any = None
    connected: bool = True
    deferred_cancels: list[dict] = field(default_factory=list)
    # Serializes command/cancel writes with detach or replacement. Broker
    # state is never held while waiting on it, so a transport callback may
    # synchronously complete() without deadlocking.
    send_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _PendingCommand:
    scope: ControllerScope
    command_id: str
    tool_call_id: Optional[str]
    event: threading.Event = field(default_factory=threading.Event)
    done: bool = False
    cancelled: bool = False
    ok: bool = False
    result: Any = None


class BrowserControlBroker:
    """Thread-safe broker core binding controllers to callers.

    ``clock`` is injectable (default ``time.monotonic``) so tests pin expiry.
    """

    def __init__(
        self,
        *,
        ticket_ttl: float = DEFAULT_TICKET_TTL,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        clock: Optional[Callable[[], float]] = None,
        developer_mode: Optional[bool] = None,
    ) -> None:
        self._ticket_ttl = ticket_ttl
        self._command_timeout = command_timeout
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.RLock()
        self._tickets: Dict[str, _TicketRecord] = {}
        self._controllers: Dict[ControllerScope, _Controller] = {}
        self._pending: Dict[str, _PendingCommand] = {}
        # None defers to live config on every selection so flipping
        # developer_mode off REVOKES raw CDP/eval from attached controllers
        # without restart; an explicit bool pins the gate (tests, multi-tenant).
        self._developer_mode_pinned: Optional[bool] = (
            None if developer_mode is None else developer_mode is True
        )
        # Artifact stores keyed by profile id; ``None`` is the default slot.
        # Per-profile stores keep multiplex profile A from pinning profile B
        # to A's physical root.
        self._artifact_stores: Dict[Optional[str], Any] = {}

    def _developer_mode_now(self) -> bool:
        """Current Developer Mode authority (live config unless pinned)."""
        if self._developer_mode_pinned is not None:
            return self._developer_mode_pinned
        try:
            return browser_control_developer_mode()
        except Exception:
            return False

    @property
    def developer_mode(self) -> bool:
        """Whether privileged capabilities may be selected/dispatched."""
        return self._developer_mode_now()

    def attach_artifact_store(
        self, store: Any, *, profile_id: Optional[str] = None
    ) -> None:
        """Attach a store exposing ``validate(artifact_id, *, scope) -> receipt``
        (raising the artifacts module's :class:`ArtifactError` subclasses).

        ``profile_id`` scopes the store to one profile on multiplex hosts;
        ``None`` registers the default store. ``store=None`` clears the slot.
        Artifact actions without a resolvable store fail closed.
        """
        if store is None:
            self._artifact_stores.pop(profile_id, None)
            return
        self._artifact_stores[profile_id] = store

    def _artifact_store_for_scope(self, scope: "ControllerScope") -> Any:
        """Exact profile-scoped store, else the default (``None``) slot so
        single-profile hosts keep the historical one-store behaviour."""
        profile = getattr(scope, "profile_id", None) or None
        store = self._artifact_stores.get(profile)
        return store if store is not None else self._artifact_stores.get(None)

    # ------------------------------------------------------------------
    # Registration tickets
    # ------------------------------------------------------------------

    def mint_ticket(self, scope: ControllerScope) -> Ticket:
        """Mint a short-lived, single-use ticket bound to ``scope``."""
        now = self._clock()
        with self._lock:
            for stale in [v for v, rec in self._tickets.items() if rec.expires_at <= now]:
                del self._tickets[stale]
            value = secrets.token_urlsafe(32)
            record = _TicketRecord(scope=scope, expires_at=now + self._ticket_ttl)
            self._tickets[value] = record
        return Ticket(value=value, expires_at=record.expires_at)

    def consume_ticket(self, value: str) -> ControllerScope:
        """Exchange a ticket for its scope exactly once.

        Raises :class:`ControllerTicketInvalid` for unknown, already-consumed,
        or expired tickets; expiry is checked against the live clock at
        consume time.
        """
        now = self._clock()
        with self._lock:
            record = self._tickets.get(value)
            if record is None:
                raise ControllerTicketInvalid("unknown ticket")
            if record.consumed:
                raise ControllerTicketInvalid("ticket already consumed")
            if now > record.expires_at:
                raise ControllerTicketInvalid("ticket expired")
            record.consumed = True
            return record.scope

    # ------------------------------------------------------------------
    # Controller registration / selection
    # ------------------------------------------------------------------

    def _controller_for_identity_locked(self, scope: ControllerScope) -> Optional[_Controller]:
        """Attached controller sharing ``scope``'s stable identity (any capabilities)."""
        return next(
            (c for c in self._controllers.values() if _same_scope_identity(c.scope, scope)),
            None,
        )

    def _lane_scopes_locked(self, session_id: str, principal_id: str, family: str) -> list[ControllerScope]:
        """Attached scopes bound to one session lane (session + principal + transport)."""
        return [
            scope
            for scope in self._controllers
            if scope.session_id == session_id
            and scope.principal_id == principal_id
            and scope.transport_family == family
        ]

    def attach(
        self,
        scope: ControllerScope,
        send: Callable[[dict], None],
        *,
        owner: Any = None,
    ) -> None:
        """Attach or refresh the controller for one stable identity.

        A same-identity reconnect refreshes the send callback and negotiated
        capabilities without cancelling pending work. A different controller
        or browser profile in the same authenticated session lane
        hard-replaces the previous identity.
        """
        while True:
            with self._lock:
                existing = self._controller_for_identity_locked(scope)
                lane_scopes = [
                    candidate
                    for candidate in self._controllers
                    if candidate.principal_id == scope.principal_id
                    and candidate.profile_id == scope.profile_id
                    and candidate.session_id == scope.session_id
                    and candidate.transport_family == scope.transport_family
                    and not _same_scope_identity(candidate, scope)
                ]
                if existing is None and not lane_scopes:
                    self._controllers[scope] = _Controller(scope=scope, send=send, owner=owner)
                    return

            # Hard replacement, not a recoverable reconnect: terminalize the
            # lane siblings before inserting so session lookup stays unique.
            if lane_scopes:
                for lane_scope in lane_scopes:
                    self.detach(lane_scope, notify_controller=False)
                continue

            with existing.send_lock:
                with self._lock:
                    if self._controllers.get(existing.scope) is not existing:
                        continue
                    self._controllers.pop(existing.scope, None)
                    existing.scope = scope
                    existing.send = send
                    existing.owner = owner
                    existing.connected = False
                    for pending in self._pending.values():
                        if _same_scope_identity(pending.scope, scope):
                            pending.scope = scope
                    deferred = list(existing.deferred_cancels)
                    existing.deferred_cancels.clear()
                    self._controllers[scope] = existing

                unsent: list[dict] = []
                for index, frame in enumerate(deferred):
                    try:
                        send(frame)
                    except Exception:
                        logger.exception("failed to flush deferred browser-controller cancel")
                        unsent = deferred[index:]
                        break
                if unsent:
                    with self._lock:
                        if self._controllers.get(scope) is existing:
                            existing.deferred_cancels = unsent[-MAX_DEFERRED_CANCELS:]
                    raise ConnectionError(
                        "browser controller reconnect could not flush deferred cancels"
                    )
                with self._lock:
                    if self._controllers.get(scope) is existing:
                        existing.connected = True
                return

    def select(self, scope: ControllerScope, capability: str) -> Optional[_Controller]:
        """Return the connected controller matching identity and capability.

        The caller's capabilities are not authoritative: identity is matched,
        then the controller's *current* negotiated set is checked. Offline
        controllers never accept new dispatches. Developer capabilities are
        additionally gated on the LIVE Developer Mode flag (unless pinned).
        """
        if capability in BROWSER_CONTROL_DEVELOPER_CAPABILITIES and not self._developer_mode_now():
            return None
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        if controller is None or not controller.connected:
            return None
        return controller if capability in controller.scope.capabilities else None

    def is_owner(self, scope: ControllerScope, owner: Any) -> bool:
        """Whether ``owner`` is the exact live transport for ``scope``.

        Independent of capabilities, so a least-privilege controller need not
        request ``controller.noop`` merely to heartbeat or complete an action.
        """
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        return controller is not None and controller.connected and controller.owner is owner

    def disconnect(
        self,
        scope: ControllerScope,
        *,
        owner: Any = _OWNER_UNSET,
    ) -> bool:
        """Mark one exact controller transport offline without cancelling work."""
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        if controller is None:
            return False
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(controller.scope) is not controller:
                    return False
                if owner is not _OWNER_UNSET and controller.owner is not owner:
                    return False
                controller.connected = False
                controller.owner = None
        return True

    def detach(
        self,
        scope: ControllerScope,
        *,
        owner: Any = _OWNER_UNSET,
        notify_controller: bool = True,
    ) -> None:
        """Remove the controller for ``scope`` and fail its pending work closed.

        Pending commands resolve cancelled (dispatchers raise
        :class:`ControllerCancelled`); a late ``complete`` returns ``False``.
        """
        with self._lock:
            controller = self._controllers.get(scope)
        if controller is None:
            return
        if owner is not _OWNER_UNSET and controller.owner != owner:
            return
        with controller.send_lock:
            with self._lock:
                if self._controllers.get(scope) is not controller:
                    return
                if owner is not _OWNER_UNSET and controller.owner != owner:
                    return
                self._controllers.pop(scope, None)
                pendings = self._pending_for_scope_locked(scope)
                for pending in pendings:
                    self._resolve_pending(pending, cancelled=True)
            # Hold the old generation's send lock through cancellation so a
            # command frame can never overtake its terminal cancel frame.
            if notify_controller:
                self._emit_cancel_frames(controller, pendings)

    # ------------------------------------------------------------------
    # Command lifecycle
    # ------------------------------------------------------------------

    def dispatch(
        self,
        scope: ControllerScope,
        *,
        action: str,
        arguments: Optional[dict] = None,
        tool_call_id: Optional[str] = None,
    ) -> Any:
        """Send one controller command and block for its completion.

        Raises :class:`ControllerUnavailable` (no exact match),
        :class:`ControllerCancelled`, :class:`ControllerTimeout`, or
        :class:`ControllerRejected` (``ok=False``). Artifact actions also
        require an attached store and an approved ``artifact_id`` argument;
        only the id travels in the frame, never the payload.
        """
        controller = self.select(scope, action)
        if controller is None:
            raise ControllerUnavailable(
                f"no controller for scope {scope!r} with capability {action!r}"
            )

        arguments = dict(arguments or {})
        if action in BROWSER_CONTROL_ARTIFACT_CAPABILITIES:
            self._validate_artifact_reference(scope, action, arguments)

        command_id = secrets.token_hex(16)
        frame = {
            "method": FRAME_COMMAND,
            "params": {
                "command_id": command_id,
                "action": action,
                "arguments": arguments,
                "controller_id": scope.controller_id,
                "browser_profile_id": scope.browser_profile_id,
                "tool_call_id": tool_call_id,
            },
        }
        pending = _PendingCommand(
            scope=controller.scope,
            command_id=command_id,
            tool_call_id=tool_call_id,
        )
        with controller.send_lock:
            with self._lock:
                # select() ran outside the send lock; revalidate the live
                # controller so disconnect/replacement can't strand a command.
                attached = self._controller_for_identity_locked(scope)
                if attached is not controller or not controller.connected:
                    raise ControllerUnavailable(
                        f"controller for scope {scope!r} detached before dispatch"
                    )
                pending.scope = controller.scope
                self._pending[command_id] = pending

            try:
                controller.send(frame)
            except Exception:
                # Never left the building: unreserve the id, surface the error.
                with self._lock:
                    self._pending.pop(command_id, None)
                raise

        if not pending.event.wait(timeout=self._command_timeout):
            timed_out = False
            with self._lock:
                # Event.wait() may return False at the exact boundary where a
                # completion already won and removed the pending command.
                if not pending.done and self._pending.get(command_id) is pending:
                    pending.done = True
                    del self._pending[command_id]
                    timed_out = True
            if timed_out:
                with controller.send_lock:
                    with self._lock:
                        active = self._controller_for_identity_locked(scope) or controller
                        if not active.connected:
                            self._defer_cancel_locked(active, pending)
                            active = None
                    if active is not None:
                        self._emit_cancel_frames(active, [pending])
                raise ControllerTimeout(
                    f"controller did not complete command {command_id!r} "
                    f"within {self._command_timeout}s"
                )

        if pending.cancelled:
            raise ControllerCancelled(f"command {command_id!r} was cancelled")
        if not pending.ok:
            raise ControllerRejected(
                f"controller rejected command {command_id!r}: {pending.result!r}"
            )
        return pending.result

    def complete(
        self,
        command_id: str,
        *,
        scope: Optional[ControllerScope] = None,
        ok: bool,
        result: Any = None,
    ) -> bool:
        """Resolve a pending command by id; ``False`` when none is pending.

        Safe from inside the controller's own ``send`` callback. Late
        completions after ``cancel``/``detach`` are ignored (``False``).
        """
        with self._lock:
            pending = self._pending.get(command_id)
            if pending is None or pending.done:
                return False
            if scope is not None and pending.scope != scope:
                return False
            pending.done = True
            pending.ok = ok is True
            pending.result = result
            del self._pending[command_id]
            pending.event.set()
        return True

    def cancel(self, scope: ControllerScope, *, tool_call_id: Optional[str]) -> bool:
        """Cancel exactly the pending command matching ``scope`` + tool_call_id.

        Emits one cancel frame; returns ``False`` when nothing matched so
        transports can answer idempotently.
        """
        with self._lock:
            controller = self._controller_for_identity_locked(scope)
        if controller is None or not controller.connected:
            return False
        with controller.send_lock:
            with self._lock:
                attached = self._controller_for_identity_locked(scope)
                if attached is not controller or not controller.connected:
                    return False
                target = next(
                    (
                        p
                        for p in self._pending.values()
                        if _same_scope_identity(p.scope, scope)
                        and p.tool_call_id == tool_call_id
                        and not p.done
                    ),
                    None,
                )
                if target is None:
                    return False
                self._resolve_pending(target, cancelled=True)
            self._emit_cancel_frames(controller, [target])
            return True

    # ------------------------------------------------------------------
    # Internals (callers hold the lock unless noted)
    # ------------------------------------------------------------------

    def _resolve_pending(self, pending: _PendingCommand, *, cancelled: bool) -> None:
        pending.cancelled = cancelled
        pending.done = True
        del self._pending[pending.command_id]
        pending.event.set()

    def _validate_artifact_reference(
        self,
        scope: ControllerScope,
        action: str,
        arguments: dict,
    ) -> None:
        """Fail closed unless ``arguments`` carries a store-approved artifact id.

        Missing store/id, traversal, expiry, checksum, or scope mismatch all
        surface as :class:`ControllerRejected` before any frame is emitted.
        """
        store = self._artifact_store_for_scope(scope)
        if store is None:
            raise ControllerRejected(f"{action} requires an attached artifact store")
        artifact_id = arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ControllerRejected(f"{action} requires a non-empty artifact_id")
        try:
            store.validate(artifact_id.strip(), scope=scope)
        except ControllerRejected:
            raise
        except Exception as exc:
            raise ControllerRejected(
                f"{action} rejected artifact reference {artifact_id!r}: {exc}"
            ) from exc

    @staticmethod
    def _cancel_frame(pending: _PendingCommand) -> dict:
        return {
            "method": FRAME_CANCEL,
            "params": {
                "command_id": pending.command_id,
                "tool_call_id": pending.tool_call_id,
            },
        }

    def _defer_cancel_locked(self, controller: _Controller, pending: _PendingCommand) -> None:
        controller.deferred_cancels.append(self._cancel_frame(pending))
        if len(controller.deferred_cancels) > MAX_DEFERRED_CANCELS:
            del controller.deferred_cancels[:-MAX_DEFERRED_CANCELS]

    def _pending_for_scope_locked(self, scope: ControllerScope) -> list[_PendingCommand]:
        return [p for p in list(self._pending.values()) if _same_scope_identity(p.scope, scope)]

    def _emit_cancel_frames(self, controller: _Controller, pendings: list[_PendingCommand]) -> None:
        """Send cancel frames (caller holds ``send_lock``, never the broker lock)."""
        for pending in pendings:
            try:
                controller.send(self._cancel_frame(pending))
            except Exception:
                logger.exception(
                    "failed to emit cancel frame for command %r", pending.command_id
                )

    # ------------------------------------------------------------------
    # Session-lane lookups / bulk teardown
    # ------------------------------------------------------------------

    @staticmethod
    def _lane_key(session_id, task_id, principal_id, transport_family) -> Optional[tuple[str, str, str]]:
        target = str(session_id or task_id or "").strip()
        principal = str(principal_id or "").strip()
        family = str(transport_family or "").strip()
        if not target or not principal or not family:
            return None
        return target, principal, family

    def scope_for_session(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        transport_family: Optional[str] = None,
    ) -> Optional[ControllerScope]:
        """Return one unambiguous attached scope for a server-owned session.

        The public session id is only a hint; the caller must also supply its
        server-derived principal and transport family. Missing identity, no
        match, or multiple matches fail closed rather than picking by order.
        """
        key = self._lane_key(session_id, task_id, principal_id, transport_family)
        if key is None:
            return None
        with self._lock:
            matches = self._lane_scopes_locked(*key)
        return matches[0] if len(matches) == 1 else None

    def lane_registered(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        transport_family: Optional[str] = None,
    ) -> bool:
        """Whether ANY controller (even offline) registered for this lane.

        Distinguishes "lane bound but currently unavailable" (fail closed;
        the extension lane stays authoritative) from "never registered"
        (caller keeps the legacy backend). Ambiguous lanes report True.
        """
        key = self._lane_key(session_id, task_id, principal_id, transport_family)
        if key is None:
            return False
        with self._lock:
            return bool(self._lane_scopes_locked(*key))

    def disconnect_owner(self, owner: Any) -> int:
        """Mark every controller owned by one lost transport offline."""
        with self._lock:
            scopes = [s for s, c in self._controllers.items() if c.owner is owner]
        return sum(int(self.disconnect(scope, owner=owner)) for scope in scopes)

    def reset(self) -> None:
        """Fail all live work closed and clear tickets (tests/shutdown)."""
        with self._lock:
            scopes = list(self._controllers)
        for scope in scopes:
            self.detach(scope)
        with self._lock:
            self._tickets.clear()
            # Pending entries whose controller a concurrent teardown removed.
            for pending in list(self._pending.values()):
                self._resolve_pending(pending, cancelled=True)

    @property
    def ticket_ttl_seconds(self) -> float:
        """Configured lifetime for newly minted one-shot tickets."""
        return self._ticket_ttl

    @property
    def pending_count(self) -> int:
        """Number of commands awaiting completion (diagnostics/tests)."""
        with self._lock:
            return len(self._pending)


_GLOBAL_BROKER = BrowserControlBroker()


def get_browser_control_broker() -> BrowserControlBroker:
    """Process-local broker shared by API and dashboard Gateway transports."""
    return _GLOBAL_BROKER
