"""Minimal, optional systemd ``sd_notify`` support for the gateway."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import socket


def _notify_socket() -> str:
    address = os.environ.get("NOTIFY_SOCKET", "").strip()
    return address if address and hasattr(socket, "AF_UNIX") else ""


def notify(message: str) -> bool:
    """Send one nonblocking sd_notify datagram when systemd configured it.

    Failures are non-fatal: a missing socket or an older platform must never prevent the gateway
    from starting.
    """
    if not (address := _notify_socket()) or not isinstance(message, str) or not message:
        return False
    try:
        payload = message.encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.setblocking(False)  # a full receiver buffer must not stall the event loop
            # systemd's ``@abstract`` notation -> Python's leading-NUL address form.
            sender.connect("\0" + address[1:] if address.startswith("@") else address)
            sender.send(payload)
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def watchdog_interval_seconds() -> float | None:
    raw = os.environ.get("WATCHDOG_USEC", "").strip() if _notify_socket() else ""
    try:
        interval = float(raw) / 1_000_000.0
    except (TypeError, ValueError):
        return None
    return interval if math.isfinite(interval) and interval > 0 else None


class SystemdWatchdog:
    """Feed systemd while the asyncio event loop continues to make progress."""

    def __init__(
        self, *, config_enabled: bool = True, lag_tolerance_seconds: float | None = None
    ) -> None:
        self._config_enabled = bool(config_enabled)
        self.interval_seconds = watchdog_interval_seconds()
        self._lag_tolerance_seconds = lag_tolerance_seconds
        self._task: asyncio.Task[None] | None = None
        self._unhealthy = self._stopping = self._stopping_notified = False

    enabled = property(lambda self: self._config_enabled and self.interval_seconds is not None)
    unhealthy = property(lambda self: self._unhealthy)
    task = property(lambda self: self._task)

    def _lag_tolerance(self) -> float:
        default = max(0.1, (self.interval_seconds or 0.0) * 0.25)
        try:
            value = float(self._lag_tolerance_seconds)
        except (TypeError, ValueError):
            return default
        return max(0.0, value) if math.isfinite(value) else default

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self._task is not None and not self._task.done():
            return True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._stopping = self._unhealthy = self._stopping_notified = False
        self._task = asyncio.create_task(self._run(), name="hermes-systemd-watchdog")
        return True

    def ready(self, status: str = "Gateway running") -> bool:
        safe_status = str(status or "Gateway running").replace("\n", " ")
        return self.enabled and notify(f"READY=1\nSTATUS={safe_status}")

    def record_tick(self, *, scheduled_at: float, now: float) -> bool:
        """Feed systemd only when the event loop woke within its lag budget."""
        if not self.enabled or self._stopping or self._unhealthy:
            return False
        try:
            lag = float(now) - float(scheduled_at)
        except (TypeError, ValueError):
            lag = float("inf")
        if not math.isfinite(lag) or lag > self._lag_tolerance():
            self._unhealthy = True
            notify("STATUS=watchdog unhealthy: event loop progress is late")
            return False
        notify("WATCHDOG=1")
        return True

    async def _run(self) -> None:
        if self.interval_seconds is None:
            return
        cadence = max(0.01, self.interval_seconds / 2.0)
        loop = asyncio.get_running_loop()
        scheduled_at = loop.time() + cadence
        with contextlib.suppress(asyncio.CancelledError):
            while not self._stopping and not self._unhealthy:
                await asyncio.sleep(max(0.0, scheduled_at - loop.time()))
                now = loop.time()
                if not self.record_tick(scheduled_at=scheduled_at, now=now):
                    return
                scheduled_at = (
                    scheduled_at + cadence if scheduled_at + cadence >= now else now + cadence
                )

    async def stop(self) -> None:
        """Stop feeding systemd and emit ``STOPPING=1`` at most once."""
        self._stopping = True
        task = self._task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None
        if self.enabled and not self._stopping_notified:
            notify("STOPPING=1")
            self._stopping_notified = True
