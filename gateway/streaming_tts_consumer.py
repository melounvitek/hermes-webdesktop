"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

Bridges the sync agent ``stream_delta_callback`` (worker thread) to a voice-capable adapter's
streaming-audio contract so playback begins while the LLM is still generating. ``on_delta``
never blocks (SentenceChunker -> thread-safe queue); the ``_run`` task on the gateway loop
drains, synthesises via a ``StreamingTTSProvider`` and writes PCM. Outcome: full success ->
``completed``; failure before any audible output -> ``suppress_whole_file=False`` (gateway
falls back to whole-file TTS); failure after partial audio -> ``partial`` + suppress (never
replay the response from the beginning).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle

logger = logging.getLogger("gateway.streaming_tts_consumer")

_ABORT = object()
_DONE = object()


class StreamingTTSConsumer:
    """Consumes LLM text deltas and produces streaming PCM audio for an adapter."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        tts_config: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider

        self._adapter = adapter
        self._chat_id = chat_id
        self._loop = loop
        self._metadata = metadata
        # Resolved once; None => inactive, gateway falls back to whole-file TTS.
        self._streamer = resolve_streaming_provider(tts_config)
        self._chunker = SentenceChunker()
        if self._streamer is not None:
            self._audio_format = AudioFormat(**{
                f: int(getattr(self._streamer, f, getattr(AudioFormat, f)))
                for f in ("sample_rate", "channels", "sample_width")
            })
        else:
            self._audio_format = audio_format or AudioFormat()
        # Thread-safe queue of completed clauses plus the _DONE/_ABORT sentinels.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)
        self._handle: Optional[StreamingTTSHandle] = None
        self._completed = self._partial = self._aborted = False
        self._finished = self._dropped = self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._strip_markdown = None  # lazily imported to avoid import cycles

    # usable streaming provider resolved
    active = property(lambda self: self._streamer is not None)
    # streaming audio fully delivered
    completed = property(lambda self: self._completed)
    # some audio was audible before a failure/drop
    partial = property(lambda self: self._partial)
    # first PCM chunk has been written
    audible = property(lambda self: bool(self._handle and self._handle.audible))
    # queue saturation dropped at least one clause
    dropped = property(lambda self: self._dropped)
    # gateway should skip whole-file TTS fallback
    suppress_whole_file = property(lambda self: self._suppress_whole_file)
    # async drain task has terminated
    done = property(lambda self: self._task is not None and self._task.done())

    def _enqueue_clauses(self, clauses, full_msg: str, *, log_errors: bool) -> None:
        try:
            for clause in clauses:
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug(full_msg)
        except Exception:
            if log_errors:
                logger.debug("streaming TTS on_delta error", exc_info=True)

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        self._enqueue_clauses(
            self._chunker.feed(text), "streaming TTS queue full, dropping clause", log_errors=True
        )

    def finish(self) -> None:
        """Signal end-of-text, flush the chunker tail, then enqueue ``_DONE`` after all flushed
        clauses so the drain loop terminates deterministically without racing a late ``on_delta``.
        """
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        self._enqueue_clauses(
            self._chunker.flush(), "streaming TTS queue full while flushing tail", log_errors=False
        )
        # The load-bearing _DONE sentinel must never be lost: evict a clause if full.
        while True:
            try:
                self._queue.put_nowait(_DONE)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._dropped = True
                except queue.Empty:
                    continue

    def start(self) -> asyncio.Task:
        """Create (once) and return the async drain task on the gateway loop."""
        if self._task is None:
            self._task = self._loop.create_task(self._run())
        return self._task

    def _settle(self, *, failed: bool) -> None:
        """Set outcome flags from what was audible: never report completion after a failure or a
        dropped clause; keep suppression whenever audio was audible (no replay from the start)."""
        audible = self._handle.audible
        degraded = failed or self._dropped
        self._completed = audible and not degraded
        if audible and degraded:
            self._partial = True
        self._suppress_whole_file = audible

    async def _run(self) -> None:
        """Drain clauses from the queue, synthesise, and write to the adapter."""
        if not self.active:
            return
        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            logger.debug("adapter %s does not support streaming TTS", getattr(self._adapter, "name", "?"))
            return
        try:
            self._handle = await self._adapter.begin_streaming_tts(
                self._chat_id, self._audio_format, metadata=self._metadata
            )
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            self._handle = None
            return
        if self._handle is None:
            return
        self._suppress_whole_file = False
        try:
            while not self._aborted:
                try:
                    item = await asyncio.to_thread(self._queue.get, True, 0.1)
                except queue.Empty:
                    continue
                if item is _ABORT or item is _DONE or self._aborted:
                    break
                if not isinstance(item, str):
                    continue
                try:
                    await self._synthesise_and_write(item)
                except Exception as exc:
                    logger.warning("streaming TTS clause failed: %s", exc)
                    self._settle(failed=True)
                    await self._safe_abort(str(exc))
                    return
            if not self._aborted and self._handle is not None:
                try:
                    await self._adapter.finish_streaming_tts(self._handle, interrupted=self._aborted)
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    self._settle(failed=True)
                    await self._safe_abort("finish_streaming_tts failed")
                else:
                    self._settle(failed=False)
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            with contextlib.suppress(Exception):
                while not self._queue.empty():
                    self._queue.get_nowait()

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        if self._handle is None or self._handle.aborted or self._streamer is None:
            return
        cleaned = self._strip_markdown_for_tts(clause)
        if not cleaned.strip():
            return
        iterator = iter(self._streamer.stream(cleaned))
        while True:
            # next() runs in a thread so a blocking provider never stalls the loop.
            chunk = await asyncio.to_thread(next, iterator, _DONE)
            if chunk is _DONE or self._aborted or self._handle.aborted:
                return
            if not chunk:
                continue
            was_audible = self._handle.audible
            await self._adapter.write_streaming_tts(self._handle, chunk)
            if not was_audible:
                self._handle.audible = True
                self._suppress_whole_file = True

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Lazy-import and apply the TTS markdown stripper."""
        if self._strip_markdown is None:
            try:
                from tools.tts_tool import _strip_markdown_for_tts as _strip
                self._strip_markdown = _strip
            except ImportError:
                self._strip_markdown = lambda t: t  # noqa: E731
        return self._strip_markdown(text).strip()

    async def _safe_abort(self, reason: str) -> None:
        """Abort the adapter stream, swallowing errors (idempotent)."""
        if self._handle is None:
            return
        try:
            with contextlib.suppress(Exception):
                await self._adapter.abort_streaming_tts(self._handle, error=reason)
        finally:
            if self._handle:
                self._handle.aborted = True

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # The load-bearing _ABORT sentinel must reach the queue even when full: evict to make room.
        for _attempt in range(3):
            try:
                self._queue.put_nowait(_ABORT)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        else:
            logger.debug("streaming TTS _ABORT sentinel could not be enqueued")
        if self._handle is not None and not self._handle.aborted:
            with contextlib.suppress(Exception):
                self._loop.call_soon_threadsafe(asyncio.create_task, self._safe_abort(reason))

    async def wait_complete(self, timeout: float = 10.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        return self._completed
