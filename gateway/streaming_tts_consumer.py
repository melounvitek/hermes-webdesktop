"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

Bridges the synchronous agent ``stream_delta_callback`` (worker thread) to a
voice-capable adapter's streaming-audio contract so playback begins while the
LLM is still generating.

Lifecycle::

    consumer = StreamingTTSConsumer(adapter, chat_id, tts_config, loop, metadata)
    agent.stream_delta_callback = consumer.on_delta   # sync, non-blocking
    ... agent runs in executor ...
    consumer.finish()                                 # signal end-of-text
    success = await consumer.wait_complete(timeout=10)
    if consumer.suppress_whole_file: ...              # skip whole-file auto-TTS
    consumer.abort("cancelled")                       # idempotent cancellation

``on_delta`` never blocks: it feeds a ``SentenceChunker`` and queues clauses on a
thread-safe ``queue.Queue``; the ``_run`` task on the gateway loop drains it,
synthesises via a ``StreamingTTSProvider`` and writes PCM to the adapter. State
is per instance (concurrent chats cannot cross-contaminate); abort is idempotent
and late chunks are dropped. Outcome contract: full success -> ``completed``;
failure before any audible output -> ``suppress_whole_file=False`` (gateway falls
back to whole-file TTS); failure after partial audio -> ``partial`` and
``suppress_whole_file=True`` (never replay the response from the beginning).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle
import contextlib

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
            self._audio_format = AudioFormat(
                sample_rate=int(getattr(self._streamer, "sample_rate", AudioFormat.sample_rate)),
                channels=int(getattr(self._streamer, "channels", AudioFormat.channels)),
                sample_width=int(getattr(self._streamer, "sample_width", AudioFormat.sample_width)),
            )
        else:
            self._audio_format = audio_format or AudioFormat()

        # Thread-safe queue of completed clauses plus the _DONE/_ABORT sentinels.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)

        self._handle: Optional[StreamingTTSHandle] = None
        self._completed = False
        self._partial = False
        self._aborted = False
        self._finished = False
        self._dropped = False
        self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._strip_markdown = None  # lazily imported to avoid import cycles

    @property
    def active(self) -> bool:  # usable streaming provider resolved
        return self._streamer is not None

    @property
    def completed(self) -> bool:  # streaming audio fully delivered
        return self._completed

    @property
    def partial(self) -> bool:  # some audio was audible before a failure/drop
        return self._partial

    @property
    def audible(self) -> bool:  # first PCM chunk has been written
        return bool(self._handle and self._handle.audible)

    @property
    def dropped(self) -> bool:  # queue saturation dropped at least one clause
        return self._dropped

    @property
    def suppress_whole_file(self) -> bool:  # gateway should skip whole-file TTS fallback
        return self._suppress_whole_file

    @property
    def done(self) -> bool:  # async drain task has terminated
        return self._task is not None and self._task.done()

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        try:
            for clause in self._chunker.feed(text):
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full, dropping clause")
        except Exception:
            logger.debug("streaming TTS on_delta error", exc_info=True)

    def finish(self) -> None:
        """Signal end-of-text, flush the chunker tail, then enqueue ``_DONE``.

        The sentinel follows all flushed clauses so the drain loop has a
        deterministic termination that cannot race a late ``on_delta``.
        """
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        try:
            for clause in self._chunker.flush():
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full while flushing tail")
        except Exception:
            pass
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
        """Set the outcome flags from what was audible.

        Never report completion after a failure or a dropped clause; keep
        suppression whenever audio was audible so the gateway does not replay
        the response from the beginning.
        """
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
                self._chat_id, self._audio_format, metadata=self._metadata,
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
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

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
            if chunk is _DONE:
                return
            if self._aborted or self._handle.aborted:
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
            await self._adapter.abort_streaming_tts(self._handle, error=reason)
        except Exception:
            pass
        finally:
            if self._handle:
                self._handle.aborted = True

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # The _ABORT sentinel is load-bearing and must reach the queue even when
        # the bounded queue is full: evict an item to make room.
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
