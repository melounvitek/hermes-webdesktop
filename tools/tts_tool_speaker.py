"""Speaker-side streaming pipeline for ``tools.tts_tool.stream_tts_to_speaker``.

Turns a queue of LLM text deltas into audio the moment each sentence is
complete. Two paths share the sentence cutter (``tools.tts_streaming``):

* :class:`_StreamerPlayback` — a registered chunked streamer (ElevenLabs,
  OpenAI, …). Every sentence gets a prefetch thread that fires the HTTP
  request immediately and buffers PCM into a per-sentence queue; one playback
  worker drains those queues in FIFO order through a sounddevice OutputStream
  (or a temp WAV + system player when PortAudio is unavailable).
* :class:`_SyncSentencePipeline` — every other provider (edge, piper,
  plugins). Per-sentence ``text_to_speech_tool`` synthesis on a single-thread
  executor, overlapped with playback so sentence n+1 synthesizes while n plays.

Seams tests monkeypatch on the origin module (``_load_tts_config``,
``_import_sounddevice``, ``text_to_speech_tool``, ``_strip_markdown_for_tts``)
are resolved through :func:`_origin` at call time.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, List, Optional

logger = logging.getLogger("tools.tts_tool")


def _origin():
    from tools import tts_tool

    return tts_tool


def _unlink_quietly(path: Optional[str]) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _align_int16_chunks(chunks: Iterable[bytes], stop_evt: threading.Event) -> Iterator[bytes]:
    """Yield int16-aligned byte chunks; a dangling odd byte is padded at the end."""
    leftover = b""
    for chunk in chunks:
        if stop_evt.is_set():
            break
        buf = leftover + chunk
        aligned_len = len(buf) - (len(buf) % 2)
        if aligned_len >= 2:
            yield buf[:aligned_len]
        leftover = buf[aligned_len:] if aligned_len < len(buf) else b""
    if leftover:
        yield b"\x00"


def _play_via_tempfile(audio_iter: Iterable[bytes], stop_evt: threading.Event, sample_rate: int = 24000) -> None:
    """Write PCM chunks to a temp WAV file and play it with the system player."""
    tmp = None
    tmp_path = None
    try:
        import wave
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            for aligned in _align_int16_chunks(audio_iter, stop_evt):
                wf.writeframes(aligned)
        # wave.open() on a file object does NOT close it. On Windows the open
        # write handle blocks the player and the unlink below (WinError 32),
        # so release it before playback.
        tmp.close()
        from tools.voice_mode import play_audio_file
        play_audio_file(tmp_path)
    except Exception as exc:
        logger.warning("Temp-file TTS fallback failed: %s", exc)
    finally:
        if tmp is not None:
            try:
                tmp.close()  # idempotent; ensures close on early error
            except Exception:
                pass
        _unlink_quietly(tmp_path)


def _drain_chunks(chunk_queue: "queue.Queue[Optional[bytes]]") -> List[bytes]:
    """Collect one sentence's PCM chunks up to the ``None`` sentinel."""
    chunks: List[bytes] = []
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            return chunks
        chunks.append(chunk)


class _SyncSentencePipeline:
    """Overlap per-sentence synthesis with playback for non-streaming providers.

    Serial synthesize-then-play added a full synthesis-time of dead air per
    sentence — for a local model at real-time-factor ~1, as long silent as
    speaking. One single-thread synthesis executor (sentences FIFO; providers
    never see concurrent calls) feeds one playback worker through a small
    bounded queue: while sentence n plays, n+1 is already synthesizing. The
    bound keeps lookahead/temp files small and gives the caller backpressure.

    ``text_to_speech_tool`` / ``play_audio_file`` are resolved late so tests
    that monkeypatch them keep working.
    """

    def __init__(self, stop_event: threading.Event, *, lookahead: int = 2):
        self._stop = stop_event
        self._queue: "queue.Queue[Optional[tuple[str, Future]]]" = queue.Queue(maxsize=max(1, lookahead))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-sync-synth")
        self._player = threading.Thread(target=self._drain, name="tts-sync-play", daemon=True)
        self._player.start()

    def speak(self, cleaned: str) -> None:
        """Queue one sentence. Blocks only when the lookahead bound is full."""
        if self._stop.is_set():
            return
        future = self._executor.submit(self._synthesize_to_tmp, cleaned)
        self._queue.put((cleaned, future))

    def close(self) -> None:
        """Flush queued sentences in order (skipped if stopped), then join."""
        self._queue.put(None)
        self._player.join()
        self._executor.shutdown(wait=True)

    def _synthesize_to_tmp(self, cleaned: str) -> Optional[str]:
        if self._stop.is_set():
            return None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            _origin().text_to_speech_tool(text=cleaned, output_path=tmp_path)
            return tmp_path
        except Exception as exc:
            logger.warning("Sync per-sentence TTS synthesis failed: %s", exc)
            _unlink_quietly(tmp_path)
            return None

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            _sentence, future = item
            tmp_path = None
            try:
                tmp_path = future.result()
                if (tmp_path and not self._stop.is_set()
                        and os.path.isfile(tmp_path)
                        and os.path.getsize(tmp_path) > 0):
                    from tools.voice_mode import play_audio_file
                    play_audio_file(tmp_path)
            except Exception as exc:
                logger.warning("Sync per-sentence TTS failed: %s", exc)
            finally:
                _unlink_quietly(tmp_path)


class _StreamerPlayback:
    """Prefetch + FIFO playback for a chunked :class:`StreamingTTSProvider`.

    ``speak(text)`` calls ``streamer.stream()`` right away and hands the
    iterator to a prefetch thread (at most 3 in flight) that buffers chunks
    into a bounded per-sentence queue; the single playback worker plays those
    queues in order, so sentence N+1 is already arriving while N plays.
    Output goes to a PortAudio stream when one could be opened, otherwise via
    temp WAV files. A failing PortAudio write is retried on a reinitialized
    stream up to ``_MAX_REINIT`` times before falling back to temp files.
    """

    _MAX_REINIT = 3
    _CHUNK_QUEUE_MAX = 64

    def __init__(self, streamer, stop_event: threading.Event):
        self.streamer = streamer
        self.stop_event = stop_event
        self.output_stream = self._open_output_stream()
        self._audio_queue: "queue.Queue[Optional[queue.Queue[Optional[bytes]]]]" = queue.Queue()
        self._prefetch_threads: List[threading.Thread] = []
        self._prefetch_sem = threading.Semaphore(3)
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()

    # -- PortAudio stream management ---------------------------------------

    def _create_output_stream(self):
        sd = _origin()._import_sounddevice()
        stream = sd.OutputStream(
            samplerate=self.streamer.sample_rate,
            channels=self.streamer.channels,
            dtype="int16",
        )
        stream.start()
        return stream

    def _open_output_stream(self):
        # On macOS skip sounddevice entirely: PortAudio/CoreAudio init triggers
        # a kTCCServiceMediaLibrary permission prompt even though output needs
        # no media-library access. None routes every sentence through the
        # tempfile -> play_audio_file -> afplay path.
        if platform.system() == "Darwin":
            return None
        try:
            return self._create_output_stream()
        except (ImportError, OSError) as exc:
            logger.debug("sounddevice not available, streamer→tempfile: %s", exc)
        except Exception as exc:
            logger.warning("sounddevice OutputStream failed: %s", exc)
        return None

    def _reinit_output_stream(self):
        """Close the broken PortAudio stream and try to create a fresh one."""
        if self.output_stream is not None:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
        try:
            self.output_stream = self._create_output_stream()
            logger.info("TTS: PortAudio output stream reinitialized after error")
        except Exception as exc:
            logger.warning("TTS: PortAudio stream reinit failed: %s", exc)
            self.output_stream = None
        return self.output_stream

    def close_output_stream(self) -> None:
        """Always release the device so a later stream can open it."""
        if self.output_stream is not None:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass

    # -- prefetch ----------------------------------------------------------

    def speak(self, text: str) -> None:
        """Start ``streamer.stream(text)`` and prefetch its chunks immediately."""
        try:
            audio_iter = self.streamer.stream(text)
        except Exception as exc:
            logger.warning("Streaming TTS synthesis failed: %s", exc)
            return
        self._prefetch_sem.acquire()
        chunk_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=self._CHUNK_QUEUE_MAX)
        self._audio_queue.put(chunk_queue)
        t = threading.Thread(target=self._consume_to_queue, args=(audio_iter, chunk_queue), daemon=True)
        self._prefetch_threads.append(t)
        t.start()

    def _consume_to_queue(self, audio_iter: Iterator[bytes], chunk_queue: "queue.Queue[Optional[bytes]]") -> None:
        try:
            for chunk in audio_iter:
                if self.stop_event.is_set():
                    logger.info(
                        "TTS CUT: prefetch cancelled (stop_event set "
                        "mid-sentence) — partial audio only"
                    )
                    break
                chunk_queue.put(chunk, timeout=30.0)
        except Exception as exc:
            logger.warning(
                "TTS CUT: streaming TTS prefetch failed mid-sentence "
                "(partial audio only): %s",
                exc,
            )
        finally:
            chunk_queue.put(None)  # sentinel: no more chunks
            self._prefetch_sem.release()

    # -- playback ----------------------------------------------------------

    def _play_sentence_via_tempfile(self, chunk_queue) -> None:
        _play_via_tempfile(iter(_drain_chunks(chunk_queue)), self.stop_event, self.streamer.sample_rate)

    def _playback_worker(self) -> None:
        """Single consumer: play audio segments from the queue in order."""
        if self.output_stream is None:
            while True:
                chunk_queue = self._audio_queue.get()
                if chunk_queue is None:
                    break
                if self.stop_event.is_set():
                    continue
                self._play_sentence_via_tempfile(chunk_queue)
            return

        import numpy as _np

        try:
            from tools.voice_mode import mark_audio_output_active
        except Exception:
            def mark_audio_output_active(_active):
                return None

        def write_pcm(stream, buf: bytes) -> None:
            stream.write(_np.frombuffer(buf, dtype="<i2").reshape(-1, 1))

        mark_audio_output_active(True)
        try:
            reinit_count = 0
            current_stream = self.output_stream
            while True:
                chunk_queue = self._audio_queue.get()
                if chunk_queue is None:
                    break
                if self.stop_event.is_set():
                    continue
                if current_stream is None:
                    self._play_sentence_via_tempfile(chunk_queue)
                    continue
                pcm_leftover = b""
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None or self.stop_event.is_set():
                        break
                    buf = pcm_leftover + chunk
                    aligned_len = len(buf) - (len(buf) % 2)
                    if aligned_len >= 2:
                        try:
                            write_pcm(current_stream, buf[:aligned_len])
                        except Exception as write_exc:
                            logger.warning(
                                "PortAudio write failed, attempting "
                                "stream reinit: %s",
                                write_exc,
                            )
                            if reinit_count < self._MAX_REINIT:
                                reinit_count += 1
                                current_stream = self._reinit_output_stream()
                                if current_stream is not None:
                                    try:
                                        write_pcm(current_stream, buf[:aligned_len])
                                    except Exception:
                                        pass
                                    pcm_leftover = buf[aligned_len:] if aligned_len < len(buf) else b""
                                    continue
                            else:
                                logger.warning(
                                    "TTS: PortAudio reinit exhausted "
                                    "after %d attempts, falling back "
                                    "to tempfile for remaining "
                                    "sentences",
                                    self._MAX_REINIT,
                                )
                                current_stream = None
                            break
                    pcm_leftover = buf[aligned_len:] if aligned_len < len(buf) else b""
        finally:
            mark_audio_output_active(False)

    def finish(self) -> None:
        """Send the end sentinel, then wait for playback and prefetch threads."""
        self._audio_queue.put(None)
        self._worker.join(timeout=300.0)
        for t in self._prefetch_threads:
            t.join(timeout=10.0)
        self.close_output_stream()


def stream_tts_to_speaker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    tts_done_event: threading.Event,
    display_callback: Optional[Callable[[str], None]] = None,
    provider: Optional[str] = None,
):
    """Consume text deltas from *text_queue*, cut them into sentences, and speak
    each one the moment it's ready — the conversational path.

    A registered streaming provider plays chunked PCM for the lowest latency;
    every other provider (edge, the default) is spoken per-sentence via the
    sync ``text_to_speech_tool`` path, so audio still starts on sentence one.

    Protocol:
        * The producer puts ``str`` deltas onto *text_queue*.
        * A ``None`` sentinel signals end-of-text (flush remaining buffer).
        * *stop_event* aborts early (barge-in / user interrupt).
        * *tts_done_event* is **set** in the ``finally`` block so callers
          waiting on it (continuous voice mode) know playback is finished.
    """
    tts_done_event.clear()
    origin = _origin()
    sync_pipeline: Optional[_SyncSentencePipeline] = None
    playback: Optional[_StreamerPlayback] = None

    try:
        tts_config = origin._load_tts_config()

        # Prefer a chunked streamer for low time-to-first-audio; otherwise
        # per-sentence sync synthesis (universal — edge + every non-streamer).
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider
        streamer = resolve_streaming_provider(tts_config, preferred=provider)

        stream_max_len = 0
        if streamer is None:
            sync_pipeline = _SyncSentencePipeline(stop_event)
        else:
            try:
                stream_max_len = origin._resolve_max_text_length(
                    provider or origin._get_provider(tts_config), tts_config
                )
            except Exception:
                stream_max_len = 0
            playback = _StreamerPlayback(streamer, stop_event)

        chunker = SentenceChunker()
        long_flush_len = 100
        queue_timeout = 0.5
        spoken_sentences: list[str] = []  # skip duplicate/near-duplicate sentences (LLM repetition)

        def _speak_sentence(sentence: str) -> None:
            if stop_event.is_set():
                return
            cleaned = origin._strip_markdown_for_tts(sentence).strip()
            if not cleaned:
                return
            cleaned_lower = cleaned.lower().rstrip(".!,")
            if any(prev.lower().rstrip(".!,") == cleaned_lower for prev in spoken_sentences):
                return
            spoken_sentences.append(cleaned)
            if display_callback is not None:
                display_callback(sentence)  # raw sentence on screen before TTS processing
            if sync_pipeline is not None:
                sync_pipeline.speak(cleaned)
                return
            if stream_max_len and len(cleaned) > stream_max_len:
                cleaned = cleaned[:stream_max_len]
            playback.speak(cleaned)

        while not stop_event.is_set():
            try:
                delta = text_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # Idle producer: flush a long buffer instead of sitting on it
                if len(chunker.buf) > long_flush_len:
                    for sentence in chunker.flush():
                        _speak_sentence(sentence)
                continue

            if delta is None:
                for sentence in chunker.flush():
                    _speak_sentence(sentence)
                break

            for sentence in chunker.feed(delta):
                _speak_sentence(sentence)

        while True:
            try:
                text_queue.get_nowait()
            except queue.Empty:
                break

    except Exception as exc:
        logger.warning("Streaming TTS pipeline error: %s", exc)
    finally:
        # Flush the sync pipeline first: queued sentences finish playing (or
        # are skipped when stop_event is set) BEFORE tts_done_event fires, so
        # continuous voice mode never reopens the mic over its own voice.
        if sync_pipeline is not None:
            try:
                sync_pipeline.close()
            except Exception:
                pass
        # The end sentinel lives in finally: so an exception in the text pump
        # still lets the playback worker exit.
        if playback is not None:
            playback.finish()
        tts_done_event.set()
