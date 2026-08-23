"""Audio capture and playback via sounddevice (PortAudio).

The microphone is captured as 16 kHz mono float32 frames pushed into an
asyncio queue. TTS audio is played through an output stream.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, SAMPLES_PER_FRAME


class AudioError(RuntimeError):
    pass


@dataclass
class AudioDevices:
    input: int | None
    output: int | None


def list_devices() -> str:
    """Return a human-readable list of audio devices."""
    lines = []
    for i, dev in enumerate(sd.query_devices()):
        lines.append(
            f"{i}: {dev['name']}  (in={dev['max_input_channels']} "
            f"out={dev['max_output_channels']}, sr={dev['default_samplerate']})"
        )
    return "\n".join(lines)


class Microphone:
    """Streams microphone audio as 16 kHz mono float32 frames."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = SAMPLES_PER_FRAME,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._stop = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            # Ignore transient over/underflow warnings; surface only hard errors.
            if status.input_overflow:
                pass
        # indata is (frames, channels); take mono channel 0, copy to float32.
        audio = np.asarray(indata[:, 0], dtype=np.float32).copy()
        self._queue.put(audio)

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise AudioError(
                f"Could not open microphone (device={self.device}): {exc}"
            ) from exc

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    async def frames(self) -> "asyncio.AsyncIterator[np.ndarray]":
        """Yield 16 kHz mono float32 frames as they arrive."""
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                frame = await loop.run_in_executor(None, self._queue.get, True, 0.1)
            except queue.Empty:
                continue
            yield frame


class AudioPlayer:
    """Plays float32 audio (16 kHz mono) through the output device."""

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise AudioError(
                f"Could not open output device (device={self.device}): {exc}"
            ) from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def play(self, audio: np.ndarray) -> None:
        """Write a chunk of float32 mono audio to the output stream (blocking)."""
        if self._stream is None:
            raise AudioError("AudioPlayer not started")
        if audio.ndim == 1:
            audio = audio[:, None]
        self._stream.write(np.asarray(audio, dtype=np.float32))

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.write(np.zeros((SAMPLES_PER_FRAME, 1), dtype=np.float32))

    def clear(self) -> None:
        """Discard any buffered audio by restarting the output stream.

        Used on barge-in so stale TTS audio stops immediately.
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.start()
