"""Text-to-speech backends.

Primary backend is Chatterbox-Turbo via `mlx-audio` (natural, expressive with
emotion tags, near real-time on Apple Silicon, local). Alternatives: kokoro
(kokoro-onnx), edge-tts (free, cloud), and piper (local). All backends expose a
common async interface that yields 16 kHz mono float32 audio chunks so the
engine can play incrementally.
"""

from __future__ import annotations

import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np

from .config import SAMPLE_RATE, TTSConfig


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample a mono float32 array to 16 kHz using linear interpolation."""
    if src_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    n_out = int(round(len(audio) * SAMPLE_RATE / src_rate))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class TTSBackend(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize `text` and return 16 kHz mono float32 audio."""

    def load(self) -> None:
        """Pre-load the model. No-op for backends with no persistent model."""

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield 16 kHz mono float32 audio chunks for `text`.

        Default implementation synthesizes the whole text and yields it as a
        single chunk. Streaming backends override this.
        """
        audio = await self.synthesize(text)
        if len(audio):
            yield audio


class ChatterboxTTSBackend(TTSBackend):
    """Chatterbox-Turbo via `mlx-audio` (natural, expressive, near real-time).

    Streams audio chunks as text is processed (MLX on Apple Silicon). Supports
    inline emotion tags like [sigh] and [laugh].
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from mlx_audio.tts import generate
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "mlx-audio is not installed. Run `uv sync` (or "
                    "`pip install mlx-audio`)."
                ) from exc
            self._model = generate.load_model(self.config.chatterbox_model)
        return self._model

    def load(self) -> None:
        """Pre-load the Chatterbox model so the first reply is instant."""
        self._load()

    async def synthesize(self, text: str) -> np.ndarray:
        chunks = [c async for c in self.stream(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        model = self._load()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            import contextlib
            import io

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    for r in model.generate(
                        text=text,
                        stream=True,
                        verbose=False,
                    ):
                        audio = np.asarray(r.audio, dtype=np.float32)
                        q.put_nowait(resample_to_16k(audio, r.sample_rate))
            except Exception as exc:  # noqa: BLE001
                q.put_nowait(exc)
            finally:
                q.put_nowait(None)

        task = loop.run_in_executor(None, _run)
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await task


class KokoroTTSBackend(TTSBackend):
    """Kokoro-82M TTS via `kokoro-onnx` (fast, natural, lightweight, offline).

    `Kokoro.create()` is a synchronous, near-real-time call returning
    (samples, sr); we run it in a thread and resample to 16 kHz. Kokoro is
    not streaming, so the base `stream()` yields the whole sentence as one
    chunk — acceptable because the engine synthesizes sentence-by-sentence.
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._kokoro = None

    def _load(self):
        if self._kokoro is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "kokoro-onnx is not installed. Run `uv sync` (or "
                    "`pip install kokoro-onnx`)."
                ) from exc
            if not self.config.kokoro_model or not self.config.kokoro_voices:
                raise FileNotFoundError(
                    "Kokoro model files not configured. Set KOKORO_MODEL and "
                    "KOKORO_VOICES, or run `python scripts/download_models.py`."
                )
            from pathlib import Path

            if not Path(self.config.kokoro_model).exists() or not Path(
                self.config.kokoro_voices
            ).exists():
                raise FileNotFoundError(
                    "Kokoro model files not found. Run "
                    "`python scripts/download_models.py`."
                )
            self._kokoro = Kokoro(self.config.kokoro_model, self.config.kokoro_voices)
        return self._kokoro

    def load(self) -> None:
        """Pre-load the Kokoro model so the first reply is instant."""
        self._load()

    async def synthesize(self, text: str) -> np.ndarray:
        kokoro = self._load()
        samples, sr = await asyncio.to_thread(
            kokoro.create,
            text,
            self.config.kokoro_voice,
            self.config.kokoro_speed,
            self.config.kokoro_lang,
        )
        if samples is None or len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        return resample_to_16k(np.asarray(samples, dtype=np.float32), int(sr))


class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge TTS (free, cloud). Output is MP3, decoded via ffmpeg."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> np.ndarray:
        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("edge-tts is not installed.") from exc

        communicate = edge_tts.Communicate(text, self.config.voice)
        mp3 = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        if not mp3:
            return np.zeros(0, dtype=np.float32)
        return await asyncio.to_thread(_decode_mp3_to_pcm, mp3)


class PiperTTSBackend(TTSBackend):
    """Local piper TTS (offline). Requires the `piper-tts` package + a voice."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> np.ndarray:
        try:
            import piper
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "piper-tts is not installed. Run `pip install piper-tts`."
            ) from exc
        # piper's API varies by version; this is a best-effort integration.
        voice = piper.PiperVoice.load(self.config.voice)
        audio = await asyncio.to_thread(_piper_synthesize, voice, text)
        return resample_to_16k(audio, 22050)


def _piper_synthesize(voice, text: str) -> np.ndarray:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=voice.config.sample_rate, channels=1, format="WAV"
    ) as f:
        for audio in voice.synthesize(text):
            f.write(audio)
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32")
    return np.asarray(data, dtype=np.float32)


def _decode_mp3_to_pcm(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes to 16 kHz mono float32 PCM using ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def create_tts(config: TTSConfig) -> TTSBackend:
    backend = config.backend.strip().lower()
    if backend == "chatterbox":
        return ChatterboxTTSBackend(config)
    if backend == "kokoro":
        return KokoroTTSBackend(config)
    if backend == "edge":
        return EdgeTTSBackend(config)
    if backend == "piper":
        return PiperTTSBackend(config)
    raise ValueError(
        f"Unknown TTS backend '{backend}'. "
        f"Choose from: chatterbox, kokoro, edge, piper."
    )
