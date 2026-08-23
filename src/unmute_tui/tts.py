"""Text-to-speech backends.

Primary backend is Marvis TTS (real-time streaming, MLX-native on Apple
Silicon). Fallbacks: edge-tts (free, cloud) and piper (local). All backends
expose a common async interface that yields 16 kHz mono float32 audio chunks so
the engine can play incrementally.
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


class MarvisTTSBackend(TTSBackend):
    """Marvis TTS (real-time streaming, MLX-native on Apple Silicon).

    Uses the `mlx-audio` package. Streams audio chunks as text is processed.
    """

    def __init__(self, config: TTSConfig, default_ref_audio: str | None = None) -> None:
        self.config = config
        self.default_ref_audio = default_ref_audio
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
            self._model = generate.load_model(self.config.marvis_model)
        return self._model

    def load(self) -> None:
        """Pre-load the Marvis model so the first reply streams immediately."""
        self._load()

    def _ref_audio(self) -> str | None:
        return self.config.ref_audio or self.default_ref_audio

    async def synthesize(self, text: str) -> np.ndarray:
        chunks = [c async for c in self.stream(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        model = self._load()
        ref_audio = self._ref_audio()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            try:
                for r in model.generate(
                    text=text,
                    stream=True,
                    ref_audio=ref_audio,
                    ref_text=self.config.ref_text,
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


def create_tts(config: TTSConfig, default_ref_audio: str | None = None) -> TTSBackend:
    backend = config.backend.strip().lower()
    if backend == "marvis":
        return MarvisTTSBackend(config, default_ref_audio=default_ref_audio)
    if backend == "edge":
        return EdgeTTSBackend(config)
    if backend == "piper":
        return PiperTTSBackend(config)
    raise ValueError(
        f"Unknown TTS backend '{backend}'. Choose from: marvis, edge, piper."
    )
