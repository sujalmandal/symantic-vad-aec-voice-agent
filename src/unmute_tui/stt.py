"""Speech-to-text backends (local, realtime).

The engine talks to STT through a small backend interface so the model can be
swapped without touching the conversation loop:

* ``push(frame)``   — feed one 16 kHz mic frame while the user speaks (streaming
  backends accumulate the turn incrementally; no-op for batch backends).
* ``partial()``     — the current partial transcript of the in-progress turn.
* ``transcribe()``  — the final transcription of a completed turn.
* ``reset()``       — start a fresh turn.

Available backends (``STT_BACKEND``):

* ``sherpa`` (default) — sherpa-onnx streaming Zipformer: true incremental
  decoding, ~0.03-0.05 RTF in int8 on CPU, far more accurate than whisper base.
* ``moonshine``       — Moonshine v2 via moonshine-voice (very low latency).
* ``parakeet``        — NVIDIA Parakeet TDT-0.6B via sherpa-onnx offline
  recognizer (best raw WER; non-streaming).
* ``faster_whisper``  — the original faster-whisper backend (fallback).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

# Tail padding fed before flushing a streaming decoder (lets trailing words
# finish decoding instead of being cut off at turn end).
_TAIL_PAD_SEC = 0.5
# Zipformer (and Parakeet) feature dimension.
_FEATURE_DIM = 80


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Transcription:
    text: str
    segments: list[Segment]
    language: str | None = None


def _as_1d_float32(audio) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    return audio


def _trailing_window(audio: np.ndarray, window_sec: float) -> np.ndarray:
    """Keep only the last `window_sec` of audio (or all of it if shorter)."""
    n = int(window_sec * SAMPLE_RATE)
    if len(audio) > n:
        return audio[-n:]
    return audio


class STTBackend(ABC):
    """Engine-facing interface implemented by every STT backend."""

    def load(self) -> None:
        """Pre-load all models (blocking); called before the event loop."""

    def push(self, frame: np.ndarray) -> None:
        """Feed one 16 kHz mono mic frame of the in-progress user turn.

        Default: no-op (batch backends re-transcribe windows on demand).
        """

    @abstractmethod
    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        """The current partial transcript of the in-progress turn.

        Streaming backends may ignore `audio`/`window_sec` and return their
        incremental result; batch backends re-transcribe the trailing window.
        """

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Final transcription of a completed turn (16 kHz mono float32)."""

    def reset(self) -> None:
        """Start a fresh turn (default: no-op)."""


# ── faster-whisper (original backend, kept as fallback) ────────────────────
class FasterWhisperBackend(STTBackend):
    """Batch transcription via faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "faster-whisper is not installed. Run `uv sync --extra stt` "
                    "(or `pip install faster-whisper`)."
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def load(self) -> None:
        self._load()

    def _transcribe(self, audio: np.ndarray) -> Transcription:
        model = self._load()
        audio = _as_1d_float32(audio)
        if len(audio) == 0:
            return Transcription("", [])
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            language=self.language,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segs = [
            Segment(text=s.text.strip(), start=float(s.start), end=float(s.end))
            for s in segments
        ]
        text = " ".join(s.text for s in segs).strip()
        return Transcription(text=text, segments=segs, language=info.language)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        return self._transcribe(audio)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        if audio is None or len(audio) == 0:
            return Transcription("", [])
        return self._transcribe(_trailing_window(_as_1d_float32(audio), window_sec))


class _SherpaModelDir:
    """Resolves the encoder/decoder/joiner/tokens files in a model folder.

    File names differ between sherpa-onnx snapshots (e.g.
    ``encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx`` vs
    ``encoder-epoch-99-avg-1.onnx``), so we glob for the first match.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)

    def _find(self, patterns: list[str]) -> Path | None:
        for pattern in patterns:
            hits = sorted(self.model_dir.glob(pattern))
            if hits:
                return hits[0]
        return None

    @property
    def encoder(self) -> Path:
        return self._require(["encoder*.int8.onnx", "encoder*.onnx"], "encoder")

    @property
    def decoder(self) -> Path:
        return self._require(["decoder*.onnx", "decoder*.int8.onnx"], "decoder")

    @property
    def joiner(self) -> Path:
        return self._require(["joiner*.int8.onnx", "joiner*.onnx"], "joiner")

    @property
    def tokens(self) -> Path:
        return self._require(["tokens.txt", "bpe.model"], "tokens")

    def _require(self, patterns: list[str], name: str) -> Path:
        hit = self._find(patterns)
        if hit is None:
            raise FileNotFoundError(
                f"Sherpa-onnx {name} file not found in {self.model_dir}. "
                "Run `uv run python scripts/download_models.py` first."
            )
        return hit


# ── sherpa-onnx streaming Zipformer (default) ──────────────────────────────
class SherpaZipformerBackend(STTBackend):
    """True streaming ASR via sherpa-onnx ``OnlineRecognizer``.

    Frames are fed as they arrive; ``partial()`` returns the recognizer's
    incremental result for the current turn, so the LLM turn orchestrator sees
    words as they are spoken instead of a re-transcribed trailing window.
    """

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 2,
        language: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.language = language
        self._recognizer = None
        self._stream = None
        self._lock = threading.Lock()

    def _load(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sherpa-onnx is not installed. Run `uv sync --extra stt` "
                "(or `pip install sherpa-onnx`)."
            ) from exc
        files = _SherpaModelDir(self.model_dir)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(  # noqa: E501
            tokens=str(files.tokens),
            encoder=str(files.encoder),
            decoder=str(files.decoder),
            joiner=str(files.joiner),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=_FEATURE_DIM,
            decoding_method="greedy_search",
            provider="cpu",
            # Model type is inferred from the ONNX files; "" selects the
            # zipformer transducer path for streaming zipformer models.
            model_type="",
            enable_endpoint_detection=False,  # engine's VAD stays turn arbiter
        )
        return self._recognizer

    def load(self) -> None:
        self._load()

    def push(self, frame: np.ndarray) -> None:
        recognizer = self._load()
        frame = _as_1d_float32(frame)
        if len(frame) == 0:
            return
        with self._lock:
            if self._stream is None:
                self._stream = recognizer.create_stream()
            self._stream.accept_waveform(SAMPLE_RATE, frame)
            while recognizer.is_ready(self._stream):
                recognizer.decode_stream(self._stream)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        recognizer = self._load()
        with self._lock:
            if self._stream is None:
                return Transcription("", [])
            text = recognizer.get_result(self._stream)
        # Zipformer emits uppercase; normalize for natural LLM input.
        return Transcription(
            text=text.strip().lower(), segments=[], language=self.language
        )

    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Finalize: feed the full turn (fresh stream) + tail padding, then drain."""
        recognizer = self._load()
        audio = _as_1d_float32(audio)
        with self._lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            pad = np.zeros(int(_TAIL_PAD_SEC * SAMPLE_RATE), dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, pad)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = recognizer.get_result(stream).strip().lower()
        self.reset()
        return Transcription(text=text, segments=[], language=self.language)

    def reset(self) -> None:
        with self._lock:
            self._stream = None


# ── sherpa-onnx offline Parakeet (best raw WER, non-streaming) ─────────────
class ParakeetBackend(STTBackend):
    """NVIDIA Parakeet TDT-0.6B via sherpa-onnx ``OfflineRecognizer``."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 2,
        language: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.language = language
        self._recognizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sherpa-onnx is not installed. Run `uv sync --extra stt` "
                "(or `pip install sherpa-onnx`)."
            ) from exc
        files = _SherpaModelDir(self.model_dir)
        # NeMo (Parakeet) int8 decoders sometimes ship without the RNNT
        # metadata sherpa-onnx needs; without it the C++ layer hard-aborts.
        # Fail with a helpful message before constructing the recognizer.
        self._check_decoder_metadata(files.decoder)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(  # noqa: E501
            encoder=str(files.encoder),
            decoder=str(files.decoder),
            joiner=str(files.joiner),
            tokens=str(files.tokens),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=_FEATURE_DIM,
            decoding_method="greedy_search",
            provider="cpu",
            # NeMo RNNT exports (Parakeet) need the nemo_transducer model type
            # so sherpa feeds the audio in the layout the encoder expects.
            model_type="nemo_transducer",
        )
        return self._recognizer

    @staticmethod
    def _check_decoder_metadata(decoder: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            return  # onnxruntime is a core dep; if absent, let load fail later
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            meta = ort.InferenceSession(
                str(decoder), sess_options=so, providers=["CPUExecutionProvider"]
            ).get_modelmeta().custom_metadata_map
        except Exception:  # noqa: BLE001 — unreadable file: fail downstream
            return
        if "vocab_size" not in meta:
            raise RuntimeError(
                f"Parakeet decoder is missing RNNT metadata ({decoder}). "
                "Rerun `uv run python scripts/download_models.py` so the "
                "metadata is patched, or use STT_BACKEND=sherpa/moonshine."
            )

    def load(self) -> None:
        self._load()

    def transcribe(self, audio: np.ndarray) -> Transcription:
        recognizer = self._load()
        audio = _as_1d_float32(audio)
        with self._lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_streams([stream])
            text = stream.result.text.strip()
        return Transcription(text=text, segments=[], language=self.language)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        if audio is None or len(audio) == 0:
            return Transcription("", [])
        return self.transcribe(_trailing_window(_as_1d_float32(audio), window_sec))


# ── Moonshine v2 (moonshine-voice) ─────────────────────────────────────────
class MoonshineBackend(STTBackend):
    """Moonshine v2 streaming STT via moonshine-voice's ``Transcriber`` API.

    Uses the documented streaming path: a ``Transcriber`` stream is fed audio
    chunks (``push``), a listener keeps the latest partial text and completed
    lines, and ``partial()``/``transcribe()`` read them out.
    """

    def __init__(self, language: str = "en", update_interval: float = 0.3) -> None:
        self.language = language
        self.update_interval = update_interval
        self._transcriber = None
        self._stream = None
        self._listener = None
        self._text = ""
        self._final_lines: list[str] = []
        self._lock = threading.Lock()

    def _load(self):
        if self._transcriber is not None:
            return self._transcriber
        try:
            from moonshine_voice import (
                Transcriber,
                TranscriptEventListener,
                get_model_for_language,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "moonshine-voice is not installed. Run `uv sync --extra stt` "
                "(or `pip install moonshine-voice`)."
            ) from exc

        class _Collector(TranscriptEventListener):
            def __init__(self, owner) -> None:
                super().__init__()
                self._owner = owner

            def on_line_text_changed(self, event) -> None:
                self._owner._on_text(event.line.text)

            def on_line_completed(self, event) -> None:
                self._owner._on_line(event.line.text)

        model_path, model_arch = get_model_for_language(self.language)
        self._transcriber = Transcriber(
            model_path=model_path, model_arch=model_arch
        )
        self._listener = _Collector(self)
        return self._transcriber

    def _ensure_stream(self):
        self._load()
        if self._stream is not None:
            return self._stream
        self._stream = self._transcriber.create_stream(
            update_interval=self.update_interval
        )
        self._stream.add_listener(self._listener)
        self._stream.start()
        return self._stream

    def _on_text(self, text: str) -> None:
        with self._lock:
            self._text = text

    def _on_line(self, text: str) -> None:
        with self._lock:
            if text.strip():
                self._final_lines.append(text.strip())
            self._text = ""

    def load(self) -> None:
        self._load()

    def push(self, frame: np.ndarray) -> None:
        stream = self._ensure_stream()
        frame = _as_1d_float32(frame)
        if len(frame) == 0:
            return
        stream.add_audio(frame.tolist(), SAMPLE_RATE)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        self._load()
        with self._lock:
            text = self._text.strip()
        return Transcription(text=text, segments=[])

    def transcribe(self, audio: np.ndarray) -> Transcription:
        self._load()
        with self._lock:
            text = (" ".join(self._final_lines) or self._text).strip()
        return Transcription(text=text, segments=[])

    def reset(self) -> None:
        with self._lock:
            self._text = ""
            self._final_lines = []
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
            self._stream = None


# ── factory ────────────────────────────────────────────────────────────────
def create_transcriber(config) -> STTBackend:
    """Build the STT backend selected by ``config.backend``."""
    backend = (config.backend or "sherpa").strip().lower()
    if backend == "sherpa":
        return SherpaZipformerBackend(
            model_dir=config.models_dir / config.model,
            num_threads=config.threads,
            language=config.language,
        )
    if backend == "parakeet":
        return ParakeetBackend(
            model_dir=config.models_dir / config.model,
            num_threads=config.threads,
            language=config.language,
        )
    if backend == "moonshine":
        return MoonshineBackend(language=config.language or "en")
    if backend == "faster_whisper":
        return FasterWhisperBackend(
            model_size=config.model, language=config.language
        )
    raise ValueError(
        f"Unknown STT_BACKEND {backend!r} "
        "(expected sherpa | moonshine | parakeet | faster_whisper)"
    )


# Backward-compatible alias for any code importing `Transcriber`.
Transcriber = STTBackend