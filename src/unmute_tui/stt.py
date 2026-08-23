"""Speech-to-text via faster-whisper (local, CTranslate2).

Transcribes a user's turn (16 kHz mono float32) into words. Used for the
conversation transcript and as the LLM's input. The semantic VAD itself is
audio-native and does not depend on this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE


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


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
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
        """Pre-load the model so the first transcription is fast."""
        self._load()

    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Transcribe a 16 kHz mono float32 array of the user's turn."""
        model = self._load()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
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
