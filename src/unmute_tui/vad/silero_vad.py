"""Silero VAD wrapper for raw speech-activity detection.

Silero VAD classifies each audio frame as speech or non-speech. It is *not*
semantic — it cannot tell whether the user has finished their turn. It is used
here only to segment the user's turn so the semantic model (Smart Turn) can be
run on the full turn recording.

Silero requires at least 512 samples per call, so we keep a rolling 512-sample
buffer and classify each 20 ms frame against the most recent 512 samples.
"""

from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE

# Silero's minimum input length at 16 kHz.
_MIN_SAMPLES = 512


class SileroVAD:
    def __init__(self, threshold: float = 0.5, energy_threshold: float = 0.01) -> None:
        self.threshold = threshold
        # Frames quieter than this RMS are treated as silence, so quiet ambient
        # noise doesn't get misclassified as speech and trigger barge-in.
        self.energy_threshold = energy_threshold
        self._model = None
        self._buf = np.zeros(0, dtype=np.float32)

    def _load(self):
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "silero-vad is not installed. Run `uv sync` (or "
                    "`pip install silero-vad`)."
                ) from exc
            self._model = load_silero_vad()
        return self._model

    def load(self) -> None:
        """Pre-load the model so the first frame is classified fast."""
        self._load()

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the 16 kHz mono float32 frame contains speech."""
        frame = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < self.energy_threshold:
            return False
        model = self._load()
        import torch

        self._buf = np.concatenate([self._buf, frame])
        if len(self._buf) < _MIN_SAMPLES:
            return False
        window = self._buf[-_MIN_SAMPLES:]
        tensor = torch.from_numpy(window).unsqueeze(0)
        with torch.no_grad():
            prob = model(tensor, SAMPLE_RATE).item()
        return prob >= self.threshold
