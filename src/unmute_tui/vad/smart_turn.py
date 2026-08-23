"""Smart Turn v3: audio-native semantic turn-end prediction.

Wraps the pipecat-ai/smart-turn-v3 ONNX model. Given a 16 kHz mono float32
recording of the user's turn, it returns the probability that the speaker has
finished their turn (a sigmoid output). It is audio-native: it uses prosody and
acoustic cues, not a transcript.

Reference: https://github.com/pipecat-ai/smart-turn
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE, SMART_TURN_WINDOW_SEC


def truncate_audio_to_last_n_seconds(
    audio: np.ndarray, n_seconds: float, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Keep only the last `n_seconds` of audio (or pad with leading zeros)."""
    n_samples = int(n_seconds * sample_rate)
    if len(audio) > n_samples:
        return audio[-n_samples:]
    if len(audio) < n_samples:
        return np.concatenate(
            [np.zeros(n_samples - len(audio), dtype=np.float32), audio]
        )
    return audio


class SmartTurn:
    """Loads the Smart Turn v3 ONNX model and predicts turn-end probability."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._session = None
        self._feature_extractor = None

    def _load(self):
        if self._session is not None:
            return self._session, self._feature_extractor

        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Smart Turn model not found at {self.model_path}. "
                "Run `python scripts/download_models.py` first."
            )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self.model_path), sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        return self._session, self._feature_extractor

    def load(self) -> None:
        """Pre-load the model so the first prediction is fast."""
        self._load()

    def predict_endpoint(self, audio: np.ndarray) -> float:
        """Return the probability (0..1) that the speaker has finished their turn.

        Args:
            audio: 16 kHz mono float32 samples of the user's current turn.
        """
        session, fe = self._load()

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = truncate_audio_to_last_n_seconds(audio, SMART_TURN_WINDOW_SEC)

        inputs = fe(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=int(SMART_TURN_WINDOW_SEC * SAMPLE_RATE),
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        outputs = session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())
