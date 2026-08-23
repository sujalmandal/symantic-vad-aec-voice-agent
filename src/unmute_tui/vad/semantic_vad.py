"""Semantic VAD orchestrator.

Combines Silero VAD (raw speech activity) with Smart Turn v3 (semantic turn-end
probability) and mirrors unmute.sh's decision logic:

* While the user is speaking, the turn-end probability is decayed toward 0.
* When silence is detected after speech, Smart Turn is run on the full turn
  recording and the resulting probability is smoothed with an exponential
  moving average.
* When the smoothed probability exceeds the threshold, a turn-end is signalled.

If the user resumes speaking before a turn-end is signalled, the semantic model
is re-run on the (now longer) full turn, per Smart Turn's guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FRAME_TIME_SEC, SAMPLE_RATE
from .exponential_moving_average import ExponentialMovingAverage
from .silero_vad import SileroVAD
from .smart_turn import SmartTurn

# Minimum silence (seconds) before we bother running the semantic model.
MIN_SILENCE_SEC = 0.2
# EMA time constants (mirror unmute's pause_prediction EMA).
EMA_ATTACK_TIME = 0.01
EMA_RELEASE_TIME = 0.01


@dataclass
class TurnEndResult:
    turn_end: bool
    probability: float  # smoothed EMA value
    raw_probability: float | None  # last Smart Turn output (None if not run)
    audio: np.ndarray | None = None  # the turn recording, when turn_end is True


class SemanticVAD:
    def __init__(
        self,
        smart_turn: SmartTurn,
        silero: SileroVAD | None = None,
        threshold: float = 0.6,
        min_silence_sec: float = MIN_SILENCE_SEC,
    ) -> None:
        self.smart_turn = smart_turn
        self.silero = silero or SileroVAD()
        self.threshold = threshold
        self.min_silence_frames = max(1, int(min_silence_sec / FRAME_TIME_SEC))

        self.ema = ExponentialMovingAverage(
            attack_time=EMA_ATTACK_TIME,
            release_time=EMA_RELEASE_TIME,
            initial_value=0.0,
        )
        self._turn_buffer: list[np.ndarray] = []
        self._was_speaking = False
        self._silence_frames = 0
        self._need_semantic = False
        self._last_raw: float | None = None

    @property
    def probability(self) -> float:
        return self.ema.value

    @property
    def is_speaking(self) -> bool:
        """Whether the user is currently speaking (last frame was speech)."""
        return self._was_speaking

    def reset(self) -> None:
        """Start a fresh turn (called after a turn-end is consumed)."""
        self._turn_buffer = []
        self._was_speaking = False
        self._silence_frames = 0
        self._need_semantic = False
        self._last_raw = None
        self.ema.value = 0.0

    def _turn_audio(self) -> np.ndarray:
        if not self._turn_buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._turn_buffer)

    def process_frame(self, frame: np.ndarray) -> TurnEndResult:
        """Feed one 16 kHz mono float32 frame; return the turn-end decision."""
        frame = np.asarray(frame, dtype=np.float32)
        self._turn_buffer.append(frame)

        is_speech = self.silero.is_speech(frame)

        if is_speech:
            self._silence_frames = 0
            if not self._was_speaking:
                # Transition silence -> speech: the user resumed; we must re-run
                # the semantic model on the full turn when they pause again.
                self._need_semantic = True
            self._was_speaking = True
            # User is actively speaking: decay turn-end probability toward 0.
            self.ema.update(dt=FRAME_TIME_SEC, new_value=0.0)
            return TurnEndResult(False, self.ema.value, self._last_raw)

        # Silence.
        self._silence_frames += 1
        was_speaking = self._was_speaking
        self._was_speaking = False

        if (was_speaking or self._need_semantic) and (
            self._silence_frames >= self.min_silence_frames
        ):
            # Run the semantic model on the full turn recording.
            audio = self._turn_audio()
            if len(audio) > 0:
                self._last_raw = self.smart_turn.predict_endpoint(audio)
                self.ema.update(dt=FRAME_TIME_SEC, new_value=self._last_raw)
            self._need_semantic = False

        turn_end = self.ema.value > self.threshold
        if turn_end:
            audio = self._turn_audio()
            self.reset()
            return TurnEndResult(True, self.ema.value, self._last_raw, audio)
        return TurnEndResult(False, self.ema.value, self._last_raw)
