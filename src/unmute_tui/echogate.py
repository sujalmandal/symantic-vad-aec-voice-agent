"""Playback-aware echo gate for barge-in without AEC.

When there is no acoustic echo canceller, the microphone picks up the bot's own
TTS played through speakers. That echo is loud enough for Silero to flag as
speech, so a naive "keep the mic live during bot speech" makes the bot interrupt
itself. This gate solves it without any DSP: a mic frame is treated as the user
only when it is clearly louder than what the bot has *just* been playing.

The bot's echo in the mic is at most ~ the playback level, so requiring the mic
to exceed the recent playback by a margin (``BARGE_IN_OVER_PLAYBACK_DB``) lets a
user speaking over the bot interrupt it while the bot's own echo never counts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# How far back (seconds) of playback to remember for the energy estimate. The
# speaker->mic echo lags the playback by only tens of ms, so 0.5s is plenty.
PLAYBACK_WINDOW_SEC = 0.5

# Small absolute floor so pure silence is never treated as user speech.
_FLOOR = 1e-4


class PlaybackEchoGate:
    """Decides whether a mic frame is the user, relative to recent playback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        margin_db: float = 6.0,
        window_sec: float = PLAYBACK_WINDOW_SEC,
    ) -> None:
        self.sample_rate = sample_rate
        self.margin_db = margin_db
        self.max_samples = int(window_sec * sample_rate)
        self._squares = 0.0  # rolling sum of mean-squared per sample
        self._count = 0
        self._window: deque[tuple[float, int]] = deque()  # (mean_sq, n)

    def add_playback(self, audio: np.ndarray) -> None:
        """Record a chunk of TTS audio the bot is about to play."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return
        sq = float(np.mean(np.square(audio)))
        n = audio.size
        self._window.append((sq, n))
        self._squares += sq * n
        self._count += n
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest samples once we exceed the window."""
        while self._count > self.max_samples and self._window:
            sq, n = self._window[0]
            drop = min(n, self._count - self.max_samples)
            self._squares -= sq * drop
            self._count -= drop
            if drop >= n:
                self._window.popleft()

    def playback_rms(self) -> float:
        """Recent RMS of what the bot is playing (0 if nothing played recently)."""
        if self._count <= 0:
            return 0.0
        return float(np.sqrt(self._squares / self._count))

    def threshold(self) -> float:
        """A mic RMS must be at least this to count as the user.

        Playback level times the margin, floored at ``_FLOOR`` so pure silence
        never counts.
        """
        play = max(self.playback_rms(), _FLOOR)
        return max(play * (10.0 ** (self.margin_db / 20.0)), _FLOOR)

    def is_user_speech(self, frame: np.ndarray) -> bool:
        """True if ``frame`` looks like the user, not the bot's own echo."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.size == 0:
            return False
        frame_rms = float(np.sqrt(np.mean(np.square(frame))))
        return frame_rms >= self.threshold()

    def reset(self) -> None:
        """Clear playback history (e.g. on barge-in)."""
        self._squares = 0.0
        self._count = 0
        self._window.clear()
