"""Bot active-listening backchannels via VAP (Voice Activity Projection).

When the user is speaking and VAP (the vendored `rvap` package) predicts the
listener should backchannel, the bot emits a short ack ("Mm-hmm", "Uh-huh")
without taking the turn. The model runs in real time (~8ms/frame) on CPU.

VAP takes stereo audio (far-end = bot, near-end = user). While the bot is
listening it is mostly silent, so the far-end channel is silence and the
near-end channel is the (AEC-cleaned) user mic. `p_bc_react` is the probability
of a reactive backchannel; when it crosses the threshold and a cooldown has
elapsed, the engine should play the ack.
"""

from __future__ import annotations

import numpy as np

from .config import BackchannelConfig, SAMPLE_RATE


class BotBackchannel:
    def __init__(self, config: BackchannelConfig) -> None:
        self.config = config
        self._vap = None
        self.frame_size = SAMPLE_RATE // config.frame_rate + 320  # 1920 @ 10Hz
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_ack = -1e9

    def _load(self):
        if self._vap is None:
            from pathlib import Path

            for path in (self.config.vap_bc_model, self.config.cpc_model):
                if not Path(path).exists():
                    raise FileNotFoundError(
                        f"VAP model not found at {path}. Run "
                        "`python scripts/download_models.py`."
                    )
            import contextlib
            import io

            from rvap.vap_bc.vap_bc_main import VAPRealTime

            # VAP prints "Load pretrained CPC"/"Froze EncoderCPC!" on load; wrap
            # so the startup log stays clean.
            with contextlib.redirect_stdout(io.StringIO()):
                self._vap = VAPRealTime(
                    self.config.vap_bc_model,
                    self.config.cpc_model,
                    "cpu",
                    self.config.frame_rate,
                    self.config.context_len_sec,
                )
        return self._vap

    def load(self) -> None:
        """Pre-load the VAP model so backchannel prediction is instant."""
        self._load()

    def push(self, frame: np.ndarray) -> float | None:
        """Feed a 16 kHz mono mic frame; return the backchannel probability
        once a full VAP frame has accumulated, else None."""
        if self._vap is None:
            self._load()
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32)])
        if len(self._buf) < self.frame_size:
            return None
        user = np.asarray(self._buf, dtype=np.float32)[-self.frame_size:]
        bot = np.zeros(self.frame_size, dtype=np.float32)  # bot is listening
        import contextlib
        import io
        import warnings

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._vap.process_vap(bot, user)
        prob = float(self._vap.result_p_bc_react[0])
        return prob

    def should_ack(self, prob: float | None, now: float) -> bool:
        return (
            prob is not None
            and prob > self.config.threshold
            and (now - self._last_ack) > self.config.cooldown_sec
        )

    def mark_acked(self, now: float) -> None:
        self._last_ack = now

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
