"""Exponential moving average with asymmetric attack/release.

Ported from unmute.sh's `unmute/stt/exponential_moving_average.py`. It smooths
a signal differently depending on whether the new value is rising (attack) or
falling (release), which lets a turn-end probability rise quickly when the user
pauses but decay slowly when they resume speaking.
"""

from __future__ import annotations

import math


class ExponentialMovingAverage:
    def __init__(
        self,
        attack_time: float,
        release_time: float,
        initial_value: float = 0.0,
    ) -> None:
        """Args:
            attack_time: seconds to reach 50% of target when rising.
            release_time: seconds to decay to 50% of target when falling.
            initial_value: starting value.
        """
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = float(initial_value)

    def update(self, *, dt: float, new_value: float) -> float:
        assert dt > 0.0, f"dt must be positive, got {dt=}"
        assert new_value >= 0.0, f"new_value must be non-negative, got {new_value=}"

        if new_value > self.value:
            alpha = 1 - math.exp(-dt / self.attack_time * math.log(2))
        else:
            alpha = 1 - math.exp(-dt / self.release_time * math.log(2))

        self.value = float((1 - alpha) * self.value + alpha * new_value)
        return self.value

    def time_to_decay_to(self, value: float) -> float:
        """Seconds for the estimate to reach `value` if it started at 1."""
        assert 0 < value < 1
        return float(-self.release_time * math.log2(value))
