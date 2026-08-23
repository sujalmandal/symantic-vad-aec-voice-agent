import math

from unmute_tui.vad.exponential_moving_average import ExponentialMovingAverage


def test_initial_value():
    ema = ExponentialMovingAverage(0.01, 0.01, initial_value=1.0)
    assert ema.value == 1.0


def test_attack_rises_toward_target():
    ema = ExponentialMovingAverage(attack_time=0.01, release_time=0.01, initial_value=0.0)
    for _ in range(1000):
        ema.update(dt=0.02, new_value=1.0)
    assert ema.value > 0.99


def test_release_decays_toward_zero():
    ema = ExponentialMovingAverage(attack_time=0.01, release_time=0.01, initial_value=1.0)
    for _ in range(1000):
        ema.update(dt=0.02, new_value=0.0)
    assert ema.value < 0.01


def test_asymmetric_attack_release():
    # Rising (attack) should move faster than falling (release) for the same dt.
    attack = ExponentialMovingAverage(0.01, 0.1, initial_value=0.0)
    release = ExponentialMovingAverage(0.1, 0.01, initial_value=0.0)
    attack.update(dt=0.02, new_value=1.0)
    release.update(dt=0.02, new_value=1.0)
    assert attack.value > release.value


def test_time_to_decay_to():
    ema = ExponentialMovingAverage(attack_time=0.01, release_time=0.1, initial_value=1.0)
    t = ema.time_to_decay_to(0.5)
    assert math.isclose(t, 0.1, rel_tol=1e-6)
