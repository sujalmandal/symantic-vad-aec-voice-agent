import numpy as np

from unmute_tui.echogate import PlaybackEchoGate


def _frame(rms, size=320):
    # White-ish noise at a target RMS, avoiding a pure DC ramp that would be
    # degenerate for RMS measurement.
    x = np.random.default_rng(0).normal(0, 1, size).astype(np.float32)
    x -= x.mean()
    cur = float(np.sqrt(np.mean(x**2)))
    if cur > 0:
        x *= rms / cur
    return x.astype(np.float32)


def test_playback_level_does_not_trigger():
    gate = PlaybackEchoGate(margin_db=6.0)
    gate.add_playback(_frame(0.2))  # bot playing at RMS 0.2
    # Mic echo at the same level as playback -> NOT the user.
    assert gate.is_user_speech(_frame(0.2)) is False
    assert gate.is_user_speech(_frame(0.3)) is False  # within margin


def test_user_louder_than_playback_triggers():
    gate = PlaybackEchoGate(margin_db=6.0)
    gate.add_playback(_frame(0.2))
    # User speaking over the bot, well above playback + 6 dB.
    assert gate.is_user_speech(_frame(0.8)) is True


def test_no_playback_passes_audible_frames():
    gate = PlaybackEchoGate(margin_db=6.0)
    assert gate.playback_rms() == 0.0
    assert gate.is_user_speech(_frame(0.1)) is True


def test_threshold_scales_with_playback():
    quiet = PlaybackEchoGate(margin_db=6.0)
    quiet.add_playback(_frame(0.02))
    loud = PlaybackEchoGate(margin_db=6.0)
    loud.add_playback(_frame(0.4))
    # Higher playback -> higher absolute threshold.
    assert loud.threshold() > quiet.threshold()


def test_reset_clears_playback():
    gate = PlaybackEchoGate(margin_db=6.0)
    gate.add_playback(_frame(0.4))
    gate.reset()
    assert gate.playback_rms() == 0.0
    assert gate.is_user_speech(_frame(0.1)) is True
