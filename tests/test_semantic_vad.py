import numpy as np

from unmute_tui.vad.semantic_vad import SemanticVAD


class FakeSilero:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.i = 0

    def is_speech(self, frame):
        d = self.decisions[self.i % len(self.decisions)]
        self.i += 1
        return d


class FakeSmartTurn:
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0

    def predict_endpoint(self, audio):
        self.calls += 1
        return self.probability


def _frame():
    return np.zeros(320, dtype=np.float32)


def test_turn_end_when_semantic_probability_high():
    # 10 speech frames then 20 silence frames; Smart Turn returns 0.9.
    silero = FakeSilero([True] * 10 + [False] * 20)
    smart = FakeSmartTurn(0.9)
    vad = SemanticVAD(smart_turn=smart, silero=silero, threshold=0.6)

    turn_ended = False
    for _ in range(30):
        result = vad.process_frame(_frame())
        if result.turn_end:
            turn_ended = True
            assert result.audio is not None
    assert turn_ended is True
    assert smart.calls >= 1


def test_no_turn_end_when_semantic_probability_low():
    silero = FakeSilero([True] * 10 + [False] * 20)
    smart = FakeSmartTurn(0.1)
    vad = SemanticVAD(smart_turn=smart, silero=silero, threshold=0.6)

    result = None
    for _ in range(30):
        result = vad.process_frame(_frame())
    assert result.turn_end is False
    assert result.audio is None


def test_resume_speaking_reruns_semantic():
    # Speak, pause (low prob, no turn end), speak again, pause (high prob -> end).
    silero = FakeSilero([True] * 10 + [False] * 20 + [True] * 10 + [False] * 20)
    smart = FakeSmartTurn(0.9)
    vad = SemanticVAD(smart_turn=smart, silero=silero, threshold=0.6)

    turn_ended = False
    for _ in range(60):
        result = vad.process_frame(_frame())
        if result.turn_end:
            turn_ended = True
    assert turn_ended is True
    # Smart Turn should have been run at least twice (once per pause).
    assert smart.calls >= 2


def test_reset_clears_buffer():
    silero = FakeSilero([True] * 10 + [False] * 20)
    smart = FakeSmartTurn(0.9)
    vad = SemanticVAD(smart_turn=smart, silero=silero, threshold=0.6)
    for _ in range(30):
        result = vad.process_frame(_frame())
        if result.turn_end:
            break
    assert vad.probability == 0.0  # reset after turn end
    assert len(vad._turn_buffer) == 0
