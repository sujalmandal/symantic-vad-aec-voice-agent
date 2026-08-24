import asyncio

import numpy as np
import pytest

from unmute_tui.config import Config
from unmute_tui.engine import ConversationEngine
from unmute_tui.turn import TurnDecision, TurnResult
from unmute_tui.vad.semantic_vad import TurnEndResult


class FakeVAD:
    def __init__(self, audio=None):
        self.is_speaking = True
        self.silence_frames = 0
        self.probability = 0.0
        self.audio = audio if audio is not None else np.zeros(3200, np.float32)

    def process_frame(self, frame):
        return TurnEndResult(False, 0.0, None)

    def get_turn_audio(self):
        return self.audio

    def reset(self):
        pass

    def load(self):
        pass


class FakeTranscriber:
    def __init__(self, text="my phone number is 412"):
        self.text = text

    def load(self):
        pass

    def push(self, frame):
        pass

    def reset(self):
        pass

    def partial(self, audio, window_sec):
        from unmute_tui.stt import Transcription

        return Transcription(text=self.text, segments=[])


class FakeLLM:
    async def stream(self, messages):
        yield ""


class FakeTTS:
    def __init__(self, audio=None):
        self.audio = audio if audio is not None else np.full(160, 0.2, np.float32)

    def load(self):
        pass

    async def stream(self, text):
        if len(self.audio):
            yield self.audio


class FakeMic:
    def start(self):
        pass

    def stop(self):
        pass

    async def frames(self):
        if False:
            yield np.zeros(320, np.float32)


class FakePlayer:
    def __init__(self):
        self.played = []

    def start(self):
        pass

    def stop(self):
        pass

    def play(self, audio):
        self.played.append(audio)

    def clear(self):
        pass


class FakeOrchestrator:
    def __init__(self, result):
        self.result = result

    def orchestration_context(self, chat_history):
        return []

    async def decide(self, partial, vad_state, context):
        return self.result


def _make_engine(config, orchestrator=None):
    events = asyncio.Queue()
    return ConversationEngine(
        config=config,
        vad=FakeVAD(),
        transcriber=FakeTranscriber(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        mic=FakeMic(),
        player=FakePlayer(),
        events=events,
        orchestrator=orchestrator,
    )


def _base_config():
    cfg = Config.from_env(env_file=None)
    cfg.vad.mute_mic_while_bot_speaking = False
    cfg.vad.barge_in_required_frames = 5
    return cfg


@pytest.mark.asyncio
async def test_barge_in_without_aec_requires_user_louder_than_playback():
    cfg = _base_config()
    engine = _make_engine(cfg)  # no AEC
    engine._set_state("bot_speaking")
    engine.uninterruptible_until = -1.0
    # The bot is playing at RMS ~0.2.
    engine.echo_gate.add_playback(np.full(160, 0.2, np.float32))

    # Mic frames at ~playback level (bot's own echo) must NOT barge in.
    for _ in range(engine._barge_in_required + 2):
        await engine._tick(TurnEndResult(False, 0.0, None), frame_rms=0.2)
    assert engine._state == "bot_speaking"

    # A clearly louder frame (real user speaking over the bot) barges in.
    for _ in range(engine._barge_in_required):
        await engine._tick(TurnEndResult(False, 0.0, None), frame_rms=0.8)
    assert engine._state == "user_speaking"


@pytest.mark.asyncio
async def test_barge_in_with_aec_uses_energy_gate_only():
    cfg = _base_config()

    class FakeAEC:
        def add_reference(self, audio):
            pass

        def process_mic(self, frame):
            return frame

        def reset(self):
            pass

    engine = _make_engine(cfg)
    engine.aec = FakeAEC()  # AEC present: echo is cancelled by AEC
    engine._set_state("bot_speaking")
    engine.uninterruptible_until = -1.0

    # With AEC, a low-energy residual frame must NOT barge in (existing gate).
    for _ in range(engine._barge_in_required + 2):
        await engine._tick(TurnEndResult(False, 0.0, None), frame_rms=0.001)
    assert engine._state == "bot_speaking"

    # Real user speech above the energy floor barges in.
    for _ in range(engine._barge_in_required):
        await engine._tick(TurnEndResult(False, 0.0, None), frame_rms=0.1)
    assert engine._state == "user_speaking"


@pytest.mark.asyncio
async def test_barge_in_cancels_committed_reply():
    cfg = _base_config()
    cfg.turn.detector = "llm"
    cfg.turn.min_partial_chars = 0
    orch = FakeOrchestrator(
        TurnResult(
            TurnDecision.RESPOND,
            response="Got it, I will save that.",
            partial="my phone number is 412",
        )
    )
    engine = _make_engine(cfg, orchestrator=orch)
    engine._set_state("user_speaking")
    await engine._poll_orchestrator()  # commits -> bot_speaking + _speak_committed
    assert engine._state == "bot_speaking"
    assert engine._response_task is not None

    engine.uninterruptible_until = -1.0
    for _ in range(engine._barge_in_required):
        await engine._tick(TurnEndResult(False, 0.0, None), frame_rms=0.8)
    assert engine._state == "user_speaking"
    # The committed-speech task was cancelled; suppress the CancelledError.
    with pytest.raises(asyncio.CancelledError):
        await engine._response_task
