import asyncio

import numpy as np
import pytest

from unmute_tui.config import Config
from unmute_tui.engine import (
    AssistantDone,
    ConversationEngine,
    StateChanged,
    UserTranscript,
)
from unmute_tui.turn import TurnDecision, TurnResult, VADState


class FakeVAD:
    def __init__(self, audio):
        self.audio = audio
        self.is_speaking = False
        self.silence_frames = 30
        self.probability = 0.0

    def get_turn_audio(self):
        return self.audio

    def reset(self):
        pass

    def load(self):
        pass


class FakeTranscriber:
    def __init__(self, text):
        self.text = text

    def load(self):
        pass

    def partial(self, audio, window_sec):
        from unmute_tui.stt import Transcription

        return Transcription(text=self.text, segments=[])


class FakeLLM:
    async def stream(self, messages):
        yield ""


class FakeTTS:
    def __init__(self, audio=None):
        self.audio = audio if audio is not None else np.zeros(160, dtype=np.float32)

    def load(self):
        pass

    async def stream(self, text):
        if len(self.audio):
            yield self.audio


class FakeMic:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    async def frames(self):
        # Never yields; run() is driven manually in tests.
        if False:
            yield np.zeros(320, dtype=np.float32)


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
    def __init__(self, result, calls=None):
        self.result = result
        self.calls = calls if calls is not None else []

    def orchestration_context(self, chat_history):
        return []

    async def decide(self, partial, vad_state, context):
        self.calls.append((partial, vad_state))
        return self.result


def _make_engine(orchestrator, partial="my phone number is 412"):
    cfg = Config.from_env(env_file=None)
    cfg.turn.detector = "llm"
    cfg.turn.min_partial_chars = 0
    events = asyncio.Queue()
    engine = ConversationEngine(
        config=cfg,
        vad=FakeVAD(np.zeros(3200, dtype=np.float32)),
        transcriber=FakeTranscriber(partial),
        llm=FakeLLM(),
        tts=FakeTTS(),
        mic=FakeMic(),
        player=FakePlayer(),
        events=events,
        orchestrator=orchestrator,
    )
    return engine, events


@pytest.mark.asyncio
async def test_responsd_commits_turn_and_speaks():
    orch = FakeOrchestrator(
        TurnResult(
            TurnDecision.RESPOND,
            response="Got it, I will save that.",
            partial="my phone number is 412",
        )
    )
    engine, events = _make_engine(orch)
    engine._set_state("user_speaking")
    await engine._poll_orchestrator()

    # The user's partial became a real message and the reply was committed.
    assert engine.chat_history[-1]["role"] == "assistant"
    assert engine.chat_history[-1]["content"] == "Got it, I will save that."
    assert any(
        m["role"] == "user" and m["content"] == "my phone number is 412"
        for m in engine.chat_history
    )

    # The committed reply is spoken and the engine returns to waiting.
    assert engine._state == "bot_speaking"
    await engine._response_task
    assert engine._state == "waiting_for_user"

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    assert any(isinstance(e, UserTranscript) for e in seen)
    assert any(isinstance(e, AssistantDone) for e in seen)


@pytest.mark.asyncio
async def test_wait_does_not_commit():
    orch = FakeOrchestrator(TurnResult(TurnDecision.WAIT, partial="partial"))
    engine, events = _make_engine(orch)
    engine._set_state("user_speaking")
    await engine._poll_orchestrator()
    assert engine._state == "user_speaking"
    assert not any(m["role"] == "assistant" for m in engine.chat_history)


@pytest.mark.asyncio
async def test_empty_respons_falls_back_without_commit():
    orch = FakeOrchestrator(TurnResult(TurnDecision.RESPOND, response="", partial="x"))
    engine, events = _make_engine(orch)
    engine._set_state("user_speaking")
    await engine._poll_orchestrator()
    assert engine._state == "user_speaking"
    assert engine._response_task is None


@pytest.mark.asyncio
async def test_short_partial_not_polled():
    class _RaiseOrch:
        async def decide(self, *a, **k):
            raise AssertionError("orchestrator should not be called for short partial")

    engine, events = _make_engine(_RaiseOrch())
    engine.config.turn.min_partial_chars = 100  # partial is much shorter
    engine._set_state("user_speaking")
    await engine._poll_orchestrator()
    assert engine._state == "user_speaking"


@pytest.mark.asyncio
async def test_state_not_user_speaking_is_ignored():
    orch = FakeOrchestrator(TurnResult(TurnDecision.RESPOND, response="nope"))
    engine, events = _make_engine(orch)
    engine._set_state("waiting_for_user")
    await engine._poll_orchestrator()
    assert engine._state == "waiting_for_user"
    assert orch.calls == []
