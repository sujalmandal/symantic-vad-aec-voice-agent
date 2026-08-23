import asyncio

import numpy as np
import pytest

from unmute_tui.config import Config
from unmute_tui.engine import (
    AssistantDone,
    ConversationEngine,
    Log,
    SessionEnd,
    StateChanged,
    UserTranscript,
)
from unmute_tui.vad.semantic_vad import TurnEndResult


class _FakeLoader:
    def load(self):
        pass


class FakeLLM:
    def __init__(self, words):
        self.words = words

    async def stream(self, messages):
        for w in self.words:
            yield w


class FakeTTS:
    def __init__(self, audio=None):
        self.audio = audio if audio is not None else np.zeros(160, dtype=np.float32)

    def load(self):
        pass

    async def synthesize(self, text):
        return self.audio

    async def stream(self, text):
        if len(self.audio):
            yield self.audio


class FakeTranscriber:
    def __init__(self, text="hello there"):
        self.text = text

    def load(self):
        pass

    def transcribe(self, audio):
        from unmute_tui.stt import Transcription

        return Transcription(text=self.text, segments=[])


class FakeMic:
    def __init__(self, frames):
        self._frames = frames

    def start(self):
        pass

    def stop(self):
        pass

    async def frames(self):
        for f in self._frames:
            yield f


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


class FakeVAD:
    def __init__(self, results=None, is_speaking=False):
        self.results = list(results or [])
        self.is_speaking = is_speaking
        self.smart_turn = _FakeLoader()
        self.silero = _FakeLoader()

    def process_frame(self, frame):
        if self.results:
            return self.results.pop(0)
        return TurnEndResult(False, 0.0, None)

    def reset(self):
        pass


def _make_engine(vad, llm=None, transcriber=None, mic=None):
    config = Config.from_env(env_file=None)
    events = asyncio.Queue()
    engine = ConversationEngine(
        config=config,
        vad=vad,
        transcriber=transcriber or FakeTranscriber(),
        llm=llm or FakeLLM(["hello", " world"]),
        tts=FakeTTS(),
        mic=mic or FakeMic([]),
        player=FakePlayer(),
        events=events,
    )
    return engine, events


@pytest.mark.asyncio
async def test_generate_response_streams_and_finalizes():
    engine, events = _make_engine(FakeVAD())
    await engine._generate_response()
    assert engine._state == "bot_speaking"
    await engine._response_task

    assert engine.chat_history[-1]["role"] == "assistant"
    assert engine.chat_history[-1]["content"] == "hello world"

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    assert any(isinstance(e, AssistantDone) for e in seen)
    assert any(isinstance(e, StateChanged) and e.state == "waiting_for_user" for e in seen)


@pytest.mark.asyncio
async def test_turn_end_transcribes_and_generates():
    # First frame triggers turn-end with audio; then silence frames.
    audio = np.zeros(1600, dtype=np.float32)
    vad = FakeVAD([TurnEndResult(True, 0.9, 0.9, audio)])
    engine, events = _make_engine(vad)
    await engine._handle_turn_end(audio)

    assert engine.chat_history[-1]["role"] == "user"
    assert engine.chat_history[-1]["content"] == "hello there"
    assert engine._state == "bot_speaking"
    await engine._response_task

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    assert any(isinstance(e, UserTranscript) and e.text == "hello there" for e in seen)


@pytest.mark.asyncio
async def test_interrupt_bot():
    engine, events = _make_engine(FakeVAD())
    # Simulate the bot mid-reply.
    engine._set_state("bot_speaking")
    engine._assistant_text = "hello"
    engine._add_message("assistant", "hello")
    await engine.interrupt_bot()
    assert engine._state == "user_speaking"
    assert engine.chat_history[-1]["role"] == "assistant"
    assert engine.chat_history[-1]["content"].endswith("—")


@pytest.mark.asyncio
async def test_goodbye_emits_session_end():
    engine, events = _make_engine(FakeVAD(), llm=FakeLLM(["Bye!"]))
    await engine._generate_response()
    await engine._response_task
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    assert any(isinstance(e, SessionEnd) for e in seen)
