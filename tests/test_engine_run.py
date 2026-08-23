import asyncio

import numpy as np
import pytest

from unmute_tui.config import Config
from unmute_tui.engine import ConversationEngine
from unmute_tui.vad.semantic_vad import TurnEndResult

from test_engine import (
    FakeLLM,
    FakeMic,
    FakePlayer,
    FakeTTS,
    FakeTranscriber,
    FakeVAD,
)


def _make_engine(vad, mic):
    config = Config.from_env(env_file=None)
    events = asyncio.Queue()
    engine = ConversationEngine(
        config=config,
        vad=vad,
        transcriber=FakeTranscriber("hello world"),
        llm=FakeLLM(["hi ", "there"]),
        tts=FakeTTS(),
        mic=mic,
        player=FakePlayer(),
        events=events,
    )
    return engine, events


@pytest.mark.asyncio
async def test_run_full_loop():
    # No turn-end; just verify the run loop starts, greets, and completes.
    vad = FakeVAD()
    mic = FakeMic([np.zeros(320, dtype=np.float32)] * 4)
    engine, events = _make_engine(vad, mic)

    await engine.run()
    # Let the greeting response task finish.
    if engine._response_task is not None:
        await engine._response_task

    roles = [m["role"] for m in engine.chat_history]
    contents = [m["content"] for m in engine.chat_history]
    assert "assistant" in roles
    # The bot's greeting reply should be present.
    assert any("hi there" in c for c in contents)
