import pytest

from unmute_tui.prompts import build_orchestrator_prompt
from unmute_tui.turn import (
    LLMTurnOrchestrator,
    TurnDecision,
    TurnResult,
    VADState,
    parse_decision,
)


def test_parse_decision_respond():
    result = parse_decision(
        '{"decision": "RESPOND", "response": "Got it."}', partial="my number is 412"
    )
    assert result.decision == TurnDecision.RESPOND
    assert result.response == "Got it."
    assert result.partial == "my number is 412"


def test_parse_decision_wait_and_think():
    assert parse_decision('{"decision":"WAIT","response":""}').decision == TurnDecision.WAIT
    assert parse_decision('{"decision":"THINK","response":""}').decision == TurnDecision.THINK


def test_parse_decision_tolerates_prose_and_fences():
    text = "```json\n{\"decision\": \"RESPOND\", \"response\": \"Sure, go ahead.\"}\n```"
    result = parse_decision(text)
    assert result.decision == TurnDecision.RESPOND
    assert result.response == "Sure, go ahead."


def test_parse_decision_unknown_or_malformed_falls_back_to_wait():
    assert parse_decision('{"decision": "MAYBE", "response": "x"}').decision == TurnDecision.WAIT
    assert parse_decision("not json at all").decision == TurnDecision.WAIT
    assert parse_decision("").decision == TurnDecision.WAIT


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.messages = None

    async def stream(self, messages):
        self.messages = messages
        yield self.text


class RaisingLLM:
    async def stream(self, messages):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_decide_returns_parsed_decision():
    llm = FakeLLM('{"decision":"RESPOND","response":"Sure."}')
    orch = LLMTurnOrchestrator(llm)
    result = await orch.decide(
        "my number is 412",
        VADState(speaking=False, silence_seconds=0.6, turn_seconds=3.0),
        [{"role": "user", "content": "hi"}],
    )
    assert result.decision == TurnDecision.RESPOND
    assert result.response == "Sure."
    # System + user messages sent to the LLM.
    assert llm.messages[0]["role"] == "system"
    assert llm.messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_decide_never_raises_and_falls_back_to_wait():
    orch = LLMTurnOrchestrator(RaisingLLM())
    result = await orch.decide("partial", VADState())
    assert result.decision == TurnDecision.WAIT


def test_build_prompt_includes_partial_and_vad_state():
    vad = VADState(speaking=False, silence_seconds=0.7, smart_turn_probability=0.83, turn_seconds=4.2)
    prompt = build_orchestrator_prompt(
        "my phone number is 412",
        vad,
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
    )
    assert "my phone number is 412" in prompt
    assert "silence_seconds=0.70" in prompt
    assert "turn_end_probability=0.83" in prompt
    assert "[user] Hi" in prompt
    assert "[assistant] Hello" in prompt


def test_orchestration_context_excludes_system_and_empty():
    llm = FakeLLM("")
    orch = LLMTurnOrchestrator(llm)
    context = orch.orchestration_context(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "  "},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert context == [{"role": "user", "content": "Hi"}]
