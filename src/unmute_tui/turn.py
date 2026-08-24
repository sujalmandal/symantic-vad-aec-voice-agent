"""LLM turn orchestrator: continuous semantic turn detection.

The "brain" of the LLM turn detector. Given a streaming partial transcript
(Process A) and VAD/audio cues (Process B), it polls the LLM to decide whether
the user has finished their turn:

* ``WAIT``   — the user is pausing briefly; keep waiting.
* ``THINK``  — the user needs more time; keep waiting patiently.
* ``RESPOND`` — the text is semantically complete AND the audio shows a
  conversational hand-off; commit the (already drafted) reply immediately.

Because the decision call also drafts the response, the engine can speak the
reply with near-zero turn-end latency — no separate turn-end transcription +
cold LLM generation. A parse error or empty reply never breaks the
conversation: it falls back to ``WAIT`` so the reliable audio silence timeout
still ends the turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_orchestrator_prompt,
)

if TYPE_CHECKING:  # pragma: no cover
    from .llm import LLM


class TurnDecision(str, Enum):
    WAIT = "WAIT"
    THINK = "THINK"
    RESPOND = "RESPOND"


@dataclass
class VADState:
    """Snapshot of the audio/turn cues fed to the orchestrator each poll."""

    speaking: bool = False
    silence_seconds: float = 0.0
    smart_turn_probability: float | None = None
    turn_seconds: float = 0.0


@dataclass
class TurnResult:
    """Outcome of one orchestrator poll."""

    decision: TurnDecision
    response: str = ""  # committed reply text when decision == RESPOND
    partial: str = ""  # the partial transcript the decision was based on


def _extract_json_object(text: str) -> dict:
    """Return the first JSON object embedded in an LLM reply, or {} if none."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_decision(text: str, partial: str = "") -> TurnResult:
    """Parse an orchestrator LLM reply into a TurnResult (degrades to WAIT)."""
    data = _extract_json_object(text)
    decision_raw = (data.get("decision") or "").strip().upper()
    try:
        decision = TurnDecision(decision_raw)
    except ValueError:
        decision = TurnDecision.WAIT
    response = (data.get("response") or "").strip()
    return TurnResult(decision=decision, response=response, partial=partial)


class LLMTurnOrchestrator:
    """Continuously polls the LLM to decide whether to respond."""

    def __init__(self, llm: "LLM") -> None:
        self.llm = llm

    def orchestration_context(self, chat_history):
        """Recent user/assistant turns fed to the orchestrator for grounding.

        Excludes the system prompt and any empty messages. Kept small so the
        orchestrator focuses on the current turn while still "thinking ahead".
        """
        context: list[dict[str, str]] = []
        for m in chat_history:
            if m["role"] == "system":
                continue
            content = (m.get("content") or "").strip()
            if content:
                context.append({"role": m["role"], "content": content})
        return context[-8:]

    async def decide(
        self,
        partial_text: str,
        vad_state: VADState | None,
        recent_turns: list[dict[str, str]] | None = None,
    ) -> TurnResult:
        """Run one poll and return the turn decision (never raises).

        Args:
            partial_text: current streaming partial transcript.
            vad_state: current VAD/audio snapshot (may be None before speech).
            recent_turns: prior conversation turns for grounding.
        """
        prompt = build_orchestrator_prompt(partial_text, vad_state, recent_turns)
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            buf = ""
            async for delta in self.llm.stream(messages):
                buf += delta
        except Exception:  # noqa: BLE001 — a failed poll must not break the chat
            return TurnResult(TurnDecision.WAIT, partial=partial_text)
        return parse_decision(buf, partial_text)
