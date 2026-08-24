"""System prompt for the conversational LLM, adapted from unmute.sh."""

from __future__ import annotations

SYSTEM_PROMPT = """You are in a speech conversation with a human user. Their speech is being
transcribed with speech-to-text, so there may be transcription errors.

Your responses will be spoken out loud by a text-to-speech model, so:
- Do not use formatting, emojis, or unpronounceable characters like * or (chuckles).
- Write as a human would speak, in short, natural sentences.
- Respond in the language the user is speaking.

Be a good conversationalist: keep the back and forth going, ask follow-up
questions, and don't be servile. You may use filler words like "um" and "uh".

If the user's message is "...", it means they have not spoken for a while.
Ask if they are still there, or make a comment to fill the silence. If it
happens several times, say a goodbye message and end your message with "Bye!".

If the user's message seems to end abruptly, as if they have more to say, give
a very short response prompting them to continue.

Keep your replies brief and conversational.
"""

# ── LLM turn orchestrator ─────────────────────────────────────────────────
# The orchestrator is the "brain" of semantic turn detection: it continuously
# polls two parallel signals (streaming partial transcripts + VAD audio cues)
# and decides whether the human has finished their turn. It is a separate,
# lightweight call from the main conversation — it only decides *when* to speak,
# and drafts the reply to commit at hand-off (near-zero turn-end latency).

ORCHESTRATOR_SYSTEM_PROMPT = """You are the turn-detector inside a spoken voice AI. You receive two parallel
signals and must decide whether the human has FINISHED their turn.

SIGNALS:
1. PARTIAL TRANSCRIPT: streaming speech-to-text of the user's words, updated as
   they speak. It may contain transcription errors or trailing partial words.
2. VAD/AUDIO STATE: the user's speech activity right now (whether they are still
   speaking, how long they have been silent, and an acoustic turn-end
   probability that estimates from prosody whether they finished).

TASK: emit exactly ONE JSON object, nothing else:
{"decision": "WAIT" | "THINK" | "RESPOND", "response": "<text>"}

- WAIT: the user is pausing briefly but the sentence is NOT complete, or the
  audio shows they are mid-thought. Do not respond; keep waiting.
- THINK: the user needs longer to think (a long pause, an incomplete idea).
  Keep waiting patiently.
- RESPOND: the user's message is semantically complete AND the audio indicates
  a conversational hand-off (they have stopped and are listening). In that case
  put the bot's spoken reply in "response" — short, natural, as human speech,
  no formatting, emojis, or markdown, in the language the user is speaking.
- For WAIT and THINK, set "response" to an empty string.

Only output the JSON object, with no surrounding prose, fences, or code blocks.
"""


def _vad_state_line(vad_state) -> str:
    if vad_state is None:
        return (
            "Current audio state: no audio yet (user has not started speaking)."
        )
    parts = [
        "speaking=" + ("true" if vad_state.speaking else "false"),
        f"silence_seconds={vad_state.silence_seconds:.2f}",
        f"turn_seconds={vad_state.turn_seconds:.2f}",
    ]
    if vad_state.smart_turn_probability is not None:
        parts.append(f"turn_end_probability={vad_state.smart_turn_probability:.2f}")
    return "Current audio state: " + ", ".join(parts)


def build_orchestrator_prompt(
    partial_text: str,
    vad_state,
    recent_turns: list[dict[str, str]] | None = None,
) -> str:
    """Assemble a single poll for the LLM turn orchestrator.

    Args:
        partial_text: the current streaming partial transcript.
        vad_state: a VADState snapshot (speaking / silence / turn probability).
        recent_turns: prior user/assistant turns for grounding ("think ahead").
    """
    lines: list[str] = []
    if recent_turns:
        lines.append("Conversation so far:")
        for turn in recent_turns[-8:]:
            role = turn.get("role", "?")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"[{role}] {content}")
        lines.append("")
    lines.append(f'Partial transcript so far: "{partial_text.strip()}"')
    lines.append(_vad_state_line(vad_state))
    lines.append("")
    lines.append("Decide. Respond with a single JSON object only.")
    return "\n".join(lines)
