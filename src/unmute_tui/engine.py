"""Conversation engine: the fluent turn-taking state machine.

Ported from unmute.sh's `UnmuteHandler`/`Chatbot`. It drives the full-duplex
loop: capture mic audio, run the semantic VAD, transcribe turns, stream the
LLM response through the TTS, and support barge-in interruption.

States: waiting_for_user -> user_speaking -> bot_speaking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from .audio import AudioPlayer, Microphone
from .config import FRAME_TIME_SEC, SAMPLE_RATE, Config
from .echogate import PlaybackEchoGate
from .llm import INTERRUPTION_CHAR, USER_SILENCE_MARKER, LLM, rechunk_to_words
from .prompts import SYSTEM_PROMPT
from .stt import Transcriber
from .tts import TTSBackend
from .turn import TurnDecision, TurnResult, VADState
from .vad import SemanticVAD

# Sentence-ending punctuation that triggers incremental TTS synthesis.
_SENTENCE_END = ".!?"
_MAX_TTS_CHUNK = 200


# ── Events emitted to the UI ────────────────────────────────────────────────
@dataclass
class StateChanged:
    state: str


@dataclass
class VADUpdate:
    probability: float
    raw_probability: float | None


@dataclass
class PartialUpdate:
    """A new partial transcript and/or the orchestrator's latest decision."""

    text: str
    decision: str = ""


@dataclass
class UserTranscript:
    text: str


@dataclass
class AssistantDelta:
    text: str


@dataclass
class AssistantDone:
    text: str


@dataclass
class Latency:
    metric: str
    value: float


@dataclass
class Log:
    message: str


@dataclass
class SessionEnd:
    reason: str


EngineEvent = (
    StateChanged
    | VADUpdate
    | PartialUpdate
    | UserTranscript
    | AssistantDelta
    | AssistantDone
    | Latency
    | Log
    | SessionEnd
)


class ConversationEngine:
    def __init__(
        self,
        config: Config,
        vad: SemanticVAD,
        transcriber: Transcriber,
        llm: LLM,
        tts: TTSBackend,
        mic: Microphone,
        player: AudioPlayer,
        events: asyncio.Queue[EngineEvent],
        aec=None,  # optional WebRTCAEC
        backchannel=None,  # optional BotBackchannel
        orchestrator=None,  # optional LLMTurnOrchestrator
    ) -> None:
        self.config = config
        self.vad = vad
        self.transcriber = transcriber
        self.llm = llm
        self.tts = tts
        self.mic = mic
        self.player = player
        self.events = events
        self.aec = aec
        self.backchannel = backchannel
        self.orchestrator = orchestrator

        self.chat_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._state = "waiting_for_user"
        self._interrupted = False
        self._response_task: asyncio.Task | None = None
        self._assistant_text = ""

        self.n_samples_received = 0
        self.audio_time = 0.0
        self.waiting_for_user_start = 0.0
        self.uninterruptible_until = 0.0
        # Consecutive speech frames required to trigger barge-in (reduces
        # echo-triggered self-interruption, since echo is usually intermittent).
        self._barge_in_frames = 0
        self._barge_in_required = config.vad.barge_in_required_frames
        self._silence_count = 0
        self._mic_muted = False
        self._partial_text = ""
        self._orchestrator_task: asyncio.Task | None = None
        # Playback-aware echo gate: keeps the bot from interrupting itself with
        # its own TTS echo when AEC is unavailable (barge-in without AEC).
        self.echo_gate = PlaybackEchoGate(
            margin_db=config.vad.barge_in_over_playback_db
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _emit(self, event: EngineEvent) -> None:
        self.events.put_nowait(event)

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self._emit(StateChanged(state))

    def _add_message(self, role: str, content: str) -> None:
        if self.chat_history and self.chat_history[-1]["role"] == role:
            self.chat_history[-1]["content"] += content
        else:
            self.chat_history.append({"role": role, "content": content})

    def _set_last_message(self, role: str, content: str) -> None:
        if self.chat_history and self.chat_history[-1]["role"] == role:
            self.chat_history[-1]["content"] = content
        else:
            self.chat_history.append({"role": role, "content": content})

    def _preprocessed_messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in self.chat_history:
            content = m["content"].strip().removesuffix(INTERRUPTION_CHAR)
            if content == "":
                continue
            if out and m["role"] == out[-1]["role"]:
                out[-1]["content"] += " " + content
            else:
                out.append({"role": m["role"], "content": content})
        if out and out[0]["role"] == "system" and (
            len(out) < 2 or out[1]["role"] == "assistant"
        ):
            out = [out[0]] + [{"role": "user", "content": "Hello."}] + out[1:]
        return out

    # ── main loop ──────────────────────────────────────────────────────────
    def load(self) -> None:
        """Pre-load all models (blocking).

        This is the "gate": it must be called before the asyncio event loop
        starts (i.e. before the TUI/headless loop runs). Loading models inside
        the event loop via threads can conflict with subprocess-spawning model
        loaders (e.g. faster-whisper), so we load synchronously up front.
        """
        self.vad.smart_turn.load()
        self.vad.silero.load()
        self.transcriber.load()
        self.tts.load()
        if self.backchannel is not None:
            self.backchannel.load()

    async def run(self) -> None:
        self.mic.start()
        self.player.start()
        self._emit(Log("Listening... (Ctrl+C to quit)"))
        if self.aec is not None:
            self._emit(Log("WebRTC AEC3 echo cancellation active."))

        # Bot greets first so the user hears the TTS working.
        self._add_message("user", "Hello!")
        await self._generate_response()

        # LLM turn orchestrator: a background loop that, while the user is
        # speaking, polls partial transcripts + VAD cues and may commit the
        # bot's reply early (near-zero turn-end latency).
        if self.orchestrator is not None and self.config.turn.detector == "llm":
            self._emit(Log("LLM turn orchestrator active (TURN_DETECTOR=llm)."))
            self._orchestrator_task = asyncio.create_task(self._orchestrator_loop())

        async for frame in self.mic.frames():
            self.n_samples_received += len(frame)
            self.audio_time = self.n_samples_received / SAMPLE_RATE
            if self._mic_muted:
                # Bot is speaking and the mic is muted (half-duplex): ignore
                # mic input so the bot doesn't hear its own TTS and interrupt
                # itself. No echo cancellation is available.
                continue
            if self.aec is not None:
                # Route mic through the AEC to remove the bot's own echo.
                clean = self.aec.process_mic(frame)
                if clean is None:
                    continue  # not enough samples accumulated yet
                frame = clean
            result = self.vad.process_frame(frame)
            self._emit(VADUpdate(result.probability, result.raw_probability))
            frame_rms = float(np.sqrt(np.mean(frame**2)))
            await self._tick(result, frame_rms)

            # Active-listening backchannels: while the user is speaking, if VAP
            # predicts a backchannel, the bot emits a short ack without taking
            # the turn (state stays user_speaking).
            if (
                self.backchannel is not None
                and self._state == "user_speaking"
                and not self._mic_muted
            ):
                prob = self.backchannel.push(frame)
                if self.backchannel.should_ack(prob, self.audio_time):
                    self.backchannel.mark_acked(self.audio_time)
                    await self._speak_backchannel()

    def _is_user_bargein(self, frame_rms: float) -> bool:
        """Whether a mic frame while the bot speaks is the user (not its echo).

        With AEC, the cleaned frame's energy above the floor is the user's voice.
        Without AEC, we also require the frame to be clearly louder than what the
        bot has just been playing (the playback-aware echo gate), so the bot's own
        TTS echo never makes it interrupt itself.
        """
        if not self.vad.is_speaking:
            return False
        if frame_rms < self.config.vad.barge_in_min_rms:
            return False
        if self.aec is not None:
            return True
        return frame_rms >= self.echo_gate.threshold()

    async def _tick(self, result, frame_rms: float = 0.0) -> None:
        if self._state == "bot_speaking":
            # Barge-in: the user starts talking over the bot. Only count a frame
            # as the user when it is genuinely the user (see _is_user_bargein):
            # with AEC the echo is cancelled; without AEC we gate against the
            # bot's own playback. Louder, sustained user speech triggers barge-in.
            if self._is_user_bargein(frame_rms):
                self._barge_in_frames += 1
            else:
                self._barge_in_frames = 0
            if (
                self._barge_in_frames >= self._barge_in_required
                and self.audio_time > self.uninterruptible_until
            ):
                await self.interrupt_bot()
            return

        if result.turn_end:
            await self._handle_turn_end(result.audio)
            return

        if self._state == "waiting_for_user":
            if self.vad.is_speaking:
                self._set_state("user_speaking")
            elif (
                self.audio_time - self.waiting_for_user_start
                > self.config.vad.user_silence_timeout
            ):
                self._silence_count += 1
                if self._silence_count >= 3:
                    self._emit(SessionEnd("user silent too long"))
                    return
                self._emit(Log("Long silence detected"))
                self._add_message("user", USER_SILENCE_MARKER)
                await self._generate_response()

    async def _handle_turn_end(self, audio: np.ndarray | None) -> None:
        self._set_state("user_speaking")
        if audio is None or len(audio) == 0:
            self._set_state("waiting_for_user")
            self.waiting_for_user_start = self.audio_time
            return
        transcription = await asyncio.to_thread(self.transcriber.transcribe, audio)
        text = transcription.text.strip()
        if not text:
            self._set_state("waiting_for_user")
            self.waiting_for_user_start = self.audio_time
            return
        self._add_message("user", text)
        self._emit(UserTranscript(text))
        self._silence_count = 0
        await self._generate_response()

    # ── LLM turn orchestrator ───────────────────────────────────────────────
    async def _orchestrator_loop(self) -> None:
        """Background loop: while the user is speaking, poll the LLM turn
        detector and commit the bot's reply when it says RESPOND."""
        cfg = self.config.turn
        try:
            while True:
                await asyncio.sleep(cfg.poll_interval_sec)
                if self._state != "user_speaking":
                    continue
                await self._poll_orchestrator()
        except asyncio.CancelledError:
            raise

    async def _poll_orchestrator(self) -> None:
        """Sample the current partial transcript + VAD state and poll the LLM."""
        if self._state != "user_speaking":
            return
        audio = self.vad.get_turn_audio()
        if len(audio) == 0:
            return
        partial = (
            await asyncio.to_thread(
                self.transcriber.partial,
                audio,
                self.config.turn.stt_poll_window_sec,
            )
        ).text.strip()
        if len(partial) < self.config.turn.min_partial_chars:
            return
        if partial == self._partial_text:
            return  # no new text since the last poll
        self._partial_text = partial
        self._emit(PartialUpdate(partial, ""))

        vad_state = VADState(
            speaking=self.vad.is_speaking,
            silence_seconds=self.vad.silence_frames * FRAME_TIME_SEC,
            smart_turn_probability=self.vad.probability,
            turn_seconds=self.audio_time,
        )
        context = self.orchestrator.orchestration_context(self.chat_history)
        result = await self.orchestrator.decide(partial, vad_state, context)
        self._emit(PartialUpdate(partial, result.decision.value))
        if result.decision == TurnDecision.RESPOND:
            await self._commit_orchestrated_turn(result)

    async def _commit_orchestrated_turn(self, result: TurnResult) -> None:
        """End the user's turn via the orchestrator and speak the reply.

        The reply was already drafted by the orchestrator, so there is no
        separate turn-end transcription + cold LLM generation — this is the
        near-zero-latency path. If no usable reply came back, we simply do
        nothing and let the audio silence timeout end the turn normally.
        """
        if self._state != "user_speaking":
            return
        text = result.response.strip()
        if not text:
            return
        # Consume the current turn recording so it isn't re-processed.
        self.vad.reset()
        user_text = result.partial.strip() or self._partial_text
        if user_text:
            self._add_message("user", user_text)
            self._emit(UserTranscript(user_text))
        self._silence_count = 0
        self._assistant_text = text
        self._set_last_message("assistant", text)
        self._emit(AssistantDone(text))
        self._set_state("bot_speaking")
        self.uninterruptible_until = (
            self.audio_time + self.config.vad.uninterruptible_by_vad_time_sec
        )
        if self.config.vad.mute_mic_while_bot_speaking and self.aec is None:
            self._mic_muted = True
        self._response_task = asyncio.create_task(self._speak_committed(text))

    async def _speak_committed(self, text: str) -> None:
        """Play an orchestrator-committed reply, then return to waiting."""
        try:
            await self._speak(text)
            if text.endswith("Bye!"):
                self._emit(SessionEnd("assistant said goodbye"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._emit(Log(f"TTS error: {exc}"))
        finally:
            if not self._interrupted:
                self._set_state("waiting_for_user")
                self.waiting_for_user_start = self.audio_time
            if self.config.vad.mute_mic_while_bot_speaking and self.aec is None:
                self._mic_muted = False
                self.vad.reset()

    # ── response generation ────────────────────────────────────────────────
    async def _generate_response(self) -> None:
        if self._state == "bot_speaking":
            return
        self._set_state("bot_speaking")
        self.uninterruptible_until = (
            self.audio_time + self.config.vad.uninterruptible_by_vad_time_sec
        )
        if self.config.vad.mute_mic_while_bot_speaking and self.aec is None:
            self._mic_muted = True
            self.vad.reset()
        self._assistant_text = ""
        self._response_task = asyncio.create_task(self._response_coro())

    async def _response_coro(self) -> None:
        messages = self._preprocessed_messages()
        words: list[str] = []
        buffer = ""
        try:
            async for word in rechunk_to_words(self.llm.stream(messages)):
                if self._interrupted:
                    break
                words.append(word)
                self._assistant_text += word
                self._emit(AssistantDelta(word))
                buffer += word
                if any(p in buffer for p in _SENTENCE_END) or len(buffer) > _MAX_TTS_CHUNK:
                    await self._speak(buffer)
                    buffer = ""
            if buffer.strip():
                await self._speak(buffer)

            text = "".join(words).strip()
            self._set_last_message("assistant", text)
            self._emit(AssistantDone(text))
            self._emit(Latency("reply_words", len(words)))
            if text.endswith("Bye!"):
                self._emit(SessionEnd("assistant said goodbye"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._emit(Log(f"LLM/TTS error: {exc}"))
        finally:
            if not self._interrupted:
                self._set_state("waiting_for_user")
                self.waiting_for_user_start = self.audio_time
            if self.config.vad.mute_mic_while_bot_speaking and self.aec is None:
                self._mic_muted = False
                self.vad.reset()

    async def _speak(self, text: str) -> None:
        async for chunk in self.tts.stream(text):
            if len(chunk):
                self.echo_gate.add_playback(chunk)
                if self.aec is not None:
                    # Feed the TTS audio as the far-end reference so the AEC can
                    # remove it from the mic signal.
                    self.aec.add_reference(chunk)
                await asyncio.to_thread(self.player.play, chunk)

    async def _speak_backchannel(self) -> None:
        """Emit a short backchannel ack ("Mm-hmm") while the user is speaking.

        Does NOT advance the conversation state (the user keeps their turn);
        the ack audio is fed to the AEC as reference so it isn't misheard as
        the user.
        """
        self._emit(Log("backchannel"))
        text = self.config.backchannel.ack_text
        async for chunk in self.tts.stream(text):
            if len(chunk):
                self.echo_gate.add_playback(chunk)
                if self.aec is not None:
                    self.aec.add_reference(chunk)
                await asyncio.to_thread(self.player.play, chunk)

    async def interrupt_bot(self) -> None:
        if self._state != "bot_speaking":
            return
        self._interrupted = True
        if self._response_task is not None:
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
        self._interrupted = False
        self.player.clear()
        if self.aec is not None:
            self.aec.reset()  # clear stale far-end reference on barge-in
        self.echo_gate.reset()  # clear the bot's own playback history
        # Start a fresh VAD turn so the user's barge-in speech (not the bot's
        # echo) is what gets transcribed.
        self.vad.reset()
        self._barge_in_frames = 0
        if self._assistant_text:
            self._set_last_message(
                "assistant", self._assistant_text + INTERRUPTION_CHAR
            )
        self._emit(Log("Interrupted by user"))
        self._set_state("user_speaking")

    async def shutdown(self) -> None:
        if self._orchestrator_task is not None:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._response_task is not None:
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
        self.mic.stop()
        self.player.stop()
