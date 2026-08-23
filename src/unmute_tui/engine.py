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
from .config import SAMPLE_RATE, Config
from .llm import INTERRUPTION_CHAR, USER_SILENCE_MARKER, LLM, rechunk_to_words
from .prompts import SYSTEM_PROMPT
from .stt import Transcriber
from .tts import TTSBackend
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
        self._barge_in_required = 10  # ~200 ms of sustained speech
        self._silence_count = 0
        self._mic_muted = False

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
        loaders (faster-whisper / mlx-audio), so we load synchronously up front.
        """
        self.vad.smart_turn.load()
        self.vad.silero.load()
        self.transcriber.load()
        self.tts.load()

    async def run(self) -> None:
        self.mic.start()
        self.player.start()
        self._emit(Log("Listening... (Ctrl+C to quit)"))
        if self.aec is not None:
            self._emit(Log("WebRTC AEC3 echo cancellation active."))

        # Bot greets first so the user hears the TTS working.
        self._add_message("user", "Hello!")
        await self._generate_response()

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
            await self._tick(result)

    async def _tick(self, result) -> None:
        if self._state == "bot_speaking":
            # Barge-in: the user starts talking over the bot. Require sustained
            # speech so the bot's own echo doesn't interrupt it.
            if self.vad.is_speaking:
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
                if self.aec is not None:
                    # Feed the TTS audio as the far-end reference so the AEC can
                    # remove it from the mic signal.
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
        if self._response_task is not None:
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
        self.mic.stop()
        self.player.stop()
