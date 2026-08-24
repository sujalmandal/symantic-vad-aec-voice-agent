#!/usr/bin/env python3
"""unmute-tui — single-file terminal replica of unmute.sh.

A full-duplex spoken conversation with an LLM, driven by a semantic VAD
(turn-end detection from speech content) and a fluent turn-taking engine with
barge-in interruption and streaming TTS (Kokoro).

Run:
    python unmute_tui.py --download-models   # one-time: fetch VAD model + voice
    python unmute_tui.py                     # TUI
    python unmute_tui.py --no-tui            # headless (prints events)
    python unmute_tui.py --list-devices      # list audio devices

Dependencies (install once):
    pip install numpy sounddevice onnxruntime transformers silero-vad \
                faster-whisper openai textual edge-tts python-dotenv kokoro-onnx

Configure via environment variables or a `.env` file (see README):
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TTS_BACKEND, VAD_THRESHOLD, ...
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import queue
import re
import subprocess
import sys
import threading
from abc import ABC, abstractmethod

# When running from the repo, expose the vendored `rvap` (VAP backchannel model)
# so bot backchannels work without installing the package.
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirror unmute.sh's kyutai_constants)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_TIME_SEC = 0.02  # 20 ms frames
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_TIME_SEC)  # 320

DEFAULT_VAD_THRESHOLD = 0.6
DEFAULT_USER_SILENCE_TIMEOUT = 7.0
DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3.0
SMART_TURN_WINDOW_SEC = 8.0

SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
KOKORO_ONNX_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)
KOKORO_ONNX_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"

INTERRUPTION_CHAR = "—"  # em-dash
USER_SILENCE_MARKER = "..."
_SENTENCE_END = ".!?"
_MAX_TTS_CHUNK = 200


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str | None = None
    thinking: str | None = None


@dataclass
class VADConfig:
    threshold: float = DEFAULT_VAD_THRESHOLD
    user_silence_timeout: float = DEFAULT_USER_SILENCE_TIMEOUT
    uninterruptible_by_vad_time_sec: float = DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC
    smart_turn_window_sec: float = SMART_TURN_WINDOW_SEC
    energy_threshold: float = 0.01
    # Hard turn-end silence fallback (reliable even if semantic model errs).
    turn_end_silence_sec: float = 0.7
    # Semantic accelerator: Smart Turn can end the turn after this much silence.
    semantic_min_silence_sec: float = 0.4
    # Barge-in robust: min RMS (filters AEC echo residual) + frames required.
    barge_in_min_rms: float = 0.03
    barge_in_required_frames: int = 15
    mute_mic_while_bot_speaking: bool = True


@dataclass
class BackchannelConfig:
    enabled: bool = False
    vap_bc_model: str = "models/vap-bc_state_dict_erica_10hz_3000msec.pt"
    cpc_model: str = "models/60k_epoch4-d0f474de.pt"
    frame_rate: int = 10
    context_len_sec: float = 3.0
    threshold: float = 0.5
    cooldown_sec: float = 3.0
    ack_text: str = "Mm-hmm."


@dataclass
class TTSConfig:
    backend: str = "chatterbox"  # chatterbox | kokoro | edge | piper
    voice: str = "en-US-AriaNeural"  # edge-tts voice / piper voice path
    # Chatterbox-Turbo (mlx-audio, MLX on Apple Silicon).
    chatterbox_model: str = "mlx-community/chatterbox-turbo-4bit"
    # Kokoro-82M (kokoro-onnx).
    kokoro_model: str = "models/kokoro-v1.0.onnx"
    kokoro_voices: str = "models/voices-v1.0.bin"
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    kokoro_lang: str = "en-us"


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None


@dataclass
class AECConfig:
    enabled: bool = True
    delay_ms: int = 30
    noise_suppression: bool = True
    ns_level: int = 1


@dataclass
class STTConfig:
    """Speech-to-text backend selection (mirrors src/unmute_tui/config.py)."""

    backend: str = "parakeet"
    model: str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    language: str | None = None
    models_dir: Path = Path("models/stt")
    threads: int = 2


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    aec: AECConfig = field(default_factory=AECConfig)
    backchannel: BackchannelConfig = field(default_factory=BackchannelConfig)
    models_dir: Path = Path("models")
    stt: STTConfig = field(default_factory=STTConfig)
    debug: bool = False

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Config":
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        models_dir = Path(os.getenv("MODELS_DIR", "models")).expanduser().resolve()
        return cls(
            llm=LLMConfig(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LLM_API_KEY", "ollama"),
                model=os.getenv("LLM_MODEL") or None,
                thinking=os.getenv("LLM_THINKING") or None,
            ),
            vad=VADConfig(
                threshold=_env_float("VAD_THRESHOLD", DEFAULT_VAD_THRESHOLD),
                user_silence_timeout=_env_float(
                    "USER_SILENCE_TIMEOUT", DEFAULT_USER_SILENCE_TIMEOUT
                ),
                uninterruptible_by_vad_time_sec=_env_float(
                    "UNINTERRUPTIBLE_BY_VAD_TIME_SEC",
                    DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC,
                ),
                energy_threshold=_env_float("VAD_ENERGY_THRESHOLD", 0.01),
                turn_end_silence_sec=_env_float("TURN_END_SILENCE_SEC", 0.7),
                semantic_min_silence_sec=_env_float(
                    "SEMANTIC_MIN_SILENCE_SEC", 0.4
                ),
                barge_in_min_rms=_env_float("BARGE_IN_MIN_RMS", 0.03),
                barge_in_required_frames=_env_int("BARGE_IN_REQUIRED_FRAMES", 15),
                mute_mic_while_bot_speaking=_env_bool(
                    "MUTE_MIC_WHILE_BOT_SPEAKING", True
                ),
            ),
            tts=TTSConfig(
                backend=os.getenv("TTS_BACKEND", "chatterbox").strip().lower(),
                voice=os.getenv("TTS_VOICE", "en-US-AriaNeural"),
                chatterbox_model=os.getenv(
                    "CHATTERBOX_MODEL", "mlx-community/chatterbox-turbo-4bit"
                ),
                kokoro_model=os.getenv(
                    "KOKORO_MODEL", "models/kokoro-v1.0.onnx"
                ),
                kokoro_voices=os.getenv(
                    "KOKORO_VOICES", "models/voices-v1.0.bin"
                ),
                kokoro_voice=os.getenv("KOKORO_VOICE", "af_heart"),
                kokoro_speed=_env_float("KOKORO_SPEED", 1.0),
                kokoro_lang=os.getenv("KOKORO_LANG", "en-us"),
            ),
            audio=AudioConfig(
                input_device=_env_int("INPUT_DEVICE", 0) or None,
                output_device=_env_int("OUTPUT_DEVICE", 0) or None,
            ),
            aec=AECConfig(
                enabled=_env_bool("AEC_ENABLED", True),
                delay_ms=_env_int("AEC_DELAY_MS", 30),
                noise_suppression=_env_bool("AEC_NOISE_SUPPRESSION", True),
                ns_level=_env_int("AEC_NS_LEVEL", 1),
            ),
            backchannel=BackchannelConfig(
                enabled=_env_bool("BACKCHANNEL_ENABLED", False),
                vap_bc_model=os.getenv(
                    "VAP_BC_MODEL",
                    "models/vap-bc_state_dict_erica_10hz_3000msec.pt",
                ),
                cpc_model=os.getenv("CPC_MODEL", "models/60k_epoch4-d0f474de.pt"),
                frame_rate=_env_int("BACKCHANNEL_FRAME_RATE", 10),
                context_len_sec=_env_float("BACKCHANNEL_CONTEXT_SEC", 3.0),
                threshold=_env_float("BACKCHANNEL_THRESHOLD", 0.5),
                cooldown_sec=_env_float("BACKCHANNEL_COOLDOWN_SEC", 3.0),
                ack_text=os.getenv("BACKCHANNEL_ACK_TEXT", "Mm-hmm."),
            ),
            models_dir=models_dir,
            stt=STTConfig(
                backend=os.getenv("STT_BACKEND", "parakeet").strip().lower(),
                model=os.getenv(
                    "STT_MODEL", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
                ),
                language=os.getenv("STT_LANGUAGE") or None,
                models_dir=Path(os.getenv("STT_MODELS_DIR", "models/stt"))
                .expanduser()
                .resolve(),
                threads=_env_int("STT_THREADS", 2),
            ),
            debug=_env_bool("DEBUG", False),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Audio (sounddevice)
# ─────────────────────────────────────────────────────────────────────────────
class AudioError(RuntimeError):
    pass


def list_devices() -> str:
    lines = []
    for i, dev in enumerate(sd.query_devices()):
        lines.append(
            f"{i}: {dev['name']}  (in={dev['max_input_channels']} "
            f"out={dev['max_output_channels']}, sr={dev['default_samplerate']})"
        )
    return "\n".join(lines)


class Microphone:
    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = SAMPLES_PER_FRAME,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time, status) -> None:
        audio = np.asarray(indata[:, 0], dtype=np.float32).copy()
        self._queue.put(audio)

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise AudioError(
                f"Could not open microphone (device={self.device}): {exc}"
            ) from exc

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    async def frames(self) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                frame = await loop.run_in_executor(None, self._queue.get, True, 0.1)
            except queue.Empty:
                continue
            yield frame


class AudioPlayer:
    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise AudioError(
                f"Could not open output device (device={self.device}): {exc}"
            ) from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def play(self, audio: np.ndarray) -> None:
        if self._stream is None:
            raise AudioError("AudioPlayer not started")
        if audio.ndim == 1:
            audio = audio[:, None]
        self._stream.write(np.asarray(audio, dtype=np.float32))

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.write(np.zeros((SAMPLES_PER_FRAME, 1), dtype=np.float32))

    def clear(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.start()


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic Echo Cancellation (WebRTC AEC3)
# ─────────────────────────────────────────────────────────────────────────────
class AECError(RuntimeError):
    pass


class WebRTCAEC:
    """WebRTC AEC3 echo cancellation via `pywebrtc-audio`.

    Same algorithm as Chrome: removes the far-end echo (the bot's own TTS) from
    the mic signal so full-duplex barge-in is possible. Needs the mic input plus
    a far-end reference (the TTS audio being played). ~52 dB echo attenuation.
    """

    BLOCK = 320  # 20 ms at 16 kHz, matches the engine's mic frames
    MAX_REF_SEC = 2.0

    def __init__(self, sample_rate=16000, delay_ms=30, noise_suppression=False, ns_level=1) -> None:
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as exc:
            raise RuntimeError(
                "pywebrtc-audio is not installed. Run `uv sync` (or "
                "`pip install pywebrtc-audio`)."
            ) from exc
        # Processing order: HP filter -> AEC3 -> NS -> AGC.
        self._ap = AudioProcessor(
            sample_rate=sample_rate,
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            ns_level=ns_level,
        )
        self._ap.stream_delay_ms = delay_ms
        self.ref_buf = np.zeros(0, dtype=np.float32)
        self.mic_buf = np.zeros(0, dtype=np.float32)

    def add_reference(self, audio) -> None:
        self.ref_buf = np.concatenate([self.ref_buf, np.asarray(audio, dtype=np.float32)])
        max_ref = int(self.MAX_REF_SEC * 16000)
        if len(self.ref_buf) > max_ref:
            self.ref_buf = self.ref_buf[-max_ref:]

    def process_mic(self, frame):
        self.mic_buf = np.concatenate([self.mic_buf, np.asarray(frame, dtype=np.float32)])
        if len(self.mic_buf) < self.BLOCK:
            return None
        mic_block = self.mic_buf[:self.BLOCK]
        self.mic_buf = self.mic_buf[self.BLOCK:]
        if len(self.ref_buf) >= self.BLOCK:
            ref_block = self.ref_buf[:self.BLOCK]
            self.ref_buf = self.ref_buf[self.BLOCK:]
        else:
            ref_block = np.zeros(self.BLOCK, dtype=np.float32)
        out = np.asarray(self._ap.process(mic_block, ref_block), dtype=np.float32)
        return out

    def reset(self) -> None:
        self._ap.reset()
        self.mic_buf = np.zeros(0, dtype=np.float32)
        self.ref_buf = np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Bot active-listening backchannels (VAP / Voice Activity Projection)
# ─────────────────────────────────────────────────────────────────────────────
class BotBackchannel:
    def __init__(self, config: BackchannelConfig) -> None:
        self.config = config
        self._vap = None
        self.frame_size = SAMPLE_RATE // config.frame_rate + 320
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_ack = -1e9

    def _load(self):
        if self._vap is None:
            from pathlib import Path

            for path in (self.config.vap_bc_model, self.config.cpc_model):
                if not Path(path).exists():
                    raise FileNotFoundError(
                        f"VAP model not found at {path}. Run "
                        "`python scripts/download_models.py`."
                    )
            import contextlib
            import io

            from rvap.vap_bc.vap_bc_main import VAPRealTime

            with contextlib.redirect_stdout(io.StringIO()):
                self._vap = VAPRealTime(
                    self.config.vap_bc_model,
                    self.config.cpc_model,
                    "cpu",
                    self.config.frame_rate,
                    self.config.context_len_sec,
                )
        return self._vap

    def load(self) -> None:
        self._load()

    def push(self, frame) -> float | None:
        if self._vap is None:
            self._load()
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32)])
        if len(self._buf) < self.frame_size:
            return None
        user = np.asarray(self._buf, dtype=np.float32)[-self.frame_size:]
        bot = np.zeros(self.frame_size, dtype=np.float32)
        import contextlib
        import io
        import warnings

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._vap.process_vap(bot, user)
        return float(self._vap.result_p_bc_react[0])

    def should_ack(self, prob, now: float) -> bool:
        return prob is not None and prob > self.config.threshold and (
            now - self._last_ack
        ) > self.config.cooldown_sec

    def mark_acked(self, now: float) -> None:
        self._last_ack = now

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic VAD
# ─────────────────────────────────────────────────────────────────────────────
class ExponentialMovingAverage:
    def __init__(self, attack_time, release_time, initial_value=0.0):
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = float(initial_value)

    def update(self, *, dt, new_value):
        assert dt > 0.0, f"dt must be positive, got {dt=}"
        assert new_value >= 0.0, f"new_value must be non-negative, got {new_value=}"
        if new_value > self.value:
            alpha = 1 - math.exp(-dt / self.attack_time * math.log(2))
        else:
            alpha = 1 - math.exp(-dt / self.release_time * math.log(2))
        self.value = float((1 - alpha) * self.value + alpha * new_value)
        return self.value


class SileroVAD:
    _MIN_SAMPLES = 512

    def __init__(self, threshold: float = 0.5, energy_threshold: float = 0.01) -> None:
        self.threshold = threshold
        # Frames quieter than this RMS are treated as silence, so quiet ambient
        # noise doesn't get misclassified as speech and trigger barge-in.
        self.energy_threshold = energy_threshold
        self._model = None
        self._buf = np.zeros(0, dtype=np.float32)

    def _load(self):
        if self._model is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as exc:
                raise RuntimeError(
                    "silero-vad is not installed. Run `pip install silero-vad`."
                ) from exc
            self._model = load_silero_vad()
        return self._model

    def load(self) -> None:
        self._load()

    def is_speech(self, frame: np.ndarray) -> bool:
        frame = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < self.energy_threshold:
            return False
        model = self._load()
        import torch

        self._buf = np.concatenate([self._buf, frame])
        if len(self._buf) < self._MIN_SAMPLES:
            return False
        window = self._buf[-self._MIN_SAMPLES:]
        tensor = torch.from_numpy(window).unsqueeze(0)
        with torch.no_grad():
            prob = model(tensor, SAMPLE_RATE).item()
        return prob >= self.threshold


def truncate_audio_to_last_n_seconds(audio, n_seconds, sample_rate=SAMPLE_RATE):
    n_samples = int(n_seconds * sample_rate)
    if len(audio) > n_samples:
        return audio[-n_samples:]
    if len(audio) < n_samples:
        return np.concatenate(
            [np.zeros(n_samples - len(audio), dtype=np.float32), audio]
        )
    return audio


class SmartTurn:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._session = None
        self._feature_extractor = None

    def _load(self):
        if self._session is not None:
            return self._session, self._feature_extractor
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Smart Turn model not found at {self.model_path}. "
                "Run `python unmute_tui.py --download-models` first."
            )
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self.model_path), sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        return self._session, self._feature_extractor

    def load(self) -> None:
        self._load()

    def predict_endpoint(self, audio: np.ndarray) -> float:
        session, fe = self._load()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = truncate_audio_to_last_n_seconds(audio, SMART_TURN_WINDOW_SEC)
        inputs = fe(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=int(SMART_TURN_WINDOW_SEC * SAMPLE_RATE),
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)
        outputs = session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())


MIN_SILENCE_SEC = 0.2
EMA_ATTACK_TIME = 0.01
EMA_RELEASE_TIME = 0.01


@dataclass
class TurnEndResult:
    turn_end: bool
    probability: float
    raw_probability: float | None
    audio: np.ndarray | None = None


class SemanticVAD:
    def __init__(
        self,
        smart_turn: SmartTurn | None = None,
        silero: SileroVAD | None = None,
        threshold: float = 0.6,
        min_silence_sec: float = MIN_SILENCE_SEC,
        turn_end_silence_sec: float = 0.7,
        semantic_min_silence_sec: float = 0.4,
    ) -> None:
        self.smart_turn = smart_turn
        self.silero = silero or SileroVAD()
        self.threshold = threshold
        self.min_silence_frames = max(1, int(min_silence_sec / FRAME_TIME_SEC))
        self.turn_end_silence_frames = max(
            self.min_silence_frames, int(turn_end_silence_sec / FRAME_TIME_SEC)
        )
        self.semantic_min_silence_frames = max(
            1, int(semantic_min_silence_sec / FRAME_TIME_SEC)
        )
        self.ema = ExponentialMovingAverage(
            attack_time=EMA_ATTACK_TIME,
            release_time=EMA_RELEASE_TIME,
            initial_value=0.0,
        )
        self._turn_buffer: list[np.ndarray] = []
        self._was_speaking = False
        self._had_speech = False
        self._silence_frames = 0
        self._need_semantic = False
        self._last_raw: float | None = None

    @property
    def probability(self) -> float:
        return self.ema.value

    @property
    def is_speaking(self) -> bool:
        return self._was_speaking

    def reset(self) -> None:
        self._turn_buffer = []
        self._was_speaking = False
        self._had_speech = False
        self._silence_frames = 0
        self._need_semantic = False
        self._last_raw = None
        self.ema.value = 0.0

    def _turn_audio(self) -> np.ndarray:
        if not self._turn_buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._turn_buffer)

    def process_frame(self, frame: np.ndarray) -> TurnEndResult:
        frame = np.asarray(frame, dtype=np.float32)
        self._turn_buffer.append(frame)
        is_speech = self.silero.is_speech(frame)
        if is_speech:
            self._silence_frames = 0
            self._had_speech = True
            if not self._was_speaking:
                self._need_semantic = True
            self._was_speaking = True
            self.ema.update(dt=FRAME_TIME_SEC, new_value=0.0)
            return TurnEndResult(False, self.ema.value, self._last_raw)
        self._silence_frames += 1
        was_speaking = self._was_speaking
        self._was_speaking = False
        if self._had_speech and (was_speaking or self._need_semantic) and (
            self._silence_frames >= self.min_silence_frames
        ):
            if self.smart_turn is not None:
                audio = self._turn_audio()
                if len(audio) > 0:
                    self._last_raw = self.smart_turn.predict_endpoint(audio)
                    self.ema.update(dt=FRAME_TIME_SEC, new_value=self._last_raw)
            self._need_semantic = False
        turn_end = False
        if self._had_speech:
            if (
                self.smart_turn is not None
                and self.ema.value > self.threshold
                and self._silence_frames >= self.semantic_min_silence_frames
            ):
                turn_end = True
            elif self._silence_frames >= self.turn_end_silence_frames:
                turn_end = True
        if turn_end:
            audio = self._turn_audio()
            self.reset()
            return TurnEndResult(True, self.ema.value, self._last_raw, audio)
        return TurnEndResult(False, self.ema.value, self._last_raw)


# ─────────────────────────────────────────────────────────────────────────────
# STT backends (pluggable local realtime speech-to-text)
# parakeet (default) | sherpa (streaming zipformer) | moonshine | faster_whisper
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Transcription:
    text: str
    segments: list[Segment]
    language: str | None = None


_TAIL_PAD_SEC = 0.5
_FEATURE_DIM = 80


def _as_1d_float32(audio) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    return audio


def _trailing_window(audio: np.ndarray, window_sec: float) -> np.ndarray:
    n = int(window_sec * SAMPLE_RATE)
    if len(audio) > n:
        return audio[-n:]
    return audio


class STTBackend(ABC):
    """Engine-facing interface implemented by every STT backend."""

    def load(self) -> None:
        """Pre-load all models (blocking); called before the event loop."""

    def push(self, frame: np.ndarray) -> None:
        """Feed one 16 kHz mono mic frame of the in-progress user turn.

        Default: no-op (batch backends re-transcribe windows on demand).
        """

    @abstractmethod
    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        """The current partial transcript of the in-progress turn."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> Transcription:
        """Final transcription of a completed turn (16 kHz mono float32)."""

    def reset(self) -> None:
        """Start a fresh turn (default: no-op)."""


class FasterWhisperBackend(STTBackend):
    """Batch transcription via faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Run "
                    "`pip install faster-whisper`."
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def load(self) -> None:
        self._load()

    def _transcribe(self, audio: np.ndarray) -> Transcription:
        model = self._load()
        audio = _as_1d_float32(audio)
        if len(audio) == 0:
            return Transcription("", [])
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            language=self.language,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segs = [
            Segment(text=s.text.strip(), start=float(s.start), end=float(s.end))
            for s in segments
        ]
        text = " ".join(s.text for s in segs).strip()
        return Transcription(text=text, segments=segs, language=info.language)

    def transcribe(self, audio: np.ndarray) -> Transcription:
        return self._transcribe(audio)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        if audio is None or len(audio) == 0:
            return Transcription("", [])
        return self._transcribe(_trailing_window(_as_1d_float32(audio), window_sec))


class _SherpaModelDir:
    """Resolves the encoder/decoder/joiner/tokens files in a model folder."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)

    def _find(self, patterns: list[str]) -> Path | None:
        for pattern in patterns:
            hits = sorted(self.model_dir.glob(pattern))
            if hits:
                return hits[0]
        return None

    @property
    def encoder(self) -> Path:
        return self._require(["encoder*.int8.onnx", "encoder*.onnx"], "encoder")

    @property
    def decoder(self) -> Path:
        return self._require(["decoder*.onnx", "decoder*.int8.onnx"], "decoder")

    @property
    def joiner(self) -> Path:
        return self._require(["joiner*.int8.onnx", "joiner*.onnx"], "joiner")

    @property
    def tokens(self) -> Path:
        return self._require(["tokens.txt", "bpe.model"], "tokens")

    def _require(self, patterns: list[str], name: str) -> Path:
        hit = self._find(patterns)
        if hit is None:
            raise FileNotFoundError(
                f"Sherpa-onnx {name} file not found in {self.model_dir}. "
                "Run `uv run python scripts/download_models.py` first."
            )
        return hit


class SherpaZipformerBackend(STTBackend):
    """True streaming ASR via sherpa-onnx OnlineRecognizer."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 2,
        language: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.language = language
        self._recognizer = None
        self._stream = None
        self._lock = threading.Lock()

    def _load(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed. Run "
                "`pip install sherpa-onnx` (or `uv sync --extra stt`)."
            ) from exc
        files = _SherpaModelDir(self.model_dir)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(files.tokens),
            encoder=str(files.encoder),
            decoder=str(files.decoder),
            joiner=str(files.joiner),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=_FEATURE_DIM,
            decoding_method="greedy_search",
            provider="cpu",
            model_type="",
            enable_endpoint_detection=False,
        )
        return self._recognizer

    def load(self) -> None:
        self._load()

    def push(self, frame: np.ndarray) -> None:
        recognizer = self._load()
        frame = _as_1d_float32(frame)
        if len(frame) == 0:
            return
        with self._lock:
            if self._stream is None:
                self._stream = recognizer.create_stream()
            self._stream.accept_waveform(SAMPLE_RATE, frame)
            while recognizer.is_ready(self._stream):
                recognizer.decode_stream(self._stream)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        recognizer = self._load()
        with self._lock:
            if self._stream is None:
                return Transcription("", [])
            text = recognizer.get_result(self._stream)
        return Transcription(
            text=text.strip().lower(), segments=[], language=self.language
        )

    def transcribe(self, audio: np.ndarray) -> Transcription:
        recognizer = self._load()
        audio = _as_1d_float32(audio)
        with self._lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            pad = np.zeros(int(_TAIL_PAD_SEC * SAMPLE_RATE), dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, pad)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            text = recognizer.get_result(stream).strip().lower()
        self.reset()
        return Transcription(text=text, segments=[], language=self.language)

    def reset(self) -> None:
        with self._lock:
            self._stream = None


class ParakeetBackend(STTBackend):
    """NVIDIA Parakeet TDT-0.6B via sherpa-onnx OfflineRecognizer."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 2,
        language: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.language = language
        self._recognizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed. Run "
                "`pip install sherpa-onnx` (or `uv sync --extra stt`)."
            ) from exc
        files = _SherpaModelDir(self.model_dir)
        self._check_decoder_metadata(files.decoder)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files.encoder),
            decoder=str(files.decoder),
            joiner=str(files.joiner),
            tokens=str(files.tokens),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=_FEATURE_DIM,
            decoding_method="greedy_search",
            provider="cpu",
            model_type="nemo_transducer",
        )
        return self._recognizer

    @staticmethod
    def _check_decoder_metadata(decoder: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            return
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            meta = ort.InferenceSession(
                str(decoder), sess_options=so, providers=["CPUExecutionProvider"]
            ).get_modelmeta().custom_metadata_map
        except Exception:
            return
        if "vocab_size" not in meta:
            raise RuntimeError(
                f"Parakeet decoder is missing RNNT metadata ({decoder}). "
                "Rerun `uv run python scripts/download_models.py` so the "
                "metadata is patched, or use STT_BACKEND=sherpa/moonshine."
            )

    def load(self) -> None:
        self._load()

    def transcribe(self, audio: np.ndarray) -> Transcription:
        recognizer = self._load()
        audio = _as_1d_float32(audio)
        with self._lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_streams([stream])
            text = stream.result.text.strip()
        return Transcription(text=text, segments=[], language=self.language)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        if audio is None or len(audio) == 0:
            return Transcription("", [])
        return self.transcribe(_trailing_window(_as_1d_float32(audio), window_sec))


class MoonshineBackend(STTBackend):
    """Moonshine v2 streaming STT via moonshine-voice's Transcriber API."""

    def __init__(self, language: str = "en", update_interval: float = 0.3) -> None:
        self.language = language
        self.update_interval = update_interval
        self._transcriber = None
        self._stream = None
        self._listener = None
        self._text = ""
        self._final_lines: list[str] = []
        self._lock = threading.Lock()

    def _load(self):
        if self._transcriber is not None:
            return self._transcriber
        try:
            from moonshine_voice import (
                Transcriber,
                TranscriptEventListener,
                get_model_for_language,
            )
        except ImportError as exc:
            raise RuntimeError(
                "moonshine-voice is not installed. Run "
                "`pip install moonshine-voice` (or `uv sync --extra stt`)."
            ) from exc

        class _Collector(TranscriptEventListener):
            def __init__(self, owner) -> None:
                super().__init__()
                self._owner = owner

            def on_line_text_changed(self, event) -> None:
                self._owner._on_text(event.line.text)

            def on_line_completed(self, event) -> None:
                self._owner._on_line(event.line.text)

        model_path, model_arch = get_model_for_language(self.language)
        self._transcriber = Transcriber(
            model_path=model_path, model_arch=model_arch
        )
        self._listener = _Collector(self)
        return self._transcriber

    def _ensure_stream(self):
        self._load()
        if self._stream is not None:
            return self._stream
        self._stream = self._transcriber.create_stream(
            update_interval=self.update_interval
        )
        self._stream.add_listener(self._listener)
        self._stream.start()
        return self._stream

    def _on_text(self, text: str) -> None:
        with self._lock:
            self._text = text

    def _on_line(self, text: str) -> None:
        with self._lock:
            if text.strip():
                self._final_lines.append(text.strip())
            self._text = ""

    def load(self) -> None:
        self._load()

    def push(self, frame: np.ndarray) -> None:
        stream = self._ensure_stream()
        frame = _as_1d_float32(frame)
        if len(frame) == 0:
            return
        stream.add_audio(frame.tolist(), SAMPLE_RATE)

    def partial(
        self, audio: np.ndarray | None = None, window_sec: float = 8.0
    ) -> Transcription:
        self._load()
        with self._lock:
            text = self._text.strip()
        return Transcription(text=text, segments=[])

    def transcribe(self, audio: np.ndarray) -> Transcription:
        self._load()
        with self._lock:
            text = (" ".join(self._final_lines) or self._text).strip()
        return Transcription(text=text, segments=[])

    def reset(self) -> None:
        with self._lock:
            self._text = ""
            self._final_lines = []
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None


def create_transcriber(config) -> STTBackend:
    """Build the STT backend selected by ``config.backend``."""
    backend = (config.backend or "parakeet").strip().lower()
    if backend == "sherpa":
        return SherpaZipformerBackend(
            model_dir=config.models_dir / config.model,
            num_threads=config.threads,
            language=config.language,
        )
    if backend == "parakeet":
        return ParakeetBackend(
            model_dir=config.models_dir / config.model,
            num_threads=config.threads,
            language=config.language,
        )
    if backend == "moonshine":
        return MoonshineBackend(language=config.language or "en")
    if backend == "faster_whisper":
        return FasterWhisperBackend(
            model_size=config.model, language=config.language
        )
    raise ValueError(
        f"Unknown STT_BACKEND {backend!r} "
        "(expected parakeet | sherpa | moonshine | faster_whisper)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end():]
            if chunk != "":
                yield prefix + chunk
            prefix = " "
    if buffer != "":
        yield prefix + buffer


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=config.base_url,
        )
        self._model: str | None = config.model

    async def _resolve_model(self) -> str:
        if self._model:
            return self._model
        models = await self._client.models.list()
        ids = [m.id for m in models.data]
        if len(ids) != 1:
            raise RuntimeError(
                f"Endpoint exposes {len(ids)} models; set LLM_MODEL explicitly. "
                f"Available: {ids}"
            )
        self._model = ids[0]
        return self._model

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        model = await self._resolve_model()
        extra_body: dict = {"reasoning_split": True}
        if self.config.thinking:
            extra_body["thinking"] = {"type": self.config.thinking}
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            temperature=0.7,
            extra_body=extra_body,
        )
        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                yield content


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# TTS (Kokoro primary; edge/piper fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    n_out = int(round(len(audio) * SAMPLE_RATE / src_rate))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class TTSBackend(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        ...

    def load(self) -> None:
        """Pre-load the model. No-op for backends with no persistent model."""

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        audio = await self.synthesize(text)
        if len(audio):
            yield audio


class ChatterboxTTSBackend(TTSBackend):
    """Chatterbox-Turbo via `mlx-audio` (natural, expressive, near real-time).

    Streams audio chunks as text is processed (MLX on Apple Silicon). Supports
    inline emotion tags like [sigh] and [laugh].
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from mlx_audio.tts import generate
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-audio is not installed. Run `uv sync` (or "
                    "`pip install mlx-audio`)."
                ) from exc
            self._model = generate.load_model(self.config.chatterbox_model)
        return self._model

    def load(self) -> None:
        self._load()

    async def synthesize(self, text: str) -> np.ndarray:
        chunks = [c async for c in self.stream(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        model = self._load()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run() -> None:
            import contextlib
            import io

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    for r in model.generate(text=text, stream=True, verbose=False):
                        audio = np.asarray(r.audio, dtype=np.float32)
                        q.put_nowait(resample_to_16k(audio, r.sample_rate))
            except Exception as exc:  # noqa: BLE001
                q.put_nowait(exc)
            finally:
                q.put_nowait(None)

        task = loop.run_in_executor(None, _run)
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await task


class KokoroTTSBackend(TTSBackend):
    """Kokoro-82M TTS via `kokoro-onnx` (fast, natural, lightweight, offline)."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._kokoro = None

    def _load(self):
        if self._kokoro is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro-onnx is not installed. Run `uv sync` (or "
                    "`pip install kokoro-onnx`)."
                ) from exc
            if not self.config.kokoro_model or not self.config.kokoro_voices:
                raise FileNotFoundError(
                    "Kokoro model files not configured. Set KOKORO_MODEL and "
                    "KOKORO_VOICES, or run `python scripts/download_models.py`."
                )
            if not Path(self.config.kokoro_model).exists() or not Path(
                self.config.kokoro_voices
            ).exists():
                raise FileNotFoundError(
                    "Kokoro model files not found. Run "
                    "`python scripts/download_models.py`."
                )
            self._kokoro = Kokoro(self.config.kokoro_model, self.config.kokoro_voices)
        return self._kokoro

    def load(self) -> None:
        self._load()

    async def synthesize(self, text: str) -> np.ndarray:
        kokoro = self._load()
        samples, sr = await asyncio.to_thread(
            kokoro.create,
            text,
            self.config.kokoro_voice,
            self.config.kokoro_speed,
            self.config.kokoro_lang,
        )
        if samples is None or len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        return resample_to_16k(np.asarray(samples, dtype=np.float32), int(sr))


class EdgeTTSBackend(TTSBackend):
    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> np.ndarray:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts is not installed.") from exc
        communicate = edge_tts.Communicate(text, self.config.voice)
        mp3 = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        if not mp3:
            return np.zeros(0, dtype=np.float32)
        return await asyncio.to_thread(_decode_mp3_to_pcm, mp3)


class PiperTTSBackend(TTSBackend):
    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> np.ndarray:
        try:
            import piper
        except ImportError as exc:
            raise RuntimeError("piper-tts is not installed.") from exc
        voice = piper.PiperVoice.load(self.config.voice)
        audio = await asyncio.to_thread(_piper_synthesize, voice, text)
        return resample_to_16k(audio, 22050)


def _piper_synthesize(voice, text: str) -> np.ndarray:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=voice.config.sample_rate, channels=1, format="WAV"
    ) as f:
        for audio in voice.synthesize(text):
            f.write(audio)
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32")
    return np.asarray(data, dtype=np.float32)


def _decode_mp3_to_pcm(mp3_bytes: bytes) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def create_tts(config: TTSConfig) -> TTSBackend:
    backend = config.backend.strip().lower()
    if backend == "chatterbox":
        return ChatterboxTTSBackend(config)
    if backend == "kokoro":
        return KokoroTTSBackend(config)
    if backend == "edge":
        return EdgeTTSBackend(config)
    if backend == "piper":
        return PiperTTSBackend(config)
    raise ValueError(
        f"Unknown TTS backend '{backend}'. "
        f"Choose from: chatterbox, kokoro, edge, piper."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation engine
# ─────────────────────────────────────────────────────────────────────────────
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
        aec=None,
        backchannel=None,
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
        self._barge_in_frames = 0
        self._barge_in_required = config.vad.barge_in_required_frames
        self._silence_count = 0
        self._mic_muted = False

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
        self._add_message("user", "Hello!")
        await self._generate_response()
        async for frame in self.mic.frames():
            self.n_samples_received += len(frame)
            self.audio_time = self.n_samples_received / SAMPLE_RATE
            if self._mic_muted:
                continue
            if self.aec is not None:
                clean = self.aec.process_mic(frame)
                if clean is None:
                    continue
                frame = clean
            result = self.vad.process_frame(frame)
            self._emit(VADUpdate(result.probability, result.raw_probability))
            frame_rms = float(np.sqrt(np.mean(frame**2)))
            await self._tick(result, frame_rms)

            if (
                self.backchannel is not None
                and self._state == "user_speaking"
                and not self._mic_muted
            ):
                prob = self.backchannel.push(frame)
                if self.backchannel.should_ack(prob, self.audio_time):
                    self.backchannel.mark_acked(self.audio_time)
                    await self._speak_backchannel()

    async def _tick(self, result, frame_rms: float = 0.0) -> None:
        if self._state == "bot_speaking":
            # Only count a frame as the user barge-in when it is above the
            # barge-in energy floor; the AEC leaves a low-energy residual of the
            # bot's own voice, which we must not treat as a barge-in.
            if self.vad.is_speaking and frame_rms >= self.config.vad.barge_in_min_rms:
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
                    self.aec.add_reference(chunk)
                await asyncio.to_thread(self.player.play, chunk)

    async def _speak_backchannel(self) -> None:
        self._emit(Log("backchannel"))
        text = self.config.backchannel.ack_text
        async for chunk in self.tts.stream(text):
            if len(chunk):
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


# ─────────────────────────────────────────────────────────────────────────────
# TUI (textual)
# ─────────────────────────────────────────────────────────────────────────────
def _vad_bar(probability: float) -> str:
    width = 20
    filled = int(round(probability * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {probability:.2f}"


class UnmuteApp(App):
    TITLE = "unmute-tui"
    CSS = """
    #status-row { height: 3; padding: 0 1; }
    #status-row Static { width: 1fr; }
    #transcript { height: 1fr; border: round $primary; }
    #current { height: 3; border: round $accent; padding: 0 1; }
    """

    def __init__(self, engine, events: asyncio.Queue[EngineEvent]) -> None:
        super().__init__()
        self.engine = engine
        self.events = events
        self._engine_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-row"):
            yield Static("state: waiting_for_user", id="state")
            yield Static("vad: " + _vad_bar(0.0), id="vad")
            yield Static("latency: -", id="latency")
        yield RichLog(id="transcript", highlight=True, markup=True, wrap=True)
        yield Static("", id="current")
        yield Footer()

    async def on_mount(self) -> None:
        self._engine_task = asyncio.create_task(self.engine.run())
        self._consumer_task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        while True:
            event = await self.events.get()
            self._handle(event)

    def _handle(self, event: EngineEvent) -> None:
        if isinstance(event, StateChanged):
            self.query_one("#state", Static).update(f"state: {event.state}")
        elif isinstance(event, VADUpdate):
            self.query_one("#vad", Static).update("vad: " + _vad_bar(event.probability))
        elif isinstance(event, UserTranscript):
            self.query_one("#transcript", RichLog).write(f"[bold cyan]You:[/] {event.text}")
        elif isinstance(event, AssistantDelta):
            self.query_one("#current", Static).update(f"[bold magenta]Bot:[/] {event.text}")
        elif isinstance(event, AssistantDone):
            self.query_one("#current", Static).update("")
            self.query_one("#transcript", RichLog).write(
                f"[bold magenta]Bot:[/] {event.text}"
            )
        elif isinstance(event, Latency):
            self.query_one("#latency", Static).update(
                f"latency: {event.metric}={event.value:.2f}"
            )
        elif isinstance(event, Log):
            self.query_one("#transcript", RichLog).write(f"[dim]{event.message}[/]")
        elif isinstance(event, SessionEnd):
            self.query_one("#transcript", RichLog).write(
                f"[bold yellow]Session ended: {event.reason}[/]"
            )
            self.call_after(self.exit, event.reason)

    async def on_unmount(self) -> None:
        if self._engine_task is not None:
            self._engine_task.cancel()
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        await self.engine.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Model download + entry point
# ─────────────────────────────────────────────────────────────────────────────
def download_models(models_dir: str | Path = "models") -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dest = models_dir / SMART_TURN_FILE
    if dest.exists():
        print(f"Smart Turn model already present: {dest}")
    else:
        print(f"Downloading {SMART_TURN_REPO}/{SMART_TURN_FILE} ...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("huggingface_hub is not installed. Run `pip install huggingface_hub`.")
            raise SystemExit(1)
        path = hf_hub_download(
            repo_id=SMART_TURN_REPO,
            filename=SMART_TURN_FILE,
            local_dir=models_dir,
        )
        print(f"Downloaded to {path}")

    # Kokoro-82M TTS ONNX model + voice embeddings.
    onnx_dest = models_dir / KOKORO_ONNX_FILE
    voices_dest = models_dir / KOKORO_VOICES_FILE
    if onnx_dest.exists():
        print(f"Kokoro ONNX model already present: {onnx_dest}")
    else:
        import urllib.request

        print(f"Downloading Kokoro ONNX model -> {onnx_dest} ...")
        urllib.request.urlretrieve(KOKORO_ONNX_URL, onnx_dest)
        print(f"Downloaded to {onnx_dest}")
    if voices_dest.exists():
        print(f"Kokoro voices already present: {voices_dest}")
    else:
        import urllib.request

        print(f"Downloading Kokoro voices -> {voices_dest} ...")
        urllib.request.urlretrieve(KOKORO_VOICES_URL, voices_dest)
        print(f"Downloaded to {voices_dest}")


def _smart_turn_path(models_dir: Path) -> Path:
    return models_dir / SMART_TURN_FILENAME


def build_engine(config: Config):
    smart_turn = SmartTurn(_smart_turn_path(config.models_dir))
    silero = SileroVAD(energy_threshold=config.vad.energy_threshold)
    vad = SemanticVAD(
        smart_turn=smart_turn,
        silero=silero,
        threshold=config.vad.threshold,
        turn_end_silence_sec=config.vad.turn_end_silence_sec,
        semantic_min_silence_sec=config.vad.semantic_min_silence_sec,
    )
    transcriber = create_transcriber(config.stt)
    llm = LLM(config.llm)
    tts = create_tts(config.tts)
    mic = Microphone(device=config.audio.input_device)
    player = AudioPlayer(device=config.audio.output_device)
    return vad, transcriber, llm, tts, mic, player


async def _headless(config: Config, engine) -> None:
    events = engine.events

    async def consume() -> None:
        while True:
            ev = await events.get()
            if isinstance(ev, VADUpdate):
                continue
            if isinstance(ev, AssistantDelta):
                print(ev.text, end="", flush=True)
            elif isinstance(ev, AssistantDone):
                print()
            elif isinstance(ev, UserTranscript):
                print(f"\n[You] {ev.text}")
            elif isinstance(ev, StateChanged):
                print(f"\n[state] {ev.state}")
            elif isinstance(ev, Log):
                print(f"\n[log] {ev.message}")
            elif isinstance(ev, SessionEnd):
                print(f"\n[session end] {ev.reason}")
                return

    try:
        await asyncio.gather(engine.run(), consume())
    except KeyboardInterrupt:
        pass
    finally:
        await engine.shutdown()


def _make_engine(config: Config):
    events: asyncio.Queue = asyncio.Queue()
    vad, transcriber, llm, tts, mic, player = build_engine(config)
    aec = None
    if config.aec.enabled:
        aec = WebRTCAEC(
            delay_ms=config.aec.delay_ms,
            noise_suppression=config.aec.noise_suppression,
            ns_level=config.aec.ns_level,
        )
    backchannel = None
    if config.backchannel.enabled:
        backchannel = BotBackchannel(config.backchannel)
    return ConversationEngine(
        config, vad, transcriber, llm, tts, mic, player, events,
        aec=aec, backchannel=backchannel,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unmute-tui")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices")
    parser.add_argument("--no-tui", action="store_true", help="headless mode")
    parser.add_argument("--config", default=".env", help="path to .env file")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="download the semantic VAD model + default voice, then exit",
    )
    args = parser.parse_args(argv)

    if args.download_models:
        download_models()
        return 0
    if args.list_devices:
        print(list_devices())
        return 0

    config = Config.from_env(args.config)
    engine = _make_engine(config)

    # Loading gate: pre-load all models before the event loop starts. This
    # avoids subprocess/thread conflicts with the asyncio loop and ensures the
    # bot never interrupts itself while a model is still loading.
    print("Loading models...", flush=True)
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1
    print("Models ready.", flush=True)

    if args.no_tui:
        try:
            asyncio.run(_headless(config, engine))
        except KeyboardInterrupt:
            pass
        return 0

    app = UnmuteApp(engine, engine.events)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
