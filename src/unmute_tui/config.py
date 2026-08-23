"""Configuration for unmute-tui.

All settings come from environment variables (optionally loaded from a `.env`
file) with sensible defaults. Thresholds mirror unmute.sh's constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Audio constants (mirror unmute's kyutai_constants).
SAMPLE_RATE = 16000
FRAME_TIME_SEC = 0.02  # 20 ms frames
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_TIME_SEC)  # 320

# Defaults mirroring unmute.sh.
DEFAULT_VAD_THRESHOLD = 0.6
DEFAULT_USER_SILENCE_TIMEOUT = 7.0
DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3.0

# Semantic VAD: how long a turn recording is kept for Smart Turn (seconds).
SMART_TURN_WINDOW_SEC = 8.0


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
    model: str | None = None  # None => auto-select if exactly one model exposed
    # MiniMax-M3 thinking control: "disabled" | "adaptive" | None (unset).
    thinking: str | None = None


@dataclass
class VADConfig:
    threshold: float = DEFAULT_VAD_THRESHOLD
    user_silence_timeout: float = DEFAULT_USER_SILENCE_TIMEOUT
    uninterruptible_by_vad_time_sec: float = DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC
    smart_turn_window_sec: float = SMART_TURN_WINDOW_SEC
    # Frames quieter than this RMS are treated as silence (filters ambient noise).
    energy_threshold: float = 0.01
    # Hard, reliable turn-end: after the user stops speaking, this much sustained
    # silence ends their turn (works even if the semantic model errs).
    turn_end_silence_sec: float = 0.7
    # Semantic accelerator: if Smart Turn is confident (prob > threshold) AND this
    # much silence has elapsed, end the turn sooner.
    semantic_min_silence_sec: float = 0.4
    # Barge-in robustness: while the bot is speaking, only count a frame as the
    # user barge-in if the AEC-cleaned mic RMS is at least this value. The AEC
    # leaves a low-energy residual of the bot's own voice; this gate ignores it
    # so the bot doesn't interrupt itself, while louder real user speech still
    # triggers barge-in.
    barge_in_min_rms: float = 0.03
    # Consecutive (energy-gated) speech frames required to trigger barge-in in
    # the bot's turn. The echo residual is intermittent; real speech is sustained.
    barge_in_required_frames: int = 15
    # Mute the mic while the bot is speaking. Without echo cancellation (e.g.
    # when not on headphones) the bot would hear its own TTS and interrupt
    # itself. Disable to allow barge-in (requires headphones/echo cancellation).
    mute_mic_while_bot_speaking: bool = True


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
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    aec: AECConfig = field(default_factory=AECConfig)
    models_dir: Path = Path("models")
    stt_model: str = "base"
    stt_language: str | None = None
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
                barge_in_required_frames=_env_int(
                    "BARGE_IN_REQUIRED_FRAMES", 15
                ),
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
            models_dir=models_dir,
            stt_model=os.getenv("STT_MODEL", "base"),
            stt_language=os.getenv("STT_LANGUAGE") or None,
            debug=_env_bool("DEBUG", False),
        )
