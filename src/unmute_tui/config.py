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
DEFAULT_UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 0.3

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
    # No-AEC barge-in margin (dB): without echo cancellation, a mic frame only
    # counts as the user when it is at least this much louder than the bot's
    # recent playback. The bot's own TTS echo sits at ~the playback level, so
    # this lets a user speaking over the bot interrupt it while preventing the
    # bot from interrupting itself.
    barge_in_over_playback_db: float = 6.0
    # Mute the mic while the bot is speaking (half-duplex). Disabled by default
    # so barge-in is live in all configs; the playback-aware echo gate keeps the
    # bot from interrupting itself when AEC is unavailable. Enable only if you
    # want a guaranteed no-self-reply half-duplex mode.
    mute_mic_while_bot_speaking: bool = False


@dataclass
class TurnConfig:
    """Turn detector selection and the LLM turn-orchestrator tuning.

    `detector`:
      - "semantic" (default): audio-only turn-end (Silero + Smart Turn v3),
        byte-for-byte the original behavior.
      - "llm": an LLM orchestrator continuously polls streaming partial
        transcripts AND VAD audio cues and can commit the bot's reply early
        (near-zero turn-end latency). The reliable audio silence timeout still
        backstops so a stalled/empty LLM never hangs the conversation.
    """

    detector: str = "semantic"  # "semantic" | "llm"
    # How often to re-transcribe the rolling turn buffer for partials (s).
    partial_poll_interval_sec: float = 0.5
    # How often to poll the LLM orchestrator while the user is speaking (s).
    poll_interval_sec: float = 0.6
    # Don't poll the LLM until the partial transcript is at least this long
    # (skips short/garbage partials).
    min_partial_chars: int = 12
    # Trailing audio window (s) re-transcribed for each partial update.
    stt_poll_window_sec: float = 8.0


@dataclass
class BackchannelConfig:
    # Bot active-listening backchannels via VAP (Voice Activity Projection).
    # When the user is speaking and VAP predicts a backchannel, the bot emits a
    # short ack ("Mm-hmm") without taking the turn.
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
    """Speech-to-text backend selection.

    `backend`:
      - "sherpa" (default): sherpa-onnx streaming Zipformer — true incremental
        streaming, RTF ~0.03-0.05 int8 on CPU, far more accurate than whisper
        base.
      - "moonshine": Moonshine v2 via moonshine-voice (very low latency).
      - "parakeet": NVIDIA Parakeet TDT-0.6B via sherpa-onnx offline recognizer
        (best raw WER; non-streaming).
      - "faster_whisper": the original faster-whisper backend (kept as
        fallback); `model` selects the size (e.g. "base", "large-v3-turbo").
    """

    backend: str = "sherpa"
    # Backend-specific model: sherpa/parakeet = model folder name under
    # `models_dir`; faster_whisper = model size.
    model: str = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
    language: str | None = None
    # Directory holding the sherpa-onnx / parakeet ONNX assets.
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
    turn: TurnConfig = field(default_factory=TurnConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    models_dir: Path = Path("models")
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
                barge_in_over_playback_db=_env_float(
                    "BARGE_IN_OVER_PLAYBACK_DB", 6.0
                ),
                mute_mic_while_bot_speaking=_env_bool(
                    "MUTE_MIC_WHILE_BOT_SPEAKING", False
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
                cpc_model=os.getenv(
                    "CPC_MODEL", "models/60k_epoch4-d0f474de.pt"
                ),
                frame_rate=_env_int("BACKCHANNEL_FRAME_RATE", 10),
                context_len_sec=_env_float("BACKCHANNEL_CONTEXT_SEC", 3.0),
                threshold=_env_float("BACKCHANNEL_THRESHOLD", 0.5),
                cooldown_sec=_env_float("BACKCHANNEL_COOLDOWN_SEC", 3.0),
                ack_text=os.getenv("BACKCHANNEL_ACK_TEXT", "Mm-hmm."),
            ),
            turn=TurnConfig(
                detector=os.getenv("TURN_DETECTOR", "semantic").strip().lower(),
                partial_poll_interval_sec=_env_float(
                    "TURN_PARTIAL_POLL_INTERVAL_SEC", 0.5
                ),
                poll_interval_sec=_env_float(
                    "TURN_ORCHESTRATOR_POLL_INTERVAL_SEC", 0.6
                ),
                min_partial_chars=_env_int("TURN_MIN_PARTIAL_CHARS", 12),
                stt_poll_window_sec=_env_float(
                    "TURN_STT_POLL_WINDOW_SEC", 8.0
                ),
            ),
            models_dir=models_dir,
            stt=STTConfig(
                backend=os.getenv("STT_BACKEND", "sherpa").strip().lower(),
                model=os.getenv(
                    "STT_MODEL", "sherpa-onnx-streaming-zipformer-en-2023-06-26"
                ),
                language=os.getenv("STT_LANGUAGE") or None,
                models_dir=Path(os.getenv("STT_MODELS_DIR", "models/stt"))
                .expanduser()
                .resolve(),
                threads=_env_int("STT_THREADS", 2),
            ),
            debug=_env_bool("DEBUG", False),
        )
