import os

from unmute_tui.config import Config


def test_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("LLM_", "VAD_", "TTS_", "MODELS_DIR", "STT_", "INPUT_", "OUTPUT_", "TURN_")):
            monkeypatch.delenv(k, raising=False)
    cfg = Config.from_env(env_file=None)
    assert cfg.llm.base_url == "http://localhost:11434/v1"
    assert cfg.vad.threshold == 0.6
    assert cfg.tts.backend == "chatterbox"
    assert cfg.stt_model == "base"
    assert cfg.turn.detector == "semantic"
    assert cfg.turn.min_partial_chars == 12
    # Barge-in is live by default in all configs (no mic muting).
    assert cfg.vad.mute_mic_while_bot_speaking is False
    assert cfg.vad.uninterruptible_by_vad_time_sec == 0.3
    assert cfg.vad.barge_in_over_playback_db == 6.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("VAD_THRESHOLD", "0.7")
    monkeypatch.setenv("TTS_BACKEND", "edge")
    monkeypatch.setenv("STT_MODEL", "small")
    monkeypatch.setenv("TURN_DETECTOR", "llm")
    monkeypatch.setenv("TURN_MIN_PARTIAL_CHARS", "5")
    cfg = Config.from_env(env_file=None)
    assert cfg.llm.base_url == "https://api.openai.com/v1"
    assert cfg.llm.api_key == "sk-test"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.vad.threshold == 0.7
    assert cfg.tts.backend == "edge"
    assert cfg.stt_model == "small"
    assert cfg.turn.detector == "llm"
    assert cfg.turn.min_partial_chars == 5
