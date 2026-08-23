import os

from unmute_tui.config import Config


def test_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("LLM_", "VAD_", "TTS_", "MODELS_DIR", "STT_", "INPUT_", "OUTPUT_")):
            monkeypatch.delenv(k, raising=False)
    cfg = Config.from_env(env_file=None)
    assert cfg.llm.base_url == "http://localhost:11434/v1"
    assert cfg.vad.threshold == 0.6
    assert cfg.tts.backend == "marvis"
    assert cfg.stt_model == "base"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("VAD_THRESHOLD", "0.7")
    monkeypatch.setenv("TTS_BACKEND", "edge")
    monkeypatch.setenv("STT_MODEL", "small")
    cfg = Config.from_env(env_file=None)
    assert cfg.llm.base_url == "https://api.openai.com/v1"
    assert cfg.llm.api_key == "sk-test"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.vad.threshold == 0.7
    assert cfg.tts.backend == "edge"
    assert cfg.stt_model == "small"
