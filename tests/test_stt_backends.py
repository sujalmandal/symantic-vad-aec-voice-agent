"""Tests for the pluggable STT backends (offline, mocked model libraries)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unmute_tui.stt import (
    FasterWhisperBackend,
    MoonshineBackend,
    ParakeetBackend,
    SherpaZipformerBackend,
    Transcriber,
    _SherpaModelDir,
    _trailing_window,
    create_transcriber,
)


# ── shared helpers ──────────────────────────────────────────────────────────
def _make_model_dir(tmp_path):
    """Create a model folder with all expected sherpa-onnx files."""
    (tmp_path / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx").write_bytes(b"")
    (tmp_path / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx").write_bytes(b"")
    (tmp_path / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx").write_bytes(b"")
    (tmp_path / "tokens.txt").write_text("")
    return tmp_path


def _cfg(backend, model="m", models_dir=None):
    return SimpleNamespace(
        backend=backend,
        model=model,
        models_dir=models_dir,
        language=None,
        threads=2,
    )


# ── fake sherpa_onnx ────────────────────────────────────────────────────────
class _FakeStream:
    def __init__(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self._decoded = False
        self.result_text = ""

    def accept_waveform(self, sample_rate, samples):
        self.audio = np.concatenate([self.audio, np.asarray(samples, np.float32)])

    def input_finished(self):
        pass


class _FakeOfflineStream(_FakeStream):
    class result:
        text = "offline final"


class _FakeRecognizer:
    last_kwargs = {}

    @classmethod
    def from_transducer_with_zipformer(cls, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()

    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        return not stream._decoded and len(stream.audio) > 0

    def decode_stream(self, stream):
        stream._decoded = True
        stream.result_text = "help me find my keys"

    def get_result(self, stream):
        return stream.result_text


class _FakeOfflineRecognizer(_FakeRecognizer):
    def create_stream(self):
        return _FakeOfflineStream()

    def decode_streams(self, streams):
        for s in streams:
            s._decoded = True


class _FakeSherpaOnnx:
    OnlineRecognizer = _FakeRecognizer
    OfflineRecognizer = _FakeOfflineRecognizer


# ── _SherpaModelDir / windowing ────────────────────────────────────────────
def test_model_dir_resolves_variable_filenames(tmp_path):
    files = _SherpaModelDir(_make_model_dir(tmp_path))
    assert "encoder" in files.encoder.name
    assert "decoder" in files.decoder.name
    assert "joiner" in files.joiner.name
    assert files.tokens.name == "tokens.txt"


def test_model_dir_missing_files_raises(tmp_path):
    files = _SherpaModelDir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = files.encoder


def test_trailing_window_truncates_and_pads():
    audio = np.ones(16000 * 10, dtype=np.float32)  # 10s
    assert len(_trailing_window(audio, 8.0)) == 16000 * 8
    short = np.ones(320, dtype=np.float32)
    assert len(_trailing_window(short, 8.0)) == 320


# ── sherpa streaming backend ───────────────────────────────────────────────
@pytest.fixture
def fake_sherpa(monkeypatch):
    fake = _FakeSherpaOnnx()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    return fake


def test_sherpa_streaming_flow(tmp_path, fake_sherpa):
    backend = SherpaZipformerBackend(model_dir=_make_model_dir(tmp_path), num_threads=1)
    backend.load()

    # No audio yet -> empty partial.
    assert backend.partial().text == ""

    # Frame-by-frame streaming partials.
    backend.push(np.zeros(320, dtype=np.float32))
    backend.push(np.zeros(320, dtype=np.float32))
    assert backend.partial().text == "help me find my keys"

    # Final turn transcription drains a fresh stream.
    tr = backend.transcribe(np.zeros(3200, dtype=np.float32))
    assert tr.text == "help me find my keys"

    # reset() starts a fresh turn: push again yields partials again.
    backend.reset()
    backend.push(np.zeros(320, dtype=np.float32))
    assert backend.partial().text == "help me find my keys"


def test_sherpa_passes_expected_kwargs(tmp_path, fake_sherpa):
    SherpaZipformerBackend(model_dir=_make_model_dir(tmp_path)).load()
    kwargs = _FakeRecognizer.last_kwargs
    assert kwargs["sample_rate"] == 16000
    assert kwargs["feature_dim"] == 80
    assert kwargs["decoding_method"] == "greedy_search"
    assert kwargs["provider"] == "cpu"
    assert kwargs["enable_endpoint"] is False


def test_sherpa_missing_model_files_raise(tmp_path, fake_sherpa):
    backend = SherpaZipformerBackend(model_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_sherpa_missing_library_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    backend = SherpaZipformerBackend(model_dir=_make_model_dir(tmp_path))
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        backend.load()


# ── parakeet offline backend ───────────────────────────────────────────────
def test_parakeet_transcribes_offline(tmp_path, fake_sherpa):
    backend = ParakeetBackend(model_dir=_make_model_dir(tmp_path))
    backend.load()
    tr = backend.transcribe(np.zeros(3200, dtype=np.float32))
    assert tr.text == "offline final"
    # partial() re-transcribes a trailing window.
    assert backend.partial(np.ones(3200, dtype=np.float32), window_sec=8.0).text == "offline final"
    assert backend.partial(None).text == ""


# ── moonshine backend ──────────────────────────────────────────────────────
class _MoonshineModule:
    class TranscriptEventListener:
        def __init__(self):
            pass

    class FakeLine:
        text = ""

    class FakeEvent:
        def __init__(self, line):
            self.line = line

    class FakeStream:
        def __init__(self):
            self.listeners = []
            self.audio_chunks = []

        def add_listener(self, listener):
            self.listeners.append(listener)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def add_audio(self, chunk, sample_rate):
            self.audio_chunks.append(chunk)

    class Transcriber:
        def __init__(self, **kwargs):
            self.stream = None

        def create_stream(self, **kwargs):
            self.stream = _MoonshineModule.FakeStream()
            return self.stream

    def get_model_for_language(self, language):
        return ("/models/moonshine-en", "moonshine")


@pytest.fixture
def fake_moonshine(monkeypatch):
    fake = _MoonshineModule()
    monkeypatch.setitem(sys.modules, "moonshine_voice", fake)
    return fake


def test_moonshine_streaming_flow(fake_moonshine):
    backend = MoonshineBackend(language="en")
    backend.load()
    stream = backend._ensure_stream()

    # Simulate the listener: partial updates + completed lines.
    backend._on_text("the first words")
    assert backend.partial().text == "the first words"
    backend._on_line("the first words")
    backend.push(np.zeros(320, dtype=np.float32))
    assert len(stream.audio_chunks) == 1

    tr = backend.transcribe(np.zeros(320, dtype=np.float32))
    assert tr.text == "the first words"

    backend.reset()
    assert backend.partial().text == ""


def test_moonshine_missing_library_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "moonshine_voice", None)
    backend = MoonshineBackend()
    with pytest.raises(RuntimeError, match="moonshine-voice is not installed"):
        backend.load()


# ── faster-whisper backend (no model load) ─────────────────────────────────
def test_faster_whisper_empty_partial():
    backend = FasterWhisperBackend(model_size="base")
    tr = backend.partial(np.zeros(0, dtype=np.float32), window_sec=8.0)
    assert tr.text == ""


# ── factory ────────────────────────────────────────────────────────────────
def test_factory_maps_backends(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    assert isinstance(
        create_transcriber(_cfg("sherpa", models_dir=model_dir.parent)), SherpaZipformerBackend
    )
    assert isinstance(
        create_transcriber(_cfg("parakeet", models_dir=model_dir.parent)), ParakeetBackend
    )
    assert isinstance(create_transcriber(_cfg("moonshine")), MoonshineBackend)
    assert isinstance(
        create_transcriber(_cfg("faster_whisper", model="base")), FasterWhisperBackend
    )


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown STT_BACKEND"):
        create_transcriber(_cfg("nope"))


def test_transcriber_alias_for_backward_compat():
    # Anything importing `Transcriber` (the old class name) gets the interface.
    assert Transcriber is not None
    assert hasattr(Transcriber, "partial")
    assert hasattr(Transcriber, "transcribe")