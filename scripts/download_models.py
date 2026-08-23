#!/usr/bin/env python3
"""Download model weights needed by unmute-tui.

Downloads:
  * Smart Turn v3 (semantic VAD) ONNX model -> models/smart-turn-v3.2-cpu.onnx
  * Kokoro-82M TTS ONNX model + voices -> models/kokoro-v1.0.onnx, voices-v1.0.bin

Silero VAD and faster-whisper weights are fetched automatically on first use by
their respective libraries. (WebRTC AEC3 needs no model.)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

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


def _download_url(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Smart Turn semantic VAD model.
    dest = models_dir / SMART_TURN_FILE
    if dest.exists():
        print(f"Smart Turn model already present: {dest}")
    else:
        print(f"Downloading {SMART_TURN_REPO}/{SMART_TURN_FILE} ...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                "huggingface_hub is not installed. Run `uv sync` (or "
                "`pip install huggingface_hub`)."
            )
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
        _download_url(KOKORO_ONNX_URL, onnx_dest)
    if voices_dest.exists():
        print(f"Kokoro voices already present: {voices_dest}")
    else:
        _download_url(KOKORO_VOICES_URL, voices_dest)


if __name__ == "__main__":
    main()
