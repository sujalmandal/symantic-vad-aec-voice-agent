#!/usr/bin/env python3
"""Download model weights needed by unmute-tui.

Downloads:
  * Smart Turn v3 (semantic VAD) ONNX model -> models/smart-turn-v3.2-cpu.onnx

Silero VAD, faster-whisper, and Marvis TTS weights are fetched automatically on
first use by their respective libraries, so they are not downloaded here.
(WebRTC AEC3 needs no model — it ships with the `pywebrtc-audio` wheel.)

Usage:
    python scripts/download_models.py [--models-dir models]
"""

from __future__ import annotations

import argparse
from pathlib import Path

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"

# Default reference voice for Marvis TTS voice cloning (from the Qwen3-TTS repo).
QWEN_DEFAULT_VOICE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
DEFAULT_VOICE_FILE = "default-voice.wav"
DEFAULT_VOICE_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)


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

    # Default Marvis TTS reference voice.
    voice_dest = models_dir / DEFAULT_VOICE_FILE
    if voice_dest.exists():
        print(f"Default voice already present: {voice_dest}")
    else:
        print(f"Downloading default voice -> {voice_dest} ...")
        import urllib.request

        urllib.request.urlretrieve(QWEN_DEFAULT_VOICE_URL, voice_dest)
        print(f"Downloaded to {voice_dest}")

    print("\nDefault reference text (set TTS_REF_TEXT to override):")
    print(f"  {DEFAULT_VOICE_TEXT}")


if __name__ == "__main__":
    main()
