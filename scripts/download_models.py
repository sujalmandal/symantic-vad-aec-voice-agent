#!/usr/bin/env python3
"""Download model weights needed by unmute-tui.

Downloads:
  * Smart Turn v3 (semantic VAD) ONNX model -> models/smart-turn-v3.2-cpu.onnx
  * Kokoro-82M TTS ONNX model + voices -> models/kokoro-v1.0.onnx, voices-v1.0.bin
  * (optional) VAP backchannel model assets -> models/*.pt
  * sherpa-onnx streaming Zipformer STT -> models/stt/<model>/ (default STT)
  * (optional) NVIDIA Parakeet TDT-0.6B STT -> models/stt/<model>/

Silero VAD weights are fetched automatically on first use by silero-vad.
"""

from __future__ import annotations

import argparse
import tarfile
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

# VAP / Voice Activity Projection (bot backchannels) model assets.
VAP_BC_URL = (
    "https://raw.githubusercontent.com/inokoj/VAP-Realtime/main/"
    "asset/vap_bc/vap-bc_state_dict_erica_10hz_3000msec.pt"
)
CPC_URL = (
    "https://raw.githubusercontent.com/inokoj/VAP-Realtime/main/"
    "asset/cpc/60k_epoch4-d0f474de.pt"
)
VAP_BC_FILE = "vap-bc_state_dict_erica_10hz_3000msec.pt"
CPC_FILE = "60k_epoch4-d0f474de.pt"

# sherpa-onnx STT models (default streaming Zipformer + optional Parakeet).
SHERPA_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
# Default: English streaming Zipformer (true streaming, ~44MB int8).
ZIPFORMER_MODEL = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
# Optional: NVIDIA Parakeet TDT-0.6B (best raw WER, non-streaming).
PARAKEET_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def _download_url(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded to {dest}")


def _download_and_extract_tarbz2(name: str, dest_dir: Path) -> None:
    """Fetch `<name>.tar.bz2` from the sherpa-onnx release and extract it."""
    tarball = dest_dir / f"{name}.tar.bz2"
    if (dest_dir / name).exists():
        print(f"Sherpa-onnx model already present: {dest_dir / name}")
    else:
        url = f"{SHERPA_RELEASE}/{name}.tar.bz2"
        try:
            _download_url(url, tarball)
            print(f"Extracting {tarball} ...")
            with tarfile.open(tarball, "r:bz2") as tf:
                tf.extractall(dest_dir)  # noqa: S202 — trusted release artifacts
            tarball.unlink()
            print(f"Extracted to {dest_dir / name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not fetch {name}: {exc}")
            if tarball.exists():
                tarball.unlink()


def _patch_nemo_decoder_metadata(model_dir: Path) -> None:
    """Patch missing RNNT metadata onto a NeMo (Parakeet) decoder ONNX.

    The sherpa-onnx release of `parakeet-tdt-0.6b-v3` carries an int8 decoder
    WITHOUT the `vocab_size`/`context_size` metadata that sherpa-onnx requires
    (it hard-aborts otherwise). The encoder has the real values; we copy them
    over so the Parakeet backend works out of the box.
    """
    decoder = model_dir / "decoder.int8.onnx"
    encoder = model_dir / "encoder.int8.onnx"
    if not decoder.exists() or not encoder.exists():
        return
    try:
        import onnx
    except ImportError:
        print(
            "Warning: `onnx` not installed; could not patch Parakeet decoder "
            "metadata (pip install onnx then rerun)."
        )
        return
    dec = onnx.load(str(decoder))
    enc = onnx.load(str(encoder))
    have = {p.key for p in dec.metadata_props}
    enc_meta = {p.key: p.value for p in enc.metadata_props}
    want = {
        "vocab_size": enc_meta.get("vocab_size", ""),
        "pred_rnn_layers": enc_meta.get("pred_rnn_layers", "2"),
        "pred_hidden": enc_meta.get("pred_hidden", "640"),
        "context_size": "2",  # NeMo transducer default
    }
    added = False
    for key, value in want.items():
        if key not in have and value:
            prop = dec.metadata_props.add()
            prop.key = key
            prop.value = value
            added = True
    if added:
        onnx.save(dec, str(decoder))
        print(f"Patched RNNT metadata onto {decoder.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--skip-stt",
        action="store_true",
        help="skip downloading the sherpa-onnx STT models",
    )
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

    # VAP backchannel model assets (optional; for bot backchannels).
    vap_dest = models_dir / VAP_BC_FILE
    cpc_dest = models_dir / CPC_FILE
    if vap_dest.exists():
        print(f"VAP-BC model already present: {vap_dest}")
    else:
        _download_url(VAP_BC_URL, vap_dest)
    if cpc_dest.exists():
        print(f"CPC encoder already present: {cpc_dest}")
    else:
        _download_url(CPC_URL, cpc_dest)

    # sherpa-onnx STT models (default streaming Zipformer + optional Parakeet).
    if not args.skip_stt:
        stt_dir = models_dir / "stt"
        stt_dir.mkdir(parents=True, exist_ok=True)
        _download_and_extract_tarbz2(ZIPFORMER_MODEL, stt_dir)
        _download_and_extract_tarbz2(PARAKEET_MODEL, stt_dir)
        # Parakeet's int8 decoder ships without RNNT metadata; patch it so the
        # backend doesn't hard-abort in sherpa-onnx.
        _patch_nemo_decoder_metadata(stt_dir / PARAKEET_MODEL)
        print(
            f"STT models in {stt_dir}: set STT_BACKEND=sherpa (default) or "
            f"STT_BACKEND=parakeet with STT_MODEL={PARAKEET_MODEL}."
        )


if __name__ == "__main__":
    main()
