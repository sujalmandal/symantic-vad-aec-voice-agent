"""WebRTC AEC3 echo cancellation via `pywebrtc-audio`.

WebRTC's AEC3 (the same algorithm Chrome uses) removes the far-end echo — the
bot's own TTS played through speakers — from the microphone signal, so the bot
doesn't hear itself and full-duplex barge-in is possible. It needs the mic input
plus a far-end reference (a loopback of what the speakers play), which the engine
has because it knows exactly what TTS audio it is playing.

Benchmark on real TTS: ~52 dB echo attenuation with ~2-3 dB user-speech loss,
far better than a small neural model.
"""

from __future__ import annotations

import numpy as np

# WebRTC AEC3 aligns frames; we process in 20 ms (320 @ 16 kHz) blocks to match
# the engine's mic frames. It also accepts longer buffers (internally frames at
# 10 ms and zero-pads).
BLOCK = 320
# Keep at most this much reference audio buffered (2 s).
MAX_REF_SEC = 2.0


class AECError(RuntimeError):
    pass


class WebRTCAEC:
    def __init__(
        self,
        sample_rate: int = 16000,
        delay_ms: int = 30,
        noise_suppression: bool = False,
        ns_level: int = 1,
    ) -> None:
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as exc:  # pragma: no cover
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

    def add_reference(self, audio: np.ndarray) -> None:
        """Feed the far-end reference (the TTS audio being played)."""
        self.ref_buf = np.concatenate([self.ref_buf, np.asarray(audio, dtype=np.float32)])
        max_ref = int(MAX_REF_SEC * 16000)
        if len(self.ref_buf) > max_ref:
            self.ref_buf = self.ref_buf[-max_ref:]

    def process_mic(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed a mic frame; return a clean 320-sample block, or None if not
        enough samples have accumulated yet."""
        self.mic_buf = np.concatenate([self.mic_buf, np.asarray(frame, dtype=np.float32)])
        if len(self.mic_buf) < BLOCK:
            return None
        mic_block = self.mic_buf[:BLOCK]
        self.mic_buf = self.mic_buf[BLOCK:]

        if len(self.ref_buf) >= BLOCK:
            ref_block = self.ref_buf[:BLOCK]
            self.ref_buf = self.ref_buf[BLOCK:]
        else:
            ref_block = np.zeros(BLOCK, dtype=np.float32)

        out = np.asarray(self._ap.process(mic_block, ref_block), dtype=np.float32)
        return out

    def reset(self) -> None:
        """Reset AEC state and clear buffers (e.g. on interruption)."""
        self._ap.reset()
        self.mic_buf = np.zeros(0, dtype=np.float32)
        self.ref_buf = np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self.reset()
