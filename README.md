# unmute-tui

A **terminal replica of [unmute.sh](https://unmute.sh)** — a full-duplex spoken
conversation with an LLM, driven by a **semantic VAD** (turn-end detection from
speech content, not just energy) and a **fluent turn-taking engine** with
barge-in interruption and streaming TTS.

You talk, the bot listens, thinks, and speaks back — and you can interrupt it
mid-sentence.

## What it replicates from unmute.sh

| unmute.sh | unmute-tui |
|-----------|------------|
| STT pause head (`prs[2]`) + EMA + threshold | **Smart Turn v3** semantic VAD + EMA + threshold |
| `waiting_for_user → user_speaking → bot_speaking` state machine | same state machine |
| Barge-in interruption (clear TTS/LLM queues) | same |
| Word-by-word streaming TTS | sentence-chunked streaming TTS |
| Long-silence `"..."` marker | same |
| Bot ends with `"Bye!"` | same |
| Any OpenAI-compatible LLM | Ollama (local) or cloud (OpenAI/OpenRouter) |
| Kyutai TTS | **Marvis TTS** (real-time streaming, MLX-native; edge-tts/piper fallbacks) |

## Architecture

```
Terminal TUI (textual) ── asyncio events ──┐
                                          ▼
              Conversation Engine (state machine)
              • barge-in  • streaming TTS  • flush  • long-silence  • goodbye
        ┌───────────────┬──────────────────┬──────────────────┐
        ▼               ▼                  ▼                  ▼
   Audio Capture    Semantic VAD        STT (words)        LLM (OpenAI-compat)
   (sounddevice)   Silero (activity)   faster-whisper      Ollama / cloud
                   + Smart Turn v3     (streaming)         ──► TTS (Marvis)
                   (turn-end prob)                          (streaming) ──► playback
```

### Semantic VAD
[Smart Turn v3](https://github.com/pipecat-ai/smart-turn) is an audio-native
model (8M params, 8MB int8 ONNX, ~10ms CPU inference) that predicts the
probability the speaker has **finished their turn** from the raw waveform —
using prosody and acoustic cues, not a transcript. It runs alongside
[Silero VAD](https://github.com/snakers4/silero-vad) (raw speech activity).
The turn-end probability is smoothed with an exponential moving average
(attack/release) and compared against a threshold (default `0.6`), mirroring
unmute's `determine_pause()` logic.

## Requirements

- macOS (Apple Silicon) or Linux, Python 3.11+
- [uv](https://docs.astral.sh/uv/) and [PortAudio](https://portaudio.com/)
  (`brew install portaudio`)
- An LLM endpoint: local [Ollama](https://ollama.com) or a cloud
  OpenAI-compatible API key
- Internet for model downloads

## Setup

```bash
# 1. Install dependencies (core + STT + dev)
uv sync --extra stt --extra dev

# 2. Download the semantic VAD model + default reference voice
uv run python scripts/download_models.py

# 3. Configure your LLM (copy and edit)
cp .env.example .env
#   - Local Ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama
#   - Cloud OpenAI:  LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...
#   - Cloud OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1  LLM_API_KEY=sk-or-...
```

Marvis TTS (the default `TTS_BACKEND`) downloads its ~500MB model on first use
via `mlx-audio`.

## Single-file version

Everything is also packed into one self-contained script, `unmute_tui.py`, so
you can run it without the package layout:

```bash
# Install dependencies once
pip install numpy sounddevice onnxruntime transformers silero-vad \
            faster-whisper openai textual edge-tts python-dotenv mlx-audio

# Download the VAD model + default voice, then run
python unmute_tui.py --download-models
python unmute_tui.py            # TUI
python unmute_tui.py --no-tui   # headless
```

## Run

```bash
uv run unmute-tui            # TUI
uv run unmute-tui --no-tui   # headless (prints events to stdout)
uv run unmute-tui --list-devices
```

## Configuration

See [`.env.example`](.env.example). Key settings:

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — OpenAI-compatible endpoint
  (e.g. MiniMax: `https://api.minimax.io/v1`, model `MiniMax-M3`).
- `LLM_THINKING` — MiniMax-M3 thinking control: `disabled` (faster, no
  chain-of-thought) or `adaptive`. Empty = default.
- `TTS_BACKEND` — `marvis` (default), `edge`, or `piper`.
- `MARVIS_MODEL` — Marvis TTS model id / local path.
- `TTS_REF_AUDIO` / `TTS_REF_TEXT` — reference voice for Marvis voice cloning
  (defaults to the bundled sample; point to your own `.wav` to clone a custom
  voice).
- `VAD_THRESHOLD` — turn-end probability threshold (default `0.6`).
- `USER_SILENCE_TIMEOUT` — seconds before the `"..."` marker (default `7.0`).
- `UNINTERRUPTIBLE_BY_VAD_TIME_SEC` — bot's protected window at turn start.
- `STT_MODEL` — faster-whisper size (`tiny`/`base`/`small`/`medium`).
- `AEC_ENABLED` / `AEC_DELAY_MS` / `AEC_NOISE_SUPPRESSION` — WebRTC AEC3
  echo cancellation (removes the bot's own TTS echo from the mic so full-duplex
  barge-in works even without headphones). On by default.

## Acoustic echo cancellation (AEC)

Without headphones, the mic picks up the bot's own TTS through the speakers and
the bot interrupts (or replies to) itself. The app uses **WebRTC AEC3** (the
same algorithm Chrome uses) to remove the echo:

- `AEC_ENABLED=true` (default) — the mic stays on while the bot speaks and AEC3
  cancels the echo so you can barge in. It feeds the TTS audio back as the
  far-end reference. Benchmarked at ~52 dB echo attenuation with ~2 dB user
  speech loss.
- `AEC_DELAY_MS` — the speaker→mic delay hint (default `30`). Helps AEC
  converge; bump if your speakers are far from the mic.
- `AEC_NOISE_SUPPRESSION=true` (default) — WebRTC noise suppression on top of
  AEC, so background noise isn't transcribed as a user turn.

If AEC is disabled (`AEC_ENABLED=false`), the app falls back to muting the mic
while the bot speaks (`MUTE_MIC_WHILE_BOT_SPEAKING=true`) — the reliable
half-duplex mode that guarantees no self-reply.

No model download or C++ build is needed — AEC3 ships with the
`pywebrtc-audio` wheel (`pip install pywebrtc-audio`).

## Tests

```bash
uv run pytest
```

## Notes & limitations

- The exact Kyutai STT/TTS models require CUDA/Linux and cannot run on this Mac;
  Smart Turn v3 and Marvis TTS are equivalent local replacements.
- Marvis TTS is MLX-native (Apple Silicon). On other platforms, set
  `TTS_BACKEND=edge` (free, cloud) or `piper` (local).
- Barge-in uses a short protected window at the start of the bot's turn to
  avoid echo-triggered self-interruption (no echo cancellation in the terminal).
