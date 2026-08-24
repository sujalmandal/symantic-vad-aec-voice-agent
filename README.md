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
| Kyutai TTS | **Chatterbox-Turbo TTS** (natural, expressive, real-time local; kokoro/edge/piper fallbacks) |

## Architecture

```
Terminal TUI (textual) ── asyncio events ──┐
                                          ▼
              Conversation Engine (state machine)
              • barge-in  • streaming TTS  • flush  • long-silence  • goodbye
        ┌───────────────┬──────────────────┬──────────────────┐
        ▼               ▼                  ▼                  ▼
   Audio Capture    Semantic VAD        STT (words)        LLM (OpenAI-compat)
   (sounddevice)   Silero (activity)   sherpa-onnx         Ollama / cloud
                   + Smart Turn v3     Zipformer           ──► TTS (Chatterbox)
                   (turn-end prob)     (streaming)         (streaming) ──► playback
```

### Semantic VAD (semantic + silence)
Turn-end detection combines two signals for reliability:
- **Silence-based** (reliable): [Silero VAD](https://github.com/snakers4/silero-vad)
  detects speech activity; once you stop speaking, ~`TURN_END_SILENCE_SEC` (0.7s)
  of sustained silence ends the turn — this always works.
- **Semantic** (accelerator): [Smart Turn v3](https://github.com/pipecat-ai/smart-turn)
  (8M params, 8MB int8 ONNX, ~10ms CPU) predicts the probability you've
  **finished your turn** from the raw waveform (prosody, not transcript). When
  it's confident (prob > `VAD_THRESHOLD`) after a short silence
  (`SEMANTIC_MIN_SILENCE_SEC`, 0.4s), it ends the turn sooner.

### LLM turn orchestrator (optional, `TURN_DETECTOR=llm`)
An experimental turn detector that mirrors the "continuous polling LLM" voice
architecture: while you are speaking, the app feeds two **parallel signals** to
an LLM **orchestrator** on every poll —

- **Process A — streaming partial transcripts**: a trailing window of the
  in-progress turn is re-transcribed (throttled at `TURN_PARTIAL_POLL_INTERVAL_SEC`)
  so the LLM sees your latest words, not a finished sentence.
- **Process B — VAD/audio cues**: your speech activity right now (speaking?,
  current silence length, and the Smart Turn acoustic turn-end probability).

The orchestrator decides one of three states each poll:

| Decision | Meaning |
|----------|---------|
| `WAIT`   | You are pausing briefly; keep waiting |
| `THINK`  | You need more time; wait patiently |
| `RESPOND`| Text is complete AND audio shows a hand-off — speak the reply now |

When it says `RESPOND` it also returns the **drafted reply**, so the bot speaks
immediately with near-zero turn-end latency (no separate turn-end transcription
+ cold LLM generation). The reliable audio silence timeout still backstops: if
the LLM stalls or the partial is too short, the turn ends normally.

Enable it with `TURN_DETECTOR=llm`; the default `semantic` is unchanged.
Tune with `TURN_PARTIAL_POLL_INTERVAL_SEC`, `TURN_ORCHESTRATOR_POLL_INTERVAL_SEC`,
`TURN_MIN_PARTIAL_CHARS`, and `TURN_STT_POLL_WINDOW_SEC`.

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

# 2. Download the semantic VAD model
uv run python scripts/download_models.py

# 3. Configure your LLM (copy and edit)
cp .env.example .env
#   - Local Ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama
#   - Cloud OpenAI:  LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...
#   - Cloud OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1  LLM_API_KEY=sk-or-...
```

Chatterbox-Turbo TTS (the default `TTS_BACKEND`) runs locally on Apple Silicon
via `mlx-audio`; its model downloads on first use. It's natural and expressive
with inline emotion tags like `[sigh]` and `[laugh]`.

## Single-file version

Everything is also packed into one self-contained script, `unmute_tui.py`, so
you can run it without the package layout:

```bash
# Install dependencies once
pip install numpy sounddevice onnxruntime transformers silero-vad \
            sherpa-onnx faster-whisper openai textual edge-tts \
            python-dotenv kokoro-onnx mlx-audio

# Download the VAD model, then run
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
- `TTS_BACKEND` — `chatterbox` (default), `kokoro`, `edge`, or `piper`.
- `CHATTERBOX_MODEL` — Chatterbox-Turbo MLX model (default `mlx-community/chatterbox-turbo-4bit`).
- `KOKORO_MODEL` / `KOKORO_VOICES` / `KOKORO_VOICE` / `KOKORO_SPEED` / `KOKORO_LANG` — Kokoro fallback.
- `VAD_THRESHOLD` — semantic turn-end probability threshold (default `0.6`).
- `TURN_END_SILENCE_SEC` / `SEMANTIC_MIN_SILENCE_SEC` — turn-end silence
  (reliable fallback) and semantic accelerator silence.
- `BARGE_IN_MIN_RMS` / `BARGE_IN_REQUIRED_FRAMES` — barge-in robustness: while
  the bot speaks, only treat the mic as the user if its AEC-cleaned RMS is at
  least `BARGE_IN_MIN_RMS` and sustained for `BARGE_IN_REQUIRED_FRAMES`. This
  ignores the AEC's low-level echo residual so the bot doesn't interrupt itself;
  louder real user speech still barges in.
- `BARGE_IN_OVER_PLAYBACK_DB` — no-AEC barge-in margin: when echo cancellation
  is unavailable, a mic frame only counts as the user if it is this much louder
  than the bot's recent playback, so the bot never interrupts itself with its
  own TTS echo. Default `6.0` dB.
- `MUTE_MIC_WHILE_BOT_SPEAKING` — half-duplex mode (mutes the mic while the bot
  speaks; guarantees no self-reply but disables barge-in). **Off by default** so
  barge-in is live in all configs.
- `BACKCHANNEL_ENABLED` / `VAP_BC_MODEL` / `CPC_MODEL` / `BACKCHANNEL_THRESHOLD`
  / `BACKCHANNEL_COOLDOWN_SEC` / `BACKCHANNEL_ACK_TEXT` — active-listening
  backchannels: while the user is speaking, VAP predicts when to backchannel and
  the bot emits a short ack ("Mm-hmm") without taking the turn. Off by default.
- `USER_SILENCE_TIMEOUT` — seconds before the `"..."` marker (default `7.0`).
- `UNINTERRUPTIBLE_BY_VAD_TIME_SEC` — bot's protected window at turn start
  (default `0.3`). Barge-in is live after this.
- `STT_BACKEND` / `STT_MODEL` / `STT_MODELS_DIR` / `STT_THREADS` — STT backend
  selection and model path (see [Speech-to-text backends](#speech-to-text-backends)).
- `AEC_ENABLED` / `AEC_DELAY_MS` / `AEC_NOISE_SUPPRESSION` — WebRTC AEC3
  echo cancellation (removes the bot's own TTS echo from the mic so full-duplex
  barge-in works even without headphones). On by default.

## Speech-to-text backends

The app ships with pluggable, fully local STT backends (select with `STT_BACKEND`):

| `STT_BACKEND` | Model | Streaming | Speed (local CPU) | Accuracy |
|---|---|---|---|---|
| `sherpa` (default) | sherpa-onnx streaming Zipformer ([docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)) | ✅ true incremental partials | RTF ≈ 0.03–0.05 int8 (~20–30× realtime) | strong conversational WER |
| `moonshine` | Moonshine v2 ([repo](https://github.com/moonshine-ai/moonshine)) | ✅ streaming | very low latency (~150 ms Small) | competitive with Whisper Large V3 |
| `parakeet` | NVIDIA Parakeet TDT-0.6B ([sherpa-onnx ONNX](https://github.com/mil-ad/parakeet-tdt-0.6b-v3-fastapi-openai)) | ❌ offline (fast) | RTF ≈ 0.05 | best raw WER (LS-clean ~2.2%) |
| `faster_whisper` | faster-whisper `base`/`large-v3-turbo` | ❌ batch/windowed | — | lower than the above |

Why `sherpa` is the default: it is the only backend with **true streaming**
decoding — mic frames are fed as they arrive and `partial()` returns the words
as they are spoken — so the LLM turn orchestrator sees the transcript with
essentially zero extra latency, and it is far more accurate than the old
whisper `base` on spontaneous speech (the "misunderstood words" problem).
(Zipformer emits uppercase; the backend lowercases it for natural LLM input.)

Setup (after `uv sync --extra stt`):

```bash
uv run python scripts/download_models.py   # fetches the Zipformer (and Parakeet) models
uv run unmute-tui                          # default STT_BACKEND=sherpa
```

Swap backends via `.env`:

```bash
STT_BACKEND=moonshine          # or sherpa | parakeet | faster_whisper
STT_MODELS_DIR=models/stt      # sherpa/parakeet ONNX assets
STT_THREADS=2                  # decoder threads
```

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

No model download or C++ build is needed — AEC3 ships with the
`pywebrtc-audio` wheel (`pip install pywebrtc-audio`).

### Barge-in without AEC

If AEC is unavailable (`AEC_ENABLED=false` or `pywebrtc-audio` not installed),
barge-in still works via a **playback-aware echo gate**: the app remembers how
loud the bot's TTS just was and only treats a mic frame as *you* when it is
`BARGE_IN_OVER_PLAYBACK_DB` louder than that recent playback. Your voice speaking
over the bot interrupts it; the bot's own echo never makes it interrupt itself.

- To force the old guaranteed half-duplex behavior instead (mute the mic while
  the bot speaks), set `MUTE_MIC_WHILE_BOT_SPEAKING=true`.
- Without AEC, a little of the bot's echo can still leak into the next turn's
  recording (inherent — there's no true echo cancellation). AEC remains
  recommended for the cleanest transcription.

## Backchannels (active listening)

While the user is speaking, the bot can emit short acknowledgments ("Mm-hmm",
"Uh-huh") without taking the turn, so it feels like it's listening. It uses
[VAP (Voice Activity Projection)](https://github.com/inokoj/VAP-Realtime) — a
real-time (~8ms/frame) model that predicts when the listener should backchannel,
from stereo audio (bot far-end + user near-end). The model code is vendored in
`src/rvap/`; enable with:

```bash
BACKCHANNEL_ENABLED=true
uv run python scripts/download_models.py   # fetches the VAP-BC + CPC models
```

The bot acks when VAP's backchannel probability exceeds `BACKCHANNEL_THRESHOLD`
(0.5) and a cooldown (`BACKCHANNEL_COOLDOWN_SEC`) has elapsed. The ack never
advances the conversation state (the user keeps their turn) and is fed to the
AEC as reference so it isn't misheard as the user.

## Tests

```bash
uv run pytest
```

## Notes & limitations

- The exact Kyutai STT/TTS models require CUDA/Linux and cannot run on this Mac;
  Smart Turn v3 and Chatterbox-Turbo are equivalent local replacements.
- Chatterbox-Turbo runs via `mlx-audio` on Apple Silicon. On other platforms, set
  `TTS_BACKEND=kokoro` (local) or `edge` (free, cloud).
- Barge-in uses WebRTC AEC3 (works without headphones). For guaranteed
  no-self-reply without AEC, set `MUTE_MIC_WHILE_BOT_SPEAKING=true`.
