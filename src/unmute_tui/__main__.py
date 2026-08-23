"""Entry point for unmute-tui.

Usage:
    unmute-tui                 # run the TUI
    unmute-tui --list-devices  # list audio devices
    unmute-tui --no-tui        # headless mode (prints events to stdout)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .audio import AudioPlayer, Microphone, list_devices
from .config import Config
from .engine import ConversationEngine
from .llm import LLM
from .stt import Transcriber
from .tts import create_tts
from .vad import SemanticVAD
from .vad.silero_vad import SileroVAD
from .vad.smart_turn import SmartTurn

SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"


def _smart_turn_path(models_dir: Path) -> Path:
    return models_dir / SMART_TURN_FILENAME


def build_engine(config: Config):
    """Construct the full engine from config (used by both TUI and headless)."""
    smart_turn = SmartTurn(_smart_turn_path(config.models_dir))
    silero = SileroVAD(energy_threshold=config.vad.energy_threshold)
    vad = SemanticVAD(
        smart_turn=smart_turn,
        silero=silero,
        threshold=config.vad.threshold,
        turn_end_silence_sec=config.vad.turn_end_silence_sec,
        semantic_min_silence_sec=config.vad.semantic_min_silence_sec,
    )
    transcriber = Transcriber(
        model_size=config.stt_model,
        language=config.stt_language,
    )
    llm = LLM(config.llm)
    tts = create_tts(config.tts)
    mic = Microphone(device=config.audio.input_device)
    player = AudioPlayer(device=config.audio.output_device)
    return vad, transcriber, llm, tts, mic, player


async def _headless(config: Config, engine: ConversationEngine) -> None:
    """Run the engine without a TUI, printing events to stdout."""
    from .engine import (
        AssistantDelta,
        AssistantDone,
        Log,
        SessionEnd,
        StateChanged,
        UserTranscript,
        VADUpdate,
    )

    events = engine.events

    async def consume() -> None:
        while True:
            ev = await events.get()
            if isinstance(ev, VADUpdate):
                continue  # too noisy
            if isinstance(ev, AssistantDelta):
                print(ev.text, end="", flush=True)
            elif isinstance(ev, AssistantDone):
                print()
            elif isinstance(ev, UserTranscript):
                print(f"\n[You] {ev.text}")
            elif isinstance(ev, StateChanged):
                print(f"\n[state] {ev.state}")
            elif isinstance(ev, Log):
                print(f"\n[log] {ev.message}")
            elif isinstance(ev, SessionEnd):
                print(f"\n[session end] {ev.reason}")
                return

    try:
        await asyncio.gather(engine.run(), consume())
    except KeyboardInterrupt:
        pass
    finally:
        await engine.shutdown()


def _make_engine(config: Config) -> ConversationEngine:
    events: asyncio.Queue = asyncio.Queue()
    vad, transcriber, llm, tts, mic, player = build_engine(config)
    aec = None
    if config.aec.enabled:
        from .aec import WebRTCAEC

        aec = WebRTCAEC(
            delay_ms=config.aec.delay_ms,
            noise_suppression=config.aec.noise_suppression,
            ns_level=config.aec.ns_level,
        )
    return ConversationEngine(
        config, vad, transcriber, llm, tts, mic, player, events, aec=aec
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unmute-tui")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices")
    parser.add_argument("--no-tui", action="store_true", help="headless mode")
    parser.add_argument("--config", default=".env", help="path to .env file")
    args = parser.parse_args(argv)

    if args.list_devices:
        print(list_devices())
        return 0

    config = Config.from_env(args.config)
    engine = _make_engine(config)

    # Loading gate: pre-load all models before the event loop starts. This
    # avoids subprocess/thread conflicts with the asyncio loop and ensures the
    # bot never interrupts itself while a model is still loading.
    print("Loading models...", flush=True)
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load models: {exc}", file=sys.stderr)
        return 1
    print("Models ready.", flush=True)

    if args.no_tui:
        try:
            asyncio.run(_headless(config, engine))
        except KeyboardInterrupt:
            pass
        return 0

    from .ui import UnmuteApp

    app = UnmuteApp(engine, engine.events)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
