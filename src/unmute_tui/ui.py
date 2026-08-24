"""Terminal UI (textual) for the conversation engine."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

from .engine import (
    AssistantDelta,
    AssistantDone,
    EngineEvent,
    Latency,
    Log,
    PartialUpdate,
    SessionEnd,
    StateChanged,
    UserTranscript,
    VADUpdate,
)


def _vad_bar(probability: float) -> str:
    width = 20
    filled = int(round(probability * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {probability:.2f}"


class UnmuteApp(App):
    TITLE = "unmute-tui"
    CSS = """
    #status-row { height: 3; padding: 0 1; }
    #status-row Static { width: 1fr; }
    #transcript { height: 1fr; border: round $primary; }
    #current { height: 3; border: round $accent; padding: 0 1; }
    """

    def __init__(self, engine, events: asyncio.Queue[EngineEvent]) -> None:
        super().__init__()
        self.engine = engine
        self.events = events
        self._engine_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-row"):
            yield Static("state: waiting_for_user", id="state")
            yield Static("vad: " + _vad_bar(0.0), id="vad")
            yield Static("latency: -", id="latency")
        yield RichLog(id="transcript", highlight=True, markup=True, wrap=True)
        yield Static("", id="current")
        yield Footer()

    async def on_mount(self) -> None:
        self._engine_task = asyncio.create_task(self.engine.run())
        self._consumer_task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        while True:
            event = await self.events.get()
            self._handle(event)

    def _handle(self, event: EngineEvent) -> None:
        if isinstance(event, StateChanged):
            self.query_one("#state", Static).update(f"state: {event.state}")
        elif isinstance(event, VADUpdate):
            self.query_one("#vad", Static).update("vad: " + _vad_bar(event.probability))
        elif isinstance(event, PartialUpdate):
            # Show the latest partial transcript / orchestrator decision dimmed.
            suffix = f" ({event.decision})" if event.decision else ""
            self.query_one("#current", Static).update(
                f"[dim]You: {event.text}{suffix}[/]"
            )
        elif isinstance(event, UserTranscript):
            self.query_one("#transcript", RichLog).write(f"[bold cyan]You:[/] {event.text}")
        elif isinstance(event, AssistantDelta):
            current = self.query_one("#current", Static)
            current.update(f"[bold magenta]Bot:[/] {event.text}")
        elif isinstance(event, AssistantDone):
            self.query_one("#current", Static).update("")
            self.query_one("#transcript", RichLog).write(
                f"[bold magenta]Bot:[/] {event.text}"
            )
        elif isinstance(event, Latency):
            self.query_one("#latency", Static).update(
                f"latency: {event.metric}={event.value:.2f}"
            )
        elif isinstance(event, Log):
            self.query_one("#transcript", RichLog).write(f"[dim]{event.message}[/]")
        elif isinstance(event, SessionEnd):
            self.query_one("#transcript", RichLog).write(
                f"[bold yellow]Session ended: {event.reason}[/]"
            )
            self.call_after(self.exit, event.reason)

    async def on_unmount(self) -> None:
        if self._engine_task is not None:
            self._engine_task.cancel()
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        await self.engine.shutdown()
