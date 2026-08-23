"""LLM client (OpenAI-compatible) with streaming and word rechunking.

Works with any OpenAI-compatible endpoint: local Ollama, vLLM, or cloud
(OpenAI / OpenRouter). The stream is rechunked to whole words so the TTS can
synthesize incrementally without splitting words (mirrors unmute's
`rechunk_to_words`).
"""

from __future__ import annotations

import re
from typing import AsyncIterator

from openai import AsyncOpenAI

from .config import LLMConfig

INTERRUPTION_CHAR = "—"  # em-dash
USER_SILENCE_MARKER = "..."

# MiniMax reasoning models wrap their chain-of-thought in markers (e.g.
# " thinking\n...\n response\n", "<thinking>...</thinking>", or
# "<think>...</think>") before the actual answer. We strip that so the voice
# assistant only speaks the final answer.
_THINKING_START_RE = re.compile(r"^\s*(?:<)?/?think(?:ing)?\b")
_THINKING_END_RE = re.compile(r"\sresponse\s*\n|</?response>|</thinking>")


async def _strip_thinking(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield a text stream with the leading chain-of-thought removed.

    If the stream begins with a thinking marker ("thinking" or "<thinking>"),
    everything up to and including the first end marker ("response" on its own
    line, a response tag, or "</thinking>") is dropped. Otherwise the stream is
    passed through unchanged.
    """
    buffer = ""
    started = False
    async for delta in iterator:
        if not started:
            buffer += delta
            if _THINKING_START_RE.match(buffer):
                m = _THINKING_END_RE.search(buffer)
                if m is not None:
                    started = True
                    rest = buffer[m.end():].lstrip("\n ")
                    if rest:
                        yield rest
                    buffer = ""
                # else keep buffering until the end marker appears
            else:
                # No thinking marker: emit everything buffered so far.
                started = True
                if buffer:
                    yield buffer
                    buffer = ""
        else:
            yield delta
    if buffer:
        yield buffer


async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Rechunk a text stream to whole words.

    Spaces are attached to the following word, so "foo bar" yields "foo", " bar".
    Multiple whitespace characters are merged into a single space.
    """
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end():]
            if chunk != "":
                yield prefix + chunk
            prefix = " "
    if buffer != "":
        yield prefix + buffer


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=config.base_url,
        )
        self._model: str | None = config.model

    async def _resolve_model(self) -> str:
        if self._model:
            return self._model
        models = await self._client.models.list()
        ids = [m.id for m in models.data]
        if len(ids) != 1:
            raise RuntimeError(
                f"Endpoint exposes {len(ids)} models; set LLM_MODEL explicitly. "
                f"Available: {ids}"
            )
        self._model = ids[0]
        return self._model

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream the assistant reply as text deltas.

        Uses MiniMax's `reasoning_split` so chain-of-thought is returned in a
        separate `reasoning_details` field and `content` holds only the spoken
        answer.
        """
        model = await self._resolve_model()
        extra_body: dict = {"reasoning_split": True}
        if self.config.thinking:
            extra_body["thinking"] = {"type": self.config.thinking}
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            temperature=0.7,
            extra_body=extra_body,
        )

        async def _raw() -> AsyncIterator[str]:
            async with stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if not content:
                        continue
                    yield content

        # reasoning_split should keep `content` clean; _strip_thinking is a
        # defensive fallback in case an endpoint ignores it.
        async for delta in _strip_thinking(_raw()):
            yield delta
