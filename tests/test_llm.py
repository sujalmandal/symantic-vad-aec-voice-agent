import pytest

from unmute_tui.llm import rechunk_to_words


async def _collect(iterator):
    return [x async for x in iterator]


@pytest.mark.asyncio
async def test_rechunk_to_words_basic():
    async def gen():
        for chunk in ["hello ", "world ", "foo bar"]:
            yield chunk

    words = await _collect(rechunk_to_words(gen()))
    assert words == ["hello", " world", " foo", " bar"]


@pytest.mark.asyncio
async def test_rechunk_to_words_merges_whitespace():
    async def gen():
        yield "a  \n b"

    words = await _collect(rechunk_to_words(gen()))
    assert words == ["a", " b"]


@pytest.mark.asyncio
async def test_rechunk_to_words_single_word():
    async def gen():
        yield "hello"

    words = await _collect(rechunk_to_words(gen()))
    assert words == ["hello"]
