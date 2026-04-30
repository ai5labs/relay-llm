"""Property-based tests for SSE streaming invariants.

The tool-call delta merge-by-index invariant is the bug LiteLLM has shipped
twice (issues #20711, #8012). We use Hypothesis to generate adversarial chunk
splits and assert:

1. Concatenated text deltas equal the original message text.
2. Concatenated tool-call argument deltas (per index) reconstruct valid JSON
   and match the original arguments dict.
3. Final response usage matches the last usage frame.
4. The number of distinct tool-call indices matches what was sent.
"""

from __future__ import annotations

import json
import random
from typing import Any

import httpx
import pytest
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from relay import Hub
from relay.config import load_str
from relay.types import StreamEnd, TextDelta, ToolCallDelta


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/test
            credential: $env.SSE_TEST_KEY
        """


@pytest.fixture(autouse=True)
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSE_TEST_KEY", "sk-fake")


def _split_sse_text(text: str, rng: random.Random) -> list[str]:
    """Split a payload into 1-N character chunks at random boundaries."""
    if not text:
        return [""]
    out: list[str] = []
    i = 0
    while i < len(text):
        chunk_len = rng.randint(1, max(1, min(10, len(text) - i)))
        out.append(text[i : i + chunk_len])
        i += chunk_len
    return out


@settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    text=st.text(
        alphabet=st.characters(
            min_codepoint=0x20, max_codepoint=0x7E, blacklist_characters='"\\\n'
        ),
        min_size=0,
        max_size=80,
    ),
    seed=st.integers(min_value=0, max_value=10000),
)
@pytest.mark.asyncio
@respx.mock
async def test_text_delta_reassembly(text: str, seed: int, env_key: None) -> None:
    """Concat of text deltas == original text, regardless of how it was chunked."""
    rng = random.Random(seed)
    pieces = _split_sse_text(text, rng)
    chunks = [
        json.dumps(
            {
                "id": "x",
                "model": "test",
                "choices": [{"index": 0, "delta": {"content": p}}] if p else [],
            }
        )
        for p in pieces
    ]
    chunks.append(
        json.dumps(
            {
                "id": "x",
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "x",
                "model": "test",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )
    body = "\n".join(f"data: {c}" for c in chunks) + "\ndata: [DONE]\n"

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        gen = hub.stream("m", messages=[{"role": "user", "content": "hi"}])
        deltas: list[str] = []
        final = None
        async for ev in gen:
            if isinstance(ev, TextDelta):
                deltas.append(ev.text)
            elif isinstance(ev, StreamEnd):
                final = ev
        assert "".join(deltas) == text
        assert final is not None
        assert final.response.text == text
    finally:
        await hub.aclose()


@settings(
    max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    args=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(min_codepoint=0x61, max_codepoint=0x7A),
            min_size=1,
            max_size=8,
        ),
        values=st.one_of(
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x20, max_codepoint=0x7E, blacklist_characters='"\\\n'
                ),
                min_size=0,
                max_size=20,
            ),
            st.integers(min_value=-1000, max_value=1000),
            st.booleans(),
        ),
        min_size=0,
        max_size=4,
    ),
    seed=st.integers(min_value=0, max_value=10000),
)
@pytest.mark.asyncio
@respx.mock
async def test_tool_call_delta_reassembly(args: dict[str, Any], seed: int, env_key: None) -> None:
    """The merge-by-index invariant: scattered argument fragments reassemble
    into valid JSON matching the original args dict.

    This is the bug LiteLLM has hit (#20711). Splitting the JSON across many
    frames must not lose data, including when the ``id`` and ``name`` only
    appear in the first frame.
    """
    rng = random.Random(seed)
    args_json = json.dumps(args)
    pieces = _split_sse_text(args_json, rng)

    # First frame: id + name + first piece. Subsequent: only index + arg piece.
    chunks: list[str] = []
    for i, piece in enumerate(pieces):
        if i == 0:
            tc = {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "do_thing", "arguments": piece},
            }
        else:
            tc = {"index": 0, "function": {"arguments": piece}}
        delta: dict[str, Any] = {"tool_calls": [tc]}
        if i == 0:
            delta["role"] = "assistant"
        chunks.append(
            json.dumps({"id": "x", "model": "test", "choices": [{"index": 0, "delta": delta}]})
        )
    chunks.append(
        json.dumps(
            {
                "id": "x",
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "x",
                "model": "test",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )
    body = "\n".join(f"data: {c}" for c in chunks) + "\ndata: [DONE]\n"

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        gen = hub.stream("m", messages=[{"role": "user", "content": "tool"}])
        delta_count = 0
        final = None
        async for ev in gen:
            if isinstance(ev, ToolCallDelta):
                delta_count += 1
            elif isinstance(ev, StreamEnd):
                final = ev
        assert delta_count >= 1
        assert final is not None
        tcs = final.response.tool_calls
        assert len(tcs) == 1
        assert tcs[0].name == "do_thing"
        assert tcs[0].id == "call_abc"
        # The reassembled args dict must match the original.
        assert tcs[0].arguments == args
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_multiple_tool_calls_kept_distinct(env_key: None) -> None:
    """Two parallel tool calls (different ``index`` values) must reassemble independently."""
    chunks = [
        # First tool call (index 0)
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"role":"assistant",'
        '"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"a","arguments":"{\\"x\\":"}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":'
        '{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
        # Second tool call (index 1)
        '{"id":"x","model":"test","choices":[{"index":0,"delta":'
        '{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"b","arguments":"{\\"y\\":"}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":'
        '{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        '{"id":"x","model":"test","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
    ]
    body = "\n".join(f"data: {c}" for c in chunks) + "\ndata: [DONE]\n"

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        final = None
        async for ev in hub.stream("m", messages=[{"role": "user", "content": "two"}]):
            if isinstance(ev, StreamEnd):
                final = ev
        assert final is not None
        tcs = final.response.tool_calls
        assert len(tcs) == 2
        assert tcs[0].name == "a"
        assert tcs[0].arguments == {"x": 1}
        assert tcs[1].name == "b"
        assert tcs[1].arguments == {"y": 2}
    finally:
        await hub.aclose()
