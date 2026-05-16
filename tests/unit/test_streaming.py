"""Streaming tests including the OpenAI tool-call delta merge-by-index invariant.

This is the bug LiteLLM has — they merge by ``id`` instead of ``index``, dropping
~90% of argument deltas. We test that we *don't* drop them.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str
from relay.types import StreamEnd, TextDelta, ToolCallDelta


def _sse(*chunks: str) -> str:
    return "\n".join(f"data: {c}" for c in chunks) + "\ndata: [DONE]\n"


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/test
            credential: $env.TEST_KEY
        """


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


@pytest.mark.asyncio
@respx.mock
async def test_streaming_text_deltas_in_order(env_key: None) -> None:
    body = _sse(
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"hello"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":" world"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        '{"id":"x","model":"test","usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        events: list = []
        gen = hub.stream("m", messages=[{"role": "user", "content": "hi"}])
        async for ev in gen:
            events.append(ev)
        text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert text_deltas == ["hello", " world"]
        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        assert ends[0].response.text == "hello world"
        assert ends[0].response.usage.input_tokens == 3
        assert ends[0].response.usage.output_tokens == 2
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_deltas_merge_by_index_not_id(env_key: None) -> None:
    """Regression for the LiteLLM #20711 bug.

    First tool-call frame includes ``id`` and ``name``; subsequent frames carry
    only ``index`` + argument deltas. A correct implementation must accumulate
    by index — keying by id (which is missing on later frames) drops the deltas.
    """
    body = _sse(
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"role":"assistant",'
        '"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":'
        '{"name":"get_weather","arguments":""}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":'
        '{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city"}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":'
        '{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\"Paris\\"}"}}]}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        '{"id":"x","model":"test","usage":{"prompt_tokens":4,"completion_tokens":8,"total_tokens":12}}',
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        events: list = []
        gen = hub.stream("m", messages=[{"role": "user", "content": "hi"}])
        async for ev in gen:
            events.append(ev)
        tc_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
        # First frame announces the tool call; the next two carry argument deltas.
        assert len(tc_deltas) >= 3
        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        tcs = ends[0].response.tool_calls
        assert len(tcs) == 1
        assert tcs[0].name == "get_weather"
        assert tcs[0].arguments == {"city": "Paris"}
        assert ends[0].response.choices[0].finish_reason == "tool_calls"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_malformed_chunk_is_skipped_not_fatal(env_key: None) -> None:
    body = _sse(
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"a"}}]}',
        "{not valid json",  # garbage frame — should be skipped
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"stop"}]}',
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        events: list = []
        gen = hub.stream("m", messages=[{"role": "user", "content": "hi"}])
        async for ev in gen:
            events.append(ev)
        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        assert ends[0].response.text == "ab"
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# PR 3: streaming guardrails + overall deadline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_stream_runs_redactor(env_key: None) -> None:
    """Redactor must mutate the outbound prompt before the provider sees it."""
    from relay.redaction import RegexRedactor

    captured: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        body = _sse(
            '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)

    redactor = RegexRedactor()
    hub = Hub.from_config(load_str(_yaml()), redactor=redactor)
    try:
        events: list = []
        async for ev in hub.stream(
            "m", messages=[{"role": "user", "content": "my SSN is 111-11-1111"}]
        ):
            events.append(ev)
        assert "111-11-1111" not in captured["body"]
        assert "REDACTED" in captured["body"]
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_stream_pre_guardrail_blocks(env_key: None) -> None:
    """A pre-call guardrail must block streaming BEFORE the SSE socket opens."""
    from relay.guardrails import BlockedKeywords, GuardrailError

    hub = Hub.from_config(
        load_str(_yaml()),
        guardrails=[BlockedKeywords(["password"], check_response=False)],
    )
    try:
        with pytest.raises(GuardrailError, match="blocked term"):
            async for _ev in hub.stream(
                "m", messages=[{"role": "user", "content": "my password is hunter2"}]
            ):
                pass
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_stream_post_guardrail_replaces_buffer(env_key: None) -> None:
    """When post-guardrails fire, the final StreamEnd must not carry the blocked text."""
    from relay.guardrails import BlockedKeywords
    from relay.types import StreamErrorEvent

    body = _sse(
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"the secret is"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":" hunter2"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    hub = Hub.from_config(
        load_str(_yaml()),
        guardrails=[BlockedKeywords(["hunter2"], check_response=True)],
    )
    try:
        events: list = []
        async for ev in hub.stream("m", messages=[{"role": "user", "content": "hi"}]):
            events.append(ev)
        errors = [e for e in events if isinstance(e, StreamErrorEvent)]
        assert errors
        assert "hunter2" in errors[0].error
        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        # Buffered text on the final response must be replaced with a block marker.
        assert "hunter2" not in ends[0].response.text
        assert "blocked" in ends[0].response.text.lower()
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_stream_overall_deadline_kills_hang(env_key: None) -> None:
    """A slow-loris provider must be killed by the wall-clock deadline."""
    import asyncio
    from collections.abc import AsyncIterator
    from typing import Any

    from relay.errors import TimeoutError as RelayTimeoutError
    from relay.providers._base import BaseProvider
    from relay.types import StreamEvent, StreamStart, TextDelta

    class _SlowProvider(BaseProvider):
        name = "openai"

        async def chat(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
            raise NotImplementedError

        async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
            yield StreamStart(id="x", model="m", provider="openai")
            for _ in range(100):
                await asyncio.sleep(0.2)
                yield TextDelta(text="byte")

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        defaults:
          stream_overall_timeout: 0.3
        models:
          m:
            target: openai/test
            credential: $env.TEST_KEY
    """
    hub = Hub.from_config(load_str(yaml))
    hub._providers["openai"] = _SlowProvider()
    try:
        with pytest.raises(RelayTimeoutError, match="exceeded overall deadline"):
            async for _ev in hub.stream("m", messages=[{"role": "user", "content": "hi"}]):
                pass
    finally:
        await hub.aclose()
