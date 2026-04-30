"""Batch API tests — OpenAI Batch + Anthropic Message Batches."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str


@pytest.fixture
def env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-fake")


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          openai-m:
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
          anthropic-m:
            target: anthropic/claude-haiku-4-5
            credential: $env.ANTHROPIC_KEY
        """


# ---------------------------------------------------------------------------
# OpenAI Batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_openai_batch_submit_uploads_jsonl_then_creates_batch(env_keys: None) -> None:
    file_route = respx.post("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json={"id": "file_abc"})
    )
    batch_route = respx.post("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "batch_xyz",
                "status": "validating",
                "request_counts": {"total": 2, "completed": 0, "failed": 0},
            },
        )
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        handle = await hub.batch.submit(
            "openai-m",
            requests=[
                {"messages": [{"role": "user", "content": "q1"}]},
                {"messages": [{"role": "user", "content": "q2"}]},
            ],
        )
        assert handle.id == "batch_xyz"
        assert handle.provider == "openai"
        assert handle.request_count == 2
        assert file_route.called
        assert batch_route.called

        # The /batches body should reference the uploaded file_id and the chat endpoint.
        import orjson

        body = orjson.loads(batch_route.calls[0].request.content)
        assert body["input_file_id"] == "file_abc"
        assert body["endpoint"] == "/v1/chat/completions"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_openai_batch_status(env_keys: None) -> None:
    respx.get("https://api.openai.com/v1/batches/batch_xyz").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "batch_xyz",
                "status": "in_progress",
                "request_counts": {"total": 5, "completed": 2, "failed": 0},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        from relay.batch import BatchHandle

        handle = BatchHandle(
            id="batch_xyz",
            provider="openai",
            alias="openai-m",
            submitted_at=0.0,
            request_count=5,
        )
        prog = await hub.batch.status(handle)
        assert prog.status == "in_progress"
        assert prog.completed == 2
        assert prog.total == 5
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_openai_batch_results_parsed(env_keys: None) -> None:
    respx.get("https://api.openai.com/v1/batches/batch_xyz").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "batch_xyz",
                "status": "completed",
                "output_file_id": "file_out",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            },
        )
    )
    output_jsonl = (
        '{"custom_id":"req-0","response":{"body":{"id":"x","model":"gpt-4o-mini",'
        '"created":1,"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},'
        '"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}}\n'
    )
    respx.get("https://api.openai.com/v1/files/file_out/content").mock(
        return_value=httpx.Response(200, text=output_jsonl)
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        from relay.batch import BatchHandle

        handle = BatchHandle(
            id="batch_xyz",
            provider="openai",
            alias="openai-m",
            submitted_at=0.0,
            request_count=1,
        )
        results = await hub.batch.results(handle)
        assert len(results) == 1
        assert results[0].custom_id == "req-0"
        assert results[0].response is not None
        assert results[0].response.text == "hello"
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# Anthropic Message Batches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_batch_submit(env_keys: None) -> None:
    route = respx.post("https://api.anthropic.com/v1/messages/batches").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msgbatch_abc",
                "processing_status": "in_progress",
                "request_counts": {"processing": 2, "succeeded": 0},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        handle = await hub.batch.submit(
            "anthropic-m",
            requests=[
                {"messages": [{"role": "user", "content": "q1"}]},
                {"messages": [{"role": "user", "content": "q2"}]},
            ],
        )
        assert handle.id == "msgbatch_abc"
        assert handle.provider == "anthropic"
        assert route.called
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_batch_results(env_keys: None) -> None:
    output = (
        '{"custom_id":"req-0","result":{"type":"succeeded","message":{"id":"msg_1","model":"claude-haiku-4-5",'
        '"content":[{"type":"text","text":"howdy"}],"stop_reason":"end_turn",'
        '"usage":{"input_tokens":3,"output_tokens":2}}}}\n'
        '{"custom_id":"req-1","result":{"type":"errored","error":{"type":"overloaded_error"}}}\n'
    )
    respx.get("https://api.anthropic.com/v1/messages/batches/msgbatch_abc/results").mock(
        return_value=httpx.Response(200, text=output)
    )

    hub = Hub.from_config(load_str(_yaml()))
    try:
        from relay.batch import BatchHandle

        handle = BatchHandle(
            id="msgbatch_abc",
            provider="anthropic",
            alias="anthropic-m",
            submitted_at=0.0,
            request_count=2,
        )
        results = await hub.batch.results(handle)
        assert len(results) == 2
        assert results[0].response is not None
        assert results[0].response.text == "howdy"
        assert results[1].error is not None
        assert "overloaded" in results[1].error
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_batch_rejects_groups() -> None:
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          a:
            target: openai/gpt-4o-mini
        groups:
          g:
            strategy: fallback
            members: [a]
    """
    hub = Hub.from_config(load_str(yaml))
    try:
        from relay.errors import ConfigError

        with pytest.raises(ConfigError, match="not groups"):
            await hub.batch.submit("g", requests=[{"messages": [{"role": "user", "content": "x"}]}])
    finally:
        await hub.aclose()
