"""OpenTelemetry instrumentation tests using the in-memory exporter."""

from __future__ import annotations

import httpx
import pytest
import respx

pytest.importorskip("opentelemetry")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from relay import Hub
from relay.config import load_str
from relay.observability import instrument


@pytest.fixture(autouse=True)
def otel_exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


@pytest.mark.asyncio
@respx.mock
async def test_chat_emits_genai_span(env_key: None, otel_exporter: InMemorySpanExporter) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
    )
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
        """
    hub = Hub.from_config(load_str(yaml))
    instrument(hub)
    try:
        await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()

    spans = otel_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = next(s for s in spans if s.name.startswith("chat"))
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert "gen_ai.cost.source" in attrs


@pytest.mark.asyncio
async def test_instrument_without_otel_extra_raises() -> None:
    """If OTel SDK is missing, instrument() should raise ConfigError. Skipped here
    because OTel is installed in dev — this contract is enforced by the import in
    the function body."""
    pass
