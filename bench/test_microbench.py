"""Microbenchmarks for Relay hot paths.

Run with: ``uv run pytest bench/ --benchmark-only``
"""

from __future__ import annotations

import json
from typing import Any

import orjson
import pytest

from relay.cache import cache_key
from relay.tools import compile_for
from relay.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    ToolDefinition,
    Usage,
)

pytest.importorskip("pytest_benchmark")


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def _make_sse_chunk(text: str) -> bytes:
    body = orjson.dumps(
        {
            "id": "x",
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": text}}],
        }
    )
    return b"data: " + body + b"\n"


def test_bench_sse_chunk_parse(benchmark: Any) -> None:
    """Parse 100 SSE chunks of ~1kb each."""
    payload = "x" * 1024
    chunk = _make_sse_chunk(payload)

    def parse_many() -> int:
        seen = 0
        for _ in range(100):
            line = chunk.decode("utf-8").rstrip("\n")
            assert line.startswith("data: ")
            obj = orjson.loads(line[6:])
            seen += len(obj["choices"][0]["delta"]["content"])
        return seen

    benchmark(parse_many)


# ---------------------------------------------------------------------------
# Tool-call merge by index
# ---------------------------------------------------------------------------


def test_bench_tool_call_merge(benchmark: Any) -> None:
    """Merge 50 argument-fragment deltas into a single tool call."""
    fragments = [json.dumps({"city": "Paris", "n": i})[i : i + 1] for i in range(50)]
    fragments[0] = '{"city": "Paris",'
    fragments[-1] = '"n": 42}'

    def merge() -> dict[str, str]:
        slot: dict[str, str] = {"id": "call_1", "name": "get_weather", "arguments": ""}
        for f in fragments:
            slot["arguments"] += f
        return slot

    benchmark(merge)


# ---------------------------------------------------------------------------
# Cache key hashing
# ---------------------------------------------------------------------------


def test_bench_cache_key(benchmark: Any) -> None:
    """Hash a 10-message chat request."""
    msgs = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}" * 10)
        for i in range(10)
    ]
    req = ChatRequest(messages=msgs, temperature=0.7, max_tokens=512)
    benchmark(lambda: cache_key("alias", req))


# ---------------------------------------------------------------------------
# Pydantic serialization
# ---------------------------------------------------------------------------


def test_bench_chatresponse_dump(benchmark: Any) -> None:
    """Serialize a ChatResponse to a dict — the cost users pay on every call."""
    resp = ChatResponse(
        id="x",
        model="alias",
        provider_model="m",
        provider="openai",
        choices=[Choice(message=Message(role="assistant", content="hello world " * 50))],
        usage=Usage(input_tokens=100, output_tokens=50),
        created_at=0.0,
        latency_ms=12.3,
    )
    benchmark(lambda: resp.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Tool schema compilation
# ---------------------------------------------------------------------------


@pytest.fixture
def tools() -> list[ToolDefinition]:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1, "maxLength": 64},
            "units": {"type": "string", "enum": ["c", "f"]},
            "options": {
                "type": "object",
                "properties": {
                    "include_humidity": {"type": "boolean"},
                    "history_days": {"type": "integer", "minimum": 0, "maximum": 30},
                },
            },
        },
        "required": ["city"],
    }
    return [
        ToolDefinition(name=f"tool_{i}", description=f"tool {i}", parameters=schema, strict=True)
        for i in range(5)
    ]


def test_bench_compile_openai(benchmark: Any, tools: list[ToolDefinition]) -> None:
    benchmark(lambda: [compile_for(t, "openai") for t in tools])


def test_bench_compile_anthropic(benchmark: Any, tools: list[ToolDefinition]) -> None:
    benchmark(lambda: [compile_for(t, "anthropic") for t in tools])


def test_bench_compile_gemini(benchmark: Any, tools: list[ToolDefinition]) -> None:
    benchmark(lambda: [compile_for(t, "google") for t in tools])
