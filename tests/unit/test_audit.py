"""Audit logging tests."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from relay import Hub
from relay.audit import AuditEvent, CallbackSink, FileSink, build_event
from relay.config import load_str
from relay.types import Message


def test_build_event_metadata_only_omits_text() -> None:
    msgs = [Message(role="user", content="email is alice@example.com")]
    ev = build_event(
        operation="chat",
        alias="m",
        provider="openai",
        model_id="gpt-4o-mini",
        messages=msgs,
        response=None,
        error=None,
        duration_ms=42.0,
        capture_messages="metadata_only",
    )
    assert ev.messages_full is None
    assert ev.messages_summary[0]["role"] == "user"
    assert "alice@example.com" not in str(ev)


def test_build_event_full_includes_text() -> None:
    msgs = [Message(role="user", content="hello")]
    ev = build_event(
        operation="chat",
        alias="m",
        provider="openai",
        model_id="gpt-4o-mini",
        messages=msgs,
        response=None,
        error=None,
        duration_ms=42.0,
        capture_messages="full",
    )
    assert ev.messages_full is not None
    assert ev.messages_full[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_file_sink_writes_jsonl(tmp_path: Any) -> None:
    path = tmp_path / "audit.log"
    sink = FileSink(path)
    ev = build_event(
        operation="chat",
        alias="x",
        provider="openai",
        model_id="m",
        messages=[Message(role="user", content="hi")],
        response=None,
        error=None,
        duration_ms=1.0,
        capture_messages="metadata_only",
    )
    await sink.emit(ev)
    await sink.emit(ev)
    text = path.read_text()
    assert text.count("\n") == 2


@pytest.mark.asyncio
@respx.mock
async def test_hub_emits_audit_event_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o-mini",
                "created": 1,
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
    captured: list[AuditEvent] = []

    async def cb(ev: AuditEvent) -> None:
        captured.append(ev)

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(load_str(yaml), audit_sinks=[CallbackSink(cb)])
    try:
        await hub.chat(
            "m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_id": "user-42"},
        )
        assert len(captured) == 1
        ev = captured[0]
        assert ev.operation == "chat"
        assert ev.user_id == "user-42"
        assert ev.input_tokens == 5
        assert ev.output_tokens == 2
        assert ev.error_type is None
        assert ev.cost_source == "snapshot"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_hub_emits_audit_event_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    captured: list[AuditEvent] = []

    async def cb(ev: AuditEvent) -> None:
        captured.append(ev)

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(load_str(yaml), audit_sinks=[CallbackSink(cb)])
    try:
        from relay.errors import AuthenticationError

        with pytest.raises(AuthenticationError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert len(captured) == 1
        assert captured[0].error_type == "AuthenticationError"
    finally:
        await hub.aclose()
