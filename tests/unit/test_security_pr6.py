"""PR 6 security tests — provider-side tool-arg validation, audit-sink failure
logging, monotonic clock in circuit breaker, sync-callback warning.
"""

from __future__ import annotations

import logging
import warnings

import httpx
import pytest
import respx

from relay import Hub
from relay.audit import AuditEvent, CallbackSink
from relay.config import load_str
from relay.errors import ToolSchemaError
from relay.types import ToolDefinition


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
    """


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


# ---------------------------------------------------------------------------
# Provider tool-arg validation (AUTH-8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_args_validated_against_declaration(env_key: None) -> None:
    """A model that returns tool arguments violating the declared JSON Schema
    must surface as ToolSchemaError so callers can fail closed."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "create_issue",
                                        # Missing required ``title``, has unknown field.
                                        "arguments": '{"random": "garbage"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    hub = Hub.from_config(load_str(_yaml()))
    tool = ToolDefinition(
        name="create_issue",
        description="",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    )
    try:
        with pytest.raises(ToolSchemaError, match="do not match declared schema"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}], tools=[tool])
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# Audit-sink failure logging + strict_audit
# ---------------------------------------------------------------------------


class _BrokenSink:
    async def emit(self, event: AuditEvent) -> None:
        raise RuntimeError("sink is down")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
@respx.mock
async def test_audit_sink_failure_logged(
    env_key: None, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    import relay.audit as audit_mod

    starting_count = audit_mod.audit_sink_failures
    hub = Hub.from_config(load_str(_yaml()), audit_sinks=[_BrokenSink()])
    try:
        with caplog.at_level(logging.WARNING, logger="relay.audit"):
            r = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert r.text == "ok"  # call still succeeds — sink failure is non-fatal
        assert any("audit_sink_failed" in rec.message for rec in caplog.records)
        assert audit_mod.audit_sink_failures > starting_count
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_strict_audit_reraises_sink_failure(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    hub = Hub.from_config(
        load_str(_yaml()),
        audit_sinks=[_BrokenSink()],
        strict_audit=True,
    )
    try:
        with pytest.raises(RuntimeError, match="sink is down"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# CallbackSink sync-warning
# ---------------------------------------------------------------------------


def test_callback_sink_warns_on_sync() -> None:
    def sync_cb(event: AuditEvent) -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CallbackSink(sync_cb)
    assert any(
        issubclass(w.category, UserWarning) and "sync callback" in str(w.message)
        for w in caught
    )


def test_callback_sink_silent_on_async() -> None:
    async def async_cb(event: AuditEvent) -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CallbackSink(async_cb)
    assert not any(
        issubclass(w.category, UserWarning) and "sync callback" in str(w.message)
        for w in caught
    )


# ---------------------------------------------------------------------------
# Circuit breaker uses time.monotonic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_uses_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """If we step the wall clock backwards/forward, the breaker's cooldown
    semantics shouldn't change — monotonic is unaffected."""
    import time as time_mod

    from relay._internal.circuit_breaker import CircuitBreaker

    calls = {"time": 0, "monotonic": 0}
    real_monotonic = time_mod.monotonic

    def fake_time() -> float:
        calls["time"] += 1
        return 0.0  # frozen wall clock

    def fake_monotonic() -> float:
        calls["monotonic"] += 1
        return real_monotonic()

    monkeypatch.setattr(time_mod, "time", fake_time)
    monkeypatch.setattr(time_mod, "monotonic", fake_monotonic)

    breaker = CircuitBreaker(failure_threshold=2, window_s=10.0, cooldown_s=1.0)
    await breaker.before("k")
    await breaker.on_failure("k")
    await breaker.on_failure("k")
    # If on_failure used time.time, the breaker's view of "now" would not
    # advance and the cooldown comparison would behave incorrectly. With
    # monotonic in place, calls["monotonic"] must be non-zero and calls
    # to time.time must NOT have been consulted by the breaker itself.
    assert calls["monotonic"] > 0
    # Note: we can't assert calls["time"] == 0 because other code (httpx,
    # logging) hits time.time. We just verify the breaker invoked monotonic.
