"""Hub.attach_router + Hub.chat_routed integration tests."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from relay import Hub
from relay.errors import ConfigError
from relay.routing import RouteDecision, RouteRequest


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          primary:
            target: openai/test-model
            credential: $env.TEST_KEY
          backup:
            target: openai/test-model-2
            credential: $env.TEST_KEY
        """


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


class _StaticRouter:
    """Test double — always returns the configured decision."""

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls: list[RouteRequest] = []

    async def route(self, request: RouteRequest) -> RouteDecision:
        self.calls.append(request)
        return self.decision

    async def aclose(self) -> None:
        return None


def _ok_response(model: str = "test-model") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model,
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
    )


@pytest.mark.asyncio
async def test_chat_routed_without_attached_router_raises_config_error(
    env_key: None,
) -> None:
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(ConfigError, match="attach_router"):
            await hub.chat_routed(
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_chat_routed_dispatches_to_chosen_alias(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    decision = RouteDecision(
        alias="primary",
        confidence=0.9,
        source="rule",
        ts=datetime.now(timezone.utc),
    )
    hub.attach_router(_StaticRouter(decision))
    try:
        resp = await hub.chat_routed(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.text == "ok"
        # Routing decision is stamped on the response metadata.
        meta = resp.metadata or {}
        assert "routing" in meta
        assert meta["routing"].alias == "primary"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_chat_routed_falls_back_through_alternates(env_key: None) -> None:
    # First call (primary) errors with 500; second call (backup) succeeds.
    route = respx.post("https://api.openai.com/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500, json={"error": {"message": "boom"}}),
        _ok_response("test-model-2"),
    ]
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    decision = RouteDecision(
        alias="primary",
        confidence=0.6,
        source="rule",
        alternates=[("backup", 0.4)],
        ts=datetime.now(timezone.utc),
    )
    hub.attach_router(_StaticRouter(decision))
    try:
        resp = await hub.chat_routed(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.text == "ok"
        # Decision still recorded, even though we used the alternate.
        meta = resp.metadata or {}
        assert meta["routing"].alias == "primary"
        assert meta["routing"].alternates == [("backup", 0.4)]
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_chat_routed_default_candidates_are_yaml_aliases(env_key: None) -> None:
    """When candidates=None, the hub should pass all configured aliases to the router."""
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    captured: list[RouteRequest] = []

    class _CaptureRouter:
        async def route(self, request: RouteRequest) -> RouteDecision:
            captured.append(request)
            return RouteDecision(
                alias="primary",
                confidence=1.0,
                source="rule",
                ts=datetime.now(timezone.utc),
            )

        async def aclose(self) -> None:
            return None

    hub.attach_router(_CaptureRouter())
    try:
        # Don't actually dispatch — we just want to inspect what was sent to the router.
        # Mock the OpenAI endpoint so the dispatch succeeds.
        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=_ok_response()
            )
            await hub.chat_routed(messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()

    assert len(captured) == 1
    assert sorted(captured[0].candidates) == ["backup", "primary"]
