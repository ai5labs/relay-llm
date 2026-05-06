"""Tests for the hosted SemanticRouter HTTP client."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay.routing import (
    RouteConstraints,
    RouterAuthError,
    RouteRequest,
    RouterQuotaError,
    RouterTimeoutError,
    RoutingError,
    SemanticRouter,
)
from relay.types import Message


@pytest.mark.asyncio
@respx.mock
async def test_success_returns_decision() -> None:
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        return_value=httpx.Response(
            200,
            json={
                "alias": "smart",
                "confidence": 0.87,
                "reasoning": "code task",
                "alternates": [["fast", 0.62]],
                "classified_intent": "code",
                "source": "hosted",
                "ts": "2026-05-06T12:34:56Z",
            },
        )
    )
    router = SemanticRouter(api_key="sk-test")
    try:
        decision = await router.route(
            RouteRequest(
                messages=[Message(role="user", content="hi")],
                candidates=["smart", "fast"],
                constraints=RouteConstraints(budget="balanced"),
            )
        )
    finally:
        await router.aclose()

    assert decision.alias == "smart"
    assert decision.confidence == pytest.approx(0.87)
    assert decision.alternates == [("fast", 0.62)]
    assert decision.classified_intent == "code"
    assert decision.source == "hosted"


@pytest.mark.asyncio
@respx.mock
async def test_401_raises_router_auth_error() -> None:
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        return_value=httpx.Response(
            401, json={"error_code": "AUTH_FAILED", "message": "bad key"}
        )
    )
    router = SemanticRouter(api_key="bad")
    try:
        with pytest.raises(RouterAuthError) as excinfo:
            await router.route(
                RouteRequest(messages=[], candidates=["smart"]),
            )
        assert excinfo.value.status_code == 401
    finally:
        await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_429_raises_router_quota_error() -> None:
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        return_value=httpx.Response(
            429, json={"error_code": "QUOTA_EXCEEDED", "message": "over limit"}
        )
    )
    router = SemanticRouter(api_key="ok")
    try:
        with pytest.raises(RouterQuotaError):
            await router.route(RouteRequest(messages=[], candidates=["smart"]))
    finally:
        await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_timeout_raises_router_timeout_error() -> None:
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )
    router = SemanticRouter(api_key="ok", timeout=1.0)
    try:
        with pytest.raises(RouterTimeoutError):
            await router.route(RouteRequest(messages=[], candidates=["smart"]))
    finally:
        await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_malformed_response_raises_routing_error() -> None:
    # 200 OK but body is not a valid decision (missing required fields).
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        return_value=httpx.Response(200, json={"not": "valid"})
    )
    router = SemanticRouter(api_key="ok")
    try:
        with pytest.raises(RoutingError):
            await router.route(RouteRequest(messages=[], candidates=["smart"]))
    finally:
        await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_500_raises_routing_error_with_status() -> None:
    respx.post("https://router.relay.ai5labs.com/v1/route").mock(
        return_value=httpx.Response(
            500, json={"error_code": "INTERNAL", "message": "boom"}
        )
    )
    router = SemanticRouter(api_key="ok")
    try:
        with pytest.raises(RoutingError) as excinfo:
            await router.route(RouteRequest(messages=[], candidates=["smart"]))
        assert excinfo.value.status_code == 500
        # 500 is not Auth/Quota/Timeout, so it stays as the base class.
        assert not isinstance(
            excinfo.value, (RouterAuthError, RouterQuotaError, RouterTimeoutError)
        )
    finally:
        await router.aclose()
