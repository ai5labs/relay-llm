"""Routing strategy + retry classification tests."""

from __future__ import annotations

import random

import pytest

from relay._internal.router import (
    DEFAULT_RETRY,
    call_group,
    order_by_strategy,
    select_members,
    with_retries,
)
from relay.config._schema import GroupMember, GroupSpec
from relay.errors import (
    AuthenticationError,
    ContextWindowError,
    RateLimitError,
)


def _group(strategy: str, *names: tuple[str, float]) -> GroupSpec:
    return GroupSpec(
        strategy=strategy,  # type: ignore[arg-type]
        members=[GroupMember(name=n, weight=w) for n, w in names],
    )


def test_fallback_preserves_order() -> None:
    g = _group("fallback", ("a", 1.0), ("b", 1.0), ("c", 1.0))
    assert order_by_strategy(select_members(g), "fallback") == ["a", "b", "c"]


def test_loadbalance_returns_a_permutation() -> None:
    g = _group("loadbalance", ("a", 1.0), ("b", 1.0), ("c", 1.0))
    rng = random.Random(42)
    out = order_by_strategy(select_members(g), "loadbalance", rng=rng)
    assert sorted(out) == ["a", "b", "c"]


def test_weighted_respects_distribution_in_aggregate() -> None:
    g = _group("weighted", ("a", 1.0), ("b", 9.0))
    rng = random.Random(0)
    first_picks: list[str] = []
    for _ in range(500):
        first_picks.append(order_by_strategy(select_members(g), "weighted", rng=rng)[0])
    # b should win significantly more often than a — weight 9:1.
    assert first_picks.count("b") > first_picks.count("a") * 4


@pytest.mark.asyncio
async def test_with_retries_retries_transient() -> None:
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("retry me", provider="x", model="m")
        return "ok"

    result = await with_retries(
        fn=fn,
        max_retries=5,
        initial_backoff=0.001,
        max_backoff=0.01,
    )
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_retries_does_not_retry_fall_back() -> None:
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        raise ContextWindowError("too big", provider="x", model="m")

    with pytest.raises(ContextWindowError):
        await with_retries(
            fn=fn,
            max_retries=5,
            initial_backoff=0.001,
            max_backoff=0.01,
        )
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_with_retries_does_not_retry_fatal() -> None:
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        raise AuthenticationError("bad key", provider="x", model="m")

    with pytest.raises(AuthenticationError):
        await with_retries(
            fn=fn,
            max_retries=5,
            initial_backoff=0.001,
            max_backoff=0.01,
        )
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_call_group_falls_back_on_context_window() -> None:
    g = _group("fallback", ("first", 1.0), ("second", 1.0))
    seen: list[str] = []

    async def call_one(alias: str) -> str:
        seen.append(alias)
        if alias == "first":
            raise ContextWindowError("nope", provider="x", model="m")
        return "ok"

    result = await call_group(
        group=g,
        call_one=call_one,
        max_retries=2,
        initial_backoff=0.001,
        max_backoff=0.01,
    )
    assert result == "ok"
    assert seen == ["first", "second"]


@pytest.mark.asyncio
async def test_call_group_propagates_fatal_immediately() -> None:
    g = _group("fallback", ("first", 1.0), ("second", 1.0))
    seen: list[str] = []

    async def call_one(alias: str) -> str:
        seen.append(alias)
        raise AuthenticationError("bad key", provider="x", model="m")

    with pytest.raises(AuthenticationError):
        await call_group(
            group=g,
            call_one=call_one,
            max_retries=2,
            initial_backoff=0.001,
            max_backoff=0.01,
        )
    # Only the first member should have been touched — fatal aborts immediately.
    assert seen == ["first"]


def test_default_retry_class_covers_expected_types() -> None:
    assert (
        RateLimitError in DEFAULT_RETRY.transient
        if isinstance(DEFAULT_RETRY.transient, tuple)
        else (RateLimitError == DEFAULT_RETRY.transient)
    )
    assert (
        ContextWindowError in DEFAULT_RETRY.fall_back
        if isinstance(DEFAULT_RETRY.fall_back, tuple)
        else (ContextWindowError == DEFAULT_RETRY.fall_back)
    )
    assert (
        AuthenticationError in DEFAULT_RETRY.fatal
        if isinstance(DEFAULT_RETRY.fatal, tuple)
        else (AuthenticationError == DEFAULT_RETRY.fatal)
    )
