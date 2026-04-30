"""Circuit breaker tests."""

from __future__ import annotations

import asyncio

import pytest

from relay._internal.circuit_breaker import CircuitBreaker, CircuitOpenError


@pytest.mark.asyncio
async def test_closed_passes_through() -> None:
    cb = CircuitBreaker(failure_threshold=2, window_s=10, cooldown_s=1)
    await cb.before("a")
    await cb.on_success("a")
    assert cb.state_for("a") == "closed"


@pytest.mark.asyncio
async def test_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3, window_s=10, cooldown_s=10)
    for _ in range(3):
        await cb.before("a")
        await cb.on_failure("a")
    assert cb.state_for("a") == "open"
    with pytest.raises(CircuitOpenError):
        await cb.before("a")


@pytest.mark.asyncio
async def test_half_open_after_cooldown() -> None:
    cb = CircuitBreaker(failure_threshold=2, window_s=10, cooldown_s=0.05)
    await cb.before("a")
    await cb.on_failure("a")
    await cb.before("a")
    await cb.on_failure("a")
    assert cb.state_for("a") == "open"

    await asyncio.sleep(0.1)
    # First call after cooldown enters half-open
    await cb.before("a")
    assert cb.state_for("a") == "half_open"
    await cb.on_success("a")
    assert cb.state_for("a") == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens() -> None:
    cb = CircuitBreaker(failure_threshold=2, window_s=10, cooldown_s=0.05)
    for _ in range(2):
        await cb.before("a")
        await cb.on_failure("a")
    await asyncio.sleep(0.1)
    await cb.before("a")  # half-open probe
    await cb.on_failure("a")  # probe failed
    assert cb.state_for("a") == "open"


@pytest.mark.asyncio
async def test_keys_independent() -> None:
    cb = CircuitBreaker(failure_threshold=2, window_s=10, cooldown_s=10)
    await cb.before("a")
    await cb.on_failure("a")
    await cb.before("a")
    await cb.on_failure("a")
    assert cb.state_for("a") == "open"
    # Other key unaffected
    await cb.before("b")
    assert cb.state_for("b") == "closed"


@pytest.mark.asyncio
async def test_failures_outside_window_drop() -> None:
    cb = CircuitBreaker(failure_threshold=3, window_s=0.05, cooldown_s=10)
    await cb.before("a")
    await cb.on_failure("a")
    await asyncio.sleep(0.1)
    # Old failure should not count
    await cb.before("a")
    await cb.on_failure("a")
    assert cb.state_for("a") == "closed"
