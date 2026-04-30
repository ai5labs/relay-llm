"""Routing strategies for groups.

Strategies
----------
* ``fallback`` — try members in order; on retryable failure move to the next.
* ``loadbalance`` — round-robin (we use random pick weighted by 1.0 for simplicity).
* ``weighted`` — random pick weighted by ``GroupMember.weight``.
* ``conditional`` — pick the first member whose ``when`` predicate matches.

Retry classification
--------------------
Errors are classified into three buckets:

* **transient** (RateLimitError, TimeoutError, network) — retry on the same
  member up to ``max_retries``, then fall back.
* **fall_back** (ContextWindowError, ContentPolicyError) — never retry on the
  same member; fall back immediately.
* **fatal** (AuthenticationError, ConfigError) — don't retry, don't fall back,
  raise immediately so the user sees the configuration problem.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from relay._internal.circuit_breaker import CircuitBreaker, CircuitOpenError
from relay.config._schema import GroupSpec, RoutingStrategy
from relay.errors import (
    AuthenticationError,
    ConfigError,
    ContentPolicyError,
    ContextWindowError,
    ProviderError,
    RateLimitError,
    RelayError,
    TimeoutError,
)


@dataclass
class RetryClass:
    transient: type[BaseException] | tuple[type[BaseException], ...]
    fall_back: type[BaseException] | tuple[type[BaseException], ...]
    fatal: type[BaseException] | tuple[type[BaseException], ...]


DEFAULT_RETRY = RetryClass(
    transient=(RateLimitError, TimeoutError, ProviderError, CircuitOpenError),
    fall_back=(ContextWindowError, ContentPolicyError),
    fatal=(AuthenticationError, ConfigError),
)


_DEFAULT_RNG = random.Random()  # noqa: S311 - load balancing, not cryptographic


def select_members(group: GroupSpec) -> list[tuple[str, float]]:
    """Return [(alias, weight), ...] honoring the strategy's selection rule."""
    pairs: list[tuple[str, float]] = []
    for m in group.members:
        # GroupSpec validator already normalized strings into GroupMember objects.
        pairs.append((m.name, m.weight))  # type: ignore[union-attr]
    return pairs


def order_by_strategy(
    pairs: list[tuple[str, float]],
    strategy: RoutingStrategy,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """Produce the call order for one request given the strategy."""
    if rng is None:
        rng = _DEFAULT_RNG
    if strategy == "fallback":
        return [name for name, _ in pairs]
    if strategy == "loadbalance":
        names = [name for name, _ in pairs]
        rng.shuffle(names)
        return names
    if strategy == "weighted":
        # Sample without replacement weighted by weight.
        remaining = list(pairs)
        out: list[str] = []
        while remaining:
            total = sum(w for _, w in remaining) or 1.0
            r = rng.uniform(0, total)
            acc = 0.0
            for i, (name, w) in enumerate(remaining):
                acc += w
                if r <= acc:
                    out.append(name)
                    remaining.pop(i)
                    break
            else:  # pragma: no cover - rounding edge
                out.append(remaining.pop()[0])
        return out
    if strategy == "conditional":
        # Conditional must be evaluated by the caller against request context;
        # here we just return the static order. Full conditional logic is v0.2.
        return [name for name, _ in pairs]
    raise ConfigError(f"unknown routing strategy: {strategy!r}")


async def with_retries(
    *,
    fn: Callable[[], Any],
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    retry_class: RetryClass = DEFAULT_RETRY,
) -> Any:
    """Run ``fn`` with classified retry semantics on a single target.

    On *fall_back* errors we re-raise immediately so the caller can pick a different
    target; on *transient* errors we back off; on *fatal* errors we re-raise.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except retry_class.fatal:
            raise
        except retry_class.fall_back:
            raise
        except retry_class.transient as e:
            if attempt >= max_retries:
                raise
            delay = min(max_backoff, initial_backoff * (2**attempt))
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = max(delay, float(e.retry_after))
            attempt += 1
            await asyncio.sleep(delay)


class _NoMembersAvailable(RelayError):
    pass


async def call_group(
    *,
    group: GroupSpec,
    call_one: Callable[[str], Any],
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    breaker: CircuitBreaker | None = None,
) -> Any:
    """Apply the group's strategy and call members until one succeeds.

    ``call_one`` is a callable taking a member alias and returning a coroutine.
    If ``breaker`` is given, members in the open state are skipped quickly and
    failures count toward opening the breaker.
    """
    order = order_by_strategy(select_members(group), group.strategy)
    if not order:
        raise _NoMembersAvailable("group has no members to call; configuration is invalid")

    last_exc: BaseException | None = None
    for alias in order:

        async def _bound_call(_alias: str = alias) -> Any:
            if breaker is not None:
                await breaker.before(_alias)
            try:
                result = await call_one(_alias)
            except DEFAULT_RETRY.transient:
                if breaker is not None:
                    await breaker.on_failure(_alias)
                raise
            except DEFAULT_RETRY.fall_back:
                # Don't penalize the breaker for ContextWindow / ContentPolicy —
                # they're request-shape problems, not provider health.
                raise
            else:
                if breaker is not None:
                    await breaker.on_success(_alias)
                return result

        try:
            return await with_retries(
                fn=_bound_call,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
            )
        except DEFAULT_RETRY.fatal:
            raise
        except DEFAULT_RETRY.fall_back as e:
            last_exc = e
            continue
        except DEFAULT_RETRY.transient as e:
            last_exc = e
            continue

    # Exhausted all members.
    if last_exc is not None:
        raise last_exc
    raise _NoMembersAvailable("all group members exhausted with no error captured")


def names_in_order(group: GroupSpec) -> Iterable[str]:
    """For diagnostics: return member names in declared order."""
    for m in group.members:
        yield m.name  # type: ignore[union-attr]
