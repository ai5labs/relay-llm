"""Unit tests for the deterministic rule-based router."""

from __future__ import annotations

import pytest

from relay.catalog._loader import BenchmarkScores, CatalogRow
from relay.routing import (
    NoCandidatesError,
    RouteConstraints,
    RouteRequest,
    RuleBasedRouter,
)
from relay.types import Message


def _row(
    slug: str,
    *,
    quality: float | None = 70.0,
    avg_cost: float | None = 5.0,
    capabilities: tuple[str, ...] = ("tools",),
    deprecated: bool = False,
    aliases: tuple[str, ...] = (),
) -> CatalogRow:
    provider, model_id = slug.split("/", 1)
    if avg_cost is not None:
        # Set input/output so cost_per_1m_avg() returns avg_cost.
        input_per_1m = avg_cost
        output_per_1m = avg_cost
    else:
        input_per_1m = None
        output_per_1m = None
    return CatalogRow(
        provider=provider,
        model_id=model_id,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
        capabilities=capabilities,
        benchmarks=BenchmarkScores(quality_index=quality) if quality is not None else None,
        deprecated=deprecated,
        aliases=aliases,
    )


def _catalog(*rows: CatalogRow) -> dict[str, CatalogRow]:
    return {r.slug: r for r in rows}


@pytest.mark.asyncio
async def test_picks_highest_quality_within_candidates() -> None:
    cat = _catalog(
        _row("openai/a", quality=80.0),
        _row("openai/b", quality=60.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[Message(role="user", content="hi")],
            candidates=["openai/a", "openai/b"],
        )
    )
    assert decision.alias == "openai/a"
    assert decision.source == "rule"
    assert decision.alternates == [("openai/b", 60.0)]


@pytest.mark.asyncio
async def test_budget_cheap_filters_above_threshold() -> None:
    cat = _catalog(
        _row("openai/expensive", quality=90.0, avg_cost=20.0),
        _row("openai/affordable", quality=70.0, avg_cost=0.5),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/expensive", "openai/affordable"],
            constraints=RouteConstraints(budget="cheap"),
        )
    )
    # ``cheap`` cap is <$1 avg/1M, so the expensive row is filtered out.
    assert decision.alias == "openai/affordable"


@pytest.mark.asyncio
async def test_capability_filter_drops_models_missing_cap() -> None:
    cat = _catalog(
        _row("openai/text-only", capabilities=("tools",), quality=90.0),
        _row("openai/vision", capabilities=("tools", "vision"), quality=70.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/text-only", "openai/vision"],
            constraints=RouteConstraints(needs=["vision"]),
        )
    )
    assert decision.alias == "openai/vision"


@pytest.mark.asyncio
async def test_exclude_models_drops_alias() -> None:
    cat = _catalog(
        _row("openai/a", quality=80.0),
        _row("openai/b", quality=60.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/a", "openai/b"],
            constraints=RouteConstraints(exclude_models=["openai/a"]),
        )
    )
    assert decision.alias == "openai/b"


@pytest.mark.asyncio
async def test_prefer_models_biases_choice() -> None:
    cat = _catalog(
        _row("openai/best", quality=78.0),
        _row("openai/preferred", quality=75.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/best", "openai/preferred"],
            constraints=RouteConstraints(prefer_models=["openai/preferred"]),
        )
    )
    # +5.0 bias on preferred (75 + 5 = 80) overtakes best (78).
    assert decision.alias == "openai/preferred"


@pytest.mark.asyncio
async def test_no_survivors_raises_no_candidates() -> None:
    cat = _catalog(_row("openai/a", capabilities=("tools",), quality=80.0))
    router = RuleBasedRouter(catalog=cat)
    with pytest.raises(NoCandidatesError):
        await router.route(
            RouteRequest(
                messages=[],
                candidates=["openai/a"],
                constraints=RouteConstraints(needs=["vision"]),
            )
        )


@pytest.mark.asyncio
async def test_alternates_ordered_descending_by_score() -> None:
    cat = _catalog(
        _row("openai/a", quality=90.0),
        _row("openai/b", quality=70.0),
        _row("openai/c", quality=50.0),
        _row("openai/d", quality=30.0),
        _row("openai/e", quality=20.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/a", "openai/b", "openai/c", "openai/d", "openai/e"],
        )
    )
    assert decision.alias == "openai/a"
    # Alternates capped at 3, descending.
    assert decision.alternates == [
        ("openai/b", 70.0),
        ("openai/c", 50.0),
        ("openai/d", 30.0),
    ]


@pytest.mark.asyncio
async def test_deterministic_for_same_input() -> None:
    cat = _catalog(
        _row("openai/a", quality=80.0),
        _row("openai/b", quality=80.0),
        _row("openai/c", quality=70.0),
    )
    router = RuleBasedRouter(catalog=cat)
    req = RouteRequest(messages=[], candidates=["openai/a", "openai/b", "openai/c"])
    d1 = await router.route(req)
    d2 = await router.route(req)
    # Tie-break by alias string is stable.
    assert d1.alias == d2.alias == "openai/a"
    assert d1.alternates == d2.alternates


@pytest.mark.asyncio
async def test_score_margin_drives_confidence() -> None:
    """Big margin → high confidence; tight race → mid confidence."""
    big_margin = _catalog(
        _row("openai/winner", quality=95.0),
        _row("openai/loser", quality=20.0),
    )
    tight = _catalog(
        _row("openai/winner", quality=70.0),
        _row("openai/loser", quality=69.0),
    )

    big = await RuleBasedRouter(catalog=big_margin).route(
        RouteRequest(messages=[], candidates=["openai/winner", "openai/loser"])
    )
    close = await RuleBasedRouter(catalog=tight).route(
        RouteRequest(messages=[], candidates=["openai/winner", "openai/loser"])
    )
    assert big.confidence > close.confidence
    assert big.confidence == 1.0  # >= 20-pt margin → cap at 1.0
    assert 0.4 < close.confidence < 0.6


@pytest.mark.asyncio
async def test_empty_candidates_uses_full_catalog() -> None:
    cat = _catalog(
        _row("openai/a", quality=80.0),
        _row("openai/b", quality=60.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(messages=[], candidates=[]),
    )
    assert decision.alias == "openai/a"


@pytest.mark.asyncio
async def test_deprecated_rows_are_skipped() -> None:
    cat = _catalog(
        _row("openai/old", quality=99.0, deprecated=True),
        _row("openai/new", quality=70.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(messages=[], candidates=["openai/old", "openai/new"]),
    )
    assert decision.alias == "openai/new"


@pytest.mark.asyncio
async def test_max_cost_per_1m_caps_choice() -> None:
    cat = _catalog(
        _row("openai/expensive", quality=95.0, avg_cost=15.0),
        _row("openai/cheap", quality=70.0, avg_cost=2.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/expensive", "openai/cheap"],
            constraints=RouteConstraints(max_cost_per_1m=5.0),
        )
    )
    assert decision.alias == "openai/cheap"


@pytest.mark.asyncio
async def test_min_quality_index_filters_below_floor() -> None:
    cat = _catalog(
        _row("openai/low", quality=40.0),
        _row("openai/high", quality=85.0),
    )
    router = RuleBasedRouter(catalog=cat)
    decision = await router.route(
        RouteRequest(
            messages=[],
            candidates=["openai/low", "openai/high"],
            constraints=RouteConstraints(min_quality_index=70),
        )
    )
    assert decision.alias == "openai/high"
