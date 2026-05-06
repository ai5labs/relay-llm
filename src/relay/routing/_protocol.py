"""Routing protocol and shared data types.

These types form the public extension surface for picking a model per call.
A :class:`Router` consumes a :class:`RouteRequest` and returns a
:class:`RouteDecision` naming one alias plus runner-up alternates.

Two implementations ship with v0.2:

* :class:`relay.routing.RuleBasedRouter` — deterministic, constraint-based
  ranking against the built-in catalog. Free, OSS, in-process.
* :class:`relay.routing.SemanticRouter` — HTTP client for the hosted
  semantic router (paid). Wire protocol documented at
  ``docs/routing/api-spec.md``.

Custom routers implement the :class:`Router` Protocol — any class with
matching ``async def route`` and ``async def aclose`` methods is accepted
by :meth:`relay.Hub.attach_router`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from relay.types import Message

Budget = Literal["cheap", "balanced", "premium"]
"""Budget tier shorthand. ``cheap`` ≈ <$1 average per 1M tokens, ``balanced``
≈ <$10 per 1M, ``premium`` = no upper bound."""

DecisionSource = Literal["rule", "hosted", "self-hosted"]
"""Where a decision originated. ``rule`` is the built-in deterministic router,
``hosted`` is the ai5labs hosted semantic router, ``self-hosted`` is a custom
deployment of the same wire protocol."""


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class RouteConstraints(_Loose):
    """Filters and biases applied to candidate models before scoring.

    All fields are optional. Unset fields impose no constraint.
    """

    budget: Budget | None = None
    """Cost tier: ``cheap`` (<$1 avg/1M), ``balanced`` (<$10 avg/1M),
    ``premium`` (no upper bound)."""

    needs: list[str] = Field(default_factory=list)
    """Required capability strings — e.g. ``["tools", "vision", "thinking"]``.
    All must be present on the candidate row."""

    exclude_models: list[str] = Field(default_factory=list)
    """Alias names to drop from the candidate set."""

    prefer_models: list[str] = Field(default_factory=list)
    """Alias names to bias toward — adds a small score boost without filtering."""

    max_cost_per_1m: float | None = None
    """Maximum acceptable average cost per 1M tokens, in USD."""

    min_quality_index: int | None = None
    """Minimum acceptable composite quality index (0-100)."""


class RouteRequest(_Loose):
    """Input to :meth:`Router.route`.

    ``candidates`` lists the aliases the caller is willing to dispatch to.
    Routers may use ``messages`` to classify intent (semantic routers) or
    ignore them entirely (rule-based routers).
    """

    messages: list[Message]
    candidates: list[str]
    """Alias names from the configured catalog to choose from. An empty list
    is interpreted as 'pick from the full built-in catalog' by routers that
    support it; otherwise routers raise :class:`NoCandidatesError`."""

    constraints: RouteConstraints | None = None
    metadata: dict[str, str] | None = None
    """Free-form key/value bag forwarded to the routing service for tracing."""


class RouteDecision(_Frozen):
    """Output of :meth:`Router.route`.

    ``alias`` is the chosen model. ``alternates`` are the next-best options
    in descending preference, suitable for fallback when the primary errors.
    """

    alias: str
    confidence: float = Field(ge=0.0, le=1.0)
    """Heuristic confidence in the choice. Rule-based routers compute this
    from the score margin between #1 and #2; hosted routers may use a
    classifier probability."""

    reasoning: str | None = None
    """Optional human-readable explanation of why this alias was chosen."""

    alternates: list[tuple[str, float]] = Field(default_factory=list)
    """Runner-up ``(alias, score)`` pairs, sorted by descending score."""

    classified_intent: str | None = None
    """Intent label assigned by the router — e.g. ``"code"``, ``"reasoning"``,
    ``"chat"``, ``"math"``, ``"vision"``. ``None`` for routers that don't
    classify."""

    source: DecisionSource
    ts: datetime


@runtime_checkable
class Router(Protocol):
    """The routing extension point.

    Implement :meth:`route` to return a :class:`RouteDecision` naming the
    chosen alias plus alternates. ``aclose`` is provided to release any
    transport resources (HTTP clients, subprocess handles, etc.); routers
    with no resources to free should still implement it as a no-op.
    """

    async def route(self, request: RouteRequest) -> RouteDecision: ...

    async def aclose(self) -> None: ...


__all__ = [
    "Budget",
    "DecisionSource",
    "RouteConstraints",
    "RouteDecision",
    "RouteRequest",
    "Router",
]
