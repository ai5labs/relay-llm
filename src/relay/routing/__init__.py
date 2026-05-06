"""Public routing extension surface.

A :class:`Router` picks one alias per call from a candidate set, optionally
guided by :class:`RouteConstraints`. Two implementations ship in v0.2:

* :class:`RuleBasedRouter` — deterministic, constraint-driven, in-process.
  Lifts the same scoring logic that powers ``relay models recommend``.
* :class:`SemanticRouter` — HTTP client for the hosted semantic router
  (paid). Wire protocol documented at ``docs/routing/api-spec.md``.

Both can be plugged into :meth:`relay.Hub.attach_router`. Custom routers
that satisfy the :class:`Router` Protocol are also accepted.
"""

from relay.routing._errors import (
    NoCandidatesError,
    RouterAuthError,
    RouterQuotaError,
    RouterTimeoutError,
    RoutingError,
)
from relay.routing._protocol import (
    RouteConstraints,
    RouteDecision,
    Router,
    RouteRequest,
)
from relay.routing._rule_based import RuleBasedRouter
from relay.routing._semantic import SemanticRouter

__all__ = [
    "NoCandidatesError",
    "RouteConstraints",
    "RouteDecision",
    "RouteRequest",
    "Router",
    "RouterAuthError",
    "RouterQuotaError",
    "RouterTimeoutError",
    "RoutingError",
    "RuleBasedRouter",
    "SemanticRouter",
]
