"""Deterministic, constraint-based router.

Lifts the scoring logic from ``relay models recommend`` into a class
implementing the :class:`Router` Protocol. The router does **not** look at
message contents — it makes a pure constraint-driven choice from the
candidate set.

Output is deterministic for a given catalog + request: ranking ties are
broken by alias slug, so two calls with the same input always return the
same decision.
"""

from __future__ import annotations

from datetime import datetime, timezone

from relay.catalog._loader import CatalogRow, get_catalog

from ._errors import NoCandidatesError
from ._protocol import RouteDecision, RouteRequest


def _avg_cost(row: CatalogRow) -> float | None:
    return row.cost_per_1m_avg()


def _budget_threshold(budget: str | None) -> float | None:
    if budget == "cheap":
        return 1.0
    if budget == "balanced":
        return 10.0
    return None  # premium or unset → no bound


def _composite_score(row: CatalogRow) -> float:
    """Quality score used when no task hint is available.

    Falls back through quality_index → arena_elo (normalised) → 0.
    """
    b = row.benchmarks
    if b is None:
        return 0.0
    if b.quality_index is not None:
        return float(b.quality_index)
    if b.arena_elo is not None:
        # Map Elo into a 0-100ish range so it's comparable to quality_index.
        return max(0.0, (float(b.arena_elo) - 1000.0) / 5.0)
    return 0.0


class RuleBasedRouter:
    """Deterministic router driven by :class:`RouteConstraints` and the catalog.

    Construct with ``catalog=None`` (the default) to use the shipped built-in
    catalog, or pass a custom mapping to test against synthetic data.

    The router scores each candidate via composite quality, applies the
    constraint filters, sorts deterministically, and returns the top alias
    plus the next three runners-up as alternates. Confidence is set from
    the score margin between the #1 and #2 picks.
    """

    def __init__(self, catalog: dict[str, CatalogRow] | None = None) -> None:
        self._catalog: dict[str, CatalogRow] = catalog if catalog is not None else get_catalog()

    async def route(self, request: RouteRequest) -> RouteDecision:
        constraints = request.constraints
        excluded = set(constraints.exclude_models) if constraints else set()
        preferred = set(constraints.prefer_models) if constraints else set()
        needs = list(constraints.needs) if constraints else []

        # Resolve candidate aliases to catalog rows.
        # ``request.candidates == []`` is documented to mean "use the full
        # catalog": skip the alias filter, take every non-deprecated row.
        if request.candidates:
            pairs: list[tuple[str, CatalogRow]] = []
            for alias in request.candidates:
                if alias in excluded:
                    continue
                row = self._resolve_alias(alias)
                if row is None or row.deprecated:
                    continue
                pairs.append((alias, row))
        else:
            pairs = [
                (row.slug, row)
                for row in self._catalog.values()
                if not row.deprecated and row.slug not in excluded
            ]

        # Capability filter: every required capability must be present.
        if needs:
            pairs = [(a, r) for a, r in pairs if all(cap in r.capabilities for cap in needs)]

        # Budget filter (cheap/balanced have an upper bound on avg cost).
        if constraints is not None:
            threshold = _budget_threshold(constraints.budget)
            if threshold is not None:
                pairs = [
                    (a, r)
                    for a, r in pairs
                    if (_avg_cost(r) is not None and (_avg_cost(r) or 0.0) < threshold)
                ]

            # Hard cost cap, if set.
            if constraints.max_cost_per_1m is not None:
                cap = constraints.max_cost_per_1m
                pairs = [
                    (a, r)
                    for a, r in pairs
                    if (_avg_cost(r) is not None and (_avg_cost(r) or 0.0) <= cap)
                ]

            # Quality floor, if set.
            if constraints.min_quality_index is not None:
                floor = float(constraints.min_quality_index)
                pairs = [(a, r) for a, r in pairs if _composite_score(r) >= floor]

        if not pairs:
            raise NoCandidatesError(
                "no candidate models satisfied the routing constraints"
            )

        # Score each survivor; bias preferred aliases.
        scored: list[tuple[str, CatalogRow, float]] = []
        for alias, row in pairs:
            score = _composite_score(row)
            if alias in preferred:
                score += 5.0  # small bias, not a hard override
            scored.append((alias, row, score))

        # Deterministic ordering: descending score, then alias slug.
        scored.sort(key=lambda t: (-t[2], t[0]))

        winner_alias, _winner_row, winner_score = scored[0]
        runners = scored[1:4]

        # Confidence from score margin between #1 and #2. Single-candidate
        # case is treated as fully confident; otherwise normalise margin to
        # a 0-1 range with a soft cap.
        if len(scored) == 1:
            confidence = 1.0
        else:
            second_score = scored[1][2]
            margin = max(0.0, winner_score - second_score)
            # Margin of 20+ points → confidence 1.0; tiny margin → ~0.5.
            confidence = min(1.0, 0.5 + margin / 40.0)

        alternates: list[tuple[str, float]] = [(a, s) for a, _r, s in runners]

        reasoning = (
            f"composite quality score {winner_score:.1f}; "
            f"margin over #2: {(winner_score - scored[1][2]) if len(scored) > 1 else 0.0:.1f}"
        )

        return RouteDecision(
            alias=winner_alias,
            confidence=confidence,
            reasoning=reasoning,
            alternates=alternates,
            classified_intent=None,
            source="rule",
            ts=datetime.now(timezone.utc),
        )

    async def aclose(self) -> None:
        # No long-lived resources to release; the catalog is process-shared.
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _resolve_alias(self, alias: str) -> CatalogRow | None:
        """Resolve ``alias`` to a catalog row.

        Tries (in order) direct slug lookup (``provider/model_id``), then
        iterates the catalog matching ``aliases`` or ``model_id``. Returns
        ``None`` if nothing matches.
        """
        row = self._catalog.get(alias)
        if row is not None:
            return row
        for r in self._catalog.values():
            if alias in r.aliases or alias == r.model_id:
                return r
        return None


__all__ = ["RuleBasedRouter"]
