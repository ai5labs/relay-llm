"""Reasoning-budget unification.

Three providers, three different APIs. We normalize on::

    reasoning="minimal" | "low" | "medium" | "high" | <int budget tokens>

Mapping
-------
=========  ==================  =====================  =====================
Relay      OpenAI              Anthropic              Gemini
=========  ==================  =====================  =====================
minimal    effort="minimal"    budget_tokens=1024     thinking_budget=1024
low        effort="low"        budget_tokens=4096     thinking_budget=4096
medium     effort="medium"     budget_tokens=16384    thinking_budget=16384
high       effort="high"       budget_tokens=24576    thinking_budget=24576
<int N>    effort=bucketed     budget_tokens=N        thinking_budget=N
=========  ==================  =====================  =====================

OpenAI's ``effort`` is the only string-only API; for integer Relay values we
bucket: <2048 → minimal, <8192 → low, <20000 → medium, else high.
"""

from __future__ import annotations

from typing import Any, Literal

ReasoningLevel = Literal["minimal", "low", "medium", "high"]
ReasoningSpec = ReasoningLevel | int

_LEVEL_TO_BUDGET: dict[str, int] = {
    "minimal": 1024,
    "low": 4096,
    "medium": 16384,
    "high": 24576,
}


def to_openai(spec: ReasoningSpec | None) -> dict[str, Any] | None:
    """Return an OpenAI ``reasoning`` body fragment, or ``None`` if no budget."""
    if spec is None:
        return None
    if isinstance(spec, str):
        return {"effort": spec}
    # Integer → bucket into one of the four levels.
    if spec < 2048:
        level = "minimal"
    elif spec < 8192:
        level = "low"
    elif spec < 20000:
        level = "medium"
    else:
        level = "high"
    return {"effort": level}


def to_anthropic(spec: ReasoningSpec | None) -> dict[str, Any] | None:
    """Return an Anthropic ``thinking`` body fragment, or ``None``."""
    if spec is None:
        return None
    budget = spec if isinstance(spec, int) else _LEVEL_TO_BUDGET[spec]
    return {"type": "enabled", "budget_tokens": budget}


def to_gemini(spec: ReasoningSpec | None) -> dict[str, Any] | None:
    """Return a Gemini ``thinking_config`` body fragment, or ``None``."""
    if spec is None:
        return None
    budget = spec if isinstance(spec, int) else _LEVEL_TO_BUDGET[spec]
    return {"thinking_budget": budget}


__all__ = [
    "ReasoningLevel",
    "ReasoningSpec",
    "to_anthropic",
    "to_gemini",
    "to_openai",
]
