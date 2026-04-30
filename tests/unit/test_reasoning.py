"""Reasoning-budget unification tests."""

from __future__ import annotations

import pytest

from relay._internal.reasoning import to_anthropic, to_gemini, to_openai


@pytest.mark.parametrize(
    ("level", "expected_effort"),
    [("minimal", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_openai_string_levels(level: str, expected_effort: str) -> None:
    assert to_openai(level) == {"effort": expected_effort}  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("budget", "expected_effort"),
    [(0, "minimal"), (1024, "minimal"), (4096, "low"), (16384, "medium"), (30000, "high")],
)
def test_openai_integer_buckets(budget: int, expected_effort: str) -> None:
    assert to_openai(budget) == {"effort": expected_effort}


@pytest.mark.parametrize(
    ("level", "expected_budget"),
    [("minimal", 1024), ("low", 4096), ("medium", 16384), ("high", 24576)],
)
def test_anthropic_string_levels(level: str, expected_budget: int) -> None:
    out = to_anthropic(level)  # type: ignore[arg-type]
    assert out == {"type": "enabled", "budget_tokens": expected_budget}


def test_anthropic_integer_passes_through() -> None:
    assert to_anthropic(8192) == {"type": "enabled", "budget_tokens": 8192}


def test_gemini_string_levels() -> None:
    assert to_gemini("medium") == {"thinking_budget": 16384}


def test_gemini_integer_passes_through() -> None:
    assert to_gemini(2000) == {"thinking_budget": 2000}


def test_none_returns_none() -> None:
    assert to_openai(None) is None
    assert to_anthropic(None) is None
    assert to_gemini(None) is None
