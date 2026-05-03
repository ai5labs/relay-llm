"""Tests for `relay models compare` and `relay models recommend`."""

from __future__ import annotations

import json

from relay._cli import main


def test_compare_shows_pricing_and_benchmarks(capsys: object) -> None:
    rc = main(["models", "compare", "anthropic/claude-sonnet-4-5", "openai/gpt-4o"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    # Must show both rows side-by-side with key fields
    assert "anthropic/claude-sonnet-4-5" in out
    assert "openai/gpt-4o" in out
    assert "MMLU" in out
    assert "GPQA" in out
    assert "HumanEval" in out
    assert "$3.00" in out  # sonnet input price
    assert "$2.50" in out  # gpt-4o input price


def test_compare_resolves_aliases(capsys: object) -> None:
    rc = main(["models", "compare", "sonnet", "4o"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "anthropic/claude-sonnet-4-5" in out
    assert "openai/gpt-4o" in out


def test_compare_reports_unknown_slug(capsys: object) -> None:
    rc = main(["models", "compare", "definitely-not-a-real-model"])
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "unknown:" in err


def test_compare_unknown_among_known_still_succeeds(capsys: object) -> None:
    rc = main(["models", "compare", "sonnet", "ghost-model"])
    assert rc == 0  # at least one matched
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "anthropic/claude-sonnet-4-5" in out


def test_compare_json(capsys: object) -> None:
    rc = main(["models", "compare", "sonnet", "--json"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["model_id"] == "claude-sonnet-4-5"
    assert "benchmarks" in data[0]
    assert data[0]["benchmarks"]["mmlu"] == 88.7


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------


def test_recommend_basic_chat(capsys: object) -> None:
    rc = main(["models", "recommend", "--task", "chat", "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Top" in out
    # At least one well-known model should appear in the top-5 chat results
    assert any(name in out for name in ("opus", "sonnet", "gpt-4o", "gemini"))


def test_recommend_code_cheap_filters_by_budget(capsys: object) -> None:
    rc = main(["models", "recommend", "--task", "code", "--budget", "cheap", "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    # Cheap budget = avg cost < $1/M; opus ($45/M avg) must NOT appear
    assert "claude-opus-4-5" not in out


def test_recommend_with_needs_capability_filter(capsys: object) -> None:
    rc = main(["models", "recommend", "--task", "chat", "--needs", "vision", "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    # All recommendations must support vision; deepseek-chat (no vision) excluded
    assert "deepseek-chat" not in out


def test_recommend_provider_filter(capsys: object) -> None:
    rc = main(
        [
            "models",
            "recommend",
            "--task",
            "chat",
            "--providers",
            "anthropic",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    # Only anthropic models should appear in the recommendations
    lines = [
        line
        for line in out.splitlines()
        if line.strip() and line.strip()[0].isdigit() and "/" in line
    ]
    for line in lines:
        assert "anthropic/" in line, f"non-anthropic in: {line}"


def test_recommend_reasoning_premium_returns_top_models(capsys: object) -> None:
    rc = main(["models", "recommend", "--task", "reasoning", "--budget", "premium", "--limit", "3"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    # Top reasoning models should include at least one of the canonical thinkers
    assert any(name in out for name in ("o1", "o3-mini", "gemini-2.5-pro", "claude-opus"))


def test_recommend_json(capsys: object) -> None:
    rc = main(
        ["models", "recommend", "--task", "code", "--budget", "cheap", "--limit", "3", "--json"]
    )
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) <= 3
    for item in data:
        assert "slug" in item
        assert "score" in item


def test_recommend_no_match_returns_error(capsys: object) -> None:
    # Impossible filter — bedrock vision with cheap budget yields nothing
    # since no bedrock model with benchmarks is < $1/M avg
    rc = main(
        [
            "models",
            "recommend",
            "--needs",
            "thinking",
            "vision",
            "--providers",
            "groq",
            "--budget",
            "cheap",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "no models matched" in err
