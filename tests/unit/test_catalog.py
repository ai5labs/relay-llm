"""Built-in catalog tests."""

from __future__ import annotations

from relay.catalog import get_catalog, lookup


def test_catalog_loads_some_rows() -> None:
    cat = get_catalog()
    assert len(cat) > 10, "catalog should ship with a non-trivial number of models"


def test_catalog_has_major_providers() -> None:
    cat = get_catalog()
    providers = {row.provider for row in cat.values()}
    assert {"openai", "anthropic", "google", "groq"} <= providers


def test_lookup_known_model() -> None:
    row = lookup("openai", "gpt-4o-mini")
    assert row is not None
    assert row.input_per_1m is not None
    assert row.output_per_1m is not None
    assert "tools" in row.capabilities


def test_lookup_unknown_returns_none() -> None:
    assert lookup("openai", "definitely-not-a-real-model") is None


def test_anthropic_models_have_thinking_capability() -> None:
    cat = get_catalog()
    sonnet = cat.get("anthropic/claude-sonnet-4-5")
    assert sonnet is not None
    assert "thinking" in sonnet.capabilities


def test_pricing_is_finite_when_present() -> None:
    """Catch obviously-wrong prices (negative, NaN, absurd values)."""
    for slug, row in get_catalog().items():
        if row.input_per_1m is not None:
            assert 0 <= row.input_per_1m <= 1000, f"{slug}: input_per_1m={row.input_per_1m}"
        if row.output_per_1m is not None:
            assert 0 <= row.output_per_1m <= 1000, f"{slug}: output_per_1m={row.output_per_1m}"
