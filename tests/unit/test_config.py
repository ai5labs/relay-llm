"""Config schema and loader tests."""

from __future__ import annotations

import os

import pytest

from relay.config import json_schema, load_str
from relay.errors import ConfigError


def test_minimal_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = load_str(
        """
        version: 1
        models:
          gpt:
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_API_KEY
        """
    )
    assert "gpt" in cfg.models
    assert cfg.models["gpt"].provider == "openai"
    assert cfg.models["gpt"].model_id == "gpt-4o-mini"


def test_missing_env_var_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(ConfigError, match="MISSING_KEY"):
        load_str(
            """
            version: 1
            models:
              x:
                target: openai/gpt-4o
                credential: $env.MISSING_KEY
            """
        )


def test_empty_env_var_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPTY_KEY", "")
    with pytest.raises(ConfigError, match="empty"):
        load_str(
            """
            version: 1
            models:
              x:
                target: openai/gpt-4o
                credential: $env.EMPTY_KEY
            """
        )


def test_target_must_have_provider_slash() -> None:
    with pytest.raises(ConfigError):
        load_str(
            """
            version: 1
            models:
              x:
                target: just-a-model-id
            """
        )


def test_group_with_unknown_member_fails() -> None:
    with pytest.raises(ConfigError, match="unknown alias"):
        load_str(
            """
            version: 1
            models:
              real:
                target: openai/gpt-4o
            groups:
              g:
                strategy: fallback
                members: [real, ghost]
            """
        )


def test_alias_collision_between_models_and_groups() -> None:
    with pytest.raises(ConfigError, match="appear in both"):
        load_str(
            """
            version: 1
            models:
              shared:
                target: openai/gpt-4o
              other:
                target: anthropic/claude-haiku-4-5
            groups:
              shared:
                strategy: fallback
                members: [other]
            """
        )


def test_group_member_string_normalizes_to_object() -> None:
    cfg = load_str(
        """
        version: 1
        models:
          a: { target: openai/gpt-4o-mini }
          b: { target: anthropic/claude-haiku-4-5 }
        groups:
          fast:
            strategy: weighted
            members:
              - a
              - { name: b, weight: 2.0 }
        """
    )
    members = cfg.groups["fast"].members
    assert members[0].name == "a"  # type: ignore[union-attr]
    assert members[0].weight == 1.0  # type: ignore[union-attr]
    assert members[1].weight == 2.0  # type: ignore[union-attr]


def test_pricing_profile_reference_validates() -> None:
    with pytest.raises(ConfigError, match="unknown pricing_profile"):
        load_str(
            """
            version: 1
            models:
              x:
                target: openai/gpt-4o
                pricing_profile: nonexistent
            """
        )


def test_json_schema_exports() -> None:
    schema = json_schema()
    assert schema["type"] == "object"
    assert "models" in schema["properties"]
    assert "groups" in schema["properties"]


def test_layered_config_overrides() -> None:
    """Smoke test for the layered loading behavior — full test in integration suite."""
    os.environ["TEST_KEY"] = "value"
    cfg = load_str(
        """
        version: 1
        models:
          x:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
        """
    )
    assert cfg.models["x"].credential == "value"
