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


# ---------------------------------------------------------------------------
# PR 2: base_url SSRF + group-cycle + weight validators
# ---------------------------------------------------------------------------


def test_base_url_rejects_metadata_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """169.254.169.254 (cloud metadata) must be rejected without opt-in."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match=r"private/link-local|http://"):
        load_str(
            """
            version: 1
            models:
              evil:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
                base_url: http://169.254.169.254/latest
            """
        )


def test_base_url_rejects_rfc1918_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match="private/link-local"):
        load_str(
            """
            version: 1
            models:
              internal:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
                base_url: https://10.0.0.1/v1
            """
        )


def test_base_url_rejects_plain_http_for_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match="http://"):
        load_str(
            """
            version: 1
            models:
              evil:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
                base_url: http://attacker.example/v1
            """
        )


def test_base_url_allows_https_external(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = load_str(
        """
        version: 1
        models:
          ok:
            target: openai/gpt-4o
            credential: $env.OPENAI_API_KEY
            base_url: https://api.openai.com/v1
        """
    )
    assert cfg.models["ok"].base_url == "https://api.openai.com/v1"


def test_base_url_allows_loopback_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-hosted vLLM / Ollama on localhost is the legitimate http use case."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = load_str(
        """
        version: 1
        models:
          vllm:
            target: openai/llama-3
            credential: $env.OPENAI_API_KEY
            base_url: http://127.0.0.1:8000/v1
        """
    )
    assert cfg.models["vllm"].base_url == "http://127.0.0.1:8000/v1"


def test_base_url_allows_private_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = load_str(
        """
        version: 1
        models:
          internal:
            target: openai/gpt-4o
            credential: $env.OPENAI_API_KEY
            base_url: https://10.0.0.5/v1
            allow_private_hosts: true
        """
    )
    assert cfg.models["internal"].allow_private_hosts is True


def test_base_url_google_requires_https(monkeypatch: pytest.MonkeyPatch) -> None:
    """google/* targets must use https even on loopback — defense-in-depth
    against query-string-key style mistakes."""
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    with pytest.raises(ConfigError, match="google models require https"):
        load_str(
            """
            version: 1
            models:
              g:
                target: google/gemini-2.5-flash
                credential: $env.GOOGLE_API_KEY
                base_url: http://127.0.0.1:9000
            """
        )


def test_group_cycle_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match="cycle detected"):
        load_str(
            """
            version: 1
            models:
              m:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
            groups:
              A:
                members: [B]
              B:
                members: [A]
            """
        )


def test_group_no_cycle_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = load_str(
        """
        version: 1
        models:
          m:
            target: openai/gpt-4o
            credential: $env.OPENAI_API_KEY
        groups:
          A:
            members: [B]
          B:
            members: [m]
        """
    )
    assert "A" in cfg.groups


def test_weight_negative_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match="non-negative"):
        load_str(
            """
            version: 1
            models:
              m:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
            groups:
              g:
                members:
                  - {name: m, weight: -1.0}
            """
        )


def test_weight_infinity_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    with pytest.raises(ConfigError, match="finite"):
        load_str(
            """
            version: 1
            models:
              m:
                target: openai/gpt-4o
                credential: $env.OPENAI_API_KEY
            groups:
              g:
                members:
                  - {name: m, weight: .inf}
            """
        )
