"""Tool-call schema compiler tests."""

from __future__ import annotations

import pytest

from relay.errors import ToolSchemaError
from relay.tools import compile_for
from relay.types import ToolDefinition


def _t(strict: bool = False, **schema_kwargs: object) -> ToolDefinition:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "city name"},
        },
        "required": ["city"],
        **schema_kwargs,
    }
    return ToolDefinition(
        name="get_weather", description="Get weather", parameters=schema, strict=strict
    )


def test_openai_basic_compile() -> None:
    out = compile_for(_t(), "openai")
    assert out["type"] == "function"
    assert out["function"]["name"] == "get_weather"
    assert out["function"]["parameters"]["type"] == "object"


def test_openai_strict_adds_additional_properties_false() -> None:
    out = compile_for(_t(strict=True), "openai")
    schema = out["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["city"]
    assert out["function"]["strict"] is True


def test_openai_strict_strips_unsupported_keywords() -> None:
    tool = ToolDefinition(
        name="search",
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string", "minLength": 3, "maxLength": 100},
            },
            "required": ["q"],
        },
        strict=True,
    )
    out = compile_for(tool, "openai")
    props = out["function"]["parameters"]["properties"]["q"]
    assert "minLength" not in props
    assert "maxLength" not in props


def test_openai_strict_compile_raises_on_unsupported() -> None:
    tool = ToolDefinition(
        name="search",
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string", "minLength": 3},
            },
            "required": ["q"],
        },
        strict=True,
    )
    with pytest.raises(ToolSchemaError, match="minLength"):
        compile_for(tool, "openai", strict_compile=True)


def test_anthropic_compile() -> None:
    out = compile_for(_t(), "anthropic")
    assert out["name"] == "get_weather"
    assert "input_schema" in out
    assert "type" in out["input_schema"]


def test_anthropic_injects_unenforced_constraints_to_description() -> None:
    tool = ToolDefinition(
        name="search",
        description="search the web",
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string", "maxLength": 100, "pattern": "^[a-z]+$"},
            },
            "required": ["q"],
        },
    )
    out = compile_for(tool, "anthropic")
    desc = out["description"]
    # The Mastra trick — constraints baked into description.
    assert "maxLength" in desc
    assert "pattern" in desc


def test_gemini_strips_unsupported_keywords() -> None:
    tool = ToolDefinition(
        name="search",
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "filter": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            },
            "additionalProperties": False,
        },
    )
    out = compile_for(tool, "google")
    schema = out["parameters"]
    # Gemini doesn't support oneOf or additionalProperties.
    assert "additionalProperties" not in schema
    props = schema.get("properties") or {}
    assert "oneOf" not in (props.get("filter") or {})
    # But should be in description as instruction.
    assert "oneOf" in out["description"] or "additionalProperties" in out["description"]


def test_gemini_strict_compile_raises() -> None:
    tool = ToolDefinition(
        name="x",
        parameters={
            "type": "object",
            "properties": {"a": {"$ref": "#/definitions/A"}},
        },
    )
    with pytest.raises(ToolSchemaError, match=r"\$ref"):
        compile_for(tool, "google", strict_compile=True)


def test_bedrock_wraps_in_toolspec() -> None:
    out = compile_for(_t(), "bedrock")
    assert "toolSpec" in out
    assert out["toolSpec"]["inputSchema"]["json"]["type"] == "object"


def test_cohere_compiles_to_parameter_definitions() -> None:
    out = compile_for(_t(), "cohere")
    assert "parameter_definitions" in out
    pd = out["parameter_definitions"]["city"]
    assert pd["type"] == "str"
    assert pd["required"] is True


def test_unknown_provider_raises() -> None:
    with pytest.raises(ToolSchemaError):
        compile_for(_t(), "fictionalprovider")


def test_groq_uses_openai_compiler() -> None:
    """All OpenAI-compat providers should produce the same shape."""
    a = compile_for(_t(), "openai")
    b = compile_for(_t(), "groq")
    c = compile_for(_t(), "fireworks")
    assert a == b == c
