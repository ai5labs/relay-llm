"""Shared JSON Schema validation for tool-call arguments.

Used by:
* :mod:`relay.mcp._manager` to validate the LLM's arguments before dispatching
  to an MCP server (CVE-2026-30623 mitigation companion).
* Provider parsers to validate ``ToolCall.arguments`` returned by the model
  against the originating :class:`relay.types.ToolDefinition.parameters`.

A schema violation is surfaced as :class:`relay.errors.ToolSchemaError` so
callers can fail closed rather than forwarding malformed payloads.
"""

from __future__ import annotations

from typing import Any

from relay.errors import ToolSchemaError


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Validate ``arguments`` against the JSON Schema ``schema``.

    Raises :class:`ToolSchemaError` on failure. A trivial / empty schema
    (``{}`` or one with no ``properties`` and no ``type``) is treated as
    "accept anything" and short-circuits without importing jsonschema.
    """
    # Cheap fast-path for the no-op case so we don't import jsonschema unless
    # there's actually a schema to enforce.
    if not schema or (
        "properties" not in schema
        and "type" not in schema
        and "required" not in schema
        and "anyOf" not in schema
        and "oneOf" not in schema
    ):
        return

    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - jsonschema is a tight dep
        raise ToolSchemaError(
            "jsonschema is required to validate tool-call arguments; "
            "install ai5labs-relay with the default extras"
        ) from e

    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as e:
        raise ToolSchemaError(
            f"tool {tool_name!r} arguments do not match declared schema: {e.message}",
            raw={"arguments": arguments, "schema": schema, "error": str(e)},
        ) from e
