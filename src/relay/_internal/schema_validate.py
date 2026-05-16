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


def _relax_for_response(schema: Any) -> Any:
    """Return a copy of ``schema`` with ``additionalProperties`` /
    ``unevaluatedProperties`` stripped recursively.

    Used for validating *response* tool-call args. The defense we care
    about is "required fields present, types right" — providers routinely
    add their own metadata fields (``__reasoning``, OpenAI strict-mode
    injected keys, vendor extensions), and a request-side schema with
    ``additionalProperties: false`` would otherwise reject every
    legitimate response. Request-side validation is unchanged.
    """
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for k, v in schema.items():
            if k in ("additionalProperties", "unevaluatedProperties"):
                continue
            out[k] = _relax_for_response(v)
        return out
    if isinstance(schema, list):
        return [_relax_for_response(v) for v in schema]
    return schema


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
    *,
    response_side: bool = False,
) -> None:
    """Validate ``arguments`` against the JSON Schema ``schema``.

    Raises :class:`ToolSchemaError` on failure. Only a literally empty
    schema (``{}``) is skipped — anything else (incl. ``allOf`` / ``enum``
    / ``$ref`` / ``additionalProperties`` / ``not`` schemas that the old
    fast-path silently waved through) goes through jsonschema. jsonschema
    is a hard runtime dependency so there's no import-cost reason to
    short-circuit on shape.

    ``response_side=True`` strips ``additionalProperties`` /
    ``unevaluatedProperties`` from the schema before validating: providers
    legitimately inject their own metadata fields and rejecting them via
    closed-schema enforcement would break working callers. Outbound
    request-side validation (MCP server dispatch) keeps the full schema.
    """
    if not schema:
        return

    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - jsonschema is a tight dep
        raise ToolSchemaError(
            "jsonschema is required to validate tool-call arguments; "
            "install ai5labs-relay with the default extras"
        ) from e

    effective = _relax_for_response(schema) if response_side else schema

    try:
        jsonschema.validate(instance=arguments, schema=effective)
    except jsonschema.ValidationError as e:
        raise ToolSchemaError(
            f"tool {tool_name!r} arguments do not match declared schema: {e.message}",
            raw={"arguments": arguments, "schema": schema, "error": str(e)},
        ) from e
