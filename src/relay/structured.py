"""Structured output — Pydantic models in, validated instances out.

Compiles a Pydantic ``BaseModel`` (or a JSON Schema dict) into the right
provider-side shape:

* **OpenAI / OpenAI-compat** — ``response_format={"type":"json_schema", ...}``
  with ``strict: True`` when the schema is OpenAI-strict-compatible.
* **Anthropic** — single-tool trick. We register a synthetic tool whose
  ``input_schema`` is the target schema and force ``tool_choice={"type":"tool",
  "name": ...}``. The response's first ``tool_use`` block is the structured
  result.
* **Gemini** — ``response_schema`` + ``response_mime_type: "application/json"``.
* **Bedrock Converse** — same single-tool trick as Anthropic.
* **Other (Mistral / Cohere / etc.)** — fall back to ``json_object`` + post-call
  validation with retry-on-failure.

Surfacing
---------
On the response, ``resp.parsed`` returns the validated Pydantic instance (or the
raw dict when input was a schema). Validation failures are retried up to
``max_attempts`` times with the validation error included as a system message.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

import orjson
from pydantic import BaseModel, ValidationError

from relay.errors import ProviderError, RelayError, ToolSchemaError
from relay.types import ChatResponse, ToolDefinition

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RelayError):
    """Failed to produce a valid structured output after retries."""


def schema_for(target: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON Schema for the target — either a Pydantic class or a dict."""
    if isinstance(target, dict):
        return target
    if isinstance(target, type) and issubclass(target, BaseModel):
        return target.model_json_schema()
    raise StructuredOutputError(
        f"target must be a Pydantic BaseModel subclass or JSON Schema dict, got {type(target)}"
    )


def name_for(target: type[BaseModel] | dict[str, Any]) -> str:
    """Stable schema name used as the tool name on Anthropic / Bedrock."""
    if isinstance(target, dict):
        return target.get("title", "Output")
    return target.__name__


def build_request_overrides(
    target: type[BaseModel] | dict[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Return kwargs to merge into a ChatRequest to enable structured output for ``provider``.

    The returned dict may set ``response_format``, ``tools``, and/or
    ``tool_choice``. Caller merges into their request.
    """
    schema = schema_for(target)
    name = name_for(target)
    p = provider.lower()

    # OpenAI-compat family that supports json_schema strict mode.
    if p in {"openai", "azure", "groq", "deepseek", "xai", "fireworks", "openrouter"}:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": True,
                },
            }
        }

    # Anthropic / Bedrock — single-tool trick.
    if p in {"anthropic", "bedrock"}:
        synthetic = ToolDefinition(
            name=name,
            description=(schema.get("description") if isinstance(schema, dict) else None)
            or f"Return a {name} object.",
            parameters=schema,
        )
        return {
            "tools": [synthetic],
            "tool_choice": {"type": "tool", "name": name}
            if p == "anthropic"
            else {"tool": {"name": name}},
        }

    # Gemini — response_schema with JSON mime type.
    if p in {"google", "vertex"}:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema},
            }
        }

    # Fallback: JSON object mode + post-validation.
    return {"response_format": {"type": "json_object"}}


def parse_response(target: type[T] | dict[str, Any], resp: ChatResponse) -> T | dict[str, Any]:
    """Extract and validate the structured payload from a chat response.

    Looks at:
    1. The first ``tool_call`` if one matches our synthetic tool name.
    2. Otherwise the response text parsed as JSON.

    :raises StructuredOutputError: if no JSON or validation fails.
    """
    expected_name = name_for(target)

    # Tool-call path (Anthropic / Bedrock single-tool trick).
    for tc in resp.tool_calls:
        if tc.name == expected_name:
            payload = tc.arguments
            return _validate(target, payload)

    # Text path (OpenAI json_schema, Gemini, json_object fallback).
    text = resp.text.strip()
    if not text:
        raise StructuredOutputError(
            "response had no text and no matching tool call",
            provider=resp.provider,
            model=resp.provider_model,
        )
    try:
        payload = orjson.loads(text)
    except orjson.JSONDecodeError as e:
        # Some providers wrap JSON in markdown fences.
        cleaned = _strip_code_fence(text)
        try:
            payload = orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            raise StructuredOutputError(
                f"response was not valid JSON: {e}",
                provider=resp.provider,
                model=resp.provider_model,
                raw=text[:1000],
            ) from e
    return _validate(target, payload)


def _validate(target: type[T] | dict[str, Any], payload: dict[str, Any]) -> T | dict[str, Any]:
    if isinstance(target, dict):
        # No Pydantic model — just hand back the dict.
        return payload
    if isinstance(target, type) and issubclass(target, BaseModel):
        try:
            return target.model_validate(payload)
        except ValidationError as e:
            raise StructuredOutputError(
                f"response did not match {target.__name__}: {e}",
                raw=payload,
            ) from e
    raise StructuredOutputError(f"target must be Pydantic class or dict, got {type(target)}")


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers some providers emit."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence + optional language tag.
        first_newline = t.find("\n")
        if first_newline > 0:
            t = t[first_newline + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# ---------------------------------------------------------------------------
# Public convenience: parse_with_retry — caller-side helper that loops
# ---------------------------------------------------------------------------


async def request_structured(
    *,
    hub: Any,
    alias: str,
    schema: type[T] | dict[str, Any],
    messages: list[dict[str, Any]],
    max_attempts: int = 2,
    **request_kwargs: Any,
) -> T | dict[str, Any]:
    """Convenience wrapper: ``hub.chat`` + structured output + retry-on-validation-error.

    Use when you don't want to thread ``response_format`` and parsing yourself.
    Looks up the target alias to know which provider rules to apply.
    """
    entry = hub.config.models.get(alias)
    if entry is None:
        if alias in hub.config.groups:
            # For groups, pick the first member's provider as the compile target.
            first_member = hub.config.groups[alias].members[0]
            member_alias = first_member.name
            entry = hub.config.models[member_alias]
        else:
            raise StructuredOutputError(f"unknown alias: {alias!r}")

    overrides = build_request_overrides(schema, provider=entry.provider)
    merged_kwargs: dict[str, Any] = {**request_kwargs, **overrides}

    last_err: Exception | None = None
    history = list(messages)
    for _attempt in range(max_attempts):
        try:
            resp = await hub.chat(alias, messages=history, **merged_kwargs)
        except ProviderError as e:
            last_err = e
            continue
        except ToolSchemaError as e:
            # Hub._validate_response_tool_calls fired against the synthetic
            # tool we registered for the structured-output single-tool trick.
            # Treat the same as a parse failure: append a corrective turn and
            # retry, matching the StructuredOutputError branch below.
            last_err = e
            history = [
                *history,
                {
                    "role": "user",
                    "content": (
                        f"Your previous tool call did not match the declared "
                        f"schema: {e}. Return a tool call whose arguments "
                        f"validate against the schema exactly."
                    ),
                },
            ]
            continue
        try:
            return parse_response(schema, resp)
        except StructuredOutputError as e:
            last_err = e
            history = [
                *history,
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed schema validation: {e}. "
                        f"Return JSON matching the schema exactly."
                    ),
                },
            ]
    raise StructuredOutputError(
        f"failed to produce valid {name_for(schema)!r} after {max_attempts} attempts: {last_err}"
    )


__all__ = [
    "StructuredOutputError",
    "build_request_overrides",
    "name_for",
    "parse_response",
    "request_structured",
    "schema_for",
]


def _re_export_for_orjson() -> None:
    """orjson is imported above; this prevents 'unused' lint when the module is
    consumed by other modules that don't trigger orjson use."""
    _ = json
