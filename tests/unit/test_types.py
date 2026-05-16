"""Tests for relay.types.Message — specifically the content-coercion validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from relay.types import (
    ImageBlock,
    Message,
    TextBlock,
    ToolCall,
)


def test_message_content_none_coerces_to_empty_string() -> None:
    """OpenAI emits ``{"role":"assistant","content":null,"tool_calls":[...]}``
    on a tool-call turn. The validator must coerce ``None`` → ``""`` so the
    Message constructs cleanly."""
    m = Message(role="assistant", content=None)
    assert m.content == ""


def test_message_content_none_with_tool_calls_roundtrip() -> None:
    """The full assistant-tool-call shape that arrives from OpenAI."""
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "x"}}],
    }
    m = Message.model_validate(raw)
    assert m.content == ""
    # _Loose accepts extras, so tool_calls round-trips on the model dump.
    dumped = m.model_dump()
    assert dumped["content"] == ""
    assert dumped["tool_calls"] == [{"id": "call_1", "name": "lookup", "arguments": {"q": "x"}}]


def test_message_content_string_passthrough() -> None:
    """The validator is a no-op for plain strings — existing behavior unchanged."""
    m = Message(role="user", content="hello")
    assert m.content == "hello"


def test_message_content_empty_string_passthrough() -> None:
    """An explicit empty string is preserved, not re-coerced."""
    m = Message(role="assistant", content="")
    assert m.content == ""


def test_message_content_list_of_blocks_passthrough() -> None:
    """A list of typed blocks validates and is preserved — existing behavior."""
    blocks: list = [
        TextBlock(text="describe this"),
        ImageBlock(url="https://example/img.png", media_type="image/png"),
    ]
    m = Message(role="user", content=blocks)
    assert isinstance(m.content, list)
    assert len(m.content) == 2
    assert m.content[0].type == "text"
    assert m.content[1].type == "image"


def test_message_content_list_via_model_validate() -> None:
    """Round-trip a dict-shaped list of blocks (the wire form)."""
    raw = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image", "url": "data:image/png;base64,abc", "media_type": "image/png"},
        ],
    }
    m = Message.model_validate(raw)
    assert isinstance(m.content, list)
    assert m.content[0].text == "look at this"  # type: ignore[union-attr]


def test_message_content_invalid_type_still_rejects() -> None:
    """The validator only handles None; bogus inputs (int, dict, etc.) still
    raise ValidationError. We're not loosening the schema, only smoothing one
    well-defined coercion."""
    with pytest.raises(ValidationError):
        Message(role="user", content=42)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Message(role="user", content={"not": "a block"})  # type: ignore[arg-type]


def test_tool_call_dict_construction() -> None:
    """Sanity: ToolCall isn't affected by the content validator on Message."""
    tc = ToolCall(id="c1", name="x", arguments={"a": 1})
    assert tc.arguments == {"a": 1}
