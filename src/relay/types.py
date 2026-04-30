"""Public types for chat requests, responses, and streaming events.

Design notes
------------
* These are Pydantic v2 models — they form the user-facing API surface. Validation
  cost on a single request/response is negligible compared to the network call.
* On the *streaming* hot path we parse SSE frames into plain dicts with ``orjson``
  and only build a Pydantic model for the final aggregate. ``StreamEvent`` here
  is a discriminated union built once per assembled event, not per token.
* Provider-specific blocks (``thinking``, ``grounding``, ``citations``) are first-class
  rather than being flattened into OpenAI deltas — this is one of Relay's core
  value props vs LiteLLM/aisuite, which lose this information.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base for immutable response models — populated once, never mutated."""

    model_config = ConfigDict(frozen=True, extra="allow")


class _Loose(BaseModel):
    """Base for request/input models — accept extras so users can pass through
    provider-specific fields without us having to know about them ahead of time."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


class TextBlock(_Loose):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(_Loose):
    type: Literal["image"] = "image"
    url: str | None = None
    """Either a URL or base64-encoded data URI."""
    media_type: str | None = None
    """e.g. ``image/png`` — required for some providers when ``url`` is base64."""


class ThinkingBlock(_Loose):
    """Anthropic-style extended thinking. Preserved across providers that emit it.

    Other libraries flatten this into the regular text stream, which loses the
    distinction between visible response text and internal reasoning.
    """

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None
    """Anthropic's signed reasoning block — must be passed back unchanged for tool flows."""


class ToolUseBlock(_Loose):
    """Assistant requesting a tool call."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(_Loose):
    """User-side block returning a tool's output to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


class CacheHintBlock(_Loose):
    """Inline marker that the prior content block is cacheable.

    Insert into a :class:`Message`'s ``content`` list. Anthropic translates
    this into ``cache_control: {type: "ephemeral"}`` on the previous block;
    OpenAI ignores it (auto-cache happens implicitly); Gemini groups preceding
    content into a ``cachedContent`` resource (v0.2). See :class:`relay.cache.CacheHint`.
    """

    type: Literal["cache_hint"] = "cache_hint"
    ttl: str = "5m"


ContentBlock: TypeAlias = (
    TextBlock | ImageBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | CacheHintBlock
)


class Message(_Loose):
    """A single message in a chat conversation.

    ``content`` may be a plain string (the common case) or a list of typed blocks
    when the message includes images, tool calls, or thinking blocks.
    """

    role: Role
    content: str | list[ContentBlock]
    name: str | None = None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class ToolDefinition(_Loose):
    """A function-style tool the model can call.

    ``parameters`` is a JSON Schema object. Relay's tool-schema normalizer will
    translate it for the target provider (OpenAI ``strict`` mode, Anthropic
    Messages, Bedrock ``toolSpec``, Gemini ``functionDeclarations``).
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False


class ToolCall(_Frozen):
    """An assistant tool invocation, surfaced on the response."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(_Loose):
    """User-side result of a tool call, fed back into the next request."""

    tool_call_id: str
    content: str | list[ContentBlock]
    is_error: bool = False


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


class ChatRequest(_Loose):
    """A normalized chat request. Adapters translate this into provider-native shape."""

    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    response_format: Literal["text", "json_object"] | dict[str, Any] | None = None
    """Pass a JSON Schema dict for structured output."""
    seed: int | None = None
    stream: bool = False
    timeout: float | None = None
    reasoning: Literal["minimal", "low", "medium", "high"] | int | None = None
    """Unified reasoning budget across providers.

    * String levels (``"minimal"``, ``"low"``, ``"medium"``, ``"high"``) map to
      OpenAI's ``reasoning.effort``, and to integer token budgets on Anthropic /
      Gemini (1k / 4k / 16k / 24k respectively).
    * Integer values are exact token budgets for Anthropic ``thinking.budget_tokens``
      and Gemini ``thinking_config.thinking_budget``; bucketed for OpenAI.
    """
    metadata: dict[str, Any] | None = None
    """Free-form key/value bag forwarded to the provider's metadata field where supported."""

    thinking: dict[str, Any] | None = None
    """Anthropic extended thinking config.

    Example: ``{"type": "enabled", "budget_tokens": 8000}``.
    """


class Usage(_Frozen):
    """Token accounting. Field semantics are normalized across providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    """Tokens served from prompt cache (Anthropic) or cached input (OpenAI/Gemini)."""
    reasoning_tokens: int = 0
    """Internal thinking tokens, when the provider reports them separately."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


class Cost(_Frozen):
    """Cost breakdown with provenance.

    Other libraries return a single float and hide where it came from. Relay
    surfaces the source so callers can decide whether to trust it.
    """

    total_usd: float
    input_usd: float = 0.0
    output_usd: float = 0.0
    cached_input_usd: float = 0.0
    reasoning_usd: float = 0.0

    source: Literal[
        "live_api",  # AWS Pricing API, Azure Retail Prices, OpenRouter live
        "snapshot",  # nightly-refreshed JSON shipped with library
        "user_override",  # explicit cost in user YAML
        "estimated",  # heuristic when exact pricing unavailable
        "unknown",  # could not be priced
    ] = "snapshot"
    fetched_at: str | None = None  # ISO 8601 timestamp; None for built-in snapshot
    confidence: Literal["exact", "list_price", "estimated", "unknown"] = "list_price"


FinishReason: TypeAlias = Literal[
    "stop",  # model finished naturally
    "length",  # hit max_tokens
    "tool_calls",  # stopped to call tools
    "content_filter",
    "error",
    "other",
]


class Choice(_Frozen):
    """One sampled completion. Most calls return a single choice."""

    index: int = 0
    message: Message
    finish_reason: FinishReason | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: list[ThinkingBlock] = Field(default_factory=list)


class ChatResponse(_Frozen):
    """Final assembled chat response, whether from a single call or aggregated stream."""

    id: str
    model: str
    """The alias from the YAML config."""
    provider_model: str
    """The actual provider-side model id used to serve the request."""
    provider: str
    choices: list[Choice]
    usage: Usage
    cost: Cost | None = None
    created_at: float
    """Unix timestamp."""
    latency_ms: float
    raw: Any = None
    """The unmodified provider response. ``None`` unless ``include_raw=True``."""

    @property
    def text(self) -> str:
        """Convenience: concatenate all text blocks of the first choice."""
        if not self.choices:
            return ""
        msg = self.choices[0].message
        if isinstance(msg.content, str):
            return msg.content
        return "".join(b.text for b in msg.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.choices[0].tool_calls if self.choices else []

    @property
    def cost_usd(self) -> float | None:
        return self.cost.total_usd if self.cost is not None else None


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class StreamStart(_Frozen):
    type: Literal["start"] = "start"
    id: str
    model: str
    provider: str


class TextDelta(_Frozen):
    type: Literal["text_delta"] = "text_delta"
    index: int = 0
    text: str


class ThinkingDelta(_Frozen):
    """Incremental thinking text — preserved as a distinct event so callers can
    render it differently (collapsed by default, etc.)."""

    type: Literal["thinking_delta"] = "thinking_delta"
    index: int = 0
    text: str


class ToolCallDelta(_Frozen):
    """Streaming chunk of a tool call. ``arguments`` is a partial JSON string;
    callers should accumulate per ``index`` and parse on completion.

    This is the field that LiteLLM gets wrong — they key by ``id`` instead of
    ``index``, dropping ~90% of argument deltas. We always key by ``index``.
    """

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class UsageDelta(_Frozen):
    type: Literal["usage"] = "usage"
    usage: Usage


class StreamEnd(_Frozen):
    type: Literal["end"] = "end"
    finish_reason: FinishReason | None = None
    response: ChatResponse
    """Fully assembled response, including final usage and cost."""


class StreamErrorEvent(_Frozen):
    type: Literal["error"] = "error"
    error: str
    code: str | None = None


StreamEvent: TypeAlias = (
    StreamStart
    | TextDelta
    | ThinkingDelta
    | ToolCallDelta
    | UsageDelta
    | StreamEnd
    | StreamErrorEvent
)


# ---------------------------------------------------------------------------
# Embeddings (scaffold for v0.2)
# ---------------------------------------------------------------------------


class EmbeddingRequest(_Loose):
    input: str | list[str]
    dimensions: int | None = None


class EmbeddingResponse(_Frozen):
    id: str
    model: str
    provider: str
    embeddings: list[list[float]]
    usage: Usage
    cost: Cost | None = None


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "ContentBlock",
    "Cost",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FinishReason",
    "ImageBlock",
    "Message",
    "Role",
    "StreamEnd",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamStart",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolDefinition",
    "ToolResult",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    "UsageDelta",
]
