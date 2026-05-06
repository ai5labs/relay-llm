"""Relay — one interface across every LLM provider.

Define your model catalog in a YAML file, then call any model by alias:

    >>> from relay import Hub
    >>> hub = Hub.from_yaml("models.yaml")
    >>> resp = await hub.chat("fast-cheap", messages=[{"role": "user", "content": "hi"}])
    >>> print(resp.text, resp.cost_usd)
"""

from relay._version import __version__
from relay.cache import Cache, CacheHint, MemoryCache
from relay.errors import (
    AuthenticationError,
    ConfigError,
    ContentPolicyError,
    ContextWindowError,
    ProviderError,
    RateLimitError,
    RelayError,
    TimeoutError,
    ToolSchemaError,
)
from relay.hub import Hub, Model
from relay.routing import (
    NoCandidatesError,
    RouteConstraints,
    RouteDecision,
    Router,
    RouterAuthError,
    RouteRequest,
    RouterQuotaError,
    RouterTimeoutError,
    RoutingError,
    RuleBasedRouter,
    SemanticRouter,
)
from relay.structured import request_structured
from relay.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    Cost,
    Message,
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)

__all__ = [
    "AuthenticationError",
    "Cache",
    "CacheHint",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "ConfigError",
    "ContentPolicyError",
    "ContextWindowError",
    "Cost",
    "Hub",
    "MemoryCache",
    "Message",
    "Model",
    "NoCandidatesError",
    "ProviderError",
    "RateLimitError",
    "RelayError",
    "RouteConstraints",
    "RouteDecision",
    "RouteRequest",
    "Router",
    "RouterAuthError",
    "RouterQuotaError",
    "RouterTimeoutError",
    "RoutingError",
    "RuleBasedRouter",
    "SemanticRouter",
    "StreamEvent",
    "ThinkingBlock",
    "TimeoutError",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolSchemaError",
    "Usage",
    "__version__",
    "request_structured",
]
