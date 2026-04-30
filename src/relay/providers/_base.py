"""Provider protocol and shared base.

Every provider adapter exposes the same surface: ``chat``, ``stream``, ``embed``.
Adapters translate the normalized :class:`ChatRequest` into the provider's native
wire format, parse the response back into Relay types, and surface errors as the
appropriate :class:`RelayError` subclass.

Design choices
--------------
* **Lazy SDK import.** Bedrock/Vertex adapters import their cloud SDK only on
  first use, so users with 20 models configured don't pay 20 import costs.
* **No Pydantic on the streaming hot path.** SSE frames are parsed with
  ``orjson`` into dicts; we only build a Pydantic model for the final aggregate.
* **Provider-specific blocks preserved.** Anthropic ``thinking``, Gemini
  ``grounding``, etc. are emitted as typed events instead of being flattened.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry
    from relay.types import ChatRequest, ChatResponse, StreamEvent


@runtime_checkable
class Provider(Protocol):
    """Contract every provider adapter implements.

    Implementations are constructed once per :class:`Hub` and shared across
    requests; they hold no per-request state except for the pooled HTTP client.
    """

    name: str
    """Stable provider id, e.g. ``"openai"``, ``"anthropic"``."""

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse: ...

    def stream(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None:
        """Release any provider-specific resources. Default impl does nothing."""
        ...


class BaseProvider:
    """Convenience base for providers that don't override every protocol method."""

    name: str = ""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
