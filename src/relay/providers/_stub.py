"""Shared stub for providers scaffolded but not yet implemented in v0.1.

Each native non-OpenAI-compat provider (Bedrock, Azure OpenAI, Google direct,
Vertex AI, Cohere) has a one-class file that subclasses this. They raise
``NotImplementedError`` with a clear roadmap message.

The architecture is in place — the entry in :mod:`relay.providers` and the
catalog rows are wired — so v0.2 can flesh these in without changing any
public API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from relay.providers._base import BaseProvider

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry
    from relay.types import ChatRequest, ChatResponse, StreamEvent


class _NotYetImplementedProvider(BaseProvider):
    """Base for providers slated for v0.2."""

    eta: str = "v0.2"

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        raise NotImplementedError(
            f"Provider {self.name!r} is scaffolded but not implemented yet "
            f"(planned: {self.eta}). Track progress at "
            f"https://github.com/ai5labs/relay-llm/issues."
        )

    def stream(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError(
            f"Provider {self.name!r} streaming is not implemented yet (planned: {self.eta})."
        )
