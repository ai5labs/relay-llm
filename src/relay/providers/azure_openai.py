"""Azure OpenAI adapter.

Azure OpenAI speaks the OpenAI wire format but with three quirks:

1. **Deployment routing.** The URL is
   ``{base_url}/openai/deployments/{deployment}/chat/completions?api-version={ver}``
   rather than ``/v1/chat/completions``. ``deployment`` is a per-tenant name,
   not the underlying model id.
2. **api-key header**, not ``Authorization: Bearer``.
3. **api-version** query parameter is required and changes feature availability.

We subclass :class:`OpenAICompatibleProvider` and override the URL/header build
steps.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from relay._internal.credentials import resolve_secret
from relay.errors import (
    AuthenticationError,
    TimeoutError,
)
from relay.providers.openai_compat import OpenAICompatibleProvider
from relay.types import (
    ChatRequest,
    ChatResponse,
    StreamEvent,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


_DEFAULT_API_VERSION = "2024-08-01-preview"


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI adapter — same wire as OpenAI, different URL + auth header."""

    def __init__(self) -> None:
        # Base URL is per-tenant; we pull it from each ModelEntry, not the class.
        super().__init__(name="azure", default_base_url="", api_key_required=True)

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        base_url = self._require_base_url(entry)
        deployment = self._require_deployment(entry)
        api_version = entry.api_version or _DEFAULT_API_VERSION
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError(
                "Azure OpenAI requires a credential",
                provider=self.name,
                model=entry.model_id,
            )

        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
        )
        auth = {"api-key": api_key}

        body = self._build_body(entry, request, stream=False)
        # Azure's deployment URL doesn't take 'model' in the body — strip it.
        body.pop("model", None)
        url = f"/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": api_version}

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=body, params=params, headers=auth)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Azure OpenAI request timed out", provider=self.name, model=entry.model_id
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000

        self._raise_for_status(resp, entry)
        data = resp.json()
        return self._parse_response(data=data, entry=entry, latency_ms=latency_ms)

    async def stream(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> AsyncIterator[StreamEvent]:
        # Reuse OpenAI-compat streaming logic; only the URL+headers differ.
        # We do the URL/header setup here, then delegate frame parsing.
        base_url = self._require_base_url(entry)
        deployment = self._require_deployment(entry)
        api_version = entry.api_version or _DEFAULT_API_VERSION
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError(
                "Azure OpenAI requires a credential",
                provider=self.name,
                model=entry.model_id,
            )

        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
        )
        auth = {"api-key": api_key}

        body = self._build_body(entry, request, stream=True)
        body.pop("model", None)
        body["stream_options"] = {"include_usage": True}
        url = f"/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": api_version}

        # Reproduce the OpenAI-compat stream loop with these custom URL/params.
        # This is intentionally a copy rather than a refactor — keeps the
        # OpenAI-compat code straightforward and avoids cross-cutting helpers.
        import time as _time
        import uuid

        from relay.types import (
            Choice,
            Message,
            StreamEnd,
            StreamStart,
            TextDelta,
            ToolCall,
            ToolCallDelta,
            Usage,
            UsageDelta,
        )

        text_buf: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage_seen: Usage = Usage()
        finish_reason: str | None = None
        response_id: str = ""
        provider_model: str = entry.model_id
        start = _time.perf_counter()

        try:
            async with client.stream("POST", url, json=body, params=params, headers=auth) as resp:
                self._raise_for_status(resp, entry, stream=True)
                yield StreamStart(
                    id=resp.headers.get("x-request-id") or uuid.uuid4().hex,
                    model=entry.target,
                    provider=self.name,
                )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue

                    response_id = response_id or chunk.get("id", "")
                    provider_model = chunk.get("model", provider_model)

                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage_seen = self._parse_usage(chunk_usage)
                        yield UsageDelta(usage=usage_seen)

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason

                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        text_buf.append(text)
                        yield TextDelta(text=text)
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        slot = tool_calls_by_index.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc_delta.get("id"):
                            slot["id"] = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        args_chunk = fn.get("arguments")
                        if args_chunk:
                            slot["arguments"] += args_chunk
                        yield ToolCallDelta(
                            index=idx,
                            id=tc_delta.get("id"),
                            name=fn.get("name"),
                            arguments_delta=args_chunk or "",
                        )
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Azure OpenAI stream timed out", provider=self.name, model=entry.model_id
            ) from e

        latency_ms = (_time.perf_counter() - start) * 1000
        from relay.providers.openai_compat import _normalize_finish_reason

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_by_index):
            slot = tool_calls_by_index[idx]
            args: dict[str, Any] = {}
            if slot["arguments"]:
                try:
                    args = orjson.loads(slot["arguments"])
                except orjson.JSONDecodeError:
                    args = {"_raw": slot["arguments"]}
            tool_calls.append(
                ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"] or "", arguments=args)
            )
        message = Message(role="assistant", content="".join(text_buf))
        choice_out = Choice(
            index=0,
            message=message,
            finish_reason=_normalize_finish_reason(finish_reason),
            tool_calls=tool_calls,
        )
        final = ChatResponse(
            id=response_id or uuid.uuid4().hex,
            model=entry.target,
            provider_model=provider_model,
            provider=self.name,
            choices=[choice_out],
            usage=usage_seen,
            cost=None,
            created_at=_time.time(),
            latency_ms=latency_ms,
        )
        yield StreamEnd(finish_reason=choice_out.finish_reason, response=final)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _require_base_url(self, entry: ModelEntry) -> str:
        if not entry.base_url:
            raise AuthenticationError(
                "Azure OpenAI requires base_url on the model entry "
                "(e.g. https://my-resource.openai.azure.com)",
                provider=self.name,
                model=entry.model_id,
            )
        return entry.base_url.rstrip("/")

    def _require_deployment(self, entry: ModelEntry) -> str:
        if not entry.deployment:
            raise AuthenticationError(
                "Azure OpenAI requires a 'deployment' field on the model entry",
                provider=self.name,
                model=entry.model_id,
            )
        return entry.deployment
