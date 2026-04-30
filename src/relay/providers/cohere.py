"""Cohere v2 chat adapter.

Cohere's v2 ``/v2/chat`` API is OpenAI-shaped at the surface but has its own
field names:

* System messages are first-class via ``role: "system"`` (modern v2 API).
* Tool calls use Cohere's own schema (handled by the tool compiler).
* Streaming events are typed (``message-start``, ``content-delta``, etc.).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from relay._internal.credentials import resolve_secret
from relay._internal.transport import parse_retry_after
from relay.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from relay.providers._base import BaseProvider
from relay.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    Message,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


_DEFAULT_BASE = "https://api.cohere.com"


class CohereProvider(BaseProvider):
    name = "cohere"

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError(
                "Cohere requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
            extra_headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        body = self._build_body(entry, request, stream=False)
        start = time.perf_counter()
        try:
            resp = await client.post("/v2/chat", json=body)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Cohere request timed out", provider=self.name, model=entry.model_id
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
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError(
                "Cohere requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
            extra_headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        body = self._build_body(entry, request, stream=True)
        start = time.perf_counter()

        text_buf: list[str] = []
        usage = Usage()
        finish_reason: str | None = None
        message_id = ""
        provider_model = entry.model_id

        try:
            async with client.stream("POST", "/v2/chat", json=body) as resp:
                self._raise_for_status(resp, entry, stream=True)
                yield StreamStart(
                    id=resp.headers.get("x-cohere-request-id") or uuid.uuid4().hex,
                    model=entry.target,
                    provider=self.name,
                )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "message-start":
                        message_id = event.get("id") or ""
                        provider_model = event.get("model", provider_model)
                    elif etype == "content-delta":
                        delta = event.get("delta", {}).get("message", {}).get("content", {})
                        text = delta.get("text") if isinstance(delta, dict) else None
                        if text:
                            text_buf.append(text)
                            yield TextDelta(text=text)
                    elif etype == "message-end":
                        delta = event.get("delta", {})
                        finish_reason = delta.get("finish_reason") or finish_reason
                        u = delta.get("usage", {}).get("billed_units", {}) if delta else {}
                        if u:
                            usage = Usage(
                                input_tokens=int(u.get("input_tokens", 0)),
                                output_tokens=int(u.get("output_tokens", 0)),
                            )
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Cohere stream timed out", provider=self.name, model=entry.model_id
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_buf)),
            finish_reason=_map_finish(finish_reason),
        )
        final = ChatResponse(
            id=message_id or uuid.uuid4().hex,
            model=entry.target,
            provider_model=provider_model,
            provider=self.name,
            choices=[choice],
            usage=usage,
            cost=None,
            created_at=time.time(),
            latency_ms=latency_ms,
        )
        yield StreamEnd(finish_reason=choice.finish_reason, response=final)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_body(
        self, entry: ModelEntry, request: ChatRequest, *, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": entry.model_id,
            "messages": [_msg_to_cohere(m) for m in request.messages],
            "stream": stream,
        }
        for k, v in entry.params.items():
            body.setdefault(k, v)
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["p"] = request.top_p
        if request.seed is not None:
            body["seed"] = request.seed
        if request.stop is not None:
            body["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.tools:
            from relay.tools import compile_all

            body["tools"] = compile_all(request.tools, "cohere")
        if request.response_format == "json_object" or (
            isinstance(request.response_format, dict)
            and request.response_format.get("type") == "json_object"
        ):
            body["response_format"] = {"type": "json_object"}
        elif isinstance(request.response_format, dict):
            body["response_format"] = request.response_format
        return body

    def _raise_for_status(
        self, resp: httpx.Response, entry: ModelEntry, *, stream: bool = False
    ) -> None:
        if resp.status_code < 400:
            return
        body: Any = None
        try:
            raw = resp.read() if stream else resp.content
            body = orjson.loads(raw) if raw else None
        except orjson.JSONDecodeError:
            body = None

        msg = "Cohere error"
        if isinstance(body, dict):
            msg = body.get("message") or body.get("detail") or msg

        kwargs = {
            "provider": self.name,
            "model": entry.model_id,
            "status_code": resp.status_code,
            "raw": body,
        }
        if resp.status_code in (401, 403):
            raise AuthenticationError(msg, **kwargs)
        if resp.status_code == 429:
            raise RateLimitError(msg, retry_after=parse_retry_after(resp.headers), **kwargs)
        raise ProviderError(msg, **kwargs)

    def _parse_response(
        self, *, data: dict[str, Any], entry: ModelEntry, latency_ms: float
    ) -> ChatResponse:
        message = data.get("message") or {}
        content = message.get("content") or []
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments")
            try:
                args = orjson.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except orjson.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        usage_in = (data.get("usage") or {}).get("billed_units") or {}
        usage = Usage(
            input_tokens=int(usage_in.get("input_tokens", 0)),
            output_tokens=int(usage_in.get("output_tokens", 0)),
        )
        finish_raw = data.get("finish_reason")
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_parts)),
            finish_reason=_map_finish(finish_raw),
            tool_calls=tool_calls,
        )
        return ChatResponse(
            id=data.get("id", ""),
            model=entry.target,
            provider_model=entry.model_id,
            provider=self.name,
            choices=[choice],
            usage=usage,
            cost=None,
            created_at=time.time(),
            latency_ms=latency_ms,
        )


def _msg_to_cohere(m: Message) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}
    parts: list[Any] = []
    for block in m.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append({"type": "text", "text": block.text})  # type: ignore[attr-defined]
    return {"role": m.role, "content": parts}


def _map_finish(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "COMPLETE": "stop",
        "STOP_SEQUENCE": "stop",
        "MAX_TOKENS": "length",
        "TOOL_CALL": "tool_calls",
        "ERROR": "error",
    }
    return mapping.get(raw, "other")
