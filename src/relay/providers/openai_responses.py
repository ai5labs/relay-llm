"""OpenAI Responses API adapter.

Lives alongside the Chat Completions adapter; users opt in by setting
``api_style: responses`` on their model entry. Default remains
``chat_completions`` for backward compatibility.

The Responses API differs from Chat Completions in three ways that matter:

1. **Endpoint**: ``POST /v1/responses`` — single endpoint for both streaming and
   non-streaming.
2. **Stateful conversations**: ``previous_response_id`` lets the server keep
   message history. Relay exposes this as ``ChatRequest.metadata.previous_response_id``.
3. **Reasoning items**: ``reasoning`` items are returned as first-class output
   types (separate from text). Required for ``o``-series chain-of-thought
   continuation.

Streaming events follow ``response.output_text.delta``, ``response.tool_call.delta``,
``response.completed`` rather than the SSE shape of Chat Completions.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from relay._internal.credentials import resolve_secret
from relay.errors import (
    AuthenticationError,
    ProviderError,
    TimeoutError,
)
from relay.providers._base import BaseProvider
from relay.providers.openai_compat import _normalize_finish_reason
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
    ThinkingBlock,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageDelta,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


_DEFAULT_BASE = "https://api.openai.com/v1"


class OpenAIResponsesProvider(BaseProvider):
    """Adapter for ``POST /v1/responses``."""

    name = "openai-responses"

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
                "OpenAI Responses requires a credential",
                provider=self.name,
                model=entry.model_id,
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )
        auth = {"authorization": f"Bearer {api_key}"}

        body = self._build_body(entry, request, stream=False)
        start = time.perf_counter()
        try:
            resp = await client.post("/responses", json=body, headers=auth)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "OpenAI Responses timed out", provider=self.name, model=entry.model_id
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
                "OpenAI Responses requires a credential",
                provider=self.name,
                model=entry.model_id,
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )
        auth = {"authorization": f"Bearer {api_key}"}

        body = self._build_body(entry, request, stream=True)
        start = time.perf_counter()

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        tool_blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        response_id = ""
        provider_model = entry.model_id
        finish_reason: str | None = None

        try:
            async with client.stream("POST", "/responses", json=body, headers=auth) as resp:
                self._raise_for_status(resp, entry, stream=True)
                yield StreamStart(
                    id=resp.headers.get("x-request-id") or uuid.uuid4().hex,
                    model=entry.target,
                    provider=self.name,
                )
                event_type = ""
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        data = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue

                    et = event_type or data.get("type") or ""
                    if et == "response.created":
                        response_id = (data.get("response") or {}).get("id", response_id)
                        provider_model = (data.get("response") or {}).get("model", provider_model)
                    elif et == "response.output_text.delta":
                        delta = data.get("delta", "")
                        if delta:
                            text_buf.append(delta)
                            yield TextDelta(text=delta)
                    elif et == "response.reasoning.delta" or et == "response.thinking.delta":
                        delta = data.get("delta", "")
                        if delta:
                            thinking_buf.append(delta)
                            yield ThinkingDelta(text=delta)
                    elif et == "response.tool_call.created":
                        item = data.get("item") or {}
                        idx = data.get("output_index", len(tool_blocks))
                        tool_blocks[idx] = {
                            "id": item.get("id", f"call_{idx}"),
                            "name": item.get("name", ""),
                            "arguments": "",
                        }
                        yield ToolCallDelta(index=idx, id=item.get("id"), name=item.get("name"))
                    elif et == "response.tool_call.delta":
                        idx = data.get("output_index", 0)
                        delta = data.get("delta", "")
                        if idx in tool_blocks:
                            tool_blocks[idx]["arguments"] += delta
                        yield ToolCallDelta(index=idx, arguments_delta=delta)
                    elif et == "response.completed":
                        r = data.get("response") or {}
                        usage_in = r.get("usage") or {}
                        usage = _parse_usage(usage_in)
                        yield UsageDelta(usage=usage)
                        finish_reason = (r.get("status") or "").replace("incomplete", "length")
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "OpenAI Responses stream timed out",
                provider=self.name,
                model=entry.model_id,
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_blocks):
            slot = tool_blocks[idx]
            args: dict[str, Any] = {}
            if slot["arguments"]:
                try:
                    args = orjson.loads(slot["arguments"])
                except orjson.JSONDecodeError:
                    args = {"_raw": slot["arguments"]}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))
        thinking = [ThinkingBlock(text="".join(thinking_buf))] if thinking_buf else []
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_buf)),
            finish_reason=_map_responses_status(finish_reason),
            tool_calls=tool_calls,
            thinking=thinking,
        )
        final = ChatResponse(
            id=response_id or uuid.uuid4().hex,
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

    def _build_body(
        self, entry: ModelEntry, request: ChatRequest, *, stream: bool
    ) -> dict[str, Any]:
        # Responses API uses ``input`` (string or array of items).
        # We pass the full message array as input items.
        input_items: list[dict[str, Any]] = []
        instructions: list[str] = []
        for m in request.messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    instructions.append(m.content)
                continue
            if isinstance(m.content, str):
                input_items.append({"role": m.role, "content": m.content})
            else:
                # Block list — translate to Responses item shape.
                parts: list[dict[str, Any]] = []
                for block in m.content:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        parts.append({"type": "input_text", "text": block.text})  # type: ignore[attr-defined]
                    elif btype == "image":
                        parts.append(
                            {
                                "type": "input_image",
                                "image_url": getattr(block, "url", "") or "",
                            }
                        )
                input_items.append({"role": m.role, "content": parts})

        body: dict[str, Any] = {
            "model": entry.model_id,
            "input": input_items,
            "stream": stream,
        }
        if instructions:
            body["instructions"] = "\n\n".join(instructions)
        if request.max_tokens is not None:
            body["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.seed is not None:
            body["seed"] = request.seed
        if request.metadata:
            body["metadata"] = {
                k: v for k, v in request.metadata.items() if k != "previous_response_id"
            }
            if "previous_response_id" in request.metadata:
                body["previous_response_id"] = request.metadata["previous_response_id"]

        if request.tools:
            from relay.tools import compile_all

            compiled_chat_tools = compile_all(request.tools, "openai")
            # Responses API uses a slightly different tool item shape.
            body["tools"] = [
                {
                    "type": "function",
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"]["parameters"],
                    **({"strict": True} if t["function"].get("strict") else {}),
                }
                for t in compiled_chat_tools
            ]
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice

        if request.reasoning is not None:
            from relay._internal.reasoning import to_openai

            r = to_openai(request.reasoning)
            if r is not None:
                body["reasoning"] = r

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
        msg = "OpenAI Responses error"
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict):
                msg = err.get("message") or msg

        kwargs = {
            "provider": self.name,
            "model": entry.model_id,
            "status_code": resp.status_code,
            "raw": body,
        }
        if resp.status_code in (401, 403):
            raise AuthenticationError(msg, **kwargs)
        raise ProviderError(msg, **kwargs)

    def _parse_response(
        self, *, data: dict[str, Any], entry: ModelEntry, latency_ms: float
    ) -> ChatResponse:
        text_parts: list[str] = []
        thinking: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []

        for item in data.get("output") or []:
            itype = item.get("type")
            if itype == "message":
                for content in item.get("content") or []:
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
            elif itype == "reasoning":
                # Reasoning summary blocks (when included).
                summary = item.get("summary") or []
                if isinstance(summary, list):
                    for s in summary:
                        text = s.get("text") if isinstance(s, dict) else None
                        if text:
                            thinking.append(
                                ThinkingBlock(text=text, signature=item.get("encrypted_content"))
                            )
            elif itype == "function_call":
                args_raw = item.get("arguments")
                try:
                    args = orjson.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except orjson.JSONDecodeError:
                    args = {"_raw": args_raw}
                tool_calls.append(
                    ToolCall(
                        id=item.get("id") or item.get("call_id") or "",
                        name=item.get("name", ""),
                        arguments=args,
                    )
                )

        usage = _parse_usage(data.get("usage") or {})
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_parts)),
            finish_reason=_map_responses_status(data.get("status")),
            tool_calls=tool_calls,
            thinking=thinking,
        )
        return ChatResponse(
            id=data.get("id", ""),
            model=entry.target,
            provider_model=data.get("model", entry.model_id),
            provider=self.name,
            choices=[choice],
            usage=usage,
            cost=None,
            created_at=float(data.get("created_at", time.time())),
            latency_ms=latency_ms,
        )


def _parse_usage(u: dict[str, Any]) -> Usage:
    in_details = u.get("input_tokens_details") or {}
    out_details = u.get("output_tokens_details") or {}
    return Usage(
        input_tokens=int(u.get("input_tokens", 0)),
        output_tokens=int(u.get("output_tokens", 0)),
        cached_input_tokens=int(in_details.get("cached_tokens", 0)),
        reasoning_tokens=int(out_details.get("reasoning_tokens", 0)),
    )


def _map_responses_status(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "other",
    }
    if raw in mapping:
        return mapping[raw]
    return _normalize_finish_reason(raw)
