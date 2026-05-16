"""Anthropic Messages API adapter.

Native, not OpenAI-compat. The Messages API has its own shape:

* ``content`` is always a list of blocks (``text``, ``thinking``, ``tool_use``, ``tool_result``).
* ``stop_reason`` values differ from OpenAI's ``finish_reason``.
* Streaming events are typed (``message_start``, ``content_block_start``,
  ``content_block_delta``, ``content_block_stop``, ``message_delta``,
  ``message_stop``).

Relay preserves Anthropic's ``thinking`` blocks rather than flattening them
into the visible text — they surface as :class:`ThinkingBlock` on the response
and as :class:`ThinkingDelta` events while streaming.
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
    ContentPolicyError,
    ContextWindowError,
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
    TextBlock,
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

_DEFAULT_BASE = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    name = "anthropic"

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
                "Anthropic requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )
        auth = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        body = self._build_body(entry, request, stream=False)
        start = time.perf_counter()
        try:
            resp = await client.post("/v1/messages", json=body, headers=auth)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Anthropic request timed out", provider=self.name, model=entry.model_id
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
                "Anthropic requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )
        auth = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        body = self._build_body(entry, request, stream=True)
        start = time.perf_counter()

        text_blocks: dict[int, list[str]] = {}
        thinking_blocks: dict[int, list[str]] = {}
        tool_blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason: str | None = None
        message_id = ""
        provider_model = entry.model_id

        try:
            async with client.stream("POST", "/v1/messages", json=body, headers=auth) as resp:
                self._raise_for_status(resp, entry, stream=True)
                yield StreamStart(
                    id=resp.headers.get("request-id") or uuid.uuid4().hex,
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
                    try:
                        data = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue

                    if event_type == "message_start":
                        msg = data.get("message", {})
                        message_id = msg.get("id", "")
                        provider_model = msg.get("model", provider_model)
                        u = msg.get("usage") or {}
                        usage = Usage(
                            input_tokens=int(u.get("input_tokens", 0)),
                            output_tokens=int(u.get("output_tokens", 0)),
                            cached_input_tokens=int(u.get("cache_read_input_tokens", 0)),
                        )
                    elif event_type == "content_block_start":
                        idx = data.get("index", 0)
                        block = data.get("content_block") or {}
                        btype = block.get("type")
                        if btype == "tool_use":
                            tool_blocks[idx] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": "",
                            }
                            yield ToolCallDelta(
                                index=idx,
                                id=block.get("id"),
                                name=block.get("name"),
                                arguments_delta="",
                            )
                    elif event_type == "content_block_delta":
                        idx = data.get("index", 0)
                        delta = data.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            text_blocks.setdefault(idx, []).append(delta.get("text", ""))
                            yield TextDelta(index=idx, text=delta.get("text", ""))
                        elif dtype == "thinking_delta":
                            thinking_blocks.setdefault(idx, []).append(delta.get("thinking", ""))
                            yield ThinkingDelta(index=idx, text=delta.get("thinking", ""))
                        elif dtype == "input_json_delta":
                            partial = delta.get("partial_json", "")
                            slot = tool_blocks.setdefault(
                                idx, {"id": "", "name": "", "arguments": ""}
                            )
                            slot["arguments"] += partial
                            yield ToolCallDelta(index=idx, arguments_delta=partial)
                    elif event_type == "message_delta":
                        delta = data.get("delta") or {}
                        if "stop_reason" in delta:
                            stop_reason = delta["stop_reason"]
                        u = data.get("usage") or {}
                        if u:
                            usage = Usage(
                                input_tokens=usage.input_tokens,
                                output_tokens=int(u.get("output_tokens", usage.output_tokens)),
                                cached_input_tokens=usage.cached_input_tokens,
                            )
                            yield UsageDelta(usage=usage)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Anthropic stream timed out", provider=self.name, model=entry.model_id
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000

        # Assemble final response.
        all_indices = sorted(set(text_blocks) | set(thinking_blocks) | set(tool_blocks))
        content: list[Any] = []
        tool_calls: list[ToolCall] = []
        thinking_out: list[ThinkingBlock] = []
        for idx in all_indices:
            if idx in text_blocks:
                content.append(TextBlock(text="".join(text_blocks[idx])))
            elif idx in thinking_blocks:
                thinking_out.append(ThinkingBlock(text="".join(thinking_blocks[idx])))
            elif idx in tool_blocks:
                slot = tool_blocks[idx]
                args: dict[str, Any] = {}
                if slot["arguments"]:
                    try:
                        args = orjson.loads(slot["arguments"])
                    except orjson.JSONDecodeError:
                        args = {"_raw": slot["arguments"]}
                tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))

        message = Message(
            role="assistant",
            content=content
            if len(content) != 1 or not isinstance(content[0], TextBlock)
            else content[0].text,
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason=_map_stop_reason(stop_reason),
            tool_calls=tool_calls,
            thinking=thinking_out,
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
        self,
        entry: ModelEntry,
        request: ChatRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        # System messages collapse into a top-level ``system`` field on Anthropic.
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue
            messages.append(_msg_to_anthropic(m))

        body: dict[str, Any] = {
            "model": entry.model_id,
            "messages": messages,
            "max_tokens": request.max_tokens or entry.params.get("max_tokens") or 4096,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.metadata is not None:
            body["metadata"] = request.metadata
        if request.thinking is not None:
            body["thinking"] = request.thinking
        elif request.reasoning is not None:
            from relay._internal.reasoning import to_anthropic

            thinking_block = to_anthropic(request.reasoning)
            if thinking_block is not None:
                body["thinking"] = thinking_block

        if request.tools:
            from relay.tools import compile_all

            body["tools"] = compile_all(request.tools, "anthropic")
        if request.tool_choice == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif request.tool_choice == "required":
            body["tool_choice"] = {"type": "any"}
        elif isinstance(request.tool_choice, dict):
            body["tool_choice"] = request.tool_choice

        # Per-model defaults applied last (don't clobber explicit request fields).
        for k, v in entry.params.items():
            body.setdefault(k, v)
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

        message = "Anthropic error"
        err_type = ""
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict):
                message = err.get("message") or message
                err_type = err.get("type", "")

        request_id = resp.headers.get("request-id")
        kwargs = {
            "provider": self.name,
            "model": entry.model_id,
            "status_code": resp.status_code,
            "request_id": request_id,
            "raw": body,
        }
        if resp.status_code in (401, 403) or err_type == "authentication_error":
            raise AuthenticationError(message, **kwargs)
        if resp.status_code == 429 or err_type == "rate_limit_error":
            raise RateLimitError(message, retry_after=parse_retry_after(resp.headers), **kwargs)
        lower = message.lower()
        if "context" in lower and ("length" in lower or "window" in lower or "tokens" in lower):
            raise ContextWindowError(message, **kwargs)
        if "content" in lower and ("policy" in lower or "filter" in lower):
            raise ContentPolicyError(message, **kwargs)
        raise ProviderError(message, **kwargs)

    def _parse_response(
        self, *, data: dict[str, Any], entry: ModelEntry, latency_ms: float
    ) -> ChatResponse:
        content_in = data.get("content") or []
        text_parts: list[str] = []
        thinking: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        for block in content_in:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking.append(
                    ThinkingBlock(
                        text=block.get("thinking", ""),
                        signature=block.get("signature"),
                    )
                )
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )

        usage_in = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_in.get("input_tokens", 0)),
            output_tokens=int(usage_in.get("output_tokens", 0)),
            cached_input_tokens=int(usage_in.get("cache_read_input_tokens", 0)),
        )

        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_parts)),
            finish_reason=_map_stop_reason(data.get("stop_reason")),
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
            created_at=time.time(),
            latency_ms=latency_ms,
        )


def _msg_to_anthropic(m: Message) -> dict[str, Any]:
    from relay.cache import _to_anthropic_cache_control
    from relay.types import CacheHintBlock

    role = "assistant" if m.role == "assistant" else "user"
    if isinstance(m.content, str):
        return {"role": role, "content": m.content}
    blocks: list[dict[str, Any]] = []
    for block in m.content:
        # Cache-control marker: attach the cache_control attribute to the
        # *previous* block, which is how Anthropic's prompt cache works.
        if isinstance(block, CacheHintBlock):
            if blocks:
                blocks[-1]["cache_control"] = _to_anthropic_cache_control(block)
            continue
        btype = getattr(block, "type", None)
        if btype == "text":
            blocks.append({"type": "text", "text": block.text})  # type: ignore[attr-defined]
        elif btype == "image":
            url = getattr(block, "url", "")
            media = getattr(block, "media_type", None)
            if url.startswith("data:"):
                # data: URI — split into media + base64.
                _, _, b64 = url.partition(",")
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media or "image/png",
                            "data": b64,
                        },
                    }
                )
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        elif btype == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.text,  # type: ignore[attr-defined]
                    "signature": getattr(block, "signature", "") or "",
                }
            )
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,  # type: ignore[attr-defined]
                    "name": block.name,  # type: ignore[attr-defined]
                    "input": block.input,  # type: ignore[attr-defined]
                }
            )
        elif btype == "tool_result":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,  # type: ignore[attr-defined]
                    "content": block.content,  # type: ignore[attr-defined]
                    **({"is_error": True} if getattr(block, "is_error", False) else {}),
                }
            )
    return {"role": role, "content": blocks}


def _map_stop_reason(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    return mapping.get(raw, "other")
