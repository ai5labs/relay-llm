"""OpenAI-compatible Chat Completions adapter.

One adapter, many providers. Handles every endpoint that speaks the OpenAI
``/v1/chat/completions`` wire format:

* OpenAI direct
* Groq, Together, DeepSeek, xAI, Mistral, Fireworks, Perplexity (managed)
* Ollama, vLLM, LM Studio (self-hosted)
* OpenRouter (aggregator)

Differences from LiteLLM
------------------------
* Streaming tool-call deltas are merged by ``index`` (not ``id``) — fixes the
  bug that drops ~90% of argument deltas in LiteLLM (issue #20711, #8012).
* No Pydantic validation on per-token SSE frames — frames are parsed with
  ``orjson`` into dicts; only the final assembled response is validated.
* Errors are mapped to the typed Relay hierarchy (``RateLimitError``,
  ``ContextWindowError``, ``ContentPolicyError``, ``AuthenticationError``,
  ``ProviderError``) with ``Retry-After`` and ``limit_type`` parsed.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from relay._internal.credentials import resolve_secret
from relay._internal.transport import detect_limit_type, parse_retry_after
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
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageDelta,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


class OpenAICompatibleProvider(BaseProvider):
    """One adapter for every OpenAI-shaped endpoint."""

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key_required: bool = True,
    ) -> None:
        self.name = name
        self._default_base_url = default_base_url
        self._api_key_required = api_key_required

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        base_url = entry.base_url or self._default_base_url
        api_key = await self._get_api_key(entry)
        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
            extra_headers=self._auth_headers(api_key),
        )

        body = self._build_body(entry, request, stream=False)
        start = time.perf_counter()
        try:
            resp = await client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"{self.name} request timed out", provider=self.name, model=entry.model_id
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000

        self._raise_for_status(resp, entry)
        data = resp.json()
        return self._parse_response(
            data=data,
            entry=entry,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    async def stream(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> AsyncIterator[StreamEvent]:
        base_url = entry.base_url or self._default_base_url
        api_key = await self._get_api_key(entry)
        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
            extra_headers=self._auth_headers(api_key),
        )

        body = self._build_body(entry, request, stream=True)
        body["stream_options"] = {"include_usage": True}
        start = time.perf_counter()

        text_buf: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage_seen: Usage = Usage()
        finish_reason: str | None = None
        response_id: str = ""
        provider_model: str = entry.model_id

        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
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
                        # Malformed frame — skip rather than abort the stream.
                        continue

                    response_id = response_id or chunk.get("id", "")
                    provider_model = chunk.get("model", provider_model)

                    # Usage frames may arrive at any time when include_usage=True.
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

                    # Tool-call deltas — merge by INDEX, not id.
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
                f"{self.name} stream timed out", provider=self.name, model=entry.model_id
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000

        # Assemble final response.
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
                ToolCall(
                    id=slot["id"] or f"call_{idx}",
                    name=slot["name"] or "",
                    arguments=args,
                )
            )

        message = Message(
            role="assistant",
            content="".join(text_buf),
        )
        choice = Choice(
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
            choices=[choice],
            usage=usage_seen,
            cost=None,  # Pricing applied by Hub layer
            created_at=time.time(),
            latency_ms=latency_ms,
        )
        yield StreamEnd(finish_reason=choice.finish_reason, response=final)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _get_api_key(self, entry: ModelEntry) -> str:
        if entry.credential is None:
            if self._api_key_required:
                raise AuthenticationError(
                    f"model {entry.target!r} has no credential and provider "
                    f"{self.name!r} requires one",
                    provider=self.name,
                )
            return ""
        return await resolve_secret(entry.credential)

    def _auth_headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            return {}
        return {"authorization": f"Bearer {api_key}"}

    def _build_body(
        self,
        entry: ModelEntry,
        request: ChatRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": entry.model_id,
            "messages": [_msg_to_openai(m) for m in request.messages],
            "stream": stream,
        }

        # Per-model defaults are applied first, then per-call overrides win.
        for k, v in entry.params.items():
            body.setdefault(k, v)

        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop"] = request.stop
        if request.seed is not None:
            body["seed"] = request.seed
        if request.metadata is not None:
            body["metadata"] = request.metadata

        if request.tools:
            from relay.tools import compile_all

            body["tools"] = compile_all(request.tools, self.name)

        if request.reasoning is not None:
            from relay._internal.reasoning import to_openai

            reasoning_block = to_openai(request.reasoning)
            if reasoning_block is not None:
                body["reasoning"] = reasoning_block
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice

        if request.response_format is not None:
            if isinstance(request.response_format, dict):
                body["response_format"] = request.response_format
            elif request.response_format == "json_object":
                body["response_format"] = {"type": "json_object"}
        return body

    def _raise_for_status(
        self,
        resp: httpx.Response,
        entry: ModelEntry,
        *,
        stream: bool = False,
    ) -> None:
        if resp.status_code < 400:
            return

        # Body parsing — small enough to read; for streams the body is the error.
        body: Any = None
        try:
            if stream:
                # In stream mode we haven't consumed yet.
                raw = resp.read()
                body = orjson.loads(raw) if raw else None
            else:
                body = resp.json()
        except (orjson.JSONDecodeError, ValueError):
            body = resp.text if not stream else None

        err_msg = self._extract_error_message(body) or f"{self.name} returned {resp.status_code}"
        request_id = resp.headers.get("x-request-id") or resp.headers.get("openai-request-id")

        if resp.status_code == 401 or resp.status_code == 403:
            raise AuthenticationError(
                err_msg,
                provider=self.name,
                model=entry.model_id,
                status_code=resp.status_code,
                request_id=request_id,
                raw=body,
            )
        if resp.status_code == 429:
            raise RateLimitError(
                err_msg,
                provider=self.name,
                model=entry.model_id,
                status_code=resp.status_code,
                request_id=request_id,
                retry_after=parse_retry_after(resp.headers),
                limit_type=detect_limit_type(body, resp.headers),
                raw=body,
            )
        if resp.status_code == 400:
            lower = err_msg.lower()
            if "context" in lower and ("length" in lower or "window" in lower or "tokens" in lower):
                raise ContextWindowError(
                    err_msg,
                    provider=self.name,
                    model=entry.model_id,
                    status_code=resp.status_code,
                    request_id=request_id,
                    raw=body,
                )
            if (
                "content_policy" in lower
                or "content policy" in lower
                or "safety" in lower
                or "harmful" in lower
            ):
                raise ContentPolicyError(
                    err_msg,
                    provider=self.name,
                    model=entry.model_id,
                    status_code=resp.status_code,
                    request_id=request_id,
                    raw=body,
                )

        raise ProviderError(
            err_msg,
            provider=self.name,
            model=entry.model_id,
            status_code=resp.status_code,
            request_id=request_id,
            raw=body,
        )

    @staticmethod
    def _extract_error_message(body: Any) -> str | None:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return err.get("message") or str(err)
            if isinstance(err, str):
                return err
            return body.get("message")
        if isinstance(body, str):
            return body[:500]
        return None

    def _parse_response(
        self,
        *,
        data: dict[str, Any],
        entry: ModelEntry,
        latency_ms: float,
    ) -> ChatResponse:
        choices_in = data.get("choices") or []
        choices_out: list[Choice] = []
        for c in choices_in:
            msg = c.get("message") or {}
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments")
                try:
                    args = orjson.loads(args_raw) if args_raw else {}
                except orjson.JSONDecodeError:
                    args = {"_raw": args_raw}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=args,
                    )
                )
            content = msg.get("content") or ""
            message = Message(role=msg.get("role", "assistant"), content=content)
            choices_out.append(
                Choice(
                    index=c.get("index", 0),
                    message=message,
                    finish_reason=_normalize_finish_reason(c.get("finish_reason")),
                    tool_calls=tool_calls,
                )
            )

        usage = self._parse_usage(data.get("usage") or {})

        return ChatResponse(
            id=data.get("id", ""),
            model=entry.target,
            provider_model=data.get("model", entry.model_id),
            provider=self.name,
            choices=choices_out,
            usage=usage,
            cost=None,
            created_at=float(data.get("created", time.time())),
            latency_ms=latency_ms,
            raw=None,
        )

    @staticmethod
    def _parse_usage(usage: dict[str, Any]) -> Usage:
        details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return Usage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_input_tokens=int(details.get("cached_tokens", 0)),
            reasoning_tokens=int(completion_details.get("reasoning_tokens", 0)),
        )


def _msg_to_openai(m: Message) -> dict[str, Any]:
    """Translate a Relay Message to the OpenAI wire format."""
    out: dict[str, Any] = {"role": m.role}
    if m.name:
        out["name"] = m.name
    if isinstance(m.content, str):
        out["content"] = m.content
    else:
        # Multi-block: collapse to OpenAI's array shape.
        parts: list[dict[str, Any]] = []
        for block in m.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append({"type": "text", "text": block.text})  # type: ignore[attr-defined]
            elif block_type == "image":
                parts.append(  # type: ignore[arg-type]
                    {
                        "type": "image_url",
                        "image_url": {"url": block.url or ""},  # type: ignore[attr-defined]
                    }
                )
            elif block_type == "tool_result":
                # OpenAI represents tool results as separate role=tool messages —
                # this should normally be flattened by the caller, but if it isn't
                # we degrade gracefully by serializing as text.
                content = block.content if isinstance(block.content, str) else str(block.content)  # type: ignore[attr-defined]
                parts.append({"type": "text", "text": content})
        out["content"] = parts
    # Cost is None on raw provider responses; the Hub adds it.
    return out


def _normalize_finish_reason(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    if raw == "stop":
        return "stop"
    if raw == "length":
        return "length"
    if raw in ("tool_calls", "function_call"):
        return "tool_calls"
    if raw == "content_filter":
        return "content_filter"
    return "other"
