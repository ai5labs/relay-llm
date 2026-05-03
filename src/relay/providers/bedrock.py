"""AWS Bedrock adapter via the Converse API.

The Converse API unifies tool-call semantics across Anthropic / Mistral / Cohere
/ Llama on Bedrock. We delegate to ``boto3`` for SigV4 signing and IAM auth —
re-implementing AWS request signing is out of scope.

Implementation notes
--------------------
* boto3 is **lazy-imported** in ``connect``; users without ``boto3`` installed
  get a clear ``ConfigError`` only when they try to actually use Bedrock.
* boto3 is sync — we run it in a thread executor to avoid blocking the loop.
* Streaming uses ``ConverseStream``, which yields a stream of typed events
  (``messageStart``, ``contentBlockDelta``, ``messageStop``, ``metadata``).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from relay.errors import (
    AuthenticationError,
    ConfigError,
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


class BedrockProvider(BaseProvider):
    name = "bedrock"

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        bedrock_client = await self._get_client(entry)
        body = _build_converse_body(entry, request)
        start = time.perf_counter()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: bedrock_client.converse(**body)
            )
        except Exception as e:
            _raise_classified(e, entry, self.name)
        latency_ms = (time.perf_counter() - start) * 1000
        return _parse_converse_response(response, entry, self.name, latency_ms)

    async def stream(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> AsyncIterator[StreamEvent]:
        bedrock_client = await self._get_client(entry)
        body = _build_converse_body(entry, request)
        start = time.perf_counter()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: bedrock_client.converse_stream(**body)
            )
        except Exception as e:
            _raise_classified(e, entry, self.name)

        yield StreamStart(id=uuid.uuid4().hex, model=entry.target, provider=self.name)

        text_buf: list[str] = []
        tool_blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        finish_reason: str | None = None
        provider_model = entry.model_id

        # boto3 EventStream is sync iterator — drain it in a thread.
        loop = asyncio.get_event_loop()
        event_iter = iter(response["stream"])

        def _next_event() -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        while True:
            event = await loop.run_in_executor(None, _next_event)
            if event is None:
                break

            if "messageStart" in event:
                pass
            elif "contentBlockStart" in event:
                idx = event["contentBlockStart"].get("contentBlockIndex", 0)
                start_data = event["contentBlockStart"].get("start") or {}
                if "toolUse" in start_data:
                    tu = start_data["toolUse"]
                    tool_blocks[idx] = {
                        "id": tu.get("toolUseId", ""),
                        "name": tu.get("name", ""),
                        "args_str": "",
                    }
                    yield ToolCallDelta(index=idx, id=tu.get("toolUseId"), name=tu.get("name"))
            elif "contentBlockDelta" in event:
                idx = event["contentBlockDelta"].get("contentBlockIndex", 0)
                delta = event["contentBlockDelta"].get("delta") or {}
                if "text" in delta:
                    text_buf.append(delta["text"])
                    yield TextDelta(text=delta["text"])
                elif "toolUse" in delta:
                    args_chunk = delta["toolUse"].get("input", "")
                    if idx in tool_blocks:
                        tool_blocks[idx]["args_str"] += args_chunk
                    yield ToolCallDelta(index=idx, arguments_delta=args_chunk)
            elif "messageStop" in event:
                finish_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                u = event["metadata"].get("usage") or {}
                usage = Usage(
                    input_tokens=int(u.get("inputTokens", 0)),
                    output_tokens=int(u.get("outputTokens", 0)),
                    cached_input_tokens=int(u.get("cacheReadInputTokens", 0)),
                )
                yield UsageDelta(usage=usage)

        latency_ms = (time.perf_counter() - start) * 1000
        import orjson

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_blocks):
            slot = tool_blocks[idx]
            args: dict[str, Any] = {}
            if slot["args_str"]:
                try:
                    args = orjson.loads(slot["args_str"])
                except orjson.JSONDecodeError:
                    args = {"_raw": slot["args_str"]}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))

        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_buf)),
            finish_reason=_map_stop_reason(finish_reason),
            tool_calls=tool_calls,
        )
        final = ChatResponse(
            id=uuid.uuid4().hex,
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

    async def _get_client(self, entry: ModelEntry) -> Any:
        try:
            import boto3  # type: ignore[import-not-found,import-untyped]
        except ImportError as e:
            raise ConfigError(
                "Bedrock requires 'boto3'. Install with: pip install ai5labs-relay[aws]"
            ) from e

        region = entry.region or "us-east-1"
        # Reuse a boto3 client per region.
        if region in self._clients:
            return self._clients[region]
        async with self._lock:
            if region in self._clients:
                return self._clients[region]
            kwargs: dict[str, Any] = {"region_name": region}
            cred = entry.credential
            if cred is not None and not isinstance(cred, str) and cred.type == "aws_profile":
                session = boto3.Session(
                    profile_name=getattr(cred, "profile", None),
                    region_name=getattr(cred, "region", None) or region,
                )
                client = session.client("bedrock-runtime")
                self._clients[region] = client
                return client
            client = boto3.client("bedrock-runtime", **kwargs)
            self._clients[region] = client
            return client


# ---------------------------------------------------------------------------
# Body builder + response parser
# ---------------------------------------------------------------------------


def _build_converse_body(entry: ModelEntry, request: ChatRequest) -> dict[str, Any]:
    system_blocks: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            if isinstance(m.content, str):
                system_blocks.append({"text": m.content})
            continue
        role = "assistant" if m.role == "assistant" else "user"
        messages.append({"role": role, "content": _content_to_bedrock(m.content)})

    body: dict[str, Any] = {
        "modelId": entry.model_id,
        "messages": messages,
    }
    if system_blocks:
        body["system"] = system_blocks

    inference: dict[str, Any] = {}
    if request.max_tokens is not None:
        inference["maxTokens"] = request.max_tokens
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    if request.top_p is not None:
        inference["topP"] = request.top_p
    if request.stop is not None:
        inference["stopSequences"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )
    if inference:
        body["inferenceConfig"] = inference

    if request.tools:
        from relay.tools import compile_all

        body["toolConfig"] = {"tools": compile_all(request.tools, "bedrock")}
        if request.tool_choice == "required":
            body["toolConfig"]["toolChoice"] = {"any": {}}
        elif request.tool_choice == "auto":
            body["toolConfig"]["toolChoice"] = {"auto": {}}
        elif isinstance(request.tool_choice, dict) and "tool" in request.tool_choice:
            body["toolConfig"]["toolChoice"] = {
                "tool": {"name": request.tool_choice["tool"].get("name", "")}
            }

    if request.reasoning is not None:
        from relay._internal.reasoning import to_anthropic

        rc = to_anthropic(request.reasoning)
        if rc is not None:
            body.setdefault("additionalModelRequestFields", {})["thinking"] = rc

    return body


def _content_to_bedrock(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append({"text": block.text})  # type: ignore[attr-defined]
        elif btype == "image":
            url = getattr(block, "url", "") or ""
            media = (getattr(block, "media_type", None) or "image/png").split("/")[-1]
            if url.startswith("data:"):
                import base64

                _, _, b64 = url.partition(",")
                parts.append(
                    {
                        "image": {
                            "format": media,
                            "source": {"bytes": base64.b64decode(b64)},
                        }
                    }
                )
        elif btype == "tool_use":
            parts.append(
                {
                    "toolUse": {
                        "toolUseId": block.id,  # type: ignore[attr-defined]
                        "name": block.name,  # type: ignore[attr-defined]
                        "input": block.input,  # type: ignore[attr-defined]
                    }
                }
            )
        elif btype == "tool_result":
            inner_content = block.content  # type: ignore[attr-defined]
            inner = (
                [{"text": inner_content}]
                if isinstance(inner_content, str)
                else [{"text": str(inner_content)}]
            )
            parts.append(
                {
                    "toolResult": {
                        "toolUseId": block.tool_use_id,  # type: ignore[attr-defined]
                        "content": inner,
                        **({"status": "error"} if getattr(block, "is_error", False) else {}),
                    }
                }
            )
    return parts


def _parse_converse_response(
    response: dict[str, Any], entry: ModelEntry, provider_name: str, latency_ms: float
) -> ChatResponse:
    output = response.get("output") or {}
    msg = output.get("message") or {}
    content_blocks = msg.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tu.get("toolUseId", ""),
                    name=tu.get("name", ""),
                    arguments=tu.get("input") or {},
                )
            )

    usage_in = response.get("usage") or {}
    usage = Usage(
        input_tokens=int(usage_in.get("inputTokens", 0)),
        output_tokens=int(usage_in.get("outputTokens", 0)),
        cached_input_tokens=int(usage_in.get("cacheReadInputTokens", 0)),
    )

    choice = Choice(
        index=0,
        message=Message(role="assistant", content="".join(text_parts)),
        finish_reason=_map_stop_reason(response.get("stopReason")),
        tool_calls=tool_calls,
    )
    return ChatResponse(
        id=response.get("ResponseMetadata", {}).get("RequestId") or uuid.uuid4().hex,
        model=entry.target,
        provider_model=entry.model_id,
        provider=provider_name,
        choices=[choice],
        usage=usage,
        cost=None,
        created_at=time.time(),
        latency_ms=latency_ms,
    )


def _map_stop_reason(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }
    return mapping.get(raw, "other")


def _raise_classified(exc: BaseException, entry: ModelEntry, provider_name: str) -> None:
    """Translate a boto3 ClientError into the Relay error hierarchy."""
    from botocore.exceptions import (  # type: ignore[import-untyped,import-not-found]
        BotoCoreError,
        ClientError,
    )

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        msg = exc.response.get("Error", {}).get("Message") or str(exc)
        kwargs = {"provider": provider_name, "model": entry.model_id, "raw": exc.response}
        if code in ("AccessDeniedException", "UnauthorizedException", "AuthFailure"):
            raise AuthenticationError(msg, **kwargs) from exc
        if code in (
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceQuotaExceededException",
        ):
            raise RateLimitError(msg, **kwargs) from exc
        if code == "ValidationException" and "context" in msg.lower():
            raise ContextWindowError(msg, **kwargs) from exc
        raise ProviderError(msg, **kwargs) from exc
    if isinstance(exc, BotoCoreError):
        raise ProviderError(
            f"Bedrock low-level error: {exc}", provider=provider_name, model=entry.model_id
        ) from exc
    if isinstance(exc, asyncio.TimeoutError):
        raise TimeoutError(
            "Bedrock request timed out", provider=provider_name, model=entry.model_id
        ) from exc
    # Last resort
    raise ProviderError(str(exc), provider=provider_name, model=entry.model_id) from exc
