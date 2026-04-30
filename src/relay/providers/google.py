"""Google Gemini direct API adapter (generativelanguage.googleapis.com).

This is the *native* Gemini client — not the OpenAI-compat shim, which loses
native server tools (Maps grounding, File Search, Computer Use) and forces
``reasoning_effort`` translation.

Auth: API key via the ``key=`` query parameter (the simplest auth Gemini supports
outside Vertex). Vertex AI's separate adapter handles ADC/service-account auth.

Wire format
-----------
* Messages → ``contents[]`` with ``role: "user"|"model"`` and ``parts[].text``.
* System messages → top-level ``systemInstruction.parts[].text``.
* Sampling params → ``generationConfig`` (``temperature``, ``topP``, ``maxOutputTokens``).
* Tools → ``tools[].functionDeclarations`` (compiled via ``relay.tools``).
* Streaming uses SSE with each chunk carrying a delta ``GenerateContentResponse``.
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
    Usage,
    UsageDelta,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


_DEFAULT_BASE = "https://generativelanguage.googleapis.com"


class GoogleProvider(BaseProvider):
    name = "google"

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
                "Gemini requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )

        body = self._build_body(entry, request)
        url = f"/v1beta/models/{entry.model_id}:generateContent"
        params = {"key": api_key}

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=body, params=params)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Gemini request timed out", provider=self.name, model=entry.model_id
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
                "Gemini requires a credential", provider=self.name, model=entry.model_id
            )
        client = await clients.get(
            provider=self.name,
            base_url=entry.base_url or _DEFAULT_BASE,
            timeout=entry.timeout,
        )

        body = self._build_body(entry, request)
        url = f"/v1beta/models/{entry.model_id}:streamGenerateContent"
        params = {"key": api_key, "alt": "sse"}
        start = time.perf_counter()

        text_buf: list[str] = []
        usage = Usage()
        finish_reason: str | None = None
        provider_model = entry.model_id
        tool_call_args: dict[int, dict[str, Any]] = {}

        try:
            async with client.stream("POST", url, json=body, params=params) as resp:
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
                    if not payload:
                        continue
                    try:
                        chunk = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue

                    candidates = chunk.get("candidates") or []
                    if candidates:
                        cand = candidates[0]
                        content = cand.get("content") or {}
                        for idx, part in enumerate(content.get("parts") or []):
                            if "text" in part:
                                text_buf.append(part["text"])
                                yield TextDelta(text=part["text"])
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                tool_call_args[idx] = {
                                    "id": fc.get("id") or f"call_{idx}",
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args") or {},
                                }
                        finish_reason = cand.get("finishReason") or finish_reason

                    usage_md = chunk.get("usageMetadata") or {}
                    if usage_md:
                        usage = _parse_usage(usage_md)
                        yield UsageDelta(usage=usage)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Gemini stream timed out", provider=self.name, model=entry.model_id
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000
        tool_calls = [
            ToolCall(id=v["id"], name=v["name"], arguments=v["args"])
            for _, v in sorted(tool_call_args.items())
        ]
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_buf)),
            finish_reason=_map_finish(finish_reason),
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
    # internals
    # ------------------------------------------------------------------

    def _build_body(self, entry: ModelEntry, request: ChatRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_text: list[str] = []
        for m in request.messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_text.append(m.content)
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": _content_to_gemini_parts(m.content)})

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_text)}]}

        gen_config: dict[str, Any] = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.stop is not None:
            gen_config["stopSequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.seed is not None:
            gen_config["seed"] = request.seed

        # Structured output: Gemini accepts a response_schema.
        if isinstance(request.response_format, dict):
            rf = request.response_format
            if rf.get("type") == "json_schema":
                schema = (rf.get("json_schema") or {}).get("schema") or {}
                gen_config["responseMimeType"] = "application/json"
                gen_config["responseSchema"] = schema
            elif rf.get("type") == "json_object":
                gen_config["responseMimeType"] = "application/json"

        # Reasoning unification
        if request.reasoning is not None:
            from relay._internal.reasoning import to_gemini

            tg = to_gemini(request.reasoning)
            if tg is not None:
                gen_config["thinkingConfig"] = tg

        # Per-model defaults
        for k, v in entry.params.items():
            gen_config.setdefault(k, v)

        if gen_config:
            body["generationConfig"] = gen_config

        if request.tools:
            from relay.tools import compile_all

            compiled = compile_all(request.tools, "google")
            body["tools"] = [{"functionDeclarations": compiled}]

        if request.tool_choice == "required":
            body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
        elif request.tool_choice == "none":
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        elif request.tool_choice == "auto":
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

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

        msg = "Gemini error"
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
        if resp.status_code == 429:
            raise RateLimitError(msg, retry_after=parse_retry_after(resp.headers), **kwargs)
        lower = msg.lower()
        if "context" in lower and ("length" in lower or "tokens" in lower):
            raise ContextWindowError(msg, **kwargs)
        raise ProviderError(msg, **kwargs)

    def _parse_response(
        self, *, data: dict[str, Any], entry: ModelEntry, latency_ms: float
    ) -> ChatResponse:
        candidates = data.get("candidates") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_raw: str | None = None
        if candidates:
            cand = candidates[0]
            finish_raw = cand.get("finishReason")
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=fc.get("id") or f"call_{len(tool_calls)}",
                            name=fc.get("name", ""),
                            arguments=fc.get("args") or {},
                        )
                    )

        usage = _parse_usage(data.get("usageMetadata") or {})
        choice = Choice(
            index=0,
            message=Message(role="assistant", content="".join(text_parts)),
            finish_reason=_map_finish(finish_raw),
            tool_calls=tool_calls,
        )
        return ChatResponse(
            id=data.get("responseId") or uuid.uuid4().hex,
            model=entry.target,
            provider_model=data.get("modelVersion", entry.model_id),
            provider=self.name,
            choices=[choice],
            usage=usage,
            cost=None,
            created_at=time.time(),
            latency_ms=latency_ms,
        )


def _content_to_gemini_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append({"text": block.text})  # type: ignore[attr-defined]
        elif btype == "image":
            url = getattr(block, "url", "") or ""
            media = getattr(block, "media_type", None) or "image/png"
            if url.startswith("data:"):
                _, _, b64 = url.partition(",")
                parts.append({"inlineData": {"mimeType": media, "data": b64}})
            else:
                parts.append({"fileData": {"fileUri": url, "mimeType": media}})
        elif btype == "tool_use":
            parts.append(
                {
                    "functionCall": {
                        "name": block.name,  # type: ignore[attr-defined]
                        "args": block.input,  # type: ignore[attr-defined]
                    }
                }
            )
        elif btype == "tool_result":
            inner = block.content if isinstance(block.content, str) else str(block.content)  # type: ignore[attr-defined]
            parts.append(
                {
                    "functionResponse": {
                        "name": "",
                        "response": {"output": inner},
                    }
                }
            )
    return parts


def _parse_usage(usage_md: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(usage_md.get("promptTokenCount", 0)),
        output_tokens=int(usage_md.get("candidatesTokenCount", 0)),
        cached_input_tokens=int(usage_md.get("cachedContentTokenCount", 0)),
        reasoning_tokens=int(usage_md.get("thoughtsTokenCount", 0)),
    )


def _map_finish(raw: str | None) -> FinishReason | None:
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "BLOCKLIST": "content_filter",
        "SPII": "content_filter",
        "MALFORMED_FUNCTION_CALL": "error",
    }
    return mapping.get(raw, "other")
