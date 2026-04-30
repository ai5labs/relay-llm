"""Vertex AI adapter (Google Cloud).

Wire format is the same as direct Gemini, but:

* URL pattern: ``{location}-aiplatform.googleapis.com/v1/projects/{project}/``
  ``locations/{location}/publishers/google/models/{model}:generateContent``
* Auth: GCP Application Default Credentials (or service-account JSON), token
  refresh handled by ``google-auth``. We delegate to the official library when
  installed; fall back to a clear error otherwise.

Subclasses :class:`GoogleProvider` to reuse the body-builder, parser, and
streaming loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from relay.errors import (
    AuthenticationError,
    ConfigError,
    TimeoutError,
)
from relay.providers.google import GoogleProvider
from relay.types import (
    ChatRequest,
    ChatResponse,
    StreamEvent,
)

if TYPE_CHECKING:
    from relay._internal.transport import HttpClientManager
    from relay.config._schema import ModelEntry


class VertexProvider(GoogleProvider):
    """Vertex AI — Gemini wire format, GCP auth, regional endpoint."""

    name = "vertex"

    def __init__(self) -> None:
        super().__init__()
        # Token cache: (project, location) -> (token, expiry).
        self._token_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._token_lock = asyncio.Lock()

    async def chat(
        self,
        *,
        entry: ModelEntry,
        request: ChatRequest,
        clients: HttpClientManager,
    ) -> ChatResponse:
        project, location = self._require_project_location(entry)
        token = await self._get_token(project)
        base_url = entry.base_url or f"https://{location}-aiplatform.googleapis.com"
        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
            extra_headers={"authorization": f"Bearer {token}"},
        )

        body = self._build_body(entry, request)
        url = (
            f"/v1/projects/{project}/locations/{location}/publishers/google/models/"
            f"{entry.model_id}:generateContent"
        )

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=body)
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Vertex request timed out", provider=self.name, model=entry.model_id
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
        # Reuse Gemini parent stream logic with our own URL/auth.
        # We can't simply call super().stream() because the URL differs.
        project, location = self._require_project_location(entry)
        token = await self._get_token(project)
        base_url = entry.base_url or f"https://{location}-aiplatform.googleapis.com"
        client = await clients.get(
            provider=self.name,
            base_url=base_url,
            timeout=entry.timeout,
            extra_headers={"authorization": f"Bearer {token}"},
        )

        body = self._build_body(entry, request)
        url = (
            f"/v1/projects/{project}/locations/{location}/publishers/google/models/"
            f"{entry.model_id}:streamGenerateContent"
        )

        # Reproduce the Gemini stream loop with the new URL.
        import uuid

        import orjson

        from relay.providers.google import _map_finish, _parse_usage
        from relay.types import (
            Choice,
            Message,
            StreamEnd,
            StreamStart,
            TextDelta,
            ToolCall,
            Usage,
            UsageDelta,
        )

        text_buf: list[str] = []
        usage = Usage()
        finish_reason: str | None = None
        provider_model = entry.model_id
        tool_call_args: dict[int, dict[str, Any]] = {}
        start = time.perf_counter()

        try:
            async with client.stream("POST", url, json=body, params={"alt": "sse"}) as resp:
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
                "Vertex stream timed out", provider=self.name, model=entry.model_id
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
    # auth
    # ------------------------------------------------------------------

    def _require_project_location(self, entry: ModelEntry) -> tuple[str, str]:
        if not entry.project:
            raise AuthenticationError(
                "Vertex requires 'project' on the model entry",
                provider=self.name,
                model=entry.model_id,
            )
        location = entry.location or entry.region or "us-central1"
        return entry.project, location

    async def _get_token(self, project: str) -> str:
        """Get an access token via Google ADC.

        Uses :mod:`google.auth` if installed; otherwise raises a config error.
        Tokens are cached in-process for ~50 minutes (vs 60-min lifetime).
        """
        cache_key = (project, "default")
        now = time.time()
        cached = self._token_cache.get(cache_key)
        if cached and cached[1] > now:
            return cached[0]

        async with self._token_lock:
            cached = self._token_cache.get(cache_key)
            if cached and cached[1] > now:
                return cached[0]
            try:
                import google.auth  # type: ignore[import-untyped,import-not-found]
                import google.auth.transport.requests  # type: ignore[import-untyped,import-not-found]
            except ImportError as e:
                raise ConfigError(
                    "Vertex AI requires google-auth. Install with: pip install relayllm[gcp]"
                ) from e

            # google-auth is sync; offload to the default executor.
            def _refresh() -> str:
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                creds.refresh(google.auth.transport.requests.Request())
                return creds.token  # type: ignore[no-any-return]

            token = await asyncio.get_event_loop().run_in_executor(None, _refresh)
            # 50 min cache.
            self._token_cache[cache_key] = (token, now + 3000)
            return token
