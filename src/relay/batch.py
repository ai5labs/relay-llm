"""Batch API wrapper.

OpenAI Batch and Anthropic Message Batches both offer ~50% cost discount on
asynchronously-processed requests with a 24-hour SLA. The integrations are
otherwise awkward — different endpoints, different file/JSONL formats, different
status fields. Relay normalizes them.

Usage::

    handle = await hub.batch.submit("smart", requests=[
        {"messages": [{"role": "user", "content": "q1"}]},
        {"messages": [{"role": "user", "content": "q2"}]},
    ])
    print(handle.id, handle.status)

    # Later — poll, or just await completion:
    results = await hub.batch.results(handle)

Limitations
-----------
* OpenAI Batch and Anthropic Message Batches only — Bedrock has its own batch
  inference workflow (deferred to v0.3).
* Tool calls and structured output supported.
* Cost computation runs through the same pricing tier as live calls.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import httpx
import orjson

from relay._internal.credentials import resolve_secret
from relay.errors import (
    AuthenticationError,
    ConfigError,
    ProviderError,
    RelayError,
)
from relay.types import ChatRequest, ChatResponse, Message, Usage

if TYPE_CHECKING:
    from relay.config._schema import ModelEntry
    from relay.hub import Hub


BatchStatus = Literal[
    "queued",
    "in_progress",
    "finalizing",
    "completed",
    "failed",
    "cancelled",
    "expired",
]


@dataclass(frozen=True, slots=True)
class BatchHandle:
    """An opaque handle for a submitted batch."""

    id: str
    """Provider-side batch id."""
    provider: str
    alias: str
    submitted_at: float
    request_count: int


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Status snapshot of a batch."""

    id: str
    status: BatchStatus
    completed: int
    failed: int
    total: int
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """One result row from a completed batch."""

    custom_id: str
    response: ChatResponse | None
    error: str | None = None


class BatchManager:
    """Hub-attached batch operations namespace.

    Constructed by :class:`Hub` and exposed as ``hub.batch``.
    """

    def __init__(self, hub: Hub) -> None:
        self._hub = hub

    async def submit(
        self,
        alias: str,
        *,
        requests: list[dict[str, Any]],
        completion_window: str = "24h",
    ) -> BatchHandle:
        """Submit a batch of chat requests.

        :param alias: A model alias from the YAML — must resolve to a single
            entry, not a group.
        :param requests: List of chat-request kwargs (``messages``,
            ``temperature``, etc.). Each gets its own ``custom_id`` automatically;
            override by including ``custom_id`` in the dict.
        :param completion_window: Provider-specific (OpenAI: ``"24h"``,
            Anthropic: ignored).
        """
        entry = self._resolve_entry(alias)
        if entry.provider in (
            "openai",
            "azure",
            "groq",
            "deepseek",
            "xai",
            "mistral",
            "fireworks",
            "perplexity",
            "openrouter",
        ):
            return await self._submit_openai(entry, alias, requests, completion_window)
        if entry.provider == "anthropic":
            return await self._submit_anthropic(entry, alias, requests)
        raise ConfigError(
            f"batch API not supported for provider {entry.provider!r} yet "
            f"(supported: openai-compat family, anthropic)"
        )

    async def status(self, handle: BatchHandle) -> BatchProgress:
        entry = self._resolve_entry(handle.alias)
        if handle.provider == "anthropic":
            return await self._status_anthropic(entry, handle)
        return await self._status_openai(entry, handle)

    async def cancel(self, handle: BatchHandle) -> BatchProgress:
        entry = self._resolve_entry(handle.alias)
        if handle.provider == "anthropic":
            return await self._cancel_anthropic(entry, handle)
        return await self._cancel_openai(entry, handle)

    async def results(self, handle: BatchHandle) -> list[BatchResult]:
        """Fetch and parse all results for a completed batch.

        Raises if the batch isn't yet complete — call :meth:`status` first or
        use :meth:`wait` to block until done.
        """
        entry = self._resolve_entry(handle.alias)
        if handle.provider == "anthropic":
            return await self._results_anthropic(entry, handle)
        return await self._results_openai(entry, handle)

    # ------------------------------------------------------------------
    # OpenAI Batch
    # ------------------------------------------------------------------

    async def _submit_openai(
        self,
        entry: ModelEntry,
        alias: str,
        requests: list[dict[str, Any]],
        completion_window: str,
    ) -> BatchHandle:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError(
                "Batch submission requires a credential", provider=entry.provider
            )

        # 1. Build a JSONL file body.
        lines: list[bytes] = []
        for i, req in enumerate(requests):
            custom_id = req.get("custom_id") or f"req-{i}"
            req_body = self._build_chat_body(entry, req)
            lines.append(
                orjson.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": req_body,
                    }
                )
            )
        jsonl = b"\n".join(lines) + b"\n"

        base_url = entry.base_url or self._default_base_url(entry.provider)
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=60.0,
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            # 2. Upload as a batch input file.
            files = {"file": ("batch.jsonl", jsonl, "application/jsonl")}
            data = {"purpose": "batch"}
            file_resp = await client.post("/files", files=files, data=data)
            if file_resp.status_code >= 400:
                raise ProviderError(
                    f"file upload failed: {file_resp.text[:500]}",
                    provider=entry.provider,
                    status_code=file_resp.status_code,
                )
            file_id = file_resp.json()["id"]

            # 3. Create the batch.
            batch_resp = await client.post(
                "/batches",
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": completion_window,
                },
            )
            if batch_resp.status_code >= 400:
                raise ProviderError(
                    f"batch create failed: {batch_resp.text[:500]}",
                    provider=entry.provider,
                    status_code=batch_resp.status_code,
                )
            batch_data = batch_resp.json()

        return BatchHandle(
            id=batch_data["id"],
            provider=entry.provider,
            alias=alias,
            submitted_at=time.time(),
            request_count=len(requests),
        )

    async def _status_openai(self, entry: ModelEntry, handle: BatchHandle) -> BatchProgress:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or self._default_base_url(entry.provider),
            timeout=30.0,
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            resp = await client.get(f"/batches/{handle.id}")
            if resp.status_code >= 400:
                raise ProviderError(
                    f"batch status failed: {resp.text[:500]}",
                    provider=entry.provider,
                    status_code=resp.status_code,
                )
            data = resp.json()
        counts = data.get("request_counts") or {}
        return BatchProgress(
            id=data["id"],
            status=_normalize_openai_status(data.get("status", "")),
            completed=int(counts.get("completed", 0)),
            failed=int(counts.get("failed", 0)),
            total=int(counts.get("total", handle.request_count)),
            expires_at=float(data.get("expires_at")) if data.get("expires_at") else None,
        )

    async def _cancel_openai(self, entry: ModelEntry, handle: BatchHandle) -> BatchProgress:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or self._default_base_url(entry.provider),
            timeout=30.0,
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            resp = await client.post(f"/batches/{handle.id}/cancel")
            if resp.status_code >= 400:
                raise ProviderError(
                    f"batch cancel failed: {resp.text[:500]}",
                    provider=entry.provider,
                    status_code=resp.status_code,
                )
        return await self._status_openai(entry, handle)

    async def _results_openai(self, entry: ModelEntry, handle: BatchHandle) -> list[BatchResult]:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or self._default_base_url(entry.provider),
            timeout=60.0,
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            status_resp = await client.get(f"/batches/{handle.id}")
            if status_resp.status_code >= 400:
                raise ProviderError(
                    "batch status failed",
                    provider=entry.provider,
                    status_code=status_resp.status_code,
                )
            status_data = status_resp.json()
            output_file_id = status_data.get("output_file_id")
            if not output_file_id:
                raise RelayError(
                    f"batch {handle.id!r} not yet complete (status={status_data.get('status')!r})"
                )
            file_resp = await client.get(f"/files/{output_file_id}/content")
            if file_resp.status_code >= 400:
                raise ProviderError(
                    "batch result file fetch failed",
                    provider=entry.provider,
                    status_code=file_resp.status_code,
                )

        results: list[BatchResult] = []
        for line in file_resp.text.splitlines():
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            custom_id = row.get("custom_id", "")
            err = row.get("error")
            if err:
                results.append(BatchResult(custom_id=custom_id, response=None, error=str(err)))
                continue
            resp_body = (row.get("response") or {}).get("body") or {}
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    response=_openai_body_to_chat_response(resp_body, entry, handle),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Anthropic Message Batches
    # ------------------------------------------------------------------

    async def _submit_anthropic(
        self, entry: ModelEntry, alias: str, requests: list[dict[str, Any]]
    ) -> BatchHandle:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        if not api_key:
            raise AuthenticationError("Anthropic batch requires a credential", provider="anthropic")

        items: list[dict[str, Any]] = []
        for i, req in enumerate(requests):
            custom_id = req.get("custom_id") or f"req-{i}"
            params = self._build_anthropic_body(entry, req)
            items.append({"custom_id": custom_id, "params": params})

        async with httpx.AsyncClient(
            base_url=entry.base_url or "https://api.anthropic.com",
            timeout=60.0,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        ) as client:
            resp = await client.post("/v1/messages/batches", json={"requests": items})
            if resp.status_code >= 400:
                raise ProviderError(
                    f"anthropic batch create failed: {resp.text[:500]}",
                    provider="anthropic",
                    status_code=resp.status_code,
                )
            data = resp.json()
        return BatchHandle(
            id=data["id"],
            provider="anthropic",
            alias=alias,
            submitted_at=time.time(),
            request_count=len(requests),
        )

    async def _status_anthropic(self, entry: ModelEntry, handle: BatchHandle) -> BatchProgress:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or "https://api.anthropic.com",
            timeout=30.0,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        ) as client:
            resp = await client.get(f"/v1/messages/batches/{handle.id}")
            if resp.status_code >= 400:
                raise ProviderError(
                    f"anthropic batch status failed: {resp.text[:500]}",
                    provider="anthropic",
                    status_code=resp.status_code,
                )
            data = resp.json()
        counts = data.get("request_counts") or {}
        succ = int(counts.get("succeeded", 0))
        return BatchProgress(
            id=data["id"],
            status=_normalize_anthropic_status(data.get("processing_status", "")),
            completed=succ,
            failed=int(counts.get("errored", 0))
            + int(counts.get("canceled", 0))
            + int(counts.get("expired", 0)),
            total=succ
            + int(counts.get("processing", 0))
            + int(counts.get("errored", 0))
            + int(counts.get("canceled", 0))
            + int(counts.get("expired", 0)),
        )

    async def _cancel_anthropic(self, entry: ModelEntry, handle: BatchHandle) -> BatchProgress:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or "https://api.anthropic.com",
            timeout=30.0,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        ) as client:
            resp = await client.post(f"/v1/messages/batches/{handle.id}/cancel")
            if resp.status_code >= 400:
                raise ProviderError(
                    f"anthropic batch cancel failed: {resp.text[:500]}",
                    provider="anthropic",
                    status_code=resp.status_code,
                )
        return await self._status_anthropic(entry, handle)

    async def _results_anthropic(self, entry: ModelEntry, handle: BatchHandle) -> list[BatchResult]:
        api_key = await resolve_secret(entry.credential) if entry.credential else ""
        async with httpx.AsyncClient(
            base_url=entry.base_url or "https://api.anthropic.com",
            timeout=120.0,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        ) as client:
            resp = await client.get(f"/v1/messages/batches/{handle.id}/results")
            if resp.status_code >= 400:
                raise ProviderError(
                    f"anthropic batch results failed: {resp.text[:500]}",
                    provider="anthropic",
                    status_code=resp.status_code,
                )
            text = resp.text

        results: list[BatchResult] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            custom_id = row.get("custom_id", "")
            result = row.get("result") or {}
            rtype = result.get("type")
            if rtype != "succeeded":
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        response=None,
                        error=str(result.get("error") or rtype or "unknown"),
                    )
                )
                continue
            msg = result.get("message") or {}
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    response=_anthropic_body_to_chat_response(msg, entry, handle),
                )
            )
        return results

    # ------------------------------------------------------------------

    def _resolve_entry(self, alias: str) -> ModelEntry:
        entry = self._hub.config.models.get(alias)
        if entry is None:
            if alias in self._hub.config.groups:
                raise ConfigError(f"batch only accepts model aliases, not groups; got {alias!r}")
            raise ConfigError(f"unknown alias: {alias!r}")
        return entry

    @staticmethod
    def _default_base_url(provider: str) -> str:
        return {
            "openai": "https://api.openai.com/v1",
            "azure": "",  # caller must set base_url
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "xai": "https://api.x.ai/v1",
            "mistral": "https://api.mistral.ai/v1",
            "fireworks": "https://api.fireworks.ai/inference/v1",
            "perplexity": "https://api.perplexity.ai",
            "openrouter": "https://openrouter.ai/api/v1",
        }.get(provider, "https://api.openai.com/v1")

    def _build_chat_body(self, entry: ModelEntry, req: dict[str, Any]) -> dict[str, Any]:
        from relay.providers.openai_compat import _msg_to_openai

        messages = req.get("messages") or []
        msg_objs = [Message.model_validate(m) if isinstance(m, dict) else m for m in messages]
        body: dict[str, Any] = {
            "model": entry.model_id,
            "messages": [_msg_to_openai(m) for m in msg_objs],
        }
        for k in ("temperature", "top_p", "max_tokens", "stop", "seed", "response_format"):
            if k in req and req[k] is not None:
                body[k] = req[k]
        if req.get("tools"):
            from relay.tools import compile_all
            from relay.types import ToolDefinition

            tools = [
                ToolDefinition.model_validate(t) if isinstance(t, dict) else t for t in req["tools"]
            ]
            body["tools"] = compile_all(tools, entry.provider)
        return body

    def _build_anthropic_body(self, entry: ModelEntry, req: dict[str, Any]) -> dict[str, Any]:
        from relay.providers.anthropic import _msg_to_anthropic

        messages = req.get("messages") or []
        msg_objs = [Message.model_validate(m) if isinstance(m, dict) else m for m in messages]
        sys_parts: list[str] = []
        anth_msgs: list[dict[str, Any]] = []
        for m in msg_objs:
            if m.role == "system":
                if isinstance(m.content, str):
                    sys_parts.append(m.content)
                continue
            anth_msgs.append(_msg_to_anthropic(m))

        params: dict[str, Any] = {
            "model": entry.model_id,
            "messages": anth_msgs,
            "max_tokens": req.get("max_tokens") or entry.params.get("max_tokens") or 4096,
        }
        if sys_parts:
            params["system"] = "\n\n".join(sys_parts)
        for k in ("temperature", "top_p"):
            if k in req and req[k] is not None:
                params[k] = req[k]
        return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_openai_status(s: str) -> BatchStatus:
    return {
        "validating": "queued",
        "in_progress": "in_progress",
        "finalizing": "finalizing",
        "completed": "completed",
        "failed": "failed",
        "expired": "expired",
        "cancelling": "in_progress",
        "cancelled": "cancelled",
    }.get(s, "queued")  # type: ignore[return-value]


def _normalize_anthropic_status(s: str) -> BatchStatus:
    return {
        "in_progress": "in_progress",
        "canceling": "in_progress",
        "ended": "completed",
    }.get(s, "queued")  # type: ignore[return-value]


def _openai_body_to_chat_response(
    body: dict[str, Any], entry: ModelEntry, handle: BatchHandle
) -> ChatResponse:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(name=entry.provider, default_base_url="")
    return provider._parse_response(data=body, entry=entry, latency_ms=0.0)


def _anthropic_body_to_chat_response(
    body: dict[str, Any], entry: ModelEntry, handle: BatchHandle
) -> ChatResponse:
    from relay.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider()
    return provider._parse_response(data=body, entry=entry, latency_ms=0.0)


__all__ = [
    "BatchHandle",
    "BatchManager",
    "BatchProgress",
    "BatchResult",
    "BatchStatus",
]


def _silence_unused_imports() -> None:
    _ = (httpx, ChatRequest, Usage, uuid)
