"""Public ``Hub`` and ``Model`` API.

This is what users actually import:

    >>> from relay import Hub
    >>> hub = Hub.from_yaml("models.yaml")
    >>> resp = await hub.chat("fast-cheap", messages=[{"role": "user", "content": "hi"}])
    >>> async for ev in hub.stream("fast-cheap", messages=[...]):
    ...     ...

The Hub composes:

* the parsed :class:`RelayConfig` (immutable registry of models, groups, profiles)
* one shared :class:`HttpClientManager` for connection pooling
* lazily-instantiated :class:`Provider` adapters (one per provider id)
* the :class:`PricingResolver` for cost computation
* the routing logic for groups
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from relay._internal.circuit_breaker import CircuitBreaker
from relay._internal.router import call_group
from relay._internal.transport import HttpClientManager
from relay.audit import AuditSink, build_event
from relay.cache import Cache, cache_key
from relay.catalog._pricing import PricingResolver
from relay.config import load as load_config
from relay.config._schema import ModelEntry, RelayConfig
from relay.errors import ConfigError
from relay.guardrails import Guardrail, GuardrailError, evaluate_post, evaluate_pre
from relay.providers import Provider, make_provider
from relay.redaction import Redactor
from relay.types import (
    ChatRequest,
    ChatResponse,
    Cost,
    Message,
    StreamEnd,
    StreamEvent,
)

if TYPE_CHECKING:
    pass


class Model:
    """A bound handle to one model alias.

    Use this in hot loops to skip the registry lookup on every call::

        model = hub.get("fast-cheap")
        for prompt in prompts:
            resp = await model.chat(messages=...)
    """

    def __init__(self, *, hub: Hub, alias: str, entry: ModelEntry) -> None:
        self._hub = hub
        self._alias = alias
        self._entry = entry

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def provider(self) -> str:
        return self._entry.provider

    @property
    def model_id(self) -> str:
        return self._entry.model_id

    @property
    def context_window(self) -> int | None:
        return self._hub._capability_for(self._entry, "context_window")  # type: ignore[return-value]

    @property
    def input_per_1m(self) -> float | None:
        return self._hub._snapshot_price_for(self._entry, "input_per_1m")

    @property
    def output_per_1m(self) -> float | None:
        return self._hub._snapshot_price_for(self._entry, "output_per_1m")

    async def chat(self, **kwargs: Any) -> ChatResponse:
        return await self._hub._chat_one(self._entry, **kwargs)

    def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        return self._hub._stream_one(self._entry, **kwargs)


class Hub:
    """The main public entry point.

    Construct via :meth:`from_yaml` (most common) or :meth:`from_config`. Always
    close the hub when done — it owns long-lived HTTP connections::

        async with Hub.from_yaml("models.yaml") as hub:
            resp = await hub.chat("fast-cheap", messages=...)
    """

    def __init__(
        self,
        config: RelayConfig,
        *,
        cache: Cache | None = None,
        redactor: Redactor | None = None,
        guardrails: list[Guardrail] | None = None,
        audit_sinks: list[AuditSink] | None = None,
    ) -> None:
        self._config = config
        self._clients = HttpClientManager(config.defaults)
        self._providers: dict[str, Provider] = {}
        self._pricing = PricingResolver(
            settings=config.catalog,
            pricing_profiles=config.pricing_profiles,
        )
        self._breaker = CircuitBreaker()
        self._cache: Cache | None = cache
        self._mcp: Any = None
        self._redactor: Redactor | None = redactor
        self._guardrails: list[Guardrail] = list(guardrails or [])
        self._audit_sinks: list[AuditSink] = list(audit_sinks or [])
        from relay.batch import BatchManager

        self.batch = BatchManager(self)
        """Batch operations namespace. See :class:`relay.batch.BatchManager`."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        *paths: str | Path,
        cache: Cache | None = None,
        redactor: Redactor | None = None,
        guardrails: list[Guardrail] | None = None,
        audit_sinks: list[AuditSink] | None = None,
    ) -> Hub:
        """Load YAML config from one or more paths and build a Hub.

        When multiple paths are given, later files override earlier ones
        (deep-merged). Useful for ``base.yaml`` + ``prod.yaml``.
        """
        return cls(
            load_config(*paths),
            cache=cache,
            redactor=redactor,
            guardrails=guardrails,
            audit_sinks=audit_sinks,
        )

    @classmethod
    def from_config(
        cls,
        config: RelayConfig,
        *,
        cache: Cache | None = None,
        redactor: Redactor | None = None,
        guardrails: list[Guardrail] | None = None,
        audit_sinks: list[AuditSink] | None = None,
    ) -> Hub:
        return cls(
            config,
            cache=cache,
            redactor=redactor,
            guardrails=guardrails,
            audit_sinks=audit_sinks,
        )

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Hub:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for p in self._providers.values():
            await p.aclose()
        await self._clients.aclose()
        if self._cache is not None:
            await self._cache.aclose()
        for sink in self._audit_sinks:
            try:
                await sink.aclose()
            except Exception:  # noqa: S112
                continue

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> RelayConfig:
        return self._config

    def list_aliases(self) -> list[str]:
        return sorted({*self._config.models, *self._config.groups})

    def attach_mcp(self, manager: Any) -> None:
        """Attach an :class:`relay.mcp.MCPManager` to this hub.

        Once attached, ``hub.mcp_tools()`` returns the cross-server tool list
        ready to pass as ``tools=`` to :meth:`chat`. ``hub.dispatch_tool_call(...)``
        runs a tool back through MCP and returns the textual result.
        """
        self._mcp = manager

    async def mcp_tools(self) -> list[Any]:
        """Convenience: return the merged tool list from the attached MCP manager."""
        if self._mcp is None:
            return []
        return await self._mcp.list_tools()

    async def dispatch_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call back to the attached MCP manager."""
        if self._mcp is None:
            raise ConfigError("no MCP manager attached; call hub.attach_mcp(manager) first")
        return (
            await self._mcp.dispatch(name, arguments)
            if hasattr(self._mcp, "dispatch")
            else await self._mcp.call_tool(name, arguments)
        )

    def get(self, alias: str) -> Model:
        """Return a bound :class:`Model` handle for an alias.

        Group aliases are *not* returned as ``Model`` handles in v0.1; call
        :meth:`chat` / :meth:`stream` with the group name instead. Trying to
        ``get()`` a group alias raises ``ConfigError``.
        """
        entry = self._config.models.get(alias)
        if entry is None:
            if alias in self._config.groups:
                raise ConfigError(
                    f"{alias!r} is a group, not a model. Call hub.chat({alias!r}, ...) directly."
                )
            raise ConfigError(f"unknown alias: {alias!r}")
        return Model(hub=self, alias=alias, entry=entry)

    # ------------------------------------------------------------------
    # call paths
    # ------------------------------------------------------------------

    async def chat(
        self,
        alias: str,
        *,
        messages: list[Message] | list[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatResponse:
        """Call a model or group by alias and return the assembled response."""
        normalized_messages = _coerce_messages(messages)
        if alias in self._config.models:
            return await self._chat_one(
                self._config.models[alias], messages=normalized_messages, **kwargs
            )
        if alias in self._config.groups:
            group = self._config.groups[alias]

            async def call_one(member_alias: str) -> ChatResponse:
                if member_alias in self._config.groups:
                    return await self.chat(member_alias, messages=normalized_messages, **kwargs)
                entry = self._config.models[member_alias]
                return await self._chat_one(entry, messages=normalized_messages, **kwargs)

            return await call_group(
                group=group,
                call_one=call_one,
                max_retries=self._config.defaults.max_retries,
                initial_backoff=self._config.defaults.retry_initial_backoff,
                max_backoff=self._config.defaults.retry_max_backoff,
                breaker=self._breaker,
            )
        raise ConfigError(f"unknown alias: {alias!r}")

    def stream(
        self,
        alias: str,
        *,
        messages: list[Message] | list[Mapping[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from a single model. Group streaming with fallback is v0.2.

        Returns an async iterator directly — callers use ``async for ev in
        hub.stream(...)`` without an ``await``.
        """
        normalized_messages = _coerce_messages(messages)
        if alias in self._config.groups:
            raise ConfigError(
                f"{alias!r} is a group; group streaming with fallback is planned for v0.2. "
                f"Stream a specific model instead."
            )
        if alias not in self._config.models:
            raise ConfigError(f"unknown alias: {alias!r}")
        entry = self._config.models[alias]
        return self._stream_one(entry, messages=normalized_messages, **kwargs)

    # ------------------------------------------------------------------
    # internal: one-model paths
    # ------------------------------------------------------------------

    async def _chat_one(
        self,
        entry: ModelEntry,
        *,
        messages: list[Message] | list[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        request = self._build_request(messages=messages, stream=False, **kwargs)

        # Redaction (pre-call). Modifies the message list in place on the request.
        redaction_count = 0
        redaction_kinds: tuple[str, ...] = ()
        if self._redactor is not None:
            result = self._redactor.redact(request.messages)
            request = request.model_copy(update={"messages": result.messages})
            redaction_count = result.redactions
            redaction_kinds = result.matched_kinds

        # Pre-call guardrails.
        violation = evaluate_pre(self._guardrails, request.messages)
        if violation is not None:
            err = GuardrailError(violation)
            await self._emit_audit(
                operation="chat",
                entry=entry,
                messages=request.messages,
                response=None,
                error=err,
                duration_ms=None,
                redaction_count=redaction_count,
                redaction_kinds=redaction_kinds,
                user_id=_extract_user_id(kwargs),
            )
            raise err

        cached_resp: ChatResponse | None = None
        ck: str | None = None
        if self._cache is not None:
            ck = cache_key(entry.target, request)
            cached_resp = await self._cache.get(ck)
            if cached_resp is not None:
                await self._emit_audit(
                    operation="chat",
                    entry=entry,
                    messages=request.messages,
                    response=cached_resp,
                    error=None,
                    duration_ms=0.0,
                    redaction_count=redaction_count,
                    redaction_kinds=redaction_kinds,
                    user_id=_extract_user_id(kwargs),
                )
                return cached_resp

        provider = self._get_provider(entry.provider, api_style=entry.api_style)
        try:
            resp = await provider.chat(entry=entry, request=request, clients=self._clients)
            priced = await self._apply_cost(resp, entry)
        except Exception as e:
            await self._emit_audit(
                operation="chat",
                entry=entry,
                messages=request.messages,
                response=None,
                error=e,
                duration_ms=None,
                redaction_count=redaction_count,
                redaction_kinds=redaction_kinds,
                user_id=_extract_user_id(kwargs),
            )
            raise

        # Post-call guardrails.
        post_violation = evaluate_post(self._guardrails, priced)
        if post_violation is not None:
            err = GuardrailError(post_violation)
            await self._emit_audit(
                operation="chat",
                entry=entry,
                messages=request.messages,
                response=priced,
                error=err,
                duration_ms=priced.latency_ms,
                redaction_count=redaction_count,
                redaction_kinds=redaction_kinds,
                user_id=_extract_user_id(kwargs),
            )
            raise err

        if self._cache is not None and ck is not None:
            await self._cache.set(ck, priced)

        await self._emit_audit(
            operation="chat",
            entry=entry,
            messages=request.messages,
            response=priced,
            error=None,
            duration_ms=priced.latency_ms,
            redaction_count=redaction_count,
            redaction_kinds=redaction_kinds,
            user_id=_extract_user_id(kwargs),
        )
        return priced

    def _stream_one(
        self,
        entry: ModelEntry,
        *,
        messages: list[Message] | list[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        request = self._build_request(messages=messages, stream=True, **kwargs)
        provider = self._get_provider(entry.provider, api_style=entry.api_style)

        async def gen() -> AsyncIterator[StreamEvent]:
            async for ev in provider.stream(  # type: ignore[attr-defined]
                entry=entry, request=request, clients=self._clients
            ):
                if isinstance(ev, StreamEnd):
                    priced = await self._apply_cost(ev.response, entry)
                    yield StreamEnd(finish_reason=ev.finish_reason, response=priced)
                else:
                    yield ev

        return gen()

    # ------------------------------------------------------------------
    # internal: helpers
    # ------------------------------------------------------------------

    def _build_request(
        self,
        *,
        messages: list[Message] | list[Mapping[str, Any]] | None,
        stream: bool,
        **kwargs: Any,
    ) -> ChatRequest:
        if messages is None:
            raise ConfigError("messages is required")
        return ChatRequest(
            messages=_coerce_messages(messages),
            stream=stream,
            **kwargs,
        )

    async def _emit_audit(
        self,
        *,
        operation: Literal["chat", "stream"],
        entry: ModelEntry,
        messages: list[Message],
        response: ChatResponse | None,
        error: BaseException | None,
        duration_ms: float | None,
        redaction_count: int,
        redaction_kinds: tuple[str, ...],
        user_id: str | None,
    ) -> None:
        if not self._audit_sinks:
            return
        ev = build_event(
            operation=operation,
            alias=entry.target,
            provider=entry.provider,
            model_id=entry.model_id,
            messages=messages,
            response=response,
            error=error,
            duration_ms=duration_ms,
            capture_messages=self._config.observability.capture_messages,
            redaction_count=redaction_count,
            redaction_kinds=redaction_kinds,
            user_id=user_id,
        )
        for sink in self._audit_sinks:
            try:
                await sink.emit(ev)
            except Exception:  # noqa: S112
                continue

    def _get_provider(self, name: str, *, api_style: str | None = None) -> Provider:
        # OpenAI api_style="responses" routes to the dedicated responses adapter
        # without needing a separate provider id in the user's YAML.
        if name == "openai" and api_style == "responses":
            key = "openai-responses"
            if key not in self._providers:
                from relay.providers.openai_responses import OpenAIResponsesProvider

                self._providers[key] = OpenAIResponsesProvider()
            return self._providers[key]
        if name not in self._providers:
            self._providers[name] = make_provider(name)
        return self._providers[name]

    async def _apply_cost(self, resp: ChatResponse, entry: ModelEntry) -> ChatResponse:
        if resp.usage.input_tokens == 0 and resp.usage.output_tokens == 0:
            return resp
        rates = await self._pricing.resolve(entry)
        if not rates.is_complete():
            return resp.model_copy(
                update={
                    "cost": Cost(
                        total_usd=0.0,
                        source=rates.source,
                        fetched_at=rates.fetched_at,
                        confidence="unknown",
                    )
                }
            )

        cached = resp.usage.cached_input_tokens
        non_cached = max(resp.usage.input_tokens - cached, 0)
        input_usd = (non_cached / 1_000_000) * (rates.input_per_1m or 0.0)
        cached_usd = (cached / 1_000_000) * (
            rates.cached_input_per_1m
            if rates.cached_input_per_1m is not None
            else (rates.input_per_1m or 0.0)
        )
        output_usd = (resp.usage.output_tokens / 1_000_000) * (rates.output_per_1m or 0.0)
        reasoning_usd = (resp.usage.reasoning_tokens / 1_000_000) * (rates.output_per_1m or 0.0)
        total = input_usd + cached_usd + output_usd + reasoning_usd

        cost = Cost(
            total_usd=total,
            input_usd=input_usd,
            output_usd=output_usd,
            cached_input_usd=cached_usd,
            reasoning_usd=reasoning_usd,
            source=rates.source,
            fetched_at=rates.fetched_at,
            confidence=rates.confidence,
        )
        return resp.model_copy(update={"cost": cost})

    def _capability_for(self, entry: ModelEntry, key: str) -> Any:
        # User override wins.
        v = getattr(entry.capabilities, key, None)
        if v is not None:
            return v
        # Catalog snapshot.
        from relay.catalog import lookup

        model_id = entry.inherit_from.split("/", 1)[1] if entry.inherit_from else entry.model_id
        row = lookup(entry.provider, model_id)
        if row is None:
            return None
        return getattr(row, key, None)

    def _snapshot_price_for(self, entry: ModelEntry, key: str) -> float | None:
        if entry.cost and key in entry.cost:
            return float(entry.cost[key])
        from relay.catalog import lookup

        model_id = entry.inherit_from.split("/", 1)[1] if entry.inherit_from else entry.model_id
        row = lookup(entry.provider, model_id)
        if row is None:
            return None
        return getattr(row, key, None)


def _extract_user_id(kwargs: dict[str, Any]) -> str | None:
    md = kwargs.get("metadata") or {}
    if isinstance(md, dict):
        uid = md.get("user_id") or md.get("user")
        if isinstance(uid, str):
            return uid
    return None


def _coerce_messages(
    messages: list[Message] | list[Mapping[str, Any]] | None,
) -> list[Message]:
    if messages is None:
        return []
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, Mapping):
            out.append(Message.model_validate(m))
        else:
            raise ConfigError(f"invalid message type: {type(m).__name__}")
    return out


_ = time  # keep ``time`` import live for downstream consumers that re-export
