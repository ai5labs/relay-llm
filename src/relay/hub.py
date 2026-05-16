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

import asyncio
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
from relay.errors import TimeoutError as RelayTimeoutError
from relay.guardrails import Guardrail, GuardrailError, evaluate_post, evaluate_pre
from relay.providers import Provider, make_provider
from relay.redaction import Redactor
from relay.routing import (
    NoCandidatesError,
    RouteConstraints,
    Router,
    RouteRequest,
)
from relay.types import (
    ChatRequest,
    ChatResponse,
    Cost,
    Message,
    StreamEnd,
    StreamErrorEvent,
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
        self._router: Router | None = None
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

    def attach_router(self, router: Router) -> None:
        """Attach a :class:`relay.routing.Router` for use with :meth:`chat_routed`.

        Idempotent — calling again replaces the previously attached router.
        The hub does **not** take ownership: callers are responsible for
        closing routers that own external resources (e.g. ``SemanticRouter``)
        unless they want the lifetime tied to the hub, in which case they
        can close it manually after :meth:`aclose`.
        """
        self._router = router

    async def chat_routed(
        self,
        messages: list[Message] | list[Mapping[str, Any]],
        candidates: list[str] | None = None,
        constraints: RouteConstraints | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Route to a model via the attached :class:`Router`, then dispatch.

        If the chosen alias errors, falls through ``decision.alternates`` in
        order until one succeeds or the list is exhausted (in which case the
        last error is re-raised). The successful :class:`RouteDecision` is
        attached to the response under ``response.metadata['routing']``.

        Raises :class:`relay.errors.ConfigError` when no router is attached.
        When ``candidates`` is ``None``, defaults to every non-deprecated
        alias declared in the loaded YAML (groups are excluded).
        """
        if self._router is None:
            raise ConfigError(
                "No router attached. Call hub.attach_router() first."
            )

        normalized_messages = _coerce_messages(messages)
        if candidates is None:
            # Default: every model alias defined in the YAML config. We don't
            # include groups since downstream dispatch is per-alias.
            candidates = list(self._config.models.keys())

        request = RouteRequest(
            messages=normalized_messages,
            candidates=candidates,
            constraints=constraints,
        )
        decision = await self._router.route(request)

        order: list[str] = [decision.alias] + [a for a, _score in decision.alternates]
        last_error: Exception | None = None
        for alias in order:
            if alias not in self._config.models:
                # Decision returned an alias the hub doesn't know about (catalog
                # slug rather than YAML alias). Skip rather than crash — the
                # rule-based router can return catalog slugs in default-catalog
                # mode, which we don't try to resolve here.
                continue
            try:
                response = await self.chat(
                    alias, messages=normalized_messages, **kwargs
                )
            except Exception as e:
                last_error = e
                continue

            # Annotate the response with the routing decision. ChatResponse is
            # frozen but allows extras, so model_copy can stamp the metadata.
            existing_meta: dict[str, Any] = {}
            for src in (getattr(response, "metadata", None),):
                if isinstance(src, dict):
                    existing_meta.update(src)
            existing_meta["routing"] = decision
            return response.model_copy(update={"metadata": existing_meta})

        if last_error is not None:
            raise last_error
        raise NoCandidatesError(
            "router returned no candidate that matches a configured model alias"
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
        trust_system: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Call a model or group by alias and return the assembled response.

        Set ``trust_system=False`` when ``messages`` originates from
        untrusted user input. Any ``role="system"`` entry then raises
        :class:`relay.errors.ConfigError` — set the developer's system
        prompt in code instead of forwarding it through.
        """
        normalized_messages = _coerce_messages(messages, trust_system=trust_system)
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
        trust_system: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from a single model. Group streaming with fallback is v0.2.

        Returns an async iterator directly — callers use ``async for ev in
        hub.stream(...)`` without an ``await``.

        ``trust_system=False`` rejects any ``role="system"`` entry; see
        :meth:`chat` for the rationale.
        """
        normalized_messages = _coerce_messages(messages, trust_system=trust_system)
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

        # Redaction (pre-call). Mutates the request; we keep the pre-redaction
        # messages around to fold into the cache key so two users whose
        # distinct PII redacts to the same placeholder don't collide.
        redaction_count = 0
        redaction_kinds: tuple[str, ...] = ()
        pre_redaction_messages: list[Message] | None = None
        if self._redactor is not None:
            pre_redaction_messages = list(request.messages)
            result = self._redactor.redact(request.messages)
            request = request.model_copy(update={"messages": result.messages})
            redaction_count = result.redactions
            redaction_kinds = result.matched_kinds

        user_id = _extract_user_id(kwargs)

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
                user_id=user_id,
            )
            raise err

        cached_resp: ChatResponse | None = None
        ck: str | None = None
        if self._cache is not None:
            ck = cache_key(
                entry.target,
                request,
                scope=user_id,
                pre_redaction_messages=pre_redaction_messages,
            )
            cached_resp = await self._cache.get(ck)
            if cached_resp is not None:
                # Post-guardrails MUST run against the cached response: if a
                # rule was tightened (new banned term, new safety policy)
                # after the entry was cached, a stale-but-now-blocked
                # response would otherwise leak. Cheap — no provider call.
                cached_violation = evaluate_post(self._guardrails, cached_resp)
                if cached_violation is not None:
                    err = GuardrailError(cached_violation)
                    await self._emit_audit(
                        operation="chat",
                        entry=entry,
                        messages=request.messages,
                        response=cached_resp,
                        error=err,
                        duration_ms=0.0,
                        redaction_count=redaction_count,
                        redaction_kinds=redaction_kinds,
                        user_id=user_id,
                    )
                    raise err
                await self._emit_audit(
                    operation="chat",
                    entry=entry,
                    messages=request.messages,
                    response=cached_resp,
                    error=None,
                    duration_ms=0.0,
                    redaction_count=redaction_count,
                    redaction_kinds=redaction_kinds,
                    user_id=user_id,
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
                user_id=user_id,
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
                user_id=user_id,
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
            user_id=user_id,
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

        # Redaction (pre-call), mirroring _chat_one. Mutates the request so
        # downstream provider never sees the pre-redaction content.
        if self._redactor is not None:
            result = self._redactor.redact(request.messages)
            request = request.model_copy(update={"messages": result.messages})

        # Pre-call guardrails — raise *before* opening the SSE socket so
        # blocked prompts never reach the model.
        pre_violation = evaluate_pre(self._guardrails, request.messages)
        if pre_violation is not None:
            raise GuardrailError(pre_violation)

        # Wall-clock deadline. Per-entry timeout wins when explicitly set;
        # otherwise fall back to the global stream_overall_timeout. A
        # malicious slow-loris provider that emits one byte per N seconds
        # can otherwise stall forever inside provider.stream's aiter_lines.
        overall_deadline = (
            entry.timeout
            if entry.timeout is not None
            else self._config.defaults.stream_overall_timeout
        )

        async def gen() -> AsyncIterator[StreamEvent]:
            # Manual wall-clock deadline implemented with per-iteration
            # wait_for — asyncio.timeout would be cleaner but only landed in
            # Python 3.11, and Relay supports 3.10.
            inner = provider.stream(  # type: ignore[attr-defined]
                entry=entry, request=request, clients=self._clients
            ).__aiter__()
            deadline = time.monotonic() + overall_deadline
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RelayTimeoutError(
                        f"stream exceeded overall deadline of {overall_deadline}s",
                        provider=entry.provider,
                        model=entry.model_id,
                    )
                try:
                    ev = await asyncio.wait_for(inner.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError as e:
                    raise RelayTimeoutError(
                        f"stream exceeded overall deadline of {overall_deadline}s "
                        "(set defaults.stream_overall_timeout or per-model timeout)",
                        provider=entry.provider,
                        model=entry.model_id,
                    ) from e

                if isinstance(ev, StreamEnd):
                    priced = await self._apply_cost(ev.response, entry)
                    # Post-call guardrails on the assembled response. On block
                    # we replace the buffered text with a marker and emit a
                    # terminating StreamErrorEvent so the caller never
                    # receives blocked content as a final response.
                    post_violation = evaluate_post(self._guardrails, priced)
                    if post_violation is not None:
                        # Marker carries only the rule id — the violation
                        # message itself often quotes the blocked term back,
                        # which would defeat the redaction.
                        priced = _strip_blocked_text(priced, post_violation.rule)
                        yield StreamErrorEvent(
                            error=post_violation.message,
                            code=post_violation.rule,
                        )
                        yield StreamEnd(
                            finish_reason="content_filter",
                            response=priced,
                        )
                        return
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


def _strip_blocked_text(response: ChatResponse, reason: str) -> ChatResponse:
    """Replace the assistant text in every choice with a block marker.

    Called when post-guardrails fire on a streamed response: the caller has
    already received text deltas, but we still owe them a final response
    object that doesn't carry the blocked content.
    """
    marker = f"[blocked by guardrail: {reason}]"
    new_choices = []
    for ch in response.choices:
        new_message = ch.message.model_copy(update={"content": marker})
        new_choices.append(
            ch.model_copy(
                update={"message": new_message, "finish_reason": "content_filter"}
            )
        )
    return response.model_copy(update={"choices": new_choices})


def _extract_user_id(kwargs: dict[str, Any]) -> str | None:
    md = kwargs.get("metadata") or {}
    if isinstance(md, dict):
        uid = md.get("user_id") or md.get("user")
        if isinstance(uid, str):
            return uid
    return None


def _coerce_messages(
    messages: list[Message] | list[Mapping[str, Any]] | None,
    *,
    trust_system: bool = True,
) -> list[Message]:
    if messages is None:
        return []
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            msg = m
        elif isinstance(m, Mapping):
            msg = Message.model_validate(m)
        else:
            raise ConfigError(f"invalid message type: {type(m).__name__}")
        if not trust_system and msg.role == "system":
            raise ConfigError(
                "messages contain a role='system' entry but trust_system=False; "
                "set the system prompt in code rather than forwarding it from "
                "untrusted user input (or call hub.chat with trust_system=True "
                "if the messages list is fully under your control)."
            )
        out.append(msg)
    return out


_ = time  # keep ``time`` import live for downstream consumers that re-export
