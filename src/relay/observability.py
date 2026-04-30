"""OpenTelemetry GenAI semantic-conventions instrumentation.

Opt-in. Calling :func:`instrument` wraps a :class:`Hub` instance so every chat
call emits a span and metrics following the OpenTelemetry GenAI semantic
conventions (the standard Datadog, Honeycomb, Langfuse, Arize, and Phoenix all
consume).

Usage::

    from relay import Hub
    from relay.observability import instrument

    hub = Hub.from_yaml("models.yaml")
    instrument(hub, capture_messages="metadata_only")

    # Now every hub.chat() / hub.stream() call emits gen_ai.* spans.

Span attributes follow ``opentelemetry.io/docs/specs/semconv/gen-ai/`` and are
opt-in for sensitive content (``capture_messages`` defaults to ``metadata_only``
— prompt and response bodies are *not* captured unless explicitly enabled).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

from relay.errors import ConfigError
from relay.types import ChatResponse, StreamEnd, StreamEvent

if TYPE_CHECKING:
    from relay.config._schema import ModelEntry
    from relay.hub import Hub


CaptureMode = Literal["never", "metadata_only", "full"]


def instrument(hub: Hub, *, capture_messages: CaptureMode = "metadata_only") -> Hub:
    """Wrap a Hub so every chat/stream call emits OTel GenAI spans + metrics.

    :param hub: The hub to instrument. Mutated in place.
    :param capture_messages:
        - ``never``: only metadata on spans (token counts, cost, latency).
        - ``metadata_only``: also include message roles + count (default).
        - ``full``: include full message contents — sensitive, opt in deliberately.
    :return: The same hub, for chaining.
    :raises ConfigError: if the OTel SDK is not installed.
    """
    try:
        from opentelemetry import metrics, trace
    except ImportError as e:
        raise ConfigError(
            "OpenTelemetry instrumentation requires the 'otel' extra. "
            "Install with: pip install relayllm[otel]"
        ) from e

    tracer = trace.get_tracer("relay", "0.1.0")
    meter = metrics.get_meter("relay", "0.1.0")
    token_histogram = meter.create_histogram(
        name="gen_ai.client.token.usage",
        description="Number of tokens used in a request.",
        unit="{token}",
    )
    duration_histogram = meter.create_histogram(
        name="gen_ai.client.operation.duration",
        description="Duration of GenAI client operations.",
        unit="s",
    )
    cost_histogram = meter.create_histogram(
        name="gen_ai.client.cost",
        description="Cost in USD of GenAI client operations.",
        unit="USD",
    )

    original_chat_one = hub._chat_one
    original_stream_one = hub._stream_one

    async def _chat_one_instrumented(entry: ModelEntry, **kwargs: Any) -> ChatResponse:
        with tracer.start_as_current_span(f"chat {entry.target}") as span:
            _set_request_attrs(span, entry, kwargs, capture_messages)
            start = time.perf_counter()
            try:
                resp = await original_chat_one(entry, **kwargs)
            except Exception as e:
                span.set_attribute("error.type", type(e).__name__)
                span.record_exception(e)
                raise
            else:
                _set_response_attrs(span, resp, capture_messages)
                _emit_metrics(
                    token_histogram,
                    duration_histogram,
                    cost_histogram,
                    entry,
                    resp,
                    time.perf_counter() - start,
                )
                return resp

    def _stream_one_instrumented(entry: ModelEntry, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        async def gen() -> AsyncIterator[StreamEvent]:
            with tracer.start_as_current_span(f"chat {entry.target}") as span:
                _set_request_attrs(span, entry, kwargs, capture_messages)
                span.set_attribute("gen_ai.operation.name", "chat")
                start = time.perf_counter()
                final_resp: ChatResponse | None = None
                try:
                    async for ev in original_stream_one(entry, **kwargs):
                        if isinstance(ev, StreamEnd):
                            final_resp = ev.response
                        yield ev
                except Exception as e:
                    span.set_attribute("error.type", type(e).__name__)
                    span.record_exception(e)
                    raise
                else:
                    if final_resp is not None:
                        _set_response_attrs(span, final_resp, capture_messages)
                        _emit_metrics(
                            token_histogram,
                            duration_histogram,
                            cost_histogram,
                            entry,
                            final_resp,
                            time.perf_counter() - start,
                        )

        return gen()

    hub._chat_one = _chat_one_instrumented  # type: ignore[method-assign]
    hub._stream_one = _stream_one_instrumented  # type: ignore[method-assign]
    return hub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_request_attrs(
    span: Any, entry: ModelEntry, kwargs: dict[str, Any], capture: CaptureMode
) -> None:
    """Populate the request side of a GenAI span."""
    span.set_attribute("gen_ai.system", entry.provider)
    span.set_attribute("gen_ai.request.model", entry.model_id)
    span.set_attribute("gen_ai.operation.name", "chat")
    if entry.target:
        span.set_attribute("gen_ai.request.target", entry.target)

    for k in ("temperature", "top_p", "max_tokens", "seed"):
        v = kwargs.get(k)
        if v is not None:
            span.set_attribute(f"gen_ai.request.{k}", v)

    messages = kwargs.get("messages") or []
    if capture == "metadata_only":
        span.set_attribute("gen_ai.request.message_count", len(messages))
        roles = []
        for m in messages:
            with suppress(Exception):
                roles.append(getattr(m, "role", None) or m.get("role"))  # type: ignore[union-attr]
        if roles:
            span.set_attribute("gen_ai.request.roles", ",".join(r for r in roles if r))
    elif capture == "full":
        with suppress(Exception):
            span.add_event(
                "gen_ai.content.prompt",
                attributes={
                    "gen_ai.prompt": str(messages)[:8000],  # truncate to keep span sane
                },
            )


def _set_response_attrs(span: Any, resp: ChatResponse, capture: CaptureMode) -> None:
    span.set_attribute("gen_ai.response.id", resp.id)
    span.set_attribute("gen_ai.response.model", resp.provider_model)
    span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
    if resp.usage.cached_input_tokens:
        span.set_attribute("gen_ai.usage.cached_input_tokens", resp.usage.cached_input_tokens)
    if resp.usage.reasoning_tokens:
        span.set_attribute("gen_ai.usage.reasoning_tokens", resp.usage.reasoning_tokens)
    if resp.cost is not None:
        span.set_attribute("gen_ai.cost.total_usd", resp.cost.total_usd)
        span.set_attribute("gen_ai.cost.source", resp.cost.source)
        span.set_attribute("gen_ai.cost.confidence", resp.cost.confidence)
    if resp.choices and resp.choices[0].finish_reason:
        span.set_attribute("gen_ai.response.finish_reason", resp.choices[0].finish_reason)
    span.set_attribute("gen_ai.client.latency_ms", resp.latency_ms)
    if capture == "full":
        span.add_event(
            "gen_ai.content.completion",
            attributes={"gen_ai.completion": resp.text[:8000]},
        )


def _emit_metrics(
    token_histogram: Any,
    duration_histogram: Any,
    cost_histogram: Any,
    entry: ModelEntry,
    resp: ChatResponse,
    duration_s: float,
) -> None:
    common = {
        "gen_ai.system": entry.provider,
        "gen_ai.request.model": entry.model_id,
        "gen_ai.response.model": resp.provider_model,
    }
    token_histogram.record(
        resp.usage.input_tokens,
        attributes={**common, "gen_ai.token.type": "input"},
    )
    token_histogram.record(
        resp.usage.output_tokens,
        attributes={**common, "gen_ai.token.type": "output"},
    )
    duration_histogram.record(duration_s, attributes=common)
    if resp.cost is not None and resp.cost.total_usd is not None:
        cost_histogram.record(resp.cost.total_usd, attributes=common)


__all__ = ["CaptureMode", "instrument"]
