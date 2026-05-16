"""Audit logging interface.

Every chat / stream call emits a structured :class:`AuditEvent` to all attached
:class:`AuditSink` implementations. Schema mirrors the OpenTelemetry GenAI
semantic conventions where possible so a SOC 2 auditor sees familiar fields.

Sinks
-----
* :class:`StdoutSink` — JSON-lines on stdout (default for development).
* :class:`FileSink` — append to a JSONL file with optional rotation.
* Pluggable: enterprise sinks (S3 + Object Lock, Splunk HEC, Datadog, Elastic)
  are 30-line plugins built against the :class:`AuditSink` Protocol.

Privacy
-------
Sinks receive *post-redaction* messages by default (see :mod:`relay.redaction`).
``capture_messages`` controls how much content gets written:

* ``"never"`` — no message bodies, only metadata.
* ``"metadata_only"`` (default) — role + length + hash, no text.
* ``"full"`` — full text. Required for some compliance regimes; also a leak risk.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from relay.types import ChatResponse, Message

logger = logging.getLogger("relay.audit")

# Module-level counter exposed for ops to scrape — incremented every time an
# attached sink raises out of emit(). Hub.from_yaml(..., strict_audit=True)
# re-raises instead of swallowing.
audit_sink_failures: int = 0


def _record_sink_failure(sink: object, exc: BaseException) -> None:
    """Log a sink failure and bump the global counter."""
    global audit_sink_failures
    audit_sink_failures += 1
    logger.warning(
        "audit_sink_failed",
        extra={"sink": type(sink).__name__, "error_type": type(exc).__name__},
        exc_info=exc,
    )

CaptureMode = Literal["never", "metadata_only", "full"]


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """A structured audit-log row."""

    timestamp_ns: int
    event_id: str
    """Random per-event ULID-ish identifier."""
    operation: Literal["chat", "stream"]
    alias: str
    provider: str
    model_id: str
    request_id: str | None = None
    user_id: str | None = None
    """Caller-supplied user attribution (via ``ChatRequest.metadata.user_id``)."""

    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str | None = None

    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    messages_summary: list[dict[str, Any]] = field(default_factory=list)
    """When capture_messages=metadata_only, list of {role, length, sha256_prefix}."""
    messages_full: list[dict[str, Any]] | None = None
    """When capture_messages=full, the full post-redaction message list."""

    response_text_summary: dict[str, Any] | None = None
    response_text: str | None = None

    redaction_count: int = 0
    redaction_kinds: tuple[str, ...] = ()


@runtime_checkable
class AuditSink(Protocol):
    """Pluggable audit sink. Async to allow network sinks."""

    async def emit(self, event: AuditEvent) -> None: ...
    async def aclose(self) -> None: ...


class StdoutSink:
    """Print events as JSON lines to stdout. Synchronous, mostly for development."""

    async def emit(self, event: AuditEvent) -> None:
        print(json.dumps(_event_to_dict(event)))

    async def aclose(self) -> None:
        return None


class FileSink:
    """Append events to a JSONL file.

    Synchronous file write under an asyncio lock to avoid interleaving on
    concurrent calls. For high-throughput production use, use an async-native
    sink (e.g., AWS Firehose / Splunk HEC) instead.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def emit(self, event: AuditEvent) -> None:
        line = json.dumps(_event_to_dict(event))
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    async def aclose(self) -> None:
        return None


class CallbackSink:
    """Forward events to a user-provided callback. Useful for tests + custom
    in-process processing pipelines.

    Prefer an async callback — a blocking sync callback runs on the event
    loop and blocks every concurrent request. If a sync callback is passed,
    a :class:`UserWarning` is emitted at construction so callers see it.
    """

    def __init__(
        self,
        callback: Callable[[AuditEvent], Awaitable[None] | None],
    ) -> None:
        if not inspect.iscoroutinefunction(callback) and inspect.isfunction(callback):
            warnings.warn(
                "CallbackSink received a sync callback; it will block the event "
                "loop and serialize concurrent audit emissions. Prefer "
                "``async def cb(event): ...`` for production sinks.",
                UserWarning,
                stacklevel=2,
            )
        self._callback = callback

    async def emit(self, event: AuditEvent) -> None:
        result = self._callback(event)
        if asyncio.iscoroutine(result):
            await result

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers — message summarization
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m.content, str):
            text = m.content
        else:
            text = " ".join(getattr(b, "text", "") for b in m.content if hasattr(b, "text"))
        out.append(
            {
                "role": m.role,
                "length": len(text),
                "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            }
        )
    return out


def _event_to_dict(event: AuditEvent) -> dict[str, Any]:
    d = asdict(event)
    d["redaction_kinds"] = list(d.get("redaction_kinds") or [])
    return d


def new_event_id() -> str:
    """Short opaque id for an audit row."""
    return hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:24]


def build_event(
    *,
    operation: Literal["chat", "stream"],
    alias: str,
    provider: str,
    model_id: str,
    messages: list[Message],
    response: ChatResponse | None,
    error: BaseException | None,
    duration_ms: float | None,
    capture_messages: CaptureMode,
    redaction_count: int = 0,
    redaction_kinds: tuple[str, ...] = (),
    user_id: str | None = None,
) -> AuditEvent:
    """Build an AuditEvent from a chat call's pre/post state."""
    msgs_summary = summarize_messages(messages)
    msgs_full: list[dict[str, Any]] | None = None
    if capture_messages == "full":
        msgs_full = []
        for m in messages:
            if isinstance(m.content, str):
                msgs_full.append({"role": m.role, "content": m.content})
            else:
                blocks: list[Any] = []
                for b in m.content:
                    if hasattr(b, "model_dump"):
                        blocks.append(b.model_dump())
                    else:
                        blocks.append(str(b))
                msgs_full.append({"role": m.role, "content": blocks})
    elif capture_messages == "never":
        msgs_summary = []

    finish_reason = None
    response_text: str | None = None
    response_summary: dict[str, Any] | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    input_tokens = 0
    output_tokens = 0
    cached_input = 0
    reasoning = 0
    if response is not None:
        finish_reason = response.choices[0].finish_reason if response.choices else None
        if capture_messages == "full":
            response_text = response.text
        elif capture_messages == "metadata_only":
            response_summary = {
                "length": len(response.text),
                "sha256_prefix": hashlib.sha256(response.text.encode("utf-8")).hexdigest()[:16],
            }
        if response.cost:
            cost_usd = response.cost.total_usd
            cost_source = response.cost.source
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached_input = response.usage.cached_input_tokens
        reasoning = response.usage.reasoning_tokens

    err_type = type(error).__name__ if error else None
    if error is None:
        err_msg = None
    elif capture_messages == "never":
        # Don't let provider response bodies attached to RelayError.raw bleed
        # into the audit row. A model could trigger a 400 whose error body
        # echoes the prompt back; ``str(error)`` would carry it along.
        status = getattr(error, "status_code", None)
        err_msg = f"{err_type}: status={status}" if status is not None else f"{err_type}"
    else:
        err_msg = str(error)

    return AuditEvent(
        timestamp_ns=time.time_ns(),
        event_id=new_event_id(),
        operation=operation,
        alias=alias,
        provider=provider,
        model_id=model_id,
        request_id=getattr(response, "id", None) if response else None,
        user_id=user_id,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input,
        reasoning_tokens=reasoning,
        cost_usd=cost_usd,
        cost_source=cost_source,
        finish_reason=finish_reason,
        error_type=err_type,
        error_message=err_msg,
        messages_summary=msgs_summary,
        messages_full=msgs_full,
        response_text_summary=response_summary,
        response_text=response_text,
        redaction_count=redaction_count,
        redaction_kinds=redaction_kinds,
    )


__all__ = [
    "AuditEvent",
    "AuditSink",
    "CallbackSink",
    "CaptureMode",
    "FileSink",
    "StdoutSink",
    "build_event",
    "summarize_messages",
]
