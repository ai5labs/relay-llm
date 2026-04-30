# Audit logging

Every chat / stream call emits a structured `AuditEvent` to all attached sinks.

```python
from relay import Hub
from relay.audit import FileSink, StdoutSink, CallbackSink

hub = Hub.from_yaml(
    "models.yaml",
    audit_sinks=[FileSink("/var/log/relay/audit.jsonl")],
)
```

## Event schema

`AuditEvent` mirrors OpenTelemetry GenAI semantic conventions:

```python
@dataclass(frozen=True, slots=True)
class AuditEvent:
    timestamp_ns: int
    event_id: str
    operation: Literal["chat", "stream"]
    alias: str
    provider: str
    model_id: str
    request_id: str | None
    user_id: str | None      # from metadata.user_id

    duration_ms: float | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    cost_source: str | None

    finish_reason: str | None
    error_type: str | None
    error_message: str | None

    messages_summary: list[dict]    # role, length, sha256_prefix
    messages_full: list[dict] | None
    response_text_summary: dict | None
    response_text: str | None

    redaction_count: int
    redaction_kinds: tuple[str, ...]
```

## Built-in sinks

```python
from relay.audit import FileSink, StdoutSink, CallbackSink

# JSON lines on stdout — dev only
StdoutSink()

# Append-only JSONL file with async lock
FileSink("/var/log/relay/audit.jsonl")

# Custom callback (sync or async)
async def my_sink(event):
    await my_kafka_producer.send("relay-audit", event)
CallbackSink(my_sink)
```

## Custom sinks

Anything that satisfies the `AuditSink` Protocol works:

```python
class S3WORMSink:
    """SOC 2-grade sink: S3 with Object Lock for tamper evidence."""

    async def emit(self, event):
        # ... write to S3 with WORM headers
        pass

    async def aclose(self):
        pass

hub = Hub.from_yaml("models.yaml", audit_sinks=[S3WORMSink()])
```

Multiple sinks are fanned out concurrently. A failing sink never breaks the call (errors are swallowed; in v0.3 they'll be reported via OTel).

## Privacy

Sinks receive *post-redaction* messages by default (see [PII redaction](redaction.md)). Capture mode is set in YAML:

```yaml
observability:
  capture_messages: metadata_only   # never | metadata_only | full
```

`metadata_only` (the default) writes role + length + sha256 prefix per message — enough to debug without leaking content. `full` is opt-in and required for some debugging or compliance regimes.

## User attribution

Pass `metadata={"user_id": "user-42"}` on a chat call to populate `event.user_id`:

```python
await hub.chat("smart", messages=[...], metadata={"user_id": "user-42"})
```
