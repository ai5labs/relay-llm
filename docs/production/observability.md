# Observability (OpenTelemetry)

Relay emits OpenTelemetry GenAI semantic-convention spans + metrics — the standard format Datadog, Honeycomb, Langfuse, Arize, and Phoenix all consume.

```bash
pip install 'ai5labs-relay[otel]'
```

```python
from relay import Hub
from relay.observability import instrument

hub = Hub.from_yaml("models.yaml")
instrument(hub, capture_messages="metadata_only")
```

Now every `hub.chat()` / `hub.stream()` call emits a `chat <model>` span with attributes:

| Attribute | Description |
|---|---|
| `gen_ai.system` | Provider id (`openai`, `anthropic`, ...) |
| `gen_ai.request.model` | Model id |
| `gen_ai.response.model` | Resolved model from the response |
| `gen_ai.request.message_count` | (metadata mode) |
| `gen_ai.request.roles` | Comma-separated roles |
| `gen_ai.usage.input_tokens` | |
| `gen_ai.usage.output_tokens` | |
| `gen_ai.usage.cached_input_tokens` | |
| `gen_ai.usage.reasoning_tokens` | |
| `gen_ai.cost.total_usd` | |
| `gen_ai.cost.source` | live_api / snapshot / user_override / unknown |
| `gen_ai.response.finish_reason` | |
| `gen_ai.client.latency_ms` | |
| `error.type` | exception class name on failure |

And metrics:

* `gen_ai.client.token.usage` (histogram, attributes: `gen_ai.token.type=input|output`)
* `gen_ai.client.operation.duration` (histogram, seconds)
* `gen_ai.client.cost` (histogram, USD)

## capture_messages

Three privacy levels for prompt/response content:

* `"never"` — only metadata on spans (token counts, cost, latency).
* `"metadata_only"` (default) — also message roles + count.
* `"full"` — full text on spans. Required for some debugging workflows; **also a leak risk** — make sure your trace backend supports redaction.

## Pairing with audit logging

OTel spans are operational telemetry; audit logs are compliance evidence. They serve different audiences. Most production deployments enable **both**:

```python
from relay import Hub
from relay.observability import instrument
from relay.audit import FileSink

hub = Hub.from_yaml("models.yaml", audit_sinks=[FileSink("/var/log/relay.audit")])
instrument(hub, capture_messages="metadata_only")
```
