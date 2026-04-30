# Guardrails

Pre-call and post-call hooks that can block requests/responses.

```python
from relay import Hub
from relay.guardrails import BlockedKeywords, MaxInputLength

hub = Hub.from_yaml(
    "models.yaml",
    guardrails=[
        MaxInputLength(max_chars=50_000),
        BlockedKeywords(["confidential", "internal-only"]),
    ],
)

# Either guardrail firing → GuardrailError raised before/after the model call.
```

## Built-in guardrails

| Class | When | Use |
|---|---|---|
| `MaxInputLength(max_chars)` | pre | Cheap sanity bound on prompt size. |
| `BlockedKeywords(terms, check_response=True)` | pre + post | Reject prompts/responses containing forbidden terms. Accepts strings or compiled regex. |

## Custom guardrails

Implement the `Guardrail` Protocol:

```python
from relay.guardrails import Guardrail, GuardrailViolation

class TopicFilter:
    name = "topic_filter"
    stage = "pre"

    def __init__(self, allowed_topics: list[str]):
        self._allowed = allowed_topics

    def check_pre(self, messages):
        text = " ".join(m.content if isinstance(m.content, str) else "" for m in messages)
        # ... your logic
        if "off-topic" in text.lower():
            return GuardrailViolation(
                rule=self.name,
                stage="pre",
                message="prompt is off-topic",
            )
        return None

    def check_post(self, response):
        return None
```

## Composition

Guardrails compose; first blocking violation wins. Pre-call guardrails run before the HTTP call (saving money on rejected prompts); post-call guardrails run before the response is returned to the caller.

## Plugins

Wrappers for [Lakera Guard](https://www.lakera.ai/), [Protect AI Rebuff](https://github.com/protectai/rebuff), and [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) are 50-line plugins against the Protocol. We'll ship reference implementations in v0.3.
