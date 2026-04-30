# Structured output

Pydantic models in, validated instances out.

```python
from pydantic import BaseModel
from relay.structured import request_structured

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

person = await request_structured(
    hub=hub,
    alias="smart",
    schema=Person,
    messages=[{"role": "user", "content": "Make up a person"}],
)

assert isinstance(person, Person)
print(person.name, person.age)
```

## Per-provider compilation

Same Pydantic model, different wire format:

| Provider | How |
|---|---|
| OpenAI / Azure / OpenAI-compat with json_schema support | `response_format={"type": "json_schema", "json_schema": {..., "strict": true}}` |
| Anthropic / Bedrock | Single-tool trick: register a synthetic tool whose `input_schema` is the target schema, force `tool_choice` to that tool. The first `tool_use` block is your structured output. |
| Gemini / Vertex | `responseSchema` + `responseMimeType: "application/json"`. |
| Cohere / fallback | `response_format={"type": "json_object"}` + post-call validation with retry-on-validation-error. |

## Retry on validation failure

`request_structured` retries up to `max_attempts` (default 2) on `ValidationError`, including the validation message in the next prompt:

```python
person = await request_structured(
    hub=hub,
    alias="smart",
    schema=Person,
    messages=[...],
    max_attempts=3,
)
```

## Manual mode

If you need finer control:

```python
from relay.structured import build_request_overrides, parse_response

overrides = build_request_overrides(Person, provider="anthropic")
resp = await hub.chat("smart", messages=[...], **overrides)
person = parse_response(Person, resp)
```
