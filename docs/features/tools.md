# Tool calling

```python
from relay.types import ToolDefinition

weather = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
    strict=True,        # OpenAI strict-mode opt-in
)

resp = await hub.chat("smart", messages=[...], tools=[weather])

for tc in resp.tool_calls:
    print(tc.name, tc.arguments)
```

## Cross-provider schema compilation

JSON Schema dialects diverge across providers. Relay's compiler translates per-provider:

| Provider | Output shape | Notable handling |
|---|---|---|
| OpenAI / OpenAI-compat | `{type: "function", function: {...}}` | Strict mode adds `additionalProperties: false`, requires all keys, strips unsupported keywords. |
| Anthropic | `{name, description, input_schema}` | Unenforceable constraints (`maxLength`, `pattern`) are baked into the description. |
| Gemini / Vertex | `{functionDeclarations: [{...}]}` | Strips `$ref`, `oneOf`, `additionalProperties` (Gemini ignores them); injects them as instruction text. |
| Bedrock | `{toolSpec: {inputSchema: {json: {...}}}}` | Schema passed through. |
| Cohere | `{parameter_definitions: {...}}` | Top-level properties only (Cohere doesn't accept JSON Schema). |

## The Mastra trick

When a JSON Schema constraint can't be expressed in the target's dialect (e.g. `maxLength` on Gemini), Relay injects it into the tool description as plain English:

```
Constraints not enforced by the target schema (please honor them anyway):
- q: maxLength=100
- q: pattern='^[a-z]+$'
```

Mastra's [published case study](https://mastra.ai/blog/mcp-tool-compatibility-layer) showed this cuts tool-call error rates from ~15% to ~3% for Gemini.

## Strict compilation

If you'd rather get a clear error than a silent downgrade:

```python
from relay.tools import compile_for

compile_for(my_tool, "google", strict_compile=True)
# Raises ToolSchemaError if the schema can't be expressed exactly.
```

## Tool choice

```python
resp = await hub.chat(
    "smart",
    messages=[...],
    tools=[weather],
    tool_choice="auto",       # default
    # tool_choice="required",  # must call a tool
    # tool_choice="none",      # never call a tool
    # tool_choice={"type": "tool", "name": "get_weather"},  # force this tool
)
```
