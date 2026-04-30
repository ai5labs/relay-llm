# Batch API

OpenAI Batch and Anthropic Message Batches both offer ~50% cost discount on requests with a 24-hour SLA. Relay normalizes both behind one API.

```python
handle = await hub.batch.submit(
    "smart",
    requests=[
        {"messages": [{"role": "user", "content": "q1"}]},
        {"messages": [{"role": "user", "content": "q2"}], "max_tokens": 256},
        {"custom_id": "important", "messages": [...]},
    ],
)
print(handle.id, handle.provider, handle.request_count)

# Poll status
prog = await hub.batch.status(handle)
print(prog.status, f"{prog.completed}/{prog.total}")

# Once complete, fetch results
results = await hub.batch.results(handle)
for r in results:
    if r.error:
        print(r.custom_id, "error:", r.error)
    else:
        print(r.custom_id, r.response.text)
```

## Cancellation

```python
prog = await hub.batch.cancel(handle)
```

## Provider support

| Provider | Endpoint | Notes |
|---|---|---|
| OpenAI / OpenAI-compat | `/v1/files` + `/v1/batches` | Two-step: upload JSONL, then create batch. |
| Anthropic | `/v1/messages/batches` | Single submit with embedded request list. |
| Bedrock | (v0.3) | Different workflow — async S3 input/output via `CreateModelInvocationJob`. |

## Custom IDs

Each request gets `custom_id: req-{i}` by default. Set your own:

```python
requests=[
    {"custom_id": "user-42", "messages": [...]},
]
```

Results are returned keyed by custom_id, so you can join back to your original IDs.
