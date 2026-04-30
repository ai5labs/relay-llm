"""Mock OpenAI-compatible server.

Returns canned ``/v1/chat/completions`` responses with controllable backend
latency. Used by the gateway-comparison harness so all gateways hit the same
synthetic backend — eliminates network noise to real providers, which would
drown out the per-gateway overhead signal.

Run::

    python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50

The ``backend-ms`` flag controls how long the server sleeps before responding,
simulating real-provider latency.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def make_app(*, backend_ms: float = 50.0, stream_chunks: int = 20) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        stream = bool(body.get("stream"))
        if stream:
            return StreamingResponse(
                _stream_response(body, backend_ms / 1000.0, stream_chunks),
                media_type="text/event-stream",
            )

        # Non-streaming: sleep, return a fixed shape.
        await asyncio.sleep(backend_ms / 1000.0)
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def _stream_response(body: dict[str, Any], backend_s: float, n_chunks: int):
    """Emit n_chunks SSE frames with ``backend_s / n_chunks`` delay between each."""
    per_chunk = backend_s / max(n_chunks, 1)
    chunk_id = "chatcmpl-mock"
    model = body.get("model", "mock")

    # First chunk — role.
    first = {
        "id": chunk_id,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
    }
    yield f"data: {orjson.dumps(first).decode()}\n\n".encode()

    for i in range(n_chunks):
        await asyncio.sleep(per_chunk)
        chunk = {
            "id": chunk_id,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": f"tok{i} "}}],
        }
        yield f"data: {orjson.dumps(chunk).decode()}\n\n".encode()

    # Final chunk — finish_reason + usage.
    final = {
        "id": chunk_id,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": n_chunks,
            "total_tokens": 10 + n_chunks,
        },
    }
    yield f"data: {orjson.dumps(final).decode()}\n\n".encode()
    yield b"data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--backend-ms",
        type=float,
        default=50.0,
        help="simulated provider latency in milliseconds",
    )
    parser.add_argument(
        "--stream-chunks",
        type=int,
        default=20,
        help="how many SSE delta chunks to emit per streaming response",
    )
    args = parser.parse_args()
    app = make_app(backend_ms=args.backend_ms, stream_chunks=args.stream_chunks)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
