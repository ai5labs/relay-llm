"""Minimal Relay quickstart.

Run with::

    GROQ_API_KEY=... python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from relay import Hub


async def main() -> None:
    async with Hub.from_yaml("examples/models.yaml") as hub:
        # 1. List what's configured.
        print("Configured aliases:")
        for alias in hub.list_aliases():
            print(f"  - {alias}")
        print()

        # 2. One-shot chat.
        resp = await hub.chat(
            "fast-cheap",
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
        )
        print(f"Reply:    {resp.text}")
        print(f"Provider: {resp.provider} ({resp.provider_model})")
        print(f"Latency:  {resp.latency_ms:.0f} ms")
        print(f"Tokens:   in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        if resp.cost:
            print(
                f"Cost:     ${resp.cost.total_usd:.6f}  "
                f"(source={resp.cost.source}, confidence={resp.cost.confidence})"
            )
        print()

        # 3. Streaming.
        print("Streaming:")
        stream = hub.stream(
            "fast-cheap",
            messages=[{"role": "user", "content": "Count to 5, one per line."}],
        )
        async for ev in stream:
            if ev.type == "text_delta":
                print(ev.text, end="", flush=True)
            elif ev.type == "end":
                print(f"\n[done in {ev.response.latency_ms:.0f} ms]")


if __name__ == "__main__":
    asyncio.run(main())
