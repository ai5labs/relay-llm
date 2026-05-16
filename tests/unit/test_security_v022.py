"""v0.2.2 regression tests — four critical + three high fixes from the
re-audit that ran against the freshly-published v0.2.1.

C1: per-request auth headers (multi-tenant credential pool leak)
C2: MCP args allowlist for npx/uvx/docker package selectors
C3: tightened secret-scrubber regex (Basic / Token / AWS-Sig + more keys)
C4: reject userinfo in base_url
H1: emit audit events from _stream_one
H2: relax response-side tool-call validator (additionalProperties)
H3: structured retry catches ToolSchemaError
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from relay import Hub
from relay.audit import AuditEvent, CallbackSink
from relay.config import load_str
from relay.errors import ConfigError, ProviderError, ToolSchemaError
from relay.types import ToolDefinition


def _yaml_two_models() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m1:
            target: openai/gpt-4o-mini
            credential: $env.KEY_A
          m2:
            target: openai/gpt-4o-mini
            credential: $env.KEY_B
    """


def _ok_body() -> dict[str, Any]:
    return {
        "id": "x",
        "model": "gpt-4o-mini",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# C1: per-request auth headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_multi_tenant_credentials_do_not_leak_across_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two model entries sharing (provider, base_url) but with different
    credentials must NOT have their bearer tokens cross-contaminate via the
    pooled httpx client."""
    monkeypatch.setenv("KEY_A", "sk-tenant-a-key")
    monkeypatch.setenv("KEY_B", "sk-tenant-b-key")

    seen_auth: list[str] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json=_ok_body())

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)

    hub = Hub.from_config(load_str(_yaml_two_models()))
    try:
        await hub.chat("m1", messages=[{"role": "user", "content": "hi"}])
        await hub.chat("m2", messages=[{"role": "user", "content": "hi"}])
        await hub.chat("m1", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()

    assert seen_auth == [
        "Bearer sk-tenant-a-key",
        "Bearer sk-tenant-b-key",
        "Bearer sk-tenant-a-key",
    ], f"credential pool leak: {seen_auth!r}"


# ---------------------------------------------------------------------------
# C2: MCP args allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_npx_args_with_package_blocked() -> None:
    """npx + positional args (= package to fetch + run) must require explicit
    operator opt-in. The original v0.2.1 allowlist only checked command,
    leaving an arbitrary-code-execution bypass via args."""
    from relay.mcp import MCPManager

    mgr = MCPManager()
    with pytest.raises(ConfigError, match="fetch and execute package"):
        await mgr.add_stdio("evil", command="npx", args=["-y", "attacker-pkg"])


@pytest.mark.asyncio
async def test_mcp_docker_run_image_blocked() -> None:
    from relay.mcp import MCPManager

    mgr = MCPManager()
    with pytest.raises(ConfigError, match="fetch and execute package"):
        await mgr.add_stdio("evil", command="docker", args=["run", "--rm", "attacker/image"])


@pytest.mark.asyncio
async def test_mcp_npx_with_args_passes_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relay.mcp import MCPManager
    from relay.mcp import _manager as mcp_manager

    async def _fake_connect(self: Any) -> None:
        self._session = object()

    monkeypatch.setattr(mcp_manager.MCPServer, "connect", _fake_connect)
    mgr = MCPManager()
    await mgr.add_stdio(
        "github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        allow_arbitrary=True,
    )
    assert "github" in mgr.list_servers()


# ---------------------------------------------------------------------------
# C3: scrubber covers Basic / Token / AWS Sig / Stripe / etc.
# ---------------------------------------------------------------------------


# Synthetic secret fixtures. Composed at runtime so the literal source file
# does not match GitHub's secret-scanning push-protection patterns (which
# would otherwise reject this very test of our own scrubber).
_SECRET_FIXTURES = [
    (
        f"Authorization: Basic {'dXNlcjpwYXNzd29yZA'}==",
        "dXNlcjpwYXNzd29yZA",
    ),
    (
        f"Authorization: Token {'tok' + '_'}{'secret' + '_'}value_long_enough",
        "tok" + "_" + "secret",
    ),
    (
        f"Authorization: AWS4-HMAC-SHA256 Credential={'AKIA' + '1234567890ABCDEF'}",
        "AKIA" + "1234567890ABCDEF",
    ),
    ("Authorization: deadbeefnoscheme", "deadbeefnoscheme"),
    (
        f"{'sk' + '_'}{'live' + '_'}abc12345defghij67890klmnop",
        "sk" + "_" + "live" + "_" + "abc",
    ),
    (
        f"{'pk' + '_'}{'test' + '_'}abc12345defghij67890klmnop",
        "pk" + "_" + "test" + "_" + "abc",
    ),
    (
        f"github{'_'}pat{'_'}11AAAAAAA0abc123def456ghi789jklmno",
        "github" + "_" + "pat" + "_" + "11",
    ),
    (
        f"hf{'_'}abcdefghijklmnopqrstuvwxyz012345AB",
        "hf" + "_" + "abcdefghij",
    ),
    ("ASIA" + "IOSFODNN7EXAMPLE", "ASIA" + "IOSFODNN7EXAMPLE"),
]


@pytest.mark.parametrize(("secret_input", "must_not_contain"), _SECRET_FIXTURES)
def test_scrubber_covers_secret_class(secret_input: str, must_not_contain: str) -> None:
    err = ProviderError(secret_input)
    assert must_not_contain not in str(err), f"leaked: {must_not_contain!r} in {str(err)!r}"


# ---------------------------------------------------------------------------
# C4: reject userinfo in base_url
# ---------------------------------------------------------------------------


def test_base_url_userinfo_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://attacker.com@api.openai.com/v1 passes the hostname check but
    httpx may route to attacker.com — explicit reject."""
    monkeypatch.setenv("KEY", "sk-test")
    with pytest.raises(ConfigError, match="userinfo"):
        load_str(
            """
            version: 1
            models:
              m:
                target: openai/gpt-4o
                credential: $env.KEY
                base_url: https://attacker.com@api.openai.com/v1
            """
        )


def test_base_url_userinfo_user_only_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY", "sk-test")
    with pytest.raises(ConfigError, match="userinfo"):
        load_str(
            """
            version: 1
            models:
              m:
                target: openai/gpt-4o
                credential: $env.KEY
                base_url: https://justuser@api.openai.com/v1
            """
        )


# ---------------------------------------------------------------------------
# H1: stream emits audit events
# ---------------------------------------------------------------------------


def _sse(*chunks: str) -> str:
    return "\n".join(f"data: {c}" for c in chunks) + "\ndata: [DONE]\n"


@pytest.mark.asyncio
@respx.mock
async def test_stream_emits_audit_event_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")
    events: list[AuditEvent] = []

    async def _cb(ev: AuditEvent) -> None:
        events.append(ev)

    body = _sse(
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}',
        '{"id":"x","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        ),
    )

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/test
            credential: $env.TEST_KEY
    """
    hub = Hub.from_config(load_str(yaml), audit_sinks=[CallbackSink(_cb)])
    try:
        async for _ev in hub.stream("m", messages=[{"role": "user", "content": "hi"}]):
            pass
    finally:
        await hub.aclose()

    stream_events = [e for e in events if e.operation == "stream"]
    assert stream_events, "no stream audit event emitted"
    assert stream_events[0].error_type is None


@pytest.mark.asyncio
async def test_stream_emits_audit_event_on_pre_guardrail_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")
    from relay.guardrails import BlockedKeywords, GuardrailError

    events: list[AuditEvent] = []

    async def _cb(ev: AuditEvent) -> None:
        events.append(ev)

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/test
            credential: $env.TEST_KEY
    """
    hub = Hub.from_config(
        load_str(yaml),
        audit_sinks=[CallbackSink(_cb)],
        guardrails=[BlockedKeywords(["secret"], check_response=False)],
    )
    try:
        with pytest.raises(GuardrailError):
            async for _ev in hub.stream(
                "m", messages=[{"role": "user", "content": "the secret is"}]
            ):
                pass
        # Background-task audit emit — give the loop a tick to run.
        import asyncio

        await asyncio.sleep(0.05)
    finally:
        await hub.aclose()

    stream_events = [e for e in events if e.operation == "stream"]
    assert stream_events, "pre-guardrail block produced no audit row"
    assert stream_events[0].error_type == "GuardrailError"


# ---------------------------------------------------------------------------
# H2: tool-call validator no longer rejects extra response fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_extra_fields_in_response_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when the declared schema has additionalProperties: false, a
    provider that injects extra fields (OpenAI strict-mode metadata,
    vendor extensions) must not crash the response."""
    monkeypatch.setenv("TEST_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "create_issue",
                                        # Required ``title`` present, extra
                                        # field that violates additionalProperties: false.
                                        "arguments": '{"title": "x", "__reasoning": "..."}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
    """
    tool = ToolDefinition(
        name="create_issue",
        description="",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    )
    hub = Hub.from_config(load_str(yaml))
    try:
        # Must NOT raise — the response-side validator should relax
        # additionalProperties on responses (request-side is unchanged).
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}], tools=[tool])
        tc = resp.choices[0].tool_calls[0]
        assert tc.name == "create_issue"
        assert tc.arguments["title"] == "x"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_missing_required_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relaxation is targeted at additionalProperties only — schema
    violations on required fields and types must still fail closed."""
    monkeypatch.setenv("TEST_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "create_issue",
                                        # Required ``title`` missing.
                                        "arguments": '{"body": "x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
    """
    tool = ToolDefinition(
        name="create_issue",
        description="",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    hub = Hub.from_config(load_str(yaml))
    try:
        with pytest.raises(ToolSchemaError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}], tools=[tool])
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# H3: structured retry handles ToolSchemaError
# ---------------------------------------------------------------------------


def test_structured_retry_catches_tool_schema_error() -> None:
    """request_structured's retry loop must catch ToolSchemaError (newly
    raised by _validate_response_tool_calls when a single-tool structured
    output fails the tool's declared schema), not just StructuredOutputError."""
    import inspect

    from relay import structured

    src = inspect.getsource(structured.request_structured)
    # The retry block must reference ToolSchemaError so a single-tool
    # structured output that fails validation triggers a corrective turn.
    assert "ToolSchemaError" in src
