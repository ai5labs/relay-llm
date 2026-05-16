"""MCP manager tests.

We don't spin up a real MCP server here — we mock the underlying session
so the namespacing, dispatch, and translation logic can be tested in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from relay.errors import ConfigError, ToolSchemaError
from relay.mcp import MCP_TOOL_RESULT_MAX_BYTES, MCPManager, MCPToolError
from relay.tools import compile_for
from relay.types import ToolDefinition


def _stub_server(server_name: str, tools: list[dict[str, Any]]) -> Any:
    """Make a fake :class:`MCPServer` that returns a fixed list of tools."""
    from relay.mcp._manager import MCPServer

    server = MCPServer(name=server_name, transport="stdio", config={"command": "/bin/true"})
    server._tools_cache = [
        ToolDefinition(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
        )
        for t in tools
    ]
    server._session = object()  # bypass the real connect()
    server.connect = AsyncMock()  # type: ignore[method-assign]
    return server


@pytest.mark.asyncio
async def test_list_tools_prefixes_with_server_name() -> None:
    mgr = MCPManager()
    mgr._servers["github"] = _stub_server(
        "github", [{"name": "create_issue", "description": "Open an issue"}]
    )
    mgr._servers["slack"] = _stub_server("slack", [{"name": "send_message"}])
    tools = await mgr.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["github__create_issue", "slack__send_message"]


@pytest.mark.asyncio
async def test_list_tools_avoids_collision_across_servers() -> None:
    mgr = MCPManager()
    mgr._servers["a"] = _stub_server("a", [{"name": "create_issue"}])
    mgr._servers["b"] = _stub_server("b", [{"name": "create_issue"}])
    tools = await mgr.list_tools()
    assert {t.name for t in tools} == {"a__create_issue", "b__create_issue"}


@pytest.mark.asyncio
async def test_call_tool_routes_to_correct_server() -> None:
    mgr = MCPManager()
    a = _stub_server("a", [])
    b = _stub_server("b", [])
    a.call_tool = AsyncMock(return_value="from-a")  # type: ignore[method-assign]
    b.call_tool = AsyncMock(return_value="from-b")  # type: ignore[method-assign]
    mgr._servers["a"] = a
    mgr._servers["b"] = b

    out = await mgr.call_tool("a__do_thing", {"x": 1})
    assert out == "from-a"
    a.call_tool.assert_awaited_with("do_thing", {"x": 1})
    b.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_call_tool_unknown_server_raises() -> None:
    mgr = MCPManager()
    with pytest.raises(MCPToolError, match="unknown MCP server"):
        await mgr.call_tool("ghost__do", {})


@pytest.mark.asyncio
async def test_call_tool_unprefixed_name_raises() -> None:
    mgr = MCPManager()
    with pytest.raises(MCPToolError, match="not server-prefixed"):
        await mgr.call_tool("bare_name", {})


@pytest.mark.asyncio
async def test_duplicate_server_name_rejected() -> None:
    mgr = MCPManager()
    mgr._servers["already"] = _stub_server("already", [])
    # ``npx`` is on the stdio allowlist; the dup check fires first regardless.
    with pytest.raises(ConfigError, match="already added"):
        await mgr.add_stdio("already", command="npx")


@pytest.mark.asyncio
async def test_mcp_tools_compile_for_each_provider() -> None:
    """An MCP-provided tool should round-trip cleanly through the schema compiler
    for every provider — that's the whole point of the universal layer."""
    mgr = MCPManager()
    mgr._servers["gh"] = _stub_server(
        "gh",
        [
            {
                "name": "create_issue",
                "description": "Open an issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title"],
                },
            }
        ],
    )
    [tool] = await mgr.list_tools()
    assert tool.name == "gh__create_issue"

    # Compile for OpenAI, Anthropic, Gemini, Bedrock — all should succeed.
    openai_shape = compile_for(tool, "openai")
    anthropic_shape = compile_for(tool, "anthropic")
    gemini_shape = compile_for(tool, "google")
    bedrock_shape = compile_for(tool, "bedrock")

    assert openai_shape["function"]["name"] == "gh__create_issue"
    assert anthropic_shape["name"] == "gh__create_issue"
    assert gemini_shape["name"] == "gh__create_issue"
    assert bedrock_shape["toolSpec"]["name"] == "gh__create_issue"


@pytest.mark.asyncio
async def test_stdio_allowlist_rejects_unknown() -> None:
    """Commands outside the allowlist must be rejected before subprocess spawn."""
    mgr = MCPManager()
    with pytest.raises(ConfigError, match="not in the allowlist"):
        await mgr.add_stdio("evil", command="/bin/sh", args=["-c", "id"])


@pytest.mark.asyncio
async def test_stdio_allowlist_accepts_npx(monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowlisted basename like ``npx`` passes the validator."""
    from relay.mcp import _manager as mcp_manager

    async def _fake_connect(self: Any) -> None:
        self._session = object()

    monkeypatch.setattr(mcp_manager.MCPServer, "connect", _fake_connect)
    mgr = MCPManager()
    await mgr.add_stdio("github", command="npx", args=["-y", "@x/y"])
    assert "github" in mgr.list_servers()


@pytest.mark.asyncio
async def test_stdio_allowlist_connect_revalidates() -> None:
    """A hand-built MCPServer with a bad command must still be rejected at
    connect() — and this defense must fire even when the `mcp` SDK is not
    installed (the validator runs before the import)."""
    from relay.mcp._manager import MCPServer

    server = MCPServer(
        name="evil",
        transport="stdio",
        config={"command": "/bin/sh", "args": ["-c", "echo pwn"]},
        allow_arbitrary_command=False,
    )
    with pytest.raises(ConfigError, match="not in the allowlist"):
        await server.connect()


@pytest.mark.asyncio
async def test_tool_args_validated_against_schema() -> None:
    """LLM-supplied arguments that violate the declared schema must be rejected."""
    mgr = MCPManager()
    server = _stub_server(
        "gh",
        [
            {
                "name": "create_issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }
        ],
    )
    server.call_tool = AsyncMock(return_value="ok")  # type: ignore[method-assign]
    mgr._servers["gh"] = server

    # Missing required ``title`` — should fail closed.
    with pytest.raises(ToolSchemaError, match="do not match declared schema"):
        await mgr.call_tool("gh__create_issue", {"body": "nope"})

    # Wrong type for an existing field.
    with pytest.raises(ToolSchemaError):
        await mgr.call_tool("gh__create_issue", {"title": 42})

    # Valid args pass through.
    out = await mgr.call_tool("gh__create_issue", {"title": "bug"})
    assert out == "ok"


@pytest.mark.asyncio
async def test_tool_result_truncated_at_cap() -> None:
    """A massive MCP tool result must be truncated with a sentinel."""
    from relay.mcp._manager import MCPServer

    class _Block:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Result:
        isError = False

        def __init__(self) -> None:
            self.content = [_Block("A" * (MCP_TOOL_RESULT_MAX_BYTES + 1024))]

    class _Session:
        async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> _Result:
            return _Result()

    server = MCPServer(name="big", transport="stdio", config={"command": "npx"})
    server._session = _Session()
    out = await server.call_tool("dump", {})
    assert out.endswith(f"[result truncated at {MCP_TOOL_RESULT_MAX_BYTES} bytes]")
    # Body before the sentinel never exceeds the cap.
    body = out.rsplit("\n[result truncated", 1)[0]
    assert len(body.encode("utf-8")) <= MCP_TOOL_RESULT_MAX_BYTES


@pytest.mark.asyncio
async def test_manager_aclose_clears_servers() -> None:
    mgr = MCPManager()
    server = _stub_server("a", [])
    server.aclose = AsyncMock()  # type: ignore[method-assign]
    mgr._servers["a"] = server
    await mgr.aclose()
    assert mgr.list_servers() == []
    server.aclose.assert_awaited()


@pytest.mark.asyncio
async def test_hub_attach_mcp() -> None:
    from relay import Hub
    from relay.config import load_str

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
    """
    hub = Hub.from_config(load_str(yaml))
    mgr = MCPManager()
    mgr._servers["a"] = _stub_server("a", [{"name": "ping"}])
    hub.attach_mcp(mgr)
    tools = await hub.mcp_tools()
    assert tools[0].name == "a__ping"
    await hub.aclose()
