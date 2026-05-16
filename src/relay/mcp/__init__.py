"""MCP (Model Context Protocol) — universal tool layer.

Lets a Relay user attach an MCP server (GitHub, Slack, Postgres, filesystem,
Playwright, etc.) and have its tools available to *any* configured model. The
MCP server speaks one protocol; Relay's :mod:`tools` schema compiler translates
each tool's JSON Schema into the target provider's native tool shape.

This means:

* You can use the GitHub MCP server's tools against an Anthropic model that
  doesn't natively speak MCP — Relay handles the translation.
* You can swap models without touching tool definitions.
* You get per-tool audit logging (what the model called, with what args) — the
  hook is in :class:`MCPManager`.

Usage
-----
::

    from relay import Hub
    from relay.mcp import MCPManager

    mcp = MCPManager()
    await mcp.add_stdio("github", command="npx",
                        args=["-y", "@modelcontextprotocol/server-github"])
    tools = await mcp.list_tools()  # all tools across all servers

    hub = Hub.from_yaml("models.yaml")
    resp = await hub.chat("smart", messages=[...], tools=tools)

    # When the model calls a tool, dispatch it back through MCP:
    for tc in resp.tool_calls:
        result = await mcp.call_tool(tc.name, tc.arguments)

The actual MCP protocol implementation uses the official ``mcp`` Python SDK
when installed; if it's not, :class:`MCPManager` raises ``ConfigError`` with
install instructions.

Design notes
------------
* **Lazy import** — the ``mcp`` package is heavy and not everyone needs it.
* **Tool name collision avoidance** — tools are exposed as ``{server}__{tool}``
  so ``github__create_issue`` doesn't collide with ``slack__create_issue``.
* **Schema translation** is reused from :mod:`relay.tools` — no duplication.
"""

from relay.mcp._manager import (
    MCP_STDIO_ALLOWED_COMMANDS,
    MCP_TOOL_RESULT_MAX_BYTES,
    MCPManager,
    MCPServer,
    MCPToolError,
)

__all__ = [
    "MCP_STDIO_ALLOWED_COMMANDS",
    "MCP_TOOL_RESULT_MAX_BYTES",
    "MCPManager",
    "MCPServer",
    "MCPToolError",
]
