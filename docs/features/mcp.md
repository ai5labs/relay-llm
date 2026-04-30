# MCP integration

Relay's universal MCP layer lets you attach any [Model Context Protocol](https://modelcontextprotocol.io) server (GitHub, Slack, Postgres, filesystem, Playwright) and use its tools against **any** configured model — including providers without native MCP support.

```bash
pip install 'relayllm[mcp]'
```

## Connect to MCP servers

```python
from relay import Hub
from relay.mcp import MCPManager

mcp = MCPManager()

# stdio transport — most servers ship as binaries
await mcp.add_stdio(
    "github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "..."},
)

# SSE transport
await mcp.add_sse("internal", url="https://mcp.acme.com/sse")

# Streamable-HTTP (mcp ≥ 1.0)
await mcp.add_streamable_http("docs", url="https://docs.mcp.acme.com")

hub = Hub.from_yaml("models.yaml")
hub.attach_mcp(mcp)
```

## Use MCP tools with any provider

```python
tools = await hub.mcp_tools()    # all servers' tools, namespaced

resp = await hub.chat(
    "smart",                        # could be Anthropic, Gemini, Bedrock — anything
    messages=[{"role": "user", "content": "Open a GitHub issue titled 'demo'"}],
    tools=tools,
)

for tc in resp.tool_calls:
    result = await hub.dispatch_tool_call(tc.name, tc.arguments)
    # Feed result back into the next round...
```

The cross-provider schema compiler (see [Tool calling](tools.md)) translates each MCP tool's JSON Schema into the target provider's dialect at request time.

## Naming

Tools are auto-namespaced by server: `github__create_issue`, `slack__send_message`, `postgres__query`. This avoids collisions across servers.

## Audit logging for MCP calls

When audit logging is enabled, each `dispatch_tool_call` invocation flows through Relay's hooks — you get a per-tool record of (server, tool, arguments, latency, success/failure) for free.
