"""MCPManager — connect to MCP servers, expose their tools to any provider.

Architecture
------------
* :class:`MCPServer` wraps one connection (stdio/sse/streamable-http transport).
  It maintains a long-lived MCP session and exposes ``list_tools()`` and
  ``call_tool(name, args)``.
* :class:`MCPManager` owns N servers, namespaces tools by server prefix, and
  dispatches inbound tool calls back to the right server.

The transport layer is delegated to the official ``mcp`` Python SDK. We don't
re-implement the protocol — we just provide a friendly surface that Relay's
chat path can consume.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from relay.errors import ConfigError, RelayError
from relay.types import ToolDefinition

if TYPE_CHECKING:
    pass


class MCPToolError(RelayError):
    """An MCP tool call failed."""


_NAME_SEP = "__"


class MCPServer:
    """A single MCP server connection.

    Construct via :class:`MCPManager` — direct instantiation is not part of the
    public API.
    """

    def __init__(
        self,
        *,
        name: str,
        transport: str,
        config: dict[str, Any],
    ) -> None:
        self.name = name
        self.transport = transport
        self.config = config
        self._session: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools_cache: list[ToolDefinition] | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish the underlying MCP session. Idempotent."""
        if self._session is not None:
            return
        async with self._lock:
            if self._session is not None:  # type: ignore[unreachable]
                return  # type: ignore[unreachable]
            try:
                from mcp import ClientSession  # type: ignore[import-not-found,import-untyped]
            except ImportError as e:
                raise ConfigError(
                    "MCP support requires the 'mcp' package. "
                    "Install with: pip install relayllm[mcp]"
                ) from e

            stack = AsyncExitStack()
            try:
                if self.transport == "stdio":
                    from mcp import (
                        StdioServerParameters,  # type: ignore[import-not-found,import-untyped]
                    )
                    from mcp.client.stdio import (
                        stdio_client,  # type: ignore[import-not-found,import-untyped]
                    )

                    params = StdioServerParameters(
                        command=self.config["command"],
                        args=self.config.get("args") or [],
                        env=self.config.get("env"),
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                elif self.transport == "sse":
                    from mcp.client.sse import (
                        sse_client,  # type: ignore[import-not-found,import-untyped]
                    )

                    read, write = await stack.enter_async_context(
                        sse_client(self.config["url"], headers=self.config.get("headers"))
                    )
                elif self.transport in ("streamable-http", "http"):
                    try:
                        from mcp.client.streamable_http import (  # type: ignore[import-not-found,import-untyped]
                            streamablehttp_client,
                        )
                    except ImportError as e:
                        raise ConfigError("streamable-http transport requires mcp>=1.0") from e
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(
                            self.config["url"], headers=self.config.get("headers")
                        )
                    )
                else:
                    raise ConfigError(
                        f"unknown MCP transport {self.transport!r}; "
                        "use 'stdio', 'sse', or 'streamable-http'"
                    )

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._exit_stack = stack
            except Exception:
                await stack.aclose()
                raise

    async def list_tools(self) -> list[ToolDefinition]:
        """Return the server's tools as Relay :class:`ToolDefinition` objects.

        Names are *not* server-prefixed here — that happens at the manager level
        when tools are merged across servers.
        """
        if self._tools_cache is not None:
            return self._tools_cache
        await self.connect()
        result = await self._session.list_tools()  # type: ignore[union-attr]
        tools: list[ToolDefinition] = []
        for t in result.tools:
            tools.append(
                ToolDefinition(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.inputSchema or {"type": "object", "properties": {}},
                )
            )
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on this server. Returns the textual result.

        :raises MCPToolError: if the tool errors or returns an error result.
        """
        await self.connect()
        try:
            result = await self._session.call_tool(name=name, arguments=arguments)  # type: ignore[union-attr]
        except Exception as e:
            raise MCPToolError(
                f"MCP server {self.name!r} tool {name!r} failed: {e}",
                provider=f"mcp:{self.name}",
            ) from e
        if getattr(result, "isError", False):
            raise MCPToolError(
                f"MCP tool {name!r} returned error result",
                provider=f"mcp:{self.name}",
                raw=result,
            )
        # Concatenate text content blocks; ignore non-text for now (v0.2 will
        # surface image/resource content as Relay content blocks).
        out_parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                out_parts.append(text)
        return "\n".join(out_parts)

    async def aclose(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._tools_cache = None


class MCPManager:
    """Owns N MCP server connections; merges their tools into one namespace.

    Tool names are prefixed with ``{server}__`` so two servers can both publish
    a ``create_issue`` tool without collision.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def list_servers(self) -> list[str]:
        return sorted(self._servers.keys())

    async def add_stdio(
        self,
        name: str,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Connect to an MCP server over stdio (most servers ship as binaries)."""
        if name in self._servers:
            raise ConfigError(f"MCP server {name!r} already added")
        server = MCPServer(
            name=name,
            transport="stdio",
            config={"command": command, "args": args or [], "env": env},
        )
        await server.connect()
        self._servers[name] = server

    async def add_sse(
        self,
        name: str,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Connect to an MCP server over Server-Sent Events."""
        if name in self._servers:
            raise ConfigError(f"MCP server {name!r} already added")
        server = MCPServer(name=name, transport="sse", config={"url": url, "headers": headers})
        await server.connect()
        self._servers[name] = server

    async def add_streamable_http(
        self,
        name: str,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Connect over the new streamable-http transport (mcp>=1.0)."""
        if name in self._servers:
            raise ConfigError(f"MCP server {name!r} already added")
        server = MCPServer(
            name=name,
            transport="streamable-http",
            config={"url": url, "headers": headers},
        )
        await server.connect()
        self._servers[name] = server

    async def list_tools(self) -> list[ToolDefinition]:
        """All tools across all servers, with server-prefixed names."""
        out: list[ToolDefinition] = []
        for server_name, server in self._servers.items():
            for t in await server.list_tools():
                prefixed = ToolDefinition(
                    name=f"{server_name}{_NAME_SEP}{t.name}",
                    description=t.description,
                    parameters=t.parameters,
                    strict=t.strict,
                )
                out.append(prefixed)
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call by its prefixed name to the right server."""
        if _NAME_SEP not in name:
            raise MCPToolError(
                f"tool name {name!r} is not server-prefixed; "
                f"expected '{{server}}{_NAME_SEP}{{tool}}'"
            )
        server_name, _, tool_name = name.partition(_NAME_SEP)
        server = self._servers.get(server_name)
        if server is None:
            raise MCPToolError(f"unknown MCP server {server_name!r}")
        return await server.call_tool(tool_name, arguments)

    async def aclose(self) -> None:
        results = await asyncio.gather(
            *(s.aclose() for s in self._servers.values()), return_exceptions=True
        )
        self._servers.clear()
        # Surface the first exception, if any, so callers can log it.
        for r in results:
            if isinstance(r, BaseException):
                raise r

    async def __aenter__(self) -> MCPManager:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
