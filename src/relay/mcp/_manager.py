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
import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from relay._internal.schema_validate import validate_tool_arguments as _validate_tool_arguments
from relay.errors import ConfigError, RelayError
from relay.types import ToolDefinition

if TYPE_CHECKING:
    pass


class MCPToolError(RelayError):
    """An MCP tool call failed."""


_NAME_SEP = "__"

# Allowlist of basenames permitted as the stdio `command` for an MCP server.
# Mirrors the LiteLLM mitigation for CVE-2026-30623 (MCP stdio command injection).
# Operators with a justified need can bypass this by passing
# ``allow_arbitrary=True`` to :meth:`MCPManager.add_stdio` (intended for tests
# and tightly-controlled deployments where the command source is trusted).
MCP_STDIO_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"npx", "uvx", "python", "python3", "node", "docker", "deno", "bun"}
)

# Commands that fetch + execute code named in args (package runners, image
# pullers, etc.). For these, allowlisting the *command* is not enough — the
# args also need explicit operator confirmation because `npx -y attacker-pkg`
# is just as much a command-injection vector as `command=/bin/sh`.
_PACKAGE_RUNNER_COMMANDS: frozenset[str] = frozenset({"npx", "uvx", "bunx", "docker"})

# Cap on the concatenated text returned by an MCP tool call. Past this the
# remainder is dropped and a sentinel is appended so the model can still reason
# about the truncation.
MCP_TOOL_RESULT_MAX_BYTES = 256 * 1024


def _basename_lower(command: str) -> str:
    """Filesystem basename, lowercased, with trailing ``.exe`` stripped."""
    base = os.path.basename(command).lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base


def _validate_stdio_command(command: str, *, allow_arbitrary: bool) -> None:
    """Reject MCP stdio commands whose basename is not on the allowlist."""
    if allow_arbitrary:
        return
    basename = _basename_lower(command)
    if basename not in MCP_STDIO_ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(MCP_STDIO_ALLOWED_COMMANDS))
        raise ConfigError(
            f"MCP stdio command {command!r} is not in the allowlist ({allowed}). "
            "Pass allow_arbitrary=True if you fully trust the source of this "
            "command (CVE-2026-30623-class command injection if untrusted)."
        )


def _extract_package_arg(command: str, args: list[str]) -> str | None:
    """For commands that fetch + execute code named in ``args`` (npx, uvx,
    docker), return the first positional argument — that's the package/image
    name that will be downloaded and run.

    Returns ``None`` for commands that don't fetch external code.
    """
    basename = _basename_lower(command)
    if basename not in _PACKAGE_RUNNER_COMMANDS:
        return None
    if basename == "docker":
        # ``docker run <image>`` / ``docker exec <ctr>``. Find the verb, then
        # the first non-flag arg after it.
        verbs = {"run", "exec", "start", "create"}
        for i, a in enumerate(args):
            if a in verbs:
                for follow in args[i + 1 :]:
                    if not follow.startswith("-"):
                        return follow
        return None
    # npx / uvx / bunx — first positional that isn't a flag is the package.
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            # ``--from foo`` style: the following token is a value, not the package.
            if a in {"--from", "--with", "--package", "-p"}:
                skip_next = True
            continue
        return a
    return None


def _validate_stdio_args(command: str, args: list[str], *, allow_arbitrary: bool) -> None:
    """Reject package-runner commands that ship an external package via args
    unless the operator opts in.

    ``allow_arbitrary=True`` is the single kill-switch — it says "I trust the
    full ``command + args`` tuple, including any package name that will be
    fetched and executed." Setting it is necessary for legitimate uses like
    ``add_stdio("github", command="npx", args=["-y",
    "@modelcontextprotocol/server-github"], allow_arbitrary=True)``.

    Without this gate, the allowlist on ``command`` alone is bypassable:
    ``command="npx", args=["-y", "attacker-pkg"]`` ships arbitrary npm code
    even though ``npx`` is allowlisted (the same bypass class as
    CVE-2026-30623 but on the args axis).
    """
    if allow_arbitrary:
        return
    pkg = _extract_package_arg(command, args)
    if pkg is None:
        return
    raise ConfigError(
        f"MCP stdio command {command!r} would fetch and execute package "
        f"{pkg!r} from external args. Pass allow_arbitrary=True if the args "
        "are hardcoded in trusted code and you understand that any positional "
        "argument here is treated as code-to-execute (npx, uvx, docker run, "
        "etc.). Never wire add_stdio to untrusted input."
    )


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
        allow_arbitrary_command: bool = False,
    ) -> None:
        self.name = name
        self.transport = transport
        self.config = config
        # Carried into connect() so a serialized config from a less-trusted
        # source cannot bypass the allowlist by mutating the dict.
        self.allow_arbitrary_command = allow_arbitrary_command
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
            # Re-validate the stdio command BEFORE importing the mcp SDK so
            # this defense fires (and is testable in CI) even when ``mcp``
            # is not installed. A deserialized YAML or hand-built MCPServer
            # cannot launch an arbitrary binary regardless of SDK presence.
            if self.transport == "stdio":
                _validate_stdio_command(
                    self.config["command"],
                    allow_arbitrary=self.allow_arbitrary_command,
                )
                _validate_stdio_args(
                    self.config["command"],
                    self.config.get("args") or [],
                    allow_arbitrary=self.allow_arbitrary_command,
                )
            try:
                from mcp import ClientSession  # type: ignore[import-not-found,import-untyped]
            except ImportError as e:
                raise ConfigError(
                    "MCP support requires the 'mcp' package. "
                    "Install with: pip install ai5labs-relay[mcp]"
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
        joined = "\n".join(out_parts)
        # Cap result size so a malicious / runaway server cannot blow the LLM's
        # context (or our memory). UTF-8 bytes, not characters.
        encoded = joined.encode("utf-8")
        if len(encoded) > MCP_TOOL_RESULT_MAX_BYTES:
            truncated = encoded[:MCP_TOOL_RESULT_MAX_BYTES].decode("utf-8", errors="ignore")
            return f"{truncated}\n[result truncated at {MCP_TOOL_RESULT_MAX_BYTES} bytes]"
        return joined

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
        allow_arbitrary: bool = False,
    ) -> None:
        """Connect to an MCP server over stdio (most servers ship as binaries).

        Two defenses run, both gated by ``allow_arbitrary=True``:

        * ``command`` basename must be in :data:`MCP_STDIO_ALLOWED_COMMANDS`
          (CVE-2026-30623 class).
        * If ``command`` is a package runner (``npx``, ``uvx``, ``bunx``,
          ``docker``) and ``args`` contains a positional argument, that
          argument is the package/image that will be fetched and executed —
          equivalent to a fresh ``command``. Reject unless
          ``allow_arbitrary=True``.

        Pass ``allow_arbitrary=True`` only for tests or trusted operators —
        ``add_stdio`` must never receive untrusted input regardless.
        """
        if name in self._servers:
            raise ConfigError(f"MCP server {name!r} already added")
        _validate_stdio_command(command, allow_arbitrary=allow_arbitrary)
        _validate_stdio_args(command, args or [], allow_arbitrary=allow_arbitrary)
        server = MCPServer(
            name=name,
            transport="stdio",
            config={"command": command, "args": args or [], "env": env},
            allow_arbitrary_command=allow_arbitrary,
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
        """Dispatch a tool call by its prefixed name to the right server.

        Arguments are validated against the originating tool's JSON Schema
        before dispatch. A schema violation raises :class:`ToolSchemaError`
        so callers can fail closed rather than forwarding garbage to a tool
        that might trust its inputs.
        """
        if _NAME_SEP not in name:
            raise MCPToolError(
                f"tool name {name!r} is not server-prefixed; "
                f"expected '{{server}}{_NAME_SEP}{{tool}}'"
            )
        server_name, _, tool_name = name.partition(_NAME_SEP)
        server = self._servers.get(server_name)
        if server is None:
            raise MCPToolError(f"unknown MCP server {server_name!r}")

        # Validate against the declared parameter schema before invoking.
        schema = await self._tool_schema(server, tool_name)
        if schema is not None:
            _validate_tool_arguments(name, arguments, schema)

        return await server.call_tool(tool_name, arguments)

    async def _tool_schema(self, server: MCPServer, tool_name: str) -> dict[str, Any] | None:
        """Return the declared JSON Schema for ``tool_name`` on ``server``."""
        for t in await server.list_tools():
            if t.name == tool_name:
                return t.parameters
        return None

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
