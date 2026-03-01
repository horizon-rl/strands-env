# MCP Environment

Base environment for connecting a Strands agent to an [MCP](https://modelcontextprotocol.io/) server. Tools exposed by the server are automatically discovered and made available to the agent.

## Setup

No additional dependencies required beyond `strands-env` (`MCPClient` is provided by `strands-agents-tools`, a core dependency).

## Usage

```python
from strands.tools.mcp import MCPClient
from strands_env.environments.mcp import MCPEnvironment

client = MCPClient(lambda: stdio_client(server_params))
env = MCPEnvironment(
    model_factory=model_factory,
    mcp_client=client,
)
await env.reset()       # starts the MCPClient
result = await env.step(action)
await env.cleanup()     # stops the MCPClient
```

## Subclassing

For environments that need to start a server before connecting, subclass `MCPEnvironment` and set `self._mcp_client` in `reset()`:

```python
class MyMCPEnvironment(MCPEnvironment):
    async def reset(self) -> None:
        # Start server, build URL, etc.
        self._mcp_client = MCPClient(lambda: streamable_http_client(url))
        await super().reset()  # starts the client

    async def cleanup(self) -> None:
        # Kill server, remove temp files, etc.
        await super().cleanup()  # stops the client
```

## Tools

All tools exposed by the MCP server are automatically discovered via `MCPClient.list_tools_sync()`.

## Reward

No built-in reward function. Supply a custom `reward_fn`.
