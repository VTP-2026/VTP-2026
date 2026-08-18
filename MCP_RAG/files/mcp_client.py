"""
mcp_client.py  --  A tiny MCP client that talks to mcp_server.py over stdio.

It launches the server as a subprocess, discovers what the server offers,
then calls a tool -- the same flow an AI host (like Claude Desktop) performs
under the hood when a model decides to use a tool.

Run:
    python mcp_client.py

You do NOT need to start the server yourself -- this client launches it.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Tell the client how to start the server process.
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


async def main():
    # Launch the server and open a read/write pipe to it.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 1. Handshake -- client and server negotiate capabilities.
            await session.initialize()

            # 2. Discover what the server exposes.
            tools = (await session.list_tools()).tools
            print("TOOLS:")
            for t in tools:
                print(f"  - {t.name}: {t.description}")

            resources = (await session.list_resources()).resources
            print("\nRESOURCES:")
            for r in resources:
                print(f"  - {r.uri}: {r.description}")

            prompts = (await session.list_prompts()).prompts
            print("\nPROMPTS:")
            for p in prompts:
                print(f"  - {p.name}: {p.description}")

            # 3. Call a tool -- this is what the model does when it needs one.
            print("\nCalling add(a=5, b=3) ...")
            result = await session.call_tool("add", {"a": 5, "b": 3})
            print("  Server returned:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
