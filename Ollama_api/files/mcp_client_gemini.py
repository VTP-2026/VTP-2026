"""
mcp_client_gemini.py  --  Let Gemini use tools from an MCP server.

The Google Gen AI SDK has built-in (experimental) MCP support: you pass the MCP
session straight into `tools`, and the SDK does the rest -- it discovers the
tools, lets the model pick one, calls it through MCP, and feeds the result back.
That whole agent loop is handled for you.

Prerequisites:
    * A Gemini API key set as GEMINI_API_KEY (https://aistudio.google.com/apikey)

Run (the server is launched for you):
    python mcp_client_gemini.py
"""

import os
import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types

GEN_MODEL = "gemini-2.5-flash"   # swap for another Gemini model if you like

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])


async def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Launch the MCP server and open a session to it.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Pass the MCP session directly as a tool. The SDK will discover the
            # server's tools and automatically call them when the model asks.
            response = await client.aio.models.generate_content(
                model=GEN_MODEL,
                contents="What is 5 + 3? Use the available tools to be sure.",
                config=types.GenerateContentConfig(
                    tools=[session],   # <- native MCP support
                ),
            )
            print("Final answer:", response.text)


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey).",
              file=sys.stderr)
        sys.exit(1)
    asyncio.run(main())
