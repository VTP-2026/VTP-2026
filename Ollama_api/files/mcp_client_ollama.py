"""
mcp_client_ollama.py  --  Let a LOCAL Ollama model use tools from an MCP server.

This is the classic agent loop, fully local:
    1. Connect to mcp_server.py and discover its tools.
    2. Hand those tools to a local Ollama chat model.
    3. If the model asks to call a tool, run it through MCP and feed the result back.
    4. The model gives a final answer using the tool's output.

Prerequisites:
    * Ollama installed and running (https://ollama.com)
    * A tool-capable model pulled, e.g.:   ollama pull llama3.1

Run (the server is launched for you):
    python mcp_client_ollama.py
"""

import asyncio
from ollama import AsyncClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GEN_MODEL = "llama3.1"   # use a model that supports tool calling (llama3.1, qwen2.5, mistral)

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])


def to_ollama_tools(mcp_tools):
    """Convert MCP tool definitions into the schema Ollama expects."""
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema,   # MCP already gives us a JSON schema
        },
    } for t in mcp_tools]


async def main():
    ollama_client = AsyncClient()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discover the server's tools and describe them to the model.
            mcp_tools = (await session.list_tools()).tools
            tools = to_ollama_tools(mcp_tools)
            print("Tools offered to the model:", [t.name for t in mcp_tools])

            messages = [{"role": "user", "content": "What is 5 + 3? Use a tool to be sure."}]

            # 2. First turn: the model may respond with tool calls.
            reply = await ollama_client.chat(model=GEN_MODEL, messages=messages, tools=tools)
            messages.append(reply.message)

            # 3. Run any requested tool calls through the MCP server.
            for call in reply.message.tool_calls or []:
                print(f"Model wants to call: {call.function.name}({call.function.arguments})")
                result = await session.call_tool(call.function.name, call.function.arguments)
                output = result.content[0].text
                print(f"  MCP server returned: {output}")
                messages.append({"role": "tool", "content": output, "tool_name": call.function.name})

            # 4. Second turn: the model answers using the tool result.
            final = await ollama_client.chat(model=GEN_MODEL, messages=messages)
            print("\nFinal answer:", final.message.content)


if __name__ == "__main__":
    asyncio.run(main())
