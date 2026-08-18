"""
mcp_server.py  --  A minimal MCP server that exposes the three MCP primitives.

MCP (Model Context Protocol) is a standard way for an AI app to connect to
external capabilities. A server can expose three kinds of things:

    TOOLS      actions the model can call        (e.g. add two numbers)
    RESOURCES  read-only data the host can load   (e.g. an app version)
    PROMPTS    reusable prompt templates          (e.g. "review this code")

Run it directly to serve over stdio (how hosts launch local servers):
    python mcp_server.py

Or test it with the included client:
    python mcp_client.py
"""

from mcp.server.fastmcp import FastMCP

# The server object. The name is how a host identifies this server.
mcp = FastMCP("Demo Server")


# ---- TOOLS -----------------------------------------------------------------
# The type hints (a: int, b: int) and the docstring become the schema the
# model sees -- you don't write any JSON schema by hand.

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b


@mcp.tool()
def word_count(text: str) -> int:
    """Count how many words are in the given text."""
    return len(text.split())


# ---- RESOURCE --------------------------------------------------------------
# Read-only data, addressed by a URI the host can load into context.

@mcp.resource("demo://app-version")
def app_version() -> str:
    """The current version of our demo app."""
    return "v1.0.0"


# ---- PROMPT ----------------------------------------------------------------
# A reusable prompt template a user can invoke.

@mcp.prompt()
def review_code(code: str) -> str:
    """Build a prompt asking the model to review a snippet of code."""
    return f"Please review this code and suggest improvements:\n\n{code}"


if __name__ == "__main__":
    mcp.run()   # defaults to the stdio transport
