# RAG & MCP — Classroom Demos

Three tiny, well-commented Python scripts for teaching two ideas:

| File | Shows |
|------|-------|
| `rag_demo.py` | A minimal **RAG** pipeline: index → retrieve → generate |
| `mcp_server.py` | A minimal **MCP** server exposing a tool, a resource, and a prompt |
| `mcp_client.py` | A client that connects to the server and calls a tool |

## Requirements

- **Python 3.10 or newer** (the MCP SDK needs 3.10+)
- The packages in `requirements.txt`
- Internet on first run (RAG downloads a small embedding model, ~80 MB, once)
- *(Optional)* an Anthropic API key — only for the "generate" step of RAG

## Setup (in VS Code's terminal)

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. install the dependencies
pip install -r requirements.txt
```

## Run the RAG demo

```bash
python rag_demo.py
```

The **retrieve** step runs locally and for free — students see the two most
relevant documents pulled from the knowledge base.

To also see Claude **generate** a grounded answer, set your key first:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
# Windows (PowerShell)
setx ANTHROPIC_API_KEY "sk-ant-..."

python rag_demo.py
```

Without a key the script still runs and just skips the generation step.

## Run the MCP demo

```bash
python mcp_client.py
```

You do **not** start the server yourself — the client launches `mcp_server.py`
as a subprocess, lists everything it exposes, then calls the `add` tool.
Expected output:

```
TOOLS:
  - add: Add two numbers together and return the result.
  - word_count: Count how many words are in the given text.

RESOURCES:
  - demo://app-version: The current version of our demo app.

PROMPTS:
  - review_code: Build a prompt asking the model to review a snippet of code.

Calling add(a=5, b=3) ...
  Server returned: 8
```

### Bonus: connect the server to Claude Desktop

`mcp_server.py` is a real MCP server. You can add it to Claude Desktop's
config (`claude_desktop_config.json`) so Claude itself can call the tools:

```json
{
  "mcpServers": {
    "demo": { "command": "python", "args": ["/full/path/to/mcp_server.py"] }
  }
}
```

## How the two connect

RAG gives a model **knowledge** (retrieved text); MCP gives a model a
standard **connection** to tools and data. A common real-world pattern is to
wrap a RAG pipeline as an MCP tool (e.g. `search_docs`) so any MCP host can
let the model search your documents on demand.
