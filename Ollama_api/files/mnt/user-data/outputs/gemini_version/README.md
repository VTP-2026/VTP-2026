# RAG & MCP — Gemini API Version

The same two demos, powered by Google's **Gemini** API.

| File | Shows |
|------|-------|
| `rag_demo.py` | RAG: Gemini embeddings + Chroma + Gemini generation |
| `mcp_server.py` | The same MCP server (unchanged — it is model-agnostic) |
| `mcp_client_gemini.py` | Gemini calling MCP tools via the SDK's built-in MCP support |

## Requirements

- **Python 3.10+**
- The packages in `requirements.txt`
- A **Gemini API key** (free tier available) from https://aistudio.google.com/apikey

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# set your key
export GEMINI_API_KEY="your-key"   # Windows (PowerShell): setx GEMINI_API_KEY "your-key"
```

## Run the RAG demo

```bash
python rag_demo.py
```

Gemini embeds the documents and the question, Chroma finds the closest matches,
and Gemini answers from them. Note the `task_type` on the embeddings
(`RETRIEVAL_DOCUMENT` for the stored text, `RETRIEVAL_QUERY` for the question) —
a small Gemini feature that improves retrieval quality.

## Run the MCP demo

```bash
python mcp_client_gemini.py
```

This one is short on purpose: the Gen AI SDK has **built-in MCP support**, so you
just pass the MCP `session` into `tools=[...]`. The SDK discovers the server's
tools, lets Gemini choose one, calls it through MCP, and returns the final answer.
Expected output:

```
Final answer: 5 + 3 = 8.
```

## Notes

- Models are swappable: change `GEN_MODEL` (e.g. `gemini-2.5-flash`) or
  `EMBED_MODEL` (`gemini-embedding-001`) at the top of each script.
- The MCP support in the SDK is marked experimental, but the `mcp_server.py` here
  is a standard server — the very same one works with the Ollama and Anthropic
  versions too.
