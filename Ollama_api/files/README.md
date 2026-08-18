# RAG & MCP — Local Version with Ollama

The same two demos as the original, but running **entirely on your own machine**
with [Ollama](https://ollama.com). No API key, no data leaves your computer.

| File | Shows |
|------|-------|
| `rag_demo.py` | Local RAG: Ollama embeddings + Chroma + a local Ollama chat model |
| `mcp_server.py` | The same MCP server (unchanged — it is model-agnostic) |
| `mcp_client_ollama.py` | A local Ollama model discovering and calling MCP tools |

## Requirements

- **Python 3.10+**
- The packages in `requirements.txt`
- The **Ollama app** installed and running: https://ollama.com
- The models pulled once from a terminal:
  ```bash
  ollama pull nomic-embed-text   # embeddings for RAG
  ollama pull llama3.2           # answer generation for RAG
  ollama pull llama3.1           # tool calling for the MCP client
  ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Ollama is running (the desktop app, or `ollama serve`). Ollama listens
on `http://localhost:11434`.

## Run the local RAG demo

```bash
python rag_demo.py
```

Everything is local: Ollama embeds the documents and the question, Chroma finds
the closest matches, and a local model answers from them.

## Run the local MCP demo

```bash
python mcp_client_ollama.py
```

The client launches `mcp_server.py`, lists its tools, hands them to a local model,
and — when the model decides to call `add` — runs that call through MCP and feeds
the result back so the model can finish its answer. Expected output:

```
Tools offered to the model: ['add', 'word_count']
Model wants to call: add({'a': 5, 'b': 3})
  MCP server returned: 8
Final answer: 5 + 3 = 8.
```

## Swapping models

Change `GEN_MODEL` / `EMBED_MODEL` at the top of each script to any model you have
pulled. For embeddings, `nomic-embed-text` (768-dim) or `mxbai-embed-large` (1024-dim)
work well. For tool calling, use a tool-capable model such as `llama3.1`, `qwen2.5`,
or `mistral`.
