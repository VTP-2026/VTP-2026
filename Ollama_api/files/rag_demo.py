"""
rag_demo.py  --  A minimal RAG pipeline running 100% LOCALLY with Ollama.

Same three steps as before, but nothing leaves your machine and there is no API key:
    1. INDEX     Ollama turns documents into embeddings; Chroma stores them
    2. RETRIEVE  find the documents most similar to a question
    3. GENERATE  a local Ollama chat model answers using only those documents

Prerequisites (one-time):
    1. Install Ollama from https://ollama.com  and make sure it is running.
    2. Pull the two models this script uses:
         ollama pull nomic-embed-text
         ollama pull llama3.2

Run it:
    python rag_demo.py
"""

import sys
import ollama
import chromadb
from chromadb.config import Settings

EMBED_MODEL = "nomic-embed-text"   # a small, fast local embedding model
GEN_MODEL   = "llama3.2"           # any local chat model you have pulled

DOCUMENTS = [
    "The Eiffel Tower is located in Paris and stands about 330 metres tall.",
    "The Great Wall of China is more than 21,000 kilometres long.",
    "Mount Everest is the highest mountain on Earth at 8,849 metres.",
    "The Amazon is the largest river in the world by water discharge.",
    "Python is a programming language created by Guido van Rossum.",
    "RAG stands for Retrieval-Augmented Generation.",
]


def embed(texts):
    """Ask the local Ollama server to embed a list of texts -> list of vectors."""
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings


def main():
    # --- 1. INDEX: embed the documents locally and store them in Chroma ---
    store = chromadb.Client(Settings(anonymized_telemetry=False)).create_collection("facts")
    store.add(
        documents=DOCUMENTS,
        ids=[f"doc{i}" for i in range(len(DOCUMENTS))],
        embeddings=embed(DOCUMENTS),          # we pass embeddings in ourselves
    )

    question = "How tall is the Eiffel Tower?"
    print(f"\nQuestion: {question}\n")

    # --- 2. RETRIEVE: embed the question and find the closest documents ---
    hits = store.query(query_embeddings=embed([question]), n_results=2)
    top_docs = hits["documents"][0]
    print("Retrieved context (most similar documents):")
    for doc in top_docs:
        print(f"  - {doc}")

    # --- 3. GENERATE: a local model answers using only the retrieved text ---
    context = "\n".join(top_docs)
    reply = ollama.chat(model=GEN_MODEL, messages=[{
        "role": "user",
        "content": (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        ),
    }])
    print(f"\n{GEN_MODEL}'s grounded answer:\n  {reply.message.content}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Most common cause: the Ollama server isn't running, or a model isn't pulled.
        print(f"\n[error] {e}\n"
              "Is Ollama running? Did you run 'ollama pull nomic-embed-text' "
              "and 'ollama pull llama3.2'?", file=sys.stderr)
        sys.exit(1)
