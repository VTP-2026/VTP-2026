"""
rag_demo.py  --  A minimal RAG pipeline using the Google Gemini API.

Same three steps, powered by Gemini:
    1. INDEX     Gemini's embedding model turns documents into vectors; Chroma stores them
    2. RETRIEVE  find the documents most similar to a question
    3. GENERATE  a Gemini chat model answers using only those documents

Prerequisites:
    * Get a free API key from Google AI Studio: https://aistudio.google.com/apikey
    * Set it as an environment variable:
        macOS / Linux:  export GEMINI_API_KEY="your-key"
        Windows (PS):   setx GEMINI_API_KEY "your-key"

Run it:
    python rag_demo.py
"""

import os
import sys
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings

EMBED_MODEL = "gemini-embedding-001"   # Gemini's text embedding model
GEN_MODEL   = "gemini-2.5-flash"       # fast, inexpensive chat model (swap if you like)

DOCUMENTS = [
    "The Eiffel Tower is located in Paris and stands about 330 metres tall.",
    "The Great Wall of China is more than 21,000 kilometres long.",
    "Mount Everest is the highest mountain on Earth at 8,849 metres.",
    "The Amazon is the largest river in the world by water discharge.",
    "Python is a programming language created by Guido van Rossum.",
    "RAG stands for Retrieval-Augmented Generation.",
]

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def embed(texts, task_type):
    """Embed a list of texts with Gemini. task_type tunes the vectors for their role."""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in resp.embeddings]


def main():
    # --- 1. INDEX: embed documents (as 'documents') and store them ---
    store = chromadb.Client(Settings(anonymized_telemetry=False)).create_collection("facts")
    store.add(
        documents=DOCUMENTS,
        ids=[f"doc{i}" for i in range(len(DOCUMENTS))],
        embeddings=embed(DOCUMENTS, "RETRIEVAL_DOCUMENT"),
    )

    question = "How tall is the Eiffel Tower?"
    print(f"\nQuestion: {question}\n")

    # --- 2. RETRIEVE: embed the question (as a 'query') and find closest docs ---
    hits = store.query(
        query_embeddings=embed([question], "RETRIEVAL_QUERY"),
        n_results=2,
    )
    top_docs = hits["documents"][0]
    print("Retrieved context (most similar documents):")
    for doc in top_docs:
        print(f"  - {doc}")

    # --- 3. GENERATE: Gemini answers using only the retrieved text ---
    context = "\n".join(top_docs)
    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=(
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        ),
    )
    print(f"\nGemini's grounded answer:\n  {resp.text}\n")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey).",
              file=sys.stderr)
        sys.exit(1)
    main()
