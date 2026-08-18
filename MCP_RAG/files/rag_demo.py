"""
rag_demo.py  --  A minimal Retrieval-Augmented Generation (RAG) demo.

The whole idea in three steps:
    1. INDEX     turn documents into vectors and store them
    2. RETRIEVE  find the documents most similar to a question
    3. GENERATE  ask an LLM to answer using ONLY those documents

Run it:
    python rag_demo.py

Notes for the classroom:
  * The INDEX and RETRIEVE steps run 100% locally and for free.
    (Chroma downloads a small embedding model on first run.)
  * The GENERATE step calls Claude. It only runs if the environment
    variable ANTHROPIC_API_KEY is set, so the demo still works and
    shows retrieval even without a key.
"""

import os
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Our tiny "knowledge base".
# In a real system these would be your PDFs, wiki pages, or support tickets,
# split into small chunks. Here we keep it to a few sentences so it's obvious.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "The Eiffel Tower is located in Paris and stands about 330 metres tall.",
    "The Great Wall of China is more than 21,000 kilometres long.",
    "Mount Everest is the highest mountain on Earth at 8,849 metres.",
    "The Amazon is the largest river in the world by water discharge.",
    "Python is a programming language created by Guido van Rossum.",
    "RAG stands for Retrieval-Augmented Generation.",
]

MODEL = "claude-haiku-4-5"   # swap for any Claude model you have access to


def build_index():
    """Step 1 -- INDEX.

    Store the documents in a local vector database. Chroma automatically
    turns each document into an embedding (a list of numbers that captures
    its meaning) using a small model that runs on your own machine.
    """
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.create_collection("facts")
    collection.add(
        documents=DOCUMENTS,
        ids=[f"doc{i}" for i in range(len(DOCUMENTS))],
    )
    return collection


def retrieve(collection, question, k=2):
    """Step 2 -- RETRIEVE.

    Embed the question the same way and return the k documents whose vectors
    are closest to it (i.e. the most similar in meaning).
    """
    results = collection.query(query_texts=[question], n_results=k)
    return results["documents"][0]          # the top-k matching documents


def generate(question, context):
    """Step 3 -- GENERATE.

    Ask Claude to answer the question using only the retrieved context.
    Returns None (and we skip it) if anthropic isn't installed or there's
    no API key -- so the retrieval half always works in class.
    """
    try:
        import anthropic
    except ImportError:
        return None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    client = anthropic.Anthropic()
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def main():
    collection = build_index()
    question = "How tall is the Eiffel Tower?"
    print(f"\nQuestion: {question}\n")

    # --- RETRIEVE ---
    top_docs = retrieve(collection, question, k=2)
    print("Retrieved context (most similar documents):")
    for doc in top_docs:
        print(f"  - {doc}")

    # --- GENERATE ---
    context = "\n".join(top_docs)
    answer = generate(question, context)
    if answer:
        print(f"\nClaude's grounded answer:\n  {answer}\n")
    else:
        print("\n(Set ANTHROPIC_API_KEY to also see Claude generate an answer "
              "from the retrieved context.)\n")


if __name__ == "__main__":
    main()
