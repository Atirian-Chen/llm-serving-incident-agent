from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_agent.rag.chunking import load_markdown_chunks  # noqa: E402
from incident_agent.rag.dense import DenseRetriever  # noqa: E402


if __name__ == "__main__":
    chunks = load_markdown_chunks()
    retriever = DenseRetriever(chunks)
    print(
        f"Indexed {len(retriever.chunks)} chunks into Chroma collection "
        f"with cosine similarity on {retriever.device}."
    )
