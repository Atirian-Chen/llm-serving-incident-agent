from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_agent.rag.retriever import ChromaBGERetriever  # noqa: E402


if __name__ == "__main__":
    retriever = ChromaBGERetriever()
    print(f"Indexed {len(retriever._chunks)} chunks into Chroma.")

