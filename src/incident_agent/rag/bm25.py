from __future__ import annotations

from typing import Any

from .chunking import DocumentChunk
from .errors import RAGDependencyError
from .tokenization import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk], *, k1: float = 1.2, b: float = 0.75) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RAGDependencyError(
                "RAG unavailable: missing dependency 'rank-bm25' for BM25 retrieval; "
                "install it with pip install rank-bm25"
            ) from exc
        self.chunks = chunks
        self._tokens = [tokenize(chunk.text + " " + chunk.source) for chunk in chunks]
        self._index = BM25Okapi(self._tokens, k1=k1, b=b)

    def retrieve(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        output: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked[:k], start=1):
            item = self.chunks[index].as_dict()
            item.update({"bm25_score": float(scores[index]), "bm25_rank": rank})
            output.append(item)
        return output
