from __future__ import annotations

import time
from typing import Any

from ..config import settings
from .bm25 import BM25Retriever
from .chunking import DocumentChunk, load_markdown_chunks
from .dense import DenseRetriever
from .errors import RAGConfigurationError
from .fusion import RRFFusion
from .reranker import CrossEncoderReranker
from .tokenization import tokenize


def _tokens(text: str) -> set[str]:
    """Backward-compatible token set helper used by the original baseline."""
    return set(tokenize(text))


class KeywordRetriever:
    """Deterministic baseline retained only for standalone retrieval evaluation."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_markdown_chunks()
        self._token_sets = [
            set(tokenize(chunk.text + " " + chunk.source, require_jieba=False))
            for chunk in self.chunks
        ]

    async def ainvoke(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(query, require_jieba=False))
        scored: list[tuple[float, int]] = []
        for index, token_set in enumerate(self._token_sets):
            overlap = len(query_tokens & token_set)
            source_bonus = sum(0.2 for token in query_tokens if token in self.chunks[index].source.casefold())
            scored.append((overlap + source_bonus, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))
        selected = [(score, self.chunks[index]) for score, index in scored if score > 0][:k]
        if not selected:
            selected = [(0.0, chunk) for chunk in self.chunks[:k]]
        output = []
        for rank, (score, chunk) in enumerate(selected, start=1):
            item = chunk.as_dict()
            item.update({"keyword_score": score, "keyword_rank": rank})
            output.append(item)
        return output


class HybridRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        *,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        tracer: Any | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_markdown_chunks()
        self.bm25 = bm25 or BM25Retriever(self.chunks, k1=1.2, b=0.75)
        self.dense = dense or DenseRetriever(self.chunks)
        self.reranker = reranker or CrossEncoderReranker()
        self.fusion = RRFFusion(settings.rag_rrf_k)
        self.tracer = tracer
        self._run_id = "unknown"

    def set_trace_context(self, run_id: str) -> None:
        self._run_id = run_id

    async def ainvoke(self, query: str, k: int = 3, **_: Any) -> list[dict[str, Any]]:
        final_k = k or settings.rag_final_top_k
        bm25_started = time.perf_counter()
        bm25_results = self.bm25.retrieve(query, settings.rag_bm25_top_k)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        dense_started = time.perf_counter()
        dense_results = await self.dense.retrieve(query, settings.rag_dense_top_k)
        dense_ms = (time.perf_counter() - dense_started) * 1000
        fusion_started = time.perf_counter()
        fused = self.fusion.fuse(bm25_results, dense_results, settings.rag_rrf_candidate_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        rerank_started = time.perf_counter()
        final = self.reranker.rerank(query, fused, final_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        payload = {
            "query": query,
            "bm25_count": len(bm25_results),
            "dense_count": len(dense_results),
            "dedup_count": len({item["chunk_id"] for item in bm25_results + dense_results}),
            "fusion_count": len(fused),
            "rerank_count": len(final),
            "final_top_k": final_k,
            "bm25_latency_ms": round(bm25_ms, 2),
            "dense_latency_ms": round(dense_ms, 2),
            "fusion_latency_ms": round(fusion_ms, 2),
            "rerank_latency_ms": round(rerank_ms, 2),
            "device": getattr(self.dense, "device", settings.rag_device),
            "embedding_model": getattr(self.dense, "model_name", settings.embedding_model),
            "reranker_model": getattr(self.reranker, "model_name", settings.rag_reranker_model),
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "rrf_results": fused,
            "final_results": final,
        }
        if self.tracer is not None:
            self.tracer.event(self._run_id, "rag.hybrid", payload)
        return final


class ChromaBGERetriever(DenseRetriever):
    """Compatibility name for the original dense-only implementation."""

    def __init__(self, chunks: list[DocumentChunk] | None = None, **kwargs: Any) -> None:
        super().__init__(chunks or load_markdown_chunks(), **kwargs)

    async def ainvoke(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        return await self.retrieve(query, k)


def build_retriever(provider: str | None = None):
    provider = (provider or settings.rag_provider).lower()
    if not settings.rag_strict:
        raise RAGConfigurationError(
            "RAG unavailable: RAG_STRICT must be true; production retrieval only supports strict hybrid RAG"
        )
    if provider != "hybrid":
        raise RAGConfigurationError(f"RAG unavailable: unsupported RAG_PROVIDER={provider!r}")
    return HybridRetriever()
