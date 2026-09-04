import asyncio

import pytest

from incident_agent.rag.bm25 import BM25Retriever
from incident_agent.rag.chunking import DocumentChunk
from incident_agent.rag.errors import RAGDependencyError
from incident_agent.rag.fusion import RRFFusion
from incident_agent.rag.reranker import CrossEncoderReranker
from incident_agent.rag.retriever import HybridRetriever
from incident_agent.rag.tokenization import tokenize


def chunks():
    return [
        DocumentChunk(
            "vLLM gpu_util 低，GPU 利用率异常，检查 scheduler queue。",
            "gpu.md",
            "metrics",
        ),
        DocumentChunk(
            "CUDA out of memory: KV Cache 分配失败，检查 gpu_memory_utilization。",
            "oom.md",
            "symptom",
        ),
        DocumentChunk(
            "NCCL tensor parallel 通信异常，检查网络和 rank。",
            "nccl.md",
            "logs",
        ),
    ]


def test_tokenization_preserves_mixed_serving_terms():
    values = tokenize("vLLM gpu_util=95%，检查 CUDA OOM 和请求排队")
    assert {"vllm", "gpu_util", "95%", "cuda", "oom", "请求排队"}.issubset(values)


def test_bm25_retrieves_relevant_chunk():
    pytest.importorskip("rank_bm25")
    pytest.importorskip("jieba")
    result = BM25Retriever(chunks()).retrieve("KV Cache 显存不足", k=2)
    assert result[0]["source"] == "oom.md"
    assert result[0]["bm25_rank"] == 1


def test_rrf_deduplicates_and_uses_rank_formula():
    fused = RRFFusion(60).fuse(
        [{"chunk_id": "a", "bm25_rank": 1, "bm25_score": 4.0, "text": "a"},
         {"chunk_id": "b", "bm25_rank": 2, "bm25_score": 3.0, "text": "b"}],
        [{"chunk_id": "a", "dense_rank": 2, "dense_score": .8, "text": "a"},
         {"chunk_id": "c", "dense_rank": 1, "dense_score": .9, "text": "c"}],
        k=10,
    )
    assert [item["chunk_id"] for item in fused] == ["a", "c", "b"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert len({item["chunk_id"] for item in fused}) == 3
    assert fused[1]["bm25_score"] is None
    assert fused[1]["bm25_rank"] is None
    assert fused[1]["dense_score"] == pytest.approx(.9)


class FakeDense:
    device = "cuda"
    model_name = "fake-embedding"

    async def retrieve(self, query, k=20):
        await asyncio.sleep(0)
        return [
            {"chunk_id": "oom.md::0::symptom", "text": "oom", "dense_score": .95, "dense_rank": 1},
            {"chunk_id": "nccl.md::0::logs", "text": "nccl", "dense_score": .7, "dense_rank": 2},
        ][:k]


class FakeBM25:
    def retrieve(self, query, k=20):
        return [
            {"chunk_id": "oom.md::0::symptom", "text": "oom", "bm25_score": 5, "bm25_rank": 1},
            {"chunk_id": "gpu.md::0::metrics", "text": "gpu", "bm25_score": 2, "bm25_rank": 2},
        ][:k]


class FakeCrossEncoder:
    model_name = "fake-reranker"

    def rerank(self, query, candidates, k=3):
        output = []
        for item in candidates:
            value = dict(item)
            value["reranker_score"] = 10.0 if value["chunk_id"].startswith("oom") else 1.0
            output.append(value)
        output.sort(key=lambda item: -item["reranker_score"])
        for rank, item in enumerate(output[:k], 1):
            item["reranker_rank"] = rank
        return output[:k]


@pytest.mark.asyncio
async def test_hybrid_default_top_three_and_stage_payload():
    tracer = []

    class Tracer:
        def event(self, run_id, name, payload):
            tracer.append((name, payload))

    retriever = HybridRetriever(
        chunks=chunks(), bm25=FakeBM25(), dense=FakeDense(), reranker=FakeCrossEncoder(), tracer=Tracer()
    )
    result = await retriever.ainvoke("CUDA OOM")
    assert len(result) == 3
    event = next(payload for name, payload in tracer if name == "rag.hybrid")
    assert event["bm25_count"] == 2
    assert event["dense_count"] == 2
    assert event["fusion_count"] == 3
    assert event["final_results"] == result


def test_strict_missing_dependency_error_is_actionable(monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ImportError("blocked for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RAGDependencyError, match="rank-bm25"):
        BM25Retriever(chunks())


def test_reranker_returns_only_requested_top_k():
    class Model:
        def predict(self, pairs):
            return [float(len(pair[1])) for pair in pairs]

    result = CrossEncoderReranker(model=Model(), device="cpu").rerank(
        "q", [{"chunk_id": str(i), "text": "x" * i} for i in range(5)], k=3
    )
    assert len(result) == 3
    assert [item["reranker_rank"] for item in result] == [1, 2, 3]
