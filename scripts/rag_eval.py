from __future__ import annotations

"""Evaluate retrieval stages without calling the online diagnosis LLM.

``--mode mock`` is deterministic and needs no model download. ``--mode real`` loads
the configured BGE/Chroma and Cross-Encoder models. ``--mode auto`` tries real mode
and records an explicit ``mock_fallback`` status if model loading is unavailable.
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_agent.rag.bm25 import BM25Retriever  # noqa: E402
from incident_agent.rag.chunking import DocumentChunk, load_markdown_chunks  # noqa: E402
from incident_agent.rag.dense import DenseRetriever  # noqa: E402
from incident_agent.rag.fusion import RRFFusion  # noqa: E402
from incident_agent.rag.reranker import CrossEncoderReranker  # noqa: E402
from incident_agent.rag.retriever import KeywordRetriever  # noqa: E402


QUERIES: list[dict[str, Any]] = [
    {"id": "q01", "query": "CUDA out of memory KV Cache 分配失败", "sources": ["cuda_oom.md", "ncuda_oom_kv_cache.md"]},
    {"id": "q02", "query": "gpu_memory_utilization 接近 1 显存不足", "sources": ["gpu_memory.md", "ncuda_oom_fragmentation.md"]},
    {"id": "q03", "query": "vLLM max_num_seqs 增大后 OOM", "sources": ["vllm_queue_batching.md", "cuda_oom.md"]},
    {"id": "q04", "query": "GPU 显存碎片 reserved allocated", "sources": ["ncuda_oom_fragmentation.md"]},
    {"id": "q05", "query": "KV Cache 使用率过高 block table", "sources": ["ncuda_oom_kv_cache.md", "kv_cache_capacity.md"]},
    {"id": "q06", "query": "PagedAttention KV cache vLLM", "sources": ["vllm_paged_attention.md"]},
    {"id": "q07", "query": "TTFT P95 首 token 延迟升高", "sources": ["ttft_tpot_p99.md", "high_ttft.md"]},
    {"id": "q08", "query": "TPOT decode 变慢 P99 长尾", "sources": ["ttft_tpot_p99.md", "prefill_decode_latency.md"]},
    {"id": "q09", "query": "vLLM waiting requests scheduler queue 排队", "sources": ["vllm_queue_batching.md", "scheduler_queue.md"]},
    {"id": "q10", "query": "continuous batching max_num_batched_tokens", "sources": ["continuous_batching.md"]},
    {"id": "q11", "query": "SGLang runtime 服务 5xx", "sources": ["sglang_runtime.md", "service_5xx_timeout.md"]},
    {"id": "q12", "query": "SGLang tokenizer backend crash", "sources": ["sglang_runtime.md", "model_loading_tokenizer.md"]},
    {"id": "q13", "query": "NCCL tensor parallel communication timeout", "sources": ["ncuda_nccl_tensor_parallel.md"]},
    {"id": "q14", "query": "NCCL network rank desync", "sources": ["ncuda_nccl_tensor_parallel.md", "node_failure_network.md"]},
    {"id": "q15", "query": "CUDA kernel launch illegal memory access", "sources": ["cuda_kernel_errors.md"]},
    {"id": "q16", "query": "Prometheus gpu_util ttft_p95_ms 指标", "sources": ["prometheus_metrics.md"]},
    {"id": "q17", "query": "日志 trace_id scheduler preempt", "sources": ["logging_observability.md", "vllm_queue_batching.md"]},
    {"id": "q18", "query": "服务超时 retry storm 限流", "sources": ["rate_limit_retry.md", "service_5xx_timeout.md"]},
    {"id": "q19", "query": "多实例负载不均 consistent hashing", "sources": ["multi_instance_load_balance.md"]},
    {"id": "q20", "query": "节点故障网络丢包 readiness", "sources": ["node_failure_network.md", "health_check_readiness.md"]},
    {"id": "q21", "query": "模型加载失败 tokenizer vocab mismatch", "sources": ["model_loading_tokenizer.md"]},
    {"id": "q22", "query": "throughput tokens per second regression", "sources": ["throughput_regression.md"]},
    {"id": "q23", "query": "tensor parallel quantization dtype memory", "sources": ["quantization_dtype.md", "ncuda_nccl_tensor_parallel.md"]},
    {"id": "q24", "query": "prefix cache hit rate low", "sources": ["prefix_cache.md", "ncaching_prefix_kv.md"]},
]


def _tokens(text: str) -> set[str]:
    # The mock dense encoder intentionally shares the production tokenizer.
    from incident_agent.rag.tokenization import tokenize

    return set(tokenize(text, require_jieba=False))


class MockDense:
    device = "mock-cpu"
    model_name = "mock-dense-token-cosine"

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self._sets = [_tokens(chunk.text + " " + chunk.source + " " + chunk.title) for chunk in chunks]

    async def retrieve(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        scored = []
        for chunk, tokens in zip(self.chunks, self._sets):
            # Cosine over binary token vectors, used only for deterministic evaluation.
            intersection = len(query_tokens & tokens)
            score = intersection / math.sqrt(max(1, len(query_tokens) * len(tokens)))
            scored.append((score, chunk))
        scored.sort(key=lambda row: (-row[0], row[1].chunk_id))
        output = []
        for rank, (score, chunk) in enumerate(scored[:k], 1):
            item = chunk.as_dict()
            item.update({"dense_score": float(score), "dense_rank": rank})
            output.append(item)
        return output


class MockReranker:
    model_name = "mock-cross-encoder-token-overlap"

    def rerank(self, query: str, candidates: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        ranked = []
        for item in candidates:
            score = len(query_tokens & _tokens(str(item.get("text", "")) + " " + str(item.get("source", ""))))
            result = dict(item)
            result["reranker_score"] = float(score)
            ranked.append(result)
        ranked.sort(key=lambda row: (-row["reranker_score"], str(row["chunk_id"])))
        for rank, item in enumerate(ranked[:k], 1):
            item["reranker_rank"] = rank
        return ranked[:k]


def _source_hits(results: list[dict[str, Any]], sources: list[str]) -> list[str]:
    allowed = set(sources)
    return [str(item["chunk_id"]) for item in results if str(item.get("source")) in allowed]


def _ndcg_at_3(results: list[dict[str, Any]], sources: list[str]) -> float:
    # Source-level labels are deduplicated so repeated chunks from one runbook
    # cannot make NDCG exceed 1.0.
    allowed = set(sources)
    seen: set[str] = set()
    hits = []
    for item in results[:3]:
        source = str(item.get("source"))
        hit = int(source in allowed and source not in seen)
        if hit:
            seen.add(source)
        hits.append(hit)
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(hits))
    ideal_count = min(3, len(sources))
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("keyword", "bm25", "dense", "rrf", "hybrid_reranker"):
        method_rows = [row for row in rows if row["method"] == name]
        if not method_rows:
            continue
        ranks = [row["first_relevant_rank"] for row in method_rows if row["first_relevant_rank"] is not None]
        result[name] = {
            "queries": len(method_rows),
            "mrr": statistics.mean((1 / rank if rank else 0.0) for rank in [row["first_relevant_rank"] for row in method_rows]),
            "ndcg_at_3": statistics.mean(row["ndcg_at_3"] for row in method_rows),
            "average_latency_ms": statistics.mean(row["latency_ms"] for row in method_rows),
            "p95_latency_ms": sorted(row["latency_ms"] for row in method_rows)[max(0, math.ceil(len(method_rows) * 0.95) - 1)],
            "relevant_found": len(ranks),
        }
        for k in (1, 3, 10):
            result[name][f"recall_at_{k}"] = statistics.mean(
                len({item["source"] for item in row["top_results"][:k]} & set(row["relevant_sources"]))
                / len(set(row["relevant_sources"]))
                for row in method_rows
            )
            result[name][f"hit_rate_at_{k}"] = statistics.mean(
                row["first_relevant_rank"] is not None and row["first_relevant_rank"] <= k
                for row in method_rows
            )
    return result


async def _run(mode: str) -> dict[str, Any]:
    chunks = load_markdown_chunks()
    keyword = KeywordRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    model_error: str | None = None
    actual_mode = mode
    dense: Any
    reranker: Any
    if mode in {"real", "auto"}:
        try:
            dense = DenseRetriever(chunks)
            reranker = CrossEncoderReranker()
        except Exception as exc:
            if mode == "real":
                raise
            actual_mode = "mock_fallback"
            model_error = f"{type(exc).__name__}: {exc}"
            dense, reranker = MockDense(chunks), MockReranker()
    else:
        dense, reranker = MockDense(chunks), MockReranker()

    rows: list[dict[str, Any]] = []
    fusion = RRFFusion()
    for case in QUERIES:
        for method in ("keyword", "bm25", "dense", "rrf", "hybrid_reranker"):
            started = time.perf_counter()
            if method == "keyword":
                results = await keyword.ainvoke(case["query"], k=10)
            elif method == "bm25":
                results = bm25.retrieve(case["query"], k=10)
            elif method == "dense":
                results = await dense.retrieve(case["query"], k=10)
            else:
                bm_results = bm25.retrieve(case["query"], k=20)
                dense_results = await dense.retrieve(case["query"], k=20)
                fused = fusion.fuse(bm_results, dense_results, k=30)
                results = fused[:10] if method == "rrf" else reranker.rerank(case["query"], fused, k=10)
            latency_ms = (time.perf_counter() - started) * 1000
            hit_ids = _source_hits(results, case["sources"])
            first_rank = next((index for index, item in enumerate(results, 1) if str(item.get("source")) in set(case["sources"])), None)
            rows.append({
                "query_id": case["id"],
                "query": case["query"],
                "relevant_sources": case["sources"],
                "method": method,
                "first_relevant_rank": first_rank,
                "relevant_chunk_ids": hit_ids,
                "ndcg_at_3": _ndcg_at_3(results, case["sources"]),
                "latency_ms": round(latency_ms, 4),
                "top_results": [
                    {"chunk_id": item.get("chunk_id"), "source": item.get("source"), "title": item.get("title"), "score": item.get("reranker_score", item.get("rrf_score", item.get("dense_score", item.get("bm25_score", item.get("keyword_score")))))}
                    for item in results[:10]
                ],
            })
    return {
        "evaluation_mode": actual_mode,
        "model_error": model_error,
        "chunk_count": len(chunks),
        "query_count": len(QUERIES),
        "models": {
            "embedding": getattr(dense, "model_name", "unknown"),
            "reranker": getattr(reranker, "model_name", "unknown"),
            "device": getattr(dense, "device", "unknown"),
        },
        "metrics": _metrics(rows),
        "queries": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RAG 检索评测",
        "",
        f"- 模式：`{payload['evaluation_mode']}`",
        f"- 知识库 chunk：{payload['chunk_count']}",
        f"- Query：{payload['query_count']}",
        f"- Embedding：`{payload['models']['embedding']}`；Reranker：`{payload['models']['reranker']}`；Device：`{payload['models']['device']}`",
        "",
        "标注粒度是 runbook source，尚不是人工逐 chunk 标注。Recall@k = 每条 query 前 k 条结果覆盖的相关 source 数 / 标注相关 source 总数，然后对该方法的 query 取平均；同一 source 只计一次。Hit@k 表示前 k 条至少命中一个相关 source 的 query 比例。",
        "MRR 在所有方法的 Top-10 上计算。NDCG@3 使用 source 去重增益，重复 source 不再得分。延迟包含各方法自身的检索阶段，不含模型加载；样本仅 24 条，不能据此推断生产性能。Keyword 是当前代码的 token-overlap 基线，不是原始中文单字分词版本。",
    ]
    if payload.get("provenance", {}).get("metrics_recomputed"):
        lines.extend(["", "本次从已保存的真实排序结果重算指标，沿用原始耗时，没有重新运行模型。旧版将命中数除以五种方法的总记录数（120），且将 Hit 误标为 Recall；此处已纠正。"])
    if payload.get("model_error"):
        lines.extend([f"- Real 模型未加载原因：`{payload['model_error']}`", "- 本结果是显式标记的 mock fallback，不代表真实模型效果。"])
    lines.extend([
        "",
        "| 方法 | Recall@1 | Recall@3 | Recall@10 | Hit@1 | Hit@3 | Hit@10 | MRR@10 | NDCG@3 | 平均 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, values in payload["metrics"].items():
        lines.append(
            f"| {name} | {values['recall_at_1']:.3f} | {values['recall_at_3']:.3f} | {values['recall_at_10']:.3f} | "
            f"{values['hit_rate_at_1']:.3f} | {values['hit_rate_at_3']:.3f} | {values['hit_rate_at_10']:.3f} | "
            f"{values['mrr']:.3f} | {values['ndcg_at_3']:.3f} | {values['average_latency_ms']:.3f} | {values['p95_latency_ms']:.3f} |"
        )
    if all(payload["metrics"]["hybrid_reranker"][metric] < payload["metrics"]["rrf"][metric] for metric in ("mrr", "ndcg_at_3")):
        lines.extend(["", "本组结果中，Cross-Encoder 的 MRR/NDCG@3 低于 RRF，不能宣称 reranker 带来了效果提升。应先复核相关性标注、chunk 粒度及重排输入，再做新的独立评测。"])
    lines.extend(["", "## Query 明细", "", "| Query | 方法 | 首个相关 rank | NDCG@3 | Top 来源 |", "| --- | --- | ---: | ---: | --- |"])
    for row in payload["queries"]:
        sources = ", ".join(str(item["source"]) for item in row["top_results"][:3])
        lines.append(f"| {row['query_id']} {row['query']} | {row['method']} | {row['first_relevant_rank'] or '未命中'} | {row['ndcg_at_3']:.3f} | {sources} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "real", "mock"), default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = asyncio.run(_run(args.mode))
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "rag_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(reports / "rag_eval.md", payload)
    print(json.dumps({"evaluation_mode": payload["evaluation_mode"], "metrics": payload["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
