from __future__ import annotations

import json

from incident_agent.fixtures import load_metrics, search_logs
from incident_agent.graph import DiagnosisEngine
from incident_agent.models import ModelDecision
from incident_agent.rag.chunking import DocumentChunk
from incident_agent.rag.retriever import HybridRetriever
from incident_agent.schemas import IncidentReport, IncidentRequest, ToolCall
from incident_agent.tracing import TraceRecorder


class FakeToolClient:
    async def call_tool(self, name, arguments, timeout=5.0):
        if name == "get_metrics_snapshot":
            return load_metrics(arguments["service_id"])
        return search_logs(arguments["service_id"], arguments["keyword"], arguments.get("limit", 20))

    async def aclose(self):
        return None


class FakeRetriever:
    """Keeps graph-flow tests independent from strict CUDA/model requirements."""

    async def ainvoke(self, query, k=3):
        return [
            {
                "chunk_id": "test-runbook::0::symptom",
                "text": f"Runbook guidance for {query}",
                "source": "test-runbook.md",
                "title": "Test runbook",
                "section": "symptom",
                "metadata": {"source": "test-runbook.md", "title": "Test runbook"},
                "bm25_score": 1.0,
                "bm25_rank": 1,
                "dense_score": 0.9,
                "dense_rank": 1,
                "rrf_score": 0.0328,
                "rrf_rank": 1,
                "reranker_score": 3.2,
                "reranker_rank": 1,
            }
        ][:k]


class FakeModel:
    """Test-only model injection; production always uses DeepSeekModel."""

    async def decide(self, request, retrieved_docs, tool_results):
        names = {item["name"] for item in tool_results}
        if "get_metrics_snapshot" not in names:
            return ModelDecision(
                [ToolCall(name="get_metrics_snapshot", arguments={"service_id": request.service_id})],
                "read metrics",
            )
        if "search_logs" not in names:
            return ModelDecision(
                [ToolCall(
                    name="search_logs",
                    arguments={"service_id": request.service_id, "keyword": "INFO", "limit": 5},
                )],
                "read logs",
            )
        return ModelDecision(text="enough evidence")

    async def finalize(self, request, retrieved_docs, tool_results, draft_text=""):
        expected = {
            "demo-oom": "cuda_oom",
            "demo-high-ttft": "high_ttft",
            "demo-prefix-cache": "low_prefix_cache_hit",
            "demo-unknown": "unknown",
        }[request.service_id]
        return IncidentReport(
            incident_type=expected,
            severity="low",
            root_cause="test result",
            evidence=[{"source": "test", "detail": "fixture evidence"}],
            recommended_actions=["collect evidence"],
            tools_used=[item["name"] for item in tool_results],
            confidence=0.5,
        )


import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_id", "symptom", "expected"),
    [
        ("demo-oom", "请求失败且 GPU 显存接近满载", "cuda_oom"),
        ("demo-high-ttft", "TTFT P95 很高且等待队列增长", "high_ttft"),
        ("demo-prefix-cache", "Prefix cache 命中率很低", "low_prefix_cache_hit"),
        ("demo-unknown", "没有明显故障信号，请分析", "unknown"),
    ],
)
async def test_end_to_end_diagnosis_with_injected_test_doubles(tmp_path, service_id, symptom, expected):
    engine = DiagnosisEngine(
        retriever=FakeRetriever(),
        model=FakeModel(),
        tool_client=FakeToolClient(),
        tracer=TraceRecorder(tmp_path / "trace.jsonl"),
        max_steps=5,
    )
    report = await engine.diagnose(IncidentRequest(service_id=service_id, symptom=symptom))
    assert report.incident_type == expected
    assert report.evidence
    assert "get_metrics_snapshot" in report.tools_used
    assert "search_logs" in report.tools_used


def test_conditional_route():
    engine = DiagnosisEngine(retriever=FakeRetriever(), model=FakeModel(), tool_client=FakeToolClient())
    assert engine.route_after_agent({"pending_tool_calls": []}) == "finalize"


class HybridTestRetriever:
    """A deterministic HybridRetriever assembled from stage-level test doubles."""

    def __init__(self, tracer):
        chunks = [
            DocumentChunk(
                "GPU OOM and KV Cache pressure: inspect gpu_memory_utilization and waiting queue.",
                "cuda_oom_runbook.md",
                "故障现象",
                chunk_id="cuda_oom_runbook.md::0",
            ),
            DocumentChunk(
                "High P99 and TTFT: compare prefill queue, decode TPOT, and batch size.",
                "latency_runbook.md",
                "指标检查",
                chunk_id="latency_runbook.md::0",
            ),
            DocumentChunk(
                "SGLang 5xx and NCCL errors: inspect worker logs, rank, and network health.",
                "runtime_runbook.md",
                "日志排障",
                chunk_id="runtime_runbook.md::0",
            ),
        ]

        class BM25:
            def retrieve(self, query, k=20):
                return [
                    {**chunks[0].as_dict(), "bm25_score": 8.0, "bm25_rank": 1},
                    {**chunks[1].as_dict(), "bm25_score": 5.0, "bm25_rank": 2},
                    {**chunks[2].as_dict(), "bm25_score": 3.0, "bm25_rank": 3},
                ][:k]

        class Dense:
            device = "test-cuda"
            model_name = "test-bge-small"

            async def retrieve(self, query, k=20):
                return [
                    {**chunks[1].as_dict(), "dense_score": 0.95, "dense_rank": 1},
                    {**chunks[0].as_dict(), "dense_score": 0.90, "dense_rank": 2},
                    {**chunks[2].as_dict(), "dense_score": 0.80, "dense_rank": 3},
                ][:k]

        class Reranker:
            model_name = "test-bge-reranker"

            def rerank(self, query, candidates, k=3):
                ranked = []
                for item in candidates:
                    result = dict(item)
                    result["reranker_score"] = float(1.0 + len(result["text"]) / 100)
                    ranked.append(result)
                ranked.sort(key=lambda item: (-item["reranker_score"], item["chunk_id"]))
                for rank, item in enumerate(ranked[:k], 1):
                    item["reranker_rank"] = rank
                return ranked[:k]

        self._inner = HybridRetriever(
            chunks=chunks,
            bm25=BM25(),
            dense=Dense(),
            reranker=Reranker(),
            tracer=tracer,
        )
        self.calls = []

    def set_trace_context(self, run_id):
        self._inner.set_trace_context(run_id)

    async def ainvoke(self, query, k=3):
        self.calls.append((query, k))
        return await self._inner.ainvoke(query, k=k)


class IncidentE2EModel:
    async def decide(self, request, retrieved_docs, tool_results):
        names = {item["name"] for item in tool_results}
        if "get_metrics_snapshot" not in names:
            return ModelDecision(
                tool_calls=[ToolCall(name="get_metrics_snapshot", arguments={"service_id": request.service_id})],
                text="need metrics",
            )
        if "search_logs" not in names:
            return ModelDecision(
                tool_calls=[ToolCall(
                    name="search_logs",
                    arguments={"service_id": request.service_id, "keyword": "error", "limit": 10},
                )],
                text="need logs",
            )
        return ModelDecision(text="evidence collected")

    async def finalize(self, request, retrieved_docs, tool_results, draft_text=""):
        symptom = request.symptom.casefold()
        if "oom" in symptom or "显存" in symptom:
            incident_type = "cuda_oom"
        elif "p99" in symptom or "延迟" in symptom or "排队" in symptom:
            incident_type = "high_ttft"
        elif "kv cache" in symptom:
            incident_type = "low_prefix_cache_hit"
        else:
            incident_type = "unknown"
        return IncidentReport(
            incident_type=incident_type,
            severity="medium",
            root_cause="test evidence",
            evidence=[{"source": "hybrid-test", "detail": "stage trace and tool evidence"}],
            recommended_actions=["inspect metrics and logs"],
            tools_used=[item["name"] for item in tool_results],
            confidence=0.8,
        )


class RecordingToolClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments, timeout=5.0):
        self.calls.append((name, dict(arguments), timeout))
        if name == "get_metrics_snapshot":
            return {"gpu_util": 0.92, "p99_ms": 1800, "kv_cache_usage": 0.94}
        return ["worker error: observed test signal"]

    async def aclose(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("symptom", "expected"),
    [
        ("GPU 显存不足 CUDA OOM", "cuda_oom"),
        ("P99 延迟升高且 TTFT 变慢", "high_ttft"),
        ("vLLM 请求排队 waiting queue 增长", "high_ttft"),
        ("SGLang 服务持续 5xx", "unknown"),
        ("KV Cache 使用率过高", "low_prefix_cache_hit"),
        ("NCCL 通信异常 timeout", "unknown"),
    ],
)
async def test_hybrid_e2e_incident_queries_keep_full_trace(tmp_path, symptom, expected):
    tracer = TraceRecorder(tmp_path / "trace.jsonl")
    retriever = HybridTestRetriever(tracer)
    tools = RecordingToolClient()
    engine = DiagnosisEngine(
        retriever=retriever,
        model=IncidentE2EModel(),
        tool_client=tools,
        tracer=tracer,
        max_steps=5,
    )
    report = await engine.diagnose(IncidentRequest(service_id="demo-unknown", symptom=symptom))
    assert report.incident_type == expected
    assert len(await retriever.ainvoke(symptom, k=3)) == 3
    assert retriever.calls[0] == (symptom, 3)
    assert [call[0] for call in tools.calls] == ["get_metrics_snapshot", "search_logs"]
    assert all(call[1]["service_id"] == "demo-unknown" for call in tools.calls)

    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    names = {record["name"] for record in records}
    assert {"rag.hybrid", "rag.final_results", "llm.request", "llm.response", "mcp.tool_result", "llm.final_response"}.issubset(names)
    hybrid = next(record["payload"] for record in records if record["name"] == "rag.hybrid")
    assert hybrid["bm25_count"] == 3 and hybrid["dense_count"] == 3
    assert len(hybrid["final_results"]) == 3
    assert all("reranker_score" in item for item in hybrid["final_results"])


@pytest.mark.asyncio
async def test_llm_failure_is_explicit_and_no_offline_fallback(tmp_path):
    class FailingModel:
        async def decide(self, request, retrieved_docs, tool_results):
            raise RuntimeError("DeepSeek connection refused")

    engine = DiagnosisEngine(
        retriever=FakeRetriever(),
        model=FailingModel(),
        tool_client=FakeToolClient(),
        tracer=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    with pytest.raises(RuntimeError, match="DeepSeek connection refused"):
        await engine.diagnose(IncidentRequest(service_id="demo-unknown", symptom="NCCL 通信异常 timeout"))
