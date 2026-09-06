from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from incident_agent.graph import DiagnosisEngine
from incident_agent.models import DeepSeekModel, LLMUnavailableError
from incident_agent.schemas import IncidentReport, IncidentRequest
from incident_agent.tracing import TraceRecorder


@pytest.fixture
def ranked_docs():
    docs = []
    for rank, name in enumerate(("queue", "batching", "prefill"), 1):
        doc = {
            "chunk_id": f"{name}.md::symptoms",
            "text": f"Inspect {name} metrics and worker logs.",
            "source": f"{name}.md",
            "title": name.title(),
            "section": "symptoms",
            "bm25_score": 10.0 - rank,
            "bm25_rank": 4 - rank,
            "dense_score": 0.95 - rank / 10,
            "dense_rank": rank,
            "rrf_score": 1 / (60 + rank) + 1 / (64 - rank),
            "rrf_rank": 4 - rank,
            "reranker_score": 0.9 - rank / 10,
            "reranker_rank": rank,
            "score": 42,
            "rank": rank,
            "debug": {"device": "cuda", "latency_ms": 12},
        }
        doc["metadata"] = {key: doc[key] for key in ("source", "title", "section")}
        docs.append(doc)
    docs[0]["metadata"].update({
        "framework": "vllm",
        "version": "0.6.3",
        "applicability": ["tensor_parallel_size > 1"],
        "updated_at": "2026-09-06",
        "bm25_score": 9.0,
        "reranker_rank": 1,
        "debug": {"rrf_score": 0.03},
    })
    docs[1]["metadata"]["version"] = None
    docs[2].pop("metadata")
    return docs


def expected_runbook(docs):
    projected = [
        {key: doc[key] for key in ("chunk_id", "text", "source", "title", "section")}
        for doc in docs
    ]
    if projected:
        projected[0]["metadata"] = {
            "framework": "vllm",
            "version": "0.6.3",
            "applicability": ["tensor_parallel_size > 1"],
            "updated_at": "2026-09-06",
        }
    return projected


@pytest.mark.parametrize("count", [0, 1, 3])
def test_json_context_preserves_ranked_evidence_without_telemetry(ranked_docs, count):
    docs = ranked_docs[:count]
    original = copy.deepcopy(docs)
    request = IncidentRequest(service_id="demo-high-ttft", symptom="vLLM queue is growing")
    tools = [{"name": "get_metrics_snapshot", "result": {"score": 0.7, "rank": 2}}]
    context = json.loads(DeepSeekModel._json_context(request, docs, tools, "inspect evidence"))

    assert context == {
        "request": request.model_dump(),
        "runbook": expected_runbook(docs),
        "tool_results": tools,
        "instruction": "inspect evidence",
    }
    assert docs == original


def test_json_context_accepts_null_metadata():
    doc = {"chunk_id": "minimal", "text": "Collect evidence", "metadata": None}
    assert DeepSeekModel._runbook_context([doc]) == [
        {"chunk_id": "minimal", "text": "Collect evidence"}
    ]
    assert doc["metadata"] is None


@pytest.mark.asyncio
async def test_graph_sends_clean_context_each_round_and_keeps_full_trace(tmp_path, ranked_docs):
    original = copy.deepcopy(ranked_docs)
    request = IncidentRequest(service_id="demo-high-ttft", symptom="vLLM queue is growing")
    arguments = {"service_id": request.service_id}
    tool_result = {"waiting_requests": 12, "p99_ms": 1800}
    tool_results = [{
        "name": "get_metrics_snapshot", "arguments": arguments, "result": tool_result,
    }]
    model = DeepSeekModel.__new__(DeepSeekModel)
    model._tool_llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{
            "id": "metrics-1", "name": "get_metrics_snapshot", "args": arguments,
        }]),
        AIMessage(content="Queue pressure observed; inspect capacity."),
    ]))
    model._structured_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=IncidentReport(
        incident_type="high_ttft",
        severity="medium",
        root_cause="Queue pressure",
        evidence=[{"source": "queue.md", "detail": "Waiting requests increased"}],
        recommended_actions=["Inspect serving capacity"],
        tools_used=[],
        confidence=0.7,
    )))
    retriever = SimpleNamespace(ainvoke=AsyncMock(return_value=ranked_docs))
    tools = SimpleNamespace(call_tool=AsyncMock(return_value=tool_result), aclose=AsyncMock())
    trace_path = tmp_path / "trace.jsonl"
    engine = DiagnosisEngine(
        retriever=retriever, model=model, tool_client=tools, tracer=TraceRecorder(trace_path),
    )
    try:
        report = await engine.diagnose(request)
    finally:
        await engine.aclose()

    retriever.ainvoke.assert_awaited_once_with(request.symptom, k=3)
    tools.call_tool.assert_awaited_once_with(
        "get_metrics_snapshot", arguments, timeout=engine.tool_timeout,
    )
    assert report.incident_type == "high_ttft"
    assert report.tools_used == ["get_metrics_snapshot"]

    messages = [call.args[0] for call in model._tool_llm.ainvoke.await_args_list]
    messages += [model._structured_llm.ainvoke.await_args.args[0]]
    assert len(messages) == 3
    for index, batch in enumerate(messages):
        context = json.loads(batch[1].content)
        assert context["request"] == request.model_dump()
        assert context["runbook"] == expected_runbook(original)
        assert len(context["runbook"]) == 3
        assert context["tool_results"] == ([] if index == 0 else tool_results)

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    retrieval = next(item["payload"] for item in records if item["name"] == "rag.final_results")
    assert retrieval["results"] == original
    raw_requests = [item["payload"] for item in records if item["name"] == "llm.raw_request"]
    assert [item["phase"] for item in raw_requests] == [
        "tool_selection", "tool_selection", "final_user_response",
    ]
    for recorded, sent in zip(raw_requests, messages, strict=True):
        assert recorded["messages"] == [model._message_dict(message) for message in sent]
    for item in records:
        if item["name"] in {"llm.request", "llm.final_response"}:
            assert item["payload"]["retrieved_docs"] == original
    assert ranked_docs == original


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["decide", "finalize"])
async def test_llm_failure_still_raises_without_fallback(ranked_docs, phase):
    model = DeepSeekModel.__new__(DeepSeekModel)
    client = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("connection refused")))
    model._tool_llm = client
    model._structured_llm = client
    request = IncidentRequest(service_id="demo-high-ttft", symptom="vLLM queue is growing")
    with pytest.raises(LLMUnavailableError, match="connection refused"):
        await getattr(model, phase)(request, ranked_docs, [])
