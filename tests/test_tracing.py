import json

import pytest

from incident_agent.graph import DiagnosisEngine
from incident_agent.schemas import IncidentReport, IncidentRequest
from incident_agent.tracing import TraceRecorder


class Retriever:
    tracer = None

    async def ainvoke(self, query, k=3):
        return [{"chunk_id": "r1", "text": "runbook", "source": "runbook.md", "title": "Runbook", "metadata": {}}]


class Model:
    async def decide(self, request, retrieved_docs, tool_results):
        from incident_agent.models import ModelDecision
        return ModelDecision(text="no tools")

    async def finalize(self, request, retrieved_docs, tool_results, draft_text=""):
        return IncidentReport(
            incident_type="unknown", severity="low", root_cause="insufficient evidence",
            evidence=[{"source": "runbook.md", "detail": "runbook"}], recommended_actions=["inspect"],
            tools_used=[], confidence=.1,
        )


class Tools:
    async def aclose(self): pass


def test_engine_injects_trace_recorder_into_hybrid_like_retriever():
    class HybridLike(Retriever):
        tracer = None

    tracer = TraceRecorder()
    retriever = HybridLike()
    DiagnosisEngine(retriever=retriever, model=Model(), tool_client=Tools(), tracer=tracer)
    assert retriever.tracer is tracer


@pytest.mark.asyncio
async def test_trace_keeps_query_rag_and_final_response(tmp_path):
    path = tmp_path / "trace.jsonl"
    engine = DiagnosisEngine(retriever=Retriever(), model=Model(), tool_client=Tools(), tracer=TraceRecorder(path))
    await engine.diagnose(IncidentRequest(service_id="demo-unknown", symptom="GPU issue"))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    names = {record["name"] for record in records}
    assert {"rag.retrieve.start", "rag.final_results", "llm.request", "llm.response", "llm.final_response", "agent.end"}.issubset(names)
    assert any(record["payload"].get("query") == "GPU issue" for record in records if record["name"] == "rag.final_results")
