from __future__ import annotations

from incident_agent.fixtures import load_metrics, search_logs
from incident_agent.graph import DiagnosisEngine
from incident_agent.models import ModelDecision
from incident_agent.schemas import IncidentReport, IncidentRequest, ToolCall
from incident_agent.tracing import TraceRecorder


class FakeToolClient:
    async def call_tool(self, name, arguments, timeout=5.0):
        if name == "get_metrics_snapshot":
            return load_metrics(arguments["service_id"])
        return search_logs(arguments["service_id"], arguments["keyword"], arguments.get("limit", 20))

    async def aclose(self):
        return None


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
    engine = DiagnosisEngine(model=FakeModel(), tool_client=FakeToolClient())
    assert engine.route_after_agent({"pending_tool_calls": []}) == "finalize"
