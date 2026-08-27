import pytest

from incident_agent.graph import DiagnosisEngine
from incident_agent.models import RuleBasedModel
from incident_agent.tracing import TraceRecorder


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
async def test_end_to_end_diagnosis(tmp_path, service_id, symptom, expected):
    engine = DiagnosisEngine(
        model=RuleBasedModel(),
        tracer=TraceRecorder(tmp_path / "trace.jsonl"),
        max_steps=5,
    )
    report = await engine.diagnose({"service_id": service_id, "symptom": symptom})
    assert report.incident_type == expected
    assert report.evidence
    assert "get_metrics_snapshot" in report.tools_used
    assert "search_logs" in report.tools_used


def test_conditional_route():
    engine = DiagnosisEngine(model=RuleBasedModel())
    assert engine.route_after_agent({"pending_tool_calls": []}) == "finalize"

