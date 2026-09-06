from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_use_per_method_queries_and_distinct_relevant_sources():
    module = load_script("rag_eval")
    rows = []
    for method in ("keyword", "bm25", "dense", "rrf", "hybrid_reranker"):
        for sources, results, rank in (
            (["a", "b"], [{"source": "a"}, {"source": "a"}, {"source": "x"}, {"source": "b"}], 1),
            (["c"], [{"source": "x"}], None),
        ):
            rows.append({"method": method, "relevant_sources": sources, "top_results": results, "first_relevant_rank": rank, "ndcg_at_3": module._ndcg_at_3(results, sources), "latency_ms": 10})
    for values in module._metrics(rows).values():
        assert values["queries"] == 2
        assert values["hit_rate_at_1"] == 0.5
        assert values["hit_rate_at_3"] == 0.5
        assert values["recall_at_1"] == 0.25
        assert values["recall_at_3"] == 0.25
        assert values["recall_at_10"] == 0.5
        assert values["mrr"] == 0.5


@pytest.mark.parametrize("hybrid_count", [0, 1, 2])
def test_e2e_requires_evidence_for_every_run(tmp_path, hybrid_count):
    module = load_script("evaluate")
    events = [{"name": "agent.start", "run_id": str(i), "payload": {}} for i in range(2)]
    events += [{"name": "rag.hybrid", "run_id": str(i), "payload": {"final_top_k": 3}} for i in range(hybrid_count)]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    info = module.inspect_rag_trace(trace)
    assert info["provider"] == ("hybrid" if hybrid_count == 2 else None)
    assert info["final_top_k"] == (3 if hybrid_count == 2 else None)
    assert "strict" not in info


def test_online_example_uses_saved_messages_and_validates_tool_arguments(tmp_path):
    module = load_script("generate_example")
    trace = ROOT / "reports/traces/online_smoke.jsonl"
    events = module.load_trace(trace)
    summary = module.summarize_trace(events)
    assert summary["all_checks_passed"]
    assert summary["new_online_requests"] == 0
    assert summary["tool_call_count"] == 6
    assert summary["sample_count"] == 1
    changed = copy.deepcopy(events)
    tool = next(item for item in changed if item["name"] == "mcp.tool_result")
    tool["payload"]["arguments"]["service_id"] = "wrong-service"
    assert not module.summarize_trace(changed)["checks"]["mcp_arguments_valid"]
    output = tmp_path / "example.md"
    module.render_example(trace, output)
    document = output.read_text(encoding="utf-8")
    assert "历史请求可能仍含上述检索字段" in document
    assert document.index("MCP 返回") < document.index("第 2 轮请求")
    assert summary["request"]["symptom"] in document
    assert summary["report"]["root_cause"] in document


def test_example_rejects_incomplete_trace(tmp_path):
    module = load_script("generate_example")
    trace = tmp_path / "bad.jsonl"
    trace.write_text(json.dumps({"name": "agent.start", "run_id": "only", "payload": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete trace"):
        module.load_trace(trace)
