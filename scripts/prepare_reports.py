"""Publish existing local evaluation evidence without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from evaluate import inspect_rag_trace, write_json, write_review
from generate_example import render_example
from rag_eval import _metrics, _ndcg_at_3, _write_markdown


ROOT = Path(__file__).resolve().parents[1]


def export_trace(source: Path, destination: Path) -> str:
    content = source.read_bytes()
    text = content.decode("utf-8-sig")
    if re.search(r"sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}", text):
        raise ValueError("Trace contains a potential credential; publication stopped")
    for line in text.splitlines():
        if line.strip():
            json.loads(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def prepare(root: Path) -> None:
    reports = root / "reports"
    traces = reports / "traces"
    reports.mkdir(exist_ok=True)

    rag_path = reports / "rag_eval.json"
    original = rag_path.read_bytes()
    rag = json.loads(original.decode("utf-8-sig"))
    for row in rag["queries"]:
        results = row["top_results"][:10]
        allowed = set(row["relevant_sources"])
        if not allowed:
            raise ValueError("RAG query has no relevance labels")
        row["first_relevant_rank"] = next((index for index, item in enumerate(results, 1) if item["source"] in allowed), None)
        row["ndcg_at_3"] = _ndcg_at_3(results, row["relevant_sources"])
    rag["metrics"] = _metrics(rag["queries"])
    rag.setdefault("provenance", {
        "metrics_recomputed": True,
        "source_report_sha256": hashlib.sha256(original).hexdigest(),
        "source_report": "reports/rag_eval.json before first recomputation",
        "new_model_executions": 0,
        "latency": "Original measured latency retained, not remeasured.",
    })
    rag["metric_definitions"] = {
        "label_level": "source (runbook), not manually annotated chunks",
        "recall_at_k": "Mean distinct relevant sources in first k results / all labeled relevant sources per query.",
        "hit_rate_at_k": "Fraction of queries with at least one relevant source in first k results, per method.",
        "mrr": "Mean reciprocal rank of first relevant result within Top-10.",
        "ndcg_at_3": "Binary source relevance with zero gain for repeated sources in the first 3 chunks.",
    }
    write_json(rag_path, rag)
    _write_markdown(reports / "rag_eval.md", rag)

    historical = root / "data/online_evaluation.json"
    legacy = json.loads(historical.read_text(encoding="utf-8-sig"))
    legacy_trace = traces / "legacy_e2e.jsonl"
    digest = export_trace(root / "data/online_evaluation_trace.jsonl", legacy_trace)
    legacy["rag"] = inspect_rag_trace(legacy_trace)
    legacy["rag"]["stage_trace"] = "traces/legacy_e2e.jsonl"
    legacy["provenance"] = {
        "kind": "historical_online_evaluation",
        "source_report": "data/online_evaluation.json",
        "source_report_sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
        "trace_sha256": digest,
        "new_online_requests": 0,
        "note": "Original 30-case outcomes preserved; unsupported hybrid/strict claims removed.",
    }
    write_json(reports / "e2e_results.json", legacy)
    write_review(reports / "e2e_results.md", legacy["metrics"], legacy["cases"], legacy["rag"])

    online_trace = traces / "online_smoke.jsonl"
    digest = export_trace(root / "data/online_current_trace.jsonl", online_trace)
    summary = render_example(online_trace, root / "example.md")
    summary["trace"] = "traces/online_smoke.jsonl"
    summary["trace_sha256"] = digest
    write_json(reports / "online_smoke.json", summary)
    lines = [
        "# 真实 Hybrid 在线单案例核验",
        "",
        "本页核验已保存的在线日志，没有再次调用 DeepSeek。该案例与历史 30 条评测分开记录。",
        "",
        f"- Run ID：`{summary['run_id']}`",
        f"- 原始运行时间：`{summary['started_at_utc']}`",
        f"- 模型：`{', '.join(summary['models']['llm'])}`；RAG device：`{summary['models']['device']}`",
        f"- 耗时：{summary['duration_ms'] / 1000:.2f} 秒；MCP 调用：{summary['tool_call_count']} 次；最终文档：{len(summary['final_results'])} 条",
        f"- 请求：{summary['request']['symptom']}",
        f"- 服务：`{summary['request']['service_id']}`（本地 fixture）",
        "- [可读完整流程](../example.md) / [原始日志](traces/online_smoke.jsonl) / [核验 JSON](online_smoke.json)",
        f"- Trace SHA-256：`{digest}`",
        "",
        "| 检查项 | 结果 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'通过' if passed else '失败'} |" for name, passed in summary["checks"].items())
    lines += [
        "",
        "核验覆盖日志中的 Hybrid、Top-3、CUDA、DeepSeek 响应、执行工具参数和最终报告；不衡量根因判断是否正确。仅一个成功案例，尚未构成六类故障或 30 条完整 Hybrid 在线回归。",
        "",
        "日志未保留全部 Cross-Encoder 候选分数及最终 HTTP 原始响应。工具的展示消息与实际 user JSON context 的区别见 example.md。",
        "",
    ]
    (reports / "online_smoke.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rag_queries": rag["query_count"], "legacy_cases": len(legacy["cases"]), "legacy_hybrid_runs": legacy["rag"]["hybrid_runs"], "online_smoke_checks": summary["checks"], "new_online_requests": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == "__main__":
    main()
