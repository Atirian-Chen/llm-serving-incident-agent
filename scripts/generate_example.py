"""Render a saved online trace without loading models or calling APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_trace(path: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    run_ids = {record["run_id"] for record in records if record.get("name") == "agent.start"}
    if run_id is None:
        if len(run_ids) != 1:
            raise ValueError("Select --run-id: the trace must identify exactly one run")
        run_id = next(iter(run_ids))
    selected = [record for record in records if record.get("run_id") == run_id]
    names = {record.get("name") for record in selected}
    required = {"agent.start", "rag.hybrid", "rag.final_results", "llm.raw_request", "llm.raw_response", "mcp.tool_result", "llm.final_response", "agent.end"}
    if not required.issubset(names):
        raise ValueError(f"Incomplete trace: missing {sorted(required - names)}")
    return selected


def payload_for(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(record["payload"] for record in records if record["name"] == name)


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    request = payload_for(records, "agent.start")["request"]
    hybrid = payload_for(records, "rag.hybrid")
    docs = payload_for(records, "rag.final_results")["results"]
    model_names = sorted({
        record["payload"]["message"]["response_metadata"]["model_name"]
        for record in records if record["name"] == "llm.raw_response"
        and record["payload"]["message"].get("response_metadata", {}).get("model_name")
    })
    tools = [record["payload"] for record in records if record["name"] == "mcp.tool_result"]
    arguments_valid = bool(tools)
    for tool in tools:
        args = tool["arguments"]
        arguments_valid &= args.get("service_id") == request["service_id"]
        arguments_valid &= tool["name"] in {"get_metrics_snapshot", "search_logs"}
        if tool["name"] == "search_logs":
            arguments_valid &= isinstance(args.get("keyword"), str) and 1 <= len(args["keyword"]) <= 120
            arguments_valid &= isinstance(args.get("limit"), int) and 1 <= args["limit"] <= 50
    checks = {
        "hybrid_trace_present": True,
        "final_top_three": len(docs) == 3 and hybrid["final_results"] == docs,
        "cuda_recorded": hybrid.get("device") == "cuda",
        "deepseek_response_metadata_present": bool(model_names) and all("deepseek" in name for name in model_names),
        "mcp_arguments_valid": bool(arguments_valid),
        "final_report_recorded": bool(payload_for(records, "llm.final_response")["response"]),
    }
    return {
        "run_id": records[0]["run_id"],
        "started_at_utc": datetime.fromtimestamp(records[0]["timestamp_ms"] / 1000, timezone.utc).isoformat(),
        "verification_mode": "saved_online_trace_audit",
        "new_online_requests": 0,
        "sample_count": 1,
        "request": request,
        "models": {"llm": model_names, "embedding": hybrid["embedding_model"], "reranker": hybrid["reranker_model"], "device": hybrid["device"]},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "duration_ms": payload_for(records, "agent.end")["duration_ms"],
        "tool_call_count": len(tools),
        "tool_calls": tools,
        "final_results": docs,
        "report": payload_for(records, "llm.final_response")["response"],
        "limitations": [
            "One saved online run, not a new 30-case evaluation.",
            "MCP reads fixture metrics/logs, not a production cluster.",
            "The trace stores reranker scores only for final Top-3, not all candidates.",
            "The final llm.raw_response is a parsed IncidentReport, not a raw HTTP response.",
            "Tool results enter the next user JSON context; the logged tool_message is not a native API ToolMessage.",
        ],
    }


def json_block(value: Any) -> list[str]:
    return ["```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""]


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def candidate_table(title: str, items: list[dict[str, Any]]) -> list[str]:
    fields = ("chunk_id", "bm25_rank", "bm25_score", "dense_rank", "dense_score", "rrf_rank", "rrf_score", "reranker_rank", "reranker_score")
    lines = [f"### {title}", "", "| chunk_id | BM rank | BM score | Dense rank | Cosine | RRF rank | RRF score | Cross rank | Cross score |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend("| " + " | ".join(_cell(item.get(field)) for field in fields) + " |" for item in items)
    return lines + [""]


def render_example(trace_path: Path, output: Path, run_id: str | None = None) -> dict[str, Any]:
    records = load_trace(trace_path, run_id)
    summary = summarize_trace(records)
    if not summary["all_checks_passed"]:
        raise ValueError(f"Trace checks failed: {summary['checks']}")
    hybrid = payload_for(records, "rag.hybrid")
    trace_link = Path(os.path.relpath(trace_path, output.parent)).as_posix()
    lines = [
        "# Hybrid RAG 真实在线流程示例",
        "",
        "本页从已保存的真实运行日志整理：CUDA BGE + Chroma + Cross-Encoder，DeepSeek 在线响应，官方 MCP SDK 调用本地 fixture 服务。没有重新请求模型，也没有使用测试替身生成本页的诊断。",
        "",
        f"- Run ID：`{summary['run_id']}`",
        f"- 原始运行时间：`{summary['started_at_utc']}`",
        f"- 原始日志：[{trace_path.name}]({trace_link})（完整内容与精度以 JSONL 为准）",
        f"- SHA-256：`{hashlib.sha256(trace_path.read_bytes()).hexdigest()}`",
        f"- 在线模型：`{', '.join(summary['models']['llm'])}`；总耗时：{summary['duration_ms'] / 1000:.2f} 秒；工具调用：{summary['tool_call_count']} 次",
        "",
        "日志边界：这里只保存了最终 Top-3 的 Cross-Encoder 分数；其余候选的精排分数未记录。最终结构化响应保存的是解析后的 IncidentReport，未保留该次 HTTP 原始响应。工具结果通过下一轮 user JSON context 送入模型，日志中的 tool_message 是展示对象，不是实际发送的原生 ToolMessage。",
        "",
        "版本说明：当前代码会在发送给 LLM 前去掉检索分数、排名及重复元数据，只保留排序后的正文、引用信息和必要元数据。下方忠实展示快照中的实际请求；历史请求可能仍含上述检索字段，不代表当前消息格式，也不会通过改写旧日志来模拟新运行。",
        "",
        "## 用户输入",
        "",
    ] + json_block(summary["request"])
    lines += [
        "## 检索过程",
        "",
        f"- Embedding：`{hybrid['embedding_model']}`；Reranker：`{hybrid['reranker_model']}`；Device：`{hybrid['device']}`",
        f"- 候选数：BM25={hybrid['bm25_count']}，Dense={hybrid['dense_count']}，去重={hybrid['dedup_count']}，RRF={hybrid['fusion_count']}，最终={hybrid['rerank_count']}",
        f"- 耗时（ms）：BM25={hybrid['bm25_latency_ms']}，Dense={hybrid['dense_latency_ms']}，RRF={hybrid['fusion_latency_ms']}，Cross-Encoder={hybrid['rerank_latency_ms']}",
        "- 表格中的 `-` 表示该阶段未召回或日志没有此项，不代表零分。",
        "",
    ]
    for title, field in (("BM25 候选", "bm25_results"), ("Dense 候选", "dense_results"), ("RRF 融合", "rrf_results"), ("Cross-Encoder 最终 Top-3", "final_results")):
        lines += candidate_table(title, hybrid[field])
    for doc in hybrid["final_results"]:
        lines += [f"### {doc['title']} / {doc['section']}", ""] + json_block(doc)
    lines += ["## LLM 与工具调用时间线", "", "按原始事件顺序展开。每轮请求的 JSON content 已格式化，便于查看实际 RAG context 和工具证据；原始序列化消息保留在 JSONL 中。", ""]
    round_index = 0
    for record in records:
        name, payload = record["name"], record["payload"]
        if name == "llm.raw_request":
            round_index += 1
            lines += [f"### 第 {round_index} 轮请求：{payload['phase']}", ""]
            for message in payload["messages"]:
                role = message.get("role", message.get("type", "unknown"))
                lines += [f"#### {role}", ""]
                content = message.get("content", "")
                try:
                    decoded = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    lines += ["```text", str(content), "```", ""]
                else:
                    lines += json_block(decoded)
        elif name == "llm.raw_response":
            label = "结构化解析结果" if payload["phase"] == "final_user_response" else "原始模型消息（含工具调用）"
            lines += [f"### 第 {round_index} 轮：{label}", ""] + json_block(payload["message"])
        elif name == "mcp.tool_result":
            lines += [f"### MCP 返回：{payload['name']}", ""] + json_block(payload)
    lines += ["## 最终用户报告", ""] + json_block(summary["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "reports/traces/online_smoke.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "example.md")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    summary = render_example(args.trace.resolve(), args.output.resolve(), args.run_id)
    print(json.dumps({"run_id": summary["run_id"], "all_checks_passed": summary["all_checks_passed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
