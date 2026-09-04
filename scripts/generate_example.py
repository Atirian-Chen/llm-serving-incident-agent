from __future__ import annotations

"""Run one inspectable Hybrid RAG flow and render its trace as example.md.

The retrieval stack is real (BGE embeddings, Chroma cosine search, and the
configured Cross-Encoder). The model and MCP client are deterministic test
doubles so this example is reproducible and never spends an online API key.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from incident_agent.config import ROOT, settings
from incident_agent.fixtures import load_metrics, search_logs
from incident_agent.graph import DiagnosisEngine
from incident_agent.models import ModelDecision
from incident_agent.rag import build_retriever
from incident_agent.schemas import IncidentReport, IncidentRequest, ToolCall
from incident_agent.tracing import TraceRecorder, record_current_trace


class ExampleTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 5.0) -> Any:
        self.calls.append({"name": name, "arguments": dict(arguments), "timeout": timeout})
        if name == "get_metrics_snapshot":
            return load_metrics(arguments["service_id"])
        return search_logs(arguments["service_id"], arguments["keyword"], arguments.get("limit", 20))

    async def aclose(self) -> None:
        return None


def _context(request: IncidentRequest, docs: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "runbook": docs,
            "tool_results": tools,
            "instruction": "先判断证据缺口，证据不足时选择一个只读 MCP 工具。",
        },
        ensure_ascii=False,
        indent=2,
    )


class ExampleModel:
    """Test-only online-model substitute that records OpenAI-shaped messages."""

    async def decide(self, request, retrieved_docs, tool_results):
        if not tool_results:
            call = ToolCall(name="get_metrics_snapshot", arguments={"service_id": request.service_id})
            response = {
                "role": "assistant",
                "content": "需要读取服务指标。",
                "tool_calls": [{"name": call.name, "arguments": call.arguments}],
            }
        elif not any(item["name"] == "search_logs" for item in tool_results):
            call = ToolCall(
                name="search_logs",
                arguments={"service_id": request.service_id, "keyword": "queue", "limit": 20},
            )
            response = {
                "role": "assistant",
                "content": "指标显示排队，需要核对日志。",
                "tool_calls": [{"name": call.name, "arguments": call.arguments}],
            }
        else:
            call = None
            response = {"role": "assistant", "content": "指标和日志证据已足够。", "tool_calls": []}
        messages = [
            {"role": "system", "content": "你是 LLM serving SRE 诊断助手。只能调用只读工具。"},
            {"role": "user", "content": _context(request, retrieved_docs, tool_results)},
        ]
        record_current_trace("llm.raw_request", {"phase": "tool_selection", "messages": messages})
        record_current_trace("llm.raw_response", {"phase": "tool_selection", "message": response})
        return ModelDecision(
            tool_calls=[call] if call else [],
            text=response["content"],
            request_messages=messages,
            response_message=response,
        )

    async def finalize(self, request, retrieved_docs, tool_results, draft_text=""):
        messages = [
            {"role": "system", "content": "只基于 runbook 和工具证据输出 IncidentReport JSON。"},
            {
                "role": "user",
                "content": _context(request, retrieved_docs, tool_results)
                + "\n模型草稿："
                + draft_text,
            },
        ]
        response = {
            "incident_type": "high_ttft",
            "severity": "medium",
            "root_cause": "prefill backlog increased waiting time",
            "evidence": [
                {"source": "demo-high-ttft metrics", "detail": "waiting_requests=64, ttft_p95_ms=1850"},
                {"source": "demo-high-ttft logs", "detail": "prefill backlog queue_time_ms=1700"},
            ],
            "recommended_actions": ["reduce admission pressure", "inspect prefill batching"],
            "tools_used": [item["name"] for item in tool_results],
            "confidence": 0.92,
        }
        record_current_trace("llm.raw_request", {"phase": "final_user_response", "messages": messages})
        record_current_trace("llm.raw_response", {"phase": "final_user_response", "message": response})
        return IncidentReport.model_validate(response)


def _short(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _candidate_table(title: str, items: list[dict[str, Any]], score_key: str, rank_key: str) -> list[str]:
    lines = [f"### {title}", "", "| rank | chunk_id | source | title | score |", "| ---: | --- | --- | --- | ---: |"]
    for item in items:
        lines.append(
            f"| {item.get(rank_key, '')} | {_short(item.get('chunk_id', ''))} | {_short(item.get('source', ''))} | "
            f"{_short(item.get('title', ''))} | {item.get(score_key, '')} |"
        )
    return lines + [""]


def _full_results(items: list[dict[str, Any]]) -> list[str]:
    lines = ["### 最终 Top-3", ""]
    for index, item in enumerate(items, 1):
        lines.extend(
            [
                f"#### {index}. {item.get('title')} ({item.get('chunk_id')})",
                f"- source: `{item.get('source')}`",
                f"- ranks: BM25={item.get('bm25_rank')}, Dense={item.get('dense_rank')}, RRF={item.get('rrf_rank')}, Reranker={item.get('reranker_rank')}",
                f"- scores: BM25={item.get('bm25_score')}, Dense={item.get('dense_score')}, RRF={item.get('rrf_score')}, Reranker={item.get('reranker_score')}",
                "- metadata:",
                "```json",
                json.dumps(item.get("metadata", {}), ensure_ascii=False, indent=2),
                "```",
                "- content:",
                "```text",
                str(item.get("text", "")),
                "```",
                "",
            ]
        )
    return lines


async def main() -> None:
    trace_path = ROOT / "data" / "example_trace.jsonl"
    trace_path.unlink(missing_ok=True)
    tracer = TraceRecorder(trace_path)
    request = IncidentRequest(
        service_id="demo-high-ttft",
        symptom="vLLM 请求排队增长，TTFT 和 P99 延迟升高，怀疑 prefill backlog",
    )
    retriever = build_retriever()
    # ``build_retriever`` is also used outside the graph, so bind the graph's
    # recorder explicitly for this inspectable run.
    retriever.tracer = tracer
    tools = ExampleTools()
    engine = DiagnosisEngine(retriever=retriever, model=ExampleModel(), tool_client=tools, tracer=tracer)
    report = await engine.diagnose(request)

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(record["name"], []).append(record)
    hybrid = by_name["rag.hybrid"][-1]["payload"]
    llm_requests = by_name.get("llm.request", [])
    raw_requests = by_name.get("llm.raw_request", [])
    raw_responses = by_name.get("llm.raw_response", [])
    tool_results = by_name.get("mcp.tool_result", [])

    lines = [
        "# Representative Hybrid RAG Trace",
        "",
        "> This is a real local Hybrid RAG run: BGE embedding + Chroma cosine retrieval + "
        "`BAAI/bge-reranker-v2-m3` on CUDA. The DeepSeek model and MCP client are deterministic "
        "test doubles, explicitly shown below, so no online API result is being represented as real.",
        "",
        "## 1. Original User Input",
        "",
        "```json",
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 2. Hybrid Retrieval",
        "",
        f"- device: `{hybrid['device']}`",
        f"- embedding model: `{hybrid['embedding_model']}`",
        f"- reranker model: `{hybrid['reranker_model']}`",
        f"- counts: BM25={hybrid['bm25_count']}, Dense={hybrid['dense_count']}, dedup={hybrid['dedup_count']}, RRF={hybrid['fusion_count']}, final={hybrid['rerank_count']}",
        f"- latency ms: BM25={hybrid['bm25_latency_ms']}, Dense={hybrid['dense_latency_ms']}, fusion={hybrid['fusion_latency_ms']}, reranker={hybrid['rerank_latency_ms']}",
        "",
    ]
    lines += _candidate_table("BM25 Top-20", hybrid["bm25_results"], "bm25_score", "bm25_rank")
    lines += _candidate_table("Dense Cosine Top-20", hybrid["dense_results"], "dense_score", "dense_rank")
    lines += _candidate_table("去重后的 RRF 候选", hybrid["rrf_results"], "rrf_score", "rrf_rank")
    lines += _full_results(hybrid["final_results"])
    lines += [
        "## 3. RAG Context Sent to the Model",
        "",
        "下面是发送给模型的完整 user message（包含请求、最终 Top-3 runbook 和当前工具结果）。",
        "",
    ]
    for index, event in enumerate(llm_requests, 1):
        lines.extend([f"### Graph LLM request {index}", "", "```json", json.dumps(event["payload"], ensure_ascii=False, indent=2), "```", ""])
    lines += ["## 4. Raw Model Messages and Tool Loop", ""]
    for index, event in enumerate(raw_requests, 1):
        lines.extend([f"### Raw request {index}", "", "```json", json.dumps(event["payload"], ensure_ascii=False, indent=2), "```", ""])
        if index <= len(raw_responses):
            lines.extend([f"### Raw response {index}", "", "```json", json.dumps(raw_responses[index - 1]["payload"], ensure_ascii=False, indent=2), "```", ""])
    for index, event in enumerate(tool_results, 1):
        lines.extend([f"### MCP tool message {index}", "", "```json", json.dumps(event["payload"], ensure_ascii=False, indent=2), "```", ""])
    lines += [
        "## 5. Final User Report",
        "",
        "```json",
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```",
        "",
        f"Trace source: `{trace_path.relative_to(ROOT)}`",
        f"Configuration: final_top_k={settings.rag_final_top_k}, bm25_top_k={settings.rag_bm25_top_k}, dense_top_k={settings.rag_dense_top_k}, strict={settings.rag_strict}",
    ]
    (ROOT / "example.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"trace": str(trace_path), "example": str(ROOT / "example.md"), "final": report.model_dump(mode="json")}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
