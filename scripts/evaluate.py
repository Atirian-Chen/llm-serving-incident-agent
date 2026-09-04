from __future__ import annotations

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

from incident_agent.config import LLMConfigurationError  # noqa: E402
from incident_agent.graph import DiagnosisEngine  # noqa: E402
from incident_agent.mcp_client import MCPToolClient  # noqa: E402
from incident_agent.models import LLMUnavailableError, build_model  # noqa: E402
from incident_agent.tracing import TraceRecorder  # noqa: E402


class RecordingMCPToolClient(MCPToolClient):
    """Run real MCP calls while retaining their parameters and returned data."""

    def __init__(self, trace_path: Path) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.trace_path = trace_path

    def _write_trace(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    async def call_tool(
        self, name: str, arguments: dict[str, Any], timeout: float = 5.0
    ) -> Any:
        started = time.perf_counter()
        record: dict[str, Any] = {"event": "tool_execution", "name": name, "arguments": arguments}
        try:
            output = await super().call_tool(name, arguments, timeout=timeout)
            record["output"] = output
            return output
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.calls.append(record)
            self._write_trace(record)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def evaluate_row(
    index: int,
    case: dict[str, Any],
    report: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    expected = {
        "incident_type": case["expected_incident_type"],
        "tools": case["expected_tools"],
        "evidence_terms": case["required_evidence_terms"],
    }
    request = {"service_id": case["service_id"], "symptom": case["symptom"]}
    if report is None:
        return {
            "index": index,
            "case_id": case["case_id"],
            "input": request,
            "expected": expected,
            "success": False,
            "error": error,
            "incident_type_correct": False,
            "tools_correct": False,
            "schema_valid": False,
            "covered_evidence_terms": [],
            "evidence_terms_covered": 0,
            "evidence_terms_total": len(expected["evidence_terms"]),
            "latency_ms": round(latency_ms, 2),
            "tool_calls": tool_calls,
            "output": None,
        }

    evidence_text = " ".join(
        str(item.get("detail", "")) for item in report.get("evidence", [])
    ).casefold()
    covered = [
        term
        for term in expected["evidence_terms"]
        if term.casefold() in evidence_text
    ]
    actual_tools = {str(name) for name in report.get("tools_used", [])}
    return {
        "index": index,
        "case_id": case["case_id"],
        "input": request,
        "expected": expected,
        "success": True,
        "error": None,
        "incident_type_correct": report["incident_type"] == expected["incident_type"],
        "tools_correct": set(expected["tools"]).issubset(actual_tools),
        "schema_valid": True,
        "covered_evidence_terms": covered,
        "evidence_terms_covered": len(covered),
        "evidence_terms_total": len(expected["evidence_terms"]),
        "latency_ms": round(latency_ms, 2),
        "tool_calls": tool_calls,
        "output": report,
    }


def summarize(rows: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    completed = len(rows)
    successful = sum(bool(row["success"]) for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    successful_latencies = [float(row["latency_ms"]) for row in rows if row["success"]]
    evidence_total = sum(int(row["evidence_terms_total"]) for row in rows)
    return {
        "evaluation_mode": "online_deepseek",
        "total_cases": total_cases,
        "completed_cases": completed,
        "successful_cases": successful,
        "failed_cases": completed - successful,
        "incident_type_accuracy": sum(bool(row["incident_type_correct"]) for row in rows) / total_cases,
        "tool_selection_accuracy": sum(bool(row["tools_correct"]) for row in rows) / total_cases,
        "schema_success_rate": sum(bool(row["schema_valid"]) for row in rows) / total_cases,
        "evidence_term_coverage": (
            sum(int(row["evidence_terms_covered"]) for row in rows) / evidence_total
            if evidence_total
            else 0.0
        ),
        "average_latency_ms_all_completed": statistics.mean(latencies) if latencies else None,
        "p95_latency_ms_all_completed": percentile_95(latencies),
        "average_latency_ms_successful": statistics.mean(successful_latencies)
        if successful_latencies
        else None,
        "p95_latency_ms_successful": percentile_95(successful_latencies),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def format_metric(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_review(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    lines = [
        "# DeepSeek 在线 Agent 评测复习记录",
        "",
        "本文件来自真实在线调用。失败案例按失败计入全部 30 个案例的准确率分母；未执行的案例不会被伪装成成功。",
        "",
        "## 汇总",
        "",
        f"- 状态：已完成 {metrics['completed_cases']}/{metrics['total_cases']} 个案例",
        f"- 成功/失败：{metrics['successful_cases']}/{metrics['failed_cases']}",
        f"- 故障类型准确率：{metrics['incident_type_accuracy']:.2%}",
        f"- 工具选择准确率：{metrics['tool_selection_accuracy']:.2%}",
        f"- Schema 成功率：{metrics['schema_success_rate']:.2%}",
        f"- 证据词覆盖率：{metrics['evidence_term_coverage']:.2%}",
        f"- 全部已完成案例平均延迟：{format_metric(metrics['average_latency_ms_all_completed'])} ms",
        f"- 全部已完成案例 P95 延迟：{format_metric(metrics['p95_latency_ms_all_completed'])} ms",
        f"- 成功案例平均延迟：{format_metric(metrics['average_latency_ms_successful'])} ms",
        f"- 成功案例 P95 延迟：{format_metric(metrics['p95_latency_ms_successful'])} ms",
        "",
    ]
    examples = [row for row in ordered if row["case_id"] in {"oom-01", "ttft-01"}]
    lines.append("## 两个完整在线案例")
    for row in examples:
        lines.extend(
            [
                "",
                f"### {row['case_id']}",
                "",
                "输入：",
                "```json",
                json.dumps(row["input"], ensure_ascii=False, indent=2),
                "```",
                "",
                "实际工具调用及返回：",
                "```json",
                json.dumps(row["tool_calls"], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
                "最终模型输出：",
                "```json",
                json.dumps(row["output"], ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## 30 条案例明细",
            "",
            "| # | case_id | 状态 | 预测 | 正确 | 工具正确 | Schema | 延迟 ms | 错误 |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in ordered:
        output = row["output"] or {}
        error = str(row["error"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['index']} | {row['case_id']} | {'成功' if row['success'] else '失败'} | "
            f"{output.get('incident_type', 'N/A')} | {'是' if row['incident_type_correct'] else '否'} | "
            f"{'是' if row['tools_correct'] else '否'} | {'是' if row['schema_valid'] else '否'} | "
            f"{float(row['latency_ms']):.2f} | {error} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(root: Path, concurrency: int) -> dict[str, Any]:
    cases = load_cases(root / "eval" / "cases.jsonl")
    output_path = root / "data" / "online_evaluation.json"
    trace_path = root / "data" / "online_evaluation_trace.jsonl"
    review_path = root / "data" / "online_evaluation_review.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()

    async def checkpoint() -> None:
        async with rows_lock:
            payload = {
                "metrics": summarize(rows, len(cases)),
                "cases": sorted(rows, key=lambda row: row["index"]),
                "rag": {
                    "provider": "hybrid",
                    "strict": True,
                    "final_top_k": 3,
                    "stage_trace": "data/online_evaluation_trace.jsonl",
                },
            }
            write_json(output_path, payload)

    async def execute(index: int, case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            tool_client = RecordingMCPToolClient(trace_path)
            tracer = TraceRecorder(trace_path)
            started = time.perf_counter()
            report: dict[str, Any] | None = None
            error: str | None = None
            try:
                engine = DiagnosisEngine(model=build_model(), tool_client=tool_client, tracer=tracer)
                result = await engine.diagnose(
                    {"service_id": case["service_id"], "symptom": case["symptom"]}
                )
                report = result.model_dump(mode="json")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if "engine" in locals():
                    await engine.aclose()
            latency_ms = (time.perf_counter() - started) * 1000
            return evaluate_row(index, case, report, tool_client.calls, latency_ms, error)

    tasks = [asyncio.create_task(execute(index, case)) for index, case in enumerate(cases, start=1)]
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        row = await task
        rows.append(row)
        await checkpoint()
        state = "OK" if row["success"] else "FAIL"
        print(f"[{completed:02d}/{len(cases)}] {row['case_id']} {state} {row['latency_ms']:.2f} ms", flush=True)

    rows = sorted(rows, key=lambda row: row["index"])
    metrics = summarize(rows, len(cases))
    payload = {
        "metrics": metrics,
        "cases": rows,
        "rag": {
            "provider": "hybrid",
            "strict": True,
            "final_top_k": 3,
            "stage_trace": "data/online_evaluation_trace.jsonl",
        },
    }
    write_json(output_path, payload)
    write_json(root / "data" / "metrics.json", payload)
    write_review(review_path, metrics, rows)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "e2e_results.json", payload)
    write_review(reports / "e2e_results.md", metrics, rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real DeepSeek evaluation set.")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    root = Path(__file__).resolve().parents[1]
    try:
        payload = asyncio.run(run(root, args.concurrency))
    except (LLMConfigurationError, LLMUnavailableError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
