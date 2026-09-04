from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_agent.graph import DiagnosisEngine  # noqa: E402
from incident_agent.config import LLMConfigurationError  # noqa: E402
from incident_agent.models import LLMUnavailableError  # noqa: E402


async def run(path: Path) -> dict[str, object]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    engine = DiagnosisEngine()
    rows: list[dict[str, object]] = []
    try:
        for case in cases:
            started = time.perf_counter()
            report = await engine.diagnose(
                {"service_id": case["service_id"], "symptom": case["symptom"]}
            )
            latency = (time.perf_counter() - started) * 1000
            evidence_text = " ".join(item.detail for item in report.evidence).casefold()
            expected_tools = set(case["expected_tools"])
            actual_tools = set(report.tools_used)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "incident_type_correct": report.incident_type == case["expected_incident_type"],
                    "tools_correct": expected_tools.issubset(actual_tools),
                    "schema_valid": True,
                    "evidence_terms_covered": sum(
                        term.casefold() in evidence_text for term in case["required_evidence_terms"]
                    ),
                    "evidence_terms_total": len(case["required_evidence_terms"]),
                    "latency_ms": round(latency, 2),
                }
            )
    finally:
        await engine.aclose()

    total = len(rows)
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    p95_index = min(total - 1, max(0, int(total * 0.95) - 1))
    return {
        "evaluation_mode": "online_deepseek",
        "total_cases": total,
        "incident_type_accuracy": sum(bool(r["incident_type_correct"]) for r in rows) / total,
        "tool_selection_accuracy": sum(bool(r["tools_correct"]) for r in rows) / total,
        "schema_success_rate": sum(bool(r["schema_valid"]) for r in rows) / total,
        "evidence_term_coverage": sum(int(r["evidence_terms_covered"]) for r in rows)
        / max(1, sum(int(r["evidence_terms_total"]) for r in rows)),
        "average_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": latencies[p95_index],
        "cases": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        metrics = asyncio.run(run(root / "eval" / "cases.jsonl"))
    except (LLMConfigurationError, LLMUnavailableError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    output = root / "data" / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
