from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schemas import IncidentReport, IncidentRequest, ToolCall


@dataclass
class ModelDecision:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""


class AgentModel(Protocol):
    async def decide(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> ModelDecision: ...

    async def finalize(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        draft_text: str = "",
    ) -> IncidentReport: ...


def infer_incident_type(request: IncidentRequest, metrics: dict[str, Any] | None, logs: list[str]) -> str:
    text = request.symptom.casefold()
    log_text = " ".join(logs).casefold()
    # Negative wording such as "没有 OOM、延迟或缓存异常" must not trigger a
    # text keyword route; metrics/logs remain the source of truth in that case.
    negated = any(marker in text for marker in ("没有", "无明显", "未发现", "not", "no "))
    if not negated and ("out of memory" in text or "oom" in text or "out of memory" in log_text):
        return "cuda_oom"
    if not negated and ("prefix" in text or "cache hit" in text or "prefix_cache_hit_rate" in log_text):
        return "low_prefix_cache_hit"
    if not negated and ("ttft" in text or "latency" in text or "queue" in text or "scheduler" in log_text):
        return "high_ttft"
    if metrics:
        if float(metrics.get("gpu_memory_utilization", 0)) >= 0.95:
            return "cuda_oom"
        if float(metrics.get("prefix_cache_hit_rate", 1)) < 0.1:
            return "low_prefix_cache_hit"
        if float(metrics.get("ttft_p95_ms", 0)) >= 1000 or int(metrics.get("waiting_requests", 0)) >= 40:
            return "high_ttft"
    return "unknown"


class RuleBasedModel:
    """Deterministic local provider for development and regression evaluation."""

    async def decide(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> ModelDecision:
        names = {result.get("name") for result in tool_results}
        if "get_metrics_snapshot" not in names:
            return ModelDecision(
                tool_calls=[ToolCall(name="get_metrics_snapshot", arguments={"service_id": request.service_id})],
                text="先读取服务指标，再结合排障手册判断故障类别。",
            )
        incident_type = infer_incident_type(
            request,
            next((r.get("result") for r in tool_results if r.get("name") == "get_metrics_snapshot"), None),
            [line for r in tool_results if r.get("name") == "search_logs" for line in r.get("result", [])],
        )
        if "search_logs" not in names:
            keywords = {
                "cuda_oom": "out of memory",
                "high_ttft": "scheduler",
                "low_prefix_cache_hit": "prefix_cache",
                "unknown": "ERROR",
            }
            return ModelDecision(
                tool_calls=[
                    ToolCall(
                        name="search_logs",
                        arguments={
                            "service_id": request.service_id,
                            "keyword": keywords[incident_type],
                            "limit": 20,
                        },
                    )
                ],
                text=f"当前信号更接近 {incident_type}，补充日志证据。",
            )
        return ModelDecision(text="已有指标、日志和手册证据，可以生成结构化诊断报告。")

    async def finalize(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        draft_text: str = "",
    ) -> IncidentReport:
        metrics = next(
            (r.get("result") for r in tool_results if r.get("name") == "get_metrics_snapshot"), None
        )
        logs = [line for r in tool_results if r.get("name") == "search_logs" for line in r.get("result", [])]
        incident_type = infer_incident_type(request, metrics, logs)
        recipes = {
            "cuda_oom": (
                "GPU 显存和 KV Cache 分配不足导致请求失败",
                "high",
                ["保留当前指标和日志证据", "降低 max_num_seqs 或 max_model_len 后回归吞吐", "复核权重、KV Cache 与 CUDA Graph 显存预算"],
            ),
            "high_ttft": (
                "Prefill 或调度排队导致首 token 延迟升高",
                "high",
                ["拆分观测 queue、prefill 和 decode 时间", "在固定流量下调整 batching/并发参数", "同时回归 TTFT P95、吞吐和错误率"],
            ),
            "low_prefix_cache_hit": (
                "共享前缀不稳定或路由/tokenizer 配置导致缓存键不一致",
                "medium",
                ["检查 system prompt 是否包含动态字段", "核对 tokenizer 版本和路由分布", "记录 hit/miss 与前缀长度后再调整"],
            ),
            "unknown": (
                "现有指标和日志不足以匹配已知故障模式",
                "low",
                ["补充服务版本、请求样本、GPU 指标和时间窗口", "不要自动修改线上参数", "将证据交给值班工程师继续排查"],
            ),
        }
        root_cause, severity, actions = recipes[incident_type]
        evidence: list[dict[str, str]] = []
        if metrics:
            for key in (
                "gpu_memory_utilization",
                "waiting_requests",
                "ttft_p95_ms",
                "prefix_cache_hit_rate",
                "kv_cache_free_blocks",
                "request_success_rate",
            ):
                if key in metrics:
                    evidence.append({"source": "metrics", "detail": f"{key}={metrics[key]}"})
        evidence.extend({"source": "logs", "detail": line} for line in logs[:4])
        for doc in retrieved_docs[:2]:
            evidence.append({"source": doc["source"], "detail": f"section={doc['section']}"})
        if not evidence:
            evidence.append({"source": "request", "detail": request.symptom})
        confidence = {"cuda_oom": 0.95, "high_ttft": 0.92, "low_prefix_cache_hit": 0.93, "unknown": 0.45}[incident_type]
        return IncidentReport(
            incident_type=incident_type,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            root_cause=root_cause,
            evidence=evidence[:12],
            recommended_actions=actions,
            tools_used=[str(r.get("name")) for r in tool_results],
            confidence=confidence,
        )


class OpenAICompatibleModel:
    """Optional provider for OpenAI-compatible vLLM/SGLang endpoints."""

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "get_metrics_snapshot",
                "description": "Read the read-only metrics snapshot for an allowlisted service.",
                "parameters": {
                    "type": "object",
                    "properties": {"service_id": {"type": "string"}},
                    "required": ["service_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_logs",
                "description": "Search read-only logs for an allowlisted service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_id": {"type": "string"},
                        "keyword": {"type": "string", "maxLength": 120},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["service_id", "keyword"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install project dependencies for MODEL_PROVIDER=openai") from exc
        self._llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=0)
        self._bound = self._llm.bind_tools(self.TOOL_DEFINITIONS)

    async def decide(self, request: IncidentRequest, retrieved_docs: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> ModelDecision:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        context = {
            "request": request.model_dump(),
            "runbook": retrieved_docs,
            "tool_results": tool_results,
            "instruction": "选择必要的只读工具；证据足够后停止调用工具。",
        }
        response = await self._bound.ainvoke([
            SystemMessage(content="你是 LLM serving SRE 诊断助手，只能调用提供的只读工具。"),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
        ])
        calls = []
        for call in getattr(response, "tool_calls", []) or []:
            name = call.get("name")
            if name in {"get_metrics_snapshot", "search_logs"}:
                calls.append(ToolCall(name=name, arguments=call.get("args", {})))
        return ModelDecision(tool_calls=calls, text=str(getattr(response, "content", "")))

    async def finalize(self, request: IncidentRequest, retrieved_docs: list[dict[str, Any]], tool_results: list[dict[str, Any]], draft_text: str = "") -> IncidentReport:
        structured = self._llm.with_structured_output(IncidentReport)
        prompt = {
            "request": request.model_dump(),
            "runbook": retrieved_docs,
            "tool_results": tool_results,
            "draft": draft_text,
            "instruction": "只输出符合 IncidentReport schema 的结果；evidence 必须引用实际工具或文档来源。",
        }
        result = await structured.ainvoke(json.dumps(prompt, ensure_ascii=False))
        return result if isinstance(result, IncidentReport) else IncidentReport.model_validate(result)


def build_model(provider: str, base_url: str | None = None, api_key: str | None = None, model: str = "local-model") -> AgentModel:
    if provider.lower() in {"openai", "vllm", "sglang"}:
        if not base_url or not api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY are required for openai provider")
        return OpenAICompatibleModel(base_url, api_key, model)
    return RuleBasedModel()
