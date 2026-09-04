from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import DeepSeekSettings, LLMConfigurationError, load_deepseek_settings
from .schemas import IncidentReport, IncidentRequest, ToolCall


class LLMUnavailableError(RuntimeError):
    """Raised when the configured online LLM cannot complete a request."""


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


class DeepSeekModel:
    """Online DeepSeek agent model using OpenAI-compatible Function Calling."""

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "get_metrics_snapshot",
                "description": "Read the read-only metrics snapshot for the current allowlisted service.",
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
                "description": "Search read-only logs for the current allowlisted service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_id": {"type": "string"},
                        "keyword": {"type": "string", "minLength": 1, "maxLength": 120},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["service_id", "keyword"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    SYSTEM_PROMPT = """你是 LLM serving SRE 诊断助手。
你只能使用已经提供的只读工具，不得修改线上配置。
先结合排障手册判断还缺少什么证据；证据不足时调用 get_metrics_snapshot 或 search_logs。
工具参数中的 service_id 必须使用当前请求的 service_id。证据足够后停止调用工具。
最终报告必须只基于请求、排障手册和工具返回的实际证据；证据不足时选择 unknown，不能把推测写成事实。
"""

    def __init__(self, config: DeepSeekSettings | None = None) -> None:
        self.config = config or load_deepseek_settings()
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency installation issue
            raise LLMConfigurationError(
                "LLM unavailable: install project dependencies with pip install -e ."
            ) from exc
        try:
            self._llm = ChatOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                model=self.config.model,
                temperature=0,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
            self._tool_llm = self._llm.bind_tools(self.TOOL_DEFINITIONS)
            self._structured_llm = self._llm.with_structured_output(
                IncidentReport,
                method="json_mode",
            )
        except Exception as exc:
            raise LLMUnavailableError(f"LLM unavailable: failed to initialize DeepSeek: {exc}") from exc

    @staticmethod
    def _json_context(
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        instruction: str,
    ) -> str:
        return json.dumps(
            {
                "request": request.model_dump(),
                "runbook": retrieved_docs,
                "tool_results": tool_results,
                "instruction": instruction,
            },
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _normalize_tool_call(request: IncidentRequest, raw: dict[str, Any]) -> ToolCall:
        name = raw.get("name")
        if name not in {"get_metrics_snapshot", "search_logs"}:
            raise LLMUnavailableError(f"LLM unavailable: model requested unsupported tool {name!r}")
        arguments = raw.get("args") or raw.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise LLMUnavailableError("LLM unavailable: model returned invalid tool arguments")
        normalized = dict(arguments)
        normalized["service_id"] = request.service_id
        if name == "search_logs":
            keyword = str(normalized.get("keyword", "")).strip()
            if not keyword or len(keyword) > 120:
                raise LLMUnavailableError("LLM unavailable: model returned invalid log search keyword")
            try:
                limit = int(normalized.get("limit", 20))
            except (TypeError, ValueError) as exc:
                raise LLMUnavailableError("LLM unavailable: model returned invalid log search limit") from exc
            if not 1 <= limit <= 50:
                raise LLMUnavailableError("LLM unavailable: model returned invalid log search limit")
            normalized.update(keyword=keyword, limit=limit)
        return ToolCall(name=name, arguments=normalized)

    async def decide(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> ModelDecision:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            response = await self._tool_llm.ainvoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(
                        content=self._json_context(
                            request,
                            retrieved_docs,
                            tool_results,
                            "选择必要的只读工具。已有足够证据时不要再调用工具。",
                        )
                    ),
                ]
            )
            calls = [
                self._normalize_tool_call(request, call)
                for call in (getattr(response, "tool_calls", None) or [])
            ]
            return ModelDecision(tool_calls=calls, text=str(getattr(response, "content", "")))
        except LLMUnavailableError:
            raise
        except Exception as exc:
            raise LLMUnavailableError(f"LLM unavailable: DeepSeek request failed: {exc}") from exc

    async def finalize(
        self,
        request: IncidentRequest,
        retrieved_docs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        draft_text: str = "",
    ) -> IncidentReport:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            prompt = self._json_context(
                request,
                retrieved_docs,
                tool_results,
                "输出 IncidentReport JSON。evidence 必须引用实际工具或文档来源，不能编造指标、日志或操作结果。"
                + f"\n模型草稿：{draft_text}",
            )
            result = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            return result if isinstance(result, IncidentReport) else IncidentReport.model_validate(result)
        except Exception as exc:
            raise LLMUnavailableError(f"LLM unavailable: DeepSeek structured output failed: {exc}") from exc


def build_model() -> AgentModel:
    """Build the mandatory online DeepSeek model; no local fallback exists."""
    try:
        return DeepSeekModel()
    except LLMConfigurationError:
        raise
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(f"LLM unavailable: failed to build model: {exc}") from exc
