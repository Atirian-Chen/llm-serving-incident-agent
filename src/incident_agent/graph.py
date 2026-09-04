from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict
from uuid import uuid4

from .config import settings
from .mcp_client import MCPToolClient
from .models import AgentModel, LLMUnavailableError, ModelDecision, build_model
from .rag import build_retriever
from .schemas import IncidentReport, IncidentRequest, ToolCall
from .tracing import TraceRecorder

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
except ImportError:  # pragma: no cover - surfaced when the engine is created
    StateGraph = None  # type: ignore
    START = "__start__"
    END = "__end__"

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except ImportError:  # pragma: no cover - surfaced when the engine is created
    MemorySaver = None  # type: ignore


class DiagnosisState(TypedDict, total=False):
    request: IncidentRequest
    retrieved_docs: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    pending_tool_calls: list[ToolCall]
    draft_text: str
    final_report: IncidentReport
    step_count: int
    run_id: str


class DiagnosisEngine:
    def __init__(
        self,
        retriever: Any | None = None,
        model: AgentModel | None = None,
        tool_client: MCPToolClient | None = None,
        tracer: TraceRecorder | None = None,
        max_steps: int | None = None,
        tool_timeout: float | None = None,
    ) -> None:
        if StateGraph is None or MemorySaver is None:
            raise LLMUnavailableError(
                "LLM unavailable: install project dependencies with pip install -e ."
            )
        self.retriever = retriever or build_retriever()
        self.model = model or build_model()
        self.tool_client = tool_client or MCPToolClient()
        self.tracer = tracer or TraceRecorder()
        self.max_steps = max_steps or settings.max_agent_steps
        self.tool_timeout = tool_timeout or settings.tool_timeout_seconds
        # MemorySaver makes each run addressable by ``run_id`` and demonstrates
        # LangGraph's checkpoint contract without introducing a database.
        self._checkpointer = MemorySaver()
        self._compiled_graph = self._build_graph()

    async def aclose(self) -> None:
        """Release MCP resources held by the client."""
        await self.tool_client.aclose()

    def _build_graph(self):
        graph = StateGraph(DiagnosisState)
        graph.add_node("retrieve_runbook", self.retrieve_runbook)
        graph.add_node("agent_reason", self.agent_reason)
        graph.add_node("call_tools", self.call_tools)
        graph.add_node("finalize_report", self.finalize_report)
        graph.add_edge(START, "retrieve_runbook")
        graph.add_edge("retrieve_runbook", "agent_reason")
        graph.add_conditional_edges(
            "agent_reason",
            self.route_after_agent,
            {"tools": "call_tools", "finalize": "finalize_report"},
        )
        graph.add_edge("call_tools", "agent_reason")
        graph.add_edge("finalize_report", END)
        return graph.compile(checkpointer=self._checkpointer)

    async def retrieve_runbook(self, state: DiagnosisState) -> dict[str, Any]:
        request = state["request"]
        run_id = state.get("run_id", "unknown")
        async with self.tracer.span(run_id, "rag.retrieve", {"query": request.symptom}):
            docs = await self.retriever.ainvoke(request.symptom, k=4)
        return {"retrieved_docs": docs}

    async def agent_reason(self, state: DiagnosisState) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        decision: ModelDecision
        async with self.tracer.span(run_id, "model.decide", {"step": state.get("step_count", 0)}):
            decision = await self.model.decide(
                state["request"], state.get("retrieved_docs", []), state.get("tool_results", [])
            )
        step_count = state.get("step_count", 0) + 1
        calls = decision.tool_calls if step_count <= self.max_steps else []
        return {"pending_tool_calls": calls, "draft_text": decision.text, "step_count": step_count}

    def route_after_agent(self, state: DiagnosisState) -> str:
        return "tools" if state.get("pending_tool_calls") else "finalize"

    async def call_tools(self, state: DiagnosisState) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        previous = list(state.get("tool_results", []))
        for call in state.get("pending_tool_calls", []):
            async with self.tracer.span(run_id, "mcp.tool_call", {"name": call.name, "arguments": call.arguments}):
                try:
                    result = await self.tool_client.call_tool(
                        call.name, call.arguments, timeout=self.tool_timeout
                    )
                    previous.append({"name": call.name, "arguments": call.arguments, "result": result})
                except Exception as exc:
                    # A failed evidence query must stop the run. Passing an
                    # empty result back to the model would allow it to invent
                    # a diagnosis from incomplete evidence.
                    raise LLMUnavailableError(
                        f"LLM unavailable: tool call failed for {call.name}: {exc}"
                    ) from exc
        return {"tool_results": previous, "pending_tool_calls": []}

    async def finalize_report(self, state: DiagnosisState) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        async with self.tracer.span(run_id, "structured_output", {}):
            report = await self.model.finalize(
                state["request"],
                state.get("retrieved_docs", []),
                state.get("tool_results", []),
                state.get("draft_text", ""),
            )
        return {"final_report": IncidentReport.model_validate(report)}

    async def diagnose(self, request: IncidentRequest | dict[str, Any]) -> IncidentReport:
        parsed = request if isinstance(request, IncidentRequest) else IncidentRequest.model_validate(request)
        run_id = str(uuid4())
        initial: DiagnosisState = {
            "request": parsed,
            "retrieved_docs": [],
            "tool_results": [],
            "pending_tool_calls": [],
            "draft_text": "",
            "step_count": 0,
            "run_id": run_id,
        }
        started = time.perf_counter()
        self.tracer.event(run_id, "agent.start", {"service_id": parsed.service_id})
        config = {"configurable": {"thread_id": run_id}}
        result = await asyncio.wait_for(
            self._compiled_graph.ainvoke(initial, config=config),
            timeout=settings.agent_timeout_seconds,
        )
        self.tracer.event(
            run_id,
            "agent.end",
            {"duration_ms": round((time.perf_counter() - started) * 1000, 2), "steps": result.get("step_count", 0)},
        )
        return IncidentReport.model_validate(result["final_report"])
