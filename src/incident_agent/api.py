from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import LLMConfigurationError, load_deepseek_settings, settings
from .graph import DiagnosisEngine
from .models import LLMUnavailableError, build_model
from .schemas import IncidentReport, IncidentRequest


def create_engine() -> DiagnosisEngine:
    return DiagnosisEngine(model=build_model())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None
    app.state.llm_error = None
    try:
        app.state.engine = create_engine()
    except (LLMConfigurationError, LLMUnavailableError) as exc:
        # Keep the HTTP process available so /health and /diagnose can expose
        # the actionable configuration error instead of hiding it at startup.
        app.state.llm_error = str(exc)
    try:
        yield
    finally:
        if app.state.engine is not None:
            await app.state.engine.aclose()


app = FastAPI(
    title="LLM Serving Incident Agent",
    version="0.1.0",
    description="Minimal LangGraph + MCP + RAG inference-service diagnosis demo.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    error = getattr(app.state, "llm_error", None)
    payload = {"status": "ok" if not error else "degraded", "llm_provider": "deepseek", "rag_provider": settings.rag_provider}
    if error:
        payload["llm_error"] = error
    else:
        try:
            payload["model"] = load_deepseek_settings().model
        except LLMConfigurationError as exc:
            payload["status"] = "degraded"
            payload["llm_error"] = str(exc)
    return payload


@app.post("/diagnose", response_model=IncidentReport)
async def diagnose(request: IncidentRequest) -> IncidentReport:
    engine: DiagnosisEngine | None = getattr(app.state, "engine", None)
    try:
        if engine is None:
            error = getattr(app.state, "llm_error", None)
            if error:
                raise LLMUnavailableError(error)
            engine = create_engine()
        return await asyncio.wait_for(engine.diagnose(request), timeout=settings.agent_timeout_seconds + 2)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="diagnosis timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (LLMConfigurationError, LLMUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"diagnosis failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run("incident_agent.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
