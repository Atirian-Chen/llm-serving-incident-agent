from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import settings
from .graph import DiagnosisEngine
from .models import build_model
from .schemas import IncidentReport, IncidentRequest


def create_engine() -> DiagnosisEngine:
    model = build_model(
        settings.model_provider,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    return DiagnosisEngine(model=model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = create_engine()
    try:
        yield
    finally:
        await app.state.engine.aclose()


app = FastAPI(
    title="LLM Serving Incident Agent",
    version="0.1.0",
    description="Minimal LangGraph + MCP + RAG inference-service diagnosis demo.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_provider": settings.model_provider, "rag_provider": settings.rag_provider}


@app.post("/diagnose", response_model=IncidentReport)
async def diagnose(request: IncidentRequest) -> IncidentReport:
    engine: DiagnosisEngine = getattr(app.state, "engine", None) or create_engine()
    try:
        return await asyncio.wait_for(engine.diagnose(request), timeout=settings.agent_timeout_seconds + 2)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="diagnosis timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"diagnosis failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run("incident_agent.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
