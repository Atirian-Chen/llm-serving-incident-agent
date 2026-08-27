from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


IncidentType = Literal["cuda_oom", "high_ttft", "low_prefix_cache_hit", "unknown"]
Severity = Literal["low", "medium", "high"]


class IncidentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    symptom: str = Field(min_length=5, max_length=2000)


class Evidence(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=1000)


class IncidentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_type: IncidentType
    severity: Severity
    root_cause: str = Field(min_length=1, max_length=1000)
    evidence: list[Evidence] = Field(min_length=1, max_length=12)
    recommended_actions: list[str] = Field(min_length=1, max_length=8)
    tools_used: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)


class ToolCall(BaseModel):
    """Provider-neutral representation of a model Function Calling request."""

    name: Literal["get_metrics_snapshot", "search_logs"]
    arguments: dict[str, object] = Field(default_factory=dict)

