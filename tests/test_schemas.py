import pytest
from pydantic import ValidationError

from incident_agent.schemas import IncidentRequest, IncidentReport


def test_request_rejects_unknown_fields_and_bad_service_id():
    with pytest.raises(ValidationError):
        IncidentRequest(service_id="../secret", symptom="valid symptom")
    with pytest.raises(ValidationError):
        IncidentRequest(service_id="demo-oom", symptom="valid symptom", extra="nope")


def test_report_is_structured_and_bounded():
    report = IncidentReport(
        incident_type="unknown",
        severity="low",
        root_cause="insufficient evidence",
        evidence=[{"source": "metrics", "detail": "gpu_memory_utilization=0.5"}],
        recommended_actions=["collect more evidence"],
        confidence=0.4,
    )
    assert report.confidence == 0.4
    with pytest.raises(ValidationError):
        IncidentReport(
            incident_type="unknown",
            severity="low",
            root_cause="bad",
            evidence=[{"source": "metrics", "detail": "x"}],
            recommended_actions=["collect"],
            confidence=2,
        )
