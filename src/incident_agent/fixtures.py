from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT


ALLOWED_SERVICES = {"demo-oom", "demo-high-ttft", "demo-prefix-cache", "demo-unknown"}
METRICS_DIR = ROOT / "fixtures" / "metrics"
LOGS_DIR = ROOT / "fixtures" / "logs"


class FixtureError(ValueError):
    pass


def validate_service_id(service_id: str) -> None:
    if service_id not in ALLOWED_SERVICES:
        raise FixtureError(f"service_id is not allowlisted: {service_id}")


def load_metrics(service_id: str) -> dict[str, Any]:
    validate_service_id(service_id)
    path = METRICS_DIR / f"{service_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def search_logs(service_id: str, keyword: str, limit: int = 20) -> list[str]:
    validate_service_id(service_id)
    if not keyword or len(keyword) > 120:
        raise FixtureError("keyword must contain 1-120 characters")
    safe_limit = min(max(int(limit), 1), 50)
    path = LOGS_DIR / f"{service_id}.log"
    keyword_lower = keyword.casefold()
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if keyword_lower in line.casefold()
    ][:safe_limit]

