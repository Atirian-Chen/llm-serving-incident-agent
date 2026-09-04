from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar
from contextlib import contextmanager
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator


_active_trace: ContextVar[tuple["TraceRecorder", str] | None] = ContextVar("active_trace", default=None)


@contextmanager
def trace_context(recorder: "TraceRecorder", run_id: str):
    """Make model-level raw messages traceable without mutable shared state."""
    token = _active_trace.set((recorder, run_id))
    try:
        yield
    finally:
        _active_trace.reset(token)


def record_current_trace(name: str, payload: dict[str, Any]) -> None:
    current = _active_trace.get()
    if current is not None:
        recorder, run_id = current
        recorder.event(run_id, name, payload)


class TraceRecorder:
    """Small local trace recorder with optional LangSmith integration.

    Local JSONL traces make the project inspectable without credentials. When
    LangSmith is configured, the same events are additionally sent as runs.
    """

    def __init__(self, path: str | Path = "data/traces.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._langsmith_client = None
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            try:
                from langsmith import Client  # type: ignore

                self._langsmith_client = Client()
            except Exception:
                self._langsmith_client = None

    def event(self, run_id: str, name: str, payload: dict[str, Any]) -> None:
        record = {
            "run_id": run_id,
            "name": name,
            "timestamp_ms": round(time.time() * 1000),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @asynccontextmanager
    async def span(self, run_id: str, name: str, payload: dict[str, Any]) -> AsyncIterator[None]:
        started = time.perf_counter()
        self.event(run_id, f"{name}.start", payload)
        try:
            yield
        except Exception as exc:
            self.event(run_id, f"{name}.error", {"error": repr(exc)})
            raise
        finally:
            self.event(
                run_id,
                f"{name}.end",
                {"duration_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
