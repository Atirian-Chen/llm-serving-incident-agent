from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "rule")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "local-model")
    rag_provider: str = os.getenv("RAG_PROVIDER", "keyword")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "5"))
    agent_timeout_seconds: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
    tool_timeout_seconds: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "5"))


settings = Settings()

