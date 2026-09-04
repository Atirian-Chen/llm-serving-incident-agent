from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional during dependency-free inspection
    pass


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    rag_provider: str = os.getenv("RAG_PROVIDER", "keyword")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "5"))
    agent_timeout_seconds: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
    tool_timeout_seconds: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "5"))


settings = Settings()


class LLMConfigurationError(RuntimeError):
    """Raised when the required online LLM configuration is missing or invalid."""


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float
    max_retries: int


def load_deepseek_settings(path: str | Path | None = None) -> DeepSeekSettings:
    """Load the mandatory DeepSeek settings from a local TOML secret file."""
    config_path = Path(
        path or os.getenv("LLM_CONFIG_FILE", str(ROOT / "config" / "llm.toml"))
    )
    if not config_path.is_file():
        raise LLMConfigurationError(
            "LLM unavailable: copy config/llm.example.toml to config/llm.toml first"
        )
    try:
        payload: dict[str, Any] = tomllib.loads(config_path.read_text(encoding="utf-8"))
        section = payload["deepseek"]
        api_key = str(section["api_key"]).strip()
        model = str(section["model"]).strip()
        base_url = str(section.get("base_url", "https://api.deepseek.com")).strip().rstrip("/")
        timeout_seconds = float(section.get("timeout_seconds", 30))
        max_retries = int(section.get("max_retries", 2))
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise LLMConfigurationError(f"LLM unavailable: invalid config file {config_path}") from exc

    if not api_key or api_key == "sk-your-deepseek-api-key":
        raise LLMConfigurationError(
            "LLM unavailable: set deepseek.api_key in config/llm.toml"
        )
    if not model:
        raise LLMConfigurationError("LLM unavailable: deepseek.model cannot be empty")
    if not base_url.startswith("https://"):
        raise LLMConfigurationError("LLM unavailable: deepseek.base_url must use HTTPS")
    if timeout_seconds <= 0 or max_retries < 0:
        raise LLMConfigurationError("LLM unavailable: invalid timeout_seconds or max_retries")

    return DeepSeekSettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
