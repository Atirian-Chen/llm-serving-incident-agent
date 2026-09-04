from pathlib import Path

import pytest

from incident_agent.config import LLMConfigurationError, load_deepseek_settings


def write_config(
    path: Path,
    *,
    api_key: str = "sk-test",
    base_url: str = "https://api.deepseek.com",
) -> None:
    path.write_text(
        "\n".join(
            [
                "[deepseek]",
                f'api_key = "{api_key}"',
                'model = "deepseek-chat"',
                f'base_url = "{base_url}"',
                "timeout_seconds = 12",
                "max_retries = 1",
            ]
        ),
        encoding="utf-8",
    )


def test_missing_config_is_explicitly_unavailable(tmp_path):
    with pytest.raises(LLMConfigurationError, match="LLM unavailable"):
        load_deepseek_settings(tmp_path / "missing.toml")


def test_empty_api_key_is_rejected(tmp_path):
    path = tmp_path / "llm.toml"
    write_config(path, api_key="")
    with pytest.raises(LLMConfigurationError, match="api_key"):
        load_deepseek_settings(path)


def test_non_https_endpoint_is_rejected(tmp_path):
    path = tmp_path / "llm.toml"
    write_config(path, base_url="http://localhost:8000/v1")
    with pytest.raises(LLMConfigurationError, match="HTTPS"):
        load_deepseek_settings(path)


def test_valid_config_is_loaded(tmp_path):
    path = tmp_path / "llm.toml"
    write_config(path)
    config = load_deepseek_settings(path)
    assert config.api_key == "sk-test"
    assert config.model == "deepseek-chat"
    assert config.timeout_seconds == 12
    assert config.max_retries == 1
