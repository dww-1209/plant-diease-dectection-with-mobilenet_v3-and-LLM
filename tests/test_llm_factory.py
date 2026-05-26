from pathlib import Path

import pytest

from plant_disease.config import Settings
from plant_disease.errors import LLMConfigError
from plant_disease.llm.factory import get_llm_service
from plant_disease.llm.mock_provider import MockProvider


def _settings(**overrides) -> Settings:
    base = dict(
        weights_path=Path("x.pth"),
        classes_txt=Path("y.txt"),
        llm_provider="mock",
    )
    base.update(overrides)
    return Settings(**base)


def test_factory_returns_mock():
    s = _settings()
    assert isinstance(get_llm_service("mock", s), MockProvider)
    assert isinstance(get_llm_service("MOCK", s), MockProvider)


def test_factory_invalid_provider_raises_config_error():
    with pytest.raises(LLMConfigError):
        get_llm_service("nope", _settings())


def test_factory_alibaba_without_key_raises_llm_config_error():
    with pytest.raises(LLMConfigError):
        get_llm_service("alibaba", _settings(dashscope_api_key=""))


def test_factory_openai_without_key_raises_llm_config_error():
    with pytest.raises(LLMConfigError):
        get_llm_service("openai", _settings(openai_api_key=""))


def test_factory_baidu_partial_key_raises_llm_config_error():
    with pytest.raises(LLMConfigError):
        get_llm_service("baidu", _settings(baidu_api_key="x", baidu_secret_key=""))
