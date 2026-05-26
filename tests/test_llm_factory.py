import pytest

from plant_disease.errors import LLMConfigError
from plant_disease.llm.factory import get_llm_service
from plant_disease.llm.mock_provider import MockProvider


def test_factory_returns_mock(monkeypatch):
    assert isinstance(get_llm_service("mock"), MockProvider)
    assert isinstance(get_llm_service("MOCK"), MockProvider)


def test_factory_invalid_provider_raises():
    with pytest.raises(ValueError):
        get_llm_service("nope")


def test_factory_alibaba_without_key_raises_llm_config_error(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        get_llm_service("alibaba")
