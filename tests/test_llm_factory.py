from pathlib import Path

import pytest

from plant_disease.config import Settings
from plant_disease.errors import LLMConfigError
from plant_disease.llm.factory import PROVIDERS, build_one_off, get_llm_service
from plant_disease.llm.mock_provider import MockProvider
from plant_disease.llm.openai_compatible import OpenAICompatibleProvider

KEY_ENV_VARS = [spec.env_var for spec in PROVIDERS.values()]


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for var in KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _settings(provider: str = "auto") -> Settings:
    return Settings(
        weights_path=Path("x.pth"),
        classes_txt=Path("y.txt"),
        llm_provider=provider,
    )


def test_factory_returns_mock():
    assert isinstance(get_llm_service("mock", _settings()), MockProvider)
    assert isinstance(get_llm_service("MOCK", _settings()), MockProvider)


def test_factory_invalid_provider_raises_config_error():
    with pytest.raises(LLMConfigError):
        get_llm_service("nope", _settings())


def test_factory_explicit_provider_without_key_raises(monkeypatch):
    with pytest.raises(LLMConfigError):
        get_llm_service("openai", _settings())


def test_factory_explicit_provider_with_key_returns_compatible(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    service = get_llm_service("deepseek", _settings())
    assert isinstance(service, OpenAICompatibleProvider)
    assert service.label == "deepseek"
    assert service.model == PROVIDERS["deepseek"].default_model


def test_auto_falls_back_to_mock_when_no_keys():
    assert isinstance(get_llm_service("auto", _settings()), MockProvider)


def test_auto_prefers_openai_over_others(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ali")
    service = get_llm_service("auto", _settings())
    assert isinstance(service, OpenAICompatibleProvider)
    assert service.label == "openai"


def test_auto_picks_first_configured_in_priority_order(monkeypatch):
    # 没有 openai key 时，落到 deepseek
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.setenv("ZHIPU_API_KEY", "sk-zhipu")
    service = get_llm_service("auto", _settings())
    assert isinstance(service, OpenAICompatibleProvider)
    assert service.label == "deepseek"


def test_build_one_off_uses_supplied_key_and_model():
    # build_one_off 不读环境变量，key/model 全靠调用方传
    service = build_one_off("openai", "sk-user-key", "gpt-5.4-mini")
    assert isinstance(service, OpenAICompatibleProvider)
    assert service.label == "openai"
    assert service.model == "gpt-5.4-mini"


def test_build_one_off_falls_back_to_default_model():
    service = build_one_off("zhipu", "sk-user", "")
    assert isinstance(service, OpenAICompatibleProvider)
    assert service.model == PROVIDERS["zhipu"].default_model


def test_build_one_off_mock_works_without_key():
    assert isinstance(build_one_off("mock", "", ""), MockProvider)


def test_build_one_off_requires_key_for_real_provider():
    with pytest.raises(LLMConfigError):
        build_one_off("openai", "", "gpt-5.5")


def test_build_one_off_rejects_unknown_provider():
    with pytest.raises(LLMConfigError):
        build_one_off("nope", "k", "m")
