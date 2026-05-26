from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, AuthenticationError

from plant_disease.errors import LLMConfigError, LLMServiceError
from plant_disease.llm.openai_compatible import OpenAICompatibleProvider


def _stream_chunks(*texts):
    """模拟 openai 流式响应：每个 chunk 是一个带 .choices[].delta.content 的对象。"""
    for text in texts:
        delta = SimpleNamespace(content=text)
        choice = SimpleNamespace(delta=delta)
        yield SimpleNamespace(choices=[choice])


def _make_provider() -> OpenAICompatibleProvider:
    with patch("plant_disease.llm.openai_compatible.OpenAI"):
        return OpenAICompatibleProvider(
            api_key="sk-test", model="test-model", base_url=None, label="test"
        )


def test_stream_yields_concatenable_chunks():
    provider = _make_provider()
    fake_create = MagicMock(return_value=_stream_chunks("Hello", " ", "World"))
    provider._client.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_create))

    out = "".join(provider.stream_treatment_advice("番茄", "早疫病", "一般", "患病"))
    assert out == "Hello World"
    fake_create.assert_called_once()
    kwargs = fake_create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == "test-model"
    # 必须是 system + user 两条消息
    messages = kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "番茄" in messages[1]["content"] and "早疫病" in messages[1]["content"]


def test_stream_skips_chunks_without_choices():
    provider = _make_provider()

    def gen():
        yield SimpleNamespace(choices=[])  # 类似最终的 usage event
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])

    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock(return_value=gen()))
    )
    assert "".join(provider.stream_treatment_advice("a", "b", "c", "d")) == "ok"


def test_get_treatment_advice_aggregates_and_strips():
    provider = _make_provider()
    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=MagicMock(return_value=_stream_chunks(" 建议", "内容 "))
        )
    )
    assert provider.get_treatment_advice("a", "b", "c", "d") == "建议内容"


def test_authentication_error_maps_to_config_error():
    provider = _make_provider()
    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=MagicMock(
                side_effect=AuthenticationError(
                    message="bad key", response=MagicMock(), body=None
                )
            )
        )
    )
    with pytest.raises(LLMConfigError):
        list(provider.stream_treatment_advice("a", "b", "c", "d"))


def test_api_error_maps_to_service_error():
    provider = _make_provider()
    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=MagicMock(side_effect=APIError("boom", request=MagicMock(), body=None))
        )
    )
    with pytest.raises(LLMServiceError):
        list(provider.stream_treatment_advice("a", "b", "c", "d"))
