"""Provider factory."""

from __future__ import annotations

from plant_disease.config import Settings
from plant_disease.errors import LLMConfigError
from plant_disease.llm.base import LLMService
from plant_disease.llm.mock_provider import MockProvider

_VALID = {"mock", "openai", "baidu", "alibaba"}


def get_llm_service(provider: str, settings: Settings) -> LLMService:
    name = provider.strip().lower()
    if name not in _VALID:
        raise LLMConfigError(f"不支持的提供商：{provider}（可选：{sorted(_VALID)}）")

    if name == "mock":
        return MockProvider()

    # Check required keys BEFORE lazy-importing the provider module so that
    # a missing optional dependency or unused provider never blocks the others.
    if name == "openai":
        if not settings.openai_api_key:
            raise LLMConfigError("缺少环境变量 OPENAI_API_KEY")
        from plant_disease.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key)

    if name == "baidu":
        if not settings.baidu_api_key or not settings.baidu_secret_key:
            raise LLMConfigError("缺少环境变量 BAIDU_API_KEY 或 BAIDU_SECRET_KEY")
        from plant_disease.llm.baidu_provider import BaiduWenxinProvider

        return BaiduWenxinProvider(
            api_key=settings.baidu_api_key,
            secret_key=settings.baidu_secret_key,
        )

    # alibaba
    if not settings.dashscope_api_key:
        raise LLMConfigError("缺少环境变量 DASHSCOPE_API_KEY")
    from plant_disease.llm.alibaba_provider import AlibabaTongyiProvider

    return AlibabaTongyiProvider(api_key=settings.dashscope_api_key)
