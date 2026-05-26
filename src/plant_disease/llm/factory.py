"""Provider factory."""

from __future__ import annotations

import os

from plant_disease.errors import LLMConfigError
from plant_disease.llm.base import LLMService
from plant_disease.llm.mock_provider import MockProvider

_VALID = {"mock", "openai", "baidu", "alibaba"}


def get_llm_service(provider: str) -> LLMService:
    name = provider.strip().lower()
    if name not in _VALID:
        raise ValueError(f"不支持的提供商：{provider}（可选：{sorted(_VALID)}）")

    if name == "mock":
        return MockProvider()

    # Check required env vars BEFORE lazy-importing the provider module,
    # so that missing config raises LLMConfigError without blowing up
    # if the optional provider module is unavailable.
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise LLMConfigError("缺少环境变量 OPENAI_API_KEY")
        from plant_disease.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=key)

    if name == "baidu":
        api_key = os.environ.get("BAIDU_API_KEY", "")
        secret = os.environ.get("BAIDU_SECRET_KEY", "")
        if not api_key or not secret:
            raise LLMConfigError("缺少环境变量 BAIDU_API_KEY 或 BAIDU_SECRET_KEY")
        from plant_disease.llm.baidu_provider import BaiduWenxinProvider

        return BaiduWenxinProvider(api_key=api_key, secret_key=secret)

    # alibaba
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    if not key:
        raise LLMConfigError("缺少环境变量 DASHSCOPE_API_KEY")
    from plant_disease.llm.alibaba_provider import AlibabaTongyiProvider

    return AlibabaTongyiProvider(api_key=key)
