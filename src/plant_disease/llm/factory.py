"""LLM provider 工厂：name → 实例。

所有真实 provider 都走 OpenAI 协议（``OpenAICompatibleProvider``），区别只是
``base_url`` + ``model`` + 读哪个环境变量。``PROVIDERS`` 把这些差异收束成一张
表，新增一家几行配置即可。

# auto fallback
``LLM_PROVIDER=auto`` 时按 ``_AUTO_ORDER`` 找第一个配了 key 的 provider；都没
配的话落到 ``mock``，保证开箱即用。显式指定的 provider 缺 key 会抛
``LLMConfigError``——避免用户以为在用 GPT 实际跑了 mock。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from plant_disease.config import Settings
from plant_disease.errors import LLMConfigError
from plant_disease.llm.base import LLMService
from plant_disease.llm.mock_provider import MockProvider
from plant_disease.llm.openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProviderSpec:
    env_var: str
    base_url: str | None
    model: str


# OpenAI 协议四件套；各家的 base_url + 默认模型在这里集中维护。
PROVIDERS: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec(
        env_var="OPENAI_API_KEY",
        base_url=None,
        model="gpt-4o-mini",
    ),
    "deepseek": _ProviderSpec(
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    ),
    "alibaba": _ProviderSpec(
        env_var="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
    ),
    "zhipu": _ProviderSpec(
        env_var="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
    ),
}

# auto 模式的优先级。GPT 第一档，mock 始终垫底。
_AUTO_ORDER = ("openai", "deepseek", "alibaba", "zhipu")


def _build(name: str) -> LLMService:
    spec = PROVIDERS[name]
    api_key = os.environ.get(spec.env_var, "")
    if not api_key:
        raise LLMConfigError(f"缺少环境变量 {spec.env_var}（provider={name}）")
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=spec.model,
        base_url=spec.base_url,
        label=name,
    )


def _auto_select() -> LLMService:
    for name in _AUTO_ORDER:
        if os.environ.get(PROVIDERS[name].env_var):
            logger.info("auto-selected LLM provider: %s", name)
            return _build(name)
    logger.warning("no LLM api key configured, falling back to mock")
    return MockProvider()


def get_llm_service(provider: str, settings: Settings) -> LLMService:
    """按名字构造 ``LLMService``。

    Args:
        provider: ``auto`` / ``mock`` / ``openai`` / ``deepseek`` / ``alibaba`` /
            ``zhipu``，大小写不敏感。``auto`` 走优先级 fallback。
        settings: 此处不读，仅为保持接口兼容（key 直接从环境变量取）。

    Raises:
        LLMConfigError: 名字不认识 / 显式指定但未配 key。
    """
    del settings  # not used; keep parameter for API stability
    name = provider.strip().lower()
    if name == "auto":
        return _auto_select()
    if name == "mock":
        return MockProvider()
    if name in PROVIDERS:
        return _build(name)
    raise LLMConfigError(
        f"不支持的提供商：{provider}（可选：auto / mock / {' / '.join(PROVIDERS)}）"
    )
