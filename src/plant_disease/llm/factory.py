"""LLM provider 工厂：name → 实例。

所有真实 provider 都走 OpenAI 协议（``OpenAICompatibleProvider``），区别只是
``base_url`` + ``model`` + 读哪个环境变量。``PROVIDERS`` 把这些差异收束成一张
表，新增一家几行配置即可。

# auto fallback
``LLM_PROVIDER=auto`` 时按 ``_AUTO_ORDER`` 找第一个配了 key 的 provider；都没
配的话落到 ``mock``，保证开箱即用。显式指定的 provider 缺 key 会抛
``LLMConfigError``——避免用户以为在用 GPT 实际跑了 mock。

# 模型默认值 + 候选清单
``models`` 字段列出每家"目前主推"的几个 model id（截至 2026-05），第一个是
``_build()`` 启动 provider 时的默认模型。前端用这个清单填充 datalist
（``GET /api/llm/providers``），但只是**建议**，用户可以手填任意字符串：
官方今天发布新 ID，前端马上能用，不必等代码改默认值。
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
class ProviderSpec:
    env_var: str
    base_url: str | None
    models: tuple[str, ...]  # 第一个为默认；其余为前端建议候选

    @property
    def default_model(self) -> str:
        return self.models[0]


# OpenAI 协议四件套；各家的 base_url + 推荐模型在这里集中维护。
# 模型名截至 2026-05 各家官网；过期了直接改这张表 / 用户也能在 web 上覆盖。
PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        env_var="OPENAI_API_KEY",
        base_url=None,
        models=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4o"),
    ),
    "deepseek": ProviderSpec(
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        models=("deepseek-v4-pro", "deepseek-v4-flash"),
    ),
    "alibaba": ProviderSpec(
        env_var="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=("qwen3.7-max", "qwen3.6-plus", "qwen3.6-flash"),
    ),
    "zhipu": ProviderSpec(
        env_var="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        models=("glm-5.1", "glm-5", "glm-4.7", "glm-4.7-flash"),
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
        model=spec.default_model,
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


def build_one_off(provider: str, api_key: str, model: str) -> LLMService:
    """按用户在 web 上手填的 provider/key/model new 一个临时 provider。

    与 ``get_llm_service`` 不同的是：
    - api_key 不从环境变量读，由调用方显式传入；
    - 不被路由层缓存（每请求 new 一份），避免不同用户的 key 互相污染。

    Args:
        provider: ``mock`` 或 ``PROVIDERS`` 里的某个名字（不接受 ``auto``，因为
            既然用户都手填了 key，意图就是用这家）。
        api_key: 用户传过来的 key。``mock`` 时可空。
        model: 模型 id，可以是 ``PROVIDERS[name].models`` 里的任一项，也可以
            是任意官方新发布的字符串。

    Raises:
        LLMConfigError: provider 名字不认识 / 非 mock 但 api_key 为空。
    """
    name = provider.strip().lower()
    if name == "mock":
        return MockProvider()
    if name not in PROVIDERS:
        raise LLMConfigError(
            f"不支持的提供商：{provider}（可选：mock / {' / '.join(PROVIDERS)}）"
        )
    if not api_key:
        raise LLMConfigError(f"{name} 需要 api_key")
    spec = PROVIDERS[name]
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=model or spec.default_model,
        base_url=spec.base_url,
        label=name,
    )
