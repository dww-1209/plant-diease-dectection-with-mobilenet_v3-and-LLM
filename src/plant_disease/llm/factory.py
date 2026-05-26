"""根据名字 + Settings 构造对应的 LLM provider 实例。"""

from __future__ import annotations

from plant_disease.config import Settings
from plant_disease.errors import LLMConfigError
from plant_disease.llm.base import LLMService
from plant_disease.llm.mock_provider import MockProvider

_VALID = {"mock", "openai", "baidu", "alibaba"}


def get_llm_service(provider: str, settings: Settings) -> LLMService:
    """工厂函数：把字符串名字翻译成实例。

    Args:
        provider: ``mock`` / ``openai`` / ``baidu`` / ``alibaba``，大小写不敏感。
        settings: 已加载的 ``Settings``，用于读取各 provider 的 API key。

    Returns:
        对应 provider 的 ``LLMService`` 实例。

    Raises:
        LLMConfigError: 名字不认识 / 必需的 key 没填。

    实现细节：除了 ``mock`` 之外，所有真实 provider 模块都是**懒加载**的——
    只有用到那家时才 import。这样万一某个 provider 模块以后引入了重型可选
    依赖，也不会拖累其他 provider。
    """
    name = provider.strip().lower()
    if name not in _VALID:
        raise LLMConfigError(f"不支持的提供商：{provider}（可选：{sorted(_VALID)}）")

    if name == "mock":
        return MockProvider()

    # Key 检查必须在 import 之前——这样 key 缺失时报清晰的 LLMConfigError，
    # 而不是底层模块 import 失败的怪异 traceback。
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
