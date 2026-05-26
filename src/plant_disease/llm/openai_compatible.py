"""统一的 OpenAI 兼容 provider。

OpenAI / DeepSeek / 通义 DashScope（兼容模式）/ 智谱 GLM 都接受 OpenAI 协议
（``/v1/chat/completions``），区别只在 ``base_url`` + 默认 ``model``。这里用
``openai.OpenAI`` 客户端跑流式 chat completion，所有四家共用同一份代码。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from openai import APIError, AuthenticationError, OpenAI, OpenAIError

from plant_disease.errors import LLMConfigError, LLMServiceError
from plant_disease.llm.base import LLMService

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMService):
    """通用 OpenAI 协议 provider。

    Args:
        api_key: API key。
        model: 模型名（如 ``gpt-4o-mini`` / ``deepseek-chat`` / ``qwen-turbo`` /
            ``glm-4-flash``）。
        base_url: 兼容端点地址；OpenAI 官方传 ``None`` 走默认即可。
        label: 出错时报错文案里的提供商名字，仅用于诊断。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        label: str = "openai",
    ) -> None:
        self.model = model
        self.label = label
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def stream_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> Iterator[str]:
        prompt = self._build_prompt(plant_class, disease_name, disease_degree, health_status)
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
            )
            for event in stream:
                # 流式响应里部分 chunk 没有 choices（如最终的 usage 事件），跳过。
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except AuthenticationError as exc:
            raise LLMConfigError(f"{self.label} 凭证无效：{exc}") from exc
        except APIError as exc:
            raise LLMServiceError(f"{self.label} API 错误：{exc}") from exc
        except OpenAIError as exc:
            raise LLMServiceError(f"{self.label} 调用失败：{exc}") from exc
