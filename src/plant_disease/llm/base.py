"""LLM provider 基类 + 共享 prompt 模板。

# 设计：流式 + 一次性聚合两条接口共存

``stream_treatment_advice`` 是底层接口，逐 chunk yield 文本，给 SSE 用。
``get_treatment_advice`` 是 ``stream_*`` 的一次性聚合，给老的 JSON 接口和
单元测试用。所有 provider 只要实现 ``stream_treatment_advice``，自动同时
支持两种调用方式。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

TREATMENT_PROMPT_TEMPLATE = (
    "你是一位专业的植物病理学专家。"
    "请根据以下信息，提供详细的植物病害治理建议：\n"
    "\n"
    "植物种类：{plant_class}\n"
    "病害名称：{disease_name}\n"
    "患病程度：{disease_degree}\n"
    "健康状况：{health_status}\n"
    "\n"
    "请提供：\n"
    "1. 病害的简要说明\n"
    "2. 具体的治理措施（包括化学防治、生物防治、农业防治等）\n"
    "3. 预防措施\n"
    "4. 注意事项\n"
    "\n"
    "请用中文回答，内容要专业、实用、易懂。"
)


class LLMService:
    """所有 LLM provider 的基类。子类实现 ``stream_treatment_advice`` 即可。"""

    def _build_prompt(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
        return TREATMENT_PROMPT_TEMPLATE.format(
            plant_class=plant_class,
            disease_name=disease_name,
            disease_degree=disease_degree,
            health_status=health_status,
        )

    def stream_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> Iterator[str]:
        """逐块产出建议文本。

        Raises:
            LLMServiceError: 网络/HTTP/响应格式错误。
        """
        raise NotImplementedError

    def get_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
        """``stream_*`` 的一次性聚合版本，给非流式调用使用。"""
        chunks = self.stream_treatment_advice(
            plant_class=plant_class,
            disease_name=disease_name,
            disease_degree=disease_degree,
            health_status=health_status,
        )
        return "".join(chunks).strip()
