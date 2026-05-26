"""LLM provider 基类 + 共享 prompt 模板。

# 设计：流式 + 一次性聚合两条接口共存

``stream_treatment_advice`` 是底层接口，逐 chunk yield 文本，给 SSE 用。
``get_treatment_advice`` 是 ``stream_*`` 的一次性聚合，给老的 JSON 接口和
单元测试用。所有 provider 只要实现 ``stream_treatment_advice``，自动同时
支持两种调用方式。

# Prompt 拆成 system + user 两条 message
- ``SYSTEM_PROMPT``：固定的角色/规范/输出要求；不随病例变化。
- ``USER_PROMPT_TEMPLATE``：每次只塞病例的四个字段（植物/病害/程度/状态）。

OpenAI 协议（含 4 家兼容端点）原生支持 messages 数组多角色。这样改的好处是：
- 模型对 system 角色权重更高，"专业、用中文、按结构作答"等约束更稳；
- 后续若想调整规范（比如"加粗关键字"），只动 SYSTEM_PROMPT 即可。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一位资深的植物病理学专家与农业技术推广员。回答时严格遵守以下规范：\n"
    "\n"
    "1. 全程使用中文，语气专业、克制、可执行，不要客套与免责声明。\n"
    "2. 严格按以下四个章节输出，每个章节都要有，章节标题保留原字面：\n"
    "   1. 病害的简要说明\n"
    "   2. 具体的治理措施（化学防治、生物防治、农业防治分点列出）\n"
    "   3. 预防措施\n"
    "   4. 注意事项\n"
    "3. 化学防治给出代表性药剂通用名 + 浓度区间或剂量参考即可，不指定品牌。\n"
    "4. 内容贴合用户给出的「患病程度」与「健康状况」做差异化建议（程度轻 →\n"
    "   优先农业/生物防治；严重 → 强调化学防治与隔离）。\n"
    "5. 如用户给的字段与常识矛盾（如健康却报严重病害），按字段表述继续给建议，\n"
    "   不质疑用户的输入。"
)

USER_PROMPT_TEMPLATE = (
    "请基于以下识别结果给出治理建议：\n"
    "\n"
    "- 植物种类：{plant_class}\n"
    "- 病害名称：{disease_name}\n"
    "- 患病程度：{disease_degree}\n"
    "- 健康状况：{health_status}"
)


class LLMService:
    """所有 LLM provider 的基类。子类实现 ``stream_treatment_advice`` 即可。"""

    def _build_messages(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> list[dict[str, str]]:
        """构造 OpenAI 协议的 messages 数组（system + user 两条）。"""
        user = USER_PROMPT_TEMPLATE.format(
            plant_class=plant_class,
            disease_name=disease_name,
            disease_degree=disease_degree,
            health_status=health_status,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

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
