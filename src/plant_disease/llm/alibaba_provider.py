"""阿里通义千问（DashScope）provider，默认 ``qwen-turbo`` 模型。"""

from __future__ import annotations

from typing import Any

from plant_disease.errors import LLMServiceError
from plant_disease.llm.base import LLMService

API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


class AlibabaTongyiProvider(LLMService):
    """通义 DashScope 原生协议（与 OpenAI 协议不同，单独实现）。"""

    def __init__(self, api_key: str, model: str = "qwen-turbo") -> None:
        super().__init__(api_key=api_key)
        self.model = model

    def _endpoint(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
        }
        return API_BASE, headers, body

    def _extract_text(self, payload: dict[str, Any]) -> str:
        # DashScope 不同模型 / 不同接入方式响应字段会变，依次兜底：
        # 1) result.output.choices[0].message.content（聊天模式标准）
        # 2) result.output.text（旧版 / 部分模型）
        # 3) result.text（极少见）
        try:
            return payload["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        text = payload.get("output", {}).get("text") or payload.get("text")
        if text:
            return text
        raise LLMServiceError(f"通义响应格式异常：{payload}")
