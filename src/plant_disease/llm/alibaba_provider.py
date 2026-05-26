"""Alibaba Tongyi Qianwen (DashScope) provider."""

from __future__ import annotations

from typing import Any

from plant_disease.errors import LLMServiceError
from plant_disease.llm.base import LLMService

API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


class AlibabaTongyiProvider(LLMService):
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
        # 标准格式优先
        try:
            return payload["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        # 备选格式
        text = payload.get("output", {}).get("text") or payload.get("text")
        if text:
            return text
        raise LLMServiceError(f"通义响应格式异常：{payload}")
