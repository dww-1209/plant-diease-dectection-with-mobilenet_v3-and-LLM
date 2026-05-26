"""OpenAI Chat Completions provider (GPT-3.5/4)."""

from __future__ import annotations

from typing import Any

from plant_disease.errors import LLMServiceError
from plant_disease.llm.base import LLMService

API_BASE = "https://api.openai.com/v1/chat/completions"
SYSTEM_PROMPT = "你是一位专业的植物病理学专家，擅长提供植物病害诊断和治理建议。"


class OpenAIProvider(LLMService):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo") -> None:
        super().__init__(api_key=api_key)
        self.model = model

    def _endpoint(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        return API_BASE, headers, body

    def _extract_text(self, payload: dict[str, Any]) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError(f"OpenAI 响应格式异常：{payload}") from exc
