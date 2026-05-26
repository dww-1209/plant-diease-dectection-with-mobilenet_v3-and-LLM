"""Base class + shared prompt for all LLM providers."""

from __future__ import annotations

import logging
from typing import Any

import requests

from plant_disease.errors import LLMServiceError

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

DEFAULT_TIMEOUT = 30


class LLMService:
    """Base provider. Subclasses implement _endpoint and _extract_text."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

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

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            raise LLMServiceError(f"调用 LLM 超时：{exc}") from exc
        except requests.HTTPError as exc:
            raise LLMServiceError(f"LLM HTTP 错误：{exc}") from exc
        except requests.RequestException as exc:
            raise LLMServiceError(f"LLM 网络错误：{exc}") from exc
        except ValueError as exc:
            raise LLMServiceError(f"LLM 响应不是合法 JSON：{exc}") from exc

    def _endpoint(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        raise NotImplementedError

    def get_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
        prompt = self._build_prompt(plant_class, disease_name, disease_degree, health_status)
        url, headers, body = self._endpoint(prompt)
        payload = self._post_json(url, headers, body)
        text = self._extract_text(payload)
        return text.strip()
