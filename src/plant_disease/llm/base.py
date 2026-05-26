"""LLM 提供商基类 + 共享 prompt 模板。

# 设计：模板方法（Template Method）模式
``LLMService.get_treatment_advice`` 把"组装 prompt → 拼请求 → 发 HTTP →
解析响应"这条链路固化下来，子类只需实现两个钩子：

- ``_endpoint(prompt)``：返回 ``(url, headers, body)`` —— 各家 API 协议不同
- ``_extract_text(response_json)``：从响应 JSON 里挖出文本 —— 各家响应字段不同

这样新增提供商只要 30 行左右；OpenAI / 百度 / 通义三家共享 prompt 文本和错
误处理逻辑，不再各自复制粘贴。
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from plant_disease.errors import LLMServiceError

logger = logging.getLogger(__name__)

# 所有 provider 共用同一份 prompt。改这里 = 改全部模型的提示。
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
    """所有 LLM provider 的基类。

    子类只需实现 ``_endpoint`` 和 ``_extract_text``，无需关心 HTTP/错误处理。
    """

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def _build_prompt(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
        """把四个字段塞进共享模板，返回最终的中文 prompt。"""
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
        """发 POST，所有 ``requests`` 异常统一翻译成 ``LLMServiceError``。

        异常顺序很重要：``Timeout`` 和 ``HTTPError`` 都是 ``RequestException`` 的
        子类，必须先 except 具体子类再 except 父类，否则永远走不到具体分支。
        """
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
            # resp.json() 解析失败时抛 ValueError（json.JSONDecodeError 的父类）
            raise LLMServiceError(f"LLM 响应不是合法 JSON：{exc}") from exc

    def _endpoint(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """子类钩子：返回 ``(url, headers, body)``，描述这家 API 怎么调。"""
        raise NotImplementedError

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        """子类钩子：从响应 JSON 中取出回复文本。"""
        raise NotImplementedError

    def get_treatment_advice(
        self,
        plant_class: str,
        disease_name: str,
        disease_degree: str,
        health_status: str,
    ) -> str:
        """对外的核心方法。其余都是私有。

        Returns:
            LLM 给出的治理建议文本（已 strip）。

        Raises:
            LLMServiceError: 网络/HTTP/响应格式错误。
        """
        prompt = self._build_prompt(plant_class, disease_name, disease_degree, health_status)
        url, headers, body = self._endpoint(prompt)
        payload = self._post_json(url, headers, body)
        text = self._extract_text(payload)
        return text.strip()
