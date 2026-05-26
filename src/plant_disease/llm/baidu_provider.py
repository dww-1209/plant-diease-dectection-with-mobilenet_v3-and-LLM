"""Baidu Wenxin (ERNIE) provider."""

from __future__ import annotations

import logging
from typing import Any

import requests

from plant_disease.errors import LLMConfigError, LLMServiceError
from plant_disease.llm.base import LLMService

logger = logging.getLogger(__name__)

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
CHAT_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"


class BaiduWenxinProvider(LLMService):
    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__(api_key=api_key)
        self.secret_key = secret_key
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        try:
            resp = requests.post(TOKEN_URL, params=params, timeout=10)
            resp.raise_for_status()
            token = resp.json().get("access_token", "")
        except requests.RequestException as exc:
            raise LLMConfigError(f"获取百度 access_token 失败：{exc}") from exc
        if not token:
            raise LLMConfigError("百度 access_token 响应为空")
        return token

    def _endpoint(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{CHAT_URL}?access_token={self.access_token}"
        headers = {"Content-Type": "application/json"}
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        return url, headers, body

    def _extract_text(self, payload: dict[str, Any]) -> str:
        text = payload.get("result")
        if not text:
            raise LLMServiceError(f"百度响应缺少 result：{payload}")
        return text
