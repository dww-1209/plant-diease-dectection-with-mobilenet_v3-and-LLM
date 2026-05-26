from unittest.mock import MagicMock, patch

import pytest
import requests

from plant_disease.errors import LLMServiceError
from plant_disease.llm.base import TREATMENT_PROMPT_TEMPLATE, LLMService


class _StubProvider(LLMService):
    """Concrete subclass for testing the base flow."""

    def _endpoint(self, prompt):
        return "http://example.com/api", {"X-K": "v"}, {"prompt": prompt}

    def _extract_text(self, payload):
        return payload["text"]


def test_template_has_all_placeholders():
    rendered = TREATMENT_PROMPT_TEMPLATE.format(
        plant_class="番茄", disease_name="早疫病",
        disease_degree="一般", health_status="患病",
    )
    assert "番茄" in rendered and "早疫病" in rendered


def test_get_treatment_advice_success():
    provider = _StubProvider(api_key="k")
    with patch.object(provider, "_post_json", return_value={"text": " 好建议 "}) as mock_post:
        out = provider.get_treatment_advice("番茄", "早疫病", "一般", "患病")
    assert out == "好建议"
    args, _ = mock_post.call_args
    assert args[0] == "http://example.com/api"


def test_post_json_timeout_raises_llm_service_error():
    provider = _StubProvider(api_key="k")
    with patch("plant_disease.llm.base.requests.post", side_effect=requests.Timeout):
        with pytest.raises(LLMServiceError):
            provider._post_json("u", {}, {})


def test_post_json_http_error_raises_llm_service_error():
    provider = _StubProvider(api_key="k")
    fake = MagicMock()
    fake.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("plant_disease.llm.base.requests.post", return_value=fake):
        with pytest.raises(LLMServiceError):
            provider._post_json("u", {}, {})
