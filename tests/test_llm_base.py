from collections.abc import Iterator

from plant_disease.llm.base import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, LLMService


class _StubProvider(LLMService):
    def stream_treatment_advice(
        self, plant_class, disease_name, disease_degree, health_status
    ) -> Iterator[str]:
        yield " 好"
        yield "建议 "


def test_user_template_has_all_placeholders():
    rendered = USER_PROMPT_TEMPLATE.format(
        plant_class="番茄",
        disease_name="早疫病",
        disease_degree="一般",
        health_status="患病",
    )
    assert "番茄" in rendered and "早疫病" in rendered


def test_build_messages_returns_system_and_user():
    msgs = LLMService()._build_messages("番茄", "早疫病", "一般", "患病")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "番茄" in msgs[1]["content"]
    assert "早疫病" in msgs[1]["content"]


def test_get_treatment_advice_aggregates_stream():
    out = _StubProvider().get_treatment_advice("番茄", "早疫病", "一般", "患病")
    assert out == "好建议"
