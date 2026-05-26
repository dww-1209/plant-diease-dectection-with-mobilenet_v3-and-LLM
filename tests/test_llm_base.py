from collections.abc import Iterator

from plant_disease.llm.base import TREATMENT_PROMPT_TEMPLATE, LLMService


class _StubProvider(LLMService):
    def stream_treatment_advice(
        self, plant_class, disease_name, disease_degree, health_status
    ) -> Iterator[str]:
        yield " 好"
        yield "建议 "


def test_template_has_all_placeholders():
    rendered = TREATMENT_PROMPT_TEMPLATE.format(
        plant_class="番茄",
        disease_name="早疫病",
        disease_degree="一般",
        health_status="患病",
    )
    assert "番茄" in rendered and "早疫病" in rendered


def test_get_treatment_advice_aggregates_stream():
    out = _StubProvider().get_treatment_advice("番茄", "早疫病", "一般", "患病")
    assert out == "好建议"
