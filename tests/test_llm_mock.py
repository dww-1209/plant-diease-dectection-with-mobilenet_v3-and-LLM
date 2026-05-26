from plant_disease.llm.mock_provider import MockProvider


def test_mock_advice_includes_inputs():
    out = MockProvider().get_treatment_advice("番茄", "早疫病", "一般", "患病")
    assert "番茄" in out and "早疫病" in out
    assert "化学防治" in out
