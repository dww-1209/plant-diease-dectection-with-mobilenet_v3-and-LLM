from pathlib import Path

import pytest

from plant_disease.config import Settings, load_settings


def test_load_settings_defaults(monkeypatch):
    for key in [
        "LLM_PROVIDER",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "BAIDU_API_KEY",
        "BAIDU_SECRET_KEY",
        "WEIGHTS_PATH",
        "CLASSES_TXT",
        "FLASK_DEBUG",
        "PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    s = load_settings()
    assert isinstance(s, Settings)
    assert s.llm_provider == "mock"
    assert s.flask_debug is False
    assert s.port == 5000
    assert s.weights_path == Path("resources/mobilenetv2_best.pth")
    assert s.classes_txt == Path("resources/actual_classed_v2.txt")
    assert s.dashscope_api_key == ""


def test_load_settings_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "alibaba")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-xyz")
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("WEIGHTS_PATH", "/tmp/w.pth")

    s = load_settings()
    assert s.llm_provider == "alibaba"
    assert s.dashscope_api_key == "sk-xyz"
    assert s.flask_debug is True
    assert s.port == 8080
    assert s.weights_path == Path("/tmp/w.pth")


@pytest.mark.parametrize(
    "value,expected", [("0", False), ("false", False), ("1", True), ("true", True), ("True", True)]
)
def test_flask_debug_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("FLASK_DEBUG", value)
    assert load_settings().flask_debug is expected
