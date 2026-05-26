import pytest

from plant_disease.config import Settings
from plant_disease.web.app import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        weights_path=tmp_path / "fake.pth",
        classes_txt=tmp_path / "fake.txt",
        llm_provider="mock",
    )


@pytest.fixture
def client(settings, monkeypatch):
    # 不真正加载模型；create_app 应吞掉初始化错误并把它放到 config["INIT_ERROR"]。
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()
