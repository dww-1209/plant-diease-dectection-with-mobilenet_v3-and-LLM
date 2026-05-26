"""Centralized environment-driven settings.

This is the only module that reads os.environ for the runtime application.
The root-level entry scripts (run_web.py, train_model.py, prepare_dataset.py)
read PLANT_DISEASE_DEBUG only for logging configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Settings:
    weights_path: Path
    classes_txt: Path
    llm_provider: str
    dashscope_api_key: str = ""
    openai_api_key: str = ""
    baidu_api_key: str = ""
    baidu_secret_key: str = ""
    flask_debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000


def load_settings() -> Settings:
    return Settings(
        weights_path=Path(os.environ.get("WEIGHTS_PATH", "resources/mobilenetv2_best.pth")),
        classes_txt=Path(os.environ.get("CLASSES_TXT", "resources/actual_classed_v2.txt")),
        llm_provider=os.environ.get("LLM_PROVIDER", "mock").strip().lower(),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        baidu_api_key=os.environ.get("BAIDU_API_KEY", ""),
        baidu_secret_key=os.environ.get("BAIDU_SECRET_KEY", ""),
        flask_debug=_bool_env("FLASK_DEBUG", False),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )
