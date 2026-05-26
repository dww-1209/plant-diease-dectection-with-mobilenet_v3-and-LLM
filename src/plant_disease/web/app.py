"""Flask 应用工厂。"""

from __future__ import annotations

import logging

from flask import Flask

from plant_disease.config import Settings
from plant_disease.errors import InferenceError
from plant_disease.model import InferenceModel
from plant_disease.web.routes import register_routes

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> Flask:
    """构造并返回 Flask app。

    "工厂模式"的好处：测试可以用一个临时 ``Settings`` 构造一个测试专用 app，
    互相不串扰；不同环境（dev / prod）也能用不同 settings 启动多份。

    重要的几个 ``app.config`` 字段：

    - ``SETTINGS``：原始 ``Settings`` 实例，路由层读它取 LLM provider 等
    - ``INFERENCE_MODEL``：``InferenceModel`` 实例，构造失败时为 ``None``
    - ``INIT_ERROR``：构造模型时的错误消息（字符串），用于在 ``/predict``
      返回 503 时告诉用户"为啥模型没起来"
    - ``MAX_CONTENT_LENGTH``：上传文件大小上限（10MB），超过 Flask 自动 413
    - ``LLM_PROVIDERS``：路由层会按需往这里塞缓存的 provider 实例（懒初始化）

    模型加载失败**不**会让 app 起不来——会被 ``InferenceError`` 兜住塞到
    ``INIT_ERROR``，Web 仍能跑（首页/关于/LLM 接口不依赖模型）。
    """
    app = Flask(__name__)
    app.config["SETTINGS"] = settings
    app.config["INIT_ERROR"] = None
    app.config["INFERENCE_MODEL"] = None
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    try:
        app.config["INFERENCE_MODEL"] = InferenceModel(
            weights_path=settings.weights_path,
            classes_txt=settings.classes_txt,
        )
    except InferenceError as exc:
        logger.exception("inference model failed to initialize")
        app.config["INIT_ERROR"] = str(exc)

    register_routes(app)
    return app
