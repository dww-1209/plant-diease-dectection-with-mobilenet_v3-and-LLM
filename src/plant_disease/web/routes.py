"""HTTP 路由层。

页面（home / nav / identify）都是渲染模板；JSON 接口两个：

- ``POST /predict``：上传图片 → 返回模型分类结果
- ``POST /get_treatment_advice``：基于分类结果调 LLM → 返回治理建议

错误统一返回 ``{"success": False, "message": "..."}``；状态码区分错误来源
（400 用户输入、500 我们的代码、502 上游 LLM、503 模型未加载）。
"""

from __future__ import annotations

import logging

from flask import Blueprint, Flask, current_app, jsonify, render_template, request

from plant_disease.errors import InferenceError, LLMConfigError, LLMServiceError
from plant_disease.llm.base import LLMService
from plant_disease.llm.factory import get_llm_service

logger = logging.getLogger(__name__)

bp = Blueprint("plant_disease", __name__)


@bp.route("/")
def home():
    """首页：项目简介 + 跳转到识别页的按钮。"""
    return render_template("home.html")


@bp.route("/nav")
def nav():
    """关于页：项目说明 + 支持的 LLM 列表。"""
    return render_template("nav.html")


@bp.route("/identify")
def identify():
    """识别页：上传 + 显示结果 + 调 LLM 拿建议。"""
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    """图片识别接口。

    Body: ``multipart/form-data``，字段名 ``image``，最大 10MB（超出 Flask 自动 413）。

    Response 形如::

        {"success": true, "data": {"class_id": 5, "probability": 0.87,
                                    "plant_class": "玉米", ...}}

    错误：
    - 503 模型未加载（启动期失败时）
    - 400 没传 image 字段 / 文件名为空
    - 500 推理过程异常
    """
    model = current_app.config.get("INFERENCE_MODEL")
    init_error = current_app.config.get("INIT_ERROR")
    if model is None:
        return jsonify({"success": False, "message": f"模型未初始化：{init_error}"}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "message": "未检测到上传文件"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "message": "文件名为空"}), 400

    try:
        result = model.predict(file.read())
        return jsonify({"success": True, "data": result})
    except InferenceError as exc:
        logger.exception("inference failed")
        return jsonify({"success": False, "message": f"预测失败：{exc}"}), 500


def _resolve_provider(provider_name: str) -> LLMService:
    """按 provider 名字取实例，进程内只 new 一次（缓存在 ``app.config``）。

    复用很关键：百度 provider 的构造函数会去换一次 access_token，每请求 new
    就等于每请求多一次 HTTP，浪费且容易触发限流。
    """
    cache: dict[str, LLMService] = current_app.config.setdefault("LLM_PROVIDERS", {})
    if provider_name in cache:
        return cache[provider_name]
    settings = current_app.config["SETTINGS"]
    service = get_llm_service(provider_name, settings)
    cache[provider_name] = service
    return service


@bp.route("/get_treatment_advice", methods=["POST"])
def get_treatment_advice():
    """LLM 治理建议接口。

    Body: ``application/json``::

        {
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "alibaba"   // 可选；不传则用 settings.llm_provider
        }

    Response: ``{"success": true, "advice": "..."}`` 或失败时 ``message`` + 状态码。

    错误：400 缺字段 / 不支持的 provider / 缺 key；502 上游 LLM 失败。
    """
    data = request.get_json(silent=True) or {}
    plant_class = data.get("plant_class", "")
    disease_name = data.get("disease_name", "")
    disease_degree = data.get("disease_degree", "")
    health_status = data.get("health_status", "")

    if not plant_class or not disease_name:
        return (
            jsonify({"success": False, "message": "缺少必要参数 plant_class 或 disease_name"}),
            400,
        )

    settings = current_app.config["SETTINGS"]
    provider_name = (data.get("provider") or settings.llm_provider or "mock").strip().lower()

    try:
        service = _resolve_provider(provider_name)
    except LLMConfigError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400

    try:
        advice = service.get_treatment_advice(
            plant_class=plant_class,
            disease_name=disease_name,
            disease_degree=disease_degree,
            health_status=health_status,
        )
        return jsonify({"success": True, "advice": advice})
    except LLMServiceError as exc:
        logger.exception("llm call failed")
        return jsonify({"success": False, "message": str(exc)}), 502


def register_routes(app: Flask) -> None:
    """把本模块的 Blueprint 挂到给定 Flask app 上。由 ``create_app`` 调用。"""
    app.register_blueprint(bp)
