"""HTTP routes."""

from __future__ import annotations

import logging
import os

from flask import (
    Blueprint,
    Flask,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from plant_disease.errors import InferenceError, LLMConfigError, LLMServiceError
from plant_disease.llm.factory import get_llm_service

logger = logging.getLogger(__name__)

bp = Blueprint("plant_disease", __name__)


@bp.route("/")
def home():
    return render_template("home.html")


@bp.route("/nav")
def nav():
    return render_template("nav.html")


@bp.route("/identify")
def identify():
    return render_template("index.html")


@bp.route("/images/<path:filename>")
def serve_image(filename: str):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(root_dir, "..", "..", ".."))
    return send_from_directory(project_root, filename)


@bp.route("/predict", methods=["POST"])
def predict():
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


@bp.route("/get_treatment_advice", methods=["POST"])
def get_treatment_advice():
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
        service = get_llm_service(provider_name)
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
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
    app.register_blueprint(bp)
