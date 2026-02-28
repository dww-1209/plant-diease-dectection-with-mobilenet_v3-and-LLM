import os
from flask import Flask, jsonify, render_template, request, send_from_directory

from model import InferenceModel
from llm_service import get_llm_service

app = Flask(__name__)

# 初始化模型（服务启动时只加载一次）
try:
    inference_model = InferenceModel()
except Exception as exc:
    inference_model = None
    init_error = str(exc)
else:
    init_error = None
 

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/nav")
def nav():
    return render_template("nav.html")


@app.route("/identify")
def identify():
    return render_template("index.html")


@app.route("/images/<filename>")
def serve_image(filename):
    """提供根目录下的图片文件"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if inference_model is None:
        return (
            jsonify({"success": False, "message": f"模型初始化失败: {init_error}"}),
            500,
        )

    if "image" not in request.files:
        return jsonify({"success": False, "message": "未检测到上传文件"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "文件名为空"}), 400

    try:
        file_bytes = file.read()
        result = inference_model.predict(file_bytes)
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        return jsonify({"success": False, "message": f"预测失败: {exc}"}), 500


@app.route("/get_treatment_advice", methods=["POST"])
def get_treatment_advice():
    """获取植物病害治理建议"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "请求数据为空"}), 400

        plant_class = data.get("plant_class", "")
        disease_name = data.get("disease_name", "")
        disease_degree = data.get("disease_degree", "")
        health_status = data.get("health_status", "")
        # 优先使用请求中的provider，否则使用环境变量，最后使用mock
        provider = data.get("provider") or os.environ.get("LLM_PROVIDER", "mock")

        if not all([plant_class, disease_name]):
            return jsonify({"success": False, "message": "缺少必要参数"}), 400

        # 获取LLM服务实例
        llm_service = get_llm_service(provider=provider)
        
        # 获取治理建议
        advice = llm_service.get_treatment_advice(
            plant_class=plant_class,
            disease_name=disease_name,
            disease_degree=disease_degree,
            health_status=health_status,
        )

        return jsonify({"success": True, "advice": advice})
    except Exception as exc:
        return jsonify({"success": False, "message": f"获取建议失败: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)  # 开启debug模式，模板修改后自动重新加载


