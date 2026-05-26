# 植物病害识别（MobileNetV2 + LLM）

基于 [百度 2018 AI 植物病害竞赛数据集](https://challenger.ai/competition/pdr2018) 训练 MobileNetV2 多分类模型，结合大语言模型（OpenAI / 百度文心 / 阿里通义）生成针对性治理建议。

## 架构

```mermaid
flowchart LR
    User([浏览器]) -- "上传图片" --> Flask
    Flask -- "图片字节" --> Inference[MobileNetV2 推理]
    Inference -- "植物 / 病害 / 程度" --> Flask
    Flask -- "POST /get_treatment_advice" --> LLM[LLM Provider]
    LLM -- "治理建议" --> Flask
    Flask -- "JSON" --> User
```

## 快速开始

```bash
git clone <your-repo>
cd plant-disease-detection
cp .env.example .env       # 按需填写 API Key
uv sync                    # 安装依赖
uv run plant-disease serve # http://localhost:5000
```

> 推理需要权重文件 `resources/mobilenetv2_best.pth`（不入仓库）。无权重时 Web 可启动，但 `/predict` 会返回 503。

## 数据准备

```
input/
├── train/<class_id>/*.jpg   # 0..60 共 61 个目录
└── val/<class_id>/*.jpg
```

辅助脚本：

```bash
uv run python -m plant_disease.data.dataset_classifier   # 按 JSON 标注归类
uv run python -m plant_disease.data.data_clean --train ... --val ...
```

## 训练

```bash
uv sync --extra train      # 装 matplotlib / scikit-learn / opencv-python
uv run plant-disease train --data-dir input --epochs 20 --batch-size 64
```

产物：`mobilenetv2_best.pth`、`artifacts/loss.png`、`artifacts/accuracy.png`。

## Web 部署

```bash
FLASK_DEBUG=0 PORT=8000 uv run plant-disease serve
```

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 首页 |
| `/identify` | GET | 上传 + 识别页面 |
| `/nav` | GET | 关于页 |
| `/predict` | POST | `multipart/form-data`，字段 `image` |
| `/get_treatment_advice` | POST | JSON `{plant_class, disease_name, disease_degree, health_status, provider?}` |

## LLM 配置

| Provider | 必需环境变量 |
|---|---|
| `mock` | 无（用于离线测试） |
| `openai` | `OPENAI_API_KEY` |
| `baidu` | `BAIDU_API_KEY`、`BAIDU_SECRET_KEY` |
| `alibaba`（推荐） | `DASHSCOPE_API_KEY` |

通过 `LLM_PROVIDER` 默认值选择，单次请求可在 body 里覆盖 `provider`。

## 项目结构

```
src/plant_disease/
├── cli.py            # `plant-disease serve | train`
├── config.py         # Settings
├── errors.py         # 自定义异常
├── model.py          # InferenceModel (MobileNetV2)
├── data/             # class_map / dataset_classifier / data_clean
├── llm/              # base + mock/openai/baidu/alibaba + factory
├── training/train.py # 训练流程
└── web/              # Flask app factory + routes
templates/  static/   # 前端
tests/                # pytest
resources/            # actual_classed_v2.txt + (.pth 不入仓库)
```

## 开发

```bash
uv sync --all-extras
uv run pytest                # 跑全部测试
uv run ruff check .          # lint
uv run black .               # 格式化
```

## 常见问题

**Q：没有 `.pth` 权重怎么办？**
A：Web 仍可启动，识别接口会返回 503；可先用 `mock` provider 体验 LLM 接口，等训练或拷贝权重后再使用完整功能。

**Q：为什么是 MobileNetV2 而不是仓库名所写的 V3？**
A：原仓库代码实际就是 V2，本次重构选择对齐到代码现状，避免误导。后续若升级 V3 是独立 PR。

**Q：Mac 没有 CUDA 能跑吗？**
A：能。`InferenceModel` 自动选择 `cuda → mps → cpu`。

## License

MIT
