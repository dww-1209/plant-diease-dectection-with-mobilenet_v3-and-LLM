# 植物病害识别项目全面重构 · 设计文档

- **日期**：2026-05-26
- **范围**：纯代码层面优化（无权重/数据）。修复架构混乱、缺失文件、重复代码，建立清晰包结构、依赖管理、测试与文档。
- **目标**：让仓库能在新机器上 `uv sync` → `uv run plant-disease serve` 正常启动 Web demo（推理因无权重而无法验证 accuracy，但代码路径完整、错误处理优雅）。

---

## 1. 现状问题清单

| # | 问题 | 影响 |
|---|---|---|
| 1 | 仓库名/README 写 MobileNetV3，`model.py` 实为 V2，`train.py` 实为 ViT-B/16 | 三处自相矛盾 |
| 2 | `app.py` 引用 `templates/{home,nav,index}.html` 全部缺失 | Web 启动 500 |
| 3 | 无 `.pth` 权重，且无任何下载/训练产出说明 | 推理不可用 |
| 4 | `class_indices_61cls.json` 0 字节空文件 | 死文件 |
| 5 | `predict.py` 含 `__main__` 后死代码、硬编码 `cuda:0` | 非 GPU/桌面端崩溃 |
| 6 | `train.py` ckpt 命名为 `mobilenetv2_finetune_best.pth` 但模型是 ViT；硬编码设备；`target_names = [str(i) for i in range(61)]` | 误导且不可移植 |
| 7 | `llm_service.py` 三个 Provider 几乎一字不差地复制 prompt | DRY 违反 |
| 8 | `app.py` `debug=True` 写死；无 `.env.example`、`.gitignore` | 生产风险 |
| 9 | `requirements.txt` 缺 `tqdm`/`matplotlib`/`scikit-learn`/`opencv-python` | 环境装不齐 |
| 10 | `read_txt.py` 没 `encoding="utf-8"` | 跨平台问题 |
| 11 | `model.py` 推理路径仍配置 `requires_grad`（推理无用） | 死代码 |
| 12 | 全局 `print` + emoji；零测试 | 工程化薄弱 |
| 13 | `data_clean.py` 注释/逻辑混乱、硬编码路径 | 维护负担 |

## 2. 决策汇总（来自头脑风暴）

| 维度 | 决策 |
|---|---|
| 模型架构 | **统一到 MobileNetV2**（README/train/model 三处对齐） |
| Web 前端 | 重做 `templates/`，**视觉好看的现代风格** |
| LLM 服务 | 抽公共 prompt 模板 + base 调用流程，保留 OpenAI/百度/通义 三家 + Mock |
| `predict.py`（Tkinter） | **删除** |
| 训练脚本 | 换 V2 + 抽函数 + argparse + 设备自动检测 |
| 工程化 | **uv** 管理依赖（`pyproject.toml` 由 `uv init` 生成）+ 重写中文 README + pytest + ruff + black |
| 重构深度 | **方案 B（合理分包）**：引入 `src/plant_disease/` 包结构，按职责分子包 |

## 3. 目标包结构

```
plant-disease-detection/
├── pyproject.toml              # uv 管理；依赖 + entry_points
├── README.md                   # 重写（中文）
├── .python-version             # uv init 自动产
├── .gitignore                  # uv init 自动产 + 补 *.pth / input/
├── .env.example                # 列 LLM_PROVIDER / DASHSCOPE_API_KEY 等
├── src/
│   └── plant_disease/
│       ├── __init__.py
│       ├── cli.py              # entry_point: serve / train 子命令
│       ├── config.py           # 读环境变量 + 默认路径（唯一入口）
│       ├── errors.py           # 自定义异常
│       ├── model.py            # InferenceModel（V2，纯推理）
│       ├── data/
│       │   ├── __init__.py
│       │   ├── class_map.py    # 替代 read_txt.py，UTF-8 安全
│       │   ├── dataset_classifier.py
│       │   └── data_clean.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py         # LLMService 基类 + 共享 prompt 模板
│       │   ├── openai_provider.py
│       │   ├── baidu_provider.py
│       │   ├── alibaba_provider.py
│       │   ├── mock_provider.py
│       │   └── factory.py      # get_llm_service
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py        # MobileNetV2，函数化 + argparse
│       └── web/
│           ├── __init__.py
│           ├── app.py          # Flask app factory
│           └── routes.py
├── resources/
│   └── actual_classed_v2.txt   # 类映射表
├── templates/                  # home.html / nav.html / index.html
├── static/                     # css / js / 图标
└── tests/
    ├── conftest.py
    ├── test_class_map.py
    ├── test_llm_base.py
    ├── test_llm_factory.py
    ├── test_llm_mock.py
    ├── test_inference_model.py
    ├── test_routes.py
    └── test_config.py
```

注：`.pth` 权重文件不入仓库，运行期通过 `WEIGHTS_PATH` 环境变量定位（默认 `resources/mobilenetv2_best.pth`）。

## 4. 核心组件与职责

### 4.1 `model.py`：`InferenceModel`

- 只做推理。删除 `requires_grad` 配置（推理不需要）。
- 设备检测优先级：`cuda → mps → cpu`（mac 用户能跑）。
- 权重路径与类映射路径通过 `config.Settings` 注入，构造函数提供默认值。
- 权重加载失败 / 文件不存在 → 抛 `InferenceError`（不打 emoji 警告吞掉）。
- 公开 API：`predict(image_bytes: bytes) -> dict`，返回字段保持现状不变：`class_id` / `probability` / `plant_class` / `health_status` / `disease_name` / `disease_degree`。

### 4.2 `llm/base.py`：`LLMService` 基类

- 模块常量 `TREATMENT_PROMPT_TEMPLATE` 作为共享 prompt。
- 基类提供模板方法：
  - `_build_prompt(plant_class, disease_name, disease_degree, health_status) -> str`
  - `_post_json(url, headers, body, timeout=30) -> dict`（统一超时/HTTP 错处理，失败抛 `LLMServiceError`）
  - `get_treatment_advice(...)`（最终公开方法，串起整个流程）
- 子类只实现两个钩子：
  - `_endpoint(prompt) -> tuple[str, dict, dict]` 返回 `(url, headers, body)`
  - `_extract_text(response_json) -> str`
- `MockProvider` 继承自 `LLMService`，覆盖 `get_treatment_advice` 返回固定文案，用于测试和未配 key 时的 fallback。

### 4.3 `llm/factory.py`

- `get_llm_service(provider: str) -> LLMService`
- 维持现签名，不破坏 `routes.get_treatment_advice` 的调用方式。
- 不支持的 provider 抛 `ValueError`。

### 4.4 `data/class_map.py`

- 替代 `read_txt.py`。所有 `open()` 加 `encoding="utf-8"`。
- 提供：
  - `CLASS_DICTS`（Plant/Healthy/Degree/Disease 四个常量字典）
  - `@dataclass ClassInfo { plant: str; health_status: str; disease_degree: str; disease_name: str }`
  - `load_class_map(path: Path) -> list[ClassInfo]`
- 文件超界索引/格式不符时返回带占位符的 `ClassInfo`，行为保持原 `_get_class_info` 兼容。

### 4.5 `web/`

- `app.py` 提供 `create_app(settings: Settings) -> Flask` 工厂。
- `routes.py` 注册 Blueprint，含 `/`、`/nav`、`/identify`、`/predict`、`/get_treatment_advice`、`/images/<filename>`。
- `debug` 来自 `Settings.flask_debug`，默认 `False`，`FLASK_DEBUG=1` 打开。
- 启动期 `InferenceModel` 实例化失败 → 存到 `app.config["INIT_ERROR"]`，路由 503 兜底。

### 4.6 `training/train.py`

- 抽函数：`build_dataloaders()`、`build_model(num_classes)`、`train_one_epoch()`、`evaluate()`、`evaluate_full(target_names)`、`plot_history()`、`main(args)`。
- argparse: `--data-dir` `--epochs` `--batch-size` `--lr` `--patience` `--ckpt-out`。
- 设备自动检测同 4.1。
- 模型从 ViT-B/16 改回 MobileNetV2；产物默认名 `mobilenetv2_best.pth`。
- `target_names` 通过 `class_map.load_class_map` 取真实类名。

### 4.7 `cli.py`

- 通过 `pyproject.toml` 配 console scripts：`plant-disease = "plant_disease.cli:main"`。
- 子命令：
  - `plant-disease serve` → 启动 Flask
  - `plant-disease train ...` → 调 `training.train.main`
- 入口配置根 logger（默认 `INFO`，`PLANT_DISEASE_DEBUG=1` 时 `DEBUG`），调用 `dotenv.load_dotenv()`。

### 4.8 `config.py`

```python
@dataclass(frozen=True)
class Settings:
    weights_path: Path
    classes_txt: Path
    llm_provider: str           # "mock" | "openai" | "baidu" | "alibaba"
    dashscope_api_key: str = ""
    openai_api_key: str = ""
    baidu_api_key: str = ""
    baidu_secret_key: str = ""
    flask_debug: bool = False
    port: int = 5000

def load_settings() -> Settings: ...
```

- 整个项目仅 `config.py`、`training/train.py` 的 argparse、`cli.py` 接触 `os.environ`。

## 5. 数据流

### 5.1 推理请求

```
浏览器上传图片
    │  POST /predict (multipart)
Flask routes.predict
    │  image_bytes
InferenceModel.predict(bytes)
    │  PIL → transform(224×224 + Normalize) → forward → softmax
    │  class_map[idx] → ClassInfo
    ▼
{class_id, probability, plant_class, health_status, disease_name, disease_degree}
    │
JSON 响应 → 前端展示
```

### 5.2 LLM 建议请求

```
前端 POST /get_treatment_advice  {plant, disease, degree, health, provider?}
    │  provider 优先级：请求体 > Settings.llm_provider > "mock"
factory.get_llm_service(provider)
    │
LLMService.get_treatment_advice(...)
    ├─ _build_prompt(...)        基类
    ├─ _endpoint(prompt)         子类
    ├─ _post_json(...)           基类（含超时/异常）
    └─ _extract_text(resp)       子类
    ▼
{success, advice}  → 前端 Markdown 渲染
```

### 5.3 训练流程

```
uv run plant-disease train --data-dir input
    │
argparse → main(args)
    │
build_dataloaders → build_model(MobileNetV2, num_classes=61)
    │
for epoch:
    train_one_epoch → evaluate → 保存 best.pth（按 val_loss）
    │
plot_history → png
evaluate_full(target_names from class_map)
```

### 5.4 启动期

```
create_app(settings):
    1. InferenceModel 加载
       失败 → app.config["INIT_ERROR"] = str(exc)，进程不挂
    2. 注册 routes.bp
    3. LLM Service 不在启动期实例化（按请求 provider 现取）
```

## 6. 错误处理

### 6.1 异常层次（`errors.py`）

```python
class PlantDiseaseError(Exception): ...
class InferenceError(PlantDiseaseError): ...
class LLMServiceError(PlantDiseaseError): ...
class LLMConfigError(LLMServiceError): ...
```

### 6.2 各层处理

| 层 | 谁报 | 处理 |
|---|---|---|
| 启动期 | `InferenceModel.__init__` | 抛 `InferenceError` → `create_app` 捕获存 config，路由层 503 兜底 |
| 请求期 - 输入 | 路由 | 缺字段/空文件 → 400 + `{"success":False,"message":...}` |
| 请求期 - 推理 | `InferenceModel.predict` | 抛 `InferenceError` → 路由 500 |
| 请求期 - LLM | `_post_json` | 网络/超时/HTTP 错 → 抛 `LLMServiceError` → 路由 502 |
| 配置缺失 | `factory` / Provider 构造 | 抛 `LLMConfigError` → 路由 400，提示具体环境变量名 |

### 6.3 日志

- 用 `logging.getLogger("plant_disease.<module>")`。
- 不再 `print` + emoji。
- `cli.py` 配根 logger；训练脚本进度仍可用 `tqdm`。

## 7. 测试设计

| 文件 | 覆盖 | 依赖权重/网络 |
|---|---|---|
| `test_class_map.py` | 正常解析 / 空文件 / 超界索引 fallback / UTF-8 字符 | 否 |
| `test_llm_base.py` | `_build_prompt` 字段完整；`_post_json` 超时/HTTP 错走 `LLMServiceError` | 否（mock requests） |
| `test_llm_factory.py` | `get_llm_service` 各 provider 类型；非法值抛 `ValueError` | 否 |
| `test_llm_mock.py` | `MockProvider.get_treatment_advice` 含输入字段 | 否 |
| `test_inference_model.py` | monkeypatch `_load_weights` + dummy state_dict；验证 `predict()` 字段/形状 | 否 |
| `test_routes.py` | Flask test client：缺文件 400 / mock 200 / 错误 provider 400 | 否 |
| `test_config.py` | 默认值 / 环境变量覆盖 | 否 |

**不测**：真模型前向、真 LLM API、训练流程。

## 8. 工具链与命令

### 8.1 `pyproject.toml`

- `[project]` 主依赖：`flask`、`torch`、`torchvision`、`pillow`、`numpy`、`requests`、`python-dotenv`、`tqdm`
- `[project.optional-dependencies].train`：`matplotlib`、`scikit-learn`、`opencv-python`
- `[project.optional-dependencies].dev`：`pytest`、`ruff`、`black`
- `[project.scripts]`：`plant-disease = "plant_disease.cli:main"`

### 8.2 常用命令

| 命令 | 用途 |
|---|---|
| `uv sync` | 安装依赖 |
| `uv run plant-disease serve` | 启动 Web |
| `uv run plant-disease train --data-dir input` | 训练 |
| `uv run pytest` | 运行测试 |
| `uv run ruff check .` | Lint |
| `uv run black .` | 格式化 |

## 9. README 大纲

1. 项目简介 + Mermaid 架构图
2. 快速开始（`uv sync` → 配 `.env` → `uv run plant-disease serve`）
3. 数据准备（百度 2018 数据集下载 + 数据清洗 + dataset_classifier）
4. 模型训练（`uv run plant-disease train`）
5. Web 部署（路由说明、截图位）
6. LLM 配置（三家 Provider 各自所需环境变量）
7. 项目结构
8. 开发（pytest / ruff / black）
9. FAQ：无权重怎么办、为什么用 V2 不用 V3、mac 用户没 cuda 等
10. License

## 10. 不做范围（明确排除）

- 不验证真实推理 accuracy（无 .pth）
- 不重训模型（无数据）
- 不做模型架构升级（V3/EfficientNet 等）
- 不引入 PyTorch Lightning / accelerate
- 不引入 pydantic（YAGNI）
- 不写 Dockerfile / GitHub Actions
- 不做前端 SPA 框架（保持原生 HTML+CSS+少量 JS）
- 不破坏 `/predict` 与 `/get_treatment_advice` 的 JSON 字段契约
