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
uv sync --all-extras       # 安装依赖（含训练 + 开发工具）
uv run plant-disease serve # http://localhost:5000
```

> 推理需要权重文件 `resources/mobilenetv2_best.pth`（不入仓库）。无权重时 Web 可启动，但 `/predict` 会返回 503。

## CLI 总览

**所有功能都通过 `plant-disease` 一个命令的子命令调用，不要直接 `python xxx.py`。**

| 子命令 | 何时用 | 关键参数 |
|---|---|---|
| `prepare-data` | 把百度原始扁平数据集按类别整理成 `ImageFolder` 结构 | `--images`、`--annotations`、`--out`、`--mode {copy,move}` |
| `clean-data` | 去重 + 删除 train/val 重叠（**破坏性**） | `--train`、`--val` |
| `train` | 训练 MobileNetV2 | `--data-dir`、`--epochs`、`--batch-size`、`--lr`、`--patience`、`--ckpt-out` |
| `serve` | 启动 Flask Web 服务 | 通过环境变量配置 `HOST` / `PORT` / `FLASK_DEBUG` |

任意子命令都可以加 `--help` 查看完整参数：

```bash
plant-disease --help
plant-disease prepare-data --help
plant-disease train --help
```

## 端到端流程

下面是从「刚下完百度数据集」到「Web 能识别 + 给建议」的完整顺序，按部就班执行即可。

```
prepare-data (train)
    ↓
prepare-data (val)
    ↓
clean-data        ← 可选
    ↓
train
    ↓
serve
```

### 步骤 1：准备环境（一次性）

```bash
uv sync --all-extras
cp .env.example .env       # 不填 key 也行，默认 LLM_PROVIDER=mock
```

### 步骤 2：归类训练集和验证集

假设数据集解压在 `~/Downloads/baidu_pdr2018/`：

```bash
plant-disease prepare-data \
    --images      ~/Downloads/baidu_pdr2018/AgriculturalDisease_trainingset/images \
    --annotations ~/Downloads/baidu_pdr2018/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json \
    --out         input/train \
    --mode        copy

plant-disease prepare-data \
    --images      ~/Downloads/baidu_pdr2018/AgriculturalDisease_validationset/images \
    --annotations ~/Downloads/baidu_pdr2018/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json \
    --out         input/val \
    --mode        copy
```

跑完后 `input/{train,val}/0..60/*.jpg` 齐全。`--mode copy` 默认保留原始数据；磁盘紧张可改 `--mode move`。

### 步骤 3：清洗（可选）

```bash
plant-disease clean-data --train input/train --val input/val
```

只在步骤 2 用了 `--mode copy` 时再跑，方便回滚。

### 步骤 4：训练

```bash
plant-disease train --data-dir input --epochs 20 --batch-size 64
```

产物：`mobilenetv2_best.pth`、`artifacts/loss.png`、`artifacts/accuracy.png`。

### 步骤 5：启动 Web

把权重移到默认位置后启动：

```bash
mv mobilenetv2_best.pth resources/mobilenetv2_best.pth
plant-disease serve
```

打开 [http://localhost:5000](http://localhost:5000) → 上传图片 → 点「获取治理建议」。

## 数据集说明

百度官方数据集刚下载下来是「扁平图像 + JSON 标注」结构：

```
AgriculturalDisease_trainingset/
├── images/*.jpg                                       # 31718 张（无类别目录）
└── AgriculturalDisease_train_annotations.json        # image_id → disease_class

AgriculturalDisease_validationset/
├── images/*.jpg
└── AgriculturalDisease_validation_annotations.json
```

而本项目训练用的是 `torchvision.datasets.ImageFolder`，需要 `train/<class_id>/*.jpg` 这种按类目分文件夹的结构。所以必须先经过 `prepare-data` 归类（见上面的端到端流程）。

清洗逻辑（`clean-data`）做两件事：
- 删除文件名带「副本」的重复图片（人工拷贝痕迹）
- 删除 train 与 val 出现的同名图片，避免数据泄漏

## HTTP 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 首页 |
| `/identify` | GET | 上传 + 识别页面 |
| `/nav` | GET | 关于页 |
| `/predict` | POST | `multipart/form-data`，字段 `image` |
| `/get_treatment_advice` | POST | JSON `{plant_class, disease_name, disease_degree, health_status, provider?}` |

`serve` 默认绑定 `127.0.0.1:5000`，通过 `HOST` / `PORT` / `FLASK_DEBUG` 环境变量调整。

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
├── cli.py            # plant-disease 入口；serve / train / prepare-data / clean-data
├── config.py         # Settings（统一读环境变量）
├── errors.py         # 自定义异常
├── model.py          # InferenceModel (MobileNetV2)
├── data/             # class_map / dataset_classifier / data_clean
├── llm/              # base + mock / openai / baidu / alibaba + factory
├── training/train.py # 训练流程
└── web/
    ├── app.py        # Flask app factory
    ├── routes.py     # 路由
    ├── templates/    # 前端 HTML
    └── static/       # CSS / JS
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
