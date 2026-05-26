"""数据集预处理入口：归类 + 清洗。

【使用前提】把百度官方数据集解压到项目根目录，目录结构应是：

    AgriculturalDisease_trainingset/
    ├── images/                                              # *.jpg
    └── AgriculturalDisease_train_annotations.json
    AgriculturalDisease_validationset/
    ├── images/
    └── AgriculturalDisease_validation_annotations.json

【运行方式】

    python prepare_dataset.py              # 默认：归类 + 清洗
    python prepare_dataset.py --no-clean   # 只归类，不清洗
    python prepare_dataset.py --move       # 用移动而非复制（节省磁盘）

跑完会得到：

    input/
    ├── train/0..60/*.jpg
    └── val/0..60/*.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from plant_disease.data.data_clean import process_repeat
from plant_disease.data.dataset_classifier import classify_dataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_RAW = REPO_ROOT / "AgriculturalDisease_trainingset"
VAL_RAW = REPO_ROOT / "AgriculturalDisease_validationset"
TRAIN_OUT = REPO_ROOT / "input" / "train"
VAL_OUT = REPO_ROOT / "input" / "val"


def _find_annotations(raw_dir: Path) -> Path:
    matches = list(raw_dir.glob("*annotation*.json"))
    if not matches:
        raise FileNotFoundError(f"在 {raw_dir} 下找不到 *annotation*.json")
    return matches[0]


def _classify_split(raw_dir: Path, out_dir: Path, mode: str) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"找不到原始数据目录：{raw_dir}\n"
            f"请把百度数据集解压到项目根目录，详见 README 的「端到端流程」一节。"
        )
    images = raw_dir / "images"
    if not images.exists():
        raise FileNotFoundError(f"{raw_dir} 下找不到 images/ 子目录")
    annotations = _find_annotations(raw_dir)

    summary = classify_dataset(
        images_dir=images, annotations=annotations, out_dir=out_dir, mode=mode
    )
    logger.info("[%s] %s", out_dir.name, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--move", action="store_true",
        help="迁移而非复制原文件（节省磁盘但破坏性）"
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="跳过清洗（不删除重名 / train-val 重叠）"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mode = "move" if args.move else "copy"

    logger.info("=== 步骤 1/2：归类训练集 ===")
    _classify_split(TRAIN_RAW, TRAIN_OUT, mode)

    logger.info("=== 步骤 2/2：归类验证集 ===")
    _classify_split(VAL_RAW, VAL_OUT, mode)

    if args.no_clean:
        logger.info("已跳过清洗（--no-clean）")
    else:
        logger.info("=== 数据清洗：去重 + 删除 train/val 重叠 ===")
        process_repeat(str(TRAIN_OUT), str(VAL_OUT))

    logger.info("✓ 完成。下一步：python train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
