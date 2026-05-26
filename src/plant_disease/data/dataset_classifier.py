"""Classify raw Baidu-2018 images into per-class subdirectories via JSON annotations.

Baidu's dataset ships as ``images/*.jpg`` plus a flat JSON file mapping each
filename to its ``disease_class`` (0-60). Both training and validation pipelines
in this repo expect ``input/{train,val}/<class_id>/*.jpg`` (PyTorch's
``ImageFolder``), so this module bridges the two layouts.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

from tqdm import tqdm

logger = logging.getLogger(__name__)

Mode = Literal["copy", "move"]


def _read_annotations(json_path: Path) -> list[dict]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info("loaded %d annotations from %s", len(data), json_path)
    return data


def classify_dataset(
    images_dir: Path,
    annotations: Path,
    out_dir: Path,
    mode: Mode = "copy",
) -> dict[str, int]:
    """按 JSON 标注把每张图放到 ``out_dir/<disease_class>/`` 下。

    Args:
        images_dir: 扁平图片目录（``AgriculturalDisease_xxxset/images``）
        annotations: 对应的 ``*annotation*.json`` 文件
        out_dir: 输出根目录，本函数会自动建子目录
        mode: ``copy`` 保留原文件（默认，可重跑），``move`` 直接迁移

    Returns:
        ``{"placed": 已放置张数, "missing": 标注里有但图片缺失张数,
           "classes": 实际见到的类别数}``

    Raises:
        ValueError: ``mode`` 不是 copy/move
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'copy' or 'move')")

    rows = _read_annotations(annotations)
    out_dir.mkdir(parents=True, exist_ok=True)

    placed = 0
    missing = 0
    seen_classes: set[int] = set()
    transfer = shutil.move if mode == "move" else shutil.copy2

    for info in tqdm(rows, desc=f"{mode} → {out_dir.name}"):
        name = info["image_id"]
        label = int(info["disease_class"])
        src = images_dir / name
        dst_dir = out_dir / str(label)
        dst_dir.mkdir(exist_ok=True)
        if not src.exists():
            logger.warning("missing source image: %s", src)
            missing += 1
            continue
        transfer(str(src), str(dst_dir / name))
        placed += 1
        seen_classes.add(label)

    summary = {"placed": placed, "missing": missing, "classes": len(seen_classes)}
    logger.info("done: %s", summary)
    return summary


def main(args) -> int:
    summary = classify_dataset(
        images_dir=Path(args.images),
        annotations=Path(args.annotations),
        out_dir=Path(args.out),
        mode=args.mode,
    )
    if summary["missing"]:
        logger.warning("%d annotated images were not found on disk", summary["missing"])
    return 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory with the flat *.jpg files")
    parser.add_argument(
        "--annotations", required=True, help="JSON file mapping image_id → disease_class"
    )
    parser.add_argument(
        "--out", required=True, help="Output root; per-class subdirs created here"
    )
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    raise SystemExit(main(parser.parse_args()))
