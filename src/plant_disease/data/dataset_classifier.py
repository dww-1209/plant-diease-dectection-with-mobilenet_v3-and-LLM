"""Classify raw competition images into per-class subdirectories using JSON annotations."""

from __future__ import annotations

import json
import logging
import os
import shutil

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClassifyAsLabel:
    @staticmethod
    def read_json(json_path: str) -> list[dict]:
        with open(json_path, encoding="utf-8") as f:
            data = json.loads(f.read())
        logger.info("loaded %d annotations from %s", len(data), json_path)
        return data

    def classify(self, img_path: str, json_path: str, out_path: str) -> None:
        """Move each image into out_path/<disease_class>/."""
        annotations = self.read_json(json_path)
        os.makedirs(out_path, exist_ok=True)

        for info in tqdm(annotations, desc="classifying"):
            name = info["image_id"]
            label = info["disease_class"]
            src = os.path.join(img_path, name)
            dst_dir = os.path.join(out_path, str(label))
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.move(src, dst_dir)
            else:
                logger.warning("missing source image: %s", src)
