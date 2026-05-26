"""MobileNetV2-based inference model."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from plant_disease.data.class_map import ClassInfo, load_class_map, lookup_class
from plant_disease.errors import InferenceError

logger = logging.getLogger(__name__)

NUM_CLASSES = 61
IMG_SIZE = 224


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceModel:
    """Loads MobileNetV2 weights once and serves predictions."""

    def __init__(
        self,
        weights_path: Path,
        classes_txt: Path,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.device = _select_device()
        logger.info("inference device: %s", self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.class_info: list[ClassInfo] = load_class_map(classes_txt)
        self.num_classes = num_classes

        self.model = self._build_model(num_classes)
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, num_classes: int) -> nn.Module:
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes),
        )
        return model

    def _load_weights(self, weights_path: Path) -> None:
        if not weights_path.exists():
            raise InferenceError(f"权重文件未找到：{weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.info(
                "weights partial match: missing=%d, unexpected=%d", len(missing), len(unexpected)
            )

    def _prepare_image(self, file_bytes: bytes) -> torch.Tensor:
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InferenceError(f"无法解码图片：{exc}") from exc
        return torch.unsqueeze(self.transform(img), dim=0)

    def predict(self, file_bytes: bytes) -> dict:
        tensor = self._prepare_image(file_bytes).to(self.device)
        try:
            with torch.no_grad():
                logits = torch.squeeze(self.model(tensor)).cpu()
                probs = torch.softmax(logits, dim=0)
                cls_idx = int(torch.argmax(probs).item())
                prob = float(probs[cls_idx].item())
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"前向推理失败：{exc}") from exc

        info = lookup_class(self.class_info, cls_idx)
        return {
            "class_id": cls_idx,
            "probability": prob,
            "plant_class": info.plant,
            "health_status": info.health_status,
            "disease_name": info.disease_name,
            "disease_degree": info.disease_degree,
        }
