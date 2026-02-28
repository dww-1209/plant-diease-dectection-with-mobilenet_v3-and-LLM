import os
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

try:
    from read_txt import read_txt_to_dict
except ImportError:
    read_txt_to_dict = None


class InferenceModel:
    """
    封装的推理模型，按需加载权重并提供预测接口。
    """

    def __init__(
        self,
        weights_path: str = "mobilenetv2_best.pth",  # 使用标准MobileNetV2权重
        #weights_path: str = 'dual_stream_mobilenet_best.pth',  # Dual-Stream需要匹配的架构
        classes_txt: str = "actual_classed_v2.txt",
    ) -> None:
        # 获取脚本所在目录，确保路径基于项目根目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 将相对路径转换为绝对路径
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(base_dir, weights_path)
        if not os.path.isabs(classes_txt):
            classes_txt = os.path.join(base_dir, classes_txt)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.class_info = self._load_class_info(classes_txt)
        self.num_classes = max(len(self.class_info), 61)  # 61 来自原脚本配置
        self.model = self._build_model(self.num_classes)
        self._load_weights(self.model, weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_class_info(self, file_path: str) -> List[Tuple[str, str, str, str]]:
        """
        读取类别信息文件。
        优先使用 read_txt_to_dict 函数（如果可用），否则使用默认的文本格式解析。
        当文件不存在或格式不符时，使用占位符。
        """
        if not os.path.exists(file_path):
            return []

        # 如果 read_txt_to_dict 可用，使用它来解析文件
        if read_txt_to_dict is not None:
            try:
                class_dict = read_txt_to_dict(file_path)
                # 将字典转换为列表，按类别ID排序
                rows: List[Tuple[str, str, str, str]] = []
                max_key = max(class_dict.keys()) if class_dict else -1
                for idx in range(max_key + 1):
                    if idx in class_dict:
                        values = class_dict[idx]
                        # values 应该是 [植物名称, 是否健康, 患病程度, 病害名称]
                        if len(values) >= 4:
                            rows.append((values[0], values[1], values[2], values[3]))
                        else:
                            rows.append((f"类别{idx}", "未知", "未知", "未知"))
                    else:
                        rows.append((f"类别{idx}", "未知", "未知", "未知"))
                return rows
            except Exception as e:
                # 如果 read_txt_to_dict 解析失败，回退到默认方法
                print(f"警告：使用 read_txt_to_dict 解析失败，回退到默认方法: {e}")

        # 默认方法：直接读取文本格式（每行4个字段）
        rows: List[Tuple[str, str, str, str]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = (
                    line.strip()
                    .replace("\t", " ")
                    .replace(",", " ")
                    .split()
                )
                if len(parts) >= 4:
                    rows.append((parts[0], parts[1], parts[2], parts[3]))
        # 如果文件存在但没有有效行，仍返回空，后续会生成占位符
        return rows

    def _build_model(self, num_classes: int) -> nn.Module:
        model = models.mobilenet_v2()

        backbone_layers = list(model.features.children())
        UNFREEZE_RATIO = 0.30
        start_unfreeze = int(len(backbone_layers) * (1 - UNFREEZE_RATIO))

        # 推理阶段冻结与否影响不大，但保持与训练代码一致的结构。
        for i, layer in enumerate(backbone_layers):
            requires = i >= start_unfreeze and not isinstance(layer, nn.BatchNorm2d)
            for param in layer.parameters():
                param.requires_grad = requires

        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes),
        )
        return model

    def _load_weights(self, model: nn.Module, weights_path: str) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件未找到：{weights_path}")
        state_dict = None
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            # 兼容旧版 torch 无 weights_only 参数
            state_dict = torch.load(weights_path, map_location=self.device)
        
        # 尝试加载权重并检查匹配情况
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 如果有很多不匹配的键，可能是架构不匹配
        if len(missing_keys) > 10 or len(unexpected_keys) > 10:
            print(f"⚠️ 警告：权重文件可能与模型架构不匹配！")
            print(f"   权重文件：{weights_path}")
            print(f"   缺失的键数量: {len(missing_keys)}")
            print(f"   意外的键数量: {len(unexpected_keys)}")
            print(f"   这可能导致识别性能下降。")
            print(f"   建议：确保权重文件与模型架构匹配。")
        elif len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"ℹ️ 信息：权重文件部分匹配（缺失{len(missing_keys)}个键，意外{len(unexpected_keys)}个键）")
        else:
            print(f"✅ 权重文件加载成功：{weights_path}")

    def _prepare_image(self, file_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        tensor = self.transform(image)
        tensor = torch.unsqueeze(tensor, dim=0)
        return tensor

    def predict(self, file_bytes: bytes) -> Dict:
        """
        输入：图片二进制。
        输出：包含类别和概率的字典。
        """
        input_tensor = self._prepare_image(file_bytes).to(self.device)
        with torch.no_grad():
            logits = torch.squeeze(self.model(input_tensor)).cpu()
            probs = torch.softmax(logits, dim=0)
            cls_idx = int(torch.argmax(probs).item())

        prob_value = float(probs[cls_idx].item())
        info = self._get_class_info(cls_idx)

        return {
            "class_id": cls_idx,
            "probability": prob_value,
            "plant_class": info["plant_class"],
            "health_status": info["health_status"],
            "disease_name": info["disease_name"],
            "disease_degree": info["disease_degree"],
        }

    def _get_class_info(self, idx: int) -> Dict[str, str]:
        if self.class_info and 0 <= idx < len(self.class_info):
            plant, healthy, disease_degree, disease_class = self.class_info[idx]
        else:
            # 占位符
            plant = f"类别{idx}"
            healthy = "未知"
            disease_class = "未知"
            disease_degree = "未知"

        return {
            "plant_class": plant,
            "health_status": healthy,
            "disease_name": disease_class,
            "disease_degree": disease_degree,
        }
