"""61 类索引 → 人类可读的中文标签的映射。

百度数据集把 61 个细分类拆成 4 个语义维度：植物种类（10）× 是否健康（2）×
病害程度（4）× 病害类型（28）。``actual_classed_v2.txt`` 给出每个 0..60 类
ID 对应这 4 个维度上的索引；本模块负责加载与查询。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PLANT_CLASS = {
    0: "苹果",
    1: "樱桃",
    2: "玉米",
    3: "葡萄",
    4: "柑桔",
    5: "桃",
    6: "辣椒",
    7: "马铃薯",
    8: "草莓",
    9: "番茄",
}
HEALTHY = {0: "未患病", 1: "患病"}
DISEASED_DEGREE = {0: "健康", 1: "一般", 2: "严重", 3: "患病但不分程度"}
DISEASED_CLASS = {
    0: "健康",
    1: "苹果黑星病",
    2: "苹果灰斑病",
    3: "苹果雪松锈病",
    4: "樱桃白粉病",
    5: "玉米灰斑病",
    6: "玉米锈病",
    7: "玉米叶斑病",
    8: "玉米花叶病毒病",
    9: "葡萄黑腐病",
    10: "葡萄轮斑病",
    11: "葡萄褐斑病",
    12: "柑桔黄龙病",
    13: "桃疮痂病",
    14: "辣椒疮痂病",
    15: "马铃薯早疫病",
    16: "马铃薯晚疫病",
    17: "草莓叶枯病",
    18: "番茄白粉病",
    19: "番茄疮痂病",
    20: "番茄早疫病",
    21: "番茄晚疫病",
    22: "番茄叶霉病",
    23: "番茄斑点病",
    24: "番茄斑枯病",
    25: "番茄红蜘蛛损伤",
    26: "番茄黄化曲叶霉病",
    27: "番茄花叶病毒病",
}

CLASS_DICTS = {
    "plant": PLANT_CLASS,
    "healthy": HEALTHY,
    "degree": DISEASED_DEGREE,
    "disease": DISEASED_CLASS,
}


@dataclass(frozen=True)
class ClassInfo:
    """单个类别 ID 解码后的 4 维语义信息。

    ``frozen=True`` 让实例不可改，避免运行时被无意中篡改影响其他调用。
    """

    plant: str
    health_status: str
    disease_degree: str
    disease_name: str


def load_class_map(path: Path) -> list[ClassInfo]:
    """读取 61 类映射表，返回按 class_id 顺序排列的 ``ClassInfo`` 列表。

    每行格式：``<class_id> <plant_idx> <healthy_idx> <degree_idx> <disease_idx>``。
    格式不对 / 字段不全的行**直接跳过**（不报错），文件不存在时返回空列表。
    这样让 Web 即使在缺类别表的情况下也能起来（虽然返回的标签会是占位符）。
    """
    if not path.exists():
        logger.warning("class map not found: %s", path)
        return []

    rows: list[ClassInfo] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                idx = [int(x) for x in parts[:5]]
            except ValueError:
                continue
            _, plant_i, healthy_i, degree_i, disease_i = idx
            rows.append(
                ClassInfo(
                    plant=PLANT_CLASS.get(plant_i, f"未知植物{plant_i}"),
                    health_status=HEALTHY.get(healthy_i, "未知"),
                    disease_degree=DISEASED_DEGREE.get(degree_i, "未知"),
                    disease_name=DISEASED_CLASS.get(disease_i, "未知"),
                )
            )
    return rows


def lookup_class(rows: list[ClassInfo], idx: int) -> ClassInfo:
    """按索引取 ``ClassInfo``，越界时返回 ``"类别{idx}" + "未知"`` 占位。

    返回占位（而非抛异常）是为了让模型推出冷门类时 UI 仍能正常显示，不至于
    因为一个 ImageFolder 顺序错位就让整个 ``/predict`` 500。
    """
    if 0 <= idx < len(rows):
        return rows[idx]
    return ClassInfo(
        plant=f"类别{idx}", health_status="未知", disease_degree="未知", disease_name="未知"
    )
