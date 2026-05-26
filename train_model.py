"""模型训练入口。

【使用前提】先跑过 prepare_dataset.py，确保 input/train 与 input/val 已就绪。

【运行方式】

    python train_model.py                    # 默认 20 epoch、batch 64
    python train_model.py --epochs 5         # 调一些超参
    python train_model.py --batch-size 16    # CPU/小显存机器
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from plant_disease.training import train as train_module

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "input"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--ckpt-out",
        default=str(REPO_ROOT / "resources" / "mobilenetv2_best.pth"),
        help="权重保存位置；默认放到 resources/ 这样 run_web.py 直接能加载",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    train_module.main(args)

    logger.info("✓ 完成。权重保存到 %s。下一步：python run_web.py", args.ckpt_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
