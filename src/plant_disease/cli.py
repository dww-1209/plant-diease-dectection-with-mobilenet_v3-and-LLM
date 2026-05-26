"""Console entry point: `plant-disease serve | train`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv


def _setup_logging() -> None:
    level = (
        logging.DEBUG
        if os.environ.get("PLANT_DISEASE_DEBUG", "").lower() in {"1", "true"}
        else logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _cmd_serve(_args: argparse.Namespace) -> int:
    from plant_disease.config import load_settings
    from plant_disease.web.app import create_app

    settings = load_settings()
    app = create_app(settings)
    # Flask's app.run is the bundled dev server — don't expose it as production.
    app.run(host=settings.host, port=settings.port, debug=settings.flask_debug)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from plant_disease.training import train as train_module

    return train_module.main(args)


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    from plant_disease.data.dataset_classifier import main as prepare_main

    return prepare_main(args)


def _cmd_clean_data(args: argparse.Namespace) -> int:
    from plant_disease.data.data_clean import process_repeat

    process_repeat(args.train, args.val)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plant-disease")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="Start the Flask web server")

    t = sub.add_parser("train", help="Train MobileNetV2 on the dataset")
    t.add_argument(
        "--data-dir", default="input", help="Dataset root (expects train/ and val/ subdirs)"
    )
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--ckpt-out", default="mobilenetv2_best.pth")

    p = sub.add_parser(
        "prepare-data",
        help="Place Baidu-2018 images into per-class subdirs based on the JSON annotations",
    )
    p.add_argument("--images", required=True, help="Directory with the flat *.jpg files")
    p.add_argument(
        "--annotations", required=True, help="JSON file mapping image_id → disease_class"
    )
    p.add_argument(
        "--out", required=True, help="Output root; per-class subdirs are created here"
    )
    p.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy keeps originals (recommended); move is faster but destructive",
    )

    c = sub.add_parser(
        "clean-data",
        help="Drop images with '副本' duplicate names and remove train/val overlap",
    )
    c.add_argument("--train", required=True)
    c.add_argument("--val", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _cmd_serve(args)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "prepare-data":
        return _cmd_prepare_data(args)
    if args.cmd == "clean-data":
        return _cmd_clean_data(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
