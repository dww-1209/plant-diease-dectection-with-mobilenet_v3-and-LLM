import json
from pathlib import Path

import pytest
from PIL import Image

from plant_disease.data.dataset_classifier import classify_dataset


def _make_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


@pytest.fixture
def baidu_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_image(images / name)

    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {"image_id": "a.jpg", "disease_class": 0},
                {"image_id": "b.jpg", "disease_class": 1},
                {"image_id": "c.jpg", "disease_class": 0},
                {"image_id": "missing.jpg", "disease_class": 2},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    return images, annotations, out


def test_classify_copy_preserves_originals(baidu_layout):
    images, annotations, out = baidu_layout
    summary = classify_dataset(images, annotations, out, mode="copy")

    assert summary == {"placed": 3, "missing": 1, "classes": 2}
    assert (out / "0" / "a.jpg").exists()
    assert (out / "0" / "c.jpg").exists()
    assert (out / "1" / "b.jpg").exists()
    # Originals still there
    assert (images / "a.jpg").exists()
    assert (images / "b.jpg").exists()


def test_classify_move_clears_source(baidu_layout):
    images, annotations, out = baidu_layout
    classify_dataset(images, annotations, out, mode="move")

    assert (out / "0" / "a.jpg").exists()
    assert not (images / "a.jpg").exists()
    assert not (images / "b.jpg").exists()


def test_classify_invalid_mode_raises(baidu_layout):
    images, annotations, out = baidu_layout
    with pytest.raises(ValueError):
        classify_dataset(images, annotations, out, mode="link")  # type: ignore[arg-type]
